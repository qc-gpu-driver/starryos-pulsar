//! RKNPU 驱动的 I/O 控制（ioctl）接口。
//!
//! # 概述
//!
//! 本模块定义了用户空间（rknn 运行时库）通过 ioctl 系统调用传递给内核驱动的
//! **数据结构**，以及对 NPU 硬件进行编程的驱动端**提交逻辑**。
//!
//! # 典型的 ioctl 流程（从用户空间角度）
//!
//! ```text
//!  rknn 运行时                          驱动（此代码）
//!  ────────────                         ──────────────────
//!  1. open("/dev/rknpu")
//!  2. ioctl(MEM_CREATE, RknpuMemCreate)  → GemPool::create()
//!     ◄── handle, dma_addr, obj_addr       分配 DMA 缓冲区
//!  3. memcpy(obj_addr, input_tensor)       CPU 写入缓冲区
//!  4. 填充任务描述符（RknpuTask[]）
//!  5. ioctl(SUBMIT, RknpuSubmit)         → submit_ioctrl()
//!     ◄── task_counter, hw_elapse_time     编程 PC，等待 IRQ
//!  6. memcpy(output, obj_addr)             CPU 读取结果
//!  7. ioctl(MEM_DESTROY, handle)         → GemPool::destroy()
//! ```
//!
//! # 多核提交
//!
//! RK3588 有多达 3 个 NPU 核心。单个 `RknpuSubmit` 可以通过 `subcore_task[5]`
//! 数组针对多个核心 — 每个条目指定任务数组的哪个切片发送到哪个核心。
//! 驱动遍历非空条目并为每个条目调用 `submit_one()`。
use crate::Vec;
use core::sync::atomic::Ordering;

use crate::status::{ActiveCoreBinding, NpuOwnerIds, TaskNpuState, TaskNpuStateKey};
use crate::{ResidentOwnerState, Rknpu, RknpuError, RknpuTask};

/// 从用户空间传递的子核心任务范围。
///
/// `RknpuSubmit::subcore_task[5]` 中的每个条目都是其中之一。
/// 它告诉驱动："将 tasks[task_start .. task_start+task_number]
/// 提交到此特定 NPU 核心。"
#[repr(C)]
#[derive(Debug, Clone, Default)]
pub struct RknpuSubcoreTask {
    /// 此核心的任务数组中第一个任务的索引。
    pub task_start: u32,
    /// 此核心应执行的连续任务数。
    pub task_number: u32,
}

/// 通过 mmap 将 GEM 缓冲区映射到用户空间的参数。
///
/// 用户空间使用 GEM 句柄调用 ioctl(MEM_MAP)，并获得可以传递给
/// `mmap()` 以映射缓冲区的 `offset`。
#[repr(C)]
#[derive(Debug, Clone, Default)]
pub struct RknpuMemMap {
    /// GEM 对象句柄（从 MEM_CREATE 获得）。
    pub handle: u32,
    /// 用于 64 位对齐的填充。
    pub reserved: u32,
    /// 驱动返回的伪文件偏移量，适用于 mmap()。
    pub offset: u64,
}

/// 主要的任务提交 ioctl 结构。
///
/// 用户空间用任务元数据填充此结构并将其传递给 ioctl(SUBMIT)。
/// 驱动使用它来：
/// 1. 在内存中定位任务描述符数组（`task_obj_addr`）
/// 2. 在每个目标核心上编程 PC 模块
/// 3. 等待完成并填充 `task_counter` / `hw_elapse_time`
///
/// # 字段摘要
///
/// ```text
///  flags            ──  JobMode 位（PC、NONBLOCK、PINGPONG 等）
///  task_obj_addr    ──  RknpuTask[] 数组的 CPU 地址
///  task_base_addr   ──  RknpuTask[] 的 DMA 地址（NPU 看到的）
///  subcore_task[5]  ──  每个核心的任务范围
///  core_mask        ──  要使用哪些核心（位掩码）
/// ```
#[repr(C)]
#[derive(Debug, Clone, Default)]
pub struct RknpuSubmit {
    /// 作业模式标志 — 参见 [`JobMode`]（PC、BLOCK、NONBLOCK、PINGPONG 等）。
    pub flags: u32,
    /// 返回超时前等待完成的最长时间（毫秒）。
    pub timeout: u32,
    /// （传统）全局任务起始索引 — 已被 subcore_task[] 取代。
    pub task_start: u32,
    /// 所有核心的任务总数。
    pub task_number: u32,
    /// 由驱动填充：实际执行了多少任务。
    pub task_counter: u32,
    /// 调度优先级提示（越低 = 优先级越高）。
    pub priority: i32,
    /// `RknpuTask[]` 数组的 CPU 虚拟地址。
    /// 驱动从此地址读取任务描述符。
    pub task_obj_addr: u64,
    /// 用于地址转换的 IOMMU 域 ID。
    pub iommu_domain_id: u32,
    /// 为 64 位对齐保留。
    pub reserved: u32,
    /// `RknpuTask[]` 数组的 DMA（总线）地址。
    /// 这是编程到 PC 的 TASK_DMA_BASE_ADDR 寄存器中的内容。
    pub task_base_addr: u64,
    /// 由驱动填充：硬件执行时间（纳秒）。
    pub hw_elapse_time: i64,
    /// 选择要使用哪个 NPU 核心的位掩码。
    pub core_mask: u32,
    /// 用于同步栅栏的文件描述符（进程间 GPU/NPU 同步）。
    pub fence_fd: i32,
    /// 每个核心的任务范围。条目 `i` 描述索引 `i` 处核心的任务。
    /// `task_number == 0` 的条目被跳过。
    pub subcore_task: [RknpuSubcoreTask; 5],
}

/// 用于分配 DMA 可访问缓冲区（GEM 对象）的 Ioctl 结构。
///
/// 用户空间填充 `size`（以及可选的 `flags`），调用 ioctl(MEM_CREATE)，
/// 驱动填充其余字段，以便用户空间知道：
/// - `handle`   — 用于未来 ioctl 调用的不透明标识符
/// - `dma_addr` — NPU 将用于读/写此缓冲区的总线地址
/// - `obj_addr` — 用于直接读/写的 CPU 虚拟地址
#[repr(C)]
#[derive(Debug, Clone, Default)]
pub struct RknpuMemCreate {
    /// 驱动分配的不透明句柄（输出）。
    pub handle: u32,
    /// 内存类型/缓存属性标志（输入）。
    pub flags: u32,
    /// 请求的分配大小（字节）（输入，应为页对齐）。
    pub size: u64,
    /// 缓冲区的 CPU 虚拟地址（输出）。
    pub obj_addr: u64,
    /// NPU 可访问的 DMA 总线地址（输出）。
    pub dma_addr: u64,
    /// 实际分配的大小，在 SRAM 路径上可能与 `size` 不同（输出）。
    pub sram_size: u64,
    /// 用于地址隔离的 IOMMU 域（输入）。
    pub iommu_domain_id: i32,
    /// 核心掩码提示（保留/填充）。
    pub core_mask: u32,
}

/// 用于刷新/使 DMA 缓冲区区域无效的 Ioctl 结构。
///
/// 当平台没有硬件一致性 DMA 时，用于确保 CPU 写入和 NPU 读取
/// （或反之）之间的缓存一致性。
#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct RknpuMemSync {
    /// 方向标志（TO_DEVICE、FROM_DEVICE 等）。
    pub flags: u32,
    /// 用于 64 位对齐的填充。
    pub reserved: u32,
    /// 要同步的缓冲区的 CPU 虚拟地址。
    pub obj_addr: u64,
    /// 从缓冲区开始的字节偏移量。
    pub offset: u64,
    /// 要同步的字节数。
    pub size: u64,
}

impl Rknpu {
    /// 在每次进入 step-submit 前处理 owner 切换。
    ///
    /// 语义：
    /// - 如果硬件当前驻留的是别的 owner，先把它最新的 submit 进度读出来并保存；
    /// - 然后把当前 owner 标记为新的 resident owner；
    /// - 后续 IRQ 和 wait 路径都据此判断“这一步任务究竟属于谁”。
    fn prepare_submit_owner(&self, submit: &RknpuSubmit, owner: NpuOwnerIds) {
        if let Some(resident) = self.shared.resident_owner() {
            if resident.owner != owner {
                let previous_context = self.read_npu_context(&resident.last_submit, resident.owner);
                let previous_completed = previous_context.driver_status.batch.completed_task_count;
                let previous_total = previous_context.submit.task_number;

                if previous_total != 0 && previous_completed < previous_total {
                    warn!(
                        "[NPU] owner switch at task boundary: prev(task={}, process={}, aspace={:#x}, progress={}/{}) -> next(task={}, process={}, aspace={:#x})",
                        resident.owner.task_id,
                        resident.owner.process_id,
                        resident.owner.address_space_id,
                        previous_completed,
                        previous_total,
                        owner.task_id,
                        owner.process_id,
                        owner.address_space_id
                    );
                }

                self.shared
                    .save_owner_context(resident.owner, previous_context);
            }
        }

        let had_saved_context = self.shared.has_owner_context(owner);
        self.shared.ensure_owner_context(owner);
        self.shared.set_resident_owner(Some(ResidentOwnerState {
            owner,
            last_submit: submit.clone(),
        }));

        debug!(
            "[NPU] submit owner prepared: task={} process={} aspace={:#x} saved_ctx={}",
            owner.task_id, owner.process_id, owner.address_space_id, had_saved_context
        );
    }

    /// 在一次 step-submit 结束后，把当前 owner 的最新 submit 快照回软件上下文表。
    fn snapshot_submit_owner_context(&self, submit: &RknpuSubmit, owner: NpuOwnerIds) {
        let current_context = self.read_npu_context(submit, owner);
        self.shared.save_owner_context(owner, current_context);
        self.shared.set_resident_owner(Some(ResidentOwnerState {
            owner,
            last_submit: submit.clone(),
        }));
    }

    /// # 工作流程
    ///
    /// 1. **刷新缓存**（`comfirm_write_all`），以便 NPU 可以看到
    ///    CPU 写入 GEM 缓冲区的所有张量数据。
    /// 2. **迭代** 5 个 `subcore_task` 槽。每个非空槽触发 `submit_one()`，
    ///    它对相应的 NPU 核心进行编程并忙等待完成。
    /// 3. **使缓存无效**（`prepare_read_all`），以便 CPU 可以读取
    ///    NPU 产生的输出张量。
    /// 4. 为用户空间填充 `task_counter` 和 `hw_elapse_time`。
    pub fn submit_ioctrl(&mut self, args: &mut RknpuSubmit) -> Result<(), RknpuError> {
        self.submit_ioctrl_with_owner(args, NpuOwnerIds::default())
    }

    /// 兼容旧语义的完整 submit 包装器。
    ///
    /// 内部反复调用 `submit_ioctrl_step_with_owner()`，直到整次 submit 的所有 task 都完成。
    pub fn submit_ioctrl_with_owner(
        &mut self,
        args: &mut RknpuSubmit,
        owner: NpuOwnerIds,
    ) -> Result<(), RknpuError> {
        loop {
            if self.submit_ioctrl_step_with_owner(args, owner)? {
                return Ok(());
            }
        }
    }

    /// 执行一次“可让出”的 submit 步进。
    ///
    /// 当前实现的抢占粒度是“单个 task 完成 IRQ 边界”：
    /// - 每次调用最多只推进当前 submit 的一个 task-batch；
    /// - `args.task_counter` 既是完成计数，也是重新进入时的恢复游标；
    /// - 返回 `Ok(true)` 表示整次 submit 已全部完成。
    pub fn submit_ioctrl_step_with_owner(
        &mut self,
        args: &mut RknpuSubmit,
        owner: NpuOwnerIds,
    ) -> Result<bool, RknpuError> {
        // 只在整次 submit 第一次进入时做写回同步；后续 step 继续沿用同一批 DMA 数据。
        if args.task_counter == 0 {
            self.gem.comfirm_write_all()?;
        }

        if args.flags & 1 << 1 > 0 {
            debug!("Nonblock task");
        }

        let task_ptr = args.task_obj_addr as *mut RknpuTask;
        // 找出这次 submit 真正参与执行的 subcore 槽位。
        let active_subcore_slots: Vec<usize> = args
            .subcore_task
            .iter()
            .enumerate()
            .filter(|(_, subcore)| subcore.task_number > 0)
            .map(|(slot, _)| slot)
            .take(self.base.len())
            .collect();

        // 空提交直接视为完成，同时把当前 owner 的软件上下文更新成“空态”。
        if active_subcore_slots.is_empty() || args.task_number == 0 {
            self.gem.prepare_read_all()?;
            args.task_counter = 0;
            args.hw_elapse_time = 0;
            self.snapshot_submit_owner_context(args, owner);
            return Ok(true);
        }

        self.prepare_submit_owner(args, owner);

        // 先把 `(owner, core)` 状态槽位预建好，避免 IRQ 上下文第一次插入 map。
        for core_slot in 0..active_subcore_slots.len() {
            self.shared.ensure_task_state_entry(TaskNpuStateKey {
                owner,
                core_slot: core_slot as u8,
            });
        }

        // 把 `task_counter` 解释为“这次 submit 已稳定完成了多少个 task”，
        // 然后据此算出本次重新进入时应该从哪个全局 task 索引继续。
        let task_iter_start = args.subcore_task[active_subcore_slots[0]].task_start as usize;
        let task_iter_end = task_iter_start
            + active_subcore_slots
                .iter()
                .map(|slot| args.subcore_task[*slot].task_number as usize)
                .sum::<usize>();
        let completed_task_count =
            (args.task_counter as usize).min(task_iter_end.saturating_sub(task_iter_start));
        let task_iter = task_iter_start + completed_task_count;

        // 如果游标已经走到末尾，本次进入只需要补一次读回同步并回填完成状态。
        if task_iter >= task_iter_end {
            if let Err(err) = self.gem.prepare_read_all() {
                self.snapshot_submit_owner_context(args, owner);
                return Err(err);
            }
            args.task_counter = (task_iter_end - task_iter_start) as u32;
            args.hw_elapse_time = (args.timeout / 2) as _;
            self.snapshot_submit_owner_context(args, owner);
            return Ok(true);
        }

        // 每次 step 最多推进一个“当前可发出的批次”：
        // - 单核 submit 时就是 1 个 task；
        // - 多核 submit 时是“每个活跃 core 各 1 个 task”。
        let task_batch = active_subcore_slots
            .len()
            .min(task_iter_end - task_iter)
            .min(self.base.len());
        let submit_tasks =
            unsafe { core::slice::from_raw_parts_mut(task_ptr.add(task_iter), task_batch) };
        let int_mask: Vec<u32> = submit_tasks.iter().map(|task| task.int_mask).collect();

        for idx in 0..task_batch {
            // Drop any stale owner/core association before draining leftover IRQs from a
            // previous batch, so the drain path cannot attribute them to the new submitter.
            self.shared.clear_active_binding(idx);
            self.base[idx].drain_pending_interrupts();
            let expected_irq_mask = submit_tasks[idx].int_mask;
            /*
            let regcmd_dma = submit_tasks[idx].regcmd_addr;
            let regcfg_offset = submit_tasks[idx].regcfg_offset;
            let regcfg_amount = submit_tasks[idx].regcfg_amount;
            */
            let binding = ActiveCoreBinding {
                key: TaskNpuStateKey {
                    owner,
                    core_slot: idx as u8,
                },
                subcore_slot: active_subcore_slots[idx] as u8,
                batch_task_start: task_iter as u32,
                batch_task_count: 1,
                current_task_index: (task_iter + idx) as u32,
                expected_irq_mask,
            };
            /*
            let mut restore_image = NpuCoreRestoreImage::default();
            // regcmd 存在 DMA/GEM 缓冲区里，这里必须先按 DMA 地址复制到 CPU 可读内存，
            // 不能把 `regcmd_addr` 直接当成本地指针去解引用。
            let regcmd_words = self.gem.copy_regcmd_words(
                regcmd_dma,
                regcfg_offset as usize,
                regcfg_amount as usize,
            );
            restore_image.task_shadow_writes = if let Some(words) = regcmd_words {
                self.base[idx].build_task_shadow_writes_from_regcmds(&words)
            } else {
                warn!(
                    "[NPU] failed to resolve regcmd DMA {:#x} (offset={:#x}, words={}) for core {} task {}",
                    regcmd_dma,
                    regcfg_offset,
                    regcfg_amount,
                    idx,
                    task_iter + idx
                );
                Vec::new()
            };
            */
            let task_state = TaskNpuState {
                key: binding.key,
                subcore_slot: binding.subcore_slot,
                batch_task_start: binding.batch_task_start,
                batch_task_count: binding.batch_task_count,
                current_task_index: binding.current_task_index,
                expected_irq_mask: binding.expected_irq_mask,
                restore_verified: true,
                ..TaskNpuState::default()
            };
            self.shared.prepare_active_binding(binding, task_state);

            // 真正启动当前 step 对应的 task。
            if let Err(err) =
                self.base[idx].start_execute_one(idx, &self.data, &mut submit_tasks[idx], args)
            {
                self.shared.clear_active_bindings(task_batch);
                self.snapshot_submit_owner_context(args, owner);
                return Err(err);
            }
        }

        if let Err(err) = self.wait_all_npucore(self.wait_fn, int_mask, submit_tasks) {
            self.shared.clear_active_bindings(task_batch);
            self.snapshot_submit_owner_context(args, owner);
            return Err(err);
        }

        // 这一步稳定完成后，更新 submit 级游标；下次重新进入时就从这里继续。
        args.task_counter = (task_iter + task_batch - task_iter_start) as u32;
        let finished = args.task_counter as usize >= (task_iter_end - task_iter_start);
        if finished {
            if let Err(err) = self.gem.prepare_read_all() {
                self.snapshot_submit_owner_context(args, owner);
                return Err(err);
            }
            args.task_counter = (task_iter_end - task_iter_start) as u32;
            args.hw_elapse_time = (args.timeout / 2) as _;
        }
        self.snapshot_submit_owner_context(args, owner);

        Ok(finished)
    }

    /// 等待所有 NPU 核心完成当前任务。
    pub fn wait_all_npucore(
        &self,
        normal_wait_fn: Option<fn()>,
        int_mask: Vec<u32>,
        submit_tasks: &mut [RknpuTask],
    ) -> Result<(), RknpuError> {
        let wait_start_idx: usize = 0; //从0开始等待
        let max_core: usize = submit_tasks.len();
        let mut done: [bool; 3] = [false; 3]; //跟踪每个核心是否完成
        if let Some(wait) = normal_wait_fn {
            debug!("[NPU]   waiting (IRQ+WFI mode)...");
            // ┌─────────────────────────────────────────────────────┐
            // │  IRQ-driven mode (CPU sleeps between checks)        │
            // │                                                     │
            // │  NPU runs → fires IRQ → handler calls               │
            // │  handle_interrupt() → stores fuzzed status to       │
            // │  irq_status atomic → CPU wakes from WFI → we       │
            // │  read the atomic here and proceed.                  │
            // └─────────────────────────────────────────────────────┘
            loop {
                let status: Vec<u32> = self
                    .base
                    .iter()
                    .map(|core| core.irq_status.load(Ordering::Acquire))
                    .collect();
                let mut int_status: [u32; 3] = [0; 3];
                for idx in wait_start_idx..max_core {
                    if status[idx] & int_mask[idx] > 0 {
                        int_status[idx] = int_mask[idx] & status[idx]; //一次最多会有一个核心完成，cpu会处理
                        self.base[idx].clean_interrupts();
                        // Reset atomic for the next batch.
                        self.base[idx].irq_status.store(0, Ordering::Release);
                        warn!("[NPU]   batch done: int_status={:#x}", int_status[idx]);
                        submit_tasks[idx].int_status = int_status[idx]; //给对应核心的单任务设置完成状态，用户空间会检查这个状态并继续处理结果
                        if let Some(binding) = self.shared.active_binding(idx) {
                            let mut states = self.shared.task_npu_state.lock();
                            if let Some(state) = states.get_mut(&binding.key) {
                                state.observed_irq_status = status[idx];
                                state.last_task_int_status = int_status[idx];
                                /*
                                // IRQ handler 在发布 `irq_status` 之前，已经完成了
                                // “快照 -> 写坏 -> 恢复 -> 读回校验” 这一整套实验路径。
                                // 这里若发现校验失败，直接让整个 submit 失败，避免继续推进脏状态。
                                if !state.restore_verified {
                                    error!(
                                        "[NPU] submit aborted: restore verify failed on core {} owner(task={}, process={}, aspace={:#x}) mismatch_mask={:#x}",
                                        idx,
                                        binding.key.owner.task_id,
                                        binding.key.owner.process_id,
                                        binding.key.owner.address_space_id,
                                        state.restore_mismatch_mask
                                    );
                                    self.shared.clear_active_bindings(max_core);
                                    return Err(RknpuError::TaskError);
                                }
                                */
                            }
                        }
                        self.shared.clear_active_binding(idx);
                        done[idx] = true; //标记这个核心完成了
                    }
                    continue; //继续检查是哪个核心触发中断
                }

                if done[..max_core].iter().filter(|status| !**status).count() == 0 {
                    break;
                }

                for idx in 0..max_core {
                    if done[idx] {
                        continue;
                    }
                    if status[idx] != 0 && (status[idx] & int_mask[idx] == 0) {
                        // 有中断但不是我们期望的 → 硬件错误
                        self.shared.clear_active_bindings(max_core);
                        return Err(RknpuError::TaskError);
                    }
                }

                // Sleep until any interrupt (including NPU) wakes the CPU.
                (wait)();
            }
        } else {
            debug!("[NPU]   waiting (busy-wait mode)...");
            // ┌─────────────────────────────────────────────────────┐
            // │  Busy-wait mode (CPU continuously polls)            │
            // │                                                     │
            // │  NPU runs → CPU continuously reads interrupt status  │
            // │  registers until it sees the expected bits set.     │
            // └─────────────────────────────────────────────────────┘
            panic!("[NPU] busy-poll mode not implemented for multi-core wait");
        };

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::RknpuSubmit;
    use crate::{
        NpuOwnerIds, Rknpu, RknpuConfig, RknpuTask, RknpuType, TaskNpuStateKey, status::NpuCtxState,
    };
    use alloc::vec::Vec;
    use core::ptr::NonNull;

    const FAKE_MMIO_LEN: usize = 0x10000;

    fn build_fake_rknpu() -> (Rknpu, Vec<Vec<u8>>) {
        let mut mmios = vec![vec![0_u8; FAKE_MMIO_LEN]; 3];
        let base_addrs = mmios
            .iter_mut()
            .map(|mmio| NonNull::new(mmio.as_mut_ptr()).unwrap())
            .collect::<Vec<_>>();
        let config = RknpuConfig {
            rknpu_type: RknpuType::Rk3588,
        };

        (Rknpu::new(&base_addrs, config), mmios)
    }

    fn fake_submit(tasks: &mut [RknpuTask], task_number: u32) -> RknpuSubmit {
        let mut submit = RknpuSubmit::default();
        submit.task_obj_addr = tasks.as_mut_ptr() as u64;
        submit.task_number = task_number;
        submit.core_mask = 0x1;
        submit.subcore_task[0].task_start = 0;
        submit.subcore_task[0].task_number = task_number;
        submit
    }

    #[test]
    fn prepare_submit_owner_saves_previous_owner_context_on_switch() {
        let (npu, _mmios) = build_fake_rknpu();
        let owner_a = NpuOwnerIds {
            task_id: 11,
            process_id: 22,
            address_space_id: 33,
        };
        let owner_b = NpuOwnerIds {
            task_id: 44,
            process_id: 55,
            address_space_id: 66,
        };
        let mut tasks = [RknpuTask::default()];
        let mut submit_a = fake_submit(&mut tasks, 1);
        submit_a.flags = 0x1;
        let mut submit_b = fake_submit(&mut tasks, 1);
        submit_b.flags = 0x2;

        npu.prepare_submit_owner(&submit_a, owner_a);
        assert_eq!(
            npu.shared
                .resident_owner()
                .expect("resident owner should be tracked")
                .owner,
            owner_a
        );

        npu.prepare_submit_owner(&submit_b, owner_b);

        let saved_a = npu
            .shared
            .owner_context(owner_a)
            .expect("previous owner context should be saved");
        let resident = npu
            .shared
            .resident_owner()
            .expect("new resident owner should be tracked");

        assert_eq!(saved_a.submit.flags, 0x1);
        assert_eq!(saved_a.submit.task_number, 1);
        assert!(matches!(
            saved_a.driver_status.ctx_state,
            NpuCtxState::Prepared
        ));
        assert_eq!(resident.owner, owner_b);
        assert_eq!(resident.last_submit.flags, 0x2);
    }

    #[test]
    fn prepare_submit_owner_does_not_overwrite_same_owner_context() {
        let (npu, _mmios) = build_fake_rknpu();
        let owner = NpuOwnerIds {
            task_id: 77,
            process_id: 88,
            address_space_id: 99,
        };
        let mut tasks = [RknpuTask::default()];
        let submit_a = fake_submit(&mut tasks, 1);
        let submit_b = fake_submit(&mut tasks, 1);

        npu.shared.ensure_task_state_entry(TaskNpuStateKey {
            owner,
            core_slot: 0,
        });
        let mut saved = crate::NpuContext::default();
        saved.submit.flags = 0x99;
        npu.shared.save_owner_context(owner, saved);

        npu.prepare_submit_owner(&submit_a, owner);
        npu.prepare_submit_owner(&submit_b, owner);

        let saved = npu
            .shared
            .owner_context(owner)
            .expect("same owner context should remain");
        assert_eq!(saved.submit.flags, 0x99);
    }
}
