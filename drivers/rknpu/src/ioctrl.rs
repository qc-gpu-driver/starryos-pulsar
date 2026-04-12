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
//!  5. ioctl(SUBMIT, RknpuSubmit)         → submit_ioctrl_step()
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
use crate::{Rknpu, RknpuError, RknpuTask};
use core::mem::size_of;

/// 从用户空间传递的子核心任务范围。
///
/// `RknpuSubmit::subcore_task[5]` 中的每个条目都是其中之一。
/// 它告诉驱动："将 tasks[task_start .. task_start+task_number]
/// 提交到此特定 NPU 核心。"
#[repr(C)]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
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

/// Parameters for destroying one DMA buffer (GEM object).
///
/// Userspace passes back the opaque `handle` received from `MEM_CREATE`.
/// `obj_addr` is kept for ABI compatibility with the Linux-style ioctl
/// contract, but the driver-side lookup is keyed by `handle`.
#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct RknpuMemDestroy {
    /// GEM object handle previously returned by `MEM_CREATE`.
    pub handle: u32,
    /// Reserved padding for 64-bit alignment.
    pub reserved: u32,
    /// CPU virtual address of the memory object (ABI compatibility only).
    pub obj_addr: u64,
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
    /// Dispatches one queued task to one hardware core.
    ///
    /// This is the driver-side execution primitive used by the queue
    /// scheduler. One call does exactly one thing:
    ///
    /// - bind one queued task to one physical core
    /// - program that core with one task descriptor
    /// - return immediately without waiting for completion
    ///
    /// The blocking behavior now lives outside this function in the OS-side
    /// queue scheduler. That layer may sleep the submitter thread, harvest IRQ
    /// completions, and dispatch follow-up tasks later on.
    ///
    /// Legacy userspace may leave `task_base_addr` as zero. The previous submit
    /// path forwarded that zero unchanged, so the queue scheduler preserves the
    /// same behavior instead of rejecting the submit.
    pub fn submit_ioctrl_step(
        &mut self,
        core_slot: usize,
        submit_flags: u32,
        task_total: u32,
        task_dma_base: u64,
        subcore_slot: u8,
        task_index: u32,
        task: &mut RknpuTask,
    ) -> Result<(), RknpuError> {
        if core_slot >= self.base.len() {
            debug!(
                "[NPU] submit_ioctrl_step rejected invalid core_slot={} base_len={} task_index={}",
                core_slot,
                self.base.len(),
                task_index
            );
            return Err(RknpuError::InvalidParameter);
        }

        if task_total == 0 {
            debug!(
                "[NPU] submit_ioctrl_step rejected empty submit core={} task_base_addr={:#x}",
                core_slot, task_dma_base
            );
            return Err(RknpuError::InvalidParameter);
        }

        if task_index >= task_total {
            debug!(
                "[NPU] submit_ioctrl_step rejected task_index={} >= task_number={} core={}",
                task_index, task_total, core_slot
            );
            return Err(RknpuError::InvalidParameter);
        }

        // `task_base_addr` belongs to the legacy DMA task-array contract. The
        // new queue path still preserves the visible field, but it may remain
        // zero while the real task descriptor is supplied through `task`.
        let task_dma_addr = if task_dma_base == 0 {
            0
        } else {
            task_dma_base + u64::from(task_index).saturating_mul(size_of::<RknpuTask>() as u64)
        };

        // Reset completion state before the task enters hardware. The final
        // `int_status` will be written back later by the scheduler harvest path.
        task.int_status = 0;
        let regcmd_addr = task.regcmd_addr;
        let regcfg_amount = task.regcfg_amount;
        let int_mask = task.int_mask;
        debug!(
            "[NPU] submit_ioctrl_step dispatch core={} subcore={} task_index={} task_number={} flags={:#x} task_base_addr={:#x} task_dma_addr={:#x} regcmd_addr={:#x} regcfg_amount={} int_mask={:#x}",
            core_slot,
            subcore_slot,
            task_index,
            task_total,
            submit_flags,
            task_dma_base,
            task_dma_addr,
            regcmd_addr,
            regcfg_amount,
            int_mask
        );

        // Drain stale completion state before rebinding this core. Otherwise an
        // old IRQ could be misattributed to the new dispatch.
        self.base[core_slot].drain_pending_interrupts();

        if let Err(err) = self.base[core_slot].start_execute_one(
            core_slot,
            &self.data,
            task,
            submit_flags,
            task_dma_addr,
        ) {
            debug!(
                "[NPU] submit_ioctrl_step start_execute_one failed core={} task_index={}: {:?}",
                core_slot, task_index, err
            );
            return Err(err);
        }

        debug!(
            "[NPU] submit_ioctrl_step programmed hardware core={} subcore={} task_index={}",
            core_slot, subcore_slot, task_index
        );

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::RknpuSubmit;
    use crate::{Rknpu, RknpuConfig, RknpuError, RknpuTask, RknpuType};
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
        submit.task_base_addr = 0x2000;
        submit.task_number = task_number;
        submit.core_mask = 0x1;
        submit.subcore_task[0].task_start = 0;
        submit.subcore_task[0].task_number = task_number;
        submit
    }

    #[test]
    fn submit_step_rejects_invalid_core_slot() {
        let (mut npu, _mmios) = build_fake_rknpu();
        let mut tasks = [RknpuTask::default()];
        let submit = fake_submit(&mut tasks, 1);

        let err = npu
            .submit_ioctrl_step(
                3,
                submit.flags,
                submit.task_number,
                submit.task_base_addr,
                0,
                0,
                &mut tasks[0],
            )
            .unwrap_err();

        assert_eq!(err, RknpuError::InvalidParameter);
    }

    #[test]
    fn submit_step_dispatches_one_task() {
        let (mut npu, _mmios) = build_fake_rknpu();
        let mut tasks = [RknpuTask {
            int_mask: 0x300,
            ..RknpuTask::default()
        }];
        let submit = fake_submit(&mut tasks, 1);

        npu.submit_ioctrl_step(
            0,
            submit.flags,
            submit.task_number,
            submit.task_base_addr,
            0,
            0,
            &mut tasks[0],
        )
        .unwrap();

        assert_eq!(tasks[0].int_status, 0);
        assert_eq!(
            npu.base[0]
                .irq_status
                .load(core::sync::atomic::Ordering::Acquire),
            0
        );
    }

    #[test]
    fn submit_step_rejects_out_of_range_task_index() {
        let (mut npu, _mmios) = build_fake_rknpu();
        let mut tasks = [RknpuTask {
            int_mask: 0x300,
            ..RknpuTask::default()
        }];
        let submit = fake_submit(&mut tasks, 1);
        let err = npu
            .submit_ioctrl_step(
                0,
                submit.flags,
                submit.task_number,
                submit.task_base_addr,
                0,
                1,
                &mut tasks[0],
            )
            .unwrap_err();

        assert_eq!(err, RknpuError::InvalidParameter);
    }

    #[test]
    fn submit_step_accepts_legacy_zero_task_base_addr() {
        let (mut npu, mmios) = build_fake_rknpu();
        let mut tasks = [RknpuTask {
            int_mask: 0x300,
            ..RknpuTask::default()
        }];
        let mut submit = fake_submit(&mut tasks, 1);
        submit.task_base_addr = 0;

        npu.submit_ioctrl_step(
            0,
            submit.flags,
            submit.task_number,
            submit.task_base_addr,
            0,
            0,
            &mut tasks[0],
        )
        .unwrap();

        let task_dma_reg = u32::from_le_bytes(mmios[0][0x34..0x38].try_into().unwrap());
        assert_eq!(task_dma_reg, 0);
    }
}
