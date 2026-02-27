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

use core::hint::spin_loop;
use core::sync::atomic::Ordering;
use crate::Vec;

use mbarrier::mb;

use crate::{
    JobMode, Rknpu, RknpuError, RknpuTask, SubmitBase, SubmitRef, registers::rknpu_fuzz_status,
};

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
    // pub fn submit_ioctrl(&mut self, args: &mut RknpuSubmit) -> Result<(), RknpuError> {
    //     self.gem.comfirm_write_all()?;
    //     let mut int_status = 0;

    //     if args.flags & 1 << 1 > 0 {
    //         debug!("Nonblock task");
    //     }
    //     let task_ptr = args.task_obj_addr as *mut RknpuTask;
    //     let mut task_iter = args.task_start as usize;
    //     let task_iter_end = task_iter + args.task_number as usize;
    //     let max_submit_number = self.data.max_submit_number as usize;

    //     while task_iter < task_iter_end {
    //         let task_number = (task_iter_end - task_iter).min(max_submit_number);
    //         let submit_tasks =
    //             unsafe { core::slice::from_raw_parts_mut(task_ptr.add(task_iter), task_number) };

    //         let job = SubmitRef {
    //             base: SubmitBase {
    //                 flags: JobMode::from_bits_retain(args.flags),
    //                 task_base_addr: args.task_base_addr as _,
    //                 core_idx: args.core_mask.trailing_zeros() as usize,
    //                 // core_idx: 0x0,
    //                 int_mask: submit_tasks.last().unwrap().int_mask,
    //                 int_clear: submit_tasks[0].int_mask,
    //                 regcfg_amount: submit_tasks[0].regcfg_amount,
    //             },
    //             task_number,
    //             regcmd_base_addr: submit_tasks[0].regcmd_addr as _,
    //         };
    //         debug!("Submit {task_number} jobs: {job:#x?}");
    //         while self.base[0].handle_interrupt() != 0 {
    //             spin_loop();
    //         }
    //         debug!("Submitting PC job...");
    //         self.base[0].submit_pc(&self.data, &job).unwrap();

    //         // Wait for completion
    //         loop {
    //             let status = self.base[0].pc().interrupt_status.get();
    //             let status = rknpu_fuzz_status(status);

    //             if status & job.base.int_mask > 0 {
    //                 int_status = job.base.int_mask & status;
    //                 break;
    //             }
    //             if status != 0 {
    //                 debug!("Interrupt status changed: {:#x}", status);
    //                 return Err(RknpuError::TaskError);
    //             }
    //         }
    //         self.base[0].pc().clean_interrupts();
    //         debug!("Job completed");
    //         submit_tasks.last_mut().unwrap().int_status = int_status;
    //         task_iter += task_number;
    //     }
    //     self.gem.prepare_read_all()?;

    //     args.task_counter = args.task_number as _;
    //     args.hw_elapse_time = (args.timeout / 2) as _;

    //     Ok(())
    // }
    /// SUBMIT ioctl 的入口点 — 将任务分派到 NPU 核心。
    ///
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
        debug!("[NPU] SUBMIT: {} tasks, flags={:#x}, core_mask={:#x}",
              args.task_number, args.flags, args.core_mask);

        // Step 1: ensure NPU can see CPU-written data
        self.gem.comfirm_write_all()?;

        if args.flags & 1 << 1 > 0 {
            debug!("Nonblock task");
        }

        let task_ptr = args.task_obj_addr as *mut RknpuTask;
        let active_subcore:Vec<&RknpuSubcoreTask> = args.subcore_task.iter().filter(|s| s.task_number > 0).collect();

        let mut task_iter = active_subcore[0].task_start as usize;
        let task_iter_end = task_iter + active_subcore.iter().map(|s| s.task_number as usize).sum::<usize>();
        //warn!("Total tasks to submit: {}, active cores: {}, max batch size: {}",
        //      args.task_number, active_subcore.len(), max_submit_number);
        while task_iter < task_iter_end {
            // Clamp batch size to hardware limit
            let task_batch = active_subcore.len().min(task_iter_end - task_iter); //每个核心1个任务，3个核心就是3个任务
            //warn!("Submitting batch of {} tasks to {} active cores", task_batch, active_subcore.len());
            let submit_tasks =
                unsafe { core::slice::from_raw_parts_mut(task_ptr.add(task_iter), task_batch) };
            let int_mask:Vec<u32> = submit_tasks.iter().map(|task| task.int_mask).collect();
            for idx in 0..active_subcore.len().min(task_batch) {
                self.base[idx].start_execute_one(idx, &self.data, &mut submit_tasks[idx],args)?;
            }
            task_iter += task_batch;
            self.wait_all_npucore(self.wait_fn, int_mask, submit_tasks)?;
        }   
        // Step 2: submit to each core that has tasks
        // for idx in 0..5 {
        //     if args.subcore_task[idx].task_number == 0 {
        //         continue;
        //     }
        //     debug!("Submitting subcore task index: {}", idx);
        //     let submitted_tasks = self.base[idx].submit_one(&self.data, self.wait_fn, idx, args)?;
        //     debug!("[NPU] core {} done: {} tasks executed", idx, submitted_tasks);
        // }

        // Step 3: ensure CPU can see NPU-written results
        self.gem.prepare_read_all()?;

        // Step 4: report back to userspace
        args.task_counter = args.task_number as _;
        args.hw_elapse_time = (args.timeout / 2) as _;

        Ok(())
    }

    /// 等待所有 NPU 核心完成当前任务。
    pub fn wait_all_npucore(&self,normal_wait_fn: Option<fn()>,int_mask:Vec<u32>,submit_tasks: &mut [RknpuTask])->Result<(), RknpuError>{
        let wait_start_idx:usize = 0; //从0开始等待
        let max_core:usize = submit_tasks.len();
        let mut done:[bool; 3] = [false; 3]; //跟踪每个核心是否完成
        
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
                        let status:Vec<u32> = self.base.iter().map(|core| core.irq_status.load(Ordering::Acquire)).collect();
                        let mut int_status:[u32; 3] = [0; 3];
                        for idx in wait_start_idx..max_core {
                            if status[idx] & int_mask[idx] > 0 {
                                 int_status[idx] = int_mask[idx] & status[idx]; //一次最多会有一个核心完成，cpu会处理
                                 self.base[idx].clean_interrupts();
                                 // Reset atomic for the next batch.
                                self.base[idx].irq_status.store(0, Ordering::Release);
                                debug!("[NPU]   batch done: int_status={:#x}", int_status[idx]);
                                submit_tasks[idx].int_status = int_status[idx]; //给对应核心的单任务设置完成状态，用户空间会检查这个状态并继续处理结果
                                done[idx] = true; //标记这个核心完成了
                            }
                            continue; //继续检查是哪个核心触发中断
                        }
                        
                        if done[..max_core].iter().filter(|status|!**status).count() == 0 {
                            break;
                        }

                        for idx in 0..max_core {
                            if done[idx] { continue; }
                            if status[idx] != 0 && (status[idx] & int_mask[idx] == 0) {
                                // 有中断但不是我们期望的 → 硬件错误
                                return Err(RknpuError::TaskError);
                            }
                        }

                        // Sleep until any interrupt (including NPU) wakes the CPU.
                        (wait)();
                    }
            }else {
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
