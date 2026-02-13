//! I/O control (ioctl) interface for the RKNPU driver.
//!
//! # Overview
//!
//! This module defines the **data structures** that userspace (the rknn runtime
//! library) passes to the kernel driver via ioctl system calls, plus the
//! driver-side **submission logic** that programs the NPU hardware.
//!
//! # Typical ioctl flow (from userspace perspective)
//!
//! ```text
//!  rknn runtime                         Driver (this code)
//!  ────────────                         ──────────────────
//!  1. open("/dev/rknpu")
//!  2. ioctl(MEM_CREATE, RknpuMemCreate)  → GemPool::create()
//!     ◄── handle, dma_addr, obj_addr       allocate DMA buffer
//!  3. memcpy(obj_addr, input_tensor)       CPU writes to buffer
//!  4. fill task descriptors (RknpuTask[])
//!  5. ioctl(SUBMIT, RknpuSubmit)         → submit_ioctrl()
//!     ◄── task_counter, hw_elapse_time     programs PC, waits for IRQ
//!  6. memcpy(output, obj_addr)             CPU reads results
//!  7. ioctl(MEM_DESTROY, handle)         → GemPool::destroy()
//! ```
//!
//! # Multi-core submission
//!
//! The RK3588 has up to 3 NPU cores.  A single `RknpuSubmit` can target
//! multiple cores via the `subcore_task[5]` array — each entry specifies
//! which slice of the task array goes to which core.  The driver iterates
//! over non-empty entries and calls `submit_one()` for each.

use core::hint::spin_loop;
use core::sync::atomic::Ordering;

use mbarrier::mb;

use crate::{
    JobMode, Rknpu, RknpuError, RknpuTask, SubmitBase, SubmitRef, registers::rknpu_fuzz_status,
};

/// Sub-core task range passed from userspace.
///
/// Each entry in `RknpuSubmit::subcore_task[5]` is one of these.
/// It tells the driver: "submit tasks[task_start .. task_start+task_number]
/// to this particular NPU core."
#[repr(C)]
#[derive(Debug, Clone, Default)]
pub struct RknpuSubcoreTask {
    /// Index of the first task in the task array for this core.
    pub task_start: u32,
    /// How many consecutive tasks this core should execute.
    pub task_number: u32,
}

/// Parameters for mapping a GEM buffer into userspace via mmap.
///
/// Userspace calls ioctl(MEM_MAP) with a GEM handle and gets back an
/// `offset` that can be passed to `mmap()` to map the buffer.
#[repr(C)]
#[derive(Debug, Clone, Default)]
pub struct RknpuMemMap {
    /// GEM object handle (obtained from MEM_CREATE).
    pub handle: u32,
    /// Padding for 64-bit alignment.
    pub reserved: u32,
    /// Fake file offset returned by the driver, suitable for mmap().
    pub offset: u64,
}

/// Main task-submission ioctl structure.
///
/// Userspace fills this with task metadata and passes it to ioctl(SUBMIT).
/// The driver uses it to:
/// 1. Locate the task descriptor array in memory (`task_obj_addr`)
/// 2. Program the PC module on each targeted core
/// 3. Wait for completion and fill in `task_counter` / `hw_elapse_time`
///
/// # Field summary
///
/// ```text
///  flags            ──  JobMode bits (PC, NONBLOCK, PINGPONG, …)
///  task_obj_addr    ──  CPU address of RknpuTask[] array
///  task_base_addr   ──  DMA address of RknpuTask[] (what the NPU sees)
///  subcore_task[5]  ──  per-core task ranges
///  core_mask        ──  which cores to use (bitmask)
/// ```
#[repr(C)]
#[derive(Debug, Clone, Default)]
pub struct RknpuSubmit {
    /// Job mode flags — see [`JobMode`] (PC, BLOCK, NONBLOCK, PINGPONG…).
    pub flags: u32,
    /// Maximum time (ms) to wait for completion before returning Timeout.
    pub timeout: u32,
    /// (Legacy) global task start index — superseded by subcore_task[].
    pub task_start: u32,
    /// Total number of tasks across all cores.
    pub task_number: u32,
    /// Filled in by the driver: how many tasks were actually executed.
    pub task_counter: u32,
    /// Scheduling priority hint (lower = higher priority).
    pub priority: i32,
    /// CPU virtual address of the `RknpuTask[]` array.
    /// The driver reads task descriptors from this address.
    pub task_obj_addr: u64,
    /// IOMMU domain id for address translation.
    pub iommu_domain_id: u32,
    /// Reserved for 64-bit alignment.
    pub reserved: u32,
    /// DMA (bus) address of the `RknpuTask[]` array.
    /// This is what gets programmed into the PC's TASK_DMA_BASE_ADDR register.
    pub task_base_addr: u64,
    /// Filled in by the driver: hardware execution time in nanoseconds.
    pub hw_elapse_time: i64,
    /// Bitmask selecting which NPU core(s) to use.
    pub core_mask: u32,
    /// File descriptor for sync fence (inter-process GPU/NPU sync).
    pub fence_fd: i32,
    /// Per-core task ranges.  Entry `i` describes the tasks for the core
    /// at index `i`.  Entries with `task_number == 0` are skipped.
    pub subcore_task: [RknpuSubcoreTask; 5],
}

/// Ioctl structure for allocating a DMA-accessible buffer (GEM object).
///
/// Userspace fills in `size` (and optionally `flags`), calls ioctl(MEM_CREATE),
/// and the driver fills in the remaining fields so userspace knows:
/// - `handle`   — opaque identifier for future ioctl calls
/// - `dma_addr` — bus address the NPU will use to read/write this buffer
/// - `obj_addr` — CPU virtual address for direct read/write
#[repr(C)]
#[derive(Debug, Clone, Default)]
pub struct RknpuMemCreate {
    /// Opaque handle assigned by the driver (output).
    pub handle: u32,
    /// Memory type / cache attribute flags (input).
    pub flags: u32,
    /// Requested allocation size in bytes (input, should be page-aligned).
    pub size: u64,
    /// CPU virtual address of the buffer (output).
    pub obj_addr: u64,
    /// DMA bus address accessible by the NPU (output).
    pub dma_addr: u64,
    /// Actual allocated size, may differ from `size` on SRAM paths (output).
    pub sram_size: u64,
    /// IOMMU domain for address isolation (input).
    pub iommu_domain_id: i32,
    /// Core mask hint (reserved/padding).
    pub core_mask: u32,
}

/// Ioctl structure for flushing / invalidating a DMA buffer region.
///
/// Used to ensure cache coherency between CPU writes and NPU reads
/// (or vice versa) when the platform doesn't have hardware-coherent DMA.
#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct RknpuMemSync {
    /// Direction flags (TO_DEVICE, FROM_DEVICE, etc.).
    pub flags: u32,
    /// Padding for 64-bit alignment.
    pub reserved: u32,
    /// CPU virtual address of the buffer to sync.
    pub obj_addr: u64,
    /// Byte offset from the start of the buffer.
    pub offset: u64,
    /// Number of bytes to synchronize.
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
    /// Entry point for the SUBMIT ioctl — dispatches tasks to NPU cores.
    ///
    /// # Workflow
    ///
    /// 1. **Flush caches** (`comfirm_write_all`) so the NPU can see all
    ///    tensor data that the CPU wrote into GEM buffers.
    /// 2. **Iterate** over the 5 `subcore_task` slots.  Each non-empty
    ///    slot triggers `submit_one()` which programs the corresponding
    ///    NPU core and busy-waits for completion.
    /// 3. **Invalidate caches** (`prepare_read_all`) so the CPU can read
    ///    the output tensors the NPU produced.
    /// 4. Fill in `task_counter` and `hw_elapse_time` for userspace.
    pub fn submit_ioctrl(&mut self, args: &mut RknpuSubmit) -> Result<(), RknpuError> {
        debug!("[NPU] SUBMIT: {} tasks, flags={:#x}, core_mask={:#x}",
              args.task_number, args.flags, args.core_mask);

        // Step 1: ensure NPU can see CPU-written data
        self.gem.comfirm_write_all()?;

        if args.flags & 1 << 1 > 0 {
            debug!("Nonblock task");
        }

        // Step 2: submit to each core that has tasks
        for idx in 0..5 {
            if args.subcore_task[idx].task_number == 0 {
                continue;
            }
            debug!("Submitting subcore task index: {}", idx);
            let submitted_tasks = self.submit_one(idx, args)?;
            debug!("[NPU] core {} done: {} tasks executed", idx, submitted_tasks);
        }

        // Step 3: ensure CPU can see NPU-written results
        self.gem.prepare_read_all()?;

        // Step 4: report back to userspace
        args.task_counter = args.task_number as _;
        args.hw_elapse_time = (args.timeout / 2) as _;

        Ok(())
    }
    /// Submit a batch of tasks to a single NPU core and wait for completion.
    ///
    /// `idx` is both the `subcore_task[]` array index AND the hardware core index.
    ///
    /// # Completion waiting modes
    ///
    /// - **IRQ-driven** (`self.wait_fn` is `Some`):
    ///   The CPU sleeps (WFI) between iterations.  When the NPU finishes, it
    ///   fires an interrupt → the GIC wakes the CPU → the IRQ handler calls
    ///   `handle_interrupt()` which stores the fuzzed status into the shared
    ///   `irq_status` atomic → the loop sees it and exits.
    ///
    /// - **Legacy busy-poll** (`self.wait_fn` is `None`):
    ///   The CPU directly reads the MMIO `INTERRUPT_STATUS` register in a
    ///   tight loop.  Simple but wastes CPU cycles.
    ///
    /// # Batching
    ///
    /// The hardware can only accept `max_submit_number` tasks at a time.
    /// If there are more tasks than that, this function loops, submitting
    /// one batch at a time and waiting for each batch to complete.
    fn submit_one(&mut self, idx: usize, args: &mut RknpuSubmit) -> Result<usize, RknpuError> {
        let task_ptr = args.task_obj_addr as *mut RknpuTask;
        let subcore = &args.subcore_task[idx];

        let mut task_iter = subcore.task_start as usize;
        let task_iter_end = task_iter + subcore.task_number as usize;
        let max_submit_number = self.data.max_submit_number as usize;

        while task_iter < task_iter_end {
            // Clamp batch size to hardware limit
            let task_number = (task_iter_end - task_iter).min(max_submit_number);
            let submit_tasks =
                unsafe { core::slice::from_raw_parts_mut(task_ptr.add(task_iter), task_number) };

            // Build the register-level submission descriptor
            let job = SubmitRef {
                base: SubmitBase {
                    flags: JobMode::from_bits_retain(args.flags),
                    task_base_addr: args.task_base_addr as _,
                    core_idx: idx,
                    // Wait for the LAST task's interrupt (pipeline completion)
                    int_mask: submit_tasks.last().unwrap().int_mask,
                    int_clear: submit_tasks[0].int_mask,
                    regcfg_amount: submit_tasks[0].regcfg_amount,
                },
                task_number,
                regcmd_base_addr: submit_tasks[0].regcmd_addr as _,
            };
            debug!("[NPU]   batch: {} tasks, regcmd@{:#x}, {} regcfg_words, int_mask={:#x}",
                  task_number, job.regcmd_base_addr, job.base.regcfg_amount, job.base.int_mask);
            debug!("Submit {task_number} jobs: {job:#x?}");

            // Drain any leftover interrupts from previous execution
            while self.base[idx].handle_interrupt() != 0 {
                spin_loop();
            }

            // Clear the shared atomic before submitting — so we only see
            // status bits from THIS submission, not leftovers.
            self.base[idx].irq_status.store(0, Ordering::Release);

            // Program PC registers and start execution
            debug!("Submitting PC job...");
            self.base[idx].submit_pc(&self.data, &job).unwrap();

            // ── Wait for completion ──────────────────────────────────────
            let int_status = if let Some(wait) = self.wait_fn {
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
                    let status = self.base[idx].irq_status.load(Ordering::Acquire);
                    if status & job.base.int_mask > 0 {
                        break job.base.int_mask & status;
                    }
                    if status != 0 {
                        debug!("Unexpected IRQ status: {:#x}", status);
                        return Err(RknpuError::TaskError);
                    }
                    // Sleep until any interrupt (including NPU) wakes the CPU.
                    (wait)();
                }
            } else {
                // ┌─────────────────────────────────────────────────────┐
                // │  Legacy busy-poll mode (direct MMIO register read)  │
                // │                                                     │
                // │  CPU spins reading INTERRUPT_STATUS until the       │
                // │  expected bits appear.  Simple but wastes cycles.   │
                // └─────────────────────────────────────────────────────┘
                loop {
                    let status = self.base[idx].pc().interrupt_status().read().bits();
                    let status = rknpu_fuzz_status(status);
                    if status & job.base.int_mask > 0 {
                        break job.base.int_mask & status;
                    }
                    if status != 0 {
                        debug!("Interrupt status changed: {:#x}", status);
                        return Err(RknpuError::TaskError);
                    }
                }
            };

            // Acknowledge the interrupt and advance to next batch.
            // In IRQ mode the handler already cleared the HW interrupt,
            // but we still clear once more to be safe.
            self.base[idx].clean_interrupts();
            // Reset atomic for the next batch.
            self.base[idx].irq_status.store(0, Ordering::Release);
            debug!("[NPU]   batch done: int_status={:#x}", int_status);
            submit_tasks.last_mut().unwrap().int_status = int_status;
            task_iter += task_number;
        }

        Ok(subcore.task_number as usize)
    }
}
