//! High-level task submission types.
//!
//! # Register command buffer
//!
//! The NPU's PC module works by DMA-reading a buffer of 64-bit **register
//! commands** (regcmds).  Each regcmd is packed by `npu_op(block, value, reg)`
//! and tells the PC: "write `value` to `reg` on `block`".
//!
//! ```text
//!  63        48 47        16 15         0
//!  ┌──────────┬─────────────┬───────────┐
//!  │  opcode  │    value    │  register │   = one u64 regcmd
//!  │ (block)  │   (32-bit)  │  (offset) │
//!  └──────────┴─────────────┴───────────┘
//! ```
//!
//! A **task** is a contiguous run of regcmds that configures the entire
//! CNA → MAC → DPU pipeline for one neural-network layer.  Multiple tasks
//! are laid out back-to-back in a single DMA buffer (`regcmd_all`).
//!
//! # Submission hierarchy
//!
//! ```text
//!  Submit                  (owns the DMA buffer + operation list)
//!    └─ SubmitRef          (lightweight view passed to register code)
//!         └─ SubmitBase    (core index, interrupt masks, flags)
//! ```

use alloc::vec::Vec;
use dma_api::{DVec, Direction};

use crate::{JobMode, op::Operation};

pub mod cna;
mod def;
pub mod dpu;
pub mod op;

/// Shared parameters for a task batch — independent of the actual regcmd data.
#[derive(Debug, Clone)]
pub struct SubmitBase {
    /// Execution mode flags (PC, BLOCK, PINGPONG, etc.).
    pub flags: JobMode,
    /// DMA address of the RknpuTask[] descriptor array (for ioctl path).
    /// Zero when using the high-level `Submit` API.
    pub task_base_addr: u32,
    /// Which NPU core to run on (0, 1, or 2).
    pub core_idx: usize,
    /// Interrupt bits to wait for — set to the last task's completion mask.
    /// e.g. 0x300 means "wait for DPU done on both ping and pong".
    pub int_mask: u32,
    /// Bits to write to INTERRUPT_CLEAR before starting.
    pub int_clear: u32,
    /// Number of 64-bit regcmd words per task (same for all tasks in a batch).
    pub regcfg_amount: u32,
}

/// Lightweight reference to a submit batch — no ownership of DMA buffers.
///
/// This is what [`RknpuCore::submit_pc`] actually consumes to program the PC.
#[derive(Debug, Clone)]
pub struct SubmitRef {
    pub base: SubmitBase,
    /// How many tasks (layers) in this batch.
    pub task_number: usize,
    /// DMA address of the regcmd buffer (start of the first task's commands).
    pub regcmd_base_addr: u32,
}

/// Owns a DMA register-command buffer and the list of operations.
///
/// Created by the high-level API (e.g. for bare-metal demos) as opposed
/// to the ioctl path which uses raw `RknpuSubmit` structs from userspace.
///
/// # Construction flow
///
/// ```text
///  1. Caller builds Vec<Operation> (e.g. several MatMul layers).
///  2. Submit::new() allocates one big DMA buffer for all regcmds.
///  3. Each Operation::fill_regcmd() writes its slice of the buffer.
///  4. The buffer is flushed (confirm_write_all) so the NPU can DMA-read it.
///  5. Rknpu::submit() converts this to a SubmitRef and calls submit_pc().
/// ```
pub struct Submit {
    pub base: SubmitBase,
    /// Contiguous DMA buffer holding all register commands for all tasks.
    /// Layout: [task0_regcmds | task1_regcmds | ... | taskN_regcmds]
    pub regcmd_all: DVec<u64>,
    /// The operation objects (one per task / layer).
    pub tasks: Vec<Operation>,
}

impl Submit {
    /// Build a submission from a list of operations.
    ///
    /// This allocates the DMA regcmd buffer, calls each operation's
    /// `fill_regcmd` to populate its slice, and flushes the cache.
    pub fn new(tasks: Vec<Operation>) -> Self {
        let base = SubmitBase {
            flags: JobMode::PC | JobMode::BLOCK | JobMode::PINGPONG,
            task_base_addr: 0,
            core_idx: 0,
            int_mask: 0x300, // wait for DPU to finish
            int_clear: 0x1ffff,
            regcfg_amount: tasks[0].reg_amount(),
        };

        // Allocate one big DMA buffer: regcfg_amount words × number of tasks
        let regcmd_all: DVec<u64> = DVec::zeros(
            u32::MAX as _,
            base.regcfg_amount as usize * tasks.len(),
            0x1000,
            Direction::Bidirectional,
        )
        .unwrap();

        assert!(
            regcmd_all.bus_addr() <= u32::MAX as u64,
            "regcmd base address exceeds u32"
        );

        // Fill each task's slice of the buffer with register commands
        let amount = base.regcfg_amount as usize;
        for (i, task) in tasks.iter().enumerate() {
            let regcmd = unsafe {
                core::slice::from_raw_parts_mut(regcmd_all.as_ptr().add(i * amount), amount)
            };
            task.fill_regcmd(regcmd);
        }
        // Flush CPU caches so the NPU can DMA-read the commands
        regcmd_all.confirm_write_all();

        Self {
            base,
            regcmd_all,
            tasks,
        }
    }

    /// Create a lightweight reference for passing to the register layer.
    pub fn as_ref(&self) -> SubmitRef {
        SubmitRef {
            base: self.base.clone(),
            task_number: self.tasks.len(),
            regcmd_base_addr: self.regcmd_all.bus_addr() as _,
        }
    }
}
