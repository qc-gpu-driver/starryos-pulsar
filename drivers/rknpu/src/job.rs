//! Job and task descriptor definitions for RKNPU hardware submission.
//!
//! # NPU execution model (for beginners)
//!
//! The RK3588 has up to **3 independent NPU cores**.  Each core contains a
//! deep pipeline of functional blocks that process neural-network layers:
//!
//! ```text
//!  ┌─────────────────────────── One NPU Core ───────────────────────────┐
//!  │                                                                    │
//!  │  PC ──► CNA ──► MAC ──► DPU ──► Output buffer                     │
//!  │  │      │       │       │                                          │
//!  │  │      │       │       └─ Post-processing (bias, activation,      │
//!  │  │      │       │          batch-norm, element-wise ops) + write   │
//!  │  │      │       └── Multiply-Accumulate (the actual compute)       │
//!  │  │      └── Convolution Neural Accelerator (load features+weights) │
//!  │  └── Program Counter (fetches & dispatches register commands)      │
//!  │                                                                    │
//!  │  Also: PPU (pooling), DDMA/SDMA (data/system DMA), GLOBAL ctrl    │
//!  └────────────────────────────────────────────────────────────────────┘
//! ```
//!
//! # How a job is submitted (PC mode)
//!
//! 1. **Userspace** (rknn runtime) compiles the neural network into a list of
//!    **tasks**.  Each task is one layer (e.g. conv2d, matmul, relu, etc.).
//!
//! 2. For each task the runtime fills an [`RknpuTask`] descriptor that tells
//!    the hardware:
//!    - Which functional blocks to enable (`enable_mask`)
//!    - Where the register configuration data lives (`regcmd_addr`)
//!    - How many register words to load (`regcfg_amount`)
//!    - Which interrupt to fire on completion (`int_mask`)
//!
//! 3. The driver collects these task descriptors into a DMA buffer and
//!    programs the **PC** (Program Counter) module with:
//!    - Base address of the register command stream
//!    - Number of tasks
//!    - Task DMA base address
//!
//! 4. The PC module autonomously fetches each task's register config via DMA,
//!    programs the CNA/MAC/DPU pipeline, and fires an interrupt when done.
//!
//! 5. The driver polls or waits for the interrupt, checks the status, and
//!    returns results to userspace.

#![allow(dead_code)]

// ── Hardware limits ──────────────────────────────────────────────────────

/// Maximum number of hardware cores supported by the RK3588 NPU IP.
/// Core 0 is always present; cores 1 and 2 are optional in some SKUs.
pub const RKNPU_MAX_CORES: usize = 3;

/// Maximum number of sub-core task groups accepted in a single submit ioctl.
/// Each sub-core task group targets one core and specifies a range of tasks.
pub const RKNPU_MAX_SUBCORE_TASKS: usize = 5;

// ── Core selection bitmasks ──────────────────────────────────────────────

/// Let the driver pick the least-busy core automatically.
pub const RKNPU_CORE_AUTO_MASK: u32 = 0x00;
/// Target core 0 (always available).
pub const RKNPU_CORE0_MASK: u32 = 0x01;
/// Target core 1 (if present).
pub const RKNPU_CORE1_MASK: u32 = 0x02;
/// Target core 2 (if present).
pub const RKNPU_CORE2_MASK: u32 = 0x04;

// ── Job flags (passed from userspace) ────────────────────────────────────

/// Use PC (Program Counter) mode — the hardware command parser fetches
/// register configs automatically.  This is the normal execution path.
pub const RKNPU_JOB_PC: u32 = 1 << 0;
/// Return immediately without waiting for the NPU to finish.
pub const RKNPU_JOB_NONBLOCK: u32 = 1 << 1;
/// Enable ping-pong double-buffering for overlapping DMA and compute.
pub const RKNPU_JOB_PINGPONG: u32 = 1 << 2;
/// Wait on an external sync fence before starting execution.
pub const RKNPU_JOB_FENCE_IN: u32 = 1 << 3;
/// Signal an external sync fence when execution completes.
pub const RKNPU_JOB_FENCE_OUT: u32 = 1 << 4;

/// Hardware task descriptor consumed by the PC module.
///
/// The PC DMA-reads an array of these from the `task_dma_base_addr`.
/// For each entry it loads `regcfg_amount` 64-bit register commands from
/// `regcmd_addr`, programs the pipeline, and waits for `int_mask` bits.
///
/// Layout must match the hardware expectation exactly (`#[repr(C, packed)]`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[repr(C, packed)]
pub struct RknpuTask {
    /// Job-level flags (see `RKNPU_JOB_*` constants).
    pub flags: u32,
    /// Index of this operation within the compiled model graph.
    pub op_idx: u32,
    /// Bitmask of functional blocks to enable for this task
    /// (e.g. CNA | MAC | DPU for a convolution layer).
    pub enable_mask: u32,
    /// Interrupt bits the driver should wait for after this task.
    /// Non-zero bits in the PC INTERRUPT_STATUS register that signal
    /// successful completion of the enabled blocks.
    pub int_mask: u32,
    /// Bits to write to INTERRUPT_CLEAR before starting this task.
    pub int_clear: u32,
    /// Filled in by the driver after completion — actual interrupt status
    /// observed, so userspace can detect partial failures.
    pub int_status: u32,
    /// Number of 64-bit register command words the PC should DMA-read
    /// and apply to the pipeline registers for this task.
    pub regcfg_amount: u32,
    /// Byte offset within the register command buffer where this task's
    /// config starts (used when multiple tasks share one buffer).
    pub regcfg_offset: u32,
    /// DMA address of the register command buffer for this task.
    /// The PC reads `regcfg_amount` entries starting here.
    pub regcmd_addr: u64,
}

/// Sub-core task range — tells the driver which slice of the task array
/// to submit to a specific NPU core.
///
/// In multi-core mode, different ranges can be dispatched to different
/// cores for parallel execution.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
#[repr(C)]
pub struct RknpuSubcoreTask {
    /// First task index in the task array.
    pub task_start: u32,
    /// Number of consecutive tasks starting from `task_start`.
    pub task_number: u32,
}

bitflags::bitflags! {
    /// Internal job submission mode flags (Rust-side representation).
    ///
    /// - `PC`       — use the hardware command parser (normal path)
    /// - `SLAVE`    — legacy software-driven mode (not used)
    /// - `BLOCK`    — wait synchronously for completion
    /// - `NONBLOCK` — return immediately, signal via fence later
    /// - `PINGPONG` — double-buffer tasks for pipelining
    #[repr(C)]
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub struct  JobMode: u32 {
        const SLAVE =  0;
        const PC = 1 << 0;
        const BLOCK = 0 << 1;
        const NONBLOCK = 1 << 1;
        const PINGPONG = 1 << 2;
        const FENCE_IN = 1 << 3;
        const FENCE_OUT = 1 << 4;
    }
}

/// Convert a core index (0, 1, 2) to the corresponding bitmask.
pub const fn core_mask_from_index(index: usize) -> u32 {
    match index {
        0 => RKNPU_CORE0_MASK,
        1 => RKNPU_CORE1_MASK,
        2 => RKNPU_CORE2_MASK,
        _ => 0,
    }
}

/// Count how many cores are selected in a bitmask (population count).
pub const fn core_count_from_mask(mask: u32) -> u32 {
    mask.count_ones()
}
