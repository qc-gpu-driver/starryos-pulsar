//! Operating System Abstraction Layer (OSAL) for RKNPU device layer.
//!
//! # Why this exists
//!
//! The RKNPU driver needs to run on different platforms (Linux kernel module,
//! bare-metal RTOS, our custom Starry OS, etc.).  Each platform has its own
//! way of allocating DMA memory, getting timestamps, and handling errors.
//!
//! This module defines **platform-agnostic type aliases** so that the rest of
//! the driver code never depends on a specific OS API.  When porting to a new
//! platform, only the concrete implementations behind these types need to
//! change — the driver logic stays the same.
//!
//! # NPU context
//!
//! The NPU communicates with system memory via DMA (Direct Memory Access).
//! The CPU prepares input tensors and register command buffers in DMA-accessible
//! memory, then tells the NPU hardware the **physical / bus address** of that
//! memory.  The types below capture those addresses in a platform-neutral way.

/// Physical address as seen by the CPU's MMU.
///
/// Used when the driver needs to map or translate memory regions.
pub type PhysAddr = u64;

/// DMA (bus) address as seen by the NPU hardware.
///
/// After allocating a DMA buffer (see [`crate::gem::GemPool`]), the driver
/// programs this address into NPU registers so the hardware knows where to
/// read input data and write output results.
pub type DmaAddr = u64;

/// Monotonic timestamp for profiling NPU job execution time.
pub type TimeStamp = u64;

/// Error types for OSAL operations.
///
/// These are low-level OS errors that can occur during memory allocation,
/// device communication, etc.  Higher-level NPU errors are in [`crate::err`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OsalError {
    /// Failed to allocate memory (DMA buffer, page table, etc.).
    OutOfMemory,
    /// Caller passed an invalid address, size, or alignment.
    InvalidParameter,
    /// A blocking operation (e.g. waiting for NPU interrupt) timed out.
    TimeoutError,
    /// The NPU hardware reported an unrecoverable error.
    DeviceError,
    /// The requested feature is not available on this NPU variant.
    NotSupported,
}
