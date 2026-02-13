//! Error types for RKNPU driver operations.
//!
//! # Overview
//!
//! Every fallible operation in the driver — memory allocation, task submission,
//! interrupt handling, etc. — returns `Result<T, RknpuError>`.  The variants
//! below cover the full spectrum of failure modes, from user mistakes
//! (wrong parameters) to hardware faults (NPU timeout, DMA error).
//!
//! These errors are designed to be **no_std compatible** (no heap allocation)
//! and map closely to the error codes used by the original Linux C driver.

use core::fmt::Display;

/// Unified error type for all RKNPU driver operations.
///
/// # Typical failure scenarios
///
/// | Scenario | Likely variant |
/// |---|---|
/// | Userspace passes a bad ioctl struct | `InvalidParameter` |
/// | DMA buffer allocation fails | `OutOfMemory` |
/// | NPU doesn't finish in time | `Timeout` |
/// | Interrupt status shows error bits | `TaskError` / `HardwareFault` |
/// | GEM handle doesn't exist | `InvalidHandle` |
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RknpuError {
    /// Caller supplied an invalid address, size, core mask, or flags value.
    InvalidParameter,
    /// NPU did not complete a task within the configured deadline.
    /// Could indicate a hang or an excessively large workload.
    Timeout,
    /// DMA buffer or internal data structure allocation failed.
    OutOfMemory,
    /// The requested action is not available on this NPU variant or config.
    NotSupported,
    /// Another task is currently running on the requested core(s).
    DeviceBusy,
    /// The NPU has not been initialized (power-on / clock enable not done).
    DeviceNotReady,
    /// Catch-all for unclassified hardware errors.
    DeviceError,
    /// The NPU reported a fatal error via its interrupt status register
    /// (e.g. AXI bus error, ECC error).
    HardwareFault,
    /// IOMMU translation fault — the NPU tried to access an unmapped address.
    IommuError,
    /// DMA transfer failed (buffer not cache-coherent, address out of range, etc.).
    DmaError,
    /// The interrupt status after task completion did not match the expected
    /// mask, indicating the task produced incorrect results or was aborted.
    TaskError,
    /// GEM (memory pool) operation failed — e.g. double-free or corruption.
    MemoryError,
    /// The GEM object handle passed by userspace does not exist in the pool.
    InvalidHandle,
    /// The resource is temporarily unavailable; caller should retry.
    TryAgain,
    /// A blocking wait was interrupted (e.g. by a signal in Linux).
    Interrupted,
    /// The caller does not have permission for this operation.
    PermissionDenied,
    /// An internal logic error in the driver — should never happen.
    InternalError,
}

impl Display for RknpuError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "{:?}", self)
    }
}

impl core::error::Error for RknpuError {}
