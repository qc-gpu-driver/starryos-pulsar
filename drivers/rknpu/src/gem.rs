//! GEM (Graphics Execution Manager) memory pool for DMA buffer management.
//!
//! # What is GEM?
//!
//! The name comes from the Linux DRM/GPU subsystem where "GEM objects" are
//! GPU-accessible memory buffers identified by integer handles.  We reuse the
//! same concept for the NPU: userspace asks the driver to allocate a chunk of
//! DMA-coherent memory, receives an opaque **handle** (u32), and later passes
//! that handle when submitting tasks so the NPU knows where the data lives.
//!
//! # Role in the NPU workflow
//!
//! ```text
//!  Userspace (rknn runtime)             Kernel / Driver
//!  ────────────────────────             ───────────────
//!  1. ioctl(MEM_CREATE, size)  ──────►  GemPool::create()
//!     ◄── handle=1, dma_addr           allocates DVec, returns handle + bus addr
//!
//!  2. write input tensor to              (CPU writes to obj_addr)
//!     obj_addr returned above
//!
//!  3. ioctl(SUBMIT, task list)  ──────►  GemPool::comfirm_write_all()
//!                                        flush CPU caches → NPU can see the data
//!                                        ... NPU runs ...
//!                                        GemPool::prepare_read_all()
//!                                        invalidate caches → CPU can see results
//!
//!  4. read output tensor from            (CPU reads from obj_addr)
//!     the same buffer
//!
//!  5. ioctl(MEM_DESTROY, handle) ────►  GemPool::destroy()
//! ```
//!
//! # Memory layout
//!
//! Each buffer is a contiguous, page-aligned, DMA-capable region (`DVec<u8>`)
//! that is accessible by **both** the CPU (virtual address) and the NPU
//! (bus / physical address).

use alloc::collections::btree_map::BTreeMap;
use dma_api::{DVec, Direction};

use crate::{
    RknpuError,
    ioctrl::{RknpuMemCreate, RknpuMemSync},
};

/// A pool of DMA buffers keyed by opaque u32 handles.
///
/// Handles are monotonically increasing integers starting from 1.
/// The pool owns every allocation and drops the underlying memory when
/// [`destroy`](GemPool::destroy) is called or the pool itself is dropped.
pub struct GemPool {
    /// handle → DMA buffer mapping.
    pool: BTreeMap<u32, DVec<u8>>,
    /// Next handle to assign (wraps on overflow, unlikely in practice).
    handle_counter: u32,
}

impl GemPool {
    pub const fn new() -> Self {
        GemPool {
            pool: BTreeMap::new(),
            handle_counter: 1,
        }
    }

    /// Allocate a new DMA buffer and return its handle + addresses.
    ///
    /// On success the following fields of `args` are filled in:
    /// - `handle`   — unique identifier for this buffer
    /// - `dma_addr` — bus address the NPU should use to access this memory
    /// - `obj_addr` — CPU virtual address for reading/writing the buffer
    /// - `sram_size`— actual allocated size (may equal requested `size`)
    pub fn create(&mut self, args: &mut RknpuMemCreate) -> Result<(), RknpuError> {
        let data = DVec::zeros(
            u32::MAX as _,
            args.size as _,
            0x1000,
            Direction::Bidirectional,
        )
        .unwrap();

        let handle = self.handle_counter;
        self.handle_counter = self.handle_counter.wrapping_add(1);

        args.handle = handle;
        args.sram_size = data.len() as _;
        args.dma_addr = data.bus_addr();
        args.obj_addr = data.as_ptr() as _;
        warn!("[NPU] MEM_CREATE: handle={}, size={:#x}, dma_addr={:#x}",
              handle, args.size, args.dma_addr);
        self.pool.insert(args.handle, data);
        Ok(())
    }

    /// Look up a buffer by handle and return its (bus_address, byte_length).
    ///
    /// Returns `None` if the handle is invalid (already destroyed or never created).
    pub fn get_phys_addr_and_size(&self, handle: u32) -> Option<(u64, usize)> {
        self.pool
            .get(&handle)
            .map(|dvec| (dvec.bus_addr(), dvec.len()))
    }

    /// Synchronize a buffer region between CPU and device.
    ///
    /// Currently a no-op because our DMA allocator uses coherent memory.
    /// On platforms with non-coherent DMA this would flush/invalidate caches
    /// for the specified byte range.
    pub fn sync(&mut self, _args: &mut RknpuMemSync) {}

    /// Free a DMA buffer.  The handle becomes invalid after this call.
    pub fn destroy(&mut self, handle: u32) {
        warn!("[NPU] MEM_DESTROY: handle={}", handle);
        self.pool.remove(&handle);
    }

    /// Flush CPU caches for **all** buffers so the NPU can read fresh data.
    ///
    /// Must be called **before** submitting a task to the NPU.
    /// This ensures any tensor data written by the CPU is visible to the
    /// NPU's DMA engine.
    pub fn comfirm_write_all(&mut self) -> Result<(), RknpuError> {
        for data in self.pool.values_mut() {
            data.confirm_write_all();
        }
        Ok(())
    }

    /// Invalidate CPU caches for **all** buffers so the CPU can read NPU output.
    ///
    /// Must be called **after** the NPU finishes a task and before userspace
    /// reads the output tensor from the buffer.
    pub fn prepare_read_all(&mut self) -> Result<(), RknpuError> {
        for data in self.pool.values_mut() {
            data.prepare_read_all();
        }
        Ok(())
    }
}

impl Default for GemPool {
    fn default() -> Self {
        Self::new()
    }
}
