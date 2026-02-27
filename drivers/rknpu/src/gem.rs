//! 用于 DMA 缓冲区管理的 GEM（图形执行管理器）内存池。
//!
//! # 什么是 GEM？
//!
//! 这个名称来自 Linux DRM/GPU 子系统，其中"GEM 对象"是由整数句柄标识的
//! GPU 可访问内存缓冲区。我们为 NPU 重用相同的概念：用户空间要求驱动分配一块
//! DMA 一致性内存，接收一个不透明的**句柄**（u32），然后在提交任务时传递该句柄，
//! 以便 NPU 知道数据在哪里。
//!
//! # 在 NPU 工作流中的角色
//!
//! ```text
//!  用户空间（rknn 运行时）             内核/驱动
//!  ────────────────────────             ───────────────
//!  1. ioctl(MEM_CREATE, size)  ──────►  GemPool::create()
//!     ◄── handle=1, dma_addr           分配 DVec，返回句柄 + 总线地址
//!
//!  2. 将输入张量写入                     （CPU 写入 obj_addr）
//!     上面返回的 obj_addr
//!
//!  3. ioctl(SUBMIT, task list)  ──────►  GemPool::comfirm_write_all()
//!                                        刷新 CPU 缓存 → NPU 可以看到数据
//!                                        ... NPU 运行 ...
//!                                        GemPool::prepare_read_all()
//!                                        使缓存无效 → CPU 可以看到结果
//!
//!  4. 从相同缓冲区读取输出张量            （CPU 从 obj_addr 读取）
//!
//!  5. ioctl(MEM_DESTROY, handle) ────►  GemPool::destroy()
//! ```
//!
//! # 内存布局
//!
//! 每个缓冲区都是一个连续的、页对齐的、支持 DMA 的区域（`DVec<u8>`），
//! 可由 **CPU**（虚拟地址）和 **NPU**（总线/物理地址）访问。

use alloc::collections::btree_map::BTreeMap;
use dma_api::{DVec, Direction};

use crate::{
    RknpuError,
    ioctrl::{RknpuMemCreate, RknpuMemSync},
};

/// 由不透明 u32 句柄索引的 DMA 缓冲区池。
///
/// 句柄是从 1 开始单调递增的整数。
/// 池拥有每个分配，并在调用 [`destroy`](GemPool::destroy) 或池本身被丢弃时
/// 释放底层内存。
pub struct GemPool {
    /// 句柄 → DMA 缓冲区映射。
    pool: BTreeMap<u32, DVec<u8>>,
    /// 要分配的下一个句柄（溢出时回绕，实际上不太可能）。
    handle_counter: u32,
}

impl GemPool {
    pub const fn new() -> Self {
        GemPool {
            pool: BTreeMap::new(),
            handle_counter: 1,
        }
    }

    /// 分配新的 DMA 缓冲区并返回其句柄 + 地址。
    ///
    /// 成功时，`args` 的以下字段将被填充：
    /// - `handle`   — 此缓冲区的唯一标识符
    /// - `dma_addr` — NPU 应用于访问此内存的总线地址
    /// - `obj_addr` — 用于读/写缓冲区的 CPU 虚拟地址
    /// - `sram_size`— 实际分配的大小（可能等于请求的 `size`）
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

    /// 通过句柄查找缓冲区并返回其（总线地址，字节长度）。
    ///
    /// 如果句柄无效（已销毁或从未创建），则返回 `None`。
    pub fn get_phys_addr_and_size(&self, handle: u32) -> Option<(u64, usize)> {
        self.pool
            .get(&handle)
            .map(|dvec| (dvec.bus_addr(), dvec.len()))
    }

    /// 在 CPU 和设备之间同步缓冲区区域。
    ///
    /// 目前是空操作，因为我们的 DMA 分配器使用一致性内存。
    /// 在具有非一致性 DMA 的平台上，这将刷新/使指定字节范围的缓存无效。
    pub fn sync(&mut self, _args: &mut RknpuMemSync) {}

    /// 释放 DMA 缓冲区。调用后句柄变为无效。
    pub fn destroy(&mut self, handle: u32) {
        warn!("[NPU] MEM_DESTROY: handle={}", handle);
        self.pool.remove(&handle);
    }

    /// 刷新**所有**缓冲区的 CPU 缓存，以便 NPU 可以读取新数据。
    ///
    /// 必须在向 NPU 提交任务**之前**调用。
    /// 这确保 CPU 写入的任何张量数据对 NPU 的 DMA 引擎可见。
    pub fn comfirm_write_all(&mut self) -> Result<(), RknpuError> {
        for data in self.pool.values_mut() {
            data.confirm_write_all();
        }
        Ok(())
    }

    /// 使**所有**缓冲区的 CPU 缓存无效，以便 CPU 可以读取 NPU 输出。
    ///
    /// 必须在 NPU 完成任务**之后**、用户空间从缓冲区读取输出张量**之前**调用。
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
