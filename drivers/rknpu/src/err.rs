//! RKNPU 驱动操作的错误类型。
//!
//! # 概述
//!
//! 驱动中的每个可能失败的操作——内存分配、任务提交、中断处理等——都返回
//! `Result<T, RknpuError>`。下面的变体涵盖了从用户错误（错误参数）到硬件故障
//! （NPU 超时、DMA 错误）的全部失败模式。
//!
//! 这些错误被设计为 **no_std 兼容**（无堆分配），并与原始 Linux C 驱动使用的
//! 错误代码紧密对应。

use core::fmt::Display;

/// 所有 RKNPU 驱动操作的统一错误类型。
///
/// # 典型失败场景
///
/// | 场景 | 可能的变体 |
/// |---|---|
/// | 用户空间传递错误的 ioctl 结构 | `InvalidParameter` |
/// | DMA 缓冲区分配失败 | `OutOfMemory` |
/// | NPU 未在时间内完成 | `Timeout` |
/// | 中断状态显示错误位 | `TaskError` / `HardwareFault` |
/// | GEM 句柄不存在 | `InvalidHandle` |
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RknpuError {
    /// 调用者提供了无效的地址、大小、核心掩码或标志值。
    InvalidParameter,
    /// NPU 未在配置的截止时间内完成任务。
    /// 可能表示挂起或工作负载过大。
    Timeout,
    /// DMA 缓冲区或内部数据结构分配失败。
    OutOfMemory,
    /// 此 NPU 变体或配置不支持所请求的操作。
    NotSupported,
    /// 另一个任务当前正在请求的核心上运行。
    DeviceBusy,
    /// NPU 尚未初始化（未完成上电/时钟使能）。
    DeviceNotReady,
    /// 未分类硬件错误的通用类型。
    DeviceError,
    /// NPU 通过其中断状态寄存器报告了致命错误
    /// （例如 AXI 总线错误、ECC 错误）。
    HardwareFault,
    /// IOMMU 转换故障 — NPU 尝试访问未映射的地址。
    IommuError,
    /// DMA 传输失败（缓冲区不是缓存一致的、地址超出范围等）。
    DmaError,
    /// 任务完成后的中断状态与预期掩码不匹配，
    /// 表示任务产生了错误结果或被中止。
    TaskError,
    /// GEM（内存池）操作失败 — 例如双重释放或损坏。
    MemoryError,
    /// 用户空间传递的 GEM 对象句柄在池中不存在。
    InvalidHandle,
    /// 资源暂时不可用；调用者应重试。
    TryAgain,
    /// 阻塞等待被中断（例如在 Linux 中被信号中断）。
    Interrupted,
    /// 调用者没有此操作的权限。
    PermissionDenied,
    /// 驱动程序中的内部逻辑错误 — 不应发生。
    InternalError,
}

impl Display for RknpuError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "{:?}", self)
    }
}

impl core::error::Error for RknpuError {}
