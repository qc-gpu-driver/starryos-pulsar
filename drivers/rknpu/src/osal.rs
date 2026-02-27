//! RKNPU 设备层的操作系统抽象层 (OSAL)。
//!
//! # 为什么需要这个模块
//!
//! RKNPU 驱动需要在不同平台上运行（Linux 内核模块、裸机 RTOS、
//! 我们的自定义 Starry OS 等）。每个平台都有自己分配 DMA 内存、
//! 获取时间戳和处理错误的方式。
//!
//! 本模块定义了**平台无关的类型别名**，使驱动代码的其余部分
//! 永远不依赖于特定的 OS API。移植到新平台时，只需要更改这些
//! 类型背后的具体实现 — 驱动逻辑保持不变。
//!
//! # NPU 上下文
//!
//! NPU 通过 DMA（直接内存访问）与系统内存通信。CPU 在 DMA 可访问的
//! 内存中准备输入张量和寄存器命令缓冲区，然后告诉 NPU 硬件该内存的
//! **物理/总线地址**。下面的类型以平台中立的方式捕获这些地址。

/// CPU 的 MMU 看到的物理地址。
///
/// 当驱动需要映射或转换内存区域时使用。
pub type PhysAddr = u64;

/// NPU 硬件看到的 DMA（总线）地址。
///
/// 分配 DMA 缓冲区后（参见 [`crate::gem::GemPool`]），驱动将此地址
/// 编程到 NPU 寄存器中，以便硬件知道从哪里读取输入数据和写入输出结果。
pub type DmaAddr = u64;

/// 用于分析 NPU 作业执行时间的单调时间戳。
pub type TimeStamp = u64;

/// OSAL 操作的错误类型。
///
/// 这些是在内存分配、设备通信等过程中可能发生的底层 OS 错误。
/// 更高级别的 NPU 错误在 [`crate::err`] 中。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OsalError {
    /// 内存分配失败（DMA 缓冲区、页表等）。
    OutOfMemory,
    /// 调用者传递了无效的地址、大小或对齐方式。
    InvalidParameter,
    /// 阻塞操作（例如等待 NPU 中断）超时。
    TimeoutError,
    /// NPU 硬件报告了不可恢复的错误。
    DeviceError,
    /// 此 NPU 变体不支持所请求的功能。
    NotSupported,
}
