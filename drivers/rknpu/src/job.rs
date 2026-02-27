//! RKNPU 硬件提交的作业和任务描述符定义。
//!
//! # NPU 执行模型（初学者）
//!
//! RK3588 有多达 **3 个独立的 NPU 核心**。每个核心包含一个处理神经网络层的
//! 深度流水线功能块：
//!
//! ```text
//!  ┌─────────────────────────── 一个 NPU 核心 ───────────────────────────┐
//!  │                                                                    │
//!  │  PC ──► CNA ──► MAC ──► DPU ──► 输出缓冲区                         │
//!  │  │      │       │       │                                          │
//!  │  │      │       │       └─ 后处理（偏置、激活、                     │
//!  │  │      │       │          批归一化、逐元素操作）+ 写入              │
//!  │  │      │       └── 乘累加（实际计算）                              │
//!  │  │      └── 卷积神经加速器（加载特征+权重）                          │
//!  │  └── 程序计数器（获取并分发寄存器命令）                              │
//!  │                                                                    │
//!  │  还有：PPU（池化）、DDMA/SDMA（数据/系统 DMA）、GLOBAL 控制         │
//!  └────────────────────────────────────────────────────────────────────┘
//! ```
//!
//! # 如何提交作业（PC 模式）
//!
//! 1. **用户空间**（rknn 运行时）将神经网络编译成**任务**列表。
//!    每个任务是一层（例如 conv2d、matmul、relu 等）。
//!
//! 2. 对于每个任务，运行时填充一个 [`RknpuTask`] 描述符，告诉硬件：
//!    - 要启用哪些功能块（`enable_mask`）
//!    - 寄存器配置数据在哪里（`regcmd_addr`）
//!    - 要加载多少寄存器字（`regcfg_amount`）
//!    - 完成时触发哪个中断（`int_mask`）
//!
//! 3. 驱动将这些任务描述符收集到 DMA 缓冲区中，并使用以下内容对
//!    **PC**（程序计数器）模块进行编程：
//!    - 寄存器命令流的基地址
//!    - 任务数量
//!    - 任务 DMA 基地址
//!
//! 4. PC 模块通过 DMA 自主获取每个任务的寄存器配置，
//!    对 CNA/MAC/DPU 流水线进行编程，并在完成时触发中断。
//!
//! 5. 驱动轮询或等待中断，检查状态，并将结果返回给用户空间。

#![allow(dead_code)]

// ── 硬件限制 ──────────────────────────────────────────────────────

/// RK3588 NPU IP 支持的最大硬件核心数。
/// 核心 0 始终存在；核心 1 和 2 在某些 SKU 中是可选的。
pub const RKNPU_MAX_CORES: usize = 3;

/// 单个提交 ioctl 中接受的最大子核心任务组数。
/// 每个子核心任务组针对一个核心并指定任务范围。
pub const RKNPU_MAX_SUBCORE_TASKS: usize = 5;

// ── 核心选择位掩码 ──────────────────────────────────────────────

/// 让驱动自动选择最不繁忙的核心。
pub const RKNPU_CORE_AUTO_MASK: u32 = 0x00;
/// 目标核心 0（始终可用）。
pub const RKNPU_CORE0_MASK: u32 = 0x01;
/// 目标核心 1（如果存在）。
pub const RKNPU_CORE1_MASK: u32 = 0x02;
/// 目标核心 2（如果存在）。
pub const RKNPU_CORE2_MASK: u32 = 0x04;

// ── 作业标志（从用户空间传递）────────────────────────────────────

/// 使用 PC（程序计数器）模式 — 硬件命令解析器自动获取寄存器配置。
/// 这是正常的执行路径。
pub const RKNPU_JOB_PC: u32 = 1 << 0;
/// 立即返回而不等待 NPU 完成。
pub const RKNPU_JOB_NONBLOCK: u32 = 1 << 1;
/// 启用乒乓双缓冲以重叠 DMA 和计算。
pub const RKNPU_JOB_PINGPONG: u32 = 1 << 2;
/// 在开始执行之前等待外部同步栅栏。
pub const RKNPU_JOB_FENCE_IN: u32 = 1 << 3;
/// 执行完成时发出外部同步栅栏信号。
pub const RKNPU_JOB_FENCE_OUT: u32 = 1 << 4;

/// PC 模块使用的硬件任务描述符。
///
/// PC 从 `task_dma_base_addr` DMA 读取这些数组。
/// 对于每个条目，它从 `regcmd_addr` 加载 `regcfg_amount` 个 64 位寄存器命令，
/// 对流水线进行编程，并等待 `int_mask` 位。
///
/// 布局必须与硬件期望完全匹配（`#[repr(C, packed)]`）。
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[repr(C, packed)]
pub struct RknpuTask {
    /// 作业级标志（参见 `RKNPU_JOB_*` 常量）。
    pub flags: u32,
    /// 此操作在编译模型图中的索引。
    pub op_idx: u32,
    /// 要为此任务启用的功能块的位掩码
    /// （例如，卷积层的 CNA | MAC | DPU）。
    pub enable_mask: u32,
    /// 此任务后驱动应等待的中断位。
    /// PC INTERRUPT_STATUS 寄存器中的非零位，表示
    /// 已启用块的成功完成。
    pub int_mask: u32,
    /// 在启动此任务之前要写入 INTERRUPT_CLEAR 的位。
    pub int_clear: u32,
    /// 完成后由驱动填充 — 观察到的实际中断状态，
    /// 以便用户空间可以检测部分失败。
    pub int_status: u32,
    /// PC 应 DMA 读取并应用于此任务的流水线寄存器的
    /// 64 位寄存器命令字数。
    pub regcfg_amount: u32,
    /// 此任务配置开始的寄存器命令缓冲区内的字节偏移量
    /// （当多个任务共享一个缓冲区时使用）。
    pub regcfg_offset: u32,
    /// 此任务的寄存器命令缓冲区的 DMA 地址。
    /// PC 从这里开始读取 `regcfg_amount` 个条目。
    pub regcmd_addr: u64,
}

/// 子核心任务范围 — 告诉驱动要将任务数组的哪个切片
/// 提交到特定的 NPU 核心。
///
/// 在多核模式下，可以将不同的范围分派到不同的核心以并行执行。
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
#[repr(C)]
pub struct RknpuSubcoreTask {
    /// 任务数组中的第一个任务索引。
    pub task_start: u32,
    /// 从 `task_start` 开始的连续任务数。
    pub task_number: u32,
}

bitflags::bitflags! {
    /// 内部作业提交模式标志（Rust 端表示）。
    ///
    /// - `PC`       — 使用硬件命令解析器（正常路径）
    /// - `SLAVE`    — 传统软件驱动模式（未使用）
    /// - `BLOCK`    — 同步等待完成
    /// - `NONBLOCK` — 立即返回，稍后通过栅栏发出信号
    /// - `PINGPONG` — 双缓冲任务以进行流水线处理
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

/// 将核心索引（0、1、2）转换为相应的位掩码。
pub const fn core_mask_from_index(index: usize) -> u32 {
    match index {
        0 => RKNPU_CORE0_MASK,
        1 => RKNPU_CORE1_MASK,
        2 => RKNPU_CORE2_MASK,
        _ => 0,
    }
}

/// 计算位掩码中选择了多少个核心（人口计数）。
pub const fn core_count_from_mask(mask: u32) -> u32 {
    mask.count_ones()
}
