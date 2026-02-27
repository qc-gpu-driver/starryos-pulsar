//! 不同瑞芯微 NPU 芯片的变体配置数据。
//!
//! # 为什么需要这个模块
//!
//! 不同的 SoC（RK3588、RK3568 等）拥有相同的 NPU IP，但参数不同：
//! 核心数量、DMA 地址宽度、PC 寄存器编码特性、带宽优先级地址等。
//!
//! [`RknpuData`] 捕获所有这些芯片特定的常量，使驱动的其余部分可以通用编写。
//! 添加新 SoC 时，只需要一个新的构造函数（如 `new_3588()`）。
//!
//! # 如何使用
//!
//! - `Rknpu::new()` 调用 `RknpuData::new(config.rknpu_type)` 获取正确的参数。
//! - `submit_pc()` 读取 `pc_data_amount_scale`、`pc_task_number_bits` 等，
//!   为目标芯片正确编码 PC 寄存器值。
//! - `clear_rw_amount()` 使用 `amount_top` / `amount_core` 偏移量。

use core::fmt::Debug;

use crate::{Rknpu, RknpuError, RknpuType};

/// 返回最低 `n` 位被设置的掩码。
/// 例如 `dma_bit_mask(40)` → 0xFF_FFFF_FFFF（40 位地址空间）。
pub(crate) const fn dma_bit_mask(n: u32) -> u64 {
    if n >= 64 {
        u64::MAX
    } else {
        (1u64 << n) - 1u64
    }
}

/// 用于读取硬件读/写字节计数统计的寄存器偏移量。
///
/// 某些 NPU 变体公开了跟踪 NPU 读/写字节数的计数器 — 对带宽分析很有用。
#[derive(Copy, Clone, Debug, Default)]
pub struct RknpuAmountData {
    /// 清除所有计数器的寄存器偏移量。
    pub offset_clr_all: u16,
    /// 读取"数据写入"计数器的寄存器偏移量。
    pub offset_dt_wr: u16,
    /// 读取"数据读取"计数器的寄存器偏移量。
    pub offset_dt_rd: u16,
    /// 读取"权重读取"计数器的寄存器偏移量。
    pub offset_wt_rd: u16,
}

/// NPU 芯片版本之间不同的芯片变体参数。
///
/// 寄存器层（`registers/mod.rs`）和提交代码（`ioctrl.rs`）
/// 查询这些值以正确编码 PC 命令。
#[derive(Debug, Clone)]
pub(crate) struct RknpuData {
    /// 带宽优先级寄存器块的 MMIO 地址。
    pub bw_priority_addr: u32,
    /// 带宽优先级寄存器块的长度（字节）。
    pub bw_priority_length: u32,
    /// NPU 可以访问的最大 DMA 地址（例如 40 位 → 1 TB）。
    pub dma_mask: u64,
    /// PC REGISTER_AMOUNTS 字段的除数。
    /// RK3588 使用 2（每个"amount"单位 = 2 个 regcmd 字）。
    pub pc_data_amount_scale: u32,
    /// TASK_CON 寄存器内任务编号字段的位宽。
    /// RK3588 = 12 位 → 每次提交最多 4095 个任务。
    pub pc_task_number_bits: u32,
    /// 提取任务编号的位掩码（例如 12 位为 0xFFF）。
    pub pc_task_number_mask: u32,
    /// PC 报告任务完成状态的寄存器偏移量。
    pub pc_task_status_offset: u32,
    /// 如果 PC 具有 DMA 控制则非零（较旧的芯片）。
    pub pc_dma_ctrl: u32,
    /// NPU 本地 SRAM 缓冲区的物理地址（如果没有则为 0）。
    pub nbuf_phyaddr: u64,
    /// NPU 本地 SRAM 缓冲区的大小（字节）。
    pub nbuf_size: u64,
    /// PC 在一批中可以接受的最大任务数。
    pub max_submit_number: u64,
    /// 可用核心的位掩码（例如 0x7 = 核心 0、1、2）。
    pub core_mask: u32,
    /// 静态 IRQ 描述符表（每个核心一个条目）。
    pub irqs: &'static [NpuIrq],
    /// 顶层读/写量计数器的偏移量（如果不支持则为 None）。
    pub amount_top: Option<RknpuAmountData>,
    /// 每核心读/写量计数器的偏移量（如果不支持则为 None）。
    pub amount_core: Option<RknpuAmountData>,
    /// 平台特定的状态初始化函数。
    pub state_init: Option<fn(&mut dyn core::any::Any) -> Result<(), RknpuError>>,
    /// 缓存散列表初始化。
    pub cache_sgt_init: Option<fn(&mut dyn core::any::Any) -> Result<(), RknpuError>>,
}

impl RknpuData {
    /// 选择正确的芯片变体参数。
    pub fn new(ty: RknpuType) -> Self {
        match ty {
            RknpuType::Rk3588 => Self::new_3588(),
        }
    }

    /// RK3588 NPU：3 个核心、40 位 DMA、12 位任务编号。
    fn new_3588() -> Self {
        Self {
            bw_priority_addr: 0x0,
            bw_priority_length: 0x0,
            dma_mask: dma_bit_mask(40),         // 40 位 → 1 TB 地址空间
            pc_data_amount_scale: 2,             // 每个 amount 单位 = 2 个 regcmd u64
            pc_task_number_bits: 12,             // TASK_CON[11:0] = 任务计数
            pc_task_number_mask: 0xfff,
            pc_task_status_offset: 0x3c,         // TASK_STATUS 寄存器偏移量
            pc_dma_ctrl: 0,                      // 无传统 DMA 控制
            irqs: RK3588_IRQS,                   // 3 个 IRQ，每个核心一个
            nbuf_phyaddr: 0,
            nbuf_size: 0,
            max_submit_number: (1u64 << 12) - 1, // 最多 4095 个任务
            core_mask: 0x7,                       // 核心 0、1、2 全部存在
            amount_top: None,                     // 读写计数器尚未连接
            amount_core: None,
            state_init: None,
            cache_sgt_init: None,
        }
    }
}

/// RK3588 的静态 IRQ 表 — 每个 NPU 核心一条中断线。
const RK3588_IRQS: &[NpuIrq] = &[
    NpuIrq {
        name: "npu0_irq",
        irq_hdl: |_, _| None,
    },
    NpuIrq {
        name: "npu1_irq",
        irq_hdl: |_, _| None,
    },
    NpuIrq {
        name: "npu2_irq",
        irq_hdl: |_, _| None,
    },
];

/// 一个 NPU 核心的中断描述符。
pub struct NpuIrq {
    /// 人类可读的名称（与设备树中断名称匹配）。
    pub name: &'static str,
    /// 当此核心的中断触发时调用的处理程序回调。
    pub irq_hdl: fn(&mut Rknpu, irq: usize) -> Option<()>,
}

impl Debug for NpuIrq {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("NpuIrq").field("name", &self.name).finish()
    }
}
