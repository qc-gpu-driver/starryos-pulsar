//! Chip-variant configuration data for different Rockchip NPU silicon.
//!
//! # Why this exists
//!
//! Different SoCs (RK3588, RK3568, …) have the same NPU IP but with
//! different parameters: number of cores, DMA address width, PC register
//! encoding quirks, bandwidth priority addresses, etc.
//!
//! [`RknpuData`] captures all these chip-specific constants so the rest of
//! the driver can be written generically.  When a new SoC is added, only
//! a new constructor (like `new_3588()`) is needed.
//!
//! # How it's used
//!
//! - `Rknpu::new()` calls `RknpuData::new(config.rknpu_type)` to get the
//!   correct parameters.
//! - `submit_pc()` reads `pc_data_amount_scale`, `pc_task_number_bits`, etc.
//!   to encode the PC register values correctly for the target chip.
//! - `clear_rw_amount()` uses `amount_top` / `amount_core` offsets.

use core::fmt::Debug;

use crate::{Rknpu, RknpuError, RknpuType};

/// Returns a mask with the lowest `n` bits set.
/// e.g. `dma_bit_mask(40)` → 0xFF_FFFF_FFFF (40-bit address space).
pub(crate) const fn dma_bit_mask(n: u32) -> u64 {
    if n >= 64 {
        u64::MAX
    } else {
        (1u64 << n) - 1u64
    }
}

/// Register offsets for reading hardware R/W byte-count statistics.
///
/// Some NPU variants expose counters that track how many bytes the NPU
/// has read/written — useful for bandwidth profiling.
#[derive(Copy, Clone, Debug, Default)]
pub struct RknpuAmountData {
    /// Register offset to clear all counters.
    pub offset_clr_all: u16,
    /// Register offset to read "data written" counter.
    pub offset_dt_wr: u16,
    /// Register offset to read "data read" counter.
    pub offset_dt_rd: u16,
    /// Register offset to read "weight read" counter.
    pub offset_wt_rd: u16,
}

/// Chip-variant parameters that differ between NPU silicon revisions.
///
/// The register layer (`registers/mod.rs`) and submission code (`ioctrl.rs`)
/// consult these values to correctly encode PC commands.
#[derive(Debug, Clone)]
pub(crate) struct RknpuData {
    /// MMIO address of the bandwidth priority register block.
    pub bw_priority_addr: u32,
    /// Length (in bytes) of the bandwidth priority register block.
    pub bw_priority_length: u32,
    /// Maximum DMA address the NPU can reach (e.g. 40-bit → 1 TB).
    pub dma_mask: u64,
    /// Divider for the PC REGISTER_AMOUNTS field.
    /// RK3588 uses 2 (each "amount" unit = 2 regcmd words).
    pub pc_data_amount_scale: u32,
    /// Bit width of the task-number field inside TASK_CON register.
    /// RK3588 = 12 bits → max 4095 tasks per submission.
    pub pc_task_number_bits: u32,
    /// Bitmask for extracting the task number (e.g. 0xFFF for 12 bits).
    pub pc_task_number_mask: u32,
    /// Register offset where the PC reports task completion status.
    pub pc_task_status_offset: u32,
    /// Non-zero if the PC has DMA control (older chips).
    pub pc_dma_ctrl: u32,
    /// Physical address of NPU-local SRAM buffer (0 if none).
    pub nbuf_phyaddr: u64,
    /// Size (bytes) of NPU-local SRAM buffer.
    pub nbuf_size: u64,
    /// Maximum number of tasks the PC can accept in one batch.
    pub max_submit_number: u64,
    /// Bitmask of available cores (e.g. 0x7 = cores 0, 1, 2).
    pub core_mask: u32,
    /// Static IRQ descriptor table (one entry per core).
    pub irqs: &'static [NpuIrq],
    /// Offsets for top-level R/W amount counters (None if unsupported).
    pub amount_top: Option<RknpuAmountData>,
    /// Offsets for per-core R/W amount counters (None if unsupported).
    pub amount_core: Option<RknpuAmountData>,
    /// Platform-specific state initialization function.
    pub state_init: Option<fn(&mut dyn core::any::Any) -> Result<(), RknpuError>>,
    /// Cache scatter-gather table initialization.
    pub cache_sgt_init: Option<fn(&mut dyn core::any::Any) -> Result<(), RknpuError>>,
}

impl RknpuData {
    /// Select the correct chip-variant parameters.
    pub fn new(ty: RknpuType) -> Self {
        match ty {
            RknpuType::Rk3588 => Self::new_3588(),
        }
    }

    /// RK3588 NPU: 3 cores, 40-bit DMA, 12-bit task numbers.
    fn new_3588() -> Self {
        Self {
            bw_priority_addr: 0x0,
            bw_priority_length: 0x0,
            dma_mask: dma_bit_mask(40),         // 40-bit → 1 TB address space
            pc_data_amount_scale: 2,             // each amount unit = 2 regcmd u64s
            pc_task_number_bits: 12,             // TASK_CON[11:0] = task count
            pc_task_number_mask: 0xfff,
            pc_task_status_offset: 0x3c,         // TASK_STATUS register offset
            pc_dma_ctrl: 0,                      // no legacy DMA control
            irqs: RK3588_IRQS,                   // 3 IRQs, one per core
            nbuf_phyaddr: 0,
            nbuf_size: 0,
            max_submit_number: (1u64 << 12) - 1, // 4095 tasks max
            core_mask: 0x7,                       // cores 0, 1, 2 all present
            amount_top: None,                     // RW counters not yet wired
            amount_core: None,
            state_init: None,
            cache_sgt_init: None,
        }
    }
}

/// Static IRQ table for the RK3588 — one interrupt line per NPU core.
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

/// Interrupt descriptor for one NPU core.
pub struct NpuIrq {
    /// Human-readable name (matches device-tree interrupt name).
    pub name: &'static str,
    /// Handler callback invoked when this core's interrupt fires.
    pub irq_hdl: fn(&mut Rknpu, irq: usize) -> Option<()>,
}

impl Debug for NpuIrq {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("NpuIrq").field("name", &self.name).finish()
    }
}
