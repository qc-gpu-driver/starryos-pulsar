//! Low-level driver for the Rockchip RK3588 NPU (Neural Processing Unit).
//!
//! # Architecture overview
//!
//! ```text
//!  ┌─────────────────────────────────────────────────────────────────────┐
//!  │                        Userspace (rknn runtime)                     │
//!  │   load model → prepare tensors → ioctl(SUBMIT) → read output       │
//!  └────────────────────────────┬────────────────────────────────────────┘
//!                               │ ioctl boundary
//!  ┌────────────────────────────▼────────────────────────────────────────┐
//!  │                      This crate (`rknpu`)                           │
//!  │                                                                     │
//!  │  lib.rs          Rknpu struct — top-level driver handle             │
//!  │  ioctrl.rs       ioctl dispatch: SUBMIT, MEM_CREATE, MEM_DESTROY   │
//!  │  gem.rs          DMA buffer pool (allocate / free / sync)           │
//!  │  job.rs          Task descriptors & job mode flags                  │
//!  │  registers/      MMIO register access via rknpu_regs (svd2rust)    │
//!  │  task/           High-level operation builders (matmul, conv, etc.) │
//!  │  osal.rs         OS-agnostic type aliases (PhysAddr, DmaAddr, …)   │
//!  │  err.rs          Unified error enum                                 │
//!  └────────────────────────────┬────────────────────────────────────────┘
//!                               │ MMIO writes
//!  ┌────────────────────────────▼────────────────────────────────────────┐
//!  │                     NPU Hardware (per core)                         │
//!  │                                                                     │
//!  │  PC  ──►  CNA  ──►  MAC  ──►  DPU  ──►  output DMA buffer          │
//!  │  cmd      load      compute    post-     (result tensor)            │
//!  │  fetch    features   (matmul)  process                              │
//!  │           + weights            (bias,                               │
//!  │                                activ,                               │
//!  │                                write)                               │
//!  └─────────────────────────────────────────────────────────────────────┘
//! ```
//!
//! # Typical end-to-end flow
//!
//! 1. **Probe** — platform code discovers the NPU via device-tree, maps its
//!    MMIO region, and calls [`Rknpu::new()`] with the base address(es).
//! 2. **MEM_CREATE** — userspace allocates DMA buffers for input/output tensors
//!    and register command streams (see [`gem::GemPool`]).
//! 3. **Prepare** — userspace (or the `task/` module) fills `RknpuTask[]`
//!    descriptors and a register command buffer that programs every pipeline
//!    stage (CNA, MAC, DPU) for each neural-network layer.
//! 4. **SUBMIT** — the driver flushes caches, programs the PC module with the
//!    command buffer address and task count, and kicks off execution
//!    (see [`ioctrl::Rknpu::submit_ioctrl`]).
//! 5. **Poll / IRQ** — the driver waits for the PC's interrupt status bits to
//!    signal completion, clears them, and returns control to userspace.
//! 6. **Read results** — userspace reads the output tensor from the DMA buffer.
//!
//! # Crate features
//!
//! - `no_std` compatible — runs on bare-metal and custom OS kernels.
//! - Register access is type-safe via the `rknpu_regs` svd2rust crate.
//! - DMA buffers are managed through the platform `dma_api` crate.

#![no_std]

extern crate alloc;
#[macro_use]
extern crate log;
use core::{
    ops::{Deref, DerefMut},
    ptr::NonNull,
};

mod config;
mod data;
mod err;
mod registers;
mod gem;
mod job;
mod osal;
mod task;
use alloc::vec::Vec;
pub use config::*;
pub use err::*;
pub use gem::*;
pub use job::*;
pub use osal::*;
use rdif_base::DriverGeneric;
pub use task::*;
pub mod ioctrl;
use crate::registers::RknpuCore;
use crate::{data::RknpuData};

const VERSION_MAJOR: u32 = 0;
const VERSION_MINOR: u32 = 9;
const VERSION_PATCH: u32 = 8;

const fn version(major: u32, minor: u32, patch: u32) -> u32 {
    major * 10000 + minor * 100 + patch
}

/// Hardware actions that can be requested via [`Rknpu::action()`].
///
/// These mirror the ioctl action codes from the Linux kernel driver.
/// Most are management / diagnostic operations rather than compute tasks.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum RknpuAction {
    GetHwVersion = 0,
    GetDrvVersion = 1,
    GetFreq = 2,
    SetFreq = 3,
    GetVolt = 4,
    SetVolt = 5,
    ActReset = 6,
    GetBwPriority = 7,
    SetBwPriority = 8,
    GetBwExpect = 9,
    SetBwExpect = 10,
    GetBwTw = 11,
    SetBwTw = 12,
    ActClrTotalRwAmount = 13,
    GetDtWrAmount = 14,
    GetDtRdAmount = 15,
    GetWtRdAmount = 16,
    GetTotalRwAmount = 17,
    GetIommuEn = 18,
    SetProcNice = 19,
    PowerOn = 20,
    PowerOff = 21,
    GetTotalSramSize = 22,
    GetFreeSramSize = 23,
    GetIommuDomainId = 24,
    SetIommuDomainId = 25,
}

/// Top-level driver handle for one RKNPU device.
///
/// Owns the MMIO register windows for all cores, the chip-specific
/// configuration data, and the GEM memory pool.  All driver operations
/// (submit, action, mem create/destroy) go through this struct.
///
/// # Completion waiting modes
///
/// By default the driver **busy-polls** the INTERRUPT_STATUS register after
/// submitting a task (backward compatible).  Call [`set_wait_fn`] to switch
/// to **interrupt-driven** mode:
///
/// ```text
///  Default (no wait_fn):   submit → loop { read MMIO register } → done
///  With wait_fn (e.g. WFI): submit → loop { check atomic; wfi() } → IRQ
///                           fires → handler stores status → CPU wakes → done
/// ```
pub struct Rknpu {
    /// Per-core register access wrappers.  `base[0]` is core 0, etc.
    /// Each `RknpuCore` holds a `NonNull<u8>` to its MMIO base and a
    /// shared `irq_status` atomic for interrupt-driven completion.
    base: Vec<RknpuCore>,
    /// Static board/SoC configuration (clock, power, IRQ routing, etc.).
    #[allow(dead_code)]
    config: RknpuConfig,
    /// Chip-variant data (RK3588 vs RK3568 etc.) — register quirks,
    /// max task counts, DMA masks, interrupt bit layouts.
    data: RknpuData,
    /// Whether an IOMMU is active between the NPU and system memory.
    iommu_enabled: bool,
    /// DMA buffer pool shared with userspace (see [`gem`] module).
    pub(crate) gem: GemPool,
    /// Optional platform-provided wait function for interrupt-driven mode.
    ///
    /// - `None` (default) → busy-poll MMIO registers directly.
    /// - `Some(wfi)` → sleep in WFI between atomic checks; the NPU IRQ
    ///   wakes the CPU, the handler stores status to the atomic, and the
    ///   submit loop sees it on the next iteration.
    wait_fn: Option<fn()>,
}

impl Rknpu {
    /// Creates a new RKNPU interface from a raw MMIO base address.
    ///
    /// # Safety
    ///
    /// The caller must ensure that `base_addr` is the correctly mapped and
    /// aligned physical address of the RKNPU register file and that it remains
    /// valid for the lifetime of the returned structure.
    pub fn new(base_addrs: &[NonNull<u8>], config: RknpuConfig) -> Self {
        let data = RknpuData::new(config.rknpu_type);

        Self {
            base: base_addrs
                .iter()
                .map(|&addr| unsafe { RknpuCore::new(addr) })
                .collect(),
            data,
            config,
            iommu_enabled: false,
            gem: GemPool::new(),
            wait_fn: None,
        }
    }

    pub fn open(&mut self) -> Result<(), RknpuError> {
        Ok(())
    }

    /// Enable interrupt-driven waiting by providing a platform sleep function.
    ///
    /// After calling this, `submit_one()` will no longer busy-poll the MMIO
    /// register.  Instead it checks the shared `irq_status` atomic and calls
    /// `f()` to sleep between checks.  The NPU interrupt wakes the CPU.
    ///
    /// # Typical usage (AArch64)
    ///
    /// ```ignore
    /// // After registering the NPU IRQ handler:
    /// npu.set_wait_fn(axcpu::wait_for_irqs);  // calls WFI instruction
    /// ```
    ///
    /// # Safety note
    ///
    /// The provided function must return when **any** interrupt fires
    /// (including the NPU interrupt).  `WFI` on AArch64 satisfies this.
    /// The platform must also register the NPU IRQ via
    /// [`new_irq_handler`](Rknpu::new_irq_handler) before calling this.
    pub fn set_wait_fn(&mut self, f: fn()) {
        warn!("[NPU] Interrupt-driven mode enabled (WFI)");
        self.wait_fn = Some(f);
    }

    #[allow(dead_code)]
    fn dma_bit_mask(&self) -> u64 {
        self.data.dma_mask
    }

    pub fn get_hw_version(&self) -> u32 {
        self.base[0].version()
    }

    pub fn clear_rw_amount(&mut self) -> Result<(), RknpuError> {
        let Some(amount_top) = self.data.amount_top else {
            warn!("RKNPU does not support read/write amount statistics");
            return Ok(());
        };

        if self.data.pc_dma_ctrl > 0 {
            let pc_data_addr = self.base[0].pc().base_address().read().bits();
            unsafe {
                self.base[0]
                    .offset_ptr::<u32>(pc_data_addr as usize)
                    .write_volatile(1);
                self.base[0]
                    .offset_ptr::<u32>(amount_top.offset_clr_all as usize)
                    .write_volatile(0x80000101);
                self.base[0]
                    .offset_ptr::<u32>(amount_top.offset_clr_all as usize)
                    .write_volatile(0x00000101);
                if let Some(amount_core) = self.data.amount_core {
                    self.base[0]
                        .offset_ptr::<u32>(amount_core.offset_clr_all as usize)
                        .write_volatile(0x80000101);
                    self.base[0]
                        .offset_ptr::<u32>(amount_core.offset_clr_all as usize)
                        .write_volatile(0x00000101);
                }
            };
        } else {
            unsafe {
                self.base[0]
                    .offset_ptr::<u32>(amount_top.offset_clr_all as usize)
                    .write_volatile(0x80000101);
                self.base[0]
                    .offset_ptr::<u32>(amount_top.offset_clr_all as usize)
                    .write_volatile(0x00000101);
                if let Some(amount_core) = self.data.amount_core {
                    self.base[0]
                        .offset_ptr::<u32>(amount_core.offset_clr_all as usize)
                        .write_volatile(0x80000101);
                    self.base[0]
                        .offset_ptr::<u32>(amount_core.offset_clr_all as usize)
                        .write_volatile(0x00000101);
                }
            }
        }

        Ok(())
    }

    /// Execute an RKNPU action based on the provided action request
    ///
    /// This function mirrors the Linux driver's rknpu_action implementation,
    /// providing a Rust-safe interface for hardware operations.
    pub fn action(&mut self, action: RknpuAction) -> Result<u32, RknpuError> {
        match action {
            RknpuAction::GetHwVersion => {
                let val = self.get_hw_version();
                Ok(val)
            }
            RknpuAction::GetDrvVersion => Ok(version(VERSION_MAJOR, VERSION_MINOR, VERSION_PATCH)),
            RknpuAction::GetFreq => {
                // TODO FPGA频率获取
                Ok(0)
            }
            RknpuAction::SetFreq => {
                // 频率设置 - 需要时钟管理
                Ok(0)
            }
            RknpuAction::GetVolt => {
                // TODO FPGA电压获取
                Ok(0)
            }
            RknpuAction::SetVolt => Err(RknpuError::InternalError),
            RknpuAction::ActReset => {
                // TODO FPGA复位操作
                Ok(0)
            }
            RknpuAction::GetBwPriority => {
                // 带宽优先级获取
                Err(RknpuError::InternalError)
            }
            RknpuAction::SetBwPriority => {
                // 带宽优先级设置
                log::warn!("SetBwPriority operation not yet implemented");
                Err(RknpuError::InternalError)
            }
            RknpuAction::GetBwExpect => {
                // 带宽期望值获取
                Err(RknpuError::InternalError)
            }
            RknpuAction::SetBwExpect => {
                // 带宽期望值设置
                log::warn!("SetBwExpect operation not yet implemented");
                Err(RknpuError::InternalError)
            }
            RknpuAction::GetBwTw => {
                // 带宽时间窗口获取
                Err(RknpuError::InternalError)
            }
            RknpuAction::SetBwTw => {
                // 带宽时间窗口设置
                Err(RknpuError::InternalError)
            }
            RknpuAction::ActClrTotalRwAmount => {
                // 清除读写总量统计
                self.clear_rw_amount()?;
                Ok(0)
            }
            RknpuAction::GetDtWrAmount => {
                // 获取设备写数据量
                warn!("Get rw_amount is not supported on this device!");
                Ok(0)
            }
            RknpuAction::GetDtRdAmount => {
                // 获取设备读数据量
                warn!("Get rw_amount is not supported on this device!");
                Ok(0)
            }
            RknpuAction::GetWtRdAmount => {
                // 获取等待读数据量
                warn!("Get rw_amount is not supported on this device!");
                Ok(0)
            }
            RknpuAction::GetTotalRwAmount => {
                // 获取总读写数据量
                warn!("Get rw_amount is not supported on this device!");
                Ok(0)
            }
            RknpuAction::GetIommuEn => {
                // 获取IOMMU启用状态
                Ok(if self.iommu_enabled { 1 } else { 0 })
            }
            RknpuAction::SetProcNice => {
                // 设置进程优先级 - 在内核空间不适用
                log::warn!("SetProcNice operation not applicable in bare metal context");
                Ok(0)
            }
            RknpuAction::PowerOn => {
                // 电源开启
                log::warn!("PowerOn operation not yet implemented");
                Ok(0)
            }
            RknpuAction::PowerOff => {
                // 电源关闭
                log::warn!("PowerOff operation not yet implemented");
                Ok(0)
            }
            RknpuAction::GetTotalSramSize => {
                // 获取总SRAM大小
                Ok(0)
            }
            RknpuAction::GetFreeSramSize => Ok(self.data.nbuf_size as u32),
            RknpuAction::GetIommuDomainId => {
                // 获取IOMMU域ID - 需要IOMMU管理
                log::warn!("GetIommuDomainId operation not yet implemented");
                Ok(0)
            }
            RknpuAction::SetIommuDomainId => {
                // 设置IOMMU域ID - 需要IOMMU管理
                // log::warn!("SetIommuDomainId operation not yet implemented");
                Ok(0)
            }
        }
    }

    /// Convenience method to check IOMMU status using action interface
    pub fn is_iommu_enabled(&self) -> bool {
        self.iommu_enabled
    }

    /// Enable or disable IOMMU
    pub fn set_iommu_enabled(&mut self, enabled: bool) {
        self.iommu_enabled = enabled;
    }

    // /// Commit a prepared job descriptor to the hardware command parser.
    // pub fn commit_job(&mut self, job: &mut RknpuJob) -> Result<(), RknpuError> {
    //     self.job_commit(job)
    // }

    // fn job_commit(&mut self, job: &mut RknpuJob) -> Result<(), RknpuError> {
    //     const CORE0_1_MASK: u32 = RKNPU_CORE0_MASK | RKNPU_CORE1_MASK;
    //     const CORE0_1_2_MASK: u32 = RKNPU_CORE0_MASK | RKNPU_CORE1_MASK | RKNPU_CORE2_MASK;

    //     match job.args.core_mask {
    //         RKNPU_CORE0_MASK => self.sub_core_submit(job, 0)?,
    //         RKNPU_CORE1_MASK => self.sub_core_submit(job, 1)?,
    //         RKNPU_CORE2_MASK => self.sub_core_submit(job, 2)?,
    //         CORE0_1_MASK => {
    //             self.sub_core_submit(job, 0)?;
    //             self.sub_core_submit(job, 1)?;
    //         }
    //         CORE0_1_2_MASK => {
    //             self.sub_core_submit(job, 0)?;
    //             self.sub_core_submit(job, 1)?;
    //             self.sub_core_submit(job, 2)?;
    //         }
    //         _ => {
    //             error!("Invalid core mask: 0x{:x}", job.args.core_mask);
    //         }
    //     }

    //     Ok(())
    // }

    /// Submit a pre-built task batch to core 0 (high-level API).
    ///
    /// Unlike `submit_ioctrl` which takes raw ioctl structs, this accepts
    /// a [`Submit`] built by the `task/` module with proper DMA buffers.
    pub fn submit(&mut self, job: &mut Submit) -> Result<(), RknpuError> {
        self.base[0].submit(&self.data, job)
    }

    /// Read and clear pending interrupts on core 0.
    ///
    /// Returns a fuzzed status bitmask — non-zero means at least one
    /// task completed (or errored) since the last call.
    pub fn handle_interrupt0(&mut self) -> u32 {
        self.base[0].handle_interrupt()
    }

    /// Create a lightweight, `Send + Sync` interrupt handler for one core.
    ///
    /// This is useful when the platform IRQ framework needs an object it
    /// can call from interrupt context without holding `&mut Rknpu`.
    pub fn new_irq_handler(&self, core_idx: usize) -> RknpuIrqHandler {
        RknpuIrqHandler(self.base[core_idx].clone())
    }
}

/// Lightweight interrupt handler for a single NPU core.
///
/// Cloned from [`Rknpu`] and safe to call from IRQ context.
/// Only reads INTERRUPT_STATUS and writes INTERRUPT_CLEAR — no
/// memory allocation or blocking.
pub struct RknpuIrqHandler(RknpuCore);

unsafe impl Send for RknpuIrqHandler {}
unsafe impl Sync for RknpuIrqHandler {}

impl RknpuIrqHandler {
    /// Read and clear pending interrupts.  Returns fuzzed status.
    pub fn handle(&self) -> u32 {
        self.0.handle_interrupt()
    }
}

impl DriverGeneric for Rknpu {
    fn open(&mut self) -> Result<(), rdif_base::KError> {
        Self::open(self).map_err(|_| rdif_base::KError::Unknown("open fail"))
    }

    fn close(&mut self) -> Result<(), rdif_base::KError> {
        Ok(())
    }
}

impl Deref for Rknpu {
    type Target = GemPool;

    fn deref(&self) -> &Self::Target {
        &self.gem
    }
}

impl DerefMut for Rknpu {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.gem
    }
}
