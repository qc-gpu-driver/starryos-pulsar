//! 瑞芯微 RK3588 NPU（神经处理单元）的低级驱动。
//!
//! # 架构概述
//!
//! ```text
//!  ┌─────────────────────────────────────────────────────────────────────┐
//!  │                        用户空间（rknn 运行时）                      │
//!  │   加载模型 → 准备张量 → ioctl(SUBMIT) → 读取输出                   │
//!  └────────────────────────────┬────────────────────────────────────────┘
//!                               │ ioctl 边界
//!  ┌────────────────────────────▼────────────────────────────────────────┐
//!  │                      本 crate（`rknpu`）                            │
//!  │                                                                     │
//!  │  lib.rs          Rknpu 结构 — 顶层驱动句柄                          │
//!  │  ioctrl.rs       ioctl 分发：SUBMIT、MEM_CREATE、MEM_DESTROY       │
//!  │  gem.rs          DMA 缓冲区池（分配/释放/同步）                      │
//!  │  job.rs          任务描述符和作业模式标志                            │
//!  │  registers/      通过 rknpu_regs（svd2rust）访问 MMIO 寄存器       │
//!  │  task/           高级操作构建器（matmul、conv 等）                  │
//!  │  osal.rs         OS 无关的类型别名（PhysAddr、DmaAddr 等）          │
//!  │  err.rs          统一错误枚举                                        │
//!  └────────────────────────────┬────────────────────────────────────────┘
//!                               │ MMIO 写入
//!  ┌────────────────────────────▼────────────────────────────────────────┐
//!  │                     NPU 硬件（每个核心）                            │
//!  │                                                                     │
//!  │  PC  ──►  CNA  ──►  MAC  ──►  DPU  ──►  输出 DMA 缓冲区            │
//!  │  命令     加载      计算       后处理    （结果张量）                │
//!  │  获取     特征      (矩阵乘)   (偏置、                              │
//!  │           + 权重              激活、                                │
//!  │                               写入)                                 │
//!  └─────────────────────────────────────────────────────────────────────┘
//! ```
//!
//! # 典型的端到端流程
//!
//! 1. **探测** — 平台代码通过设备树发现 NPU，映射其 MMIO 区域，
//!    并使用基地址调用 [`Rknpu::new()`]。
//! 2. **MEM_CREATE** — 用户空间为输入/输出张量和寄存器命令流
//!    分配 DMA 缓冲区（参见 [`gem::GemPool`]）。
//! 3. **准备** — 用户空间（或 `task/` 模块）填充 `RknpuTask[]` 描述符
//!    和寄存器命令缓冲区，为每个神经网络层对每个流水线阶段
//!    （CNA、MAC、DPU）进行编程。
//! 4. **SUBMIT** — 驱动刷新缓存，按单 core / 单 task dispatch 对 PC 模块进行编程并启动执行
//!    （参见 [`ioctrl::Rknpu::submit_ioctrl_step`]）。
//! 5. **轮询/IRQ** — 驱动等待 PC 的中断状态位发出完成信号，
//!    清除它们，并将控制权返回给用户空间。
//! 6. **读取结果** — 用户空间从 DMA 缓冲区读取输出张量。
//!
//! # Crate 特性
//!
//! - `no_std` 兼容 — 在裸机和自定义 OS 内核上运行。
//! - 通过 `rknpu_regs` svd2rust crate 进行类型安全的寄存器访问。
//! - DMA 缓冲区通过平台 `dma_api` crate 管理。

#![no_std]

extern crate alloc;
#[macro_use]
extern crate log;
use core::{
    ops::{Deref, DerefMut},
    ptr::NonNull,
    sync::atomic::Ordering,
};

mod config;
mod data;
mod err;
mod gem;
mod job;
mod osal;
mod registers;
pub mod status;
mod task;
use alloc::vec::Vec;
pub use config::*;
pub use err::*;
pub use gem::*;
pub use job::*;
pub use osal::*;
use rdif_base::DriverGeneric;
pub use status::*;
pub use task::*;
pub mod ioctrl;
use crate::data::RknpuData;
use crate::registers::RknpuCore;

const VERSION_MAJOR: u32 = 0;
const VERSION_MINOR: u32 = 9;
const VERSION_PATCH: u32 = 8;

const fn version(major: u32, minor: u32, patch: u32) -> u32 {
    major * 10000 + minor * 100 + patch
}

/// 可以通过 [`Rknpu::action()`] 请求的硬件操作。
///
/// 这些镜像了 Linux 内核驱动的 ioctl 操作代码。
/// 大多数是管理/诊断操作而不是计算任务。
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

/// 一个 RKNPU 设备的顶层驱动句柄。
///
/// 拥有所有核心的 MMIO 寄存器窗口、芯片特定的配置数据和 GEM 内存池。
/// 所有驱动操作（提交、操作、内存创建/销毁）都通过此结构进行。
///
/// # 完成等待模式
///
/// 默认情况下，驱动在提交任务后**忙轮询** INTERRUPT_STATUS 寄存器
/// （向后兼容）。调用 [`set_wait_fn`] 切换到**中断驱动**模式：
///
/// ```text
///  默认（无 wait_fn）：   submit → loop { 读取 MMIO 寄存器 } → 完成
///  使用 wait_fn（例如 WFI）：submit → loop { 检查原子变量; wfi() } → IRQ
///                           触发 → 处理程序存储状态 → CPU 唤醒 → 完成
/// ```
pub struct Rknpu {
    /// 每个核心的寄存器访问包装器。`base[0]` 是核心 0，依此类推。
    /// 每个 `RknpuCore` 持有指向其 MMIO 基地址的 `NonNull<u8>` 和
    /// 用于中断驱动完成的共享 `irq_status` 原子变量。
    base: Vec<RknpuCore>,
    /// 静态板/SoC 配置（时钟、电源、IRQ 路由等）。
    #[allow(dead_code)]
    config: RknpuConfig,
    /// 芯片变体数据（RK3588 vs RK3568 等）— 寄存器特性、
    /// 最大任务计数、DMA 掩码、中断位布局。
    data: RknpuData,
    /// NPU 和系统内存之间是否有活动的 IOMMU。
    iommu_enabled: bool,
    /// Current logical IOMMU domain id exposed through the action ioctl.
    ///
    /// StarryOS does not yet wire this to a real domain-switching backend, but
    /// keeping the state here preserves the userspace-visible contract and
    /// mirrors the Linux driver shape.
    iommu_domain_id: i32,
    /// DMA buffer pool shared with userspace (see [`gem`] module).
    pub(crate) gem: GemPool,
    /// 用于中断驱动模式的可选平台提供的等待函数。
    ///
    /// - `None`（默认）→ 直接忙轮询 MMIO 寄存器。
    /// - `Some(wfi)` → 在原子检查之间在 WFI 中休眠；NPU IRQ
    ///   唤醒 CPU，处理程序将状态存储到原子变量，
    ///   提交循环在下一次迭代时看到它。
    wait_fn: Option<fn()>,
}

impl Rknpu {
    /// 从原始 MMIO 基地址创建新的 RKNPU 接口。
    ///
    /// # 安全性
    ///
    /// 调用者必须确保 `base_addr` 是正确映射和对齐的 RKNPU 寄存器文件的
    /// 物理地址，并且在返回结构的生命周期内保持有效。
    pub fn new(base_addrs: &[NonNull<u8>], config: RknpuConfig) -> Self {
        let data = RknpuData::new(config.rknpu_type);

        Self {
            base: base_addrs
                .iter()
                .enumerate()
                .map(|(core_slot, &addr)| unsafe { RknpuCore::new(addr, core_slot as u8) })
                .collect(),
            data,
            config,
            iommu_enabled: false,
            iommu_domain_id: 0,
            gem: GemPool::new(),
            wait_fn: None,
        }
    }

    pub fn open(&mut self) -> Result<(), RknpuError> {
        Ok(())
    }

    /// 通过提供平台休眠函数来启用中断驱动等待。
    ///
    /// 调用此函数后，`submit_one()` 将不再忙轮询 MMIO 寄存器。
    /// 相反，它检查共享的 `irq_status` 原子变量，并在检查之间调用 `f()` 休眠。
    /// NPU 中断唤醒 CPU。
    ///
    /// # 典型用法（AArch64）
    ///
    /// ```ignore
    /// // 注册 NPU IRQ 处理程序后：
    /// npu.set_wait_fn(axcpu::wait_for_irqs);  // 调用 WFI 指令
    /// ```
    ///
    /// # 安全注意事项
    ///
    /// 提供的函数必须在**任何**中断触发时返回（包括 NPU 中断）。
    /// AArch64 上的 `WFI` 满足此要求。
    /// 平台还必须在调用此函数之前通过 [`new_irq_handler`](Rknpu::new_irq_handler)
    /// 注册 NPU IRQ。
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

    fn read_core_u32(&self, core_slot: usize, offset: u16) -> u32 {
        unsafe {
            self.base[core_slot]
                .offset_ptr::<u32>(offset as usize)
                .as_ptr()
                .read_volatile()
        }
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

    pub fn get_rw_amount(
        &self,
        dt_wr: Option<&mut u32>,
        dt_rd: Option<&mut u32>,
        wt_rd: Option<&mut u32>,
    ) -> Result<(), RknpuError> {
        let Some(amount_top) = self.data.amount_top else {
            warn!("Get rw_amount is not supported on this device!");
            return Ok(());
        };

        let amount_scale = self.data.pc_data_amount_scale;
        let amount_core = self.data.amount_core;

        if let Some(dt_wr) = dt_wr {
            *dt_wr = self
                .read_core_u32(0, amount_top.offset_dt_wr)
                .saturating_mul(amount_scale);
            if let Some(amount_core) = amount_core {
                *dt_wr = dt_wr.saturating_add(
                    self.read_core_u32(0, amount_core.offset_dt_wr)
                        .saturating_mul(amount_scale),
                );
            }
        }

        if let Some(dt_rd) = dt_rd {
            *dt_rd = self
                .read_core_u32(0, amount_top.offset_dt_rd)
                .saturating_mul(amount_scale);
            if let Some(amount_core) = amount_core {
                *dt_rd = dt_rd.saturating_add(
                    self.read_core_u32(0, amount_core.offset_dt_rd)
                        .saturating_mul(amount_scale),
                );
            }
        }

        if let Some(wt_rd) = wt_rd {
            *wt_rd = self
                .read_core_u32(0, amount_top.offset_wt_rd)
                .saturating_mul(amount_scale);
            if let Some(amount_core) = amount_core {
                *wt_rd = wt_rd.saturating_add(
                    self.read_core_u32(0, amount_core.offset_wt_rd)
                        .saturating_mul(amount_scale),
                );
            }
        }

        Ok(())
    }

    pub fn get_total_rw_amount(&self) -> Result<u32, RknpuError> {
        let Some(_amount_top) = self.data.amount_top else {
            warn!("Get total_rw_amount is not supported on this device!");
            return Ok(0);
        };

        let mut dt_wr = 0;
        let mut dt_rd = 0;
        let mut wt_rd = 0;
        self.get_rw_amount(Some(&mut dt_wr), Some(&mut dt_rd), Some(&mut wt_rd))?;
        Ok(dt_wr.saturating_add(dt_rd).saturating_add(wt_rd))
    }

    pub fn soft_reset(&mut self) -> Result<(), RknpuError> {
        // Linux performs a full reset-controller dance here. StarryOS does not
        // yet have that platform plumbing in this crate, so the best-effort
        // reset path clears every core's pending IRQ state and published
        // completion shadow.
        for core in &self.base {
            core.clean_interrupts();
            core.drain_pending_interrupts();
            core.irq_status.store(0, Ordering::Release);
        }
        Ok(())
    }

    /// 根据提供的操作请求执行 RKNPU 操作
    ///
    /// 此函数镜像了 Linux 驱动的 rknpu_action 实现，
    /// 为硬件操作提供 Rust 安全接口。
    pub fn action(&mut self, action: RknpuAction, value: u32) -> Result<u32, RknpuError> {
        match action {
            RknpuAction::GetHwVersion => Ok(self.get_hw_version()),
            RknpuAction::GetDrvVersion => Ok(version(VERSION_MAJOR, VERSION_MINOR, VERSION_PATCH)),
            RknpuAction::GetFreq => {
                warn!("GetFreq requires platform clock integration");
                Err(RknpuError::NotSupported)
            }
            RknpuAction::SetFreq => {
                warn!("SetFreq requires platform clock integration");
                let _ = value;
                Err(RknpuError::NotSupported)
            }
            RknpuAction::GetVolt => {
                warn!("GetVolt requires platform regulator integration");
                Err(RknpuError::NotSupported)
            }
            RknpuAction::SetVolt => {
                warn!("SetVolt requires platform regulator integration");
                let _ = value;
                Err(RknpuError::NotSupported)
            }
            RknpuAction::ActReset => {
                self.soft_reset()?;
                Ok(0)
            }
            RknpuAction::GetBwPriority => {
                warn!("GetBwPriority requires a mapped bw_priority register window");
                Err(RknpuError::NotSupported)
            }
            RknpuAction::SetBwPriority => {
                warn!("SetBwPriority requires a mapped bw_priority register window");
                let _ = value;
                Err(RknpuError::NotSupported)
            }
            RknpuAction::GetBwExpect => {
                warn!("GetBwExpect requires a mapped bw_priority register window");
                Err(RknpuError::NotSupported)
            }
            RknpuAction::SetBwExpect => {
                warn!("SetBwExpect requires a mapped bw_priority register window");
                let _ = value;
                Err(RknpuError::NotSupported)
            }
            RknpuAction::GetBwTw => {
                warn!("GetBwTw requires a mapped bw_priority register window");
                Err(RknpuError::NotSupported)
            }
            RknpuAction::SetBwTw => {
                warn!("SetBwTw requires a mapped bw_priority register window");
                let _ = value;
                Err(RknpuError::NotSupported)
            }
            RknpuAction::ActClrTotalRwAmount => {
                self.clear_rw_amount()?;
                Ok(0)
            }
            RknpuAction::GetDtWrAmount => {
                let mut out = 0;
                self.get_rw_amount(Some(&mut out), None, None)?;
                Ok(out)
            }
            RknpuAction::GetDtRdAmount => {
                let mut out = 0;
                self.get_rw_amount(None, Some(&mut out), None)?;
                Ok(out)
            }
            RknpuAction::GetWtRdAmount => {
                let mut out = 0;
                self.get_rw_amount(None, None, Some(&mut out))?;
                Ok(out)
            }
            RknpuAction::GetTotalRwAmount => self.get_total_rw_amount(),
            RknpuAction::GetIommuEn => Ok(if self.iommu_enabled { 1 } else { 0 }),
            RknpuAction::SetProcNice => {
                let nice = i32::from_ne_bytes(value.to_ne_bytes());
                warn!(
                    "SetProcNice({}) is not applicable in the bare-metal driver context",
                    nice
                );
                Ok(0)
            }
            RknpuAction::PowerOn => {
                warn!("PowerOn requires platform PM integration");
                Err(RknpuError::NotSupported)
            }
            RknpuAction::PowerOff => {
                warn!("PowerOff requires platform PM integration");
                Err(RknpuError::NotSupported)
            }
            RknpuAction::GetTotalSramSize => Ok(self.data.nbuf_size as u32),
            RknpuAction::GetFreeSramSize => Ok(self.data.nbuf_size as u32),
            RknpuAction::GetIommuDomainId => Ok(self.iommu_domain_id as u32),
            RknpuAction::SetIommuDomainId => {
                let domain_id = i32::from_ne_bytes(value.to_ne_bytes());
                if !(0..16).contains(&domain_id) {
                    return Err(RknpuError::InvalidParameter);
                }
                self.iommu_domain_id = domain_id;
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

    /// 将预构建的任务批次提交到核心 0（高级 API）。
    ///
    /// 与 ioctl 路径的单 task 步进不同，
    /// 这接受由 `task/` 模块构建的具有适当 DMA 缓冲区的 [`Submit`]。
    pub fn submit(&mut self, job: &mut Submit) -> Result<(), RknpuError> {
        self.base[0].submit(&self.data, self.wait_fn, job)
    }

    /// Harvests every per-core completion currently published by IRQ handlers.
    ///
    /// The scheduler calls this from normal task context. Each returned record
    /// clears the corresponding active per-core dispatch slot.
    pub fn harvest_completed_dispatches(&mut self) -> Vec<CoreCompletion> {
        let mut completed = Vec::new();

        for core_slot in 0..self.base.len().min(NPU_MAX_CORES) {
            let irq_status = self.base[core_slot].irq_status.load(Ordering::Acquire);
            if irq_status == 0 {
                continue;
            }

            self.base[core_slot].irq_status.store(0, Ordering::Release);
            debug!(
                "[NPU] harvest_completed_dispatches core{} irq_status={:#x} observed={:#x}",
                core_slot, irq_status, irq_status
            );

            completed.push(CoreCompletion {
                core_slot: core_slot as u8,
                observed_irq_status: irq_status,
            });
        }

        completed
    }

    /// 读取并清除核心 0 上的挂起中断。
    ///
    /// 返回模糊状态位掩码 — 非零表示自上次调用以来至少有一个
    /// 任务完成（或出错）。
    pub fn handle_interrupt0(&mut self) -> u32 {
        self.base[0].handle_interrupt()
    }

    /// 为一个核心创建轻量级的 `Send + Sync` 中断处理程序。
    ///
    /// 当平台 IRQ 框架需要一个可以从中断上下文调用的对象而不持有
    /// `&mut Rknpu` 时，这很有用。
    pub fn new_irq_handler(&self, core_idx: usize) -> RknpuIrqHandler {
        RknpuIrqHandler(self.base[core_idx].clone())
    }
}

/// 单个 NPU 核心的轻量级中断处理程序。
///
/// 从 [`Rknpu`] 克隆，可以安全地从 IRQ 上下文调用。
/// 仅读取 INTERRUPT_STATUS 并写入 INTERRUPT_CLEAR —
/// 没有内存分配或阻塞。
pub struct RknpuIrqHandler(RknpuCore);

unsafe impl Send for RknpuIrqHandler {}
unsafe impl Sync for RknpuIrqHandler {}

impl RknpuIrqHandler {
    /// 读取并清除挂起的中断。返回模糊状态。
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

#[cfg(test)]
mod tests {
    use super::*;

    fn build_test_npu(core_count: usize) -> Rknpu {
        let mut mmios = (0..core_count)
            .map(|_| vec![0_u8; 0x10000])
            .collect::<Vec<_>>();
        let base_addrs = mmios
            .iter_mut()
            .map(|mmio| NonNull::new(mmio.as_mut_ptr()).unwrap())
            .collect::<Vec<_>>();
        let config = RknpuConfig {
            rknpu_type: RknpuType::Rk3588,
        };
        Rknpu::new(&base_addrs, config)
    }

    #[test]
    fn harvest_completed_dispatches_returns_raw_core_status() {
        let mut npu = build_test_npu(1);
        npu.base[0].irq_status.store(0x300, Ordering::Release);

        let completed = npu.harvest_completed_dispatches();

        assert_eq!(completed.len(), 1);
        assert_eq!(completed[0].core_slot, 0);
        assert_eq!(completed[0].observed_irq_status, 0x300);
        assert_eq!(npu.base[0].irq_status.load(Ordering::Acquire), 0);
    }

    #[test]
    fn harvest_completed_dispatches_collects_multiple_cores() {
        let mut npu = build_test_npu(2);

        npu.base[0].irq_status.store(0x100, Ordering::Release);
        npu.base[1].irq_status.store(0x300, Ordering::Release);

        let completed = npu.harvest_completed_dispatches();

        assert_eq!(completed.len(), 2);
        assert_eq!(completed[0].core_slot, 0);
        assert_eq!(completed[0].observed_irq_status, 0x100);
        assert_eq!(completed[1].core_slot, 1);
        assert_eq!(completed[1].observed_irq_status, 0x300);
        assert_eq!(npu.base[0].irq_status.load(Ordering::Acquire), 0);
        assert_eq!(npu.base[1].irq_status.load(Ordering::Acquire), 0);
    }

    #[test]
    fn action_iommu_domain_round_trip_uses_input_value() {
        let mut npu = build_test_npu(1);

        assert_eq!(npu.action(RknpuAction::GetIommuDomainId, 0).unwrap(), 0);
        assert_eq!(npu.action(RknpuAction::SetIommuDomainId, 7).unwrap(), 0);
        assert_eq!(npu.action(RknpuAction::GetIommuDomainId, 0).unwrap(), 7);
    }

    #[test]
    fn action_rejects_out_of_range_iommu_domain() {
        let mut npu = build_test_npu(1);

        let err = npu.action(RknpuAction::SetIommuDomainId, 16).unwrap_err();

        assert_eq!(err, RknpuError::InvalidParameter);
    }

    #[test]
    fn action_reports_total_sram_from_variant_data() {
        let mut npu = build_test_npu(1);
        npu.data.nbuf_size = 0x4000;

        assert_eq!(
            npu.action(RknpuAction::GetTotalSramSize, 0).unwrap(),
            0x4000
        );
        assert_eq!(npu.action(RknpuAction::GetFreeSramSize, 0).unwrap(), 0x4000);
    }

    #[test]
    fn soft_reset_clears_published_irq_state() {
        let mut npu = build_test_npu(2);
        npu.base[0].irq_status.store(0x100, Ordering::Release);
        npu.base[1].irq_status.store(0x300, Ordering::Release);

        npu.soft_reset().unwrap();

        assert_eq!(npu.base[0].irq_status.load(Ordering::Acquire), 0);
        assert_eq!(npu.base[1].irq_status.load(Ordering::Acquire), 0);
    }
}
