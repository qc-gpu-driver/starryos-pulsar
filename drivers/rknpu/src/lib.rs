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
//! 4. **SUBMIT** — 驱动刷新缓存，使用命令缓冲区地址和任务计数对 PC 模块
//!    进行编程，并启动执行（参见 [`ioctrl::Rknpu::submit_ioctrl`]）。
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

    /// 根据提供的操作请求执行 RKNPU 操作
    ///
    /// 此函数镜像了 Linux 驱动的 rknpu_action 实现，
    /// 为硬件操作提供 Rust 安全接口。
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

    /// 将预构建的任务批次提交到核心 0（高级 API）。
    ///
    /// 与接受原始 ioctl 结构的 `submit_ioctrl` 不同，
    /// 这接受由 `task/` 模块构建的具有适当 DMA 缓冲区的 [`Submit`]。
    pub fn submit(&mut self, job: &mut Submit) -> Result<(), RknpuError> {
        self.base[0].submit(&self.data, job)
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
