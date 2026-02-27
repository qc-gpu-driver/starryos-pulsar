//! 高级任务提交类型。
//!
//! # 寄存器命令缓冲区
//!
//! NPU 的 PC 模块通过 DMA 读取 64 位**寄存器命令**（regcmds）缓冲区来工作。
//! 每个 regcmd 由 `npu_op(block, value, reg)` 打包，并告诉 PC：
//! "将 `value` 写入 `block` 上的 `reg`"。
//!
//! ```text
//!  63        48 47        16 15         0
//!  ┌──────────┬─────────────┬───────────┐
//!  │  opcode  │    value    │  register │   = 一个 u64 regcmd
//!  │ (block)  │   (32-bit)  │  (offset) │
//!  └──────────┴─────────────┴───────────┘
//! ```
//!
//! 一个**任务**是一系列连续的 regcmds，为一个神经网络层配置整个
//! CNA → MAC → DPU 流水线。多个任务在单个 DMA 缓冲区（`regcmd_all`）中
//! 背靠背排列。
//!
//! # 提交层次结构
//!
//! ```text
//!  Submit                  (拥有 DMA 缓冲区 + 操作列表)
//!    └─ SubmitRef          (传递给寄存器代码的轻量级视图)
//!         └─ SubmitBase    (核心索引、中断掩码、标志)
//! ```

use alloc::vec::Vec;
use dma_api::{DVec, Direction};

use crate::{JobMode, op::Operation};

pub mod cna;
mod def;
pub mod dpu;
pub mod op;

/// 任务批次的共享参数 — 独立于实际的 regcmd 数据。
#[derive(Debug, Clone)]
pub struct SubmitBase {
    /// 执行模式标志（PC、BLOCK、PINGPONG 等）。
    pub flags: JobMode,
    /// RknpuTask[] 描述符数组的 DMA 地址（用于 ioctl 路径）。
    /// 使用高级 `Submit` API 时为零。
    pub task_base_addr: u32,
    /// 要在哪个 NPU 核心上运行（0、1 或 2）。
    pub core_idx: usize,
    /// 要等待的中断位 — 设置为最后一个任务的完成掩码。
    /// 例如 0x300 表示"等待 ping 和 pong 上的 DPU 完成"。
    pub int_mask: u32,
    /// 启动前要写入 INTERRUPT_CLEAR 的位。
    pub int_clear: u32,
    /// 每个任务的 64 位 regcmd 字数（批次中所有任务相同）。
    pub regcfg_amount: u32,
}

/// 提交批次的轻量级引用 — 不拥有 DMA 缓冲区。
///
/// 这是 [`RknpuCore::submit_pc`] 实际使用来编程 PC 的内容。
#[derive(Debug, Clone)]
pub struct SubmitRef {
    pub base: SubmitBase,
    /// 此批次中有多少任务（层）。
    pub task_number: usize,
    /// regcmd 缓冲区的 DMA 地址（第一个任务命令的开始）。
    pub regcmd_base_addr: u32,
}

/// 拥有 DMA 寄存器命令缓冲区和操作列表。
///
/// 由高级 API 创建（例如用于裸机演示），而不是使用来自用户空间的
/// 原始 `RknpuSubmit` 结构的 ioctl 路径。
///
/// # 构造流程
///
/// ```text
///  1. 调用者构建 Vec<Operation>（例如几个 MatMul 层）。
///  2. Submit::new() 为所有 regcmds 分配一个大的 DMA 缓冲区。
///  3. 每个 Operation::fill_regcmd() 写入其缓冲区切片。
///  4. 刷新缓冲区（confirm_write_all），以便 NPU 可以 DMA 读取它。
///  5. Rknpu::submit() 将其转换为 SubmitRef 并调用 submit_pc()。
/// ```
pub struct Submit {
    pub base: SubmitBase,
    /// 保存所有任务的所有寄存器命令的连续 DMA 缓冲区。
    /// 布局：[task0_regcmds | task1_regcmds | ... | taskN_regcmds]
    pub regcmd_all: DVec<u64>,
    /// 操作对象（每个任务/层一个）。
    pub tasks: Vec<Operation>,
}

impl Submit {
    /// 从操作列表构建提交。
    ///
    /// 这会分配 DMA regcmd 缓冲区，调用每个操作的 `fill_regcmd` 
    /// 来填充其切片，并刷新缓存。
    pub fn new(tasks: Vec<Operation>) -> Self {
        let base = SubmitBase {
            flags: JobMode::PC | JobMode::BLOCK | JobMode::PINGPONG,
            task_base_addr: 0,
            core_idx: 0,
            int_mask: 0x300, // 等待 DPU 完成
            int_clear: 0x1ffff,
            regcfg_amount: tasks[0].reg_amount(),
        };

        // 分配一个大的 DMA 缓冲区：regcfg_amount 字 × 任务数
        let regcmd_all: DVec<u64> = DVec::zeros(
            u32::MAX as _,
            base.regcfg_amount as usize * tasks.len(),
            0x1000,
            Direction::Bidirectional,
        )
        .unwrap();

        assert!(
            regcmd_all.bus_addr() <= u32::MAX as u64,
            "regcmd 基地址超过 u32"
        );

        // 用寄存器命令填充缓冲区的每个任务切片
        let amount = base.regcfg_amount as usize;
        for (i, task) in tasks.iter().enumerate() {
            let regcmd = unsafe {
                core::slice::from_raw_parts_mut(regcmd_all.as_ptr().add(i * amount), amount)
            };
            task.fill_regcmd(regcmd);
        }
        // 刷新 CPU 缓存，以便 NPU 可以 DMA 读取命令
        regcmd_all.confirm_write_all();

        Self {
            base,
            regcmd_all,
            tasks,
        }
    }

    /// 创建用于传递给寄存器层的轻量级引用。
    pub fn as_ref(&self) -> SubmitRef {
        SubmitRef {
            base: self.base.clone(),
            task_number: self.tasks.len(),
            regcmd_base_addr: self.regcmd_all.bus_addr() as _,
        }
    }
}
