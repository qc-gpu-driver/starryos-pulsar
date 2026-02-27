//! NPU 操作抽象 — 每个变体将高级神经网络运算符（matmul、conv2d、pooling 等）
//! 映射到 PC 模块 DMA 读取并应用于 CNA/MAC/DPU 流水线的 64 位寄存器命令序列。
//!
//! # 添加新操作
//!
//! 1. 在 `op/` 下创建新文件（例如 `op/conv2d.rs`）。
//! 2. 实现 [`OperationTrait::fill_regcmd`] — 这会用 `npu_op(block, value, register)` 
//!    条目填充 `&mut [u64]` 切片。
//! 3. 向 [`Operation`] 添加变体并在 `fill_regcmd`/`reg_amount` 中连接它。

use crate::op::matmul::MatMul;

pub mod matmul;

/// NPU 硬件支持的数据精度模式。
///
/// 这些值被写入 CNA、MAC（核心）和 DPU 寄存器配置的 
/// `proc_precision` / `in_precision` / `out_precision` 字段。
/// NPU 流水线会自动在精度之间转换。
#[allow(unused)]
#[derive(Clone, Copy, Debug)]
#[repr(u8)]
pub(crate) enum Precision {
    /// 8 位有符号整数 — 量化推理最常用。
    Int8 = 0x0,
    /// 16 位 IEEE 754 半精度浮点数。
    Float16 = 0x2,
    /// 32 位有符号整数 — 用于累加器输出（例如 matmul i8×i8 → i32）。
    Int32 = 0x4,
    /// 32 位 IEEE 754 单精度浮点数。
    Float32 = 0x5,
}

/// 具有其 DMA 缓冲区（输入、权重、输出）的具体 NPU 操作。
///
/// 目前仅实现了 i8 矩阵乘法。
/// 未来的变体：Conv2d、Pooling、Elementwise 等。
pub enum Operation {
    /// 矩阵乘法：A(m×k, i8) × B(k×n, i8) → C(m×n, i32)。
    MatMulu8(MatMul<i8, i32>),
}

impl Operation {
    /// 此操作需要的 64 位寄存器命令字数。
    ///
    /// PC 模块从 regcmd 缓冲区为每个任务读取恰好这么多条目。
    /// 目前所有操作都使用 112 个字（完整的 CNA + MAC + DPU 寄存器集），
    /// 但对于更简单的操作，这可能会有所不同。
    pub fn reg_amount(&self) -> u32 {
        112
    }

    /// 用此操作的硬件指令填充预分配的寄存器命令缓冲区。
    ///
    /// 每个 `u64` 条目由 [`npu_op(block, value, register)`] 打包，
    /// 并告诉 PC 要写入哪个寄存器以及什么值。
    pub fn fill_regcmd(&self, regcmd: &mut [u64]) {
        match self {
            Operation::MatMulu8(op) => {
                op.fill_regcmd(regcmd);
            }
        }
    }
}

/// 每个 NPU 操作都必须实现的 trait。
///
/// 单个方法 `fill_regcmd` 负责：
/// 1. 从操作参数计算所有 CNA/MAC/DPU 寄存器值。
/// 2. 将它们打包成提供的切片中的 `npu_op()` 格式。
pub trait OperationTrait {
    fn fill_regcmd(&self, regcmd: &mut [u64]);
}
