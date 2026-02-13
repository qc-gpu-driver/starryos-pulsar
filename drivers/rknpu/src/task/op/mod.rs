//! NPU operation abstraction — each variant maps a high-level NN operator
//! (matmul, conv2d, pooling, …) to a sequence of 64-bit register commands
//! that the PC module DMA-reads and applies to the CNA/MAC/DPU pipeline.
//!
//! # Adding a new operation
//!
//! 1. Create a new file under `op/` (e.g. `op/conv2d.rs`).
//! 2. Implement [`OperationTrait::fill_regcmd`] — this fills a `&mut [u64]`
//!    slice with `npu_op(block, value, register)` entries.
//! 3. Add a variant to [`Operation`] and wire it up in `fill_regcmd`/`reg_amount`.

use crate::op::matmul::MatMul;

pub mod matmul;

/// Data precision modes supported by the NPU hardware.
///
/// These values are written into the `proc_precision` / `in_precision` /
/// `out_precision` fields of CNA, MAC (core), and DPU register configs.
/// The NPU pipeline converts between precisions automatically.
#[allow(unused)]
#[derive(Clone, Copy, Debug)]
#[repr(u8)]
pub(crate) enum Precision {
    /// 8-bit signed integer — most common for quantized inference.
    Int8 = 0x0,
    /// 16-bit IEEE 754 half-precision float.
    Float16 = 0x2,
    /// 32-bit signed integer — used for accumulator outputs (e.g. matmul i8×i8 → i32).
    Int32 = 0x4,
    /// 32-bit IEEE 754 single-precision float.
    Float32 = 0x5,
}

/// A concrete NPU operation with its DMA buffers (input, weight, output).
///
/// Currently only i8 matrix multiplication is implemented.
/// Future variants: Conv2d, Pooling, Elementwise, etc.
pub enum Operation {
    /// Matrix multiply: A(m×k, i8) × B(k×n, i8) → C(m×n, i32).
    MatMulu8(MatMul<i8, i32>),
}

impl Operation {
    /// Number of 64-bit register command words this operation needs.
    ///
    /// The PC module reads exactly this many entries from the regcmd buffer
    /// for each task.  Currently all operations use 112 words (the full
    /// CNA + MAC + DPU register set), but this could vary for simpler ops.
    pub fn reg_amount(&self) -> u32 {
        112
    }

    /// Fill a pre-allocated register command buffer with the hardware
    /// instructions for this operation.
    ///
    /// Each `u64` entry is packed by [`npu_op(block, value, register)`]
    /// and tells the PC which register to write and what value.
    pub fn fill_regcmd(&self, regcmd: &mut [u64]) {
        match self {
            Operation::MatMulu8(op) => {
                op.fill_regcmd(regcmd);
            }
        }
    }
}

/// Trait that every NPU operation must implement.
///
/// The single method `fill_regcmd` is responsible for:
/// 1. Computing all CNA/MAC/DPU register values from the operation parameters.
/// 2. Packing them into `npu_op()` format in the provided slice.
pub trait OperationTrait {
    fn fill_regcmd(&self, regcmd: &mut [u64]);
}
