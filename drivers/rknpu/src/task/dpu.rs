//! DPU (Data Processing Unit) descriptor.
//!
//! The DPU is the **post-processing** stage after MAC computation.
//! It can apply bias (BS), batch normalization (BN), element-wise ops (EW),
//! activation functions (ReLU via LUT), output type conversion, and finally
//! DMA-write the result to the output buffer.
//!
//! # Post-processing pipeline inside the DPU
//!
//! ```text
//!  MAC output ──► BS (bias/scale) ──► BN (batch-norm) ──► EW (element-wise)
//!                                                              │
//!                                                              ▼
//!                                        output convert ◄── LUT (activation)
//!                                              │
//!                                              ▼
//!                                      WDMA (write to DRAM)
//! ```
//!
//! Each sub-block (BS, BN, EW) can be individually **bypassed** by setting
//! its `*_bypass` flag to 1.  For a raw matmul without any post-processing,
//! all bypass flags are set to 1 and the MAC output goes straight to WDMA.

/// DPU (Data Processing Unit) descriptor for one layer.
///
/// # Bypass flags
///
/// For a pure matmul / conv without post-processing, set all `*_bypass = 1`:
/// - `bs_bypass`  — skip bias/scale
/// - `bn_bypass`  — skip batch normalization
/// - `ew_bypass`  — skip element-wise operations
/// - `od_bypass`  — skip output data reorder
///
/// # Output
///
/// - `dst_base_addr` — DMA bus address where the result tensor is written
/// - `*_wdma` fields — dimensions for the write-DMA engine
/// - `out_cvt_scale` — output type conversion scale (e.g. i32 → i8)
#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct NpuDpuDesc {
    pub burst_len: u8,        // 0x400C — AXI burst length for output DMA
    pub conv_mode: u8,        // 0x400C — must match CNA's conv_mode
    pub output_mode: u8,      // 0x400C — 0x2 = write to memory
    pub flying_mode: u8,      // 0x400C — 0 = from MAC, 1 = from RDMA
    pub out_precision: u8,    // 0x4010 — output data precision
    pub in_precision: u8,     // 0x4010 — input data precision (from MAC)
    pub proc_precision: u8,   // 0x4010 — processing precision
    pub dst_base_addr: u32,   // 0x4020 — DMA address of output tensor
    pub dst_surf_stride: u32, // 0x4024 — bytes between output channel surfaces
    pub width: u16,           // 0x4030 — output width  (0-indexed)
    pub height: u16,          // 0x4034 — output height (0-indexed)
    pub channel: u16,         // 0x403C — output channels (0-indexed)
    pub bs_bypass: u8,        // 0x4040 — bypass bias/scale block
    pub bs_alu_bypass: u8,    // 0x4040 — bypass BS ALU sub-unit
    pub bs_mul_bypass: u8,    // 0x4040 — bypass BS multiplier
    pub bs_relu_bypass: u8,   // 0x4040 — bypass BS ReLU activation
    pub od_bypass: u8,        // 0x4050 — bypass output data reorder
    pub size_e_2: u8,         // 0x4050 — OW size exponent 2
    pub size_e_1: u8,         // 0x4050 — OW size exponent 1
    pub size_e_0: u8,         // 0x4050 — OW size exponent 0
    pub channel_wdma: u16,    // 0x4058 — WDMA output channels (0-indexed)
    pub height_wdma: u16,     // 0x405C — WDMA output height (0-indexed)
    pub width_wdma: u16,      // 0x405C — WDMA output width  (0-indexed)
    pub bn_relu_bypass: u8,   // 0x4060 — bypass BN ReLU
    pub bn_mul_bypass: u8,    // 0x4060 — bypass BN multiplier
    pub bn_alu_bypass: u8,    // 0x4060 — bypass BN ALU
    pub bn_bypass: u8,        // 0x4060 — bypass entire batch-norm block
    pub ew_bypass: u8,        // 0x4070 — bypass entire element-wise block
    pub ew_op_bypass: u8,     // 0x4070 — bypass EW operation
    pub ew_lut_bypass: u8,    // 0x4070 — bypass EW LUT (activation)
    pub ew_op_cvt_bypass: u8, // 0x4070 — bypass EW operand conversion
    pub ew_relu_bypass: u8,   // 0x4070 — bypass EW ReLU
    pub fp32tofp16_en: u8,    // 0x4084 — enable FP32→FP16 output conversion
    pub out_cvt_scale: u16,   // 0x4084 — output conversion scale factor
    pub surf_add: u32,        // 0x40C0 — surface address adder
}
