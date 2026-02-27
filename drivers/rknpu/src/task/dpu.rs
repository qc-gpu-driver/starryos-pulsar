//! DPU（数据处理单元）描述符。
//!
//! DPU 是 MAC 计算后的**后处理**阶段。
//! 它可以应用偏置（BS）、批归一化（BN）、逐元素操作（EW）、
//! 激活函数（通过 LUT 的 ReLU）、输出类型转换，最后
//! 通过 DMA 将结果写入输出缓冲区。
//!
//! # DPU 内部的后处理流水线
//!
//! ```text
//!  MAC 输出 ──► BS（偏置/缩放）──► BN（批归一化）──► EW（逐元素）
//!                                                              │
//!                                                              ▼
//!                                        输出转换 ◄── LUT（激活）
//!                                              │
//!                                              ▼
//!                                      WDMA（写入 DRAM）
//! ```
//!
//! 每个子块（BS、BN、EW）都可以通过将其 `*_bypass` 标志设置为 1 来单独**旁路**。
//! 对于没有任何后处理的原始矩阵乘法，所有旁路标志都设置为 1，
//! MAC 输出直接进入 WDMA。

/// 一层的 DPU（数据处理单元）描述符。
///
/// # 旁路标志
///
/// 对于没有后处理的纯矩阵乘法/卷积，将所有 `*_bypass = 1`：
/// - `bs_bypass`  — 跳过偏置/缩放
/// - `bn_bypass`  — 跳过批归一化
/// - `ew_bypass`  — 跳过逐元素操作
/// - `od_bypass`  — 跳过输出数据重排序
///
/// # 输出
///
/// - `dst_base_addr` — 写入结果张量的 DMA 总线地址
/// - `*_wdma` 字段 — 写 DMA 引擎的维度
/// - `out_cvt_scale` — 输出类型转换缩放（例如 i32 → i8）
#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct NpuDpuDesc {
    pub burst_len: u8,        // 0x400C — 输出 DMA 的 AXI 突发长度
    pub conv_mode: u8,        // 0x400C — 必须匹配 CNA 的 conv_mode
    pub output_mode: u8,      // 0x400C — 0x2 = 写入内存
    pub flying_mode: u8,      // 0x400C — 0 = 来自 MAC，1 = 来自 RDMA
    pub out_precision: u8,    // 0x4010 — 输出数据精度
    pub in_precision: u8,     // 0x4010 — 输入数据精度（来自 MAC）
    pub proc_precision: u8,   // 0x4010 — 处理精度
    pub dst_base_addr: u32,   // 0x4020 — 输出张量的 DMA 地址
    pub dst_surf_stride: u32, // 0x4024 — 输出通道表面之间的字节数
    pub width: u16,           // 0x4030 — 输出宽度（0 索引）
    pub height: u16,          // 0x4034 — 输出高度（0 索引）
    pub channel: u16,         // 0x403C — 输出通道数（0 索引）
    pub bs_bypass: u8,        // 0x4040 — 旁路偏置/缩放块
    pub bs_alu_bypass: u8,    // 0x4040 — 旁路 BS ALU 子单元
    pub bs_mul_bypass: u8,    // 0x4040 — 旁路 BS 乘法器
    pub bs_relu_bypass: u8,   // 0x4040 — 旁路 BS ReLU 激活
    pub od_bypass: u8,        // 0x4050 — 旁路输出数据重排序
    pub size_e_2: u8,         // 0x4050 — OW 大小指数 2
    pub size_e_1: u8,         // 0x4050 — OW 大小指数 1
    pub size_e_0: u8,         // 0x4050 — OW 大小指数 0
    pub channel_wdma: u16,    // 0x4058 — WDMA 输出通道数（0 索引）
    pub height_wdma: u16,     // 0x405C — WDMA 输出高度（0 索引）
    pub width_wdma: u16,      // 0x405C — WDMA 输出宽度（0 索引）
    pub bn_relu_bypass: u8,   // 0x4060 — 旁路 BN ReLU
    pub bn_mul_bypass: u8,    // 0x4060 — 旁路 BN 乘法器
    pub bn_alu_bypass: u8,    // 0x4060 — 旁路 BN ALU
    pub bn_bypass: u8,        // 0x4060 — 旁路整个批归一化块
    pub ew_bypass: u8,        // 0x4070 — 旁路整个逐元素块
    pub ew_op_bypass: u8,     // 0x4070 — 旁路 EW 操作
    pub ew_lut_bypass: u8,    // 0x4070 — 旁路 EW LUT（激活）
    pub ew_op_cvt_bypass: u8, // 0x4070 — 旁路 EW 操作数转换
    pub ew_relu_bypass: u8,   // 0x4070 — 旁路 EW ReLU
    pub fp32tofp16_en: u8,    // 0x4084 — 启用 FP32→FP16 输出转换
    pub out_cvt_scale: u16,   // 0x4084 — 输出转换缩放因子
    pub surf_add: u32,        // 0x40C0 — 表面地址加法器
}
