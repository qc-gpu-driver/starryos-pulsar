//! CNA、MAC（核心）和 PC 流水线阶段的描述符结构。
//!
//! 这些是单层所需内容的**软件端**描述。
//! `matmul.rs`（或未来的 `conv2d.rs`）代码从高级参数（M、K、N、精度等）填充这些结构，
//! 然后 `gen_matul()` 通过 `npu_op()` 将它们打包成 64 位寄存器命令。
//!
//! # NPU 流水线回顾
//!
//! ```text
//!  输入张量                               输出张量
//!      │                                       ▲
//!      ▼                                       │
//!  ┌───────┐   ┌───────┐   ┌───────┐   ┌──────┴──┐
//!  │  CNA  │──►│  MAC  │──►│  DPU  │──►│  WDMA   │
//!  │ (加载 │   │(计算)  │   │(后处  │   │ (写入   │
//!  │ 特征+ │   │ 矩阵乘/│   │ 理)   │   │ 结果)   │
//!  │ 权重) │   │ 卷积  │   │       │   │         │
//!  └───────┘   └───────┘   └───────┘   └─────────┘
//! ```

/// CNA（卷积神经加速器）描述符。
///
/// 为一层配置**输入特征加载**和**权重加载**。
/// 十六进制注释（例如 `// 0x100C`）指示字段映射到哪个 MMIO 寄存器 — 
/// 在交叉引用 TRM 时很有用。
///
/// # 关键概念
///
/// - **datain_***: 输入特征张量的维度（W×H×C）
/// - **weight_***: 权重/内核张量的维度和大小
/// - **CBUF**: 特征数据和权重之间共享的片上缓冲区；
///   `data_bank` + `weight_bank` 必须适合 `NPU_CBUF_BANKS`（12）
/// - **feature_base_addr**: 输入张量的 DMA 总线地址
/// - **decompress_addr0**: 权重张量的 DMA 总线地址
///   （命名为"decompress"是因为硬件可以即时解压缩）
/// - **cvt_***: 输入数据类型转换（缩放/旁路）
/// - **fc_***: 全连接层特定控制
#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct NpuCnaDesc {
    pub enable: u8,
    pub conv_mode: u8,                // 0x100C — 卷积模式（直接/winograd）
    pub in_precision: u8,             // 0x100C — 输入数据精度（Int8/FP16/…）
    pub proc_precision: u8,           // 0x100C — 处理精度
    pub kernel_groups: u8,            // 0x1010 — 内核组数
    pub feature_grains: u16,          // 0x1010 — 特征数据粒度
    pub conv_y_stride: u8,            // 0x1014 — 垂直步长
    pub conv_x_stride: u8,            // 0x1014 — 水平步长
    pub datain_width: u16,            // 0x1020 — 输入特征宽度
    pub datain_height: u16,           // 0x1020 — 输入特征高度
    pub datain_channel: u16,          // 0x1024 — 输入特征通道数
    pub dataout_width: u16,           // 0x1028 — 输出特征宽度
    pub dataout_atomics: u32,         // 0x102C — 输出原子计数（W×H）
    pub weight_bytes: u32,            // 0x1030 — 权重总大小（字节）
    pub weight_bytes_per_kernel: u32, // 0x1034 — 每个输出通道的权重字节数
    pub weight_width: u8,             // 0x1038 — 内核宽度
    pub weight_height: u8,            // 0x1038 — 内核高度
    pub weight_kernels: u16,          // 0x1038 — 输出通道数（N）
    pub weight_bank: u8,              // 0x1040 — 为权重分配的 CBUF 存储体
    pub data_bank: u8,                // 0x1040 — 为特征分配的 CBUF 存储体
    pub data_entries: u16,            // 0x1044 — CBUF 中的特征条目
    pub data_sign: u8,                // 0x104c — 有符号/无符号标志
    pub cvt_type: u8,                 // 0x104c — 转换类型
    pub cvt_bypass: u8,               // 0x104c — 旁路输入转换
    pub cvt_scale0: u16,              // 0x1050 — 通道组 0 缩放
    pub cvt_scale1: u16,              // 0x1054 — 通道组 1 缩放
    pub cvt_scale2: u16,              // 0x1058 — 通道组 2 缩放
    pub cvt_scale3: u16,              // 0x105C — 通道组 3 缩放
    pub fc_skip_en: u8,               // 0x1060 — 全连接跳过使能
    pub data_offset: u16,             // 0x1064 — 数据起始偏移量
    pub pad_left: u8,                 // 0x1068 — 左侧零填充
    pub pad_top: u8,                  // 0x1068 — 顶部零填充
    pub feature_base_addr: u32,       // 0x1070 — 输入特征数据的 DMA 地址
    pub weight_offset: u16,           // 0x1074 — 权重起始偏移量
    pub weight_burst_len: u8,         // 0x1078 — 权重 DMA 的 AXI 突发长度
    pub data_burst_len: u8,           // 0x1078 — 特征 DMA 的 AXI 突发长度
    pub line_stride: u32,             // 0x107C — 连续行之间的字节数
    pub surf_stride: i32,             // 0x1080 — 通道表面之间的字节数
    pub dma_width: u16,               // 0x1084 — DMA 传输宽度
    pub dma_height: u16,              // 0x1084 — DMA 传输高度
    pub dma_channel: u16,             // 0x1088 — DMA 传输通道数
    pub decompress_addr0: u32,        // 0x1110 — 权重数据的 DMA 地址
    pub dataout_height: u16,
}

/// MAC（乘累加核心）描述符。
///
/// MAC 单元执行实际的矩阵乘法/卷积计算。
/// 其配置相对简单 — 主要是输出维度和精度。
/// 繁重的工作（输入路由）由 CNA 处理。
#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct NpuCoreDesc {
    pub proc_precision: u8,   // 0x3010 — 计算精度
    pub qd_en: u8,            // 0x3010 — 量化/反量化使能
    pub dataout_height: u16,  // 0x3014 — 输出高度（0 索引：实际值-1）
    pub dataout_width: u16,   // 0x3014 — 输出宽度（0 索引：实际值-1）
    pub dataout_channel: u16, // 0x3018 — 输出通道数（0 索引：实际值-1）
}

/// PC（程序计数器）描述符 — 很少手动填充。
///
/// PC 模块通常由寄存器层中的 `submit_pc()` 直接编程，
/// 而不是通过此结构。
#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct NpuPcDesc {
    pub pc_source_addr: u32, // 0x0010 — regcmd 缓冲区基地址
    pub pc_data_amount: u32, // 0x0014 — 每个任务的 regcmd 字数
}

/// 一个 CNA+MAC 任务的原始 64 位寄存器命令数组（112 个条目）。
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct NpuCnaCoreTask {
    pub ops: [u64; 112],
}
