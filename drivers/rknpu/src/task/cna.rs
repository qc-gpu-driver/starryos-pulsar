//! Descriptor structs for the CNA, MAC (core), and PC pipeline stages.
//!
//! These are **software-side** descriptions of what a single layer needs.
//! The `matmul.rs` (or future `conv2d.rs`) code fills these structs from
//! high-level parameters (M, K, N, precision, etc.), then `gen_matul()`
//! packs them into 64-bit register commands via `npu_op()`.
//!
//! # NPU pipeline recap
//!
//! ```text
//!  Input tensor                           Output tensor
//!      │                                       ▲
//!      ▼                                       │
//!  ┌───────┐   ┌───────┐   ┌───────┐   ┌──────┴──┐
//!  │  CNA  │──►│  MAC  │──►│  DPU  │──►│  WDMA   │
//!  │ (load │   │(compute)  │(post- │   │ (write  │
//!  │ feat+ │   │ matmul/│   │ proc) │   │ result) │
//!  │ weight)│   │ conv  │   │       │   │         │
//!  └───────┘   └───────┘   └───────┘   └─────────┘
//! ```

/// CNA (Convolution Neural Accelerator) descriptor.
///
/// Configures **input feature loading** and **weight loading** for one layer.
/// The hex comments (e.g. `// 0x100C`) indicate which MMIO register the
/// field maps to — useful when cross-referencing the TRM.
///
/// # Key concepts
///
/// - **datain_***: dimensions of the input feature tensor (W×H×C)
/// - **weight_***: dimensions and size of the weight/kernel tensor
/// - **CBUF**: on-chip buffer shared between feature data and weights;
///   `data_bank` + `weight_bank` must fit within `NPU_CBUF_BANKS` (12)
/// - **feature_base_addr**: DMA bus address of the input tensor
/// - **decompress_addr0**: DMA bus address of the weight tensor
///   (named "decompress" because the HW can decompress on-the-fly)
/// - **cvt_***: input data type conversion (scale/bypass)
/// - **fc_***: fully-connected layer specific controls
#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct NpuCnaDesc {
    pub enable: u8,
    pub conv_mode: u8,                // 0x100C — convolution mode (direct/winograd)
    pub in_precision: u8,             // 0x100C — input data precision (Int8/FP16/…)
    pub proc_precision: u8,           // 0x100C — processing precision
    pub kernel_groups: u8,            // 0x1010 — number of kernel groups
    pub feature_grains: u16,          // 0x1010 — feature data granularity
    pub conv_y_stride: u8,            // 0x1014 — vertical stride
    pub conv_x_stride: u8,            // 0x1014 — horizontal stride
    pub datain_width: u16,            // 0x1020 — input feature width
    pub datain_height: u16,           // 0x1020 — input feature height
    pub datain_channel: u16,          // 0x1024 — input feature channels
    pub dataout_width: u16,           // 0x1028 — output feature width
    pub dataout_atomics: u32,         // 0x102C — output atomic count (W×H)
    pub weight_bytes: u32,            // 0x1030 — total weight size in bytes
    pub weight_bytes_per_kernel: u32, // 0x1034 — weight bytes per output channel
    pub weight_width: u8,             // 0x1038 — kernel width
    pub weight_height: u8,            // 0x1038 — kernel height
    pub weight_kernels: u16,          // 0x1038 — number of output channels (N)
    pub weight_bank: u8,              // 0x1040 — CBUF banks allocated for weights
    pub data_bank: u8,                // 0x1040 — CBUF banks allocated for features
    pub data_entries: u16,            // 0x1044 — feature entries in CBUF
    pub data_sign: u8,                // 0x104c — signed/unsigned flag
    pub cvt_type: u8,                 // 0x104c — conversion type
    pub cvt_bypass: u8,               // 0x104c — bypass input conversion
    pub cvt_scale0: u16,              // 0x1050 — channel group 0 scale
    pub cvt_scale1: u16,              // 0x1054 — channel group 1 scale
    pub cvt_scale2: u16,              // 0x1058 — channel group 2 scale
    pub cvt_scale3: u16,              // 0x105C — channel group 3 scale
    pub fc_skip_en: u8,               // 0x1060 — fully-connected skip enable
    pub data_offset: u16,             // 0x1064 — data start offset
    pub pad_left: u8,                 // 0x1068 — left zero-padding
    pub pad_top: u8,                  // 0x1068 — top zero-padding
    pub feature_base_addr: u32,       // 0x1070 — DMA address of input feature data
    pub weight_offset: u16,           // 0x1074 — weight start offset
    pub weight_burst_len: u8,         // 0x1078 — AXI burst length for weight DMA
    pub data_burst_len: u8,           // 0x1078 — AXI burst length for feature DMA
    pub line_stride: u32,             // 0x107C — bytes between consecutive lines
    pub surf_stride: i32,             // 0x1080 — bytes between channel surfaces
    pub dma_width: u16,               // 0x1084 — DMA transfer width
    pub dma_height: u16,              // 0x1084 — DMA transfer height
    pub dma_channel: u16,             // 0x1088 — DMA transfer channels
    pub decompress_addr0: u32,        // 0x1110 — DMA address of weight data
    pub dataout_height: u16,
}

/// MAC (Multiply-Accumulate Core) descriptor.
///
/// The MAC unit does the actual matrix multiply / convolution computation.
/// Its config is relatively simple — mainly output dimensions and precision.
/// The heavy lifting (input routing) is handled by CNA.
#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct NpuCoreDesc {
    pub proc_precision: u8,   // 0x3010 — compute precision
    pub qd_en: u8,            // 0x3010 — quantize/dequantize enable
    pub dataout_height: u16,  // 0x3014 — output height (0-indexed: actual-1)
    pub dataout_width: u16,   // 0x3014 — output width  (0-indexed: actual-1)
    pub dataout_channel: u16, // 0x3018 — output channels (0-indexed: actual-1)
}

/// PC (Program Counter) descriptor — rarely filled manually.
///
/// The PC module is usually programmed directly by `submit_pc()` in the
/// register layer, not through this struct.
#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct NpuPcDesc {
    pub pc_source_addr: u32, // 0x0010 — regcmd buffer base address
    pub pc_data_amount: u32, // 0x0014 — number of regcmd words per task
}

/// Raw 64-bit register command array for one CNA+MAC task (112 entries).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct NpuCnaCoreTask {
    pub ops: [u64; 112],
}
