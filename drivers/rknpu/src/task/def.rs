#![allow(dead_code)]

pub const DIRECT_CONVOLUTION: u8 = 0x0;
pub const NPU_CBUF_BANK_SIZE: u16 = 32768;
pub const NPU_CBUF_BANKS: u16 = 12;

pub const fn npu_op(op: u32, value: u32, reg: u32) -> u64 {
    ((op as u64 & 0xFFFF) << 48) | ((value as u64 & 0xFFFF_FFFF) << 16) | reg as u64
}

// 根据 TRM V1.0 2022-03-09 的寄存器及描述（可能含糊或缺失）
pub const PC_OPERATION_ENABLE: u32 = 0x0008; // 操作使能
pub const PC_BASE_ADDRESS: u32 = 0x0010; // PC 地址寄存器
pub const PC_REGISTER_AMOUNTS: u32 = 0x0014; // 每个任务的寄存器数量

pub const CNA_S_POINTER: u32 = 0x1004; // 单寄存器组指针
pub const CNA_CONV_CON1: u32 = 0x100C; // 卷积控制寄存器1
pub const CNA_CONV_CON2: u32 = 0x1010; // 卷积控制寄存器2
pub const CNA_CONV_CON3: u32 = 0x1014; // 卷积控制寄存器3
pub const CNA_DATA_SIZE0: u32 = 0x1020; // 特征数据大小控制寄存器0
pub const CNA_DATA_SIZE1: u32 = 0x1024; // 特征数据大小控制寄存器1
pub const CNA_DATA_SIZE2: u32 = 0x1028; // 特征数据大小控制寄存器2
pub const CNA_DATA_SIZE3: u32 = 0x102C; // 特征数据大小控制寄存器3
pub const CNA_WEIGHT_SIZE0: u32 = 0x1030; // 权重大小控制 0
pub const CNA_WEIGHT_SIZE1: u32 = 0x1034; // 权重大小控制 1
pub const CNA_WEIGHT_SIZE2: u32 = 0x1038; // 权重大小控制 2
pub const CNA_CBUF_CON0: u32 = 0x1040; // CBUF 控制寄存器 0
pub const CNA_CBUF_CON1: u32 = 0x1044; // CBUF 控制寄存器 1
pub const CNA_CVT_CON0: u32 = 0x104C; // 输入转换控制寄存器0
pub const CNA_CVT_CON1: u32 = 0x1050; // 输入转换控制寄存器1
pub const CNA_CVT_CON2: u32 = 0x1054; // 输入转换控制寄存器2
pub const CNA_CVT_CON3: u32 = 0x1058; // 输入转换控制寄存器3
pub const CNA_CVT_CON4: u32 = 0x105C; // 输入转换控制寄存器4
pub const CNA_FC_CON0: u32 = 0x1060; // 全连接控制寄存器0
pub const CNA_FC_CON1: u32 = 0x1064; // 全连接控制寄存器1
pub const CNA_PAD_CON0: u32 = 0x1068; // 填充控制寄存器0
pub const CNA_FEATURE_DATA_ADDR: u32 = 0x1070; // 输入特征数据的基地址
pub const CNA_FC_CON2: u32 = 0x1074; // 全连接控制寄存器2
pub const CNA_DMA_CON0: u32 = 0x1078; // AXI 控制寄存器 0
pub const CNA_DMA_CON1: u32 = 0x107C; // AXI 控制寄存器 1
pub const CNA_DMA_CON2: u32 = 0x1080; // AXI 控制寄存器 2
pub const CNA_FC_DATA_SIZE0: u32 = 0x1084; // 全连接数据大小控制寄存器0
pub const CNA_FC_DATA_SIZE1: u32 = 0x1088; // 全连接数据大小控制寄存器1
pub const CNA_DCOMP_CTRL: u32 = 0x1100; // 权重解压缩控制寄存器
pub const CNA_DCOMP_REGNUM: u32 = 0x1104; // 权重解压缩寄存器数量
pub const CNA_DCOMP_ADDR0: u32 = 0x1110; // 权重的基地址
pub const CNA_DCOMP_AMOUNT: u32 = 0x1140; // 第 0 次解压缩的权重解压缩量
pub const CNA_DCOMP_AMOUNT1: u32 = 0x1144; // 第 1 次解压缩的权重解压缩量
pub const CNA_DCOMP_AMOUNT2: u32 = 0x1148; // 第 2 次解压缩的权重解压缩量
pub const CNA_DCOMP_AMOUNT3: u32 = 0x114C; // 第 3 次解压缩的权重解压缩量
pub const CNA_DCOMP_AMOUNT4: u32 = 0x1150; // 第 4 次解压缩的权重解压缩量
pub const CNA_DCOMP_AMOUNT5: u32 = 0x1154; // 第 5 次解压缩的权重解压缩量
pub const CNA_DCOMP_AMOUNT6: u32 = 0x1158; // 第 6 次解压缩的权重解压缩量
pub const CNA_DCOMP_AMOUNT7: u32 = 0x115C; // 第 7 次解压缩的权重解压缩量
pub const CNA_DCOMP_AMOUNT8: u32 = 0x1160; // 第 8 次解压缩的权重解压缩量
pub const CNA_DCOMP_AMOUNT9: u32 = 0x1164; // 第 9 次解压缩的权重解压缩量
pub const CNA_DCOMP_AMOUNT10: u32 = 0x1168; // 第 10 次解压缩的权重解压缩量
pub const CNA_DCOMP_AMOUNT11: u32 = 0x116C; // 第 11 次解压缩的权重解压缩量
pub const CNA_DCOMP_AMOUNT12: u32 = 0x1170; // 第 12 次解压缩的权重解压缩量
pub const CNA_DCOMP_AMOUNT13: u32 = 0x1174; // 第 13 次解压缩的权重解压缩量
pub const CNA_DCOMP_AMOUNT14: u32 = 0x1178; // 第 14 次解压缩的权重解压缩量
pub const CNA_DCOMP_AMOUNT15: u32 = 0x117C; // 第 15 次解压缩的权重解压缩量
pub const CNA_CVT_CON5: u32 = 0x1180; // 输入转换控制寄存器5
pub const CNA_PAD_CON1: u32 = 0x1184; // 填充控制寄存器1

pub const CORE_S_POINTER: u32 = 0x3004; // 单寄存器组指针
pub const CORE_MISC_CFG: u32 = 0x3010; // 杂项配置寄存器
pub const CORE_DATAOUT_SIZE_0: u32 = 0x3014; // 输出的特征大小寄存器 0
pub const CORE_DATAOUT_SIZE_1: u32 = 0x3018; // 输出的特征大小寄存器 1
pub const CORE_CLIP_TRUNCATE: u32 = 0x301C; // 移位值寄存器
pub const CORE_3030: u32 = 0x3030; // 似乎没有文档记录，是否需要？

pub const DPU_S_POINTER: u32 = 0x4004; // 单寄存器组指针
pub const DPU_FEATURE_MODE_CFG: u32 = 0x400C; // 特征模式的配置
pub const DPU_DATA_FORMAT: u32 = 0x4010; // 数据格式的配置
pub const DPU_OFFSET_PEND: u32 = 0x4014; // 偏移挂起的值
pub const DPU_DST_BASE_ADD: u32 = 0x4020; // 目标基地址
pub const DPU_DST_SURF_STRIDE: u32 = 0x4024; // 目标表面大小
pub const DPU_DATA_CUBE_WIDTH: u32 = 0x4030; // 输入立方体的宽度
pub const DPU_DATA_CUBE_HEIGHT: u32 = 0x4034; // 输入立方体的高度
pub const DPU_DATA_CUBE_NOTCH_ADDR: u32 = 0x4038; // 输入立方体的缺口信号
pub const DPU_DATA_CUBE_CHANNEL: u32 = 0x403C; // 输入立方体的通道
pub const DPU_BS_CFG: u32 = 0x4040; // BS 的配置
pub const DPU_BS_ALU_CFG: u32 = 0x4044; // BS ALU 的配置
pub const DPU_BS_MUL_CFG: u32 = 0x4048; // BS MUL 的配置
pub const DPU_BS_RELUX_CMP_VALUE: u32 = 0x404C; // RELUX 比较的值
pub const DPU_BS_OW_CFG: u32 = 0x4050; // BS OW 的配置
pub const DPU_BS_OW_OP: u32 = 0x4054; // BS OW 的 Ow 操作
pub const DPU_WDMA_SIZE_0: u32 = 0x4058; // WDMA 的大小 0
pub const DPU_WDMA_SIZE_1: u32 = 0x405C; // WDMA 的大小 1
pub const DPU_BN_CFG: u32 = 0x4060; // BN 的配置
pub const DPU_BN_ALU_CFG: u32 = 0x4064; // BN ALU 的配置
pub const DPU_BN_MUL_CFG: u32 = 0x4068; // BN MUL 的配置
pub const DPU_BN_RELUX_CMP_VALUE: u32 = 0x406C; // RELUX 比较的值
pub const DPU_EW_CFG: u32 = 0x4070; // EW 的配置
pub const DPU_EW_CVT_OFFSET_VALUE: u32 = 0x4074; // EW 输入转换的偏移
pub const DPU_EW_CVT_SCALE_VALUE: u32 = 0x4078; // EW 输入转换的缩放
pub const DPU_EW_RELUX_CMP_VALUE: u32 = 0x407C; // RELUX 比较的值
pub const DPU_OUT_CVT_OFFSET: u32 = 0x4080; // 输出转换器的偏移
pub const DPU_OUT_CVT_SCALE: u32 = 0x4084; // 输出转换器的缩放
pub const DPU_OUT_CVT_SHIFT: u32 = 0x4088; // 输出转换器的移位
pub const DPU_EW_OP_VALUE_0: u32 = 0x4090; // 配置 EW 的操作数0
pub const DPU_EW_OP_VALUE_1: u32 = 0x4094; // 配置 EW 的操作数1
pub const DPU_EW_OP_VALUE_2: u32 = 0x4098; // 配置 EW 的操作数2
pub const DPU_EW_OP_VALUE_3: u32 = 0x409C; // 配置 EW 的操作数3
pub const DPU_EW_OP_VALUE_4: u32 = 0x40A0; // 配置 EW 的操作数4
pub const DPU_EW_OP_VALUE_5: u32 = 0x40A4; // 配置 EW 的操作数5
pub const DPU_EW_OP_VALUE_6: u32 = 0x40A8; // 配置 EW 的操作数6
pub const DPU_EW_OP_VALUE_7: u32 = 0x40AC; // 配置 EW 的操作数7
pub const DPU_SURFACE_ADD: u32 = 0x40C0; // 表面加法器的值
pub const DPU_40C4: u32 = 0x40C4; // 未记录
pub const DPU_LUT_ACCESS_CFG: u32 = 0x4100; // LUT 访问地址和类型
pub const DPU_LUT_ACCESS_DATA: u32 = 0x4104; // LUT 访问数据的配置
pub const DPU_LUT_CFG: u32 = 0x4108; // LUT 的配置
pub const DPU_LUT_INFO: u32 = 0x410C; // LUT 信息寄存器
pub const DPU_LUT_LE_START: u32 = 0x4110; // LE LUT 起始点
pub const DPU_LUT_LE_END: u32 = 0x4114; // LE LUT 结束点
pub const DPU_LUT_LO_START: u32 = 0x4118; // LO LUT 起始点
pub const DPU_LUT_LO_END: u32 = 0x411C; // LO LUT 结束点
pub const DPU_LUT_LE_SLOPE_SCALE: u32 = 0x4120; // LE LUT 斜率缩放
pub const DPU_LUT_LE_SLOPE_SHIFT: u32 = 0x4124; // LE LUT 斜率移位
pub const DPU_LUT_LO_SLOPE_SCALE: u32 = 0x4128; // LO LUT 斜率缩放
pub const DPU_LUT_LO_SLOPE_SHIFT: u32 = 0x412C; // LO LUT 斜率移位

// NPU 能力仅限于以下单元
pub const BLOCK_PC: u32 = 0x0100;
pub const BLOCK_CNA: u32 = 0x0200;
pub const BLOCK_CORE: u32 = 0x0800;
pub const BLOCK_DPU: u32 = 0x1000;
pub const BLOCK_DPU_RDMA: u32 = 0x2000;
pub const BLOCK_PPU: u32 = 0x4000;
pub const BLOCK_PPU_RDMA: u32 = 0x8000;

pub const PC_OP_01: u32 = 0x01; // 寄存器 ??
pub const PC_OP_40: u32 = 0x40; // ??
pub const PC_OP_ENABLE: u32 = 0x80; // 启用块

pub const OP_REG_PC: u32 = BLOCK_PC | PC_OP_01; // ??
pub const OP_REG_CNA: u32 = BLOCK_CNA | PC_OP_01; // ??
pub const OP_REG_CORE: u32 = BLOCK_CORE | PC_OP_01; // ??
pub const OP_REG_DPU: u32 = BLOCK_DPU | PC_OP_01; // ??

pub const OP_40: u32 = PC_OP_40 | PC_OP_01; // ??
pub const OP_ENABLE: u32 = PC_OP_ENABLE | PC_OP_01; // ??
pub const OP_NONE: u32 = 0x0; // ??

pub const PC_ENABLE: u32 = 0x01; // 为此任务启用
pub const PC_ENABLE_CNA: u32 = 0x04; // ?? 中断
pub const PC_ENABLE_DPU: u32 = 0x08; // ?? 中断
pub const PC_ENABLE_PPU: u32 = 0x10; // ?? 中断
