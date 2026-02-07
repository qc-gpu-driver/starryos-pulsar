//! DRM ioctl 编码解析与分发机制
//!
//! Linux DRM ioctl 命令编码 (32-bit):
//!   [31:30] 方向 dir   (0=none, 1=write, 2=read, 3=read+write)
//!   [29:16] 参数大小 size
//!   [15:8]  类型 type  (DRM = 'd' = 0x64)
//!   [7:0]   编号 nr

/// DRM ioctl 类型字节 'd'
pub const DRM_IOCTL_TYPE: u8 = b'd';

/// 驱动私有 ioctl 起始编号
pub const DRM_COMMAND_BASE: u8 = 0x40;
/// 驱动私有 ioctl 结束编号
pub const DRM_COMMAND_END: u8 = 0xA0;

// ---- ioctl 编码解析 ----

/// 提取 ioctl 编号 (bits [7:0])
pub fn ioctl_nr(cmd: u32) -> u8 {
    (cmd & 0xFF) as u8
}

/// 提取 ioctl 类型 (bits [15:8])
pub fn ioctl_type(cmd: u32) -> u8 {
    ((cmd >> 8) & 0xFF) as u8
}

/// 提取参数大小 (bits [29:16])
pub fn ioctl_size(cmd: u32) -> usize {
    ((cmd >> 16) & 0x3FFF) as usize
}

/// 提取方向 (bits [31:30])
pub fn ioctl_dir(cmd: u32) -> u8 {
    ((cmd >> 30) & 0x3) as u8
}

/// 是否是 DRM 类型的 ioctl
pub fn is_drm_ioctl(cmd: u32) -> bool {
    ioctl_type(cmd) == DRM_IOCTL_TYPE
}

/// 是否是驱动私有 ioctl (nr >= 0x40)
pub fn is_driver_ioctl(cmd: u32) -> bool {
    let nr = ioctl_nr(cmd);
    nr >= DRM_COMMAND_BASE && nr < DRM_COMMAND_END
}

// ---- ioctl 分发 ----

/// 驱动 ioctl 处理函数签名
/// 参数: (nr - DRM_COMMAND_BASE 后的偏移, arg 用户态指针)
/// 返回: Ok(返回值) 或 Err
pub type DrmIoctlHandler = fn(nr: u8, arg: usize) -> Result<usize, i32>;

/// DRM ioctl 分发表条目
pub struct DrmIoctlEntry {
    /// 驱动名称（调试用）
    pub name: &'static str,
    /// 驱动私有 ioctl 处理函数
    pub handler: DrmIoctlHandler,
}

/// 最大注册驱动数
const MAX_DRM_DRIVERS: usize = 8;

/// 已注册驱动数量
static mut DRIVER_COUNT: usize = 0;
/// 驱动分发表
static mut DRIVERS: [Option<DrmIoctlEntry>; MAX_DRM_DRIVERS] = [const { None }; MAX_DRM_DRIVERS];

/// 注册一个 DRM 驱动的 ioctl handler
///
/// # Safety
/// 必须在单线程初始化阶段调用
pub unsafe fn register_driver(entry: DrmIoctlEntry) -> Result<(), &'static str> {
    if DRIVER_COUNT >= MAX_DRM_DRIVERS {
        return Err("DRM driver table full");
    }
    DRIVERS[DRIVER_COUNT] = Some(entry);
    DRIVER_COUNT += 1;
    Ok(())
}

/// 分发 DRM ioctl
/// 对于驱动私有 ioctl，依次尝试已注册的驱动
pub fn dispatch_ioctl(cmd: u32, arg: usize) -> Result<usize, i32> {
    if !is_drm_ioctl(cmd) {
        return Err(-1); // 不是 DRM ioctl
    }

    let nr = ioctl_nr(cmd);

    if !is_driver_ioctl(cmd) {
        // TODO: 处理 DRM core ioctl (nr < 0x40)
        // 如 DRM_IOCTL_VERSION, DRM_IOCTL_GEM_CLOSE 等
        return Err(-1);
    }

    // 驱动私有 ioctl: nr >= 0x40
    let driver_nr = nr - DRM_COMMAND_BASE;

    unsafe {
        for i in 0..DRIVER_COUNT {
            if let Some(ref entry) = DRIVERS[i] {
                let result = (entry.handler)(driver_nr, arg);
                if result != Err(-1) {
                    return result;
                }
            }
        }
    }

    Err(-1) // 没有驱动处理
}
