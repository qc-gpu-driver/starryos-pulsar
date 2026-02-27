//! 从 C 语言 `struct rknpu_config` 翻译而来的 RKNPU 配置绑定。
//!
//! 本模块提供了 `#[repr(C)]` 的 Rust 等价结构，适用于 FFI
//! 或直接翻译内核风格的配置数据。

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum RknpuType {
    Rk3588,
}

#[derive(Debug, Clone)]
pub struct RknpuConfig {
    pub rknpu_type: RknpuType,
}
