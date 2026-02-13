# ArceOS AArch64 动态平台实现 🦀

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Rust](https://img.shields.io/badge/rust-2024+-orange.svg)](https://www.rust-lang.org/)
[![Platform](https://img.shields.io/badge/platform-AArch64-green.svg)](#)

## 📋 目录
- [项目简介](#项目简介)
- [功能特性](#功能特性)
- [快速开始](#快速开始)
  - [环境要求](#环境要求)
  - [安装步骤](#安装步骤)
  - [基本使用](#基本使用)
- [项目结构](#项目结构)
- [核心模块](#核心模块)
  - [启动初始化](#启动初始化)
  - [内存管理](#内存管理)
  - [中断处理](#中断处理)
  - [SMP多核支持](#smp多核支持)
  - [时间管理](#时间管理)
  - [电源控制](#电源控制)
  - [设备树解析](#设备树解析)
- [特性支持](#特性支持)
- [开发指南](#开发指南)
  - [构建项目](#构建项目)
  - [配置说明](#配置说明)
- [许可证](#许可证)

## 📖 项目简介

本项目是 ArceOS 在 AArch64 架构下的动态平台实现，主要用于支持 ArceOS Hypervisor 运行环境。该平台提供了完整的底层硬件抽象层，包括启动初始化、内存管理、中断处理、SMP多核支持、时间管理、电源控制及设备树解析等功能。

通过启用 `hv` 特性，可以支持 EL2 虚拟化环境，适配 Hypervisor 场景。项目基于 Rust 2024 Edition 开发，采用 `no_std` 设计，完全适用于裸机和嵌入式环境。

## ✨ 功能特性

- 🧠 **完整的AArch64支持**: 针对 ARMv8 架构进行了完整优化，支持现代 ARM64 处理器
- ⚡ **EL2虚拟化支持**: 通过 `hv` 特性启用，支持 Hypervisor 运行环境
- 🔌 **中断处理系统**: 支持 GICv2 和 GICv3 中断控制器
- 🏔 **SMP多核支持**: 支持多核处理器启动和管理
- 🕐 **时间管理系统**: 提供高精度时间管理和定时器支持
- 💾 **内存管理**: 完善的物理内存和虚拟内存管理机制
- 📡 **设备树解析**: 支持设备树(blob)解析，自动识别硬件配置
- 🔧 **电源管理**: 提供基础的电源控制功能
- 📦 **no_std兼容**: 完全不依赖标准库，适用于裸机环境
- 🛡️ **类型安全**: 基于 Rust 语言特性，确保硬件操作的安全性和可靠性

## 🚀 快速开始

### 🛠 环境要求

- Rust 2024 Edition (nightly)
- AArch64 开发环境
- 支持的 ARM64 硬件平台或模拟器 (QEMU)
- ArceOS hypervisor 分支

### 📦 安装步骤

1. 安装 Rust 工具链：

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

2. 添加 AArch64 目标：

```bash
rustup target add aarch64-unknown-none-softfloat
```

3. 克隆项目并初始化：

```bash
git clone <repository-url>
cd axplat-aarch64-dyn
```

### 📝 基本使用

本平台与 ArceOS 及衍生 OS 项目配合使用：

```bash
# 获取 ArceOS hypervisor 分支
git clone https://github.com/arceos-hypervisor/arceos.git
cd arceos

# 把本平台添加到Cargo.toml
[target.'cfg(target_arch = "aarch64")'.dependencies]
axplat-aarch64-dyn = { git = "https://github.com/arceos-hypervisor/axplat-aarch64-dyn.git", features = ["irq"]}

# 配置平台
make APP_FEATURES=dyn MYPLAT=axplat-aarch64-dyn xxx
```

## 📁 项目结构

```
src/
├── lib.rs              # 主入口和核心功能
├── boot.rs             # 系统启动入口
├── init.rs             # 初始化流程
├── mem.rs              # 内存管理
├── irq/                # 中断处理逻辑
│   ├── mod.rs          # 中断模块主文件
│   ├── v2.rs           # GICv2 中断控制器
│   └── v3.rs           # GICv3 中断控制器
├── smp.rs              # 多核处理器支持
├── time.rs             # 时间与定时器
├── power.rs            # 电源管理
├── fdt.rs              # 设备树解析
├── driver.rs           # 设备驱动接口
└── console.rs          # 控制台输出

Cargo.toml              # 项目配置文件
axconfig.toml           # 平台配置文件
build.rs                # 构建脚本
rust-toolchain.toml     # Rust 工具链配置
link.ld                 # 链接脚本
```

## 🧱 核心模块

### 启动初始化

启动模块负责系统启动时的早期初始化工作，包括：
- CPU 状态设置
- 早期页表配置
- 异常和中断处理程序设置
- 早期控制台初始化

### 内存管理

内存管理模块提供：
- 物理内存区域识别和管理
- 虚拟地址到物理地址的转换
- 保留内存区域处理
- MMIO 区域映射

### 中断处理

中断处理模块支持：
- GICv2/GICv3 中断控制器初始化
- 中断使能/禁用控制
- 中断处理程序注册和分发
- IPI (处理器间中断) 支持

### SMP多核支持

SMP模块提供多核处理器支持：
- 次级核心启动流程
- 多核初始化同步
- 核间通信机制

### 时间管理

时间管理模块提供：
- 系统时间获取
- 定时器中断处理
- 高精度时间管理

### 电源控制

电源控制模块提供基础的电源管理功能：
- 系统关机
- 重启支持

### 设备树解析

设备树解析模块：
- 解析设备树(blob)信息
- 提取硬件配置信息
- 中断配置解析

## 🔧 特性支持

| 特性 | 描述 |
|------|------|
| **`hv`** | 启用 EL2 虚拟化支持，适配 Hypervisor 环境 |
| **`smp`** | 启用 SMP 多核支持 |
| **`irq`** | 启用中断处理支持 |
| **`fp-simd`** | 启用浮点和 SIMD 指令支持 |

## 🛠 开发指南

### 构建项目

```bash
# 构建项目
cargo build

# 构建时启用特定特性
cargo build --features "hv,smp,irq"
```

### 配置说明

平台配置文件 `axconfig.toml` 包含以下主要配置项：
- 架构和平台标识
- CPU 核心数量
- 内存布局配置
- 设备规格配置
- 中断号配置

## 📄 许可证

本项目采用 MIT 许可证 - 查看 [LICENSE](LICENSE) 文件了解详情