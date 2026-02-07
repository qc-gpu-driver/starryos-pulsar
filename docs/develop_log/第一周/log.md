# 第一周开发日志

## 工作总结
这周完成了 NPU 驱动开发的准备工作，包括开发板验证、闭源库逆向文档整理、驱动骨架搭建。内容如下：
**开发板验证通过** — 在 OrangePi 5 Plus（RK3588）上部署 StarryOS（NPU 版本），成功跑通 RKNN 推理测例
**闭源库逆向文档整理** — 整理了 `librknnrt.so` 逆向成果、完整寄存器语义、模型推理全链条，发布至 [GitHub Pages](https://qc-gpu-driver.github.io/starryos-pulsar/documents/docs.html)
**驱动骨架搭建** — 创建 `axnpu-rknn` 独立 crate，编写 DTB 设备探测代码（`dtbparse.rs`），通过 `rdrive` 框架自动匹配 `rockchip,rk3588-rknn`
**基础知识补充** — 学习 NPU 原理、AI 模型本质、RKNN Toolkit2 套件与仿真器

---

## 开发板验证

搭载 RK3588 芯片的 **OrangePi 5 Plus** 开发板到了之后，将带有 NPU 支持的 StarryOS 版本（[atomgit 仓库](https://atomgit.com/aios-porting/3906cf9f2ccbf898bd3512d5862ef92e)）部署到开发板上，**成功跑通测例**，验证了 NPU 驱动在 StarryOS 上的可行性。

---

## 闭源库逆向与文档整理

花了 3 天时间系统整理了以下内容：

- **`librknnrt.so` 闭源库逆向成果** — 梳理了用户态库的内部调用流程、ioctl 命令、内存管理机制
- **寄存器语义** — 从官方 TRM 手册和 [RKNN 开源内核驱动](https://github.com/rockchip-linux/kernel)（C 语言版本）中，整理了完整的寄存器描述和语义
- **模型推理全链条** — 从 `rknn_init` 到 `rknn_destroy` 的完整生命周期，包括任务提交流程、状态机、DMA Fence 路径等

> **文档链接**：[https://qc-gpu-driver.github.io/starryos-pulsar/documents/docs.html](https://qc-gpu-driver.github.io/starryos-pulsar/documents/docs.html)

这份文档不仅方便自己开发时查阅，也相当于一份**社区参考文档**，我也会一直维护和验证的，方便更多对 RKNN NPU 驱动感兴趣的开发者查阅和使用。

---

## 驱动骨架编写

最初尝试直接在 `arceos/modules/` 下创建 `axnpu-rknn` 驱动模块，但遇到了较多的**循环依赖问题**（`axdriver → axfs → axmm → axdma → axdriver`），参考 StarryOS（NPU 版本）的做法，将驱动模块移动到 Starry 根目录下独立开发，等完善后再集成回 arceos。

编写了 `dtbparse.rs`，利用 ArceOS 的 `rdrive` 框架通过设备树（DTB）自动探测 NPU 设备：

```rust
module_driver! {
    name: "RKNPU",
    level: ProbeLevel::PostKernel,
    priority: ProbePriority::DEFAULT,
    probe_kinds: &[
        ProbeKind::Fdt {
            compatibles: &["rockchip,rk3588-rknn"],
            on_probe: probe_rknpu
        }
    ],
}

fn probe_rknpu(info: FdtInfo<'_>, dev: PlatformDevice) -> Result<(), OnProbeError> {
    let name = info.node.name();
    // 提取 MMIO 基址: 0xfdab0000, 大小: 0x9000
    let mut regs = info.node.reg().ok_or_else(|| { /* ... */ })?;
    let base_reg = regs.next().ok_or_else(|| { /* ... */ })?;
    let mmio_base = base_reg.address as usize;
    let mmio_size = base_reg.size.unwrap_or(0x9000);
    // 提取中断号: SPI 110, 111, 112（3 个 NPU 核心）
    // ...
    Ok(())
}
```

`module_driver!` 宏会将驱动自动注册到 `rdrive` 框架，内核启动时遍历 DTB，匹配 `compatible = "rockchip,rk3588-rknn"` 后自动调用 `probe_rknpu`，无需手动调用。

---

## 其它

刚接到 NPU 驱动开发任务时，虽然听说过 NPU，但对它实际是什么、解决什么问题、工作原理、如何运行都不了解。花了一天时间系统学习了以下内容：

- **NPU 是什么** — 神经网络处理单元，专为矩阵运算和推理加速设计的硬件
- **AI 模型的本质** — 本质上是一个函数,是大量矩阵乘法配上激活函数。训练是拟合参数，推理是执行前向传播
- **模型格式与转换** — ONNX → RKNN 格式转换，量化，以及模型在 NPU 上的执行流程
- **驱动在整个链条中的角色** — 用户态库（`librknnrt.so`）通过 ioctl 与内核驱动通信，驱动负责任务调度、DMA 搬运、寄存器操作

上 Rockchip 官网下载了 [rknn-toolkit2](https://github.com/airockchip/rknn-toolkit2) 套件，发现里面包含一个 **仿真器**，可以在 x86 主机上模拟 RKNPU 的部分功能。阅读了 demo 代码，理解了 RKNN 推理的基本 API 调用流程：

```
rknn_init()来初始化npu执行上下文 → rknn_inputs_set()读写npu寄存器设置输出输出参数 → rknn_run()写npu寄存器提醒npu开始工作和运行 → rknn_outputs_get()npu通过中断通知操作系统，用户库从约定位置读取结果 → rknn_destroy()销毁释放npu执行上下文
```

---
