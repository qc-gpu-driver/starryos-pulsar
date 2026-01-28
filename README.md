# [基于国产 AI 芯片的 Rust 内核组件开发赛](https://competition.atomgit.com/competitionInfo?id=7f4ba773dfa6396f824a3074192ebcde#heading-0-0)

## 目录

```
.
├── imgs
├── demo                    # NPU相关的应用示例源码
│   ├── npu_benchmark       # benchmark 测试程序源码
│   ├── npu_llama           # LLAMA 测试程序源码
│   └── yolov8              # Yolo v8 模型源码
├── deploy                  # 部署工具及镜像
│   ├── config              # 烧写配置文件
│   ├── images              # 磁盘（根文件系统）镜像
│   ├── loader              # 引导程序镜像
│   └── RKDevTool_v3.37     # 镜像烧写工具
├── axcpu                   # 扩展完善后的 axcpu 组件
├── axplat-aarch64-dyn      # 动态平台适配层
├── drivers                 # 为适配香橙派而实现的驱动源码
│   ├── arm-scmi            # 时钟驱动源码
│   ├── rk3588-clk          # 时钟驱动源码
│   ├── rknpu               # NPU 驱动源码
│   ├── rockchip-pm         # 电源驱动源码
│   ├── sdmmc               # 存储驱动源码
│   └── some-serial         # 串口驱动源码
├── rdrive                  # Rust 动态驱动框架
├── ostool                  # 为了方便在开发板上调试而开发的辅助调试工具源码
├── StarryOS                # 适配后的香橙派上可运行的 StarryOS 源码
├── license
├── README.md
├── 决赛-技术报告.md
├── 决赛-源代码.md
└── 初赛-文档.md
```

## 文档

- [初赛文档](./初赛-文档.md) - 项目初期方案设计

- [决赛技术报告](./决赛-技术报告.md) - 完整技术实现报告

- 演示视频 - 仓库中无法直接播放视频，下载仓库后详见 `./imgs` 目录下的 `*.mp4` 文件

## 性能测试

演示视频参见 `imgs\perf.mp4`，执行结果如下：

<img src="./imgs/perf.png" width="50%" height="50%">

## 功能测试

演示视频参见 `imgs/llama.mp4`，执行结果如下：

<img src="./imgs/llama.png" width="50%" height="50%">

演示视频参见 `imgs/yolov8.mp4`，执行结果如下：

<img src="./imgs/yolov8_result.png" width="50%" height="50%">
