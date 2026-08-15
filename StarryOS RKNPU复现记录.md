****

**StarryOS RKNPU** **统一仓库复现说明**

用于记录 starryos-pulsar 与 rknpu-starry-adapter 的统一整合过程，明确成功复现所采用的历史版本、整合后的目录职责、Cargo 配置修改以及调度优先级测试修正。版本关系以 GitHub 提交记录为准，配置内容以下载的仓库的 Cargo.toml、Cargo.lock 和测试代码为准。

当前统一仓库已经完成编译、仿真测试和 RK3588 板端运行验证。调度测试修正后，仿真测试能够全部通过，后续开发可以基于该版本继续进行。

## 1. 复现成功使用的版本

本次统一仓库复现以 starryos-pulsar 的整合提交 4aee945… 为主基线；该提交中的适配层指针固定到 rknpu-starry-adapter 的 af2cefe…。

| **仓库**                           | **版本**      | **用途与结论**                                               |
| ---------------------------------- | ------------- | ------------------------------------------------------------ |
| qc-gpu-driver/starryos-pulsar      | 4aee945ad87c… | 主仓库整合版本；加入 RKNPU、适配层及 StarryOS 接入配置。     |
| qc-gpu-driver/rknpu-starry-adapter | af2cefe81c02… | 成功复现采用的适配层基线；其本地 path 依赖与统一仓库目录一致。 |
| LM-cell/RKNPU                      | 个人统一仓库  | 后续驱动分析、测试修正与板端实验的开发基线。                 |

 

## 2. 整合后的目录与职责

```
starryos-pulsar/
 ├── StarryOS/           # 内核、设备接口与 ioctl 接入
 ├── drivers/rknpu/        # RKNPU 核心驱动、任务与调度逻辑
 ├── drivers/rknpu-starry-adapter/ # StarryOS 与 RKNPU 的桥接层
 ├── drivers/rknpu-regs/      # NPU 寄存器描述与访问
 ├── drivers/rockchip-pm/     # RK3588 电源域支持
 ├── drivers/rk3588-clk/      # RK3588 时钟支持
 ├── rdrive/            	  # 设备驱动框架
 └── demo/npu_benchmark/      # 单核、三核与并发测试
```

rknpu 负责寄存器访问、任务描述、内存管理和调度等核心功能；rknpu-starry-adapter 负责将 StarryOS 的设备节点、DRM/RKNPU ioctl、用户内存、DMA 映射、IRQ、等待机制和后台 Worker 接入核心驱动。

## 3. Cargo 配置修改

| 配置文件                                  | 主要修改                                                     |
| ----------------------------------------- | ------------------------------------------------------------ |
| `StarryOS/Cargo.toml`                     | 将原先通过 Git 获取的 `rknpu` 改为统一仓库内的相对路径依赖 `../drivers/rknpu`；加入 `../drivers/rknpu-starry-adapter`，并仅在 `aarch64` 目标下启用适配层；通过 `[patch.crates-io]` 将 `rdrive`、`rdif-*` 和 `pcie` 统一指向仓库内源码。 |
| `StarryOS/api/Cargo.toml`                 | 移除 API 层对 `rdrive` 和 `rknpu` 的直接通用依赖，改为在 `aarch64` 目标下通过 `../../drivers/rknpu-starry-adapter` 接入 RKNPU。 |
| `StarryOS/crates/axdriver-dyn/Cargo.toml` | 将原先通过 Git 获取的 `rockchip-pm` 和 `rk3588-clk` 改为统一仓库内的相对路径依赖 `../../../drivers/rockchip-pm` 和 `../../../drivers/rk3588-clk`；同时移除对 `rknpu` Git 仓库的直接依赖。 |
| `drivers/rknpu/Cargo.toml`                | 移除 `starryos` feature 以及 `rdrive`、`rockchip-pm`、`rk3588-clk`、`axklib`、`axtask` 等平台相关可选依赖，将 StarryOS 平台接入职责转移到适配层。 |
| `drivers/rknpu-starry-adapter/Cargo.toml` | 按统一仓库目录使用相对路径依赖，包括 `../rknpu`、`../../rdrive/rdrive`、`../rockchip-pm` 和 `../../StarryOS/...`；`axerrno` 保持 `0.1`。 |
| `StarryOS/Cargo.lock`                     | 根据统一仓库内的 `path` 和 `[patch.crates-io]` 配置重新解析依赖，并加入 `rknpu-starry-adapter`。 |

改为本地依赖”是指：Cargo 不再从 GitHub 或 crates.io 下载源码，而是直接使用统一仓库中指定目录的源码，通过 `path` 配置引用。因为如果有的文件因为网络问题下载不下来 ，只能下载到本地进行。 

本文中的 `path` 均为相对于对应 `Cargo.toml` 的仓库内相对路径，以统一仓库的目录结构为准。

## 4. 调度测试修正

测试 running_submit_is_prioritized_before_new_ready_submit 的目标是：旧 Submit 已经从 Ready 进入 Running 后，即使新来的 Ready Submit 优先级数值更高，空闲核心仍应优先继续推进 Running Submit。要验证这一策略，必须先把旧 Submit 入队并向 core0 派发第一条 lane，再加入新 Submit。

```
let running_id = queue.enqueue(running);
 
 // 旧 Submit 先获得 core0，真实进入 Running
 assert_eq!(
   queue.reserve_next_dispatch(0).unwrap().task_id,
   running_id
 );
 
 // 此后才加入新的高优先级 Ready Submit
 let ready_id = queue.enqueue(ready);
 
 // core1 应继续选择 Running Submit
 assert_eq!(
   queue.reserve_next_dispatch(1).unwrap().task_id,
   running_id
 );
 assert_ne!(running_id, ready_id);
```

**对应提交说明** 公开提交 8a6dc53… 记录了测试调整意图，但代码先使用 running_id、后定义并入队，顺序仍然错误。调度测试修正后，仿真测试能够全部通过。

# 5. 参考仓库与提交

• [starryos-pulsar 上游仓库](https://github.com/qc-gpu-driver/starryos-pulsar)

• [主仓库整合提交 4aee945…](https://github.com/qc-gpu-driver/starryos-pulsar/commit/4aee945ad87cddf30d56b693bc4de0181eb0598a) — 本次成功复现主基线

• [rknpu-starry-adapter 上游仓库](https://github.com/qc-gpu-driver/rknpu-starry-adapter)

• [适配层代码基线 af2cefe…](https://github.com/qc-gpu-driver/rknpu-starry-adapter/commit/af2cefe81c02ae187decf09c844d551ca7e19b28) — 本次实际采用

• [个人统一仓库 LM-cell/RKNPU](https://github.com/LM-cell/RKNPU)—个人已经整理好的可开发版本

• [调度测试公开调整记录 8a6dc53…](https://github.com/LM-cell/RKNPU/commit/8a6dc5300017383a87dc1b66b18ad921b4d3f9d8) — 调度测试的修正