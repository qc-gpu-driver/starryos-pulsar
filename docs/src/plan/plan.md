# RK3588 NPU 驱动开发路线图

## 进度总览
> **当前状态**：电源域管理 (Power Domain)
> **目标**：将添加了npu驱动的starry编译拷贝到开发板验证探测

## 实际计划

## 第一周：环境验证与基础初始化

### 1. 理论与环境准备
- [x] **阅读手册与 Demo**
    - [x] 研读 RK3588 技术参考手册中 NPU 章节。
    - [x] 分析 RKNN Toolkit2 的官方 Demo，理解从模型加载到推理的完整数据流
- [x] **实体机环境验证** (基于 StarryOS 比赛版)
    - [x] 编译并烧录比赛版本的 StarryOS 代码到 RK3588 实体板。
    - [x] 成功运行一个简单的文本生成模型，确保硬件和工具链正常。
- [x] **整理 StarryOS 对 RK3588 NPU 驱动的逆向成果**
    - [x] **寄存器地图**
        - [x] 按模块梳理：PC / CORE / CNA / DPU / PPU / SDMA / DDMA / GLOBAL
        - [x] 对每个寄存器记录：offset、字段含义、读写属性、默认值/复位值、关联流程
        - [x] 标注来源：TRM / Linux rknpu 驱动
    - [x] **提交协议与数据结构**
        - [x] 整理 `DRM_IOCTL_RKNPU_*` ioctl 列表与语义（Action / MemCreate / MemMap / MemDestroy / MemSync / Submit）
        - [x] 对齐结构体布局：`rknpu_mem_create` / `rknpu_mem_map` / `rknpu_task` / `rknpu_submit`（字段意义与对齐）
        - [x] 梳理 `mmap(offset)` 规则（handle 与 offset 的编码/解码约定）
    - [x] **任务提交流程（时序 + 状态机）**
        - [x] 从“用户态提交”到“硬件执行完成”的完整时序
        - [x] 失败路径：超时、异常中断状态、非法参数


### 2. 驱动框架搭建 (在StarryOS主线上完成)
- [x] **创建模块骨架**
    - [x] 在StarryOS组织的`arceos` 仓库中创建 `rknpu` 独立模块。
    - [x] 配置 `Cargo.toml` 依赖。
- [x] **NPU探测**
    - [x] 实现 FDT (设备树) 解析，获取 NPU 寄存器基地址和中断号。
- [ ] **电源域管理 (Power Domain)**
    - [ ] 对接 PMU 驱动，按顺序开启 NPU 电源域：
        - [ ] `NPUTOP`
        - [ ] `NPU0` (Core 0)
        - [ ] `NPU1` (Core 1)
        - [ ] `NPU2` (Core 2)

### 3. 用户态接口适配 (DRM)

- [ ] **DRM 框架（StarryOS 主线）**
    - [x] **DRM ioctl 基础解析**
        - [x] 实现 ioctl 编码解析：`ioctl_nr` / `is_driver_ioctl` / `io_size`
    - [ ] **设备节点与基础握手**
        - [ ] `/dev/dri/card0`：实现 `DRM_IOCTL_VERSION`，返回 Rockchip 相关信息以通过初始化探测
        - [ ] `/dev/dri/card1`：实现 `DRM_IOCTL_VERSION`，driver name 为 `rknpu`
    - [ ] **RKNPU driver ioctls（初期基于闭源库验证成果）**
        - [ ] `DRM_IOCTL_RKNPU_MEM_CREATE`：返回 handle + dma_addr（设备可访问地址）
        - [ ] `DRM_IOCTL_RKNPU_MEM_MAP`：返回可用于 `mmap` 的 offset
        - [ ] `mmap`：根据 offset 找回 handle，并映射出可读写的用户态地址
        - [ ] `DRM_IOCTL_RKNPU_SUBMIT`：最小可用的同步提交（先跑通）
        - [ ] `DRM_IOCTL_RKNPU_MEM_DESTROY`：释放 handle 对应的内存对象（避免泄漏）
        - [ ] `DRM_IOCTL_RKNPU_MEM_SYNC`：实现 cache sync 语义（至少对常用 flags 生效）
        - [ ] `DRM_IOCTL_RKNPU_ACTION`：实现最小集合（GetDrvVersion/GetHwVersion/ActReset/GetIommuEn）
    - [ ] **基于逆向成果编写纯驱动（内核态自测），再次验证上述成果**
        - [ ] **目标**：不靠 `librknnrt.so`，自己在内核里“拼出一份 NPU 能执行的命令”，跑通一次计算
        - [ ] **待完成**
---

## 第二周：核心逻辑与任务提交

### 1. 硬件抽象层 (HAL)
- [ ] **寄存器定义**
    - [ ] 封装 NPU 控制寄存器 (PC_DATA, PC_TASK_CONTROL 等)。
    - [ ] 封装中断状态寄存器 (INT_STATUS, INT_CLEAR)。
- [ ] **低级操作封装**
    - [ ] 实现 `npu_reset()` (软复位)。
    - [ ] 实现 `npu_start()` (启动计算)。

### 2. 任务调度逻辑
- [ ] **IOCTL 接口对接**
    - [ ] 在 `StarryOS` 中处理来自 `card1` 的 IOCTL 请求。
    - [ ] 解析用户态传递的 `RKNPU_SUBMIT` 结构体。
- [ ] **任务提交 (Job Submission)**
    - [ ] 申请内核 DMA 内存存放 Command Buffer。
    - [ ] 将用户态指令流写入硬件寄存器。
    - [ ] 触发 NPU 硬件运行。
    - [ ] 任务完成判定：中断状态位/完成标志的明确化（文档与代码一致）
    - [ ] 超时与异常路径：可观测日志 + 合理错误码
    - [ ] 多核最小策略：至少支持 core0/core1/core2 单核提交与轮转

---

## 待定计划 (Backlog)
> *随着开发深入，以下内容将补充细节*

- [ ] **中断处理 (IRQ Handler)**：实现异步等待机制，替代轮询。
- [ ] **IOMMU/SMMU 支持**：实现虚实地址转换，支持非连续内存。
- [ ] **多核调度策略**：如何分配任务给 3 个 NPU 核心。