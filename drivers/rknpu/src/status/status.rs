//! RKNPU 任务级状态闭包。
//!
//! 这里定义的类型只覆盖“恢复某个正在执行的 NPU 任务”所必需的驱动动态态，
//! 不试图镜像整个设备生命周期。
//!
//! 完整性约束：
//! - 仅保存 [`crate::ioctrl::RknpuSubmit`] 和 IRQ 状态仍然不完整。
//! - 必须额外保存执行进度游标、active core/subcore 映射、owner 绑定，
//!   以及 `PC_TASK_STATUS` / `*_S_STATUS` / `*_S_POINTER` 这类硬件自推进寄存器，
//!   才能覆盖 IRQ 边界恢复所需的最小闭包。

use alloc::vec::Vec;
use core::array;

use super::NpuOwnerIds;

/// RK3588 RKNPU 当前支持的最大核心数。
pub const NPU_MAX_CORES: usize = 3;

/// 无效 core/subcore 槽位标记。
pub const INVALID_CORE_SLOT: u8 = 0xff;

/// NPU 上下文的粗粒度阶段。
///
/// 这些状态只描述“某个任务的 NPU 恢复闭包”处于什么阶段，不描述电源、
/// workqueue 或 PM runtime 等设备外层状态。
#[repr(C)]
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub enum NpuCtxState {
    /// 默认空态，还未和任何提交任务绑定。
    #[default]
    Empty,
    /// 已构造上下文，但尚未真正启动硬件执行。
    Prepared,
    /// NPU 当前正在执行该上下文对应的任务流。
    Running,
    /// 已请求在下一个 IRQ 边界保存，但硬件尚未到达边界。
    WaitingIrqBoundary,
    /// 已到达 IRQ 边界，允许切走 owner 并保存闭包。
    BoundaryReady,
    /// 所需状态已完整保存在软件侧，后续可恢复。
    Saved,
    /// 正在把软件侧保存的闭包重新装回 NPU 执行流。
    Resuming,
    /// 整个任务流已完成，不再需要恢复。
    Completed,
    /// 当前执行流出现错误，软件侧闭包保留为故障快照。
    Faulted,
    /// 发生 timeout/soft reset 后，原执行流已丢失，不能保证精确恢复。
    ResetLost,
}

/// 提交批次级执行进度。
///
/// 这些字段直接对应当前 `submit_ioctrl()` 同步循环里的批次游标：
/// - 来源：`task_iter`、`task_iter_end`、`task_batch`
/// - 更新时间：每轮批次启动前/完成后
/// - 恢复用途：确定恢复时从哪一个任务索引继续重放剩余 descriptor
#[repr(C)]
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct NpuBatchProgress {
    /// 当前批次全局游标。
    ///
    /// 来源：`submit_ioctrl()` 局部变量 `task_iter`
    /// 更新时间：每次推进批次后
    /// 恢复用途：表示下一次需要提交的全局任务索引。
    pub task_iter: u32,
    /// 本次提交流的全局结束位置。
    ///
    /// 来源：`submit_ioctrl()` 局部变量 `task_iter_end`
    /// 更新时间：构造 active subcore 后一次性确定
    /// 恢复用途：用于判断是否还有剩余任务需要恢复。
    pub task_iter_end: u32,
    /// 当前批次的起始索引。
    ///
    /// 来源：当前轮 `task_iter`
    /// 更新时间：每轮批次开始前
    /// 恢复用途：重建当前批次 descriptor 范围。
    pub current_batch_start: u32,
    /// 当前批次包含的任务数。
    ///
    /// 来源：当前轮 `task_batch`
    /// 更新时间：每轮批次开始前
    /// 恢复用途：恢复时重放该批次剩余任务。
    pub current_batch_count: u32,
    /// 已确认完成的任务总数。
    ///
    /// 来源：由 `task_iter` 或 `task_counter` 快照推导
    /// 更新时间：每轮批次完成或 IRQ 边界到达时
    /// 恢复用途：定位最后一个稳定保存点。
    pub completed_task_count: u32,
    /// 尚未完成的任务总数。
    ///
    /// 来源：`task_iter_end - task_iter`
    /// 更新时间：每轮批次推进后
    /// 恢复用途：判断是否还能继续恢复剩余任务流。
    pub remaining_task_count: u32,
}

/// PC 模块的任务级寄存器快照。
///
/// 这组寄存器不是“设备全局生命周期”状态，而是当前任务流在某个硬件 core
/// 上的提交入口和推进游标。TRM 第 36.4.2 节把它们定义为 PC block 的主控制面。
#[repr(C)]
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct NpuPcRegisterSnapshot {
    /// `RKNN_pc_operation_enable` 的最近一次软件写值。
    ///
    /// 来源：提交/恢复路径写 PC 启动脉冲时的软件镜像
    /// 更新时间：每次写 `operation_enable` 前后
    /// 恢复用途：恢复时重放“重新启动 PC 拉取 descriptor”的最后一步。
    pub operation_enable: u32,
    /// `RKNN_pc_base_address` 原始值。
    ///
    /// 来源：TRM `RKNN_pc_base_address`
    /// 更新时间：每次批次重编程 PC 时
    /// 恢复用途：恢复 descriptor 拉取入口，不依赖外层临时局部变量。
    pub base_address: u32,
    /// `RKNN_pc_register_amounts` 原始值。
    ///
    /// 来源：TRM `RKNN_pc_register_amounts`
    /// 更新时间：每次批次重编程 PC 时
    /// 恢复用途：恢复“每个 task 需要拉取多少 register cmd”。
    pub register_amounts: u32,
    /// `RKNN_pc_interrupt_mask` 原始值。
    ///
    /// 来源：TRM `RKNN_pc_interrupt_mask`
    /// 更新时间：每次批次启动前
    /// 恢复用途：保证恢复后等待的仍是切换出去时那一组中断语义。
    pub interrupt_mask: u32,
    /// `RKNN_pc_interrupt_status` 快照。
    ///
    /// 来源：TRM `RKNN_pc_interrupt_status`
    /// 更新时间：到达保存点或处理中断时读取
    /// 恢复用途：判断哪些事件已经到达稳定边界，避免重复消费旧事件。
    pub interrupt_status: u32,
    /// `RKNN_pc_interrupt_raw_status` 快照。
    ///
    /// 来源：TRM `RKNN_pc_interrupt_raw_status`
    /// 更新时间：到达保存点时读取
    /// 恢复用途：区分 mask 后状态与硬件原始事件，辅助边界恢复和故障诊断。
    pub interrupt_raw_status: u32,
    /// `RKNN_pc_task_con` 原始值。
    ///
    /// 来源：TRM `RKNN_pc_task_con`
    /// 更新时间：每次批次重编程 PC 时
    /// 恢复用途：恢复 task 数量和 ping-pong 工作方式。
    pub task_con: u32,
    /// `RKNN_pc_task_dma_base_addr` 原始值。
    ///
    /// 来源：TRM `RKNN_pc_task_dma_base_addr`
    /// 更新时间：每次批次重编程 PC 时
    /// 恢复用途：恢复 feature/weight/DPU/PPU DMA 地址基底。
    pub task_dma_base_addr: u32,
    /// `RKNN_pc_task_status` 快照。
    ///
    /// 来源：TRM `RKNN_pc_task_status`
    /// 更新时间：到达保存点时读取
    /// 恢复用途：保存当前 task counter、首尾 task operating/fetching 状态；
    /// 这是 `RknpuSubmit + IRQ 状态` 之外必须额外补上的硬件进度寄存器。
    pub task_status: u32,
}

/// 各计算/搬运 block 的 ping-pong 寄存器快照。
///
/// TRM 明确指出 `*_s_status` 和 `*_s_pointer` 由硬件在执行过程中推进。
/// 即使驱动没有在每次切换时显式重写，它们也与当前任务流紧密绑定。
#[repr(C)]
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct NpuBlockRegisterSnapshot {
    /// `*_s_status` 快照。
    ///
    /// 来源：TRM `*_s_status`
    /// 更新时间：到达保存点时读取
    /// 恢复用途：判断 executer 0/1 哪组正在执行、哪组在等待；
    /// 这是 ping-pong 执行流的硬件真状态，不可由 `RknpuSubmit` 单独推回。
    pub s_status: u32,
    /// `*_s_pointer` 快照。
    ///
    /// 来源：TRM `*_s_pointer`
    /// 更新时间：提交配置或到达保存点时读取
    /// 恢复用途：恢复 pointer/executer 组选择、ping-pong 使能与模式。
    pub s_pointer: u32,
    /// `*_operation_enable` 快照或最近一次软件写镜像。
    ///
    /// 来源：TRM `*_operation_enable`
    /// 更新时间：块启动或到达保存点时
    /// 恢复用途：恢复各 block 的 enable 闭包，保证恢复后的块级启动关系一致。
    pub operation_enable: u32,
}

/// GLOBAL block 的任务级快照。
#[repr(C)]
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct NpuGlobalRegisterSnapshot {
    /// `GLOBAL.OPERATION_ENABLE` 快照。
    ///
    /// 来源：TRM `Combine Operation Enable`
    /// 更新时间：组合执行流启动或到达保存点时
    /// 恢复用途：恢复跨 block 的 combine-op 使能闭包，避免只恢复单块状态。
    pub operation_enable: u32,
}

/// 单个硬件 core 的可见寄存器闭包。
///
/// 完整性结论：
/// - 仅保存 `RknpuSubmit + irq 状态` 仍不完整；
/// - 还必须补上 `PC_TASK_STATUS`、各 block `S_STATUS/S_POINTER` 这类
///   会被硬件自推进的寄存器；
/// - 再叠加执行进度游标、active core/subcore 映射和 owner 绑定后，
///   才能覆盖 IRQ 边界恢复所需的最小闭包。
#[repr(C)]
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct NpuCoreRegisterSnapshot {
    /// PC block 快照。
    pub pc: NpuPcRegisterSnapshot,
    /// CNA block 快照。
    pub cna: NpuBlockRegisterSnapshot,
    /// MAC/CORE block 快照。
    pub mac: NpuBlockRegisterSnapshot,
    /// DPU block 快照。
    pub dpu: NpuBlockRegisterSnapshot,
    /// DPU_RDMA block 快照。
    pub dpu_rdma: NpuBlockRegisterSnapshot,
    /// PPU block 快照。
    pub ppu: NpuBlockRegisterSnapshot,
    /// PPU_RDMA block 快照。
    pub ppu_rdma: NpuBlockRegisterSnapshot,
    /// GLOBAL block 快照。
    pub global: NpuGlobalRegisterSnapshot,
}

/// 单核心视角下的驱动动态态。
///
/// 这些字段对应“某个 core 当前替哪个 subcore 槽跑到哪里了”。
/// 这里不保存整个 `RknpuSubmit`，只保存恢复本 core 执行流需要的局部快照。
#[repr(C)]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NpuCoreDriverStatus {
    /// 硬件核心槽位。
    ///
    /// 来源：active core 集合和 `self.base[idx]`
    /// 更新时间：建立 active core/subcore 映射时
    /// 恢复用途：决定恢复时把状态装回哪个硬件 core。
    pub core_slot: u8,
    /// 对应的 `subcore_task[]` 槽位。
    ///
    /// 来源：active subcore 选择结果
    /// 更新时间：建立 active core/subcore 映射时
    /// 恢复用途：从原始 `RknpuSubmit::subcore_task[]` 中恢复该 core 的任务范围。
    pub subcore_slot: u8,
    /// 该核心是否参与了当前提交。
    ///
    /// 来源：`subcore_task[idx].task_number > 0`
    /// 更新时间：建立 active core 集合时
    /// 恢复用途：跳过未参与本次任务流的 core。
    pub enabled: bool,
    /// 该核心当前是否持有 in-flight 执行流。
    ///
    /// 来源：批次已下发但尚未到稳定边界
    /// 更新时间：启动批次后置位，保存/完成后清除
    /// 恢复用途：区分“仅配置过”与“仍在执行中”。
    pub inflight: bool,
    /// 该核心是否正在等待 IRQ 完成。
    ///
    /// 来源：`wait_all_npucore()` 进入等待后
    /// 更新时间：批次启动后置位，看到预期 IRQ 后清除
    /// 恢复用途：恢复时判断该 core 是否应继续从 IRQ 边界推进。
    pub waiting_irq: bool,
    /// 是否已经观察到该核心的 IRQ。
    ///
    /// 来源：`irq_status` 原子或轮询寄存器
    /// 更新时间：检测到预期 IRQ 时
    /// 恢复用途：作为边界到达的 checkpoint 标记。
    pub irq_seen: bool,
    /// 驱动是否已清除该核心的硬件中断。
    ///
    /// 来源：`clean_interrupts()`
    /// 更新时间：执行中断清理后
    /// 恢复用途：避免恢复时重复依赖旧的 pending 中断。
    pub irq_cleared: bool,
    /// 是否已到达允许切换的 IRQ 边界。
    ///
    /// 来源：预期 IRQ 完成并通过状态校验后
    /// 更新时间：边界到达时
    /// 恢复用途：只有该位成立时，当前 core 的执行流才可安全保存。
    pub boundary_ready: bool,
    /// 该核心视角下是否发生故障。
    ///
    /// 来源：IRQ 状态不匹配或等待失败
    /// 更新时间：检测到异常时
    /// 恢复用途：阻止把故障态误判为可恢复态。
    pub faulted: bool,
    /// 该核心当前处理到的任务索引。
    ///
    /// 来源：批次内当前 `task_iter + core offset`
    /// 更新时间：每轮批次开始前
    /// 恢复用途：定位当前 core 对应的 descriptor。
    pub current_task_index: u32,
    /// 该核心当前批次起始索引。
    ///
    /// 来源：当前轮批次起点
    /// 更新时间：每轮批次开始前
    /// 恢复用途：恢复该 core 的局部批次范围。
    pub batch_task_start: u32,
    /// 该核心当前批次任务数。
    ///
    /// 来源：当前轮批次对该 core 分配的任务数
    /// 更新时间：每轮批次开始前
    /// 恢复用途：重放该 core 尚未完成的 descriptor。
    pub batch_task_count: u32,
    /// 该核心已完成任务数。
    ///
    /// 来源：等待路径和 `task_counter` 快照
    /// 更新时间：每次 core 完成边界后
    /// 恢复用途：判断该 core 还剩多少任务未恢复。
    pub completed_task_count: u32,
    /// 该核心剩余任务数。
    ///
    /// 来源：局部批次范围减去完成数
    /// 更新时间：每次 core 推进后
    /// 恢复用途：判断恢复时是否还需继续下发该 core。
    pub remaining_task_count: u32,
    /// 该核心期待的完成中断掩码。
    ///
    /// 来源：当前批次 `RknpuTask::int_mask`
    /// 更新时间：每轮批次开始前
    /// 恢复用途：恢复后继续等待正确的 IRQ 边界。
    pub expected_int_mask: u32,
    /// 驱动最近一次观察到的原始 IRQ 状态。
    ///
    /// 来源：`irq_status` 原子或轮询寄存器
    /// 更新时间：每次看到中断状态变化时
    /// 恢复用途：作为边界快照和故障分析信息。
    pub observed_irq_status: u32,
    /// 最近一个任务描述符写回的 `int_status`。
    ///
    /// 来源：`submit_tasks[idx].int_status`
    /// 更新时间：确认批次完成并回写 task 时
    /// 恢复用途：保持用户态/驱动态对最后完成状态的一致视图。
    pub last_task_int_status: u32,
    /// 当前 core 的任务级硬件寄存器快照。
    ///
    /// 来源：TRM 第 36.4.2 节定义的 PC/CNA/MAC/DPU/DPU_RDMA/PPU/PPU_RDMA/GLOBAL
    /// 可见寄存器，以及驱动对这些寄存器的最后一次软件写镜像
    /// 更新时间：每次建立保存点时
    /// 恢复用途：补全 `RknpuSubmit` 之外那些由硬件自推进、却仍与任务强绑定的状态。
    pub regs: NpuCoreRegisterSnapshot,
}

impl Default for NpuCoreDriverStatus {
    fn default() -> Self {
        Self {
            core_slot: INVALID_CORE_SLOT,
            subcore_slot: INVALID_CORE_SLOT,
            enabled: false,
            inflight: false,
            waiting_irq: false,
            irq_seen: false,
            irq_cleared: false,
            boundary_ready: false,
            faulted: false,
            current_task_index: 0,
            batch_task_start: 0,
            batch_task_count: 0,
            completed_task_count: 0,
            remaining_task_count: 0,
            expected_int_mask: 0,
            observed_irq_status: 0,
            last_task_int_status: 0,
            regs: NpuCoreRegisterSnapshot::default(),
        }
    }
}

/// 驱动层和当前任务执行流动态强相关的最小闭包。
///
/// 该结构不重复存储整包 [`crate::ioctrl::RknpuSubmit`]，而是专门保存：
/// - 提交执行过程中的局部进度游标
/// - active core/subcore 映射
/// - 等待 IRQ 路径得到的边界状态
/// - 外层 task/process/address-space 绑定
///
/// 这组字段和 `RknpuSubmit` 一起，构成 IRQ 边界恢复所需的最小闭包。
#[repr(C)]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NPUDriverStatus {
    /// 当前上下文阶段。
    ///
    /// 来源：驱动状态机判断结果
    /// 更新时间：提交、到达边界、保存、恢复、完成或故障时
    /// 恢复用途：决定当前闭包是否允许恢复。
    pub ctx_state: NpuCtxState,
    /// 当前是否已经拥有稳定保存点。
    ///
    /// 来源：IRQ 边界是否到达并完成快照
    /// 更新时间：边界保存成功时
    /// 恢复用途：阻止从非稳定执行点恢复。
    pub savepoint_valid: bool,
    /// 当前闭包是否还能恢复继续运行。
    ///
    /// 来源：状态机结论
    /// 更新时间：故障、reset 或完成时
    /// 恢复用途：快速拒绝不可恢复上下文。
    pub can_resume: bool,
    /// 外层 owner 任务 ID。
    ///
    /// 来源：外层调度器/TCB
    /// 更新时间：上下文绑定到任务时
    /// 恢复用途：把 NPU 闭包重新归还给正确的任务。
    pub owner_task_id: u64,
    /// 外层 owner 进程 ID。
    ///
    /// 来源：外层进程管理器
    /// 更新时间：上下文绑定到进程时
    /// 恢复用途：进程级资源回收和审计。
    pub owner_process_id: u64,
    /// 外层 owner 地址空间 ID。
    ///
    /// 来源：外层 mm/asid 管理器
    /// 更新时间：绑定地址空间时
    /// 恢复用途：恢复前确认地址空间和 DMA 映射归属。
    pub owner_address_space_id: u64,
    /// 当前任务流绑定的 IOMMU domain。
    ///
    /// 来源：`RknpuSubmit::iommu_domain_id`
    /// 更新时间：接收 submit 后镜像一次
    /// 恢复用途：恢复时快速切回正确地址空间，不依赖外层 submit 解引用。
    pub iommu_domain_id: u32,
    /// 当前任务流使用的 core mask。
    ///
    /// 来源：`RknpuSubmit::core_mask`
    /// 更新时间：接收 submit 后镜像一次
    /// 恢复用途：恢复时快速重建 active core 集合。
    pub core_mask: u32,
    /// 参与当前执行流的 active core 数量。
    ///
    /// 来源：`subcore_task[]` 过滤结果
    /// 更新时间：构造 active subcore/core 映射时
    /// 恢复用途：恢复时重建批次与等待范围。
    pub active_core_count: u8,
    /// 当前等待路径从哪个 core 索引开始检查。
    ///
    /// 来源：`wait_all_npucore()` 局部变量 `wait_start_idx`
    /// 更新时间：进入等待路径时
    /// 恢复用途：恢复等待逻辑时保持相同的 core 扫描起点。
    pub wait_start_idx: u8,
    /// 当前仍由执行流持有的核心集合。
    ///
    /// 来源：active core 集合和 inflight 状态
    /// 更新时间：批次启动/保存/完成时
    /// 恢复用途：判断哪些 core 还需要重新装载状态。
    pub running_core_mask: u32,
    /// 当前处于等待 IRQ 状态的核心集合。
    ///
    /// 来源：`waiting_irq` 汇总
    /// 更新时间：进入和退出等待路径时
    /// 恢复用途：恢复后继续在相同 core 上等待边界。
    pub waiting_core_mask: u32,
    /// 已完成预期 IRQ 的核心集合。
    ///
    /// 来源：`irq_seen` 汇总
    /// 更新时间：每个 core 到达边界时
    /// 恢复用途：识别哪些 core 已可安全保存。
    pub irq_done_core_mask: u32,
    /// 提交路径对外可见的任务计数快照。
    ///
    /// 来源：`RknpuSubmit::task_counter`
    /// 更新时间：边界到达或提交结束时
    /// 恢复用途：把恢复前后的任务完成进度保持一致。
    pub task_counter_snapshot: u32,
    /// 硬件耗时快照。
    ///
    /// 来源：`RknpuSubmit::hw_elapse_time`
    /// 更新时间：边界到达或提交结束时
    /// 恢复用途：保留恢复前的执行时间统计。
    pub hw_elapse_time_snapshot: i64,
    /// 批次级全局执行进度。
    pub batch: NpuBatchProgress,
    /// 每个硬件 core 的局部快照。
    pub cores: [NpuCoreDriverStatus; NPU_MAX_CORES],
}

impl Default for NPUDriverStatus {
    fn default() -> Self {
        Self {
            ctx_state: NpuCtxState::Empty,
            savepoint_valid: false,
            can_resume: false,
            owner_task_id: 0,
            owner_process_id: 0,
            owner_address_space_id: 0,
            iommu_domain_id: 0,
            core_mask: 0,
            active_core_count: 0,
            wait_start_idx: 0,
            running_core_mask: 0,
            waiting_core_mask: 0,
            irq_done_core_mask: 0,
            task_counter_snapshot: 0,
            hw_elapse_time_snapshot: 0,
            batch: NpuBatchProgress::default(),
            cores: array::from_fn(|_| NpuCoreDriverStatus::default()),
        }
    }
}

/// 一条可回写的任务级寄存器写入。
///
/// 这里保存的是“来自 regcmd、可在 IRQ 边界后重新覆盖回硬件”的那部分任务窗口，
/// 不包含 `*_s_status` / `interrupt_status` 这类只读观察态寄存器。
#[repr(C)]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct NpuTaskRegisterWrite {
    /// 寄存器块 opcode 的 block 部分。
    pub block: u16,
    /// 绝对寄存器偏移（例如 `0x100c` / `0x4020`）。
    pub offset: u16,
    /// 需要回写的任务配置值。
    pub value: u32,
}

/// 单核心“可恢复镜像”。
///
/// 设计要点：
/// - `task_shadow_writes` 保存 task regcmd 里的可回写任务窗口；
/// - `*_s_pointer` / `pc_task_con` 会在恢复前做消毒，避免 W1C/clear 位被错误重放；
/// - 所有 `operation_enable` 只恢复为 0，不在 IRQ 边界重放启动脉冲。
#[repr(C)]
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct NpuCoreRestoreImage {
    pub pc_operation_enable: u32,
    pub pc_base_address: u32,
    pub pc_register_amounts: u32,
    pub pc_interrupt_mask: u32,
    pub pc_task_con: u32,
    pub pc_task_dma_base_addr: u32,
    pub cna_s_pointer: u32,
    pub mac_s_pointer: u32,
    pub dpu_s_pointer: u32,
    pub dpu_rdma_s_pointer: u32,
    pub ppu_s_pointer: u32,
    pub ppu_rdma_s_pointer: u32,
    pub cna_operation_enable: u32,
    pub mac_operation_enable: u32,
    pub dpu_operation_enable: u32,
    pub dpu_rdma_operation_enable: u32,
    pub ppu_operation_enable: u32,
    pub ppu_rdma_operation_enable: u32,
    pub global_operation_enable: u32,
    pub unsafe_snapshot: bool,
    pub task_shadow_writes: Vec<NpuTaskRegisterWrite>,
}

/// `(owner, core)` 级别的最近一次 IRQ 边界快照索引键。
#[repr(C)]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, PartialOrd, Ord)]
pub struct TaskNpuStateKey {
    pub owner: NpuOwnerIds,
    pub core_slot: u8,
}

/// 当前正在某个硬件 core 上运行的 owner/batch/task 绑定。
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ActiveCoreBinding {
    pub key: TaskNpuStateKey,
    pub subcore_slot: u8,
    pub batch_task_start: u32,
    pub batch_task_count: u32,
    pub current_task_index: u32,
    pub expected_irq_mask: u32,
}

impl Default for ActiveCoreBinding {
    fn default() -> Self {
        Self {
            key: TaskNpuStateKey {
                owner: NpuOwnerIds::default(),
                core_slot: INVALID_CORE_SLOT,
            },
            subcore_slot: INVALID_CORE_SLOT,
            batch_task_start: 0,
            batch_task_count: 0,
            current_task_index: 0,
            expected_irq_mask: 0,
        }
    }
}

/// 驱动共享 `task_npu_state` map 里的最近一次任务级快照。
#[repr(C)]
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct TaskNpuState {
    pub key: TaskNpuStateKey,
    pub subcore_slot: u8,
    pub batch_task_start: u32,
    pub batch_task_count: u32,
    pub current_task_index: u32,
    pub expected_irq_mask: u32,
    pub observed_irq_status: u32,
    pub last_task_int_status: u32,
    pub regs: NpuCoreRegisterSnapshot,
    pub restore_image: NpuCoreRestoreImage,
    pub restore_verified: bool,
    pub restore_mismatch_mask: u64,
}
