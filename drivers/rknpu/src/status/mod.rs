//! RKNPU 在线状态读取与任务级状态模型。

use alloc::vec::Vec;
use core::{array, ptr, sync::atomic::Ordering};

use crate::{Rknpu, RknpuTask, ioctrl::RknpuSubmit, registers::RknpuCore};

pub mod status;

pub use status::*;

/// 外层任务/进程/地址空间的 opaque owner 绑定。
#[repr(C)]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, PartialOrd, Ord)]
pub struct NpuOwnerIds {
    pub task_id: u64,
    pub process_id: u64,
    pub address_space_id: u64,
}

impl Rknpu {
    /// 读取当前 live 硬件/驱动态，并与调用者提供的 submit/owner 组合成任务级闭包。
    pub fn read_npu_driver_status(
        &self,
        submit: &RknpuSubmit,
        owner: NpuOwnerIds,
    ) -> NPUDriverStatus {
        let mapped_subcore_slots =
            collect_active_subcore_slots(submit, self.base.len().min(NPU_MAX_CORES));
        let active_core_count = mapped_subcore_slots.len() as u8;
        let task_count = submit.task_number as usize;
        let submit_tasks = read_submit_tasks(submit);
        let completed_task_count = completed_task_count(submit_tasks);

        let mut live_regs: [Option<NpuCoreRegisterSnapshot>; NPU_MAX_CORES] =
            array::from_fn(|_| None);
        let mut observed_irq_status = [0_u32; NPU_MAX_CORES];
        let mut inflight_flags = [false; NPU_MAX_CORES];

        for core_slot in 0..mapped_subcore_slots.len() {
            let regs = read_core_register_snapshot(&self.base[core_slot]);
            inflight_flags[core_slot] = is_inflight_snapshot(&regs);
            observed_irq_status[core_slot] =
                self.base[core_slot].irq_status.load(Ordering::Acquire);
            live_regs[core_slot] = Some(regs);
        }

        let inflight_core_count = inflight_flags[..mapped_subcore_slots.len()]
            .iter()
            .filter(|&&flag| flag)
            .count() as u8;
        let batch = derive_progress(
            task_count,
            completed_task_count,
            active_core_count,
            inflight_core_count,
        );

        let mut cores = array::from_fn(|_| NpuCoreDriverStatus::default());
        for core_slot in 0..mapped_subcore_slots.len() {
            cores[core_slot] = derive_core_status(
                core_slot,
                Some(mapped_subcore_slots[core_slot]),
                task_count,
                submit_tasks,
                active_core_count as usize,
                &batch,
                live_regs[core_slot].clone().unwrap_or_default(),
                observed_irq_status[core_slot],
            );
        }

        let mut running_core_mask = 0_u32;
        let mut waiting_core_mask = 0_u32;
        let mut irq_done_core_mask = 0_u32;
        let mut any_faulted = false;

        for (core_slot, core) in cores.iter().enumerate() {
            if core.enabled && core.inflight {
                running_core_mask |= 1 << core_slot;
            }
            if core.enabled && core.waiting_irq {
                waiting_core_mask |= 1 << core_slot;
            }
            if core.enabled && core.irq_seen {
                irq_done_core_mask |= 1 << core_slot;
            }
            if core.enabled && core.faulted {
                any_faulted = true;
            }
        }

        let savepoint_valid = running_core_mask != 0 && irq_done_core_mask == running_core_mask;
        let can_resume = savepoint_valid && !any_faulted;
        let ctx_state = derive_ctx_state(
            task_count,
            completed_task_count,
            running_core_mask != 0,
            waiting_core_mask != 0,
            savepoint_valid,
            any_faulted,
        );

        NPUDriverStatus {
            ctx_state,
            savepoint_valid,
            can_resume,
            owner_task_id: owner.task_id,
            owner_process_id: owner.process_id,
            owner_address_space_id: owner.address_space_id,
            iommu_domain_id: submit.iommu_domain_id,
            core_mask: submit.core_mask,
            active_core_count,
            wait_start_idx: 0,
            running_core_mask,
            waiting_core_mask,
            irq_done_core_mask,
            task_counter_snapshot: submit.task_counter,
            hw_elapse_time_snapshot: submit.hw_elapse_time,
            batch,
            cores,
        }
    }
}

pub(crate) fn read_pc_register_snapshot(core: &RknpuCore) -> NpuPcRegisterSnapshot {
    let pc = core.pc();

    NpuPcRegisterSnapshot {
        operation_enable: pc.operation_enable().read().bits(),
        base_address: pc.base_address().read().bits(),
        register_amounts: pc.register_amounts().read().bits(),
        interrupt_mask: pc.interrupt_mask().read().bits(),
        interrupt_status: pc.interrupt_status().read().bits(),
        interrupt_raw_status: pc.interrupt_raw_status().read().bits(),
        task_con: pc.task_con().read().bits(),
        task_dma_base_addr: pc.task_dma_base_addr().read().bits(),
        task_status: pc.task_status().read().bits(),
    }
}

pub(crate) fn read_block_register_snapshot(
    s_status: u32,
    s_pointer: u32,
    operation_enable: u32,
) -> NpuBlockRegisterSnapshot {
    NpuBlockRegisterSnapshot {
        s_status,
        s_pointer,
        operation_enable,
    }
}

pub(crate) fn read_cna_register_snapshot(core: &RknpuCore) -> NpuBlockRegisterSnapshot {
    let cna = core.cna();
    read_block_register_snapshot(
        cna.s_status().read().bits(),
        cna.s_pointer().read().bits(),
        cna.operation_enable().read().bits(),
    )
}

pub(crate) fn read_mac_register_snapshot(core: &RknpuCore) -> NpuBlockRegisterSnapshot {
    let mac = core.mac();
    read_block_register_snapshot(
        mac.s_status().read().bits(),
        mac.s_pointer().read().bits(),
        mac.operation_enable().read().bits(),
    )
}

pub(crate) fn read_dpu_register_snapshot(core: &RknpuCore) -> NpuBlockRegisterSnapshot {
    let dpu = core.dpu();
    read_block_register_snapshot(
        dpu.s_status().read().bits(),
        dpu.s_pointer().read().bits(),
        dpu.operation_enable().read().bits(),
    )
}

pub(crate) fn read_dpu_rdma_register_snapshot(core: &RknpuCore) -> NpuBlockRegisterSnapshot {
    let dpu_rdma = core.dpu_rdma();
    read_block_register_snapshot(
        dpu_rdma.s_status().read().bits(),
        dpu_rdma.s_pointer().read().bits(),
        dpu_rdma.operation_enable().read().bits(),
    )
}

pub(crate) fn read_ppu_register_snapshot(core: &RknpuCore) -> NpuBlockRegisterSnapshot {
    let ppu = core.ppu();
    read_block_register_snapshot(
        ppu.s_status().read().bits(),
        ppu.s_pointer().read().bits(),
        ppu.operation_enable().read().bits(),
    )
}

pub(crate) fn read_ppu_rdma_register_snapshot(core: &RknpuCore) -> NpuBlockRegisterSnapshot {
    let ppu_rdma = core.ppu_rdma();
    read_block_register_snapshot(
        ppu_rdma.s_status().read().bits(),
        ppu_rdma.s_pointer().read().bits(),
        ppu_rdma.operation_enable().read().bits(),
    )
}

pub(crate) fn read_global_register_snapshot(core: &RknpuCore) -> NpuGlobalRegisterSnapshot {
    NpuGlobalRegisterSnapshot {
        operation_enable: core.global().operation_enable().read().bits(),
    }
}

pub(crate) fn read_core_register_snapshot(core: &RknpuCore) -> NpuCoreRegisterSnapshot {
    NpuCoreRegisterSnapshot {
        pc: read_pc_register_snapshot(core),
        cna: read_cna_register_snapshot(core),
        mac: read_mac_register_snapshot(core),
        dpu: read_dpu_register_snapshot(core),
        dpu_rdma: read_dpu_rdma_register_snapshot(core),
        ppu: read_ppu_register_snapshot(core),
        ppu_rdma: read_ppu_rdma_register_snapshot(core),
        global: read_global_register_snapshot(core),
    }
}

pub(crate) fn read_submit_tasks(submit: &RknpuSubmit) -> &[RknpuTask] {
    if submit.task_obj_addr == 0 || submit.task_number == 0 {
        return &[];
    }

    unsafe {
        core::slice::from_raw_parts(
            submit.task_obj_addr as *const RknpuTask,
            submit.task_number as usize,
        )
    }
}

pub(crate) fn task_at(tasks: &[RknpuTask], idx: usize) -> Option<RknpuTask> {
    if idx >= tasks.len() {
        return None;
    }

    Some(unsafe { ptr::read_unaligned(tasks.as_ptr().add(idx)) })
}

pub(crate) fn collect_active_subcore_slots(submit: &RknpuSubmit, max_cores: usize) -> Vec<usize> {
    submit
        .subcore_task
        .iter()
        .enumerate()
        .filter(|(_, subcore)| subcore.task_number > 0)
        .map(|(slot, _)| slot)
        .take(max_cores)
        .collect()
}

fn completed_task_count(tasks: &[RknpuTask]) -> u32 {
    (0..tasks.len())
        .filter_map(|idx| task_at(tasks, idx))
        .filter(|task| task.int_status & task.int_mask != 0)
        .count() as u32
}

fn is_inflight_snapshot(regs: &NpuCoreRegisterSnapshot) -> bool {
    regs.pc.task_status != 0
        || regs.cna.s_status != 0
        || regs.mac.s_status != 0
        || regs.dpu.s_status != 0
        || regs.dpu_rdma.s_status != 0
        || regs.ppu.s_status != 0
        || regs.ppu_rdma.s_status != 0
}

fn distributed_task_count(total_tasks: usize, active_core_count: usize, core_slot: usize) -> u32 {
    if active_core_count == 0 || core_slot >= active_core_count {
        return 0;
    }

    let base = total_tasks / active_core_count;
    let remainder = total_tasks % active_core_count;

    (base + usize::from(core_slot < remainder)) as u32
}

fn derive_progress(
    task_count: usize,
    completed_task_count: u32,
    active_core_count: u8,
    inflight_core_count: u8,
) -> NpuBatchProgress {
    let total_task_count = task_count as u32;
    let completed_task_count = completed_task_count.min(total_task_count);
    let remaining_task_count = total_task_count.saturating_sub(completed_task_count);
    let current_batch_count = if inflight_core_count != 0 {
        inflight_core_count as u32
    } else {
        remaining_task_count.min(active_core_count as u32)
    };

    NpuBatchProgress {
        task_iter: completed_task_count.saturating_add(inflight_core_count as u32),
        task_iter_end: total_task_count,
        current_batch_start: completed_task_count,
        current_batch_count,
        completed_task_count,
        remaining_task_count,
    }
}

fn derive_ctx_state(
    task_count: usize,
    completed_task_count: u32,
    any_inflight: bool,
    any_waiting: bool,
    savepoint_valid: bool,
    any_faulted: bool,
) -> NpuCtxState {
    if task_count == 0 {
        NpuCtxState::Empty
    } else if completed_task_count == task_count as u32 {
        NpuCtxState::Completed
    } else if any_faulted {
        NpuCtxState::Faulted
    } else if savepoint_valid {
        NpuCtxState::BoundaryReady
    } else if any_inflight && any_waiting {
        NpuCtxState::WaitingIrqBoundary
    } else if any_inflight {
        NpuCtxState::Running
    } else {
        NpuCtxState::Prepared
    }
}

fn derive_core_status(
    core_slot: usize,
    mapped_subcore_slot: Option<usize>,
    task_count: usize,
    tasks: &[RknpuTask],
    active_core_count: usize,
    batch: &NpuBatchProgress,
    regs: NpuCoreRegisterSnapshot,
    observed_irq_status: u32,
) -> NpuCoreDriverStatus {
    let Some(subcore_slot) = mapped_subcore_slot else {
        return NpuCoreDriverStatus::default();
    };

    let inflight = is_inflight_snapshot(&regs);
    let completed_for_core = distributed_task_count(
        batch.completed_task_count as usize,
        active_core_count,
        core_slot,
    );
    let total_assigned_to_core = distributed_task_count(task_count, active_core_count, core_slot);
    let mut current_task_index = batch.current_batch_start.saturating_add(core_slot as u32);
    let batch_task_count = if current_task_index < task_count as u32 {
        1
    } else {
        current_task_index = 0;
        0
    };
    let expected_int_mask = if batch_task_count != 0 {
        task_at(tasks, current_task_index as usize)
            .map(|task| task.int_mask)
            .unwrap_or(0)
    } else {
        0
    };
    let irq_seen = (observed_irq_status & expected_int_mask) != 0;
    let faulted = observed_irq_status != 0 && (observed_irq_status & expected_int_mask) == 0;
    let boundary_ready = irq_seen && !faulted;
    let last_task_int_status = if completed_for_core == 0 {
        0
    } else {
        let last_completed_index =
            core_slot + (completed_for_core as usize - 1) * active_core_count;
        task_at(tasks, last_completed_index)
            .map(|task| task.int_status)
            .unwrap_or(0)
    };

    NpuCoreDriverStatus {
        core_slot: core_slot as u8,
        subcore_slot: subcore_slot as u8,
        enabled: true,
        inflight,
        waiting_irq: inflight && observed_irq_status == 0,
        irq_seen,
        irq_cleared: observed_irq_status == 0,
        boundary_ready,
        faulted,
        current_task_index,
        batch_task_start: batch.current_batch_start,
        batch_task_count,
        completed_task_count: completed_for_core,
        remaining_task_count: total_assigned_to_core.saturating_sub(completed_for_core),
        expected_int_mask,
        observed_irq_status,
        last_task_int_status,
        regs,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn derive_progress_matches_submit_model() {
        let progress = derive_progress(5, 2, 3, 2);

        assert_eq!(progress.task_iter, 4);
        assert_eq!(progress.task_iter_end, 5);
        assert_eq!(progress.current_batch_start, 2);
        assert_eq!(progress.current_batch_count, 2);
        assert_eq!(progress.completed_task_count, 2);
        assert_eq!(progress.remaining_task_count, 3);
    }

    #[test]
    fn distributed_task_count_round_robins() {
        assert_eq!(distributed_task_count(5, 3, 0), 2);
        assert_eq!(distributed_task_count(5, 3, 1), 2);
        assert_eq!(distributed_task_count(5, 3, 2), 1);
    }

    #[test]
    fn derive_core_status_marks_fault_on_unexpected_irq() {
        let batch = derive_progress(3, 1, 2, 1);
        let regs = NpuCoreRegisterSnapshot {
            pc: NpuPcRegisterSnapshot {
                task_status: 1,
                ..Default::default()
            },
            ..Default::default()
        };
        let tasks = [
            RknpuTask {
                int_mask: 0x3,
                int_status: 0x3,
                ..Default::default()
            },
            RknpuTask {
                int_mask: 0xc,
                ..Default::default()
            },
            RknpuTask {
                int_mask: 0x30,
                ..Default::default()
            },
        ];

        let status = derive_core_status(0, Some(0), 3, &tasks, 2, &batch, regs, 0x10);

        assert!(status.enabled);
        assert!(status.inflight);
        assert!(status.faulted);
        assert!(!status.boundary_ready);
        assert_eq!(status.expected_int_mask, 0xc);
    }

    #[test]
    fn derive_ctx_state_covers_core_transitions() {
        assert_eq!(
            derive_ctx_state(0, 0, false, false, false, false),
            NpuCtxState::Empty
        );
        assert_eq!(
            derive_ctx_state(2, 0, false, false, false, false),
            NpuCtxState::Prepared
        );
        assert_eq!(
            derive_ctx_state(2, 0, true, true, false, false),
            NpuCtxState::WaitingIrqBoundary
        );
        assert_eq!(
            derive_ctx_state(2, 1, true, false, true, false),
            NpuCtxState::BoundaryReady
        );
        assert_eq!(
            derive_ctx_state(2, 1, true, false, false, true),
            NpuCtxState::Faulted
        );
        assert_eq!(
            derive_ctx_state(2, 2, false, false, false, false),
            NpuCtxState::Completed
        );
    }
}
