//! NPU 上下文闭包。
//!
//! `NpuContext` 固定由两部分组成：
//! - 原始 [`crate::ioctrl::RknpuSubmit`]：完整保存用户提交参数
//! - [`crate::status::NPUDriverStatus`]：保存驱动动态态、执行流闭包和
//!   任务级硬件寄存器快照
//!
//! 设计约束：
//! - 不在这里保存电源状态、workqueue 状态、GEM 池全局状态或队列长度
//! - `RknpuSubmit` 必须整包保存，不拆字段
//! - `NPUDriverStatus` 只保存恢复执行流所需的驱动动态态

use alloc::string::String;
use core::fmt::Write;

use crate::{
    Rknpu,
    ioctrl::RknpuSubmit,
    status::{NPUDriverStatus, NpuOwnerIds},
};

/// 一个可脱离当前执行流保存/恢复的 NPU 任务闭包。
#[repr(C)]
#[derive(Debug, Clone, Default)]
pub struct NpuContext {
    /// 用户/驱动原始提交参数的完整镜像。
    ///
    /// 来源：`submit_ioctrl(&mut RknpuSubmit)` 的入参
    /// 恢复用途：重新构建任务描述符流、IOMMU domain 和 core 配置
    pub submit: RknpuSubmit,
    /// 驱动层与当前任务执行流动态强相关的最小闭包。
    ///
    /// 来源：提交路径局部变量、等待 IRQ 路径和外层 task/process 绑定
    /// 恢复用途：在 IRQ 边界恢复时决定从哪里继续执行
    pub driver_status: NPUDriverStatus,
}

impl Rknpu {
    /// 读取当前 live NPU 状态，并与调用者提供的 submit/owner 组合成完整 `NpuContext`。
    pub fn read_npu_context(&self, submit: &RknpuSubmit, owner: NpuOwnerIds) -> NpuContext {
        NpuContext {
            submit: submit.clone(),
            driver_status: self.read_npu_driver_status(submit, owner),
        }
    }

    /// 读取完整的进程级 NPU 状态。
    ///
    /// 该结构按约定同时保留一份独立的 `driver_status` 和一份 `npu_context`，
    /// 其中 `npu_context.driver_status` 必须与顶层 `driver_status` 保持一致。
    pub fn read_process_npu_state(
        &self,
        submit: &RknpuSubmit,
        owner: NpuOwnerIds,
    ) -> ProcessNpuState {
        let driver_status = self.read_npu_driver_status(submit, owner);
        let npu_context = NpuContext {
            submit: submit.clone(),
            driver_status: driver_status.clone(),
        };

        ProcessNpuState {
            driver_status,
            npu_context,
        }
    }
}

/// 完整的进程级 NPU 状态描述。
#[repr(C)]
#[derive(Debug, Clone)]
pub struct ProcessNpuState {
    pub driver_status: NPUDriverStatus,
    pub npu_context: NpuContext,
}

impl ProcessNpuState {
    /// 生成人类可读的完整状态快照。
    pub fn format_pretty(&self) -> String {
        let mut output = String::new();
        let driver_status = &self.driver_status;
        let submit = &self.npu_context.submit;

        let _ = writeln!(output, "Process NPU State");
        let _ = writeln!(
            output,
            "  owner: task={} process={} addr_space={}",
            driver_status.owner_task_id,
            driver_status.owner_process_id,
            driver_status.owner_address_space_id
        );
        let _ = writeln!(
            output,
            "  ctx: state={:?} can_resume={} savepoint_valid={}",
            driver_status.ctx_state, driver_status.can_resume, driver_status.savepoint_valid
        );
        let _ = writeln!(
            output,
            "  masks: core_mask={:#x} active={} running={:#x} waiting={:#x} irq_done={:#x}",
            driver_status.core_mask,
            driver_status.active_core_count,
            driver_status.running_core_mask,
            driver_status.waiting_core_mask,
            driver_status.irq_done_core_mask
        );

        let _ = writeln!(output, "batch");
        let _ = writeln!(
            output,
            "  task_iter={} task_iter_end={} current_batch_start={} current_batch_count={} completed={} remaining={}",
            driver_status.batch.task_iter,
            driver_status.batch.task_iter_end,
            driver_status.batch.current_batch_start,
            driver_status.batch.current_batch_count,
            driver_status.batch.completed_task_count,
            driver_status.batch.remaining_task_count
        );

        let _ = writeln!(output, "submit");
        let _ = writeln!(
            output,
            "  flags={:#x} timeout={} task_number={} task_counter={} hw_elapse_time={} task_base_addr={:#x} task_obj_addr={:#x} iommu_domain_id={}",
            submit.flags,
            submit.timeout,
            submit.task_number,
            submit.task_counter,
            submit.hw_elapse_time,
            submit.task_base_addr,
            submit.task_obj_addr,
            submit.iommu_domain_id
        );

        for (core_slot, core) in driver_status.cores.iter().enumerate() {
            let _ = writeln!(output, "core{}", core_slot);
            let _ = writeln!(
                output,
                "  flags: enabled={} inflight={} waiting_irq={} irq_seen={} irq_cleared={} boundary_ready={} faulted={}",
                core.enabled,
                core.inflight,
                core.waiting_irq,
                core.irq_seen,
                core.irq_cleared,
                core.boundary_ready,
                core.faulted
            );
            let _ = writeln!(
                output,
                "  task: subcore_slot={} current_task_index={} batch_task_start={} batch_task_count={} completed={} remaining={} expected_int_mask={:#x} observed_irq_status={:#x} last_task_int_status={:#x}",
                core.subcore_slot,
                core.current_task_index,
                core.batch_task_start,
                core.batch_task_count,
                core.completed_task_count,
                core.remaining_task_count,
                core.expected_int_mask,
                core.observed_irq_status,
                core.last_task_int_status
            );
            let _ = writeln!(
                output,
                "  pc: op_en={:#x} base={:#x} amount={:#x} int_mask={:#x} int_status={:#x} int_raw={:#x} task_con={:#x} task_dma_base={:#x} task_status={:#x}",
                core.regs.pc.operation_enable,
                core.regs.pc.base_address,
                core.regs.pc.register_amounts,
                core.regs.pc.interrupt_mask,
                core.regs.pc.interrupt_status,
                core.regs.pc.interrupt_raw_status,
                core.regs.pc.task_con,
                core.regs.pc.task_dma_base_addr,
                core.regs.pc.task_status
            );
            let _ = writeln!(
                output,
                "  cna: s_status={:#x} s_pointer={:#x} op_en={:#x}",
                core.regs.cna.s_status, core.regs.cna.s_pointer, core.regs.cna.operation_enable
            );
            let _ = writeln!(
                output,
                "  mac: s_status={:#x} s_pointer={:#x} op_en={:#x}",
                core.regs.mac.s_status, core.regs.mac.s_pointer, core.regs.mac.operation_enable
            );
            let _ = writeln!(
                output,
                "  dpu: s_status={:#x} s_pointer={:#x} op_en={:#x}",
                core.regs.dpu.s_status, core.regs.dpu.s_pointer, core.regs.dpu.operation_enable
            );
            let _ = writeln!(
                output,
                "  dpu_rdma: s_status={:#x} s_pointer={:#x} op_en={:#x}",
                core.regs.dpu_rdma.s_status,
                core.regs.dpu_rdma.s_pointer,
                core.regs.dpu_rdma.operation_enable
            );
            let _ = writeln!(
                output,
                "  ppu: s_status={:#x} s_pointer={:#x} op_en={:#x}",
                core.regs.ppu.s_status, core.regs.ppu.s_pointer, core.regs.ppu.operation_enable
            );
            let _ = writeln!(
                output,
                "  ppu_rdma: s_status={:#x} s_pointer={:#x} op_en={:#x}",
                core.regs.ppu_rdma.s_status,
                core.regs.ppu_rdma.s_pointer,
                core.regs.ppu_rdma.operation_enable
            );
            let _ = writeln!(
                output,
                "  global: op_en={:#x}",
                core.regs.global.operation_enable
            );
        }

        output
    }

    /// 通过日志打印完整状态。
    pub fn dump_pretty(&self) {
        warn!("\n{}", self.format_pretty());
    }
}

#[cfg(test)]
mod tests {
    use super::{NpuContext, ProcessNpuState};
    use crate::{
        Rknpu, RknpuConfig, RknpuType,
        ioctrl::RknpuSubmit,
        status::{INVALID_CORE_SLOT, NpuCtxState, NpuOwnerIds},
    };
    use alloc::vec::Vec;
    use core::ptr::NonNull;

    const FAKE_MMIO_LEN: usize = 0x10000;

    fn build_fake_rknpu() -> (Rknpu, Vec<Vec<u8>>) {
        let mut mmios = vec![vec![0_u8; FAKE_MMIO_LEN]; 3];
        let base_addrs = mmios
            .iter_mut()
            .map(|mmio| NonNull::new(mmio.as_mut_ptr()).unwrap())
            .collect::<Vec<_>>();
        let config = RknpuConfig {
            rknpu_type: RknpuType::Rk3588,
        };

        (Rknpu::new(&base_addrs, config), mmios)
    }

    #[test]
    fn default_context_starts_empty() {
        let ctx = NpuContext::default();

        assert!(matches!(ctx.driver_status.ctx_state, NpuCtxState::Empty));
        assert!(!ctx.driver_status.savepoint_valid);
        assert_eq!(ctx.driver_status.active_core_count, 0);

        for core in &ctx.driver_status.cores {
            assert!(core.core_slot == INVALID_CORE_SLOT || !core.enabled);
            assert_eq!(core.regs.pc.task_status, 0);
            assert_eq!(core.regs.pc.interrupt_status, 0);
            assert_eq!(core.regs.cna.s_status, 0);
            assert_eq!(core.regs.global.operation_enable, 0);
        }
    }

    #[test]
    fn read_process_state_keeps_driver_status_consistent() {
        let (npu, _mmios) = build_fake_rknpu();
        let submit = RknpuSubmit::default();
        let owner = NpuOwnerIds {
            task_id: 1,
            process_id: 2,
            address_space_id: 3,
        };
        let process = npu.read_process_npu_state(&submit, owner);

        assert_eq!(process.driver_status, process.npu_context.driver_status);
        assert_eq!(process.npu_context.submit.flags, submit.flags);
        assert_eq!(process.npu_context.submit.task_number, submit.task_number);
        assert_eq!(
            process.npu_context.submit.task_obj_addr,
            submit.task_obj_addr
        );
        assert_eq!(
            process.npu_context.submit.task_base_addr,
            submit.task_base_addr
        );
        assert_eq!(process.npu_context.submit.core_mask, submit.core_mask);
    }

    #[test]
    fn format_pretty_contains_summary_sections() {
        let state = ProcessNpuState {
            driver_status: Default::default(),
            npu_context: Default::default(),
        };
        let pretty = state.format_pretty();

        assert!(pretty.contains("Process NPU State"));
        assert!(pretty.contains("batch"));
        assert!(pretty.contains("submit"));
        assert!(pretty.contains("core0"));
        assert!(pretty.contains("core1"));
        assert!(pretty.contains("core2"));
    }
}
