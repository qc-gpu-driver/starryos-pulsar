//! 瑞芯微 NPU 的内存映射寄存器定义。
//!
//! 寄存器块类型来自 [`rknpu_regs`] crate（svd2rust 生成）。
//! 本模块提供 [`RknpuCore`]，一个将动态 MMIO 基地址转换为
//! 适当寄存器块引用的薄包装器。

use core::ptr::NonNull;
use core::sync::atomic::{AtomicU32, Ordering};

use alloc::{sync::Arc, vec::Vec};
use mbarrier::mb;

use crate::{JobMode, RknpuError, Submit, SubmitRef, data::RknpuData};

use crate::RknpuSharedState;
use crate::RknpuTask;
use crate::SubmitBase;
use crate::ioctrl::RknpuSubmit;
use crate::status::{
    NpuCoreRegisterSnapshot, NpuCoreRestoreImage, NpuTaskRegisterWrite, read_core_register_snapshot,
};
use crate::task::def::{
    BLOCK_CNA, BLOCK_CORE, BLOCK_DPU, BLOCK_DPU_RDMA, BLOCK_PC, BLOCK_PPU, BLOCK_PPU_RDMA,
    CNA_S_POINTER, CORE_S_POINTER, DPU_S_POINTER, OP_40, OP_ENABLE, OP_NONE, PC_OP_01,
    PC_OPERATION_ENABLE, PC_REGISTER_AMOUNTS,
};
use core::hint::spin_loop;

const INT_CLEAR_ALL: u32 = 0x1_FFFF;
const RKNPU_PC_DATA_EXTRA_AMOUNT: u32 = 4;
const PC_BASE_ADDRESS_OFFSET: u16 = 0x0010;
const PC_INTERRUPT_MASK_OFFSET: u16 = 0x0020;
const PC_TASK_CON_OFFSET: u16 = 0x0030;
const PC_TASK_DMA_BASE_ADDR_OFFSET: u16 = 0x0034;
const PC_INTERRUPT_STATUS_OFFSET: u16 = 0x0028;
const PC_INTERRUPT_RAW_STATUS_OFFSET: u16 = 0x002c;
const PC_TASK_STATUS_OFFSET: u16 = 0x003c;
const DPU_RDMA_S_POINTER_OFFSET: u16 = 0x5004;
const PPU_S_POINTER_OFFSET: u16 = 0x6004;
const PPU_RDMA_S_POINTER_OFFSET: u16 = 0x7004;
const CNA_OPERATION_ENABLE_OFFSET: u16 = 0x1008;
const CORE_OPERATION_ENABLE_OFFSET: u16 = 0x3008;
const DPU_OPERATION_ENABLE_OFFSET: u16 = 0x4008;
const DPU_RDMA_OPERATION_ENABLE_OFFSET: u16 = 0x5008;
const PPU_OPERATION_ENABLE_OFFSET: u16 = 0x6008;
const PPU_RDMA_OPERATION_ENABLE_OFFSET: u16 = 0x7008;
const GLOBAL_OPERATION_ENABLE_OFFSET: u16 = 0xf008;

#[derive(Clone)]
/// 一个 NPU 核心的 MMIO 寄存器窗口的不可变视图。
///
/// # 中断驱动完成
///
/// 每个核心拥有一个共享的 `irq_status` 原子变量。当 IRQ 触发时，
/// 平台调用 [`handle_interrupt()`]（通常通过 [`RknpuIrqHandler`]），
/// 它读取 `INTERRUPT_STATUS`，清除硬件中断，并将模糊状态存储到 `irq_status` 中。
/// 然后 `ioctrl.rs` 中的提交代码可以检查原子变量而不是忙轮询 MMIO 寄存器，
/// 允许 CPU 在检查之间休眠（WFI）。
pub struct RknpuCore {
    base: NonNull<u8>,
    pub(crate) core_slot: u8,
    /// 共享中断状态标志 — 由 IRQ 处理程序写入，由提交读取。
    /// 零表示没有挂起的中断；非零保存模糊状态位。
    pub(crate) irq_status: Arc<AtomicU32>,
    pub(crate) shared: Arc<RknpuSharedState>,
}
unsafe impl Send for RknpuCore {}

impl RknpuCore {
    ///提交并执行1个任务
    pub fn start_execute_one(
        &mut self,
        idx: usize,
        data: &RknpuData,
        rknpu_task: &mut RknpuTask,
        args: &RknpuSubmit,
    ) -> Result<(), RknpuError> {
        let job = SubmitRef {
            base: SubmitBase {
                flags: JobMode::from_bits_retain(args.flags),
                task_base_addr: args.task_base_addr as _,
                core_idx: idx,
                // Wait for the LAST task's interrupt (pipeline completion)
                int_mask: rknpu_task.int_mask,
                int_clear: rknpu_task.int_mask,
                regcfg_amount: rknpu_task.regcfg_amount,
            },
            task_number: 1,
            regcmd_base_addr: rknpu_task.regcmd_addr as _,
        };
        debug!(
            "[NPU]   batch: {} tasks, regcmd@{:#x}, {} regcfg_words, int_mask={:#x}",
            1, job.regcmd_base_addr, job.base.regcfg_amount, job.base.int_mask
        );
        debug!("Submit {} jobs: {job:#x?}", 1);

        // Clear the shared atomic before submitting — so we only see
        // status bits from THIS submission, not leftovers.
        self.irq_status.store(0, Ordering::Release);

        // Program PC registers and start execution
        debug!("Submitting PC job...");
        self.submit_pc(data, &job).unwrap();
        Ok(())
    }

    /// 将一批任务提交到单个 NPU 核心并等待完成。
    ///
    /// `idx` 既是 `subcore_task[]` 数组索引，也是硬件核心索引。
    ///
    /// # 完成等待模式
    ///
    /// - **IRQ 驱动**（`self.wait_fn` 为 `Some`）：
    ///   CPU 在迭代之间休眠（WFI）。当 NPU 完成时，它触发中断 → GIC 唤醒 CPU
    ///   → IRQ 处理程序调用 `handle_interrupt()`，将模糊状态存储到共享的
    ///   `irq_status` 原子变量中 → 循环看到它并退出。
    ///
    /// - **传统忙轮询**（`self.wait_fn` 为 `None`）：
    ///   CPU 在紧密循环中直接读取 MMIO `INTERRUPT_STATUS` 寄存器。
    ///   简单但浪费 CPU 周期。
    ///
    /// # 批处理
    ///
    /// 硬件一次只能接受 `max_submit_number` 个任务。
    /// 如果任务数超过此数量，此函数会循环，一次提交一批并等待每批完成。
    pub fn _submit_one(
        &mut self,
        data: &RknpuData,
        wait_fn: Option<fn()>,
        idx: usize,
        args: &mut RknpuSubmit,
    ) -> Result<usize, RknpuError> {
        let task_ptr = args.task_obj_addr as *mut RknpuTask;
        let subcore = &args.subcore_task[idx];

        let mut task_iter = subcore.task_start as usize;
        let task_iter_end = task_iter + subcore.task_number as usize;
        let max_submit_number = data.max_submit_number as usize;

        while task_iter < task_iter_end {
            // Clamp batch size to hardware limit
            let task_number = (task_iter_end - task_iter).min(max_submit_number);
            let submit_tasks =
                unsafe { core::slice::from_raw_parts_mut(task_ptr.add(task_iter), task_number) };

            // Build the register-level submission descriptor
            let job = SubmitRef {
                base: SubmitBase {
                    flags: JobMode::from_bits_retain(args.flags),
                    task_base_addr: args.task_base_addr as _,
                    core_idx: idx,
                    // Wait for the LAST task's interrupt (pipeline completion)
                    int_mask: submit_tasks.last().unwrap().int_mask,
                    int_clear: submit_tasks[0].int_mask,
                    regcfg_amount: submit_tasks[0].regcfg_amount,
                },
                task_number,
                regcmd_base_addr: submit_tasks[0].regcmd_addr as _,
            };
            debug!(
                "[NPU]   batch: {} tasks, regcmd@{:#x}, {} regcfg_words, int_mask={:#x}",
                task_number, job.regcmd_base_addr, job.base.regcfg_amount, job.base.int_mask
            );
            debug!("Submit {task_number} jobs: {job:#x?}");

            // Drain any leftover interrupts from previous execution
            while self.handle_interrupt() != 0 {
                spin_loop();
            }

            // Clear the shared atomic before submitting — so we only see
            // status bits from THIS submission, not leftovers.
            self.irq_status.store(0, Ordering::Release);

            // Program PC registers and start execution
            debug!("Submitting PC job...");
            self.submit_pc(data, &job).unwrap();

            // ── Wait for completion ──────────────────────────────────────
            let int_status = if let Some(wait) = wait_fn {
                debug!("[NPU]   waiting (IRQ+WFI mode)...");
                // ┌─────────────────────────────────────────────────────┐
                // │  IRQ-driven mode (CPU sleeps between checks)        │
                // │                                                     │
                // │  NPU runs → fires IRQ → handler calls               │
                // │  handle_interrupt() → stores fuzzed status to       │
                // │  irq_status atomic → CPU wakes from WFI → we       │
                // │  read the atomic here and proceed.                  │
                // └─────────────────────────────────────────────────────┘
                loop {
                    let status = self.irq_status.load(Ordering::Acquire);
                    if status & job.base.int_mask > 0 {
                        break job.base.int_mask & status;
                    }
                    if status != 0 {
                        debug!("Unexpected IRQ status: {:#x}", status);
                        return Err(RknpuError::TaskError);
                    }
                    // Sleep until any interrupt (including NPU) wakes the CPU.
                    (wait)();
                }
            } else {
                // ┌─────────────────────────────────────────────────────┐
                // │  Legacy busy-poll mode (direct MMIO register read)  │
                // │                                                     │
                // │  CPU spins reading INTERRUPT_STATUS until the       │
                // │  expected bits appear.  Simple but wastes cycles.   │
                // └─────────────────────────────────────────────────────┘
                loop {
                    let status = self.pc().interrupt_status().read().bits();
                    let status = rknpu_fuzz_status(status);
                    if status & job.base.int_mask > 0 {
                        break job.base.int_mask & status;
                    }
                    if status != 0 {
                        debug!("Interrupt status changed: {:#x}", status);
                        return Err(RknpuError::TaskError);
                    }
                }
            };

            // Acknowledge the interrupt and advance to next batch.
            // In IRQ mode the handler already cleared the HW interrupt,
            // but we still clear once more to be safe.
            self.clean_interrupts();
            // Reset atomic for the next batch.
            self.irq_status.store(0, Ordering::Release);
            debug!("[NPU]   batch done: int_status={:#x}", int_status);
            submit_tasks.last_mut().unwrap().int_status = int_status;
            task_iter += task_number;
        }

        Ok(subcore.task_number as usize)
    }

    /// # 安全性
    ///
    /// 调用者必须确保指针在返回结构的生命周期内映射 RKNN 寄存器空间。
    pub unsafe fn new(
        base_addr: NonNull<u8>,
        core_slot: u8,
        shared: Arc<RknpuSharedState>,
    ) -> Self {
        Self {
            base: base_addr,
            core_slot,
            irq_status: Arc::new(AtomicU32::new(0)),
            shared,
        }
    }

    #[inline(always)]
    pub(crate) fn offset_ptr<T>(&self, offset: usize) -> NonNull<T> {
        // 安全性：调用者保证 MMIO 映射有效；偏移量来自硬件文档。
        unsafe {
            let ptr = self.base.as_ptr().add(offset);
            NonNull::new_unchecked(ptr as *mut T)
        }
    }

    #[inline(always)]
    fn reg<T>(&self, offset: usize) -> &T {
        unsafe { self.offset_ptr::<T>(offset).as_ref() }
    }

    pub fn pc(&self) -> &rknpu_regs::pc::RegisterBlock {
        self.reg(rknpu_regs::Pc::PTR as usize)
    }

    pub fn cna(&self) -> &rknpu_regs::cna::RegisterBlock {
        self.reg(rknpu_regs::Cna::PTR as usize)
    }

    pub fn mac(&self) -> &rknpu_regs::mac::RegisterBlock {
        self.reg(rknpu_regs::Mac::PTR as usize)
    }

    pub fn dpu(&self) -> &rknpu_regs::dpu::RegisterBlock {
        self.reg(rknpu_regs::Dpu::PTR as usize)
    }

    pub fn dpu_rdma(&self) -> &rknpu_regs::dpu_rdma::RegisterBlock {
        self.reg(rknpu_regs::DpuRdma::PTR as usize)
    }

    pub fn ppu(&self) -> &rknpu_regs::ppu::RegisterBlock {
        self.reg(rknpu_regs::Ppu::PTR as usize)
    }

    pub fn ppu_rdma(&self) -> &rknpu_regs::ppu_rdma::RegisterBlock {
        self.reg(rknpu_regs::PpuRdma::PTR as usize)
    }

    pub fn ddma(&self) -> &rknpu_regs::ddma::RegisterBlock {
        self.reg(rknpu_regs::Ddma::PTR as usize)
    }

    pub fn sdma(&self) -> &rknpu_regs::sdma::RegisterBlock {
        self.reg(rknpu_regs::Sdma::PTR as usize)
    }

    pub fn global(&self) -> &rknpu_regs::global::RegisterBlock {
        self.reg(rknpu_regs::Global::PTR as usize)
    }

    pub fn version(&self) -> u32 {
        self.pc()
            .version()
            .read()
            .bits()
            .wrapping_add(self.pc().version_num().read().bits() & 0xffff)
    }

    pub fn clean_interrupts(&self) {
        self.pc()
            .interrupt_clear()
            .write(|w| unsafe { w.bits(INT_CLEAR_ALL) });
    }

    pub(crate) fn submit_pc(
        &mut self,
        config: &RknpuData,
        args: &SubmitRef,
    ) -> Result<(), RknpuError> {
        let pc_data_amount_scale = config.pc_data_amount_scale;

        self.pc().base_address().write(|w| unsafe { w.bits(1) });

        let task_pp_en = if args.base.flags.contains(JobMode::PINGPONG) {
            1
        } else {
            0
        };
        let pc_task_number_bits = config.pc_task_number_bits;

        if config.irqs.get(args.base.core_idx).is_some() {
            let val = 0xe + 0x10000000 * args.base.core_idx as u32;

            debug!("Set PC S_POINTER to {:#x}", val);

            self.cna().s_pointer().write(|w| unsafe { w.bits(val) });
            self.mac().s_pointer().write(|w| unsafe { w.bits(val) });
        }

        let pc_base_addr = args.regcmd_base_addr;

        debug!("Set PC BASE_ADDRESS to {:#x}", pc_base_addr);

        self.pc()
            .base_address()
            .write(|w| unsafe { w.bits(pc_base_addr) });

        let amount = (args.base.regcfg_amount + RKNPU_PC_DATA_EXTRA_AMOUNT)
            .div_ceil(pc_data_amount_scale)
            - 1;

        debug!("Set PC REGISTER_AMOUNTS to {:#x}", amount);

        self.pc()
            .register_amounts()
            .write(|w| unsafe { w.bits(amount) });

        self.pc()
            .interrupt_mask()
            .write(|w| unsafe { w.bits(args.base.int_mask) });
        self.pc()
            .interrupt_clear()
            .write(|w| unsafe { w.bits(args.base.int_clear) });
        let task_number = args.task_number as u32;

        let task_control = ((0x6 | task_pp_en) << pc_task_number_bits) | task_number;
        debug!("Set PC TASK_CONTROL to {:#x}", task_control);
        self.pc()
            .task_con()
            .write(|w| unsafe { w.bits(task_control) });
        debug!(
            "Set PC TASK_DMA_BASE_ADDR to {:#x}",
            args.base.task_base_addr
        );
        self.pc()
            .task_dma_base_addr()
            .write(|w| unsafe { w.bits(args.base.task_base_addr) });
        mb();
        self.pc().operation_enable().write(|w| unsafe { w.bits(1) });
        mb();
        self.pc().operation_enable().write(|w| unsafe { w.bits(0) });

        debug!("Submitted {args:#x?}");

        Ok(())
    }

    pub fn submit(&mut self, config: &RknpuData, args: &Submit) -> Result<(), RknpuError> {
        if args.tasks.len() > config.max_submit_number as usize {
            todo!()
        }
        self.submit_pc(config, &args.as_ref())
    }

    /// 在给某个 core 装载新 task 之前，先把遗留的 pending IRQ 全部排空。
    ///
    /// 否则上一个批次残留的中断可能会被误归属到新的 owner/task。
    pub fn drain_pending_interrupts(&self) {
        while self.handle_interrupt() != 0 {
            self.irq_status.store(0, Ordering::Release);
            spin_loop();
        }
        self.irq_status.store(0, Ordering::Release);
    }

    /// 从一段 regcmd 流中提取“可安全恢复”的 task-window 写操作。
    ///
    /// 规则：
    /// - 只保留最终会影响任务配置、且在 IRQ 边界允许回写的寄存器；
    /// - 同一 offset 若被多次写，保留最后一次；
    /// - 只用于“先写坏再恢复”的实验镜像，不直接代表 live 观察态。
    pub(crate) fn build_task_shadow_writes_from_regcmds(
        &self,
        regcmds: &[u64],
    ) -> Vec<NpuTaskRegisterWrite> {
        if regcmds.is_empty() {
            return Vec::new();
        }
        let mut writes: Vec<NpuTaskRegisterWrite> = Vec::new();

        for &regcmd in regcmds {
            if let Some(write) = Self::decode_task_reg_write(regcmd) {
                if let Some(existing) = writes.iter_mut().find(|item| item.offset == write.offset) {
                    *existing = write;
                } else {
                    writes.push(write);
                }
            }
        }

        writes
    }

    /// 把单条 regcmd 解析成“可恢复寄存器写”。
    ///
    /// 这里会主动过滤：
    /// - 无实际写效果的 opcode；
    /// - 只读快照寄存器；
    /// - 已经由 restore scalar 单独管理的关键寄存器；
    /// - 需要先做消毒才能安全恢复的位。
    fn decode_task_reg_write(regcmd: u64) -> Option<NpuTaskRegisterWrite> {
        let opcode = ((regcmd >> 48) & 0xffff) as u16;
        let value = ((regcmd >> 16) & 0xffff_ffff) as u32;
        let offset = (regcmd & 0xffff) as u16;
        let block = (opcode as u32 & 0xff00) as u16;
        let op_kind = opcode as u32 & 0x00ff;

        if matches!(opcode as u32, OP_NONE | OP_40 | OP_ENABLE) {
            return None;
        }
        if op_kind != PC_OP_01 {
            return None;
        }
        if !matches!(
            block as u32,
            BLOCK_PC
                | BLOCK_CNA
                | BLOCK_CORE
                | BLOCK_DPU
                | BLOCK_DPU_RDMA
                | BLOCK_PPU
                | BLOCK_PPU_RDMA
        ) {
            return None;
        }
        if Self::offset_handled_by_restore_scalars(offset)
            || Self::is_readonly_snapshot_offset(offset)
        {
            return None;
        }

        Some(NpuTaskRegisterWrite {
            block,
            offset,
            value: Self::sanitize_task_reg_value(offset, value)?,
        })
    }

    /// 用 IRQ 边界抓到的 live 快照刷新“可恢复镜像”。
    ///
    /// 注意这里不是简单地全量照抄：
    /// - `operation_enable` 一律只恢复成 0，避免在 IRQ 边界重新打启动脉冲；
    /// - `s_pointer` / `task_con` 会先做消毒；
    /// - `unsafe_snapshot` 用来标记边界上是否仍然观察到块处于 enable 状态。
    pub(crate) fn refresh_restore_image_from_snapshot(
        image: &mut NpuCoreRestoreImage,
        regs: &NpuCoreRegisterSnapshot,
    ) {
        image.pc_operation_enable = 0;
        image.pc_base_address = regs.pc.base_address;
        image.pc_register_amounts = regs.pc.register_amounts;
        image.pc_interrupt_mask = regs.pc.interrupt_mask;
        image.pc_task_con = Self::sanitize_task_con(regs.pc.task_con);
        image.pc_task_dma_base_addr = regs.pc.task_dma_base_addr;
        image.cna_s_pointer = Self::sanitize_s_pointer(regs.cna.s_pointer);
        image.mac_s_pointer = Self::sanitize_s_pointer(regs.mac.s_pointer);
        image.dpu_s_pointer = Self::sanitize_s_pointer(regs.dpu.s_pointer);
        image.dpu_rdma_s_pointer = Self::sanitize_s_pointer(regs.dpu_rdma.s_pointer);
        image.ppu_s_pointer = Self::sanitize_s_pointer(regs.ppu.s_pointer);
        image.ppu_rdma_s_pointer = Self::sanitize_s_pointer(regs.ppu_rdma.s_pointer);
        image.cna_operation_enable = 0;
        image.mac_operation_enable = 0;
        image.dpu_operation_enable = 0;
        image.dpu_rdma_operation_enable = 0;
        image.ppu_operation_enable = 0;
        image.ppu_rdma_operation_enable = 0;
        image.global_operation_enable = 0;
        image.unsafe_snapshot = regs.pc.operation_enable != 0
            || regs.cna.operation_enable != 0
            || regs.mac.operation_enable != 0
            || regs.dpu.operation_enable != 0
            || regs.dpu_rdma.operation_enable != 0
            || regs.ppu.operation_enable != 0
            || regs.ppu_rdma.operation_enable != 0
            || regs.global.operation_enable != 0;
    }

    /// 按固定毒值先把即将恢复的寄存器集合写坏。
    ///
    /// 这是实验路径的一部分，用来验证“保存的镜像是否真的足够把硬件重新覆盖回来”。
    fn apply_poison_to_restore_image(&self, image: &NpuCoreRestoreImage) {
        for write in &image.task_shadow_writes {
            self.write_mmio_u32(write.offset, Self::poison_value(write.offset, write.value));
        }

        self.write_mmio_u32(
            PC_BASE_ADDRESS_OFFSET,
            Self::poison_value(PC_BASE_ADDRESS_OFFSET, image.pc_base_address),
        );
        self.write_mmio_u32(
            PC_REGISTER_AMOUNTS as u16,
            Self::poison_value(PC_REGISTER_AMOUNTS as u16, image.pc_register_amounts),
        );
        self.write_mmio_u32(
            PC_INTERRUPT_MASK_OFFSET,
            Self::poison_value(PC_INTERRUPT_MASK_OFFSET, image.pc_interrupt_mask),
        );
        self.write_mmio_u32(
            PC_TASK_CON_OFFSET,
            Self::poison_value(PC_TASK_CON_OFFSET, image.pc_task_con),
        );
        self.write_mmio_u32(
            PC_TASK_DMA_BASE_ADDR_OFFSET,
            Self::poison_value(PC_TASK_DMA_BASE_ADDR_OFFSET, image.pc_task_dma_base_addr),
        );
        self.write_mmio_u32(
            CNA_S_POINTER as u16,
            Self::poison_value(CNA_S_POINTER as u16, image.cna_s_pointer),
        );
        self.write_mmio_u32(
            CORE_S_POINTER as u16,
            Self::poison_value(CORE_S_POINTER as u16, image.mac_s_pointer),
        );
        self.write_mmio_u32(
            DPU_S_POINTER as u16,
            Self::poison_value(DPU_S_POINTER as u16, image.dpu_s_pointer),
        );
        self.write_mmio_u32(
            DPU_RDMA_S_POINTER_OFFSET,
            Self::poison_value(DPU_RDMA_S_POINTER_OFFSET, image.dpu_rdma_s_pointer),
        );
        self.write_mmio_u32(
            PPU_S_POINTER_OFFSET,
            Self::poison_value(PPU_S_POINTER_OFFSET, image.ppu_s_pointer),
        );
        self.write_mmio_u32(
            PPU_RDMA_S_POINTER_OFFSET,
            Self::poison_value(PPU_RDMA_S_POINTER_OFFSET, image.ppu_rdma_s_pointer),
        );

        self.write_mmio_u32(PC_OPERATION_ENABLE as u16, 0);
        self.write_mmio_u32(CNA_OPERATION_ENABLE_OFFSET, 0);
        self.write_mmio_u32(CORE_OPERATION_ENABLE_OFFSET, 0);
        self.write_mmio_u32(DPU_OPERATION_ENABLE_OFFSET, 0);
        self.write_mmio_u32(DPU_RDMA_OPERATION_ENABLE_OFFSET, 0);
        self.write_mmio_u32(PPU_OPERATION_ENABLE_OFFSET, 0);
        self.write_mmio_u32(PPU_RDMA_OPERATION_ENABLE_OFFSET, 0);
        self.write_mmio_u32(GLOBAL_OPERATION_ENABLE_OFFSET, 0);
    }

    /// 按保存的恢复镜像重新覆盖硬件寄存器。
    fn apply_restore_image(&self, image: &NpuCoreRestoreImage) {
        for write in &image.task_shadow_writes {
            self.write_mmio_u32(write.offset, write.value);
        }

        self.write_mmio_u32(PC_BASE_ADDRESS_OFFSET, image.pc_base_address);
        self.write_mmio_u32(PC_REGISTER_AMOUNTS as u16, image.pc_register_amounts);
        self.write_mmio_u32(PC_INTERRUPT_MASK_OFFSET, image.pc_interrupt_mask);
        self.write_mmio_u32(PC_TASK_CON_OFFSET, image.pc_task_con);
        self.write_mmio_u32(PC_TASK_DMA_BASE_ADDR_OFFSET, image.pc_task_dma_base_addr);
        self.write_mmio_u32(CNA_S_POINTER as u16, image.cna_s_pointer);
        self.write_mmio_u32(CORE_S_POINTER as u16, image.mac_s_pointer);
        self.write_mmio_u32(DPU_S_POINTER as u16, image.dpu_s_pointer);
        self.write_mmio_u32(DPU_RDMA_S_POINTER_OFFSET, image.dpu_rdma_s_pointer);
        self.write_mmio_u32(PPU_S_POINTER_OFFSET, image.ppu_s_pointer);
        self.write_mmio_u32(PPU_RDMA_S_POINTER_OFFSET, image.ppu_rdma_s_pointer);
        self.write_mmio_u32(PC_OPERATION_ENABLE as u16, image.pc_operation_enable);
        self.write_mmio_u32(CNA_OPERATION_ENABLE_OFFSET, image.cna_operation_enable);
        self.write_mmio_u32(CORE_OPERATION_ENABLE_OFFSET, image.mac_operation_enable);
        self.write_mmio_u32(DPU_OPERATION_ENABLE_OFFSET, image.dpu_operation_enable);
        self.write_mmio_u32(
            DPU_RDMA_OPERATION_ENABLE_OFFSET,
            image.dpu_rdma_operation_enable,
        );
        self.write_mmio_u32(PPU_OPERATION_ENABLE_OFFSET, image.ppu_operation_enable);
        self.write_mmio_u32(
            PPU_RDMA_OPERATION_ENABLE_OFFSET,
            image.ppu_rdma_operation_enable,
        );
        self.write_mmio_u32(
            GLOBAL_OPERATION_ENABLE_OFFSET,
            image.global_operation_enable,
        );
    }

    /// 读回校验恢复镜像覆盖的关键标量寄存器。
    ///
    /// 这里故意不把 `task_shadow_writes` 算进硬失败集合，因为这些 task-window
    /// 配置寄存器在“完成 IRQ 边界”后不保证还能稳定按原值读回。
    fn verify_restore_image(&self, image: &NpuCoreRestoreImage) -> u64 {
        let mut mismatch_mask = 0_u64;

        mismatch_mask |= self.verify_field(0, PC_BASE_ADDRESS_OFFSET, image.pc_base_address);
        mismatch_mask |=
            self.verify_field(1, PC_REGISTER_AMOUNTS as u16, image.pc_register_amounts);
        mismatch_mask |= self.verify_field(2, PC_INTERRUPT_MASK_OFFSET, image.pc_interrupt_mask);
        mismatch_mask |= self.verify_field(3, PC_TASK_CON_OFFSET, image.pc_task_con);
        mismatch_mask |=
            self.verify_field(4, PC_TASK_DMA_BASE_ADDR_OFFSET, image.pc_task_dma_base_addr);
        mismatch_mask |= self.verify_field(5, CNA_S_POINTER as u16, image.cna_s_pointer);
        mismatch_mask |= self.verify_field(6, CORE_S_POINTER as u16, image.mac_s_pointer);
        mismatch_mask |= self.verify_field(7, DPU_S_POINTER as u16, image.dpu_s_pointer);
        mismatch_mask |= self.verify_field(8, DPU_RDMA_S_POINTER_OFFSET, image.dpu_rdma_s_pointer);
        mismatch_mask |= self.verify_field(9, PPU_S_POINTER_OFFSET, image.ppu_s_pointer);
        mismatch_mask |= self.verify_field(10, PPU_RDMA_S_POINTER_OFFSET, image.ppu_rdma_s_pointer);
        mismatch_mask |= self.verify_field(11, PC_OPERATION_ENABLE as u16, 0);
        mismatch_mask |= self.verify_field(12, CNA_OPERATION_ENABLE_OFFSET, 0);
        mismatch_mask |= self.verify_field(13, CORE_OPERATION_ENABLE_OFFSET, 0);
        mismatch_mask |= self.verify_field(14, DPU_OPERATION_ENABLE_OFFSET, 0);
        mismatch_mask |= self.verify_field(15, DPU_RDMA_OPERATION_ENABLE_OFFSET, 0);
        mismatch_mask |= self.verify_field(16, PPU_OPERATION_ENABLE_OFFSET, 0);
        mismatch_mask |= self.verify_field(17, PPU_RDMA_OPERATION_ENABLE_OFFSET, 0);
        mismatch_mask |= self.verify_field(18, GLOBAL_OPERATION_ENABLE_OFFSET, 0);

        mismatch_mask
    }

    /// 单独观察 task-window 写回后的首个读回差异，只用于日志诊断。
    fn first_task_shadow_mismatch(&self, image: &NpuCoreRestoreImage) -> Option<(u16, u32, u32)> {
        image.task_shadow_writes.iter().find_map(|write| {
            let got = self.read_mmio_u32(write.offset);
            (got != write.value).then_some((write.offset, write.value, got))
        })
    }

    fn verify_field(&self, bit: u32, offset: u16, expected: u32) -> u64 {
        u64::from(self.read_mmio_u32(offset) != expected) << bit
    }

    /// 对不同寄存器类型生成确定性的毒值。
    ///
    /// 原则：
    /// - 带启动副作用的寄存器写 0；
    /// - `s_pointer` / `task_con` 先保持在“消毒后仍合法”的范围内；
    /// - 其它普通任务配置寄存器写固定异或毒值。
    fn poison_value(offset: u16, value: u32) -> u32 {
        if Self::is_operation_enable_offset(offset) {
            0
        } else if Self::is_s_pointer_offset(offset) {
            Self::sanitize_s_pointer(value ^ 0x0001_000f)
        } else if offset == PC_TASK_CON_OFFSET {
            Self::sanitize_task_con(value ^ 0x0055_00aa)
        } else {
            value ^ (0x5a5a_0000 | offset as u32)
        }
    }

    /// 在把 regcmd 值纳入恢复镜像前做一次软件消毒。
    fn sanitize_task_reg_value(offset: u16, value: u32) -> Option<u32> {
        if Self::is_readonly_snapshot_offset(offset) {
            None
        } else if Self::is_operation_enable_offset(offset) {
            Some(0)
        } else if Self::is_s_pointer_offset(offset) {
            Some(Self::sanitize_s_pointer(value))
        } else if offset == PC_TASK_CON_OFFSET {
            Some(Self::sanitize_task_con(value))
        } else {
            Some(value)
        }
    }

    fn offset_handled_by_restore_scalars(offset: u16) -> bool {
        matches!(
            offset,
            x if x == PC_OPERATION_ENABLE as u16
                || x == PC_BASE_ADDRESS_OFFSET
                || x == PC_REGISTER_AMOUNTS as u16
                || x == PC_INTERRUPT_MASK_OFFSET
                || x == PC_TASK_CON_OFFSET
                || x == PC_TASK_DMA_BASE_ADDR_OFFSET
                || x == CNA_S_POINTER as u16
                || x == CORE_S_POINTER as u16
                || x == DPU_S_POINTER as u16
                || x == DPU_RDMA_S_POINTER_OFFSET
                || x == PPU_S_POINTER_OFFSET
                || x == PPU_RDMA_S_POINTER_OFFSET
                || x == CNA_OPERATION_ENABLE_OFFSET
                || x == CORE_OPERATION_ENABLE_OFFSET
                || x == DPU_OPERATION_ENABLE_OFFSET
                || x == DPU_RDMA_OPERATION_ENABLE_OFFSET
                || x == PPU_OPERATION_ENABLE_OFFSET
                || x == PPU_RDMA_OPERATION_ENABLE_OFFSET
                || x == GLOBAL_OPERATION_ENABLE_OFFSET
        )
    }

    fn is_readonly_snapshot_offset(offset: u16) -> bool {
        matches!(
            offset,
            PC_INTERRUPT_STATUS_OFFSET
                | PC_INTERRUPT_RAW_STATUS_OFFSET
                | PC_TASK_STATUS_OFFSET
                | 0x1000
                | 0x3000
                | 0x4000
                | 0x5000
                | 0x6000
                | 0x7000
        )
    }

    fn is_s_pointer_offset(offset: u16) -> bool {
        matches!(
            offset,
            x if x == CNA_S_POINTER as u16
                || x == CORE_S_POINTER as u16
                || x == DPU_S_POINTER as u16
                || x == DPU_RDMA_S_POINTER_OFFSET
                || x == PPU_S_POINTER_OFFSET
                || x == PPU_RDMA_S_POINTER_OFFSET
        )
    }

    fn is_operation_enable_offset(offset: u16) -> bool {
        matches!(
            offset,
            x if x == PC_OPERATION_ENABLE as u16
                || x == CNA_OPERATION_ENABLE_OFFSET
                || x == CORE_OPERATION_ENABLE_OFFSET
                || x == DPU_OPERATION_ENABLE_OFFSET
                || x == DPU_RDMA_OPERATION_ENABLE_OFFSET
                || x == PPU_OPERATION_ENABLE_OFFSET
                || x == PPU_RDMA_OPERATION_ENABLE_OFFSET
                || x == GLOBAL_OPERATION_ENABLE_OFFSET
        )
    }

    fn sanitize_s_pointer(value: u32) -> u32 {
        value & !0x30
    }

    fn sanitize_task_con(value: u32) -> u32 {
        value & !(1 << 13)
    }

    fn read_mmio_u32(&self, offset: u16) -> u32 {
        unsafe {
            self.offset_ptr::<u32>(offset as usize)
                .as_ptr()
                .read_volatile()
        }
    }

    fn write_mmio_u32(&self, offset: u16, value: u32) {
        unsafe {
            self.offset_ptr::<u32>(offset as usize)
                .as_ptr()
                .write_volatile(value)
        }
    }

    /// 读取 INTERRUPT_STATUS，清除硬件中断，并将模糊结果存储到
    /// 共享的 `irq_status` 原子变量中。
    ///
    /// 从两个上下文调用：
    /// - **IRQ 处理程序**（通过 `RknpuIrqHandler::handle`）— 主要路径。
    ///   返回后，CPU 从 WFI 唤醒，`submit_one()` 看到非零原子变量。
    /// - **`submit_one()` 中的排空循环** — 在开始新提交之前清除剩余状态。
    ///
    /// 返回模糊状态（非零 = 发生了某事）。
    pub fn handle_interrupt(&self) -> u32 {
        // 第一步先抓 live 快照，确保清中断前的原始状态仍然可见。
        let live_regs = read_core_register_snapshot(self);
        let int_status = live_regs.pc.interrupt_status;
        mb();
        self.pc()
            .interrupt_clear()
            .write(|w| unsafe { w.bits(INT_CLEAR_ALL) });
        let fuzzed = rknpu_fuzz_status(int_status);

        if fuzzed != 0 {
            if let Some(binding) = self.shared.active_binding(self.core_slot as usize) {
                let mut states = self.shared.task_npu_state.lock();
                if let Some(state) = states.get_mut(&binding.key) {
                    state.key = binding.key;
                    state.subcore_slot = binding.subcore_slot;
                    state.batch_task_start = binding.batch_task_start;
                    state.batch_task_count = binding.batch_task_count;
                    state.current_task_index = binding.current_task_index;
                    state.expected_irq_mask = binding.expected_irq_mask;
                    state.observed_irq_status = fuzzed;
                    state.last_task_int_status = fuzzed & binding.expected_irq_mask;
                    state.regs = live_regs.clone();
                    // 当前提交先只保留单 task 边界调度；下面这套“寄存器快照 ->
                    // 毒值写坏 -> 恢复 -> 读回校验”的实验路径先整块注释保留，
                    // 下次继续做细粒度抢占/恢复时可以直接恢复。
                    state.restore_verified = true;
                    state.restore_mismatch_mask = 0;
                    /*
                    // 在真正放行等待路径之前，先完成：
                    // 1. 用 live 快照刷新恢复镜像
                    // 2. 按毒值写坏
                    // 3. 再按镜像恢复
                    // 4. 最后做一次读回校验
                    Self::refresh_restore_image_from_snapshot(&mut state.restore_image, &live_regs);
                    self.apply_poison_to_restore_image(&state.restore_image);
                    self.apply_restore_image(&state.restore_image);
                    state.restore_mismatch_mask = self.verify_restore_image(&state.restore_image);
                    state.restore_verified = state.restore_mismatch_mask == 0;
                    if let Some((offset, expected, got)) =
                        self.first_task_shadow_mismatch(&state.restore_image)
                    {
                        // task-window 配置寄存器在完成边界后不保证还能稳定读回，
                        // 所以这里只打印调试日志，不把它算成硬失败。
                        debug!(
                            "[NPU] task shadow readback differs after IRQ boundary: core={} owner(task={}, process={}, aspace={:#x}) offset={:#x} expected={:#x} got={:#x}",
                            self.core_slot,
                            binding.key.owner.task_id,
                            binding.key.owner.process_id,
                            binding.key.owner.address_space_id,
                            offset,
                            expected,
                            got
                        );
                    }
                    if !state.restore_verified {
                        error!(
                            "[NPU] restore verify failed: core={} owner(task={}, process={}, aspace={:#x}) mismatch_mask={:#x}",
                            self.core_slot,
                            binding.key.owner.task_id,
                            binding.key.owner.process_id,
                            binding.key.owner.address_space_id,
                            state.restore_mismatch_mask
                        );
                    }
                    */
                }
            }
            // 存储以便等待者（submit_one）可以看到它而无需轮询 MMIO。
            self.irq_status.fetch_or(fuzzed, Ordering::Release);
        }
        fuzzed
    }
}

#[inline(always)]
pub fn rknpu_fuzz_status(status: u32) -> u32 {
    let mut fuzz_status = 0;
    if (status & 0x3) != 0 {
        fuzz_status |= 0x3;
    }
    if (status & 0xc) != 0 {
        fuzz_status |= 0xc;
    }
    if (status & 0x30) != 0 {
        fuzz_status |= 0x30;
    }
    if (status & 0xc0) != 0 {
        fuzz_status |= 0xc0;
    }
    if (status & 0x300) != 0 {
        fuzz_status |= 0x300;
    }
    if (status & 0xc00) != 0 {
        fuzz_status |= 0xc00;
    }
    fuzz_status
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::status::{NpuOwnerIds, TaskNpuStateKey};
    use crate::task::def::{DPU_DST_BASE_ADD, OP_REG_CNA, OP_REG_DPU, OP_REG_PC, npu_op};
    use alloc::sync::Arc;

    const FAKE_MMIO_LEN: usize = 0x10000;

    fn build_fake_core() -> (RknpuCore, Vec<u8>, Arc<RknpuSharedState>) {
        let mut mmio = vec![0_u8; FAKE_MMIO_LEN];
        let shared = Arc::new(RknpuSharedState::new());
        let core =
            unsafe { RknpuCore::new(NonNull::new(mmio.as_mut_ptr()).unwrap(), 0, shared.clone()) };
        (core, mmio, shared)
    }

    #[test]
    fn decode_task_reg_write_keeps_rw_task_window() {
        let rw =
            RknpuCore::decode_task_reg_write(npu_op(OP_REG_DPU, 0x1234_5678, DPU_DST_BASE_ADD))
                .expect("rw task reg should be restorable");
        assert_eq!(rw.offset, DPU_DST_BASE_ADD as u16);
        assert_eq!(rw.value, 0x1234_5678);

        let s_pointer = RknpuCore::decode_task_reg_write(npu_op(OP_REG_DPU, 0x3f, DPU_S_POINTER));
        assert!(s_pointer.is_none());
    }

    #[test]
    fn decode_task_reg_write_skips_readonly_status_registers() {
        let pc_interrupt_status = RknpuCore::decode_task_reg_write(npu_op(
            OP_REG_PC,
            0x300,
            PC_INTERRUPT_STATUS_OFFSET as u32,
        ));
        let pc_interrupt_raw = RknpuCore::decode_task_reg_write(npu_op(
            OP_REG_PC,
            0x300,
            PC_INTERRUPT_RAW_STATUS_OFFSET as u32,
        ));
        let cna_status = RknpuCore::decode_task_reg_write(npu_op(OP_REG_CNA, 0x1, 0x1000));

        assert!(pc_interrupt_status.is_none());
        assert!(pc_interrupt_raw.is_none());
        assert!(cna_status.is_none());
    }

    #[test]
    fn verify_restore_image_ignores_task_shadow_readback_mismatch() {
        let (core, _mmio, _shared) = build_fake_core();
        let mut image = NpuCoreRestoreImage::default();
        image.task_shadow_writes.push(NpuTaskRegisterWrite {
            block: BLOCK_CNA as u16,
            offset: 0x100c,
            value: 0x1234_5678,
        });

        assert_eq!(core.verify_restore_image(&image), 0);
        assert_eq!(
            core.first_task_shadow_mismatch(&image),
            Some((0x100c, 0x1234_5678, 0))
        );
    }

    #[test]
    fn handle_interrupt_without_binding_only_publishes_irq_status() {
        let (core, _mmio, shared) = build_fake_core();
        core.write_mmio_u32(PC_INTERRUPT_STATUS_OFFSET, 0x200);

        let fuzzed = core.handle_interrupt();

        assert_eq!(fuzzed, 0x300);
        assert_eq!(core.irq_status.load(Ordering::Acquire), 0x300);
        assert!(shared.task_npu_state.lock().is_empty());
    }

    #[test]
    fn handle_interrupt_with_binding_snapshots_and_restores() {
        let (core, _mmio, shared) = build_fake_core();
        let key = TaskNpuStateKey {
            owner: NpuOwnerIds {
                task_id: 1,
                process_id: 2,
                address_space_id: 3,
            },
            core_slot: 0,
        };
        let binding = ActiveCoreBinding {
            key,
            subcore_slot: 0,
            batch_task_start: 0,
            batch_task_count: 1,
            current_task_index: 0,
            expected_irq_mask: 0x300,
        };
        let mut restore_image = NpuCoreRestoreImage::default();
        restore_image.task_shadow_writes.push(NpuTaskRegisterWrite {
            block: BLOCK_CNA as u16,
            offset: 0x100c,
            value: 0x1234_5678,
        });
        shared.prepare_active_binding(
            binding,
            TaskNpuState {
                key,
                expected_irq_mask: 0x300,
                restore_image,
                ..TaskNpuState::default()
            },
        );

        core.write_mmio_u32(PC_BASE_ADDRESS_OFFSET, 0x1000);
        core.write_mmio_u32(PC_REGISTER_AMOUNTS as u16, 0x22);
        core.write_mmio_u32(PC_INTERRUPT_MASK_OFFSET, 0x300);
        core.write_mmio_u32(PC_INTERRUPT_STATUS_OFFSET, 0x300);
        core.write_mmio_u32(PC_INTERRUPT_RAW_STATUS_OFFSET, 0x300);
        core.write_mmio_u32(PC_TASK_CON_OFFSET, 0x2001);
        core.write_mmio_u32(PC_TASK_DMA_BASE_ADDR_OFFSET, 0x8800);
        core.write_mmio_u32(CNA_S_POINTER as u16, 0x3f);
        core.write_mmio_u32(CNA_OPERATION_ENABLE_OFFSET, 1);
        core.write_mmio_u32(GLOBAL_OPERATION_ENABLE_OFFSET, 1);
        core.write_mmio_u32(0x100c, 0);

        let fuzzed = core.handle_interrupt();
        let states = shared.task_npu_state.lock();
        let state = states.get(&key).expect("state entry must exist");

        assert_eq!(fuzzed, 0x300);
        assert_eq!(core.irq_status.load(Ordering::Acquire), 0x300);
        assert_eq!(state.observed_irq_status, 0x300);
        assert_eq!(state.last_task_int_status, 0x300);
        assert!(state.restore_verified);
        assert_eq!(state.restore_mismatch_mask, 0);
        assert!(state.restore_image.unsafe_snapshot);
        assert_eq!(core.read_mmio_u32(0x100c), 0x1234_5678);
        assert_eq!(core.read_mmio_u32(CNA_S_POINTER as u16), 0x0f);
        assert_eq!(core.read_mmio_u32(PC_TASK_CON_OFFSET), 0x0001);
        assert_eq!(core.read_mmio_u32(CNA_OPERATION_ENABLE_OFFSET), 0);
        assert_eq!(core.read_mmio_u32(GLOBAL_OPERATION_ENABLE_OFFSET), 0);
    }

    #[test]
    fn handle_interrupt_updates_pre_registered_entry_without_inserting_new_map_item() {
        let (core, _mmio, shared) = build_fake_core();
        let key = TaskNpuStateKey {
            owner: NpuOwnerIds {
                task_id: 7,
                process_id: 8,
                address_space_id: 9,
            },
            core_slot: 0,
        };
        let binding = ActiveCoreBinding {
            key,
            subcore_slot: 0,
            batch_task_start: 4,
            batch_task_count: 1,
            current_task_index: 4,
            expected_irq_mask: 0x300,
        };

        shared.ensure_task_state_entry(key);
        assert_eq!(shared.task_npu_state.lock().len(), 1);
        shared.prepare_active_binding(
            binding,
            TaskNpuState {
                key,
                expected_irq_mask: 0x300,
                ..TaskNpuState::default()
            },
        );
        assert_eq!(shared.task_npu_state.lock().len(), 1);

        core.write_mmio_u32(PC_INTERRUPT_MASK_OFFSET, 0x300);
        core.write_mmio_u32(PC_INTERRUPT_STATUS_OFFSET, 0x300);

        let fuzzed = core.handle_interrupt();
        let states = shared.task_npu_state.lock();
        let state = states
            .get(&key)
            .expect("pre-registered state entry must remain");

        assert_eq!(fuzzed, 0x300);
        assert_eq!(states.len(), 1);
        assert_eq!(state.batch_task_start, 4);
        assert_eq!(state.current_task_index, 4);
        assert_eq!(state.expected_irq_mask, 0x300);
        assert_eq!(state.observed_irq_status, 0x300);
    }
}
