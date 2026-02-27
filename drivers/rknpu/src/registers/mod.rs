//! 瑞芯微 NPU 的内存映射寄存器定义。
//!
//! 寄存器块类型来自 [`rknpu_regs`] crate（svd2rust 生成）。
//! 本模块提供 [`RknpuCore`]，一个将动态 MMIO 基地址转换为
//! 适当寄存器块引用的薄包装器。

use ::core::ptr::NonNull;
use ::core::sync::atomic::{AtomicU32, Ordering};

use alloc::sync::Arc;
use mbarrier::mb;

use crate::{JobMode, RknpuError, Submit, SubmitRef, data::RknpuData};

use crate::ioctrl::RknpuSubmit;
use crate::RknpuTask;
use crate::Rknpu;
use crate::SubmitBase;
use core::hint::spin_loop;

const INT_CLEAR_ALL: u32 = 0x1_FFFF;
const RKNPU_PC_DATA_EXTRA_AMOUNT: u32 = 4;

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
    /// 共享中断状态标志 — 由 IRQ 处理程序写入，由提交读取。
    /// 零表示没有挂起的中断；非零保存模糊状态位。
    pub(crate) irq_status: Arc<AtomicU32>,
}
unsafe impl Send for RknpuCore {}

impl RknpuCore {




    ///提交并执行1个任务
    pub fn start_execute_one(&mut self, idx:usize,data: &RknpuData,rknpu_task: &mut RknpuTask,args: &RknpuSubmit) -> Result<(), RknpuError> {
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
            debug!("[NPU]   batch: {} tasks, regcmd@{:#x}, {} regcfg_words, int_mask={:#x}",
                  1, job.regcmd_base_addr, job.base.regcfg_amount, job.base.int_mask);
            debug!("Submit {} jobs: {job:#x?}", 1);

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
    pub fn submit_one(&mut self,data: &RknpuData,wait_fn:Option<fn()>, idx: usize, args: &mut RknpuSubmit) -> Result<usize, RknpuError> {
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
            debug!("[NPU]   batch: {} tasks, regcmd@{:#x}, {} regcfg_words, int_mask={:#x}",
                  task_number, job.regcmd_base_addr, job.base.regcfg_amount, job.base.int_mask);
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
    pub unsafe fn new(base_addr: NonNull<u8>) -> Self {
        Self {
            base: base_addr,
            irq_status: Arc::new(AtomicU32::new(0)),
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
        self.pc().version().read().bits()
            .wrapping_add(self.pc().version_num().read().bits() & 0xffff)
    }

    pub fn clean_interrupts(&self) {
        self.pc().interrupt_clear().write(|w| unsafe { w.bits(INT_CLEAR_ALL) });
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

        self.pc().base_address().write(|w| unsafe { w.bits(pc_base_addr) });

        let amount = (args.base.regcfg_amount + RKNPU_PC_DATA_EXTRA_AMOUNT)
            .div_ceil(pc_data_amount_scale)
            - 1;

        debug!("Set PC REGISTER_AMOUNTS to {:#x}", amount);

        self.pc().register_amounts().write(|w| unsafe { w.bits(amount) });

        self.pc().interrupt_mask().write(|w| unsafe { w.bits(args.base.int_mask) });
        self.pc().interrupt_clear().write(|w| unsafe { w.bits(args.base.int_clear) });
        let task_number = args.task_number as u32;

        let task_control = ((0x6 | task_pp_en) << pc_task_number_bits) | task_number;
        debug!("Set PC TASK_CONTROL to {:#x}", task_control);
        self.pc().task_con().write(|w| unsafe { w.bits(task_control) });
        debug!(
            "Set PC TASK_DMA_BASE_ADDR to {:#x}",
            args.base.task_base_addr
        );
        self.pc().task_dma_base_addr().write(|w| unsafe { w.bits(args.base.task_base_addr) });
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
        let int_status = self.pc().interrupt_status().read().bits();
        mb();
        self.pc().interrupt_clear().write(|w| unsafe { w.bits(INT_CLEAR_ALL) });
        let fuzzed = rknpu_fuzz_status(int_status);
        if fuzzed != 0 {
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
