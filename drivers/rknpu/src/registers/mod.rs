//! Memory-mapped register definitions for the Rockchip NPU.
//!
//! Register block types come from the [`rknpu_regs`] crate (svd2rust generated).
//! This module provides [`RknpuCore`], a thin wrapper that casts a dynamic MMIO
//! base address to the appropriate register block references.

use ::core::ptr::NonNull;
use ::core::sync::atomic::{AtomicU32, Ordering};

use alloc::sync::Arc;
use mbarrier::mb;

use crate::{JobMode, RknpuError, Submit, SubmitRef, data::RknpuData};

const INT_CLEAR_ALL: u32 = 0x1_FFFF;
const RKNPU_PC_DATA_EXTRA_AMOUNT: u32 = 4;

#[derive(Clone)]
/// Immutable view over one NPU core's MMIO register window.
///
/// # Interrupt-driven completion
///
/// Each core owns a shared `irq_status` atomic.  When an IRQ fires, the
/// platform calls [`handle_interrupt()`] (typically via [`RknpuIrqHandler`]),
/// which reads `INTERRUPT_STATUS`, clears the hardware interrupt, and
/// stores the fuzzed status into `irq_status`.  The submission code in
/// `ioctrl.rs` can then check the atomic instead of busy-polling the
/// MMIO register, allowing the CPU to sleep (WFI) between checks.
pub struct RknpuCore {
    base: NonNull<u8>,
    /// Shared interrupt status flag — written by IRQ handler, read by submit.
    /// Zero means no interrupt pending; non-zero holds fuzzed status bits.
    pub(crate) irq_status: Arc<AtomicU32>,
}
unsafe impl Send for RknpuCore {}

impl RknpuCore {
    /// # Safety
    ///
    /// Caller must ensure the pointer maps the RKNN register space for the
    /// lifetime of the returned structure.
    pub unsafe fn new(base_addr: NonNull<u8>) -> Self {
        Self {
            base: base_addr,
            irq_status: Arc::new(AtomicU32::new(0)),
        }
    }

    #[inline(always)]
    pub(crate) fn offset_ptr<T>(&self, offset: usize) -> NonNull<T> {
        // SAFETY: caller guarantees the MMIO mapping is valid; offsets come
        // from the hardware documentation.
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

    /// Read INTERRUPT_STATUS, clear the hardware interrupt, and store the
    /// fuzzed result into the shared `irq_status` atomic.
    ///
    /// Called from two contexts:
    /// - **IRQ handler** (via `RknpuIrqHandler::handle`) — the primary path.
    ///   After this returns, the CPU wakes from WFI and `submit_one()` sees
    ///   the non-zero atomic.
    /// - **Drain loop** in `submit_one()` — to clear leftover status before
    ///   starting a new submission.
    ///
    /// Returns the fuzzed status (non-zero = something happened).
    pub fn handle_interrupt(&self) -> u32 {
        let int_status = self.pc().interrupt_status().read().bits();
        mb();
        self.pc().interrupt_clear().write(|w| unsafe { w.bits(INT_CLEAR_ALL) });
        let fuzzed = rknpu_fuzz_status(int_status);
        if fuzzed != 0 {
            // Store so the waiter (submit_one) can see it without polling MMIO.
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
