use core::cell::UnsafeCell;
use crate::RknpuIrqHandler;

/// Mutable storage slot used to hold one installed IRQ handler.
pub struct IrqSlot(pub UnsafeCell<Option<RknpuIrqHandler>>);

/// The slot may be shared across interrupt and probe contexts.
unsafe impl Sync for IrqSlot {}

/// Global per-core IRQ handler table populated during probe.
pub static NPU_IRQ_HANDLERS: [IrqSlot; 3] = [
    IrqSlot(UnsafeCell::new(None)),
    IrqSlot(UnsafeCell::new(None)),
    IrqSlot(UnsafeCell::new(None)),
];

/// Static trampoline table registered with the platform IRQ framework.
pub const NPU_IRQ_FNS: [fn(); 3] = [
    handle_npu_irq_core0,
    handle_npu_irq_core1,
    handle_npu_irq_core2,
];


/// IRQ entry point for NPU core 0.
///
/// The platform IRQ framework can only register a plain `fn()`, so probe code
/// stores the real [`RknpuIrqHandler`] in a global slot first. This lightweight
/// trampoline then forwards the interrupt in IRQ context. If the slot has not
/// been initialized yet, the interrupt is ignored.
fn handle_npu_irq_core0() {
    unsafe {
        if let Some(h) = &*NPU_IRQ_HANDLERS[0].0.get() {
            h.handle();
        }
    }
}

/// IRQ entry point for NPU core 1.
///
/// This is the core-1 variant of [`handle_npu_irq_core0`]. It reads the handler
/// stored for core 1 and forwards the interrupt if the slot is populated.
fn handle_npu_irq_core1() {
    unsafe {
        if let Some(h) = &*NPU_IRQ_HANDLERS[1].0.get() {
            h.handle();
        }
    }
}

/// IRQ entry point for NPU core 2.
///
/// This is the final per-core trampoline in the three-core RK3588 layout. It
/// transfers the platform callback to the core-2 [`RknpuIrqHandler`].
fn handle_npu_irq_core2() {
    unsafe {
        if let Some(h) = &*NPU_IRQ_HANDLERS[2].0.get() {
            h.handle();
        }
    }
}
