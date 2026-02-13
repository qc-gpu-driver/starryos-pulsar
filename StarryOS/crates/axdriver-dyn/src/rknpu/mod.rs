use alloc::vec::Vec;
use core::cell::UnsafeCell;

use log::{info, warn};
use rdrive::{PlatformDevice, module_driver, probe::OnProbeError, register::FdtInfo};
use rknpu::{Rknpu, RknpuConfig, RknpuIrqHandler, RknpuType};
use rockchip_pm::{PD, RockchipPM};

use crate::iomap;

// ── NPU 中断异步机制 ────────────────────────────────────────────────
//
// 问题：axklib::irq::register 只接受 fn(usize)（函数指针），不能传闭包。
// 但 RknpuIrqHandler 是有状态的结构体（里面有 MMIO 指针 + Arc<AtomicU32>）。
//
// 解决：用全局 static 存放每个核心的 RknpuIrqHandler，然后用 3 个独立的
// fn(usize) 分别访问对应的 static slot。
//
// 安全性：probe 时写入一次，之后只在 IRQ 上下文读取。单核初始化，无竞争。
// ────────────────────────────────────────────────────────────────────

/// Wrapper to make `UnsafeCell<Option<RknpuIrqHandler>>` Sync.
/// Safe because: written once during probe (single-threaded init),
/// then only read from IRQ context (no concurrent writes).
struct IrqSlot(UnsafeCell<Option<RknpuIrqHandler>>);
unsafe impl Sync for IrqSlot {}

static NPU_IRQ_HANDLERS: [IrqSlot; 3] = [
    IrqSlot(UnsafeCell::new(None)),
    IrqSlot(UnsafeCell::new(None)),
    IrqSlot(UnsafeCell::new(None)),
];

/// 每个核心的 IRQ handler 函数（fn() 函数指针，匹配 axklib::IrqHandler）
fn handle_npu_irq_core0() {
    unsafe {
        if let Some(h) = &*NPU_IRQ_HANDLERS[0].0.get() {
            h.handle();
        }
    }
}
fn handle_npu_irq_core1() {
    unsafe {
        if let Some(h) = &*NPU_IRQ_HANDLERS[1].0.get() {
            h.handle();
        }
    }
}
fn handle_npu_irq_core2() {
    unsafe {
        if let Some(h) = &*NPU_IRQ_HANDLERS[2].0.get() {
            h.handle();
        }
    }
}

const NPU_IRQ_FNS: [fn(); 3] = [
    handle_npu_irq_core0,
    handle_npu_irq_core1,
    handle_npu_irq_core2,
];

/// 解析 ARM GIC 设备树中断描述 [type, number, flags] → GIC IRQ 号
///
/// - type 0 (SPI): IRQ = number + 32
/// - type 1 (PPI): IRQ = number + 16
fn fdt_irq_to_gic_num(cells: &[u32]) -> usize {
    let irq_type = cells[0];
    let irq_num = cells[1] as usize;
    match irq_type {
        0 => irq_num + 32, // SPI (Shared Peripheral Interrupt)
        1 => irq_num + 16, // PPI (Private Peripheral Interrupt)
        _ => panic!("Unknown GIC interrupt type: {}", irq_type),
    }
}

/// WFI + 短暂让步 — CPU 睡眠直到中断唤醒，然后短暂 busy_wait
/// 让 timer 中断有机会触发调度（处理 Ctrl+C 等信号）。
///
/// 工作原理：
/// 1. WFI → CPU 睡觉，NPU 中断或 timer 中断都能唤醒
/// 2. 唤醒后 busy_wait(10μs) → 如果 timer 中断到了，
///    preempt 调度器会抢占当前任务，让信号处理有机会运行
/// 3. 如果 NPU 已完成（atomic 非零），submit_one 直接退出循环
///
/// 代价：每次唤醒多 ~10μs，NPU 任务通常 >100μs，影响可忽略
#[cfg(target_arch = "aarch64")]
fn wfi_and_relax() {
    unsafe { core::arch::asm!("wfi") };
    axklib::time::busy_wait(core::time::Duration::from_micros(10));
}

module_driver!(
    name: "Rockchip NPU",
    level: ProbeLevel::PostKernel,
    priority: ProbePriority::DEFAULT,
    probe_kinds: &[
        ProbeKind::Fdt {
            compatibles: &["rockchip,rk3588-rknpu"],
            on_probe: probe
        }
    ],
);

fn probe(info: FdtInfo<'_>, plat_dev: PlatformDevice) -> Result<(), OnProbeError> {
    let mut config = None;
    for c in info.node.compatibles() {
        if c == "rockchip,rk3588-rknpu" {
            config = Some(RknpuConfig {
                rknpu_type: RknpuType::Rk3588,
            });
            break;
        }
    }

    let config = config.expect("Unsupported RKNPU compatible");
    let regs = info.node.reg().unwrap();

    let mut base_regs = Vec::new();
    let page_size = 0x1000;
    for reg in regs {
        let start_raw = reg.address as usize;
        let end = start_raw + reg.size.unwrap_or(0x1000);

        let start = start_raw & !(page_size - 1);
        let offset = start_raw - start;
        let end = (end + page_size - 1) & !(page_size - 1);
        let size = end - start;

        base_regs.push(unsafe { iomap(start as _, size)?.add(offset) });
    }

    enable_pm();

    info!("NPU power enabled");

    #[allow(unused_mut)] // mut needed for set_wait_fn on aarch64
    let mut npu = Rknpu::new(&base_regs, config);

    // ── 注册 NPU 中断 ──────────────────────────────────────────────
    //
    // 流程：
    //  1. 从 FDT 解析每个核心的中断号（RK3588 有 3 个 SPI 中断）
    //  2. 从 Rknpu 取出轻量级 IRQ handler（共享 Arc<AtomicU32>）
    //  3. 存入全局 static（因为 axklib 只接受 fn 指针）
    //  4. 注册到 GIC 中断框架
    //  5. 启用 WFI 等待模式
    //
    // 之后 NPU 完成任务时：
    //  NPU 触发中断 → GIC 路由 → handle_npu_irq_coreN() →
    //  RknpuIrqHandler::handle() → 读寄存器 → 写 irq_status atomic →
    //  CPU 从 WFI 醒来 → submit_one 看到 atomic 非零 → 完成
    // ────────────────────────────────────────────────────────────────
    let interrupts = info.interrupts();
    for (i, irq_cells) in interrupts.iter().enumerate() {
        if i >= 3 {
            break;
        }
        let gic_irq = fdt_irq_to_gic_num(irq_cells);

        // 取出 IRQ handler 并存入全局 static
        let handler = npu.new_irq_handler(i);
        unsafe { *NPU_IRQ_HANDLERS[i].0.get() = Some(handler) };

        // 注册到平台 IRQ 框架（同时自动 enable 该中断线）
        axklib::irq::register(gic_irq, NPU_IRQ_FNS[i]);
        warn!("[NPU] Core {} IRQ registered: GIC #{}", i, gic_irq);
    }

    // 启用中断驱动等待模式（WFI 替代忙轮询）
    #[cfg(target_arch = "aarch64")]
    npu.set_wait_fn(wfi_and_relax);

    plat_dev.register(npu);
    warn!("NPU registered successfully");
    Ok(())
}

fn enable_pm() {
    // RK3588 NPU 相关电源域 ID

    /// NPU 主电源域
    pub const NPU: PD = PD(8);
    /// NPU TOP 电源域  
    pub const NPUTOP: PD = PD(9);
    /// NPU1 电源域
    pub const NPU1: PD = PD(10);
    /// NPU2 电源域
    pub const NPU2: PD = PD(11);

    let mut pm = rdrive::get_one::<RockchipPM>().unwrap().lock().unwrap();

    pm.power_domain_on(NPUTOP).unwrap();
    pm.power_domain_on(NPU).unwrap();
    pm.power_domain_on(NPU1).unwrap();
    pm.power_domain_on(NPU2).unwrap();
}
