use alloc::format;
use rdrive::module_driver;
use rdrive::register::FdtInfo;
use rdrive::PlatformDevice;
use rdrive::probe::OnProbeError;

module_driver! {
    name: "RKNPU",
    level: ProbeLevel::PostKernel,
    priority: ProbePriority::DEFAULT,
    probe_kinds: &[
        ProbeKind::Fdt {
            compatibles: &["rockchip,rk3588-rknn"],
            on_probe: probe_rknpu
        }
    ],
}


fn probe_rknpu(info: FdtInfo<'_>, dev: PlatformDevice) -> Result<(), OnProbeError> {
    // 1. 节点名
    let name = info.node.name();
    log::info!("[RKNPU] probing node: {}", name);

    // 2. compatible 字符串列表
    if let Some(compatible) = info.node.compatible() {
        for c in compatible {
            if let Ok(cp) = c {
                log::info!("[RKNPU] compatible match: {}", cp);
            }
        }
    }

    // 3. reg — MMIO 基址和大小
    //    DTB 里 rknpu 节点: reg = <0x0 0xfdab0000 0x0 0x9000>
    //    解析后: address = 0xfdab0000, size = 0x9000
    let mut regs = info.node.reg().ok_or_else(|| {
        OnProbeError::other(format!("[RKNPU] node '{}' has no reg property", name))
    })?;

    let base_reg = regs.next().ok_or_else(|| {
        OnProbeError::other(format!("[RKNPU] node '{}' reg is empty", name))
    })?;

    let mmio_base = base_reg.address as usize;
    let mmio_size = base_reg.size.unwrap_or(0x9000);
    log::info!("[RKNPU] MMIO base: {:#x}, size: {:#x}", mmio_base, mmio_size);

    // 4. interrupts — 3 个 SPI 中断（对应 3 个 NPU 核心）
    //    DTB 里: interrupts = <GIC_SPI 110 IRQ_TYPE_LEVEL_HIGH>,
    //                         <GIC_SPI 111 IRQ_TYPE_LEVEL_HIGH>,
    //                         <GIC_SPI 112 IRQ_TYPE_LEVEL_HIGH>;
    //    每组 3 个 cell: [type, number, flags]
    if let Some(interrupts) = info.node.interrupts() {
        for (i, irq) in interrupts.enumerate() {
            let cells: alloc::vec::Vec<u32> = irq.collect();
            log::info!("[RKNPU] IRQ[{}]: {:?}", i, cells);
        }
    }

    // TODO: iomap 映射 MMIO 区域，初始化 NPU 硬件
    // let vaddr = iomap(PhysAddr::from(mmio_base), mmio_size)?;
    // let npu = RkNpu::new(vaddr, mmio_size);
    // dev.register(npu);

    log::info!("[RKNPU] probe complete for '{}'", name);
    Ok(())
}