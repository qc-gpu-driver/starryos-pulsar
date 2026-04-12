use alloc::vec;
use core::{
    any::Any,
    convert::TryFrom,
    ffi::{CStr, c_char, c_ulong},
    mem,
};

use axfs_ng_vfs::{DeviceId, NodeFlags, VfsError, VfsResult};
use memory_addr::{MemoryAddr, PhysAddrRange};
use rknpu::{
    RknpuAction, RknpuQueuedSubmit, RknpuTask,
    ioctrl::{RknpuMemCreate, RknpuMemDestroy, RknpuMemMap, RknpuSubmit},
};
use starry_core::vfs::DeviceMmap;

use super::{
    card0::{RknpuCmd, copy_from_user, copy_to_user},
    drm::DrmVersion,
    rknpu_scheduler::{enqueue_submit, take_terminal_submit, wait_for_submit},
};
use crate::vfs::{
    DeviceOps,
    dev::drm::{io_size, ioctl_nr, is_driver_ioctl},
};

/// Driver name for DRM device
const DRM1_NAME: &CStr = c"rknpu";
/// Driver date for DRM device
const DRM1_DATE: &CStr = c"20240828";
/// Driver description for DRM device
const DRM1_DESC: &CStr = c"RKNPU driver";

/// Device ID for /dev/dri/card1
pub const CARD1_SYSTEM_DEVICE_ID: DeviceId = DeviceId::new(0xe2, 1);

/// Device ID for /dev/rknpu (pick an unused major/minor)
pub const RKNPU_DEVICE_ID: DeviceId = DeviceId::new(251, 0);

/// Page shift constant (4KB pages)
const PAGE_SHIFT: u32 = 12;
/// Maximum ioctl command number
const MAX_IOCTL_NR: u32 = 0xcf;
/// Stack data buffer size
const STACK_DATA_SIZE: usize = 128;
/// DRM ioctl version command number
const DRM_IOCTL_VERSION_NR: u32 = 0;
/// DRM ioctl get unique command number
const DRM_IOCTL_GET_UNIQUE_NR: u32 = 1;
/// DRM ioctl gem flink command number
const DRM_IOCTL_GEM_FLINK_NR: u32 = 10;
/// DRM ioctl prime handle to fd command number
const DRM_IOCTL_PRIME_HANDLE_TO_FD_NR: u32 = 0x2d;

/// DRM_IOCTL_VERSION ioctl argument type
#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct DrmUnique {
    /// Length of unique string identifier
    pub unique_len: c_ulong,
    /// Pointer to user-space buffer holding unique name for driver
    /// instantiation
    pub unique: *mut c_char,
}

/// Represents an RKNPU user action with flags and value
#[repr(C)]
#[derive(Debug, Copy, Clone)]
struct RknpuUserAction {
    /// Action flags
    pub flags: RknpuAction,
    /// Action value
    pub value: u32,
}

impl RknpuUserAction {
    /// Creates a new RknpuUserAction with default values
    pub fn default() -> Self {
        Self {
            flags: RknpuAction::GetDrvVersion,
            value: 0,
        }
    }
}

/// DRM card1 device implementation
pub struct Card1;

impl Card1 {
    /// Creates a new /dev/dri/card1 device.
    pub fn new() -> Card1 {
        Self
    }
}

impl Default for Card1 {
    fn default() -> Self {
        Self::new()
    }
}

impl DeviceOps for Card1 {
    /// Reads data from the device (not supported for card1)
    fn read_at(&self, _buf: &mut [u8], _offset: u64) -> VfsResult<usize> {
        trace!("dri: read_at called");
        // card1 heap devices are not meant to be read directly
        Err(VfsError::InvalidInput)
    }

    /// Writes data to the device (not supported for card1)
    fn write_at(&self, _buf: &[u8], _offset: u64) -> VfsResult<usize> {
        trace!("dri: write_at called");
        // card1 heap devices are not meant to be written directly
        Err(VfsError::InvalidInput)
    }

    /// Handles ioctl commands for the device
    fn ioctl(&self, cmd: u32, arg: usize) -> VfsResult<usize> {
        if arg == 0 {
            warn!("[rknpu]: ioctl received null arg pointer");
            return Err(VfsError::InvalidData);
        }
        let nr = ioctl_nr(cmd);
        info!("card1: cmd {cmd:#x}, nr {nr:#x}, arg {arg:#x}");

        let is_driver_ioctl = is_driver_ioctl(ioctl_nr(cmd));
        info!("card1: is_driver_ioctl = {}", is_driver_ioctl);

        if is_driver_ioctl {
            if let Ok(op) = RknpuCmd::try_from(nr) {
                rknpu_driver_ioctl(op, arg)?;
            } else {
                warn!("Unknown RKNPU cmd: {:#x}", cmd);
                return Err(VfsError::BadIoctl);
            }
        } else {
            assert!(nr <= MAX_IOCTL_NR, "card1: unsupported ioctl nr {nr}");
            let mut stack_data = [0u8; STACK_DATA_SIZE];

            let in_size = io_size(cmd) as usize;
            let out_size = in_size;

            copy_from_user(stack_data.as_mut_ptr(), arg as _, in_size)?;
            match nr {
                DRM_IOCTL_VERSION_NR => {
                    info!("drm get version");
                    drm_version(&mut stack_data)?;
                }
                DRM_IOCTL_GET_UNIQUE_NR => {
                    info!("drm get unique");
                    drm_get_unique(&mut stack_data)?;
                }
                DRM_IOCTL_GEM_FLINK_NR => {
                    drm_gem_flink_ioctl(&mut stack_data)?;
                }
                DRM_IOCTL_PRIME_HANDLE_TO_FD_NR => {
                    drm_prime_handle_to_fd_ioctl(&mut stack_data)?;
                }

                _ => {
                    panic!("card1: unsupported ioctl nr {nr:#x}");
                }
            }
            copy_to_user(arg as _, stack_data.as_mut_ptr(), out_size)?;
        }

        Ok(0)
    }

    /// Returns a reference to the object as Any for dynamic type checking
    fn as_any(&self) -> &dyn Any {
        self
    }

    /// Returns the node flags for the device
    fn flags(&self) -> NodeFlags {
        NodeFlags::NON_CACHEABLE
    }

    /// Maps device memory to user space
    fn mmap(&self, offset: u64) -> starry_core::vfs::DeviceMmap {
        const PAGE_SHIFT: u32 = 12;
        const PAGE_SIZE: usize = 4096;

        let handle = (offset >> PAGE_SHIFT) as u32;

        with_npu(|rknpu_dev| {
            match rknpu_dev.get_phys_addr_and_size(handle) {
                Some((phys_addr, size)) => {
                    let range_size = if size < PAGE_SIZE {
                        PAGE_SIZE
                    } else {
                        size.align_up(PAGE_SIZE) // 向上对齐
                    };

                    info!(
                        "card1: mmap handle={}, phys={:#x}, orig_size={:#x}, range_size={:#x}",
                        handle, phys_addr, size, range_size
                    );

                    Ok(DeviceMmap::Physical(PhysAddrRange::new(
                        (phys_addr as usize).into(),
                        (phys_addr as usize + range_size).into(),
                    )))
                }
                None => {
                    warn!("card1: mmap invalid handle={}", handle);
                    Ok(DeviceMmap::None)
                }
            }
        })
        .unwrap_or(DeviceMmap::None)
    }
}

/// Gets a reference to the NPU device
pub fn npu() -> Result<rdrive::DeviceGuard<::rknpu::Rknpu>, VfsError> {
    rdrive::get_one()
        .ok_or(VfsError::NotFound)?
        .lock()
        .map_err(|_| VfsError::AddressInUse)
}

/// Executes a function with the NPU device
pub fn with_npu<F, R>(f: F) -> Result<R, VfsError>
where
    F: FnOnce(&mut ::rknpu::Rknpu) -> Result<R, VfsError>,
{
    let mut npu = npu()?;
    f(&mut npu)
}

/// Handles RKNPU action ioctl commands
pub fn rknpu_driver_ioctl(op: RknpuCmd, arg: usize) -> VfsResult<usize> {
    info!("rknpu_driver_ioctl: op = {:?}", op);
    match op {
        RknpuCmd::Submit => {
            // This ioctl still looks blocking from userspace, but internally
            // the calling thread no longer advances the submit by repeatedly
            // dispatching tasks itself.
            //
            // The new lifecycle is:
            //   1. copy the userspace submit/task array into kernel shadow memory
            //   2. enqueue one whole submit into the global NPU scheduler
            //   3. put only this submitter thread to sleep
            //   4. let the worker thread dispatch and harvest work per core
            //   5. wake this thread once the whole submit becomes terminal
            //
            // While this thread sleeps, other threads continue running
            // normally. They may enter the same ioctl path and enqueue new NPU
            // submits, which is how multiple processes share the device now.
            let mut submit_args = RknpuSubmit::default();
            copy_from_user(
                &mut submit_args as *mut _ as *mut u8,
                arg as *const u8,
                mem::size_of::<RknpuSubmit>(),
            )?;
            info!("rknpu submit ioctl {submit_args:#x?}");

            if submit_args.task_number == 0 || submit_args.task_obj_addr == 0 {
                warn!(
                    "rknpu invalid submit header: task_number={}, task_obj_addr={:#x}, \
                     task_base_addr={:#x}",
                    submit_args.task_number, submit_args.task_obj_addr, submit_args.task_base_addr,
                );
                warn!("rknpu submit ioctl rejected invalid submit header");
                return Err(VfsError::InvalidData);
            }

            if submit_args.task_base_addr == 0 {
                debug!(
                    "rknpu submit header keeps legacy zero task_base_addr, scheduler will \
                     preserve zero DMA base"
                );
            }

            let user_task_obj_addr = submit_args.task_obj_addr;
            let task_bytes = (submit_args.task_number as usize)
                .checked_mul(mem::size_of::<RknpuTask>())
                .ok_or(VfsError::InvalidData)?;
            let mut tasks = vec![RknpuTask::default(); submit_args.task_number as usize];
            copy_from_user(
                tasks.as_mut_ptr() as *mut u8,
                user_task_obj_addr as *const u8,
                task_bytes,
            )?;

            // `enqueue_submit()` only inserts the task into the scheduler and
            // kicks the worker if needed. It does not run the whole submission
            // inline on behalf of this syscall thread.
            warn!(
                "[rknpu-submit] queueing blocking submit task_number={} core_mask={:#x} \
                 timeout={} task_base_addr={:#x} user_task_obj_addr={:#x}",
                submit_args.task_number,
                submit_args.core_mask,
                submit_args.timeout,
                submit_args.task_base_addr,
                user_task_obj_addr
            );
            let queue_task_id = enqueue_submit(RknpuQueuedSubmit::new(submit_args.clone(), tasks))
                .inspect_err(|e| warn!("[rknpu-submit] enqueue_submit failed: {:?}", e))?;

            warn!(
                "[rknpu-submit] enqueued queue_task={} and entering blocking wait",
                queue_task_id
            );

            // Sleep until the queue entry is terminal. This blocks only the
            // current thread. The worker keeps driving the NPU, and other CPU
            // threads remain free to run or enqueue more NPU work.
            wait_for_submit(queue_task_id).inspect_err(|e| {
                warn!(
                    "[rknpu-submit] wait_for_submit failed for queue_task={}: {:?}",
                    queue_task_id, e
                )
            })?;

            warn!(
                "[rknpu-submit] blocking wait finished for queue_task={}, collecting terminal \
                 snapshot",
                queue_task_id
            );

            // After wake-up the queue entry is already terminal. Copy the
            // kernel-owned shadow task array back into the caller's task
            // buffer, then write the final submit header back to the ioctl
            // argument so userspace sees the completed counters/status fields.
            let finished = take_terminal_submit(queue_task_id).inspect_err(|e| {
                warn!(
                    "[rknpu-submit] take_terminal_submit failed for queue_task={}: {:?}",
                    queue_task_id, e
                )
            })?;
            let mut finished_submit = finished.submit;
            finished_submit.task_obj_addr = user_task_obj_addr;

            warn!(
                "[rknpu-submit] terminal queue_task={} task_counter={} last_error={:?}",
                queue_task_id, finished_submit.task_counter, finished.last_error
            );

            copy_to_user(
                user_task_obj_addr as *mut u8,
                finished.tasks.as_ptr() as *const u8,
                task_bytes,
            )?;

            copy_to_user(
                arg as *mut u8,
                &finished_submit as *const _ as *const u8,
                mem::size_of::<RknpuSubmit>(),
            )?;

            if let Some(err) = finished.last_error {
                warn!("rknpu submit ioctl completed with driver error: {:?}", err);
                return Err(VfsError::InvalidData);
            }
        }
        RknpuCmd::MemCreate => {
            info!("rknpu mem_create ioctl");
            let mut mem_create_args = RknpuMemCreate::default();

            copy_from_user(
                &mut mem_create_args as *mut _ as *mut u8,
                arg as *const u8,
                mem::size_of::<RknpuMemCreate>(),
            )?;

            with_npu(|rknpu_dev| {
                rknpu_dev
                    .create(&mut mem_create_args)
                    .map_err(|_| VfsError::InvalidData)
            })
            .inspect_err(|e| warn!("rknpu mem_create ioctl failed: {:?}", e))?;

            copy_to_user(
                arg as *mut u8,
                &mem_create_args as *const _ as *const u8,
                mem::size_of::<RknpuMemCreate>(),
            )?;
        }
        RknpuCmd::MemMap => {
            info!("rknpu mem_map ioctl");
            let mut mem_map = RknpuMemMap::default();
            copy_from_user(
                &mut mem_map as *mut _ as *mut u8,
                arg as *const u8,
                mem::size_of::<RknpuMemMap>(),
            )?;

            with_npu(|rknpu_dev| {
                if rknpu_dev.get_phys_addr_and_size(mem_map.handle).is_some() {
                    mem_map.offset = (mem_map.handle as u64) << PAGE_SHIFT;

                    info!(
                        "mem_map: handle={} -> offset=0x{:x}",
                        mem_map.handle, mem_map.offset
                    );
                    Ok(())
                } else {
                    warn!("mem_map: invalid handle={}", mem_map.handle);
                    Err(VfsError::InvalidData)
                }
            })
            .inspect_err(|e| warn!("rknpu mem_map ioctl failed: {:?}", e))?;

            copy_to_user(
                arg as *mut u8,
                &mem_map as *const _ as *const u8,
                mem::size_of::<RknpuMemMap>(),
            )?;
        }
        RknpuCmd::MemDestroy => {
            let mut mem_destroy = RknpuMemDestroy::default();
            copy_from_user(
                &mut mem_destroy as *mut _ as *mut u8,
                arg as *const u8,
                mem::size_of::<RknpuMemDestroy>(),
            )?;

            warn!(
                "[rknpu] mem_destroy ioctl handle={} obj_addr={:#x}",
                mem_destroy.handle, mem_destroy.obj_addr
            );

            with_npu(|rknpu_dev| {
                if rknpu_dev
                    .get_phys_addr_and_size(mem_destroy.handle)
                    .is_none()
                {
                    warn!(
                        "[rknpu] mem_destroy ignored unknown handle={} obj_addr={:#x}",
                        mem_destroy.handle, mem_destroy.obj_addr
                    );
                    return Ok(());
                }

                rknpu_dev.destroy(mem_destroy.handle);
                Ok(())
            })
            .inspect_err(|e| warn!("rknpu mem_destroy ioctl failed: {:?}", e))?;
        }
        RknpuCmd::MemSync => {
            info!("rknpu mem_sync ioctl");
        }
        _ => {
            info!("rknpu action ioctl");
            let mut action = RknpuUserAction::default();
            copy_from_user(
                &mut action as *mut _ as *mut u8,
                arg as *const u8,
                mem::size_of::<RknpuUserAction>(),
            )?;

            info!(
                "rknpu action ioctl: flags = {:?}, value = {}",
                action.flags, action.value
            );

            with_npu(|rknpu_dev| {
                let val = rknpu_dev
                    .action(action.flags, action.value)
                    .map_err(|_| VfsError::InvalidData)?;
                action.value = val;
                Ok(())
            })
            .inspect_err(|e| warn!("rknpu action ioctl failed: {:?}", e))?;

            copy_to_user(
                arg as *mut u8,
                &action as *const _ as *const u8,
                mem::size_of::<RknpuUserAction>(),
            )?;
        }
    }
    Ok(0)
}

/// Handles RKNPU memory create ioctl command
pub fn rknpu_mem_create_ioctl(arg: usize) -> VfsResult<usize> {
    let mut mem_create_args = RknpuMemCreate::default();

    copy_from_user(
        &mut mem_create_args as *mut _ as *mut u8,
        arg as *const u8,
        mem::size_of::<RknpuMemCreate>(),
    )?;

    with_npu(|rknpu_dev| {
        rknpu_dev
            .create(&mut mem_create_args)
            .map_err(|_| VfsError::InvalidData)
    })
    .inspect_err(|e| warn!("rknpu mem_create ioctl failed: {:?}", e))?;

    copy_to_user(
        arg as *mut u8,
        &mem_create_args as *const _ as *const u8,
        mem::size_of::<RknpuMemCreate>(),
    )?;
    Ok(0)
}

/// DRM_IOCTL_GEM_FLINK ioctl argument type
#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
struct DrmGemFlink {
    /// GEM handle
    handle: u32,
    /// GEM name
    name: u32,
}

/// Handles DRM GEM flink ioctl command
fn drm_gem_flink_ioctl(data: &mut [u8]) -> VfsResult<usize> {
    let data = unsafe { &mut *(data.as_mut_ptr() as *mut DrmGemFlink) };
    info!("drm_gem_flink_ioctl called: {:#?}", data);
    Err(VfsError::NotFound)
}

/// DRM prime handle structure
#[repr(C)]
#[derive(Debug, Clone, Copy)]
struct DrmPrimeHande {
    /// Handle
    handle: u32,
    /// Flags
    flags: u32,
    /// File descriptor
    fd: i32,
}

/// Handles DRM prime handle to fd ioctl command
fn drm_prime_handle_to_fd_ioctl(data: &mut [u8]) -> VfsResult<usize> {
    let data = unsafe { &mut *(data.as_mut_ptr() as *mut DrmPrimeHande) };
    info!("drm_prime_handle_to_fd_ioctl {data:#x?}");
    data.fd = 1; // 返回一个假的 fd
    Ok(0)
}

/// Rust implementation of Linux kernel's drm_copy_field function
///
/// This function safely copies a string value to user space buffer,
/// similar to the Linux kernel implementation with proper error handling.
unsafe fn drm_copy_field(
    buf: *mut u8,
    buf_len: &mut c_ulong,
    value: *const u8,
) -> Result<(), axio::Error> {
    // Handle NULL value case - same as kernel's WARN_ONCE check
    if value.is_null() {
        warn!("[drm_copy_field] BUG: the value to copy was not set!");
        *buf_len = 0;
        return Ok(());
    }

    // Calculate actual string length using C string semantics
    let mut len = 0;
    unsafe {
        let mut ptr = value;
        while *ptr != 0 {
            len += 1;
            ptr = ptr.add(1);
        }
    }

    // Get the original buffer size
    let original_buf_len = *buf_len;

    // Update user's buffer length with actual string length (same as kernel)
    *buf_len = len;

    // Don't overflow user buffer - limit copy to available space
    let copy_len = if len > original_buf_len {
        original_buf_len
    } else {
        len
    };

    // Finally, try filling in the userbuf (same logic as kernel)
    if copy_len > 0 && !buf.is_null() {
        copy_to_user(buf as _, value, copy_len as _)?;
    }

    Ok(())
}

/// Sets the DRM version information for the device
pub fn drm_version(data: &mut [u8]) -> VfsResult<()> {
    let data = unsafe { &mut *(data.as_mut_ptr() as *mut DrmVersion) };
    info!("drm_version called: {:?}", data);

    // Set version information
    data.version_major = 0;
    data.version_minor = 9;
    data.version_patchlevel = 8;

    // Use drm_copy_field to handle string copying properly
    unsafe {
        // Copy driver name
        let ret = drm_copy_field(data.name, &mut data.name_len, DRM1_NAME.as_ptr());
        if let Err(e) = ret {
            warn!("[drm_version] Failed to copy driver name: {:?}", e);
            return Err(VfsError::InvalidData);
        }

        // Copy driver date
        let ret = drm_copy_field(
            data.date as *mut u8,
            &mut data.date_len,
            DRM1_DATE.as_ptr() as *const u8,
        );
        if let Err(e) = ret {
            warn!("[drm_version] Failed to copy driver date: {:?}", e);
            return Err(VfsError::InvalidData);
        }

        // Copy driver description
        let ret = drm_copy_field(data.desc, &mut data.desc_len, DRM1_DESC.as_ptr());
        if let Err(e) = ret {
            warn!("[drm_version] Failed to copy driver description: {:?}", e);
            return Err(VfsError::InvalidData);
        }
    }

    info!(
        "[drm_version] Set driver info: name_len={}, date_len={}, desc_len={}",
        data.name_len, data.date_len, data.desc_len
    );

    Ok(())
}

/// DRM_GET_UNIQUE ioctl handler
///
/// This function handles DRM_IOCTL_GET_UNIQUE requests, returning the unique
/// identifier for the DRM device (typically a bus ID or similar identifier).
pub fn drm_get_unique(data: &mut [u8]) -> VfsResult<()> {
    let unique_data = unsafe { &mut *(data.as_mut_ptr() as *mut DrmUnique) };
    info!("drm_get_unique called: {:?}", unique_data);

    unique_data.unique_len = 0;

    Ok(())
}

/// DRM_SET_UNIQUE ioctl handler (stub implementation)
///
/// This function handles DRM_IOCTL_SET_UNIQUE requests. For this
/// implementation, we return success but don't actually set the unique
/// identifier, as this is typically not used/needed in embedded systems.
pub fn drm_set_unique(data: &mut [u8]) -> VfsResult<()> {
    let unique_data = unsafe { &*(data.as_ptr() as *const DrmUnique) };
    info!("drm_set_unique called: {:?}", unique_data);

    // For this implementation, we just log the attempt and return success
    // In a real implementation, this would validate and store the unique ID
    warn!("[drm_set_unique] Setting unique identifier is not supported in this implementation");

    Ok(())
}
