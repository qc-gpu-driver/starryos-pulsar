use alloc::sync::Arc;
use core::mem::size_of;

use axerrno::{AxError, AxResult};
use axfs_ng_vfs::VfsError;
use axtask::current;
use rknpu::{NpuOwnerIds, ioctrl::RknpuSubmit};
use starry_core::task::AsThread;

use crate::vfs::dev::{card0::copy_from_user, card1::with_npu};

fn current_npu_owner_ids() -> NpuOwnerIds {
    let curr = current();
    let thr = curr.as_thread();

    NpuOwnerIds {
        task_id: curr.id().as_u64(),
        process_id: thr.proc_data.proc.pid() as u64,
        address_space_id: Arc::as_ptr(&thr.proc_data.aspace) as usize as u64,
    }
}

fn map_vfs_error(err: VfsError) -> AxError {
    match err {
        VfsError::NotFound => AxError::NoSuchDevice,
        VfsError::AddressInUse => AxError::ResourceBusy,
        VfsError::InvalidData => AxError::BadAddress,
        _ => AxError::InvalidInput,
    }
}

/// 打印当前任务视角下的完整 NPU 状态。
///
/// - `submit_user_ptr == 0` 时，使用默认空 submit
/// - 否则把用户态 `RknpuSubmit` 拷入内核，再结合当前线程 owner 信息构造状态
pub fn sys_dump_npu_status(submit_user_ptr: usize) -> AxResult<isize> {
    let owner = current_npu_owner_ids();
    let mut submit = RknpuSubmit::default();

    if submit_user_ptr != 0 {
        copy_from_user(
            &mut submit as *mut _ as *mut u8,
            submit_user_ptr as *const u8,
            size_of::<RknpuSubmit>(),
        )
        .map_err(|_| AxError::BadAddress)?;
    }

    trace!(
        "sys_dump_npu_status <= submit_ptr={:#x}, task_id={}, process_id={}, \
         address_space_id={:#x}",
        submit_user_ptr, owner.task_id, owner.process_id, owner.address_space_id
    );

    with_npu(|rknpu_dev| {
        let state = rknpu_dev.read_process_npu_state(&submit, owner);
        state.dump_pretty();
        Ok(())
    })
    .map_err(map_vfs_error)?;

    Ok(0)
}
