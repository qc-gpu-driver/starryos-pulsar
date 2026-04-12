//! Queue-side data structures for the RKNPU task scheduler.
//!
//! The queue tracks one whole blocking userspace submit per entry, while the
//! scheduler dispatches one task at a time to whichever hardware core is idle.
//! Queue state therefore only owns submit progress. The scheduler owns the
//! per-core in-flight bindings.

#![allow(dead_code)]

use crate::{
    RKNPU_MAX_SUBCORE_TASKS, RknpuError, RknpuTask, core_mask_from_index,
    ioctrl::{RknpuSubcoreTask, RknpuSubmit},
};
use alloc::{
    boxed::Box,
    collections::{BTreeMap, VecDeque},
    vec::Vec,
};

/// Unique identifier assigned to one queued NPU submission.
pub type RknpuQueueTaskId = u64;

/// Lifecycle state of one queued NPU submission.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum RknpuQueueTaskState {
    /// The task can be scheduled immediately.
    #[default]
    Ready,
    /// At least one hardware core is currently executing a task from this submit.
    Running,
    /// Every dispatchable task in this submit completed successfully.
    Completed,
    /// The submit encountered an error and should not be scheduled again.
    Faulted,
}

/// Minimal immutable scheduler view derived from one ioctl submit.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SubmitMeta {
    pub flags: u32,
    pub priority: i32,
    pub core_mask: u32,
    pub task_dma_base: u64,
    pub lane_ranges: [RknpuSubcoreTask; RKNPU_MAX_SUBCORE_TASKS],
    pub task_total: u32,
}

impl SubmitMeta {
    pub fn from_submit(submit: &RknpuSubmit, task_total: u32) -> Self {
        let mut lane_ranges = submit.subcore_task;
        let all_empty = lane_ranges.iter().all(|lane| lane.task_number == 0);
        if all_empty && task_total > 0 {
            lane_ranges[0] = RknpuSubcoreTask {
                task_start: 0,
                task_number: task_total,
            };
        }

        Self {
            flags: submit.flags,
            priority: submit.priority,
            core_mask: submit.core_mask,
            task_dma_base: submit.task_base_addr,
            lane_ranges,
            task_total,
        }
    }

    pub fn lane_range(&self, subcore_slot: usize) -> Option<RknpuSubcoreTask> {
        self.lane_ranges
            .get(subcore_slot)
            .copied()
            .filter(|lane| lane.task_number > 0)
    }
}

/// Immutable reply-only fields preserved from the original ioctl header.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
struct SubmitReplyState {
    timeout: u32,
    task_start: u32,
    iommu_domain_id: u32,
    reserved: u32,
    fence_fd: i32,
}

impl SubmitReplyState {
    fn from_submit(submit: &RknpuSubmit) -> Self {
        Self {
            timeout: submit.timeout,
            task_start: submit.task_start,
            iommu_domain_id: submit.iommu_domain_id,
            reserved: submit.reserved,
            fence_fd: submit.fence_fd,
        }
    }
}

/// Queue-owned input built from the submit ioctl boundary.
#[derive(Debug, Clone)]
pub struct RknpuQueuedSubmit {
    pub meta: SubmitMeta,
    reply: SubmitReplyState,
    pub tasks: Vec<RknpuTask>,
}

impl RknpuQueuedSubmit {
    pub fn new(submit: RknpuSubmit, tasks: Vec<RknpuTask>) -> Self {
        let task_total = tasks.len() as u32;
        Self {
            meta: SubmitMeta::from_submit(&submit, task_total),
            reply: SubmitReplyState::from_submit(&submit),
            tasks,
        }
    }
}

/// One queue entry managed by the NPU scheduler.
#[derive(Debug, Clone)]
pub struct RknpuQueueTask {
    pub id: RknpuQueueTaskId,
    pub state: RknpuQueueTaskState,
    pub meta: SubmitMeta,
    reply: SubmitReplyState,
    pub tasks: Vec<RknpuTask>,
    pub subcore_cursors: [u32; RKNPU_MAX_SUBCORE_TASKS],
    pub subcore_running_mask: u8,
    pub completed_task_count: u32,
    pub inflight_core_mask: u8,
    pub last_error: Option<RknpuError>,
    ready_queued: bool,
}

impl RknpuQueueTask {
    pub fn new(id: RknpuQueueTaskId, queued_submit: RknpuQueuedSubmit) -> Self {
        Self {
            id,
            state: RknpuQueueTaskState::Ready,
            meta: queued_submit.meta,
            reply: queued_submit.reply,
            tasks: queued_submit.tasks,
            subcore_cursors: [0; RKNPU_MAX_SUBCORE_TASKS],
            subcore_running_mask: 0,
            completed_task_count: 0,
            inflight_core_mask: 0,
            last_error: None,
            ready_queued: false,
        }
    }

    fn completion_target(&self) -> u32 {
        let total = self
            .meta
            .lane_ranges
            .iter()
            .map(|lane| lane.task_number)
            .sum::<u32>();
        if total == 0 {
            self.meta.task_total
        } else {
            total.min(self.meta.task_total)
        }
    }

    pub fn build_submit(&self) -> RknpuSubmit {
        let mut submit = RknpuSubmit::default();
        submit.flags = self.meta.flags;
        submit.timeout = self.reply.timeout;
        submit.task_start = self.reply.task_start;
        submit.task_number = self.meta.task_total;
        submit.task_counter = self.completed_task_count.min(self.completion_target());
        submit.priority = self.meta.priority;
        submit.task_obj_addr = 0;
        submit.iommu_domain_id = self.reply.iommu_domain_id;
        submit.reserved = self.reply.reserved;
        submit.task_base_addr = self.meta.task_dma_base;
        submit.hw_elapse_time = if self.last_error.is_some() {
            -1
        } else if matches!(self.state, RknpuQueueTaskState::Completed) {
            self.completed_task_count as i64
        } else {
            0
        };
        submit.core_mask = self.meta.core_mask;
        submit.fence_fd = self.reply.fence_fd;
        submit.subcore_task = self.meta.lane_ranges;
        submit
    }

    pub fn is_terminal(&self) -> bool {
        matches!(self.state, RknpuQueueTaskState::Completed)
            || matches!(self.state, RknpuQueueTaskState::Faulted) && self.inflight_core_mask == 0
    }

    pub fn has_dispatchable_work(&self) -> bool {
        self.next_dispatchable().is_some()
    }

    pub fn allows_core(&self, core_slot: usize) -> bool {
        let mask = self.meta.core_mask;
        mask == 0 || (mask & core_mask_from_index(core_slot)) != 0
    }

    pub fn next_dispatchable(&self) -> Option<(usize, usize)> {
        if matches!(
            self.state,
            RknpuQueueTaskState::Completed | RknpuQueueTaskState::Faulted
        ) {
            return None;
        }

        for subcore_slot in 0..RKNPU_MAX_SUBCORE_TASKS {
            let bit = 1u8 << subcore_slot;
            if self.subcore_running_mask & bit != 0 {
                continue;
            }

            let Some(range) = self.meta.lane_range(subcore_slot) else {
                continue;
            };
            let cursor = self.subcore_cursors[subcore_slot];
            if cursor >= range.task_number {
                continue;
            }

            let task_index = range.task_start as usize + cursor as usize;
            if task_index >= self.tasks.len() {
                continue;
            }

            return Some((subcore_slot, task_index));
        }

        None
    }

    pub fn reserve_dispatch(&mut self, core_slot: usize, subcore_slot: usize) {
        self.state = RknpuQueueTaskState::Running;
        self.subcore_running_mask |= 1u8 << subcore_slot;
        self.inflight_core_mask |= 1u8 << core_slot;
    }

    pub fn release_failed_dispatch(
        &mut self,
        core_slot: usize,
        subcore_slot: usize,
        err: RknpuError,
    ) {
        self.subcore_running_mask &= !(1u8 << subcore_slot);
        self.inflight_core_mask &= !(1u8 << core_slot);
        self.state = RknpuQueueTaskState::Faulted;
        self.last_error = Some(err);
    }

    pub fn complete_dispatch(
        &mut self,
        core_slot: usize,
        subcore_slot: usize,
        _task_index: u32,
        _last_task_int_status: u32,
        task_error: bool,
    ) {
        let was_faulted = matches!(self.state, RknpuQueueTaskState::Faulted);

        self.inflight_core_mask &= !(1u8 << core_slot);
        self.subcore_running_mask &= !(1u8 << subcore_slot);

        if let Some(cursor) = self.subcore_cursors.get_mut(subcore_slot) {
            *cursor = cursor.saturating_add(1);
        }
        self.completed_task_count = self.completed_task_count.saturating_add(1);

        if task_error {
            self.state = RknpuQueueTaskState::Faulted;
            self.last_error = Some(RknpuError::TaskError);
            return;
        }

        if was_faulted {
            self.state = RknpuQueueTaskState::Faulted;
            return;
        }

        if self.completed_task_count >= self.completion_target() {
            self.state = RknpuQueueTaskState::Completed;
        } else if self.inflight_core_mask == 0 {
            self.state = RknpuQueueTaskState::Ready;
        } else {
            self.state = RknpuQueueTaskState::Running;
        }
    }

    pub fn mark_faulted(&mut self, err: RknpuError) {
        self.state = RknpuQueueTaskState::Faulted;
        self.last_error = Some(err);
    }
}

/// One dispatch reservation selected for one queued submit.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RknpuDispatchReservation {
    pub queue_task_id: RknpuQueueTaskId,
    pub task_index: u32,
    pub subcore_slot: u8,
}

/// Ready queues for the RKNPU scheduler.
#[derive(Debug, Default)]
pub struct RknpuTaskQueue {
    next_task_id: RknpuQueueTaskId,
    tasks: BTreeMap<RknpuQueueTaskId, Box<RknpuQueueTask>>,
    ready: BTreeMap<i32, VecDeque<RknpuQueueTaskId>>,
}

impl RknpuTaskQueue {
    pub const fn new() -> Self {
        Self {
            next_task_id: 1,
            tasks: BTreeMap::new(),
            ready: BTreeMap::new(),
        }
    }

    pub fn is_idle(&self) -> bool {
        self.ready.values().all(VecDeque::is_empty) && !self.has_inflight()
    }

    pub fn has_inflight(&self) -> bool {
        self.tasks.values().any(|task| task.inflight_core_mask != 0)
    }

    pub fn enqueue(&mut self, queued_submit: RknpuQueuedSubmit) -> RknpuQueueTaskId {
        let task_id = self.next_task_id;
        self.next_task_id = self.next_task_id.saturating_add(1);

        let task = Box::new(RknpuQueueTask::new(task_id, queued_submit));
        self.tasks.insert(task_id, task);
        self.push_ready_task_by_id(task_id);
        task_id
    }

    pub fn task(&self, task_id: RknpuQueueTaskId) -> Option<&RknpuQueueTask> {
        self.tasks.get(&task_id).map(Box::as_ref)
    }

    pub fn task_mut(&mut self, task_id: RknpuQueueTaskId) -> Option<&mut RknpuQueueTask> {
        self.tasks.get_mut(&task_id).map(Box::as_mut)
    }

    pub fn is_terminal(&self, task_id: RknpuQueueTaskId) -> bool {
        self.tasks
            .get(&task_id)
            .is_some_and(|task| task.is_terminal())
    }

    pub fn take_terminal_task(&mut self, task_id: RknpuQueueTaskId) -> Option<RknpuQueueTask> {
        if !self.is_terminal(task_id) {
            return None;
        }
        self.tasks.remove(&task_id).map(|task| *task)
    }

    pub fn reserve_next_dispatch(&mut self, core_slot: usize) -> Option<RknpuDispatchReservation> {
        let task_id = self.pop_next_ready_task_id_for_core(core_slot)?;
        let (subcore_slot, task_index, requeue) = {
            let task = self.tasks.get_mut(&task_id)?;
            let (subcore_slot, task_index) = task.next_dispatchable()?;
            task.reserve_dispatch(core_slot, subcore_slot);
            (subcore_slot, task_index, task.has_dispatchable_work())
        };

        if requeue {
            self.push_ready_task_by_id(task_id);
        }

        let reservation = RknpuDispatchReservation {
            queue_task_id: task_id,
            task_index: task_index as u32,
            subcore_slot: subcore_slot as u8,
        };

        if let Some(task) = self.tasks.get(&task_id) {
            debug!(
                "[rknpu-queue] reserve_next_dispatch queue_task={} core={} subcore={} task_index={} completed={} inflight_core_mask={:#x} subcore_running_mask={:#x} requeue={}",
                reservation.queue_task_id,
                core_slot,
                reservation.subcore_slot,
                reservation.task_index,
                task.completed_task_count,
                task.inflight_core_mask,
                task.subcore_running_mask,
                requeue
            );
        }

        Some(reservation)
    }

    pub fn fail_dispatch(
        &mut self,
        reservation: RknpuDispatchReservation,
        core_slot: usize,
        err: RknpuError,
    ) -> Option<RknpuQueueTaskId> {
        let task = self.tasks.get_mut(&reservation.queue_task_id)?;
        task.release_failed_dispatch(core_slot, reservation.subcore_slot as usize, err);
        debug!(
            "[rknpu-queue] fail_dispatch queue_task={} core={} subcore={} task_index={} err={:?}",
            reservation.queue_task_id,
            core_slot,
            reservation.subcore_slot,
            reservation.task_index,
            err
        );
        task.is_terminal().then_some(reservation.queue_task_id)
    }

    pub fn complete_dispatch(
        &mut self,
        reservation: RknpuDispatchReservation,
        core_slot: usize,
        last_task_int_status: u32,
        task_error: bool,
    ) -> Option<RknpuQueueTaskId> {
        let task = self.tasks.get_mut(&reservation.queue_task_id)?;
        task.complete_dispatch(
            core_slot,
            reservation.subcore_slot as usize,
            reservation.task_index,
            last_task_int_status,
            task_error,
        );

        debug!(
            "[rknpu-queue] complete_dispatch queue_task={} core={} subcore={} task_index={} completed={} inflight_core_mask={:#x} subcore_running_mask={:#x} state={:?} task_error={} next_dispatchable={}",
            reservation.queue_task_id,
            core_slot,
            reservation.subcore_slot,
            reservation.task_index,
            task.completed_task_count,
            task.inflight_core_mask,
            task.subcore_running_mask,
            task.state,
            task_error,
            task.has_dispatchable_work()
        );

        if task.is_terminal() {
            debug!(
                "[rknpu-queue] queue_task={} reached terminal state={:?} last_error={:?} task_counter={}",
                reservation.queue_task_id, task.state, task.last_error, task.completed_task_count
            );
            return Some(reservation.queue_task_id);
        }

        let should_requeue = !task.ready_queued && task.has_dispatchable_work();
        if should_requeue {
            self.push_ready_task_by_id(reservation.queue_task_id);
        }

        None
    }

    pub fn mark_task_faulted(
        &mut self,
        task_id: RknpuQueueTaskId,
        err: RknpuError,
    ) -> Option<RknpuQueueTaskId> {
        let task = self.tasks.get_mut(&task_id)?;
        task.mark_faulted(err);
        debug!(
            "[rknpu-queue] mark_task_faulted queue_task={} state={:?} inflight_core_mask={:#x} err={:?}",
            task_id, task.state, task.inflight_core_mask, err
        );
        task.is_terminal().then_some(task_id)
    }

    fn push_ready_task_by_id(&mut self, task_id: RknpuQueueTaskId) {
        let Some((priority, should_queue)) = self.tasks.get(&task_id).map(|task| {
            (
                task.meta.priority,
                !task.ready_queued && !task.is_terminal() && task.has_dispatchable_work(),
            )
        }) else {
            return;
        };
        if !should_queue {
            return;
        }
        if let Some(task) = self.tasks.get_mut(&task_id) {
            task.ready_queued = true;
        }
        self.ready.entry(priority).or_default().push_back(task_id);
    }

    fn pop_next_ready_task_id_for_core(&mut self, core_slot: usize) -> Option<RknpuQueueTaskId> {
        let priorities = self.ready.keys().copied().collect::<Vec<_>>();
        for priority in priorities {
            let len = self.ready.get(&priority).map_or(0, VecDeque::len);
            if len == 0 {
                continue;
            }

            let mut deferred = Vec::new();
            let mut selected = None;

            for _ in 0..len {
                let task_id = match self
                    .ready
                    .get_mut(&priority)
                    .and_then(|queue| queue.pop_front())
                {
                    Some(task_id) => task_id,
                    None => break,
                };
                let decision = match self.tasks.get_mut(&task_id) {
                    Some(task) => {
                        task.ready_queued = false;
                        if task.is_terminal() || !task.has_dispatchable_work() {
                            0_u8
                        } else if !task.allows_core(core_slot) {
                            task.ready_queued = true;
                            1_u8
                        } else {
                            2_u8
                        }
                    }
                    None => 0_u8,
                };

                match decision {
                    0 => {}
                    1 => deferred.push(task_id),
                    2 => {
                        selected = Some(task_id);
                        break;
                    }
                    _ => unreachable!(),
                }
            }

            if let Some(queue) = self.ready.get_mut(&priority) {
                for task_id in deferred.into_iter().rev() {
                    queue.push_front(task_id);
                }
            }

            if let Some(task_id) = selected {
                return Some(task_id);
            }
        }
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fake_submit(
        task_number: u32,
        priority: i32,
        subcore_tasks: &[(usize, u32, u32)],
    ) -> RknpuSubmit {
        let mut submit = RknpuSubmit::default();
        submit.timeout = 1000;
        submit.priority = priority;
        submit.task_number = task_number;
        for (slot, start, number) in subcore_tasks {
            submit.subcore_task[*slot].task_start = *start;
            submit.subcore_task[*slot].task_number = *number;
        }
        submit
    }

    fn fake_submit_with_core_mask(
        task_number: u32,
        priority: i32,
        core_mask: u32,
        subcore_tasks: &[(usize, u32, u32)],
    ) -> RknpuSubmit {
        let mut submit = fake_submit(task_number, priority, subcore_tasks);
        submit.core_mask = core_mask;
        submit
    }

    fn fake_task(int_mask: u32) -> RknpuTask {
        RknpuTask {
            int_mask,
            ..RknpuTask::default()
        }
    }

    #[test]
    fn queued_submit_builds_submit_meta() {
        let tasks = vec![RknpuTask::default(), RknpuTask::default()];
        let queued = RknpuQueuedSubmit::new(fake_submit(2, 0, &[(0, 0, 2)]), tasks);

        assert_eq!(queued.meta.task_total, 2);
        assert_eq!(queued.meta.task_dma_base, 0);
        assert_eq!(queued.meta.lane_range(0).unwrap().task_number, 2);
    }

    #[test]
    fn empty_subcore_layout_defaults_to_slot_zero_range() {
        let mut queue = RknpuTaskQueue::new();
        let submit = RknpuQueuedSubmit::new(
            fake_submit(2, 0, &[]),
            vec![fake_task(0x100), fake_task(0x200)],
        );

        queue.enqueue(submit);
        let first = queue.reserve_next_dispatch(0).unwrap();

        assert_eq!(first.subcore_slot, 0);
        assert_eq!(first.task_index, 0);
    }

    #[test]
    fn lower_priority_value_runs_first() {
        let mut queue = RknpuTaskQueue::new();

        let low = RknpuQueuedSubmit::new(fake_submit(1, 10, &[(0, 0, 1)]), vec![fake_task(0x100)]);
        let high = RknpuQueuedSubmit::new(fake_submit(1, -5, &[(0, 0, 1)]), vec![fake_task(0x200)]);

        let low_id = queue.enqueue(low);
        let high_id = queue.enqueue(high);

        assert_eq!(
            queue.reserve_next_dispatch(0).unwrap().queue_task_id,
            high_id
        );
        assert_eq!(
            queue.reserve_next_dispatch(1).unwrap().queue_task_id,
            low_id
        );
    }

    #[test]
    fn same_submit_can_fill_multiple_idle_cores() {
        let mut queue = RknpuTaskQueue::new();
        let submit = RknpuQueuedSubmit::new(
            fake_submit(2, 0, &[(0, 0, 1), (1, 1, 1)]),
            vec![fake_task(0x100), fake_task(0x200)],
        );

        let task_id = queue.enqueue(submit);
        let first = queue.reserve_next_dispatch(0).unwrap();
        let second = queue.reserve_next_dispatch(1).unwrap();

        assert_eq!(first.queue_task_id, task_id);
        assert_eq!(second.queue_task_id, task_id);
        assert_ne!(first.subcore_slot, second.subcore_slot);
    }

    #[test]
    fn core_mask_binds_submit_to_matching_physical_core() {
        let mut queue = RknpuTaskQueue::new();
        let submit = RknpuQueuedSubmit::new(
            fake_submit_with_core_mask(1, 0, 0x2, &[(0, 0, 1)]),
            vec![fake_task(0x100)],
        );

        queue.enqueue(submit);

        assert!(queue.reserve_next_dispatch(0).is_none());
        let dispatch = queue.reserve_next_dispatch(1).unwrap();
        assert_eq!(dispatch.task_index, 0);
    }

    #[test]
    fn bound_high_priority_submit_is_not_stolen_by_other_cores() {
        let mut queue = RknpuTaskQueue::new();
        let high = RknpuQueuedSubmit::new(
            fake_submit_with_core_mask(1, -10, 0x2, &[(0, 0, 1)]),
            vec![fake_task(0x100)],
        );
        let low = RknpuQueuedSubmit::new(
            fake_submit_with_core_mask(1, 0, 0x1, &[(0, 0, 1)]),
            vec![fake_task(0x200)],
        );

        let high_id = queue.enqueue(high);
        let low_id = queue.enqueue(low);

        assert_eq!(
            queue.reserve_next_dispatch(0).unwrap().queue_task_id,
            low_id
        );
        assert_eq!(
            queue.reserve_next_dispatch(1).unwrap().queue_task_id,
            high_id
        );
    }

    #[test]
    fn same_subcore_slot_is_not_dispatched_twice_concurrently() {
        let mut queue = RknpuTaskQueue::new();
        let submit = RknpuQueuedSubmit::new(
            fake_submit(2, 0, &[(0, 0, 2)]),
            vec![fake_task(0x100), fake_task(0x100)],
        );

        queue.enqueue(submit);
        let first = queue.reserve_next_dispatch(0).unwrap();

        assert_eq!(first.subcore_slot, 0);
        assert!(queue.reserve_next_dispatch(1).is_none());
    }

    #[test]
    fn completion_requeues_followup_work_and_terminal_task_can_be_taken() {
        let mut queue = RknpuTaskQueue::new();
        let submit = RknpuQueuedSubmit::new(
            fake_submit(2, 0, &[(0, 0, 2)]),
            vec![fake_task(0x100), fake_task(0x100)],
        );

        let task_id = queue.enqueue(submit);
        let first = queue.reserve_next_dispatch(0).unwrap();

        assert!(queue.complete_dispatch(first, 0, 0x100, false).is_none());

        let second = queue.reserve_next_dispatch(0).unwrap();
        let terminal = queue.complete_dispatch(second, 0, 0x100, false);

        assert_eq!(terminal, Some(task_id));
        let finished = queue.take_terminal_task(task_id).unwrap();
        assert_eq!(finished.state, RknpuQueueTaskState::Completed);
        assert_eq!(finished.build_submit().task_counter, 2);
    }

    #[test]
    fn faulted_submit_waits_for_other_inflight_cores_before_becoming_terminal() {
        let mut queue = RknpuTaskQueue::new();
        let submit = RknpuQueuedSubmit::new(
            fake_submit(2, 0, &[(0, 0, 1), (1, 1, 1)]),
            vec![fake_task(0x100), fake_task(0x200)],
        );

        let task_id = queue.enqueue(submit);
        let first = queue.reserve_next_dispatch(0).unwrap();
        let second = queue.reserve_next_dispatch(1).unwrap();

        assert!(
            queue
                .fail_dispatch(first, 0, RknpuError::InternalError)
                .is_none()
        );
        assert!(!queue.is_terminal(task_id));

        let terminal = queue.complete_dispatch(second, 1, 0x200, false);

        assert_eq!(terminal, Some(task_id));
        let finished = queue.take_terminal_task(task_id).unwrap();
        assert_eq!(finished.state, RknpuQueueTaskState::Faulted);
        assert_eq!(finished.last_error, Some(RknpuError::InternalError));
    }

    #[test]
    fn completed_submit_preserves_reply_fields() {
        let mut submit = fake_submit(2, -3, &[(0, 0, 2)]);
        submit.flags = 0x55aa;
        submit.timeout = 77;
        submit.task_start = 9;
        submit.iommu_domain_id = 11;
        submit.reserved = 12;
        submit.task_base_addr = 0x8800;
        submit.core_mask = 0x7;
        submit.fence_fd = 13;

        let queued = RknpuQueuedSubmit::new(submit, vec![fake_task(0x100), fake_task(0x200)]);
        let task = RknpuQueueTask::new(1, queued);
        let rebuilt = task.build_submit();

        assert_eq!(rebuilt.flags, 0x55aa);
        assert_eq!(rebuilt.timeout, 77);
        assert_eq!(rebuilt.task_start, 9);
        assert_eq!(rebuilt.priority, -3);
        assert_eq!(rebuilt.iommu_domain_id, 11);
        assert_eq!(rebuilt.reserved, 12);
        assert_eq!(rebuilt.task_base_addr, 0x8800);
        assert_eq!(rebuilt.core_mask, 0x7);
        assert_eq!(rebuilt.fence_fd, 13);
        assert_eq!(rebuilt.task_obj_addr, 0);
    }

    #[test]
    fn partial_lane_layout_keeps_original_completion_target() {
        let queued = RknpuQueuedSubmit::new(
            fake_submit(4, 0, &[(0, 1, 2)]),
            vec![
                fake_task(0x100),
                fake_task(0x100),
                fake_task(0x100),
                fake_task(0x100),
            ],
        );
        let task = RknpuQueueTask::new(1, queued);

        assert_eq!(task.completion_target(), 2);
    }
}
