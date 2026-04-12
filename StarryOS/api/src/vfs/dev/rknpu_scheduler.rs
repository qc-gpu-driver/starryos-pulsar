//! Blocking-submit / asynchronous-dispatch scheduler for the StarryOS NPU path.
//!
//! The external contract remains a blocking `Submit` ioctl: the ioctl thread
//! enqueues one whole userspace submit, waits, and receives the completed
//! `RknpuSubmit` header only after the submit reaches a terminal state.
//!
//! Internally the scheduler is intentionally split into three ownership layers:
//!
//! - the queue owns submit-level progress and fairness state
//! - the scheduler owns the per-core in-flight dispatch bindings
//! - the driver owns only raw hardware state and raw per-core IRQ publication
//!
//! This separation is the key invariant of the current design. Queue entries do
//! not carry reverse pointers to hardware cores, and the driver does not carry
//! queue-specific metadata such as `queue_task_id` or `subcore_slot`. The
//! scheduler is the only layer that can join "core X just completed" back to
//! "this queue task and this exact `RknpuTask` completed".
//!
//! The steady-state flow is:
//!
//! 1. `card1` validates the ioctl payload, materializes `RknpuTask[]`, and
//!    builds one immutable `RknpuQueuedSubmit`.
//! 2. `enqueue()` inserts that submit into the queue, installs a per-submit
//!    waiter, and kicks the background worker.
//! 3. The worker reserves work for idle cores, records an `InflightDispatch`
//!    entry, flushes DMA visibility once per submit dispatch wave, and calls
//!    into the driver for one core / one task dispatch.
//! 4. IRQ handlers publish only raw per-core completion status. The worker
//!    harvests those raw completions, looks up the matching `InflightDispatch`,
//!    updates the queue, and writes back the task's interrupt shadow.
//! 5. Once a submit becomes terminal, the scheduler prepares DMA buffers for
//!    CPU reads, wakes the waiter, and hands a `CompletedSubmit` back to the
//!    ioctl boundary.
//!
//! Two synchronization primitives are used on purpose:
//!
//! - `NpuSubmitWaiter` is a per-submit blocking primitive for the ioctl thread
//! - `kick` is a global event that only wakes the single background worker
//!
//! The worker never waits on individual submits, and submitters never drive
//! scheduling work themselves.

use alloc::{
    collections::{BTreeMap, BTreeSet},
    string::ToString,
    sync::Arc,
    vec::Vec,
};
use core::sync::atomic::{AtomicBool, Ordering};

use axfs_ng_vfs::{VfsError, VfsResult};
use axsync::Mutex;
use axtask::{future::block_on, yield_now};
use event_listener::Event;
use lazy_static::lazy_static;
use rknpu::{
    NPU_MAX_CORES, RknpuDispatchReservation, RknpuError, RknpuQueueTaskId, RknpuQueuedSubmit,
    RknpuTask, RknpuTaskQueue, ioctrl::RknpuSubmit,
};
use starry_core::futex::WaitQueue;

use super::card1::npu;

/// Terminal scheduler result returned to the ioctl path.
///
/// The scheduler returns a compact boundary object instead of exposing
/// `RknpuQueueTask` directly. That keeps queue internals private while still
/// allowing `card1` to:
///
/// - copy the rebuilt `RknpuSubmit` header back to userspace
/// - copy the final `RknpuTask[]` shadow, including updated interrupt status
/// - surface the last scheduler/driver error if the submit faulted
pub struct CompletedSubmit {
    /// Rebuilt ABI-facing submit header returned to the ioctl boundary.
    pub submit: RknpuSubmit,
    /// Final task shadow copied back to userspace, including `int_status`.
    pub tasks: Vec<RknpuTask>,
    /// Last scheduler/driver error recorded for this submit, if any.
    pub last_error: Option<RknpuError>,
}

/// Scheduler-owned binding between one hardware core and one queued task.
///
/// This record exists only while the hardware may still complete that dispatch.
/// It is the scheduler's private join key from a raw `CoreCompletion` back to
/// queue state.
///
/// `task_ptr` points into the queue-owned boxed task storage. That address
/// stays stable while the queue entry remains alive, so the scheduler can
/// temporarily drop the mutex while submitting to hardware and later write back
/// the task's interrupt shadow during harvest.
#[derive(Clone, Copy)]
struct InflightDispatch {
    /// Queue entry that owns this hardware dispatch.
    queue_task_id: RknpuQueueTaskId,
    /// Physical NPU core slot currently running this dispatch.
    core_slot: usize,

    /// Logical lane inside the submit that this task belongs to.
    subcore_slot: u8,
    /// Absolute task index inside the submit task array.
    task_index: u32,
    /// Stable pointer to the queue-owned task shadow written back on harvest.
    task_ptr: usize,
    /// IRQ bits expected for this task on completion.
    expected_irq_mask: u32,
}

impl InflightDispatch {
    /// Rebuild the queue-side reservation token associated with this dispatch.
    fn reservation(self) -> RknpuDispatchReservation {
        RknpuDispatchReservation {
            queue_task_id: self.queue_task_id,
            task_index: self.task_index,
            subcore_slot: self.subcore_slot,
        }
    }
}

/// Per-submit blocking primitive used by the ioctl thread.
///
/// Each blocking submit gets a dedicated waiter so wake-ups remain strictly
/// targeted to that submit. Timeouts are still carried in the ioctl header for
/// ABI compatibility, but timeout policy is not implemented here; this waiter
/// only models "terminal" vs "not terminal yet".
struct NpuSubmitWaiter {
    /// Set to true exactly once when the submit reaches terminal state.
    done: AtomicBool,
    /// Futex-backed wait queue used by the blocking ioctl thread.
    wait_queue: WaitQueue,
}

impl NpuSubmitWaiter {
    /// Wait mask used for all submit wait/wake operations on this waiter.
    const WAIT_MASK: u32 = u32::MAX;

    /// Create a fresh waiter for one blocking submit.
    fn new() -> Self {
        Self {
            done: AtomicBool::new(false),
            wait_queue: WaitQueue::new(),
        }
    }

    /// Block the caller until the submit becomes terminal or the wait errors.
    fn wait(&self) -> VfsResult<()> {
        // `WaitQueue` may wake spuriously, so the predicate is rechecked with an
        // acquire load before and after each sleep.
        while !self.done.load(Ordering::Acquire) {
            self.wait_queue
                .wait_if(Self::WAIT_MASK, None, || !self.done.load(Ordering::Acquire))?;
        }
        Ok(())
    }

    /// Mark the submit as terminal and wake every thread blocked on it.
    fn complete(&self) {
        // Publish terminal state before waking the blocked submitter.
        self.done.store(true, Ordering::Release);
        self.wait_queue.wake(usize::MAX, Self::WAIT_MASK);
    }
}

/// Mutable scheduler state protected by one mutex.
///
/// This lock deliberately guards both queue progress and the scheduler-owned
/// in-flight table. The worker publishes an `InflightDispatch` entry under the
/// mutex before programming hardware, and harvest removes that entry under the
/// same mutex before the queue is allowed to become terminal and get removed.
struct NpuSchedulerState {
    /// Queue-owned submit/task state and fairness bookkeeping.
    queue: RknpuTaskQueue,
    /// Per-submit waiters keyed by queue task id.
    waiters: BTreeMap<RknpuQueueTaskId, Arc<NpuSubmitWaiter>>,
    /// Scheduler-owned in-flight dispatch table, indexed by physical core.
    inflight: [Option<InflightDispatch>; NPU_MAX_CORES],
    /// Whether the singleton worker thread has already been spawned.
    worker_started: bool,
}

impl NpuSchedulerState {
    /// Construct empty queue, waiter map, and in-flight table state.
    fn new() -> Self {
        Self {
            queue: RknpuTaskQueue::new(),
            waiters: BTreeMap::new(),
            inflight: [None; NPU_MAX_CORES],
            worker_started: false,
        }
    }

    /// Return true if any hardware core is still owned by the scheduler.
    fn has_inflight(&self) -> bool {
        self.inflight.iter().any(Option::is_some)
    }
}

/// Global NPU scheduler shared by submit threads and the worker thread.
///
/// The scheduler intentionally has only one worker. `kick` is therefore a
/// coarse "there may be work" event, not a work-counting semaphore.
pub struct NpuScheduler {
    /// Shared mutable scheduler state used by submit threads and the worker.
    state: Mutex<NpuSchedulerState>,
    /// Global "there may be work" event for the single worker thread.
    kick: Event,
}

impl NpuScheduler {
    /// Build the singleton scheduler instance used by the device layer.
    fn new() -> Self {
        Self {
            state: Mutex::new(NpuSchedulerState::new()),
            kick: Event::new(),
        }
    }

    /// Insert one queued submit, install its waiter, and wake the worker.
    ///
    /// The ioctl thread remains responsible only for enqueueing and later
    /// blocking on the returned queue id. All actual scheduling work is driven
    /// by the background worker.
    pub fn enqueue(&self, queued_submit: RknpuQueuedSubmit) -> VfsResult<RknpuQueueTaskId> {
        let waiter = Arc::new(NpuSubmitWaiter::new());
        let submit_snapshot = queued_submit.meta;
        let (task_id, spawn_worker) = {
            let mut state = self.state.lock();
            let spawn_worker = !state.worker_started;
            if spawn_worker {
                state.worker_started = true;
            }

            let task_id = state.queue.enqueue(queued_submit);
            // Install the waiter before dropping the lock so a very fast
            // terminal transition cannot race with `wait()`.
            state.waiters.insert(task_id, waiter);
            (task_id, spawn_worker)
        };

        warn!(
            "[rknpu-scheduler] enqueue queue_task={} priority={} task_number={} core_mask={:#x} \
             task_base_addr={:#x} spawn_worker={} subcore0=({}, {}) subcore1=({}, {}) \
             subcore2=({}, {})",
            task_id,
            submit_snapshot.priority,
            submit_snapshot.task_total,
            submit_snapshot.core_mask,
            submit_snapshot.task_dma_base,
            spawn_worker,
            submit_snapshot.lane_ranges[0].task_start,
            submit_snapshot.lane_ranges[0].task_number,
            submit_snapshot.lane_ranges[1].task_start,
            submit_snapshot.lane_ranges[1].task_number,
            submit_snapshot.lane_ranges[2].task_start,
            submit_snapshot.lane_ranges[2].task_number
        );

        // Spawn the singleton worker once, then always kick after dropping the
        // lock so the worker never races with partially-published state.
        if spawn_worker {
            warn!("[rknpu-scheduler] spawning worker thread");
            axtask::spawn(worker_main, "rknpu-scheduler".to_string());
        }

        // One relaxed notify is enough because there is exactly one worker.
        self.kick.notify_relaxed(1);
        Ok(task_id)
    }

    /// Block the caller until the specified queued submit becomes terminal.
    pub fn wait(&self, task_id: RknpuQueueTaskId) -> VfsResult<()> {
        let waiter = {
            let state = self.state.lock();
            state
                .waiters
                .get(&task_id)
                .cloned()
                .ok_or(VfsError::NotFound)?
        };

        // From this point on the submit thread is passive. All progress is made
        // by the worker, which will complete this waiter when the submit turns
        // terminal.
        warn!("[rknpu-scheduler] wait start queue_task={}", task_id);
        if let Err(err) = waiter.wait() {
            warn!(
                "[rknpu-scheduler] wait interrupted queue_task={} err={:?}",
                task_id, err
            );
            self.abort_wait(task_id);
            return Err(err);
        }
        warn!("[rknpu-scheduler] wait done queue_task={}", task_id);
        Ok(())
    }

    /// Consume one terminal queue entry and rebuild the ioctl-facing result.
    ///
    /// This is the only place where queue-internal state is converted back into
    /// a public `CompletedSubmit`.
    pub fn take_terminal_task(&self, task_id: RknpuQueueTaskId) -> VfsResult<CompletedSubmit> {
        let mut state = self.state.lock();
        let task = state
            .queue
            .take_terminal_task(task_id)
            .ok_or(VfsError::InvalidData)?;
        state.waiters.remove(&task_id);

        // Rebuild the ABI-facing header only at the terminal boundary. The
        // scheduler does not mutate an ioctl-owned `RknpuSubmit` in place while
        // the submit is in flight.
        let submit = task.build_submit();
        let tasks = task.tasks;
        let last_error = task.last_error;
        Ok(CompletedSubmit {
            submit,
            tasks,
            last_error,
        })
    }

    /// Check whether the queue still contains any live or terminal-pending work.
    fn has_work(&self) -> bool {
        let state = self.state.lock();
        !state.queue.is_idle()
    }

    /// Sleep until at least one queued submit is available for the worker.
    ///
    /// The double-check around `listen()` avoids losing a kick that races with
    /// the transition from "observed idle" to "going to sleep".
    fn wait_for_work(&self) {
        loop {
            if self.has_work() {
                return;
            }

            let listener = self.kick.listen();

            // Double-check after installing the listener so we do not lose a
            // wake-up that raced with the sleep transition.
            if self.has_work() {
                return;
            }

            block_on(listener);
        }
    }

    /// Wake submit waiters for terminal queue entries and optionally prepare DMA
    /// buffers for CPU reads first.
    ///
    /// `prepare_read` is used only on the normal completion path. Failure to
    /// make the buffers CPU-visible is converted into a queue fault before the
    /// waiter is released.
    fn wake_terminal_tasks(&self, task_ids: Vec<RknpuQueueTaskId>, prepare_read: bool) {
        if task_ids.is_empty() {
            return;
        }

        warn!(
            "[rknpu-scheduler] wake_terminal_tasks ids={:?} prepare_read={} count={}",
            task_ids,
            prepare_read,
            task_ids.len()
        );

        let prepare_error = if prepare_read {
            // Terminal submits must not wake their waiter until CPU-side reads
            // are safe. This stays a scheduler concern; the queue does not
            // persist a separate "read prepared" state bit.
            with_npu_driver(|rknpu_dev| rknpu_dev.prepare_read_all()).err()
        } else {
            None
        };

        let waiters = {
            let mut state = self.state.lock();
            let mut waiters = Vec::with_capacity(task_ids.len());

            for task_id in task_ids {
                // Read-prepare failure is terminal for the submit because the
                // userspace-visible results can no longer be trusted.
                if prepare_error.is_some() {
                    state
                        .queue
                        .mark_task_faulted(task_id, RknpuError::MemoryError);
                }

                // The waiter may already be gone if the submit thread aborted
                // its wait. In that case the queue entry is cleaned up locally
                // once it is known to be terminal.
                if let Some(waiter) = state.waiters.get(&task_id).cloned() {
                    waiters.push(waiter);
                } else if state.queue.is_terminal(task_id) {
                    let _ = state.queue.take_terminal_task(task_id);
                }
            }

            waiters
        };

        for waiter in waiters {
            waiter.complete();
        }
    }

    /// Roll back one failed dispatch and propagate any resulting terminal wake.
    fn fail_dispatch(&self, inflight: InflightDispatch, err: RknpuError) {
        // Roll back scheduler-owned state first, then let the queue advance its
        // own state machine from the same reservation token.
        let terminal_ids = {
            let mut state = self.state.lock();
            state.inflight[inflight.core_slot] = None;
            state
                .queue
                .fail_dispatch(inflight.reservation(), inflight.core_slot, err)
                .into_iter()
                .collect::<Vec<_>>()
        };

        self.wake_terminal_tasks(terminal_ids, false);
    }

    /// Dump a verbose in-flight snapshot when the worker appears to be stalled.
    fn log_inflight_snapshot(
        &self,
        inflight_yields: u32,
        harvest_rounds: u32,
        dispatch_rounds: u32,
    ) {
        // Diagnostic helper used when the worker spins waiting for hardware
        // completions longer than expected.
        let state = self.state.lock();
        let mut printed = false;

        for core_slot in 0..NPU_MAX_CORES {
            let Some(inflight) = state.inflight[core_slot] else {
                continue;
            };
            printed = true;

            if let Some(queue_task) = state.queue.task(inflight.queue_task_id) {
                warn!(
                    "[rknpu-scheduler] inflight snapshot yields={} harvest_rounds={} \
                     dispatch_rounds={} core={} queue_task={} subcore={} task_index={} \
                     expected_irq_mask={:#x} state={:?} completed={} / {} \
                     inflight_core_mask={:#x} subcore_running_mask={:#x} cursors={:?} \
                     last_error={:?}",
                    inflight_yields,
                    harvest_rounds,
                    dispatch_rounds,
                    core_slot,
                    inflight.queue_task_id,
                    inflight.subcore_slot,
                    inflight.task_index,
                    inflight.expected_irq_mask,
                    queue_task.state,
                    queue_task.completed_task_count,
                    queue_task.meta.task_total,
                    queue_task.inflight_core_mask,
                    queue_task.subcore_running_mask,
                    queue_task.subcore_cursors,
                    queue_task.last_error
                );
            } else {
                warn!(
                    "[rknpu-scheduler] inflight snapshot yields={} harvest_rounds={} \
                     dispatch_rounds={} core={} queue_task={} subcore={} task_index={} \
                     expected_irq_mask={:#x} queue_task_missing=true",
                    inflight_yields,
                    harvest_rounds,
                    dispatch_rounds,
                    core_slot,
                    inflight.queue_task_id,
                    inflight.subcore_slot,
                    inflight.task_index,
                    inflight.expected_irq_mask
                );
            }
        }

        if !printed {
            warn!(
                "[rknpu-scheduler] inflight snapshot yields={} harvest_rounds={} \
                 dispatch_rounds={} no_inflight_entries=true",
                inflight_yields, harvest_rounds, dispatch_rounds
            );
        }
    }

    /// Harvest raw per-core completions from the driver and advance queue state.
    ///
    /// Returns true if at least one completion was consumed. The driver reports
    /// only raw core completion information; this function is where raw IRQ
    /// state is joined back to queue semantics through `InflightDispatch`.
    fn harvest_completed_cores(&self) -> bool {
        let completions =
            match with_npu_driver(|rknpu_dev| Ok(rknpu_dev.harvest_completed_dispatches())) {
                Ok(completions) => completions,
                Err(err) => {
                    warn!(
                        "[rknpu-scheduler] failed to harvest completed cores: {:?}",
                        err
                    );
                    return false;
                }
            };

        if completions.is_empty() {
            return false;
        }

        let terminal_ids = {
            let mut state = self.state.lock();
            let mut terminal_ids = Vec::new();

            for completion in completions {
                let core_slot = completion.core_slot as usize;
                // The driver publishes only raw per-core completion state. The
                // scheduler recovers queue semantics by looking up the matching
                // in-flight entry for that core.
                let Some(inflight) = state.inflight.get_mut(core_slot).and_then(Option::take)
                else {
                    debug!(
                        "[rknpu-scheduler] harvested completion on core={} without inflight \
                         dispatch observed={:#x}",
                        core_slot, completion.observed_irq_status
                    );
                    continue;
                };

                // Reapply the per-task expected mask here rather than in the
                // driver so the driver stays free of queue/task ownership.
                let last_task_int_status =
                    completion.observed_irq_status & inflight.expected_irq_mask;
                let task_error = completion.observed_irq_status != 0 && last_task_int_status == 0;
                let task_ptr = inflight.task_ptr as *mut RknpuTask;
                unsafe {
                    // The userspace-visible task shadow is updated here rather
                    // than in the driver so the driver stays free of queue
                    // ownership and task pointer bookkeeping.
                    core::ptr::addr_of_mut!((*task_ptr).int_status)
                        .write_unaligned(last_task_int_status);
                }

                debug!(
                    "[rknpu-scheduler] harvested completion core={} queue_task={} task_index={} \
                     subcore={} observed={:#x} last_task_int_status={:#x} task_error={}",
                    core_slot,
                    inflight.queue_task_id,
                    inflight.task_index,
                    inflight.subcore_slot,
                    completion.observed_irq_status,
                    last_task_int_status,
                    task_error
                );

                // Completing the queue reservation may also make the whole
                // submit terminal, in which case the queue returns its id.
                if let Some(task_id) = state.queue.complete_dispatch(
                    inflight.reservation(),
                    core_slot,
                    last_task_int_status,
                    task_error,
                ) {
                    terminal_ids.push(task_id);
                }
            }

            terminal_ids
        };

        self.wake_terminal_tasks(terminal_ids, true);
        true
    }

    /// Try to dispatch one task onto every currently idle core.
    ///
    /// Returns true if at least one dispatch was issued. Reservation, in-flight
    /// publication, DMA visibility confirmation, driver submission, and failure
    /// rollback all happen here because this is the only place that owns the
    /// transition from "queued" to "running on hardware".
    fn dispatch_idle_cores(&self) -> bool {
        // Deduplicate cache-flush visibility work within one worker sweep. A
        // submit may fan out to multiple idle cores, but its buffers only need
        // to be confirmed once before that dispatch wave.
        let mut dispatched = false;
        let mut confirmed_submit_ids = BTreeSet::new();
        let mut terminal_ids = Vec::new();

        for core_slot in 0..NPU_MAX_CORES {
            let setup = {
                let mut state = self.state.lock();
                if state.inflight[core_slot].is_some() {
                    None
                } else {
                    if let Some(reservation) = state.queue.reserve_next_dispatch(core_slot) {
                        let mut rollback_err = None;
                        // Materialize the queue reservation into the exact
                        // scheduler-owned dispatch record needed by the driver
                        // and by later harvest.
                        let prepared = if let Some(queue_task) =
                            state.queue.task_mut(reservation.queue_task_id)
                        {
                            let meta = queue_task.meta;
                            if let Some(task) =
                                queue_task.tasks.get_mut(reservation.task_index as usize)
                            {
                                let expected_irq_mask = task.int_mask;
                                let inflight = InflightDispatch {
                                    queue_task_id: reservation.queue_task_id,
                                    core_slot,
                                    subcore_slot: reservation.subcore_slot,
                                    task_index: reservation.task_index,
                                    task_ptr: task as *mut RknpuTask as usize,
                                    expected_irq_mask,
                                };
                                Some((inflight, meta))
                            } else {
                                rollback_err = Some(RknpuError::InvalidParameter);
                                None
                            }
                        } else {
                            rollback_err = Some(RknpuError::InternalError);
                            None
                        };

                        if let Some((inflight, meta)) = prepared {
                            // Publish the in-flight binding before touching the
                            // driver so any later raw core completion can always
                            // be mapped back into queue state.
                            state.inflight[core_slot] = Some(inflight);
                            debug!(
                                "[rknpu-scheduler] reserved dispatch queue_task={} core={} \
                                 subcore={} task_index={}",
                                inflight.queue_task_id,
                                core_slot,
                                inflight.subcore_slot,
                                inflight.task_index
                            );
                            Some((inflight, meta))
                        } else {
                            // Reservation failed to materialize into a concrete
                            // dispatch. Roll the queue reservation back through
                            // the queue state machine using the same token.
                            if let Some(err) = rollback_err {
                                if let Some(task_id) =
                                    state.queue.fail_dispatch(reservation, core_slot, err)
                                {
                                    terminal_ids.push(task_id);
                                }
                            }
                            None
                        }
                    } else {
                        None
                    }
                }
            };

            let Some((inflight, meta)) = setup else {
                continue;
            };

            // `confirm_write_all()` is per-submit visibility work, so one
            // worker sweep deduplicates it even if this submit fans out to
            // multiple hardware cores immediately.
            if confirmed_submit_ids.insert(inflight.queue_task_id) {
                debug!(
                    "[rknpu-scheduler] confirm_write_all queue_task={}",
                    inflight.queue_task_id
                );
                if let Err(err) = with_npu_driver(|rknpu_dev| rknpu_dev.comfirm_write_all()) {
                    warn!(
                        "[rknpu-scheduler] confirm_write_all failed for queue_task={}: {:?}",
                        inflight.queue_task_id, err
                    );
                    self.fail_dispatch(inflight, err);
                    continue;
                }
            }

            // The driver receives only the fields required to program hardware.
            // Queue/scheduler metadata stays in `InflightDispatch`.
            let submit_result = unsafe {
                let task_ptr = inflight.task_ptr as *mut RknpuTask;
                // `task_ptr` points into queue-owned task storage whose address
                // remains stable for the lifetime of the queue entry.
                with_npu_driver(|rknpu_dev| {
                    rknpu_dev.submit_ioctrl_step(
                        core_slot,
                        meta.flags,
                        meta.task_total,
                        meta.task_dma_base,
                        inflight.subcore_slot,
                        inflight.task_index,
                        &mut *task_ptr,
                    )
                })
            };

            match submit_result {
                Ok(()) => {
                    debug!(
                        "[rknpu-scheduler] dispatched queue_task={} core={} subcore={} \
                         task_index={}",
                        inflight.queue_task_id,
                        core_slot,
                        inflight.subcore_slot,
                        inflight.task_index
                    );
                    dispatched = true;
                }
                Err(err) => {
                    warn!(
                        "[rknpu-scheduler] dispatch failed for queue_task={} core={} task={}: {:?}",
                        inflight.queue_task_id, core_slot, inflight.task_index, err
                    );
                    self.fail_dispatch(inflight, err);
                }
            }
        }

        self.wake_terminal_tasks(terminal_ids, false);
        dispatched
    }

    /// Abort the waiter side of a submit without forcibly cancelling hardware.
    ///
    /// This is used when the blocking wait itself fails. The queue entry is
    /// faulted and later cleaned up once any already-running hardware drains.
    fn abort_wait(&self, task_id: RknpuQueueTaskId) {
        let mut state = self.state.lock();
        state.waiters.remove(&task_id);

        if state.queue.is_terminal(task_id) {
            let _ = state.queue.take_terminal_task(task_id);
            return;
        }

        // An interrupted waiter does not forcibly cancel hardware that is
        // already running. The queue entry is faulted and will become terminal
        // once any outstanding in-flight cores drain.
        let terminal = state
            .queue
            .mark_task_faulted(task_id, RknpuError::Interrupted);
        if terminal.is_some() {
            let _ = state.queue.take_terminal_task(task_id);
        }
    }
}

lazy_static! {
    static ref NPU_SCHEDULER: NpuScheduler = NpuScheduler::new();
}

/// Singleton worker loop that drives the scheduler state machine.
///
/// The worker alternates between two responsibilities:
/// 1. harvest completions from cores that already finished
/// 2. dispatch new work onto cores that became idle
///
/// It only sleeps on `kick` when the queue is fully idle. If hardware is still
/// in flight, it yields instead so IRQ handling can run and the next iteration
/// can harvest the completed cores.
fn worker_main() {
    warn!("[rknpu-scheduler] worker thread started");
    loop {
        // Sleep until at least one submit is queued or requeued. Once awake,
        // keep draining progress until the current wave is exhausted.
        NPU_SCHEDULER.wait_for_work();
        let (has_inflight_on_wake, has_work_on_wake) = {
            let state = NPU_SCHEDULER.state.lock();
            (state.has_inflight(), !state.queue.is_idle())
        };
        warn!(
            "[rknpu-scheduler] worker awakened has_inflight={} has_work={}",
            has_inflight_on_wake, has_work_on_wake
        );
        let mut harvest_rounds = 0u32;
        let mut dispatch_rounds = 0u32;
        let mut inflight_yields = 0u32;
        let mut stalled_yields = 0u32;

        loop {
            // One iteration tries both halves of the state machine:
            // 1. harvest anything the hardware already finished
            // 2. dispatch any newly idle cores
            let harvested = NPU_SCHEDULER.harvest_completed_cores();
            let dispatched = NPU_SCHEDULER.dispatch_idle_cores();

            if harvested {
                harvest_rounds = harvest_rounds.saturating_add(1);
            }
            if dispatched {
                dispatch_rounds = dispatch_rounds.saturating_add(1);
            }

            if harvested || dispatched {
                continue;
            }

            let (has_inflight, has_work) = {
                let state = NPU_SCHEDULER.state.lock();
                (state.has_inflight(), !state.queue.is_idle())
            };

            if has_inflight {
                inflight_yields = inflight_yields.saturating_add(1);
                if inflight_yields == 1 || inflight_yields % 64 == 0 {
                    warn!(
                        "[rknpu-scheduler] worker waiting on inflight hardware yields={} \
                         harvest_rounds={} dispatch_rounds={}",
                        inflight_yields, harvest_rounds, dispatch_rounds
                    );
                    NPU_SCHEDULER.log_inflight_snapshot(
                        inflight_yields,
                        harvest_rounds,
                        dispatch_rounds,
                    );
                }

                // Yield rather than sleeping on `kick`: there is already live
                // hardware owned by this worker, so progress now depends on IRQ
                // handling and subsequent harvest, not on a fresh submit.
                yield_now();
                continue;
            }

            if !has_work {
                warn!(
                    "[rknpu-scheduler] worker drained current wakeup and is returning to sleep \
                     harvest_rounds={} dispatch_rounds={} inflight_yields={} stalled_yields={}",
                    harvest_rounds, dispatch_rounds, inflight_yields, stalled_yields
                );
                break;
            }

            // The queue still contains work, but nothing progressed in this
            // iteration. Yield once so other tasks and IRQ handling can change
            // the scheduling conditions before the next worker sweep.
            stalled_yields = stalled_yields.saturating_add(1);
            if stalled_yields == 1 || stalled_yields % 64 == 0 {
                warn!(
                    "[rknpu-scheduler] worker still has queued work but no dispatch was possible \
                     yields={} harvest_rounds={} dispatch_rounds={}",
                    stalled_yields, harvest_rounds, dispatch_rounds
                );
            }

            // This path means the queue is non-idle but currently cannot make
            // progress in this iteration, for example due to scheduling
            // constraints or a just-faulted submit waiting for cleanup.
            yield_now();
        }
    }
}

/// Public enqueue entry used by the `card1` ioctl path.
pub fn enqueue_submit(queued_submit: RknpuQueuedSubmit) -> VfsResult<RknpuQueueTaskId> {
    NPU_SCHEDULER.enqueue(queued_submit)
}

/// Public blocking wait entry used after one submit has been enqueued.
pub fn wait_for_submit(task_id: RknpuQueueTaskId) -> VfsResult<()> {
    NPU_SCHEDULER.wait(task_id)
}

/// Public terminal collection entry used by the ioctl path after wait returns.
pub fn take_terminal_submit(task_id: RknpuQueueTaskId) -> VfsResult<CompletedSubmit> {
    NPU_SCHEDULER.take_terminal_task(task_id)
}

/// Borrow the global RKNPU device and execute one scheduler/driver operation.
///
/// This helper centralizes the conversion from VFS-side device acquisition
/// failure into the driver's `RknpuError::DeviceBusy`.
fn with_npu_driver<F, R>(f: F) -> Result<R, RknpuError>
where
    F: FnOnce(&mut rknpu::Rknpu) -> Result<R, RknpuError>,
{
    let mut guard = npu().map_err(|_| RknpuError::DeviceBusy)?;
    f(&mut guard)
}
