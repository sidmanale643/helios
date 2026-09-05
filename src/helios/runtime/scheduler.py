import time
from collections import deque
from collections.abc import Callable
from concurrent.futures import Future
from dataclasses import dataclass, field
from threading import Condition, Thread
from typing import Generic, TypeVar

Payload = TypeVar("Payload")
Result = TypeVar("Result")


class QueueFullError(RuntimeError):
    pass


class SchedulerClosedError(RuntimeError):
    pass


@dataclass
class Job(Generic[Payload, Result]):
    payload: Payload
    request_ids: tuple[str, ...]
    batchable: bool = True
    enqueued_at: float = field(default_factory=time.perf_counter)
    future: Future[Result] = field(default_factory=Future)


class Scheduler(Generic[Payload, Result]):
    def __init__(
        self,
        execute: Callable[[tuple[Job[Payload, Result], ...]], tuple[Result, ...]],
        can_add: Callable[
            [tuple[Job[Payload, Result], ...], Job[Payload, Result]], bool
        ],
        *,
        max_batch_size: int,
        max_queue_size: int,
        batch_wait_seconds: float,
    ) -> None:
        self._execute = execute
        self._can_add = can_add
        self._max_batch_size = max_batch_size
        self._max_queue_size = max_queue_size
        self._batch_wait_seconds = batch_wait_seconds
        self._condition = Condition()
        self._waiting: deque[Job[Payload, Result]] = deque()
        self._active: tuple[Job[Payload, Result], ...] = ()
        self._closed = False
        self._worker = Thread(target=self._run, name="helios-scheduler", daemon=True)
        self._worker.start()

    def submit(self, job: Job[Payload, Result]) -> Result:
        return self.enqueue(job).result()

    def enqueue(self, job: Job[Payload, Result]) -> Future[Result]:
        with self._condition:
            if self._closed:
                raise SchedulerClosedError("The generation scheduler is closed.")
            self._discard_cancelled()
            if len(self._waiting) >= self._max_queue_size:
                raise QueueFullError("The generation waiting queue is full.")
            self._waiting.append(job)
            self._condition.notify()
        return job.future

    def snapshot(self) -> dict[str, object]:
        with self._condition:
            return {
                "waiting": [
                    request_id
                    for job in self._waiting
                    for request_id in job.request_ids
                ],
                "active": [
                    request_id for job in self._active for request_id in job.request_ids
                ],
                "max_batch_size": self._max_batch_size,
                "max_queue_size": self._max_queue_size,
                "batch_wait_ms": self._batch_wait_seconds * 1_000,
            }

    def close(self) -> None:
        with self._condition:
            if self._closed:
                return
            self._closed = True
            waiting = tuple(self._waiting)
            self._waiting.clear()
            self._condition.notify_all()
        error = SchedulerClosedError("The generation scheduler is closed.")
        for job in waiting:
            if not job.future.done():
                job.future.set_exception(error)
        self._worker.join()

    def _run(self) -> None:
        while True:
            with self._condition:
                while not self._waiting and not self._closed:
                    self._condition.wait()
                self._discard_cancelled()
                if self._closed and not self._waiting:
                    return
                if not self._waiting:
                    continue
                self._wait_for_batch()
                self._discard_cancelled()
                if self._closed and not self._waiting:
                    return
                if not self._waiting:
                    continue
                active = self._promote()
                self._active = active

            try:
                results = self._execute(active)
                if len(results) != len(active):
                    raise RuntimeError(
                        "The scheduler executor returned the wrong result count."
                    )
            except Exception as error:  # noqa: BLE001
                for job in active:
                    if not job.future.done():
                        job.future.set_exception(error)
            else:
                for job, result in zip(active, results, strict=True):
                    if not job.future.done():
                        job.future.set_result(result)
            finally:
                with self._condition:
                    self._active = ()
                    self._condition.notify_all()

    def _wait_for_batch(self) -> None:
        first = self._waiting[0]
        if not first.batchable or self._max_batch_size == 1:
            return
        deadline = first.enqueued_at + self._batch_wait_seconds
        while len(self._waiting) < self._max_batch_size and not self._closed:
            remaining = deadline - time.perf_counter()
            if remaining <= 0:
                return
            self._condition.wait(remaining)

    def _promote(self) -> tuple[Job[Payload, Result], ...]:
        first = self._waiting.popleft()
        active = [first]
        if not first.batchable:
            return tuple(active)
        while self._waiting and len(active) < self._max_batch_size:
            candidate = self._waiting[0]
            if not candidate.batchable or not self._can_add(tuple(active), candidate):
                break
            active.append(self._waiting.popleft())
        return tuple(active)

    def _discard_cancelled(self) -> None:
        self._waiting = deque(
            job for job in self._waiting if not job.future.cancelled()
        )
