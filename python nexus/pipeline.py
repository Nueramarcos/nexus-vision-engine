"""
Nexus - Pipeline
Multi-stage chaining, per-job retry logic, and a dead-letter queue.
"""

import threading
import queue
import time
import logging
from dataclasses import dataclass, field
from typing import Any, Callable
from enum import Enum, auto

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(threadName)s] %(levelname)s %(message)s",
)
log = logging.getLogger(__name__)


# ── Job model ─────────────────────────────────────────────────────────────────

class JobStatus(Enum):
    PENDING   = auto()
    RUNNING   = auto()
    DONE      = auto()
    FAILED    = auto()
    RETRYING  = auto()
    DEAD      = auto()   # exhausted all retries → dead-letter


@dataclass
class Job:
    id: str
    payload: Any
    status: JobStatus = JobStatus.PENDING
    result: Any = None
    error: str | None = None
    attempt: int = 0
    max_retries: int = 3
    _stage: int = field(default=0, repr=False)   # which stage produced this job

    @property
    def exhausted(self) -> bool:
        return self.attempt >= self.max_retries


# ── Stage ─────────────────────────────────────────────────────────────────────

@dataclass
class Stage:
    """One processing step in a multi-stage pipeline."""
    name: str
    worker_fn: Callable[[Job], Any]
    num_workers: int = 2
    maxsize: int = 0
    retry_delay: float = 0.1   # seconds between retries


# ── Dead-letter queue ─────────────────────────────────────────────────────────

class DeadLetterQueue:
    """Collects jobs that have exhausted all retries."""

    def __init__(self):
        self._jobs: list[Job] = []
        self._lock = threading.Lock()

    def put(self, job: Job) -> None:
        job.status = JobStatus.DEAD
        with self._lock:
            self._jobs.append(job)
        log.warning("DLQ ← job %s after %d attempts: %s", job.id, job.attempt, job.error)

    def drain(self) -> list[Job]:
        with self._lock:
            jobs, self._jobs = self._jobs, []
        return jobs

    @property
    def size(self) -> int:
        with self._lock:
            return len(self._jobs)


# ── Single-stage worker ───────────────────────────────────────────────────────

class _StageRunner:
    """Internal: runs one Stage with its own thread pool and queues."""

    def __init__(
        self,
        stage: Stage,
        out_queue: "queue.Queue[Job | None] | None",
        dlq: DeadLetterQueue,
    ):
        self._stage = stage
        self._in: queue.Queue[Job | None] = queue.Queue(maxsize=stage.maxsize)
        self._out = out_queue          # None for the final stage
        self._dlq = dlq
        self._completed: list[Job] = []
        self._lock = threading.Lock()
        self._workers: list[threading.Thread] = []

    def start(self) -> "_StageRunner":
        for i in range(self._stage.num_workers):
            t = threading.Thread(
                target=self._loop,
                name=f"{self._stage.name}-w{i}",
                daemon=True,
            )
            t.start()
            self._workers.append(t)
        return self

    def stop(self) -> None:
        for _ in self._workers:
            self._in.put(None)
        for t in self._workers:
            t.join()
        self._workers.clear()

    def submit(self, job: Job) -> None:
        self._in.put(job)

    def join(self) -> None:
        self._in.join()

    @property
    def completed(self) -> list[Job]:
        with self._lock:
            return list(self._completed)

    # ── internals ──────────────────────────────────────────────────────────

    def _loop(self) -> None:
        while True:
            item = self._in.get()
            if item is None:
                self._in.task_done()
                break
            self._process(item)
            self._in.task_done()

    def _process(self, job: Job) -> None:
        job.status = JobStatus.RUNNING
        job.attempt += 1
        log.info("[%s] attempt=%d job=%s", self._stage.name, job.attempt, job.id)

        try:
            job.result = self._stage.worker_fn(job)
            job.status = JobStatus.DONE
            job._stage += 1
            log.info("[%s] done job=%s result=%r", self._stage.name, job.id, job.result)

            if self._out is not None:
                # reset attempt counter for the next stage
                job.attempt = 0
                job.status = JobStatus.PENDING
                self._out.put(job)
            else:
                with self._lock:
                    self._completed.append(job)

        except Exception as exc:
            job.error = str(exc)
            log.warning("[%s] error job=%s: %s", self._stage.name, job.id, exc)

            if job.exhausted:
                self._dlq.put(job)
            else:
                # Retry: re-queue after a short delay
                job.status = JobStatus.RETRYING
                delay = self._stage.retry_delay
                log.info("[%s] retry job=%s in %.2fs (attempt %d/%d)",
                         self._stage.name, job.id, delay, job.attempt, job.max_retries)

                def _retry(j=job):
                    time.sleep(delay)
                    j.status = JobStatus.PENDING
                    self._in.put(j)
                    # we must re-add a task to the queue's unfinished count
                    # because task_done() is called after this method returns

                threading.Thread(target=_retry, daemon=True).start()


# ── Multi-stage Pipeline ──────────────────────────────────────────────────────

class Pipeline:
    """
    Chain multiple Stages so output of stage N feeds input of stage N+1.
    Failed jobs that exhaust retries go to the DeadLetterQueue.

    Usage:
        stages = [
            Stage("parse",   parse_fn,   num_workers=2),
            Stage("enrich",  enrich_fn,  num_workers=3),
            Stage("persist", persist_fn, num_workers=1),
        ]
        p = Pipeline(stages)
        p.start()
        p.submit(Job(id="1", payload=raw_data))
        p.join()
        p.stop()
        print(p.completed)   # final-stage successes
        print(p.dlq.drain()) # permanent failures
    """

    def __init__(self, stages: list[Stage]):
        if not stages:
            raise ValueError("Pipeline needs at least one stage")
        self.dlq = DeadLetterQueue()
        self._stages = stages
        self._runners: list[_StageRunner] = []
        self._running = False

    def start(self) -> "Pipeline":
        if self._running:
            return self
        # Build runners back-to-front so each has a reference to the next queue
        runners: list[_StageRunner] = []
        next_q: "queue.Queue | None" = None
        for stage in reversed(self._stages):
            runner = _StageRunner(stage, next_q, self.dlq)
            next_q = runner._in
            runners.append(runner)
        self._runners = list(reversed(runners))
        for r in self._runners:
            r.start()
        self._running = True
        log.info("Pipeline started (%d stage(s))", len(self._stages))
        return self

    def stop(self) -> None:
        for r in self._runners:
            r.stop()
        self._running = False
        log.info("Pipeline stopped")

    def join(self) -> None:
        """Block until all jobs have cleared every stage."""
        for r in self._runners:
            r.join()

    def submit(self, job: Job) -> None:
        if not self._running:
            raise RuntimeError("Call start() first")
        self._runners[0].submit(job)

    @property
    def completed(self) -> list[Job]:
        """Jobs that made it through every stage successfully."""
        return self._runners[-1].completed


# ── Demo ──────────────────────────────────────────────────────────────────────

_call_counts: dict[str, int] = {}
_lock = threading.Lock()

def _flaky(job: Job, fail_until: int) -> str:
    """Fails for the first `fail_until` attempts, then succeeds."""
    with _lock:
        _call_counts[job.id] = _call_counts.get(job.id, 0) + 1
        count = _call_counts[job.id]
    if count <= fail_until:
        raise ValueError(f"transient error (attempt {count})")
    return f"{job.payload}_processed"


def main():
    stages = [
        Stage(
            "stage1-validate",
            worker_fn=lambda job: str(job.payload).strip(),
            num_workers=2,
            retry_delay=0.05,
        ),
        Stage(
            "stage2-transform",
            # job-3 will fail once, job-4 will exhaust retries (max_retries=3, fails 4x)
            worker_fn=lambda job: _flaky(
                job,
                fail_until=(1 if job.id == "job-3" else 4 if job.id == "job-4" else 0),
            ),
            num_workers=2,
            retry_delay=0.05,
        ),
        Stage(
            "stage3-persist",
            worker_fn=lambda job: f"saved:{job.result}",
            num_workers=1,
            retry_delay=0.05,
        ),
    ]

    pipeline = Pipeline(stages)
    pipeline.start()

    for i in range(6):
        pipeline.submit(Job(id=f"job-{i}", payload=f"  task_{i}  ", max_retries=3))

    pipeline.join()
    pipeline.stop()

    print("\n── Completed ──")
    for job in sorted(pipeline.completed, key=lambda j: j.id):
        print(f"  {job.id}: {job.result}")

    print("\n── Dead-letter queue ──")
    dead = pipeline.dlq.drain()
    if dead:
        for job in dead:
            print(f"  {job.id}: {job.error} (after {job.attempt} attempts)")
    else:
        print("  (empty)")


if __name__ == "__main__":
    main()
