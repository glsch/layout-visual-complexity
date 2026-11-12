"""Running the steps: every step at once, each in worker processes, every page moving on as soon as it's ready.

The main process owns the page records; workers only process batches. A batch that fails is retried page by page, so
one bad page fails alone; a batch too big for its device is halved; a worker that dies is replaced. Results never
depend on the number of workers, or on the order pages arrive in.
"""

import contextlib
import multiprocessing
import queue
import signal
import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from multiprocessing.connection import Connection
from multiprocessing.connection import wait as wait_for
from typing import Any

from layout_visual_complexity.logs import logger
from layout_visual_complexity.pipeline.step import IMAGE, OutOfMemory, Step, usage
from layout_visual_complexity.segmentation.records import PageRecord, Status

MAX_RESTARTS = 5


class RunError(RuntimeError):
    """A run that can't go on: a step that can't load, or a worker that keeps dying."""


@dataclass(frozen=True)
class Placement:
    """Where a step's workers run, and how many."""

    stage: str
    device: str
    workers: int


def work(step: Step, device: str, worker: int, tasks: Connection, results: Connection, control: Connection) -> None:
    """A worker process: load the step once, then process batches until told to stop.

    Each message carries the process's CPU seconds and peak memory so far, for the run's report.
    """
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    began = time.perf_counter()
    try:
        step.load(device)
    except Exception as error:
        results.send(("load failed", worker, f"{type(error).__name__}: {error}", usage()))
        return
    results.send(("ready", worker, time.perf_counter() - began, usage()))
    parent = multiprocessing.parent_process()
    stopping = False
    while True:
        if not tasks.poll(1.0):
            if parent is not None and not parent.is_alive():
                return
            continue
        try:
            task = tasks.recv()
        except EOFError:
            return
        if task is None:
            return
        batch, pages = task
        stopping = stopping or control.poll()
        if stopping:
            results.send(("skipped", worker, batch, usage()))
            continue
        results.send(("started", worker, batch, usage()))
        began = time.perf_counter()
        try:
            done = step.process(pages)
        except OutOfMemory as error:
            results.send(("error", worker, batch, "memory", str(error) or "out of memory", usage()))
        except Exception as error:
            results.send(("error", worker, batch, "error", f"{type(error).__name__}: {error}", usage()))
        else:
            results.send(("done", worker, batch, done, time.perf_counter() - began, usage()))


def feed(connection: Connection, outbox: "queue.Queue[Any]") -> None:
    """Send a worker its tasks from a thread, so the main process never waits on a busy worker's pipe."""
    while True:
        item = outbox.get()
        try:
            connection.send(item)
        except (OSError, EOFError):
            return
        if item is None:
            return


@dataclass
class Batch:
    id: int
    stage: str
    pages: list[str]


@dataclass
class Worker:
    id: int
    stage: str
    device: str
    process: Any
    results: Connection
    control: Connection
    tasks: Connection
    outbox: "queue.Queue[Any]"
    feeder: threading.Thread
    in_flight: dict[int, Batch] = field(default_factory=dict)
    current: int | None = None
    usage: dict[str, float] = field(default_factory=dict)
    busy: float = 0.0

    def close(self) -> None:
        for connection in (self.results, self.control, self.tasks):
            connection.close()


@dataclass
class Pending:
    """Pages waiting for one step, oldest first; `alone` pages go in batches of their own."""

    pages: deque[str] = field(default_factory=deque)
    since: float | None = None
    alone: set[str] = field(default_factory=set)


@dataclass
class StageUsage:
    """What a stage's workers used: CPU seconds, the largest peak memory of one worker, and seconds spent on
    batches."""

    cpu: float = 0.0
    peak_memory: float = 0.0
    busy: float = 0.0
    load: float = 0.0
    workers: int = 0
    device: str = ""


def upstream_of(steps: dict[str, Step]) -> dict[str, list[str]]:
    """For each step, the earlier steps whose products it needs."""
    producer: dict[str, str] = {}
    upstream: dict[str, list[str]] = {}
    for name, step in steps.items():
        needed = []
        for need in sorted(step.needs - {IMAGE}):
            if need not in producer:
                raise RunError(f"step {name!r} needs {need}, which no step before it produces")
            needed.append(producer[need])
        upstream[name] = needed
        for product in step.produces:
            producer[product] = name
    return upstream


class Scheduler:
    """Runs the steps over the pages handed to it.

    `on_page` is called with each page's record once every step is done with it (or failed), `on_document` once all
    of a document's pages are.
    """

    def __init__(
        self,
        steps: dict[str, Step],
        placements: list[Placement],
        on_page: Callable[[PageRecord], None] | None = None,
        on_document: Callable[[str], None] | None = None,
        batch_wait: float = 0.3,
        in_flight: int = 2,
    ) -> None:
        self.steps = steps
        for name, step in steps.items():
            step.name = name
        self.order = list(steps)
        self.upstream = upstream_of(steps)
        self.placements = placements
        self.on_page = on_page
        self.on_document = on_document
        self.batch_wait = batch_wait
        self.in_flight = in_flight
        self.context = multiprocessing.get_context("spawn")
        self.stopping = False
        self.images: queue.Queue[tuple[str, Any]] = queue.Queue()
        self.records: dict[str, PageRecord] = {}
        self.documents: dict[str, list[str]] = {}
        self.finished: dict[str, dict[str, str]] = {}
        self.queued: set[tuple[str, str]] = set()
        self.pending = {name: Pending() for name in self.order}
        self.batch_size = {name: step.batch_size for name, step in steps.items()}
        self.workers: dict[int, Worker] = {}
        self.restarts: dict[str, int] = {}
        self.next_id = 0
        self.unfinished = 0
        self.unfinished_in: dict[str, int] = {}
        self.remaining = {name: 0 for name in self.order}
        self.started: set[str] = set()
        self.usage = {name: StageUsage() for name in self.order}
        self.retired: dict[str, StageUsage] = {name: StageUsage() for name in self.order}

    # Setting up.

    def _start_worker(self, name: str, device: str) -> Worker:
        self.next_id += 1
        task_reader, task_writer = self.context.Pipe(duplex=False)
        result_reader, result_writer = self.context.Pipe(duplex=False)
        control_reader, control_writer = self.context.Pipe(duplex=False)
        process = self.context.Process(
            target=work,
            args=(self.steps[name], device, self.next_id, task_reader, result_writer, control_reader),
            name=f"{name}-{device}-{self.next_id}",
            daemon=True,
        )
        process.start()
        for end in (task_reader, result_writer, control_reader):
            end.close()  # the worker's ends: closing ours lets a dead worker's pipe show its end
        outbox: queue.Queue[Any] = queue.Queue()
        feeder = threading.Thread(target=feed, args=(task_writer, outbox), name=f"feed-{self.next_id}", daemon=True)
        feeder.start()
        worker = Worker(self.next_id, name, device, process, result_reader, control_writer, task_writer, outbox, feeder)
        self.workers[worker.id] = worker
        return worker

    def _start_workers(self, name: str) -> None:
        """A step's workers, started the first time it has pages to do."""
        if name in self.started:
            return
        self.started.add(name)
        for placement in self.placements:
            if placement.stage == name:
                for _ in range(placement.workers):
                    self._start_worker(name, placement.device)
                self.usage[name].device = placement.device
                self.usage[name].workers += placement.workers

    def add_document(self, name: str, records: list[PageRecord]) -> None:
        """A document's pages to run, in order."""
        keys = []
        for record in records:
            self.records[record.page] = record
            self.finished[record.page] = {}
            keys.append(record.page)
        self.documents[name] = keys
        self.unfinished_in[name] = len(keys)
        for stage in self.order:
            self.remaining[stage] += len(keys)
        self.unfinished += len(keys)

    def image_ready(self, page: str, image: Any) -> None:
        """Called from any thread when a page's image is ready (a record of it), or failed (a string: the reason)."""
        self.images.put((page, image))

    # Moving pages along.

    def _finish(self, page: str, stage: str, state: str, reason: str | None, seconds: float) -> None:
        record = self.records[page]
        record.status[stage] = Status(state, reason)  # type: ignore[arg-type]
        record.seconds[stage] = record.seconds.get(stage, 0.0) + seconds
        self.finished[page][stage] = state
        if len(self.finished[page]) == len(self.order):
            self.unfinished -= 1
            document = record.document
            self.unfinished_in[document] -= 1
            if self.on_page:
                self.on_page(record)
            if self.unfinished_in[document] == 0 and self.on_document:
                self.on_document(document)

    def _advance(self, page: str) -> None:
        """Queue the page for every step whose inputs are ready."""
        record = self.records[page]
        for name in self.order:
            if name in self.finished[page] or (page, name) in self.queued:
                continue
            if any(up not in self.finished[page] for up in self.upstream[name]):
                continue
            self.remaining[name] -= 1
            failed = [up for up in self.upstream[name] if self.finished[page][up] == "failed"]
            if failed:
                self._finish(page, name, "failed", self._inherited(record, failed[0]), 0.0)
                continue
            self.queued.add((page, name))
            pending = self.pending[name]
            pending.pages.append(page)
            pending.since = pending.since or time.monotonic()

    @staticmethod
    def _inherited(record: PageRecord, upstream: str) -> str:
        reason = record.status[upstream].reason or ""
        return reason if reason.endswith(" failed") else f"{upstream} failed"

    def _dispatch(self) -> None:
        now = time.monotonic()
        for name in self.order:
            pending = self.pending[name]
            if not pending.pages:
                continue
            self._start_workers(name)
            size = self.batch_size[name]
            full = len(pending.pages) >= size
            waited = pending.since is not None and now - pending.since >= self.batch_wait
            if not (full or waited or self.remaining[name] == 0):
                continue
            workers = sorted(
                (w for w in self.workers.values() if w.stage == name and len(w.in_flight) < self.in_flight),
                key=lambda w: len(w.in_flight),
            )
            for worker in workers:
                batch = self._take(pending, size)
                if not batch:
                    break
                self._send(worker, name, batch)
            pending.since = now if pending.pages else None

    @staticmethod
    def _take(pending: Pending, size: int) -> list[str]:
        taken: list[str] = []
        skipped: list[str] = []
        while pending.pages and len(taken) < size:
            page = pending.pages.popleft()
            if page in pending.alone:
                if taken:
                    skipped.append(page)
                    continue
                taken.append(page)
                break
            taken.append(page)
        pending.pages.extendleft(reversed(skipped))
        return taken

    def _send(self, worker: Worker, name: str, pages: list[str]) -> None:
        self.next_id += 1
        batch = Batch(self.next_id, name, pages)
        worker.in_flight[batch.id] = batch
        copies = [PageRecord.from_dict(self.records[page].to_dict()) for page in pages]
        worker.outbox.put((batch.id, copies))

    # Results.

    def _handle(self, message: tuple) -> None:
        kind, worker_id = message[0], message[1]
        worker = self.workers.get(worker_id)
        if worker is None:
            return
        worker.usage = message[-1]
        if kind == "ready":
            self.usage[worker.stage].load = max(self.usage[worker.stage].load, message[2])
            return
        if kind == "started":
            worker.current = message[2]
            return
        if kind == "skipped":
            worker.in_flight.pop(message[2], None)
            return
        if kind == "load failed":
            raise RunError(f"step {worker.stage!r} couldn't load on {worker.device}: {message[2]}")
        batch = worker.in_flight.pop(message[2], None)
        worker.current = None
        if batch is None:
            return
        if kind == "done":
            worker.busy += message[4]
            self._done(batch, message[3], message[4])
        else:
            self._failed(batch, message[3], message[4])

    def _done(self, batch: Batch, pages: list[PageRecord], seconds: float) -> None:
        step = self.steps[batch.stage]
        if len(pages) != len(batch.pages):
            self._failed(batch, "error", f"the step returned {len(pages)} pages for {len(batch.pages)}")
            return
        for page, result in zip(batch.pages, pages, strict=True):
            record = self.records[page]
            step.merge(record, result)
            status = record.status[batch.stage]
            self.queued.discard((page, batch.stage))
            self._finish(page, batch.stage, status.state, status.reason, seconds / len(pages))
            if not self.stopping:
                self._advance(page)

    def _failed(self, batch: Batch, kind: str, message: str) -> None:
        pending = self.pending[batch.stage]
        if kind == "memory" and len(batch.pages) > 1:
            self.batch_size[batch.stage] = max(1, len(batch.pages) // 2)
            logger.warning("%s: %s; batches of %d from now on", batch.stage, message, self.batch_size[batch.stage])
            pending.pages.extendleft(reversed(batch.pages))
            return
        if len(batch.pages) > 1:
            logger.info("%s: a batch failed (%s); trying its pages one by one", batch.stage, message)
            pending.alone.update(batch.pages)
            pending.pages.extendleft(reversed(batch.pages))
            return
        self._give_up(batch.pages[0], batch.stage, message)

    def _give_up(self, page: str, stage: str, reason: str) -> None:
        logger.warning("%s: %s failed: %s", page, stage, reason)
        self.queued.discard((page, stage))
        self._finish(page, stage, "failed", reason, 0.0)
        if not self.stopping:
            self._advance(page)

    def _retire(self, worker: Worker) -> None:
        """Keep what a worker that's gone used, for the report."""
        used = self.retired[worker.stage]
        used.cpu += worker.usage.get("cpu", 0.0)
        used.busy += worker.busy
        used.peak_memory = max(used.peak_memory, worker.usage.get("peak_memory", 0.0))

    def _check_workers(self) -> None:
        dead = [worker for worker in self.workers.values() if not worker.process.is_alive()]
        for worker in dead:
            self._drain(worker)
            del self.workers[worker.id]
            self._retire(worker)
            worker.close()
            code = worker.process.exitcode
            for batch in worker.in_flight.values():
                if batch.id == worker.current:
                    self._failed(batch, "error", f"the worker stopped (exit code {code})")
                else:
                    self.pending[batch.stage].pages.extendleft(reversed(batch.pages))
            restarts = self.restarts.get(worker.stage, 0) + 1
            self.restarts[worker.stage] = restarts
            if self.stopping:
                continue
            if restarts > MAX_RESTARTS:
                raise RunError(
                    f"step {worker.stage!r}: workers stopped {restarts} times; the last exit code was {code}"
                )
            logger.warning(
                "%s: a worker on %s stopped (exit code %s); starting another", worker.stage, worker.device, code
            )
            self._start_worker(worker.stage, worker.device)

    def _take_images(self) -> None:
        while True:
            try:
                page, image = self.images.get_nowait()
            except queue.Empty:
                return
            record = self.records[page]
            if isinstance(image, str):
                for name in self.order:
                    if name not in self.finished[page] and (page, name) not in self.queued:
                        self.remaining[name] -= 1
                        self._finish(page, name, "failed", image, 0.0)
                continue
            record.image = str(image.path)
            record.width, record.height = image.width, image.height
            record.full_width, record.full_height = image.full_width, image.full_height
            record.seconds["fetch"] = getattr(image, "seconds", 0.0)
            self._advance(page)

    def all_done(self) -> bool:
        return self.unfinished == 0

    def cpu_seconds(self) -> float:
        """CPU seconds used so far by the workers, those gone included."""
        live = sum(worker.usage.get("cpu", 0.0) for worker in self.workers.values())
        return live + sum(used.cpu for used in self.retired.values())

    def stage_usage(self) -> dict[str, StageUsage]:
        """Per stage: CPU seconds, peak memory of its busiest worker, seconds on batches, workers and device."""
        found = {}
        for name in self.order:
            total = StageUsage(
                cpu=self.retired[name].cpu,
                peak_memory=self.retired[name].peak_memory,
                busy=self.retired[name].busy,
                load=self.usage[name].load,
                workers=self.usage[name].workers,
                device=self.usage[name].device,
            )
            for worker in self.workers.values():
                if worker.stage == name:
                    total.cpu += worker.usage.get("cpu", 0.0)
                    total.busy += worker.busy
                    total.peak_memory = max(total.peak_memory, worker.usage.get("peak_memory", 0.0))
            found[name] = total
        return found

    def run(self, tick: Callable[[], None] | None = None) -> bool:
        """Run until every page is done. Returns False if stopped by Ctrl-C; what's done is kept either way."""
        try:
            while not self.all_done():
                self._take_images()
                self._dispatch()
                self._receive(0.02)
                self._check_workers()
                if tick:
                    tick()
            return True
        except KeyboardInterrupt:
            logger.warning("stopping: finishing the batches under way; running the same command again resumes")
            self._wind_down()
            return False
        finally:
            self._stop_workers()

    def _receive(self, wait: float) -> int:
        by_connection = {worker.results: worker for worker in self.workers.values()}
        handled = 0
        for connection in wait_for(list(by_connection), timeout=wait):
            handled += self._drain(by_connection[connection])  # type: ignore[index]
        return handled

    def _drain(self, worker: Worker) -> int:
        handled = 0
        while True:
            try:
                if not worker.results.poll():
                    return handled
                message = worker.results.recv()
            except (EOFError, OSError):
                return handled
            self._handle(message)
            handled += 1

    def _wind_down(self, patience: float = 10.0) -> None:
        """After Ctrl-C: take back the batches not started, and keep those under way when they finish."""
        self.stopping = True
        for worker in self.workers.values():
            while True:
                try:
                    task = worker.outbox.get_nowait()
                except queue.Empty:
                    break
                worker.in_flight.pop(task[0], None)
            with contextlib.suppress(OSError):
                worker.control.send("stop")
        deadline = time.monotonic() + patience
        try:
            while any(w.in_flight for w in self.workers.values()) and time.monotonic() < deadline:
                self._receive(0.05)
                self._check_workers()
        except KeyboardInterrupt:
            pass

    def _stop_workers(self) -> None:
        for worker in self.workers.values():
            worker.outbox.put(None)
        deadline = time.monotonic() + 5
        for worker in list(self.workers.values()):
            worker.process.join(max(0.0, deadline - time.monotonic()))
            if worker.process.is_alive():
                worker.process.terminate()
                worker.process.join(1)
            self._drain(worker)
            self._retire(worker)
            worker.close()
            worker.feeder.join(1)
        self.workers.clear()
