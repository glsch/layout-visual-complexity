"""How busy the machine is during a run: the cores (from the CPU time of the run's processes), the GPU and memory
(sampled from the system), every few seconds.

The GPU's busy share and the memory in use are read on macOS (`ioreg`, `sysctl`) and Linux (`nvidia-smi`,
`/proc/meminfo`); elsewhere they're left out.
"""

import itertools
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field


def cores() -> int:
    return os.cpu_count() or 1


def total_memory() -> int | None:
    """The machine's memory in bytes."""
    try:
        if sys.platform == "darwin":
            return int(subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True, text=True).stdout)
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    except (OSError, ValueError):
        return None


def gpu_busy() -> float | None:
    """The GPU's busy share now, from 0 to 1: Apple's `Device Utilization %`, or NVIDIA's utilisation."""
    try:
        if sys.platform == "darwin":
            output = subprocess.run(
                ["ioreg", "-r", "-d", "1", "-w", "0", "-c", "IOAccelerator"], capture_output=True, text=True, timeout=5
            ).stdout
            found = re.search(r'"Device Utilization %"=(\d+)', output)
            return int(found[1]) / 100 if found else None
        if shutil.which("nvidia-smi"):
            output = subprocess.run(
                ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
                capture_output=True,
                text=True,
                timeout=5,
            ).stdout
            values = [float(line) for line in output.split()]
            return sum(values) / len(values) / 100 if values else None
    except (OSError, subprocess.SubprocessError, ValueError):
        return None
    return None


def memory_available() -> int | None:
    """Bytes of memory the system says are available: macOS's memory status level (the share `memory_pressure`
    reports free), or Linux's `MemAvailable`."""
    try:
        if sys.platform == "darwin":
            level = subprocess.run(
                ["sysctl", "-n", "kern.memorystatus_level"], capture_output=True, text=True, timeout=5
            )
            total = total_memory()
            return int(int(level.stdout) / 100 * total) if total else None
        with open("/proc/meminfo") as stream:
            values = {line.split(":")[0]: int(line.split()[1]) * 1024 for line in stream}
        return values["MemAvailable"]
    except (OSError, subprocess.SubprocessError, ValueError, KeyError):
        return None


def memory_in_use() -> int | None:
    """Bytes of memory in use: the machine's less what the system says is available."""
    total, available = total_memory(), memory_available()
    return total - available if total is not None and available is not None else None


def cpu_time(text: str) -> float:
    """Seconds from `ps`'s CPU time: `[[days-]hours:]minutes:seconds`."""
    days, _, clock = text.strip().rpartition("-")
    seconds = 0.0
    for part in clock.split(":"):
        seconds = seconds * 60 + float(part)
    return seconds + 86400 * int(days or 0)


def process_tree_cpu(pid: int | None = None) -> float | None:
    """CPU seconds used so far by a process (this one by default) and its children, from `ps`."""
    pid = pid or os.getpid()
    try:
        output = subprocess.run(["ps", "-A", "-o", "pid=,ppid=,time="], capture_output=True, text=True, timeout=5)
        total = 0.0
        for line in output.stdout.splitlines():
            fields = line.split()
            if len(fields) == 3 and pid in (int(fields[0]), int(fields[1])):
                total += cpu_time(fields[2])
        return total
    except (OSError, subprocess.SubprocessError, ValueError):
        return None


@dataclass
class Sample:
    """One reading: seconds since the start, CPU seconds of the run's processes as `ps` sees them (None if it
    can't) and as the workers report them (each at the end of its batches), the GPU's busy share, the memory in use."""

    seconds: float
    tree: float | None
    reported: float
    gpu: float | None
    memory: int | None


@dataclass
class Monitor:
    """Samples the run every `interval` seconds: the CPU seconds of this process and its children (from `ps`, or
    else from `cpu_seconds`), the GPU's busy share and the memory in use."""

    cpu_seconds: Callable[[], float]
    interval: float = 2.0
    samples: list[Sample] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._loop, name="monitor", daemon=True)
        self.began = time.monotonic()
        self.baseline_memory = memory_in_use()

    def start(self) -> "Monitor":
        self.began = time.monotonic()
        self.baseline_cpu = process_tree_cpu() or 0.0
        self.thread.start()
        return self

    def stop(self) -> None:
        self.stop_event.set()
        self.thread.join(10)
        self._sample()

    def _sample(self) -> None:
        tree = process_tree_cpu()
        tree = tree - self.baseline_cpu if tree is not None else None
        self.samples.append(
            Sample(time.monotonic() - self.began, tree, float(self.cpu_seconds()), gpu_busy(), memory_in_use())
        )

    def _loop(self) -> None:
        while not self.stop_event.wait(self.interval):
            try:
                self._sample()
            except Exception:  # a failed reading must not stop the run
                continue

    def summary(self) -> dict[str, float | int | None]:
        """The cores' busy share over the run and at its busiest interval, the GPU's mean and peak busy share, and
        the most memory in use, with what was in use before the run."""
        if len(self.samples) < 1:
            return {}
        last = self.samples[-1]
        wall = max(last.seconds, 1e-9)
        # The whole run's CPU: `ps` sees the workers while they run, their own reports count them once they've stopped.
        total = max([last.reported] + [sample.tree for sample in self.samples if sample.tree is not None])
        # The busiest interval, from `ps` alone: the workers report a batch's CPU only when it ends.
        shares = []
        timed = [sample for sample in self.samples if sample.tree is not None]
        for previous, sample in itertools.pairwise(timed):
            if sample.seconds > previous.seconds and sample.tree >= previous.tree:  # type: ignore[operator]
                shares.append((sample.tree - previous.tree) / ((sample.seconds - previous.seconds) * cores()))  # type: ignore[operator]
        gpu = [sample.gpu for sample in self.samples if sample.gpu is not None]
        memory = [sample.memory for sample in self.samples if sample.memory is not None]
        return {
            "cores": cores(),
            "cores_busy": total / (wall * cores()),
            "cores_busy_peak": max(shares) if shares else None,
            "gpu_busy": sum(gpu) / len(gpu) if gpu else None,
            "gpu_busy_peak": max(gpu) if gpu else None,
            "memory_peak": max(memory) if memory else None,
            "memory_before": self.baseline_memory,
            "memory_total": total_memory(),
            "samples": len(self.samples),
        }
