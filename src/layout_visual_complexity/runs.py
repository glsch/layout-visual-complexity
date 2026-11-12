"""The record of a run: its configuration as YAML, and a `manifest.json` with its inputs (and their checksums), its
outputs, the configuration, the versions of the package and its libraries, and when it started and ended.

Paths are written as given, or relative to the working folder, or from `~`, never as a home folder's full path.
"""

import dataclasses
import hashlib
import json
import platform
import subprocess
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path
from typing import Any

import yaml

import layout_visual_complexity

LIBRARIES = [
    "numpy",
    "pandas",
    "scipy",
    "scikit-learn",
    "matplotlib",
    "jsonargparse",
    "PyYAML",
    "httpx2",
    "opencv-python-headless",
    "Pillow",
    "pyarrow",
    "shapely",
    "torch",
    "kraken",
    "ultralytics",
]


def shown_path(path: Path | str) -> str:
    """A path as it may be written down: relative to the working folder if inside it, else from `~`, else as is."""
    path = Path(path)
    if not path.is_absolute():
        return path.as_posix()
    resolved = path.resolve()
    for base, prefix in ((Path.cwd().resolve(), ""), (Path.home().resolve(), "~/")):
        if resolved.is_relative_to(base):
            relative = resolved.relative_to(base).as_posix()
            return f"{prefix}{relative}" if prefix else relative
    return path.as_posix()


def plain(value: Any) -> Any:
    """A configuration value as YAML and JSON can hold it."""
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {f.name: plain(getattr(value, f.name)) for f in dataclasses.fields(value)}
    if isinstance(value, Path):
        return shown_path(value)
    if isinstance(value, dict):
        return {str(key): plain(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [plain(item) for item in value]
    return value


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def versions() -> dict[str, str]:
    """The package's version, Python's, and the installed libraries'."""
    found = {"layout-visual-complexity": layout_visual_complexity.__version__, "python": platform.python_version()}
    for name in LIBRARIES:
        try:
            found[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            continue
    return found


def commit() -> str | None:
    """The repository's commit, with `-dirty` when the package's own files differ from it, if it runs from a clone.

    Only the package counts: a run's outputs may be tracked files that the run itself rewrites."""
    folder = Path(__file__).resolve().parent
    try:
        head = subprocess.run(
            ["git", "-C", str(folder), "rev-parse", "HEAD"], capture_output=True, text=True, timeout=10, check=True
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "-C", str(folder), "status", "--porcelain", "--untracked-files=no", "--", "."],
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None
    return f"{head}-dirty" if dirty else head


def now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


class Run:
    """Collects what a run read and wrote; `finish` writes its configuration and manifest."""

    def __init__(self, command: str, configuration: object) -> None:
        self.command = command
        self.configuration = plain(configuration)
        self.started = now()
        self.inputs: list[dict[str, Any]] = []
        self.outputs: list[str] = []
        self.summary: dict[str, Any] = {}

    def read(self, path: Path) -> Path:
        """Record an input file, with its checksum."""
        path = Path(path)
        self.inputs.append({"path": shown_path(path), "sha256": sha256(path), "bytes": path.stat().st_size})
        return path

    def wrote(self, paths: list[Path] | Path) -> None:
        for path in [paths] if isinstance(paths, Path) else paths:
            self.outputs.append(shown_path(path))

    def finish(self, folder: Path) -> list[Path]:
        """Write `config.yaml` and `manifest.json` into `folder`."""
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        config_path, manifest_path = folder / "config.yaml", folder / "manifest.json"
        config_path.write_text(yaml.safe_dump(self.configuration, sort_keys=False, allow_unicode=True), "utf-8")
        manifest = {
            "command": self.command,
            "started": self.started,
            "ended": now(),
            "inputs": self.inputs,
            "outputs": self.outputs,
            "configuration": self.configuration,
            "versions": versions(),
            "commit": commit(),
            "platform": platform.platform(terse=True),
            "summary": plain(self.summary),
        }
        manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False, default=str) + "\n", "utf-8")
        return [config_path, manifest_path]
