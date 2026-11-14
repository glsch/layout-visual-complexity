"""The `lvc cache` command: the user cache's parts, where they are and how big, and clearing them by kind."""

import shutil
from dataclasses import dataclass, field
from pathlib import Path

from layout_visual_complexity.download import DEFAULT_CACHE
from layout_visual_complexity.runs import shown_path

# The cache's kinds, by folder: what each holds.
KINDS = {
    "images": "page images at the long edge asked for, bounded (images.cache_gb), the oldest leaving first",
    "info": "IIIF image services' descriptions (info.json) and image URLs' full sizes",
    "manifests": "IIIF manifests and other JSON fetched",
    "thumbnails": "small page images drawn in figures",
    "tokens": "ViT patch tokens for training and evaluating the head (lvc embed)",
    "compare": "pages scored by lvc compare",
    "records": "catalogue records fetched for the dataset's preparation",
    "zenodo": "the dataset fetched from Zenodo",
    "ultralytics": "the regions model's settings, kept by ultralytics",
}


@dataclass(frozen=True)
class CacheSettings:
    """The user cache: its parts by kind, where they are and how big; and clearing them.

    Args:
        cache: The cache's folder.
        clear: Kinds to clear (`images`, `manifests`, `tokens`, …), or `all`.
    """

    cache: Path = DEFAULT_CACHE
    clear: list[str] = field(default_factory=list)


def size(folder: Path) -> tuple[int, int]:
    """Files and bytes under a folder."""
    files = total = 0
    for path in folder.rglob("*"):
        if path.is_file():
            files += 1
            total += path.stat().st_size
    return files, total


def parts(cache: Path) -> list[tuple[str, Path, int, int, str]]:
    """Each kind the cache holds, known ones first: its name, folder, files, bytes and what it holds."""
    found = []
    others = (
        sorted(path.name for path in cache.iterdir() if path.is_dir() and path.name not in KINDS)
        if cache.is_dir()
        else []
    )
    for name in [*KINDS, *others]:
        folder = cache / name
        if folder.is_dir():
            files, total = size(folder)
            found.append((name, folder, files, total, KINDS.get(name, "")))
    return found


def run_cache(settings: CacheSettings) -> list[tuple[str, Path, int, int, str]]:
    """Clear the kinds asked for, then print the cache's parts with their sizes. Returns them."""
    cache = Path(settings.cache).expanduser()
    asked = set(settings.clear)
    present = {name for name, *_ in parts(cache)}
    if "all" in asked:
        asked = present
    unknown = sorted(asked - present - set(KINDS))
    if unknown:
        raise ValueError(f"no such kind in the cache: {', '.join(unknown)}; it holds {', '.join(sorted(present))}")
    for name in sorted(asked & present):
        shutil.rmtree(cache / name)
        print(f"cleared {name}")
    found = parts(cache)
    print(f"{shown_path(cache)}:")
    for name, _, files, total, what in found:
        print(f"  {name:12} {total / 2**20:10,.1f} MB {files:8,} files  {what}")
    print(f"  {'in all':12} {sum(item[3] for item in found) / 2**20:10,.1f} MB")
    return found
