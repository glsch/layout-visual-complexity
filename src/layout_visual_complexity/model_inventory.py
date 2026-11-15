"""The `lvc models` command: what a scoring run uses, where each model is, and whether it is there and the one
expected.

- the regions model (YOLO), from the configured path, checked against the checksum of the one this study used;
- Kraken's segmentation model, its default as installed with Kraken, or the configured one;
- the ViT backbone at its pinned revision, in the Hugging Face cache, as the shipped head names it;
- the rankers: the ensemble and its members, each checked against the checksum the ensemble recorded.
"""

import os
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

from layout_visual_complexity.runs import sha256, shown_path
from layout_visual_complexity.score import ScoreSettings
from layout_visual_complexity.scoring import ENSEMBLE, ModelError, ScoringModels, shipped_folder

# The regions model this study used, by its file name: its SHA-256. It can't ship with the package (PyPI refuses
# files over 100 MB); it will be fetched from its archive once it is published.
KNOWN = {"medieval-yolov11x.pt": "deb69b38b3e360184e97d15801b7383754b1dc7f2559e4a51126559289d6533d"}
REGIONS_HINT = (
    "put medieval-yolov11x.pt in models/ (or give its path as regions.model); it will be downloadable once it is "
    "published"
)


@dataclass
class Item:
    """One model: what it is, where, whether it is there, its checksum, and what to do."""

    what: str
    where: str
    present: bool
    sha256: str | None = None
    matches: bool | None = None
    note: str = ""
    fetched: bool = False  # fetched on the first run when missing


def hub_cache() -> Path:
    """Where the Hugging Face hub keeps models: `HF_HUB_CACHE`, else `HF_HOME/hub`, else `~/.cache/huggingface/hub`."""
    if os.environ.get("HF_HUB_CACHE"):
        return Path(os.environ["HF_HUB_CACHE"]).expanduser()
    home = os.environ.get("HF_HOME")
    return (Path(home).expanduser() if home else Path("~/.cache/huggingface").expanduser()) / "hub"


def backbone_item(tokens: dict[str, Any]) -> Item:
    """The backbone's snapshot at its revision in the Hugging Face cache (a local folder is taken as it is)."""
    name, revision = str(tokens["backbone"]), tokens.get("revision")
    what = f"ViT backbone {name} at {revision or 'its latest revision'}"
    if Path(name).expanduser().is_dir():
        return Item(what, shown_path(Path(name).expanduser()), True, note="a local folder")
    folder = hub_cache() / f"models--{name.replace('/', '--')}" / "snapshots" / str(revision)
    weights = [folder / file for file in ("model.safetensors", "pytorch_model.bin") if (folder / file).exists()]
    present = (folder / "config.json").exists() and bool(weights)
    note = "" if present else "downloaded from Hugging Face on the first run (it needs the vit extra)"
    return Item(what, shown_path(folder), present, note=note, fetched=True)


def regions_item(path: Path) -> Item:
    path = Path(path)
    if not path.is_file():
        return Item("regions model (YOLO)", shown_path(path), False, note=REGIONS_HINT)
    found = sha256(path)
    expected = KNOWN.get(path.name)
    note = "" if expected else "not the model this study used: its scores aren't the article's"
    return Item("regions model (YOLO)", shown_path(path), True, found, found == expected if expected else None, note)


def lines_item(path: Path | None) -> Item:
    if path is not None:
        found = Path(path).is_file()
        return Item("lines model (Kraken)", shown_path(path), found, sha256(Path(path)) if found else None)
    try:
        engine = f"Kraken {version('kraken')}"
        from importlib import resources

        default = Path(str(resources.files("kraken").joinpath("blla.mlmodel")))
    except (PackageNotFoundError, ModuleNotFoundError):
        return Item("lines model (Kraken's default)", "kraken", False, note="install the segment extra")
    present = default.is_file()
    return Item("lines model (Kraken's default)", shown_path(default), present,
                sha256(default) if present else None, note=f"as installed with {engine}")  # fmt: skip


def ranker_items(folder: Path | None) -> list[Item]:
    folder = Path(folder) if folder is not None else shipped_folder()
    shipped = folder == shipped_folder()
    label = "the package's" if shipped else "given"
    path = folder / ENSEMBLE
    if not path.is_file():
        return [Item(f"rankers ({label})", shown_path(folder), False, note="no ensemble.json")]
    import json

    description = json.loads(path.read_text(encoding="utf-8"))
    items = [Item(f"ensemble ({label})", shown_path(path), True, sha256(path))]
    for name, member in description["members"].items():
        file = folder / member["file"]
        if not file.is_file():
            items.append(Item(f"ranker {name}", shown_path(file), False, note="missing"))
            continue
        found = sha256(file)
        items.append(Item(f"ranker {name}", shown_path(file), True, found, found == member["sha256"]))
    return items


def inventory(settings: ScoreSettings) -> list[Item]:
    """Every model the scoring run these settings describe would use."""
    items = [regions_item(settings.regions.model), lines_item(settings.lines.model)]
    try:
        models = ScoringModels.load(settings.models)
    except ModelError:
        models = None
    if models is not None:
        items.append(backbone_item(models.head.tokens))
    items += ranker_items(settings.models)
    return items


def run_models(settings: ScoreSettings) -> list[Item]:
    """List the models and print them, each with where it is, whether it is there and the one expected."""
    items = inventory(settings)
    for item in items:
        state = "present" if item.present else "missing"
        if item.matches is True:
            state += ", checksum matches"
        elif item.matches is False:
            state += ", checksum DIFFERS"
        checksum = f"\n    sha256 {item.sha256}" if item.sha256 else ""
        note = f"\n    {item.note}" if item.note else ""
        print(f"{item.what}: {state}\n    {item.where}{checksum}{note}")
    if any((not item.present and not item.fetched) or item.matches is False for item in items):
        raise RuntimeError("some models are missing or differ; see above")
    return items
