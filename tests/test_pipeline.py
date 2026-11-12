"""The pipeline with fake steps: no models, real worker processes."""

import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest

from layout_visual_complexity.download import ImageSettings
from layout_visual_complexity.segment import SegmentSettings, place, run_segment
from layout_visual_complexity.segmentation.fake import FakeFolioBox, FakeLines, FakeRegions, FakeSettings
from layout_visual_complexity.segmentation.folio import FolioSettings
from layout_visual_complexity.segmentation.steps import FolioStep, segmentation_steps
from layout_visual_complexity.segmentation.store import read_layout, read_pages


def pages(folder: Path, name: str, count: int) -> Path:
    """`count` synthetic pages: a bright leaf, a little askew, with writing, on a dark ground."""
    target = folder / name
    target.mkdir(parents=True, exist_ok=True)
    for number in range(1, count + 1):
        image = np.full((600, 450, 3), 30, np.uint8)
        corners = np.array([[40 + number, 30], [410, 35 + number], [405, 570], [45, 565]], np.int32)
        cv2.fillPoly(image, [corners], (230, 220, 200))
        for y in range(80, 520, 20 + number):
            cv2.line(image, (80, y), (370, y), (50, 40, 40), 4)
        cv2.imwrite(str(target / f"{number:03d}r.jpg"), image)
    return target


def steps(workers: int | None = 1, batch: int = 3, **fake: object) -> dict:
    """Fake YOLO and Kraken steps around the real folio crop, seeded by the fake folio box. `fake` settings go to every
    fake step but the lines."""
    settings = FakeSettings(workers=workers, batch_size=batch, **fake)  # type: ignore[arg-type]
    return {
        "folio_box": FakeFolioBox(settings),
        "folio": FolioStep(FolioSettings(workers=workers, batch_size=batch), seeded=True),
        "regions": FakeRegions(settings),
        "lines": FakeLines(FakeSettings(workers=workers, batch_size=batch)),
    }


@pytest.fixture
def sources(tmp_path: Path) -> list[str]:
    return [str(pages(tmp_path / "images", "ms_a", 5)), str(pages(tmp_path / "images", "ms_b", 4))]


def run(tmp_path: Path, name: str, sources: list[str], chosen: dict, **settings: object) -> dict:
    options = SegmentSettings(sources=sources, output=tmp_path / name, cache=tmp_path / "cache", jobs=2, **settings)  # type: ignore[arg-type]
    return run_segment(options, chosen)


def layout(folder: Path) -> dict[str, pd.DataFrame]:
    return {path.stem: read_layout(path) for path in sorted((folder / "layout").glob("*.parquet"))}


def outcome(folder: Path) -> pd.DataFrame:
    table = read_pages(folder / "pages.parquet")
    kept = [column for column in table.columns if not column.endswith("_seconds") and column != "source"]
    return table[kept].sort_values("page").reset_index(drop=True)


def test_the_same_results_with_one_worker_or_many_and_in_any_order(tmp_path: Path, sources: list[str]) -> None:
    one = run(tmp_path, "one", sources, steps(workers=1, batch=1))
    many = run(tmp_path, "many", list(reversed(sources)), steps(workers=3, batch=2))
    assert one["done"] == many["done"] == 9
    first, second = layout(tmp_path / "one"), layout(tmp_path / "many")
    assert list(first) == list(second) == ["msa", "msb"]
    for document in first:
        pd.testing.assert_frame_equal(first[document], second[document])
    pd.testing.assert_frame_equal(outcome(tmp_path / "one"), outcome(tmp_path / "many"))
    assert set(outcome(tmp_path / "one")["folio_method"]) == {"otsu"}


def test_without_yolos_box_the_folio_crop_runs_alone(tmp_path: Path, sources: list[str]) -> None:
    chosen = steps()
    del chosen["folio_box"]
    chosen["folio"] = FolioStep(FolioSettings(workers=1))
    summary = run(tmp_path, "out", sources, chosen)
    assert summary["done"] == 9 and not summary["folio_box_used"]
    table = outcome(tmp_path / "out")
    assert set(table["folio_method"]) == {"otsu"} and table["folio_iou"].isna().all()
    assert "folio_box" not in set(read_layout(tmp_path / "out" / "layout" / "msa.parquet")["kind"])
    report = (tmp_path / "out" / "report.md").read_text()
    assert "Pages where YOLO's box" not in report and "| folio_box |" not in report


def test_a_failing_page_fails_alone(tmp_path: Path, sources: list[str]) -> None:
    summary = run(tmp_path, "out", sources, steps(batch=3, fail_pages=["msa_2r"], raise_pages=["msb_3r"]))
    failed = dict(summary["failed"])
    assert set(failed) == {"msa_2r", "msb_3r"}
    assert failed["msb_3r"] == (
        "folio_box: RuntimeError: fake error on msb_3r; folio: folio_box failed;"
        " regions: RuntimeError: fake error on msb_3r"
    )
    assert summary["done"] == 7
    table = read_layout(tmp_path / "out" / "layout" / "msa.parquet")
    assert "msa_2r" not in set(table["page"]) and {"msa_1r", "msa_3r"} <= set(table["page"])


def test_a_worker_that_dies_takes_only_its_page_down(tmp_path: Path, sources: list[str]) -> None:
    summary = run(tmp_path, "out", sources, steps(batch=1, crash_pages=["msa_4r"]))
    assert [page for page, _ in summary["failed"]] == ["msa_4r"]
    assert "the worker stopped" in summary["failed"][0][1]
    assert summary["done"] == 8


def test_a_batch_too_big_for_its_device_is_halved(tmp_path: Path, sources: list[str]) -> None:
    summary = run(tmp_path, "out", sources, steps(batch=4, max_batch=1))
    assert summary["done"] == 9 and not summary["failed"]


def test_resume_skips_what_is_done_and_retries_what_failed(tmp_path: Path, sources: list[str]) -> None:
    marker = tmp_path / "failing"
    marker.touch()
    chosen = {"fail_pages": ["msb_2r"], "fail_while": str(marker)}
    first = run(tmp_path, "out", sources, steps(**chosen))
    assert first["done"] == 8 and [page for page, _ in first["failed"]] == ["msb_2r"]
    marker.unlink()
    second = run(tmp_path, "out", sources, steps(workers=2, **chosen))
    assert second["pages_this_run"] == 1 and second["done"] == 9 and not second["failed"]
    assert set(read_layout(tmp_path / "out" / "layout" / "msb.parquet")["page"]) == {f"msb_{n}r" for n in range(1, 5)}
    third = run(tmp_path, "out", sources, steps(**chosen))
    assert third["pages_this_run"] == 0 and third["done"] == 9
    other = run(tmp_path, "out", sources, steps(**chosen), images=ImageSettings(long_edge=400))
    assert other["pages_this_run"] == 9


def test_a_limit_takes_the_first_pages(tmp_path: Path, sources: list[str]) -> None:
    summary = run(tmp_path, "out", sources, steps(), limit=6)
    assert summary["pages"] == 6
    assert set(read_pages(tmp_path / "out" / "pages.parquet")["page"]) == {f"msa_{n}r" for n in range(1, 6)} | {
        "msb_1r"
    }


def test_alto_on_request(tmp_path: Path, sources: list[str]) -> None:
    run(tmp_path, "out", sources[:1], steps(), alto=True)
    files = sorted(path.name for path in (tmp_path / "out" / "alto" / "msa").iterdir())
    assert files == [f"msa_{n}r.xml" for n in range(1, 6)]
    report = (tmp_path / "out" / "report.md").read_text()
    assert "## Stages" in report and "## The folio crop" in report
    manifest = json.loads((tmp_path / "out" / "manifest.json").read_text())
    assert manifest["command"] == "segment"
    assert manifest["summary"]["steps"]["lines"] == {"version": "FakeLines 1", "device": "cpu", "workers": 1}


def test_workers_share_the_cpu_and_fit_in_memory() -> None:
    from layout_visual_complexity.segmentation.lines import LineSettings
    from layout_visual_complexity.segmentation.regions import FolioBoxSettings, RegionSettings

    chosen = segmentation_steps(FolioBoxSettings(), FolioSettings(), RegionSettings(), LineSettings())
    assert list(chosen) == ["folio", "regions", "line_maps", "lines"]
    assert chosen["folio"].needs == {"image"}
    seeded = segmentation_steps(
        FolioBoxSettings(model=Path("box.pt")), FolioSettings(), RegionSettings(), LineSettings()
    )
    assert list(seeded)[:2] == ["folio_box", "folio"] and seeded["folio"].needs == {"image", "folio_box"}
    on_cpu = {p.stage: (p.device, p.workers) for p in place(chosen, "cpu", 14, 64_000)}
    assert on_cpu["lines"][1] > on_cpu["regions"][1] >= 1 and all(device == "cpu" for device, _ in on_cpu.values())
    tight = {p.stage: p.workers for p in place(chosen, "cpu", 14, 12_000)}
    assert tight["lines"] < on_cpu["lines"][1]
    assert sum(tight[name] * chosen[name].memory("cpu") for name in tight) <= 12_000
