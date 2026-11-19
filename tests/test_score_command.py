"""`lvc score` and `lvc compare` on synthetic pages: fake YOLO and Kraken steps around the real folio crop, the real
feature families and the real ViT head stage on a tiny random backbone, on the CPU."""

import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest

from layout_visual_complexity.compare import CompareSettings, run_compare
from layout_visual_complexity.evaluation.model import PairwiseModel, TrainedModel
from layout_visual_complexity.evaluation.vit_head import HeadSettings, HeadWeights, TrainedHead
from layout_visual_complexity.features.contract import load_families
from layout_visual_complexity.manuscript_reports import (
    CollectionSettings,
    load_collection,
    make,
    pages_sentence,
    scale_sentence,
)
from layout_visual_complexity.score import ScoreSettings, read_scores, run_score
from layout_visual_complexity.score_steps import FeaturesStep, FeatureStepSettings, TokensStep, TokenStepSettings
from layout_visual_complexity.scoring import ScoringModels
from layout_visual_complexity.segmentation.fake import FakeFolioBox, FakeLines, FakeRegions, FakeSettings
from layout_visual_complexity.segmentation.folio import FolioSettings
from layout_visual_complexity.segmentation.steps import FolioStep

pytest.importorskip("torch")


def pages(folder: Path, count: int, shift: int = 0) -> Path:
    """`count` synthetic pages: a bright leaf with writing, its lines further apart page by page."""
    folder.mkdir(parents=True, exist_ok=True)
    for number in range(1, count + 1):
        image = np.full((600, 450, 3), 30, np.uint8)
        corners = np.array([[40 + number, 30], [410, 35 + number], [405, 570], [45, 565]], np.int32)
        cv2.fillPoly(image, [corners], (230, 220, 200))
        for y in range(80, 520, 14 + 3 * number + shift):
            cv2.line(image, (80 + 5 * number, y), (370, y + number), (50, 40, 40), 3)
        cv2.imwrite(str(folder / f"{number:03d}r.jpg"), image)
    return folder


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rankers(folder: Path, backbone: Path) -> Path:
    """Made-up rankers: a pairwise model on the geometric features and a head on the tiny backbone's tokens."""
    folder.mkdir(parents=True, exist_ok=True)
    (family,) = load_families(["geometry"], {})
    columns = family.columns()
    rng = np.random.default_rng(0)
    model = PairwiseModel(columns, np.zeros(len(columns)), np.full(len(columns), 10.0), np.zeros(len(columns)),
                          rng.normal(size=len(columns)))  # fmt: skip
    TrainedModel(model, [family.describe()], 0.0, 1.0, 1.0, 1.0, {}).save(folder / "geometry.json")
    tokens = {"backbone": str(backbone), "revision": None, "block": 2, "long_edge": 56, "crop": "folio",
              "patch_size": 14, "dim": 16}  # fmt: skip
    weights = HeadWeights(rng.normal(size=(16, 3)), rng.normal(size=3), rng.normal(size=3), 0.0)
    TrainedHead(weights, tokens, asdict(HeadSettings()), 1.0, 0.5, 2.0, {}).save(folder / "head.npz")
    members = {
        name: {"file": file, "sha256": sha256(folder / file), "weight": 0.5, "scale_training": 1.5, "scale": 1.2}
        for name, file in (("geometry", "geometry.json"), ("vit", "head.npz"))
    }
    description = {
        "model": "ensemble",
        "members": members,
        "combined": {"mean": 0.0, "sd": 0.8},
        "scale": 1.1,
        "scale_training": 1.3,
        "elo": {
            "intercept": 1500.0,
            "slope": 160.0,
            "correlation": 0.8,
            "elo_mean": 1500.0,
            "elo_sd": 200.0,
            "score_sd": 1.0,
            "pages": 10,
        },  # fmt: skip
        "trained_on": {"pages": 10, "judgments": 50, "left_out": []},
    }
    (folder / "ensemble.json").write_text(json.dumps(description))
    return folder


def steps(models: ScoringModels, workers: int) -> dict:
    fake = FakeSettings(workers=workers, batch_size=2)
    features = FeatureStepSettings(families=["geometry"], scored=["geometry"], workers=workers)
    tokens = TokenStepSettings(models.files()["vit"], local_only=True, workers=workers)
    return {
        "folio_box": FakeFolioBox(fake),
        "folio": FolioStep(FolioSettings(workers=workers, batch_size=2), seeded=True),
        "regions": FakeRegions(fake),
        "lines": FakeLines(fake),
        "features": FeaturesStep(features),
        "tokens": TokensStep(tokens),
    }


@pytest.fixture
def scoring(tmp_path: Path, tiny_vit: Path) -> tuple[ScoringModels, list[str]]:
    sources = [str(pages(tmp_path / "images" / "ms_a", 4)), str(pages(tmp_path / "images" / "ms_b", 3, shift=6))]
    return ScoringModels.load(rankers(tmp_path / "rankers", tiny_vit)), sources


def score(tmp_path: Path, name: str, sources: list[str], models: ScoringModels, workers: int, **more: object) -> dict:
    settings = ScoreSettings(sources=sources, output=tmp_path / name, cache=tmp_path / "cache", jobs=3, device="cpu",
                             **more)  # type: ignore[arg-type]  # fmt: skip
    return run_score(settings, steps(models, workers), models=models)


def test_scores_dont_depend_on_the_workers_or_the_order_and_a_rerun_resumes(
    tmp_path: Path, scoring: tuple[ScoringModels, list[str]]
) -> None:
    models, sources = scoring
    one = score(tmp_path, "one", sources, models, 1, maps=True)
    many = score(tmp_path, "many", list(reversed(sources)), models, 2, maps=True)
    assert one["done"] == many["done"] == 7 and not one["failed"]
    first = read_scores(tmp_path / "one").sort_values("page").reset_index(drop=True)
    second = read_scores(tmp_path / "many").sort_values("page").reset_index(drop=True)
    pd.testing.assert_frame_equal(first, second)
    assert list(first.columns) == ["page", "document", "label", "canvas", "order", "geometry", "vit", "ensemble",
                                   "expected_elo"]  # fmt: skip
    assert first[["geometry", "vit", "ensemble", "expected_elo"]].notna().all().all()
    np.testing.assert_allclose(first["expected_elo"], 1500 + 160 * first["ensemble"])
    again = score(tmp_path, "one", sources, models, 2, maps=True)
    assert again["pages_this_run"] == 0
    pd.testing.assert_frame_equal(read_scores(tmp_path / "one").sort_values("page").reset_index(drop=True), first)


def test_other_rankers_score_the_same_pages_beside_the_runs(
    tmp_path: Path, scoring: tuple[ScoringModels, list[str]]
) -> None:
    models, sources = scoring
    settings = ScoreSettings(sources=sources, output=tmp_path / "run", cache=tmp_path / "cache", jobs=3, device="cpu",
                             other_models={"same": models.folder})  # fmt: skip
    chosen = steps(models, 1)
    chosen["tokens"] = TokensStep(TokenStepSettings(models.files()["vit"], [models.files()["vit"]], local_only=True))
    run_score(settings, chosen, models=models)
    found = pd.read_csv(tmp_path / "run" / "scores_same.tsv", sep="\t")
    pd.testing.assert_frame_equal(found, pd.read_csv(tmp_path / "run" / "scores.tsv", sep="\t"))


def test_a_score_run_keeps_no_image_and_no_token(tmp_path: Path, scoring: tuple[ScoringModels, list[str]]) -> None:
    models, sources = scoring
    score(tmp_path, "run", sources, models, 1, maps=True)
    folder = tmp_path / "run"
    kept = [path for path in folder.rglob("*") if path.is_file()]
    assert not [path for path in kept if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".tif", ".tiff"}]
    assert not list((folder / ".maps").glob("*"))
    assert not (tmp_path / "cache" / "tokens").exists()
    maps = sorted((folder / "maps").glob("*.npz"))
    assert [path.stem for path in maps] == ["msa", "msb"]
    with np.load(maps[0]) as stored:
        grids = [stored[name] for name in stored.files if not name.endswith(".box")]
        assert grids and all(grid.dtype == np.float16 and grid.ndim == 2 for grid in grids)
        assert all(16 not in grid.shape for grid in grids)  # patch scores, never the 16-wide tokens
    for name in ("scores.tsv", "features.parquet", "pages.parquet", "manifest.json", "config.yaml", "report.md"):
        assert (folder / name).is_file()
    manifest = json.loads((folder / "manifest.json").read_text())
    read = {Path(item["path"]).name: item["sha256"] for item in manifest["inputs"]}
    assert read["ensemble.json"] and read["geometry.json"] and read["head.npz"]


def test_profiles_and_the_overview_are_drawn_from_a_score_run(
    tmp_path: Path, scoring: tuple[ScoringModels, list[str]]
) -> None:
    models, sources = scoring
    score(tmp_path, "run", sources, models, 1, profiles=True)
    figures = tmp_path / "run" / "figures"
    assert sorted(path.name for path in figures.glob("*.svg")) == [
        "manuscript_profile_msa.svg", "manuscript_profile_msb.svg", "manuscripts_overview.svg"
    ]  # fmt: skip
    collection = load_collection(tmp_path / "run", None, 50, 1)
    settings = CollectionSettings(tmp_path / "run", bootstrap=50, cache=tmp_path / "cache")
    for output in ("manuscript_strips", "manuscript_overview", "manuscripts_table"):
        assert make(output, settings, collection, tmp_path / "report").paths


def test_the_profiles_captions_open_with_the_manuscript_as_catalogued(
    tmp_path: Path, scoring: tuple[ScoringModels, list[str]]
) -> None:
    models, sources = scoring
    score(tmp_path, "run", sources, models, 1)
    catalogue = tmp_path / "catalogue.tsv"
    catalogue.write_text(
        "# Two manuscripts as catalogued.\nmanuscript\tshelfmark\tcatalogue_number\tdate\tfolios\tmaterial\ttitle\n"
        "msa\tMS A\tK1\tXV in. + XV2\t12\tpaper\t[Textus varii]\nmsb\tMS B\tK2\t1463\t8\tparchment\t\n",
        encoding="utf-8",
    )
    collection = load_collection(tmp_path / "run", catalogue, 50, 1)
    settings = CollectionSettings(tmp_path / "run", catalogue=catalogue, bootstrap=50, cache=tmp_path / "cache",
                                  catalogue_credit="Catalogue facts from a catalogue.",
                                  title_credit="the title from the record", repository="Here, A Library")  # fmt: skip
    make("manuscript_profiles", settings, collection, tmp_path / "report")
    figures = tmp_path / "report" / "figures"
    first = (figures / "manuscript_profile_msa.tex").read_text(encoding="utf-8")
    assert (
        "\\caption{Here, A Library, MS A, paper, 12 leaves, early 15th century and second half of the 15th century: "
        "[Textus varii]. Its " in first
    )
    assert "Catalogue facts from a catalogue, no. K1; the title from the record." in first
    second = (figures / "manuscript_profile_msb.tex").read_text(encoding="utf-8")
    assert "\\caption{Here, A Library, MS B, parchment, 8 leaves, 1463. Its " in second
    assert "Catalogue facts from a catalogue, no. K2." in second and "title from" not in second


def test_pages_with_too_few_lines_are_left_out_and_counted_in_the_captions(
    tmp_path: Path, scoring: tuple[ScoringModels, list[str]]
) -> None:
    models, sources = scoring
    score(tmp_path, "run", sources, models, 1)
    store = tmp_path / "run" / "pages.parquet"
    pages = pd.read_parquet(store)
    pages.loc[pages["page"].isin(["msa_2r", "msb_1r"]), "lines"] = 2  # the fake Kraken finds six on every page
    pages.to_parquet(store)
    kept = load_collection(tmp_path / "run", None, 50, 1, min_lines=5)
    assert len(kept.scores) == 5 and not kept.scores["page"].isin(["msa_2r", "msb_1r"]).any()
    assert sum(kept.blank.values()) == 2 and set(kept.blank.values()) == {1}
    assert "The 2 leaves on which Kraken finds fewer than 5 text lines" in pages_sentence(kept)
    document = kept.scores.loc[kept.scores["page"].str.startswith("msa"), "document"].iloc[0]
    assert "Its 1 leaf on which Kraken finds fewer than 5" in pages_sentence(kept, document)
    assert "left out" not in pages_sentence(load_collection(tmp_path / "run", None, 50, 1, min_lines=2))
    assert "Layout (geometry) and ViT rankers" in scale_sentence(kept)


def test_compare_gives_complementary_probabilities_in_either_order(
    tmp_path: Path, scoring: tuple[ScoringModels, list[str]]
) -> None:
    models, sources = scoring
    first, second = str(Path(sources[0]) / "001r.jpg"), str(Path(sources[1]) / "003r.jpg")
    shared = {"output": tmp_path / "compare", "cache": tmp_path / "cache", "jobs": 3, "device": "cpu"}
    forward = run_compare(CompareSettings(first=first, second=second, **shared), steps(models, 1), models=models)  # type: ignore[arg-type]
    backward = run_compare(CompareSettings(first=second, second=first, **shared), steps(models, 1), models=models)  # type: ignore[arg-type]
    for name in ("geometry", "vit", "ensemble"):
        assert forward["probabilities"][name] + backward["probabilities"][name] == pytest.approx(1.0)
    with pytest.raises(ValueError, match="same page"):
        run_compare(CompareSettings(first=first, second=first, **shared), steps(models, 1), models=models)  # type: ignore[arg-type]
