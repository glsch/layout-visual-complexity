"""The feature contract and the machinery of `lvc features`: names, needs, the configuration, resume, workers."""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml
from PIL import Image

from layout_visual_complexity import cli
from layout_visual_complexity.evaluation.model import FeatureMismatch, TrainedModel, fit_pairwise_model
from layout_visual_complexity.features import contract
from layout_visual_complexity.features.contract import Family, FeatureError, Group, load_families, load_family
from layout_visual_complexity.features.extraction import (
    FeaturesSettings,
    family_names,
    measure_page,
    read_feature_table,
    run_features,
)
from layout_visual_complexity.features.page import Page
from layout_visual_complexity.segmentation.store import write_layout_rows, write_pages


@dataclass(frozen=True)
class DarknessSettings:
    factor: float = 1.0


class Darkness(Family):
    """A test family: the image's mean darkness, times a factor."""

    name = "darkness"
    Settings = DarknessSettings

    @property
    def needs(self):
        return frozenset({"image"})

    def groups(self) -> list[Group]:
        return [Group("Test", "Darkness", "Test__Darkness", ("mean",), "The image's mean darkness.")]

    def measure(self, page: Page) -> list[dict[str, float]]:
        return [{"Test__Darkness__mean": self.settings.factor * float(255 - page.image.mean())}]


class Brightness(Darkness):
    name = "brightness"

    def groups(self) -> list[Group]:
        return [Group("Test", "Brightness", "Test__Brightness", ("mean",), "The image's mean brightness.")]

    def measure(self, page: Page) -> list[dict[str, float]]:
        return [{"Test__Brightness__mean": float(page.image.mean())}]


class BadlyNamed(Darkness):
    def groups(self) -> list[Group]:
        return [Group("Test", "Bad", "test__bad", ("Mean",), "A badly named column.")]


def test_the_built_in_families_names_are_unique_and_well_formed() -> None:
    families = load_families(["geometry", "layout"])
    columns = [column for family in families for column in family.columns()]
    assert len(columns) == len(set(columns)) == 105
    assert all(contract.COLUMN.match(column) for column in columns)
    with pytest.raises(FeatureError, match="isn't named"):
        contract.check_columns([BadlyNamed()])
    with pytest.raises(FeatureError, match="comes from both"):
        contract.check_columns([Darkness(), Darkness()])


def test_families_are_picked_by_short_name_or_python_path_with_their_settings() -> None:
    assert family_names("geometry, layout") == ["geometry", "layout"] == family_names(["geometry,layout"])
    assert load_family("geometry").name == "geometry"
    found = load_family(f"{__name__}.Darkness", {"factor": 2})
    assert isinstance(found, Darkness) and found.settings.factor == 2
    layout = load_family("layout", {"flavours": ["mass"], "resolution": 128})
    assert layout.settings.flavours == ("mass",) and layout.settings.resolution == 128
    with pytest.raises(FeatureError, match="no setting"):
        load_family("geometry", {"min_line": 3})
    with pytest.raises(FeatureError, match="unknown feature family"):
        load_family("colour")
    with pytest.raises(FeatureError, match="not chosen"):
        load_families(["geometry"], {"layout": {"resolution": 64}})


def test_a_familys_hash_follows_its_settings() -> None:
    assert load_family("geometry").settings_hash() == load_family("geometry").settings_hash()
    assert load_family("geometry").settings_hash() != load_family("geometry", {"min_lines": 4}).settings_hash()


def test_what_families_need_is_prepared_once_per_page(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "1r.png"
    Image.fromarray(np.full((40, 30, 3), 200, np.uint8)).save(path)
    opened = []
    original = Image.open
    monkeypatch.setattr(Image, "open", lambda *a, **k: opened.append(a) or original(*a, **k))
    page = Page("ms_1r", 30, 40, image_path=path)
    values, seconds, errors = measure_page(page, [Darkness(), Brightness()])
    assert len(opened) == 1 and not errors
    assert values == {"Test__Darkness__mean": 55.0, "Test__Brightness__mean": 200.0}
    assert set(seconds) == {"prepare", "darkness", "brightness"}


def test_a_family_failing_on_a_page_leaves_its_columns_missing() -> None:
    values, _, errors = measure_page(Page("ms_1r", 30, 40), [Darkness()])
    assert np.isnan(values["Test__Darkness__mean"]) and set(errors) == {"darkness"}


def test_a_model_refuses_features_whose_settings_hash_differs(tmp_path: Path) -> None:
    pages = [f"p{n}" for n in range(6)]
    features = pd.DataFrame({"Test__Darkness__mean": np.arange(6.0)}, index=pages)
    judgments = pd.DataFrame({"left": pages[:5], "right": pages[1:], "chosen": pages[1:]})
    family = Darkness()
    model = TrainedModel(fit_pairwise_model(features, judgments), [family.describe()], 0.0, 1.0, 1.0, 1.0, {})
    model.check({"darkness": family.settings_hash()}, list(features.columns))
    with pytest.raises(FeatureMismatch, match="other settings"):
        model.scores(features, {"darkness": Darkness(DarknessSettings(2.0)).settings_hash()})
    with pytest.raises(FeatureMismatch, match="no family"):
        model.check({}, list(features.columns))
    with pytest.raises(FeatureMismatch, match="lack"):
        model.check({"darkness": family.settings_hash()}, [])
    assert TrainedModel.load(model.save(tmp_path / "model.json")).to_dict() == model.to_dict()


def box(x0: float, y0: float, x1: float, y1: float) -> list[list[int]]:
    return [[round(x0), round(y0)], [round(x1), round(y0)], [round(x1), round(y1)], [round(x0), round(y1)]]


def write_store(folder: Path, pages: int = 5) -> Path:
    """A segmentation run of one manuscript's pages, with their images in a folder its configuration names."""
    rng = np.random.default_rng(3)
    images = folder / "images" / "ms"
    images.mkdir(parents=True)
    rows, page_rows = [], []
    for number in range(1, pages + 1):
        page = f"ms_{number}r"
        pixels = np.full((300, 200, 3), 235, np.uint8)
        layout = [{"page": page, "kind": "folio", "class": "otsu", "polygon": box(5, 5, 195, 295), "id": "folio"}]
        layout.append({"page": page, "kind": "region", "class": "MainZone", "polygon": box(20, 20, 180, 280),
                       "id": "r1", "confidence": 0.9})  # fmt: skip
        for line in range(3 + number):
            y = 40 + 30 * line
            end = 180 - rng.integers(0, 60)
            pixels[y - 10 : y, 20:end] = 40
            layout.append({"page": page, "kind": "line", "class": "line", "polygon": box(20, y - 10, end, y + 2),
                           "baseline": [[20, y], [int(end), y + int(rng.integers(-2, 3))]], "region": "r1",
                           "id": f"l{line + 1}"})  # fmt: skip
        Image.fromarray(pixels).save(images / f"{number}r.png")
        rows += [{"baseline": None, "confidence": None, "region": None} | row for row in layout]
        page_rows.append({"page": page, "document": "ms", "width": 200, "height": 300, "status": "done"})
    write_layout_rows(folder / "layout" / "ms.parquet", rows)
    write_pages(folder / "pages.parquet", page_rows)
    (folder / "config.yaml").write_text(yaml.safe_dump({"sources": [str(images)], "cache": str(folder / "cache")}))
    return folder


def test_results_are_the_same_with_one_worker_and_with_many(tmp_path: Path) -> None:
    store = write_store(tmp_path / "segment")
    tables = []
    for jobs in (1, 3):
        settings = FeaturesSettings(store, tmp_path / f"features_{jobs}", ["geometry", "layout"], jobs=jobs, batch=2)
        summary = run_features(settings)
        assert summary["measured"] == 5 and not summary["failed"]
        tables.append(read_feature_table(tmp_path / f"features_{jobs}" / "features.parquet"))
    one, many = tables
    pd.testing.assert_frame_equal(one.values, many.values)
    assert list(one.values.index) == [f"ms_{n}r" for n in range(1, 6)]
    assert one.values.shape == (5, 105) and one.values.notna().all().all()
    assert one.columns["Layout__Symmetry__left_right_ink"] == "layout"
    assert {family["name"] for family in one.families} == {"geometry", "layout"}


def test_a_rerun_resumes_and_new_settings_measure_again(tmp_path: Path) -> None:
    store = write_store(tmp_path / "segment", pages=3)
    output = tmp_path / "features"
    assert run_features(FeaturesSettings(store, output, ["geometry"], jobs=1))["measured"] == 3
    assert run_features(FeaturesSettings(store, output, ["geometry"], jobs=1))["measured"] == 0
    changed = FeaturesSettings(store, output, ["geometry"], {"geometry": {"min_lines": 4}}, jobs=1)
    assert run_features(changed)["measured"] == 3
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["summary"]["families"][0]["settings"]["min_lines"] == 4
    assert "seconds_per_page" in manifest["summary"] and (output / "report.md").is_file()


def test_lvc_features_from_the_command_line(tmp_path: Path) -> None:
    store = write_store(tmp_path / "segment", pages=2)
    output = tmp_path / "features"
    arguments = ["--quiet", "features", "--layout", str(store), "--output", str(output), "--families", "layout"]
    assert cli.main([*arguments, "--jobs", "1", "--parameters", "{layout: {resolution: 64}}"]) == 0
    table = read_feature_table(output / "features.parquet")
    assert table.values.shape == (2, 22)
    assert table.families[0]["settings"]["resolution"] == 64
