"""`lvc report`: the geometry tables and figures from small synthetic evaluations, features and pages."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml
from PIL import Image

from layout_visual_complexity.evaluate import EvaluateSettings, run_evaluate
from layout_visual_complexity.evaluation.folds import FoldSettings
from layout_visual_complexity.features.contract import load_families
from layout_visual_complexity.features.extraction import write_feature_table
from layout_visual_complexity.ranking.dataset import read_dataset
from layout_visual_complexity.reports import ALL_OUTPUTS, ReportSettings, correlations, run_report
from layout_visual_complexity.segmentation.store import write_layout_rows, write_pages


def box(x0: int, y0: int, x1: int, y1: int) -> list[list[int]]:
    return [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]


def write_inputs(folder: Path, dataset_folder: Path) -> dict[str, Path]:
    """Features of the dataset's pages (random, one that follows the scores), the scores, two evaluations, and a
    segmentation run of one of the pages with its image."""
    dataset = read_dataset(dataset_folder)
    pages = dataset.images["image"].tolist()
    families = load_families(["geometry", "layout"])
    rng = np.random.default_rng(2)
    columns = [column for family in families for column in family.columns()]
    values = pd.DataFrame(rng.normal(size=(len(pages), len(columns))), index=pages, columns=columns)
    values["AreaFeature__LineSimilarity__dice_median"] = -np.arange(len(pages), dtype=float)
    features = write_feature_table(folder / "features.parquet", values, families)
    scores = folder / "scores.tsv"
    pd.DataFrame({"image": pages, "elo_mean": 1500 + 20 * np.arange(len(pages))}).to_csv(scores, sep="\t", index=False)
    evaluations = {}
    for name, chosen in (("geometry", "geometry"), ("geometry_layout", "geometry,layout")):
        evaluations[name] = folder / name
        folds = FoldSettings(2, 2, 3)
        settings = EvaluateSettings(dataset_folder, scores, folder / name, features, chosen, folds=folds)
        run_evaluate(settings)
    segment = folder / "segment"
    images = segment / "images" / "msa"
    images.mkdir(parents=True)
    pixels = np.full((600, 400, 3), 235, np.uint8)
    rows = [{"page": "msa_1r", "kind": "folio", "class": "otsu", "polygon": box(10, 10, 390, 590), "id": "folio"},
            {"page": "msa_1r", "kind": "region", "class": "MainZone", "polygon": box(40, 40, 360, 560), "id": "r1"},
            {"page": "msa_1r", "kind": "region", "class": "GraphicZone", "polygon": box(60, 400, 200, 500), "id": "r2"},
            {"page": "msa_1r", "kind": "region", "class": "DropCapitalZone", "polygon": box(45, 45, 75, 90),
             "id": "r3"}]  # fmt: skip
    for number in range(10):
        y = 70 + 32 * number
        end = 350 - 13 * (number % 4)
        pixels[y - 14 : y, 50:end] = 50
        baseline = [[50, y], [end, y + number % 3 - 1]]
        rows.append({"page": "msa_1r", "kind": "line", "class": "line", "polygon": box(50, y - 14, end, y + 3),
                     "baseline": baseline, "region": "r1", "id": f"l{number + 1}"})  # fmt: skip
    Image.fromarray(pixels).save(images / "1r.png")
    write_layout_rows(segment / "layout" / "msa.parquet", [{"baseline": None, "confidence": None, "region": None}
                                                          | row for row in rows])  # fmt: skip
    write_pages(segment / "pages.parquet", [{"page": "msa_1r", "document": "msa", "width": 400, "height": 600,
                                            "status": "done"}])  # fmt: skip
    configuration = {"sources": [str(images)], "cache": str(folder / "cache")}
    (segment / "config.yaml").write_text(yaml.safe_dump(configuration))
    return {"features": features, "scores": scores, "segment": segment, **evaluations}


def test_the_geometry_tables_and_figures(tmp_path: Path, dataset_folder: Path) -> None:
    inputs = write_inputs(tmp_path / "inputs", dataset_folder)
    output = tmp_path / "results"
    groups = ["AreaFeature__WrittenSpaceRatio", "AreaFeature__GraphicZoneArea", "LineFeature__ZigzagScore"]
    settings = ReportSettings(
        output,
        [output for output in ALL_OUTPUTS if output.startswith("geometry")],
        dataset=dataset_folder,
        scores=inputs["scores"],
        features=inputs["features"],
        layout=inputs["segment"],
        evaluations={"geometry": inputs["geometry"], "geometry_layout": inputs["geometry_layout"]},
        labels={"geometry": "Geometry", "geometry_layout": "Geometry + Layout"},
        pages=dict.fromkeys(groups, "msa_1r"),
        bootstrap=50,
        image_credit="Synthetic pages.",
    )
    summary = run_report(settings)
    for stem in ("geometry_by_elo_gap", "geometry_feature_hierarchy", "geometry_feature_correlations"):
        assert (output / "tables" / f"{stem}.tsv").is_file() and (output / "tables" / f"{stem}.tex").is_file()
    for stem in ("geometry_feature_correlations", "geometry_features_on_page", "geometry_features_on_schematic_page",
                 "geometry_feature_families"):  # fmt: skip
        for suffix in (".svg", ".png", ".tex"):
            assert (output / "figures" / f"{stem}{suffix}").is_file()

    by_gap = pd.read_csv(output / "tables" / "geometry_by_elo_gap.tsv", sep="\t")
    assert by_gap["gap"].tolist()[-2:] == ["overall", "srocc"]
    assert {"consistency", "geometry", "geometry_low", "geometry_layout_high"} <= set(by_gap.columns)
    hierarchy = pd.read_csv(output / "tables" / "geometry_feature_hierarchy.tsv", sep="\t")
    assert len(hierarchy) == 15 and hierarchy["flavours"].sum() == 105
    assert "83 + 22" in (output / "tables" / "geometry_feature_hierarchy.tex").read_text()
    found = pd.read_csv(output / "tables" / "geometry_feature_correlations.tsv", sep="\t")
    assert len(found) == 105 and found["best"].sum() == 15
    best = found.set_index("column").loc["AreaFeature__LineSimilarity__dice_median"]
    assert best["rho"] == -1 and best["best"]
    caption = (output / "figures" / "geometry_feature_correlations.tex").read_text()
    assert "the median Dice similarity of two line polygons centred on each other" in caption
    on_page = (output / "figures" / "geometry_features_on_page.tex").read_text()
    assert "msa\\_1r" in on_page and "Synthetic pages." in on_page
    assert summary["geometry_features_on_page"]["pages"]["LineFeature__ZigzagScore"] == "msa_1r"


def test_correlation_intervals_resample_manuscripts() -> None:
    pages = [f"m{n // 4}_{n}" for n in range(40)]
    elo = pd.Series(np.arange(40.0), index=pages)
    values = pd.DataFrame({"A__B__up": np.arange(40.0), "A__B__flat": np.ones(40)}, index=pages)
    manuscripts = pd.Series([page.split("_")[0] for page in pages], index=pages)
    found = correlations(values, elo, manuscripts, 100, 1).set_index("column")
    assert found.at["A__B__up", "rho"] == pytest.approx(1) and found.at["A__B__up", "low"] == pytest.approx(1)
    assert np.isnan(found.at["A__B__flat", "rho"])
