"""The methods' names as the reproduction's figures and tables show them, and as their captions define them."""

import re
from pathlib import Path

import pytest
import yaml

from layout_visual_complexity.method_reports import Evaluations, family_columns, in_sentence

CONFIGS = Path(__file__).resolve().parents[1] / "reproduce" / "configs"
OLD_NAMES = re.compile(r"as published|Geometry with|Geometry \+|[Rr]eal colour|Layout family|^Geometry\b")


def labels(name: str) -> dict[str, str]:
    return yaml.safe_load((CONFIGS / f"{name}.yaml").read_text(encoding="utf-8")).get("labels") or {}


def test_no_label_table_holds_an_old_name() -> None:
    found = {path.stem: labels(path.stem) for path in CONFIGS.glob("*.yaml")}
    assert sum(len(table) for table in found.values()) > 100
    old = [(name, label) for name, table in found.items() for label in table.values() if OLD_NAMES.search(label)]
    assert old == []


def test_the_figures_name_the_methods_as_decided() -> None:
    assert labels("figure_accuracy_by_elo_gap") == {
        "line_texture": "Line texture",
        "colour": "Ink colour",
        "geometry": "Layout (geometry)",
        "vit": "ViT",
        "ensemble": "Ensemble",
    }
    information = labels("figure_baselines_by_information")
    assert information["kyle_davidson"] == "Kyle–Davidson" and information["line_texture"] == "Line texture"
    assert information["geometry_layout"] == "Layout (geometry with Kyle–Davidson)"
    assert information["ensemble"] == "Ensemble: Layout (geometry) and ViT"
    within = labels("table_within_and_across_manuscripts")
    assert within["ensemble_layout"] == "Ensemble with Layout (Kyle–Davidson)"


def test_two_families_are_defined_by_their_columns_each() -> None:
    summary = {
        "families": ["geometry", "layout"],
        "columns": 105,
        "measures": [
            {"prefix": "AreaFeature__WrittenSpaceRatio", "columns": 83},
            {"prefix": "Layout__Clutter", "columns": 22},
        ],
    }
    assert family_columns(summary, ["geometry", "layout"]) == {"geometry": 83, "layout": 22}
    evaluations = object.__new__(Evaluations)
    defined = evaluations.define({"summary": summary, "configuration": {}})
    assert defined == (
        "the pairwise logistic model on the 83 geometric features and the 22 of Layout (Kyle--Davidson) (105 columns)"
    )
    assert family_columns(summary | {"columns": 100}, ["geometry", "layout"]) == {}


@pytest.mark.parametrize(
    ("name", "within"),
    [("Image only", "image only"), ("Kyle–Davidson", "Kyle–Davidson"), ("MSSC, scale 1", "MSSC, scale 1"),
     ("Line similarity, median IoU", "line similarity, median IoU")],
)  # fmt: skip
def test_names_within_a_sentence_keep_their_capitals_only_where_they_are_names(name: str, within: str) -> None:
    assert in_sentence(name) == within
