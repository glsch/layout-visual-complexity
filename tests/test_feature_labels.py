"""Every feature a figure or table names has a readable label: its feature and flavour in words, not its code name."""

import pytest

from layout_visual_complexity.features.contract import load_family

SHOWN = {
    "LineFeature__LineLength__var_coef": "Line length, variation",
    "AreaFeature__LineSimilarity__iou_median": "Line similarity, median IoU",
    "LineFeature__InterlinearDistance__min_gap_std": "Interlinear distance, variation",
    "LineFeature__ZigzagScore__std_right": "Zigzag, right ends",
    "LineFeature__LineParallelism__pairwise_angle_diff_mean": "Line parallelism, mean angle difference",
    "Layout__Symmetry__top_bottom_mass": "Symmetry, top and bottom",
    "Layout__Regularity__vertical_ink": "Regularity, vertical",
    "Layout__BlankSpace__largest_empty": "Blank space, largest",
    "AreaFeature__WrittenSpaceRectangularity__outside_ratio": "Rectangularity, outside share",
    "Layout__Balance__text_block_mass": "Balance, text block",
    "AreaFeature__WrittenSpaceRatio__convexity": "Written space, convexity",
    "AreaFeature__DropCapitalArea__drop_capital_num_isolated_instances": "Drop capitals, separate",
    "Layout__Entropy__zone_types": "Entropy, zone types",
    "AreaFeature__GraphicZoneArea__graphic_area_ratio_std": "Graphic zones, area variation",
    "Layout__Clutter__zones": "Clutter, zones",
}


@pytest.mark.parametrize("name", ["geometry", "layout", "generic", "trivial"])
def test_every_column_has_a_readable_label(name: str) -> None:
    family = load_family(name)
    for column in family.columns():
        label = family.label(column)
        assert label and "__" not in label and "_" not in label, (column, label)
        assert label[0].isupper()


def test_the_features_figure_7_shows_are_labelled_as_its_caption_names_them() -> None:
    families = {"Layout": load_family("layout"), "AreaFeature": load_family("geometry")}
    families["LineFeature"] = families["AreaFeature"]
    assert {column: families[column.split("__")[0]].label(column) for column in SHOWN} == SHOWN


def test_flavours_apart_from_the_shared_ones_have_labels_of_their_own() -> None:
    """Only the six definitions of the interlinear distance, and the written mass and the ink of Layout's measures,
    share labels: a caption says which is shown."""
    geometry = load_family("geometry")
    shared = {"LineFeature__InterlinearDistance"}
    labels = [geometry.label(column) for column in geometry.columns() if column.rsplit("__", 1)[0] not in shared]
    assert len(labels) == len(set(labels))
    layout = load_family("layout")
    kept = [column for column in layout.columns() if not column.endswith("_ink")]
    assert len({layout.label(column) for column in kept}) == len(kept)
    generic = load_family("generic")
    assert [generic.label(column) for column in generic.columns()][:3] == [
        "Edge-orientation entropy", "MSSC, scale 1", "MSSC, scale 2"
    ]  # fmt: skip
    assert load_family("trivial").label("Trivial__InkCoverage__share") == "Ink coverage"
