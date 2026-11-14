"""Line texture as the article publishes it, on synthetic line crops and pages with known answers."""

import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from layout_visual_complexity.features import line_texture
from layout_visual_complexity.features.contract import aggregate
from layout_visual_complexity.features.extraction import FeaturesSettings, read_feature_table, run_features
from layout_visual_complexity.features.line_texture import LineTexture, LineTextureSettings
from layout_visual_complexity.features.page import Page, TextLine
from test_features_contract import write_store

# Appendix B's 38 measures, in the order the article's columns have them.
ARTICLE_ORDER = [
    "superpixels_count",
    "superpixels_area_mean",
    "superpixels_area_std",
    "superpixels_area_min",
    "superpixels_area_max",
    "superpixels_area_entropy",
    "superpixels_centroid_x_mean",
    "superpixels_centroid_x_std",
    "superpixels_centroid_y_mean",
    "superpixels_centroid_y_std",
    "superpixels_radius_mean",
    "superpixels_radius_std",
    "superpixels_weighted_centroid_x",
    "superpixels_weighted_centroid_y",
    "superpixels_channel_r_mean",
    "superpixels_channel_r_std",
    "superpixels_channel_r_min",
    "superpixels_channel_r_max",
    "superpixels_channel_r_weighted_mean",
    "superpixels_channel_r_weighted_std",
    "superpixels_channel_g_mean",
    "superpixels_channel_g_std",
    "superpixels_channel_g_min",
    "superpixels_channel_g_max",
    "superpixels_channel_g_weighted_mean",
    "superpixels_channel_g_weighted_std",
    "superpixels_channel_b_mean",
    "superpixels_channel_b_std",
    "superpixels_channel_b_min",
    "superpixels_channel_b_max",
    "superpixels_channel_b_weighted_mean",
    "superpixels_channel_b_weighted_std",
    "superpixels_corr_x_r",
    "superpixels_corr_x_g",
    "superpixels_corr_x_b",
    "superpixels_corr_y_r",
    "superpixels_corr_y_g",
    "superpixels_corr_y_b",
]


def box(x0: float, y0: float, x1: float, y1: float) -> np.ndarray:
    return np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=float)


def written_crop(seed: int = 0, height: int = 40, width: int = 300) -> np.ndarray:
    """A line of dark strokes on parchment, with some noise."""
    rng = np.random.default_rng(seed)
    crop = np.full((height, width, 3), (230, 220, 195), np.uint8)
    for x in range(10, width - 10, 14):
        crop[12:30, x : x + 4] = (40, 35, 30)
    noise = rng.integers(-6, 7, crop.shape)
    return np.clip(crop.astype(int) + noise, 0, 255).astype(np.uint8)


def test_the_38_columns_are_named_as_the_article_names_them() -> None:
    family = LineTexture()
    assert len(ARTICLE_ORDER) == 38
    assert family.columns() == [f"LineTextureFeature__LineSuperpixels__{key}" for key in ARTICLE_ORDER]
    assert sum(len(keys) for _, keys, _ in line_texture.TEXTURE_FAMILIES) == 38
    assert [len(keys) for _, keys, _ in line_texture.TEXTURE_FAMILIES] == [1, 5, 8, 18, 6]
    assert family.scope == "line" and family.aggregation(family.columns()[0]) == "mean"


def test_gamma_truncates_to_whole_grey_levels() -> None:
    table = line_texture.gamma_table(1.5)
    exact = 255 * (np.arange(256) / 255) ** 1.5
    assert table.dtype == np.uint8 and table[0] == 0 and table[255] == 255
    assert (table == np.floor(exact + 1e-9)).all()
    assert table[128] == 90 and exact[128] == pytest.approx(90.68, abs=0.01)  # truncated, not rounded to 91


def test_a_flat_crop_gives_zero_responses_and_correlations_of_zero() -> None:
    flat = np.full((30, 200, 3), 180, np.uint8)
    responses = line_texture.vertical_sobel(flat, 5)
    assert responses.dtype == np.float32 and not responses.any()
    values = line_texture.line_measures(flat, LineTextureSettings())
    assert values is not None and list(values) == ARTICLE_ORDER
    for channel in "rgb":
        assert values[f"superpixels_channel_{channel}_mean"] == 0
        assert values[f"superpixels_channel_{channel}_weighted_std"] == 0
        for axis in "xy":
            assert values[f"superpixels_corr_{axis}_{channel}"] == 0


def test_slic_gives_about_the_superpixels_asked_for() -> None:
    """At most the 50 asked for where SLIC's regular grid fits the crop; on long thin crops its grid's step, rounded
    to whole pixels, can place more centres (60 on a 40 × 300 crop), as in the earlier code."""
    square = line_texture.line_measures(written_crop(0, height=100, width=100), LineTextureSettings())
    assert square is not None and 2 <= square["superpixels_count"] <= 50
    for seed, width in ((0, 300), (1, 800), (2, 120)):
        values = line_texture.line_measures(written_crop(seed, width=width), LineTextureSettings())
        assert values is not None
        assert 2 <= values["superpixels_count"] <= 64
        assert values["superpixels_area_mean"] == pytest.approx(1 / values["superpixels_count"], rel=1e-5)


def test_the_measures_of_known_superpixels() -> None:
    labels = np.zeros((4, 6), np.int32)
    labels[:, 3:] = 1
    image = np.zeros((4, 6, 3), np.float32)
    image[:, 3:, 0] = 2.0
    image[0, 0, 1] = 4.0
    values = line_texture.superpixel_measures(image, labels)
    assert values is not None
    assert values["superpixels_count"] == 2
    assert values["superpixels_area_mean"] == pytest.approx(0.5)
    assert values["superpixels_area_entropy"] == pytest.approx(1.0)
    assert values["superpixels_centroid_x_mean"] == pytest.approx((1 / 5 + 4 / 5) / 2)
    assert values["superpixels_weighted_centroid_x"] == pytest.approx(0.5 * 1 + 0.5 * 4)
    assert values["superpixels_channel_r_mean"] == pytest.approx(1.0)
    assert values["superpixels_channel_r_weighted_std"] == pytest.approx(1.0)
    assert values["superpixels_channel_g_max"] == pytest.approx(4.0)
    assert values["superpixels_corr_x_r"] == pytest.approx(1.0)
    assert values["superpixels_corr_y_r"] == 0


def lined_page(tmp_path: Path, lines: list[np.ndarray]) -> Page:
    """A 600 × 400 page of strokes whose text lines have the polygons given."""
    pixels = np.full((400, 600, 3), (230, 220, 195), np.uint8)
    for y in range(40, 380, 40):
        for x in range(40, 560, 12):
            pixels[y : y + 14, x : x + 3] = (40, 35, 30)
    path = tmp_path / "page.png"
    Image.fromarray(pixels).save(path)
    text_lines = [
        TextLine(f"l{number}", polygon, np.array([[polygon[0, 0], polygon[-1, 1]], [polygon[1, 0], polygon[-1, 1]]]),
                 None)
        for number, polygon in enumerate(lines)
    ]  # fmt: skip
    return Page("ms_1r", 600, 400, box(0, 0, 600, 400), lines=text_lines, image_path=path)


def test_the_pages_value_is_the_mean_over_its_lines(tmp_path: Path) -> None:
    polygons = [box(30, 34, 570, 58), box(30, 74, 400, 98), box(100, 114, 570, 138)]
    page = lined_page(tmp_path, polygons)
    family = LineTexture()
    rows = family.measure(page)
    assert len(rows) == 3
    values = aggregate(family, rows)
    for column in family.columns():
        assert values[column] == pytest.approx(np.mean([row[column] for row in rows]))
    table = line_texture.gamma_table(1.5)
    first = line_texture.line_measures(table[page.image[34:58, 30:570]], LineTextureSettings())
    assert rows[0] == {f"{line_texture.PREFIX}__{key}": value for key, value in first.items()}


def test_a_page_without_a_valid_line_has_no_values(tmp_path: Path) -> None:
    family = LineTexture()
    for lines in ([], [np.array([[10.0, 10.0], [20.0, 10.0]])], [box(700, 500, 800, 520)]):
        page = lined_page(tmp_path, lines)
        assert family.measure(page) == []
        assert all(math.isnan(value) for value in aggregate(family, family.measure(page)).values())


def test_the_same_values_with_one_worker_and_with_many(tmp_path: Path) -> None:
    store = write_store(tmp_path / "segment", pages=4)
    tables = []
    for jobs in (1, 3):
        settings = FeaturesSettings(store, tmp_path / f"features_{jobs}", ["line_texture"], jobs=jobs, batch=1)
        assert not run_features(settings)["failed"]
        tables.append(read_feature_table(tmp_path / f"features_{jobs}" / "features.parquet").values)
    pd.testing.assert_frame_equal(*tables)
    assert tables[0].shape == (4, 38) and tables[0].notna().all().all()
