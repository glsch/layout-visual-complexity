"""The normalised folio, the ink layer and the line height, on synthetic pages with known answers."""

import math
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from layout_visual_complexity.features.folio_image import (
    InkLayerSettings,
    ink_layer,
    line_spacing,
    normalised_folio,
    paper_surface,
)
from layout_visual_complexity.features.page import Page, TextLine

WIDTH, HEIGHT, SPACING = 400, 600, 30


def box(x0: float, y0: float, x1: float, y1: float) -> np.ndarray:
    return np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=float)


def lit_page(tmp_path: Path, lines: bool = True) -> tuple[Page, np.ndarray]:
    """A page lit unevenly, its paper from 170 on the left to 245 on the right, with dark strokes 6 pixels high
    every 30 pixels; and where the strokes are."""
    shade = np.linspace(170, 245, WIDTH)[None, :, None]
    pixels = np.repeat(np.repeat(shade, HEIGHT, axis=0), 3, axis=2)
    strokes = np.zeros((HEIGHT, WIDTH), bool)
    baselines = []
    for y in range(60, HEIGHT - 60, SPACING):
        strokes[y - 6 : y, 50:350] = True
        baselines.append(TextLine(f"l{y}", box(50, y - 12, 350, y + 2), np.array([[50.0, y], [350.0, y]]), "r1"))
    pixels[strokes] = 0.35 * pixels[strokes]
    path = tmp_path / "page.png"
    Image.fromarray(np.clip(pixels, 0, 255).astype(np.uint8)).save(path)
    page = Page("ms_1r", WIDTH, HEIGHT, box(0, 0, WIDTH, HEIGHT), lines=baselines if lines else [], image_path=path)
    return page, strokes


def test_the_ink_layer_finds_dark_strokes_on_an_unevenly_lit_page(tmp_path: Path) -> None:
    page, strokes = lit_page(tmp_path)
    layer = ink_layer(page, InkLayerSettings(long_edge=HEIGHT))
    assert layer.ink.shape == strokes.shape
    assert layer.ink[strokes].mean() > 0.95
    assert layer.ink[~strokes].mean() < 0.01
    assert layer.coverage == pytest.approx(strokes.mean(), rel=0.05)
    left, right = layer.ink[:, :200], layer.ink[:, 200:]
    assert left[strokes[:, :200]].mean() == pytest.approx(right[strokes[:, 200:]].mean(), abs=0.03)


def test_flattening_evens_out_the_light(tmp_path: Path) -> None:
    page, strokes = lit_page(tmp_path)
    folio = normalised_folio(page, HEIGHT)
    paper = folio.lab[..., 0][~strokes]
    assert paper.std() < 1.0
    raw = np.asarray(Image.open(page.image_path).convert("L"), float)[~strokes]
    assert raw.std() > 15


def test_the_ink_layer_is_made_once_per_page_and_settings(tmp_path: Path) -> None:
    page, _ = lit_page(tmp_path)
    first = ink_layer(page, InkLayerSettings(long_edge=HEIGHT))
    assert ink_layer(page, InkLayerSettings(long_edge=HEIGHT)) is first
    assert ink_layer(page, InkLayerSettings(long_edge=300)) is not first


def test_line_height_from_the_text_lines(tmp_path: Path) -> None:
    page, _ = lit_page(tmp_path)
    layer = ink_layer(page, InkLayerSettings(long_edge=300))
    assert layer.line_height_source == "lines"
    assert layer.line_height == pytest.approx(SPACING * 300 / HEIGHT)


def test_line_height_from_the_inks_profile_without_lines(tmp_path: Path) -> None:
    page, _ = lit_page(tmp_path, lines=False)
    layer = ink_layer(page, InkLayerSettings(long_edge=HEIGHT))
    assert layer.line_height_source == "profile"
    assert layer.line_height == pytest.approx(SPACING, abs=1)


def test_line_spacing_skips_lines_that_dont_overlap_and_needs_three() -> None:
    column = [np.array([[0.0, y], [100.0, y]]) for y in (10, 50, 90, 130)]
    other = [np.array([[300.0, y], [400.0, y]]) for y in (12, 52, 92, 132)]
    assert line_spacing(column + other) == pytest.approx(40)
    assert math.isnan(line_spacing(column[:2]))


def test_a_flat_page_gives_a_flat_surface() -> None:
    lightness = np.full((64, 48), 80.0, np.float32)
    surface = paper_surface(lightness, np.ones_like(lightness, bool))
    assert np.allclose(surface, 80.0, atol=1e-3)
