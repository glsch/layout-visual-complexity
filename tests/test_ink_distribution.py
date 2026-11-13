"""Ink distribution, on synthetic pages and patterns with known answers."""

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from layout_visual_complexity.features import ink
from layout_visual_complexity.features.contract import aggregate
from layout_visual_complexity.features.ink import InkDistribution, InkSettings
from layout_visual_complexity.features.page import Page, TextLine
from layout_visual_complexity.features.trivial import Trivial, TrivialSettings


def box(x0: float, y0: float, x1: float, y1: float) -> np.ndarray:
    return np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=float)


def written_page(tmp_path: Path, red: bool = True) -> Page:
    """A parchment-coloured 400 × 600 page with black strokes (a tenth of the folio) every 30 pixels and, at the
    top, a red stroke."""
    pixels = np.full((600, 400, 3), (232, 222, 196), np.uint8)
    lines = []
    for y in range(90, 570, 30):
        pixels[y - 6 : y, 50:350] = (35, 30, 25)
        lines.append(TextLine(f"l{y}", box(50, y - 12, 350, y + 2), np.array([[50.0, y], [350.0, y]]), "r1"))
    if red:
        pixels[54:60, 50:350] = (190, 35, 30)
    path = tmp_path / "page.png"
    Image.fromarray(pixels).save(path)
    return Page("ms_1r", 400, 600, box(0, 0, 400, 600), lines=lines, image_path=path)


def measure(page: Page, **settings: object) -> dict[str, float]:
    family = InkDistribution(InkSettings(long_edge=600, **settings))
    return aggregate(family, family.measure(page))


def test_coverage_of_a_known_pattern(tmp_path: Path) -> None:
    page = written_page(tmp_path, red=False)
    values = measure(page)
    assert values["Ink__Coverage__share"] == pytest.approx(16 * 6 * 300 / (400 * 600), rel=0.05)
    trivial = Trivial(TrivialSettings(long_edge=600))
    found = aggregate(trivial, trivial.measure(page))
    assert found["Trivial__InkCoverage__share"] == values["Ink__Coverage__share"]
    assert found["Trivial__TextLines__count"] == 16
    assert values["Ink__LineHeight__from_lines"] == 1
    assert values["Ink__LineHeight__folio_share"] == pytest.approx(30 / 600)


def test_a_uniform_pattern_has_a_lacunarity_of_one() -> None:
    inside = np.ones((90, 90), bool)
    assert ink.lacunarity(np.ones((90, 90), bool), inside, 10) == pytest.approx(1)
    checkerboard = (np.indices((90, 90)).sum(axis=0) % 2).astype(bool)
    assert ink.lacunarity(checkerboard, inside, 10) == pytest.approx(1)
    clumped = np.zeros((90, 90), bool)
    clumped[:30, :30] = True
    assert ink.lacunarity(clumped, inside, 10) > 3
    assert np.isnan(ink.lacunarity(clumped, inside, 200))


def test_fragmentation_counts_known_masses() -> None:
    inside = np.ones((200, 200), bool)
    masses = np.zeros((200, 200), bool)
    for x, y in ((20, 20), (120, 30), (60, 150)):
        masses[y : y + 10, x : x + 10] = True
        masses[y + 12 : y + 20, x : x + 10] = True  # split by a gap the closing bridges
    assert ink.fragmentation(masses, inside, 10) == pytest.approx(3 / (200 * 200 / 10**2))


def test_a_red_stroke_is_chromatic_ink_of_the_red_class(tmp_path: Path) -> None:
    values = measure(written_page(tmp_path))
    red_share = 6 * 300 / (17 * 6 * 300)
    assert values["Ink__ChromaticShare__share"] == pytest.approx(red_share, rel=0.15)
    assert values["Ink__HueClasses__red"] == pytest.approx(values["Ink__ChromaticShare__share"])
    assert values["Ink__HueClasses__blue"] == values["Ink__HueClasses__gold"] == values["Ink__HueClasses__green"] == 0
    assert values["Ink__HueClasses__count"] == 1
    plain = measure(written_page(tmp_path, red=False))
    assert plain["Ink__ChromaticShare__share"] < 0.01 and plain["Ink__HueClasses__count"] == 0


def test_hue_classes_wrap_round_360() -> None:
    hue = np.array([350.0, 10.0, 70.0, 150.0, 250.0])
    assert ink.hue_class(hue, ink.HUE_CLASSES["red"]).tolist() == [True, True, False, False, False]
    assert ink.hue_class(hue, ink.HUE_CLASSES["blue"]).tolist() == [False, False, False, False, True]
