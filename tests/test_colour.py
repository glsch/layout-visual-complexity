"""Real colour, on synthetic pages and colours with known answers."""

import math
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from layout_visual_complexity.features import colour as colour_module
from layout_visual_complexity.features.colour import Colour, ColourSettings
from layout_visual_complexity.features.contract import aggregate, load_family
from layout_visual_complexity.features.folio_image import InkLayer, NormalisedFolio, ink_layer
from layout_visual_complexity.features.ink import InkDistribution, InkSettings, ink_colours
from layout_visual_complexity.features.ink_colour import HUE_CLASSES, InkColour, neutral_lab
from layout_visual_complexity.features.page import Page, TextLine, Zone

PARCHMENT, BLACK, RED, BLUE = (232, 222, 196), (35, 30, 25), (190, 35, 30), (40, 60, 170)


def box(x0: float, y0: float, x1: float, y1: float) -> np.ndarray:
    return np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=float)


def page_with(tmp_path: Path, strokes: list[tuple[int, tuple]], regions: list[Zone] = (), headings=()) -> Page:
    """A 400 × 600 parchment page with a text line every 30 pixels from y = 90, each written in the colour given."""
    pixels = np.full((600, 400, 3), PARCHMENT, np.uint8)
    lines = []
    for number, (y, ink) in enumerate(strokes):
        pixels[y - 6 : y, 60:340] = ink
        lines.append(TextLine(f"l{number}", box(60, y - 12, 340, y + 2), np.array([[60.0, y], [340.0, y]]), None))
    for zone, ink in regions:
        if ink is not None:
            x0, y0 = zone.polygon.min(axis=0).astype(int)
            x1, y1 = zone.polygon.max(axis=0).astype(int)
            pixels[y0 + 4 : y1 - 4, x0 + 4 : x1 - 4][::3] = ink
    path = tmp_path / f"page_{len(list(tmp_path.iterdir()))}.png"
    Image.fromarray(pixels).save(path)
    zones = [zone for zone, _ in regions]
    return Page("ms_1r", 400, 600, box(0, 0, 400, 600), zones, lines, list(headings), image_path=path)


def measure(page: Page) -> dict[str, float]:
    family = Colour(ColourSettings(long_edge=600))
    return aggregate(family, family.measure(page))


def test_the_hue_entropy_of_one_hue_and_of_every_hue() -> None:
    assert colour_module.hue_entropy(np.full(50, 5.0), 24) == 0
    assert colour_module.hue_entropy(np.array([]), 24) == 0
    even = np.repeat(np.arange(24) * 15 + 7.5, 10)
    assert colour_module.hue_entropy(even, 24) == pytest.approx(math.log2(24))
    assert colour_module.hue_entropy(np.array([10.0, 20.0, 359.9]), 24) == pytest.approx(math.log2(3))


def known_colour(chroma: np.ndarray, hue: np.ndarray) -> InkColour:
    """Ink colours given pixel by pixel, all of them ink, one line height of 1."""
    shape = chroma.shape
    folio = NormalisedFolio(np.zeros((*shape, 3), np.uint8), np.zeros((*shape, 3), np.float32), np.ones(shape, bool),
                            1.0, (90.0, 0.0, 0.0))  # fmt: skip
    return InkColour(InkLayer(folio, np.ones(shape, bool), 1.0, "lines"), chroma, hue)


def test_the_chroma_statistics_of_a_known_mix() -> None:
    chroma = np.concatenate([np.full(10, 30.0), np.full(10, 40.0), np.full(10, 50.0), np.full(70, 5.0)])[None, :]
    hue = np.concatenate([np.full(20, 20.0), np.full(10, 250.0), np.full(70, 20.0)])[None, :]
    colour = known_colour(chroma, hue)
    settings = ColourSettings()
    chromatic = colour.chromatic(settings.chroma)
    found = colour_module.hue_measures(colour, colour.layer.ink, chromatic, settings)
    assert found["Colour__ChromaticShare__share"] == pytest.approx(0.3)
    assert found["Colour__Saturation__median"] == pytest.approx(40)
    assert found["Colour__Saturation__spread"] == pytest.approx(20)
    assert found["Colour__HueClasses__red"] == pytest.approx(0.2)
    assert found["Colour__HueClasses__blue"] == pytest.approx(0.1)
    assert found["Colour__HueClasses__count"] == 2
    assert found["Colour__HueEntropy__bits"] == pytest.approx(-(2 / 3 * math.log2(2 / 3) + 1 / 3 * math.log2(1 / 3)))
    plain = colour_module.hue_measures(colour, colour.layer.ink, np.zeros_like(chromatic), settings)
    assert plain["Colour__ChromaticShare__share"] == 0 and plain["Colour__HueEntropy__bits"] == 0
    assert math.isnan(plain["Colour__Saturation__median"])


def test_three_hue_classes() -> None:
    assert HUE_CLASSES == {"red": (330.0, 60.0), "green": (110.0, 200.0), "blue": (200.0, 330.0)}
    hues = [column for column in Colour().columns() if "HueClasses" in column]
    assert hues == ["Colour__HueClasses__red", "Colour__HueClasses__green", "Colour__HueClasses__blue",
                    "Colour__HueClasses__count"]  # fmt: skip
    assert len(Colour().columns()) == 17


def test_ink_of_hue_80_and_chroma_60_isnt_chromatic() -> None:
    colour = known_colour(np.full((1, 20), 60.0), np.full((1, 20), 80.0))
    settings = ColourSettings()
    chromatic = colour.chromatic(settings.chroma)
    assert not chromatic.any()
    found = colour_module.hue_measures(colour, colour.layer.ink, chromatic, settings)
    assert found["Colour__ChromaticShare__share"] == 0 and found["Colour__HueEntropy__bits"] == 0
    assert all(found[f"Colour__HueClasses__{name}"] == 0 for name in HUE_CLASSES)
    assert found["Colour__HueClasses__count"] == 0 and math.isnan(found["Colour__Saturation__median"])
    ink = ink_colours(colour, settings.chroma, settings.excluded_hues, settings.class_area)
    assert ink["Ink__ChromaticShare__share"] == 0 and ink["Ink__HueClasses__count"] == 0
    assert all(ink[f"Ink__HueClasses__{name}"] == 0 for name in HUE_CLASSES)


def test_red_green_and_blue_ink_in_their_classes() -> None:
    hue = np.array([[20.0] * 10 + [345.0] * 10 + [150.0] * 10 + [250.0] * 10])
    colour = known_colour(np.full(hue.shape, 40.0), hue)
    settings = ColourSettings()
    found = colour_module.hue_measures(colour, colour.layer.ink, colour.chromatic(settings.chroma), settings)
    assert found["Colour__ChromaticShare__share"] == 1
    assert [found[f"Colour__HueClasses__{name}"] for name in HUE_CLASSES] == [0.5, 0.25, 0.25]
    assert found["Colour__HueClasses__count"] == 3
    ink = ink_colours(colour, settings.chroma, settings.excluded_hues, settings.class_area)
    assert [ink[f"Ink__HueClasses__{name}"] for name in HUE_CLASSES] == [0.5, 0.25, 0.25]


def test_the_settings_hashes_change_with_the_excluded_hues() -> None:
    for name, settings in (("ink", InkSettings), ("colour", ColourSettings)):
        family = load_family(name)
        assert family.describe()["settings"]["excluded_hues"] == [60.0, 110.0]
        assert load_family(name, {"excluded_hues": [60.0, 110.0]}).settings_hash() == family.settings_hash()
        assert load_family(name, {"excluded_hues": [50.0, 120.0]}).settings_hash() != family.settings_hash()
        assert load_family(name, {"excluded_hues": []}).settings_hash() != family.settings_hash()
        assert settings().excluded_hues == (60.0, 110.0)


def text(lines: int = 10, red: tuple[int, ...] = ()) -> list[tuple[int, tuple]]:
    return [(90 + 30 * number, RED if number in red else BLACK) for number in range(lines)]


def test_a_red_zone_among_black_ones_gives_its_share_to_its_kind(tmp_path: Path) -> None:
    main = Zone("r1", "MainZone", box(50, 70, 350, 400))
    numbering = Zone("r2", "NumberingZone", box(300, 20, 380, 60))
    margin = Zone("r3", "MarginTextZone", box(355, 100, 395, 300))
    page = page_with(tmp_path, text(), [(main, None), (numbering, RED), (margin, BLACK)])
    values = measure(page)
    assert values["Colour__Zones__numbering"] == pytest.approx(1, abs=0.05)
    assert values["Colour__Zones__main"] == pytest.approx(0, abs=0.01)
    assert values["Colour__Zones__margins"] == pytest.approx(0, abs=0.01)
    assert math.isnan(values["Colour__Zones__decoration"]) and math.isnan(values["Colour__Zones__headings"])
    assert values["Colour__HueClasses__red"] == pytest.approx(values["Colour__ChromaticShare__share"], rel=0.05)


def test_a_red_heading_line_is_counted_among_the_headings(tmp_path: Path) -> None:
    heading = Zone("y1", "HeadingLine", box(55, 75, 345, 95))
    page = page_with(tmp_path, text(red=(0,)), headings=[heading])
    values = measure(page)
    assert values["Colour__Zones__headings"] == pytest.approx(1, abs=0.05)
    assert values["Colour__Zones__rest"] == pytest.approx(0, abs=0.01)


def test_a_stamps_red_ink_is_left_out(tmp_path: Path) -> None:
    stamp = Zone("r1", "StampZone", box(100, 420, 300, 560))
    page = page_with(tmp_path, text(), [(stamp, RED)])
    values = measure(page)
    assert values["Colour__ChromaticShare__share"] == pytest.approx(0, abs=0.005)
    assert values["Colour__HueClasses__count"] == 0
    ink = InkDistribution(InkSettings(long_edge=600))
    assert aggregate(ink, ink.measure(page))["Ink__ChromaticShare__share"] > 0.1


def test_one_red_line_among_black_ones(tmp_path: Path) -> None:
    page = page_with(tmp_path, text(10, red=(4,)))
    values = measure(page)
    assert values["Colour__Lines__rubric_share"] == pytest.approx(0.1)
    assert values["Colour__Lines__spread"] == pytest.approx(np.std([1.0] + [0.0] * 9), abs=0.02)
    plain = measure(page_with(tmp_path, text(10)))
    assert plain["Colour__Lines__rubric_share"] == 0 and plain["Colour__Lines__spread"] == pytest.approx(0, abs=0.01)


def test_a_red_drop_capital_is_a_coloured_initial(tmp_path: Path) -> None:
    red = Zone("r1", "DropCapitalZone", box(10, 80, 55, 140))
    black = Zone("r2", "DropCapitalZone", box(10, 200, 55, 260))
    blue = Zone("r3", "GraphicZone", box(10, 300, 55, 360))
    page = page_with(tmp_path, text(), [(red, RED), (black, BLACK), (blue, BLUE)])
    values = measure(page)
    assert values["Colour__ColouredInitials__count"] == 2
    assert values["Colour__Zones__decoration"] == pytest.approx(2 / 3, abs=0.05)
    assert values["Colour__HueClasses__blue"] > 0 and values["Colour__HueClasses__red"] > 0


def test_the_ink_familys_colour_columns_are_unchanged_by_the_shared_code(tmp_path: Path) -> None:
    page = page_with(tmp_path, text(10, red=(2, 5)))
    settings = InkSettings(long_edge=600)
    ink = InkDistribution(settings)
    values = aggregate(ink, ink.measure(page))
    layer = ink_layer(page, settings)
    lab = neutral_lab(layer.folio.rgb, layer.folio.inside & ~layer.ink)
    a, b = lab[..., 1][layer.ink], lab[..., 2][layer.ink]
    hue = np.degrees(np.arctan2(b, a)) % 360
    chromatic = (np.hypot(a, b) >= settings.chroma) & ~((hue >= 60) & (hue < 110))
    red = chromatic & ((hue >= 330) | (hue < 60))
    assert values["Ink__ChromaticShare__share"] == chromatic.sum() / layer.ink.sum()
    assert values["Ink__HueClasses__red"] == red.sum() / layer.ink.sum()
    found = measure(page)
    for name in ("ChromaticShare__share", "HueClasses__red", "HueClasses__blue", "HueClasses__count"):
        assert found[f"Colour__{name}"] == values[f"Ink__{name}"]
