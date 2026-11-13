"""Kyle–Davidson's measures as published, on synthetic images with known answers."""

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from layout_visual_complexity.features import kyle_davidson
from layout_visual_complexity.features.contract import aggregate
from layout_visual_complexity.features.kyle_davidson import KyleDavidson, KyleDavidsonSettings
from layout_visual_complexity.features.page import Page

RNG = np.random.default_rng(11)


def test_a_two_colour_image_gives_two_regions_after_the_normalised_cut() -> None:
    image = np.zeros((120, 160, 3), np.uint8)
    image[:, :80] = (200, 40, 40)
    image[:, 80:] = (40, 60, 200)
    labels = kyle_davidson.normalised_cut(image, KyleDavidsonSettings(superpixels=60))
    assert len(np.unique(labels)) == 2
    assert len(np.unique(labels[:, :70])) == 1 and len(np.unique(labels[:, 90:])) == 1


def test_a_mirrored_patch_gives_full_symmetry_and_noise_less() -> None:
    half = RNG.random((8, 4, 3))
    patch = np.concatenate([half, half[:, ::-1]], axis=1)
    patch = np.concatenate([patch, patch[::-1]], axis=0)
    across, down = kyle_davidson.patch_symmetry(patch[None])
    assert across[0] == pytest.approx(1) and down[0] == pytest.approx(1)
    noise_across, noise_down = kyle_davidson.patch_symmetry(RNG.random((1, 16, 16, 3)))
    assert noise_across[0] < 0.8 and noise_down[0] < 0.8


def test_an_image_mirrored_in_every_patch_scores_full_symmetry() -> None:
    inside = np.ones((80, 80), bool)
    rows = RNG.integers(0, 255, (80, 1, 3)).astype(np.uint8)
    stripes = np.repeat(rows, 80, axis=1)  # each row one colour: every patch mirrors left to right
    assert kyle_davidson.symmetry(np.full((80, 80, 3), 90, np.uint8), inside, 20, 0.9) == pytest.approx(1)
    across = kyle_davidson.symmetry(stripes, inside, 20, 0.9)
    noise = kyle_davidson.symmetry(RNG.integers(0, 255, (80, 80, 3)).astype(np.uint8), inside, 20, 0.9)
    assert noise < across < 1


def test_a_flat_image_has_no_hsv_entropy() -> None:
    inside = np.ones((64, 64), bool)
    assert kyle_davidson.hsv_entropy(np.full((64, 64, 3), (180, 150, 100), np.uint8), inside, 4) == (0, 0, 0)
    noisy = kyle_davidson.hsv_entropy(RNG.integers(0, 255, (64, 64, 3)).astype(np.uint8), inside, 4)
    assert all(value > 3 for value in noisy)


def test_the_family_measures_a_page(tmp_path: Path) -> None:
    pixels = np.full((300, 200, 3), 225, np.uint8)
    pixels[40:260:20, 30:170] = 40
    pixels[50:90, 30:70] = (190, 30, 30)
    path = tmp_path / "page.png"
    Image.fromarray(pixels).save(path)
    folio = np.array([[0, 0], [200, 0], [200, 300], [0, 300]], float)
    page = Page("ms_1r", 200, 300, folio, image_path=path)
    family = KyleDavidson(KyleDavidsonSettings(long_edge=150, superpixels=60, patch_sizes=(40, 20, 10)))
    values = aggregate(family, family.measure(page))
    assert family.columns() == list(values)
    assert values["KyleDavidson__Clutter__regions"] >= 2
    assert all(0 < values[f"KyleDavidson__Symmetry__{size}"] <= 1 for size in ("large", "medium", "small"))
    assert values["KyleDavidson__Entropy__value"] > 0
