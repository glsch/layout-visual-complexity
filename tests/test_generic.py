"""The generic image measures on synthetic images with known answers, and the patch codebook."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from layout_visual_complexity.codebook import CodebookSettings, run_codebook
from layout_visual_complexity.features import generic
from layout_visual_complexity.features.extraction import FeaturesSettings, read_feature_table, run_features
from layout_visual_complexity.features.folio_image import NormalisedFolio
from layout_visual_complexity.features.generic import Codebook, Generic, GenericSettings
from test_features_contract import write_store

RNG = np.random.default_rng(5)
FLAT = np.full((128, 96), 0.8)
NOISE = RNG.random((128, 96))


def test_a_flat_image_compresses_best_and_has_no_colourfulness() -> None:
    assert generic.compressibility(FLAT, 16) > generic.compressibility(NOISE, 16)
    grey = np.full((50, 3), 128, np.uint8)
    assert generic.colourfulness(grey) == 0
    coloured = np.array([[200, 40, 40], [40, 40, 200]] * 25, np.uint8)
    assert generic.colourfulness(coloured) > 50


def test_parallel_stripes_give_low_edge_orientation_entropy() -> None:
    stripes = np.zeros((128, 96))
    stripes[::8] = 1.0
    stripes[1::8] = 1.0
    inside = np.ones_like(stripes, bool)
    striped = generic.orientation_entropy(stripes, inside, 24, 120_000, 10_000)
    noisy = generic.orientation_entropy(NOISE, inside, 24, 120_000, 10_000)
    assert striped < 0.5
    assert noisy > 4.0


def test_mssc_grows_with_structure() -> None:
    edge = np.zeros((128, 96))
    edge[:, 45:] = 1.0  # one edge, off the poolings' grid
    blocks = np.kron(RNG.integers(0, 2, (16, 12)).astype(float), np.ones((8, 8)))  # many, of the same contrast
    flat, one, many = (sum(generic.mssc(image, 6)) for image in (FLAT, edge, blocks))
    assert flat == 0
    assert 0 < one < many
    assert generic.mssc(blocks, 6)[:3] == pytest.approx([0, 0, 0])
    assert np.argmax(generic.mssc(NOISE, 6)) == 0
    assert np.isnan(generic.mssc(FLAT, 7)[-1])


def test_feature_congestion_is_higher_on_a_cluttered_image() -> None:
    inside = np.ones(FLAT.shape, bool)
    quiet = np.stack([FLAT * 100, np.zeros_like(FLAT), np.zeros_like(FLAT)], axis=-1).astype(np.float32)
    busy = np.stack([NOISE * 100, RNG.normal(0, 20, FLAT.shape), RNG.normal(0, 20, FLAT.shape)], axis=-1)
    calm, cluttered = generic.feature_congestion(quiet, inside, 3), generic.feature_congestion(busy, inside, 3)
    assert all(a < b for a, b in zip(calm, cluttered, strict=True))


def test_the_codebook_counts_the_codewords_a_page_uses() -> None:
    image = np.zeros((64, 64, 3), np.uint8)
    image[:, :32] = (230, 220, 200)
    image[:, 32:] = (40, 30, 20)
    folio = NormalisedFolio(image, np.zeros((64, 64, 3), np.float32), np.ones((64, 64), bool), 1.0, (0, 0, 0))
    histograms = generic.patch_histograms(folio, 16, 8)
    assert histograms.shape == (16, 24)
    assert np.allclose(histograms.reshape(16, 3, 8).sum(axis=2), 1)
    centroids = generic.learn_codewords([histograms], 2, seed=1)
    codebook = Codebook(centroids, 16, 8, 64, {})
    used, entropy = generic.codebook_variety(codebook.assign(histograms), 2, 0.01)
    assert used == 2 and entropy == pytest.approx(1.0)


def coloured_store(folder: Path) -> Path:
    """A synthetic segmentation run whose pages carry coloured initials of their own colours."""
    store = write_store(folder, pages=4)
    colours = [(200, 30, 30), (30, 60, 200), (40, 160, 60), (200, 160, 40)]
    for number, colour in enumerate(colours, start=1):
        path = store / "images" / "ms" / f"{number}r.png"
        pixels = np.asarray(Image.open(path).convert("RGB")).copy()
        pixels[30 : 30 + 20 * number, 25:60] = colour
        Image.fromarray(pixels).save(path)
    return store


def test_the_codebook_and_its_use_are_the_same_with_any_number_of_workers(tmp_path: Path) -> None:
    store = coloured_store(tmp_path / "segment")
    codebooks = []
    for jobs in (1, 2):
        settings = CodebookSettings(store, tmp_path / f"codebook_{jobs}", codewords=4, patch=16, bins=8,
                                    long_edge=300, samples=40, seed=7, jobs=jobs)  # fmt: skip
        assert run_codebook(settings)["pages"] == 4
        codebooks.append(Codebook.load(tmp_path / f"codebook_{jobs}" / "codebook.npz"))
    assert np.array_equal(codebooks[0].centroids, codebooks[1].centroids)
    again = run_codebook(CodebookSettings(store, tmp_path / "again", codewords=4, patch=16, bins=8, long_edge=300,
                                          samples=40, seed=7, jobs=2))  # fmt: skip
    assert again["patches"] == 160
    assert np.array_equal(Codebook.load(tmp_path / "again" / "codebook.npz").centroids, codebooks[0].centroids)

    path = str(tmp_path / "codebook_1" / "codebook.npz")
    tables = []
    for jobs in (1, 3):
        parameters = {"generic": {"codebook": path, "long_edge": 300}}
        run_features(FeaturesSettings(store, tmp_path / f"features_{jobs}", ["generic"], parameters, jobs=jobs))
        tables.append(read_feature_table(tmp_path / f"features_{jobs}" / "features.parquet").values)
    pd.testing.assert_frame_equal(tables[0], tables[1])
    assert tables[0].notna().all().all()
    assert (tables[0]["Generic__CodebookVariety__codewords"] >= 1).all()


def test_the_familys_hash_follows_the_codebooks_contents_not_its_path(tmp_path: Path) -> None:
    centroids = RNG.random((3, 6)).astype(np.float32)
    first = Codebook(centroids, 8, 2, 300, {}).save(tmp_path / "a" / "codebook.npz")
    second = Codebook(centroids, 8, 2, 300, {}).save(tmp_path / "b" / "codebook.npz")
    other = Codebook(centroids + 1, 8, 2, 300, {}).save(tmp_path / "c" / "codebook.npz")
    hashes = [Generic(GenericSettings(long_edge=300, codebook=str(path))).settings_hash()
              for path in (first, second, other)]  # fmt: skip
    assert hashes[0] == hashes[1] != hashes[2]


def test_a_missing_codebook_is_refused(tmp_path: Path) -> None:
    family = Generic(GenericSettings(codebook=str(tmp_path / "none.npz")))
    with pytest.raises(ValueError, match="lvc codebook"):
        family.codebook  # noqa: B018
