"""The local ViT head: its scores, its training with RankNet's loss, its folds in worker processes, what it refuses,
its maps and the head that comes with the package."""

from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from conftest import write_token_set
from layout_visual_complexity.evaluation.folds import manuscript_folds
from layout_visual_complexity.evaluation.vit_head import (
    HeadRanker,
    HeadSettings,
    HeadWeights,
    PackedTokens,
    TrainedHead,
    local_map,
    shipped_head,
    train_head,
    unit,
)
from layout_visual_complexity.features.vit import TokenGrid, TokenMismatch, read_token_set
from layout_visual_complexity.ranking.dataset import read_dataset

pytest.importorskip("torch")


def weights(dim: int = 4, hidden: int = 3, bias: float = 0.0, seed: int = 0) -> HeadWeights:
    rng = np.random.default_rng(seed)
    return HeadWeights(rng.normal(size=(dim, hidden)), rng.normal(size=hidden), rng.normal(size=hidden), bias)


def test_softplus_keeps_every_patch_score_non_negative() -> None:
    tokens = np.random.default_rng(1).normal(size=(500, 4))
    for bias in (-50.0, 0.0, 5.0):
        scores = weights(bias=bias).patch_scores(tokens)
        assert (scores >= 0).all() and np.isfinite(scores).all()


def test_a_pages_score_is_the_mean_of_its_patches() -> None:
    rng = np.random.default_rng(2)
    values = rng.normal(size=(9, 4)).astype(np.float32)
    packed = PackedTokens(values, np.array([0, 4, 9]), ["a", "b"])
    head = weights()
    patch = head.patch_scores(values)
    np.testing.assert_allclose(packed.page_scores(head), [patch[:4].mean(), patch[4:].mean()], rtol=1e-6)


def ranknet_loss(scores: np.ndarray, chosen: np.ndarray, other: np.ndarray) -> float:
    return float(np.mean(np.logaddexp(0.0, scores[other] - scores[chosen])))


def test_ranknets_loss_falls_on_judgments_where_one_page_always_wins() -> None:
    rng = np.random.default_rng(3)
    values = unit(rng.normal(size=(40, 6)))
    packed = PackedTokens(values, np.arange(0, 41, 8), ["a", "b", "c", "d", "e"])
    chosen = np.zeros(40, dtype=np.int64)
    other = np.tile(np.arange(1, 5), 10)
    settings = HeadSettings(hidden=8, epochs=1, learning_rate=0.01, seed=4)
    _, before = train_head(packed, chosen, other, settings, seed=4)
    _, after = train_head(packed, chosen, other, HeadSettings(hidden=8, epochs=150, learning_rate=0.01), seed=4)
    assert ranknet_loss(after, chosen, other) < ranknet_loss(before, chosen, other) - 0.2
    assert after.argmax() == 0


def strengths(dataset_folder: Path) -> tuple[dict[str, float], object]:
    dataset = read_dataset(dataset_folder)
    pages = dataset.images["image"].tolist()
    return dict(zip(pages, np.linspace(-2, 2, len(pages)), strict=True)), dataset


def test_the_same_seed_gives_the_same_weights_with_any_number_of_workers(tmp_path: Path, dataset_folder: Path) -> None:
    strength, dataset = strengths(dataset_folder)
    token_set = read_token_set(write_token_set(tmp_path / "tokens", strength))
    judgments = dataset.judgments.reset_index(drop=True)  # type: ignore[attr-defined]
    pages = list(strength)
    folds = manuscript_folds(pages, dataset.images.set_index("image")["manuscript"], judgments, 2, 2, 1)  # type: ignore[attr-defined]
    found, weights_found = [], []
    for jobs in (1, 3):
        ranker = HeadRanker(token_set, pages, HeadSettings(hidden=4, epochs=30, seed=5), jobs)
        try:
            found.append(ranker.fold_scores(folds, judgments))
            chosen, other = ranker.positions(judgments.iloc[folds[0].train])
            trained = ranker.train_many([(chosen, other, seed) for seed in (1, 2, 3)])
            weights_found.append([item["weights"] for item in trained])
        finally:
            ranker.close()
    assert ranker.workers == 3
    for one, many in zip(found[0], found[1], strict=True):
        pd.testing.assert_series_equal(one, many, check_exact=True)
    for one, many in zip(*weights_found, strict=True):
        for name in ("first", "first_bias", "second"):
            np.testing.assert_array_equal(getattr(one, name), getattr(many, name))
        assert one.second_bias == many.second_bias
    assert not np.array_equal(weights_found[0][0].first, weights_found[0][1].first)
    first = found[0][0]
    assert first is not None and first.index.tolist()[: len(folds[0].train_pages)] == list(folds[0].train_pages)


def trained_head(tokens: dict, dim: int = 4) -> TrainedHead:
    return TrainedHead(weights(dim), tokens, asdict(HeadSettings()), 1.0, 0.5, 2.0, {})


def test_a_head_refuses_tokens_made_otherwise(tmp_path: Path, dataset_folder: Path) -> None:
    strength, _ = strengths(dataset_folder)
    token_set = read_token_set(write_token_set(tmp_path / "tokens", strength, dim=4))
    head = trained_head(token_set.identity())
    assert head.scores(token_set).shape == (len(strength),)
    for name, value in [("block", 1), ("long_edge", 518), ("crop", "page"), ("backbone", "facebook/dinov2-small")]:
        other = trained_head(token_set.identity() | {name: value})
        with pytest.raises(TokenMismatch, match=name):
            other.scores(token_set)
    saved = TrainedHead.load(head.save(tmp_path / "head.npz"))
    assert saved.tokens == head.tokens and saved.scale == 2.0
    np.testing.assert_allclose(saved.weights.first, head.weights.first.astype(np.float32))


def test_a_local_map_places_each_patchs_score_on_the_pages_grid() -> None:
    tokens = np.zeros((6, 4))
    tokens[4, 0] = 3.0
    head = TrainedHead(HeadWeights(np.eye(4)[:, :1], np.zeros(1), np.ones(1), 0.0), {}, {"unit_length": False},
                       0.0, 1.0, 1.0, {})  # fmt: skip
    grid = TokenGrid(tokens, 2, 3, (100.0, 200.0, 400.0, 400.0))
    found = local_map(head, grid)
    assert found.shape == (2, 3) and np.unravel_index(found.argmax(), found.shape) == (1, 1)
    np.testing.assert_allclose(grid.patch_boxes()[1 * 3 + 1], [200, 300, 300, 400])


def test_the_shipped_head_loads_from_the_package_and_scores_a_page() -> None:
    head = shipped_head()
    assert head.tokens["backbone"] == "facebook/dinov2-small" and head.tokens["crop"] == "folio"
    assert head.tokens["block"] == 12 and head.tokens["long_edge"] == 518 and head.tokens["dim"] == 384
    assert head.trained_on["pages"] == 397 and head.settings["epochs"] == 180
    tokens = np.random.default_rng(6).normal(size=(37 * 26, 384)).astype(np.float16)
    scores = head.patch_scores(TokenGrid(tokens, 37, 26, (0.0, 0.0, 1300.0, 1850.0)))
    assert scores.shape == (962,) and (scores >= 0).all() and np.isfinite(scores.mean())
