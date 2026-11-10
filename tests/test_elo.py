import numpy as np
import pandas as pd
import pytest

from layout_visual_complexity.ranking import bradley_terry
from layout_visual_complexity.ranking.elo import EloSettings, expected, permutation, permuted_scores, play
from layout_visual_complexity.ranking.reliability import spearman
from layout_visual_complexity.ranking.scores import compute_scores, read_scores, write_scores


def test_the_articles_worked_example() -> None:
    # A (item 0) at 1450 and B (item 1) at 1750, K = 100: A wins, then B; and the other way round.
    initial = np.array([1450.0, 1750.0])
    winners, losers = np.array([0, 1]), np.array([1, 0])
    assert expected(1450.0, 1750.0) == pytest.approx(0.151, abs=5e-4)
    first = play(winners, losers, np.array([[0, 1]]), 2, EloSettings(), initial)[0]
    second = play(winners, losers, np.array([[1, 0]]), 2, EloSettings(), initial)[0]
    assert first == pytest.approx([1502.81, 1697.19], abs=0.005)
    assert second == pytest.approx([1521.90, 1678.10], abs=0.005)


def test_scores_keep_their_sum() -> None:
    rng = np.random.default_rng(0)
    winners, losers = rng.integers(0, 5, 50), rng.integers(5, 10, 50)
    scores = play(winners, losers, np.arange(50)[None, :], 10)
    assert scores.sum() == pytest.approx(10 * 1500)


def test_permutations_are_seeded_by_index() -> None:
    assert np.array_equal(permutation(20, 7, 3), np.random.default_rng([7, 3]).permutation(20))
    assert not np.array_equal(permutation(20, 7, 3), permutation(20, 7, 4))


def test_permutations_give_the_same_scores_with_one_and_four_workers() -> None:
    rng = np.random.default_rng(1)
    winners, losers = rng.integers(0, 30, 400), rng.integers(0, 30, 400)
    keep = winners != losers
    one = permuted_scores(winners[keep], losers[keep], 30, 100, seed=5, jobs=1)
    four = permuted_scores(winners[keep], losers[keep], 30, 100, seed=5, jobs=4)
    assert one.shape == (100, 30)
    assert np.array_equal(one, four)


def simulated(strengths: np.ndarray, judgments: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    first = rng.integers(0, len(strengths), judgments)
    second = (first + rng.integers(1, len(strengths), judgments)) % len(strengths)
    p = 1 / (1 + np.exp(strengths[second] - strengths[first]))
    first_wins = rng.random(judgments) < p
    return np.where(first_wins, first, second), np.where(first_wins, second, first)


def test_bradley_terry_recovers_known_strengths() -> None:
    strengths = np.linspace(-2, 2, 12)
    winners, losers = simulated(strengths, 20_000, seed=2)
    estimate = bradley_terry.log_strengths(winners, losers, len(strengths), prior=1.0)
    assert estimate == pytest.approx(strengths - strengths.mean(), abs=0.1)


def test_bradley_terry_stays_finite_for_an_image_that_always_wins() -> None:
    estimate = bradley_terry.log_strengths(np.array([0, 0, 0]), np.array([1, 2, 1]), 4)
    assert np.isfinite(estimate).all()
    assert estimate[0] == estimate.max()


def test_bradley_terry_on_elos_scale() -> None:
    scores = bradley_terry.on_elo_scale(np.array([-np.log(10) / 2, np.log(10) / 2]))
    assert scores == pytest.approx([1300.0, 1700.0])


def test_scores_table_and_distributions_round_trip(tmp_path) -> None:
    strengths = np.linspace(-1.5, 1.5, 8)
    winners, losers = simulated(strengths, 600, seed=3)
    images = [f"p{n}" for n in range(8)]
    left = np.where(np.arange(600) % 2 == 0, winners, losers)
    right = np.where(np.arange(600) % 2 == 0, losers, winners)
    judgments = pd.DataFrame(
        {
            "left": [images[i] for i in left],
            "right": [images[i] for i in right],
            "chosen": [images[i] for i in winners],
        }
    )
    scores = compute_scores(judgments, images, permutations=30, seed=1, jobs=1)
    table = scores.table
    assert table["comparisons"].sum() == 1200 and table["wins"].sum() == 600
    assert (table["elo_q05"] <= table["elo_median"]).all() and (table["elo_median"] <= table["elo_q95"]).all()
    assert spearman(table["elo_mean"].to_numpy(), strengths) > 0.9
    assert spearman(table["bt_score"].to_numpy(), table["elo_mean"].to_numpy()) > 0.9
    write_scores(scores, tmp_path, seed=1)
    again = read_scores(tmp_path)
    assert np.array_equal(again.distributions, scores.distributions)
    assert again.table["image"].tolist() == images
