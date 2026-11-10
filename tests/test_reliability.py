import numpy as np
import pandas as pd
import pytest

from layout_visual_complexity.ranking.elo import permuted_scores
from layout_visual_complexity.ranking.graph import describe
from layout_visual_complexity.ranking.reliability import (
    pair_agreement,
    position_bias,
    possible_pairs_below,
    reliability_by_gap,
    spearman,
    wilson_interval,
)
from layout_visual_complexity.ranking.split_half import SplitHalfSettings, split_half_correlations, split_persons

# Scores a 1500, b 1450, c 1700: the pair (a, b) is 50 apart, (a, c) 200, (b, c) 250.
SCORES = pd.Series({"a": 1500.0, "b": 1450.0, "c": 1700.0})
JUDGMENTS = pd.DataFrame(
    {
        "left": ["a", "b", "a", "a", "c", "b"],
        "right": ["b", "a", "b", "b", "a", "c"],
        "chosen": ["a", "b", "a", "b", "c", "b"],
        "annotator": ["x", "y", "z", "x", "x", "y"],
    }
)


def test_consistency_and_agreement_by_gap() -> None:
    table = reliability_by_gap(JUDGMENTS, SCORES).set_index("gap")
    # Below 100: four judgments of (a, b), two chose a (the higher).
    assert table.loc["<100", "annotated"] == 4 and table.loc["<100", "consistency"] == 0.5
    # (a, b) was judged by x (a, then b), y (b) and z (a): five pairs of judgments by different annotators, two
    # of them agreeing (x's a with z's a, x's b with y's b).
    assert table.loc["<100", "rejudged"] == 1 and table.loc["<100", "agreement"] == pytest.approx(0.4)
    assert table.loc["200-300", "annotated"] == 2 and table.loc["200-300", "consistency"] == 0.5
    assert table.loc["overall", "annotated"] == 6 and table.loc["overall", "consistency"] == pytest.approx(3 / 6)
    assert table["possible"].tolist() == [1, 0, 2, 0, 0, 3]


def test_pairs_by_one_annotator_alone_are_not_rejudged() -> None:
    alone = pd.DataFrame({"left": ["a", "b"], "right": ["b", "a"], "chosen": ["a", "a"], "annotator": ["x", "x"]})
    assert pair_agreement(alone).empty


def test_possible_pairs_count_ties_and_bounds() -> None:
    scores = np.array([0.0, 0.0, 50.0, 150.0])
    assert possible_pairs_below(scores, 0.0) == 0
    assert possible_pairs_below(scores, 100.0) == 3
    assert possible_pairs_below(scores, np.inf) == 6


def test_position_bias_overall_and_per_annotator() -> None:
    bias = position_bias(JUDGMENTS).set_index("annotator")
    # Five of the six chose the left image; x's second judgment chose the right one.
    assert bias.loc["overall", "left"] == pytest.approx(5 / 6)
    assert bias.loc["x", "judgments"] == 3 and bias.loc["x", "left"] == pytest.approx(2 / 3)
    low, high = wilson_interval(50, 100)
    assert low < 0.5 < high and high - low == pytest.approx(0.19, abs=0.01)


def test_the_comparison_graph() -> None:
    summary = describe(JUDGMENTS, ["a", "b", "c", "d"], anchor_factor=1.0)
    assert summary.pairs == 3 and summary.components == 2 and summary.largest_component == 3
    assert summary.density == pytest.approx(3 / 6)
    # Comparisons: a 5, b 5, c 2, d 0; median 3.5: a and b are anchors.
    assert summary.anchors == ("a", "b")


def test_halves_balance_judgments_or_persons() -> None:
    counts = pd.Series([100, 1, 1, 1, 1, 96], index=list("abcdef"))
    half = split_persons(counts, np.random.default_rng(0), "judgments")
    loads = [counts[half == side].sum() for side in (0, 1)]
    assert abs(loads[0] - loads[1]) <= 6
    half = split_persons(counts, np.random.default_rng(0), "persons")
    assert (half == 0).sum() == 3


def simulate_judgments(strengths: np.ndarray, annotators: int, each: int, noise: bool, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    images = [f"p{n:02d}" for n in range(len(strengths))]
    rows = []
    for annotator in range(annotators):
        for _ in range(each):
            first, second = rng.choice(len(strengths), 2, replace=False)
            if noise:
                first_wins = rng.random() < 1 / (1 + np.exp(strengths[second] - strengths[first]))
            else:
                first_wins = strengths[first] > strengths[second]
            chosen = first if first_wins else second
            rows.append((images[first], images[second], images[chosen], f"annotator_{annotator:03d}"))
    return pd.DataFrame(rows, columns=["left", "right", "chosen", "annotator"])


def mean_elo(judgments: pd.DataFrame, images: list[str], seed: int) -> np.ndarray:
    index = {image: n for n, image in enumerate(images)}
    other = judgments["right"].where(judgments["chosen"] == judgments["left"], judgments["left"])
    scores = permuted_scores(
        judgments["chosen"].map(index).to_numpy(), other.map(index).to_numpy(), len(images), 20, seed, jobs=1
    )
    return scores.mean(axis=0)


def test_the_split_half_ceiling_matches_a_known_reliability() -> None:
    strengths = np.linspace(-1.5, 1.5, 40)
    images = [f"p{n:02d}" for n in range(40)]
    settings = SplitHalfSettings(repeats=8, permutations=20, seed=3)
    judgments = simulate_judgments(strengths, 50, 30, noise=True, seed=1)
    result = split_half_correlations(judgments, images, settings, jobs=1)
    estimate = float(result["spearman_brown"].median())
    # The reliability, known by simulation: the correlation between the scores of two independent sets of
    # judgments of the same size, drawn from the same strengths.
    replications = [
        spearman(
            mean_elo(simulate_judgments(strengths, 50, 30, True, 100 + 2 * n), images, n),
            mean_elo(simulate_judgments(strengths, 50, 30, True, 101 + 2 * n), images, n),
        )
        for n in range(4)
    ]
    assert estimate == pytest.approx(np.mean(replications), abs=0.08)
    assert (result["judgments_a"] + result["judgments_b"] == 1500).all()


def test_the_split_half_ceiling_is_near_one_without_noise_and_near_zero_at_random() -> None:
    strengths = np.linspace(-1.5, 1.5, 30)
    images = [f"p{n:02d}" for n in range(30)]
    settings = SplitHalfSettings(repeats=4, permutations=10, seed=1)
    clean = simulate_judgments(strengths, 40, 30, noise=False, seed=2)
    assert split_half_correlations(clean, images, settings, jobs=1)["spearman_brown"].median() > 0.9
    # Without signal, the halves' correlation varies around zero from one set of judgments to the next.
    at_random = [
        split_half_correlations(simulate_judgments(np.zeros(30), 40, 30, True, seed), images, settings, jobs=1)
        for seed in range(8)
    ]
    assert abs(np.mean([result["spearman"].median() for result in at_random])) < 0.1
