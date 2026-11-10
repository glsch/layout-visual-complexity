"""How reliable the judgments are, by the gap between the two images' final scores.

- consistency: the share of judgments that chose the image with the higher final score (ties count half);
- agreement: over image pairs judged by at least two different annotators, the share of pairs of their judgments
  (one judgment from each of two different annotators) that chose the same image;
- position bias: the share of judgments that chose the image on the left.
"""

import math

import numpy as np
import pandas as pd

DEFAULT_GAP_EDGES = (100.0, 200.0, 300.0, 400.0)
OVERALL = "overall"


def gap_bins(edges: tuple[float, ...] = DEFAULT_GAP_EDGES) -> list[tuple[float, float]]:
    """The bins the edges make, from 0 to infinity: [0, 100), [100, 200), …, [400, ∞)."""
    bounds = [0.0, *edges, math.inf]
    return list(zip(bounds[:-1], bounds[1:], strict=True))


def gap_label(low: float, high: float) -> str:
    """`<100`, `100-200`, `>=400`."""
    if low == 0:
        return f"<{high:g}"
    if math.isinf(high):
        return f">={low:g}"
    return f"{low:g}-{high:g}"


def gap_bin(gaps: np.ndarray, edges: tuple[float, ...] = DEFAULT_GAP_EDGES) -> np.ndarray:
    """Each gap's bin, from 0."""
    return np.searchsorted(np.asarray(edges, dtype=float), np.asarray(gaps, dtype=float), side="right")


def other_image(judgments: pd.DataFrame) -> pd.Series:
    """The image not chosen."""
    return judgments["right"].where(judgments["chosen"] == judgments["left"], judgments["left"])


def agrees_with_scores(judgments: pd.DataFrame, scores: pd.Series) -> np.ndarray:
    """1 where the chosen image has the higher score, 0.5 on a tie, 0 otherwise."""
    chosen = judgments["chosen"].map(scores).to_numpy(dtype=float)
    other = other_image(judgments).map(scores).to_numpy(dtype=float)
    return (chosen > other) + 0.5 * (chosen == other)


def judgment_gaps(judgments: pd.DataFrame, scores: pd.Series) -> np.ndarray:
    """The absolute difference between the two images' scores, per judgment."""
    return np.abs(judgments["left"].map(scores).to_numpy(dtype=float) - judgments["right"].map(scores).to_numpy())


def possible_pairs_below(scores: np.ndarray, gap: float) -> int:
    """How many pairs of distinct items have scores less than `gap` apart."""
    ordered = np.sort(np.asarray(scores, dtype=float))
    if math.isinf(gap):
        return len(ordered) * (len(ordered) - 1) // 2
    reach = np.searchsorted(ordered, ordered + gap, side="left")
    return int(np.sum(np.maximum(reach - np.arange(len(ordered)) - 1, 0)))


def _pairs(count: pd.Series) -> pd.Series:
    return count * (count - 1) // 2


def pair_agreement(judgments: pd.DataFrame) -> pd.DataFrame:
    """Per image pair judged by at least two annotators: the pairs of judgments by different annotators, and how
    many of them chose the same image. Columns: `first`, `second`, `annotators`, `judgment_pairs`, `agreeing`."""
    left, right = judgments["left"].to_numpy(dtype=object), judgments["right"].to_numpy(dtype=object)
    first = np.where(left < right, left, right)
    frame = pd.DataFrame(
        {
            "first": first,
            "second": np.where(left < right, right, left),
            "annotator": judgments["annotator"].to_numpy(dtype=object),
            "a": (judgments["chosen"].to_numpy(dtype=object) == first).astype(int),
        }
    )
    frame["b"] = 1 - frame["a"]
    per_annotator = frame.groupby(["first", "second", "annotator"], sort=False).agg(
        n=("a", "size"), a=("a", "sum"), b=("b", "sum")
    )
    for column in ["n", "a", "b"]:
        per_annotator[f"{column}_within"] = _pairs(per_annotator[column])
    per_pair = per_annotator.groupby(level=["first", "second"], sort=False).agg(
        annotators=("n", "size"), **{column: (column, "sum") for column in per_annotator.columns}
    )
    per_pair["judgment_pairs"] = _pairs(per_pair["n"]) - per_pair["n_within"]
    per_pair["agreeing"] = _pairs(per_pair["a"]) - per_pair["a_within"] + _pairs(per_pair["b"]) - per_pair["b_within"]
    rejudged = per_pair[per_pair["annotators"] >= 2].reset_index()
    return rejudged[["first", "second", "annotators", "judgment_pairs", "agreeing"]]


def reliability_by_gap(
    judgments: pd.DataFrame, scores: pd.Series, edges: tuple[float, ...] = DEFAULT_GAP_EDGES
) -> pd.DataFrame:
    """Table 2 of the article: per gap bin and overall, the possible pairs among the scored images, the judgments,
    their consistency, the pairs judged by two annotators or more, and their agreement.

    `scores` holds one final score per image; all of them count for the possible pairs.
    """
    gaps = judgment_gaps(judgments, scores)
    hits = agrees_with_scores(judgments, scores)
    bins = gap_bin(gaps, edges)
    rejudged = pair_agreement(judgments)
    rejudged_bins = gap_bin(np.abs(rejudged["first"].map(scores) - rejudged["second"].map(scores)), edges)
    values = scores.to_numpy(dtype=float)
    rows = []
    for number, (low, high) in enumerate(gap_bins(edges)):
        here, pairs_here = bins == number, rejudged_bins == number
        judged_pairs = rejudged.loc[pairs_here, "judgment_pairs"].sum()
        rows.append(
            {
                "gap": gap_label(low, high),
                "possible": possible_pairs_below(values, high) - possible_pairs_below(values, low),
                "annotated": int(here.sum()),
                "consistency": float(hits[here].mean()) if here.any() else math.nan,
                "rejudged": int(pairs_here.sum()),
                "agreement": float(rejudged.loc[pairs_here, "agreeing"].sum() / judged_pairs)
                if judged_pairs
                else math.nan,
            }
        )
    total_pairs = rejudged["judgment_pairs"].sum()
    rows.append(
        {
            "gap": OVERALL,
            "possible": possible_pairs_below(values, math.inf),
            "annotated": len(judgments),
            "consistency": float(hits.mean()) if len(hits) else math.nan,
            "rejudged": len(rejudged),
            "agreement": float(rejudged["agreeing"].sum() / total_pairs) if total_pairs else math.nan,
        }
    )
    return pd.DataFrame(rows)


def wilson_interval(successes: int, trials: int, z: float = 1.959963984540054) -> tuple[float, float]:
    """The Wilson score interval of a proportion, 95% by default."""
    if trials == 0:
        return math.nan, math.nan
    p = successes / trials
    centre = (p + z * z / (2 * trials)) / (1 + z * z / trials)
    half = z * math.sqrt(p * (1 - p) / trials + z * z / (4 * trials * trials)) / (1 + z * z / trials)
    return centre - half, centre + half


def position_bias(judgments: pd.DataFrame) -> pd.DataFrame:
    """The share of judgments that chose the left image, with its 95% interval: overall, then per annotator."""
    left = judgments["chosen"] == judgments["left"]
    groups = [(OVERALL, left)] + [
        (name, left[judgments["annotator"] == name]) for name in judgments["annotator"].unique()
    ]
    rows = []
    for name, chose_left in groups:
        low, high = wilson_interval(int(chose_left.sum()), len(chose_left))
        share = float(chose_left.mean()) if len(chose_left) else math.nan
        rows.append({"annotator": name, "judgments": len(chose_left), "left": share, "low": low, "high": high})
    return pd.DataFrame(rows)


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    """Spearman's rank correlation, with tied values given their mean rank."""
    ranks_a = pd.Series(np.asarray(a, dtype=float)).rank().to_numpy()
    ranks_b = pd.Series(np.asarray(b, dtype=float)).rank().to_numpy()
    if len(ranks_a) < 2 or np.std(ranks_a) == 0 or np.std(ranks_b) == 0:
        return math.nan
    return float(np.corrcoef(ranks_a, ranks_b)[0, 1])
