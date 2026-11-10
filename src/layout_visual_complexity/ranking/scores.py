"""Scores from judgments: Elo over permutations, summarised per image, and Bradley–Terry beside them."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from layout_visual_complexity.ranking import bradley_terry
from layout_visual_complexity.ranking.elo import EloSettings, permuted_scores
from layout_visual_complexity.ranking.graph import comparison_counts, win_counts
from layout_visual_complexity.ranking.reliability import other_image

SCORE_COLUMNS = [
    "image",
    "elo_mean",
    "elo_sd",
    "elo_q05",
    "elo_q25",
    "elo_median",
    "elo_q75",
    "elo_q95",
    "comparisons",
    "wins",
    "bt_score",
]


@dataclass
class Scores:
    """Per-image scores, and the Elo scores of every permutation (permutations × images, in the table's order)."""

    table: pd.DataFrame
    distributions: np.ndarray


def indices(judgments: pd.DataFrame, images: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """The chosen and the other image of each judgment, as indices into `images`."""
    index = {image: number for number, image in enumerate(images)}
    winners = judgments["chosen"].map(index)
    losers = other_image(judgments).map(index)
    if winners.isna().any() or losers.isna().any():
        raise ValueError("some judged images aren't in the image list")
    return winners.to_numpy(dtype=np.intp), losers.to_numpy(dtype=np.intp)


def summarise(distributions: np.ndarray, images: list[str]) -> pd.DataFrame:
    """Mean, standard deviation and quantiles of each image's scores over the permutations."""
    quantiles = np.quantile(distributions, [0.05, 0.25, 0.5, 0.75, 0.95], axis=0)
    return pd.DataFrame(
        {
            "image": images,
            "elo_mean": distributions.mean(axis=0),
            "elo_sd": distributions.std(axis=0, ddof=1) if len(distributions) > 1 else np.zeros(len(images)),
            "elo_q05": quantiles[0],
            "elo_q25": quantiles[1],
            "elo_median": quantiles[2],
            "elo_q75": quantiles[3],
            "elo_q95": quantiles[4],
        }
    )


def compute_scores(
    judgments: pd.DataFrame,
    images: list[str],
    permutations: int,
    seed: int,
    elo: EloSettings | None = None,
    bt_prior: float = 1.0,
    jobs: int = 0,
) -> Scores:
    """Elo over `permutations` seeded permutations, its summary per image, and Bradley–Terry on Elo's scale."""
    elo = elo or EloSettings()
    winners, losers = indices(judgments, images)
    distributions = permuted_scores(winners, losers, len(images), permutations, seed, elo, jobs)
    table = summarise(distributions, images)
    table["comparisons"] = comparison_counts(judgments, images).to_numpy()
    table["wins"] = win_counts(judgments, images).to_numpy()
    logs = bradley_terry.log_strengths(winners, losers, len(images), prior=bt_prior)
    table["bt_score"] = bradley_terry.on_elo_scale(logs, centre=elo.start, scale=elo.scale)
    return Scores(table=table[SCORE_COLUMNS], distributions=distributions)


def write_scores(scores: Scores, folder: Path, seed: int) -> list[Path]:
    """`scores.tsv` and `score_distributions.npz` (arrays `scores`, `images` and `seed`) in `folder`."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    table_path = folder / "scores.tsv"
    scores.table.to_csv(table_path, sep="\t", index=False, lineterminator="\n", float_format="%.6f")
    array_path = folder / "score_distributions.npz"
    np.savez_compressed(
        array_path,
        scores=scores.distributions,
        images=np.array(scores.table["image"].tolist(), dtype=str),
        seed=np.array(seed),
    )
    return [table_path, array_path]


def read_scores(folder: Path) -> Scores:
    """The scores a ranking run wrote."""
    folder = Path(folder)
    table = pd.read_csv(folder / "scores.tsv", sep="\t")
    table["image"] = table["image"].astype(object)
    with np.load(folder / "score_distributions.npz") as data:
        distributions = data["scores"]
        if list(data["images"]) != table["image"].tolist():
            raise ValueError(f"{folder}: the distributions and the score table list different images")
    return Scores(table=table, distributions=distributions)
