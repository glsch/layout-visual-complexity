"""The rankers that score pages: the geometry ranker, the local ViT head and their ensemble, as `lvc train` makes them
(`ensemble.json` beside its members' files). The package comes with them, trained on the dataset's 397 pages and
all their judgments.

A page's scores: each member's z-score (its raw score less the training pages' mean, over their spread), the
ensemble's (the members' weighted sum, z-scored the same way) and the ensemble's expected Elo score. The probability
that one page is judged more complex than another is a sigmoid of the difference of their z-scores, by a scale fitted
on out-of-fold scores, so it holds on new pages. Each member refuses inputs made otherwise than its own: features
whose families' settings differ, tokens from another backbone, block, input size or crop.
"""

import json
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from layout_visual_complexity.evaluation import metrics
from layout_visual_complexity.evaluation.calibration import EloScale
from layout_visual_complexity.evaluation.model import TrainedModel
from layout_visual_complexity.evaluation.vit_head import TrainedHead
from layout_visual_complexity.runs import sha256

ENSEMBLE = "ensemble.json"


class ModelError(ValueError):
    """Rankers that can't be used: a member missing, or changed since the ensemble was made."""


def shipped_folder() -> Path:
    """The folder of the rankers that come with the package."""
    return Path(str(resources.files("layout_visual_complexity") / "heads"))


@dataclass
class ScoringModels:
    """An ensemble and its members, read from its folder."""

    folder: Path
    description: dict[str, Any]
    features: TrainedModel
    head: TrainedHead
    names: tuple[str, str]
    weights: np.ndarray
    elo_scale: EloScale | None

    @classmethod
    def load(cls, folder: Path | None = None) -> "ScoringModels":
        """The ensemble in `folder` (the package's by default), each member checked against its checksum."""
        folder = Path(folder) if folder is not None else shipped_folder()
        path = folder / ENSEMBLE
        if not path.is_file():
            raise ModelError(f"{folder} holds no {ENSEMBLE}: make one with lvc train --ensemble")
        description = json.loads(path.read_text(encoding="utf-8"))
        features = head = None
        names: dict[str, str] = {}
        for name, member in description["members"].items():
            file = folder / member["file"]
            if not file.is_file():
                raise ModelError(f"the ensemble's member {name} ({file}) isn't there")
            if sha256(file) != member["sha256"]:
                raise ModelError(f"the ensemble's member {name} ({file}) changed since the ensemble was made")
            if file.suffix == ".npz":
                head, names["head"] = TrainedHead.load(file), name
            else:
                features, names["features"] = TrainedModel.load(file), name
        if features is None or head is None or len(description["members"]) != 2:
            raise ModelError("scoring needs an ensemble of one pairwise model on features and one local ViT head")
        weights = np.array([description["members"][names[kind]]["weight"] for kind in ("features", "head")])
        elo = EloScale.from_dict(description["elo"]) if "elo" in description else None
        return cls(folder, description, features, head, (names["features"], names["head"]), weights, elo)

    def files(self) -> dict[str, Path]:
        """Every file the rankers are read from, by what it is."""
        found = {"ensemble": self.folder / ENSEMBLE}
        for name, member in self.description["members"].items():
            found[name] = self.folder / member["file"]
        return found

    def families(self) -> list[dict[str, Any]]:
        """The feature families the pairwise model needs, as its feature table recorded them."""
        return list(self.features.families)

    def scale(self, name: str) -> float:
        """The scale of win probabilities of a member, by name, or of the `ensemble`."""
        if name == "ensemble":
            return float(self.description["scale"])
        member = self.description["members"][name]
        return float(member.get("scale", member["scale_training"]))

    def combine(self, feature_z: pd.Series, head_z: pd.Series) -> pd.Series:
        """The ensemble's z-score from its members'."""
        combined = self.weights[0] * feature_z + self.weights[1] * head_z
        spread = self.description["combined"]
        return (combined - spread["mean"]) / (spread["sd"] if spread["sd"] > 0 else 1.0)

    def head_z(self, raw: pd.Series) -> pd.Series:
        """The head's z-score from its raw page score, the mean of the page's patch scores."""
        return (raw - self.head.score_mean) / (self.head.score_sd if self.head.score_sd > 0 else 1.0)

    def scores(self, features: pd.DataFrame, hashes: dict[str, str], head_raw: pd.Series) -> pd.DataFrame:
        """Each page's z-score by both members and the ensemble, and its expected Elo score when the ensemble has an
        Elo scale. `features` is pages × columns, `hashes` their families' settings hashes, `head_raw` each page's
        raw head score."""
        first, second = self.names
        table = pd.DataFrame(index=features.index)
        table[first] = self.features.scores(features, hashes)
        table[second] = self.head_z(head_raw.reindex(features.index))
        table["ensemble"] = self.combine(table[first], table[second])
        if self.elo_scale is not None:
            table["expected_elo"] = self.elo_scale(table["ensemble"])
        return table

    def probability(self, first: pd.Series | dict, second: pd.Series | dict) -> dict[str, float]:
        """The probability that the first page is judged more complex than the second, by each member and the
        ensemble, from their z-scores (as `scores` gives them)."""
        return {
            name: float(metrics.sigmoid(self.scale(name) * (float(first[name]) - float(second[name]))))
            for name in (*self.names, "ensemble")
        }
