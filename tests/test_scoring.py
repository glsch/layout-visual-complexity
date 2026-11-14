"""The rankers that score pages: the shipped ones, the calibrated scale, the Elo scale, leaving manuscripts out, and
the probability that one page is judged more complex than another."""

import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from layout_visual_complexity.evaluation import calibration, metrics
from layout_visual_complexity.evaluation.model import FeatureMismatch
from layout_visual_complexity.features.vit import TokenMismatch
from layout_visual_complexity.ranking.dataset import read_dataset
from layout_visual_complexity.scoring import ModelError, ScoringModels, shipped_folder
from layout_visual_complexity.train import without_manuscripts


def judged(scores: pd.Series, scale: float, count: int, seed: int) -> pd.DataFrame:
    """Judgments between random pairs of pages, the first chosen with probability sigmoid(scale × difference)."""
    rng = np.random.default_rng(seed)
    pages = scores.index.to_numpy()
    left, right = rng.choice(pages, count), rng.choice(pages, count)
    keep = left != right
    left, right = left[keep], right[keep]
    p = metrics.sigmoid(scale * (scores[left].to_numpy() - scores[right].to_numpy()))
    chosen = np.where(rng.random(len(p)) < p, left, right)
    return pd.DataFrame({"judgment": np.arange(len(chosen)).astype(str), "left": left, "right": right,
                         "chosen": chosen})  # fmt: skip


def test_the_scale_fitted_on_out_of_fold_scores_recovers_a_known_scale() -> None:
    scores = pd.Series(np.random.default_rng(0).normal(size=300), index=[f"p{i}" for i in range(300)])
    judgments = judged(scores, 1.7, 20000, 1)
    assert calibration.out_of_fold_scale(scores, judgments) == pytest.approx(1.7, abs=0.08)
    # Judgments between pages without scores are left out.
    extra = pd.DataFrame({"judgment": ["x"], "left": ["p0"], "right": ["unknown"], "chosen": ["unknown"]})
    with_extra = calibration.out_of_fold_scale(scores, pd.concat([judgments, extra], ignore_index=True))
    assert with_extra == pytest.approx(calibration.out_of_fold_scale(scores, judgments))


def test_held_out_probabilities_are_calibrated_on_judgments_the_scale_never_saw() -> None:
    pages = [f"p{i}" for i in range(200)]
    scores = pd.Series(np.random.default_rng(2).normal(size=200), index=pages)
    judgments = judged(scores, 1.2, 8000, 3)
    groups = pd.Series([f"m{i % 10}" for i in range(200)], index=pages)
    p = calibration.held_out_scale_probabilities(scores, judgments, groups)
    assert np.isfinite(p).all()
    assert calibration.calibration_error(p) < 0.03
    curve = calibration.calibration_curve(p)
    assert curve["judgments"].sum() == 2 * len(p)


def test_the_elo_scale_is_the_regression_of_the_elo_scores_on_the_scores() -> None:
    rng = np.random.default_rng(4)
    z = pd.Series(rng.normal(size=500), index=[f"p{i}" for i in range(500)])
    elo = 1500 + 170 * z + rng.normal(scale=60, size=500)
    scale = calibration.fit_elo_scale(z, elo)
    assert scale.slope == pytest.approx(170, abs=10) and scale.intercept == pytest.approx(1500, abs=8)
    assert scale.slope == pytest.approx(scale.correlation * scale.elo_sd / scale.score_sd)
    assert float(scale(1.0)) == pytest.approx(scale.intercept + scale.slope)
    assert calibration.EloScale.from_dict(scale.to_dict()) == scale


def test_leaving_manuscripts_out_drops_their_pages_and_every_judgment_touching_them(dataset_folder: Path) -> None:
    dataset = read_dataset(dataset_folder)
    kept = without_manuscripts(dataset, ["msa"])
    assert not kept.images["manuscript"].eq("msa").any()
    left_out = set(dataset.images.loc[dataset.images["manuscript"] == "msa", "image"])
    touching = dataset.judgments["left"].isin(left_out) | dataset.judgments["right"].isin(left_out)
    assert touching.any()
    assert len(kept.judgments) == int((~touching).sum())
    assert not (kept.judgments["left"].isin(left_out) | kept.judgments["right"].isin(left_out)).any()
    assert "msa" not in set(kept.manuscripts["manuscript"])
    with pytest.raises(ValueError, match="nope"):
        without_manuscripts(dataset, ["nope"])


def test_the_shipped_rankers_load_and_check_their_inputs() -> None:
    models = ScoringModels.load()
    assert models.names == ("geometry", "vit")
    assert models.description["elo"]["slope"] > 0
    assert set(models.description["trained_on"]) >= {"pages", "judgments"}
    assert models.description["trained_on"]["pages"] == 397
    assert [family["name"] for family in models.families()] == ["geometry"]
    columns = models.features.model.columns
    features = pd.DataFrame(np.zeros((2, len(columns))), index=["a", "b"], columns=columns)
    good = {"geometry": models.families()[0]["hash"]}
    head_raw = pd.Series([4.0, 5.0], index=["a", "b"])
    scores = models.scores(features, good, head_raw)
    assert list(scores.columns) == ["geometry", "vit", "ensemble", "expected_elo"]
    with pytest.raises(FeatureMismatch):
        models.scores(features, {"geometry": "0" * 16}, head_raw)
    with pytest.raises(FeatureMismatch):
        models.scores(features.drop(columns=columns[:2]), good, head_raw)
    with pytest.raises(TokenMismatch):
        models.head.check(dict(models.head.tokens) | {"long_edge": 224})


def test_rankers_whose_members_changed_are_refused(tmp_path: Path) -> None:
    folder = tmp_path / "rankers"
    shutil.copytree(shipped_folder(), folder)
    description = json.loads((folder / "ensemble.json").read_text())
    member = folder / description["members"]["geometry"]["file"]
    member.write_text(member.read_text().replace('"regularisation": 1.0', '"regularisation": 2.0'))
    with pytest.raises(ModelError, match="changed"):
        ScoringModels.load(folder)
    (folder / "ensemble.json").unlink()
    with pytest.raises(ModelError, match="ensemble.json"):
        ScoringModels.load(folder)


def test_the_probability_that_a_page_is_judged_more_complex_is_symmetric() -> None:
    models = ScoringModels.load()
    first = {"geometry": 0.8, "vit": -0.2, "ensemble": 0.35}
    second = {"geometry": -0.4, "vit": 0.6, "ensemble": 0.1}
    forward, backward = models.probability(first, second), models.probability(second, first)
    for name in ("geometry", "vit", "ensemble"):
        assert forward[name] + backward[name] == pytest.approx(1.0)
    assert forward["geometry"] > 0.5 > forward["vit"]
    assert models.probability(first, first)["ensemble"] == pytest.approx(0.5)
