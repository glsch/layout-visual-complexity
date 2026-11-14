"""The comparison graph: images are nodes, judged pairs are edges."""

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class GraphSummary:
    """What the comparison graph looks like.

    `density` is the share of all image pairs judged at least once. Anchors are the images compared more than
    `anchor_factor` times the median number of comparisons.
    """

    images: int
    judgments: int
    pairs: int
    density: float
    components: int
    largest_component: int
    comparisons_min: int
    comparisons_median: float
    comparisons_mean: float
    comparisons_max: int
    anchor_factor: float
    anchors: tuple[str, ...]


def comparison_counts(judgments: pd.DataFrame, images: list[str]) -> pd.Series:
    """How often each image was shown, in the order of `images`."""
    shown = pd.concat([judgments["left"], judgments["right"]]).value_counts()
    return shown.reindex(images, fill_value=0).astype(int)


def win_counts(judgments: pd.DataFrame, images: list[str]) -> pd.Series:
    """How often each image was chosen, in the order of `images`."""
    return judgments["chosen"].value_counts().reindex(images, fill_value=0).astype(int)


def component_labels(first: np.ndarray, second: np.ndarray, items: int) -> np.ndarray:
    """Each item's connected component, numbered from 0 by its smallest item."""
    parent = np.arange(items)

    def root(item: int) -> int:
        while parent[item] != item:
            parent[item] = parent[parent[item]]
            item = parent[item]
        return item

    for a, b in zip(first.tolist(), second.tolist(), strict=True):
        ra, rb = root(a), root(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)
    roots = np.array([root(item) for item in range(items)])
    _, labels = np.unique(roots, return_inverse=True)
    return labels


def describe(judgments: pd.DataFrame, images: list[str], anchor_factor: float = 3.0) -> GraphSummary:
    """Components, density, comparisons per image and the anchors of the judgments' graph over `images`."""
    index = {image: number for number, image in enumerate(images)}
    left = judgments["left"].map(index).to_numpy(dtype=np.intp)
    right = judgments["right"].map(index).to_numpy(dtype=np.intp)
    pairs = len({(min(a, b), max(a, b)) for a, b in zip(left.tolist(), right.tolist(), strict=True)})
    labels = component_labels(left, right, len(images))
    counts = comparison_counts(judgments, images)
    median = float(counts.median())
    anchors = tuple(counts.index[counts > anchor_factor * median])
    possible = len(images) * (len(images) - 1) / 2
    return GraphSummary(
        images=len(images),
        judgments=len(judgments),
        pairs=pairs,
        density=pairs / possible if possible else float("nan"),
        components=int(labels.max() + 1) if len(images) else 0,
        largest_component=int(np.bincount(labels).max()) if len(images) else 0,
        comparisons_min=int(counts.min()),
        comparisons_median=median,
        comparisons_mean=float(counts.mean()),
        comparisons_max=int(counts.max()),
        anchor_factor=anchor_factor,
        anchors=anchors,
    )


def batches_by_first_judgment(judgments: pd.DataFrame, images: list[str], starts: dict[str, str]) -> pd.Series:
    """Each image's batch: the last of the batches, given by their first days (ISO dates, `{name: day}`), that began
    on or before the day the image was first judged. Without batches, every image is in one, `Images`. A categorical
    series, its categories in the order of their first days; images never judged are missing."""
    if not starts:
        return pd.Series(pd.Categorical(["Images"] * len(images)), index=images)
    ordered = sorted(starts.items(), key=lambda item: item[1])
    shown = pd.concat(
        [judgments[["left", "date"]].set_axis(["image", "date"], axis=1),
         judgments[["right", "date"]].set_axis(["image", "date"], axis=1)]
    )  # fmt: skip
    first = shown.groupby("image")["date"].min().reindex(images)
    days = [day for _, day in ordered]
    position = np.searchsorted(np.asarray(days, dtype=object), first.fillna("").to_numpy(dtype=object), side="right")
    names = [name for name, _ in ordered]
    batch = [names[number - 1] if number > 0 and isinstance(day, str) else None
             for number, day in zip(position, first, strict=True)]  # fmt: skip
    return pd.Series(pd.Categorical(batch, categories=names), index=images)
