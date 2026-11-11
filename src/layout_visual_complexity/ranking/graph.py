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
