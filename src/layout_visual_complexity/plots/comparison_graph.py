"""The comparison graph as a figure: images as nodes placed by a seeded force layout, judged pairs as edges."""

import numpy as np
import pandas as pd


def force_layout(first: np.ndarray, second: np.ndarray, items: int, seed: int, iterations: int = 300) -> np.ndarray:
    """Positions of `items` nodes joined by edges (`first`[e], `second`[e]): Fruchterman and Reingold's layout.

    Nodes repel each other with a force k²/d and edges pull with d²/k, where k is the ideal distance; each step
    moves a node at most as far as a temperature that cools linearly to zero. Seeded, so the layout is the same on
    every run.
    """
    rng = np.random.default_rng(seed)
    position = rng.uniform(-1.0, 1.0, size=(items, 2))
    if items < 2:
        return position
    ideal = np.sqrt(4.0 / items)
    temperature = 0.2
    cooling = temperature / iterations
    for _ in range(iterations):
        delta = position[:, None, :] - position[None, :, :]
        distance = np.maximum(np.sqrt((delta**2).sum(axis=-1)), 1e-3)
        np.fill_diagonal(distance, np.inf)
        displacement = (ideal**2 * delta / distance[..., None] ** 2).sum(axis=1)
        edge = position[first] - position[second]
        length = np.maximum(np.sqrt((edge**2).sum(axis=-1)), 1e-3)
        pull = edge * (length / ideal)[:, None]
        np.add.at(displacement, first, -pull)
        np.add.at(displacement, second, pull)
        size = np.maximum(np.sqrt((displacement**2).sum(axis=-1)), 1e-12)
        position += displacement / size[:, None] * np.minimum(size, temperature)[:, None]
        temperature -= cooling
    return position


def plot_comparison_graph(
    judgments: pd.DataFrame, images: list[str], scores: pd.Series, anchors: tuple[str, ...], seed: int
):
    """Nodes coloured by score and sized by comparisons, anchors outlined; edges for the pairs judged. Full width."""
    from matplotlib.collections import LineCollection

    from layout_visual_complexity.plots import style

    index = {image: number for number, image in enumerate(images)}
    left = judgments["left"].map(index).to_numpy(dtype=np.intp)
    right = judgments["right"].map(index).to_numpy(dtype=np.intp)
    pairs = np.unique(np.stack([np.minimum(left, right), np.maximum(left, right)], axis=1), axis=0)
    position = force_layout(pairs[:, 0], pairs[:, 1], len(images), seed)
    comparisons = np.bincount(np.concatenate([left, right]), minlength=len(images))
    size = 3 + 30 * comparisons / max(comparisons.max(), 1)
    anchor = np.isin(np.asarray(images, dtype=object), list(anchors))

    figure = style.new_figure("full", 12.5)
    axes = figure.subplots()
    segments = np.stack([position[pairs[:, 0]], position[pairs[:, 1]]], axis=1)
    axes.add_collection(LineCollection(segments, colors=style.SECONDARY, linewidths=0.5, alpha=0.12, zorder=1))
    points = axes.scatter(
        position[:, 0],
        position[:, 1],
        c=scores.reindex(images).to_numpy(dtype=float),
        s=size,
        cmap=style.score_colormap(),
        edgecolors=np.where(anchor, style.TEXT, "white"),
        linewidths=np.where(anchor, 0.8, 0.5),
        zorder=2,
    )
    axes.set_aspect("equal")
    axes.axis("off")
    bar = figure.colorbar(points, ax=axes, fraction=0.04, pad=0.02, shrink=0.7)
    bar.set_label("Elo score (mean over permutations)")
    bar.outline.set_linewidth(0.6)
    return figure
