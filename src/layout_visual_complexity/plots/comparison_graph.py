"""The comparison graph as a figure: images as nodes placed by a seeded force layout, judged pairs as edges."""

import numpy as np
import pandas as pd

# The campaign of the sessions in which both batches of images were judged, as the published dataset names it.
BRIDGE = "bridge"


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


def lengthwise(position: np.ndarray) -> np.ndarray:
    """Positions turned about their centre so that their longest extent, their first principal axis, runs along x."""
    centred = position - position.mean(axis=0)
    if len(centred) < 2:
        return centred
    _, vectors = np.linalg.eigh(np.cov(centred.T))
    major = vectors[:, -1] * (1 if vectors[np.argmax(np.abs(vectors[:, -1])), -1] > 0 else -1)
    return centred @ np.array([[major[0], -major[1]], [major[1], major[0]]])


def graph_layout(judgments: pd.DataFrame, images: list[str], seed: int) -> tuple[np.ndarray, np.ndarray]:
    """The judged pairs (rows of two image numbers, in the order of `images`) and the nodes' positions, lengthwise."""
    index = {image: number for number, image in enumerate(images)}
    left = judgments["left"].map(index).to_numpy(dtype=np.intp)
    right = judgments["right"].map(index).to_numpy(dtype=np.intp)
    pairs = np.unique(np.stack([np.minimum(left, right), np.maximum(left, right)], axis=1), axis=0)
    return pairs, lengthwise(force_layout(pairs[:, 0], pairs[:, 1], len(images), seed))


def plot_comparison_graph(
    judgments: pd.DataFrame,
    images: list[str],
    scores: pd.Series,
    anchors: tuple[str, ...],
    batches: pd.Series,
    seed: int,
):
    """Two panels with the same layout of nodes, sized by comparisons, anchors ringed: (a) coloured by batch, the
    pairs judged in the bridge sessions apart from the others; (b) coloured by score. Full width.

    `batches` gives each image's batch, its categories in their order; the batches take the style's batch colours.
    """
    from matplotlib.collections import LineCollection
    from matplotlib.lines import Line2D

    from layout_visual_complexity.plots import style

    pairs, position = graph_layout(judgments, images, seed)
    index = {image: number for number, image in enumerate(images)}
    left = judgments["left"].map(index).to_numpy(dtype=np.intp)
    right = judgments["right"].map(index).to_numpy(dtype=np.intp)
    comparisons = np.bincount(np.concatenate([left, right]), minlength=len(images))
    size = 2 + 22 * comparisons / max(comparisons.max(), 1)
    anchor = np.isin(np.asarray(images, dtype=object), list(anchors))
    bridged = judgments["campaign"].eq(BRIDGE).to_numpy() if "campaign" in judgments else np.zeros(len(left), bool)
    bridge_pairs = {(min(a, b), max(a, b)) for a, b, here in zip(left, right, bridged, strict=True) if here}
    in_bridge = np.array([(a, b) in bridge_pairs for a, b in pairs.tolist()], dtype=bool)
    segments = np.stack([position[pairs[:, 0]], position[pairs[:, 1]]], axis=1)

    span = np.ptp(position, axis=0)
    panel_width = 7.1  # cm
    graph_height = min(panel_width * span[1] / max(span[0], 1e-9), 6.0) + 0.3
    legend_height, title = 1.45, 0.45
    width_cm = style.FULL_WIDTH / style.CENTIMETRE
    height_cm = title + graph_height + legend_height
    figure = style.new_figure("full", height_cm, constrained=False)
    axes = [
        figure.add_axes((left / width_cm, legend_height / height_cm, panel_width / width_cm, graph_height / height_cm))
        for left in (0.15, width_cm - 0.15 - panel_width)
    ]
    batch_of = batches.reindex(images)
    names = [name for name in batches.cat.categories if (batch_of == name).any()]
    colour_of = dict(zip(names, style.BATCHES, strict=False))
    node_colours = [
        np.array([colour_of.get(name, style.SECONDARY) for name in batch_of], dtype=object),
        scores.reindex(images).to_numpy(dtype=float),
    ]
    for number, (panel, colours) in enumerate(zip(axes, node_colours, strict=True)):
        edges = segments[~in_bridge] if number == 0 else segments
        panel.add_collection(LineCollection(edges, colors=style.SECONDARY, linewidths=0.5, alpha=0.12, zorder=1))
        if number == 0 and in_bridge.any():
            bridge_edges = LineCollection(
                segments[in_bridge], colors=style.HIGHLIGHT, linewidths=0.5, alpha=0.25, zorder=1
            )
            panel.add_collection(bridge_edges)
        options = {"cmap": style.score_colormap()} if number == 1 else {}
        points = panel.scatter(
            position[:, 0], position[:, 1], c=colours, s=size, edgecolors="white", linewidths=0.3, zorder=2, **options
        )
        panel.scatter(
            position[anchor, 0], position[anchor, 1], s=size[anchor] * 1.9, facecolors="none",
            edgecolors=style.TEXT, linewidths=0.45, zorder=3,
        )  # fmt: skip
        panel.set_aspect("equal", adjustable="datalim")
        panel.axis("off")
        figure.text(
            (0.15 if number == 0 else width_cm - 0.15 - panel_width) / width_cm, 1 - 0.05 / height_cm,
            f"({chr(ord('a') + number)})", ha="left", va="top", fontweight="bold",
        )  # fmt: skip

    def node(colour: str) -> Line2D:
        return Line2D([], [], marker="o", linestyle="", markersize=4, markerfacecolor=colour, markeredgecolor="white")

    handles = [node(colour_of[name]) for name in names]
    labels = [f"{name} ({int((batch_of == name).sum())})" for name in names]
    handles += [
        Line2D([], [], color=style.SECONDARY, linewidth=1, alpha=0.6),
        Line2D([], [], color=style.HIGHLIGHT, linewidth=1, alpha=0.8),
        Line2D([], [], marker="o", linestyle="", markersize=5, markerfacecolor="none", markeredgecolor=style.TEXT,
               markeredgewidth=0.6),
    ]  # fmt: skip
    labels += ["Main campaign", "Bridge sessions", "Anchor"]
    figure.legend(
        handles, labels, loc="upper left", bbox_to_anchor=(0.1 / width_cm, (legend_height - 0.05) / height_cm),
        ncol=2, handlelength=1.4, columnspacing=1.2, borderaxespad=0, borderpad=0,
    )  # fmt: skip
    bar_axes = figure.add_axes(
        ((width_cm - 0.15 - panel_width + 1.9) / width_cm, (legend_height - 0.45) / height_cm,
         (panel_width - 2.6) / width_cm, 0.2 / height_cm)
    )  # fmt: skip
    bar = figure.colorbar(points, cax=bar_axes, orientation="horizontal")
    bar.set_label("Elo score (mean over permutations)")
    bar.outline.set_linewidth(0.6)
    return figure
