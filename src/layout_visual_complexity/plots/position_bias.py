"""Position bias as a funnel: each annotator's share of left choices against their number of judgments, within the
bounds of chance around one half."""

import numpy as np
import pandas as pd


def plot_position_bias_funnel(annotators: pd.DataFrame, overall: float):
    """`annotators` has one row per annotator: `judgments`, `left` (the share choosing the left image) and `outside`
    (whether it lies outside the bounds of chance). The bounds, the overall share as a line, the annotators outside in
    a second colour. Half width."""
    from layout_visual_complexity.plots import style
    from layout_visual_complexity.ranking.reliability import chance_bounds

    judgments = annotators["judgments"].to_numpy(dtype=float)
    outside = annotators["outside"].to_numpy(dtype=bool)
    figure = style.new_figure("half", 8.4)
    axes = figure.subplots()
    grid = np.geomspace(1, max(judgments.max(), 1) * 1.5, 300)
    low, high = chance_bounds(grid)
    axes.fill_between(grid, np.clip(low, 0, 1), np.clip(high, 0, 1), color=style.GRID, linewidth=0, zorder=1,
                      label="Bounds of chance")  # fmt: skip
    axes.axhline(0.5, color=style.METHODS["chance"], linewidth=0.6, linestyle="--", zorder=2)
    axes.axhline(overall, color=style.METHODS["vit"], linewidth=0.8, zorder=2, label="Overall share")
    shares = annotators["left"].to_numpy(dtype=float)
    axes.scatter(judgments[~outside], shares[~outside], s=9, color=style.SECONDARY, linewidths=0, zorder=3,
                 label="Annotators")  # fmt: skip
    axes.scatter(judgments[outside], shares[outside], s=11, color=style.HIGHLIGHT, linewidths=0, zorder=4,
                 label="Outside the bounds")  # fmt: skip
    axes.set_xscale("log")
    axes.set_xlim(0.7, grid[-1])
    ticks = [10**power for power in range(int(np.log10(grid[-1])) + 1)]
    axes.set_xticks(ticks, [f"{tick:,}" for tick in ticks])
    axes.minorticks_off()
    axes.set_ylim(-0.03, 1.03)
    axes.set_xlabel("Judgments by the annotator")
    axes.set_ylabel("Share choosing the left image")
    style.light_grid(axes)
    axes.legend(loc="upper center", bbox_to_anchor=(0.45, -0.18), ncol=2, handlelength=1.2, columnspacing=0.8)
    return figure
