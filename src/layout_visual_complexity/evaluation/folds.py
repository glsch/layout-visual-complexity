"""Cross-validation folds over pages, annotators or categories.

A fold trains on the judgments whose two pages are both training pages, and tests on those whose two pages are both
test pages; judgments across the two sides are left out. Folds by annotator split the judgments by person instead,
and keep every page on both sides.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Fold:
    """One split: the pages and the judgments (row positions in the judgment table) on each side."""

    kind: str
    repeat: int
    index: int
    label: str
    train_pages: np.ndarray
    test_pages: np.ndarray
    train: np.ndarray
    test: np.ndarray


def judgments_within(judgments: pd.DataFrame, pages: np.ndarray) -> np.ndarray:
    """The row positions of the judgments whose two pages are both in `pages`."""
    inside = judgments["left"].isin(pages).to_numpy() & judgments["right"].isin(pages).to_numpy()
    return np.flatnonzero(inside)


def assign_greedily(sizes: pd.Series, k: int, rng: np.random.Generator) -> dict[object, int]:
    """Each group's fold: groups in a seeded random order, each to the fold with the least load so far."""
    order = rng.permutation(np.array(sorted(sizes.index), dtype=object))
    load = np.zeros(k)
    assigned = {}
    for group in order:
        fold = int(np.argmin(load))
        assigned[group] = fold
        load[fold] += sizes[group]
    return assigned


def _page_fold(
    kind: str, repeat: int, index: int, label: str, pages: np.ndarray, test: np.ndarray, judgments: pd.DataFrame
) -> Fold:
    train_pages, test_pages = pages[~test], pages[test]
    return Fold(
        kind=kind,
        repeat=repeat,
        index=index,
        label=label,
        train_pages=train_pages,
        test_pages=test_pages,
        train=judgments_within(judgments, train_pages),
        test=judgments_within(judgments, test_pages),
    )


def manuscript_folds(
    pages: list[str], manuscripts: pd.Series, judgments: pd.DataFrame, k: int = 10, repeats: int = 10, seed: int = 1
) -> list[Fold]:
    """Repeated k-fold over pages, each manuscript in one fold, balanced by pages. `manuscripts` maps page → id."""
    pages_array = np.asarray(pages, dtype=object)
    groups = manuscripts.reindex(pages_array)
    if groups.isna().any():
        raise ValueError(f"{int(groups.isna().sum())} pages have no manuscript")
    sizes = groups.value_counts()
    k = max(2, min(k, len(sizes)))
    folds = []
    for repeat in range(repeats):
        assigned = assign_greedily(sizes, k, np.random.default_rng([seed, repeat]))
        fold_of_page = groups.map(assigned).to_numpy()
        for index in range(k):
            test = fold_of_page == index
            folds.append(_page_fold("manuscripts", repeat, index, str(index + 1), pages_array, test, judgments))
    return folds


def category_folds(pages: list[str], categories: pd.Series, judgments: pd.DataFrame) -> list[Fold]:
    """One fold per category: its pages are the test pages, all others train."""
    pages_array = np.asarray(pages, dtype=object)
    groups = categories.reindex(pages_array)
    folds = []
    for index, category in enumerate(sorted(groups.dropna().unique())):
        test = (groups == category).to_numpy()
        folds.append(_page_fold("categories", 0, index, str(category), pages_array, test, judgments))
    return folds


def annotator_folds(
    pages: list[str], judgments: pd.DataFrame, k: int = 10, repeats: int = 1, seed: int = 1
) -> list[Fold]:
    """Repeated k-fold over persons, balanced by judgments: a fold tests on its persons' judgments and trains on the
    others'. Every page is on both sides."""
    pages_array = np.asarray(pages, dtype=object)
    sizes = judgments["annotator"].value_counts()
    k = max(2, min(k, len(sizes)))
    folds = []
    for repeat in range(repeats):
        assigned = assign_greedily(sizes, k, np.random.default_rng([seed, repeat]))
        fold_of_judgment = judgments["annotator"].map(assigned).to_numpy()
        for index in range(k):
            folds.append(
                Fold(
                    kind="annotators",
                    repeat=repeat,
                    index=index,
                    label=str(index + 1),
                    train_pages=pages_array,
                    test_pages=pages_array,
                    train=np.flatnonzero(fold_of_judgment != index),
                    test=np.flatnonzero(fold_of_judgment == index),
                )
            )
    return folds
