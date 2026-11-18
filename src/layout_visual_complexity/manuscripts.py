"""Manuscripts seen through their pages' scores: each one's profile along its pages, its mean and the spread of its
pages' scores with intervals from resampling its pages, the overview of a collection, and the collection's table
beside its catalogue's facts.

A profile shows a manuscript's pages in its order, each coloured by its predicted score on the Elo scale, the scores
with their rolling median, and its three most and three least complex pages. The overview sets each manuscript's mean
score against the interquartile range of its pages' scores: how complex it is on average, and how varied. The
readings test what the overview suggests against the catalogue: the number of hands against the mean and the
spread, composite volumes against homogeneous ones, and the colour measures against the catalogue's colours.
"""

import io
import math
import re
import zlib
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from layout_visual_complexity.ranking.reliability import spearman

YEAR = re.compile(r"(?<!\d)1[0-9]{3}(?!\d)")
SCORE = "expected_elo"


# The catalogue.


def read_catalogue(path: Path) -> pd.DataFrame:
    """A collection's catalogue facts (`#` lines are comments), indexed by manuscript id."""
    table = pd.read_csv(path, sep="\t", comment="#", dtype=str, keep_default_na=False).set_index("manuscript")
    for column in ("folios", "hands", "texts_coarse", "texts_fine", "colours"):
        if column in table.columns:
            table[column] = pd.to_numeric(table[column], errors="coerce")
    if "homogeneous" in table.columns:
        table["homogeneous"] = pd.to_numeric(table["homogeneous"], errors="coerce")
    return table


def short_date(text: str) -> str:
    """A catalogue's date, shortened: the earliest and latest year it names, else its centuries as written."""
    years = sorted(int(year) for year in YEAR.findall(str(text)))
    if not years:
        return " ".join(str(text).split())
    return str(years[0]) if years[0] == years[-1] else f"{years[0]}–{years[-1]}"


CENTURY = re.compile(r"^(?P<century>[IVX]+)\.?\s*(?P<part>in\.|med\.|ex\.|[12](?![0-9]))?\s*(?P<rest>.*)$")
CENTURY_PARTS = {
    None: "{n} century",
    "in.": "early {n} century",
    "med.": "mid-{n} century",
    "ex.": "late {n} century",
    "1": "first half of the {n} century",
    "2": "second half of the {n} century",
}
ROMAN = {"I": 1, "V": 5, "X": 10}


def roman(text: str) -> int:
    values = [ROMAN[character] for character in text]
    return sum(
        -value if value < following else value for value, following in zip(values, [*values[1:], 0], strict=True)
    )


def ordinal(number: int) -> str:
    return f"{number}{'th' if 10 <= number % 100 <= 20 else {1: 'st', 2: 'nd', 3: 'rd'}.get(number % 10, 'th')}"


def listed(items: Sequence[str]) -> str:
    """`a, b and c`."""
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + f" and {items[-1]}"


def date_part_in_words(part: str) -> str:
    """One part of a catalogue's date in words: `XV2 (parts dated 1461, 1491)` → `second half of the 15th century
    (parts dated 1461 and 1491)`; years and spans of years stay as they are."""
    part = " ".join(part.split()).strip(" ,")
    bracket = re.search(r"\((.*)\)\s*$", part)
    note = ""
    if bracket:
        part, note = part[: bracket.start()].strip(), bracket.group(1)
        items = [item.strip() for item in re.split(r",\s*|\s+and\s+", note) if item.strip()]
        lead = re.match(r"^((?:a |one )?parts? dated )(.*)$", items[0]) if items else None
        if lead and len(items) > 1:
            note = lead.group(1) + listed([lead.group(2), *items[1:]])
        elif note == "med.?":
            note = "perhaps mid-century"
    if "/" in part and all(CENTURY.match(piece) for piece in part.split("/")):
        first, second = (ordinal(roman(piece.strip(". "))) for piece in part.split("/"))
        words = f"turn of the {first} and {second} centuries"
    elif found := CENTURY.match(part):
        words = CENTURY_PARTS[found.group("part")].format(n=ordinal(roman(found.group("century"))))
        words = f"{words} {found.group('rest')}".strip()
    else:
        words = part
    return f"{words} ({note})" if note else words


def date_in_words(text: str) -> str:
    """A catalogue's date in words, its parts in its order: `XV in. + XV2` → `early 15th century and second half of
    the 15th century`."""
    parts = [date_part_in_words(part) for part in str(text).split("+") if part.strip(" ,")]
    return listed(parts) if parts else ""


def languages(text: str) -> str:
    """ISO 639-3 codes as a list in words: `lat|ces` → `lat, ces`."""
    return ", ".join(code for code in str(text).split("|") if code)


def shelfmark_order(catalogue: pd.DataFrame) -> list[str]:
    """Manuscripts in the catalogue's order: by catalogue number (`K5` before `K12`), else as listed."""
    if "catalogue_number" not in catalogue.columns:
        return list(catalogue.index)
    numbers = catalogue["catalogue_number"].str.extract(r"(\d+)")[0].astype(float)
    return list(numbers.sort_values(kind="stable").index)


# A manuscript's pages.


def interquartile_range(values: np.ndarray) -> float:
    """The 75th percentile less the 25th, interpolated linearly."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return math.nan
    low, high = np.percentile(values, [25, 75])
    return float(high - low)


def rolling_median(values: Sequence[float], window: int = 9) -> np.ndarray:
    """The median over `window` pages centred on each page; at the ends, over the pages there are."""
    series = pd.Series(np.asarray(values, dtype=float))
    return series.rolling(window, center=True, min_periods=1).median().to_numpy()


def extremes(scores: pd.Series, count: int = 3) -> tuple[list[str], list[str]]:
    """The `count` most and least complex pages (by score, then by their place for ties), each in the
    manuscript's order as `scores` gives it."""
    place = pd.Series(np.arange(len(scores)), index=scores.index)
    ranked = sorted(scores.index, key=lambda page: (-scores[page], place[page]))
    most = ranked[:count]
    least = [page for page in reversed(ranked) if page not in most][:count]
    return sorted(most, key=place.get), sorted(least, key=place.get)


def generator(seed: int, name: str) -> np.random.Generator:
    """Draws for one manuscript, the same whatever the others or their order."""
    return np.random.default_rng([seed, zlib.crc32(name.encode())])


@dataclass(frozen=True)
class Resampled:
    """A manuscript's statistics in each resample of its pages: the mean and the interquartile range."""

    means: np.ndarray
    ranges: np.ndarray


def resample_pages(values: np.ndarray, resamples: int, rng: np.random.Generator) -> Resampled:
    """The mean and interquartile range over resamples of the pages, drawn with replacement."""
    values = np.asarray(values, dtype=float)
    draws = rng.integers(0, len(values), size=(resamples, len(values)))
    drawn = values[draws]
    low, high = np.percentile(drawn, [25, 75], axis=1)
    return Resampled(drawn.mean(axis=1), high - low)


def manuscript_statistics(
    scores: pd.DataFrame, column: str = SCORE, resamples: int = 1000, seed: int = 1
) -> tuple[pd.DataFrame, dict[str, Resampled]]:
    """Per manuscript (`document`): its pages, mean, quartiles, interquartile range, and the 95% intervals of the
    mean and the range from resampling its pages, taken in their order (`order`, then `page`), so the intervals
    don't depend on the order rows come in. Returns the table and each manuscript's resamples."""
    rows, drawn = [], {}
    for document, group in scores.groupby("document", sort=False):
        group = group.sort_values([name for name in ("order", "page") if name in group.columns], kind="stable")
        values = group[column].to_numpy(dtype=float)
        values = values[np.isfinite(values)]
        found = resample_pages(values, resamples, generator(seed, str(document)))
        drawn[str(document)] = found
        q25, median, q75 = np.percentile(values, [25, 50, 75])
        rows.append(
            {
                "manuscript": str(document),
                "pages": len(values),
                "mean": float(values.mean()),
                "mean_low": float(np.percentile(found.means, 2.5)),
                "mean_high": float(np.percentile(found.means, 97.5)),
                "q25": float(q25),
                "median": float(median),
                "q75": float(q75),
                "iqr": float(q75 - q25),
                "iqr_low": float(np.percentile(found.ranges, 2.5)),
                "iqr_high": float(np.percentile(found.ranges, 97.5)),
            }
        )
    return pd.DataFrame(rows).set_index("manuscript"), drawn


def colour_pages(features: pd.DataFrame) -> pd.Series:
    """Whether each page has colour: a rubric line (at least half of its ink chromatic) or a coloured initial, by
    the colour family's measures; missing where the family wasn't measured."""
    needed = ["Colour__Lines__rubric_share", "Colour__ColouredInitials__count"]
    if any(column not in features.columns for column in needed):
        return pd.Series(np.nan, index=features.index)
    rubric, initials = (features[column] for column in needed)
    found = (rubric.fillna(0) > 0) | (initials.fillna(0) >= 1)
    return found.astype(float).where(rubric.notna() | initials.notna())


# The readings.


def bootstrap_rho(x: pd.Series, y: pd.Series, resamples: int, seed: int, name: str) -> dict[str, float]:
    """Spearman's ρ of two columns over the manuscripts, with its 95% interval from resampling the manuscripts."""
    both = pd.concat([x.rename("x"), y.rename("y")], axis=1).dropna()
    xs, ys = both["x"].to_numpy(dtype=float), both["y"].to_numpy(dtype=float)
    rng = generator(seed, name)
    values = []
    for _ in range(resamples):
        chosen = rng.integers(0, len(xs), len(xs))
        if len(np.unique(xs[chosen])) > 1 and len(np.unique(ys[chosen])) > 1:
            values.append(spearman(xs[chosen], ys[chosen]))
    low, high = np.percentile(values, [2.5, 97.5]) if values else (math.nan, math.nan)
    return {"value": spearman(xs, ys), "low": float(low), "high": float(high), "manuscripts": len(both)}


def group_difference(values: pd.Series, groups: pd.Series, resamples: int, seed: int, name: str) -> dict[str, Any]:
    """The mean of `values` in group 0 (composite) less group 1 (homogeneous), with its 95% interval from
    resampling the manuscripts within each group; manuscripts without a group are left out."""
    both = pd.concat([values.rename("value"), groups.rename("group")], axis=1).dropna()
    composite = both.loc[both["group"] == 0, "value"].to_numpy(dtype=float)
    homogeneous = both.loc[both["group"] == 1, "value"].to_numpy(dtype=float)
    rng = generator(seed, name)
    differences = [
        rng.choice(composite, len(composite)).mean() - rng.choice(homogeneous, len(homogeneous)).mean()
        for _ in range(resamples)
    ]
    low, high = np.percentile(differences, [2.5, 97.5])
    return {
        "value": float(composite.mean() - homogeneous.mean()),
        "low": float(low),
        "high": float(high),
        "composite_mean": float(composite.mean()),
        "homogeneous_mean": float(homogeneous.mean()),
        "composite_median": float(np.median(composite)),
        "homogeneous_median": float(np.median(homogeneous)),
        "composite": len(composite),
        "homogeneous": len(homogeneous),
        "left_out": sorted(set(values.index) - set(both.index)),
    }


def places(statistics: pd.DataFrame, drawn: dict[str, Resampled], which: Sequence[str]) -> list[dict[str, Any]]:
    """Each manuscript's place among them all by interquartile range, 1 the most varied, with its 95% interval
    from the resamples of every manuscript's pages (the n-th resample of each taken together)."""
    names = list(statistics.index)
    ranges = np.column_stack([drawn[name].ranges for name in names])
    ranks = (-ranges).argsort(axis=1).argsort(axis=1) + 1
    observed = statistics["iqr"].rank(ascending=False, method="min").astype(int)
    found = []
    for name in which:
        if name not in names:
            continue
        column = ranks[:, names.index(name)]
        low, high = np.percentile(column, [2.5, 97.5])
        found.append({"manuscript": name, "place": int(observed[name]), "low": float(low), "high": float(high),
                      "of": len(names), "iqr": float(statistics.at[name, "iqr"])})  # fmt: skip
    return found


def readings(
    statistics: pd.DataFrame,
    drawn: dict[str, Resampled],
    catalogue: pd.DataFrame,
    colour_share: pd.Series,
    named: Sequence[str],
    resamples: int,
    seed: int,
) -> pd.DataFrame:
    """The article's readings as numbers, each with its 95% interval: ρ of the number of hands with the mean and with
    the interquartile range, the composite volumes' range less the homogeneous ones', the named manuscripts' places
    by range, and ρ of the colour share with the catalogue's colours."""
    facts = catalogue.reindex(statistics.index)
    rows = []

    def add(reading: str, found: dict[str, Any], **more: Any) -> None:
        rows.append({"reading": reading, **{key: found.get(key) for key in ("value", "low", "high")}, **more})

    for column, words in (("mean", "mean"), ("iqr", "interquartile range")):
        found = bootstrap_rho(facts["hands"], statistics[column], resamples, seed, f"hands-{column}")
        add(f"Spearman's rho, hands and {words}", found, manuscripts=found["manuscripts"])
    found = group_difference(statistics["iqr"], facts["homogeneous"], resamples, seed, "composite-iqr")
    add("Interquartile range, composite less homogeneous (mean)", found, manuscripts=found["composite"]
        + found["homogeneous"], note=f"composite {found['composite']} (mean {found['composite_mean']:.1f}, median "
        f"{found['composite_median']:.1f}), homogeneous {found['homogeneous']} (mean {found['homogeneous_mean']:.1f}, "
        f"median {found['homogeneous_median']:.1f}); left out: {', '.join(found['left_out']) or 'none'}")  # fmt: skip
    for place in places(statistics, drawn, named):
        rows.append({"reading": f"Place by interquartile range, {place['manuscript']} (1 the most varied)",
                     "value": place["place"], "low": place["low"], "high": place["high"],
                     "manuscripts": place["of"], "note": f"interquartile range {place['iqr']:.1f}"})  # fmt: skip
    if colour_share.notna().any():
        found = bootstrap_rho(facts["colours"], colour_share.reindex(statistics.index), resamples, seed, "colour")
        add("Spearman's rho, pages with colour and the catalogue's colours", found, manuscripts=found["manuscripts"])
    return pd.DataFrame(rows)


# Thumbnails.


def iiif_thumbnail(service: str, size: int) -> str:
    """A IIIF Image API request for the image fitted in a `size` × `size` square."""
    return f"{service.rstrip('/')}/full/!{size},{size}/0/default.jpg"


def fetch_thumbnail(service: str, size: int, client: Any, cache: Any) -> np.ndarray:
    """A page's thumbnail from its image service, through the thumbnail cache."""
    import matplotlib.image

    from layout_visual_complexity.download import cached_get

    data = cached_get(client, cache, iiif_thumbnail(service, size))
    return matplotlib.image.imread(io.BytesIO(data), format="jpg")
