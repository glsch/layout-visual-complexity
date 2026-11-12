"""The contract every feature family follows.

A family names its columns `Family__Feature__Flavour`, declares its scope (`page`, `region` or `line`: the level its
rows come at) and what it needs prepared (the image, the folio crop, the regions, the lines, YOLO's lines), and says
how its rows are aggregated to the page, column by column. Its settings are a frozen dataclass; their hash, with the
family's class and version, identifies how its columns were computed, so a trained model can refuse features computed
differently.

Families are picked by short name (the built-in ones) or by Python path (`package.module.Class`).
"""

import dataclasses
import hashlib
import importlib
import json
import math
import re
import typing
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, ClassVar, Literal

import numpy as np

Scope = Literal["page", "region", "line"]
Need = Literal["image", "folio_crop", "regions", "lines", "yolo_lines"]
Aggregation = Literal["median", "mean", "sum", "min", "max"]

COLUMN = re.compile(r"^[A-Z][A-Za-z0-9]*__[A-Z][A-Za-z0-9]*__[a-z0-9]+(?:_[a-z0-9]+)*$")
BUILT_IN = {
    "geometry": "layout_visual_complexity.features.geometry.Geometry",
    "layout": "layout_visual_complexity.features.layout.Layout",
}
AGGREGATIONS = {"median": np.median, "mean": np.mean, "sum": np.sum, "min": np.min, "max": np.max}


class FeatureError(ValueError):
    """A family that can't be loaded or doesn't keep the contract. The message says which and why."""


@dataclass(frozen=True)
class Group:
    """One feature of a family with its flavours, as a table of the family's features describes it: the family's
    and the feature's names in words, the columns' common prefix (`Family__Feature`), the flavours, and a
    definition."""

    family: str
    feature: str
    prefix: str
    flavours: tuple[str, ...]
    definition: str

    @property
    def columns(self) -> list[str]:
        return [f"{self.prefix}__{flavour}" for flavour in self.flavours]


@dataclass(frozen=True)
class NoSettings:
    """A family without settings."""


class Family:
    """A family of features. Subclasses set `name`, `scope`, `version` and `Settings`, and implement `groups` and
    `measure`; `needs` and `aggregation` may depend on the settings.

    `measure` returns rows: one for the page, or one per region or line measured (a family may add a page row to its
    region rows). Each row maps some of the family's columns to numbers; the machinery aggregates the rows to the
    page, column by column, as `aggregation` says, leaving out missing values.
    """

    name: ClassVar[str] = ""
    scope: ClassVar[Scope] = "page"
    version: ClassVar[str] = "1"
    Settings: ClassVar[type] = NoSettings

    def __init__(self, settings: object | None = None) -> None:
        self.settings = settings if settings is not None else self.Settings()

    @property
    def needs(self) -> frozenset[Need]:
        return frozenset()

    def groups(self) -> list[Group]:
        raise NotImplementedError

    def columns(self) -> list[str]:
        return [column for group in self.groups() for column in group.columns]

    def aggregation(self, column: str) -> Aggregation:
        return "median"

    def words(self, column: str) -> str:
        """What a column measures, in plain words for a caption; by default its feature's definition."""
        for group in self.groups():
            if column in group.columns:
                return group.definition
        raise FeatureError(f"{self.name} has no column {column!r}")

    def measure(self, page: Any) -> list[dict[str, float]]:
        raise NotImplementedError

    @classmethod
    def path(cls) -> str:
        return f"{cls.__module__}.{cls.__qualname__}"

    def settings_dict(self) -> dict[str, Any]:
        return plain_settings(self.settings)

    def settings_hash(self) -> str:
        """What decides the family's values: its class, version and settings, as 16 hexadecimal digits."""
        payload = {"class": self.path(), "version": self.version, "settings": self.settings_dict()}
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]

    def describe(self) -> dict[str, Any]:
        """The family as a run records it."""
        return {
            "name": self.name,
            "class": self.path(),
            "version": self.version,
            "scope": self.scope,
            "settings": self.settings_dict(),
            "hash": self.settings_hash(),
        }


def plain_settings(settings: object) -> dict[str, Any]:
    """Settings as JSON holds them: tuples as lists."""
    if not dataclasses.is_dataclass(settings):
        raise FeatureError(f"a family's settings must be a dataclass, not {type(settings).__name__}")
    return json.loads(json.dumps(dataclasses.asdict(settings), default=str))  # type: ignore[arg-type]


def build_settings(settings_type: type, parameters: Mapping[str, Any] | None) -> object:
    """A family's settings from configuration values: unknown names refused, lists turned into tuples where the
    settings hold tuples."""
    parameters = dict(parameters or {})
    fields = {field.name: field for field in dataclasses.fields(settings_type)}
    unknown = sorted(set(parameters) - set(fields))
    if unknown:
        raise FeatureError(f"{settings_type.__name__} has no setting {', '.join(unknown)}")
    hints = typing.get_type_hints(settings_type)
    for name, value in parameters.items():
        if typing.get_origin(hints[name]) is tuple and isinstance(value, list):
            parameters[name] = tuple(value)
    return settings_type(**parameters)


def family_class(name: str) -> type[Family]:
    """A family's class: a built-in one by short name, any other by Python path."""
    path = BUILT_IN.get(name, name)
    module_name, _, attribute = path.rpartition(".")
    if not module_name:
        known = ", ".join(sorted(BUILT_IN))
        raise FeatureError(f"unknown feature family {name!r}: give one of {known}, or a Python path")
    try:
        found = getattr(importlib.import_module(module_name), attribute)
    except (ImportError, AttributeError) as error:
        raise FeatureError(f"can't load the feature family {name!r}: {error}") from None
    if not (isinstance(found, type) and issubclass(found, Family)):
        raise FeatureError(f"{name!r} is not a feature family (a subclass of {Family.path()})")
    return found


def load_family(name: str, parameters: Mapping[str, Any] | None = None) -> Family:
    """A family, by short name or Python path, with its settings from `parameters`."""
    cls = family_class(name)
    return cls(build_settings(cls.Settings, parameters))


def load_families(names: Sequence[str], parameters: Mapping[str, Mapping[str, Any]] | None = None) -> list[Family]:
    """The families named, each once, with their settings: `parameters` holds each family's by the name it's given
    under. Checks that their columns are well formed and that no two families share one."""
    parameters = dict(parameters or {})
    unused = sorted(set(parameters) - set(names))
    if unused:
        raise FeatureError(f"settings given for families not chosen: {', '.join(unused)}")
    families = [load_family(name, parameters.get(name)) for name in dict.fromkeys(names)]
    check_columns(families)
    return families


def check_columns(families: Sequence[Family]) -> None:
    """Every column named `Family__Feature__Flavour`, and no column twice."""
    seen: dict[str, str] = {}
    for family in families:
        for column in family.columns():
            if not COLUMN.match(column):
                raise FeatureError(f"{family.name}: the column {column!r} isn't named Family__Feature__flavour")
            if column in seen:
                raise FeatureError(f"the column {column!r} comes from both {seen[column]} and {family.name}")
            seen[column] = family.name


def aggregate(family: Family, rows: Sequence[Mapping[str, float]]) -> dict[str, float]:
    """A family's rows as one page row: each column by its aggregation over the rows that have a finite value; NaN
    where none has."""
    columns = family.columns()
    known = set(columns)
    for row in rows:
        unknown = set(row) - known
        if unknown:
            raise FeatureError(f"{family.name} measured columns it doesn't declare: {', '.join(sorted(unknown))}")
    if family.scope == "page" and len(rows) > 1:
        raise FeatureError(f"{family.name} has page scope but measured {len(rows)} rows")
    page: dict[str, float] = {}
    for column in columns:
        values = [float(row[column]) for row in rows if column in row and row[column] is not None]
        values = [value for value in values if math.isfinite(value)]
        page[column] = float(AGGREGATIONS[family.aggregation(column)](values)) if values else math.nan
    return page
