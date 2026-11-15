"""The `lvc` command line. Settings come from YAML files (`--config`, several combine, later ones win) and options,
which override them."""

import dataclasses
import logging
import sys
import typing
from collections.abc import Callable

import docstring_parser
from jsonargparse import ActionConfigFile, ActionYesNo, ArgumentParser, Namespace

import layout_visual_complexity
from layout_visual_complexity import logs


def judgment_commands() -> dict[str, tuple[type, Callable[..., object]]]:
    """The judgments commands, by name: their settings and what runs them. Imported here, not at the top, so that
    worker processes, which import this module again, start quickly."""
    from layout_visual_complexity.judgments import (
        ImportSettings,
        RankSettings,
        ReportSettings,
        run_import,
        run_rank,
        run_report,
    )

    return {
        "import": (ImportSettings, run_import),
        "rank": (RankSettings, run_rank),
        "report": (ReportSettings, run_report),
    }


def segment_command() -> tuple[type, Callable[..., object]]:
    """The segment command's settings and what runs it, imported when needed."""
    from layout_visual_complexity.segment import SegmentSettings, run_segment

    return SegmentSettings, run_segment


def features_command() -> tuple[type, Callable[..., object]]:
    """The features command's settings and what runs it, imported when needed."""
    from layout_visual_complexity.features.extraction import FeaturesSettings, run_features

    return FeaturesSettings, run_features


def score_command() -> tuple[type, Callable[..., object]]:
    """The score command's settings and what runs it, imported when needed."""
    from layout_visual_complexity.score import ScoreSettings, run_score

    return ScoreSettings, run_score


def compare_command() -> tuple[type, Callable[..., object]]:
    """The compare command's settings and what runs it, imported when needed."""
    from layout_visual_complexity.compare import CompareSettings, run_compare

    return CompareSettings, run_compare


def models_command() -> tuple[type, Callable[..., object]]:
    """The models command's settings (a scoring run's) and what runs it, imported when needed."""
    from layout_visual_complexity.model_inventory import run_models
    from layout_visual_complexity.score import ScoreSettings

    return ScoreSettings, run_models


def cache_command() -> tuple[type, Callable[..., object]]:
    """The cache command's settings and what runs it, imported when needed."""
    from layout_visual_complexity.caches import CacheSettings, run_cache

    return CacheSettings, run_cache


def codebook_command() -> tuple[type, Callable[..., object]]:
    """The codebook command's settings and what runs it, imported when needed."""
    from layout_visual_complexity.codebook import CodebookSettings, run_codebook

    return CodebookSettings, run_codebook


def embed_command() -> tuple[type, Callable[..., object]]:
    """The embed command's settings and what runs it, imported when needed."""
    from layout_visual_complexity.embed import EmbedSettings, run_embed

    return EmbedSettings, run_embed


def evaluate_command() -> tuple[type, Callable[..., object]]:
    """The evaluate command's settings and what runs it, imported when needed."""
    from layout_visual_complexity.evaluate import EvaluateSettings, run_evaluate

    return EvaluateSettings, run_evaluate


def train_command() -> tuple[type, Callable[..., object]]:
    """The train command's settings and what runs it, imported when needed."""
    from layout_visual_complexity.train import TrainSettings, run_train

    return TrainSettings, run_train


def report_command() -> tuple[type, Callable[..., object]]:
    """The report command's settings and what runs it, imported when needed."""
    from layout_visual_complexity.reports import ReportSettings, run_report

    return ReportSettings, run_report


def first_paragraph(text: str | None) -> str:
    return " ".join((text or "").strip().split("\n\n")[0].split())


def settings_parser(settings: type, flags: tuple[str, ...] = ()) -> ArgumentParser:
    """A command's options from its settings dataclass; the boolean settings named in `flags` are flags (`--alto`,
    `--no_alto`) rather than options taking `true` or `false`."""
    parser = ArgumentParser(description=first_paragraph(settings.__doc__))
    parser.add_argument(
        "--config", action=ActionConfigFile, help="A YAML configuration file; give several to combine them."
    )
    parser.add_class_arguments(settings, skip=set(flags))
    documented = docstring_parser.parse(settings.__doc__ or "")
    for name in flags:
        described = next((item.description for item in documented.params if item.arg_name == name), None)
        default = next(item.default for item in dataclasses.fields(settings) if item.name == name)
        parser.add_argument(f"--{name}", action=ActionYesNo, default=default, help=described)
    return parser


def parser() -> ArgumentParser:
    """The command line: `lvc score`, `lvc compare`, `lvc models`, `lvc cache`, `lvc judgments import|rank|report`,
    `lvc segment`, `lvc codebook`, `lvc features`, `lvc embed`, `lvc evaluate`, `lvc train` and `lvc report`."""
    top = ArgumentParser(prog="lvc", description=layout_visual_complexity.__doc__)
    top.add_argument("--version", action="version", version=layout_visual_complexity.__version__)
    top.add_argument("--quiet", action="store_true", help="Show only warnings and errors.")
    commands = top.add_subcommands(dest="command", required=True)
    judgments = ArgumentParser(description="Pairwise judgments: import them, rank the images, report on them.")
    commands.add_subcommand("judgments", judgments, help="Pairwise judgments: import, rank, report.")
    actions = judgments.add_subcommands(dest="action", required=True)
    for name, (settings, _) in judgment_commands().items():
        actions.add_subcommand(name, settings_parser(settings), help=first_paragraph(settings.__doc__))
    settings, _ = segment_command()
    segment = settings_parser(settings, flags=("alto",))
    segment.add_argument(
        "given_sources",
        nargs="*",
        default=[],
        metavar="SOURCES",
        help="Sources to segment; they replace the configuration's.",
    )
    commands.add_subcommand("segment", segment, help=first_paragraph(settings.__doc__))
    settings, _ = score_command()
    score = settings_parser(settings, flags=("alto", "maps", "profiles", "local_only"))
    score.add_argument(
        "given_sources",
        nargs="*",
        default=[],
        metavar="SOURCES",
        help="Sources to score; they replace the configuration's.",
    )
    commands.add_subcommand("score", score, help=first_paragraph(settings.__doc__))
    settings, _ = compare_command()
    compare = settings_parser(settings, flags=("alto", "maps", "profiles", "local_only"))
    compare.add_argument("pages", nargs="*", default=[], metavar="PAGE", help="The two pages: first, then second.")
    commands.add_subcommand("compare", compare, help=first_paragraph(settings.__doc__))
    settings, _ = models_command()
    commands.add_subcommand(
        "models",
        settings_parser(settings, flags=("alto", "maps", "profiles", "local_only")),
        help="The models a scoring run uses: where each is, and whether it is there and the one expected.",
    )
    settings, _ = cache_command()
    commands.add_subcommand("cache", settings_parser(settings), help=first_paragraph(settings.__doc__))
    settings, _ = codebook_command()
    commands.add_subcommand("codebook", settings_parser(settings), help=first_paragraph(settings.__doc__))
    settings, _ = features_command()
    commands.add_subcommand("features", settings_parser(settings), help=first_paragraph(settings.__doc__))
    settings, _ = embed_command()
    commands.add_subcommand("embed", settings_parser(settings), help=first_paragraph(settings.__doc__))
    settings, _ = evaluate_command()
    commands.add_subcommand("evaluate", settings_parser(settings), help=first_paragraph(settings.__doc__))
    settings, _ = train_command()
    commands.add_subcommand("train", settings_parser(settings), help=first_paragraph(settings.__doc__))
    settings, _ = report_command()
    commands.add_subcommand("report", settings_parser(settings), help=first_paragraph(settings.__doc__))
    return top


def build(settings: type, values: Namespace | dict) -> object:
    """A settings dataclass from parsed values, nested settings included."""
    values = values.as_dict() if isinstance(values, Namespace) else dict(values)
    hints = typing.get_type_hints(settings)
    arguments = {}
    for item in dataclasses.fields(settings):
        if item.name not in values:
            continue
        value = values[item.name]
        if dataclasses.is_dataclass(hints[item.name]) and isinstance(value, dict | Namespace):
            value = build(hints[item.name], value)
        arguments[item.name] = value
    return settings(**arguments)


def main(argv: list[str] | None = None) -> int:
    arguments = parser().parse_args(argv)
    logs.setup(logging.WARNING if arguments.quiet else logging.INFO)
    if arguments.command in ("segment", "score"):
        settings, run = segment_command() if arguments.command == "segment" else score_command()
        given = arguments[arguments.command]
        values = given.as_dict()
        if values.pop("given_sources", None):
            values["sources"] = list(given.given_sources)
        chosen = build(settings, values)
    elif arguments.command == "compare":
        settings, run = compare_command()
        values = arguments.compare.as_dict()
        pages = values.pop("pages", None) or []
        if pages:
            if len(pages) != 2:
                print("lvc: error: compare takes two pages", file=sys.stderr)
                return 2
            values["first"], values["second"] = pages
        chosen = build(settings, values)
    elif arguments.command == "models":
        settings, run = models_command()
        chosen = build(settings, arguments.models)
    elif arguments.command == "cache":
        settings, run = cache_command()
        chosen = build(settings, arguments.cache)
    elif arguments.command == "codebook":
        settings, run = codebook_command()
        chosen = build(settings, arguments.codebook)
    elif arguments.command == "features":
        settings, run = features_command()
        chosen = build(settings, arguments.features)
    elif arguments.command == "embed":
        settings, run = embed_command()
        chosen = build(settings, arguments.embed)
    elif arguments.command == "evaluate":
        settings, run = evaluate_command()
        chosen = build(settings, arguments.evaluate)
    elif arguments.command == "train":
        settings, run = train_command()
        chosen = build(settings, arguments.train)
    elif arguments.command == "report":
        settings, run = report_command()
        chosen = build(settings, arguments.report)
    else:
        action = arguments.judgments.action
        settings, run = judgment_commands()[action]
        chosen = build(settings, arguments.judgments[action])
    try:
        run(chosen)
    except (OSError, ValueError, RuntimeError) as error:
        print(f"lvc: error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
