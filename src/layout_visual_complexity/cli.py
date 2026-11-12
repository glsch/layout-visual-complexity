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


def evaluate_command() -> tuple[type, Callable[..., object]]:
    """The evaluate command's settings and what runs it, imported when needed."""
    from layout_visual_complexity.evaluate import EvaluateSettings, run_evaluate

    return EvaluateSettings, run_evaluate


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
    """The command line: `lvc judgments import|rank|report`, `lvc segment`, `lvc features` and `lvc evaluate`."""
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
    settings, _ = features_command()
    commands.add_subcommand("features", settings_parser(settings), help=first_paragraph(settings.__doc__))
    settings, _ = evaluate_command()
    commands.add_subcommand("evaluate", settings_parser(settings), help=first_paragraph(settings.__doc__))
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
    if arguments.command == "segment":
        settings, run = segment_command()
        values = arguments.segment.as_dict()
        if values.pop("given_sources", None):
            values["sources"] = list(arguments.segment.given_sources)
        chosen = build(settings, values)
    elif arguments.command == "features":
        settings, run = features_command()
        chosen = build(settings, arguments.features)
    elif arguments.command == "evaluate":
        settings, run = evaluate_command()
        chosen = build(settings, arguments.evaluate)
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
