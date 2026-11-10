import json
from pathlib import Path

import pytest
import yaml

import layout_visual_complexity
from conftest import Study
from layout_visual_complexity import cli


def test_the_command_line_shows_its_version(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit):
        cli.main(["--version"])
    assert layout_visual_complexity.__version__ in capsys.readouterr().out


@pytest.mark.parametrize("arguments", [["--help"], ["judgments", "--help"], ["judgments", "rank", "--help"]])
def test_help(arguments: list[str], capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as stop:
        cli.main(arguments)
    assert stop.value.code == 0
    assert "usage: lvc" in capsys.readouterr().out


def test_import_rank_and_report_on_a_tiny_dataset(study: Study, tmp_path: Path) -> None:
    dataset, ranking, results = tmp_path / "dataset", tmp_path / "ranking", tmp_path / "results"
    config = tmp_path / "dataset.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "export": str(study.export),
                "mapping": str(study.mapping),
                "metadata": str(study.metadata),
                "merges": str(study.merges),
                "output": str(dataset),
                "canvases": False,
            }
        )
    )
    overrides = tmp_path / "sources.yaml"
    overrides.write_text("exclude_sources: [exhibition]\n")
    assert cli.main(["--quiet", "judgments", "import", "--config", str(config), "--config", str(overrides)]) == 0
    written = yaml.safe_load((dataset / "config.yaml").read_text())
    assert written["exclude_sources"] == ["exhibition"] and written["canvases"] is False

    rank = ["judgments", "rank", "--dataset", str(dataset), "--output", str(ranking)]
    assert cli.main(["--quiet", *rank, "--permutations", "30", "--seed", "4", "--jobs", "1"]) == 0
    manifest = json.loads((ranking / "manifest.json").read_text())
    assert manifest["summary"]["permutations"] == 30 and manifest["configuration"]["seed"] == 4

    report = ["judgments", "report", "--dataset", str(dataset), "--ranking", str(ranking), "--output", str(results)]
    small = ["--folds.k", "2", "--folds.repeats", "1", "--split_half.repeats", "3", "--split_half.permutations", "5"]
    assert cli.main(["--quiet", *report, *small, "--thumbnails", "0", "--jobs", "1", "--name", "tiny"]) == 0
    for stem in [
        "tables/dataset_categories",
        "tables/reliability_by_elo_gap",
        "tables/position_bias",
        "tables/consistency_reference_by_fold",
        "tables/split_half_ceiling",
        "tables/comparison_graph",
    ]:
        assert (results / f"{stem}.tsv").is_file() and (results / f"{stem}.tex").is_file()
    for stem in ["figures/comparison_graph", "figures/elo_score_distributions"]:
        for suffix in (".svg", ".png", ".tex"):
            assert (results / f"{stem}{suffix}").is_file()
    assert (results / "runs" / "tiny" / "manifest.json").is_file()
    assert r"\caption" in (results / "tables" / "reliability_by_elo_gap.tex").read_text()


def test_errors_are_reported_without_a_traceback(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    arguments = ["judgments", "rank", "--dataset", str(tmp_path / "missing"), "--output", str(tmp_path / "out")]
    assert cli.main(["--quiet", *arguments]) == 1
    assert "lvc: error:" in capsys.readouterr().err
