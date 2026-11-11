import json
from pathlib import Path

import pandas as pd
from matplotlib.figure import Figure

from layout_visual_complexity.outputs import LatexTable, escape, gap_latex, save_figure, write_table
from layout_visual_complexity.runs import Run, shown_path


def test_tables_are_written_as_tsv_and_latex_with_the_caption_above(tmp_path: Path) -> None:
    table = pd.DataFrame({"gap": ["<100", "overall"], "value": [0.5, 0.75]})
    latex = LatexTable(["Gap", "Value"], [[gap_latex("<100"), "0.500"], ["Overall", "0.750"]], "A & B", "tab:x")
    tsv, tex = write_table(table, tmp_path / "tables" / "x", latex)
    assert pd.read_csv(tsv, sep="\t")["value"].tolist() == [0.5, 0.75]
    text = tex.read_text()
    assert text.index(r"\caption") < text.index(r"\begin{tabular}") and r"\toprule" in text
    assert escape("A & B_c 50%") == r"A \& B\_c 50\%"
    assert gap_latex(">=400") == r"$\geq 400$" and gap_latex("100-200") == "100--200"


def test_figures_are_svg_png_and_latex_with_the_caption_below(tmp_path: Path) -> None:
    figure = Figure()
    figure.subplots().plot([0, 1], [0, 1])
    svg, png, tex = save_figure(figure, tmp_path / "figures" / "line", "A line.", "fig:line")
    assert svg.read_text().startswith("<?xml") and png.read_bytes()[:4] == b"\x89PNG"
    text = tex.read_text()
    assert "figures/line.png" in text and text.index(r"\includegraphics") < text.index(r"\caption")
    figure = Figure()
    figure.subplots().plot([0, 1], [0, 1])
    again = save_figure(figure, tmp_path / "again" / "line", "A line.", "fig:line")[0]
    assert again.read_text() == svg.read_text()


def test_the_manifest_records_inputs_and_hides_home_folders(tmp_path: Path) -> None:
    source = tmp_path / "input.txt"
    source.write_text("data")
    run = Run("test", {"cache": Path("~/.cache/x"), "path": source})
    run.read(source)
    run.summary["count"] = 3
    config, manifest = run.finish(tmp_path / "run")
    record = json.loads(manifest.read_text())
    assert record["inputs"][0]["sha256"] == "3a6eb0790f39ac87c94f3856b2dd2c5d110e6811602261a9a923d3bb23adc8b7"
    assert record["summary"] == {"count": 3} and record["ended"] >= record["started"]
    assert "~/.cache/x" in config.read_text()
    assert shown_path(Path.home() / "projects" / "x") == "~/projects/x"
