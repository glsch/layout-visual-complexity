"""Research outputs: each table as TSV and a LaTeX `table`, each figure as SVG, a 300-dpi PNG and a LaTeX `figure`."""

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

LATEX_SPECIAL = {
    "\\": r"\textbackslash{}",
    "&": r"\&",
    "%": r"\%",
    "$": r"\$",
    "#": r"\#",
    "_": r"\_",
    "{": r"\{",
    "}": r"\}",
    "~": r"\textasciitilde{}",
    "^": r"\textasciicircum{}",
    "\u2013": "--",
    "\u2014": "---",
    "\u2212": "$-$",
}


def escape(text: object) -> str:
    """Text with LaTeX's special characters escaped."""
    return "".join(LATEX_SPECIAL.get(character, character) for character in str(text))


def number(value: float | int, digits: int = 3) -> str:
    """A number for a table: integers with thousands separators, others with `digits` decimals, `--` if missing."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "--"
    if isinstance(value, int) or (isinstance(value, float) and value.is_integer() and digits == 0):
        return f"{int(value):,}"
    return f"{value:,.{digits}f}"


def gap_latex(label: str) -> str:
    """A gap bin's label for LaTeX: `<100`, `100--200`, `$\\geq 400$`, `Overall`."""
    if label == "overall":
        return "Overall"
    if label.startswith(">="):
        return rf"$\geq {label[2:]}$"
    if label.startswith("<"):
        return f"$<${label[1:]}"
    return label.replace("-", "--")


@dataclass
class LatexTable:
    """A LaTeX table: headers, rows of cells already formatted, a caption above, and rules.

    Columns are centred but the first, which is left-aligned, unless `align` says otherwise. `rules_before` holds the
    row numbers a `\\midrule` goes above.
    """

    headers: Sequence[str]
    rows: Sequence[Sequence[str]]
    caption: str
    label: str
    align: str | None = None
    rules_before: Sequence[int] = field(default_factory=tuple)

    def render(self) -> str:
        align = self.align or "l" + "c" * (len(self.headers) - 1)
        lines = [
            r"\begin{table}[htbp]",
            r"  \centering",
            rf"  \caption{{{self.caption}}}",
            rf"  \label{{{self.label}}}",
            rf"  \begin{{tabular}}{{{align}}}",
            r"    \toprule",
            "    " + " & ".join(self.headers) + r" \\",
            r"    \midrule",
        ]
        for index, row in enumerate(self.rows):
            if index in self.rules_before:
                lines.append(r"    \midrule")
            lines.append("    " + " & ".join(row) + r" \\")
        lines += [r"    \bottomrule", r"  \end{tabular}", r"\end{table}", ""]
        return "\n".join(lines)


def write_table(table: pd.DataFrame, stem: Path, latex: LatexTable | None, float_format: str = "%.6f") -> list[Path]:
    """`<stem>.tsv` and `<stem>.tex`; without `latex`, the TSV alone, and a `<stem>.tex` left from before is removed."""
    stem = Path(stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    tsv, tex = stem.with_suffix(".tsv"), stem.with_suffix(".tex")
    table.to_csv(tsv, sep="\t", index=False, lineterminator="\n", float_format=float_format)
    if latex is None:
        tex.unlink(missing_ok=True)
        return [tsv]
    tex.write_text(latex.render(), encoding="utf-8")
    return [tsv, tex]


def save_figure(figure, stem: Path, caption: str, label: str) -> list[Path]:
    """`<stem>.svg`, `<stem>.png` at 300 dpi, and `<stem>.tex`, a LaTeX figure with its caption below.

    The figure is saved at its own size, its print size, and the LaTeX figure includes the PNG at that width, so it's
    never scaled. The PNG is named by its path relative to the output folder (the stem's grandparent).
    """
    import matplotlib

    stem = Path(stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    svg, png, tex = stem.with_suffix(".svg"), stem.with_suffix(".png"), stem.with_suffix(".tex")
    with matplotlib.rc_context({"svg.hashsalt": "layout-visual-complexity", "svg.fonttype": "none"}):
        figure.savefig(svg, metadata={"Date": None})
    figure.savefig(png, dpi=300)
    width = figure.get_size_inches()[0] * 2.54
    relative = f"{stem.parent.name}/{png.name}"
    tex.write_text(
        "\n".join(
            [
                r"\begin{figure}[htbp]",
                r"  \centering",
                rf"  \includegraphics[width={width:.2f}cm]{{{relative}}}",
                rf"  \caption{{{caption}}}",
                rf"  \label{{{label}}}",
                r"\end{figure}",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return [svg, png, tex]
