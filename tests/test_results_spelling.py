"""The prose of the reproduction's outputs spells "judgement": every caption and table in `results/` (LaTeX) and every
label of its figures (SVG). Names keep "judgment": files, columns, options, keys and ids."""

import re
from pathlib import Path

RESULTS = Path(__file__).resolve().parents[1] / "results"
# A word holding "judgment", with what joins a name's parts: underscores (escaped in LaTeX), dots and dashes.
WORD = re.compile(r"[\w\\.\-]*[Jj]udgment[\w\\.\-]*")


def prose(word: str) -> bool:
    """Whether a word holding "judgment" is prose rather than a name (`judgments.tsv`, `judgments\\_per\\_fold`,
    `--thumbnail_trios`)."""
    return not ("_" in word or ".tsv" in word or word.startswith("-"))


def labels(svg: str) -> list[str]:
    return re.findall(r"<text\b[^>]*>([^<]*)</text>", svg)


def test_the_outputs_prose_says_judgement() -> None:
    latex = sorted(RESULTS.glob("figures/*.tex")) + sorted(RESULTS.glob("tables/*.tex"))
    svgs = sorted(RESULTS.glob("figures/*.svg"))
    assert len(latex) > 50 and len(svgs) > 15
    found = [(path.name, word) for path in latex for word in WORD.findall(path.read_text(encoding="utf-8"))
             if prose(word)]  # fmt: skip
    found += [(path.name, word) for path in svgs for text in labels(path.read_text(encoding="utf-8"))
              for word in WORD.findall(text) if prose(word)]  # fmt: skip
    assert found == []
    shown = " ".join(path.read_text(encoding="utf-8") for path in latex)
    assert shown.count("judgement") > 50


def test_names_keep_their_spelling() -> None:
    assert not prose("judgments.tsv") and not prose("judgments\\_per\\_fold") and prose("judgments.")
