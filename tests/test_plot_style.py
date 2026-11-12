"""The figures' style: drawn at their print size, in STIXGeneral, and included in LaTeX at that size."""

from pathlib import Path

import pytest
from matplotlib.font_manager import findfont
from PIL import Image

from layout_visual_complexity.outputs import save_figure
from layout_visual_complexity.plots import style


@pytest.mark.parametrize(("width", "centimetres"), [("full", 15.0), ("half", 7.2)])
def test_figures_are_drawn_at_their_print_size_in_stix(width: str, centimetres: float, tmp_path: Path) -> None:
    figure = style.new_figure(width, 6)
    axes = figure.subplots()
    axes.plot([0, 1], [0, 1])
    label = axes.set_xlabel("Elo gap")
    assert style.print_width_cm(figure) == pytest.approx(centimetres)
    assert figure.get_size_inches()[1] * 2.54 == pytest.approx(6)
    assert label.get_fontname() == "STIXGeneral" and label.get_fontsize() == 9
    assert axes.get_xticklabels()[0].get_fontsize() == 8
    assert "STIX" in findfont(label.get_fontproperties())
    assert "DejaVuSansMono" in findfont(style.monospace())
    svg, png, tex = save_figure(figure, tmp_path / "figures" / "line", "A line.", "fig:line")
    assert f"width={centimetres:.2f}cm" in tex.read_text()
    assert "STIXGeneral" in svg.read_text()
    with Image.open(png) as image:
        assert image.size[0] == pytest.approx(centimetres / 2.54 * 300, abs=1)


def test_every_method_has_its_colour_and_panels_their_letters() -> None:
    assert style.METHODS["geometry"] == "#E69F00" and style.METHODS["vit"] == "#0072B2"
    figure = style.new_figure("full", 5)
    axes = figure.subplots(1, 2)
    style.panel_letters(axes, ["Accuracy", ""])
    assert axes[0].get_title(loc="left").endswith("Accuracy") and axes[1].get_title(loc="left") == "(b)"
