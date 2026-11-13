"""Trivial: the two measures that need almost nothing, as the lowest baselines.

- Ink coverage: the ink layer's share of the folio (`folio_image.py`).
- Text lines: the number of the segmenter's text lines on the folio.
"""

from dataclasses import dataclass

from layout_visual_complexity.features.contract import Family, Group, Need
from layout_visual_complexity.features.folio_image import InkLayerSettings, ink_layer
from layout_visual_complexity.features.page import Page


@dataclass(frozen=True)
class TrivialSettings:
    """The ink layer coverage is measured on.

    Args:
        long_edge: The folio crop's long edge, in pixels.
        window: Sauvola's window, as a share of the crop's shorter edge.
        k: Sauvola's k.
        speck: Ink components smaller than this share of the folio's pixels are removed.
    """

    long_edge: int = InkLayerSettings.long_edge
    window: float = InkLayerSettings.window
    k: float = InkLayerSettings.k
    speck: float = InkLayerSettings.speck


class Trivial(Family):
    """Ink coverage and the number of text lines."""

    name = "trivial"
    scope = "page"
    version = "1"
    Settings = TrivialSettings

    @property
    def needs(self) -> frozenset[Need]:
        return frozenset({"image", "folio_crop", "ink", "lines"})

    def groups(self) -> list[Group]:
        settings: TrivialSettings = self.settings  # type: ignore[assignment]
        return [
            Group(
                "Trivial",
                "Ink coverage",
                "Trivial__InkCoverage",
                ("share",),
                "Share of the folio covered by ink.",
                f"Sauvola, window {settings.window:g} of the shorter edge, k = {settings.k:g}, on the flattened "
                f"lightness at {settings.long_edge:,} px",
                "Sauvola and Pietikäinen 2000",
            ),
            Group(
                "Trivial",
                "Number of text lines",
                "Trivial__TextLines",
                ("count",),
                "Number of text lines on the folio.",
                "Kraken's lines whose baseline's midpoint lies on the folio",
                "Kiessling 2019",
            ),
        ]

    def words(self, column: str) -> str:
        return {
            "Trivial__InkCoverage__share": "ink coverage, the share of the folio covered by ink",
            "Trivial__TextLines__count": "the number of text lines on the folio",
        }.get(column) or super().words(column)

    def prepare(self, page: Page) -> None:
        ink_layer(page, self.settings)

    def measure(self, page: Page) -> list[dict[str, float]]:
        layer = ink_layer(page, self.settings)
        lines = float(len(page.folio_lines))
        return [{"Trivial__InkCoverage__share": layer.coverage, "Trivial__TextLines__count": lines}]
