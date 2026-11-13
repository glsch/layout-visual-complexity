"""The mean token: the mean of a page's patch tokens from a ViT's token set (`lvc embed`), one column a dimension,
for a linear pairwise model, the simplest learned baseline."""

from dataclasses import dataclass

import numpy as np

from layout_visual_complexity.features.contract import Family, FeatureError, Group, Need
from layout_visual_complexity.features.page import Page


@dataclass(frozen=True)
class MeanTokenSettings:
    """The token width.

    Args:
        dimensions: The tokens' width (384 for ViT-S); a token set of another width is refused.
    """

    dimensions: int = 384


class MeanToken(Family):
    """The mean of the page's patch tokens."""

    name = "mean_token"
    scope = "page"
    version = "1"
    Settings = MeanTokenSettings

    @property
    def needs(self) -> frozenset[Need]:
        return frozenset({"tokens"})

    def groups(self) -> list[Group]:
        settings: MeanTokenSettings = self.settings  # type: ignore[assignment]
        width = len(str(settings.dimensions - 1))
        flavours = tuple(f"d{number:0{width}d}" for number in range(settings.dimensions))
        return [
            Group("Mean token", "Dimension", "MeanToken__Dimension", flavours,
                  "The mean of the page's patch tokens, one column a dimension.",
                  f"{settings.dimensions} dimensions", "Oquab et al. 2024"),
        ]  # fmt: skip

    def words(self, column: str) -> str:
        return f"dimension {int(column.rsplit('__', 1)[1][1:])} of the mean patch token"

    def measure(self, page: Page) -> list[dict[str, float]]:
        grid = page.tokens
        if grid is None:
            raise FeatureError(f"{page.page} has no tokens")
        tokens = np.asarray(grid.tokens, dtype=np.float64)
        if tokens.ndim != 2 or tokens.shape[1] != self.settings.dimensions:
            raise FeatureError(f"mean_token: tokens {tokens.shape[-1]} wide, not {self.settings.dimensions}")
        return [dict(zip(self.columns(), tokens.mean(axis=0).tolist(), strict=True))]
