"""The real backbone, DINOv2 ViT-S/14 at its pinned revision, with the head that comes with the package. Skipped when
the `vit` extra isn't installed or the backbone isn't in the Hugging Face cache; nothing is downloaded."""

import importlib.util

import numpy as np
import pytest

from layout_visual_complexity.evaluation.vit_head import shipped_head
from layout_visual_complexity.features.vit import Backbone, TokenGrid, TokenSettings

INSTALLED = all(importlib.util.find_spec(name) for name in ("torch", "transformers"))


def cached_backbone() -> Backbone | None:
    if not INSTALLED:
        return None
    try:
        return Backbone(TokenSettings(), "cpu", local_only=True)
    except OSError:
        return None


pytestmark = pytest.mark.requires_models


def test_dinov2_gives_37_patches_along_the_long_edge_and_the_shipped_head_scores_them() -> None:
    backbone = cached_backbone()
    if backbone is None:
        pytest.skip("needs the vit extra and facebook/dinov2-small in the Hugging Face cache")
    assert (backbone.patch_size, backbone.dim, backbone.blocks) == (14, 384, 12)
    image = np.random.default_rng(1).integers(0, 255, (1500, 1000, 3), dtype=np.uint8)
    tokens, rows, columns = backbone.tokens(image)
    assert (rows, columns) == (37, 25) and tokens.shape == (37 * 25, 384)
    head = shipped_head()
    head.check(backbone.describe() | {"crop": "folio"})
    scores = head.patch_scores(TokenGrid(tokens, rows, columns, (0.0, 0.0, 1000.0, 1500.0)))
    assert scores.shape == (925,) and (scores >= 0).all()
