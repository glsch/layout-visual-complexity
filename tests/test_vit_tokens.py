"""Patch tokens from a ViT, their cache and `lvc embed`, with a tiny random DINOv2 for the backbone."""

from pathlib import Path

import numpy as np
import pytest

from layout_visual_complexity import cli
from layout_visual_complexity.embed import EmbedSettings, run_embed
from layout_visual_complexity.features.contract import Family, Group
from layout_visual_complexity.features.extraction import FeaturesSettings, read_feature_table, run_features
from layout_visual_complexity.features.vit import (
    Backbone,
    TokenCache,
    TokenGrid,
    TokenSettings,
    grid_shape,
    model_input,
    read_token_set,
)
from test_features_contract import write_store

pytest.importorskip("transformers")


def settings(folder: Path, **changes: object) -> TokenSettings:
    return TokenSettings(**({"backbone": str(folder), "revision": None, "long_edge": 56} | changes))  # type: ignore[arg-type]


def page_image(height: int = 300, width: int = 200) -> np.ndarray:
    rng = np.random.default_rng(1)
    return rng.integers(0, 255, (height, width, 3), dtype=np.uint8)


def test_the_grid_follows_the_input_size_and_the_patch_size(tiny_vit: Path) -> None:
    assert grid_shape(1500, 1000, 518, 14) == (37, 25)
    assert grid_shape(1000, 1500, 518, 14) == (25, 37)
    backbone = Backbone(settings(tiny_vit), "cpu")
    tokens, rows, columns = backbone.tokens(page_image())
    assert (rows, columns) == (4, 3) and tokens.shape == (12, 16)
    tokens, rows, columns = Backbone(settings(tiny_vit, long_edge=112), "cpu").tokens(page_image())
    assert (rows, columns) == (8, 5) and tokens.shape == (40, 16)
    values, rows, columns = model_input(page_image(), 112, 14, (0.5,) * 3, (0.25,) * 3)
    assert values.shape == (3, 8 * 14, 5 * 14)


def test_the_class_token_is_dropped_and_the_tokens_are_the_last_blocks_after_its_norm(tiny_vit: Path) -> None:
    import torch

    backbone = Backbone(settings(tiny_vit), "cpu")
    values, rows, columns = model_input(page_image(), 56, 14, backbone.mean, backbone.std)
    with torch.no_grad():
        output = backbone.model(pixel_values=torch.from_numpy(values)[None], output_hidden_states=True)
    tokens, _, _ = backbone.tokens(page_image())
    assert output.last_hidden_state.shape[1] == rows * columns + 1
    np.testing.assert_allclose(tokens, output.last_hidden_state[0, 1:].numpy(), atol=1e-5)
    first, _, _ = Backbone(settings(tiny_vit, layer=1), "cpu").tokens(page_image())
    with torch.no_grad():
        expected = backbone.model.layernorm(output.hidden_states[1])[0, 1:].numpy()
    np.testing.assert_allclose(first, expected, atol=1e-5)
    with pytest.raises(ValueError, match="no block 3"):
        Backbone(settings(tiny_vit, layer=3), "cpu")


def test_the_cache_is_keyed_by_backbone_revision_layer_size_and_crop(tmp_path: Path) -> None:
    base = TokenSettings()
    changed = [
        TokenSettings(backbone="facebook/dinov3-vits16-pretrain-lvd1689m"),
        TokenSettings(revision="0" * 40),
        TokenSettings(layer=11),
        TokenSettings(long_edge=592),
        TokenSettings(crop="page"),
    ]
    keys = {base.key()} | {other.key() for other in changed}
    assert len(keys) == 6 and TokenSettings().key() == base.key()
    grid = TokenGrid(np.ones((6, 4), np.float32), 2, 3, (1.0, 2.0, 31.0, 22.0))
    cache = TokenCache(tmp_path, base)
    cache.write("ms_1r", grid, "an image", "cpu")
    assert TokenCache(tmp_path, TokenSettings(layer=11)).read("ms_1r") is None
    assert cache.read("ms_1r", "another image") is None
    found = cache.read("ms_1r", "an image")
    assert found is not None and found.tokens.dtype == np.float16 and found.box == grid.box
    assert (found.rows, found.columns) == (2, 3)


def test_patch_boxes_are_placed_on_the_page_in_the_full_images_pixels() -> None:
    grid = TokenGrid(np.zeros((6, 4)), 2, 3, (100.0, 200.0, 400.0, 400.0))
    boxes = grid.patch_boxes()
    np.testing.assert_allclose(boxes[0], [100, 200, 200, 300])
    np.testing.assert_allclose(boxes[2], [300, 200, 400, 300])
    np.testing.assert_allclose(boxes[5], [300, 300, 400, 400])


def test_lvc_embed_caches_the_folio_crops_tokens_and_a_rerun_keeps_them(tmp_path: Path, tiny_vit: Path) -> None:
    store = write_store(tmp_path / "segment", pages=3)
    chosen = EmbedSettings(store, tmp_path / "tokens", settings(tiny_vit), "cpu", tmp_path / "user_cache", threads=2)
    summary = run_embed(chosen)
    assert summary["embedded"] == 3 and not summary["failed"]
    assert run_embed(chosen)["embedded"] == 0
    token_set = read_token_set(tmp_path / "tokens")
    assert token_set.description["block"] == 2 and token_set.description["dim"] == 16
    assert list(token_set.pages.index) == ["ms_1r", "ms_2r", "ms_3r"]
    grid = token_set.grid("ms_2r")
    assert grid.box == (5.0, 5.0, 195.0, 295.0) and (grid.rows, grid.columns) == (4, 3)
    whole = EmbedSettings(store, tmp_path / "whole", settings(tiny_vit, crop="page"), "cpu", tmp_path / "user_cache")
    assert run_embed(whole)["embedded"] == 3
    assert read_token_set(tmp_path / "whole").grid("ms_2r").box == (0.0, 0.0, 200.0, 300.0)


def test_lvc_embed_from_the_command_line(tmp_path: Path, tiny_vit: Path) -> None:
    store = write_store(tmp_path / "segment", pages=2)
    arguments = ["--quiet", "embed", "--layout", str(store), "--output", str(tmp_path / "tokens"), "--device", "cpu"]
    arguments += ["--tokens.backbone", str(tiny_vit), "--tokens.revision", "null", "--tokens.long_edge", "28"]
    assert cli.main([*arguments, "--token_cache", str(tmp_path / "user_cache")]) == 0
    assert read_token_set(tmp_path / "tokens").grid("ms_1r").tokens.shape == (2, 16)


class MeanToken(Family):
    """A test family that needs the tokens: their mean first dimension."""

    name = "mean_token"

    @property
    def needs(self) -> frozenset:
        return frozenset({"tokens"})

    def groups(self) -> list[Group]:
        return [Group("Test", "Tokens", "Test__Tokens", ("first_mean",), "The tokens' mean first dimension.")]

    def measure(self, page: object) -> list[dict[str, float]]:
        return [{"Test__Tokens__first_mean": float(page.tokens.tokens[:, 0].astype(float).mean())}]  # type: ignore[attr-defined]


def test_a_family_that_needs_the_tokens_gets_them_from_a_token_set(tmp_path: Path, tiny_vit: Path) -> None:
    store = write_store(tmp_path / "segment", pages=2)
    run_embed(EmbedSettings(store, tmp_path / "tokens", settings(tiny_vit), "cpu", tmp_path / "user_cache"))
    family = f"{MeanToken.__module__}.MeanToken"
    with pytest.raises(ValueError, match="token set"):
        run_features(FeaturesSettings(store, tmp_path / "features", [family], jobs=1))
    summary = run_features(FeaturesSettings(store, tmp_path / "features", [family], jobs=1, tokens=tmp_path / "tokens"))
    assert summary["measured"] == 2 and not summary["failed"]
    values = read_feature_table(tmp_path / "features" / "features.parquet").values["Test__Tokens__first_mean"]
    expected = read_token_set(tmp_path / "tokens").grid("ms_1r").tokens[:, 0].astype(float).mean()
    assert values["ms_1r"] == pytest.approx(expected)
