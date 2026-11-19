"""The reproduction's targets: what `make -C reproduce all` runs, and which recipe sees the gated model's token."""

import re
import shutil
import subprocess
from pathlib import Path

import pytest

REPRODUCE = Path(__file__).resolve().parents[1] / "reproduce"
NEEDS_TOKEN = ("embed_dataset_dinov3.yaml", "evaluate_vit_dinov3.yaml", "table_vit_backbones.yaml")


def recipes(target: str) -> str:
    """Every recipe `target` would run, whatever is built already (`make -n -B`)."""
    if shutil.which("make") is None:
        pytest.skip("make isn't installed")
    done = subprocess.run(["make", "-n", "-B", "-C", str(REPRODUCE), target], capture_output=True, text=True)
    assert done.returncode == 0, done.stderr
    return done.stdout


def test_all_draws_the_trios_and_needs_no_token() -> None:
    made = recipes("all")
    assert "figure_elo_score_bands.yaml" in made and "figure_elo_score_bands_trios.yaml" in made
    assert [name for name in NEEDS_TOKEN if name in made] == []


def test_the_backbones_table_makes_dinov3s_tokens_and_evaluation_first() -> None:
    made = recipes("table-vit-backbones")
    assert [made.index(name) for name in NEEDS_TOKEN] == sorted(made.index(name) for name in NEEDS_TOKEN)
    assert "evaluate_vit.yaml" in made


def test_the_token_is_exported_to_the_tokens_recipe_alone() -> None:
    lines = (REPRODUCE / "Makefile").read_text(encoding="utf-8").splitlines()
    assert [line for line in lines if re.search(r"\bexport\b.*\bHF_TOKEN\b", line)] == [
        "$(TOKENS_DINOV3): export HF_TOKEN := $(HF_TOKEN)"
    ]
