import json
import re
from pathlib import Path

import pandas as pd
import pytest

from conftest import MERGES, PERSONS, SPELLINGS, Study
from layout_visual_complexity.download import HttpSettings
from layout_visual_complexity.judgments import ImportSettings, run_import
from layout_visual_complexity.ranking.annotators import fold_spelling
from layout_visual_complexity.ranking.dataset import read_dataset


def settings(study: Study, output: Path, **changes) -> ImportSettings:
    values = {
        "export": study.export,
        "mapping": study.mapping,
        "metadata": study.metadata,
        "merges": study.merges,
        "output": output,
        "exclude_sources": ["exhibition"],
        "cache": output.parent / "cache",
        "http": HttpSettings(interval=0.0, backoff=0.0),
    }
    return ImportSettings(**(values | changes))


def name_patterns() -> list[re.Pattern]:
    """Every spelling, person and folded form, and each word of four letters or more in them, case-insensitive."""
    texts = set(SPELLINGS) | set(PERSONS) | set(MERGES) | {fold_spelling(s) for s in SPELLINGS + PERSONS}
    words = {word for text in texts for word in re.findall(r"\w{4,}", text)}
    words |= {fold_spelling(word) for word in words}
    return [re.compile(re.escape(text), re.IGNORECASE) for text in texts | words]


def test_the_import_writes_no_names(study: Study, tmp_path: Path, server) -> None:
    output = tmp_path / "dataset"
    run_import(settings(study, output), transport=server.transport())
    written = [path for path in output.rglob("*") if path.is_file()]
    assert {path.name for path in written} >= {"judgments.tsv", "images.tsv", "manuscripts.tsv", "README.md"}
    patterns = name_patterns()
    for path in written:
        text = path.read_text(encoding="utf-8")
        hits = [pattern.pattern for pattern in patterns if pattern.search(text)]
        assert not hits, f"{path.name} holds {hits}"


def test_the_dataset_has_pseudonyms_sessions_and_canvases(study: Study, tmp_path: Path, server) -> None:
    output = tmp_path / "dataset"
    summary = run_import(settings(study, output), transport=server.transport())
    dataset = read_dataset(output)
    judgments = dataset.judgments
    assert list(judgments.columns) == [
        "judgment",
        "session",
        "annotator",
        "source",
        "date",
        "left",
        "right",
        "chosen",
        "response_ms",
    ]
    # Six spellings, two pairs of them one person each: four persons. The fourth judged only in the exhibition
    # sessions, which are gone, so gets no pseudonym.
    assert set(judgments["source"]) == {"", "bridge"}
    assert judgments["annotator"].nunique() == 3
    assert judgments["annotator"].iloc[0] == "annotator_001"
    assert ((judgments["chosen"] == judgments["left"]) | (judgments["chosen"] == judgments["right"])).all()
    # Ten sessions kept of twelve, 12 rated judgments each; skipped and calibration ones dropped.
    assert summary["sessions"] == 10 and len(judgments) == 120
    images = dataset.images
    assert len(images) == 12
    resolved = images[images["canvas"].notna()]
    assert len(resolved) == 11
    assert images.loc[images["canvas"].isna(), "image"].tolist() == ["MS_D_0002r"]
    assert resolved["image_service"].str.startswith("https://iiif.example.org/images/").all()
    assert resolved["width"].eq(1000).all()
    unresolved = pd.read_csv(output / "unresolved_canvases.tsv", sep="\t")
    assert unresolved.loc[0, "matches"] == 2
    manuscripts = dataset.manuscripts
    assert manuscripts["pages"].tolist() == [3, 3, 3, 3]
    assert "catalogue_text" not in manuscripts.columns
    manifest = json.loads((output / "manifest.json").read_text())
    assert [entry["path"] for entry in manifest["inputs"]][0].endswith("export.json")
    assert all(len(entry["sha256"]) == 64 for entry in manifest["inputs"])
    assert manifest["versions"]["layout-visual-complexity"]


def test_manifests_are_fetched_once(study: Study, tmp_path: Path, server) -> None:
    run_import(settings(study, tmp_path / "first"), transport=server.transport())
    run_import(settings(study, tmp_path / "second"), transport=server.transport())
    assert len(server.requests) == 4


def test_overrides_give_canvases_before_labels(study: Study, tmp_path: Path, server) -> None:
    overrides = tmp_path / "overrides.tsv"
    overrides.write_text(
        "image\tcanvas\tmanifest\tnote\n"
        "MS_D_0002r\thttps://iiif.example.org/msd/canvas/4\t\ttwo carry 2r; the second\n"
        "MS_A_0001r\t\t\tleft empty: matched by label\n"
        "MS_B_0001v\tnone\t\tthe label matches, but the page isn't in this manifest\n",
        encoding="utf-8",
    )
    output = tmp_path / "dataset"
    summary = run_import(
        settings(study, output, canvas_overrides=overrides, notes=["Categories were checked by hand."]),
        transport=server.transport(),
    )
    images = read_dataset(output).images.set_index("image")
    assert images.loc["MS_D_0002r", "canvas"] == "https://iiif.example.org/msd/canvas/4"
    assert images.loc["MS_A_0001r", "canvas"] == "https://iiif.example.org/msa/canvas/1"
    assert images["canvas"].isna().sum() == 1 and pd.isna(images.loc["MS_B_0001v", "canvas"])
    assert summary["canvases"]["overrides"] == 2 and summary["canvases"]["unresolved"] == 1
    assert (output / "README.md").read_text().rstrip().endswith("Categories were checked by hand.")


def test_overrides_may_move_a_page_to_another_manifest(study: Study, tmp_path: Path, server) -> None:
    overrides = tmp_path / "overrides.tsv"
    overrides.write_text(
        "image\tcanvas\tmanifest\nMS_D_0002r\thttps://iiif.example.org/msc/canvas/3\thttps://iiif.example.org/msc/manifest\n",
        encoding="utf-8",
    )
    output = tmp_path / "dataset"
    run_import(settings(study, output, canvas_overrides=overrides), transport=server.transport())
    image = read_dataset(output).images.set_index("image").loc["MS_D_0002r"]
    assert image["iiif_manifest"] == "https://iiif.example.org/msc/manifest"
    assert image["image_service"] == "https://iiif.example.org/images/msc/3"


def test_an_override_naming_a_canvas_not_in_the_manifest_is_refused(study: Study, tmp_path: Path, server) -> None:
    overrides = tmp_path / "overrides.tsv"
    overrides.write_text("image\tcanvas\nMS_D_0002r\thttps://iiif.example.org/msd/canvas/99\n", encoding="utf-8")
    with pytest.raises(ValueError, match="isn't in"):
        run_import(settings(study, tmp_path / "dataset", canvas_overrides=overrides), transport=server.transport())
