import json
from pathlib import Path

import pytest

from conftest import FakeZenodo, published, write_published
from layout_visual_complexity.download import DownloadError, HttpSettings
from layout_visual_complexity.judgments import ImportSettings, run_import
from layout_visual_complexity.ranking.dataset import DatasetError, read_dataset, validate_dataset
from layout_visual_complexity.zenodo import record_of


def settings(source: str | Path, output: Path, **changes) -> ImportSettings:
    values = {"source": str(source), "output": output, "cache": output.parent / "cache"}
    values["http"] = HttpSettings(interval=0.0, backoff=0.0)
    return ImportSettings(**(values | changes))


def test_a_folder_is_checked_and_copied(dataset_folder: Path, tmp_path: Path) -> None:
    output = tmp_path / "dataset"
    summary = run_import(settings(dataset_folder, output))
    for name in ("README.md", "judgments.tsv", "images.tsv", "manuscripts.tsv"):
        assert (output / name).read_bytes() == (dataset_folder / name).read_bytes()
    assert (summary["judgments"], summary["images"], summary["manuscripts"]) == (160, 12, 4)
    manifest = json.loads((output / "manifest.json").read_text())
    assert all(len(entry["sha256"]) == 64 for entry in manifest["inputs"])


def test_the_published_structure_reads_back_as_written(dataset_folder: Path) -> None:
    dataset = read_dataset(dataset_folder)
    validate_dataset(dataset)
    assert str(dataset.images["width"].dtype) == "Int64"
    assert dataset.manuscripts.loc[0, "date_edtf"] == "1401/1500"
    assert dataset.manuscripts.loc[0, "secondary_categories"] == ""


def refused(tmp_path: Path, change, message: str) -> None:
    dataset = published()
    change(dataset)
    folder = write_published(tmp_path / "broken", dataset)
    with pytest.raises(DatasetError, match=message):
        run_import(settings(folder, tmp_path / "out"))


def test_a_missing_column_is_refused(tmp_path: Path) -> None:
    refused(tmp_path, lambda d: d.images.pop("image_url"), "missing image_url")


def test_a_judgment_naming_an_unknown_page_is_refused(tmp_path: Path) -> None:
    refused(tmp_path, lambda d: d.judgments.loc.__setitem__((0, "left"), "nope_1r"), "not in images.tsv")


def test_a_malformed_date_is_refused(tmp_path: Path) -> None:
    refused(tmp_path, lambda d: d.manuscripts.loc.__setitem__((0, "date_edtf"), "about 1450"), "EDTF")


def test_other_inconsistencies_are_refused() -> None:
    dataset = published()
    dataset.judgments.loc[0, "right"] = dataset.judgments.loc[0, "left"]
    dataset.judgments.loc[0, "chosen"] = dataset.judgments.loc[0, "left"]
    dataset.images.loc[1, "canvas"] = ""
    dataset.manuscripts.loc[2, "languages"] = "la"
    dataset.manuscripts.loc[3, "category"] = ""
    with pytest.raises(DatasetError) as caught:
        validate_dataset(dataset)
    message = str(caught.value)
    for part in ("against itself", "no canvas", "ISO 639-3", "no category"):
        assert part in message


@pytest.mark.parametrize("date", ["1461", "1505~", "1450?", "1450%", "1434/1466", "1450/..", "../1480", ""])
def test_the_edtf_forms_used_are_accepted(date: str) -> None:
    dataset = published()
    dataset.manuscripts.loc[0, "date_edtf"] = date
    validate_dataset(dataset)


def test_zenodo_dois_are_recognised() -> None:
    assert record_of("10.5281/zenodo.1234") == ("zenodo.org", "1234")
    assert record_of("https://doi.org/10.5072/zenodo.99") == ("sandbox.zenodo.org", "99")
    assert record_of("data/folder") is None


def test_a_doi_fetches_the_record_s_files(dataset_folder: Path, tmp_path: Path) -> None:
    server = FakeZenodo(dataset_folder)
    output = tmp_path / "dataset"
    summary = run_import(settings("10.5281/zenodo.1234", output), transport=server.transport())
    assert summary["doi"] == "10.5281/zenodo.1234"
    assert (output / "judgments.tsv").read_bytes() == (dataset_folder / "judgments.tsv").read_bytes()
    fetched = len(server.requests)
    run_import(settings("10.5281/zenodo.1234", tmp_path / "again"), transport=server.transport())
    assert len(server.requests) == fetched + 1  # the record again, but no file


def test_a_file_whose_md5_differs_is_refused(dataset_folder: Path, tmp_path: Path) -> None:
    server = FakeZenodo(dataset_folder, corrupt="images.tsv")
    with pytest.raises(DownloadError, match="MD5"):
        run_import(settings("10.5281/zenodo.1234", tmp_path / "dataset"), transport=server.transport())


def test_a_source_that_is_neither_is_refused(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="neither a folder nor"):
        run_import(settings(tmp_path / "nowhere", tmp_path / "dataset"))
