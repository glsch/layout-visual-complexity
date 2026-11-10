import json
from pathlib import Path

import pytest

from layout_visual_complexity.ranking.dataset import Rules, build_judgments
from layout_visual_complexity.ranking.export import ExportError, image_id, read_export


def write(path: Path, sessions: list[dict]) -> Path:
    path.write_text(json.dumps({"sessions": sessions}), encoding="utf-8")
    return path


def rated(chosen: str, other: str, chosen_side: str = "left", calibration: bool = False) -> dict:
    other_side = "right" if chosen_side == "left" else "left"
    return {
        "entropyRated": True,
        "isCalibration": calibration,
        "thinkingTime": 1234,
        "winnerHigherEntropy": {"shelfmark": "s", "url": f"http://x/images/{chosen}.jpg", "position": chosen_side},
        "loserLowerEntropy": {"shelfmark": "s", "url": f"http://x/images/{other}.jpg", "position": other_side},
    }


SKIPPED = {"entropyRated": False, "isCalibration": False, "folio1": None, "folio2": None}


def session(identifier: str, combinations: list[dict], source: str | None = None, when: str = "01") -> dict:
    return {
        "sessionID": identifier,
        "user": "someone",
        "source": source,
        "timestamp": f"2025-03-{when}T10:00:00.000Z",
        "completed": True,
        "combinations": combinations,
    }


def test_image_ids_are_file_names_without_extension() -> None:
    assert image_id("http://x/images/VI._B._9._50v.jpg") == "VI._B._9._50v"
    assert image_id("http://x/images/II_D_1a_(II_D_1_1)_19v.jpg") == "II_D_1a_(II_D_1_1)_19v"
    assert image_id("http://x/images/a%20b.jpg") == "a b"


def test_rated_skipped_and_calibration_combinations_are_told_apart(tmp_path: Path) -> None:
    path = write(
        tmp_path / "e.json", [session("s1", [rated("a", "b", "right"), SKIPPED, rated("c", "d", calibration=True)])]
    )
    frame = read_export(path)
    assert frame["kind"].tolist() == ["rated", "skipped", "calibration"]
    first = frame.iloc[0]
    assert (first["left"], first["right"], first["chosen"], first["response_ms"]) == ("b", "a", "a", 1234)
    assert frame.iloc[1][["left", "right", "chosen"]].isna().all()
    assert str(frame["started"].dt.tz) == "UTC"


def test_rules_drop_skipped_calibration_and_excluded_sources(tmp_path: Path) -> None:
    sessions = [
        session("s1", [rated("a", "b"), SKIPPED, rated("a", "b", calibration=True)]),
        session("s2", [rated("c", "d"), rated("a", "c")], source="exhibition", when="02"),
        session("s3", [rated("b", "d")], source="bridge", when="03"),
    ]
    frame = read_export(write(tmp_path / "e.json", sessions))
    kept = build_judgments(frame, {}, Rules(exclude_sources=["exhibition"]))
    assert len(kept.table) == 2
    assert kept.table["source"].tolist() == ["", "bridge"]
    counts = kept.counts.set_index("source")
    assert counts.loc["total", "rated"] == 4 and counts.loc["total", "skipped"] == 1
    assert counts.loc["total", "calibration"] == 1 and counts.loc["exhibition", "kept"] == 0

    with_calibration = build_judgments(frame, {}, Rules(exclude_calibration=False))
    assert len(with_calibration.table) == 5
    assert with_calibration.table["calibration"].sum() == 1


def test_sessions_are_numbered_by_start_and_dated_to_the_day(tmp_path: Path) -> None:
    sessions = [session("zzz", [rated("a", "b")], when="05"), session("aaa", [rated("c", "d")], when="02")]
    kept = build_judgments(read_export(write(tmp_path / "e.json", sessions)), {}, Rules())
    assert kept.table["session"].tolist() == ["session_0001", "session_0002"]
    assert kept.table["chosen"].tolist() == ["c", "a"]
    assert kept.table["date"].tolist() == ["2025-03-02", "2025-03-05"]


def test_an_export_without_sessions_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "e.json"
    path.write_text("[]", encoding="utf-8")
    with pytest.raises(ExportError):
        read_export(path)


def test_a_rated_combination_with_two_left_images_is_refused(tmp_path: Path) -> None:
    broken = rated("a", "b")
    broken["loserLowerEntropy"]["position"] = "left"
    with pytest.raises(ExportError, match="positions"):
        read_export(write(tmp_path / "e.json", [session("s1", [broken])]))
