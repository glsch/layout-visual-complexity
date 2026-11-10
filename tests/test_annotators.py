from pathlib import Path

import pandas as pd
import pytest

from layout_visual_complexity.ranking.annotators import fold_spelling, person_key, pseudonyms, read_merge_table
from layout_visual_complexity.ranking.dataset import Rules, build_judgments


def test_folding_drops_accents_case_spaces_and_punctuation() -> None:
    assert fold_spelling("Jan Hajič") == fold_spelling("jan_hajic") == "janhajic"
    assert fold_spelling("Ukázková, Zdeňka") == "ukazkovazdenka"
    assert fold_spelling("  Ö. Ünal-3 ") == "ounal3"


def test_the_merge_table_maps_folded_spellings_to_persons(tmp_path: Path) -> None:
    path = tmp_path / "merges.tsv"
    path.write_text("spelling\tperson\tcheck\nUkázková, Zdeňka\tZ. U.\tx\nzdenka ukazkova\tZ. U.\t\n", "utf-8")
    merges = read_merge_table(path)
    assert merges == {"ukazkovazdenka": "Z. U.", "zdenkaukazkova": "Z. U."}
    assert person_key("Zdeňka Ukázková", merges) == ("table", "Z. U.")
    assert person_key("Ukázková Zdeňka", merges) == ("table", "Z. U.")
    assert person_key("Jan Hajič", merges) == ("folded", "janhajic")


def test_a_folded_spelling_with_two_persons_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "merges.tsv"
    path.write_text("spelling\tperson\nJan Hajič\tA\njan hajic\tB\n", "utf-8")
    with pytest.raises(ValueError, match="two persons"):
        read_merge_table(path)


def test_pseudonyms_follow_first_appearance() -> None:
    assert pseudonyms(["b", "a", "b", "c"]) == {"b": "annotator_001", "a": "annotator_002", "c": "annotator_003"}
    assert pseudonyms(range(1000))[999] == "annotator_1000"


def combinations(rows: list[tuple[str, str, str]]) -> pd.DataFrame:
    """Rated combinations from (session id, start day, spelling)."""
    return pd.DataFrame(
        {
            "session_id": [r[0] for r in rows],
            "user": [r[2] for r in rows],
            "source": [None] * len(rows),
            "started": pd.to_datetime([f"2025-01-{r[1]}" for r in rows], utc=True),
            "completed": True,
            "position": 0,
            "kind": "rated",
            "left": "a",
            "right": "b",
            "chosen": "a",
            "response_ms": pd.array([100] * len(rows), dtype="Int64"),
        }
    )


def test_merged_spellings_share_a_pseudonym_numbered_by_first_session() -> None:
    frame = combinations(
        [("s9", "03", "Petr"), ("s1", "01", "jan_hajic"), ("s5", "02", "Jan Hajič"), ("s2", "01", "X")]
    )
    kept = build_judgments(frame, {"x": "Petr Person", "petr": "Petr Person"}, Rules())
    by_session = dict(zip(kept.table["session"], kept.table["annotator"], strict=True))
    # Sessions by start, then by the app's id: s1 and s2 start the same day.
    assert by_session == {
        "session_0001": "annotator_001",
        "session_0002": "annotator_002",
        "session_0003": "annotator_001",
        "session_0004": "annotator_002",
    }
    assert kept.persons == 2
