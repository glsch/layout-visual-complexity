"""The difficulty experiment's parts: ground truth read from ALTO and from a `metadata.jsonl`, spreads split, zones
and lines matched, character errors counted, and `lvc difficulty` on synthetic pages with fake models."""

import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest

from layout_visual_complexity import ground_truth as truth
from layout_visual_complexity.difficulty import Found, measure_page
from layout_visual_complexity.layout_matching import baseline_distance, box_iou, line_height, match, polygon_iou
from layout_visual_complexity.spreads import Gutter, SpreadSettings, find_gutter, split_page, split_spreads
from layout_visual_complexity.text_errors import edit_distance, normalise, rate


def box(x0: float, y0: float, x1: float, y1: float) -> list[tuple[float, float]]:
    return [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]


def flat(points: list[tuple[float, float]]) -> str:
    return " ".join(f"{x:g} {y:g}" for x, y in points)


def alto(image: str, blocks: list[tuple[str, str, list, list]], tags: dict[str, str], size: tuple[int, int]) -> str:
    """An ALTO 4 file: `blocks` as (id, tag reference, polygon, lines), each line (id, tag reference, polygon,
    baseline, text)."""
    tag_lines = "".join(f'<OtherTag ID="{key}" LABEL="{label}"/>' for key, label in tags.items())
    body = ""
    for block_id, reference, polygon, lines in blocks:
        refs = f' TAGREFS="{reference}"' if reference else ""
        body += f'<TextBlock ID="{block_id}"{refs}><Shape><Polygon POINTS="{flat(polygon)}"/></Shape>'
        for line_id, line_ref, line_polygon, baseline, text in lines:
            line_refs = f' TAGREFS="{line_ref}"' if line_ref else ""
            body += (f'<TextLine ID="{line_id}"{line_refs} BASELINE="{flat(baseline)}"><Shape><Polygon '
                     f'POINTS="{flat(line_polygon)}"/></Shape><String CONTENT="{text}"/></TextLine>')  # fmt: skip
        body += "</TextBlock>"
    return (
        '<alto xmlns="http://www.loc.gov/standards/alto/ns-v4#"><Description><sourceImageInformation><fileName>'
        f"{image}</fileName></sourceImageInformation></Description><Tags>{tag_lines}</Tags><Layout>"
        f'<Page WIDTH="{size[0]}" HEIGHT="{size[1]}"><PrintSpace>{body}</PrintSpace></Page></Layout></alto>'
    )


def write_page(folder: Path, stem: str, size: tuple[int, int] = (600, 800), spread: bool = False) -> Path:
    """A synthetic page (or spread) with an ALTO file beside it: a main zone of text lines and a margin note."""
    folder.mkdir(parents=True, exist_ok=True)
    width, height = size
    image = np.full((height, width, 3), 30, np.uint8)
    halves = [(0, width // 2 - 4), (width // 2 + 4, width)] if spread else [(0, width)]
    blocks = []
    for number, (x0, x1) in enumerate(halves, start=1):
        cv2.rectangle(image, (x0 + 10, 10), (x1 - 10, height - 10), (230, 220, 200), -1)
        left, right = x0 + 60, x1 - 120
        lines = []
        for row in range(6):
            y = 120 + 60 * row
            cv2.line(image, (left, y), (right, y), (40, 30, 30), 4)
            lines.append((f"p{number}l{row}", "LT1", box(left, y - 30, right, y + 8), [(left, y), (right, y)],
                          f"line {row} of page {number}"))  # fmt: skip
        blocks.append((f"p{number}main", "BT2", box(left - 10, 80, right + 10, 480), lines))
        note = (f"p{number}note", "LT1", box(right + 20, 140, x1 - 20, 170), [(right + 20, 160), (x1 - 20, 160)],
                "nota")  # fmt: skip
        blocks.append((f"p{number}margin", "BT3", box(right + 15, 130, x1 - 15, 180), [note]))
    cv2.imwrite(str(folder / f"{stem}.jpg"), image)
    tags = {"BT2": "MainZone#1", "BT3": "MarginTextZone:note", "LT1": "DefaultLine"}
    path = folder / f"{stem}.xml"
    path.write_text(alto(f"{stem}.jpg", blocks, tags, size), encoding="utf-8")
    return path


# Ground truth.


def test_alto_is_read_with_its_own_tags_and_classes_without_subtypes(tmp_path: Path) -> None:
    first = write_page(tmp_path / "ms_one", "f1r")
    other = tmp_path / "ms_one" / "f2r.xml"
    # Another file numbers the same classes otherwise.
    text = first.read_text().replace('"BT2"', '"BT9"').replace("f1r.jpg", "f2r.jpg")
    other.write_text(text.replace('ID="BT3" LABEL="MarginTextZone:note"', 'ID="BT3" LABEL="MainZone"'))
    (tmp_path / "ms_one" / "f2r.jpg").write_bytes((tmp_path / "ms_one" / "f1r.jpg").read_bytes())
    page = truth.read_alto(first)
    assert page.page == "msone_f1r" and page.document == "msone" and (page.width, page.height) == (600, 800)
    assert [zone.cls for zone in page.zones] == ["MainZone", "MarginTextZone"]
    assert len(page.lines) == 7 and {line.cls for line in page.lines} == {"DefaultLine"}
    assert page.lines[0].text == "line 0 of page 1" and page.lines[0].zone_cls == "MainZone"
    assert page.lines[-1].zone_cls == "MarginTextZone" and page.transcribed
    second = truth.read_alto(other)
    assert [zone.cls for zone in second.zones] == ["MainZone", "MainZone"]


def test_a_line_without_a_class_is_a_default_line_and_a_block_without_one_no_zone(tmp_path: Path) -> None:
    (tmp_path / "ms").mkdir()
    cv2.imwrite(str(tmp_path / "ms" / "a.jpg"), np.zeros((100, 100, 3), np.uint8))
    line = ("l1", "", box(10, 10, 90, 30), [(10, 25), (90, 25)], "text")
    (tmp_path / "ms" / "a.xml").write_text(alto("a.jpg", [("b1", "", box(5, 5, 95, 95), [line])], {}, (100, 100)))
    page = truth.read_alto(tmp_path / "ms" / "a.xml")
    assert not page.zones and page.lines[0].cls == "DefaultLine" and page.lines[0].zone is None


def test_metadata_reads_each_image_once_and_only_the_named_manuscripts(tmp_path: Path) -> None:
    folder = tmp_path / "test"
    for name in ("ms-a", "ms-b"):
        (folder / name).mkdir(parents=True)
        cv2.imwrite(str(folder / name / "page-001.jpg"), np.zeros((50, 40, 3), np.uint8))
    objects = {"id": ["b1", "l1", "l2"], "bbox": [[0, 0, 40, 50]] * 3,
               "polygons": [[0, 0, 40, 0, 40, 50, 0, 50], [2, 2, 38, 2, 38, 9, 2, 9], [2, 12, 38, 12, 38, 19, 2, 19]],
               "category": ["MainZone#2", "DefaultLine", "DropCapitalLine"], "type": ["block", "line", "line"],
               "parent": ["", "b1", ""], "baseline": [[], [2, 8, 38, 8], [2, 18, 38, 18]]}  # fmt: skip
    rows = [{"file_name": "ms-a/page-001.jpg", "width": 40, "height": 50, "objects": objects, "century": 14},
            {"file_name": "ms-b/page-001.jpg", "width": 40, "height": 50, "objects": objects, "century": 15},
            {"file_name": "ms-a/page-001.jpg", "width": 40, "height": 50, "objects": {**objects, "id": ["x", "y", "z"]},
             "century": 14}]  # fmt: skip
    (folder / "metadata.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    pages = truth.read_metadata(folder / "metadata.jsonl", ["ms-a"])
    assert [page.page for page in pages] == ["msa_page-001"]
    page = pages[0]
    assert [zone.cls for zone in page.zones] == ["MainZone"] and [zone.id for zone in page.zones] == ["b1"]
    assert [(line.cls, line.zone_cls, line.text) for line in page.lines] == [
        ("DefaultLine", "MainZone", None), ("DropCapitalLine", None, None)
    ]  # fmt: skip
    assert not page.transcribed
    assert len(truth.read_metadata(folder / "metadata.jsonl")) == 2
    with pytest.raises(ValueError, match="no images of ms-c"):
        truth.read_metadata(folder / "metadata.jsonl", ["ms-c"])


# Spreads.


def test_a_spread_splits_with_its_ground_truth_moved_and_an_empty_half_left_out(tmp_path: Path) -> None:
    page = truth.read_alto(write_page(tmp_path / "ms", "f1v-2r", (1200, 800), spread=True))
    split = split_page(page, Gutter(600, True, 0.5))
    left, right = split.halves
    assert (left.page, right.page) == ("ms_f1v-2r_left", "ms_f1v-2r_right")
    assert left.region == (0, 0, 600, 800) and right.region == (600, 0, 1200, 800) and right.size == (600, 800)
    assert len(left.lines) == len(right.lines) == 7 and len(left.zones) == len(right.zones) == 2
    original = next(line for line in page.lines if line.id == "p2l0")
    moved = next(line for line in right.lines if line.id == "p2l0")
    assert moved.polygon[0] == (original.polygon[0][0] - 600, original.polygon[0][1])
    assert not split.crossing and not split.empty
    # A line across the split is listed; a half without lines is left out.
    wide = truth.Line("across", "DefaultLine", box(500, 600, 700, 640), [(500, 630), (700, 630)], "x")
    crossing = split_page(truth.Page(**{**page.__dict__, "lines": [*page.lines, wide]}), Gutter(600, True, 0.5))
    assert [line for line, _ in crossing.crossing] == ["across"] and crossing.crossing[0][1] == pytest.approx(0.5)
    lonely = truth.Page(**{**page.__dict__, "lines": [line for line in page.lines if line.id.startswith("p1")]})
    found = split_page(lonely, Gutter(600, True, 0.5))
    assert [half.page for half in found.halves] == ["ms_f1v-2r_left"] and found.empty == ["ms_f1v-2r_right"]


def test_the_gutter_is_found_on_a_synthetic_spread_and_falls_back_to_the_middle() -> None:
    image = np.full((400, 700, 3), 20, np.uint8)
    cv2.rectangle(image, (20, 20), (680, 380), (235, 225, 205), -1)
    for y in range(60, 340, 12):  # writing on both pages
        cv2.line(image, (60, y), (300, y), (40, 30, 30), 3)
        cv2.line(image, (420, y), (650, y), (40, 30, 30), 3)
    plain = image.copy()
    cv2.rectangle(image, (378, 20), (384, 380), (70, 65, 60), -1)  # the gutter's shadow, off the middle
    column, found, depth, _ = find_gutter(image)
    assert found and 377 <= column <= 385 and depth > 0.5
    # A column of initials beside the text is no gutter: without a shadow, the split falls back to the middle.
    for y in (60, 160, 260):
        cv2.rectangle(plain, (405, y), (418, y + 25), (40, 30, 30), -1)
    column, found, depth, _ = find_gutter(plain)
    assert not found and column == 350 and depth < SpreadSettings().contrast


def test_spreads_are_split_by_their_ids_or_every_wide_image(tmp_path: Path) -> None:
    pages = [truth.read_alto(write_page(tmp_path / "ms", "f1v-2r", (1200, 800), spread=True)),
             truth.read_alto(write_page(tmp_path / "ms", "f3r"))]  # fmt: skip
    kept, splits = split_spreads(pages, SpreadSettings(pages="auto"))
    assert [page.page for page in kept] == ["ms_f1v-2r_left", "ms_f1v-2r_right", "ms_f3r"]
    assert splits[0].gutter.found and 590 <= splits[0].gutter.x <= 610
    kept, splits = split_spreads(pages, SpreadSettings(pages=["ms_f3r"]))
    assert [page.page for page in kept] == ["ms_f1v-2r", "ms_f3r_left", "ms_f3r_right"]


# Matching and errors.


def hand_made() -> truth.Page:
    """A page with known answers: two zones; three text lines, a drop capital's line, all 40 pixels high."""
    lines = [truth.Line(f"t{n}", "DefaultLine", box(100, 100 * n, 500, 100 * n + 40), [(100, 100 * n + 35),
                        (500, 100 * n + 35)], text, "z1", "MainZone")
             for n, text in ((1, "abcd"), (2, "efgh"), (3, "ijkl"))]  # fmt: skip
    lines.append(truth.Line("d1", "DropCapitalLine", box(40, 100, 90, 180), [(40, 170), (90, 170)], "A", "z1",
                            "MainZone"))  # fmt: skip
    zones = [truth.Zone("z1", "MainZone", box(30, 90, 510, 350)), truth.Zone("z2", "MarginTextZone",
                                                                             box(600, 90, 700, 300))]  # fmt: skip
    return truth.Page("ms_p1", "ms", Path("p1.jpg"), 800, 1000, zones, lines, label="p1")


def test_zones_and_lines_match_one_to_one_and_drop_capital_lines_count_neither_way() -> None:
    page = hand_made()
    found = Found(
        zones=[("MainZone", (35, 95, 505, 345)), ("MarginTextZone", (100, 800, 200, 900))],
        lines=[(box(100, 105, 500, 145), [(100, 140), (500, 140)]),  # t1, its baseline 5 px off
               (box(100, 200, 500, 240), [(100, 235), (500, 235)]),  # t2, exactly
               (box(42, 102, 88, 178), [(42, 170), (88, 170)]),  # the drop capital's line
               (box(100, 600, 500, 640), [(100, 635), (500, 635)])],  # invented
        texts=["abxd", "efgh", "A", "zz"],
    )  # fmt: skip
    measured = measure_page(page, found, {"t1": "abcd", "t2": "efg", "t3": "ijkl"}).row
    assert (measured["zones_matched"], measured["zones_invented"], measured["zones_missed"]) == (1, 1, 1)
    assert measured["zone_f1"] == pytest.approx(0.5)
    assert (measured["lines_matched"], measured["lines_invented"], measured["lines_missed"]) == (2, 1, 1)
    assert measured["line_f1"] == pytest.approx(4 / 6)
    assert measured["line_height"] == pytest.approx(40) and measured["lines"] == 3
    assert measured["baseline_distance"] == pytest.approx((5 / 40 + 0) / 2)
    # On the ground-truth lines: one deletion in 12 characters.
    assert measured["cer_lines"] == pytest.approx(1 / 12)
    # End to end: a substitution, the missed line's 4 characters and the invented line's 2.
    assert measured["cer_end_to_end"] == pytest.approx((1 + 0 + 4 + 2) / 12)
    assert measured["main_lines"] == 3 and measured["main_line_f1"] == pytest.approx(4 / 5)
    assert measured["margin_lines"] == 0 and np.isnan(measured["margin_line_f1"])
    assert measured["lines_outside"] == 1 and measured["zones_outside"] == 1
    without = measure_page(page, Found(found.zones, found.lines)).row
    assert np.isnan(without["cer_lines"]) and without["line_f1"] == measured["line_f1"]


TEXTS = ["alpha one", "beta two", "gamma three", "delta four", "epsilon five", "zeta six", "eta seven"]


def stacked(order: list[int], texts: list[str] | None = None) -> tuple[truth.Page, Found]:
    """A page of text lines one under the other, the n-th 100 pixels below the (n-1)-th, listed in the ground truth in
    `order`; each found exactly by a found line, in the same order, whose texts are left to the test."""
    texts = texts or TEXTS
    lines = [truth.Line(f"t{n}", "DefaultLine", box(100, 100 * n, 500, 100 * n + 40), [(100, 100 * n + 35),
                        (500, 100 * n + 35)], texts[n], "z1", "MainZone") for n in order]  # fmt: skip
    page = truth.Page("ms_p1", "ms", Path("p1.jpg"), 800, 1000, [truth.Zone("z1", "MainZone", box(50, 50, 550, 900))],
                      lines, label="p1")  # fmt: skip
    found = Found([], [(line.polygon, line.baseline) for line in lines])
    return page, found


def checked(page: truth.Page, found: Found, readings: list[str], recognised: dict[str, str] | None = None) -> dict:
    found.texts = readings
    own = recognised if recognised is not None else {line.id: line.text or "" for line in page.lines}
    return measure_page(page, found, own).row


def test_a_page_with_three_shifted_texts_is_flagged_and_one_with_two_isnt() -> None:
    page, found = stacked(list(range(7)))
    # Lines 1 to 3 read the next line's text: their ground truth sits one line off.
    three = checked(page, found, [TEXTS[0], TEXTS[2], TEXTS[3], TEXTS[4], TEXTS[4], TEXTS[5], TEXTS[6]])
    assert three["lines_shifted"] == 3 and three["lines_read_empty"] == 0 and three["texts_misplaced"]
    two = checked(page, found, [TEXTS[0], TEXTS[2], TEXTS[3], TEXTS[3], TEXTS[4], TEXTS[5], TEXTS[6]])
    assert two["lines_shifted"] == 2 and not two["texts_misplaced"]
    # A reading close to its own text isn't shifted, even when a neighbour's is like it.
    near = checked(page, found, [TEXTS[0], "beta twx", TEXTS[2], TEXTS[3], TEXTS[4], TEXTS[5], TEXTS[6]])
    assert near["lines_shifted"] == 0


def test_neighbours_are_taken_in_the_ground_truths_order() -> None:
    # The ground truth lists the lines 0, 2, 1, 3, …: line 0's neighbours are 2 alone, though line 1 lies under it.
    page, found = stacked([0, 2, 1, 3, 4, 5, 6])
    readings = [line.text for line in page.lines]
    readings[0] = TEXTS[1]  # the line under it on the page, two away in the ground truth
    assert checked(page, found, list(readings))["lines_shifted"] == 0
    readings[0] = TEXTS[2]  # the next in the ground truth
    assert checked(page, found, list(readings))["lines_shifted"] == 1


def test_an_empty_reference_and_an_unmatched_line_are_skipped() -> None:
    texts = list(TEXTS)
    texts[1] = "  "
    page, found = stacked(list(range(7)), texts)
    readings = [TEXTS[0], TEXTS[2], TEXTS[3], TEXTS[4], TEXTS[4], TEXTS[5], TEXTS[6]]
    # Line 1 has no reference, so it counts neither way; lines 2 and 3 are shifted.
    assert checked(page, found, list(readings))["lines_shifted"] == 2
    # Without a found line on line 3, only line 2 is.
    unmatched = Found([], found.lines[:3] + found.lines[4:])
    assert checked(page, unmatched, readings[:3] + readings[4:])["lines_shifted"] == 1


def test_lines_read_empty_through_their_own_polygon_while_the_found_line_reads_text() -> None:
    page, found = stacked(list(range(7)))
    own = {line.id: line.text or "" for line in page.lines}
    for empty in ("t0", "t1", "t2"):
        own[empty] = " "
    flagged = checked(page, found, list(TEXTS), own)
    assert flagged["lines_read_empty"] == 3 and flagged["lines_shifted"] == 0 and flagged["texts_misplaced"]
    # Where the found line reads nothing either, the line doesn't count.
    readings = list(TEXTS)
    readings[2] = ""
    two = checked(page, found, readings, own)
    assert two["lines_read_empty"] == 2 and not two["texts_misplaced"]
    # An unmatched line doesn't count.
    unmatched = Found([], found.lines[1:])
    assert checked(page, unmatched, list(TEXTS[1:]), own)["lines_read_empty"] == 2
    # Without recognition the counts are missing and the page isn't flagged.
    plain = measure_page(page, Found([], found.lines)).row
    assert np.isnan(plain["lines_shifted"]) and np.isnan(plain["lines_read_empty"]) and not plain["texts_misplaced"]


def test_the_thresholds_are_settings() -> None:
    from layout_visual_complexity.difficulty import GroundTruthChecks

    page, found = stacked(list(range(7)))
    found.texts = [TEXTS[0], TEXTS[2], TEXTS[3], TEXTS[3], TEXTS[4], TEXTS[5], TEXTS[6]]
    own = {line.id: line.text or "" for line in page.lines}
    assert measure_page(page, found, own, checks=GroundTruthChecks(lines=2)).row["texts_misplaced"]
    # A CER against the neighbour's text under 0 can't be.
    assert measure_page(page, found, own, checks=GroundTruthChecks(shifted_neighbour=0.0)).row["lines_shifted"] == 0


def test_pages_whose_ground_truth_may_leave_text_out() -> None:
    from layout_visual_complexity.difficulty import leaves_text_out, texts_misplaced

    rows = pd.DataFrame({"lines_found": [100, 110, 40, 30, 0, np.nan], "lines_outside": [10, 11, 5, 4, 0, np.nan],
                         "texts_misplaced": [True, False, "True", "False", np.nan, False]})  # fmt: skip
    assert leaves_text_out(rows).tolist() == [True, True, True, False, False, False]
    assert leaves_text_out(rows, lines=6).tolist() == [True, True, False, False, False, False]
    assert leaves_text_out(rows, share=0.105).tolist() == [False, False, True, False, False, False]
    assert texts_misplaced(rows).tolist() == [True, False, True, False, False, False]
    assert not texts_misplaced(rows.drop(columns="texts_misplaced")).any()


def test_matching_iou_and_baselines() -> None:
    assert box_iou(np.array([[0, 0, 10, 10]]), np.array([[5, 0, 15, 10], [20, 20, 30, 30]])).tolist() == [[1 / 3, 0]]
    iou = polygon_iou([box(0, 0, 10, 10), box(0, 0, 10, 4)], [box(0, 0, 10, 4), box(0, 0, 10, 10)])
    assert iou[0, 1] == pytest.approx(1) and iou[1, 0] == pytest.approx(1) and iou[0, 0] == pytest.approx(0.4)
    assert match(iou) == [(0, 1, pytest.approx(1)), (1, 0, pytest.approx(1))]
    assert match(np.array([[0.4]])) == []
    assert baseline_distance([(0, 0), (100, 0)], [(0, 3), (100, 3)]) == pytest.approx(3)
    assert baseline_distance([(0, 0), (100, 0)], [(0, 0), (50, 0)]) == pytest.approx((0 + 25 * 51 / 101 + 0) / 2, 0.1)
    assert line_height(box(0, 0, 100, 20), [(0, 18), (100, 18)]) == pytest.approx(20)


def test_edit_distance_and_the_character_error_rate() -> None:
    assert edit_distance("kitten", "sitting") == 3 and edit_distance("", "abc") == 3 and edit_distance("ab", "ab") == 0
    assert edit_distance("ꝯ̃a", "ꝯa") == 1
    assert normalise("  éte   d'\tun  ") == "éte d' un"
    assert normalise("a b  c", spaces=False) == "abc"
    assert rate(3, 12) == 0.25 and np.isnan(rate(0, 0))
    rng = np.random.default_rng(3)
    for _ in range(200):
        first = "".join(rng.choice(list("abcꝯ "), rng.integers(0, 30)))
        second = "".join(rng.choice(list("abcꝯ "), rng.integers(0, 30)))
        table = np.zeros((len(first) + 1, len(second) + 1), int)
        table[:, 0], table[0, :] = range(len(first) + 1), range(len(second) + 1)
        for i in range(1, len(first) + 1):
            for j in range(1, len(second) + 1):
                table[i, j] = min(table[i - 1, j] + 1, table[i, j - 1] + 1,
                                  table[i - 1, j - 1] + (first[i - 1] != second[j - 1]))  # fmt: skip
        assert edit_distance(first, second) == table[-1, -1]


# The command, on synthetic pages.


@pytest.fixture
def difficulty_inputs(tmp_path: Path, tiny_vit: Path) -> tuple[list[str], Path]:
    from test_score_command import rankers

    sources = tmp_path / "sources"
    write_page(sources / "ms_one", "f1r")
    write_page(sources / "ms_one", "f2r", (620, 820))
    write_page(sources / "ms_two", "f5v-6r", (1200, 800), spread=True)
    return [str(sources)], rankers(tmp_path / "rankers", tiny_vit)


def difficulty_run(tmp_path: Path, name: str, sources: list[str], models_folder: Path, workers: int) -> dict:
    from layout_visual_complexity.difficulty import DifficultySettings, read_pages, recognition_pages, run_difficulty
    from layout_visual_complexity.scoring import ScoringModels
    from layout_visual_complexity.segmentation.fake import FakeRecognition, FakeSettings
    from test_score_command import steps

    settings = DifficultySettings(sources=sources, output=tmp_path / name, cache=tmp_path / "cache", jobs=3,
                                  device="cpu", spreads=SpreadSettings(pages="auto"))  # fmt: skip
    models = ScoringModels.load(models_folder)
    chosen = steps(models, workers)
    pages, _ = read_pages(settings, 1)
    chosen["recognition"] = FakeRecognition(FakeSettings(workers=workers, batch_size=2), recognition_pages(pages))
    return run_difficulty(settings, chosen, models=models)


def test_difficulty_rows_dont_depend_on_the_workers_a_rerun_resumes_and_nothing_is_kept_but_tables(
    tmp_path: Path, difficulty_inputs: tuple[list[str], Path]
) -> None:
    sources, models_folder = difficulty_inputs
    one = difficulty_run(tmp_path, "one", sources, models_folder, 1)
    many = difficulty_run(tmp_path, "many", sources, models_folder, 2)
    assert one["done"] == many["done"] == 4 and not one["failed"]
    first = pd.read_csv(tmp_path / "one" / "difficulty.tsv", sep="\t")
    second = pd.read_csv(tmp_path / "many" / "difficulty.tsv", sep="\t")
    pd.testing.assert_frame_equal(first.drop(columns="image"), second.drop(columns="image"))
    assert first["page"].tolist() == ["msone_f1r", "msone_f2r", "mstwo_f5v-6r_left", "mstwo_f5v-6r_right"]
    assert first[["expected_elo", "zone_f1", "line_f1", "cer_lines", "cer_end_to_end"]].notna().all().all()
    assert (first["lines"] == 7).all() and set(first["half"].fillna("")) == {"", "left", "right"}
    spreads = pd.read_csv(tmp_path / "one" / "spreads.tsv", sep="\t")
    assert spreads["page"].tolist() == ["mstwo_f5v-6r"] and spreads["found"].all()
    again = difficulty_run(tmp_path, "one", sources, models_folder, 2)
    assert again["pages_this_run"] == 0
    pd.testing.assert_frame_equal(pd.read_csv(tmp_path / "one" / "difficulty.tsv", sep="\t"), first)
    folder = tmp_path / "one"
    kept = [path for path in folder.rglob("*") if path.is_file()]
    assert not [path for path in kept if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".tif", ".tiff"}]
    assert not (tmp_path / "cache" / "tokens").exists()
    lines = pd.read_parquet(folder / "lines.parquet")
    assert set(lines["source"]) == {"truth", "found"} and lines.loc[lines["source"] == "truth", "text"].notna().all()
    zones = pd.read_parquet(folder / "zones.parquet")
    assert set(zones["class"]) <= {"MainZone", "MarginTextZone"}
    manifest = json.loads((folder / "manifest.json").read_text())
    assert sum(Path(item["path"]).suffix == ".xml" for item in manifest["inputs"]) == 3
