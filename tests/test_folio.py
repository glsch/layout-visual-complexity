import cv2
import numpy as np
import pytest

from layout_visual_complexity.segmentation.folio import FolioSettings, find_folio, polygon_mask


def page(corners: list[tuple[int, int]], size: tuple[int, int] = (900, 1200), lines: bool = True) -> np.ndarray:
    """A bright leaf with the given corners on a dark ground, with dark lines of writing on it."""
    width, height = size
    image = np.full((height, width, 3), 35, np.uint8)
    cv2.fillPoly(image, [np.array(corners, np.int32)], (232, 222, 200))
    if lines:
        xs = [x for x, _ in corners]
        ys = [y for _, y in corners]
        for y in range(min(ys) + 80, max(ys) - 80, 28):
            cv2.line(image, (min(xs) + 90, y), (max(xs) - 90, y), (60, 50, 40), 7)
    return image


def iou(first: list[tuple[float, float]], second: list[tuple[float, float]], shape: tuple[int, int]) -> float:
    a, b = polygon_mask(np.array(first), shape), polygon_mask(np.array(second), shape)
    return float(np.logical_and(a, b).sum() / np.logical_or(a, b).sum())


def test_a_bright_skewed_quadrilateral_on_a_dark_ground_is_found() -> None:
    corners = [(120, 70), (800, 110), (770, 1130), (95, 1090)]
    image = page(corners)
    box = (80.0, 60.0, 820.0, 1150.0)
    folio = find_folio(image, box)
    assert folio.method == "otsu"
    assert len(folio.quadrilateral) == 4
    assert iou(folio.quadrilateral, corners, image.shape[:2]) > 0.95
    assert folio.iou is not None and 0.85 < folio.iou < 1.0


def test_the_ink_on_the_leaf_doesnt_make_otsus_region_implausible() -> None:
    corners = [(60, 50), (840, 50), (840, 1150), (60, 1150)]
    image = page(corners)
    for y in range(130, 1080, 14):  # dense writing: much of the leaf is dark
        cv2.line(image, (150, y), (750, y), (40, 30, 30), 9)
    assert find_folio(image, (50.0, 40.0, 850.0, 1160.0)).method == "otsu"


@pytest.mark.parametrize(
    "corners",
    [
        [(0, 0), (900, 0), (900, 1200), (0, 1200)],  # the leaf fills the image: over 98% of it
        [(400, 500), (520, 500), (520, 640), (400, 640)],  # a small bright patch: under half of it
    ],
)
def test_an_implausible_otsu_region_falls_back_to_the_box(corners: list[tuple[int, int]]) -> None:
    box = (100.0, 100.0, 800.0, 1100.0)
    folio = find_folio(page(corners, lines=False), box)
    assert folio.method == "yolo"
    assert folio.quadrilateral == [(100.0, 100.0), (800.0, 100.0), (800.0, 1100.0), (100.0, 1100.0)]


def test_a_region_far_from_a_rectangle_falls_back_to_the_box() -> None:
    triangle = [(80, 80), (860, 600), (80, 1150)]
    folio = find_folio(page(triangle, lines=False), (60.0, 60.0, 870.0, 1160.0))
    assert folio.method == "yolo"


def test_no_box_gives_the_whole_image() -> None:
    folio = find_folio(page([(120, 70), (800, 110), (770, 1130), (95, 1090)]), None)
    assert folio.method == "image"
    assert folio.quadrilateral == [(0.0, 0.0), (900.0, 0.0), (900.0, 1200.0), (0.0, 1200.0)]
    assert folio.iou is None


def test_a_box_covering_half_the_leaf_gives_the_whole_leaf_and_a_low_iou() -> None:
    corners = [(100, 80), (800, 80), (800, 1120), (100, 1120)]
    image = page(corners)
    folio = find_folio(image, (100.0, 80.0, 450.0, 1120.0), FolioSettings())
    assert folio.method == "otsu"
    assert iou(folio.quadrilateral, corners, image.shape[:2]) > 0.95
    assert folio.iou is not None and folio.iou < FolioSettings().disagreement
