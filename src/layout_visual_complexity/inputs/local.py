"""Local images: one image, or the images of a folder in natural order (`2.jpg` before `10.jpg`)."""

import re
from pathlib import Path

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".jp2", ".webp", ".bmp"}


def natural_key(text: str) -> tuple[tuple[int, int | str], ...]:
    """Sort key putting `2.jpg` before `10.jpg`."""
    return tuple(
        (0, int(part)) if part.isdigit() else (1, part.casefold()) for part in re.split(r"(\d+)", text) if part
    )


def is_image(path: Path) -> bool:
    return path.suffix.lower() in IMAGE_SUFFIXES


def images_in(folder: Path) -> list[Path]:
    """The images in a folder, in natural order; other files (ALTO beside them, say) are left out."""
    found = [child for child in folder.iterdir() if child.is_file() and not child.name.startswith(".")]
    return sorted((child for child in found if is_image(child)), key=lambda path: natural_key(path.name))
