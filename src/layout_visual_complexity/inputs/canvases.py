"""Finding each page's canvas in its IIIF manifest by its folio label.

Labels are compared normalised: case, spaces and leading zeros ignored, and a leading `folio`, `fol.`, `f.` or `p.`
(with or without the dot) dropped before a digit; `r` and `v` are kept. So `0008v` matches `8v`, `fol. 8v`,
`f. 8 v` and `008v`. Exactly one canvas must match; pages with none or several are left unresolved, with the
nearest labels listed for review.
"""

import difflib
import re
from dataclasses import dataclass

import pandas as pd

from layout_visual_complexity.download import Cache, HttpClient, cached_json
from layout_visual_complexity.inputs.iiif import Canvas, canvases
from layout_visual_complexity.logs import logger

PREFIX = re.compile(r"^(folio|fol\.?|f\.?|p\.?)(?=\d)")
LEADING_ZEROS = re.compile(r"^0+(?=\d)")
NEAREST = 5


def normalise_label(label: str) -> str:
    """A folio label for comparison: `fol. 008 v` → `8v`."""
    text = "".join(label.casefold().split())
    text = PREFIX.sub("", text)
    return LEADING_ZEROS.sub("", text)


@dataclass(frozen=True)
class CanvasMatch:
    """The canvases whose label matches a page's, and the nearest labels when not exactly one does."""

    label: str
    matches: tuple[Canvas, ...]
    nearest: tuple[Canvas, ...]

    @property
    def canvas(self) -> Canvas | None:
        return self.matches[0] if len(self.matches) == 1 else None


def match_label(label: str, found: list[Canvas], nearest: int = NEAREST) -> CanvasMatch:
    """The canvases labelled `label`; when there isn't exactly one, the `nearest` closest labels too."""
    wanted = normalise_label(label)
    matches = tuple(canvas for canvas in found if canvas.label and normalise_label(canvas.label) == wanted)
    if len(matches) == 1:
        return CanvasMatch(label, matches, ())
    if matches:
        return CanvasMatch(label, matches, matches)
    labelled = [canvas for canvas in found if canvas.label]
    keys = [normalise_label(canvas.label or "") for canvas in labelled]
    close = difflib.get_close_matches(wanted, keys, n=nearest, cutoff=0.0)
    picked = []
    for key in close:
        canvas = labelled[keys.index(key)]
        if canvas not in picked:
            picked.append(canvas)
    return CanvasMatch(label, (), tuple(picked))


def describe_canvases(found: tuple[Canvas, ...]) -> str:
    """`8v (canvas 17); 9r (canvas 18)`."""
    return "; ".join(f"{canvas.label} (canvas {canvas.position})" for canvas in found)


def resolve_canvases(
    images: pd.DataFrame, client: HttpClient, cache: Cache
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    """Fill `canvas`, `image_service`, `width` and `height` of the images whose label matches one canvas.

    Returns the images, the unresolved ones (image, manifest, label, matches, nearest labels), and counts.
    """
    images = images.copy()
    unresolved = []
    manifests: dict[str, list[Canvas]] = {}
    licences: dict[str, str] = {}
    fetched_before = len(client.records)
    for url in sorted(images["iiif_manifest"].unique()):
        manifest = cached_json(client, cache, url)
        manifests[url] = canvases(manifest, url)
        licence = manifest.get("license") or manifest.get("rights") or ""
        licences[url] = licence if isinstance(licence, str) else ";".join(map(str, licence))
    for position, row in images.iterrows():
        match = match_label(str(row["folio_label"]), manifests[row["iiif_manifest"]])
        canvas = match.canvas
        if canvas is None:
            unresolved.append(
                {
                    "image": row["image"],
                    "manifest": row["iiif_manifest"],
                    "label": row["folio_label"],
                    "matches": len(match.matches),
                    "nearest": describe_canvases(match.nearest),
                }
            )
            continue
        images.at[position, "canvas"] = canvas.id
        images.at[position, "image_service"] = canvas.image_service
        images.at[position, "width"] = canvas.width
        images.at[position, "height"] = canvas.height
    unresolved_table = pd.DataFrame(unresolved, columns=["image", "manifest", "label", "matches", "nearest"])
    licence_counts = pd.Series(licences).value_counts().to_dict()
    counts = {
        "manifests": len(manifests),
        "manifests_downloaded": len(client.records) - fetched_before,
        "canvases_found": int(images["canvas"].notna().sum()),
        "unresolved": len(unresolved_table),
        "unresolved_without_match": int((unresolved_table["matches"] == 0).sum()),
        "unresolved_with_several": int((unresolved_table["matches"] > 1).sum()),
        "manifest_licences": {str(key): int(value) for key, value in licence_counts.items()},
    }
    logger.info("canvases: %d found, %d unresolved", counts["canvases_found"], counts["unresolved"])
    return images, unresolved_table, counts
