"""A small synthetic study: an export with annotators' names, its images, manuscripts, a merge table, and a fake
IIIF server for the manifests and thumbnails."""

import io
import json
from dataclasses import dataclass
from pathlib import Path

import httpx2
import numpy as np
import pytest

# Spellings as an app might record them: the first two fold together, the next two are merged by the table.
SPELLINGS = ["Jan Hajič", "jan_hajic", "Ukázková, Zdeňka", "Zdenka Ukazkova", "Anna Nováková", "Petr Svoboda"]
MERGES = {"Ukázková, Zdeňka": "Zdeňka Ukázková", "Zdenka Ukazkova": "Zdeňka Ukázková"}
PERSONS = ["Zdeňka Ukázková"]
MANUSCRIPTS = {"msa": ("MS A", "A"), "msb": ("MS B", "A"), "msc": ("MS C", "B"), "msd": ("MS D", "B")}
FOLIOS = ["0001r", "0001v", "0002r"]
HOST = "https://iiif.example.org"


@dataclass
class Study:
    folder: Path
    export: Path
    mapping: Path
    metadata: Path
    merges: Path
    images: list[str]


def image_ids() -> list[str]:
    return [f"{MANUSCRIPTS[m][0].replace(' ', '_')}_{folio}" for m in MANUSCRIPTS for folio in FOLIOS]


def combination(rng: np.random.Generator, images: list[str], strength: dict[str, float]) -> dict:
    first, second = rng.choice(images, size=2, replace=False)
    p = 1 / (1 + np.exp(strength[second] - strength[first]))
    chosen, other = (first, second) if rng.random() < p else (second, first)
    left_is_chosen = rng.random() < 0.5
    url = "http://app.example.org/images/{}.jpg"
    return {
        "entropyRated": True,
        "isCalibration": False,
        "thinkingTime": int(rng.integers(500, 5000)),
        "winnerHigherEntropy": {
            "shelfmark": "x",
            "url": url.format(chosen),
            "position": "left" if left_is_chosen else "right",
        },
        "loserLowerEntropy": {
            "shelfmark": "x",
            "url": url.format(other),
            "position": "right" if left_is_chosen else "left",
        },
    }


def write_study(folder: Path, seed: int = 7) -> Study:
    rng = np.random.default_rng(seed)
    images = image_ids()
    strength = {image: float(value) for image, value in zip(images, np.linspace(-2, 2, len(images)), strict=True)}
    sessions = []
    sources = [None, None, None, "bridge", None, "exhibition", None, None, "bridge", None, None, "exhibition"]
    for number, source in enumerate(sources):
        combinations = [combination(rng, images, strength) for _ in range(12)]
        combinations.insert(3, {"entropyRated": False, "isCalibration": False, "folio1": None, "folio2": None})
        if number % 4 == 0:
            calibration = combination(rng, images[:2], strength)
            calibration["isCalibration"] = True
            combinations.insert(0, calibration)
        sessions.append(
            {
                "sessionID": f"app-{1000 - number}",
                "user": SPELLINGS[number % len(SPELLINGS)],
                "source": source,
                "timestamp": f"2025-03-{1 + number:02d}T10:{number:02d}:00.000Z",
                "completed": True,
                "combinations": combinations,
            }
        )
    export = folder / "export.json"
    export.write_text(json.dumps({"sessions": sessions}, ensure_ascii=False), encoding="utf-8")
    mapping = {
        "images": {
            f"{image}.jpg": {
                "filename": f"{image}.jpg",
                "shelfmark": MANUSCRIPTS[key][0],
                "normalized_shelfmark": key,
                "folio_or_page_from_filename": image.rsplit("_", 1)[1],
                "iiif_manifest_url": f"{HOST}/{key}/manifest",
            }
            for key in MANUSCRIPTS
            for image in images
            if image.startswith(MANUSCRIPTS[key][0].replace(" ", "_") + "_")
        }
    }
    mapping_path = folder / "mapping.json"
    mapping_path.write_text(json.dumps(mapping), encoding="utf-8")
    metadata_path = folder / "metadata.jsonl"
    with metadata_path.open("w", encoding="utf-8") as stream:
        for key, (shelfmark, category) in MANUSCRIPTS.items():
            record = {
                "normalized_shelfmark": key,
                "shelfmark": shelfmark,
                "category": category,
                "category_name": {"A": "Handbooks", "B": "Religious"}[category],
                "secondary_categories": [],
                "canvas_count": 4,
                "title": f"Title of {shelfmark}",
                "date": "1450-1460",
                "location": None,
                "material": "paper",
                "height_mm": 300,
                "width_mm": 210,
                "repository": "A library",
                "city": "Praha",
                "iiif_manifest_url": f"{HOST}/{key}/manifest",
                "catalogue_text": "Free text that must not be published.",
            }
            stream.write(json.dumps(record) + "\n")
    merges_path = folder / "merges.tsv"
    lines = ["spelling\tperson\treason"] + [f"{s}\t{p}\tsame person" for s, p in MERGES.items()]
    merges_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return Study(folder, export, mapping_path, metadata_path, merges_path, images)


def manifest(key: str) -> dict:
    """A IIIF Presentation 2 manifest: a cover, then the folios. Manuscript D's has folio 2r twice."""
    labels = ["FC", "1r", "fol. 1v", "f. 2 r", "3r"] if key != "msd" else ["FC", "1r", "1v", "2r", "2r"]
    canvases = []
    for number, label in enumerate(labels):
        service = f"{HOST}/images/{key}/{number}"
        canvases.append(
            {
                "@id": f"{HOST}/{key}/canvas/{number}",
                "@type": "sc:Canvas",
                "label": label,
                "width": 1000,
                "height": 1500,
                "images": [
                    {
                        "@type": "oa:Annotation",
                        "motivation": "sc:painting",
                        "resource": {
                            "@id": f"{service}/full/full/0/default.jpg",
                            "@type": "dctypes:Image",
                            "width": 1000,
                            "height": 1500,
                            "service": {"@context": "http://iiif.io/api/image/2/context.json", "@id": service},
                        },
                    }
                ],
            }
        )
    return {
        "@context": "http://iiif.io/api/presentation/2/context.json",
        "@id": f"{HOST}/{key}/manifest",
        "@type": "sc:Manifest",
        "license": "https://creativecommons.org/licenses/by-nc/4.0/",
        "sequences": [{"canvases": canvases}],
    }


def jpeg() -> bytes:
    import matplotlib.image

    buffer = io.BytesIO()
    matplotlib.image.imsave(buffer, np.full((20, 14, 3), 0.8), format="jpg")
    return buffer.getvalue()


class FakeServer:
    """Answers manifest and image requests, and counts them."""

    def __init__(self) -> None:
        self.requests: list[str] = []
        self.image = jpeg()

    def __call__(self, request: httpx2.Request) -> httpx2.Response:
        url = str(request.url)
        self.requests.append(url)
        if url.endswith("/manifest"):
            return httpx2.Response(200, json=manifest(url.split("/")[-2]))
        if url.endswith("default.jpg"):
            return httpx2.Response(200, content=self.image, headers={"content-type": "image/jpeg"})
        return httpx2.Response(404)

    def transport(self) -> httpx2.MockTransport:
        return httpx2.MockTransport(self)


@pytest.fixture
def study(tmp_path: Path) -> Study:
    folder = tmp_path / "inputs"
    folder.mkdir()
    return write_study(folder)


@pytest.fixture
def server() -> FakeServer:
    return FakeServer()
