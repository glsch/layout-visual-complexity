"""A small synthetic dataset in the published structure, and a fake Zenodo record of it."""

import hashlib
import json
from pathlib import Path

import httpx2
import numpy as np
import pandas as pd
import pytest

from layout_visual_complexity.ranking.dataset import Dataset, write_dataset

MANUSCRIPTS = {"msa": "A", "msb": "A", "msc": "B", "msd": "B"}
FOLIOS = ["1r", "1v", "2r"]
HOST = "https://iiif.example.org"


def published(seed: int = 7) -> Dataset:
    """Four manuscripts of three pages each, and 160 judgments drawn from strengths that grow with the page."""
    rng = np.random.default_rng(seed)
    pages = [(m, f) for m in MANUSCRIPTS for f in FOLIOS]
    ids = [f"{m}_{f}" for m, f in pages]
    images = pd.DataFrame(
        {
            "image": ids,
            "manuscript": [m for m, _ in pages],
            "folio": [f"000{f}" for _, f in pages],
            "canvas": [f"{HOST}/{m}/canvas/{f}" for m, f in pages],
            "image_url": [f"{HOST}/images/{m}/{f}/full/full/0/default.jpg" for m, f in pages],
            "image_service": [f"{HOST}/images/{m}/{f}" for m, f in pages],
            "width": pd.array([1000] * len(pages), dtype="Int64"),
            "height": pd.array([1500] * len(pages), dtype="Int64"),
            "work": ["Summa"] * len(pages),
            "text": [""] * len(pages),
            "text_language": ["lat"] * len(pages),
        }
    )
    manuscripts = pd.DataFrame(
        [
            {
                "manuscript": m,
                "shelfmark": m.upper(),
                "library": "A library",
                "city": "Praha",
                "country": "Česko",
                "title": f"Title of {m}",
                "date": "15. století",
                "date_not_before": 1401,
                "date_not_after": 1500,
                "date_edtf": "1401/1500",
                "place_of_origin": "Čechy",
                "material": "paper",
                "height_mm": 300,
                "width_mm": 210,
                "languages": "lat",
                "texts": 1,
                "category": category,
                "secondary_categories": "",
                "pages": len(FOLIOS),
                "canvases": 5,
                "iiif_manifest": f"{HOST}/{m}/manifest",
                "catalogue_record": f"https://catalogue.example.org/{m}",
                "mirabile_id": "",
                "biblissima_id": "",
                "hscensus_id": "",
                "licence": "https://creativecommons.org/licenses/by-nc/4.0/",
            }
            for m, category in MANUSCRIPTS.items()
        ]
    )
    for column in ("date_not_before", "date_not_after", "height_mm", "width_mm", "texts", "pages", "canvases"):
        manuscripts[column] = manuscripts[column].astype("Int64")
    strength = np.linspace(-2, 2, len(ids))
    rows = []
    for number in range(160):
        first, second = rng.choice(len(ids), 2, replace=False)
        chosen = first if rng.random() < 1 / (1 + np.exp(strength[second] - strength[first])) else second
        rows.append(
            {
                "judgment": f"judgment_{number + 1:05d}",
                "session": f"session_{number // 16 + 1:04d}",
                "annotator": f"annotator_{number // 32 + 1:03d}",
                "campaign": "main" if number < 120 else "bridge",
                "date": "2025-03-01",
                "left": ids[first],
                "right": ids[second],
                "chosen": ids[chosen],
                "response_ms": int(rng.integers(500, 5000)),
            }
        )
    judgments = pd.DataFrame(rows)
    judgments["response_ms"] = judgments["response_ms"].astype("Int64")
    return Dataset(judgments, images, manuscripts)


def write_published(folder: Path, dataset: Dataset | None = None) -> Path:
    write_dataset(dataset or published(), folder, "# A synthetic dataset\n")
    return folder


class FakeZenodo:
    """Serves one record, 10.5281/zenodo.1234, listing the files of a folder; `corrupt` names a file whose listed
    MD5 is wrong."""

    def __init__(self, folder: Path, corrupt: str | None = None) -> None:
        self.folder, self.corrupt = folder, corrupt
        self.requests: list[str] = []

    def __call__(self, request: httpx2.Request) -> httpx2.Response:
        url = str(request.url)
        self.requests.append(url)
        if url == "https://zenodo.org/api/records/1234":
            files = []
            for path in sorted(self.folder.iterdir()):
                md5 = hashlib.md5(path.read_bytes()).hexdigest()
                if path.name == self.corrupt:
                    md5 = "0" * 32
                link = f"https://zenodo.org/api/records/1234/files/{path.name}/content"
                files.append({"key": path.name, "checksum": f"md5:{md5}", "links": {"content": link}})
            return httpx2.Response(200, content=json.dumps({"id": 1234, "files": files}).encode())
        prefix = "https://zenodo.org/api/records/1234/files/"
        if url.startswith(prefix):
            name = url.removeprefix(prefix).removesuffix("/content")
            return httpx2.Response(200, content=(self.folder / name).read_bytes())
        return httpx2.Response(404)

    def transport(self) -> httpx2.MockTransport:
        return httpx2.MockTransport(self)


@pytest.fixture
def dataset_folder(tmp_path: Path) -> Path:
    return write_published(tmp_path / "published")
