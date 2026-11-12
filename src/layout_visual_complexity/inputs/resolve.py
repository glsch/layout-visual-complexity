"""From sources to documents and their pages: kinds recognised, manifests read, pages selected and named."""

import json
import re
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlsplit

from layout_visual_complexity.inputs import Document, InputError, PageSource, iiif
from layout_visual_complexity.inputs.lists import Entry, is_page_table, read_list, read_page_table
from layout_visual_complexity.inputs.local import IMAGE_SUFFIXES, images_in, is_image
from layout_visual_complexity.inputs.selection import (
    document_id,
    label_filter,
    page_ids,
    passes,
    select,
)

FetchJson = Callable[[str], dict[str, Any]]

# A IIIF Image API request: the service, then region, size, rotation, and quality with the format.
IMAGE_REQUEST = re.compile(r"^(?P<service>.+?)/(?:full|square|\d+,\d+,\d+,\d+|pct:[^/]+)/[^/]+/!?\d+/\w+\.\w+$")
GENERIC_SEGMENTS = {"manifest", "manifest.json", "iiif", "presentation", "v2", "v3", "2", "3", "api", "info.json"}
LIST_SUFFIXES = {".txt", ".tsv", ".list"}


@dataclass(frozen=True)
class InputSettings:
    """Which pages to take from the sources.

    Args:
        pages: A selection for the sources that don't give their own: pages and ranges by index from 1 (`5-12`) or by
            canvas label (`10r-20v`), comma-separated; all pages when empty. A label of digits only is written
            `label:12`.
        label_filter: A regular expression a page's folio (its label in lower case, without spaces or leading zeros)
            must match as a whole, such as `\\d+[rv]` for leaves; every page when unset.
        languages: Languages to prefer in IIIF labels and metadata, after `none`.
    """

    pages: str = ""
    label_filter: str | None = None
    languages: list[str] = field(default_factory=lambda: ["en"])


def url_identifier(url: str) -> str:
    """The identifier in a URL: the `IIIF` query value (Phaidra), or the last path segment that isn't generic."""
    parts = urlsplit(url)
    query = parse_qs(parts.query)
    if "IIIF" in query:
        return query["IIIF"][0]
    for segment in reversed([part for part in parts.path.split("/") if part]):
        if segment.lower() not in GENERIC_SEGMENTS:
            return segment.removesuffix(".json")
    return parts.netloc


def is_image_service_info(data: dict[str, Any]) -> bool:
    contexts = " ".join(str(value) for value in iiif._list(data.get("@context")))
    return "iiif.io/api/image" in contexts or str(data.get("protocol", "")) == "http://iiif.io/api/image"


class Resolver:
    """Reads sources into documents. Remote JSON comes through `fetch_json`, which caches it."""

    def __init__(self, settings: InputSettings, fetch_json: FetchJson | None = None, threads: int = 8) -> None:
        self.settings = settings
        self.fetch_json = fetch_json
        self.threads = threads
        self.filter = label_filter(settings.label_filter)

    def _json(self, url: str) -> dict[str, Any]:
        if self.fetch_json is None:
            raise InputError(f"{url}: reading remote sources needs a connection")
        return self.fetch_json(url)

    def resolve(self, sources: list[str]) -> list[Document]:
        """The documents of all sources, in order; documents of the same id merge, their pages each once."""
        entries = [Entry(source) for source in sources]
        with ThreadPoolExecutor(max_workers=self.threads, thread_name_prefix="inputs") as pool:
            found = list(pool.map(self.entry, entries))
        return merge([document for documents in found for document in documents])

    def entry(self, entry: Entry) -> list[Document]:
        try:
            if "://" in entry.source and not entry.source.startswith("file:"):
                return self._remote(entry)
            return self._local(entry)
        except InputError as error:
            message = str(error)
            raise InputError(message if message.startswith(entry.source) else f"{entry.source}: {message}") from None

    # What to take from a document.

    def _positions(self, entry: Entry, labels: list[str | None], canvases: list[str | None]) -> list[int]:
        """The positions an entry asks for, with the default selection if it asks for none, then filtered."""
        chosen: set[int] = set()
        if entry.everything:
            chosen.update(select(self.settings.pages, labels))
        else:
            if entry.selection.strip():
                chosen.update(select(entry.selection, labels))
            for canvas in entry.canvases:
                matches = [
                    i for i, other in enumerate(canvases, start=1) if other and other.rstrip("/") == canvas.rstrip("/")
                ]
                if len(matches) != 1:
                    raise InputError(f"no single canvas {canvas!r}: {len(matches)} found")
                chosen.add(matches[0])
            for label in entry.labels:
                chosen.update(select(f"label:{label}", labels))
        return [position for position in sorted(chosen) if passes(labels[position - 1], self.filter)]

    # Remote sources.

    def _remote(self, entry: Entry) -> list[Document]:
        url = entry.source
        request = IMAGE_REQUEST.match(url)
        if request:
            return [self._service(entry, request["service"])]
        if Path(urlsplit(url).path).suffix.lower() in IMAGE_SUFFIXES:
            return [self._image_url(entry)]
        if url.endswith("/info.json") or "IIIF=" in urlsplit(url).query:
            return [self._service(entry, url.removesuffix("/info.json"))]
        data = self._json(url)
        if iiif.presentation_version(data) or iiif._type(data) == "Manifest":
            return [self._manifest(entry, data, url)]
        if is_image_service_info(data):
            return [self._service(entry, iiif._id(data) or url.removesuffix("/info.json"))]
        raise InputError("neither a IIIF manifest nor an image service")

    def _manifest(self, entry: Entry, data: dict[str, Any], url: str) -> Document:
        languages = self.settings.languages
        try:
            canvases = iiif.canvases(data, url, languages)
        except iiif.ManifestError as error:
            raise InputError(str(error)) from None
        shelfmark = iiif.shelfmark(data, languages)
        document = document_id(shelfmark or url_identifier(url))
        labels = [canvas.label for canvas in canvases]
        ids = page_ids(document, labels)
        pages = []
        for position in self._positions(entry, labels, [canvas.id for canvas in canvases]):
            canvas = canvases[position - 1]
            if not (canvas.image_service or canvas.image_url):
                raise InputError(f"canvas {position} ({canvas.label}) has no image")
            pages.append(
                PageSource(
                    page=ids[position - 1],
                    document=document,
                    position=position,
                    source=url,
                    label=canvas.label,
                    canvas=canvas.id,
                    image_service=canvas.image_service,
                    image_api=canvas.image_api,
                    image_url=None if canvas.image_service else canvas.image_url,
                    width=canvas.width,
                    height=canvas.height,
                )
            )
        return Document(document, url, pages, iiif.text(data.get("label"), languages))

    def _service(self, entry: Entry, service: str) -> Document:
        document = document_id(url_identifier(service))
        page = PageSource(
            page=page_ids(document, [None])[0],
            document=document,
            position=1,
            source=entry.source,
            image_service=service,
        )
        return Document(document, entry.source, [page])

    def _image_url(self, entry: Entry) -> Document:
        path = Path(urlsplit(entry.source).path)
        document = document_id(path.parent.name or urlsplit(entry.source).netloc)
        page = PageSource(
            page=page_ids(document, [path.stem])[0],
            document=document,
            position=1,
            source=entry.source,
            label=path.stem,
            image_url=entry.source,
        )
        return Document(document, entry.source, [page])

    # Local sources.

    def _local(self, entry: Entry) -> list[Document]:
        path = Path(urlsplit(entry.source).path) if entry.source.startswith("file:") else Path(entry.source)
        if not path.exists():
            raise InputError("no such file or folder")
        if path.is_dir():
            return [self._folder(entry, path)]
        suffix = path.suffix.lower()
        if suffix == ".json":
            return [self._manifest(entry, json.loads(path.read_text(encoding="utf-8")), path.resolve().as_uri())]
        if is_image(path):
            return [self._local_image(entry, path)]
        if suffix == ".tsv" and is_page_table(path):
            return self._page_table(path)
        if suffix in LIST_SUFFIXES:
            listed = read_list(path)
            with ThreadPoolExecutor(max_workers=self.threads, thread_name_prefix="inputs") as pool:
                found = list(pool.map(self.entry, listed))
            return [document for documents in found for document in documents]
        raise InputError("not an image, a folder, a IIIF manifest, a list or a page table")

    def _folder(self, entry: Entry, folder: Path) -> Document:
        files = images_in(folder)
        if not files:
            raise InputError("the folder holds no images")
        document = document_id(folder.name)
        stems = [file.stem for file in files]
        ids = page_ids(document, stems)
        pages = [
            PageSource(
                page=ids[p - 1],
                document=document,
                position=p,
                source=str(folder),
                label=stems[p - 1],
                path=str(files[p - 1]),
            )
            for p in self._positions(entry, list(stems), [None] * len(files))
        ]
        return Document(document, str(folder), pages)

    def _local_image(self, entry: Entry, path: Path) -> Document:
        document = document_id(path.resolve().parent.name)
        page = PageSource(
            page=page_ids(document, [path.stem])[0],
            document=document,
            position=1,
            source=str(path),
            label=path.stem,
            path=str(path),
        )
        return Document(document, str(path), [page])

    def _page_table(self, path: Path) -> list[Document]:
        documents: dict[str, Document] = {}
        for row in read_page_table(path):
            label = row.get("folio") or None
            if not passes(label, self.filter):
                continue
            document = documents.setdefault(row["manuscript"], Document(row["manuscript"], str(path), []))
            document.pages.append(
                PageSource(
                    page=row["image"],
                    document=row["manuscript"],
                    position=len(document.pages) + 1,
                    source=str(path),
                    label=label,
                    canvas=row.get("canvas") or None,
                    image_service=row.get("image_service") or None,
                    image_url=None if row.get("image_service") else row.get("image_url") or None,
                    width=int(row["width"]) if row.get("width") else None,
                    height=int(row["height"]) if row.get("height") else None,
                )
            )
        return list(documents.values())


def merge(documents: list[Document]) -> list[Document]:
    """Documents of the same id as one, in the order first seen; a page named twice for the same image counts once,
    for two different images it's refused."""
    merged: dict[str, Document] = {}
    images: dict[str, str] = {}
    for document in documents:
        target = merged.setdefault(document.id, replace(document, pages=[]))
        for page in document.pages:
            if page.page in images:
                if images[page.page] != page.image:
                    raise InputError(f"two images are both named {page.page!r}: {images[page.page]} and {page.image}")
                continue
            images[page.page] = page.image
            target.pages.append(page)
    return list(merged.values())
