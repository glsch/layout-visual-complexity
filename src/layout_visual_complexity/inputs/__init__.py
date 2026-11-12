"""Where pages come from: IIIF manifests and image services, image URLs, local images and folders, lists of these,
and a dataset's page table. Each page keeps its source, canvas, image service and the full image's size."""

from dataclasses import dataclass


class InputError(ValueError):
    """A source, selection or list that can't be read as given. The message says which and why."""


@dataclass(frozen=True)
class PageSource:
    """One page and where its image comes from.

    `page` is its id, `document` its document's. The image is a IIIF image service (with its Image API version), an
    image URL fetched as it is, or a local file. `width` and `height` are the full image's, when the source gives
    them; coordinates are given in those pixels.
    """

    page: str
    document: str
    position: int
    source: str
    label: str | None = None
    canvas: str | None = None
    image_service: str | None = None
    image_api: int = 2
    image_url: str | None = None
    path: str | None = None
    width: int | None = None
    height: int | None = None

    @property
    def image(self) -> str:
        """What identifies the image: its service, URL or file."""
        return self.image_service or self.image_url or self.path or ""


@dataclass
class Document:
    """A document's pages, in order: a manifest's canvases, a folder's images, or a table's rows."""

    id: str
    source: str
    pages: list[PageSource]
    label: str | None = None
