"""IIIF Presentation manifests, versions 2 and 3: canvases, their labels, image services and sizes."""

import json
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any
from urllib.parse import urljoin

JSON = dict[str, Any]

LABEL_KEYS = {"label", "folio", "foliation", "page", "pagination", "seite", "blatt", "page number"}


class ManifestError(ValueError):
    """A manifest this code can't read."""


@dataclass(frozen=True)
class Canvas:
    """One canvas of a manifest: its position (from 1), id, label, image service and the full image's size."""

    position: int
    id: str | None
    label: str | None
    image_service: str | None
    image_url: str | None
    width: int | None
    height: int | None


def _list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else ([] if value is None else [value])


def _id(value: Any) -> str:
    if isinstance(value, str):
        return value
    return str(value.get("id") or value.get("@id") or "") if isinstance(value, dict) else ""


def _type(value: JSON) -> str:
    kind = value.get("type") or value.get("@type") or ""
    return str(kind[0] if isinstance(kind, list) and kind else kind).split(":")[-1]


def presentation_version(data: JSON) -> int | None:
    """2 or 3 for a IIIF Presentation resource, None for anything else."""
    contexts = " ".join(str(context) for context in _list(data.get("@context")))
    if "iiif.io/api/presentation/3" in contexts:
        return 3
    if "iiif.io/api/presentation/2" in contexts:
        return 2
    return None


def text(value: Any, languages: Sequence[str] = ("en",)) -> str | None:
    """A label or value as text. Language maps: `none` first, then the given languages, then the first present."""
    if value is None:
        return None
    if isinstance(value, str):
        return " ".join(value.split()) or None
    if isinstance(value, list):
        if value and all(isinstance(item, dict) and "@value" in item for item in value):
            for language in (None, *languages):
                chosen = [item for item in value if item.get("@language") == language]
                if chosen:
                    return text([item["@value"] for item in chosen], languages)
            return text(value[0]["@value"], languages)
        parts = [part for part in (text(item, languages) for item in value) if part]
        return "; ".join(dict.fromkeys(parts)) or None
    if isinstance(value, dict):
        if "@value" in value:
            return text(value["@value"], languages)
        for language in ("none", *languages):
            if language in value:
                return text(value[language], languages)
        return text(next(iter(value.values()), None), languages)
    return str(value)


def _canvas_items(manifest: JSON) -> list[JSON]:
    if presentation_version(manifest) == 3 or (_type(manifest) == "Manifest" and "items" in manifest):
        items = manifest.get("items")
    else:
        sequences = _list(manifest.get("sequences"))
        items = sequences[0].get("canvases") if sequences and isinstance(sequences[0], dict) else None
    if not isinstance(items, list) or not items:
        raise ManifestError("the manifest has no canvases")
    return [item for item in items if isinstance(item, dict)]


def _metadata_label(canvas: JSON, languages: Sequence[str]) -> str | None:
    for entry in _list(canvas.get("metadata")):
        if isinstance(entry, dict) and (text(entry.get("label"), languages) or "").casefold() in LABEL_KEYS:
            return text(entry.get("value"), languages)
    return None


def _painting_images(canvas: JSON) -> list[JSON]:
    resources: list[JSON] = []
    if "images" in canvas:
        for annotation in _list(canvas.get("images")):
            if isinstance(annotation, dict) and isinstance(annotation.get("resource"), dict):
                resources.append(annotation["resource"])
    else:
        for page in _list(canvas.get("items")):
            if not isinstance(page, dict):
                continue
            for annotation in _list(page.get("items")):
                if not isinstance(annotation, dict):
                    continue
                motivations = [str(item).split(":")[-1] for item in _list(annotation.get("motivation", "painting"))]
                if "painting" in motivations:
                    resources.extend(body for body in _list(annotation.get("body")) if isinstance(body, dict))
    images = []
    for resource in resources:
        if _type(resource) == "Choice":
            choices = [choice for choice in _list(resource.get("default")) + _list(resource.get("items")) if choice]
            resource = choices[0] if choices and isinstance(choices[0], dict) else {}
        is_image = _type(resource) in {"Image", ""} or str(resource.get("format", "")).startswith("image/")
        if resource and is_image:
            images.append(resource)
    return images


def _service(resource: JSON, base: str) -> tuple[str | None, JSON]:
    """The image service's URL and the service as the manifest describes it."""
    for service in _list(resource.get("service")):
        if not isinstance(service, dict):
            continue
        signature = json.dumps(service, sort_keys=True).casefold()
        if "auth" in signature and "imageservice" not in signature:
            continue
        if any(mark in signature for mark in ("imageservice", "iiif.io/api/image", "level0", "level1", "level2")):
            return urljoin(base, _id(service)).rstrip("/"), service
    return None, {}


def _size(value: Any) -> int | None:
    try:
        number = int(value)
    except (TypeError, ValueError):
        return None
    return number if number > 0 else None


def canvases(manifest: JSON, base: str = "", languages: Sequence[str] = ("en",)) -> list[Canvas]:
    """The manifest's canvases, in order, with their labels, image services and sizes.

    The size comes from the image service's description, then the painting image, then the canvas.
    """
    found = []
    for position, item in enumerate(_canvas_items(manifest), 1):
        images = _painting_images(item)
        resource = images[0] if len(images) == 1 else {}
        service, described = _service(resource, base) if resource else (None, {})
        image_url = urljoin(base, _id(resource)) if _id(resource) else None
        sized = next(
            (c for c in (described, resource, item) if _size(c.get("width")) and _size(c.get("height"))),
            {},
        )
        found.append(
            Canvas(
                position=position,
                id=urljoin(base, _id(item)) if _id(item) else None,
                label=text(item.get("label"), languages) or _metadata_label(item, languages),
                image_service=service,
                image_url=image_url,
                width=_size(sized.get("width")),
                height=_size(sized.get("height")),
            )
        )
    return found
