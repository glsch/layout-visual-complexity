"""Polite HTTP downloads, with retries and caches on disk: hosts are fetched from at once, each at a steady pace with
a few requests in flight. Page images come at a long edge of 2,000 px, never more than the image's own, into a cache
of bounded size."""

import hashlib
import io
import json
import math
import os
import random
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import httpx2

import layout_visual_complexity
from layout_visual_complexity.logs import logger

REPOSITORY = "https://github.com/glsch/layout-visual-complexity"
CONTACT_VARIABLE = "LVC_CONTACT"
DEFAULT_CACHE = Path("~/.cache/layout-visual-complexity")
MAX_WAIT = 600.0
MAX_BACKOFF = 60.0


class DownloadError(RuntimeError):
    """A download that failed for good. The message says why."""


@dataclass(frozen=True)
class HttpSettings:
    """How to fetch from servers.

    Args:
        contact: Who to contact about these requests, sent in the User-Agent: an address or a project's name. When
            unset, the `LVC_CONTACT` environment variable.
        interval: Seconds between the starts of two requests to the same host.
        connections: Requests in flight at once to the same host.
        retries: Attempts after the first, for network errors, server errors, 429 and 503.
        backoff: Seconds before the first retry, doubled at each one, up to a minute.
        timeout: Seconds to wait for a server.
    """

    contact: str | None = None
    interval: float = 0.5
    connections: int = 2
    retries: int = 5
    backoff: float = 1.0
    timeout: float = 60.0


def user_agent(contact: str | None = None) -> str:
    """`layout-visual-complexity/<version> (+<repository>; <contact>)`."""
    contact = (contact or os.environ.get(CONTACT_VARIABLE, "")).strip()
    suffix = f"; {contact}" if contact else ""
    return f"layout-visual-complexity/{layout_visual_complexity.__version__} (+{REPOSITORY}{suffix})"


def retry_after(response: httpx2.Response) -> float | None:
    """The wait a `Retry-After` header asks for, in seconds."""
    value = response.headers.get("retry-after", "").strip()
    if not value:
        return None
    if value.isdigit():
        return min(float(value), MAX_WAIT)
    try:
        return min(max(parsedate_to_datetime(value).timestamp() - time.time(), 0.0), MAX_WAIT)
    except (TypeError, ValueError):
        return None


class Cache:
    """Downloaded files on disk, named by the SHA-256 of their URL."""

    def __init__(self, folder: Path, suffix: str = "") -> None:
        self.folder = Path(folder).expanduser()
        self.suffix = suffix

    def path(self, url: str) -> Path:
        return self.folder / f"{hashlib.sha256(url.encode()).hexdigest()}{self.suffix}"

    def get(self, url: str) -> bytes | None:
        path = self.path(url)
        return path.read_bytes() if path.is_file() else None

    def put(self, url: str, data: bytes) -> None:
        path = self.path(url)
        path.parent.mkdir(parents=True, exist_ok=True)
        partial = path.with_name(path.name + ".part")
        partial.write_bytes(data)
        partial.replace(path)


@dataclass
class RequestRecord:
    """One request as it went, for timings in reports."""

    host: str
    url: str
    seconds: float
    status: int | None
    size: int


class Host:
    """One host's pace: at most `connections` requests in flight, their starts at least `interval` apart."""

    def __init__(self, name: str, connections: int, interval: float) -> None:
        self.name = name
        self.connections = max(1, connections)
        self.interval = interval
        self.active = 0
        self.next_start = 0.0
        self.condition = threading.Condition()

    @contextmanager
    def slot(self) -> Iterator[None]:
        """Wait for a free connection and the host's turn, then hold the connection."""
        with self.condition:
            self.condition.wait_for(lambda: self.active < self.connections)
            self.active += 1
            now = time.monotonic()
            start = max(now, self.next_start)
            self.next_start = start + self.interval
        try:
            if start > now:
                time.sleep(start - now)
            yield
        finally:
            with self.condition:
                self.active -= 1
                self.condition.notify_all()

    def slow_down(self, wait: float, floor: float) -> None:
        """After a 429 or 503: half the connections and twice the interval for the rest of the run."""
        with self.condition:
            self.connections = max(1, self.connections // 2)
            self.interval = min(max(self.interval * 2, floor), MAX_BACKOFF)
            self.next_start = max(self.next_start, time.monotonic() + wait)
        logger.warning(
            "%s asked us to slow down: now %d connection(s), %.1f s between requests",
            self.name,
            self.connections,
            self.interval,
        )


class HttpClient:
    """GET requests, safe to call from several threads: hosts are fetched from at once, each at its own pace. Use it
    as a context manager.

    A 429 or 503 halves the host's connections and doubles its interval for the rest of the run, and waits as the
    server asks.
    """

    def __init__(self, settings: HttpSettings | None = None, transport: httpx2.BaseTransport | None = None) -> None:
        self.settings = settings or HttpSettings()
        self.client = httpx2.Client(
            http2=True,
            timeout=httpx2.Timeout(self.settings.timeout),
            headers={"User-Agent": user_agent(self.settings.contact)},
            follow_redirects=True,
            transport=transport,
            limits=httpx2.Limits(max_connections=None, max_keepalive_connections=32),
        )
        self.hosts: dict[str, Host] = {}
        self.records: list[RequestRecord] = []
        self.lock = threading.Lock()

    def __enter__(self) -> "HttpClient":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def close(self) -> None:
        self.client.close()

    def host(self, name: str) -> Host:
        with self.lock:
            if name not in self.hosts:
                self.hosts[name] = Host(name, self.settings.connections, self.settings.interval)
            return self.hosts[name]

    def _record(self, record: RequestRecord) -> None:
        with self.lock:
            self.records.append(record)

    def get(self, url: str) -> bytes:
        """The body of a URL, retrying network errors, server errors, 429 and 503."""
        host = self.host(urlsplit(url).netloc)
        reason = "no attempt made"
        for attempt in range(self.settings.retries + 1):
            wait = min(self.settings.backoff * 2.0**attempt, MAX_BACKOFF) * random.uniform(0.75, 1.25)
            with host.slot():
                began = time.monotonic()
                try:
                    response = self.client.get(url)
                except httpx2.TransportError as error:
                    response = None
                    reason = f"network error: {type(error).__name__}"
                seconds = time.monotonic() - began
            if response is None:
                self._record(RequestRecord(host.name, url, seconds, None, 0))
                logger.info("%s: %s, attempt %d", url, reason, attempt + 1)
                time.sleep(wait)
                continue
            status = response.status_code
            self._record(RequestRecord(host.name, url, seconds, status, len(response.content)))
            if status in (429, 503):
                host.slow_down(retry_after(response) or wait, self.settings.backoff)
                reason = f"refused: {status}"
                continue
            if status >= 500:
                reason = f"server error: {status}"
                time.sleep(wait)
                continue
            if status >= 400:
                raise DownloadError(f"{url}: refused: {status}")
            return response.content
        raise DownloadError(f"{url}: {reason}")


def cached_get(client: HttpClient, cache: Cache, url: str) -> bytes:
    """A URL's body from the cache, or downloaded into it."""
    data = cache.get(url)
    if data is None:
        data = client.get(url)
        cache.put(url, data)
    return data


def cached_json(client: HttpClient, cache: Cache, url: str) -> dict[str, Any]:
    """A JSON object from the cache, or downloaded into it. Invalid JSON isn't kept."""
    data = cached_get(client, cache, url)
    try:
        value = json.loads(data)
    except ValueError:
        cache.path(url).unlink(missing_ok=True)
        raise DownloadError(f"{url}: not JSON") from None
    if not isinstance(value, dict):
        cache.path(url).unlink(missing_ok=True)
        raise DownloadError(f"{url}: not a JSON object")
    return value


# Page images.

LONG_EDGE = 2000


@dataclass(frozen=True)
class ImageSettings:
    """How page images are fetched and kept.

    Args:
        long_edge: The long edge asked for, in pixels; an image smaller than that comes at its own size.
        cache_gb: Gigabytes the image cache keeps at most; the oldest images leave first.
        quality: JPEG quality of the images made smaller here, from image URLs or local files larger than the long
            edge.
    """

    long_edge: int = LONG_EDGE
    cache_gb: float = 2.0
    quality: int = 90


@dataclass(frozen=True)
class PageImage:
    """A page's image on disk, as the stages work on it: its size, the full image's, and what was asked for."""

    path: Path
    width: int
    height: int
    full_width: int
    full_height: int
    request: str
    cached: bool


def size_request(width: int, height: int, long_edge: int, image_api: int = 2) -> str:
    """The size of a IIIF Image API request: the long edge at `long_edge`, or the image's own size if it's smaller."""
    if max(width, height) <= long_edge:
        return "max" if image_api >= 3 else "full"
    return f"{long_edge}," if width >= height else f",{long_edge}"


def image_request(service: str, width: int, height: int, long_edge: int, image_api: int = 2) -> str:
    """The URL of a page image from its IIIF image service, at the long edge asked for."""
    return f"{service.rstrip('/')}/full/{size_request(width, height, long_edge, image_api)}/0/default.jpg"


def limited_size(width: int, height: int, info: dict[str, object]) -> tuple[int, int]:
    """The largest size a service gives, from its `maxWidth`, `maxHeight` and `maxArea`, keeping the image's shape."""
    scale = 1.0
    max_width, max_height, max_area = (info.get(key) for key in ("maxWidth", "maxHeight", "maxArea"))
    if isinstance(max_width, int) and max_width < width:
        scale = min(scale, max_width / width)
    if isinstance(max_height, int) and max_height < height:
        scale = min(scale, max_height / height)
    if isinstance(max_area, int) and max_area < width * height:
        scale = min(scale, math.sqrt(max_area / (width * height)))
    return max(1, math.floor(width * scale)), max(1, math.floor(height * scale))


class ImageCache:
    """Page images on disk, named by the SHA-256 of what was asked, at most `limit` bytes, the oldest leaving first.

    Images in use are pinned and never leave; `release` unpins them. Safe to use from several threads.
    """

    def __init__(self, folder: Path, limit: int) -> None:
        self.folder = Path(folder).expanduser()
        self.folder.mkdir(parents=True, exist_ok=True)
        self.limit = limit
        self.lock = threading.Lock()
        self.pins: dict[Path, int] = {}
        self.entries: dict[Path, tuple[float, int]] = {}
        for entry in os.scandir(self.folder):
            if entry.is_file() and entry.name.endswith(".jpg"):
                stat = entry.stat()
                self.entries[Path(entry.path)] = (stat.st_mtime, stat.st_size)

    @property
    def size(self) -> int:
        with self.lock:
            return sum(size for _, size in self.entries.values())

    def path(self, key: str) -> Path:
        return self.folder / f"{hashlib.sha256(key.encode()).hexdigest()}.jpg"

    def get(self, key: str) -> Path | None:
        """The image for a key, pinned, if it's there."""
        path = self.path(key)
        with self.lock:
            if path not in self.entries or not path.is_file():
                self.entries.pop(path, None)
                return None
            self.pins[path] = self.pins.get(path, 0) + 1
        return path

    def put(self, key: str, data: bytes) -> Path:
        """Keep an image, pinned, and make room for it."""
        path = self.path(key)
        partial = path.with_name(f"{path.name}.{threading.get_ident()}.part")
        partial.write_bytes(data)
        partial.replace(path)
        with self.lock:
            self.entries[path] = (time.time(), len(data))
            self.pins[path] = self.pins.get(path, 0) + 1
            self._evict()
        return path

    def release(self, path: Path) -> None:
        """Unpin an image: it may leave when the cache needs room."""
        with self.lock:
            if path in self.pins:
                self.pins[path] -= 1
                if self.pins[path] <= 0:
                    del self.pins[path]
            self._evict()

    def _evict(self) -> None:
        total = sum(size for _, size in self.entries.values())
        if total <= self.limit:
            return
        for path in sorted(self.entries, key=lambda item: self.entries[item][0]):
            if total <= self.limit:
                break
            if path in self.pins:
                continue
            total -= self.entries.pop(path)[1]
            path.unlink(missing_ok=True)


def _decoded_size(data: bytes) -> tuple[int, int]:
    """Decode an image fully, so a truncated one is refused, and give its size."""
    from PIL import Image

    try:
        with Image.open(io.BytesIO(data)) as image:
            image.load()
            return image.size
    except Exception as error:
        raise DownloadError(f"not a readable image: {type(error).__name__}: {error}") from None


def _smaller(data: bytes | Path, long_edge: int, quality: int) -> tuple[bytes, tuple[int, int], tuple[int, int]]:
    """An image made to fit the long edge, as JPEG, with its new size and its own."""
    from PIL import Image

    source = io.BytesIO(data) if isinstance(data, bytes) else data
    with Image.open(source) as image:
        image.load()
        own = image.size
        scale = long_edge / max(own)
        size = (max(1, round(own[0] * scale)), max(1, round(own[1] * scale)))
        smaller = image.convert("RGB").resize(size, Image.Resampling.LANCZOS, reducing_gap=3.0)
    buffer = io.BytesIO()
    smaller.save(buffer, "JPEG", quality=quality)
    return buffer.getvalue(), size, own


def _image_size(path: Path) -> tuple[int, int]:
    from PIL import Image

    with Image.open(path) as image:
        return image.size


class ImageFetcher:
    """Gets page images ready: from IIIF image services at the long edge asked for, from image URLs and local files,
    made smaller here when they're larger than the long edge. Safe to use from several threads."""

    def __init__(self, client: HttpClient, cache: Path, settings: ImageSettings | None = None) -> None:
        self.client = client
        self.settings = settings or ImageSettings()
        self.images = ImageCache(Path(cache) / "images", int(self.settings.cache_gb * 1e9))
        self.info = Cache(Path(cache) / "info", ".json")

    def service_size(self, service: str) -> tuple[int, int, dict[str, object]]:
        info = cached_json(self.client, self.info, f"{service.rstrip('/')}/info.json")
        width, height = info.get("width"), info.get("height")
        if not (isinstance(width, int) and isinstance(height, int)):
            raise DownloadError(f"{service}: info.json gives no size")
        return width, height, info

    def from_service(self, service: str, width: int | None, height: int | None, image_api: int) -> PageImage:
        info: dict[str, object] = {}
        if not (width and height):
            width, height, info = self.service_size(service)
        asked = limited_size(width, height, info) if info else (width, height)
        url = image_request(service, *asked, self.settings.long_edge, image_api)
        path = self.images.get(url)
        if path is None:
            data = self.client.get(url)
            _decoded_size(data)
            path = self.images.put(url, data)
        fetched = _image_size(path)
        if url.split("/")[-3] in ("full", "max") and asked == (width, height):
            width, height = fetched  # the full image itself, whatever the manifest said
        elif abs(fetched[0] / fetched[1] - width / height) > 0.02 * width / height and not info:
            # The manifest's size doesn't fit the image: the service's own description decides.
            width, height, _ = self.service_size(service)
        if max(fetched) > self.settings.long_edge:
            self.images.release(path)
            data, fetched, _ = _smaller(path, self.settings.long_edge, self.settings.quality)
            path = self.images.put(f"{url}#long_edge={self.settings.long_edge}", data)
        return PageImage(path, *fetched, width, height, url, True)

    def from_url(self, url: str) -> PageImage:
        """An image URL fetched as it is; the full image's size is kept beside it, in the description cache."""
        key = f"{url}#long_edge={self.settings.long_edge}"
        path = self.images.get(key)
        own = self.info.get(key)
        if path is not None and own is not None:
            width, height = json.loads(own)
            return PageImage(path, *_image_size(path), width, height, url, True)
        if path is not None:
            self.images.release(path)
        data = self.client.get(url)
        width, height = _decoded_size(data)
        if max(width, height) > self.settings.long_edge:
            data, _, _ = _smaller(data, self.settings.long_edge, self.settings.quality)
        path = self.images.put(key, data)
        self.info.put(key, json.dumps([width, height]).encode())
        return PageImage(path, *_image_size(path), width, height, url, True)

    def from_file(self, file: Path) -> PageImage:
        file = Path(file)
        if not file.is_file():
            raise DownloadError(f"{file}: no such image")
        own = _image_size(file)
        if max(own) <= self.settings.long_edge:
            return PageImage(file, *own, *own, str(file), False)
        stat = file.stat()
        key = f"{file.resolve()}#{stat.st_size}#{stat.st_mtime_ns}#long_edge={self.settings.long_edge}"
        path = self.images.get(key)
        if path is None:
            data, _, _ = _smaller(file, self.settings.long_edge, self.settings.quality)
            path = self.images.put(key, data)
        return PageImage(path, *_image_size(path), *own, str(file), True)

    def release(self, image: PageImage) -> None:
        """The page is done with its image: the cache may let it go."""
        if image.cached:
            self.images.release(image.path)
