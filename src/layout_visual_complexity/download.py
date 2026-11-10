"""Polite HTTP downloads: one request at a time per host, at a steady pace, with retries and a cache on disk."""

import hashlib
import json
import os
import random
import time
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
        retries: Attempts after the first, for network errors, server errors, 429 and 503.
        backoff: Seconds before the first retry, doubled at each one, up to a minute.
        timeout: Seconds to wait for a server.
    """

    contact: str | None = None
    interval: float = 0.5
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


class HttpClient:
    """GET requests, one at a time, each host's at least `interval` apart. Use it as a context manager.

    A 429 or 503 doubles the host's interval for the rest of the run and waits as the server asks.
    """

    def __init__(self, settings: HttpSettings | None = None, transport: httpx2.BaseTransport | None = None) -> None:
        self.settings = settings or HttpSettings()
        self.client = httpx2.Client(
            http2=True,
            timeout=httpx2.Timeout(self.settings.timeout),
            headers={"User-Agent": user_agent(self.settings.contact)},
            follow_redirects=True,
            transport=transport,
        )
        self.intervals: dict[str, float] = {}
        self.next_start: dict[str, float] = {}
        self.records: list[RequestRecord] = []

    def __enter__(self) -> "HttpClient":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def close(self) -> None:
        self.client.close()

    def _wait_turn(self, host: str) -> None:
        now = time.monotonic()
        start = max(now, self.next_start.get(host, 0.0))
        if start > now:
            time.sleep(start - now)
        self.next_start[host] = start + self.intervals.setdefault(host, self.settings.interval)

    def _slow_down(self, host: str, wait: float) -> None:
        self.intervals[host] = min(max(self.intervals.get(host, 0.0) * 2, self.settings.backoff), MAX_BACKOFF)
        self.next_start[host] = max(self.next_start.get(host, 0.0), time.monotonic() + wait)
        logger.warning("%s asked us to slow down: now %.1f s between requests", host, self.intervals[host])

    def get(self, url: str) -> bytes:
        """The body of a URL, retrying network errors, server errors, 429 and 503."""
        host = urlsplit(url).netloc
        reason = "no attempt made"
        for attempt in range(self.settings.retries + 1):
            wait = min(self.settings.backoff * 2.0**attempt, MAX_BACKOFF) * random.uniform(0.75, 1.25)
            self._wait_turn(host)
            began = time.monotonic()
            try:
                response = self.client.get(url)
            except httpx2.TransportError as error:
                reason = f"network error: {type(error).__name__}"
                self.records.append(RequestRecord(host, url, time.monotonic() - began, None, 0))
                logger.info("%s: %s, attempt %d", url, reason, attempt + 1)
                time.sleep(wait)
                continue
            status = response.status_code
            self.records.append(RequestRecord(host, url, time.monotonic() - began, status, len(response.content)))
            if status in (429, 503):
                asked = retry_after(response)
                self._slow_down(host, asked or wait)
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
