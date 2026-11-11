"""A dataset's files from its Zenodo record, by DOI, each checked against the MD5 the record gives."""

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path

from layout_visual_complexity.download import DownloadError, HttpClient
from layout_visual_complexity.logs import logger

# Zenodo's DOIs: 10.5281/zenodo.<record> on zenodo.org, 10.5072/zenodo.<record> on its sandbox.
DOI = re.compile(r"^(?:https?://(?:dx\.)?doi\.org/|doi:)?10\.(5281|5072)/zenodo\.(\d+)$", re.IGNORECASE)
HOSTS = {"5281": "zenodo.org", "5072": "sandbox.zenodo.org"}


@dataclass(frozen=True)
class RecordFile:
    """A file of a record: its name, where to get it, and its MD5."""

    name: str
    url: str
    md5: str


def record_of(doi: str) -> tuple[str, str] | None:
    """The host and record id a Zenodo DOI names, or None for anything else."""
    match = DOI.match(doi.strip())
    return (HOSTS[match.group(1)], match.group(2)) if match else None


def record_files(client: HttpClient, host: str, record: str) -> list[RecordFile]:
    """The files a record lists, from Zenodo's API."""
    data = json.loads(client.get(f"https://{host}/api/records/{record}"))
    entries = data.get("files")
    if isinstance(entries, dict):
        entries = entries.get("entries") or []
        entries = list(entries.values()) if isinstance(entries, dict) else entries
    files = []
    for entry in entries or []:
        name = entry.get("key") or entry.get("filename")
        links = entry.get("links") or {}
        url = links.get("content") or links.get("self") or links.get("download")
        md5 = str(entry.get("checksum", "")).removeprefix("md5:")
        if not (name and url and md5):
            raise DownloadError(f"record {record}: a file without a name, link or checksum")
        files.append(RecordFile(name, url, md5))
    if not files:
        raise DownloadError(f"record {record} lists no files")
    return files


def md5_of(path: Path) -> str:
    digest = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fetch_record(doi: str, client: HttpClient, cache: Path) -> Path:
    """Fetch a record's files into `cache/<host>/<record>/`, keeping those already there whose MD5 matches.

    Raises `DownloadError` when a file's MD5 differs from the record's."""
    location = record_of(doi)
    if location is None:
        raise ValueError(f"not a Zenodo DOI: {doi}")
    host, record = location
    folder = Path(cache).expanduser() / host / record
    folder.mkdir(parents=True, exist_ok=True)
    for item in record_files(client, host, record):
        path = folder / item.name
        if path.is_file() and md5_of(path) == item.md5:
            continue
        data = client.get(item.url)
        if hashlib.md5(data, usedforsecurity=False).hexdigest() != item.md5:
            raise DownloadError(f"{item.name}: its MD5 differs from the record's")
        partial = path.with_name(path.name + ".part")
        partial.write_bytes(data)
        partial.replace(path)
        logger.info("fetched %s from record %s", item.name, record)
    return folder
