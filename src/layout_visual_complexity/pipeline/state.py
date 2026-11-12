"""A run's state: one SQLite file in the output folder, written by the main process only.

Each finished page is kept with two keys: the run's settings, and those with the page's own image. A rerun skips the
pages done under the same key and does the rest again.
"""

import json
import sqlite3
import time
import zlib
from collections.abc import Iterator
from pathlib import Path

from layout_visual_complexity.segmentation.records import PageRecord

SCHEMA = """
CREATE TABLE IF NOT EXISTS pages (
    page TEXT PRIMARY KEY, document TEXT, key TEXT, settings TEXT, status TEXT, source TEXT, record BLOB,
    finished REAL
);
CREATE INDEX IF NOT EXISTS pages_by_settings ON pages (settings, document);
"""


class State:
    """The pages a run's output folder holds: their keys, statuses, sources and records."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(self.path, timeout=30)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA synchronous=NORMAL")
        self.db.executescript(SCHEMA)

    def close(self) -> None:
        self.db.commit()
        self.db.close()

    def commit(self) -> None:
        self.db.commit()

    def done(self) -> dict[str, str]:
        """The pages done, with their keys."""
        return dict(self.db.execute("SELECT page, key FROM pages WHERE status = 'done'"))

    def save(self, record: PageRecord, key: str, settings: str, source: dict[str, object]) -> None:
        reasons = [f"{name}: {record.status[name].reason}" for name in record.failed]
        status = "done" if not reasons else "; ".join(reasons)
        blob = zlib.compress(json.dumps(record.to_dict(), separators=(",", ":")).encode(), 6)
        self.db.execute(
            "INSERT OR REPLACE INTO pages VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (record.page, record.document, key, settings, status, json.dumps(source), blob, time.time()),
        )

    def rows(self, settings: str, document: str | None = None) -> Iterator[tuple[PageRecord, str, dict[str, object]]]:
        """Records made under the run's settings, with their statuses and sources: all, or one document's."""
        query = "SELECT record, status, source FROM pages WHERE settings = ?"
        arguments: tuple[str, ...] = (settings,)
        if document is not None:
            query += " AND document = ?"
            arguments += (document,)
        for blob, status, source in self.db.execute(query + " ORDER BY document, page", arguments):
            yield PageRecord.from_dict(json.loads(zlib.decompress(blob))), status, json.loads(source)

    def documents(self, settings: str) -> list[str]:
        """The documents with pages done under the run's settings."""
        query = "SELECT DISTINCT document FROM pages WHERE settings = ? AND status = 'done'"
        return sorted(document for (document,) in self.db.execute(query, (settings,)))
