"""Reading the annotation app's export: its sessions, and the pairs shown in them.

A session holds combinations, one per pair shown. A rated one has `entropyRated: true`, the chosen image under
`winnerHigherEntropy` and the other under `loserLowerEntropy` (each with `url` and `position`), `thinkingTime` in
milliseconds and `isCalibration`. A skipped one has `entropyRated: false`. The field names come from the app.
"""

import json
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlsplit

import pandas as pd

RATED, SKIPPED, CALIBRATION = "rated", "skipped", "calibration"
KINDS = (RATED, SKIPPED, CALIBRATION)
COLUMNS = [
    "session_id",
    "user",
    "source",
    "started",
    "completed",
    "position",
    "kind",
    "left",
    "right",
    "chosen",
    "response_ms",
]


class ExportError(ValueError):
    """An export this code can't read."""


def image_id(url: str) -> str:
    """The image id: the URL's file name without its extension."""
    name = unquote(urlsplit(url).path.rsplit("/", 1)[-1])
    stem, dot, _ = name.rpartition(".")
    return stem if dot else name


def _rated(combination: dict[str, Any], where: str) -> tuple[str, str, str, int | None]:
    """Left image, right image, chosen image and response time of a rated combination."""
    try:
        chosen, other = combination["winnerHigherEntropy"], combination["loserLowerEntropy"]
        sides = {chosen["position"]: image_id(chosen["url"]), other["position"]: image_id(other["url"])}
    except (KeyError, TypeError) as error:
        raise ExportError(f"{where}: a rated combination without {error}") from None
    if set(sides) != {"left", "right"}:
        raise ExportError(f"{where}: positions {sorted(sides)}, expected left and right")
    time = combination.get("thinkingTime")
    return sides["left"], sides["right"], image_id(chosen["url"]), int(time) if time is not None else None


def read_export(path: Path) -> pd.DataFrame:
    """One row per combination, in the export's order, with its session's fields.

    Columns: `session_id`, `user`, `source`, `started` (UTC), `completed`, `position` (in the session, from 0),
    `kind` (rated, skipped or calibration), `left`, `right`, `chosen`, `response_ms`. Skipped combinations have no
    images.
    """
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    sessions = data.get("sessions") if isinstance(data, dict) else None
    if not isinstance(sessions, list):
        raise ExportError(f"{path}: expected an object with a list of sessions")
    rows = []
    seen: set[str] = set()
    for number, session in enumerate(sessions):
        try:
            session_id = str(session["sessionID"])
            fields = (session_id, session["user"], session.get("source"), session["timestamp"])
        except (KeyError, TypeError) as error:
            raise ExportError(f"{path}: session {number} has no {error}") from None
        if session_id in seen:
            raise ExportError(f"{path}: session {session_id} appears twice")
        seen.add(session_id)
        for position, combination in enumerate(session.get("combinations") or []):
            where = f"session {session_id}, combination {position}"
            if combination.get("entropyRated"):
                kind = CALIBRATION if combination.get("isCalibration") else RATED
                left, right, chosen, response = _rated(combination, where)
            else:
                kind, left, right, chosen, response = SKIPPED, None, None, None, None
            rows.append((*fields, bool(session.get("completed")), position, kind, left, right, chosen, response))
    frame = pd.DataFrame(rows, columns=["session_id", "user", "source", "timestamp", *COLUMNS[4:]])
    frame.insert(3, "started", pd.to_datetime(frame.pop("timestamp"), utc=True, format="ISO8601"))
    frame["response_ms"] = frame["response_ms"].astype("Int64")
    return frame[COLUMNS]
