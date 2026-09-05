"""Load the model catalog and derive one observation per release day.

The catalog preserves individual models, including models released on the same
day. Forecasting works with distinct release days so that multiple models in a
single release do not create artificial zero-day intervals.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from urllib.parse import urlsplit

import pandas as pd


DEFAULT_DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "releases.json"
_DATE_PATTERN = re.compile(r"[0-9]{4}-[0-9]{2}-[0-9]{2}\Z")
_ID_PATTERN = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*\Z")
_RELEASE_FIELDS = {"id", "name", "date", "source_url", "notes"}


@dataclass(frozen=True)
class Release:
    """One recorded model release, with an ID independent of list position."""

    id: str
    name: str
    date: date
    source_url: str | None
    notes: str


def parse_date(value: str) -> date:
    """Parse an exact ISO calendar date; reject times and abbreviated dates."""
    if not isinstance(value, str) or not _DATE_PATTERN.fullmatch(value):
        raise ValueError("date must be a string in YYYY-MM-DD format")
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(f"invalid calendar date: {value!r}") from exc


def _validate_source_url(value: object) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value or any(c.isspace() for c in value):
        raise ValueError("source_url must be null or an HTTP(S) URL")
    try:
        parsed = urlsplit(value)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            raise ValueError
        # Accessing port also rejects malformed and out-of-range port numbers.
        _ = parsed.port
    except ValueError as exc:
        raise ValueError("source_url must be null or a valid HTTP(S) URL") from exc
    return value


def load_releases(path: str | Path = DEFAULT_DATA_PATH) -> list[Release]:
    """Validate a schema-v1 catalog and return records sorted by date and ID.

    ``source_url: null`` describes a record without a supplied source; loading
    a catalog does not verify that its names or release dates are factual.
    Record errors identify the zero-based JSON array index, e.g. releases[2].
    """
    path = Path(path)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read release catalog {path}: {exc}") from exc

    if not isinstance(payload, dict):
        raise ValueError("release catalog must be a JSON object")
    unknown = payload.keys() - {"schema_version", "notes", "releases"}
    if unknown:
        raise ValueError(f"unknown catalog fields: {', '.join(sorted(unknown))}")
    if type(payload.get("schema_version")) is not int or payload["schema_version"] != 1:
        raise ValueError("unsupported schema_version; expected integer 1")
    if "notes" in payload and not isinstance(payload["notes"], str):
        raise ValueError("catalog notes must be a string")
    if not isinstance(payload.get("releases"), list):
        raise ValueError("catalog releases must be an array")

    records: list[Release] = []
    ids: dict[str, int] = {}
    identities: dict[tuple[str, date], int] = {}
    for index, item in enumerate(payload["releases"]):
        try:
            if not isinstance(item, dict):
                raise ValueError("must be an object")
            missing = _RELEASE_FIELDS - item.keys()
            extra = item.keys() - _RELEASE_FIELDS
            if missing:
                raise ValueError(f"missing fields: {', '.join(sorted(missing))}")
            if extra:
                raise ValueError(f"unknown fields: {', '.join(sorted(extra))}")
            record_id = item["id"]
            if not isinstance(record_id, str) or not _ID_PATTERN.fullmatch(record_id):
                raise ValueError("id must be a nonempty lowercase alphanumeric slug")
            name = item["name"]
            if not isinstance(name, str) or not name.strip() or name != name.strip():
                raise ValueError("name must be a nonempty string without surrounding whitespace")
            released_on = parse_date(item["date"])
            source_url = _validate_source_url(item["source_url"])
            if not isinstance(item["notes"], str):
                raise ValueError("notes must be a string")
            if record_id in ids:
                raise ValueError(f"duplicate id {record_id!r}; first used at releases[{ids[record_id]}]")
            identity = (name, released_on)
            if identity in identities:
                raise ValueError(
                    f"duplicate name/date pair; first used at releases[{identities[identity]}]"
                )
            ids[record_id] = index
            identities[identity] = index
            records.append(Release(record_id, name, released_on, source_url, item["notes"]))
        except ValueError as exc:
            raise ValueError(f"releases[{index}]: {exc}") from exc
    return sorted(records, key=lambda record: (record.date, record.id))


def releases_to_frame(releases: list[Release], as_of: date) -> pd.DataFrame:
    """Build chronological event features from releases on/before ``as_of``.

    Same-day names are joined with `` / `` in stable ID order. The first
    interval is missing because there is no preceding observed release. At
    least two distinct observed release days are required for forecasting.
    """
    if type(as_of) is not date:
        raise ValueError("as_of must be a datetime.date (use parse_date for YYYY-MM-DD strings)")
    by_day: dict[date, list[str]] = {}
    for release in sorted(releases, key=lambda record: (record.date, record.id)):
        if release.date <= as_of:
            by_day.setdefault(release.date, []).append(release.name)
    if len(by_day) < 2:
        raise ValueError(
            f"At least two distinct release dates on or before {as_of.isoformat()} are required; "
            f"found {len(by_day)}"
        )
    frame = pd.DataFrame({
        "version": [" / ".join(names) for names in by_day.values()],
        # Second precision handles all valid Python calendar dates, including
        # dates outside pandas' narrower nanosecond timestamp range.
        "date": pd.array(list(by_day), dtype="datetime64[s]"),
    })
    frame["days_since_start"] = (frame["date"] - frame["date"].iloc[0]).dt.days
    frame["month"] = frame["date"].dt.month
    frame["interval_days"] = frame["days_since_start"].diff()
    return frame
