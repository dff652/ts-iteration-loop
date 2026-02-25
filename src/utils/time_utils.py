"""
UTC time helpers.

Use these helpers instead of datetime.utcnow() to avoid deprecated APIs while
keeping existing naive-UTC storage semantics for DB DateTime columns.
"""

from __future__ import annotations

from datetime import UTC, datetime


def utc_now() -> datetime:
    """Return current UTC datetime (timezone-aware)."""
    return datetime.now(UTC)


def utc_now_naive() -> datetime:
    """Return current UTC datetime as naive datetime (legacy DB compatibility)."""
    return utc_now().replace(tzinfo=None)


def utc_iso_z(timespec: str = "seconds") -> str:
    """Return current UTC timestamp in ISO-8601 with `Z` suffix."""
    return utc_now().isoformat(timespec=timespec).replace("+00:00", "Z")
