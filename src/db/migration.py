"""
Lightweight DB migration runner.

Why this exists:
- `Base.metadata.create_all()` can create missing tables, but cannot evolve existing schema safely.
- This runner applies ordered SQL migrations and records applied versions.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Tuple

from sqlalchemy import text

from src.db.database import engine


MIGRATIONS_DIR = Path(__file__).resolve().parent / "migrations"


@dataclass(frozen=True)
class MigrationFile:
    version: str
    filename: str
    path: Path


def _ensure_migrations_dir() -> None:
    MIGRATIONS_DIR.mkdir(parents=True, exist_ok=True)


def _ensure_schema_migrations_table() -> None:
    ddl = """
    CREATE TABLE IF NOT EXISTS schema_migrations (
        version TEXT PRIMARY KEY,
        filename TEXT NOT NULL,
        checksum TEXT NOT NULL,
        applied_at TEXT NOT NULL
    )
    """
    with engine.begin() as conn:
        conn.execute(text(ddl))


def _discover_migration_files() -> List[MigrationFile]:
    _ensure_migrations_dir()
    files = []
    for path in sorted(MIGRATIONS_DIR.glob("*.sql")):
        stem = path.stem
        # Convention: 0001_xxx.sql
        if "_" not in stem:
            continue
        version = stem.split("_", 1)[0]
        files.append(MigrationFile(version=version, filename=path.name, path=path))
    return files


def _read_applied() -> dict:
    _ensure_schema_migrations_table()
    with engine.begin() as conn:
        rows = conn.execute(text("SELECT version, checksum FROM schema_migrations")).fetchall()
    return {str(r[0]): str(r[1]) for r in rows}


def _checksum(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def _split_sql_statements(sql_text: str) -> List[str]:
    # We keep this simple because current migrations are straightforward DDL.
    parts = [p.strip() for p in sql_text.split(";")]
    return [p for p in parts if p]


def _apply_one(migration: MigrationFile) -> None:
    sql_text = migration.path.read_text(encoding="utf-8")
    checksum = _checksum(sql_text)
    statements = _split_sql_statements(sql_text)

    with engine.begin() as conn:
        for stmt in statements:
            conn.exec_driver_sql(stmt)
        conn.execute(
            text(
                """
                INSERT INTO schema_migrations(version, filename, checksum, applied_at)
                VALUES (:version, :filename, :checksum, :applied_at)
                """
            ),
            {
                "version": migration.version,
                "filename": migration.filename,
                "checksum": checksum,
                "applied_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
            },
        )


def migration_status() -> Tuple[List[MigrationFile], List[MigrationFile]]:
    files = _discover_migration_files()
    applied = _read_applied()
    applied_files: List[MigrationFile] = []
    pending_files: List[MigrationFile] = []
    for mf in files:
        if mf.version in applied:
            applied_files.append(mf)
        else:
            pending_files.append(mf)
    return applied_files, pending_files


def apply_pending_migrations() -> List[str]:
    """
    Apply pending SQL migrations in version order.
    Returns list of applied migration filenames.
    """
    files = _discover_migration_files()
    applied = _read_applied()

    applied_now: List[str] = []
    for mf in files:
        if mf.version in applied:
            # If file changed after being applied, fail fast.
            current_checksum = _checksum(mf.path.read_text(encoding="utf-8"))
            if current_checksum != applied[mf.version]:
                raise RuntimeError(
                    f"Migration checksum mismatch for version {mf.version}: {mf.filename}"
                )
            continue

        _apply_one(mf)
        applied_now.append(mf.filename)

    return applied_now
