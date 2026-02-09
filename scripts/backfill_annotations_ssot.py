#!/usr/bin/env python3
"""
Backfill annotation JSON files into DB-first annotation tables.

Usage:
  python scripts/backfill_annotations_ssot.py
  python scripts/backfill_annotations_ssot.py --user douff
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from configs.settings import settings
from src.db.database import SessionLocal, init_db
from src.utils.annotation_store import upsert_annotation


def main() -> int:
    parser = argparse.ArgumentParser(description="Backfill annotation JSON into DB SSOT tables.")
    parser.add_argument("--user", default=settings.DEFAULT_USER, help="annotation user name")
    parser.add_argument("--root", default=settings.ANNOTATIONS_ROOT, help="annotations root directory")
    args = parser.parse_args()

    ann_dir = Path(args.root) / args.user
    if not ann_dir.exists():
        print(f"skip: annotation dir not found: {ann_dir}")
        return 0

    init_db()
    db = SessionLocal()
    inserted = 0
    skipped = 0
    try:
        for path in sorted(ann_dir.glob("*.json")):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except Exception as e:
                print(f"skip: {path.name} ({e})")
                skipped += 1
                continue

            filename = payload.get("filename") or path.stem
            upsert_annotation(db, args.user, str(filename), payload)
            inserted += 1
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()

    print(f"done: upserted={inserted}, skipped={skipped}, dir={ann_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
