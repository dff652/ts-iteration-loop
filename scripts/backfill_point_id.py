#!/usr/bin/env python3
"""
Backfill point_id columns in DB (scheme-1: normalized point name).
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.db.database import SessionLocal, init_db
from src.utils.point_id_backfill import backfill_point_ids


def main() -> int:
    init_db()
    with SessionLocal() as db:
        stats = backfill_point_ids(db)

    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    report_dir = Path("artifacts") / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / f"backfill_point_id_{ts}.json"
    report_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps({"success": True, "report": str(report_path), "stats": stats}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
