#!/usr/bin/env python3
"""
Run or inspect DB migrations.

Usage:
  python scripts/db_migrate.py --status
  python scripts/db_migrate.py --apply
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.db.database import init_db
from src.db.migration import apply_pending_migrations, migration_status


def main() -> int:
    parser = argparse.ArgumentParser(description="DB migration tool")
    parser.add_argument("--status", action="store_true", help="show migration status")
    parser.add_argument("--apply", action="store_true", help="apply pending migrations")
    args = parser.parse_args()

    if not args.status and not args.apply:
        args.status = True

    # Ensure tables exist for current model set before schema patch migrations.
    init_db()

    if args.status:
        applied, pending = migration_status()
        print(f"Applied: {len(applied)}")
        for mf in applied:
            print(f"  - {mf.filename}")
        print(f"Pending: {len(pending)}")
        for mf in pending:
            print(f"  - {mf.filename}")

    if args.apply:
        applied_now = apply_pending_migrations()
        if applied_now:
            print("Applied migrations:")
            for name in applied_now:
                print(f"  - {name}")
        else:
            print("No pending migrations.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
