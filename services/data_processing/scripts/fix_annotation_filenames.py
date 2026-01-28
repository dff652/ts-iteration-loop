#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fix annotation JSON `filename` fields by matching CSV filenames.
Default: strict exact match. If no exact match, use fuzzy matching by point id.
"""

import argparse
import json
import os
import re
import shutil
import sys
from pathlib import Path
from typing import List, Optional

# Ensure project root is on sys.path for settings import
PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from configs.settings import settings


def _parse_csv_dirs(csv_dir_arg: str) -> List[Path]:
    if not csv_dir_arg:
        return []
    parts = [p.strip() for p in csv_dir_arg.split(",") if p.strip()]
    return [Path(p) for p in parts]


def _extract_point_from_filename(csv_name: str) -> str:
    name = csv_name
    if name.endswith(".csv"):
        name = name[:-4]
    name = re.sub(r"_\d{8}_\d{6}_to_\d{8}_\d{6}.*$", "", name)
    name = re.sub(r"_\d{8}.*$", "", name)
    name = re.sub(r"_trend(_resid)?$", "", name)

    parts = [p for p in name.split("_") if p]
    idx = 0
    if parts and parts[0] in {"chatts", "qwen", "global", "global_chatts", "globalchatts"}:
        idx += 1
    if idx < len(parts) and parts[idx] in {"m4", "minmax", "none"}:
        idx += 1
    if idx < len(parts) and re.fullmatch(r"\d+(\.\d+)?", parts[idx]):
        idx += 1
    if idx < len(parts) and re.fullmatch(r"\d+", parts[idx]):
        idx += 1
    point = "_".join(parts[idx:]) if idx < len(parts) else name
    return point


def _build_csv_index(csv_dirs: List[Path]) -> dict:
    index = {}
    for root in csv_dirs:
        if not root.exists():
            continue
        for p in root.rglob("*.csv"):
            if p.is_file() and p.name not in index:
                index[p.name] = p
    return index


def _find_fuzzy_candidates(point: str, csv_dirs: List[Path]) -> List[Path]:
    matches: List[Path] = []
    if not point:
        return matches
    for root in csv_dirs:
        if not root.exists():
            continue
        for p in root.glob("*.csv"):
            if point in p.name:
                matches.append(p)
    return matches


def _choose_best_candidate(original_name: str, candidates: List[Path]) -> Optional[Path]:
    if not candidates:
        return None

    prefixes = ["_chatts_", "_qwen_", "global_chatts_", "chatts_", "qwen_", "timer_"]
    prefix = next((p for p in prefixes if original_name.startswith(p)), "")

    def score(p: Path):
        name = p.name
        pref = 1 if (prefix and prefix in name) else 0
        try:
            mtime = p.stat().st_mtime
        except OSError:
            mtime = 0
        return (pref, mtime, -len(name))

    return sorted(candidates, key=score, reverse=True)[0]


def main():
    parser = argparse.ArgumentParser(description="Fix annotation JSON filename fields.")
    parser.add_argument("--ann_dir", type=str, default=None)
    parser.add_argument("--csv_dir", type=str, default=None, help="CSV root directories (comma-separated)")
    parser.add_argument("--dry_run", type=str, default="false")
    parser.add_argument("--backup", type=str, default="true")
    args = parser.parse_args()

    ann_dir = Path(args.ann_dir or (Path(settings.ANNOTATIONS_ROOT) / settings.DEFAULT_USER))
    csv_dirs = _parse_csv_dirs(args.csv_dir or "")
    if not csv_dirs:
        csv_dirs = [Path(settings.DATA_DOWNSAMPLED_DIR), Path(settings.DATA_INFERENCE_DIR) / "chatts"]

    dry_run = args.dry_run.lower() == "true"
    backup = args.backup.lower() == "true"

    if not ann_dir.exists():
        print(f"[ERR] ann_dir not found: {ann_dir}")
        return

    csv_index = _build_csv_index(csv_dirs)
    backup_dir = ann_dir / "_backup" if backup else None
    if backup_dir:
        backup_dir.mkdir(parents=True, exist_ok=True)

    changed = 0
    unchanged = 0
    missing = 0

    for ann_file in sorted(ann_dir.glob("*.json")):
        try:
            data = json.loads(ann_file.read_text())
        except Exception:
            print(f"[SKIP] invalid json: {ann_file.name}")
            continue

        original = data.get("filename")
        if not original:
            print(f"[SKIP] no filename: {ann_file.name}")
            continue

        # Exact match
        if original in csv_index:
            unchanged += 1
            continue

        point = _extract_point_from_filename(original)
        candidates = _find_fuzzy_candidates(point, csv_dirs)
        best = _choose_best_candidate(original, candidates)
        if not best:
            missing += 1
            print(f"[MISS] {ann_file.name} -> {original}")
            continue

        if not dry_run:
            if backup_dir:
                shutil.copy2(ann_file, backup_dir / ann_file.name)
            data["filename"] = best.name
            ann_file.write_text(json.dumps(data, ensure_ascii=False, indent=2))

        changed += 1
        print(f"[FIX] {ann_file.name}: {original} -> {best.name}")

    print(f"\nSummary: changed={changed}, unchanged={unchanged}, missing={missing}, dry_run={dry_run}")
    if backup_dir and changed and not dry_run:
        print(f"Backup dir: {backup_dir}")


if __name__ == "__main__":
    main()
