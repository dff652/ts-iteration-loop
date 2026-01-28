#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ChatTS data processing pipeline.
"""

from __future__ import annotations

import json
import os
import re
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import convert_annotations as conv_mod
import fix_jsonl_format
import split_anomalies


def build_csv_index(csv_roots: List[Path]) -> Dict[str, Path]:
    """Index all CSV files under csv_roots (recursive) by filename."""
    index: Dict[str, Path] = {}
    for root in csv_roots:
        if not root.exists():
            continue
        for p in root.rglob("*.csv"):
            if p.is_file() and p.name not in index:
                index[p.name] = p
    return index


def candidate_csv_names(image_path: str) -> List[str]:
    """Generate candidate csv filenames from image path."""
    base = os.path.basename(image_path)
    if base.endswith(".jpg"):
        csv_name = base[:-4] + ".csv"
    elif base.endswith(".png"):
        csv_name = base[:-4] + ".csv"
    else:
        csv_name = base.replace(".jpg", ".csv").replace(".png", ".csv")

    candidates = [csv_name, f"数据集{csv_name}"]

    if csv_name.startswith("gdsh_second_"):
        candidates.append(csv_name.replace("gdsh_second_", ""))

    if "NB.LJSJ" in csv_name:
        candidates.append("数据集whlj_ljsj_" + csv_name)

    return candidates


def read_timeseries_values(csv_path: Path) -> List[float]:
    """Read timeseries values from CSV. Prefer 'value' column, fallback to second column."""
    import csv as _csv

    values: List[float] = []
    with csv_path.open("r", encoding="utf-8") as f:
        reader = _csv.DictReader(f)
        if not reader.fieldnames:
            return values
        fieldnames = reader.fieldnames
        if "value" in fieldnames:
            value_field = "value"
        elif len(fieldnames) > 1:
            value_field = fieldnames[1]
        else:
            value_field = fieldnames[-1]

        for row in reader:
            try:
                v = float(row.get(value_field, ""))
                values.append(v)
            except (TypeError, ValueError):
                continue
    return values


def _extract_point_from_filename(csv_name: str) -> str:
    name = csv_name
    if name.endswith(".csv"):
        name = name[:-4]

    # remove trailing date segments
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


def _match_csv_path(
    csv_name: str,
    csv_dirs: List[Path],
    csv_index: Dict[str, Path],
    allow_fuzzy: bool,
) -> Optional[tuple]:
    if csv_name in csv_index:
        p = csv_index[csv_name]
        return p, p.parent
    if not allow_fuzzy:
        return None

    point = _extract_point_from_filename(csv_name)
    if not point:
        return None

    candidates: List[Path] = []
    for root in csv_dirs:
        if not root.exists():
            continue
        for p in root.glob("*.csv"):
            if point in p.name:
                candidates.append(p)

    if not candidates:
        return None

    prefixes = ["_chatts_", "_qwen_", "global_chatts_", "chatts_", "qwen_", "timer_"]
    prefix = next((p for p in prefixes if csv_name.startswith(p)), "")

    def score(p: Path):
        name = p.name
        pref = 1 if (prefix and prefix in name) else 0
        try:
            mtime = p.stat().st_mtime
        except OSError:
            mtime = 0
        return (pref, mtime, -len(name))

    best = sorted(candidates, key=score, reverse=True)[0]
    return best, best.parent


def build_chatts_jsonl(
    ann_dir: Path,
    csv_dirs: List[Path],
    output_jsonl: Path,
    dry_run: bool = False,
    missing_log: Optional[Path] = None,
    allow_fuzzy: bool = False,
) -> Tuple[int, int, dict, set]:
    """Build ChatTS JSONL directly from annotation JSONs and CSVs."""
    if not ann_dir.exists():
        raise FileNotFoundError(f"Annotation dir not found: {ann_dir}")

    csv_index = build_csv_index(csv_dirs)

    processed = 0
    missing = 0
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)

    missing_records: List[str] = []
    source_counts: dict = {}
    ts_len_set: set = set()

    out_handle = None
    if not dry_run:
        output_jsonl.parent.mkdir(parents=True, exist_ok=True)
        out_handle = output_jsonl.open("w", encoding="utf-8")

    for ann_file in sorted(ann_dir.glob("*.json")):
        try:
            ann_data = json.loads(ann_file.read_text())
        except Exception:
            missing += 1
            missing_records.append(f"{ann_file.name}\tINVALID_JSON")
            continue

        csv_name = ann_data.get("filename")
        if not csv_name:
            missing += 1
            missing_records.append(f"{ann_file.name}\tNO_FILENAME")
            continue

        matched = _match_csv_path(csv_name, csv_dirs, csv_index, allow_fuzzy)
        if not matched:
            missing += 1
            missing_records.append(f"{ann_file.name}\tNO_CSV\t{csv_name}")
            continue
        csv_path, csv_parent = matched

        values = read_timeseries_values(csv_path)
        if not values:
            missing += 1
            missing_records.append(f"{ann_file.name}\tEMPTY_CSV\t{csv_path}")
            continue
        ts_len_set.add(len(values))

        conv = conv_mod.convert_annotation_to_conversation(ann_data, image_path="")
        conversations = conv.get("conversations", [])
        prompt = ""
        response = ""
        for c in conversations:
            if c.get("from") == "user":
                prompt = c.get("value", "").replace("<image>", "<ts><ts/>")
            elif c.get("from") == "assistant":
                response = c.get("value", "")

        if not prompt or not response:
            missing += 1
            missing_records.append(f"{ann_file.name}\tNO_PROMPT_OR_RESPONSE\t{csv_path}")
            continue

        if not dry_run:
            entry = {
                "input": prompt,
                "output": response,
                "timeseries": [values],
            }
            out_handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
        processed += 1
        source_counts[str(csv_parent)] = source_counts.get(str(csv_parent), 0) + 1

    if out_handle:
        out_handle.close()

    if missing_log and missing_records:
        missing_log.parent.mkdir(parents=True, exist_ok=True)
        with missing_log.open("w", encoding="utf-8") as f:
            for line in missing_records:
                f.write(line + "\n")

    return processed, missing, source_counts, ts_len_set


def _summarize_source(csv_dirs: List[Path], source_counts: dict) -> str:
    if not source_counts:
        return "unknown"
    labels = {}
    for root in csv_dirs:
        root_str = str(root)
        if "inference" in root_str and "chatts" in root_str:
            labels[root_str] = "inference_chatts"
        elif "downsampled" in root_str:
            labels[root_str] = "downsampled"
        else:
            labels[root_str] = "other"

    used = set()
    for root_str in source_counts.keys():
        used.add(labels.get(root_str, "other"))

    if len(used) == 1:
        return list(used)[0]
    return "mixed"


def _summarize_len(ts_len_set: set) -> str:
    if not ts_len_set:
        return "unknown"
    if len(ts_len_set) == 1:
        return str(next(iter(ts_len_set)))
    return "var"


def run_chatts_pipeline(
    ann_dir: Path,
    csv_dirs: List[Path],
    output_dir: Path,
    split_enabled: bool,
    fix_enabled: bool,
    dry_run: bool,
    strict_match: bool,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir = output_dir / "_pipeline_tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    train_jsonl = output_dir / "train.jsonl"
    missing_log = output_dir / "missing_records.log"
    processed, missing, source_counts, ts_len_set = build_chatts_jsonl(
        ann_dir,
        csv_dirs,
        train_jsonl,
        dry_run=dry_run,
        missing_log=missing_log,
        allow_fuzzy=not strict_match,
    )
    if dry_run:
        print(f"[DRY_RUN] processed={processed}, missing={missing}, log={missing_log}")
        return {
            "processed": processed,
            "missing": missing,
            "missing_log": missing_log,
        }

    print(f"[OK] train.jsonl generated: {train_jsonl} (processed={processed}, missing={missing})")

    target_jsonl = train_jsonl
    final_count = processed
    if split_enabled:
        split_jsonl = output_dir / "train_split.jsonl"
        split_stats = split_anomalies.process_jsonl_file(str(train_jsonl), str(split_jsonl))
        final_count = split_stats.get("split_samples", final_count)
        print(f"[OK] train_split.jsonl generated: {split_jsonl}")
        target_jsonl = split_jsonl

    if fix_enabled:
        fixed_tmp = tmp_dir / f"{target_jsonl.stem}_fixed.jsonl"
        fix_jsonl_format.fix_jsonl_format(str(target_jsonl), str(fixed_tmp))
        fixed_tmp.replace(target_jsonl)
        print(f"[OK] format fixed: {target_jsonl}")

    source_label = _summarize_source(csv_dirs, source_counts)
    len_label = _summarize_len(ts_len_set)
    split_label = "split" if split_enabled else "nosplit"
    date_label = datetime.now().strftime("%Y%m%d")
    named_filename = f"chatts_tune_{len_label}_n{final_count}_{source_label}_{split_label}_{date_label}.jsonl"
    named_path = output_dir / named_filename
    shutil.copyfile(target_jsonl, named_path)
    print(f"[OK] named output: {named_path}")

    return {
        "processed": processed,
        "missing": missing,
        "train_jsonl": train_jsonl,
        "named_path": named_path,
    }
