#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unified pipeline entry for converting annotation results into training JSONL.
Supports ChatTS/Qwen with modular pipelines.
"""

import argparse
import sys
from pathlib import Path

# Ensure project root is on sys.path for settings import
PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from configs.settings import settings

# Reuse existing scripts (load via sys.path to avoid import issues)
DATA_PROCESSING_DIR = Path(__file__).resolve().parent
SCRIPTS_DIR = DATA_PROCESSING_DIR / "scripts"
TRANS_DIR = SCRIPTS_DIR / "transformation"
PRE_DIR = SCRIPTS_DIR / "preprocessing"
PIPELINES_DIR = DATA_PROCESSING_DIR / "pipelines"
for _p in (str(TRANS_DIR), str(PRE_DIR), str(PIPELINES_DIR), str(DATA_PROCESSING_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from pipelines import chatts as chatts_pipeline
from pipelines import qwen as qwen_pipeline


def _parse_csv_dirs(csv_dir_arg: str):
    if not csv_dir_arg:
        return []
    parts = [p.strip() for p in csv_dir_arg.split(",") if p.strip()]
    return [Path(p) for p in parts]


def main():
    parser = argparse.ArgumentParser(description="Unified data pipeline for ChatTS/Qwen.")
    parser.add_argument("--model_family", choices=["chatts", "qwen"], default="chatts")
    parser.add_argument("--ann_dir", type=str, default=None, help="Annotation JSON directory")
    parser.add_argument("--image_dir", type=str, default=None, help="Image directory")
    parser.add_argument("--csv_dir", type=str, default=None, help="CSV root directory (comma-separated for multiple)")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory")
    parser.add_argument("--split", type=str, default="false", help="true/false")
    parser.add_argument("--fix", type=str, default="true", help="true/false")
    parser.add_argument("--dry_run", type=str, default="false", help="true/false")
    parser.add_argument("--strict", type=str, default="true", help="true/false (strict filename match)")
    args = parser.parse_args()

    split_enabled = args.split.lower() == "true"
    fix_enabled = args.fix.lower() == "true"
    dry_run = args.dry_run.lower() == "true"
    strict_match = args.strict.lower() == "true"

    ann_dir = Path(args.ann_dir or (Path(settings.ANNOTATIONS_ROOT) / settings.DEFAULT_USER))
    image_dir = Path(args.image_dir or settings.DATA_IMAGES_DIR)
    csv_dirs = _parse_csv_dirs(args.csv_dir or "")
    if not csv_dirs:
        csv_dirs = [Path(settings.DATA_DOWNSAMPLED_DIR)]

    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = Path(settings.DATA_TRAINING_CHATTS_DIR if args.model_family == "chatts" else settings.DATA_TRAINING_QWEN_DIR)

    output_dir.mkdir(parents=True, exist_ok=True)

    if args.model_family == "qwen":
        qwen_pipeline.run_qwen_pipeline(
            ann_dir=ann_dir,
            image_dir=image_dir,
            output_dir=output_dir,
        )
        return

    chatts_pipeline.run_chatts_pipeline(
        ann_dir=ann_dir,
        csv_dirs=csv_dirs,
        output_dir=output_dir,
        split_enabled=split_enabled,
        fix_enabled=fix_enabled,
        dry_run=dry_run,
        strict_match=strict_match,
    )


if __name__ == "__main__":
    main()
