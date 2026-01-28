#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Qwen data processing pipeline.
"""

from __future__ import annotations

from pathlib import Path

import convert_annotations as conv_mod


def run_qwen_pipeline(
    ann_dir: Path,
    image_dir: Path,
    output_dir: Path,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    out_json = output_dir / "converted_data.json"
    conv_mod.convert_annotations(
        input_dir=str(ann_dir),
        output_file=str(out_json),
        image_dir=str(image_dir),
        format_type="qwen",
        csv_src_dir=None,
    )

    final_path = conv_mod.find_latest_auto_file(str(output_dir), "qwen")
    if final_path:
        print(f"[OK] Qwen data saved: {final_path}")
        return Path(final_path)

    print(f"[OK] Qwen data saved: {out_json}")
    return out_json
