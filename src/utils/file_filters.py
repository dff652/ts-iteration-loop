"""
Shared filename filtering helpers for UI and adapters.
"""

from typing import Iterable

# Prefixes for inference results or generated intermediates.
_INFERENCE_RESULT_PREFIXES: Iterable[str] = (
    "global_chatts_",
    "chatts_",
    "_chatts_",
    "timer_",
    "adtk_hbos_",
    "ensemble_",
    "_qwen",
    "qwen_",
)

_RESULT_KEYWORDS_BY_METHOD = {
    "chatts": ("qwen",),   # chatts 结果不应包含 qwen 关键词
    "qwen": ("qwen",),     # qwen 结果必须包含 qwen 关键词
    "timer": ("timer",),   # timer 结果必须包含 timer 关键词
}


def is_inference_or_generated_csv(filename: str) -> bool:
    """Return True if the CSV name looks like an inference/generated file."""
    name = filename.lower()
    for prefix in _INFERENCE_RESULT_PREFIXES:
        if name.startswith(prefix):
            return True
    return "_trend_resid" in name


def match_result_method(filename: str, method: str) -> bool:
    """Return True if filename matches result method filtering rules."""
    name = filename.lower()
    method = (method or "").lower()

    # Default: allow if method not in rules.
    if method not in _RESULT_KEYWORDS_BY_METHOD:
        return True

    keyword = _RESULT_KEYWORDS_BY_METHOD[method]
    if method == "chatts":
        return keyword[0] not in name
    return keyword[0] in name
