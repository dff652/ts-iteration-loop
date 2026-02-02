"""Evaluation subpackage providing a base interface, registry, and algorithms."""

from .base import BaseEvaluator
from .registry import EVALUATOR_REGISTRY, register_evaluator, get_evaluator

__all__ = [
    "BaseEvaluator",
    "EVALUATOR_REGISTRY",
    "register_evaluator",
    "get_evaluator",
]

# Trigger registrations for built-in evaluators
from . import simple_accuracy  # noqa: F401
from . import lb_eval  # noqa: F401


