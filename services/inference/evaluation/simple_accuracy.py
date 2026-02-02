from __future__ import annotations

from typing import Any, Dict

import numpy as np

from .base import BaseEvaluator
from .registry import register_evaluator


@register_evaluator("simple_accuracy")
class SimpleAccuracyEvaluator(BaseEvaluator):
    """A minimal evaluator that computes accuracy for 1D label arrays.

    This is a reference implementation showing how to plug into the registry.
    It attempts to coerce inputs to NumPy arrays and computes mean equality.
    """

    def evaluate(self, y_true: Any, y_pred: Any, **kwargs: Any) -> Dict[str, Any]:
        """Compute simple accuracy.

        Parameters
        ----------
        y_true : Any
            Ground-truth labels; will be coerced to a 1D NumPy array.
        y_pred : Any
            Predicted labels; will be coerced to a 1D NumPy array.
        **kwargs : Any
            Unused. Present for signature compatibility.

        Returns
        -------
        Dict[str, Any]
            Dictionary with a single key "accuracy".
        """

        y_true_arr = np.asarray(y_true).ravel()
        y_pred_arr = np.asarray(y_pred).ravel()
        if y_true_arr.shape != y_pred_arr.shape:
            raise ValueError(f"Shape mismatch: y_true{y_true_arr.shape} vs y_pred{y_pred_arr.shape}")
        accuracy = float(np.mean(y_true_arr == y_pred_arr)) if y_true_arr.size > 0 else 0.0
        return {"accuracy": accuracy}


