from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict


class BaseEvaluator(ABC):
    """Abstract base class for label evaluation algorithms.

    Subclasses should implement the ``evaluate`` method.

    Notes
    -----
    This interface is intentionally minimal to allow flexible inputs.
    """

    @abstractmethod
    def evaluate(self, y_true: Any, y_pred: Any, **kwargs: Any) -> Dict[str, Any]:
        """Compute evaluation metrics.

        Parameters
        ----------
        y_true : Any
            Ground-truth labels or structured references.
        y_pred : Any
            Predicted labels or outputs to be evaluated.
        **kwargs : Any
            Optional algorithm-specific parameters.

        Returns
        -------
        Dict[str, Any]
            Mapping from metric names to computed values.
        """


