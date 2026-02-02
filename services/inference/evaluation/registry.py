from __future__ import annotations

from typing import Any, Callable, Dict, Type

from .base import BaseEvaluator


EVALUATOR_REGISTRY: Dict[str, Type[BaseEvaluator]] = {}


def register_evaluator(name: str) -> Callable[[Type[BaseEvaluator]], Type[BaseEvaluator]]:
    """Class decorator to register an evaluator by name.

    Parameters
    ----------
    name : str
        Unique name to register the evaluator under.

    Returns
    -------
    Callable[[Type[BaseEvaluator]], Type[BaseEvaluator]]
        Decorator that registers the class and returns it unchanged.
    """

    def decorator(cls: Type[BaseEvaluator]) -> Type[BaseEvaluator]:
        if not issubclass(cls, BaseEvaluator):
            raise TypeError(f"{cls.__name__} must inherit from BaseEvaluator")
        if name in EVALUATOR_REGISTRY:
            raise KeyError(f"Evaluator '{name}' is already registered")
        EVALUATOR_REGISTRY[name] = cls
        return cls

    return decorator


def get_evaluator(name: str, /, **init_kwargs: Any) -> BaseEvaluator:
    """Instantiate a registered evaluator by name.

    Parameters
    ----------
    name : str
        Registered evaluator name.
    **init_kwargs : Any
        Keyword arguments forwarded to the evaluator's constructor.

    Returns
    -------
    BaseEvaluator
        Instantiated evaluator.
    """

    if name not in EVALUATOR_REGISTRY:
        raise KeyError(f"Evaluator '{name}' is not registered. Available: {list(EVALUATOR_REGISTRY)}")
    evaluator_cls = EVALUATOR_REGISTRY[name]
    return evaluator_cls(**init_kwargs)


