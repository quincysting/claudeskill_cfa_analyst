"""Registry of CFA calculations.

Every public calculation is decorated with ``@formula`` so the CLI (``scripts/cfa.py``) can list,
describe and call it by name. Names are global across modules, so they must be unique.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

TOPICS = ("quant", "economics", "fra", "corporate_finance", "equity", "fixed_income", "derivatives",
          "alternatives", "portfolio", "pwm")


@dataclass(frozen=True)
class Formula:
    name: str
    topic: str
    func: Callable
    formula: str      # the textbook formula, plain text, e.g. "P = sum CF_t / (1 + y/m)^t"
    curriculum: str   # where the CFA curriculum teaches it, e.g. "L1 FI: Introduction to Fixed-Income Valuation"


REGISTRY: dict[str, Formula] = {}


def formula(topic: str, formula: str = "", curriculum: str = "") -> Callable[[Callable], Callable]:
    """Register a calculation. Returns the function unchanged."""
    if topic not in TOPICS:
        raise ValueError(f"unknown topic {topic!r}; use one of {TOPICS}")

    def wrap(func: Callable) -> Callable:
        name = func.__name__
        if name in REGISTRY and REGISTRY[name].func.__module__ != func.__module__:
            raise ValueError(f"duplicate formula name {name!r}: {REGISTRY[name].func.__module__} and {func.__module__}")
        REGISTRY[name] = Formula(name, topic, func, formula, curriculum)
        return func

    return wrap
