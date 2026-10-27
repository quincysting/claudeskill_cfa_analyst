"""Fixed income (CFA Levels I-III) — one namespace for every `fixed_income` formula.

The implementations live in the fi_* modules (pricing, yields, curve, risk, credit, trees, securitized,
portfolio); this module re-exports them so `from cfa_calc import fixed_income as fi; fi.bond_price(...)` works.
"""
from __future__ import annotations

from . import fi_credit, fi_curve, fi_portfolio, fi_pricing, fi_risk, fi_securitized, fi_trees, fi_yields  # noqa: F401
from ._registry import REGISTRY

__all__ = sorted(n for n, f in REGISTRY.items() if f.topic == "fixed_income")
globals().update({n: REGISTRY[n].func for n in __all__})
