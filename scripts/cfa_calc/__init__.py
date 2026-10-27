"""Deterministic CFA calculation library. Importing the package registers every module's formulas."""
from __future__ import annotations

import importlib
import pkgutil

from ._registry import REGISTRY, TOPICS, Formula, formula

for _mod in sorted(m.name for m in pkgutil.iter_modules(__path__) if not m.name.startswith("_")):
    importlib.import_module(f"{__name__}.{_mod}")

# every formula is also a package attribute, so scripts need not know which module holds it:
#   import cfa_calc as cfa; cfa.bond_price(...); cfa.concentrated_position_strategies(...)
globals().update({_name: _f.func for _name, _f in REGISTRY.items()})

__all__ = ["REGISTRY", "TOPICS", "Formula", "formula", *sorted(REGISTRY)]
