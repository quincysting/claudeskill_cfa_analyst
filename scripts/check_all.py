"""Verify the skill: every engine and tool test, registry integrity, and the knowledge-base index.

  python scripts/check_all.py          exits non-zero on any failure
"""
from __future__ import annotations

import inspect
import subprocess
import sys
from pathlib import Path

SKILL = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SKILL / "scripts"))


def registry_problems() -> list[str]:
    from cfa_calc import REGISTRY, TOPICS
    problems = []
    for name, f in REGISTRY.items():
        doc = inspect.getdoc(f.func) or ""
        if f.topic not in TOPICS:
            problems.append(f"{name}: unknown topic {f.topic}")
        if not doc.strip():
            problems.append(f"{name}: no docstring")
        if not f.formula and not f.curriculum:
            problems.append(f"{name}: neither formula nor curriculum source given")
    return problems


def main() -> int:
    from cfa_calc import REGISTRY
    by_topic: dict[str, int] = {}
    for f in REGISTRY.values():
        by_topic[f.topic] = by_topic.get(f.topic, 0) + 1
    print(f"engine: {len(REGISTRY)} formulas  " + "  ".join(f"{k}={v}" for k, v in sorted(by_topic.items())))
    probs = registry_problems()
    for p in probs:
        print("registry:", p)
    if not (SKILL / "data" / "kb.sqlite").exists():
        print("knowledge base index missing: python scripts/kb.py build")
    rc = subprocess.call([sys.executable, "-m", "pytest", str(SKILL / "tests"), "-q", "-p", "no:cacheprovider"],
                         cwd=SKILL)
    ok = rc == 0 and not probs
    print("ALL CHECKS PASSED" if ok else "CHECKS FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
