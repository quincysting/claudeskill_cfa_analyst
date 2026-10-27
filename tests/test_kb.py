"""Knowledge-base index: known queries find the right readings, results are deterministic, exams are held out."""
import json
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest

SKILL = Path(__file__).resolve().parent.parent
KB = SKILL / "scripts" / "kb.py"
DB = SKILL / "data" / "kb.sqlite"

pytestmark = pytest.mark.skipif(not DB.exists(), reason="index not built: python scripts/kb.py build")


def run(*args: str) -> str:
    out = subprocess.run([sys.executable, str(KB), *args], capture_output=True, text=True, encoding="utf-8")
    assert out.returncode == 0, out.stderr
    return out.stdout


def search(query: str, *flags: str) -> list[dict]:
    return json.loads(run("search", query, "--json", *flags))


@pytest.mark.parametrize("query, flags, expect", [
    ("macaulay duration modified duration", ("--level", "1", "--series", "curriculum-2014"), "Fixed-Income"),
    ("free cash flow to the firm", ("--level", "2", "--series", "curriculum-2014"), "Free-Cash-Flow"),
    ("contingent immunization", ("--level", "3",), "Fixed-Income"),
    ("put-call parity", ("--level", "1", "--series", "curriculum-2014"), "Derivatives"),
    ("material nonpublic information mosaic theory", (), "Ethics"),
    ("residual income model persistence factor", ("--level", "2",), "Residual"),
    ("time-weighted rate of return", ("--level", "1",), "discounted-cash-flow-applications"),  # L1 Quant R6
    ("久期", (), "Fixed-Income"),
])
def test_queries_find_their_readings(query, flags, expect):
    hits = search(query, "-k", "5", *flags)
    assert hits, f"no hits for {query}"
    assert any(expect.lower() in h["file"].lower() for h in hits), [h["file"] for h in hits]


def test_deterministic():
    assert search("cost of capital beta", "-k", "8") == search("cost of capital beta", "-k", "8")


def test_mock_exams_held_out():
    con = sqlite3.connect(DB)
    assert con.execute("SELECT count(*) FROM docs WHERE kind = 'exam' OR file LIKE '%/Exams/%'").fetchone()[0] == 0


def test_los_and_readings():
    assert "candidate should be able to" in run("los", "--level", "2", "--reading", "30").lower()
    lines = run("readings", "--level", "1", "--edition", "2014").splitlines()
    assert sum(1 for ln in lines if ln.startswith("L1 2014")) == 63


def test_show_page():
    hit = search("Macaulay duration", "--level", "1", "--series", "curriculum-2014", "-k", "1")[0]
    text = run("show", hit["file"], "--page", str(hit["page"]))
    assert f"<!-- page {hit['page']} -->" in text and "duration" in text.lower()
