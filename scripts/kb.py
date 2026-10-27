"""Search the CFA knowledge base: curriculum 2013/2014 (Levels I-III), Kaplan/Schweser notes, Chinese notes,
Secret Sauce, glossaries and the Standards of Practice Handbook, as converted to Markdown.

  python kb.py search "macaulay duration" [--level 1] [--topic fixed] [--series curriculum-2014] [--lang zh] [-k 8]
  python kb.py show Level-1/08-Fixed-Income/Curriculum-2014/SS16-R55-....md --page 512 [--context 1]
  python kb.py los --level 2 --reading 30 [--edition 2014]      learning outcome statements of a reading
  python kb.py readings [--level 3] [--topic ethics]           reading list (level, session, reading, title, file)
  python kb.py stats
  python kb.py build [--md E:/CLAUDE/CFA/MD] [--include-exams]  rebuild data/kb.sqlite from the Markdown

Ranking is SQLite FTS5 BM25 (title x3, heading x2, body x1), ties broken by file and page, so the same query always
returns the same passages. Chinese text is indexed per character, so Chinese queries work too. Mock exams are left
out of the index by default: they are the held-out test set for this skill.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sqlite3
import sys
from pathlib import Path

SKILL = Path(__file__).resolve().parent.parent
DB = SKILL / "data" / "kb.sqlite"
DEFAULT_MD = os.environ.get("CFA_KB", "E:/CLAUDE/CFA/MD")
PAGE = re.compile(r"<!-- page (\d+) -->")
HEADING = re.compile(r"^#{1,4}\s+(.+?)\s*$", re.M)
CJK = re.compile(r"([\u3400-\u9fff\uf900-\ufaff])")
WORD = re.compile(r"[\w\u3400-\u9fff]+")


def cjk_split(text: str) -> str:
    """Put spaces around CJK characters so the unicode61 tokenizer indexes them one by one."""
    return CJK.sub(r" \1 ", text)


def front_matter(text: str) -> tuple[dict, str]:
    if not text.startswith("---"):
        return {}, text
    end = text.find("\n---", 3)
    meta = {}
    for line in text[3:end].splitlines():
        if ":" in line:
            k, v = line.split(":", 1)
            try:
                meta[k.strip()] = json.loads(v.strip())
            except json.JSONDecodeError:
                meta[k.strip()] = v.strip()
    return meta, text[end + 4:]


def connect() -> sqlite3.Connection:
    if not DB.exists():
        raise SystemExit(f"knowledge-base index missing: {DB}\nbuild it with: python {Path(__file__).name} build")
    return sqlite3.connect(DB)


# --- build -------------------------------------------------------------------------------------
def build(md_root: Path, include_exams: bool) -> None:
    catalog = json.loads((md_root / "catalog.json").read_text(encoding="utf-8"))
    DB.parent.mkdir(parents=True, exist_ok=True)
    tmp = DB.with_suffix(".tmp")
    tmp.unlink(missing_ok=True)
    con = sqlite3.connect(tmp)
    con.executescript("""
        CREATE TABLE docs(id INTEGER PRIMARY KEY, file TEXT UNIQUE, title TEXT, level INT, topic TEXT, series TEXT,
                          kind TEXT, edition TEXT, language TEXT, reading INT, study_session INT);
        CREATE TABLE pages(id INTEGER PRIMARY KEY, doc_id INT, page INT, heading TEXT, text TEXT);
        CREATE INDEX pages_doc ON pages(doc_id, page);
        CREATE VIRTUAL TABLE fts USING fts5(title, heading, body, content='',
                                            tokenize='porter unicode61 remove_diacritics 2');
    """)
    n_docs = n_pages = 0
    for c in sorted(catalog, key=lambda c: c["file"]):
        if c.get("kind") == "exam" and not include_exams:
            continue
        path = md_root / c["file"]
        meta, body = front_matter(path.read_text(encoding="utf-8"))
        parts = PAGE.split(body)
        chunks = [(int(parts[i]), parts[i + 1]) for i in range(1, len(parts) - 1, 2)] or [(0, body)]
        series = c["file"].split("/")[2] if c["file"].count("/") >= 3 else c["file"].split("/")[1] \
            if c["file"].count("/") == 2 else "Reference"
        cur = con.execute("INSERT INTO docs(file,title,level,topic,series,kind,edition,language,reading,study_session)"
                          " VALUES (?,?,?,?,?,?,?,?,?,?)",
                          (c["file"], c.get("title"), c.get("level") or 0, c.get("topic") or "", series,
                           c.get("kind") or "", str(c.get("edition") or ""), c.get("language") or "",
                           c.get("reading"), c.get("study_session")))
        doc_id = cur.lastrowid
        n_docs += 1
        heading = ""
        for page, text in chunks:
            text = text.strip()
            if len(text) < 40:
                continue
            heads = HEADING.findall(text)
            page_heading = heads[0] if heads else heading
            heading = heads[-1] if heads else heading
            pid = con.execute("INSERT INTO pages(doc_id,page,heading,text) VALUES (?,?,?,?)",
                              (doc_id, page, page_heading[:200], text)).lastrowid
            con.execute("INSERT INTO fts(rowid,title,heading,body) VALUES (?,?,?,?)",
                        (pid, cjk_split(c.get("title") or ""), cjk_split(page_heading), cjk_split(text)))
            n_pages += 1
    con.execute("INSERT INTO fts(fts) VALUES ('optimize')")
    con.commit()
    con.execute("VACUUM")
    con.close()
    tmp.replace(DB)
    print(f"indexed {n_docs} files, {n_pages} pages -> {DB} ({DB.stat().st_size / 2**20:.0f} MB)")


# --- search ------------------------------------------------------------------------------------
def fts_query(q: str, mode: str) -> str:
    terms = [t for t in WORD.findall(cjk_split(q).lower()) if t]
    quoted = [f'"{t}"' for t in terms]
    return (" AND " if mode == "and" else " OR ").join(quoted)


def filters(args) -> tuple[str, list]:
    sql, params = [], []
    if args.level is not None:
        sql.append("d.level = ?"); params.append(args.level)
    if args.topic:
        sql.append("(lower(d.topic) LIKE ? OR lower(d.file) LIKE ?)"); params += [f"%{args.topic.lower()}%"] * 2
    if args.series:
        sql.append("lower(d.series) LIKE ?"); params.append(f"%{args.series.lower()}%")
    if args.lang:
        sql.append("d.language = ?"); params.append(args.lang)
    if args.kind:
        sql.append("d.kind = ?"); params.append(args.kind)
    return (" AND " + " AND ".join(sql)) if sql else "", params


def snippet(text: str, query: str, width: int = 420) -> str:
    flat = " ".join(text.split())
    terms = sorted({t for t in WORD.findall(query.lower()) if len(t) > 1}, key=len, reverse=True)
    low = flat.lower()
    pos = min((low.find(t) for t in terms if low.find(t) >= 0), default=0)
    start = max(0, pos - width // 3)
    return ("…" if start else "") + flat[start:start + width] + ("…" if start + width < len(flat) else "")


def search(args) -> None:
    con = connect()
    where, params = filters(args)
    rows = []
    for mode in ("and", "or"):  # all terms first; fall back to any term
        q = fts_query(args.query, mode)
        if not q:
            raise SystemExit("empty query")
        rows = con.execute(
            f"""SELECT d.file, p.page, p.heading, d.level, d.topic, d.series, d.reading, p.text,
                       bm25(fts, 3.0, 2.0, 1.0) AS score
                FROM fts JOIN pages p ON p.id = fts.rowid JOIN docs d ON d.id = p.doc_id
                WHERE fts MATCH ? {where}
                ORDER BY score, d.file, p.page LIMIT ?""", [q, *params, args.k]).fetchall()
        if len(rows) >= min(args.k, 3) or mode == "or":
            break
    if args.json:
        print(json.dumps([{"file": r[0], "page": r[1], "heading": r[2], "level": r[3], "topic": r[4],
                           "series": r[5], "reading": r[6], "snippet": snippet(r[7], args.query)} for r in rows],
                         ensure_ascii=False, indent=1))
        return
    if not rows:
        print("no matches; try fewer or different words, or drop filters")
    for i, (file, page, heading, level, topic, series, reading, text, score) in enumerate(rows, 1):
        tag = " · ".join(x for x in [f"L{level}" if level else "", topic, series, f"R{reading}" if reading else ""] if x)
        print(f"[{i}] {file}  p.{page}  ({tag})")
        if heading:
            print(f"    § {heading}")
        print(f"    {snippet(text, args.query)}\n")


def show(args) -> None:
    con = connect()
    doc = con.execute("SELECT id, title FROM docs WHERE file = ? OR file LIKE ?",
                      (args.file, f"%{args.file}")).fetchone()
    if not doc:
        raise SystemExit(f"no indexed file matches {args.file!r}")
    lo, hi = (args.page - args.context, args.page + args.context) if args.page is not None else (0, 10**9)
    rows = con.execute("SELECT page, text FROM pages WHERE doc_id = ? AND page BETWEEN ? AND ? ORDER BY page",
                       (doc[0], lo, hi)).fetchall()
    if args.page is None:
        rows = rows[: args.max_pages]
    print(f"# {doc[1]}\n")
    for page, text in rows:
        print(f"<!-- page {page} -->\n{text}\n")


def los(args) -> None:
    con = connect()
    sql = ("SELECT d.file, d.title, p.page, p.text FROM docs d JOIN pages p ON p.doc_id = d.id "
           "WHERE d.kind = 'curriculum' AND d.level = ? AND d.reading = ? AND d.edition LIKE ? "
           "AND upper(p.text) LIKE '%CANDIDATE SHOULD BE ABLE TO%' ORDER BY d.edition DESC, p.page LIMIT 2")
    rows = con.execute(sql, (args.level, args.reading, f"%{args.edition or ''}%")).fetchall()
    if not rows:
        raise SystemExit("no learning outcome statements found for that level/reading")
    print(f"# {rows[0][1]}  ({rows[0][0]} p.{rows[0][2]})\n")
    for _, _, page, text in rows:
        if rows[0][2] <= page <= rows[0][2] + 1:
            block = text[text.upper().find("CANDIDATE SHOULD BE ABLE TO"):]
            print(re.split(r"\n#+ (?!Mastery)", block, maxsplit=1)[0].strip(), "\n")


def readings(args) -> None:
    con = connect()
    where, params = ["d.kind = 'curriculum'"], []
    if args.level is not None:
        where.append("d.level = ?"); params.append(args.level)
    if args.topic:
        where.append("lower(d.file) LIKE ?"); params.append(f"%{args.topic.lower()}%")
    if args.edition:
        where.append("d.edition = ?"); params.append(args.edition)
    for row in con.execute(f"""SELECT level, edition, study_session, reading, title, file FROM docs d
                               WHERE {' AND '.join(where)} AND reading IS NOT NULL
                               ORDER BY level, edition DESC, reading""", params):
        print(f"L{row[0]} {row[1]} SS{row[2] or 0:02d} R{row[3]:02d}  {row[4]}\n      {row[5]}")


def stats(_args) -> None:
    con = connect()
    for row in con.execute("""SELECT d.level, d.series, count(DISTINCT d.id), count(p.id) FROM docs d
                              JOIN pages p ON p.doc_id = d.id GROUP BY 1, 2 ORDER BY 1, 2"""):
        print(f"L{row[0]}  {row[1]:28} {row[2]:4} files {row[3]:6} pages")


def main(argv: list[str]) -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("search")
    s.add_argument("query")
    s.add_argument("-k", type=int, default=8)
    s.add_argument("--level", type=int)
    s.add_argument("--topic")
    s.add_argument("--series", help="e.g. curriculum-2014, kaplan, schweser, zh, secret, glossary")
    s.add_argument("--lang", help="en or zh")
    s.add_argument("--kind", help="curriculum, notes, summary, glossary, reference, guide, exercise")
    s.add_argument("--json", action="store_true")
    s = sub.add_parser("show")
    s.add_argument("file")
    s.add_argument("--page", type=int)
    s.add_argument("--context", type=int, default=0)
    s.add_argument("--max-pages", type=int, default=6)
    s = sub.add_parser("los")
    s.add_argument("--level", type=int, required=True)
    s.add_argument("--reading", type=int, required=True)
    s.add_argument("--edition")
    s = sub.add_parser("readings")
    s.add_argument("--level", type=int)
    s.add_argument("--topic")
    s.add_argument("--edition")
    sub.add_parser("stats")
    s = sub.add_parser("build")
    s.add_argument("--md", default=DEFAULT_MD)
    s.add_argument("--include-exams", action="store_true")
    args = ap.parse_args(argv)
    if args.cmd == "build":
        build(Path(args.md), args.include_exams)
    else:
        {"search": search, "show": show, "los": los, "readings": readings, "stats": stats}[args.cmd](args)


if __name__ == "__main__":
    main(sys.argv[1:])
