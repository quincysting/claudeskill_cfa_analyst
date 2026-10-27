# cfa-analyst — a CFA senior-analyst skill for Claude

A Claude skill that works like a CFA-charterholder senior financial analyst: it grounds every method in the CFA
Program curriculum, computes every number with a tested deterministic engine, and pairs the quantitative work with the
curriculum's qualitative frameworks. It reads, audits and improves Excel models and financial statements.

## Install (Claude Code)

```bash
git clone https://github.com/quincysting/claudeskill_cfa_analyst ~/.claude/skills/cfa-analyst
pip install numpy scipy openpyxl xlrd pytest     # Python 3.10+
python ~/.claude/skills/cfa-analyst/scripts/check_all.py
```

The skill triggers on investment or financial analysis, Excel model or statement review, and CFA exam questions
(English or Chinese).

### Knowledge base (bring your own materials)

The curriculum search (`scripts/kb.py`) needs an index built from **your own licensed copy** of the study materials;
none of that content is in this repository. Convert your PDFs to Markdown (one file per reading, `<!-- page N -->`
markers, YAML front matter with level/kind/edition/reading, plus a `catalog.json`; the layout produced by
[the converter this was built with](https://github.com/opendatalab/MinerU) and an organizing script), then:

```bash
python scripts/kb.py build --md /path/to/MD        # or set CFA_KB=/path/to/MD
python scripts/kb.py search "contingent immunization" --level 3
```

Without an index the calculation engine, Excel tools and playbooks still work; only curriculum citations are missing.

## What's inside

| Part | Purpose |
|---|---|
| `SKILL.md` | Operating model: numbers only from tested code, curriculum-grounded methods with page citations, quantitative + qualitative, explicit assumptions and sensitivities, CFA Standards of Professional Conduct, routing to playbooks. |
| `scripts/cfa_calc/`, `scripts/cfa.py` | **493 formulas** across every CFA topic: quant 71, economics 61, FRA 41, corporate finance 35, equity 49, fixed income 72, derivatives 62, alternatives 24, portfolio 61, private wealth 17. `python scripts/cfa.py search duration` · `describe bond_price` · `bond_price face=100 coupon_rate=0.05 ytm=0.06 years=10 freq=2`; in Python `import cfa_calc as cfa`. |
| `scripts/xlsx_tool.py` | Excel/CSV: `inspect`, `dump` (values + formulas), `extract` (statement tables → canonical items, English and Chinese labels), `ratios`, `audit` (formula errors, hardcodes, broken row patterns, accounting identities, DCF logic such as FCFF from pre-tax EBIT, terminal-value discounting, WACC consistency), `review` (findings workbook), `patch` (corrected copy with change log). Never modifies the input. |
| `scripts/kb.py` | Curriculum search with file + page citations (SQLite FTS5, deterministic BM25 ranking). |
| `references/` | 13 analyst playbooks (quant, economics, FRA, corporate finance, equity, fixed income, derivatives, alternatives, portfolio, performance & risk, Excel, ethics, conventions) and `calc-catalog.md`. |
| `tests/` | 691 unit tests; expected values come from CFA curriculum worked examples (cited by reading and page) and standard textbooks. |
| `evals/` | Skill-creator eval cases (prompts, input workbooks, graders). |

## Verification

- **Unit tests:** 691, against curriculum worked examples: `python scripts/check_all.py`.
- **Independent check:** a separate re-implementation of 68 high-use functions ran 5,197 checks. These were textbook
  cases, seeded random inputs, and identities such as put-call parity, price ↔ yield round trips, FCFF from EBIT =
  FCFF from CFO, and Gordon-growth limits. It found 3 defects and 12 convention issues; all were fixed with regression
  tests, and a re-run showed 0 failures.
- **Eval benchmark** (with vs. without the skill, independent graders, second round of 10 tasks):

| Task | With skill | Without |
|---|---|---|
| Excel: review and fix a flawed DCF (9 planted errors) | 14/14 | 12/14 |
| Excel: credit + earnings-quality assessment from statements | 9/9 | 7/9 |
| Bond book: analytics, rate shock, futures hedge | 12/12 | 11/12 |
| Equity collar: BSM, zero-cost strike, client issues | 9/9 | 8/9 |
| IPS for a retired couple | 9/9 | 8/9 |
| Ethics case (six items) | 12/12 | 11/12 |
| Callable bond on a calibrated binomial tree, OAS, duration | 8/8 | 7/8 |
| CVA with hazard rates | 7/7 | 6/7 |
| PE waterfall with catch-up and clawback | 7/7 | 6/7 |
| 30 held-out Level I mock questions (case not published) | 29/30 | 29/30 |
| **Pass rate** | **99.7%** | **88.3%** |

Claude without the skill already handles well-specified problems well. The measurable edge is determinism and
grounding: every figure is reproducible from tested code, and methods carry verified curriculum citations.

## Limits

- **Curriculum vintage:** built against the 2013/2014 curriculum. Methods are durable; newer rules (SOFR, IFRS 16,
  GIPS 2020, current tax law) and live market data must come from the user or a live source.
- **Engine scope:** annual binomial trees; no CDS pricing or Dickey-Fuller/Durbin-Watson table lookups (these are
  covered qualitatively).
- **Not investment advice.**

## Notice

CFA® and Chartered Financial Analyst® are trademarks of CFA Institute. This project is not affiliated with or
endorsed by CFA Institute, Kaplan or any other publisher. It contains no curriculum text. Citations name readings and
pages so users can look them up in their own copy.
