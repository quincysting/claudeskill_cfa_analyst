# Playbook: reading, analyzing and improving Excel workbooks

People bring their own spreadsheets: a company's statements, a DCF or LBO model, a portfolio's holdings and returns,
a bond schedule, a budget. The job is to understand the workbook as it is, judge it against CFA standards, quantify
what is wrong, and leave the user with something better, without ever damaging their file.

## Contents
1. Workflow
2. What the tool checks for you
3. CFA review checklists by workbook type (what only an analyst can check)
4. Recompute and reconcile
5. Improving the workbook
6. Deliverable

## 1. Workflow

| Step | Command | Why |
|---|---|---|
| Inspect | `python scripts/xlsx_tool.py inspect FILE` | sheets, size, formula counts, error cells, external links, hidden sheets/rows, charts/pivots present |
| Read | `python scripts/xlsx_tool.py dump FILE --sheet S [--range A1:K60]` | every cell's value and, for formulas, the formula: you see how numbers are built, not just what they are |
| Ratios | `python scripts/xlsx_tool.py ratios FILE` | the full CFA ratio set per period from the extracted statements (prior period = beginning balance) |
| Structure | `python scripts/xlsx_tool.py extract FILE [--json]` | period x line-item tables with labels mapped to canonical items (`revenue`, `ebit`, `cfo`, `total_assets`, ...), English or Chinese labels |
| Audit | `python scripts/xlsx_tool.py audit FILE [--json]` | formula and model-integrity findings, most severe first |
| Recompute | `scripts/cfa.py` / `cfa_calc` (section 4) | independent figures to reconcile against the model |
| Improve | `xlsx_tool.py review` and, when asked, `xlsx_tool.py patch` | a review workbook, and a corrected copy, never the original |

Read before judging: dump the sheets the question is about and trace the key outputs back to their inputs. Spreadsheets
lie in mundane ways: a typed number over a formula, a range that misses a row, a percentage stored as 8 instead of 0.08.

`.xlsx/.xlsm` are read twice, once for formulas and once for cached values. `.xls` and `.csv` give values only, so
formula checks do not apply; say so. If a formula cell has no cached value (`formulas_without_cached_value` in
inspect), the file was written by a program that did not recalculate; ask the user to open and save it in Excel, or
recompute the figures with the engine.

## 2. What the tool checks for you (`audit`)

- Formula errors (#REF!, #DIV/0!, #VALUE!, #N/A, #NAME?), external links and references, hidden sheets and rows.
- Hardcoded numbers inside formulas (`=D4*1.06`), typed values over a formula row, formulas that break their row's
  pattern (the odd one out among neighbors).
- Accounting identities on extracted tables: assets = liabilities + equity, CFO + CFI + CFF = change in cash, gross
  profit ties, and equity roll-forward (prior equity + net income - dividends; a gap means buybacks, issuance, OCI
  or statements that are incomplete).
- Valuation sanity: terminal growth vs discount rate and vs long-run nominal GDP, rates stored as whole numbers,
  several WACC figures that disagree, WACC weights at book value, terminal value share of EV, FCFF built from pre-tax
  EBIT, terminal value discounted over fewer periods than the forecast.

A workbook of typed values (no formulas) gets only the identity and valuation checks, so a short findings list there
is not a clean bill of health: the analysis in sections 3 and 4 still has to be done.

Treat its findings as leads to verify, not verdicts: open the cells (`dump --range`) and confirm before reporting.
It cannot see intent (an override may be deliberate and documented) and it checks only what it can recognise.

## 3. CFA review checklists (what only an analyst can check)

### DCF / valuation models (L2 Equity: Free Cash Flow Valuation, DDM, Residual Income; L1/L2 Cost of Capital)
- **Cash-flow definition matches the discount rate.** FCFF = EBIT(1 - t) + D&A - FCInv - WCInv, discounted at WACC;
  FCFE = FCFF - Int(1 - t) + net borrowing, at the cost of equity. Mixing them is the classic error.
- **Tax rate.** Marginal rate on EBIT for NOPAT; the effective rate only if it is sustainable.
- **WACC.** Market-value (target) weights; after-tax cost of debt at the current yield, not the coupon; cost of
  equity from CAPM with a consistent risk-free rate (currency and maturity); beta relevered for the target capital
  structure (Hamada); country risk where relevant.
- **Terminal value.** Gordon g below the discount rate and at or below long-run nominal GDP; terminal reinvestment
  consistent with growth (reinvestment rate = g / ROIC); a normalized final-year cash flow (no growth capex spike,
  margins at steady state); exit multiple cross-checked against the implied multiple. Discount TV from the end of the
  final forecast year. If TV is above about 75% of EV, show the sensitivity grid.
- **Timing.** End-of-year vs mid-year convention applied consistently; stub periods for valuation dates within a year.
- **Bridge to equity.** EV - debt (incl. leases if FCFF excludes lease payments) - preferred - noncontrolling interest +
  cash and non-operating assets; diluted shares by the treasury stock method; options and SBC treated consistently.
- **Consistency.** Nominal cash flows with nominal rates (or real with real); the currency of cash flows matches the rate.
- **Sanity.** Revenue growth vs industry and GDP; margins vs history and peers; capex vs D&A in steady state; NWC as a
  share of revenue stable unless explained.

### Financial statements (L1/L2 FRA)
- Run the full ratio set (activity, liquidity, solvency, profitability, DuPont). Read trends, not single years.
- Earnings quality: CFO vs net income, accruals ratio, DSO/DIO/DPO trends, revenue recognized faster than cash
  (receivables growing faster than sales), inventory build-up, capitalized costs, one-offs, reserve releases, changes
  in accounting estimates. Use the FRA playbook's red-flag list.
- Analyst adjustments: LIFO to FIFO, operating leases, pension deficits, non-recurring items, goodwill impairment, FX.
- IFRS vs US GAAP differences that move the ratios (interest and dividends in the cash flow statement, inventory write-down reversals, revaluation).

### Portfolio and performance workbooks (L1/L3 Portfolio Management, GIPS)
- Returns computed correctly: time-weighted for managers (external flows neutralized), money-weighted for investors;
  geometric linking; annualization only for periods over a year.
- Risk-adjusted measures with the right risk-free rate and risk measure (Sharpe on total risk, Treynor on beta,
  information ratio on active risk); √t scaling only for i.i.d. returns.
- Weights sum to 1; covariance matrix symmetric and positive semidefinite; benchmark appropriate (SAMURAI properties).
- GIPS: composites, dispersion, fees (gross vs net), required disclosures.

### Fixed-income schedules (L1/L2/L3 Fixed Income)
- Day count and accrued interest; full vs flat price; yield compounding frequency; settlement vs coupon dates.
- Duration: modified for option-free bonds, effective for bonds with options; convexity sign; portfolio duration
  weighted by market value.

### LBO / budgets / forecasts
- Debt schedule ties to cash flow; interest on average balances (watch circularity, check the iteration switch);
  IRR and MOIC from dated cash flows; covenant tests.
- Budget vs actual variances, driver logic, seasonality.

## 4. Recompute and reconcile

Recompute the outputs that matter with the engine from the workbook's own inputs, then reconcile:

| Item | Model | Engine (as modelled) | Engine (corrected) | Difference | Cause |
|---|---|---|---|---|---|

- "As modelled" reproduces the workbook's logic, which proves you read it right. "Corrected" applies the fixes. The
  difference column shows each fix's value impact. Apply fixes one at a time when the user needs the attribution.
- Find functions with `python scripts/cfa.py search <words>`, for example `search fcff`, `search ratio`, `search dupont`,
  `search wacc`, `search duration`, `search sharpe`.
- For multi-step work write one short script that imports `cfa_calc`, prints every intermediate, and keep it with
  the deliverable so the numbers can be re-run.

## 5. Improving the workbook

- **Never modify the user's file.** `xlsx_tool.py` refuses to write over its input.
- **Review workbook** (`xlsx_tool.py review FILE --findings findings.json [--metrics metrics.json] --out FILE-review.xlsx`):
  a findings sheet (severity, cell, issue, fix, CFA reference) and a recomputed-metrics sheet. Add `cfa_reference`
  to each finding you write yourself.
- **Corrected copy** (`xlsx_tool.py patch FILE --changes changes.json --out FILE-corrected.xlsx`), when the user wants
  fixes applied: each change is `{"sheet", "cell", "value", "note"}`. The value is a formula that references input
  cells (`=F4*Assumptions!$B$4`), never a pasted number. Each changed cell gets a comment and a row on the
  "CFA Changes" log sheet. Excel recalculates on open. If the tool warns that charts or pivots are not carried over,
  tell the user and prefer the review workbook.
- **New models**: follow model conventions: inputs in labelled cells (blue), formulas black, one assumption per cell,
  formulas consistent along each row, checks row (BS balances, cash ties). For a full build from scratch use the
  `xlsx` skill (recalculation, formatting), `pitch-agent:dcf-model` / `3-statement-model` / `lbo-model`, or
  `aswath-finance` (Damodaran valuation with a live Excel export), and use this skill to review the result.

## 6. Deliverable

1. What the workbook is and how it flows (inputs → calculations → outputs), one paragraph.
2. Findings ranked by severity, each with the cell, the evidence, the fix and the CFA reference.
3. Reconciliation table and the corrected headline numbers (value per share, IRR, ratios), with a sensitivity grid
   for the key drivers.
4. Analyst view: what the numbers say about the business or portfolio, risks, and what would change the conclusion.
5. Files produced (review workbook, corrected copy, the recompute script) with paths.
