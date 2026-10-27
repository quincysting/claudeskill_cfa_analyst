# Playbook: financial reporting and analysis

Use this playbook when a question turns on what a company's statements really say: ratios and DuPont, inventory
methods, capitalization and depreciation, deferred taxes, bonds and leases on the issuer's books, pensions, EPS, cash
flow and free cash flow, earnings quality, equity-method and acquisition accounting, or foreign currency translation.
Every number comes from `cfa_calc.fra` (`python scripts/cfa.py list fra`). The judgement around those numbers comes
from the curriculum frameworks in section 4.

## Contents
1. When to use
2. Inputs to gather
3. Quantitative workflow (engine functions)
4. Qualitative frameworks from the curriculum
5. Interpretation, red flags and exam traps
6. Deliverable outline
7. Knowledge-base paths

## 1. When to use

| Scenario | Typical question |
|---|---|
| Ratio analysis (L1 R28) | "How has liquidity / efficiency / leverage changed?" "Decompose ROE." "Common-size this statement." |
| Inventories (L1 R29, L2 R17) | FIFO vs LIFO vs weighted average; LIFO to FIFO restatement; LIFO liquidation; NRV write-downs |
| Long-lived assets (L1 R30, L2 R18) | Capitalize or expense; capitalized interest; depreciation methods; asset age; impairment |
| Income taxes (L1 R31) | DTA/DTL from carrying amount vs tax base; tax expense; rate change; ETR reconciliation |
| Liabilities and leases (L1 R32, L2 R18) | Bond amortization at the effective rate; finance vs operating lease; capitalizing operating leases |
| Pensions (L2 R20) | Funded status; periodic pension cost under IFRS and US GAAP; analyst adjustments to P&L and cash flow |
| EPS (L1 R25) | Weighted shares; basic and diluted EPS; if-converted; treasury stock method; antidilution |
| Cash flow (L1 R27) | Indirect CFO; FCFF/FCFE; cash flow ratios |
| Earnings quality (L1 R33-R34, L2 R22-R23) | Accruals ratios; Beneish M-score; Altman Z; red flags; cash flow shenanigans |
| Intercorporate (L2 R19) | Equity method income and carrying value; goodwill; full vs partial goodwill and NCI; goodwill impairment |
| Multinational (L2 R21) | Transaction gains; current rate vs temporal exposure; translation adjustment; ratio distortions |

Not covered by the engine: bank and insurer CAMELS-type ratios (state the definitions from the reading and compute
with `financial_ratios` pieces or plain division in a script), share-based compensation fair values (option pricing
lives in `derivatives`), and perpetual-inventory LIFO (the engine is periodic; perpetual LIFO needs the transaction
sequence: walk it in a short script).

## 2. Inputs to gather

- **Statements for at least two periods**: income statement, balance sheet (beginning and ending balances for
  averages), cash flow statement, and the notes (inventory, PP&E, taxes, debt, leases, pensions, segment, EPS).
- **Reporting framework**: IFRS or US GAAP, and the year; several answers differ by framework (section 4.3).
- **Policies and estimates**: inventory method and LIFO reserve, depreciation methods and useful lives, capitalized
  interest and development costs, lease classification, pension discount rate / expected return / salary growth.
- **Tax data**: statutory rate, taxes payable, deferred tax balances, the ETR reconciliation table.
- **Capital data**: weighted shares, dilutive instruments (conversion ratios, coupons, option strikes, average price).
- **Market data** where needed: share price, market value of equity (Altman Z), exchange rates (spot, average,
  historical) for translation.
- Canonical line-item names in `xlsx_tool.py extract` (revenue, cogs, ebit, interest_expense, pretax_income,
  net_income, cash, receivables, inventory, payables, current_assets, current_liabilities, ppe, total_assets,
  total_debt, total_equity, cfo, capex) are the parameter names of `financial_ratios`: pass extracted columns straight in.

## 3. Quantitative workflow (engine functions)

1. **Normalize first, ratios second.** Decide which analyst adjustments apply (step 9) before computing ratios
   you will compare across companies.
2. **Ratio bundle**: `financial_ratios(...)` with ending and `*_begin` balances (averages are used automatically).
   Returns activity (turnovers, DOH/DSO/DPO, cash conversion cycle), liquidity (current, quick, cash, defensive
   interval), solvency (D/A, D/capital, D/E, financial leverage), coverage (interest, fixed charge) and profitability
   (margins, ROA, operating ROA, return on total capital, ROE). Days given directly: `cash_conversion_cycle`.
3. **DuPont**: `dupont_roe(net_income, revenue, total_assets, total_equity, pretax_income, ebit)` for the three-
   and five-way split. Explain the ROE change by the component that moved.
4. **Common-size**: `common_size_statement(items, "revenue")` (income statement) or `"total_assets"` (balance sheet).
5. **Inventories**
   - From cost layers: `inventory_cost_flow(purchases, units_sold, beginning_inventory, sale_price)`.
   - LIFO user vs FIFO peer: `lifo_to_fifo(lifo_inventory, reserve_end, reserve_begin, lifo_cogs, net_income,
     tax_rate)`; adjust equity by reserve x (1 - t) and liabilities (deferred tax) by reserve x t.
   - Falling LIFO reserve: quantify the boost with `lifo_liquidation_profit(layers, replacement_cost, tax_rate)`.
   - NRV / market tests: `inventory_writedown(cost, nrv, standard, replacement_cost, normal_profit)`.
6. **Long-lived assets**
   - Depreciation: `depreciation_schedule(cost, salvage, life, method)` (straight line, DDB with switch,
     SYD, units of production). Changing an estimate: rerun on the remaining book value and life (prospective).
   - Capitalize vs expense: `capitalize_vs_expense(amount, life, tax_rate)`; borrowing costs:
     `capitalized_borrowing_costs`; coverage including capitalized interest: `capitalized_interest_coverage`.
   - Asset base: `fixed_asset_age(gross_ppe, accumulated_depreciation, depreciation_expense)`.
   - Impairment: `asset_impairment(...)` (IFRS one step; US GAAP recoverability then fair value);
     goodwill: `goodwill_impairment(...)`.
7. **Income taxes**
   - Balances: `deferred_tax_items(items, tax_rate)` (kind = asset or liability, carrying amount vs tax base).
   - Expense: `income_tax_expense(taxes_payable, dtl_begin, dtl_end, dta_begin, dta_end, pretax_income)`.
   - Rate change: `tax_rate_change_effect(diff_begin, diff_end, old_rate, new_rate)`.
   - ETR: `effective_tax_rate_reconciliation(pretax_income, statutory_rate, items, reported_tax_expense)`.
8. **Debt, leases, pensions**
   - Bonds: `bond_effective_interest_schedule(face, coupon_rate, market_rate, years, freq)`.
   - Finance lease (lessee): `finance_lease_lessee(payment, rate, periods, payments_in_advance)`.
   - Operating leases as debt: rate from capital leases `lease_implicit_rate(pv, payments)`, then
     `operating_lease_capitalization(payments, rate, total_liabilities, total_equity)`.
   - Pensions: `db_pension_funded_status`; `pension_periodic_cost` (total, IFRS P&L / OCI, US GAAP P&L);
     `pension_obligation_unit_credit` (how a DBO builds); `pension_income_adjustment` (service cost operating,
     interest cost to interest expense, actual return to investment income); `pension_cash_flow_adjustment`
     (excess contributions as debt repayment).
9. **Analyst adjustments** (L1 R35 section 6; L2 R24 case study 2): LIFO to FIFO (step 5), capitalize operating
   leases (step 8), add capitalized interest back to interest (step 6), remove goodwill for tangible ratios,
   pension deficit as debt, reclassify pension components, treat securitized receivables as debt (add back to CFO
   the cash pulled forward). Recompute `financial_ratios` on the adjusted numbers.
10. **EPS**: `weighted_average_shares(beginning, changes, split_factor)` then
    `eps_basic_diluted(net_income, shares, preferred_dividends, convertible_preferred, convertible_debt, options,
    average_market_price, tax_rate)`. Antidilutive instruments are excluded and listed.
11. **Cash flow**: `cfo_indirect_method(...)`; `free_cash_flow_fcff_fcfe(tax_rate, fcinv, wcinv, interest,
    net_borrowing, cfo | net_income | ebit | ebitda)`; `cash_flow_ratios(cfo, ...)`.
12. **Earnings quality**: `accruals_ratios(...)` (balance sheet and cash flow versions, two periods),
    `beneish_m_score(current, prior)`, `altman_z_score(...)`. Trend CFO / net income alongside.
13. **Intercorporate**: `equity_method_investment(price, ownership, investee NI, dividends, book value,
    fair_value_adjustments, unrealized_profit)`; `acquisition_goodwill(price, ownership, fv_net_assets,
    fv_subsidiary)`.
14. **Multinational**: `fx_transaction_gain_loss`; exposure and sign with `fx_translation_exposure`; the amount with
    `fx_translation_adjustment(beginning_exposure, rate_begin, rate_end, flows)` (current rate: CTA in OCI;
    temporal: remeasurement gain/loss in net income).
15. For multi-step work write one script that imports `cfa_calc.fra` and prints every intermediate value; report
    rounded figures and say what was rounded.

## 4. Qualitative frameworks from the curriculum

### 4.1 Reporting quality and the quality-of-earnings spectrum
- **Two dimensions** (L2 R23 Evaluating Financial Reporting Quality): reporting quality (does it faithfully
  represent the economics?) and earnings quality (are the earnings sustainable and adequate?). Accruals are where
  discretion lives: the accrual component of earnings is less persistent than the cash component and mean-reverts
  faster (R23 sections 2-3), so high accruals ratios predict lower future earnings.
- **Spectrum** (the framing used in later L2 curricula, consistent with R23): GAAP, decision-useful, sustainable
  and adequate returns -> GAAP but unsustainable or inadequate -> biased accounting choices (aggressive or
  conservative) -> biased choices that manage earnings -> departures from GAAP (non-compliant) -> fictitious
  transactions. Conservative bias is still bias: "cookie jar" reserves reverse into future earnings.
- **Why managers bias** (R23 2.3; L1 R33 fraud triangle): capital-market incentives (meet consensus, smooth
  earnings, equity issuance, compensation), contracts (debt covenants, bonuses), and opportunity (weak audit
  committee, complex transactions, related parties), plus rationalization. Disciplining mechanisms: auditors,
  regulators, boards, short sellers, analysts (R23 2.4).

### 4.2 Red flags and accounting shenanigans
- **Revenue** (L1 R33; L2 R23 4.1): recognition before delivery or acceptance, bill-and-hold, channel stuffing,
  barter or round-trip sales, gross vs net, nonoperating gains in revenue; DSO rising, receivables growing faster
  than sales, revenue growth out of line with peers, fourth-quarter surprises.
- **Expenses** (R23 4.2): capitalizing operating costs (WorldCom), long useful lives, low bad-debt or warranty
  reserves, recurring "nonrecurring" restructuring charges, aggressive pension assumptions (high discount rate,
  high expected return), inventory growing faster than sales (DOH rising).
- **Balance sheet** (R23 4.3): off-balance-sheet debt (operating leases, SPEs/VIEs, take-or-pay, guarantees),
  goodwill and intangibles never impaired, equity-method entities that hide leverage (Enron, L1 R33).
- **Cash flow statement** (L1 R34 Accounting Shenanigans on the Cash Flow Statement; R23 4.4): stretching
  payables, supplier financing classed as operating, securitizing receivables, tax benefits from stock options in
  CFO, buybacks to offset dilution (true employee cost in financing), capitalizing costs into investing, CFO lagging
  net income for several years. CFO can be managed too: "dispel the myth about cash flows".
- **Lessons** (L2 R22 The Lessons We Learn): understand what you are looking at; read the fine print; if it is too
  good to be true, it may be; follow the money; understand the risks.

### 4.3 IFRS vs US GAAP differences that move numbers (2013-2014 curriculum)
| Area | IFRS | US GAAP |
|---|---|---|
| Inventory | Lower of cost and NRV; reversals allowed; LIFO prohibited | LIFO allowed; lower of cost or market (2014 text); no reversals |
| PP&E / intangibles | Revaluation model allowed; component depreciation required | Cost model only |
| Development costs | Capitalized once criteria met | Expensed (except software) |
| Borrowing costs | Net of investment income on borrowed funds | Gross |
| Impairment | One step: CA vs recoverable (higher of FVLCS and VIU); reversals allowed (not goodwill) | Recoverability test with undiscounted CF, then fair value; no reversals; goodwill two-step (2014) |
| Leases (pre-IFRS 16/ASC 842) | Substance-based finance lease | Four bright-line tests; operating leases off balance sheet |
| Pensions | P&L: service cost + net interest at discount rate; remeasurements in OCI, never recycled | P&L: service + interest - expected return + amortizations; corridor |
| Interest / dividends in cash flow | Interest paid: CFO or CFF; dividends paid: CFO or CFF | Interest paid: CFO; dividends paid: CFF |
| Goodwill / NCI | Full or partial goodwill | Full goodwill only |
| Extraordinary items | Prohibited | Allowed (pre-2015) |
| Deferred tax | Net non-current | Current / non-current split (pre-2015) |

### 4.4 Analyst adjustments framework (L1 R35 section 6; L2 R24 Integration case studies)
Follow the six-phase framework (L1 R22; L2 R24): purpose, collect data, process (adjust), analyze, conclude and
communicate, follow up. Adjust for: inventory method (LIFO reserve), PP&E (useful lives, impairments,
capitalization policy), goodwill (exclude for tangible ratios), off-balance-sheet financing (leases, take-or-pay,
securitizations), pensions (deficit as debt, reclassify components), equity-method vs consolidated presentation,
and translation method. Adjust both companies consistently, then compare.

### 4.5 Multinational translation rules (L2 R21)
- Functional currency = local -> current rate method: assets and liabilities at the current rate, income at the
  average rate, equity at historical rates; exposure = net assets; CTA in OCI; most ratios preserved (turnovers mix
  current and average rates).
- Functional currency = parent's (or highly inflationary under US GAAP) -> temporal method: monetary items at the
  current rate, nonmonetary at historical rates (COGS and depreciation follow), exposure = net monetary assets;
  remeasurement gain/loss in net income; all ratios distorted.
- IFRS highly inflationary: restate for inflation, then translate at the current rate.

## 5. Interpretation, red flags and exam traps

- **Averages**: turnovers and returns use average balances when both years are given; DSO uses revenue, DOH uses
  COGS; days payable uses purchases if given (COGS otherwise). State which you used.
- **DuPont signs**: interest burden and tax burden are ratios below 1; "higher burden" means a lower ratio.
- **LIFO in rising prices**: lower inventory and higher COGS -> higher turnover, lower margins, lower current ratio,
  higher D/E than FIFO; lower taxes and higher CFO. A falling reserve hints at liquidation and an unsustainable boost.
- **Capitalizing** raises current NI, CFO and assets; lowers CFI; equal totals over the life; higher early ROA/ROE
  is followed by lower later ratios when spending stops.
- **Accelerated depreciation** lowers early NI and book value, raises later; total depreciation is unchanged.
  DDB ignores salvage in the rate but never goes below it.
- **Tax**: asset CA > TB or liability CA < TB -> DTL; a rate cut reduces DTLs (gain) and DTAs (loss). A DTL that
  is not expected to reverse is closer to equity.
- **Bonds**: interest expense = carrying amount x market rate at issue, not the coupon; discount amortization adds to
  CFO under the indirect method.
- **Leases**: finance lease raises total expense early, lowers it later, raises CFO (principal is CFF), raises D/E.
- **Pensions**: total periodic cost is the same under IFRS and US GAAP; only the P&L/OCI split differs. A higher
  discount rate lowers the obligation; a higher expected return (US GAAP) lowers P&L cost only.
- **EPS**: stock splits and stock dividends are retroactive; options are dilutive only if average price > strike;
  test each instrument for antidilution; convertible debt add-back is after tax.
- **FCFF from CFO** must not subtract WCInv again (already in CFO); FCFE adds net borrowing.
- **Accruals**: compare scaled ratios across companies, never the raw aggregate accruals.
- **Translation**: exposure sign x currency direction gives the sign of the result; current rate gain goes to OCI,
  temporal to NI. Receivables turnover is the only Canadaco ratio the method did not change (R21).

## 6. Deliverable outline

1. **Question and conclusion** in two sentences (e.g., "Adjusted for LIFO and leases, A is more levered than B").
2. **Data and adjustments**: sources, period, framework; each adjustment with its engine function and amount.
3. **Ratio table**: reported vs adjusted, two or more periods and peers; DuPont bridge for ROE changes.
4. **Quality assessment**: accruals ratios, CFO / NI trend, M-score / Z-score, red flags found (cite the note).
5. **Framework effects**: IFRS vs US GAAP or method choices that drive differences.
6. **Implications**: valuation, credit, covenant headroom, forecast adjustments; what to monitor next.
7. **Appendix**: the script and its printed intermediate values; rounding stated.

## 7. Knowledge-base paths (E:\CLAUDE\CFA\MD)

- L1 curriculum: `Level-1/04-Financial-Reporting/Curriculum-2014/` R22 (framework), R25 income statement and EPS
  (`SS08-R25-understanding-income-statements.md`, EPS examples p204-211), R27 cash flow (p311 indirect, p322 FCFF/FCFE,
  p323 ratios), R28 ratios (`SS08-R28-financial-analysis-techniques.md`, p360 activity, p384 five-way DuPont, p393
  Altman), R29 inventories (p416 cost flows, p421 write-downs), R30 long-lived assets (p448, p454, p466), R31 taxes
  (p495, p502, p504), R32 liabilities and leases (p535-536 bonds, p550 finance lease), R33 red flags
  (`SS10-R33-financial-reporting-quality-red-flags-and-accounting-warning-signs.md`, p589-590), R34 cash flow
  shenanigans (`SS10-R34-accounting-shenanigans-on-the-cash-flow-statement.md`), R35 applications (p663 asset age).
- Kaplan notes: `Level-1/04-Financial-Reporting/Kaplan-Notes-2014/SS08-R28-financial-analysis-techniques.md`
  (p158-166 worked ratio and DuPont examples).
- L2 curriculum: `Level-2/04-Financial-Reporting/Curriculum-2014/` R17 inventories (p29 CAT LIFO to FIFO, p37 LIFO
  liquidation), R18 long-lived assets and leases (p69, p74, p101, p107-109), R19 intercorporate (p144-168),
  R20 pensions (p198-218, practice problems p229-234), R21 multinational (p238, p270-277), R22 lessons, R23
  reporting quality (p348-351 accruals), R24 integration case studies.
- Tests with every golden value: `tests/test_fra.py`.
