# Playbook: alternative investments

Use this playbook for real estate (direct and REITs), private equity and venture capital, hedge funds and funds of
funds, and commodity futures, from L1 fee arithmetic to L3 due diligence and portfolio roles. Every number comes from
`cfa_calc.alternatives` (`python scripts/cfa.py list alternatives`), plus the DDM functions in `cfa_calc.equity` for
REIT dividend models. The judgement comes from the curriculum frameworks in section 4.

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
| Property valuation | "What is this building worth?": NOI, direct capitalization, DCF with a terminal cap rate, UK term-and-reversion or layer method, GIM |
| Property financing | Maximum loan under LTV and DSCR, cash-on-cash return, leveraged vs unleveraged IRR |
| REITs | NAV per share, FFO/AFFO, P/FFO and P/AFFO value, two-step DDM, premium/discount to NAV |
| Private equity funds | Carried interest under deal-by-deal or whole-of-fund waterfalls with hurdle, catch-up and clawback; the fund NAV table; DPI/RVPI/TVPI; gross and net IRR |
| Venture capital deals | Post/pre-money, ownership fraction, shares and price per round, failure risk, dilution across rounds |
| Hedge funds | Management and incentive fees (gross or net of management fee, hard or soft hurdle, high-water mark), net returns, fund-of-funds layering |
| Commodities | Total return = spot + roll + collateral, roll return in price units, backwardation and contango, basis |
| Smoothed data | Unsmooth appraisal-based (NCREIF, PE NAV) returns before using volatility or correlations |

Not here: commodity forward pricing and swaps (derivatives), GIPS treatment of PE performance (portfolio/ethics),
private company equity valuation (equity playbook, section G).

## 2. Inputs to gather

- **Real estate.** Rent roll or potential gross income, other income, vacancy and collection loss, operating expenses
  (fixed and as % of EGI, management fee), lease terms and review dates, ERV, going-in and terminal cap rates from
  comparables, discount rate, holding period, NOI in the year after sale, selling costs. Financing: appraised
  value, LTV and DSCR limits, rate, interest-only or amortizing, term.
- **REITs.** Last-12-month NOI, straight-line rent, acquisitions during the year, NOI growth, market cap rate, other
  tangible assets (exclude goodwill and intangibles), market value of debt, other liabilities (exclude deferred
  taxes), shares; net income, real estate D&A, deferred tax, gains and losses on sales, recurring capex and leasing
  costs; peer P/FFO and P/AFFO multiples; dividend forecasts, beta, Rf, ERP.
- **Private equity fund.** Committed capital, calls by year, operating results (realized + unrealized), distributions,
  management fee rate and basis (committed, paid-in or NAV), carry rate, hurdle, catch-up, waterfall type, clawback
  terms; for each deal: invested, proceeds, holding period.
- **Venture capital.** Exit value (or exit metric x multiple) and timing, target return, investment per round, timing
  of rounds, founders' shares, probability of failure, expected dilution (retention ratio), option pool.
- **Hedge fund.** Beginning capital (net of past fees), gross ending value, fee rates, the basis of the management fee
  (beginning or end AUM), whether the incentive is net of the management fee, hurdle (hard or soft), high-water mark;
  for a FoF, both fee layers.
- **Commodities.** Spot change, the same contract's futures prices at both dates, near and next contract prices at
  the roll, T-bill rate, index weights and rebalancing.

## 3. Quantitative workflow

Run each step through the engine and print the intermediate values (conventions.md). Round only when presenting.

**A. Private real estate (L2 R40)**
1. `real_estate_noi`: PGI → vacancy → EGI → operating expenses → NOI (before debt service, depreciation and income tax).
2. `real_estate_direct_capitalization`: value = NOI1 / cap rate. With only a discount rate and growth, cap = r - g.
   Divide rent by the all risks yield for UK fully let property.
3. `real_estate_dcf_value`: PV of NOI for the holding period plus PV of resale = NOI_(n+1) / terminal cap rate, net of
   selling costs. Report the implied going-in cap rate.
4. UK leases: `real_estate_term_and_reversion` (term rent at a lower rate, ERV capitalized at the ARY and deferred)
   or `real_estate_layer_method` (contract rent in perpetuity plus the deferred top slice).
5. Quick screen: `real_estate_gross_income_multiplier`, which ignores vacancy and expenses, so use it only as a cross-check.
6. Financing: `real_estate_loan_sizing` (the lower of the LTV and DSCR amounts binds), then
   `real_estate_investment_returns` for the equity dividend rate (cash-on-cash), DSCR, LTV, leveraged and
   unleveraged IRR. Positive leverage requires the unleveraged IRR to exceed the cost of debt.
7. Smoothed index data: `unsmooth_appraisal_returns` before you use volatility or correlation in an allocation.

**B. REITs (L2 R41)**
1. `reit_ffo_affo`: FFO = NI + real estate D&A + deferred tax - gains + losses; AFFO = FFO - straight-line rent -
   recurring capex and leasing costs; per share, plus payout on FFO and on AFFO.
2. `reit_nav_per_share`: capitalize next-12-month cash NOI at the market cap rate, add other tangible assets,
   subtract debt and other liabilities. Pass `price` to get P/NAV.
3. `reit_relative_value`: peer P/FFO and P/AFFO multiples x forecast FFO/AFFO per share, and the REIT's own multiples
   at the market price.
4. DDM: `ddm_multistage_value` (equity module) with explicit dividend forecasts and terminal growth; required return
   from `equity_required_return_capm`. Triangulate NAV, multiples and DDM.

**C. Private equity (L2 R42; L1 R63)**
1. Carry with dated cash flows (the usual LPA case: contributions and distributions on dates, hurdle accruing on unreturned capital plus unpaid pref): `pe_waterfall_cash_flows`, which also gives LP net IRR, DPI/RVPI/TVPI and, with `deal_by_deal_gp_carry`, the clawback. Use it whenever early exits return capital before later deals finish: per-deal hurdle accrual (below) then overstates the hurdle.
   Carry by deal (curriculum examples, holding periods only): `pe_distribution_waterfall`. Tiers: return of capital → preferred return (compound hurdle) → GP catch-up
   (`catch_up` = GP share of that tier; 0 = hard hurdle) → carry split. `method="deal_by_deal"` reports the clawback
   against the whole-fund entitlement. `method="whole_of_fund"` with `whole_fund_basis="committed"` (all committed
   capital back first, curriculum alternative 1), `"invested"`, or `"threshold"` (carry once value ≥ 120% of invested,
   alternative 2).
2. Fund accounting: `pe_fund_nav_table` gives paid-in, management fees, NAV before distributions, carry (starts when
   NAV before distributions passes committed capital), NAV after, DPI, RVPI, TVPI, and gross and net IRR in the
   curriculum's convention.
3. Multiples from summary data: `pe_fund_performance_multiples` (DPI, RVPI, TVPI, PIC).
4. IRR of any fund or deal cash flows: `pe_fund_irr` (add `terminal_nav` for a since-inception IRR; set
   `periods_per_year` for quarterly flows).
5. VC method: `vc_method_single_round` (POST = V/(1+r)^t, PRE, F = I/POST, new shares = x F/(1 - F), price). Pass
   `exit_metric` and `exit_multiple` for the terminal value, `prob_failure` (adjusted r = (1+r)/(1-q) - 1) and
   `retention_ratio` for expected dilution. Several rounds: `vc_method_multiple_rounds`, working back from the exit
   with the compound discount factor of each interval. Also run a scenario-weighted exit value at a lower rate.

**D. Hedge funds (L1 R63/R66; L2 R43)**
1. `hedge_fund_fees` for one period. Chain years by passing `ending_value_net` as the next `beginning_value` and
   `high_water_mark_end` as the next `high_water_mark`. Options: `management_fee_basis`,
   `incentive_net_of_management_fee`, `hurdle_rate` with `hurdle_type` (hard: only the excess; soft: the whole gain once
   beaten).
2. `fund_of_hedge_funds_net_return`: two fee layers; report the fee-layering cost.
3. Multi-year net performance: arithmetic and geometric means of the chained net returns (quant module), plus
   total fees paid.

**E. Commodities (L2 R44; L3 R26)**
1. `commodity_futures_total_return`: spot + roll + collateral (+ rebalancing); the excess return is spot + roll.
2. `commodity_roll_return_monetary`: change in the futures price minus change in the spot price (L3 exhibits).
3. `commodity_term_structure`: roll yield (F_near - F_far)/F_near, the shape of the curve, and the basis.

## 4. Qualitative frameworks from the curriculum

**Due diligence: the eight checkpoints (L3 R26 Example 2, UVIMCO).** (1) Market opportunity: what inefficiency, and
why does it exist? (2) Investment process: who does it best, and what is their edge? (3) Organization: staffing,
stability, compensation, turnover, succession. (4) People: experience, integrity, references, background searches.
(5) Terms and structure: fees, alignment, capacity, fit of the vehicle to the opportunity. (6) Service providers:
lawyers, auditors, prime brokers, lenders. (7) Documents: read the PPM and the audits. (8) Write-up: a formal
recommendation. For private wealth clients add taxes, suitability, communication, decision risk and concentrated
positions.
- **Hedge funds (L1 R63 §3.5; L3 R26 §6.3.3).** Strategy and process, competitive advantage, track record (how
  returns are calculated, before or after fees), size and age, management style, key-person risk, reputation,
  investor relations, plans for growth, risk management systems, leverage, counterparty and prime-broker risk, an
  independent auditor, and valuation policy (bid/ask vs mid, liquidity haircuts, trading vs reporting NAV).
  Redemption terms: lockup, notice period, gates. A short notice period plus illiquid holdings invites forced
  liquidation (L1 R63 Example 4).
- **Private equity (L2 R42 §3.3; L3 R26 §4.3.3).** Persistence of GP returns (top-quartile funds stay on top), the
  wide spread between funds, and illiquidity. Review fund terms (management fee, transaction fees, carry, ratchet,
  hurdle, key-man clause, clawback, waterfall, no-fault divorce, co-investment rights) and the valuation policy.
  At company level: market prospects, management, operations, and the financial and legal review.
- **Real estate (L2 R40 §9, Example 26; approaches §§5-8).** Due diligence uncovers legal, environmental, physical and title
  problems the seller did not disclose: lease review, rent roll and cash flow history, operating expenses,
  environmental and physical surveys, zoning, property tax, title and liens. Use three approaches: income, cost and
  sales comparison; reconcile them.
- **REITs (L2 R41 §3.4).** Remaining lease terms, inflation protection in leases, in-place vs market rents, costs of
  re-leasing, tenant concentration and credit, new competitive supply, balance sheet (leverage, cost of debt,
  maturities), and management.

**Liquidity, valuation lag and smoothing (L2 R40 §11, L3 R26 §§3.2.3, 4.2.3).** Appraisal-based indices (NCREIF,
IPD) and GP-reported PE NAVs lag and smooth true values. They understate volatility and correlation, and so overstate
diversification and Sharpe ratios. Unsmooth, or use transaction-based or REIT-based proxies. IRR assumes
liquidity the fund does not have; interim NAVs depend on GP valuation policy and are staler in down markets. Hedge
fund indices suffer from self-selection, backfill (instant history) and survivorship bias (L2 R43); returns are
non-normal (negative skew, fat tails); and capacity limits erode alpha.

**Fees.** "2 and 20" for hedge funds, "1 and 10" on top for FoFs, PE management fees of 1.5-2.5% of committed
capital and 20% carry. Fee drag explains much of the gap between gross and net returns and the J-curve in early PE
fund years. The incentive fee is an option on fund performance and can reward risk-taking; a high-water mark stops
investors paying twice for the same gains, and a clawback restores the agreed profit split.

**Role in a portfolio (L3 R26).**
- Direct real estate: income and diversification, a partial inflation hedge; lumpy, illiquid, with high
  transaction costs. REITs are levered and equity-like in the short run.
- Private equity: a growth (return-enhancing) role with an illiquidity premium; J-curve; commit by vintage year and
  diversify by stage, strategy and GP.
- Commodities: an inflation hedge (spot is the most inflation-sensitive part), diversification from equities and
  bonds, positive event risk; returns depend heavily on roll yield and collateral yield.
- Hedge funds: diversification or alpha depending on strategy; check the factor (beta) exposures behind "alpha".
- Managed futures and distressed securities: trend and crisis diversification; event-driven, illiquid exposure.
- Fit to the investor: time horizon, liquidity needs, size (fund minimums, due diligence costs), taxes, governance capacity.

**Commodity term structure (L2 R44; L3 R26 §5.2).** Backwardation (futures below spot: hedging pressure,
convenience yield, producers' real options) gives positive roll yield to longs; contango (storage and financing
costs) gives negative roll. Index construction (weights, roll schedule, rebalancing) changes the return a lot.

## 5. Interpretation, red flags and exam traps

- **Cap rate vs discount rate.** Cap = r - g only for constant growth; the terminal cap applies to NOI in year
  n + 1; a lower going-in cap than terminal cap means growth is expected or rents are below market.
- **NOI excludes** debt service, depreciation and income tax; vacancy applies to PGI incl. other income (curriculum
  Example 12).
- **Loan sizing.** Compute both LTV and DSCR amounts; the smaller binds. Interest-only: loan = (NOI/DSCR)/rate.
- **Cash-on-cash is not IRR.** It ignores appreciation and the sale. Leverage raises the IRR only if the property's
  return exceeds the cost of debt.
- **FFO vs AFFO.** AFFO is the better measure of economic income (it deducts recurring capex and straight-line rent)
  but is less standardized. NAV ignores going-concern value and management skill; P/NAV premiums and discounts persist.
- **Waterfalls.** Deal-by-deal pays the GP earlier and needs a clawback; whole-of-fund protects LPs. The hurdle is
  compound, on capital over time. With full catch-up and a large enough profit, the GP ends with exactly the carry
  % of all profit; without catch-up, carry applies only above the preferred return.
- **Fund NAV table.** Fees on paid-in vs committed capital change everything; carry starts only when NAV before
  distributions exceeds committed capital (total return method, alternative 1). DPI is realized, RVPI unrealized
  (GP-valued); TVPI = DPI + RVPI. Interpret by vintage and against peer quartiles.
- **VC method.** Use the exit value in a success scenario with a high rate, or the expected exit value with a lower
  rate; do not discount the expected value at a "VC" rate (double counting). Ignoring future dilution overpays
  (use the retention ratio). F ≥ 100% means a negative-NPV deal.
- **Hedge fund fees.** Read the basis: end vs beginning AUM; incentive gross vs net of the management fee; hard vs
  soft hurdle; the HWM is measured on net-of-fee value. No incentive fee is earned below the HWM even in a positive
  year. Gross returns overstate investor returns; index returns are net of fees but biased upward.
- **Fund of funds.** Two fee layers; check that "diversification" is not just double fees on correlated strategies.
- **Roll yield sign.** Backwardation → positive roll for long positions. R44 Eq. 6 expresses (F_new - F_old)/F_old,
  whose negative value is a positive roll return.
- **Smoothing bias.** Low reported volatility and correlation are partly artifacts; unsmoothed NCREIF volatility is
  more than double the smoothed figure (L3 R26: 3.37% → 8.95%).

## 6. Deliverable outline

1. **Recommendation.** Invest / pass / allocation size, and the vehicle (direct, fund, FoF, REIT, futures index).
2. **Opportunity and role.** Why this asset class or strategy, its expected role (return, income, diversification,
   inflation hedge), and liquidity fit.
3. **Valuation or return analysis.** Engine outputs: NOI and value by two approaches; or NAV/FFO/AFFO/DDM; or the
   fund table with DPI/RVPI/TVPI and net IRR; or the deal's post/pre-money and ownership; or net-of-fee returns.
4. **Fees and terms.** Fee drag, waterfall type, hurdle, catch-up, HWM, clawback, lockups and gates.
5. **Due diligence findings.** The eight checkpoints, the asset-specific items, open questions.
6. **Risks.** Liquidity, leverage, valuation lag and smoothing, key person, concentration, data biases, scenarios.
7. **Appendix.** Engine calls and full-precision outputs; state the rounding.

## 7. Knowledge-base paths (E:\CLAUDE\CFA\MD)

| Topic | Path |
|---|---|
| L1 overview: hedge funds, PE, real estate, commodities, fees | `Level-1/10-Alternative-Investments/Curriculum-2014/SS18-R63-introduction-to-alternative-investments.md` |
| L1 (2013 edition, full AWJ fee solutions) | `Level-1/10-Alternative-Investments/Curriculum-2013/SS18-R66-introduction-to-alternative-investments.md` |
| L1 investing in commodities (2013) | `Level-1/10-Alternative-Investments/Curriculum-2013/SS18-R67-investing-in-commodities.md` |
| Private real estate | `Level-2/10-Alternative-Investments/Curriculum-2014/SS13-R40-private-real-estate-investments.md` |
| REITs and REOCs | `Level-2/10-Alternative-Investments/Curriculum-2014/SS13-R41-publicly-traded-real-estate-securities.md` |
| Private equity valuation and VC method | `Level-2/10-Alternative-Investments/Curriculum-2014/SS13-R42-private-equity-valuation.md` |
| Hedge funds survey (biases, factor models) | `Level-2/10-Alternative-Investments/Curriculum-2014/SS13-R43-investing-in-hedge-funds-a-survey.md` |
| Commodity investing primer | `Level-2/10-Alternative-Investments/Curriculum-2014/SS13-R44-a-primer-on-commodity-investing.md` |
| L3 alternatives portfolio management | `Level-3/09-Alternative-Investments/Curriculum-2014/SS13-R26-alternative-investments-portfolio-management.md` |
| L3 commodity forwards and swaps (2013) | `Level-3/09-Alternative-Investments/Curriculum-2013/SS13-R32-swaps.md`, `SS13-R33-commodity-forwards-and-futures.md` |

Kaplan/Schweser L1 summaries: `Level-1/10-Alternative-Investments/Kaplan-Notes-2014/`, `Schweser-Notes-2013/`.
Do not open `Level-1/Exams/`.
