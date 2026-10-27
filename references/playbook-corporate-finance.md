# Playbook: corporate finance

Use this playbook when a question is about how a company invests, finances and pays out: project evaluation and
capital budgeting, cost of capital, operating and financial leverage, capital structure, dividends and buybacks,
working capital and short-term financing, corporate governance, and mergers and acquisitions. Every number comes
from `cfa_calc.corpfin` (`python scripts/cfa.py list corporate_finance`); ratio inputs come from `cfa_calc.fra`.
The judgement around those numbers comes from the curriculum frameworks in section 4.

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
| Project decision (L1 R36, L2 R25) | "Should we invest?" NPV, IRR, MIRR, payback, PI; expansion or replacement cash flows; MACRS |
| Ranking and constraints (L1 R36, L2 R25) | NPV vs IRR conflicts, crossover rate, unequal lives (EAA, LCM chains), capital rationing |
| Risk and flexibility (L2 R25) | Real options (timing, abandonment, expansion, flexibility), inflation consistency |
| Hurdle rate (L1 R37) | WACC; cost of debt, preferred and equity; pure-play beta; country risk; MCC break points; flotation |
| Risk of the income stream (L1 R38) | DOL, DFL, DTL; operating and total breakeven |
| Financing mix (L2 R26) | MM with and without taxes, static trade-off, target structure, rating effects |
| Payout (L1 R39, L2 R27) | Tax systems, buyback EPS/BVPS math, Lintner target payout, dividend safety and FCFE coverage |
| Liquidity (L1 R40) | Cash conversion cycle, cost of trade credit, short-term borrowing cost, money market yields |
| Governance (L1 R41, L2 R28) | Board independence, committees, shareowner rights, conflicts of interest (qualitative) |
| M&A (L2 R29) | HHI test, bootstrapping, exchange ratios, gains to target and acquirer, DCF and comparables value |

## 2. Inputs to gather

- **Project data**: initial fixed capital and NWC, sales, cash operating costs, depreciation schedule (method, tax
  class), tax rate, life, terminal salvage and book value, old-asset sale price and book value (replacement), the
  required return, any real options and their cost.
- **Capital structure**: target weights (market values, not book), before-tax cost of debt (bond price, coupon,
  maturity; or the yield on comparably rated debt), preferred dividend and price, share price, dividends, growth,
  beta (or comparables' betas, D/E and tax rates), risk-free rate, equity risk premium, sovereign spreads and
  volatilities for country risk, financing tiers for MCC, flotation cost.
- **Operating structure**: unit price, variable cost, fixed cost, interest, units.
- **Payout**: earnings, dividends, buybacks, CFO, capex, net borrowing, tax regime and investor tax rates.
- **Working capital**: days of inventory, receivables and payables; trade terms; borrowing quotes and fees.
- **M&A**: pre-deal prices and shares of both firms, synergy estimate, consideration (cash per share, exchange
  ratio), market shares in the relevant market, comparable companies or transactions, target FCF forecast and WACC.
- **Governance documents**: proxy statement, board and committee composition, bylaws, takeover defenses.

## 3. Quantitative workflow (engine functions)

1. **Build cash flows, not earnings.** `project_cash_flows(fc_inv, nwc_inv, sales, cash_opex, depreciation,
   tax_rate, life, salvage_terminal, book_value_terminal, salvage_initial, book_value_initial, rate)` returns the
   outlay, annual after-tax OCF, terminal non-operating cash flow, the vector, NPV, IRR and the PV of depreciation
   tax savings. Accelerated depreciation: `macrs_depreciation(base, class)` and pass the list. Replacement: pass
   incremental sales, costs and depreciation plus the old asset's sale price and book value.
2. **Decision metrics**: `capital_project_metrics(cash_flows, rate, reinvestment_rate, shares_outstanding)` gives
   NPV, IRR (all roots if multiple), MIRR, PI, payback, discounted payback and NPV per share.
   Accounting return if asked: `average_accounting_rate_of_return`.
3. **Mutually exclusive projects**: `npv_crossover_rate(cf_a, cf_b)`; unequal lives with replacement:
   `unequal_lives_comparison({name: cash_flows}, rate)` (EAA and LCM chain agree); fixed budget:
   `capital_rationing(projects, budget)`.
4. **Inflation**: discount nominal flows at nominal rates; convert with `project_real_nominal_rate`.
5. **Real options**: `real_option_adjusted_npv(npv_dcf, option_value, option_cost)`; abandonment from scenarios:
   `abandonment_option_npv(outlay, scenarios, abandonment_value, year, rate)`. Option pricing (binomial,
   Black-Scholes) lives in `derivatives`.
6. **Cost of capital**
   - Debt: `cost_of_debt_after_tax(tax_rate, price, face, coupon_rate, years, freq)` or `comparable_yield=`.
   - Preferred: `cost_of_preferred_stock(dividend, price)`.
   - Equity: `cost_of_equity_capm(rf, beta, erp, country_risk_premium)`, `cost_of_equity_ddm(price, g, d1 | d0,
     retention, roe, flotation)`, `cost_of_equity_bond_yield_plus(yield, premium)`; triangulate the three.
   - Beta for a project or private firm: `beta_pure_play(target_de, target_tax, comparables=[...])`.
   - Country risk: `country_risk_premium(sovereign_yield, developed_yield, equity_vol, bond_vol)`.
   - WACC: `weighted_average_cost_of_capital(rd, re, tax, weight_debt | debt_value+equity_value | de_ratio)`.
   - Growing budgets: `marginal_cost_of_capital_schedule(weights, tiers)` and accept projects whose IRR exceeds
     the MCC of the tranche that funds them.
   - Flotation costs: `npv_with_flotation_costs(cash_flows, rate, flotation_cost, tax_rate, deductible)`
     (preferred over inflating the cost of equity).
7. **Leverage**: `degree_of_leverage(units, price, variable_cost, fixed_cost, interest)` for DOL, DFL, DTL and
   breakeven units and revenue.
8. **Capital structure**: `mm_firm_value(ebit, r0, debt, rd, tax_rate, pv_financial_distress_costs)` for VU, VL,
   cost of equity and WACC (static trade-off when distress costs are given); `mm_cost_of_equity(r0, rd, de, t)`.
   Scan several debt levels to locate the WACC minimum (L1 R37 Example 13, L2 R26 Example 4).
9. **Payout**: `dividend_effective_tax_rate(system, tc, ti, tc_distributed)`;
   `share_repurchase_effects(shares, ni, price, amount, after_tax_cost_of_funds, book_value_equity)`;
   `lintner_target_dividend(last_div, last_eps, expected_eps, target_payout, adjustment_years)`;
   `dividend_coverage(ni, dividends, cfo, fcinv, net_borrowing, buybacks)`.
10. **Working capital**: `cash_conversion_cycle` (in `fra`; or `financial_ratios` from statements);
    `cost_of_trade_credit(discount, discount_days, payment_day)`;
    `short_term_borrowing_cost(amount, rate, term, discount_basis, commitment/dealer/backup fees)`;
    `short_term_investment_yields(face, days, price | discount_rate)`.
11. **M&A**: `hhi_merger_analysis(shares, merging)`; `merger_bootstrap_eps(...)`;
    `merger_bid_evaluation(acquirer_value, acquirer_shares, target_value, target_shares, synergies,
    cash_per_target_share, exchange_ratio)`; target value by `ma_dcf_target_value(fcfs, wacc, g | multiple)` and
    `ma_comparables_valuation(target_metrics, comparables | multiples, weights, takeover_premiums)`. Bid range:
    from the target's pre-deal value (minimum) to pre-deal value plus synergies (maximum).
12. For multi-step work write one script that imports `cfa_calc.corpfin` (and `fra`), prints every intermediate
    value, and states rounding.

## 4. Qualitative frameworks from the curriculum

### 4.1 Capital allocation discipline (L1 R36 section 2; L2 R25 sections 2, 7.6)
- Process: idea generation -> analysis -> planning the budget -> monitoring and post-audit.
- Principles: decisions on after-tax incremental cash flows; timing matters; opportunity costs included; sunk
  costs and financing costs excluded (financing is in the discount rate); externalities and cannibalization counted.
- Project types: replacement (maintain), replacement (cost reduction), expansion, new products and markets,
  mandatory (regulatory, safety), other (pet projects, R&D).
- Rules: NPV governs; IRR's reinvestment assumption and scale/timing differences cause ranking conflicts; multiple
  or no IRR with non-conventional flows; payback is liquidity, not value.
- Pitfalls (L2 R25 Table 27): ignoring economic responses, template misuse, pet projects, deciding on EPS/NI/ROE,
  using IRR, bad cash flow accounting, overhead costs, wrong risk-adjusted rate, spending the budget because it is
  available, failing to consider investment alternatives, handling sunk and opportunity costs incorrectly.

### 4.2 Corporate governance (L1 R41 The Corporate Governance of Listed Companies; L2 R28 Corporate Governance)
- Objectives: mitigate conflicts between managers, directors and shareholders; ensure assets are used in
  shareholders' interest. Agency problems: manager-shareholder (perquisites, empire building, entrenchment,
  short-termism) and director-shareholder (lack of independence, interlocks, related parties).
- Board evaluation checklist (L2 R28 section 5): majority independent; independent chair or lead director;
  qualified and engaged directors (limited other boards, attendance); annual election (no staggered board);
  annual self-assessment; independent-only sessions; independent audit committee with financial experts and
  internal audit access; independent nominating and compensation committees; ability to hire outside counsel;
  published governance policies; disclosure and transparency; related-party transactions reviewed; responsiveness
  to shareholder proxy votes.
- Shareowner rights (L1 R41): one share one vote, confidential and remote voting, cumulative voting, proposals,
  say-on-pay, takeover defenses that entrench management, dual-class structures.
- Management: code of ethics, pay aligned with long-term value (watch option repricing, excessive severance),
  insider trading policy. Weak governance raises the cost of capital and the risk of the accounting red flags in
  `playbook-fra.md`.

### 4.3 Capital structure decision factors (L2 R26)
- Theory: MM I/II without taxes (irrelevance; re rises linearly with D/E), with taxes (value rises by tD), plus
  costs of financial distress, agency costs of debt and equity, and asymmetric information (pecking order:
  internal funds -> debt -> equity; issuing equity signals overvaluation). Static trade-off: optimal where the
  marginal tax shield equals marginal expected distress costs.
- Practice (section 3): debt ratings and their cost; capital structure over time, versus peers with similar
  business risk, and company specifics (cash flow volatility, tangible assets, growth options, governance);
  financial flexibility. International (3.3): institutional and legal environment (legal system, agency
  costs, bankruptcy codes), financial markets and banking (bank-centered systems use more debt), macro
  (inflation, GDP growth), tax regime.

### 4.4 Payout policy theories (L2 R27 sections 2-4)
- MM irrelevance (homemade dividends), bird-in-the-hand (Gordon, Lintner), tax argument (prefer low payout when
  dividends are taxed more heavily), clientele effects, signaling (initiations and increases are good news; cuts are
  bad unless they fund value-creating investment), agency (dividends and buybacks discipline free cash flow;
  bondholder-shareholder conflict over payouts).
- Factors: investment opportunities, earnings volatility, financial flexibility, taxes, flotation costs,
  contractual and legal restrictions (covenants, impairment of capital rules).
- Policies: stable (Lintner adjustment), constant payout ratio, residual; global trends of lower payout ratios and
  more buybacks. Buyback vs dividend: flexibility, tax, signaling of undervaluation, offsetting option dilution,
  changing leverage. Safety: payout, coverage, and FCFE / (dividends + buybacks).

### 4.5 M&A motives and defenses (L2 R29)
- Motives (section 3): synergies (cost, revenue), growth, market power, unique capabilities, diversification
  (questionable: investors diversify cheaper), bootstrapping EPS (illusory), managers' incentives (size, pay),
  tax benefits, unlocking hidden value, cross-border (market imperfections, policy, technology, differentiation,
  following clients). Industry life cycle shapes the merger type.
- Forms and payment (section 4): stock vs asset purchase; cash vs stock (stock shares synergy risk with the
  target; cash signals confidence); friendly vs hostile (bear hug, tender offer, proxy fight).
- Pre-offer defenses: poison pills (flip-in, flip-over), poison puts, state of incorporation with restrictive
  laws, staggered board, restricted voting rights, supermajority provisions, fair price amendments, golden
  parachutes. Post-offer: litigation, greenmail, share repurchase, leveraged recapitalization, "just say no",
  crown jewel, Pac-Man, white knight, white squire.
- Regulation (section 6): antitrust (HHI thresholds: below 1,000 unconcentrated; 1,000-1,800 with a change of 100+
  possible challenge; above 1,800 with a change of 50+ challenge), securities law (Williams Act disclosure).
- Evidence (section 8, p286-287): target shareholders earn about 30% premiums over the pre-announcement price;
  the acquirer's price falls 1-3% on average; both do better around cash offers than share offers; acquirers tend
  to underperform over the long run (winner's curse, hubris). Restructuring: divestitures, equity carve-outs,
  spin-offs, split-offs, liquidations.

## 5. Interpretation, red flags and exam traps

- NPV in currency, IRR in percent: choose by NPV when they conflict; IRR assumes reinvestment at the IRR.
- Do not discount financing flows; interest is in the WACC. Exclude sunk costs; include NWC recovery and after-tax
  salvage (tax on gain = t x (salvage - book value)).
- Replacement outlay subtracts the after-tax proceeds of the old asset; depreciation is incremental.
- Payback ignores later cash flows and time value; discounted payback still ignores flows after recovery.
- Unequal lives matter only if projects will be replaced; otherwise compare plain NPVs.
- WACC uses target market-value weights and marginal (new) costs, after-tax debt; never the coupon rate on old debt.
- Unlever with the comparable's D/E and tax rate, relever with the project's. Country risk premium multiplies beta
  in the curriculum's form (re = rf + beta(ERP + CRP)).
- MCC break point = amount where a source's cost changes / that source's weight.
- DOL is high near breakeven; DTL = DOL x DFL; operating breakeven excludes interest, total breakeven includes it.
- MM with taxes: re still rises with leverage, but less than without taxes; WACC falls; ignoring distress costs,
  value is maximized at 100% debt. Static trade-off adds the cost of distress.
- Buybacks raise EPS only if the earnings yield exceeds the after-tax cost of funds; BVPS falls when price > BVPS.
- Lintner: the curriculum example (Luna) applies the target payout to the earnings change; the classic form moves a
  fraction of the gap to the target dividend; the engine returns both.
- Trade credit cost uses days beyond the discount period, compounded; borrowing costs divide by usable proceeds.
- Stock offers dilute: price paid depends on the post-merger share price; the gains to target and acquirer always
  sum to the synergies.

## 6. Deliverable outline

1. **Recommendation** (accept/reject, target structure, payout change, bid range) and the key number.
2. **Assumptions table**: cash flows, rates, tax, life, weights, sources; what is estimated vs observed.
3. **Calculation trail**: engine function per step with the printed result.
4. **Sensitivity**: NPV to discount rate (profile, crossover), to sales and costs; WACC to beta and ERP; deal value
   to synergies and premium.
5. **Qualitative overlay**: strategic fit, real options, governance quality, capital structure constraints,
   signaling and tax effects of payout, M&A motive credibility and defenses.
6. **Risks and monitoring**: post-audit plan, covenants, rating thresholds, integration milestones.

## 7. Knowledge-base paths (E:\CLAUDE\CFA\MD)

- L1 curriculum `Level-1/05-Corporate-Finance/Curriculum-2014/`: R36 capital budgeting
  (`SS11-R36-capital-budgeting.md`, p26-36), R37 cost of capital (`SS11-R37-cost-of-capital.md`, p55-83), R38
  leverage (p108-119), R39 dividends and buybacks basics (p145-148), R40 working capital (p167, p174, p192-198),
  R41 corporate governance manual (`SS11-R41-the-corporate-governance-of-listed-companies-a-manual-for-investors.md`).
- L2 curriculum `Level-2/05-Corporate-Finance/Curriculum-2014/`: R25 capital budgeting (`SS08-R25-capital-budgeting.md`,
  p45-56 cash flows, MACRS, replacement, unequal lives, rationing; p66-71 real options and pitfalls), R26 capital
  structure (p113-126), R27 dividends and buybacks analysis (p150-180), R28 corporate governance (p199-225),
  R29 mergers and acquisitions (`SS09-R29-mergers-and-acquisitions.md`, p252-286).
- Kaplan / Schweser notes for the same readings sit beside the curriculum folders (`Kaplan-Notes-2014`,
  `Schweser-Notes-2013`).
- Tests with every golden value: `tests/test_corpfin.py`.
