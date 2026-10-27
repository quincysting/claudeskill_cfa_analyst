# Playbook: equity investments, valuation and equity portfolio management

Use this playbook when a question is about what a share, an index or an equity portfolio is worth or how it
performed: margin trades and index arithmetic, required returns, DDM, free cash flow, multiples, residual income,
private company valuation, or active equity management. Every number comes from `cfa_calc.equity`
(`python scripts/cfa.py list equity`). The judgement around those numbers comes from the curriculum frameworks in
section 4.

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
| Trading mechanics (L1) | "What is my return on a margin purchase?" "At what price do I get a margin call?" |
| Index arithmetic (L1) | Price, total return and weights of a price-, equal-, cap- or float-weighted index; divisor after a split |
| Required return (L2) | CAPM, Fama-French or Pastor-Stambaugh, build-up for a private firm, bond-yield-plus-premium, ERP estimates |
| Intrinsic value (L1-L2) | Gordon, two-stage, H-model, three-stage DDM; FCFF/FCFE; residual income; preferred stock |
| Relative value (L1-L2) | Justified P/E, P/B, P/S, P/CF; PEG; EV/EBITDA; portfolio P/E; earnings surprise |
| Market-implied views | Implied growth, implied return (IRR) at today's price, PVGO share of price |
| Private companies (L2) | Capitalized cash flow, excess earnings, DLOC/DLOM, control premium |
| Active management (L3) | Active return, tracking risk, IR, fundamental law, core-satellite risk budget, true vs misfit return |

Not here: the cost of capital for capital budgeting (corporate finance), FRA quality-of-earnings adjustments
(fra module; do them first, then value), Sharpe/Treynor/attribution of whole portfolios (portfolio module).

## 2. Inputs to gather

- **Timing of the cash flow.** D0 (just paid, trailing) or D1 (next year); FCFF0 or FCFF1; trailing EPS (E0) or
  forward EPS (E1). Write the date of each number. This one choice causes most valuation errors.
- **Discount rate matched to the cash flow.** Dividends and FCFE at the required return on equity r; FCFF at WACC
  (after-tax cost of debt, market-value or target weights, preferred included). Nominal with nominal, real with real.
- **Growth.** State the source: g = b x ROE (sustainable), analyst consensus, GDP-linked, or the market-implied rate.
  For multistage models, collect the length of each stage and whether growth falls abruptly (two-stage) or
  linearly (H-model).
- **Required-return inputs.** Risk-free rate (long government bond yield for CAPM; a short rate for Fama-French
  RMRF), beta (raw or Blume-adjusted 2/3 x raw + 1/3), ERP and its estimation method, size and company-specific
  premia, the company's bond YTM.
- **FCF inputs.** Net income to common, noncash charges (depreciation, amortization, impairments; subtract noncash
  gains), after-tax interest, preferred dividends, FCInv (capex net of asset sales), WCInv (excluding cash and
  short-term debt), net borrowing, tax rate, shares outstanding, market value of debt and preferred, nonoperating cash.
- **RI inputs.** Book value per share B0, EPS or ROE forecasts, dividends or payout, cost of equity, persistence
  factor (0 to 1), clean-surplus violations (OCI items, buybacks at non-book prices).
- **Index inputs.** Prices at both dates, dividends, shares outstanding, float fraction, current divisor.
- **Private companies.** Normalized earnings (owner compensation, related-party rent, one-offs), the basis of the
  value (controlling or minority, marketable or not), control premia from transactions, DLOM evidence.
- **L3.** Portfolio and benchmark return series, IC, breadth, transfer coefficient, manager weights, alphas,
  tracking risks, normal (style) benchmark returns.

## 3. Quantitative workflow

Run each step through the engine and print the intermediate values (conventions.md). Round only when presenting.

**A. Trading and indices (L1 R46-R47)**
1. `equity_margin_purchase_return`: give `leverage_ratio` or `initial_margin`, the call money rate, dividends and
   commission per share. Compare `total_return` with `leveraged_price_return` to show how much interest and costs cost.
2. `equity_margin_call_price`: long P0(1 - IM)/(1 - MM); `position="short"` for P0(1 + IM)/(1 + MM).
3. `equity_index_return` with `weighting` = price, equal, market_cap or float_adjusted. Report price return,
   income return and total return, and the beginning weights, which explain why the methods differ.
4. `price_weighted_divisor_adjustment` after a split, stock dividend or constituent change. The index value must not
   move.

**B. Required return (L2 R31)**
1. CAPM: `equity_required_return_capm`. Multifactor (FFM, PSM, BIRR): `equity_required_return_multifactor` with
   names, so each factor's contribution is visible.
2. Private or small firms: `equity_required_return_build_up` (add `beta=` for the modified CAPM).
   Cross-check with `equity_required_return_bond_yield_plus` (company YTM + 3-4% in U.S. practice).
3. ERP: `equity_risk_premium_historical` (state arithmetic or geometric and the sample),
   `equity_risk_premium_gordon` (forward D/P + long-term growth - long bond yield) and
   `equity_risk_premium_supply_side` (Ibbotson-Chen). Report the range of estimates, not one number.
4. WACC for FCFF: `equity_valuation_wacc` (supports preferred).

**C. Dividend discount models (L1 R51, L2 R35)**
1. Pick the growth pattern: constant → `ddm_gordon_value`; abrupt drop → `ddm_two_stage_value`;
   three distinct rates → `ddm_three_stage_value`; linear fade → `ddm_h_model_value`; high growth then a linear fade →
   `ddm_three_stage_declining_value`; any explicit path (zeros for non-payers, a P/E-based terminal value, a
   liquidating dividend) → `ddm_multistage_value` with `terminal_growth` or `terminal_value`.
2. Preferred: `equity_preferred_stock_value` (perpetual, or `years`/`par`/`freq` for retractable or term issues).
3. Growth input: `ddm_sustainable_growth` from b x ROE, ROA x leverage or the PRAT/DuPont components.
4. Reverse-engineer the market: `ddm_implied_growth`, `ddm_gordon_implied_return`, `ddm_h_model_implied_return`,
   `ddm_multistage_implied_return` (IRR). Compare with your own estimates.
5. Decompose: `equity_pvgo_decomposition`. A large PVGO share means the price rests on growth you have to defend.
6. Sensitivity: `equity_gordon_sensitivity_table` over a grid of r and g (also works for FCFE0 or FCFF0).
   Always show it; single-stage values move sharply as r - g narrows.

**D. Free cash flow (L2 R36)**
1. `equity_fcff_fcfe` with `start` = net_income, cfo, ebit or ebitda. Running two starts that agree is a cheap
   check on your inputs.
2. Forecasts tied to sales: `equity_fcf_from_sales_forecast`, one year at a time, with margins, the incremental fixed
   and working capital rates per unit of sales increase, and the target debt ratio for FCFE.
3. Value: `equity_fcff_valuation` (firm value; subtract debt and preferred, add nonoperating assets) or
   `equity_fcfe_valuation` (equity directly). Single-stage with `fcff0`/`fcfe0`; multistage with `forecast_*`. Pass
   `terminal_*` when year n+1 differs from year n x (1 + g), e.g. investment falls when growth slows.
4. Report the PV of the terminal value as a share of total value.

**E. Market multiples (L1 R51, L2 R37)**
1. Justified multiples from fundamentals: `equity_justified_pe` (leading and trailing), `equity_justified_pb`,
   `equity_justified_ps`, `equity_justified_price_to_cash_flow`. For a DCF value, the justified multiple is
   value / forecast metric.
2. Comparables: compute the peer benchmark with `equity_multiple_harmonic_mean`. The weighted harmonic mean equals
   the portfolio or index multiple; also report the median. Then adjust for growth (`equity_peg_ratio`), risk and quality.
3. Enterprise value: `equity_enterprise_value` (market value of debt, preferred and minority interest, less cash and
   investments) gives EV/EBITDA, EV/EBIT and EV/Sales.
4. Momentum: `equity_earnings_surprise_sue` (SUE, or the surprise scaled by the dispersion of analyst forecasts).

**F. Residual income (L2 R38)**
1. Period RI: `equity_residual_income` (equity charge) or `equity_economic_value_added` (NOPAT minus a capital charge;
   MVA if market value is given).
2. Constant ROE and g: `equity_residual_income_single_stage` gives value, justified P/B and, with `price`, the
   implied RI growth rate.
3. Multistage: `equity_residual_income_valuation` with EPS and/or ROE per period, dividends or payout, and a
   `terminal` of none, perpetuity, persistence (with omega) or premium. Check clean surplus: B_t = B_(t-1) + E_t - D_t.
4. Cross-check: under consistent assumptions, RI value = DDM value (the Bugg test in tests/test_equity.py).

**G. Private companies (L2 R39)**
1. Normalize earnings first (owner pay, related-party rent, nonrecurring items), then pick the method:
   `private_company_capitalized_cash_flow` (stable growth), FCFF/FCFE valuation from D (high or changing growth), or
   `private_company_excess_earnings` (very small firms, intangibles).
2. Discount rate: build-up (B2), adding size and company-specific premia as the facts justify.
3. Discounts: `private_company_valuation_discounts`. DLOC = 1 - 1/(1 + control premium). The discounts multiply
   (1 - (1 - DLOC)(1 - DLOM)); they are not added. Apply DLOC only when the value is on a control basis and the
   interest is a minority one.

**H. Equity portfolio management (L3 R23)**
1. `equity_active_return_risk`: mean active return, tracking risk and IR from return series (annualize with
   `periods_per_year`).
2. `equity_fundamental_law`: IR ≈ TC x IC x √BR. Compare strategies with high breadth and low IC against the reverse;
   add `active_risk` to get the expected active return.
3. `equity_manager_portfolio_active_risk`: core-satellite risk budget. Alphas add by weight; tracking risks combine
   with correlations (uncorrelated by default).
4. `equity_manager_true_misfit`: true active return against the normal benchmark, misfit against the investor's
   benchmark, and the true IR.

## 4. Qualitative frameworks from the curriculum

**Valuation process (L2 R30, Equity Valuation: Applications and Processes).** (1) Understand the business: industry
prospects, competitive position, strategy, and the quality of financial reports. (2) Forecast performance.
(3) Select the model. (4) Convert forecasts into a value with sensitivity analysis. (5) Apply the conclusion and
communicate it. Watch for quality-of-earnings warning signs: aggressive revenue recognition, capitalized costs,
falling reserves, CFO well below net income, and frequent nonrecurring charges. Value definitions: intrinsic,
going-concern, liquidation, fair market value, investment value.

**Industry analysis (L1 R50; L2 R32 Porter, R33-R34).**
- Porter's five forces (L2 R32): threat of new entrants, bargaining power of suppliers and of buyers, threat of
  substitutes, and rivalry. Industry profitability depends on the structure, not on fast growth or government.
- Industry life cycle (L1 R50 §5.1.5): embryonic → growth → shakeout → mature → decline. Map the stage to DDM stage
  lengths and to the payout path.
- Strategic analysis (L1 R50 §5.1): barriers to entry, concentration, capacity, stability of market share, price
  competition. External influences: macroeconomic, technological, demographic, governmental, social.
- Peer groups (L1 R50 §4.4): GICS, RGS and ICB classifications, adjusted for business mix and sensitivity to the cycle
  (cyclical or defensive). A bad peer group spoils every comparable multiple.
- Company analysis checklist (L1 R50 Exhibit 8): corporate profile, industry characteristics, demand and supply
  analysis, pricing, financial ratios, competitive position. L2 R34 adds forecasting revenue (top-down, bottom-up,
  hybrid), costs, working capital and capex, and the effect of competition on the forecasts.

**Competitive position and strategy (L2 R33, "Your strategy needs a strategy").** Classical, adaptive, shaping and
visionary strategies, chosen by how predictable and how malleable the environment is. Ask whether the moat behind a
high ROE can last; that sets how long the high-growth stage runs and how strongly RI persists.

**Choosing the model (L2 R35 §2.2, R36, R38 §4.2).**
- DDM: the company pays dividends, the dividend policy bears a consistent relation to profitability, and the investor
  holds a minority (noncontrol) stake.
- FCFE/FCFF: the company pays no dividends or dividends differ from capacity to pay, FCF tracks profitability, or the
  investor takes a control perspective (takeover). Use FCFF when leverage is high or changing, or FCFE is negative.
- Residual income: the company pays no or unpredictable dividends, FCF is negative over the forecast horizon, or the
  terminal value is very uncertain. Avoid it when clean surplus fails badly or book value and ROE cannot be forecast.
- Multiples: for relative value, communication and the terminal value. They depend on a sound peer group and on the
  market pricing the peers correctly.
- Asset-based: when assets have observable market values (financials, resources, liquidation). It gives a floor
  when intangibles matter.
- Private firms: the income, market (guideline public companies or transactions) and asset approaches. Weigh them by
  stage, data and purpose (transaction, tax, litigation, financial reporting).

**Quality checks before you sign off.** g < r and g ≤ long-run nominal GDP in the terminal stage; the terminal ROE
is plausible (it fades toward r); the payout is consistent with g = b x ROE; justified multiples are close to peers
or the gap is explained; FCF signs are sensible; the share of value in the terminal value is reported; at least
two models are triangulated.

**Active management (L3 R23).** Passive (tracking risk under 1%), enhanced (1-2%, IR around 0.75) or active
(4%+, IR around 0.5) by market efficiency and the investor's beliefs. Full replication, stratified sampling or
optimization for indexing. Style (value, growth, market-oriented; holdings-based vs returns-based analysis; style
drift). Long-short and market-neutral portfolios. Core-satellite with a completeness fund. Portable alpha. Manager
search: questionnaire, fees, and the sell discipline.

## 5. Interpretation, red flags and exam traps

- **D0 vs D1.** Gordon needs D1. If the question says "just paid", multiply by (1 + g). The same goes for FCFF0 and
  FCFE0 and for trailing vs leading P/E: leading = payout/(r - g); trailing = payout x (1 + g)/(r - g).
- **g ≥ r.** The engine raises an error. A negative r - g is no answer; change the model or the stage.
- **Terminal value timing.** The Gordon value at t = n uses D_(n+1) and is discounted n periods, not n + 1.
- **FCFE is not dividends.** FCFE is capacity to pay; dividends are a board decision. Buybacks, dividends and share
  issues do not change FCF; leverage does.
- **FCFF pitfalls.** Add back after-tax interest (not pre-tax); add back preferred dividends to FCFF; with an
  EBITDA start add the depreciation tax shield (Dep x t); with a CFO start do not deduct WCInv again. Never
  discount NI or EBITDA as if it were cash flow. Discount FCFF at WACC, FCFE at r.
- **Equity from FCFF.** Firm value minus the market value of debt and preferred, plus nonoperating assets.
- **Multiples.** Negative earnings make P/E meaningless: use inverse E/P ranking or EV/Sales. Compare like with like
  (trailing vs leading, diluted, normalized over the cycle). PEG assumes P/E is linear in growth and ignores risk and
  how long growth lasts. Use the harmonic mean for averages of P/E.
- **Residual income.** The equity charge uses beginning book value. Clean-surplus violations (OCI) distort RI.
  Persistence omega near 1 means RI lasts; near 0 means it fades quickly. Value is recognized earlier than in a DDM,
  so the terminal value is a small share.
- **Index arithmetic.** In a price-weighted index the high-priced stocks dominate, and splits change the weights; equal
  weighting needs rebalancing and tilts to small caps; cap weighting tilts to momentum and overvalued names. The
  total return index exceeds the price index by reinvested income.
- **Margin.** Leverage magnifies the price return; interest and commissions make the realized return worse than
  leverage x price return.
- **Discounts.** Never add DLOC and DLOM; never apply DLOC to a value that is already on a minority basis.
- **IR and the fundamental law.** Breadth means independent decisions, not just names covered; constraints (TC < 1)
  cut the IR.
- **ERP.** Arithmetic and geometric historical estimates differ, survivorship bias raises historical estimates, and
  the GGM estimate depends on the consensus growth figure.

## 6. Deliverable outline

1. **Recommendation and value range.** Point estimate, range from the sensitivity table, price, upside or downside.
2. **Business and industry.** Five forces, life-cycle stage, competitive position, strategy, peer group.
3. **Required return.** The models used, their inputs and the estimate chosen, with why.
4. **Valuation.** The primary model (why it fits), its inputs table, the engine outputs (value, PV of the terminal value
   and its share, implied multiples), and a cross-check model (DDM vs FCFE vs RI vs multiples).
5. **Market-implied view.** Implied growth or IRR at today's price, the PVGO share, and whether they are plausible.
6. **Risks and sensitivities.** r and g grid, key drivers, quality-of-earnings flags.
7. **Appendix.** Engine calls with arguments and full-precision outputs. State the rounding used.

For L3 mandates: active return, tracking risk and IR versus objective; risk budget across managers;
true vs misfit decomposition; recommendation on allocation or replacement.

## 7. Knowledge-base paths (E:\CLAUDE\CFA\MD)

| Topic | Path |
|---|---|
| Margin, orders, market structure | `Level-1/07-Equity/Curriculum-2014/SS13-R46-market-organization-and-structure.md` |
| Index construction and returns | `Level-1/07-Equity/Curriculum-2014/SS13-R47-security-market-indices.md` |
| Market efficiency | `Level-1/07-Equity/Curriculum-2014/SS13-R48-market-efficiency.md` |
| Equity securities overview | `Level-1/07-Equity/Curriculum-2014/SS14-R49-overview-of-equity-securities.md` |
| Industry and company analysis | `Level-1/07-Equity/Curriculum-2014/SS14-R50-introduction-to-industry-and-company-analysis.md` |
| Valuation basics (DDM, multiples, EV, asset-based) | `Level-1/07-Equity/Curriculum-2014/SS14-R51-equity-valuation-concepts-and-basic-tools.md` |
| Valuation process, quality of earnings | `Level-2/07-Equity/Curriculum-2014/SS10-R30-equity-valuation-applications-and-processes.md` |
| Return concepts, ERP, CAPM/FFM/build-up | `Level-2/07-Equity/Curriculum-2014/SS10-R31-return-concepts.md` |
| Porter's five forces | `Level-2/07-Equity/Curriculum-2014/SS11-R32-the-five-competitive-forces-that-shape-strategy.md` |
| Strategy selection | `Level-2/07-Equity/Curriculum-2014/SS11-R33-your-strategy-needs-a-strategy.md` |
| Industry and company forecasting | `Level-2/07-Equity/Curriculum-2014/SS11-R34-industry-and-company-analysis.md` |
| Discounted dividend valuation | `Level-2/07-Equity/Curriculum-2014/SS11-R35-discounted-dividend-valuation.md` |
| Free cash flow valuation | `Level-2/07-Equity/Curriculum-2014/SS12-R36-free-cash-flow-valuation.md` |
| Market multiples, EV, momentum | `Level-2/07-Equity/Curriculum-2014/SS12-R37-market-based-valuation-price-and-enterprise-value-multiples.md` |
| Residual income | `Level-2/07-Equity/Curriculum-2014/SS12-R38-residual-income-valuation.md` |
| Private company valuation | `Level-2/07-Equity/Curriculum-2014/SS12-R39-private-company-valuation.md` |
| Equity portfolio management | `Level-3/08-Equity-Portfolio-Management/Curriculum-2014/SS11-R23-equity-portfolio-management.md` |
| International equity benchmarks | `Level-3/08-Equity-Portfolio-Management/Curriculum-2014/SS12-R24-international-equity-benchmarks.md` |
| Corporate performance, governance, ethics | `Level-3/08-Equity-Portfolio-Management/Curriculum-2014/SS12-R25-corporate-performance-governance-and-business-ethics.md` |

Kaplan/Schweser summaries for L1 sit beside the curriculum (`Level-1/07-Equity/Kaplan-Notes-2014/`,
`Schweser-Notes-2013/`). Do not open `Level-1/Exams/`.
