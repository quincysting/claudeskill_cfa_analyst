# Playbook: performance evaluation, GIPS and risk management

Use this playbook to measure returns, attribute and appraise performance, check GIPS calculations and presentations,
and measure and govern risk (VaR, stress tests, credit exposure, risk budgets). Every number comes from the engine:
`python scripts/cfa.py list portfolio` (modules `portfolio_performance`, `portfolio_risk`, plus `portfolio` for active
management). Section 4 covers the judgement around those numbers. IPS, allocation and trading are in
`playbook-portfolio.md`.

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
| Return measurement | "What did the account earn?" TWR vs MWR, Modified Dietz, linking, annualizing, composite returns |
| GIPS | Is this calculation or presentation compliant? Dispersion, large cash flows, gross vs net, carve-outs, required disclosures |
| Benchmarks | Is the benchmark valid? Style vs active return, factor-model benchmarks |
| Attribution | Where did the excess return come from? Macro (sponsor decisions), micro (Brinson), international (currency), factor tilts |
| Appraisal | Was it skill? Sharpe, Treynor, Jensen, M^2, information and appraisal ratios, Sortino, drawdowns |
| Active management | How much active risk, and what IR is plausible? Fundamental law, Treynor-Black, manager risk budgets |
| Risk | VaR at a horizon and confidence; ES; component VaR; stress scenarios; credit exposure of derivatives; capital allocation |

## 2. Inputs to gather

- **Valuations and flows.** Collect the value at the start, at every external cash flow and at the end. For TWR the
  engine wants the value just *after* each flow plus the flow amount. Record flow dates (day of period, calendar days)
  and whether flows are start- or end-of-day.
- **Returns.** Collect period returns of portfolio, benchmark and market (same frequency), the risk-free rate, beta,
  standard deviations and residual SD, tracking error, and the minimum acceptable return.
- **Attribution data.**
  - Sector weights and returns for portfolio and benchmark.
  - For macro attribution: policy weights by asset category, category benchmark returns, manager weights within each
    category, manager benchmark and actual returns, total fund return.
  - For international attribution: country weights, local returns, index returns and currency moves.
- **Composite data.** Portfolio returns for portfolios held the full year, with beginning values. Also gross returns,
  fee schedule, firm and composite assets, and number of portfolios.
- **Risk data.** Collect the position values, the expected returns and SDs (state the period: annual, monthly), the
  correlations, and the confidence level and horizon.
  - Historical VaR needs the return history.
  - Stress tests need the shock scenarios.
  - Derivative credit exposure needs the contract terms: spot, forward price, rates, time remaining.

## 3. Quantitative workflow

Run each step in the engine and print the intermediate values. Round only when presenting.

**A. Return measurement (L3 R34, GIPS R35)**
1. Use `gips_time_weighted_return` to chain-link subperiods valued at each external flow (`market_values` +
   `cash_flows`), or to link any series of period returns (`subperiod_returns`, which also gives the LIRR).
2. Use `gips_money_weighted_return` for the IRR with flows at any time: daily rate, period MWR, annualized. Use it for
   the since-inception IRR of private equity and real estate.
3. Use `gips_modified_dietz` for day-weighted flows (end-of-day by default; `flow_timing="start_of_day"` or
   `"midpoint"` for the original Dietz).
4. Use `gips_composite_return` to asset-weight portfolio returns (beginning values, or beginning + weighted flows).
5. Use `gips_annualized_return` for geometric annualization. Its flag `gips_may_annualize` is False for periods under
   one year.
6. Use `gips_net_of_fees_returns` to get net from gross (arithmetic or geometric fee deduction).
7. Use `gips_composite_dispersion` for:
   - equal-weighted SD (n - 1) and population SD;
   - asset-weighted SD;
   - high-low range;
   - quartiles by the (n + 1) rule, and the IQR.

**B. Benchmarks and attribution (L3 R34; 2013 R42 global)**
1. Use `manager_style_active_decomposition`: P = M + S + A, with S = B - M and A = P - B.
2. Use `factor_model_benchmark_skill` for expected returns from a factor benchmark, the exposure-driven difference
   and skill.
3. Use `brinson_attribution` for allocation, selection and interaction by sector. Pass `method="brinson_fachler"`
   (curriculum: pure sector allocation vs the total benchmark) or `"bhb"`. Pass segments missing from the benchmark
   (cash) with their own return as the benchmark return.
4. Use `macro_attribution` to layer the fund-sponsor decisions: net contributions, risk-free asset, asset category,
   benchmarks (misfit), investment managers, allocation effects (residual).
5. Use `international_performance_attribution` to split into benchmark return, market allocation, currency
   allocation, yield and selection.
6. Use `factor_active_risk_return` (in `portfolio`) for active return from factor tilts vs selection, and active risk
   from factor vs specific sources (FMCAR).

**C. Appraisal and active management (L1 R44, L3 R34, L2 R58-R59, R55)**
1. Use `risk_adjusted_performance` for Sharpe, Treynor, Jensen's alpha and CAPM return. It gives `m_squared` both as
   a return (L3: rf + Sharpe x sd_m) and as `m_squared_excess` over the market (L1). It also gives the information
   ratio (active return / TE) and the appraisal ratio (alpha / residual SD).
2. Use `active_return_tracking_error` for active return, TE (sample SD) and IR from return series, with optional
   annualization.
3. Use `sortino_ratio_from_returns`. Its downside deviation versus the MAR uses n - 1, as in the curriculum.
4. Use `drawdown_ratios` for maximum drawdown, RoMAD and Calmar.
5. Use `fundamental_law_active_management` for:
   - IR = TC x IC x sqrt(BR);
   - optimal active risk IR/SR_B x sd_B;
   - combined Sharpe sqrt(SR_B^2 + IR^2);
   - Grinold's omega* = IR/(2 lambda) and VA* = IR^2/(4 lambda), with lambda in percent units.
6. Use `treynor_black_active_weights` for active weights proportional to alpha/sigma_e^2, w0, the beta-adjusted w*,
   and the Sharpe gain.
7. Use `active_risk_budget_managers` to split a tracking-error budget across uncorrelated managers in proportion to
   their IRs.

**D. Risk measurement (L3 R27)**
1. Use `var_parametric`.
   - Inputs: a single asset (`expected_return`, `std_dev`) or a portfolio (`weights`, `expected_returns`, `cov` or
     `sds` + `corr`).
   - Horizon: `input_periods_per_year` and `horizon_periods_per_year` (250 daily, 52 weekly, 12 monthly).
   - Defaults: the mean is included and z is the curriculum's (1.65 at 95%, 2.33 at 99%). Pass `z` for exact
     quantiles.
   - It also returns the normal expected shortfall.
2. Use `var_sqrt_time_scaling` to scale a VaR by sqrt(T). This is valid only with a zero mean
   (`include_mean=False` in `var_parametric`).
3. Use `var_historical` for the k-th worst return, with k = (1 - c) x n rounded half up (the curriculum's rule). It also
   gives the average of the adjacent observations (the curriculum's alternative) and historical ES. Use it on simulated
   outcomes too.
4. Use `var_monte_carlo` for seeded normal simulation of VaR and ES.
5. Use `var_component_parametric` for component VaR (sums to VaR), marginal VaR, incremental VaR (VaR with vs
   without), and the diversification benefit.
6. Use `stress_test_scenarios` to revalue the portfolio under named shocks and find the worst case.
7. Use `credit_risk_forward` for the value of a forward to each side, who bears credit risk, and current vs potential
   exposure. FX forwards use `foreign_rate`.
8. Use `risk_budget_allocation` for return on capital, return on VaR, and combined VaR under a correlation
   assumption (perfect correlation by default).

## 4. Qualitative frameworks from the curriculum

### 4.1 Performance evaluation (L3 R34)
- **Three parts.** Measurement (what was the return), attribution (why), appraisal (was it skill).
- **TWR vs MWR.** Use TWR to judge a manager who does not control external flows (GIPS standard). MWR (IRR) suits
  investors who control flows, and private equity and real estate closed-end funds.
- **Benchmark properties.** A valid benchmark is:
  - Specified in advance and appropriate;
  - Measurable and unambiguous;
  - Reflective of current investment opinions;
  - Accountable (the manager accepts it) and investable (SAMURAI).
- **Benchmark types, best last.** Absolute returns, manager universes (median: not specified in advance, not
  investable, survivorship bias), broad market indexes, style indexes, factor-model-based, returns-based (style
  analysis), custom security-based (normal portfolio).
- **Benchmark quality tests.**
  - The portfolio's beta against the benchmark is near 1.
  - Active return is uncorrelated with style return.
  - TE against the benchmark is lower than against alternatives.
  - High coverage and low turnover.
  - Active positions are positive.
- **Macro vs micro attribution.**
  - Macro measures the sponsor's decisions: policy allocation, benchmark misfit, manager selection, tactical
    allocation.
  - Micro measures the manager's: sector allocation and security selection (Brinson), factor bets, fixed-income
    effects.
  - Fixed-income attribution splits the external interest-rate effect (expected and unexpected) from management
    effects: duration, convexity, yield-curve shape, sector/quality, bond selection, trading.
- **Appraisal.**
  - Sharpe for a portfolio that is the investor's whole risky position.
  - Treynor or Jensen for one well-diversified part of a larger fund (systematic risk is what matters).
  - IR for active managers measured against a benchmark. M^2 restates Sharpe in return units.
  - Quality-control charts plot value-added against confidence bands. Bands narrow with sqrt(time).
- **Manager continuation.** Null hypothesis: no skill.
  - Type I error: keeping or hiring a manager with no skill.
  - Type II error: firing or not hiring a manager with skill.
  - Use a written continuation policy and watch lists; avoid knee-jerk firing (costly and often mean-reverting).

### 4.2 GIPS (L3 R35)
- **Scope.** Compliance is firm-wide. There is no partial compliance and no "in compliance except".
  - Every actual, fee-paying, discretionary portfolio must be in at least one composite, defined by strategy.
  - Composites may not be cherry-picked.
- **Calculation rules.**
  - Total returns, time-weighted with actual returns.
  - Fair value; trade-date accounting; accrual for fixed income.
  - Monthly valuation from 2001; on the date of large cash flows from 2010.
  - Composite returns asset-weighted, monthly from 2010.
  - Daily-weighted flow approximations (Modified Dietz) at least since 2005 when portfolios are not revalued.
- **Fees.** Gross-of-fees is net of trading expenses. Net-of-fees also deducts management fees (including performance
  fees). Bundled and wrap fees need specific treatment. Custody is not a trading cost.
- **Carve-outs.** Since 2010 a carve-out needs its own actually managed cash. Before that, cash had to be allocated in a
  timely and consistent way and the policy disclosed.
- **Presentation.**
  - At least 5 years (or since inception), building to 10.
  - Annual composite and benchmark returns.
  - Number of portfolios, with dispersion if there are six or more.
  - Composite assets, and firm assets or the composite share.
  - 3-year annualized ex post SD of composite and benchmark (from 2011).
  - The compliance statement.
  - Periods under a year are not annualized. Non-compliant periods after 2000 cannot be linked.
- **Special cases and verification.**
  - Real estate: external valuation; income and capital returns.
  - Private equity: since-inception IRR.
  - Verification is firm-wide and recommended. It tests composite construction and policies, not single-composite
    performance.

### 4.3 Risk management (L3 R27)
- **Risk governance.** Centralized (enterprise risk management: firm-wide view, netting, diversification) or
  decentralized. The process is: set policies and tolerance, identify, measure, monitor, adjust.
- **Risk taxonomy.**
  - Financial: market, credit, liquidity.
  - Nonfinancial: operational, model, settlement, regulatory, legal, tax, accounting, sovereign, political.
- **VaR.** A minimum loss at a probability over a horizon, not a maximum.

  | Method | Strengths | Weaknesses |
  |---|---|---|
  | Analytical | Easy | Normality, linear exposures |
  | Historical | No distribution assumption | Past may not repeat |
  | Monte Carlo | Flexible (options, non-normal) | Model and computation |

  - Backtest: a 5% daily VaR should be exceeded about 12-13 days a year.
  - Extensions: CFaR and EaR (cash flow or earnings at risk), TVaR/ES (losses beyond VaR), incremental VaR.
- **Stress testing.**
  - Scenario analysis: historical scenarios, or stylized ones such as the Derivatives Policy Group set (yield curve
    +/-100 bp and twists, volatility +/-20%, equity +/-10%, currencies +/-6% or 20%, swap spreads +/-20 bp).
  - Stressing models: factor push, maximum loss optimization, worst case.
- **Credit risk of derivatives.**
  - Current credit risk exists only when a payment is due. Potential credit risk is the market value to the party that
    is owed.
  - Forwards: risk is borne by the party with positive value.
  - Swaps: risk peaks mid-life (interest-rate and equity swaps) or later (currency swaps, because of the principal
    exchange).
  - Options: only the buyer bears it.
  - Mitigants: exposure limits, marking to market, collateral, netting, minimum credit standards, special-purpose
    vehicles, credit derivatives.
- **Capital allocation and limits.**
  - Allocation bases: nominal, VaR-based, maximum-loss.
  - Limits: risk budgets, position limits, liquidity limits, performance stop-outs.
- **Risk-adjusted performance.** Sharpe (total SD), RAROC (expected return / capital at risk), RoMAD
  (return / maximum drawdown), Sortino (downside deviation versus the MAR).

### 4.4 Active management (L2 R58-R59; L2 2013 R55)
- **Fundamental law.** Breadth must count *independent* bets: correlated bets add less, and the IC falls. The transfer
  coefficient (constraints such as long-only) scales IR down.
- **Value added.** Value added peaks at omega* = IR/(2 lambda). Doubling active risk beyond that destroys utility.
- **Treynor-Black.** Weights each security by alpha/residual variance. The active portfolio's weight then adjusts for
  its beta. Shrink alphas by forecast quality before use.

## 5. Interpretation, red flags and exam traps

- **Sharpe vs Treynor.** Different measures can rank managers differently (R34 Q22). Choose by context: total
  portfolio (Sharpe) vs part of a diversified fund (Treynor or alpha). Say which and why.
- **M^2 has two forms.** L3 gives a return (rf + Sharpe x sd_m) to compare with the market return. L1 gives the excess
  over the market. Report which.
- **VaR horizon scaling.**
  - The curriculum scales the mean by t and sigma by sqrt(t), and *includes* the mean.
  - Scaling a VaR figure by sqrt(t) is only right when the mean is zero.
  - Use 250 trading days unless told otherwise.
  - z = 1.65 and 2.33 in the book (exact 1.645 and 2.326). State the convention.
- **Historical VaR order statistic.** 5% of 240 is the 12th worst; 1% of 240 = 2.4 rounds to the 2nd worst (the
  alternative averages the 2nd and 3rd). 1% of 60 = 0.6 is the worst observation.
- **Wording.** "1% VaR" and "99% VaR" are the same number. VaR is a minimum loss in the tail, not a maximum.
- **TWR traps.**
  - The value used for the subperiod return excludes the flow: (MV_after - CF - MV_prior)/MV_prior.
  - TWR and MWR diverge when large flows precede strong or weak returns. Do not judge a manager on MWR.
- **Modified Dietz weights.** (CD - D)/CD for end-of-day flows, (CD - D + 1)/CD for start-of-day. The original Dietz
  assumes the midpoint.
- **Dispersion.** GIPS equal-weighted SD uses n - 1 (with n the dispersion is lower). Only portfolios held the full
  period count. Quartiles use (n + 1) x y/100 in the descending array.
- **Brinson.**
  - Curriculum allocation is (w_p - w_b)(R_b,i - R_b), measured against the *total* benchmark.
  - Interaction uses (w_p - w_b)(R_p,i - R_b,i).
  - Selection uses benchmark weights.
  - The three sum to active return. "Trading and other" is the gap between buy-and-hold and actual return.
- **Macro attribution order.** Risk-free, then asset category (policy minus rf), then benchmarks (misfit), then
  managers, then allocation effects (residual from actual vs policy weights).
- **Style vs active.** Style S = B - M is the sponsor's choice. Only A = P - B is the manager's.
- **Credit risk.** A forward with negative value to you is credit risk to your counterparty, not to you. A swap's
  credit risk before a payment date is potential, not current.
- **Information ratio.** Annualize with mean x periods and TE x sqrt(periods). An IR from fewer than a few years of data
  is mostly noise (R34 Exhibit 19).

## 6. Deliverable outline

**Performance review**
1. Return summary: TWR by period, annualized (no annualizing under one year), versus the benchmark and market, gross
   and net.
2. Benchmark assessment against SAMURAI, quality tests, and the style vs active split.
3. Attribution: macro layers for a sponsor, Brinson or factor attribution for a manager, currency effects if
   international.
4. Risk-adjusted appraisal (Sharpe, Treynor, alpha, M^2, IR, Sortino, drawdown), with the chosen headline measure
   justified.
5. Conclusion: skill vs luck (IR horizon), continuation decision, watch-list triggers.

**GIPS check.** Cover firm definition, composite construction, the return method (TWR, large cash flows), fees, carve-out
cash, the required presentation items (dispersion, 3-year SD, number of portfolios, assets) and disclosures. List the
gaps found.

**Risk report**
1. VaR (method, horizon, confidence, mean included or not) and ES.
2. Component and incremental VaR by position.
3. Stress scenarios with worst case and the factors behind it.
4. Credit exposure by counterparty (current vs potential) and mitigants.
5. Capital and risk budget usage (return on capital, return on VaR).
6. Limit breaches and recommended actions.

## 7. Knowledge-base paths (E:\CLAUDE\CFA\MD)

- Performance evaluation: `Level-3/13-Performance-Evaluation/Curriculum-2014/SS17-R34-evaluating-portfolio-performance.md`.
  Global attribution: `Curriculum-2013/SS17-R42-global-performance-evaluation.md`.
- GIPS: `Level-3/14-GIPS/Curriculum-2014/SS18-R35-overview-of-the-global-investment-performance-standards.md`.
- Risk management: `Level-3/10-Risk-Management/Curriculum-2014/SS14-R27-risk-management.md` (2013: `SS14-R34`).
  Currency management: `SS14-R28-currency-management-an-introduction.md`.
- Performance measures (L1): `Level-1/06-Portfolio-Management/Curriculum-2014/SS12-R44-portfolio-risk-and-return-part-ii.md`.
- Active management (L2): `Level-2/06-Portfolio-Management/Curriculum-2014/SS18-R58-*.md` and `SS18-R59-*.md`.
  Treynor-Black: `Curriculum-2013/SS18-R55-the-theory-of-active-portfolio-management.md`.
- Sortino and downside deviation (alternatives): `Level-3/09-Alternative-Investments/Curriculum-2014/SS13-R26-*.md`.
