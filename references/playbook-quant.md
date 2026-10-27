# Playbook: Quantitative Methods

Engine modules: `scripts/cfa_calc/tvm.py` (time value, NPV/IRR, returns, money market),
`stats.py` (descriptive statistics, risk ratios, probability, counting), `inference.py` (distributions,
confidence intervals, hypothesis tests, Monte Carlo), `regression.py` (correlation, OLS, diagnostics, time series).
Find any function with `python scripts/cfa.py search <word>` and read its units with `describe <name>`.
Every number in the answer comes from one of these calls. Show the call and its output.

Knowledge base (KB) root: `E:/CLAUDE/CFA/MD`. Curriculum 2014 reading numbers are used below.
- L1: `Level-1/02-Quantitative-Methods/Curriculum-2014/` R5 TVM, R6 DCF applications, R7 statistics,
  R8 probability, R9 distributions, R10 sampling, R11 hypothesis testing.
- L2: `Level-2/02-Quantitative-Methods/Curriculum-2014/` R11 correlation and regression,
  R12 multiple regression, R13 time series.
- Faster review: the `Kaplan-Notes-2014/` and `Schweser-Notes-2013/` folders next to each curriculum folder.

## 1. When to use

- Valuing a cash-flow stream: a loan, lease, pension liability, structured settlement, retirement plan,
  an annuity versus a lump sum, a project's NPV and IRR.
- Measuring performance: holding-period, money-weighted and time-weighted returns; annualizing; comparing a
  manager's record with a benchmark or peers (quartile ranks, Sharpe ratio, safety-first).
- Pricing and comparing short-term instruments: T-bills, commercial paper, CDs quoted on different bases.
- Describing a return history: mean, dispersion, downside risk, skewness, fat tails.
- Scenario and probability work: expected values from scenario trees, Bayes updates when news arrives,
  portfolio risk from a covariance matrix, default and shortfall probabilities.
- Testing a claim with data: "the fund beats its benchmark", "volatility rose", "the strategy's mean return
  differs", "these two series are correlated".
- Building or auditing a forecasting model: factor or style regressions, beta estimation, earnings or sales
  trends, autoregressive models of inflation, margins or growth.

## 2. Inputs to gather (and defaults)

| Task | Must have | Ask or assume |
|---|---|---|
| TVM | amounts, timing (t = 0 first), rate, compounding | Stated rate or EAR? Payments in arrears unless told "due"/BGN. State the compounding assumption. |
| NPV / IRR | full cash-flow vector incl. CF0, discount rate | Mixed-sign flows: expect several or no IRRs; ask for the hurdle and reinvestment rates (MIRR). |
| Portfolio returns | values just before each external flow, flow dates and amounts | Use TWR for the manager, MWR for the investor's own outcome. Period length to annualize. |
| Money market | face, price or quoted yield, days to maturity | Quote basis (bank discount on face/360 vs money market on price/360 vs 365). |
| Descriptive stats | the raw series, units (percent or decimal) | Sample (default) or full population? Period frequency for annualizing. |
| Probabilities | priors and likelihoods, or joint table | Events must be mutually exclusive and exhaustive; probabilities must sum to 1. |
| Hypothesis test | statistic inputs, H0 value, alpha, tail | Default alpha 0.05, two-tailed, unless the claim is directional ("greater than"). |
| Regression | aligned y and x series, n, frequency | Check for look-ahead, survivorship, overlapping observations, structural breaks. |
| Time series | the series in time order | Is it trending, mean-reverting or a random walk? Seasonal frequency (4, 12). |

Defaults the engine uses: rates are decimals, `freq` is periods per year, cash-flow lists start at t = 0 with
outflows negative, money-market conventions are actual/360 unless stated, sample (n - 1) variance, alpha 0.05,
two-tailed tests, Monte Carlo seed 42.

## 3. Quantitative workflow

### 3.1 Time value of money (L1 R5)
1. Lay out a timeline; index today t = 0 and convert the quoted rate to the period rate:
   `effective_annual_rate`, `stated_annual_rate` (any frequency or continuous).
2. Single amounts: `fv_lump_sum`, `pv_lump_sum` (`freq`, `continuous=True`).
3. Level streams: `annuity_pv`, `annuity_fv` (`due=True` for payments today). Unknown payment, number of
   periods or rate: `tvm_solve(solve_for=...)` with calculator signs (money out negative).
4. Perpetuities and deferred streams: `perpetuity_pv(first_payment_period=k, growth=g)`; growing finite
   streams `growing_annuity_value`; irregular streams `uneven_cash_flows_value`.
5. Loans: `loan_amortization_schedule` for the payment, interest/principal split and remaining balance.
6. Multi-stage goals (retirement funding): value every leg at one common date, then solve the gap with
   `tvm_solve("pmt", ...)`. Print each intermediate value.

### 3.2 Capital budgeting and return measurement (L1 R6)
1. `npv(rate, cash_flows)`; accept if NPV > 0; for mutually exclusive projects rank by NPV.
2. `irr(cash_flows)`; read `note` and `irrs`. If several or none, report NPV (and `mirr` with the firm's
   finance and reinvestment rates) instead of a single IRR.
3. Investor return: `money_weighted_return(cash_flows, freq)`. Manager return: `time_weighted_return`
   from subperiod returns or begin/end values; annualize with `years=` or `annualize_return`.
4. Single period: `holding_period_return`.
5. Money market: `money_market_yields(days, face, price=... or any one yield)` returns BDY, HPY, EAY, MMY,
   the semiannual bond-equivalent yield and the 365-day add-on yield. Discount future cash with the HPY.

### 3.3 Describing returns and risk (L1 R7)
1. `descriptive_statistics(series)`: mean, median, modes, range, MAD, sample and population variance and
   standard deviation, CV, sample skewness and excess kurtosis (curriculum formulas).
2. Compound growth: `geometric_mean_return` (report with the arithmetic mean). Averages of ratios applied to
   a fixed amount (cost averaging): `mean_measures` (harmonic). Portfolio or index return: weighted mean.
3. Position in a distribution: `quantile_curriculum(data, p)` (position (n + 1)p, interpolate).
4. Downside risk: `semivariance(data, target=...)`. Distribution-free bounds: `chebyshev_inequality`.
5. Risk-adjusted: `sharpe_ratio_historical`, `coefficient_of_variation`, `roy_safety_first_ratio`
   (also gives the normal shortfall probability).

### 3.4 Probability (L1 R8-R9)
1. Two events: `probability_rules` (union, joint, conditional, independence check).
2. Scenarios: `total_probability` (also works for expected values), `bayes_update` for new information.
3. Discrete outcomes: `expected_value_variance`; joint tables: `joint_probability_covariance`.
4. Portfolios: `portfolio_expected_return_variance(weights, expected_returns, covariance_matrix)`; it also
   returns the correlation matrix and splits variance into own-variance and covariance terms.
5. Counting: `counting_rules` (factorial, nCr, nPr, multinomial). Success counts: `binomial_probability`.
   Price lattices: `binomial_tree`. One-period credit pricing: `one_period_default_risk_premium`.

### 3.5 Distributions, estimation and tests (L1 R9-R11)
1. Normal: `normal_probability` (z-scores included), `normal_quantile`. Prices: `lognormal_moments`;
   returns: `continuously_compounded_return`, `volatility_from_prices` (annualize by sqrt of periods).
2. Estimation: `standard_error_of_mean`, `confidence_interval_mean(distribution="t" or "z")`.
3. Testing, five steps every time: state H0/Ha and the tail; pick the statistic; set alpha; compute; decide
   and interpret economically.
   - one mean: `t_test_mean` (variance unknown) or `z_test_mean` (known variance or given standard error);
   - two independent means: `t_test_two_means_pooled` (equal variances) or `t_test_two_means_unequal`;
   - dependent samples (same period, overlapping portfolios): `t_test_paired`;
   - one variance: `chi_square_test_variance`; two variances: `f_test_variances` (larger variance on top);
   - correlation: `correlation_t_test`; non-normal or ranked data: `spearman_rank_correlation`.
4. Simulation: `monte_carlo_terminal_wealth(..., seed=...)` for terminal-value distributions and shortfall
   probabilities; always report the seed and the number of trials.

### 3.6 Regression (L2 R11-R12)
1. Look at the data, then `sample_correlation` (with its t-test).
2. Fit: `ols_regression(y, x)`; x as a dict of named columns. Read in order: F-test of the whole model,
   then each coefficient's t-stat and p-value, then R-squared, adjusted R-squared and SEE.
3. Reported output only: `regression_anova_from_summary` (any two of RSS, SSE, SST, or R-squared with one),
   `regression_coefficient_t_test` (e.g. H0: beta = 1).
4. Forecast: `ols_regression(..., x_new=...)` gives the prediction interval and the interval for the mean;
   from summary statistics use `simple_regression_prediction_interval`.
5. Qualitative regressors: `make_dummy_variables(categories, base=...)` (n - 1 dummies).
6. Diagnostics: `breusch_pagan_test` (heteroskedasticity), `durbin_watson_statistic` (serial correlation,
   not with lagged y), `variance_inflation_factors` (multicollinearity).

### 3.7 Time series (L2 R13)
1. Plot; fit `trend_regression` (linear for constant change, `log_linear=True` for constant growth).
   Forecast from reported coefficients with `trend_forecast`.
2. If the trend residuals are serially correlated, move to an AR model: `autoregressive_model_fit(series,
   lags=[1])`; check `residual_autocorrelations` (t = r / (1/sqrt(T))). Significant lags: add lags; a
   significant seasonal lag (4 quarterly, 12 monthly): use `lags=[1, 4]` or `[1, 12]`.
3. Mean reversion: `mean_reverting_level` = b0/(1 - sum b). Forecast with `autoregressive_forecast` (chain
   rule; uncertainty grows with the horizon).
4. Compare models out of sample with `forecast_rmse` (lowest wins). Check for ARCH with `arch1_test`.

## 4. Judgment framework (curriculum)

- **Time value (L1 R5):** A rate is a required return, a discount rate or an opportunity cost, built from the
  real risk-free rate plus inflation, default, liquidity and maturity premiums. Match the rate to the risk and
  currency of the flows, and to their frequency. KB: `.../SS02-R05-the-time-value-of-money.md`.
- **NPV over IRR (L1 R6):** NPV measures the wealth added; IRR assumes reinvestment at the IRR and fails on
  scale, timing and non-conventional flows. With mutually exclusive projects follow NPV.
  KB: `.../SS02-R06-discounted-cash-flow-applications.md`.
- **Which return (L1 R6; L3 performance evaluation and GIPS):** TWR judges the manager because the client
  controls the flows; MWR shows the investor's own experience. GIPS requires TWR.
- **Which average (L1 R7):** geometric for past compound performance and multi-period forecasts, arithmetic
  for a single-period expected return, harmonic for cost averaging.
  KB: `.../SS02-R07-statistical-concepts-and-market-returns.md`.
- **Risk is more than standard deviation (L1 R7, R9):** check skewness, excess kurtosis and downside
  semideviation; Sharpe assumes symmetric returns and misleads for option-like strategies ("picking up coins in
  front of a bulldozer"). Safety-first suits a hard return floor.
- **Probability discipline (L1 R8):** list mutually exclusive, exhaustive scenarios; make probabilities
  consistent (total probability rule); update with Bayes rather than intuition.
  KB: `.../SS02-R08-probability-concepts.md`.
- **Sampling and bias (L1 R10):** data-mining, sample-selection, survivorship, look-ahead and time-period
  biases invalidate significance; prefer out-of-sample evidence and an economic rationale.
  KB: `.../SS03-R10-sampling-and-estimation.md`.
- **Statistical vs economic significance (L1 R11):** a significant result can vanish after transaction costs,
  taxes and risk adjustment (Dow-10 example). KB: `.../SS03-R11-hypothesis-testing.md`.
- **Regression assumptions (L2 R11-R12):** linearity, independent regressors, zero-mean homoskedastic
  uncorrelated normal errors. Violations: heteroskedasticity (use robust/White standard errors), serial
  correlation (Hansen/Newey-West standard errors or a different model), multicollinearity (drop or combine
  variables). Misspecification: omitted variables, wrong functional form, pooled data from different
  populations, lagged dependent variable with serial correlation, forecasting the past, measurement error.
  KB: `Level-2/.../SS03-R11-correlation-and-regression.md`, `.../SS03-R12-multiple-regression-and-issues-in-regression-analysis.md`.
- **Time series (L2 R13):** AR models need covariance stationarity. A random walk (unit root, b1 = 1) has no
  mean-reverting level: test with Dickey-Fuller and model first differences. Regressing one non-stationary
  series on another is valid only if they are cointegrated. Coefficients can be unstable across regimes:
  prefer the sample period that reflects the current regime. KB: `.../SS03-R13-time-series-analysis.md`.

## 5. Interpretation, red flags and exam traps

How a senior analyst reads the output:
- State the result in the decision's units ("the annuity is worth 2.27m today versus a 2.0m lump sum"),
  then the assumption that drives it (rate, timing, distribution).
- A p-value is the smallest alpha at which H0 is rejected; it is not the probability H0 is true. "Fail to
  reject" is not "accept".
- Report the interval, not just the point: confidence intervals for means, prediction intervals for forecasts.
- R-squared says how much variation is explained, not whether the model is right. Use adjusted R-squared to
  compare models with different numbers of regressors on the same y and sample.

Red flags:
- IRR quoted on cash flows with more than one sign change; `irr` returns `note` with the count.
- MWR far from TWR: large flows at good or bad times; do not credit or blame the manager for timing.
- Excess kurtosis > 1 or strong skew in 100+ observations: normal-based VaR and Sharpe understate tail risk.
- High R-squared and significant F with insignificant t-stats: multicollinearity (check VIF).
- DW well below 2 or significant residual autocorrelations: standard errors too small, t-stats inflated.
- Breusch-Pagan rejects: t-tests and F-test unreliable until standard errors are corrected.
- A great in-sample fit with a poor out-of-sample RMSE: overfitting or regime change.
- An AR coefficient near 1, a trend in the mean or variance: probable unit root.

Common mistakes and CFA exam traps:
- Sample (n - 1) versus population (n) divisors; the engine reports both in `descriptive_statistics`.
- One- versus two-tailed: a two-tailed test at alpha uses alpha/2 in each tail; the F-test of overall
  regression significance and the Breusch-Pagan test are one-tailed.
- t versus z: unknown variance and a small sample means t with n - 1 df; paired tests use n - 1 pairs;
  pooled tests n1 + n2 - 2; correlation and simple regression n - 2; multiple regression n - k - 1.
- Stated versus effective rates, periodic versus annual; annuity due (BGN) versus ordinary; the first payment
  of a perpetuity starting at t is valued at t - 1.
- Bank discount yield uses face value and 360 days and understates the return; discount cash flows with the HPY.
- Geometric mean <= arithmetic mean (equal only with no variability); harmonic <= geometric.
- Chebyshev holds for any distribution (at least 75% within 2 sd); the normal's 95% within 1.96 sd does not.
- Negative Sharpe ratios cannot be ranked; Roy's ratio ranks by the lowest shortfall probability.
- Positive skew: mean > median > mode; leptokurtic: fatter tails than normal.
- Continuously compounded returns add across periods; holding-period returns compound. Volatility scales with
  the square root of time.
- Correlation measures only linear association; outliers and spurious correlation mislead; correlation is
  not causation.
- Dummy-variable trap: n categories need n - 1 dummies with an intercept.
- Durbin-Watson is invalid with a lagged dependent variable; use residual autocorrelation t-tests for AR models.
- Log-linear trend forecasts must be exponentiated; a linear trend on exponential growth leaves patterned residuals.
- Type I error = rejecting a true H0 (probability alpha); Type II = failing to reject a false H0; power = 1 - P(Type II).

## 6. Deliverable outline (client or IC memo)

1. **Question and answer first**: the decision and the headline number with its unit and date
   (e.g. "NPV +89.6m at an 11% cost of capital; IRR 14.5%; recommend proceeding").
2. **Inputs and assumptions**: data source and period, rate and compounding, sample size, frequency,
   any flows netted or excluded; what was assumed rather than given.
3. **Method**: the engine calls in order with their key outputs (a short table), and the curriculum basis.
4. **Results**: point estimates with intervals; test statistic, critical value, p-value and decision for
   every claim tested; for regressions the coefficient table, R-squared/adjusted R-squared, F and SEE.
5. **Diagnostics and robustness**: sensitivity to the rate or key input (re-run the engine at +/- 1 point),
   regression diagnostics (BP, DW or residual autocorrelations, VIF), out-of-sample RMSE, Monte Carlo seed.
6. **Risks and limitations**: distribution assumptions, small samples, biases (survivorship, look-ahead,
   data mining), regime change, multiple IRRs, statistical versus economic significance.
7. **Recommendation and next steps**: the action, triggers to revisit it, data still needed.
