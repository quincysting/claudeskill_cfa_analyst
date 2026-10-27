# Playbook: portfolio planning, asset allocation and private wealth

Use this playbook to build or review an investment policy statement (IPS), set capital market expectations, recommend a
strategic asset allocation, and handle taxes, estates, concentrated positions, institutional clients, rebalancing and
trading. Every number comes from the engine: `python scripts/cfa.py list portfolio` and `python scripts/cfa.py list pwm`
(modules `portfolio`, `portfolio_allocation`, `portfolio_trading`, `pwm`, `pwm_planning`). Section 4 covers the
judgement around those numbers. Performance, GIPS and VaR are in `playbook-performance-risk.md`.

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
| IPS | "What return does this client need? Can they bear the risk? What are the constraints?" |
| Portfolio math | Portfolio risk and return, utility, the CAL/CML, CAPM, beta, APT, the minimum-variance or tangency mix |
| Capital market expectations | "What should equities and bonds return over 10 years in this market?" |
| Strategic allocation | "Which allocation meets the return target with acceptable risk?" (MVO, corner portfolios, safety-first, ALM, Black-Litterman) |
| Private wealth | After-tax accumulation, asset location, tax-loss harvesting, gifts vs bequests, core capital, human capital, retirement income, concentrated stock |
| Institutions | DB and DC pensions, foundations, endowments, insurers, banks |
| Implementation | Rebalancing policy (corridors, CPPI, constant mix) and trading costs (implementation shortfall, effective spread, VWAP) |

## 2. Inputs to gather

- **Client facts.** Age and stage of life, family, health, employment and its stability, and income and its correlation
  with markets (human capital). Collect assets by account type (taxable, tax-deferred, tax-exempt) with cost bases,
  liabilities, and spending needs in money, with timing. Note one-off outflows (house, gift, business), the emergency
  reserve, pensions and other income, and expected inheritances.
- **Willingness.** Collect the client's statements, questionnaire results, past behaviour in drawdowns and any biases
  observed (section 4.4).
- **Taxes.** Collect rates on interest, dividends and realized gains, the wealth tax, the estate and gift taxes, the
  withdrawal rate T_n of tax-deferred accounts, and the shares of return taxed each way (p_i, p_d, p_g).
- **Institution facts.**
  - DB pension: plan assets, liabilities (PBO) and discount rate, liability duration, workforce age, retired-lives
    share, sponsor balance sheet and profitability, and correlation of the sponsor's business with plan assets.
  - Foundation or endowment: spending rule, payout requirement, cost inflation (for example HEPI), and the share of the
    operating budget it funds.
  - Insurer: liability duration, reserve crediting rate, surplus.
  - Bank: durations and values of assets and liabilities.
- **Market inputs.** Use decimals for expected returns, volatilities, correlations (or covariances), the risk-free rate
  and inflation. For CME collect dividend yield, buybacks, real growth, the P/E view, the GIM Sharpe ratio, the degree
  of integration and illiquidity premiums.
- **Units check.** The L3 curriculum writes utility as E(R) - 0.005 R_A sigma^2 with percents. The engine uses decimals
  and 0.5 A sigma^2, which gives the same ranking: 9.7% and 15% become `0.097` and `0.15`.

## 3. Quantitative workflow

Run each step in the engine and print the intermediate values. Round only when presenting.

**A. Return objective and risk capacity (L3 R10, R15)**
1. Use `ips_required_return` for the spending-based requirement. Pass `spending_need`, `portfolio_value`,
   `other_income`, `reserved_outflows`, `inflation`, `fees` and `tax_rate`. Use `method="additive"` for the
   individual cases and `"multiplicative"` for foundations and endowments (exact compounding).
2. Use `ips_required_return_for_goal` for an accumulation goal ("GBP 2m in 18 years while withdrawing GBP 26k a year")
   or a funding target such as a pledge or a pension shortfall. It gives the after-tax rate and the pretax gross-up.
3. Use `return_conversions` to move between pretax and after-tax, and between nominal and real.
4. Use `estate_core_capital` to separate core capital from excess capital (survival-weighted PV of spending, joint
   survival for couples). Use `retirement_annuity` to price an income stream or a sustainable withdrawal.
   Use `retirement_monte_carlo` (seeded) for probability of success and ruin by year.
5. Use `human_capital_value` for PV of earnings, economic net worth and the human-capital share. Bond-like human capital
   supports more equity in the financial portfolio.

**B. Portfolio math (L1 R43-R44, L2 R57)**
1. Use `portfolio_return_risk` for E(R), sigma and risk contributions of N assets. Inputs are `cov`, or `sds` with
   `corr` (a scalar works for two assets).
2. Use `global_minimum_variance_portfolio` and `tangency_portfolio` for the unconstrained closed forms.
   Use `capital_allocation_line` for points on the CAL or CML. Use `optimal_risky_allocation` for
   y* = (E(R_p) - r_f)/(A sigma^2).
3. Use `utility_ranking` to pick among candidate portfolios. For CAPM and SML alpha use `capm_sml_expected_return`.
   Use `capm_beta_from_moments` for beta from covariance or correlation, plus adjusted beta. Use
   `systematic_risk_decomposition` for systematic vs nonsystematic risk.
4. Use `apt_expected_return` for APT. Use `apt_solve_factor_premiums` to back out rf and lambdas from portfolios.
   Use `apt_arbitrage_check` for an arbitrage or tracking portfolio. Use `factor_active_risk_return` for active return
   and active risk by factor.

**C. Capital market expectations (L3 R17-R18)**
1. Equities: `cme_grinold_kroner` (income + nominal earnings growth + repricing) and `cme_gordon_growth_erp`.
2. Risk premiums across markets: use `cme_singer_terhaar`. It blends the integrated premium
   (sigma x rho x Sharpe_GIM) and the segmented premium (sigma x Sharpe_GIM) by degree of integration, then adds
   illiquidity.
3. Bonds: `cme_bond_risk_premium_buildup`.
4. Covariances: use `cme_multifactor_covariance` (BFB' + residual) and `cme_shrinkage_estimate` toward a target.
   Use `cme_unsmoothed_volatility` to de-smooth appraisal-based real estate or private equity returns before using them.

**D. Strategic asset allocation (L3 R19)**
1. Use `mvo_optimize` to optimize. `long_only=True` is the default (SLSQP; deterministic). The objective is either
   `target_return` or `risk_aversion`. For bounds use `min_weights` or `max_weights`. This reproduces the curriculum's
   corner-portfolio frontiers.
2. Use `corner_portfolio_combination` to interpolate between adjacent corner portfolios. Its SD is a linear upper bound.
   Treat the risk-free asset as a corner with SD 0.
3. Choose with `utility_ranking` (risk aversion) or `asset_allocation_safety_first` (Roy, shortfall threshold R_L).
   Check the shortfall probability. For pensions use `surplus_return_risk` (surplus return, surplus volatility,
   funded ratio, surplus utility).
4. Use `add_asset_class_test` to decide whether a new asset class adds value (Sharpe_new > Sharpe_p x corr).
5. Robustness: `mvo_resampled_weights` (seeded resampling). Use `black_litterman_implied_returns`
   (Pi = lambda Cov w) and then `black_litterman_posterior` with views (P, Q, Omega).

**E. Taxes and estates (L3 R11-R12)**
1. Use `tax_future_value` with `regime` set to accrual, deferred, wealth, blended, tax_deferred_account or tax_exempt.
   It returns FV, tax drag and accrual-equivalent return and tax rate. For blended returns supply p_i, p_d, p_g, the
   rates and the cost basis.
2. Use `after_tax_asset_allocation` to restate holdings after tax (TDA x (1 - T_n)) and recompute the weights.
   Use `tax_loss_harvesting` for current savings and the hold-vs-harvest comparison.
3. Use `gift_vs_bequest_relative_value` with `gift_type` set to tax_free, recipient_pays, donor_pays or charitable.
   Use `generation_skipping_value` and `double_taxation_rate` (credit, exemption, deduction).

**F. Concentrated positions (L3 R13)**
1. Use `concentrated_position_strategies` to compare collar floor and cap, payoffs at chosen terminal prices,
   monetization proceeds (LTV x value) with carry cost, and the after-tax outright sale.

**G. Institutions (L3 R15-R16)**
1. Use `pension_funded_status` for surplus and funded ratio, valuing liabilities from benefits if needed.
   Use `ips_required_return_for_goal` for the return needed to close a gap given contributions.
2. Use `endowment_spending_rule` (simple, rolling, geometric smoothing) and `ips_required_return`
   (multiplicative: spending x inflation x costs).
3. Use `bank_leverage_adjusted_duration_gap` for D_A - (L/A) D_L and the change in equity value for a rate shock.

**H. Rebalancing and trading (L3 R32-R33)**
1. Use `rebalancing_corridor_check` for breaches and trades. `rebalance_to` is target, edge or halfway; widths can be
   absolute or `corridor_pct_of_target`.
2. Use `cppi_exposure` for m(A - F). Use `rebalancing_strategy_paths` to compare buy-and-hold, constant-mix and CPPI
   paths over a price path.
3. Trading costs: `implementation_shortfall` (explicit + realized P/L + delay + missed trade; per-fill arrival prices;
   sells flip sign; market-adjusted), `effective_spread` and `vwap_comparison`.

## 4. Qualitative frameworks from the curriculum

### 4.1 The IPS: objectives and constraints (L3 R10 individuals, R15 institutions; L1 R45)
- **Return objective.** Separate the *required* return (to meet goals) from the *desired* return. State it as
  nominal or real, and pretax or after-tax.
- **Risk objective.** Assess both:
  - Ability: horizon, wealth relative to needs, liquidity needs, stability of income, insurance coverage.
  - Willingness: attitudes, statements and behaviour.
  - When they conflict, the overall risk tolerance is the *lower* of the two. Educate the client where willingness is
    below ability.
  - A required return the risk tolerance cannot support means the goals must change (spend less, save more, delay).
- **Constraints (TTLLU).**
  - Time horizon: single or multistage, and stage of life.
  - Taxes: account types, regime, estate.
  - Liquidity: ongoing spending, emergency reserve (often 3-12 months), planned outflows, illiquid holdings.
  - Legal and regulatory: trusts, ERISA, the prudent investor rule, insurance regulation.
  - Unique circumstances: ESG exclusions, concentrated or restricted stock, family business, health, gifting.
- **Investor types.**

  | Investor | Return driver | Risk | Liquidity / horizon | Taxes / legal |
  |---|---|---|---|---|
  | Individual | Spending, goals, inflation | Ability and willingness, stage of life | Reserve, life events, multistage | Taxable; trusts, estate |
  | DB pension | Discount rate on liabilities (fully funded); more if underfunded and able | Surplus, sponsor strength, sponsor-plan correlation, workforce age, retired lives | Benefits net of contributions; long unless mature | Exempt; ERISA, prudent expert |
  | DC pension | Participant-chosen | Participant | Participant | Plan provides choice and education |
  | Foundation | Payout (e.g. 5%) + inflation + costs | Above average; perpetual | Spending; low | Mostly exempt; excise tax; prudent investor |
  | Endowment | Spending rule + higher-education inflation + costs | Above average if budget reliance is low | Smoothed spending | Exempt; UPMIFA |
  | Life insurer | Reserve crediting rate + spread | Low; ALM, reinvestment, credit, valuation | Disintermediation risk; segmented horizons | Taxable; heavily regulated (RBC, eligible assets) |
  | Non-life insurer | Total return; underwriting cycle | Low to moderate; shorter liabilities | High (claims uncertainty) | Taxable; regulated |
  | Bank | Spread over funding; manage LADG | Interest rate, credit, liquidity | High | Regulated (risk-based capital) |

### 4.2 Capital market expectations (L3 R17)
- **Methods.** Historical statistics (with shrinkage), DCF (Gordon, Grinold-Kroner), risk premium build-up,
  equilibrium (Singer-Terhaar), surveys and panels, and judgement. Link forecasts to the business cycle, inflation and
  monetary and fiscal policy (Taylor rule).
- **Pitfalls.**
  - Data errors and biases: transcription, survivorship, appraisal smoothing (understates volatility and correlation).
  - Regime change and nonstationarity.
  - Ex post risk as a biased proxy for ex ante risk; data-mining and time-period bias.
  - Ignoring conditioning information; misreading correlations.
  - Psychological traps: anchoring, status quo, confirming evidence, overconfidence, prudence, recallability.
  - Model uncertainty.

### 4.3 Asset allocation, rebalancing and execution (L3 R19, R32-R33)
- **Allocation.**
  - Strategic (policy) vs tactical, and asset-only vs ALM (surplus).
  - Approaches: MVO and its instability (resampling, Black-Litterman, constraints); ALM for liability-driven investors
    (pensions, insurers); experience-based rules.
  - Asset class criteria: homogeneous, mutually exclusive, diversifying, together covering most investable wealth,
    able to absorb meaningful investment.
  - Selection: the efficient mix must meet the return requirement and constraints. Then pick by utility or safety-first,
    and check liquidity and concentration.
- **Rebalancing.** Calendar vs percentage-of-portfolio corridors.
  - Wider corridors when transaction costs, risk tolerance or correlation with the rest of the portfolio are higher.
  - Narrower corridors when the asset class or the rest of the portfolio is more volatile.
  - Perold-Sharpe strategies:

    | Strategy | Payoff | Does well in | Risk tolerance |
    |---|---|---|---|
    | Buy-and-hold | Linear | Trends | Rises with wealth |
    | Constant mix | Concave (buys dips) | Oscillating markets | Proportional to wealth |
    | CPPI (m > 1) | Convex (sells dips) | Trends | Falls sharply with the cushion |

- **Trading.**
  - Explicit costs: commissions, taxes, fees. Implicit costs: spread, market impact, delay, missed trades.
  - Benchmarks: implementation shortfall is preferred for decision-to-execution cost. VWAP can be gamed and ignores
    unfilled orders.
  - Trader motivations and order styles: information (urgent, market orders), value (patient, limit orders), liquidity
    (minimize cost, crossing/algorithms), passive (low cost).
  - Algorithmic trading: logical participation (VWAP, TWAP, IS) vs opportunistic. Best execution is a process, not a
    single price.

### 4.4 Behavioral finance (L3 R7-R9)
- **Cognitive errors.** *Moderate* these through education and process.
  - Belief perseverance: conservatism, confirmation, representativeness, illusion of control, hindsight.
  - Information processing: anchoring and adjustment, mental accounting, framing, availability.
- **Emotional biases.** Usually *adapt* to these.
  - Loss aversion, overconfidence, self-control, status quo, endowment, regret aversion.
- **Rule of thumb.** Moderate when wealth is low relative to needs (standard-of-living risk), when biases are cognitive,
  or both. Adapt when wealth is high and the biases are emotional.
- **Behavioral investor types.** Passive preserver, friendly follower, independent individualist, active accumulator.
- **Goals-based investing.** Behavioral portfolio theory uses a layered pyramid (safety, income, aspiration). It
  connects to the personal, market and aspirational risk buckets of concentrated-wealth planning (R13).
- **Process biases.** Analyst forecasts (overconfidence, illusion of knowledge, anchoring), committee groupthink, and
  market anomalies.

### 4.5 Private wealth specifics (L3 R11-R14)
- **Taxes.**
  - Accrual taxation drag grows with horizon and return. Deferred capital-gains drag equals t_cg regardless of horizon.
  - Asset location: tax-inefficient assets (taxable bonds, high-turnover strategies) go in tax-advantaged accounts when
    the investor can choose.
  - TDA vs tax-exempt account: compare today's tax rate T_0 with the withdrawal rate T_n.
  - Tax-loss harvesting adds value through deferral and rate differences, not by eliminating tax.
  - Holding-period management: long-term vs short-term rates.
- **Estates.**
  - Core capital (survival-weighted spending plus reserve) versus excess capital that can be transferred.
  - Lifetime gifts beat bequests when the recipient's after-tax growth is higher, or when gift tax is below estate tax.
  - Generation skipping avoids a layer of transfer tax.
  - Other tools: trusts (revocable/irrevocable, fixed/discretionary), foundations, life insurance, valuation discounts,
    spousal exemptions.
  - Double-tax relief: credit, exemption, deduction.
- **Lifetime planning.**
  - Human capital and financial capital together form total wealth. Equity-like or market-correlated human capital
    calls for less equity.
  - Life insurance hedges the loss of human capital (mortality risk). Annuities hedge longevity risk.
- **Concentrated positions.**
  - Goals: reduce risk, raise cash, minimize tax.
  - Tools: staged sale, exchange fund (basis carries over), completion portfolio, collars and prepaid variable forwards
    with loans, short against the box, total return swaps, cross-hedges.
  - Private business: sale, leveraged recapitalization, ESOP, personal line of credit.
  - Real estate: mortgage, sale-leaseback, donor-advised fund.
  - Watch the tax treatment of hedges (constructive sale rules) and emotional attachment.

## 5. Interpretation, red flags and exam traps

- **Ability vs willingness.** Never let a high required return override a low ability or willingness. Report the
  conflict and resolve it toward the lower tolerance.
- **Additive vs multiplicative.**
  - The curriculum's individual cases add spending rate + inflation, then gross up for tax: r_pretax = r_at/(1 - t).
  - Foundations and endowments compound: (1+s)(1+i)(1+c) - 1.
  - State which you used. The two differ by roughly the cross-products.
- **Net need and base.** Subtract other income (pensions, salary, gift funds) from spending. Remove near-term outflows and
  reserves from the investable base before dividing.
- **Utility units.** 0.005 x A x sigma^2 with percents equals 0.5 x A x sigma^2 with decimals. Do not mix them.
- **Safety-first.** R_L is the minimum acceptable return (for example spending / assets). The allocation with the
  highest SF ratio wins, even if another has the higher mean.
- **Corner portfolios.** Interpolated SD overstates true risk (an upper bound). Weights interpolate exactly. An
  allocation with no asset class at zero weight is not a corner portfolio.
- **CME.**
  - Grinold-Kroner uses *real* earnings growth plus inflation. A buyback reduces shares outstanding, so dS < 0 adds
    return.
  - Singer-Terhaar adds the illiquidity premium to both the integrated and segmented premiums.
  - De-smooth appraisal data before trusting low volatility or correlation.
- **Taxes.**
  - With a cost basis below market value, subtract (1 - B) t_cg.
  - The tax drag of a deferred gain does not grow with time; accrual drag does.
  - The TDA balance is not all the client's money: the government owns T_n of it.
- **Estates.** Joint survival for a couple is p1 + p2 - p1 p2. Discount real spending at the real rate:
  (1 + nominal)/(1 + inflation) - 1.
- **Pensions.** The required return for a fully funded plan is the liability discount rate. Underfunding plus a weak
  sponsor lowers the ability to take risk. A high retired-lives share raises liquidity needs and shortens the horizon.
- **Rebalancing.** CPPI with m > 1 sells into declines and buys into rallies, so it loses in choppy markets. Constant
  mix is the opposite.
- **Implementation shortfall.**
  - Realized P/L is measured from the price when the order was released (the prior close), not the decision price.
  - The move from the decision price to the release price is the delay cost.
  - Missed-trade cost uses the cancellation (or final) price.
  - Sells flip all signs.

## 6. Deliverable outline

**Investment policy statement**
1. Client profile: facts, goals, balance sheet including human capital.
2. Return objective, with a worked calculation from the engine (state additive or multiplicative, pretax or after-tax,
   nominal or real).
3. Risk objective: ability, willingness, overall tolerance and any conflict.
4. Constraints (TTLLU), each with specifics.
5. Strategic allocation table with expected return, SD, Sharpe, shortfall probability and how it meets the objectives.
6. Rebalancing policy (corridors, review frequency) and monitoring triggers.

**Asset allocation recommendation.** Set out the CME table (method per asset class), the efficient frontier or corner
portfolios, the candidate mixes, the selection criterion (utility, safety-first or surplus), the chosen weights,
sensitivity to CME errors (resampled or BL), and implementation notes.

**Wealth or estate plan.** Show core vs excess capital, account location, tax drag by account, transfer strategy
(gift vs bequest RV, trusts, generation skipping), the concentrated-position plan, and insurance and annuity needs.

**Trade cost review.** Show the implementation shortfall decomposition by order, spreads and VWAP comparison, the
attribution of costs to delay vs impact, and recommendations (order type, algorithm, venue).

## 7. Knowledge-base paths (E:\CLAUDE\CFA\MD)

- L1 portfolio: `Level-1/06-Portfolio-Management/Curriculum-2014/SS12-R43..R45-*.md`.
- L2 portfolio: `Level-2/06-Portfolio-Management/Curriculum-2014/SS18-R57-portfolio-concepts.md`, `SS18-R60-*.md`
  (IPS). The 2013 file `SS18-R55-the-theory-of-active-portfolio-management.md` covers Treynor-Black.
- Behavioral: `Level-3/02-Behavioral-Finance/Curriculum-2014/SS03-R07..R09-*.md`.
- Private wealth: `Level-3/03-Private-Wealth-Management/Curriculum-2014/SS04-R10` (individuals), `R11` (taxes),
  `R12` (estate), `R13` (concentrated positions), `R14` (lifetime advice).
- Institutions: `Level-3/04-Institutional-Investors/Curriculum-2014/SS05-R15`, `R16`. Also
  `Curriculum-2013/SS05-R17-allocating-shareholder-capital-to-pension-plans.md`.
- CME: `Level-3/05-Economics-and-Capital-Market-Expectations/Curriculum-2014/SS06-R17`, `SS07-R18`.
- Asset allocation: `Level-3/06-Asset-Allocation/Curriculum-2014/SS08-R19-asset-allocation.md`. Also
  `Curriculum-2013/SS08-R22-the-case-for-international-diversification.md`.
- Trading and rebalancing: `Level-3/12-Trading-Monitoring-Rebalancing/Curriculum-2014/SS16-R32`, `SS16-R33`.
