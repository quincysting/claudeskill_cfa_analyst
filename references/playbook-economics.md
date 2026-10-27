# Playbook: economics, growth and currencies

Use this playbook when a question turns on how an economy, a market or a currency behaves: elasticities and pricing,
market power, GDP and inflation, policy settings, long-run growth, trade, or exchange rates. Every number comes from
`cfa_calc.econ` (`python scripts/cfa.py list economics`). The judgement around those numbers comes from the curriculum
frameworks in section 4.

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
| Pricing and demand | "Will a price rise lift revenue?" "Are these goods substitutes?" "What does a tax or price cap do?" |
| Firm economics | Breakeven or shutdown, profit-maximizing output, cost curves, input mix, market power, merger screening (HHI) |
| Macro read | GDP by the expenditure and income approaches, deflator or CPI inflation, sectoral balances, multipliers, where are we in the cycle? |
| Policy stance | Neutral rate, Taylor rule, the policy mix and what it means for bonds, equities and the currency |
| Growth and long-horizon CME | Potential GDP, growth accounting, steady state, sustainability of growth, equity returns linked to GDP |
| Trade | Comparative advantage, tariff and quota welfare, trade balance response to the currency (Marshall-Lerner, J-curve) |
| FX | Cross rates, bid-offer and arbitrage, forwards and forward points, mark-to-market of forwards, parity-based forecasts, carry trades, real exchange rates and fair value |

Not here: Grinold-Kroner, Singer-Terhaar and other capital market expectation models (portfolio module), bond
yield mathematics (fixed income), and currency hedging of portfolios (portfolio and derivatives).

## 2. Inputs to gather

- **Quote convention first.** Write every pair as PRICE/BASE ("USD/EUR 1.10" means 1 EUR costs 1.10 USD). Note bid and
  offer, the pip size (0.0001; 0.01 for JPY) and the points scale (10,000; 100 for JPY).
- **Rates.** Use annualized decimals and the day count (Actual/360 for money markets, Actual/365 for GBP). Know whether
  each rate is nominal or real, and whether a figure is a forecast or a trend.
- **Demand and supply.** For a linear function, collect the intercept and the slopes, and confirm the form:
  Q = a + bP for the R13 functions, or the inverse P = a - bQ for the R16 market-structure functions. For a schedule,
  collect quantities with prices, TR or TC.
- **National accounts.** Use one currency and one units basis (millions), and say whether figures are nominal or real.
  Watch for the statistical discrepancy and for government investment recorded separately.
- **Growth.** Collect the capital share alpha (1 minus the labor share of factor cost), growth in capital, labor
  (hours) and TFP, the saving and depreciation rates, and labor force growth.
- **Policy.** Collect the neutral real rate, the inflation target, current or forecast inflation, the output gap or
  the growth gap, and the actual policy rate.
- **Trade.** Collect export and import values, demand elasticities as positive magnitudes, output per worker for the
  comparative advantage, and the world and tariff prices with quantities at each.

## 3. Quantitative workflow

Run each step through the engine and print the intermediate values (conventions.md). Round only when presenting.

**A. Demand, pricing and welfare (L1 R13)**
1. Use `elasticity_point` for an elasticity from a slope, or `elasticity_arc` from two observations. Use
   `demand_function_elasticities` for own-price, income and cross-price elasticities at once, with labels: elastic
   or inelastic, normal or inferior, substitutes or complements.
2. Use `total_revenue_test` to see whether total revenue rose or fell when price changed.
3. Use `supply_demand_equilibrium` for P*, Q*, consumer and producer surplus, and the excess demand at any price.
   Use `consumer_surplus_linear` for demand alone (two-part tariff pricing).
4. Use `price_control_effects` for a floor or ceiling (deadweight loss, shortage or glut) and `per_unit_tax_effects`
   for tax incidence. The burden falls on the steeper (less elastic) side.

**B. The firm and market structure (L1 R15-R16)**
1. Use `revenue_schedule` for TR, AR and MR, and `cost_schedule` for TC, AFC, AVC, ATC and MC. Shutdown is at min AVC;
   the perfect-competition breakeven price is min ATC.
2. Use `profit_schedule` for the breakeven points (lower and upper) and the profit maximum (MR = MC).
   `shutdown_decision` gives the short-run versus long-run call (TR against TVC and TC).
3. Use `product_schedule` for AP, MP and the onset of diminishing returns. `input_productivity_ratios` gives MP/P,
   MRP and MRP/P; hire first the input with the highest ratio, and the optimum is reached when every MRP/P = 1.
4. For pricing power, use `monopoly_price_from_elasticity` (MR = P(1 - 1/|E|)), `linear_market_structure_outcomes`
   (competitive versus Cournot versus monopoly) and `monopoly_optimum_quadratic_cost`.
5. For merger screening, `concentration_measures` gives CR_N and HHI before and after the merger (`hhi_points` is on
   the x10,000 scale).

**C. Macro accounts, prices and policy (L1 R17-R19, L2 R14, L3 R17)**
1. Use `gdp_expenditure_approach` and `gdp_income_approach` (they should reconcile up to the statistical
   discrepancy), and `personal_income_measures` for personal income, taxes, household saving and the saving rate.
2. Use `gdp_deflator` for the deflator, inflation and real growth; `price_indices` for Laspeyres, Paasche and Fisher
   indices (Laspeyres overstates cost-of-living inflation through substitution bias); and `labor_market_rates`.
3. Use `sectoral_balances` for S = I + (G - T) + (X - M): who finances a fiscal deficit, private saving or foreign
   saving (a trade deficit)?
4. Use `is_curve_linear` for a structural IS model and its spending multiplier; `fiscal_multiplier` gives
   1/[1 - MPC(1 - t)] (the balanced-budget multiplier is 1).
5. On the money side, use `money_multiplier`, `quantity_theory_of_money` / `quantity_theory_growth`, and
   `fisher_relation` (exact and approximate; omit inflation to get breakeven inflation from nominal and real yields).
6. For policy stance, `neutral_policy_rate` compares the policy rate with neutral. `taylor_rule_rate` is the L2 form
   (inflation and output gap); `taylor_rule_growth_form` is the L3 CME form (growth and inflation forecasts against
   trend and target). `unit_labor_cost` shows whether wages are running ahead of productivity.

**D. Growth (L1 R17, L2 R15, L3 R18)**
1. `potential_gdp_growth` is labor force growth plus labor productivity growth, with the growth gap against actual or
   forecast GDP.
2. `growth_accounting` gives output growth from the inputs, or the Solow residual (TFP) from output growth. It reports
   capital deepening and the factor contributions.
3. `capital_stock_path` builds the capital stock by perpetual inventory when only GDP, the investment share and the
   depreciation share are known.
4. `neoclassical_steady_state` gives sustainable growth theta/(1 - alpha) + n and the steady-state Y/K;
   `neoclassical_transition_growth` gives growth off the steady state after a change in the saving rate, with the
   level effects on k and y.
5. `cobb_douglas_output` gives output, y, k, MPK and MPL. `gdp_compound_growth` gives multi-decade growth
   comparisons. `equity_growth_decomposition` splits equity price growth into nominal GDP growth, the change in the
   profit share and the change in P/E.

**E. Trade (L1 R20-R21)**
1. `comparative_advantage` gives opportunity costs, comparative and absolute advantage, and the range of terms of
   trade at which both countries gain.
2. `tariff_welfare_effects` gives the consumer surplus loss, the producer surplus gain, revenue (or quota rent) and
   the deadweight loss.
3. `marshall_lerner_condition` gives the ML value, whether it is satisfied, and the export share needed to satisfy it.

**F. Currencies (L1 R21, L2 R14)**
1. Spot: `fx_cross_rate` derives a cross rate from any two quotes, inverting a leg with bid and offer swapped.
   `fx_triangular_arbitrage` tests a dealer's cross quote. `fx_bid_offer_spread` and `fx_percent_change` give the
   spread and the base-currency and price-currency moves, which are not symmetric.
2. Forwards: `fx_forward_cip` gives the CIP forward, premium and points, and the arbitrage direction if a market
   forward is given. `fx_forward_outright` gives all-in bid and offer from points, a percentage premium, or an FX
   swap at mid spot. `fx_forward_mark_to_market` prices the offsetting trade and discounts at the rate of the
   cash-flow currency.
3. Expectations: `uip_expected_spot` (a risk-neutral benchmark; the forward is not a good forecast),
   `ppp_relative_expected_spot` and `ppp_absolute` (long-run anchor and misvaluation), and
   `international_fisher_effect`.
4. Real rates and value: `real_exchange_rate` and `real_exchange_rate_change` measure purchasing power.
   `real_fx_rate_model` gives q = q_bar + (r_B - r_P) - (phi_B - phi_P); feed it real-rate gaps from
   `taylor_rule_rate`.
5. Strategy: `carry_trade_return` gives the net return of the funding/investing pair (quoted FUNDING/INVEST).

## 4. Qualitative frameworks from the curriculum

**Business cycle (L1 R18; L3 R17 Sec 4.1).** The L3 phases are initial recovery, early upswing, late upswing,
slowdown and recession (L3 R17 p.66-67). Inventories and capital spending swing most. Use indicators by timing:
leading (building permits, the yield spread, new orders, equity prices), coincident (payrolls, industrial
production) and lagging (unemployment duration, unit labor cost, inventory-to-sales ratio). Unemployment lags
because firms hoard labor. Inflation accelerates late in the cycle once the output gap closes. Map each phase to
markets: in the late upswing short rates rise, bond yields climb and equities are volatile; in recession short rates
fall, the curve steepens and equities bottom before the economy does. The level of rates matters, not only their
direction: a cut from a high level stimulates more than a cut near zero (liquidity trap).

**Monetary and fiscal interaction (L1 R19 Sec 4, p.420-423).** The four policy mixes:

| Mix | Output | Interest rates | Composition |
|---|---|---|---|
| Easy fiscal, tight monetary | up | up | public sector share rises |
| Tight fiscal, easy monetary | up/mixed | down | private sector share rises |
| Easy, easy | strongly up | down if the monetary effect dominates | both sectors grow |
| Tight, tight | down | up if the monetary effect dominates | both shrink |

Add Ricardian equivalence: a deficit financed by debt may not stimulate if households save for future taxes. Fiscal
policy works with lags (recognition, action, impact), and the structural (cyclically adjusted) deficit, not the
headline deficit, shows the stance. Monetary policy loses traction at the zero bound and when credibility is low;
quantitative easing is the fallback. Monetary policy cannot have a fixed exchange rate, free capital flows and
independence all at once (the impossible trinity, L2 R14 p.578).

**FX forecasting approaches.** Use them together; each answers a different horizon.
- *Parity conditions (L2 R14 Sec 3).* CIP is an arbitrage condition and always holds. UIP, PPP and the international
  Fisher effect are equilibrium conditions that fail in the short run but anchor the long run. If all held, the
  forward rate, the rate differential and the inflation differential would each equal the expected change in spot
  (Exhibit 4, p.554-555).
- *PPP and fair value (L2 R14 Sec 3.2; L3 R17 Sec 4.6.9.1).* PPP works over five years or more and says little at
  horizons under three years. Combine it with the macroeconomic balance approach (the current account level that is
  sustainable). The four-cell box (Example 7, p.558-559): overvalued on PPP with a deficit means depreciate;
  undervalued on PPP with a surplus means appreciate; mixed signals are ambiguous.
- *Balance-of-payments flows (L2 R14 Sec 5).* Current-account channels are flow supply and demand, portfolio balance
  and debt sustainability. Capital flows dominate in the short and medium term: real rate differentials, carry and
  equity flows. Persistent deficits point to eventual depreciation; surplus countries face pressure to appreciate.
- *Asset market and Mundell-Fleming (L2 R14 Sec 6.1, p.577-580).* With high capital mobility, restrictive monetary
  policy means appreciation and expansionary fiscal policy means appreciation. Easy fiscal with tight money is the
  most bullish mix; tight fiscal with easy money is the most bearish; uniform mixes are ambiguous. With low capital
  mobility trade flows dominate: a uniformly restrictive mix is bullish and a uniformly expansionary mix is bearish.
  Long run (portfolio balance, Sec 6.5): sustained deficits raise the required risk premium, and the currency gives
  back its short-run gains.
- *Monetary approach and Dornbusch (Sec 6.2).* With flexible prices, money growth leads to proportional depreciation.
  With sticky prices, a monetary expansion makes the currency overshoot its long-run PPP path in the short run.
- *Taylor-rule view (Sec 6.3, Eq. 17).* Inflation above target means a higher real policy rate and a stronger
  currency, the opposite of the PPP intuition. Judge a currency by its expected real-rate differential and its risk
  premium.
- *L3 CME summary (L3 R17 Sec 4.6.9, p.113-115).* Four approaches are used: PPP, relative economic strength
  (investment flows), capital flows (FDI and equity; can reverse the usual rate-currency link), and saving-investment
  imbalances. Also weigh government intervention, which mostly fails against fundamentals without capital controls.
- *Technical, flow and positioning signals (L2 R14 Sec 9).* Use these for timing and crowding; carry trades carry
  crash risk (negative skew) when volatility spikes.

**Growth analysis (L2 R15; L3 R18 Sec 2).** First check the preconditions: saving and investment, financial markets,
property rights and rule of law, openness, education and health, and the tax and regulatory regime. Then decompose
growth into labor (population, participation, hours, immigration), capital deepening, which hits diminishing
returns near the steady state, and TFP, the only source of sustained per-capita growth in the neoclassical model.
In endogenous growth, investment in R&D and human capital can raise the growth rate permanently through spillovers.
Convergence: absolute (all countries), conditional (same fundamentals) or club (only countries with the right
institutions). A capital-deepening-driven economy, such as Spain 2000-2009 or China, should be expected to slow
toward its steady state. In the long run, equity price growth tracks nominal GDP; profit share and P/E cannot trend
forever (L2 R15 p.629-630).

**Regulation (L2 R16, qualitative).** Classify each rule: prudential, conduct, competition (antitrust: HHI, abuse of
dominance), or for externalities and information asymmetry. Assess intended effects against unintended ones
(regulatory capture, regulatory arbitrage, competition between regulators) and against compliance cost. Use
cost-benefit analysis, which is easier to apply in retrospect than in prospect. The value effects reach industries
through compliance costs, barriers to entry (incumbents can benefit) and pricing limits.

## 5. Interpretation, red flags and exam traps

- **Quote direction.** A rise in USD/EUR is a EUR appreciation and a USD depreciation, and the two percentages differ
  (`fx_percent_change`). "The forward is at a discount" refers to the base currency.
- **Inverting a quote** swaps bid and offer: the inverse bid is 1/offer. A cross rate made by multiplying two bids
  without first aligning the currencies is the classic wrong answer.
- **Which side to deal on.** Selling the base currency uses the bid (spot and points); buying it uses the offer. An
  FX swap uses mid spot with sided points. Forward points are not annualized.
- **CIP versus UIP.** CIP is arbitrage (the hedged return equals the domestic rate). UIP is a risk-neutral
  expectation that empirically fails, which is why carry trades pay. The higher-yield currency trades at a forward
  discount.
- **Mark-to-market currency.** Discount at the interest rate of the currency the net cash flow is in.
- **Elasticity signs.** Own-price elasticity is negative; the curriculum compares magnitudes. Income elasticity
  separates normal (> 0) from inferior (< 0) goods; cross-price elasticity separates substitutes (> 0) from
  complements (< 0). Total revenue moves with price only when demand is inelastic, and a monopolist never prices on
  the inelastic part of the demand curve.
- **Linear-demand forms.** R13 writes Q = a + bP; R16 often inverts to P = a - bQ. MR has twice the slope of the
  inverse demand curve.
- **Shutdown versus exit.** Operate in the short run if TR covers TVC; exit in the long run unless TR covers TC.
  Ignore fixed (sunk) costs for the shutdown decision.
- **Concentration measures.** CR_N barely moves when large firms merge; HHI moves. Neither measure captures entry
  barriers or contestability.
- **Real versus nominal.** Use the GDP deflator for economy-wide prices and CPI for consumers. Nominal growth is
  approximately real growth plus inflation; use the exact form (`fisher_relation`) for large rates.
- **Multipliers** assume spare capacity and fixed prices. Near full employment, crowding out and inflation erode
  them.
- **Taylor rule forms.** The L2 form adds current inflation to the neutral real rate; the L3 form starts from a
  neutral nominal rate and uses 0.5 weights on growth and inflation gaps. Do not mix them.
- **Steady state.** Higher saving raises the level of k and y, not the steady-state growth rate. The growth jump
  after the change is transitional.
- **Marshall-Lerner.** Inelastic trade (as for commodities) can make a depreciation worsen the balance. Even when the
  condition holds, the J-curve means the balance may worsen first.
- **Real exchange rate.** A higher q (PRICE/BASE x CPI_base/CPI_price) is a real appreciation of the base currency.
  For a resident of the price-currency country, foreign goods have become dearer.

## 6. Deliverable outline

1. **Question and conclusion:** the answer in one or two sentences, for example "AUD is about 10% overvalued on PPP,
   carry still positive; expect depreciation over 3-5 years, not 12 months".
2. **Data and conventions:** sources and dates, the quote convention (PRICE/BASE), units, the day count, and which
   figures are forecasts or assumptions.
3. **Calculations:** each engine call with its inputs and key outputs, in a table. State the rounding used.
4. **Framework reading:** the cycle phase, the policy mix and stance, the growth drivers, and the FX approach used
   for each horizon.
5. **Implications:** for rates, the curve, equities, credit and the currency; include the scenario or sensitivity
   that would change the call.
6. **Risks and red flags:** the parity conditions that are failing, positioning and crowding, policy credibility,
   data revisions and structural breaks.

## 7. Knowledge-base paths (E:/CLAUDE/CFA/MD)

| Topic | Level and reading | Path |
|---|---|---|
| Demand, supply, elasticity, surplus | L1 R13 | `Level-1/03-Economics/Curriculum-2014/SS04-R13-demand-and-supply-analysis-introduction.md` |
| Consumer demand, income and substitution effects | L1 R14 | `Level-1/03-Economics/Curriculum-2014/SS04-R14-demand-and-supply-analysis-consumer-demand.md` |
| Firm costs, breakeven, productivity | L1 R15 | `Level-1/03-Economics/Curriculum-2014/SS04-R15-demand-and-supply-analysis-the-firm.md` |
| Market structures, concentration | L1 R16 | `Level-1/03-Economics/Curriculum-2014/SS04-R16-the-firm-and-market-structures.md` |
| GDP, AD-AS, growth sources | L1 R17 | `Level-1/03-Economics/Curriculum-2014/SS05-R17-aggregate-output-prices-and-economic-growth.md` |
| Business cycles, inflation, indicators | L1 R18 | `Level-1/03-Economics/Curriculum-2014/SS05-R18-understanding-business-cycles.md` |
| Monetary and fiscal policy | L1 R19 | `Level-1/03-Economics/Curriculum-2014/SS05-R19-monetary-and-fiscal-policy.md` |
| Trade, capital flows, BoP | L1 R20 | `Level-1/03-Economics/Curriculum-2014/SS06-R20-international-trade-and-capital-flows.md` |
| FX basics, forwards, regimes, ML | L1 R21 | `Level-1/03-Economics/Curriculum-2014/SS06-R21-currency-exchange-rates.md` |
| FX determination and forecasting | L2 R14 | `Level-2/03-Economics/Curriculum-2014/SS04-R14-currency-exchange-rates-determination-and-forecasting.md` |
| Economic growth | L2 R15 | `Level-2/03-Economics/Curriculum-2014/SS04-R15-economic-growth-and-the-investment-decision.md` |
| Regulation | L2 R16 | `Level-2/03-Economics/Curriculum-2014/SS04-R16-economics-of-regulation.md` |
| CME: cycle, Taylor rule, FX approaches | L3 R17 | `Level-3/05-Economics-and-Capital-Market-Expectations/Curriculum-2014/SS06-R17-capital-market-expectations.md` |
| Growth accounting to equity value | L3 R18 | `Level-3/05-Economics-and-Capital-Market-Expectations/Curriculum-2014/SS07-R18-equity-market-valuation.md` |

Study-note versions (Kaplan 2014, Schweser 2013) sit beside each L1 curriculum folder and are useful for concise
summaries. 2013 curriculum copies use the same reading numbers at L1 and L2 (L3 numbering differs: R18-R20). Never
open `Level-1/Exams/`; it is held out for evaluation.
