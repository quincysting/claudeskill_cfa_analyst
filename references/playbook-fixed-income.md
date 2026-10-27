# Playbook: Fixed Income (Levels I-III)

Engine modules: `scripts/cfa_calc/fi_pricing.py` (prices, accrued interest, matrix pricing, horizon return),
`fi_yields.py` (YTM/YTC/YTW, periodicity, money market, FRNs), `fi_curve.py` (spot/par/forward curves, spreads,
rolldown, yield volatility), `fi_risk.py` (duration, convexity, money duration, key rates, gaps),
`fi_credit.py` (expected loss, CVA, structural/reduced-form models, ratios), `fi_trees.py` (binomial trees,
callable/putable, OAS, effective duration, capped floaters, pathwise, convertibles), `fi_securitized.py` (SMM/CPR,
PSA, mortgage cash flows, WAL), `fi_portfolio.py` (immunization, futures, return decomposition, leverage, repo,
currency hedging, tracking risk). `fixed_income.py` re-exports all of them:
`from cfa_calc import fixed_income as fi`. List with `python scripts/cfa.py list fixed_income`; read units with
`describe <name>`. Every number in an answer comes from one of these calls; show the call and its output.

Knowledge base (KB) root `E:/CLAUDE/CFA/MD` (2014 reading numbers):
- L1 `Level-1/08-Fixed-Income/Curriculum-2014/`: R52 defining elements, R53 markets and funding, R54 valuation,
  R55 risk and return, R56 credit analysis. Older framing: `Curriculum-2013/` R56-R59.
- L2 `Level-2/08-Fixed-Income/Curriculum-2014/`: R45 credit analysis models, R46 term structure and volatility,
  R47 bonds with embedded options, R48-R50 MBS/ABS and their valuation.
- L3 `Level-3/07-Fixed-Income-Portfolio-Management/Curriculum-2014/`: R20 FI PM part I (indexing, immunization),
  R21 relative value for credit, R22 FI PM part II (leverage, derivatives, international, EMD, manager selection).
- Faster review: `Kaplan-Notes-2014/`, `Schweser-Notes-2013/` next to each L1 curriculum folder.

## Contents
1. When to use
2. Inputs to gather
3. Quantitative workflow (A pricing/yields, B curve, C rate risk, D credit, E options, F MBS, G portfolio)
4. Qualitative framework (credit, curve strategies, LDI, benchmarks, relative value)
5. Interpretation, red flags and exam traps
6. Deliverable outline

## 1. When to use

- Pricing or yielding a bond, note, bill, CD, repo or FRN; reconciling a dealer quote (flat vs full price).
- Building or reading a yield curve: spot, par, forward, swap curves; spreads (G, I, Z, OAS); rolldown trades.
- Measuring and managing interest-rate risk: duration, convexity, PVBP, key rates, hedging with futures.
- Credit work: an issuer's capacity, a bond's expected loss and fair spread, rating-migration risk.
- Bonds with options: callable, putable, step-up, capped floaters, convertibles; MBS prepayment and average life.
- Liability-driven investing: immunizing a pension or insurance liability, contingent immunization, surplus risk.
- Portfolio decisions: expected-return decomposition, leverage via repo, currency hedging, benchmark selection.

## 2. Inputs to gather

| Task | Must have | Ask or assume (state it) |
|---|---|---|
| Price / yield | coupon rate, maturity, frequency, price or yield | Is the quote flat or full? Settlement date and day count (govts act/act, US corporates 30/360). Yield periodicity = coupon frequency unless told. |
| Money market | days, quoted rate, basis | Discount (bills, CP, BAs) or add-on (CDs, repo, Libor)? 360 or 365 days? |
| FRN | index, quoted margin, price or required margin, reset frequency | Flat index assumed (simplified model); between resets the next coupon is fixed. |
| Curve | par yields or coupon-bond prices of consecutive maturities, or spot/forward rates | Rate periodicity; annual vs semiannual periods. Benchmark: government or swap curve. |
| Duration / convexity | cash-flow terms and yield, or three model prices | Yield vs curve measure; shift size; option-free or not. Position size for money duration. |
| Credit | PD (or hazard), recovery/LGD, risk-free curve, coupon schedule | Annual or per-period hazard; exposure definition; recovery on exposure. |
| Embedded options | par curve, rate volatility, call/put schedule, market price | Exercise rule (call if value > call price); volatility is the key unobservable: run 2 values. |
| MBS | balance, WAC, pass-through rate, WAM, loan age, PSA or CPR | Seasoning shifts the PSA ramp. Servicing spread = WAC - pass-through. |
| Immunization / LDI | liability amounts and dates, discount curve, candidate bonds | Parallel-shift assumption; rebalancing frequency; liability valuation curve. |
| Futures overlay | portfolio value and duration, target, CTD duration and price, conversion factor | Yield beta (1 if unknown, state it). |

## 3. Quantitative workflow

**A. Pricing and yields (L1 R54).**
1. Coupon date: `bond_price` (zero: `coupon_rate=0`, fractional years allowed). Spot curve: `bond_price_spot_curve`
   (also returns the YTM). Forward curve: `fi_spot_from_forwards` then `bond_price_spot_curve`.
2. Between coupons: `bond_price_settlement` (dates + day count) or `bond_price_full_flat` (N and t/T);
   accrued interest alone: `bond_accrued_interest`. Full = flat + accrued; duration uses the full price.
3. Yield: `bond_ytm` (use `fraction` with a full price between coupons); `bond_yield_to_call_put`;
   `bond_yield_to_worst` with `call_schedule=[[year, price], ...]`; `bond_current_and_simple_yield`.
4. Compare yields only at the same periodicity: `yield_periodicity_convert` (0 = continuous; money-market
   periodicity is 365/days).
5. Odd dates, portfolios, MBS: `cash_flow_yield` (street yield = scheduled times, true yield = actual times).
6. Illiquid bond: `bond_matrix_pricing` with comparables `[[years, coupon, price], ...]`.
7. Money market: `money_market_price`, `money_market_rate`, `money_market_bey` (365-day add-on and semiannual basis).
8. FRNs: `frn_price`, `frn_discount_margin`, `frn_duration` (index duration ~ time to reset; spread duration ~ maturity).
9. Realized return: `bond_horizon_return` (reinvestment + sale; splits capital gain from the carrying value).

**B. Term structure and spreads (L1 R54, L2 R46).**
1. Bootstrap: `fi_spot_bootstrap` (par yields, or coupons + prices); it also returns discount factors and
   one-period forwards. Spot -> par (= swap rates on the swap curve): `fi_par_curve`. Forwards: `fi_forward_rate`
   ("AyBy"), `fi_forward_curve`.
2. Spreads: `bond_g_i_spread` (G over govts, I over swaps, interpolated), `bond_z_spread` (over the spot curve),
   OAS in E.3; market stress indicators: `swap_ted_ois_spreads` (swap spread, TED, Libor-OIS).
3. Riding the curve: `bond_rolldown_return` with the curve's yield at the shorter maturity; feed the rolldown into
   `bond_expected_return_decomposition`.
4. Rate volatility for trees and risk: `yield_volatility_historical` (log changes, annualize with 250/260/365).

**C. Interest-rate risk (L1 R55, L2 R46, L3 R20/R22).**
1. Option-free bond: `bond_duration_convexity` (Macaulay, modified, approximate, money duration, PVBP, convexity,
   all annualized on the full price; `dy` sets the bump for approximate measures).
2. Model prices (callables, MBS, liabilities): `fi_approx_duration_convexity(pv0, pv_minus, pv_plus, delta)` =
   effective duration/convexity; trees give these prices in E.4.
3. Scenario: `bond_price_change_estimate` (-D dy + 1/2 C dy^2; pass `price` for a money change). Works for spread
   changes with spread duration.
4. Portfolio: `bond_portfolio_duration` (market-value weights; spread duration with Treasuries at 0),
   `bond_money_duration` (position money duration, 100 bp dollar duration, PVBP).
5. Non-parallel shifts: `key_rate_durations_spot` (per bond), `key_rate_price_change` (portfolio scenario).
6. Horizon and balance-sheet gaps: `duration_gap_horizon` (Macaulay - horizon), `asset_liability_duration_gap`
   (surplus/equity duration, surplus change), `foreign_bond_duration` (country beta).
7. Keeping risk constant: `dollar_duration_rebalance` (after time/yield drift), `duration_neutral_swap` (bond swaps).

**D. Credit (L1 R56, L2 R45).**
1. First pass: `credit_expected_loss` (EL = PD x LGD; spread ~ PD x LGD).
2. Capacity: `credit_ratios` (coverage, debt/EBITDA gross and net, FFO/debt, debt/capital, leverage through each
   tranche for high yield).
3. Bond valuation with default: `credit_cva_bond` (hazard rate per year, recovery on exposure; returns VND, CVA,
   fair value, YTM and credit spread, plus the year-by-year exposure/POD/POS table).
4. Models: `credit_structural_merton` (equity = call on assets; PD from actual drift, PV of EL from risk-neutral),
   `credit_reduced_form_constant` (constant intensity: spread = lambda x gamma),
   `credit_pv_expected_loss_curve` (risk-free and credit-spread term structures, continuous).
5. Rating migration: `credit_migration_expected_return` (probability-weighted -D ds + 1/2 C ds^2).

**E. Embedded options and arbitrage-free valuation (L2 R47).**
1. `binomial_tree_calibrate(par_rates, volatility)` (annual, lognormal, r_low x e^(2 sigma k), iterative).
   Check: an option-free bond values the same as on the spot curve.
2. `binomial_tree_bond_value`: option-free, callable (`call_schedule`), putable (`put_schedule`), both, step-ups
   (`coupons=[...]`). Option value = straight - callable (call), putable - straight (put).
3. `binomial_tree_oas(price, tree, ...)`: OAS for a market price; compare with the required OAS of peers.
4. `binomial_tree_effective_duration(par_rates, volatility, ..., oas, shift)`: shifts the par curve, recalibrates,
   keeps OAS, returns V0, V+, V-, effective duration and convexity.
5. Floaters with caps/floors: `binomial_tree_capped_floater`. Pathwise check: `binomial_tree_pathwise_value`.
   Monte Carlo (concept): simulate many rate paths calibrated to the curve, value path-dependent cash flows (MBS),
   average; add a constant spread so the model price equals market (the OAS); results carry sampling error.
6. Convertibles: `convertible_bond_metrics` (conversion value, floor = max(conversion, straight), market conversion
   price and premium, income differential, payback, premium over straight value).

**F. Securitized (L2 R48-R50).**
1. Prepayment speed: `mbs_smm_cpr`, `mbs_psa_speed` (age = seasoning + month).
2. Schedule: `mbs_cash_flows` (mortgage payment, net interest, scheduled principal, prepayment, WAL); compare WAL
   across PSA speeds to show contraction and extension risk. Payment alone: `mbs_mortgage_payment`.
3. Tranches/ABS: `mbs_weighted_average_life` on any principal vector; pool stats `mbs_pool_wac_wam`.
   Tranche concepts (qualitative): sequential-pay classes redistribute prepayment timing; PAC tranches get
   stability inside the PAC band while support tranches absorb contraction and extension; credit tranching
   (senior/mezzanine/equity) redistributes default losses; Z-bonds accrue; IO/PO strips have opposite prepayment
   sensitivity (IO gains when prepayments slow).
4. MBS yield: `cash_flow_yield` on the projected cash flows; MBS duration from model prices via
   `fi_approx_duration_convexity` (expect negative convexity).

**G. Portfolio management (L3 R20-R22).**
1. Single liability: `immunization_single_liability` (required PV, target Macaulay duration = horizon, liability
   convexity). Choose assets with PV >= PV_L, duration = horizon and the lowest convexity above the liability's
   (bullet around the horizon), e.g. via `barbell_bullet_duration_match`.
2. Multiple liabilities: `immunization_multiple_liabilities` checks PV, duration match and dispersion/bracketing.
   Cash-flow matching is the zero-reinvestment-risk alternative (higher cost).
3. Contingent immunization: `contingent_immunization` (required terminal value at the safety-net rate, required
   assets at today's immunizable rate, dollar safety margin; immunize when the margin hits zero).
4. Duration overlay: `futures_duration_hedge_contracts` (BPV hedge ratio with CTD and conversion factor; target 0
   for a full hedge; yield beta for cross hedges).
5. Expected return: `bond_rolldown_return` -> `bond_expected_return_decomposition` (yield income + rolldown
   + rate/spread effect - expected credit loss + currency).
6. Breakeven: `breakeven_yield_spread_change` (yield advantage over the horizon / longer duration).
7. Leverage: `leveraged_bond_return`, `bond_repo_interest` (haircut), equity duration via
   `asset_liability_duration_gap`.
8. International: `currency_hedged_bond_return` (hedged ~ local return + i_d - i_f; hedge if the forward premium
   beats the expected currency return), `foreign_bond_duration`.
9. Indexing: `bond_index_tracking_risk`; key-rate and spread-duration matching in C.4-C.5.

## 4. Qualitative framework

**Credit analysis (L1 R56 §5, KB p.548-600).** The four Cs: *capacity* (industry structure and cyclicality,
competitive position, operating history, cash-flow generation; ratios: EBITDA/interest, EBIT/interest,
debt/EBITDA, FFO/debt, debt/capital, FCF after dividends/debt), *collateral* (asset quality, goodwill,
depreciation vs capex), *covenants* (affirmative: maintain insurance, pay taxes; negative: limits on debt,
liens, asset sales, dividends, change of control put), *character* (management strategy, track record,
accounting aggressiveness, governance, past bankruptcy). Seniority ranking drives recovery (secured > senior
unsecured > subordinated); notching widens below investment grade; agency ratings lag market spreads.
- *High yield (§7.1):* liquidity and refinancing risk, leverage through each debt tranche (`credit_ratios`),
  structural subordination at holdco, covenant strength, equity-like valuation (EV/EBITDA) cross-check.
- *Sovereign (§7.2):* institutional/political profile, economic structure and growth, external position
  (reserve currency status), fiscal flexibility and debt burden, monetary flexibility; local- vs foreign-currency
  ratings (foreign usually lower).
- *Municipal (§7.3):* general obligation (tax base, budget balance, pensions/OPEB) vs revenue bonds
  (project economics, debt-service coverage); tax-equivalent yield comparison.
- *L2 models (R45):* ratings (stable but lag), structural models (option analogy, need asset value/volatility,
  balance-sheet dependent), reduced-form (hazard from market/state variables, fits any liability structure).
  Credit spread = expected loss + liquidity premium (+ risk premium).

**Yield-curve strategies (L3 R20 §3.3, R21; L2 R46).** Shape views: level (duration), slope (steepener/flattener),
curvature (butterfly). Bullet vs barbell vs ladder: at equal duration a barbell has more convexity (gains on
large parallel moves and flattening) but higher immunization risk; a bullet wins on steepening around its
maturity. Riding the yield curve works when the curve is upward sloping and expected to stay put (implied forwards
are breakeven rates). Theories: pure expectations (forwards = expected spots), liquidity preference (term premium),
preferred habitat/segmentation. Relative value for credit (R21): sector rotation, new-issue swaps,
credit-upside/defense trades, curve-adjustment and structure trades, spread tools (mean reversion, quality
spreads, percent yield spreads), callable vs bullet structure.

**Liability-driven investing (L3 R20 §4).** Classical immunization needs PV match, duration = horizon, minimum
convexity/dispersion, and rebalancing as time passes and rates move (cost trade-off). Multiple liabilities: PV
match, composite duration match, asset dispersion wider than liabilities' (bracketing). Risks: non-parallel
shifts (immunization risk M^2 = PV-weighted variance of payment times around the horizon), contingent claims
(calls, prepayments), caps on floaters, default, liquidity. Contingent immunization requires immunizable rate >
safety-net rate (cushion spread). Surplus (economic) duration matters for pensions, not asset duration alone.

**Benchmark selection (L3 R20 §3.2.1, KB p.25-26).** Match market-value risk (duration), income risk (long
portfolios lock in income), credit risk (quality and issuer diversification) and liability framework risk (long
liabilities -> long index); taxable investors compare after tax. Indexing spectrum: pure index, enhanced
(matching primary risk factors: duration, key rates/cash-flow distribution, sector/quality weights, sector and
quality spread-duration contributions, call exposure, issuer exposure), active. Empirical duration (concept):
regress observed price changes on benchmark yield changes; useful when spreads are correlated with rates (high
yield, equity-like credits, MBS) and analytical duration overstates sensitivity.

## 5. Interpretation, red flags and exam traps

- **Macaulay vs modified vs effective.** Macaulay = weighted time (years; use for horizon/immunization);
  modified = % price change per 1.00 yield change; effective = curve sensitivity for bonds with options.
  Callables and MBS: never use modified/Macaulay; yield to worst is a quote convention, not a risk measure.
- **Per period vs annual.** Divide annual rates by freq inside discounting; annualize per-period duration by /freq
  and convexity by /freq^2. Money-market rates are simple (365/days periodicity); compare at one periodicity.
- **Flat vs full price.** Quotes are flat; you pay full = flat + accrued; duration, convexity and money duration
  are on the full price. At YTM = coupon rate the flat price between coupons is slightly below par.
- **Convexity scaling.** This engine uses the L1 convention: Con = (PV- + PV+ - 2PV0)/(dy^2 PV0) with 1/2 x C x dy^2.
  The 2014 L2 reading (Fabozzi) divides by 2V0 dy^2 and drops the 1/2; Bloomberg prints convexity / 100. Do not mix.
  (The L1 credit reading's "-4.765%" example omits the 1/2; its own Exhibit 17 applies it: `bond_price_change_estimate`.)
- **Negative convexity.** Callables near the call price and MBS: upside capped, effective convexity < 0; putables
  always positive. Higher volatility lowers callable value and OAS, raises putable value.
- **OAS depends on the volatility assumption and benchmark.** Compare with the required OAS of comparable issues;
  OAS = Z-spread - option cost (in bp) for a callable. Positive OAS vs the issuer's own curve = cheap.
- **Spot vs YTM vs par.** YTM is a cash-flow-weighted average of spots; two different curves can give the same YTM.
  Forward rates are breakeven reinvestment rates, not forecasts (unless pure expectations holds).
- **Duration gap.** Macaulay > horizon: price risk dominates (higher rates hurt); < horizon: reinvestment risk.
- **FRNs.** Index duration is tiny; spread (credit) duration is near maturity: credit downgrades still hurt.
- **Credit.** PD alone misranks risk; expected loss (and its PV) includes LGD. Spread ~ PD x LGD ignores liquidity
  and risk premia. Ratings lag; check covenants and structural subordination before trusting an issuer rating.
- **MBS.** Prepayments are borrowers' call options: faster speeds shorten WAL (contraction when rates fall),
  slower lengthen it (extension). PSA multiples scale the CPR, not the SMM.
- **Leverage.** Magnifies both ways; equity duration (D_A A - D_L L)/E can be several times asset duration.
  Repo rate falls with better collateral, delivery to the lender and special (scarce) collateral.
- **Hedging.** Futures hedge ratio uses the CTD's dollar duration divided by its conversion factor; cross hedges
  add basis risk and yield-beta error. Hedged foreign return ~ domestic rate + local risk premium.
- Known engine conventions: binomial trees and `credit_cva_bond` are annual-step; 30/360 is US bond basis
  (no February end-of-month rule); matrix pricing averages comparables' yields by maturity then interpolates
  linearly; `mbs_cash_flows` reproduces the L2 R48 Exhibit 3 cash flows exactly, while the printed WAL table is
  ~0.01-0.02 years lower.

## 6. Deliverable outline

1. **Question and answer first**: the price/yield/spread/duration/decision in one sentence, units stated
   (decimal vs %, bp, per 100 par, annualized with periodicity).
2. **Inputs and conventions**: the bond terms, settlement and day count, curve and its periodicity, assumptions
   (volatility, PSA, recovery, parallel shift).
3. **Calculations**: each engine call with its output (rounded for display; say how).
4. **Risk profile**: duration (which kind), convexity, key-rate or spread exposures, scenario table
   (+/-100 bp, steepener/flattener, spread widening), option and credit effects.
5. **Relative value or recommendation**: vs benchmark/peers (spread vs history, OAS vs required, breakeven),
   with the qualitative case (four Cs, covenants, curve view, liability fit).
6. **Red flags and sensitivities**: what would change the conclusion (volatility, prepayments, rating migration,
   liquidity, FX), and data you assumed rather than observed.
7. **Sources**: curriculum readings and KB paths used.
