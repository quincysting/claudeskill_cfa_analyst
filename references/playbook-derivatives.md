# Playbook: Derivatives (Levels I-III)

Engine modules (topic `derivatives`, list with `python scripts/cfa.py list derivatives`):
- `scripts/cfa_calc/derivatives.py`: forwards, futures, FRAs, swaps, swaptions (plus the shared helpers).
- `scripts/cfa_calc/derivatives_options.py`: bounds, parity, binomial, BSM, Greeks, Black, option strategies.
- `scripts/cfa_calc/derivatives_risk.py`: L3 hedging with futures, forwards, options and swaps.

Import what you need: `from cfa_calc import derivatives as dv, derivatives_options as do, derivatives_risk as dr`.
Read units with `python scripts/cfa.py describe <name>`. Every number in an answer comes from a call; show the call
and its output. Historical volatility for BSM inputs: `volatility_from_prices` (quant module).

Knowledge base (KB) root `E:/CLAUDE/CFA/MD` (2014 reading numbers; 2013 numbering in `Curriculum-2013/`):
- L1 `Level-1/09-Derivatives/Curriculum-2014/`: R57 markets and instruments, R58 forwards, R59 futures,
  R60 options, R61 swaps, R62 option strategies. Faster review: `Kaplan-Notes-2014/`, `Schweser-Notes-2013/`.
- L2 `Level-2/09-Derivatives/Curriculum-2014/`: R51 forwards, R52 futures, R53 options, R54 swaps,
  R55 interest rate derivative instruments, R56 credit default swaps.
- L3 `Level-3/11-Risk-Management-Applications-of-Derivatives/Curriculum-2014/`: R29 forward and futures strategies,
  R30 option strategies, R31 swap strategies. Related: `Level-3/07-Fixed-Income-Portfolio-Management/` R22 (CTD
  duration hedges, protective puts), `Level-3/10-Risk-Management/` R28 (2014) / R35 (2013) currency management.

## Contents
1. When to use
2. Inputs to gather
3. Quantitative workflow (A forwards/futures, B FRAs, C swaps, D option pricing, E Greeks and hedging,
   F strategies, G L3 futures overlays, H L3 interest rate risk on loans, I L3 swaps, J currency)
4. Qualitative framework (choosing the instrument, credit and liquidity risk, hedge effectiveness, accounting/tax)
5. Interpretation, red flags and exam traps
6. Deliverable outline

## 1. When to use

- **Valuation**: price a forward, futures, FRA, swap, option or swaption; mark an existing contract to market;
  check a quote for arbitrage.
- **Hedging**: lock in a price, rate or exchange rate; change a portfolio's beta or duration; floor or cap a loan rate;
  hedge a foreign asset's market and currency risk.
- **Speculation / views**: express a directional or volatility view with spreads, straddles, strangles, butterflies;
  measure max gain, max loss and breakevens before trading.
- **Synthetic positions**: equitize cash, create synthetic cash, pre-invest expected inflows, shift stock/bond mix,
  replicate an option (parity) or a forward (put-call-forward parity), convert fixed to floating with swaps.

## 2. Inputs to gather

| Task | Must have | Ask or assume (state it) |
|---|---|---|
| Forward / futures price | spot, rate, time, income or costs | Compounding: 2014 curriculum uses (1+r)^T; continuous when yields are continuous. Dividends as amounts and dates, a yield, or an FV? Bond price clean or full? |
| Forward value | original contract price, current spot, rate, time left | Which side (long/short)? Remaining dividends only (paid after t). |
| Currency | spot quote direction, both rates, time | Quote is domestic per foreign (price/base). LIBOR-style rates: `compounding="simple"` with years = days/360. |
| FRA | spot LIBORs to h and h+m days, or the FRA rate | 360-day basis; payoff discounted at the underlying rate; long = pays fixed. |
| Swap | term structure (LIBOR spot rates + days, or discount factors), fixed rate, last floating reset | Payment frequency and day count; which side pays fixed; notional. |
| Option price | S, X, T, r, sigma, dividends | Continuous r for BSM (r_c = ln(1+r)); European vs American; discrete up/down or CRR with sigma. |
| Strategy | strikes, premiums, S0, S_T | Long or short each leg; same expiration. |
| Beta / duration overlay | portfolio value, current and target beta or duration, futures price (x multiplier), futures beta or implied duration | Yield beta (1 if unknown); CTD price, duration and conversion factor for the CTD form. |
| Loan hedges | notional, spread, days, LIBOR now and at reset, option strike and premium | Premium financed at current LIBOR + spread; 360-day accrual, 365-day annualization. |
| Currency hedge | foreign value, beta, index futures, forward rate, both rates | Hedge the hedged value V(1+r_f)^T or the principal V? |

## 3. Quantitative workflow

**A. Forwards and futures (L2 R51, R52; L1 R58-R59).**
1. Price: `forward_price` covers every carry variant: no income; `income_cash_flows=[[amount, years], ...]` or
   `pv_income` (discrete dividends, coupons); `fv_income`/`fv_costs` (curriculum's FV form, storage costs);
   `income_yield` (dividend yield, convenience yield, foreign rate); `cost_yield`; `compounding`.
2. Value during the life: `forward_value` (pass remaining dividends; `years_remaining=0` at expiration;
   `position="short"` flips the sign). Value at inception is zero by construction.
3. Currency: `currency_forward_price` (covered interest parity, also `forward_premium`) and `currency_forward_value`.
4. Bonds: `bond_forward_price` (full price, coupons by PV or FV, accrued interest at 0 and T, conversion factor ->
   quoted futures price). Delivery: `bond_futures_invoice_price`; cheapest-to-deliver: highest
   `bond_futures_implied_repo_rate` across deliverable bonds. T-bill futures: `tbill_futures_price`.
5. Arbitrage and basis: `forward_arbitrage` returns the theoretical price, cash-and-carry or reverse strategy,
   profit at expiration, implied return and basis (S - F). Basis converges to 0 at expiration.
6. Margin: `futures_margin_account` (daily table, variation margin to the initial level, trigger price).
   Value of a futures before marking to market = (current futures price - last settlement) x multiplier.

**B. FRAs (L2 R51).** `fra_rate` (h, h+m days, 360 basis, "3x9" notation) -> `fra_value` on day g ->
`fra_settlement` at expiration (discounted at the underlying LIBOR; long receives when LIBOR > FRA rate).

**C. Swaps (L2 R54; L1 R61).**
1. Fixed rate: `swap_fixed_rate` from LIBOR spot rates and days or discount factors. Returns periodic and annual
   rates, the implied forward (FRA) rates and the FRA-portfolio check (sum fwd x B = 1 - B_n). Currency swap:
   one call per currency; equity swap fixed rate = plain-vanilla rate.
2. Value later: `interest_rate_swap_value` (floating leg = (1 + next payment) x B_1; fixed leg = bond with par),
   `currency_swap_value` (all four fixed/floating combinations; foreign notional = domestic / S0),
   `equity_swap_value` (equity vs fixed, floating or another index; equity leg = S_t / S_last reset).
3. Swaptions: `swaption_payoff` (payer = put on a bond with coupon = strike; receiver = call; pass the market swap
   rate if quoted, otherwise it is computed from the discount factors).

**D. Option pricing (L2 R53; L1 R60).**
1. Moneyness and time value: `option_intrinsic_time_value`. Bounds: `option_bounds` (European vs American).
2. Parity: `put_call_parity` (solve any leg; mispricing and arbitrage when all are given),
   `put_call_forward_parity` (synthetic forward/put).
3. Binomial: `binomial_option_price` (u, d, r per period; hedge ratio n, bond B with c = nS - B, value and
   hedge-ratio trees; `american=True` lists early-exercise nodes). N-step: `binomial_crr_option_price`
   (u = e^(sigma sqrt dt), dividend yield, American). Rate trees (bond options, caplets, floorlets):
   `binomial_interest_rate_option`; cap = sum of caplets.
4. BSM: `bsm_option_price` (call and put, N(d1), N(d2); `dividend_yield` or `pv_income`; currency: foreign rate
   as yield). Futures/forward options: `black_futures_option_price`. Interest rate options:
   `black_interest_rate_option` (lag discount and m/360 scaling) and `black_cap_floor_price`.
   Payoff check: `interest_rate_option_payoff`. Implied volatility: `bsm_implied_volatility`.

**E. Greeks and dynamic hedging (L2 R53; L3 R30 Section 4).** `bsm_greeks` (delta, gamma, theta, vega, rho with
per-day / per-1% versions). Shares to hedge an option book and the rebalance trade: `delta_hedge`
(`option_position` negative when short; or `share_position` to get options needed). Gamma: `delta_gamma_neutral_hedge`
(one option + underlying, or two options).

**F. Strategies at expiration (L3 R30; L1 R62).** Each returns value at expiration, profit, initial cost, max gain,
max loss (inf = unlimited) and breakevens: `option_payoff_profit`, `covered_call`, `protective_put`, `bull_spread`,
`bear_spread` (calls or puts), `butterfly_spread`, `collar`, `straddle`, `strangle`; `box_spread` adds the
arbitrage test vs (X2 - X1)/(1+r)^T. Anything else: `option_strategy_expiration` with a list of legs.
Zero-cost collar: `zero_cost_collar_call_strike` (BSM put premium -> call strike).

**G. L3 futures overlays (R29; FI R22).**
1. Beta: `equity_futures_contracts_beta` (target 0 = hedge; multiplier for quoted index prices).
2. Duration: `bond_futures_contracts_duration` (implied futures duration, yield beta) or
   `bond_futures_contracts_ctd` (CTD duration, price and conversion factor; BPVs returned).
3. Synthetic positions: `synthetic_equity_cash_futures` (`direction="equitize"` for a synthetic index fund /
   equitized cash, `"to_cash"` for synthetic cash; rounding gives the effective amount).
4. Asset allocation: `futures_asset_allocation` (stock<->bond shift plus optional new beta/duration; components
   rounded as in the curriculum). Pre-investing: `preinvest_futures_contracts`.
5. Protective puts on bond futures: `bond_futures_protective_put` (puts = full-hedge futures count; floor is a
   duration approximation, say so).

**H. L3 interest rate risk on loans (R29, R30).** Single loan with an FRA: `fra_hedged_loan_rate`. With an
interest rate call (borrower) or put (lender): `interest_rate_option_loan` (effective annual rate, premium
compounded at LIBOR + spread). Floating-rate loans with caps, floors or collars: `floating_rate_loan_cap_floor`.

**I. L3 swap strategies (R31).** `swap_converted_rate` (floating -> fixed: swap rate + spread; fixed -> floating:
LIBOR + coupon - swap rate). Duration: `swap_duration_approx` (floating ~ half the period, fixed ~ 75% of maturity
per curriculum) then `swap_notional_for_duration`.

**J. Currency hedging (R29; R28/R35).** Receivable/payable: lock in amount x `currency_forward_price`. Foreign
equity: `foreign_equity_currency_hedge` (index futures for local market risk, forward on V(1+r_f)^T or on the
principal; outcome comparison). Regression hedge: `minimum_variance_hedge_ratio`; partial hedge return:
`currency_hedged_return`.

## 4. Qualitative framework

**Choosing the instrument (L3 R29 "Final comments"; L2 R51 "The role of forward markets", R52 "The role of futures
markets and exchanges"; L1 R57).**
| Need | Forward / swap (OTC) | Futures (exchange) | Options |
|---|---|---|---|
| Exact date, amount, underlying | customized: best fit | standardized: basis and date mismatch | customized OTC or listed |
| Credit risk | bilateral; grows with MTM value; collateral / marking to market mitigates | clearinghouse guarantee; daily settlement | buyer bears seller's credit risk (OTC); writer bears none after premium |
| Cash flows | none until settlement (swaps: periodic) | daily variation margin: liquidity planning | premium up front; loss capped for the buyer |
| Payoff | linear, symmetric | linear, symmetric | nonlinear: keeps upside, costs premium |
| Liquidity | dealer-dependent; terminate by offsetting contract | deep in nearby contracts, thin in distant ones | listed: strike/expiry grid; OTC: dealer |
| Regulation / policy | light; may be banned for credit-risky instruments | regulated; some policies ban full leverage | some policies allow options but not leverage |

- Many periodic exposures (a floating loan): a swap or cap/floor beats a strip of single FRAs or options.
- One future exposure: FRA, forward or a single interest rate option. Uncertain whether the exposure occurs (a loan
  commitment, a bid): options, because a forward could leave an unhedged position (L3 R29 Section 2.1).
- Beta, duration and allocation changes: futures are cheaper (stock index derivative costs are reported about 95%
  lower than trading the stocks) and less disruptive to asset-class managers (L3 R29 final comments).

**Hedge effectiveness and basis risk.**
- A hedge swaps price risk for basis risk: the spot-futures relation can change before the hedge is lifted; cross
  hedges (corporate bonds with Treasury futures, small caps with S&P futures) add spread risk (L3 FI R22 5.3.4).
- Beta and duration are estimates; the ex post beta/duration will differ (L3 R29 Exhibit 2: target 7.50, realized
  7.70). Check ex post: effective beta = portfolio return / market return; effective duration = -%change / dy.
- Yield beta corrects for portfolio yields that move more or less than the futures' implied yield.
- Delta hedges drift: rebalance as S and time change; large gamma (at the money, near expiry) makes delta hedges
  fail; vega risk remains even when delta- and gamma-neutral (L3 R30 Section 4).
- Swapping a floating loan to fixed stabilizes cash flows but raises market-value duration (sixfold in L3 R31 IBP):
  a cash-flow hedge, a value speculation.
- Hedging foreign equity returns locks in the foreign risk-free rate; adding the currency forward locks in the
  domestic rate. Leaving the local market unhedged while hedging currency is only approximate (hedge an expected or
  minimum value) (L3 R29 Section 5.3).

**Accounting and tax notes from the curriculum.** Derivatives are marked to market on the balance sheet and income
statement, which is one reason to compute the value of forwards and swaps during their lives (L2 R51 Section 4.1;
L3 R31 Section 2.1). Translation exposure is an accounting issue, transaction exposure an economic one; the readings
hedge the latter (L3 R29 Section 5). Equity swaps on a concentrated holding keep voting rights but can create cash
calls when the stock outperforms; in the US a synthetic sale through a swap ends the holding period, so there is no
tax advantage, and an insider's swap must be reported (L3 R31 Sections 4.1, 4.4).

## 5. Interpretation, red flags and exam traps

- **Compounding.** 2014 forwards, bounds, parity and binomial use (1+r)^T with T = days/365; BSM and Black need
  continuous r_c = ln(1+r). LIBOR, FRA, swap and interest rate option accruals use simple days/360. Mixing them is the
  most common wrong answer; `compounding` makes it explicit.
- **360 vs 365.** FRA and swap discount factors: 1/(1 + L days/360). Effective annual loan rates: ^(365/days).
  The Black caplet lag discount in the 2014 reading uses e^(-F m/365) with accrual m/360.
- **Long vs short.** Forward value formulas give the long's value; the short is the negative. FRA long = pays fixed.
  Payer swaption = pay fixed = put on a bond. Short options: max gain is the premium, max loss can be unlimited.
- **Which dividends.** Only cash flows between today and expiration; at valuation date t only those still to come.
- **Forward price is not a forecast.** F > S just means carry (r > yield); currency premium follows rate differential.
- **Futures margin.** Margin calls restore the initial (not maintenance) level and are set by the prior day's
  balance, regardless of the next price move. Trigger price move = initial - maintenance (per unit).
- **Moneyness traps.** Puts are in the money when S < X. American calls on non-dividend stocks are never exercised
  early (= European); American puts can be. Intrinsic value uses X, the European lower bound uses PV(X).
- **Curriculum rounding.** The readings read N(d) from a two-decimal table and round intermediate values; the engine
  uses exact N(d) (e.g. L2 R53 Example 11: 15.06 in the text vs 15.23 exact). Report the exact number and note the
  difference if it matters.
- **Contract counts.** Use the unrounded count to explain, the rounded count for P&L. Sign: + buy, - sell.
  Multiply quoted index futures by the multiplier (S&P 1225 x $250). Bond futures: implied duration form (R29) vs
  CTD/conversion-factor form (FI R22): do not mix inputs.
- **Swap value signs.** Pay-fixed gains when rates rise. Currency swaps also move with the exchange rate.
- **Red flags in a client's hedge:** hedge size from an outdated beta or duration; futures expiring well after the
  horizon (liquidity, basis); options bought when premium budget makes the hedge negative-carry for years; a
  "zero-cost" collar presented as costless (it gives up upside); swaps sized on notional rather than duration.

## 6. Deliverable outline

1. **Question and position**: what is hedged or valued, side (long/short), horizon, constraints (policy, liquidity).
2. **Inputs**: table of every input with source; conventions chosen (compounding, day basis, quote direction) and
   assumptions (yield beta, dividend treatment).
3. **Calculations**: each engine call and its key outputs (price, value, contracts, Greeks, effective rate), in order.
4. **Result**: the number(s) with units and sign; contracts rounded with action (buy/sell); for strategies, the
   payoff table across S_T scenarios plus max gain, max loss, breakevens.
5. **Interpretation**: what the result means (mispricing and arbitrage trade, locked-in rate, residual risks:
   basis, credit, gamma/vega, model and rounding differences vs curriculum).
6. **Recommendation and monitoring**: instrument choice and why (Section 4 table), rebalancing triggers, ex post
   checks (effective beta/duration), accounting/tax considerations if relevant.
