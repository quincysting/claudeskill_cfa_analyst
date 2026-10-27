"""Derivatives I: forwards, futures, FRAs and swaps (CFA L1-L2, 2013/2014 curriculum).

Options live in derivatives_options.py and L3 risk-management applications in derivatives_risk.py.
Shared helpers (underscore names, not registered) live here and are imported by those modules.

Conventions: rates are decimals; `years` is time in years; `compounding` is "discrete" ((1+r)^T, the
2014 curriculum default), "continuous" (e^(rT)) or "simple" (1 + rT, money-market/LIBOR style).
Day-count money-market rates use `day_basis` (360 unless stated). Currency quotes are domestic per
unit of foreign currency (price/base), as in the curriculum.
"""
from __future__ import annotations

import math

from ._registry import formula

R51 = "L2 Derivatives: Forward Markets and Contracts (2014 R51; L1 2014 R58)"
R52 = "L2 Derivatives: Futures Markets and Contracts (2014 R52; L1 2014 R59)"
R54 = "L2 Derivatives: Swap Markets and Contracts (2014 R54; L1 2014 R61)"
R55 = "L2 Derivatives: Interest Rate Derivative Instruments (2014 R55)"

COMPOUNDING = ("discrete", "continuous", "simple")


# ---------------------------------------------------------------- shared helpers
def _check_compounding(compounding: str) -> None:
    if compounding not in COMPOUNDING:
        raise ValueError(f"compounding must be one of {COMPOUNDING}, got {compounding!r}")


def _growth(rate: float, years: float, compounding: str = "discrete") -> float:
    """Future value of 1 after `years` at `rate` under the given compounding convention."""
    _check_compounding(compounding)
    if compounding == "discrete":
        if rate <= -1:
            raise ValueError("rate must be greater than -100%")
        return (1 + rate) ** years
    if compounding == "continuous":
        return math.exp(rate * years)
    g = 1 + rate * years
    if g <= 0:
        raise ValueError("1 + rate * years must be positive")
    return g


def _nonneg(name: str, x: float) -> float:
    if x < 0:
        raise ValueError(f"{name} must be non-negative, got {x}")
    return x


def _pos(name: str, x: float) -> float:
    if x <= 0:
        raise ValueError(f"{name} must be positive, got {x}")
    return x


def _sign(position: str, long: str = "long", short: str = "short") -> int:
    if position == long:
        return 1
    if position == short:
        return -1
    raise ValueError(f"position must be {long!r} or {short!r}, got {position!r}")


def _pv_flows(flows, rate: float, compounding: str) -> float:
    """PV of [[amount, years_from_now], ...]; flows at or before now are ignored (already paid)."""
    if not flows:
        return 0.0
    total = 0.0
    for item in flows:
        if len(item) != 2:
            raise ValueError("cash flows must be [amount, years] pairs")
        amount, t = float(item[0]), float(item[1])
        if t > 0:
            total += amount / _growth(rate, t, compounding)
    return total


def _discount_factors(discount_factors=None, spot_rates=None, days=None, day_basis: float = 360) -> list:
    """Discount factors given directly, or from LIBOR-style add-on spot rates: B = 1/(1 + L days/basis)."""
    if discount_factors is not None:
        dfs = [float(b) for b in discount_factors]
    else:
        if spot_rates is None or days is None or len(spot_rates) != len(days):
            raise ValueError("give discount_factors, or spot_rates and days of equal length")
        dfs = [1 / (1 + float(r) * float(d) / day_basis) for r, d in zip(spot_rates, days)]
    if not dfs or any(b <= 0 for b in dfs):
        raise ValueError("discount factors must be a non-empty list of positive numbers")
    return dfs


# ---------------------------------------------------------------- forwards and futures pricing
@formula("derivatives",
         formula="F = (S - PV(income) + PV(costs)) x G(r,T) x G(cost_yield,T)/G(income_yield,T) + FV(costs) - FV(income)",
         curriculum=R51 + "; " + R52)
def forward_price(spot: float, rate: float, years: float, income_cash_flows: list | None = None,
                  pv_income: float = 0.0, pv_costs: float = 0.0, fv_income: float = 0.0, fv_costs: float = 0.0,
                  income_yield: float = 0.0, cost_yield: float = 0.0, compounding: str = "discrete") -> dict:
    """No-arbitrage forward/futures price (cost of carry) with any mix of income, costs and yields.

    One formula covers the curriculum variants: no income (S(1+r)^T); discrete dividends or coupons
    (pass income_cash_flows or pv_income); FV of dividends/coupons (fv_income); storage costs (pv_costs,
    fv_costs or cost_yield); convenience yield, dividend yield or foreign interest rate (income_yield);
    continuous versions (compounding="continuous", e.g. S e^((r - q)T)). Futures are priced like forwards
    (constant rates assumed).

    Args:
        spot: spot price S0 (for a bond: full price including accrued interest).
        rate: risk-free rate (annual, per `compounding`).
        years: time to expiration T.
        income_cash_flows: [[amount, years_from_now], ...] dividends/coupons paid on or before expiration;
            discounted at `rate`. A flow dated after `years` raises ValueError (it belongs to the holder after
            delivery and must not be deducted); flows at t <= 0 are treated as already paid.
        pv_income, pv_costs: present values of benefits/costs over the contract life.
        fv_income, fv_costs: future values (at T) of benefits/costs.
        income_yield: dividend yield, convenience yield or foreign rate (same compounding as `rate`).
        cost_yield: storage cost as a yield.
        compounding: "discrete" (curriculum default), "continuous" or "simple".

    Returns:
        forward_price, net_spot (S - PV income + PV costs), pv_income_total, growth_factor.

    Example:
        forward_price(spot=62.50, rate=0.042, years=200/365,
                      income_cash_flows=[[0.75, 50/365], [0.75, 140/365]]) -> 62.41 (L2 R51 Example 2)
    """
    _pos("spot", spot)
    _nonneg("years", years)
    pv_i = _pv_flows(income_cash_flows, rate, compounding) + pv_income  # also checks [amount, t] pairs
    late = [t for _, t in (income_cash_flows or []) if float(t) > years + 1e-12]
    if late:
        raise ValueError(f"income/cost cash flows dated after expiration (T = {years}) at t = {late}: they do not "
                         "affect the forward price; drop them")
    net_spot = spot - pv_i + pv_costs
    g = _growth(rate, years, compounding) * _growth(cost_yield, years, compounding) / _growth(
        income_yield, years, compounding)
    f = net_spot * g + fv_costs - fv_income
    return {"forward_price": f, "net_spot": net_spot, "pv_income_total": pv_i, "growth_factor": g}


@formula("derivatives",
         formula="V_t(long) = (F_t - F_0) / G(r, T-t) = S_t - PV(income) - F_0/(1+r)^(T-t); short = -long",
         curriculum=R51)
def forward_value(spot: float, forward_price_initial: float, rate: float, years_remaining: float,
                  position: str = "long", income_cash_flows: list | None = None, pv_income: float = 0.0,
                  pv_costs: float = 0.0, fv_income: float = 0.0, fv_costs: float = 0.0,
                  income_yield: float = 0.0, cost_yield: float = 0.0, compounding: str = "discrete",
                  notional: float = 1.0) -> dict:
    """Value of a forward contract at inception, during its life, or at expiration (years_remaining=0).

    Args:
        spot: current spot price S_t.
        forward_price_initial: contract price F(0,T) agreed at inception.
        rate: current risk-free rate.
        years_remaining: T - t (0 at expiration, where value = S_T - F).
        position: "long" or "short" (short value = -long value).
        income_cash_flows: remaining dividends/coupons [[amount, years_from_now], ...], none after expiration
            (a later date raises ValueError).
        pv_income, pv_costs, fv_income, fv_costs, income_yield, cost_yield, compounding: as in forward_price.
        notional: units of the underlying (value is multiplied by it).

    Returns:
        value (for `position`, times notional), value_long_per_unit, current_forward_price, pv_forward_price.

    Example:
        forward_value(spot=102, forward_price_initial=105, rate=0.05, years_remaining=0.75) -> 0.7728
        (L2 R51 Section 4.1)
    """
    s = _sign(position)
    fp = forward_price(spot=spot, rate=rate, years=years_remaining, income_cash_flows=income_cash_flows,
                       pv_income=pv_income, pv_costs=pv_costs, fv_income=fv_income, fv_costs=fv_costs,
                       income_yield=income_yield, cost_yield=cost_yield, compounding=compounding)
    disc = _growth(rate, years_remaining, compounding)
    v_long = (fp["forward_price"] - forward_price_initial) / disc
    return {"value": s * v_long * notional, "value_long_per_unit": v_long,
            "current_forward_price": fp["forward_price"], "pv_forward_price": forward_price_initial / disc}


@formula("derivatives", formula="F = S0 x G(r_dom,T) / G(r_for,T)  (covered interest rate parity)", curriculum=R51)
def currency_forward_price(spot: float, domestic_rate: float, foreign_rate: float, years: float,
                           compounding: str = "discrete") -> dict:
    """Currency forward/futures rate by covered interest rate parity (quote: domestic per foreign).

    Default is annual compounding, F = S (1 + r_dom)^T / (1 + r_for)^T, as in the derivatives readings; continuous
    is also available. For money-market (LIBOR) rates with simple act/360 interest and forward points (economics/FX
    readings), use economics' `fx_forward_cip`; compounding="simple" here with years = days/360 gives the same
    forward when both currencies use a 360-day basis.

    Args:
        spot: spot rate, domestic currency per unit of foreign currency.
        domestic_rate, foreign_rate: risk-free rates (same compounding).
        years: time to expiration. Use compounding="simple" with years = days/360 for LIBOR-style rates.
        compounding: "discrete", "continuous" or "simple".

    Returns:
        forward_price, forward_premium (F/S - 1; positive = foreign currency at a premium), foreign_units_to_buy
        (1/G(r_for,T), units of foreign currency that grow to 1 at T).

    Example:
        currency_forward_price(spot=0.5987, domestic_rate=0.055, foreign_rate=0.0475, years=180/365) -> 0.6008
        (L2 R51 Section 4.4)
    """
    fp = forward_price(spot=spot, rate=domestic_rate, years=years, income_yield=foreign_rate,
                       compounding=compounding)["forward_price"]
    return {"forward_price": fp, "forward_premium": fp / spot - 1,
            "foreign_units_to_buy": 1 / _growth(foreign_rate, years, compounding)}


@formula("derivatives", formula="V_t(long) = S_t/G(r_for,T-t) - F_0/G(r_dom,T-t)", curriculum=R51)
def currency_forward_value(spot: float, forward_price_initial: float, domestic_rate: float, foreign_rate: float,
                           years_remaining: float, position: str = "long", notional_foreign: float = 1.0,
                           compounding: str = "discrete") -> dict:
    """Mark-to-market value of a currency forward (long = agreed to buy the foreign currency).

    Args:
        spot: current spot rate (domestic per foreign). forward_price_initial: contract rate.
        domestic_rate, foreign_rate: current risk-free rates. years_remaining: T - t.
        position: "long" or "short". notional_foreign: foreign currency units.
        compounding: "discrete", "continuous" or "simple".

    Returns:
        value (domestic currency, for `position` and notional), value_long_per_unit, current_forward_price.

    Example:
        currency_forward_value(spot=1.72, forward_price_initial=1.75, domestic_rate=0.051, foreign_rate=0.062,
                               years_remaining=11/12) -> value_long_per_unit -0.0443 (L2 R51 Example 5B)
    """
    r = forward_value(spot=spot, forward_price_initial=forward_price_initial, rate=domestic_rate,
                      years_remaining=years_remaining, position=position, income_yield=foreign_rate,
                      compounding=compounding, notional=notional_foreign)
    return {"value": r["value"], "value_long_per_unit": r["value_long_per_unit"],
            "current_forward_price": r["current_forward_price"]}


@formula("derivatives",
         formula="Full F = (B0 + AI0 - PV(coupons)) (1+r)^T - FV(coupons); clean F = full F - AI_T; quoted futures = clean F / CF",
         curriculum=R52 + "; " + R51)
def bond_forward_price(bond_price: float, rate: float, years: float, coupons: list | None = None,
                       fv_coupons: float = 0.0, accrued_interest_0: float = 0.0, accrued_interest_T: float = 0.0,
                       conversion_factor: float = 1.0, compounding: str = "discrete") -> dict:
    """Forward/futures price of a coupon bond, with accrued interest and a delivery conversion factor.

    Args:
        bond_price: bond price today; clean if accrued_interest_0 is given, otherwise the full price.
        rate: risk-free rate. years: time to forward/futures expiration.
        coupons: coupons paid before expiration, [[amount, years_from_now], ...] (PV method); a coupon dated
            after `years` raises ValueError.
        fv_coupons: alternatively the future value at expiration of those coupons (FV method).
        accrued_interest_0: accrued interest today (added to a clean bond_price).
        accrued_interest_T: accrued interest at expiration (removed to get the clean/quoted price).
        conversion_factor: CF of the deliverable (cheapest-to-deliver) bond; 1 for a single deliverable bond.
        compounding: "discrete", "continuous" or "simple".

    Returns:
        quoted_futures_price (clean forward / CF), full_forward_price, clean_forward_price, pv_coupons.

    Example:
        bond_forward_price(bond_price=1.0132, rate=0.07, years=1, fv_coupons=0.0815, conversion_factor=1.0372)
        -> quoted_futures_price 0.9667 (L2 R52 Example 4)
    """
    _pos("conversion_factor", conversion_factor)
    fp = forward_price(spot=bond_price + accrued_interest_0, rate=rate, years=years, income_cash_flows=coupons,
                       fv_income=fv_coupons, compounding=compounding)
    full = fp["forward_price"]
    clean = full - accrued_interest_T
    return {"quoted_futures_price": clean / conversion_factor, "full_forward_price": full,
            "clean_forward_price": clean, "pv_coupons": fp["pv_income_total"]}


@formula("derivatives", formula="Invoice = contract size x (futures price/100) x CF + accrued interest", curriculum=R55)
def bond_futures_invoice_price(futures_price: float, conversion_factor: float, accrued_interest: float = 0.0,
                               contract_size: float = 100_000) -> dict:
    """Amount the long pays the short on delivery of a Treasury bond into a futures contract.

    Args:
        futures_price: settlement price per 100 of par (97-24 = 97.75).
        conversion_factor: CF of the bond delivered.
        accrued_interest: accrued interest on the delivered bond, in currency per contract.
        contract_size: par value per contract (100,000 for CBOT T-bond futures).

    Returns:
        invoice_price, converted_price (contract size x price/100 x CF).

    Example:
        bond_futures_invoice_price(futures_price=105.5, conversion_factor=1.2) -> converted 126,600
        (L2 R55 Section 2)
    """
    _pos("futures_price", futures_price)
    _pos("conversion_factor", conversion_factor)
    conv = contract_size * futures_price / 100 * conversion_factor
    return {"invoice_price": conv + accrued_interest, "converted_price": conv}


@formula("derivatives",
         formula="Implied repo = (proceeds - cost)/cost x 360/days; proceeds = F x CF + AI_del + coupon + reinvestment",
         curriculum=R55)
def bond_futures_implied_repo_rate(bond_price: float, accrued_interest_paid: float, futures_price: float,
                                   conversion_factor: float, accrued_interest_at_delivery: float,
                                   days_to_delivery: float, interim_coupon: float = 0.0,
                                   days_coupon_reinvested: float = 0.0, reinvestment_rate: float = 0.0,
                                   day_basis: float = 360) -> dict:
    """Implied repo rate of a cash-and-carry trade in a deliverable bond (highest = cheapest to deliver).

    Args:
        bond_price: clean price paid for the bond (per 100). accrued_interest_paid: AI paid at purchase.
        futures_price: futures price (per 100). conversion_factor: CF of this bond.
        accrued_interest_at_delivery: AI received on delivery. days_to_delivery: holding period (days_1).
        interim_coupon: coupon received before delivery. days_coupon_reinvested: days_2.
        reinvestment_rate: term repo rate used to reinvest the coupon (money-market, day_basis).
        day_basis: 360.

    Returns:
        implied_repo_rate, dollar_return, proceeds, cost, converted_price.

    Example:
        bond_futures_implied_repo_rate(bond_price=107, accrued_interest_paid=3.8904, futures_price=96,
            conversion_factor=1.1111, accrued_interest_at_delivery=1.1507, days_to_delivery=82,
            interim_coupon=5, days_coupon_reinvested=42, reinvestment_rate=0.038) -> 0.0771 (L2 R55)
    """
    _pos("days_to_delivery", days_to_delivery)
    cost = bond_price + accrued_interest_paid
    _pos("cost", cost)
    conv = futures_price * conversion_factor
    reinv = interim_coupon * reinvestment_rate * days_coupon_reinvested / day_basis
    proceeds = conv + accrued_interest_at_delivery + interim_coupon + reinv
    ret = proceeds - cost
    return {"implied_repo_rate": ret / cost * day_basis / days_to_delivery, "dollar_return": ret,
            "proceeds": proceeds, "cost": cost, "converted_price": conv}


@formula("derivatives", formula="f0(h) = B0(h+m)/B0(h); B0(j) = 1 - r_d(j) j/360; implied discount = (1 - f) 360/m",
         curriculum=R52)
def tbill_futures_price(discount_rate_short: float, days_short: float, discount_rate_long: float,
                        days_long: float, day_basis: float = 360) -> dict:
    """T-bill futures price from the discount rates on the bills maturing at futures expiry (h) and h+m.

    Args:
        discount_rate_short: discount rate on the h-day bill. days_short: h.
        discount_rate_long: discount rate on the (h+m)-day bill. days_long: h + m.
        day_basis: 360.

    Returns:
        futures_price (per 1 par), implied_discount_rate, bill_price_short, bill_price_long, days_underlying.

    Example:
        tbill_futures_price(discount_rate_short=0.05, days_short=50, discount_rate_long=0.046, days_long=140)
        -> 0.9889 (L2 R52 Example 3)
    """
    m = days_long - days_short
    _pos("days_long - days_short", m)
    b_h = 1 - discount_rate_short * days_short / day_basis
    b_hm = 1 - discount_rate_long * days_long / day_basis
    _pos("bill price", min(b_h, b_hm))
    f = b_hm / b_h
    return {"futures_price": f, "implied_discount_rate": (1 - f) * day_basis / m, "bill_price_short": b_h,
            "bill_price_long": b_hm, "days_underlying": m}


@formula("derivatives",
         formula="Theoretical F vs market F: cash-and-carry if F_mkt > F*, reverse if F_mkt < F*; profit at T = |F_mkt - F*|; basis = S - F",
         curriculum=R51 + "; " + R52)
def forward_arbitrage(spot: float, market_forward_price: float, rate: float, years: float,
                      income_cash_flows: list | None = None, pv_income: float = 0.0, pv_costs: float = 0.0,
                      fv_income: float = 0.0, fv_costs: float = 0.0, income_yield: float = 0.0,
                      cost_yield: float = 0.0, compounding: str = "discrete") -> dict:
    """Arbitrage check for a forward/futures quote: strategy, profit, implied rate and basis.

    Cash-and-carry (market price too high): buy the asset (net of carry), sell the forward.
    Reverse cash-and-carry (too low): short the asset, invest proceeds, buy the forward.

    Args:
        spot: spot price. market_forward_price: quoted forward/futures price.
        rate, years, income/cost arguments, compounding: as in forward_price.

    Returns:
        strategy, theoretical_price, mispricing (market - theoretical), profit_at_expiration (per unit, >= 0),
        pv_profit, implied_return (period return of buying net spot and delivering at the market price),
        implied_return_annual, basis (spot - market forward).

    Example:
        forward_arbitrage(spot=125.72, market_forward_price=140, rate=0.05625, years=0.75)
        -> implied_return 0.1136, annual 0.1543 (L2 R51 Example 1B)
    """
    _pos("years", years)
    fp = forward_price(spot=spot, rate=rate, years=years, income_cash_flows=income_cash_flows,
                       pv_income=pv_income, pv_costs=pv_costs, fv_income=fv_income, fv_costs=fv_costs,
                       income_yield=income_yield, cost_yield=cost_yield, compounding=compounding)
    f_star = fp["forward_price"]
    g = _growth(rate, years, compounding)
    net_investment = f_star / g  # PV of what is needed today to deliver one unit at T
    diff = market_forward_price - f_star
    if abs(diff) < 1e-12:
        strategy = "none: market price equals the no-arbitrage price"
    elif diff > 0:
        strategy = "cash-and-carry: buy the asset (borrow), sell the forward/futures"
    else:
        strategy = "reverse cash-and-carry: short the asset (lend proceeds), buy the forward/futures"
    period_ret = market_forward_price / net_investment - 1
    if compounding == "continuous":
        annual = math.log(1 + period_ret) / years
    elif compounding == "simple":
        annual = period_ret / years
    else:
        annual = (1 + period_ret) ** (1 / years) - 1
    return {"strategy": strategy, "theoretical_price": f_star, "mispricing": diff,
            "profit_at_expiration": abs(diff), "pv_profit": abs(diff) / g, "implied_return": period_ret,
            "implied_return_annual": annual, "basis": spot - market_forward_price}


@formula("derivatives",
         formula="Gain_t = sign x contracts x multiplier x (P_t - P_t-1); deposit to initial margin if balance < maintenance",
         curriculum=R52)
def futures_margin_account(settlement_prices: list, contracts: float, initial_margin: float,
                           maintenance_margin: float, position: str = "long", multiplier: float = 1.0) -> dict:
    """Daily mark-to-market of a futures margin account, with margin calls (variation margin).

    The position opens at settlement_prices[0] (day 0) with the initial margin deposited. When an ending
    balance is below the maintenance level, the next day's deposit (variation margin) restores the balance
    to the initial margin level, regardless of that next day's price change.

    Args:
        settlement_prices: daily settlement prices, day 0 first.
        contracts: number of contracts (positive).
        initial_margin, maintenance_margin: per contract.
        position: "long" or "short". multiplier: units of underlying per contract (price x multiplier).

    Returns:
        ending_balance, total_deposits, total_gain, margin_call_trigger_price, rows
        (day, beginning_balance, deposit, settlement_price, price_change, gain_loss, ending_balance).

    Example:
        futures_margin_account(settlement_prices=[100, 99.2, 96, 101, 103.5, 103, 104], contracts=10,
                               initial_margin=5, maintenance_margin=3) -> deposits 50 + 40, ending 130,
        trigger 98 (L2 R52 Exhibit 1)
    """
    s = _sign(position)
    _pos("contracts", contracts)
    if not 0 <= maintenance_margin <= initial_margin:
        raise ValueError("need 0 <= maintenance_margin <= initial_margin")
    if len(settlement_prices) < 1:
        raise ValueError("settlement_prices needs at least the day-0 price")
    init, maint = initial_margin * contracts, maintenance_margin * contracts
    rows = [{"day": 0, "beginning_balance": 0.0, "deposit": init, "settlement_price": settlement_prices[0],
             "price_change": 0.0, "gain_loss": 0.0, "ending_balance": init}]
    for day in range(1, len(settlement_prices)):
        prev = rows[-1]["ending_balance"]
        deposit = init - prev if prev < maint else 0.0
        chg = settlement_prices[day] - settlement_prices[day - 1]
        gain = s * contracts * multiplier * chg
        rows.append({"day": day, "beginning_balance": prev, "deposit": deposit,
                     "settlement_price": settlement_prices[day], "price_change": chg, "gain_loss": gain,
                     "ending_balance": prev + deposit + gain})
    trigger = settlement_prices[0] - s * (initial_margin - maintenance_margin) / multiplier
    return {"ending_balance": rows[-1]["ending_balance"], "total_deposits": sum(r["deposit"] for r in rows),
            "total_gain": sum(r["gain_loss"] for r in rows), "margin_call_trigger_price": trigger, "rows": rows}


# ---------------------------------------------------------------- FRAs
@formula("derivatives",
         formula="FRA(0,h,m) = [(1 + L(h+m)(h+m)/360) / (1 + L(h) h/360) - 1] x 360/m",
         curriculum=R51)
def fra_rate(rate_short: float, days_short: float, rate_long: float, days_long: float,
             day_basis: float = 360) -> dict:
    """No-arbitrage FRA rate (implied LIBOR forward rate) from two add-on spot rates.

    Args:
        rate_short: L0(h), the rate to FRA expiration. days_short: h.
        rate_long: L0(h+m), the rate to maturity of the underlying deposit. days_long: h + m.
        day_basis: 360 (LIBOR/Euribor convention).

    Returns:
        fra_rate (annualized), days_underlying (m), periodic_rate, notation (e.g. "3x9" in months, 30-day months).

    Example:
        fra_rate(rate_short=0.056, days_short=90, rate_long=0.06, days_long=270) -> 0.0611 (L2 R51 Section 4.3)
    """
    m = days_long - days_short
    _pos("days_long - days_short", m)
    _pos("days_short", days_short)
    periodic = (1 + rate_long * days_long / day_basis) / (1 + rate_short * days_short / day_basis) - 1
    return {"fra_rate": periodic * day_basis / m, "days_underlying": m, "periodic_rate": periodic,
            "notation": f"{round(days_short / 30)}x{round(days_long / 30)}"}


@formula("derivatives",
         formula="V_g = 1/(1 + L_g(h-g)(h-g)/360) - (1 + FRA m/360)/(1 + L_g(h+m-g)(h+m-g)/360)",
         curriculum=R51)
def fra_value(fra_rate: float, rate_to_expiry: float, days_to_expiry: float, rate_to_maturity: float,
              days_to_maturity: float, notional: float = 1.0, position: str = "long",
              day_basis: float = 360) -> dict:
    """Market value of an existing FRA during its life (long = pays fixed FRA rate, receives LIBOR).

    Args:
        fra_rate: contract FRA rate. rate_to_expiry, days_to_expiry: L_g(h-g) and h-g.
        rate_to_maturity, days_to_maturity: L_g(h+m-g) and h+m-g.
        notional: notional principal. position: "long" or "short". day_basis: 360.

    Returns:
        value (for `position`), value_per_unit_long.

    Example:
        fra_value(fra_rate=0.0611, rate_to_expiry=0.059, days_to_expiry=65, rate_to_maturity=0.065,
                  days_to_maturity=245) -> 0.0026 per 1 notional (L2 R51 Section 4.3)
    """
    s = _sign(position)
    m = days_to_maturity - days_to_expiry
    _pos("days_to_maturity - days_to_expiry", m)
    v = 1 / (1 + rate_to_expiry * days_to_expiry / day_basis) - (1 + fra_rate * m / day_basis) / (
        1 + rate_to_maturity * days_to_maturity / day_basis)
    return {"value": s * v * notional, "value_per_unit_long": v}


@formula("derivatives",
         formula="Payment at expiration = NP x (L - FRA)(m/360) / (1 + L m/360); long receives when L > FRA",
         curriculum=R51)
def fra_settlement(notional: float, fra_rate: float, underlying_rate: float, days_underlying: float,
                   position: str = "long", day_basis: float = 360) -> dict:
    """FRA settlement at expiration (the interest difference discounted at the underlying rate).

    Args:
        notional: notional principal. fra_rate: contract rate.
        underlying_rate: LIBOR/Euribor observed at expiration for the m-day deposit.
        days_underlying: m. position: "long" or "short". day_basis: 360.

    Returns:
        settlement (received if positive, for `position`), undiscounted_amount (if paid at the deposit maturity).

    Example:
        fra_settlement(notional=40_000_000, fra_rate=0.0525, underlying_rate=0.06, days_underlying=180)
        -> 145,631 (L3 R29 Exhibit 1)
    """
    s = _sign(position)
    _pos("days_underlying", days_underlying)
    acc = days_underlying / day_basis
    undiscounted = notional * (underlying_rate - fra_rate) * acc
    return {"settlement": s * undiscounted / (1 + underlying_rate * acc), "undiscounted_amount": s * undiscounted}


# ---------------------------------------------------------------- swaps
@formula("derivatives",
         formula="annual FS = (1 - B(h_n)) / sum B(h_j) x a_j, a_j = (h_j - h_(j-1))/360; periodic FS(0,n,m) = "
                 "(1 - B(h_n)) / sum B(h_j) = annual x m/360 for equal periods; B = 1/(1 + L days/360)",
         curriculum=R54)
def swap_fixed_rate(discount_factors: list | None = None, spot_rates: list | None = None,
                    days: list | None = None, day_basis: float = 360, periods_per_year: float | None = None) -> dict:
    """Plain-vanilla swap fixed rate from discount factors or a LIBOR spot term structure.

    Also the fixed rate of each leg of a currency swap (price each currency's swap separately) and of
    a pay-fixed equity swap. The swap rate equals the par bond coupon and the DF-weighted average of the
    implied forward rates (a swap is a portfolio of FRAs / off-market forwards).

    The annual rate weights each discount factor by its period's accrual fraction, so it is right for unequal
    periods (e.g. a short first stub); with equal periods it equals the curriculum's FS x 360/m.

    Args:
        discount_factors: B0(h_1..h_n). Or spot_rates (add-on LIBOR for each payment date) with days.
        days: days to each payment date (h_1..h_n), e.g. [90, 180, 270, 360]; accrual_j = (h_j - h_(j-1))/day_basis.
        day_basis: 360. periods_per_year: only when days are not given (discount_factors alone): equal periods
            with accrual 1/periods_per_year. Days take precedence when both are given.

    Returns:
        fixed_rate_annual, fixed_rate_periodic ((1 - B_n)/sum B, the per-period rate for equal periods; with
        unequal periods each fixed payment is fixed_rate_annual x accrual_j), accrual_fractions, discount_factors,
        sum_discount_factors, implied_forward_rates (periodic forward rate for each period, i.e. the FRA rates),
        pv_floating_via_forwards (sum fwd x B = 1 - B_n, the FRA-portfolio view of the floating leg).

    Example:
        swap_fixed_rate(spot_rates=[0.0345, 0.0358, 0.0370, 0.0375], days=[90, 180, 270, 360])
        -> periodic 0.0092, annual 0.0368 (L2 R54 Section 4.2.1)
    """
    dfs = _discount_factors(discount_factors, spot_rates, days, day_basis)
    total = sum(dfs)
    if days is not None:
        if len(days) != len(dfs):
            raise ValueError("days needs one value per discount factor")
        acc = [(float(d) - float(p)) / day_basis for d, p in zip(days, [0.0] + list(days[:-1]))]
        if any(a <= 0 for a in acc):
            raise ValueError("days must be positive and increasing")
    elif periods_per_year is not None:
        acc = [1 / _pos("periods_per_year", periods_per_year)] * len(dfs)
    else:
        raise ValueError("give days, or periods_per_year when passing discount_factors without days")
    fwds = [dfs[0] ** -1 - 1] + [dfs[i - 1] / dfs[i] - 1 for i in range(1, len(dfs))]
    return {"fixed_rate_periodic": (1 - dfs[-1]) / total,
            "fixed_rate_annual": (1 - dfs[-1]) / sum(b * a for b, a in zip(dfs, acc)),
            "accrual_fractions": acc, "discount_factors": dfs, "sum_discount_factors": total,
            "implied_forward_rates": fwds, "pv_floating_via_forwards": sum(f * b for f, b in zip(fwds, dfs))}


def _fixed_leg(c_periodic: float, dfs: list) -> float:
    return c_periodic * sum(dfs) + dfs[-1]


def _floating_leg(next_payment: float, dfs: list) -> float:
    return (1 + next_payment) * dfs[0]


@formula("derivatives",
         formula="V(pay fixed) = (1 + L_last x m/360) B(next) - [FS x sum B + B(last)]  (per 1 notional)",
         curriculum=R54)
def interest_rate_swap_value(fixed_rate: float, floating_rate_set: float, period_fraction: float,
                             discount_factors: list | None = None, spot_rates: list | None = None,
                             days: list | None = None, day_basis: float = 360, notional: float = 1.0,
                             position: str = "pay_fixed") -> dict:
    """Value of an interest rate swap after inception: floating leg minus fixed leg (bonds with par).

    Args:
        fixed_rate: annual swap fixed rate (periodic payment = fixed_rate x period_fraction).
        floating_rate_set: annual LIBOR set at the last reset (determines the next floating payment).
        period_fraction: m/360 of a payment period (0.25 quarterly, 0.5 semiannual).
        discount_factors: B_t to each remaining payment; or spot_rates and days (days from today).
        day_basis: 360. notional: notional principal. position: "pay_fixed" or "receive_fixed".

    Returns:
        value (for `position`, times notional), pv_fixed_leg, pv_floating_leg (per 1, incl. notional), discount_factors.

    Example:
        interest_rate_swap_value(fixed_rate=0.0368, floating_rate_set=0.0345, period_fraction=0.25,
                                 spot_rates=[0.0425, 0.0432, 0.0437, 0.0444], days=[30, 120, 210, 300])
        -> 0.0047 per 1 to the fixed payer (L2 R54 Section 4.2.1)
    """
    s = _sign(position, "pay_fixed", "receive_fixed")
    dfs = _discount_factors(discount_factors, spot_rates, days, day_basis)
    pv_fix = _fixed_leg(fixed_rate * period_fraction, dfs)
    pv_flt = _floating_leg(floating_rate_set * period_fraction, dfs)
    return {"value": s * (pv_flt - pv_fix) * notional, "pv_fixed_leg": pv_fix, "pv_floating_leg": pv_flt,
            "discount_factors": dfs}


@formula("derivatives",
         formula="V = receive leg - pay leg; foreign leg in domestic = PV_foreign x (NP_dom/S_0) x S_t",
         curriculum=R54)
def currency_swap_value(spot_rate_initial: float, spot_rate_now: float, period_fraction: float,
                        domestic_fixed_rate: float = 0.0, foreign_fixed_rate: float = 0.0,
                        domestic_floating_rate_set: float = 0.0, foreign_floating_rate_set: float = 0.0,
                        domestic_discount_factors: list | None = None, foreign_discount_factors: list | None = None,
                        domestic_spot_rates: list | None = None, foreign_spot_rates: list | None = None,
                        days: list | None = None, day_basis: float = 360, notional_domestic: float = 1.0,
                        pay_leg: str = "domestic_fixed", receive_leg: str = "foreign_fixed") -> dict:
    """Value of a currency swap (both notionals exchanged) for any fixed/floating leg combination.

    The foreign notional is notional_domestic / spot_rate_initial. Each leg is valued per unit of its own
    currency (fixed: c x sum B + B_n; floating: (1 + next payment) x B_1), then converted at today's spot.
    Price a new currency swap by calling swap_fixed_rate on each currency's term structure.

    Args:
        spot_rate_initial, spot_rate_now: domestic per foreign at inception and today.
        period_fraction: payment period as a fraction of a year (m/360).
        domestic_fixed_rate, foreign_fixed_rate: annual fixed rates of the swap.
        domestic_floating_rate_set, foreign_floating_rate_set: annual rates set at the last reset.
        domestic_discount_factors / domestic_spot_rates, foreign_...: term structures to the remaining dates.
        days: days to remaining payments (needed with spot rates). notional_domestic: domestic notional.
        pay_leg, receive_leg: each one of "domestic_fixed", "domestic_floating", "foreign_fixed", "foreign_floating"
            (one domestic, one foreign).

    Returns:
        value (domestic currency), leg_values (all four legs in domestic currency, notionals included),
        all_combinations (pay domestic / receive foreign for the four swaps), notional_foreign.

    Example:
        L2 R54 Example 5B: pay $ fixed 7.84%, receive EUR fixed 6.48%, S0 0.75, St 0.70 -> -0.0712 per $1
    """
    legs_ok = {"domestic_fixed", "domestic_floating", "foreign_fixed", "foreign_floating"}
    if pay_leg not in legs_ok or receive_leg not in legs_ok:
        raise ValueError(f"pay_leg and receive_leg must be in {sorted(legs_ok)}")
    if pay_leg.split("_")[0] == receive_leg.split("_")[0]:
        raise ValueError("one leg must be domestic and the other foreign")
    _pos("spot_rate_initial", spot_rate_initial)
    _pos("spot_rate_now", spot_rate_now)
    d_dfs = _discount_factors(domestic_discount_factors, domestic_spot_rates, days, day_basis)
    f_dfs = _discount_factors(foreign_discount_factors, foreign_spot_rates, days, day_basis)
    nf = notional_domestic / spot_rate_initial
    fx = nf * spot_rate_now
    legs = {
        "domestic_fixed": _fixed_leg(domestic_fixed_rate * period_fraction, d_dfs) * notional_domestic,
        "domestic_floating": _floating_leg(domestic_floating_rate_set * period_fraction, d_dfs) * notional_domestic,
        "foreign_fixed": _fixed_leg(foreign_fixed_rate * period_fraction, f_dfs) * fx,
        "foreign_floating": _floating_leg(foreign_floating_rate_set * period_fraction, f_dfs) * fx,
    }
    combos = {f"pay_{p}_receive_{r}": legs[r] - legs[p] for p in ("domestic_fixed", "domestic_floating")
              for r in ("foreign_fixed", "foreign_floating")}
    return {"value": legs[receive_leg] - legs[pay_leg], "leg_values": legs, "all_combinations": combos,
            "notional_foreign": nf}


@formula("derivatives",
         formula="V(receive equity) = S_t/S_0 - pay leg; fixed leg = FS sum B + B_n; floating = (1+next) B_1; other equity = S2_t/S2_0",
         curriculum=R54)
def equity_swap_value(index_last_reset: float, index_now: float, pay_leg: str = "fixed", fixed_rate: float = 0.0,
                      floating_rate_set: float = 0.0, period_fraction: float = 0.0,
                      other_index_last_reset: float | None = None, other_index_now: float | None = None,
                      discount_factors: list | None = None, spot_rates: list | None = None,
                      days: list | None = None, day_basis: float = 360, notional: float = 1.0,
                      position: str = "receive_equity") -> dict:
    """Value of an equity swap: receive the equity return vs pay fixed, floating, or another equity return.

    Args:
        index_last_reset: equity index at the last settlement (or at inception). index_now: index today.
        pay_leg: "fixed", "floating" or "equity" (the leg paid by the receive-equity party).
        fixed_rate: annual fixed rate (periodic = fixed_rate x period_fraction); the same as the
            plain-vanilla swap rate (swap_fixed_rate).
        floating_rate_set: annual rate set at the last reset. period_fraction: m/360.
        other_index_last_reset, other_index_now: the paid index, for pay_leg="equity".
        discount_factors / spot_rates + days: term structure to remaining payments (fixed/floating legs).
        notional: notional principal. position: "receive_equity" or "pay_equity".

    Returns:
        value (for `position`, times notional), equity_leg, pay_leg_value (per 1 notional).

    Example:
        equity_swap_value(index_last_reset=10033.27, index_now=9955.14, pay_leg="fixed", fixed_rate=0.0784,
                          period_fraction=0.5, spot_rates=[0.071, 0.074], days=[90, 270]) -> -0.0309
        (L2 R54 Example 6B)
    """
    s = _sign(position, "receive_equity", "pay_equity")
    _pos("index_last_reset", index_last_reset)
    eq = index_now / index_last_reset
    if pay_leg == "fixed":
        other = _fixed_leg(fixed_rate * period_fraction, _discount_factors(discount_factors, spot_rates, days, day_basis))
    elif pay_leg == "floating":
        other = _floating_leg(floating_rate_set * period_fraction,
                              _discount_factors(discount_factors, spot_rates, days, day_basis))
    elif pay_leg == "equity":
        if not other_index_last_reset or other_index_now is None:
            raise ValueError("pay_leg='equity' needs other_index_last_reset and other_index_now")
        other = other_index_now / other_index_last_reset
    else:
        raise ValueError("pay_leg must be 'fixed', 'floating' or 'equity'")
    return {"value": s * (eq - other) * notional, "equity_leg": eq, "pay_leg_value": other}


@formula("derivatives",
         formula="Payer payoff = NP max(0, FS - x)(m/360) sum B; receiver = NP max(0, x - FS)(m/360) sum B; = put/call on a par-x coupon bond",
         curriculum=R54)
def swaption_payoff(exercise_rate: float, period_fraction: float, discount_factors: list | None = None,
                    spot_rates: list | None = None, days: list | None = None, market_swap_rate: float | None = None,
                    notional: float = 1.0, swaption_type: str = "payer", day_basis: float = 360) -> dict:
    """Value at expiration of a European payer or receiver swaption (cash-settled annuity).

    A payer swaption is a put on a bond with coupon = exercise rate; a receiver swaption is a call on it.

    Args:
        exercise_rate: annual strike swap rate. period_fraction: m/360 of the underlying swap's periods.
        discount_factors / spot_rates + days: term structure at expiration for the swap payment dates.
        market_swap_rate: annual market swap rate at expiration; computed from the discount factors if omitted.
        notional: notional principal. swaption_type: "payer" or "receiver". day_basis: 360.

    Returns:
        payoff, market_swap_rate, payment_per_period, equivalent_bond_value (coupon = exercise rate, par 1),
        sum_discount_factors.

    Example:
        swaption_payoff(exercise_rate=0.036, period_fraction=0.25, market_swap_rate=0.0368, notional=20e6,
                        discount_factors=[0.9914, 0.9824, 0.9730, 0.9639]) -> 15,643 (L2 R54 Section 6.3)
    """
    if swaption_type not in ("payer", "receiver"):
        raise ValueError("swaption_type must be 'payer' or 'receiver'")
    _pos("period_fraction", period_fraction)
    dfs = _discount_factors(discount_factors, spot_rates, days, day_basis)
    if market_swap_rate is None:
        market_swap_rate = (1 - dfs[-1]) / sum(dfs) / period_fraction
    diff = market_swap_rate - exercise_rate if swaption_type == "payer" else exercise_rate - market_swap_rate
    per_period = notional * max(0.0, diff) * period_fraction
    return {"payoff": per_period * sum(dfs), "market_swap_rate": market_swap_rate,
            "payment_per_period": per_period, "equivalent_bond_value": _fixed_leg(exercise_rate * period_fraction, dfs),
            "sum_discount_factors": sum(dfs)}
