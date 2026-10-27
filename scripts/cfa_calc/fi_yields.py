"""Fixed income: yield measures, money-market rates and floating-rate notes (L1)."""
from __future__ import annotations

import math

from ._registry import formula
from .fi_pricing import L1_RISK, L1_VAL, _level_cfs, _periods, _pv, _rate_ok, _solve


@formula("fixed_income", formula="solve P = sum PMT/(1+r)^(t-f) + Redemption/(1+r)^(N-f); YTM = r x m",
         curriculum=L1_VAL)
def bond_ytm(price: float, coupon_rate: float, years: float, freq: int = 2, face: float = 100.0,
             redemption: float | None = None, fraction: float = 0.0) -> dict:
    """Yield to maturity (IRR) of a fixed-rate bond, stated for periodicity `freq`.

    Args:
        price: full price per `face` (equals flat price on a coupon date).
        coupon_rate: annual coupon rate. years: years to maturity from the last coupon date.
        freq: coupons per year. face: par value.
        redemption: amount repaid at the end (default face; use the call/put price for yield-to-call/put).
        fraction: t/T elapsed in the current coupon period (0 on a coupon date).

    Returns:
        ytm (annual, periodicity freq), yield_per_period, effective_annual_yield.

    Example:
        4-year 5% annual priced at 105 -> ytm 3.634%
    """
    if price <= 0:
        raise ValueError("price must be positive")
    n = _periods(years, freq)
    cfs = _level_cfs(face, coupon_rate, n, freq, redemption)
    r = _solve(lambda y: _pv(cfs, y, fraction) - price, what="yield to maturity")
    return {"ytm": r * freq, "yield_per_period": r, "effective_annual_yield": (1 + r) ** freq - 1}


@formula("fixed_income", formula="solve P = sum PMT/(1+r)^t + Exercise price/(1+r)^T_exercise",
         curriculum=L1_VAL)
def bond_yield_to_call_put(price: float, coupon_rate: float, years_to_exercise: float, exercise_price: float,
                           freq: int = 2, face: float = 100.0) -> dict:
    """Yield to call (or to put): the YTM computed to an exercise date with the call/put price as redemption.

    Args:
        price: price per `face` on a coupon date. coupon_rate: annual coupon.
        years_to_exercise: years to the call (put) date. exercise_price: call (put) price per `face`.
        freq: coupons per year. face: par value.

    Returns:
        yield (annual, periodicity freq), yield_per_period.

    Example:
        7y 8% annual at 105, first call in 4 years at 102 -> yield to first call 6.975%
    """
    out = bond_ytm(price, coupon_rate, years_to_exercise, freq, face, redemption=exercise_price)
    return {"yield": out["ytm"], "yield_per_period": out["yield_per_period"]}


@formula("fixed_income", formula="YTW = min(YTM, yield to each call date)", curriculum=L1_VAL)
def bond_yield_to_worst(price: float, coupon_rate: float, years: float, call_schedule: list, freq: int = 2,
                        face: float = 100.0) -> dict:
    """Yield to worst of a callable bond: the lowest of the yields to each call date and the yield to maturity.

    Args:
        price: price per `face` on a coupon date. coupon_rate: annual coupon. years: years to maturity.
        call_schedule: list of [years_to_call, call_price].
        freq: coupons per year. face: par value.

    Returns:
        yield_to_worst, worst_date_years, yields (list of [years, redemption, yield]), ytm.

    Example:
        7y 8% at 105, calls 102 @4y, 101 @5y, 100 @6y -> YTW 6.953% (third call)
    """
    rows = [[float(t), float(k), bond_yield_to_call_put(price, coupon_rate, t, k, freq, face)["yield"]]
            for t, k in call_schedule]
    ytm = bond_ytm(price, coupon_rate, years, freq, face)["ytm"]
    rows.append([float(years), float(face), ytm])
    worst = min(rows, key=lambda r: r[2])
    return {"yield_to_worst": worst[2], "worst_date_years": worst[0], "yields": rows, "ytm": ytm}


@formula("fixed_income", formula="Current yield = annual coupon / flat price; "
         "Simple yield = (annual coupon + (F - P)/years) / flat price", curriculum=L1_VAL)
def bond_current_and_simple_yield(coupon_rate: float, flat_price: float, face: float = 100.0,
                                  years: float | None = None) -> dict:
    """Current yield and (if years given) simple yield with straight-line amortization of discount/premium.

    Args:
        coupon_rate: annual coupon rate. flat_price: flat (clean) price per `face`.
        face: par value. years: years to maturity (needed for the simple yield).

    Returns:
        current_yield, simple_yield (None when years is not given).

    Example:
        2% coupon priced at 95 -> current yield 2.105%
    """
    if flat_price <= 0:
        raise ValueError("flat_price must be positive")
    cpn = coupon_rate * face
    simple = None if years is None else (cpn + (face - flat_price) / years) / flat_price
    return {"current_yield": cpn / flat_price, "simple_yield": simple}


@formula("fixed_income", formula="(1 + APR_m/m)^m = (1 + APR_n/n)^n ; continuous: e^APR", curriculum=L1_VAL)
def yield_periodicity_convert(rate: float, from_freq: float, to_freq: float) -> dict:
    """Convert an annual rate between periodicities (compounding frequencies); 0 means continuous compounding.

    Args:
        rate: annual stated rate, decimal. from_freq, to_freq: periods per year (may be fractional,
            e.g. 365/90 for a money-market rate); 0 = continuous.

    Returns:
        rate (converted), effective_annual_rate.

    Example:
        3.582% semiannual -> 3.566% quarterly, 3.556% monthly
    """
    if from_freq < 0 or to_freq < 0:
        raise ValueError("frequencies must be >= 0 (0 = continuous)")
    growth = math.exp(rate) if from_freq == 0 else (1 + rate / from_freq) ** from_freq
    if growth <= 0:
        raise ValueError("rate per period must exceed -100%")
    out = math.log(growth) if to_freq == 0 else to_freq * (growth ** (1 / to_freq) - 1)
    return {"rate": out, "effective_annual_rate": growth - 1}


@formula("fixed_income", formula="solve Price = sum CF_i/(1+y/m)^(m x t_i)", curriculum=L1_RISK)
def cash_flow_yield(price: float, cash_flows: list, times: list, freq: float = 2) -> dict:
    """IRR of arbitrary dated cash flows: portfolio cash-flow yield, MBS cash-flow yield, street vs true yield.

    Street-convention yield uses the scheduled payment times; the true yield uses the actual
    (weekend/holiday-adjusted) times, so it is never higher.

    Args:
        price: amount paid today (positive).
        cash_flows: future receipts. times: their times in years from today.
        freq: periodicity of the stated yield.

    Returns:
        yield (annual, periodicity freq), yield_per_period, effective_annual_yield.

    Example:
        pay 19.6m, receive 10m at 1y and 100m at 30y (annual) -> 7.8611%
    """
    if len(cash_flows) != len(times) or not cash_flows:
        raise ValueError("cash_flows and times must be non-empty and the same length")
    if any(t < 0 for t in times):
        raise ValueError("times must be >= 0")
    r = _solve(lambda y: sum(cf / (1 + y) ** (freq * t) for cf, t in zip(cash_flows, times)) - price,
               what="cash flow yield")
    return {"yield": r * freq, "yield_per_period": r, "effective_annual_yield": (1 + r) ** freq - 1}


_BASES = ("discount", "add_on")


def _mm_check(days: float, basis: str, days_in_year: float) -> None:
    if days <= 0 or days_in_year <= 0:
        raise ValueError("days and days_in_year must be positive")
    if basis not in _BASES:
        raise ValueError(f"basis must be one of {_BASES}")


@formula("fixed_income", formula="Discount: PV = FV(1 - Days/Year x DR); Add-on: PV = FV/(1 + Days/Year x AOR)",
         curriculum=L1_VAL)
def money_market_price(days: float, rate: float, basis: str = "discount", days_in_year: float = 360,
                       fv: float = 100.0, principal: float | None = None) -> dict:
    """Price of a money-market instrument quoted on a discount-rate or add-on-rate basis.

    Args:
        days: days from settlement to maturity. rate: quoted annual rate, decimal.
        basis: "discount" (T-bills, CP, BAs) or "add_on" (CDs, repos, Libor).
        days_in_year: 360 or 365 per the quote convention.
        fv: amount paid at maturity (face for discount instruments, redemption incl. interest for add-on).
        principal: add-on only: give the principal instead to get the redemption amount.

    Returns:
        price, redemption, interest.

    Example:
        91-day T-bill USD10m at 2.25% discount (360) -> price 9,943,125
    """
    _mm_check(days, basis, days_in_year)
    f = days / days_in_year * rate
    if principal is not None:
        if basis != "add_on":
            raise ValueError("principal is only used with basis='add_on'")
        fv = principal * (1 + f)
        return {"price": principal, "redemption": fv, "interest": fv - principal}
    pv = fv * (1 - f) if basis == "discount" else fv / (1 + f)
    return {"price": pv, "redemption": fv, "interest": fv - pv}


@formula("fixed_income", formula="DR = (Year/Days)(FV-PV)/FV ; AOR = (Year/Days)(FV-PV)/PV", curriculum=L1_VAL)
def money_market_rate(pv: float, fv: float, days: float, basis: str = "add_on", days_in_year: float = 365) -> dict:
    """Quoted money-market rate implied by a purchase price and an amount received (discount or add-on basis).

    Args:
        pv: amount invested. fv: amount received (maturity value or sale price).
        days: holding days. basis: "discount" or "add_on". days_in_year: 360 or 365.

    Returns:
        rate.

    Example:
        CAD10,000,000 -> 10,060,829 in 45 days, add-on 365 -> 4.934%
    """
    _mm_check(days, basis, days_in_year)
    if pv <= 0 or fv <= 0:
        raise ValueError("pv and fv must be positive")
    base = fv if basis == "discount" else pv
    return {"rate": days_in_year / days * (fv - pv) / base}


@formula("fixed_income", formula="BEY = (365/Days)(FV-PV)/PV ; then periodicity 365/Days -> 2",
         curriculum=L1_VAL)
def money_market_bey(rate: float, days: float, basis: str = "discount", days_in_year: float = 360) -> dict:
    """Bond-equivalent yield (365-day add-on rate) of a money-market quote, plus its semiannual-bond-basis rate.

    Args:
        rate: quoted annual rate. days: days to maturity.
        basis: "discount" or "add_on". days_in_year: 360 or 365 for the quote.

    Returns:
        bond_equivalent_yield, price_per_100, semiannual_bond_basis_yield.

    Example:
        180-day, 4.33% discount (360) -> BEY 4.487%
    """
    px = money_market_price(days, rate, basis, days_in_year, fv=100.0)["price"]
    bey = money_market_rate(px, 100.0, days, "add_on", 365)["rate"]
    sbb = yield_periodicity_convert(bey, 365 / days, 2)["rate"]
    return {"bond_equivalent_yield": bey, "price_per_100": px, "semiannual_bond_basis_yield": sbb}


def _frn_cfs(index: float, quoted_margin: float, n: int, freq: float, face: float) -> list:
    return _level_cfs(face, index + quoted_margin, n, freq)


@formula("fixed_income", formula="PV = sum [(Index+QM) F/m]/(1+(Index+DM)/m)^t + F/(1+(Index+DM)/m)^N",
         curriculum=L1_VAL)
def frn_price(index: float, quoted_margin: float, discount_margin: float, years: float, freq: int = 4,
              face: float = 100.0) -> dict:
    """Price of a floating-rate note on a reset date (simplified model: flat reference rate).

    Args:
        index: reference rate (e.g. Libor), annual decimal. quoted_margin: spread paid over the index.
        discount_margin: required margin. years: years to maturity. freq: resets per year. face: par.

    Returns:
        price, coupon_per_period, discount_rate_per_period.

    Example:
        2y, 6m Libor 1.25% + 0.50%, DM 0.40% -> 100.196
    """
    n = _periods(years, freq)
    r = (index + discount_margin) / freq
    _rate_ok(r)
    cfs = _frn_cfs(index, quoted_margin, n, freq, face)
    return {"price": _pv(cfs, r), "coupon_per_period": cfs[0], "discount_rate_per_period": r}


@formula("fixed_income", formula="solve PV = FRN pricing equation for DM", curriculum=L1_VAL)
def frn_discount_margin(price: float, index: float, quoted_margin: float, years: float, freq: int = 4,
                        face: float = 100.0) -> dict:
    """Discount (required) margin of a floating-rate note given its price on a reset date.

    Args:
        price: price per `face`. index: reference rate. quoted_margin: QM. years: to maturity.
        freq: resets per year. face: par.

    Returns:
        discount_margin, discount_margin_bp, discount_rate_per_period.

    Example:
        5y, 3m Libor 1.10% + 0.75%, price 95.50 -> DM 1.718%
    """
    n = _periods(years, freq)
    cfs = _frn_cfs(index, quoted_margin, n, freq, face)
    r = _solve(lambda y: _pv(cfs, y) - price, what="discount rate")
    dm = r * freq - index
    return {"discount_margin": dm, "discount_margin_bp": dm * 1e4, "discount_rate_per_period": r}


@formula("fixed_income", formula="Rate MacDur ~ time to next reset; spread duration = -(dP/dDM)/P",
         curriculum=L1_RISK)
def frn_duration(years: float, index: float, quoted_margin: float, discount_margin: float, freq: int = 4,
                 fraction: float = 0.0, face: float = 100.0, dy: float = 0.0001) -> dict:
    """Interest-rate (index) duration and spread duration of a floating-rate note.

    The index duration is short (time to the next reset) because coupons reset; the spread duration
    is long (close to a fixed-rate bond of the same maturity) because the quoted margin is fixed.

    Args:
        years: years to maturity from the last reset. index, quoted_margin, discount_margin: annual decimals.
        freq: resets per year. fraction: t/T elapsed since the last reset. face: par. dy: DM bump.

    Returns:
        index_macaulay_duration, index_modified_duration, spread_duration, full_price.

    Example:
        5y quarterly FRN at par on a reset date -> index duration 0.25, spread duration ~4.7
    """
    if not 0 <= fraction < 1:
        raise ValueError("fraction must be in [0, 1)")
    n = _periods(years, freq)
    cfs = _frn_cfs(index, quoted_margin, n, freq, face)

    def full(dm: float) -> float:
        r = (index + dm) / freq
        _rate_ok(r)
        return _pv(cfs, r, fraction)

    p0 = full(discount_margin)
    mac = (1 - fraction) / freq
    return {"index_macaulay_duration": mac, "index_modified_duration": mac / (1 + (index + discount_margin) / freq),
            "spread_duration": (full(discount_margin - dy) - full(discount_margin + dy)) / (2 * dy * p0),
            "full_price": p0}
