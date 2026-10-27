"""Fixed income: bond pricing, accrued interest, matrix pricing, horizon returns (L1).

Shared helpers used by the other fi_* modules live here (underscore names, not registered).
"""
from __future__ import annotations

import calendar
import datetime as _dt

from scipy.optimize import brentq

from ._registry import formula

L1_VAL = "L1 FI: Introduction to Fixed-Income Valuation (2014 R54)"
L1_RISK = "L1 FI: Understanding Fixed-Income Risk and Return (2014 R55)"


# ---------------------------------------------------------------- helpers
def _periods(years: float, freq: float, allow_fraction: bool = False) -> float:
    if freq <= 0:
        raise ValueError("freq must be a positive number of periods per year")
    if years <= 0:
        raise ValueError("years must be positive")
    n = years * freq
    if allow_fraction:
        return n
    if abs(n - round(n)) > 1e-9:
        raise ValueError(f"years * freq must be a whole number of periods, got {n}")
    return int(round(n))


def _rate_ok(r_per_period: float) -> None:
    if r_per_period <= -1:
        raise ValueError("yield per period must be greater than -100%")


def _level_cfs(face: float, coupon_rate: float, n: int, freq: float, redemption: float | None = None) -> list:
    pmt = face * coupon_rate / freq
    cfs = [pmt] * n
    cfs[-1] += face if redemption is None else redemption
    return cfs


def _pv(cfs: list, r: float, fraction: float = 0.0) -> float:
    """PV of cash flows at periods 1..n (per-period rate r), settlement `fraction` into period 1."""
    return sum(cf / (1 + r) ** (k - fraction) for k, cf in enumerate(cfs, start=1))


def _solve(fn, lo: float = -0.99, hi: float = 10.0, what: str = "rate") -> float:
    try:
        return brentq(fn, lo, hi, xtol=1e-15, maxiter=1000)
    except ValueError as exc:  # no sign change: inputs inconsistent
        raise ValueError(f"could not solve for {what}: no solution in [{lo}, {hi}] ({exc})") from exc


def _pv_dur_conv(cfs: list, times: list, ytm: float, freq: float) -> dict:
    """PV, Macaulay duration (years), convexity (years^2) of cash flows at times (years), stated yield."""
    r = ytm / freq
    _rate_ok(r)
    pvs = [cf / (1 + r) ** (freq * t) for cf, t in zip(cfs, times)]
    pv = sum(pvs)
    if pv == 0:
        raise ValueError("present value of the cash flows is zero")
    tp = [freq * t for t in times]
    mac = sum(t * v for t, v in zip(tp, pvs)) / pv / freq
    conv = sum(t * (t + 1) * v for t, v in zip(tp, pvs)) / (pv * (1 + r) ** 2) / freq ** 2
    disp = sum((t / freq - mac) ** 2 * v for t, v in zip(tp, pvs)) / pv
    return {"pv": pv, "macaulay_duration": mac, "modified_duration": mac / (1 + r), "convexity": conv,
            "dispersion": disp}


def _date(x) -> _dt.date:
    if isinstance(x, _dt.date):
        return x
    try:
        return _dt.date.fromisoformat(str(x))
    except ValueError as exc:
        raise ValueError(f"dates must be ISO strings YYYY-MM-DD, got {x!r}") from exc


def _add_months(d: _dt.date, months: int) -> _dt.date:
    y, m = divmod(d.month - 1 + months, 12)
    year, month = d.year + y, m + 1
    return _dt.date(year, month, min(d.day, calendar.monthrange(year, month)[1]))


def _days_30_360(d1: _dt.date, d2: _dt.date) -> int:
    """US 30/360 (bond basis): day 31 -> 30; end day 31 -> 30 when start day is 30 or 31."""
    a, b = min(d1.day, 30), d2.day
    if b == 31 and a == 30:
        b = 30
    return 360 * (d2.year - d1.year) + 30 * (d2.month - d1.month) + (b - a)


def _coupon_dates(settlement: _dt.date, maturity: _dt.date, freq: int) -> tuple:
    if freq not in (1, 2, 3, 4, 6, 12):
        raise ValueError("freq must divide 12 (1, 2, 3, 4, 6, 12) for date-based calculations")
    if settlement >= maturity:
        raise ValueError("settlement must be before maturity")
    step, k = 12 // freq, 1
    while _add_months(maturity, -k * step) > settlement:
        k += 1
    return _add_months(maturity, -k * step), _add_months(maturity, -(k - 1) * step), k


_DAY_COUNTS = ("30/360", "actual/actual", "actual/360", "actual/365")


# ---------------------------------------------------------------- pricing
@formula("fixed_income", formula="P = sum_{t=1..N} (c*F/m)/(1+y/m)^t + F/(1+y/m)^N", curriculum=L1_VAL)
def bond_price(face: float = 100.0, coupon_rate: float = 0.0, ytm: float = 0.0, years: float = 1.0,
               freq: int = 2) -> dict:
    """Price of an option-free bond on a coupon date from its yield to maturity (zero-coupon when coupon_rate=0).

    Args:
        face: par value.
        coupon_rate: annual coupon rate, decimal.
        ytm: annual yield to maturity (stated rate, compounded `freq` times a year), decimal.
        years: years to maturity; years * freq must be whole unless coupon_rate = 0 (zero-coupon).
        freq: coupons (compounding periods) per year.

    Returns:
        price, pv_coupons, pv_face, periods, coupon_per_period, yield_per_period.

    Example:
        bond_price(face=100, coupon_rate=0.05, ytm=0.06, years=10, freq=2) -> price 92.56
    """
    zero = coupon_rate == 0
    n = _periods(years, freq, allow_fraction=zero)
    r = ytm / freq
    _rate_ok(r)
    pmt = face * coupon_rate / freq
    pv_f = face / (1 + r) ** n
    pv_c = 0.0 if zero else _pv([pmt] * n, r)
    return {"price": pv_c + pv_f, "pv_coupons": pv_c, "pv_face": pv_f, "periods": n,
            "coupon_per_period": pmt, "yield_per_period": r}


@formula("fixed_income", formula="P = sum CF_t/(1+z_t/m)^t ; YTM solves P = sum CF_t/(1+y/m)^t", curriculum=L1_VAL)
def bond_price_spot_curve(spot_rates: list, coupon_rate: float = 0.0, face: float = 100.0, freq: int = 1,
                          cash_flows: list | None = None) -> dict:
    """No-arbitrage price of a bond from a spot (zero) curve, with the implied yield to maturity.

    Args:
        spot_rates: spot rate for each period 1..N (annual stated rates compounded `freq` times a year), decimals.
        coupon_rate: annual coupon rate (ignored when cash_flows is given).
        face: par value.
        freq: periods per year.
        cash_flows: optional explicit cash flows for periods 1..N (overrides coupon/face).

    Returns:
        price, ytm, discount_factors, pv_cash_flows, cash_flows.

    Example:
        spots 2%, 3%, 4%, 5% annual coupon 3-year -> price 102.960, ytm 3.935% (L1 R54 §2.4)
    """
    if not spot_rates:
        raise ValueError("spot_rates must be a non-empty list")
    cfs = list(cash_flows) if cash_flows is not None else _level_cfs(face, coupon_rate, len(spot_rates), freq)
    if len(cfs) != len(spot_rates):
        raise ValueError("cash_flows and spot_rates must have the same length")
    dfs = []
    for k, z in enumerate(spot_rates, start=1):
        _rate_ok(z / freq)
        dfs.append((1 + z / freq) ** -k)
    pvs = [cf * df for cf, df in zip(cfs, dfs)]
    price = sum(pvs)
    r = _solve(lambda y: _pv(cfs, y) - price, what="yield to maturity")
    return {"price": price, "ytm": r * freq, "discount_factors": dfs, "pv_cash_flows": pvs, "cash_flows": cfs}


@formula("fixed_income", formula="PV_full = PV_last_coupon*(1+r)^(t/T); AI = (t/T)*PMT; PV_flat = PV_full - AI",
         curriculum=L1_VAL)
def bond_price_full_flat(coupon_rate: float, ytm: float, periods: int, fraction: float, freq: int = 2,
                         face: float = 100.0) -> dict:
    """Full (dirty), accrued interest and flat (clean) price of a bond between coupon dates.

    Args:
        coupon_rate: annual coupon rate, decimal.
        ytm: annual yield (stated, `freq` compounding), decimal.
        periods: N, coupon periods to maturity counted from the last coupon date.
        fraction: t/T, fraction of the current coupon period elapsed (0 <= t/T < 1).
        freq: coupons per year.
        face: par value.

    Returns:
        full_price, accrued_interest, flat_price, pv_at_last_coupon.

    Example:
        5% semiannual, N=18, t/T=88/181, ytm 4.80% -> full 102.624323, AI 1.215470, flat 101.408853
    """
    if not 0 <= fraction < 1:
        raise ValueError("fraction must be in [0, 1)")
    n = int(periods)
    if n < 1 or n != periods:
        raise ValueError("periods must be a positive integer")
    r = ytm / freq
    _rate_ok(r)
    pv0 = _pv(_level_cfs(face, coupon_rate, n, freq), r)
    full = pv0 * (1 + r) ** fraction
    ai = fraction * face * coupon_rate / freq
    return {"full_price": full, "accrued_interest": ai, "flat_price": full - ai, "pv_at_last_coupon": pv0}


@formula("fixed_income", formula="AI = (t/T) x coupon per period; 30/360 or actual/actual day counts",
         curriculum=L1_VAL)
def bond_accrued_interest(last_coupon: str, settlement: str, next_coupon: str, coupon_rate: float,
                          freq: int = 2, day_count: str = "actual/actual", face: float = 100.0) -> dict:
    """Accrued interest between coupon dates under a day-count convention.

    Args:
        last_coupon, settlement, next_coupon: ISO dates (YYYY-MM-DD).
        coupon_rate: annual coupon rate, decimal.
        freq: coupons per year.
        day_count: "30/360" (US bond basis), "actual/actual", "actual/360" or "actual/365"
            (the last two accrue coupon_rate * face * days / 360 or 365, the FRN/money-market style).
        face: par value.

    Returns:
        accrued_interest, days_accrued (t), days_in_period (T), fraction (t/T).

    Example:
        15 May -> 27 Jun, 4.375% semiannual: actual/actual 43/184 -> 0.511209; 30/360 42/180 -> 0.510417
    """
    d0, ds, d1 = _date(last_coupon), _date(settlement), _date(next_coupon)
    if not d0 <= ds < d1:
        raise ValueError("need last_coupon <= settlement < next_coupon")
    if day_count not in _DAY_COUNTS:
        raise ValueError(f"day_count must be one of {_DAY_COUNTS}")
    pmt = face * coupon_rate / freq
    if day_count == "30/360":
        t, big_t = _days_30_360(d0, ds), 360 / freq
        ai = t / big_t * pmt
    else:
        t, big_t = (ds - d0).days, (d1 - d0).days
        ai = t / big_t * pmt if day_count == "actual/actual" else \
            face * coupon_rate * t / (360 if day_count == "actual/360" else 365)
    return {"accrued_interest": ai, "days_accrued": t, "days_in_period": big_t, "fraction": t / big_t}


@formula("fixed_income", formula="coupon schedule from maturity; PV_full = PV x (1+r)^(t/T); flat = full - AI",
         curriculum=L1_VAL)
def bond_price_settlement(settlement: str, maturity: str, coupon_rate: float, ytm: float, freq: int = 2,
                          day_count: str = "actual/actual", face: float = 100.0) -> dict:
    """Full, accrued and flat price for a settlement date, building the coupon schedule back from maturity.

    Args:
        settlement, maturity: ISO dates (YYYY-MM-DD). Coupons fall on the maturity day-of-month every 12/freq months.
        coupon_rate: annual coupon rate. ytm: annual yield (street convention, `freq` compounding).
        freq: coupons per year (must divide 12).
        day_count: "30/360", "actual/actual", "actual/360", "actual/365" (sets t/T and accrued interest).
        face: par value.

    Returns:
        full_price, accrued_interest, flat_price, fraction, periods, last_coupon, next_coupon.

    Example:
        settle 2015-06-18, maturity 2026-09-19, 6%, ytm 5.80%, 30/360 -> full 103.108770, flat 101.625437
    """
    ds, dm = _date(settlement), _date(maturity)
    last, nxt, n = _coupon_dates(ds, dm, freq)
    ai = bond_accrued_interest(last.isoformat(), ds.isoformat(), nxt.isoformat(), coupon_rate, freq, day_count, face)
    frac = ai["fraction"] if day_count in ("30/360", "actual/actual") else (ds - last).days / (nxt - last).days
    px = bond_price_full_flat(coupon_rate, ytm, n, frac, freq, face)
    return {"full_price": px["full_price"], "accrued_interest": ai["accrued_interest"],
            "flat_price": px["full_price"] - ai["accrued_interest"], "fraction": frac, "periods": n,
            "last_coupon": last.isoformat(), "next_coupon": nxt.isoformat()}


@formula("fixed_income", formula="YTM of comparables -> average by maturity -> linear interpolation -> price",
         curriculum=L1_VAL)
def bond_matrix_pricing(target_coupon_rate: float, target_years: float, comparables: list, freq: int = 2,
                        face: float = 100.0) -> dict:
    """Matrix pricing of an illiquid bond from comparable bonds' prices.

    Args:
        target_coupon_rate: annual coupon of the bond to price.
        target_years: its years to maturity.
        comparables: list of [years, coupon_rate, price] for similar-credit bonds (same `freq`).
        freq: coupons per year. face: par value.

    Returns:
        price, estimated_ytm, comparable_ytms, average_ytm_by_maturity.

    Example:
        2y 3% @98.5, 2y 5% @102.25, 5y 2% @90.25, 5y 4% @99.125 -> 3y 4% bond: ytm 3.9318%, price 100.191
    """
    by_m: dict = {}
    ytms = []
    for years, cpn, price in comparables:
        n = _periods(years, freq)
        cfs = _level_cfs(face, cpn, n, freq)
        y = _solve(lambda r: _pv(cfs, r) - price, what="comparable yield") * freq
        ytms.append(y)
        by_m.setdefault(float(years), []).append(y)
    avg = {m: sum(v) / len(v) for m, v in sorted(by_m.items())}
    mats = list(avg)
    if target_years in avg or len(mats) == 1:
        est = avg.get(float(target_years), avg[mats[0]])
    else:  # linear interpolation; beyond the ends, extrapolate from the two nearest maturities
        lo = max((m for m in mats if m < target_years), default=None)
        hi = min((m for m in mats if m > target_years), default=None)
        if lo is None:
            lo, hi = mats[0], mats[1]
        elif hi is None:
            lo, hi = mats[-2], mats[-1]
        est = avg[lo] + (target_years - lo) / (hi - lo) * (avg[hi] - avg[lo])
    px = bond_price(face, target_coupon_rate, est, target_years, freq)["price"]
    return {"price": px, "estimated_ytm": est, "comparable_ytms": ytms, "average_ytm_by_maturity": avg}


@formula("fixed_income", formula="Total = FV(reinvested coupons) + sale price; horizon yield = (Total/P0)^(1/n) - 1",
         curriculum=L1_RISK)
def bond_horizon_return(purchase_price: float, coupon_rate: float, years: float, horizon_years: float,
                        reinvestment_rate: float, sale_yield: float, freq: int = 1, face: float = 100.0) -> dict:
    """Realized horizon yield of a fixed-rate bond: reinvested coupons plus sale price (sources of return).

    Args:
        purchase_price: price paid per `face` on a coupon date.
        coupon_rate: annual coupon rate. years: years to maturity at purchase.
        horizon_years: holding period (<= years); bond is sold right after the coupon at the horizon.
        reinvestment_rate: annual rate at which coupons are reinvested (stated, `freq` compounding).
        sale_yield: yield to maturity at the sale date.
        freq: coupons per year. face: par value.

    Returns:
        horizon_yield (annual stated), fv_coupons, interest_on_interest, sale_price, carrying_value,
        capital_gain (sale price minus constant-yield carrying value), total_return_amount, original_ytm.

    Example:
        10y 8% annual bought at 85.503075, sold after 4y with rates at 11.40% -> horizon yield 9.67%
    """
    n, nh = _periods(years, freq), _periods(horizon_years, freq)
    if nh > n:
        raise ValueError("horizon_years cannot exceed years to maturity")
    pmt = face * coupon_rate / freq
    rr = reinvestment_rate / freq
    _rate_ok(rr)
    fv_c = sum(pmt * (1 + rr) ** (nh - k) for k in range(1, nh + 1))
    cfs = _level_cfs(face, coupon_rate, n, freq)
    y0 = _solve(lambda r: _pv(cfs, r) - purchase_price, what="original yield")
    rem = n - nh
    sale = face if rem == 0 else _pv(_level_cfs(face, coupon_rate, rem, freq), sale_yield / freq)
    carry = face if rem == 0 else _pv(_level_cfs(face, coupon_rate, rem, freq), y0)
    total = fv_c + sale
    hy = (total / purchase_price) ** (1 / nh) - 1
    return {"horizon_yield": hy * freq, "fv_coupons": fv_c, "interest_on_interest": fv_c - pmt * nh,
            "sale_price": sale, "carrying_value": carry, "capital_gain": sale - carry,
            "total_return_amount": total, "original_ytm": y0 * freq}
