"""Time value of money, NPV/IRR, return measurement and money-market yields (L1 Quant R5-R6).

Rates are annual decimals compounded `freq` times a year unless a parameter says it is per period.
Cash-flow lists start at t = 0; outflows negative, inflows positive.
"""
from __future__ import annotations

import math

import numpy as np
from scipy.optimize import brentq

from ._registry import formula

_R5 = "L1 Quant R5: The Time Value of Money"
_R6 = "L1 Quant R6: Discounted Cash Flow Applications"


def _num(name: str, x) -> float:
    try:
        v = float(x)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be a number, got {x!r}") from None
    if not math.isfinite(v):
        raise ValueError(f"{name} must be finite, got {x!r}")
    return v


def _freq(freq) -> int:
    if int(freq) != freq or freq < 1:
        raise ValueError(f"freq must be a positive whole number of periods per year, got {freq!r}")
    return int(freq)


def _n_periods(years, freq: int) -> float:
    n = _num("years", years) * freq
    if n < 0:
        raise ValueError("years must be >= 0")
    return n


def _whole(n: float, what: str) -> int:
    if abs(n - round(n)) > 1e-9:
        raise ValueError(f"{what} must be a whole number of periods, got {n}")
    return int(round(n))


def _flows(cash_flows, min_len: int = 2) -> np.ndarray:
    cf = np.array([_num("cash_flows", c) for c in cash_flows], dtype=float)
    if cf.size < min_len:
        raise ValueError(f"cash_flows needs at least {min_len} values (first one at t = 0)")
    return cf


def _pv_flows(cf: np.ndarray, r: float) -> float:
    if r <= -1:
        raise ValueError("rate must be greater than -100%")
    return float(np.sum(cf / (1.0 + r) ** np.arange(cf.size)))


# ---------------------------------------------------------------- lump sums and annuities

@formula("quant", formula="FV = PV(1 + r/m)^(mN); continuous: FV = PV e^(rN)", curriculum=_R5)
def fv_lump_sum(pv: float, rate: float, years: float, freq: int = 1, continuous: bool = False) -> dict:
    """Future value of a single amount.

    Args:
        pv: amount invested today.
        rate: stated annual rate, decimal.
        years: years invested.
        freq: compounding periods per year (ignored if continuous).
        continuous: True for continuous compounding.

    Returns:
        fv, growth_factor, periodic_rate, periods.

    Example:
        fv_lump_sum(pv=5_000_000, rate=0.07, years=5) -> fv 7,012,758.65 (CFA L1 R5 Example 1)
    """
    pv, rate, m = _num("pv", pv), _num("rate", rate), _freq(freq)
    n = _n_periods(years, m)
    if continuous:
        g = math.exp(rate * float(years))
        return {"fv": pv * g, "growth_factor": g, "periodic_rate": None, "periods": None}
    if rate / m <= -1:
        raise ValueError("rate/freq must be greater than -100%")
    g = (1 + rate / m) ** n
    return {"fv": pv * g, "growth_factor": g, "periodic_rate": rate / m, "periods": n}


@formula("quant", formula="PV = FV (1 + r/m)^(-mN); continuous: PV = FV e^(-rN)", curriculum=_R5)
def pv_lump_sum(fv: float, rate: float, years: float, freq: int = 1, continuous: bool = False) -> dict:
    """Present value of a single future amount.

    Args:
        fv: amount received in `years`.
        rate: stated annual discount rate, decimal.
        years: years until receipt.
        freq: compounding periods per year (ignored if continuous).
        continuous: True for continuous compounding.

    Returns:
        pv, discount_factor, periodic_rate, periods.

    Example:
        pv_lump_sum(fv=100_000, rate=0.08, years=6) -> pv 63,016.96 (CFA L1 R5 Example 8)
    """
    out = fv_lump_sum(1.0, rate, years, freq, continuous)
    df = 1.0 / out["growth_factor"]
    return {"pv": _num("fv", fv) * df, "discount_factor": df, "periodic_rate": out["periodic_rate"],
            "periods": out["periods"]}


def _annuity_factors(i: float, n: float) -> tuple[float, float]:
    """(PV factor, FV factor) of an ordinary annuity of 1 per period."""
    if abs(i) < 1e-15:
        return n, n
    return (1 - (1 + i) ** -n) / i, ((1 + i) ** n - 1) / i


@formula("quant", formula="PV = A[1 - (1+r)^-N]/r; annuity due x (1+r)", curriculum=_R5)
def annuity_pv(payment: float, rate: float, years: float, freq: int = 1, due: bool = False) -> dict:
    """Present value of a level annuity (ordinary, or annuity due with first payment today).

    Args:
        payment: level payment per period.
        rate: stated annual rate, decimal; periodic rate = rate / freq.
        years: years of payments; number of payments = years * freq.
        freq: payments (and compounding periods) per year.
        due: True if payments are at the start of each period.

    Returns:
        pv, annuity_factor, periodic_rate, periods. For payment, periods or rate use tvm_solve.

    Example:
        annuity_pv(payment=1000, rate=0.12, years=5) -> pv 3,604.78 (CFA L1 R5 Example 11)
    """
    m = _freq(freq)
    i, n = _num("rate", rate) / m, _n_periods(years, m)
    f = _annuity_factors(i, n)[0] * ((1 + i) if due else 1)
    return {"pv": _num("payment", payment) * f, "annuity_factor": f, "periodic_rate": i, "periods": n}


@formula("quant", formula="FV = A[(1+r)^N - 1]/r; annuity due x (1+r)", curriculum=_R5)
def annuity_fv(payment: float, rate: float, years: float, freq: int = 1, due: bool = False) -> dict:
    """Future value of a level annuity at the end of the last period.

    Args:
        payment: level payment per period.
        rate: stated annual rate, decimal; periodic rate = rate / freq.
        years: years of payments; number of payments = years * freq.
        freq: payments per year.
        due: True if payments are at the start of each period.

    Returns:
        fv, annuity_factor, periodic_rate, periods.

    Example:
        annuity_fv(payment=20000, rate=0.09, years=30) -> fv 2,726,150.77 (CFA L1 R5 Example 7)
    """
    m = _freq(freq)
    i, n = _num("rate", rate) / m, _n_periods(years, m)
    f = _annuity_factors(i, n)[1] * ((1 + i) if due else 1)
    return {"fv": _num("payment", payment) * f, "annuity_factor": f, "periodic_rate": i, "periods": n}


@formula("quant", formula="PV(1+i)^N + PMT(1+i*due)[(1+i)^N - 1]/i + FV = 0 (calculator sign convention)",
         curriculum=_R5)
def tvm_solve(solve_for: str, periods: float | None = None, rate: float | None = None, pv: float = 0.0,
              pmt: float = 0.0, fv: float = 0.0, freq: int = 1, due: bool = False) -> dict:
    """Financial-calculator TVM solver: find pv, fv, pmt, periods or rate given the other four.

    Uses the calculator sign convention: money paid out is negative, money received positive
    (e.g. a loan received is pv=+100000, the payments come back negative).

    Args:
        solve_for: one of "pv", "fv", "pmt", "periods", "rate".
        periods: total number of periods N (not years).
        rate: stated annual rate, decimal; periodic rate i = rate / freq.
        pv, pmt, fv: present value, level payment per period, future value.
        freq: periods per year (calculator P/Y = C/Y).
        due: True for payments at the start of each period (BGN mode).

    Returns:
        the solved quantity first, then periods, rate, periodic_rate, pv, pmt, fv,
        and effective_annual_rate when solving for rate.

    Example:
        tvm_solve("pmt", periods=360, rate=0.08, pv=100000, freq=12) -> pmt -733.76 (CFA L1 R5 Example 20)
    """
    targets = ("pv", "fv", "pmt", "periods", "rate")
    if solve_for not in targets:
        raise ValueError(f"solve_for must be one of {targets}")
    m = _freq(freq)
    pv, pmt, fv = _num("pv", pv), _num("pmt", pmt), _num("fv", fv)
    if solve_for != "periods":
        if periods is None:
            raise ValueError("periods is required")
        n = _num("periods", periods)
        if n < 0:
            raise ValueError("periods must be >= 0")
    if solve_for != "rate":
        if rate is None:
            raise ValueError("rate is required")
        i = _num("rate", rate) / m
        if i <= -1:
            raise ValueError("rate/freq must be greater than -100%")
    b = 1 if due else 0

    def resid(i_: float, n_: float) -> float:
        g = (1 + i_) ** n_
        s = n_ if abs(i_) < 1e-15 else (g - 1) / i_
        return pv * g + pmt * (1 + i_ * b) * s + fv

    if solve_for == "fv":
        fv = -(resid(i, n) - fv)
    elif solve_for == "pv":
        g = (1 + i) ** n
        pv = -(resid(i, n) - pv * g) / g
    elif solve_for == "pmt":
        s = _annuity_factors(i, n)[1] * (1 + i * b)
        if s == 0:
            raise ValueError("cannot solve for pmt with zero periods")
        pmt = -(pv * (1 + i) ** n + fv) / s
    elif solve_for == "periods":
        if abs(i) < 1e-15:
            if pmt == 0:
                raise ValueError("no solution: zero rate and zero payment")
            n = -(pv + fv) / pmt
        else:
            a = pmt * (1 + i * b) / i
            if pv + a == 0 or (a - fv) / (pv + a) <= 0:
                raise ValueError("no solution for periods: check the signs of pv, pmt and fv")
            n = math.log((a - fv) / (pv + a)) / math.log(1 + i)
        if n < 0:
            raise ValueError(f"no non-negative solution for periods (got {n:.6g}); check signs")
    else:  # rate
        grid = np.concatenate([np.linspace(-0.99, 1.0, 400), np.geomspace(1.0, 100.0, 200)[1:]])
        vals = [resid(x, n) for x in grid]
        root = None
        for a_, b_, fa, fb in zip(grid[:-1], grid[1:], vals[:-1], vals[1:]):
            if fa == 0:
                root = a_
                break
            if fa * fb < 0:
                root = brentq(resid, a_, b_, args=(n,), xtol=1e-14, maxiter=500)
                break
        if root is None:
            raise ValueError("no rate solves the TVM equation; check the signs of pv, pmt and fv")
        i = root
    out = {"pv": pv, "fv": fv, "pmt": pmt, "periods": n, "rate": i * m}
    res = {solve_for: out[solve_for]}
    res.update({"periods": n, "rate": i * m, "periodic_rate": i, "pv": pv, "pmt": pmt, "fv": fv})
    if solve_for == "rate":
        res["effective_annual_rate"] = (1 + i) ** m - 1
    return res


@formula("quant", formula="PV_(t-1) = C_t/(r - g); PV_0 = PV_(t-1)/(1+r)^(t-1)", curriculum=_R5)
def perpetuity_pv(payment: float, rate: float, growth: float = 0.0, first_payment_period: int = 1) -> dict:
    """Present value of a level or constant-growth perpetuity, optionally starting in the future.

    Args:
        payment: first payment (at period `first_payment_period`).
        rate: discount rate per period, decimal.
        growth: constant growth rate of payments per period (0 for a level perpetuity).
        first_payment_period: period of the first payment (1 = one period from now).

    Returns:
        pv (at t = 0), pv_one_period_before_first_payment, valuation_period.

    Example:
        perpetuity_pv(payment=100, rate=0.05, first_payment_period=5) -> pv 1,645.40 (CFA L1 R5 Example 15)
    """
    c, r, g = _num("payment", payment), _num("rate", rate), _num("growth", growth)
    t = _whole(_num("first_payment_period", first_payment_period), "first_payment_period")
    if t < 1:
        raise ValueError("first_payment_period must be >= 1")
    if r <= g:
        raise ValueError("rate must exceed growth for a perpetuity to have a finite value")
    pv_prior = c / (r - g)
    return {"pv": pv_prior / (1 + r) ** (t - 1), "pv_one_period_before_first_payment": pv_prior,
            "valuation_period": t - 1}


@formula("quant", formula="PV = C1/(r-g)[1 - ((1+g)/(1+r))^N]; r = g: N C1/(1+r); FV = PV(1+r)^N",
         curriculum=_R5)
def growing_annuity_value(payment: float, rate: float, growth: float, periods: int, due: bool = False) -> dict:
    """Present and future value of an annuity whose payments grow at a constant rate.

    Args:
        payment: first payment (at t = 1, or t = 0 if due).
        rate: discount rate per period, decimal.
        growth: payment growth rate per period, decimal.
        periods: number of payments.
        due: True if the first payment is today.

    Returns:
        pv, fv (at the end of the last period), last_payment.

    Example:
        growing_annuity_value(payment=100, rate=0.1, growth=0.05, periods=3) -> pv 260.52
    """
    c, r, g = _num("payment", payment), _num("rate", rate), _num("growth", growth)
    n = _whole(_num("periods", periods), "periods")
    if r <= -1 or g <= -1:
        raise ValueError("rate and growth must be greater than -100%")
    if abs(r - g) < 1e-15:
        pv = n * c / (1 + r)
    else:
        pv = c / (r - g) * (1 - ((1 + g) / (1 + r)) ** n)
    if due:
        pv *= 1 + r
    return {"pv": pv, "fv": pv * (1 + r) ** n, "last_payment": c * (1 + g) ** (n - 1)}


@formula("quant", formula="PV = sum CF_t/(1+r/m)^t; FV_N = sum CF_t (1+r/m)^(N-t)", curriculum=_R5)
def uneven_cash_flows_value(rate: float, cash_flows: list, freq: int = 1) -> dict:
    """Present value (t = 0) and future value (at the last cash flow) of an uneven cash-flow series.

    Args:
        rate: stated annual rate, decimal; periodic rate = rate / freq.
        cash_flows: one per period, first at t = 0 (use 0 if there is none).
        freq: periods per year.

    Returns:
        pv, fv, horizon_periods, pv_each.

    Example:
        uneven_cash_flows_value(rate=0.05, cash_flows=[0, 1000, 2000, 4000, 5000, 6000])
        -> pv 15,036.46, fv 19,190.76 (CFA L1 R5 Tables 2-3)
    """
    cf = _flows(cash_flows, 1)
    i = _num("rate", rate) / _freq(freq)
    pv_each = cf / (1 + i) ** np.arange(cf.size)
    pv = float(pv_each.sum())
    n = cf.size - 1
    return {"pv": pv, "fv": pv * (1 + i) ** n, "horizon_periods": n, "pv_each": pv_each.tolist()}


# ---------------------------------------------------------------- NPV, IRR, MIRR

@formula("quant", formula="NPV = sum_{t=0..N} CF_t/(1+r)^t", curriculum=_R6)
def npv(rate: float, cash_flows: list) -> dict:
    """Net present value of a cash-flow series; accept if NPV > 0.

    Args:
        rate: discount rate per period (opportunity cost of capital), decimal.
        cash_flows: per period, first at t = 0 (usually the negative investment).

    Returns:
        npv, pv_future_flows (t >= 1), profitability_index (pv_future_flows / -CF0 when CF0 < 0).

    Example:
        npv(rate=0.11, cash_flows=[-1000, 294.8, 294.8, 294.8, 294.8, 294.8]) -> npv 89.55 (CFA L1 R6 Example 3)
    """
    cf = _flows(cash_flows)
    r = _num("rate", rate)
    total = _pv_flows(cf, r)
    pv_future = total - cf[0]
    return {"npv": total, "pv_future_flows": pv_future,
            "profitability_index": pv_future / -cf[0] if cf[0] < 0 else None}


def _irr_roots(cf: np.ndarray) -> list[float]:
    """All rates r > -1 where NPV = 0, found by a grid scan plus Brent's method on each bracket."""
    t = np.arange(cf.size)
    n = cf.size - 1

    def h(r: float) -> float:  # same sign and roots as NPV, without overflow for r near -1
        if r >= 0:
            return float(np.sum(cf * (1.0 / (1 + r)) ** t))  # underflows to 0, never overflows
        return float(np.sum(cf * (1 + r) ** (n - t)))

    grid = np.concatenate([np.linspace(-0.999, 1.0, 4000), np.geomspace(1.0, 1e4, 600)[1:]])
    vals = np.array([h(x) for x in grid])
    roots = []
    for k in range(grid.size - 1):
        if vals[k] == 0:
            roots.append(float(grid[k]))
        elif vals[k] * vals[k + 1] < 0:
            roots.append(brentq(h, grid[k], grid[k + 1], xtol=1e-15, rtol=1e-14, maxiter=500))
    return roots


@formula("quant", formula="sum_{t=0..N} CF_t/(1+IRR)^t = 0", curriculum=_R6)
def irr(cash_flows: list, freq: int = 1) -> dict:
    """Internal rate of return, robust to non-conventional cash flows.

    Scans for every sign change of NPV over r in (-99.9%, 1,000,000%) and refines each with Brent's
    method. With one root, `irr` is that root. With several (possible when the cash flows change sign
    more than once) or none, `irr` is None and `note` says why; rely on NPV in those cases.

    Args:
        cash_flows: per period, first at t = 0.
        freq: periods per year, used only to annualize: (1 + irr)^freq - 1.

    Returns:
        irr (periodic), irr_annualized, irrs (all roots found), sign_changes, note.

    Example:
        irr([-1000, 294.8, 294.8, 294.8, 294.8, 294.8]) -> irr 0.145012 (CFA L1 R6 Example 3)
    """
    cf = _flows(cash_flows)
    m = _freq(freq)
    nz = cf[cf != 0]
    changes = int(np.sum(np.sign(nz[1:]) != np.sign(nz[:-1])))
    roots = _irr_roots(cf) if changes else []
    if len(roots) == 1:
        r, note = roots[0], "unique IRR"
    elif not roots:
        r = None
        note = ("no IRR: cash flows never change sign" if changes == 0
                else "no IRR: NPV never crosses zero for r > -100%; use NPV")
    else:
        r, note = None, f"{len(roots)} IRRs (cash flows change sign {changes} times); IRR is ambiguous, use NPV"
    return {"irr": r, "irr_annualized": None if r is None else (1 + r) ** m - 1, "irrs": roots,
            "sign_changes": changes, "note": note}


@formula("quant", formula="MIRR = [FV(inflows @ reinvest) / PV(outflows @ finance)]^(1/N) - 1",
         curriculum="L1 Corporate Finance: Capital Budgeting (MIRR); L1 Quant R6 (IRR reinvestment assumption)")
def mirr(cash_flows: list, finance_rate: float, reinvest_rate: float) -> dict:
    """Modified IRR: outflows discounted at the finance rate, inflows compounded at the reinvestment rate.

    Args:
        cash_flows: per period, first at t = 0.
        finance_rate: rate for discounting negative flows to t = 0, decimal per period.
        reinvest_rate: rate for compounding positive flows to t = N, decimal per period.

    Returns:
        mirr, pv_outflows (positive number), fv_inflows, periods.

    Example:
        mirr([-120000, 39000, 30000, 21000, 37000, 46000], finance_rate=0.10, reinvest_rate=0.12)
        -> 0.126094 (Microsoft Excel MIRR documentation example)
    """
    cf = _flows(cash_flows)
    fr, rr = _num("finance_rate", finance_rate), _num("reinvest_rate", reinvest_rate)
    n = cf.size - 1
    t = np.arange(cf.size)
    pv_out = float(-np.sum(np.where(cf < 0, cf, 0) / (1 + fr) ** t))
    fv_in = float(np.sum(np.where(cf > 0, cf, 0) * (1 + rr) ** (n - t)))
    if pv_out <= 0 or fv_in <= 0:
        raise ValueError("MIRR needs at least one negative and one positive cash flow")
    return {"mirr": (fv_in / pv_out) ** (1 / n) - 1, "pv_outflows": pv_out, "fv_inflows": fv_in, "periods": n}


@formula("quant", formula="PMT = P i/[1 - (1+i)^-N]; interest_t = i * balance_(t-1)", curriculum=_R5)
def loan_amortization_schedule(principal: float, rate: float, years: float, freq: int = 12) -> dict:
    """Level-payment loan: payment and full amortization schedule.

    Args:
        principal: amount borrowed.
        rate: stated annual rate, decimal; periodic rate = rate / freq.
        years: term; number of payments = years * freq (must be whole).
        freq: payments per year.

    Returns:
        payment, total_interest, total_paid, periods, schedule (rows: period, beginning_balance, payment,
        interest, principal, ending_balance).

    Example:
        loan_amortization_schedule(principal=100000, rate=0.08, years=30) -> payment 733.76 (CFA L1 R5 Example 20)
    """
    p = _num("principal", principal)
    if p <= 0:
        raise ValueError("principal must be positive")
    m = _freq(freq)
    i = _num("rate", rate) / m
    n = _whole(_n_periods(years, m), "years * freq")
    if n < 1:
        raise ValueError("the loan needs at least one payment")
    pay = p / _annuity_factors(i, n)[0]
    bal, rows = p, []
    for k in range(1, n + 1):
        interest = bal * i
        princ = pay - interest
        rows.append({"period": k, "beginning_balance": bal, "payment": pay, "interest": interest,
                     "principal": princ, "ending_balance": bal - princ})
        bal -= princ
    total = pay * n
    return {"payment": pay, "total_interest": total - p, "total_paid": total, "periods": n, "schedule": rows}


@formula("quant", formula="EAR = (1 + r_s/m)^m - 1; continuous: EAR = e^r_s - 1", curriculum=_R5)
def effective_annual_rate(stated_rate: float, freq: int = 1, continuous: bool = False) -> dict:
    """Effective annual rate from a stated (nominal) annual rate.

    Args:
        stated_rate: stated annual rate, decimal.
        freq: compounding periods per year.
        continuous: True for continuous compounding.

    Returns:
        effective_annual_rate, periodic_rate.

    Example:
        effective_annual_rate(stated_rate=0.08, freq=2) -> 0.0816 (CFA L1 R5 Section 3.3)
    """
    r = _num("stated_rate", stated_rate)
    if continuous:
        return {"effective_annual_rate": math.expm1(r), "periodic_rate": None}
    m = _freq(freq)
    if r / m <= -1:
        raise ValueError("stated_rate/freq must be greater than -100%")
    return {"effective_annual_rate": (1 + r / m) ** m - 1, "periodic_rate": r / m}


@formula("quant", formula="r_s = m[(1+EAR)^(1/m) - 1]; continuous: r_s = ln(1+EAR)", curriculum=_R5)
def stated_annual_rate(effective_rate: float, freq: int = 1, continuous: bool = False) -> dict:
    """Stated annual rate (compounded `freq` times, or continuously) equivalent to an effective annual rate.

    Args:
        effective_rate: effective annual rate, decimal.
        freq: compounding periods per year.
        continuous: True to return the continuously compounded rate.

    Returns:
        stated_rate, periodic_rate.

    Example:
        stated_annual_rate(effective_rate=0.0816, freq=2) -> 0.08 (CFA L1 R5 Section 3.3)
    """
    e = _num("effective_rate", effective_rate)
    if e <= -1:
        raise ValueError("effective_rate must be greater than -100%")
    if continuous:
        return {"stated_rate": math.log1p(e), "periodic_rate": None}
    m = _freq(freq)
    p = (1 + e) ** (1 / m) - 1
    return {"stated_rate": p * m, "periodic_rate": p}


# ---------------------------------------------------------------- return measurement

@formula("quant", formula="HPR = (P1 - P0 + D1)/P0", curriculum=_R6)
def holding_period_return(begin_value: float, end_value: float, income: float = 0.0) -> dict:
    """Holding period return over one period.

    Args:
        begin_value: price or portfolio value at the start.
        end_value: price or value at the end.
        income: cash distributions received during the period.

    Returns:
        holding_period_return, price_return, income_return.

    Example:
        holding_period_return(begin_value=200, end_value=225, income=5) -> 0.15 (CFA L1 R6 Section 3.1)
    """
    p0, p1, d = _num("begin_value", begin_value), _num("end_value", end_value), _num("income", income)
    if p0 == 0:
        raise ValueError("begin_value must be non-zero")
    return {"holding_period_return": (p1 - p0 + d) / p0, "price_return": (p1 - p0) / p0, "income_return": d / p0}


@formula("quant", formula="MWR = IRR of portfolio flows: sum CF_t/(1+MWR)^t = 0", curriculum=_R6)
def money_weighted_return(cash_flows: list, freq: int = 1) -> dict:
    """Money-weighted rate of return: the IRR of the investor's portfolio cash flows.

    Contributions (including the initial value) are negative; withdrawals, income taken out and the
    ending value are positive. Net all flows that fall on the same date.

    Args:
        cash_flows: net flow per period, first at t = 0.
        freq: periods per year, to annualize: (1 + mwr)^freq - 1.

    Returns:
        money_weighted_return (periodic), annualized, irrs, note.

    Example:
        money_weighted_return([-200, -220, 480]) -> 0.0939 (CFA L1 R6 Section 3.1)
    """
    out = irr(cash_flows, freq)
    if out["irr"] is None:
        raise ValueError(f"money-weighted return undefined: {out['note']}")
    return {"money_weighted_return": out["irr"], "annualized": out["irr_annualized"], "irrs": out["irrs"],
            "note": out["note"]}


@formula("quant", formula="TWR = prod(1 + r_i) - 1; annualized = (1 + TWR)^(1/years) - 1", curriculum=_R6)
def time_weighted_return(subperiod_returns: list | None = None, begin_values: list | None = None,
                         end_values: list | None = None, years: float | None = None) -> dict:
    """Time-weighted rate of return: link subperiod holding-period returns.

    Give either `subperiod_returns`, or `begin_values` (value just after each external flow) with
    `end_values` (value just before the next flow, including income).

    Args:
        subperiod_returns: HPR of each subperiod, decimals.
        begin_values: value at the start of each subperiod, after contributions/withdrawals.
        end_values: value at the end of each subperiod, before the next flow.
        years: length of the whole measurement period, to annualize.

    Returns:
        time_weighted_return (whole period), annualized (if years given), subperiod_returns.

    Example:
        time_weighted_return(subperiod_returns=[0.20, 0.05, 0.12, -0.10]) -> 0.27 (CFA L1 R6 Example 4)
    """
    if subperiod_returns is None:
        if begin_values is None or end_values is None or len(begin_values) != len(end_values):
            raise ValueError("give subperiod_returns, or begin_values and end_values of equal length")
        b = np.array([_num("begin_values", v) for v in begin_values])
        e = np.array([_num("end_values", v) for v in end_values])
        if np.any(b == 0):
            raise ValueError("begin_values must be non-zero")
        r = e / b - 1
    else:
        r = np.array([_num("subperiod_returns", v) for v in subperiod_returns])
    if r.size == 0 or np.any(r <= -1):
        raise ValueError("need at least one subperiod return, each greater than -100%")
    twr = float(np.prod(1 + r) - 1)
    ann = None
    if years is not None:
        y = _num("years", years)
        if y <= 0:
            raise ValueError("years must be positive")
        ann = (1 + twr) ** (1 / y) - 1
    return {"time_weighted_return": twr, "annualized": ann, "subperiod_returns": r.tolist()}


@formula("quant", formula="r_annual = (1 + r_period)^(periods per year) - 1", curriculum=_R6)
def annualize_return(period_return: float, periods_per_year: float | None = None, days: float | None = None,
                     days_in_year: float = 365) -> dict:
    """Annualize a holding-period return by compounding (and show the simple, non-compounded version).

    Args:
        period_return: return over one period, decimal.
        periods_per_year: number of such periods in a year (e.g. 12 for a monthly return), or
        days: length of the period in days (periods_per_year = days_in_year / days).
        days_in_year: 365 by default (use 360 for money-market conventions).

    Returns:
        annualized (compound), annualized_simple, periods_per_year.

    Example:
        annualize_return(period_return=0.0628, periods_per_year=3) -> 0.2005 (CFA L1 R6 Example 5)
    """
    r = _num("period_return", period_return)
    if r <= -1:
        raise ValueError("period_return must be greater than -100%")
    if (periods_per_year is None) == (days is None):
        raise ValueError("give exactly one of periods_per_year or days")
    m = _num("periods_per_year", periods_per_year) if days is None else _num("days_in_year", days_in_year) / _num("days", days)
    if m <= 0:
        raise ValueError("periods per year must be positive")
    return {"annualized": (1 + r) ** m - 1, "annualized_simple": r * m, "periods_per_year": m}


@formula("quant", formula="r_BD = (D/F)(360/t); HPY = D/P0; EAY = (1+HPY)^(365/t) - 1; r_MM = HPY(360/t)",
         curriculum=_R6)
def money_market_yields(days: float, face: float = 100.0, price: float | None = None,
                        bank_discount_yield: float | None = None, holding_period_yield: float | None = None,
                        effective_annual_yield: float | None = None, money_market_yield: float | None = None) -> dict:
    """All money-market yield measures of a pure-discount instrument from its price or any one yield.

    Give exactly one of price, bank_discount_yield, holding_period_yield, effective_annual_yield or
    money_market_yield; everything else is derived through the holding period yield.

    Args:
        days: days to maturity t.
        face: face (par) value.
        price, bank_discount_yield, holding_period_yield, effective_annual_yield, money_market_yield:
            the one known quantity (yields as decimals).

    Returns:
        bank_discount_yield, holding_period_yield, effective_annual_yield, money_market_yield,
        bey_semiannual (2 x semiannual effective rate, the bond-equivalent convention),
        add_on_yield_365 (HPY x 365/t), price, discount.

    Example:
        money_market_yields(days=150, face=100000, price=98000) -> BDY 0.048, HPY 0.020408,
        EAY 0.050388, MMY 0.04898 (CFA L1 R6 Examples 6-7)
    """
    t, f = _num("days", days), _num("face", face)
    if t <= 0 or f <= 0:
        raise ValueError("days and face must be positive")
    given = {"price": price, "bank_discount_yield": bank_discount_yield,
             "holding_period_yield": holding_period_yield, "effective_annual_yield": effective_annual_yield,
             "money_market_yield": money_market_yield}
    known = {k: _num(k, v) for k, v in given.items() if v is not None}
    if len(known) != 1:
        raise ValueError("give exactly one of " + ", ".join(given))
    (k, v), = known.items()
    if k == "price":
        p = v
    elif k == "bank_discount_yield":
        p = f - v * f * t / 360
    else:
        hpy = {"holding_period_yield": lambda: v, "effective_annual_yield": lambda: (1 + v) ** (t / 365) - 1,
               "money_market_yield": lambda: v * t / 360}[k]()
        p = f / (1 + hpy)
    if p <= 0:
        raise ValueError("implied price is not positive")
    hpy = (f - p) / p
    eay = (1 + hpy) ** (365 / t) - 1
    return {"bank_discount_yield": (f - p) / f * 360 / t, "holding_period_yield": hpy,
            "effective_annual_yield": eay, "money_market_yield": hpy * 360 / t,
            "bey_semiannual": 2 * ((1 + eay) ** 0.5 - 1), "add_on_yield_365": hpy * 365 / t,
            "price": p, "discount": f - p}
