"""Fixed income: term structure (spot, par, forward curves), spreads, rolldown, yield volatility (L1-L2)."""
from __future__ import annotations

import math
import statistics

from ._registry import formula
from .fi_pricing import L1_VAL, _level_cfs, _periods, _pv, _rate_ok, _solve

L2_TS = "L2 FI: Term Structure and Volatility of Interest Rates (2014 R46) / Arbitrage-Free Valuation"


def _dfs_to_spots(dfs: list, freq: float) -> list:
    return [freq * (df ** (-1 / k) - 1) for k, df in enumerate(dfs, start=1)]


def _spots_to_dfs(spots: list, freq: float) -> list:
    out = []
    for k, z in enumerate(spots, start=1):
        _rate_ok(z / freq)
        out.append((1 + z / freq) ** -k)
    return out


@formula("fixed_income", formula="DF_k = (P_k/F - c_k sum_{j<k} DF_j)/(1 + c_k); z_k = m(DF_k^(-1/k) - 1)",
         curriculum=L2_TS)
def fi_spot_bootstrap(coupon_rates: list, prices: list | None = None, freq: int = 1, face: float = 100.0) -> dict:
    """Bootstrap spot rates and discount factors from par yields or coupon-bond prices of consecutive maturities.

    Args:
        coupon_rates: annual coupon rate of the bond maturing at period 1, 2, ..., N (par yields if prices omitted).
        prices: price of each bond per `face` (default: all at par, so coupon_rates are par rates).
        freq: periods per year. face: par value.

    Returns:
        spot_rates (annual, periodicity freq), discount_factors, forward_rates (one-period forwards).

    Example:
        par 3.5%, 4.2%, 4.7%, 5.2% (annual) -> spots 3.5000%, 4.2148%, 4.7352%, 5.2706%
    """
    prices = [face] * len(coupon_rates) if prices is None else list(prices)
    if len(prices) != len(coupon_rates) or not prices:
        raise ValueError("coupon_rates and prices must be non-empty and the same length")
    dfs: list = []
    for c, p in zip(coupon_rates, prices):
        ck = c / freq
        df = (p / face - ck * sum(dfs)) / (1 + ck)
        if df <= 0:
            raise ValueError("inputs imply a non-positive discount factor")
        dfs.append(df)
    fwds = [freq * ((dfs[k - 1] if k else 1.0) / dfs[k] - 1) for k in range(len(dfs))]
    return {"spot_rates": _dfs_to_spots(dfs, freq), "discount_factors": dfs, "forward_rates": fwds}


@formula("fixed_income", formula="par_N = m (1 - DF_N) / sum_{k<=N} DF_k (also the swap rate on a swap spot curve)",
         curriculum=L1_VAL)
def fi_par_curve(spot_rates: list, freq: int = 1) -> dict:
    """Par yields (and swap rates, when the spot curve is the swap/Libor curve) implied by spot rates.

    Args:
        spot_rates: spot rates for periods 1..N (annual, periodicity freq). freq: periods per year.

    Returns:
        par_rates, discount_factors, annuity_factors (sum of DFs to each maturity).

    Example:
        spots 5.263%, 5.616%, 6.359%, 7.008% -> par 5.263%, 5.606%, 6.306%, 6.899%
    """
    dfs = _spots_to_dfs(spot_rates, freq)
    ann, pars, s = [], [], 0.0
    for df in dfs:
        s += df
        ann.append(s)
        pars.append(freq * (1 - df) / s)
    return {"par_rates": pars, "discount_factors": dfs, "annuity_factors": ann}


@formula("fixed_income", formula="(1+z_A/m)^(mA) (1+f/m)^(m(B-A)) = (1+z_B/m)^(mB)", curriculum=L1_VAL)
def fi_forward_rate(spot_short: float, years_short: float, spot_long: float, years_long: float,
                    freq: int = 2) -> dict:
    """Implied forward rate between two maturities ("AyBy" notation: A years forward, B-A year tenor).

    Args:
        spot_short: spot rate to `years_short` (annual, periodicity freq). years_short: A (0 allowed).
        spot_long: spot rate to `years_long`. years_long: B > A. freq: periodicity of all rates.

    Returns:
        forward_rate (annual, periodicity freq), forward_per_period, tenor_years.

    Example:
        3y 3.65%, 4y 4.18% semiannual -> 3y1y forward 5.778%
    """
    if years_long <= years_short or years_short < 0:
        raise ValueError("need 0 <= years_short < years_long")
    a, b = years_short * freq, years_long * freq
    g = (1 + spot_long / freq) ** b / (1 + spot_short / freq) ** a
    f = g ** (1 / (b - a)) - 1
    return {"forward_rate": f * freq, "forward_per_period": f, "tenor_years": years_long - years_short}


@formula("fixed_income", formula="f_k = m[(1+z_k/m)^k/(1+z_{k-1}/m)^(k-1) - 1]", curriculum=L1_VAL)
def fi_forward_curve(spot_rates: list, freq: int = 1) -> dict:
    """One-period forward curve implied by a spot curve (first element = the one-period spot rate).

    Args:
        spot_rates: spot rates for periods 1..N. freq: periods per year.

    Returns:
        forward_rates (annual, periodicity freq), discount_factors.

    Example:
        spots 3.5%, 4.2148%, 4.7352%, 5.2706% -> forwards 3.500%, 4.935%, 5.784%, 6.893%
    """
    dfs = _spots_to_dfs(spot_rates, freq)
    fwds = [freq * ((dfs[k - 1] if k else 1.0) / dfs[k] - 1) for k in range(len(dfs))]
    return {"forward_rates": fwds, "discount_factors": dfs}


@formula("fixed_income", formula="(1+z_N/m)^N = prod_{k=1..N}(1+f_k/m) (spot = geometric mean of forwards)",
         curriculum=L1_VAL)
def fi_spot_from_forwards(forward_rates: list, freq: int = 1) -> dict:
    """Spot rates and discount factors from a sequence of one-period forward rates (0y1y, 1y1y, ...).

    Args:
        forward_rates: one-period forward rates, first = current one-period spot. freq: periods per year.

    Returns:
        spot_rates, discount_factors.

    Example:
        forwards 1.88%, 2.77%, 3.54%, 4.12% -> spots 1.88%, 2.3240%, 2.7278%, 3.0741%
    """
    dfs, g = [], 1.0
    for f in forward_rates:
        _rate_ok(f / freq)
        g *= 1 + f / freq
        dfs.append(1 / g)
    return {"spot_rates": _dfs_to_spots(dfs, freq), "discount_factors": dfs}


@formula("fixed_income", formula="G-spread = YTM - govt yield; I-spread = YTM - swap rate (interpolated)",
         curriculum=L1_VAL)
def bond_g_i_spread(bond_ytm: float, benchmark_yield: float | None = None, bond_years: float | None = None,
                    benchmark_years: list | None = None, benchmark_yields: list | None = None) -> dict:
    """Yield spread over a benchmark: G-spread (government curve) or I-spread (swap curve), interpolating if needed.

    Args:
        bond_ytm: the bond's yield to maturity.
        benchmark_yield: benchmark yield of matching maturity, or give the curve below instead.
        bond_years: bond maturity (for interpolation). benchmark_years / benchmark_yields: benchmark curve
            points (government yields for a G-spread, swap rates for an I-spread). Linear interpolation.

    Returns:
        spread, spread_bp, benchmark_yield.

    Example:
        4y corporate 2.36%, govt 3y 0.75% / 5y 1.45% -> interpolated 1.10%, spread 126 bp
    """
    if benchmark_yield is None:
        if bond_years is None or not benchmark_years or benchmark_yields is None:
            raise ValueError("give benchmark_yield or bond_years with benchmark_years/benchmark_yields")
        pts = sorted(zip(benchmark_years, benchmark_yields))
        xs = [p[0] for p in pts]
        if not xs[0] <= bond_years <= xs[-1]:
            raise ValueError("bond_years is outside the benchmark curve; no extrapolation")
        for (x0, y0), (x1, y1) in zip(pts, pts[1:] + pts[-1:]):
            if x0 <= bond_years <= x1:
                benchmark_yield = y0 if x1 == x0 else y0 + (bond_years - x0) / (x1 - x0) * (y1 - y0)
                break
    s = bond_ytm - benchmark_yield
    return {"spread": s, "spread_bp": s * 1e4, "benchmark_yield": benchmark_yield}


@formula("fixed_income", formula="solve P = sum CF_t/(1 + (z_t + Z)/m)^t for Z", curriculum=L1_VAL)
def bond_z_spread(price: float, spot_rates: list, coupon_rate: float = 0.0, freq: int = 1, face: float = 100.0,
                  cash_flows: list | None = None) -> dict:
    """Zero-volatility (static) spread: constant spread over every benchmark spot rate that reprices the bond.

    Args:
        price: full price per `face`. spot_rates: benchmark spot rates for periods 1..N (annual, periodicity freq).
        coupon_rate: annual coupon (ignored if cash_flows given). freq: periods per year. face: par.
        cash_flows: optional explicit cash flows for periods 1..N.

    Returns:
        z_spread, z_spread_bp.

    Example:
        2y 6% annual at 100.125, spots 2.10% / 3.635% -> Z = 234.22 bp
    """
    cfs = list(cash_flows) if cash_flows is not None else _level_cfs(face, coupon_rate, len(spot_rates), freq)
    if len(cfs) != len(spot_rates):
        raise ValueError("cash_flows and spot_rates must have the same length")

    def f(z: float) -> float:
        return sum(cf / (1 + (s + z) / freq) ** k for k, (cf, s) in enumerate(zip(cfs, spot_rates), 1)) - price

    lo = -freq * 0.99 - min(spot_rates)
    z = _solve(f, lo=lo, hi=5.0, what="Z-spread")
    return {"z_spread": z, "z_spread_bp": z * 1e4}


@formula("fixed_income", formula="Swap spread = swap rate - govt yield; TED = Libor - T-bill; Libor-OIS = Libor - OIS",
         curriculum=L2_TS)
def swap_ted_ois_spreads(swap_rate: float | None = None, treasury_yield: float | None = None,
                         libor: float | None = None, tbill_yield: float | None = None,
                         ois_rate: float | None = None) -> dict:
    """Money-market and swap spread indicators of bank credit and funding stress (decimals and bp).

    Args:
        swap_rate / treasury_yield: same-tenor swap rate and government yield (swap spread).
        libor / tbill_yield: same-tenor Libor and T-bill yield (TED spread).
        libor / ois_rate: same-tenor Libor and overnight indexed swap rate (Libor-OIS spread).

    Returns:
        swap_spread, ted_spread, libor_ois_spread and *_bp versions (None when inputs missing).

    Example:
        7y swap 6.0% vs Treasury 5.4% -> swap spread 60 bp
    """
    def d(a, b):
        return None if a is None or b is None else a - b

    out = {"swap_spread": d(swap_rate, treasury_yield), "ted_spread": d(libor, tbill_yield),
           "libor_ois_spread": d(libor, ois_rate)}
    if all(v is None for v in out.values()):
        raise ValueError("give at least one complete pair of rates")
    out.update({k + "_bp": None if v is None else v * 1e4 for k, v in list(out.items())})
    return out


@formula("fixed_income", formula="Rolldown = (P(N-h, y_rolled) - P(N, y0))/P0; HPR = (P1 + coupons - P0)/P0",
         curriculum="L2/L3 FI: riding the yield curve; rolling yield = yield income + rolldown return")
def bond_rolldown_return(coupon_rate: float, years: float, ytm_begin: float, horizon_years: float,
                         ytm_end: float, freq: int = 2, face: float = 100.0) -> dict:
    """Riding the yield curve: holding-period return when the bond rolls down an unchanged upward-sloping curve.

    Args:
        coupon_rate: annual coupon. years: maturity today. ytm_begin: yield today.
        horizon_years: holding period. ytm_end: yield at the horizon for the shorter remaining maturity
            (read from today's curve if the curve is unchanged). freq: coupons per year. face: par.

    Returns:
        price_begin, price_end, coupon_income, rolldown_return (price change / P0), yield_income
        (coupons / P0), rolling_yield (yield income + rolldown), holding_period_return.

    Example:
        5y 3% at 3.0% held 1y, 4y yield 2.5% -> rolldown ~ +1.88%
    """
    n, h = _periods(years, freq), _periods(horizon_years, freq)
    if h >= n:
        raise ValueError("horizon must be shorter than maturity")
    p0 = _pv(_level_cfs(face, coupon_rate, n, freq), ytm_begin / freq)
    p1 = _pv(_level_cfs(face, coupon_rate, n - h, freq), ytm_end / freq)
    cpn = face * coupon_rate / freq * h
    roll = (p1 - p0) / p0
    return {"price_begin": p0, "price_end": p1, "coupon_income": cpn, "rolldown_return": roll,
            "yield_income": cpn / p0, "rolling_yield": cpn / p0 + roll, "holding_period_return": (p1 + cpn - p0) / p0}


@formula("fixed_income", formula="X_t = ln(y_t/y_{t-1}); daily sd = stdev(X); annual = sd x sqrt(days)",
         curriculum=L2_TS)
def yield_volatility_historical(yields: list, days_in_year: float = 250) -> dict:
    """Historical yield volatility from daily yields (log relative changes, sample standard deviation).

    Args:
        yields: consecutive daily yields (any consistent unit). days_in_year: 250, 260 or 365 to annualize.

    Returns:
        daily_std (decimal), annual_std (decimal), mean_change, observations.

    Example:
        26 daily Treasury-zero yields (L2 R46 Exhibit 8) -> daily 0.6360%, annual (250 d) 10.06%
    """
    if len(yields) < 3 or any(y <= 0 for y in yields):
        raise ValueError("need at least 3 positive yields")
    x = [math.log(b / a) for a, b in zip(yields, yields[1:])]
    sd = statistics.stdev(x)
    return {"daily_std": sd, "annual_std": sd * math.sqrt(days_in_year), "mean_change": statistics.fmean(x),
            "observations": len(x)}
