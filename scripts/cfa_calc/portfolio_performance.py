"""Performance measurement, attribution and appraisal; GIPS return calculations (CFA L1-L3). Topic ``portfolio``.

Valuation convention for time-weighted returns: `market_values[i]` is the value on date i just AFTER that date's
external cash flow `cash_flows[i]` (contributions +, withdrawals -), as in the curriculum's examples.
"""
from __future__ import annotations

import math

import numpy as np
from scipy.optimize import brentq

from ._registry import formula
from .portfolio import _f, _vec

EPP = "L3 PM: Evaluating Portfolio Performance"
GIPS = "L3 PM: Global Investment Performance Standards"
GPE = "L3 PM (2013): Global Performance Evaluation"


def _opt_vec(x, name: str) -> np.ndarray:
    """Optional list of flows/dates/weights: None or [] both mean none."""
    return np.array([]) if x is None or len(x) == 0 else _vec(x, name)


# ---------------------------------------------------------------- return measurement
@formula("portfolio", formula="r_i = (MV_i - CF_i - MV_(i-1))/MV_(i-1); TWR = prod(1 + r_i) - 1", curriculum=EPP + "; " + GIPS)
def gips_time_weighted_return(market_values: list | None = None, cash_flows: list | None = None,
                              subperiod_returns: list | None = None, years: float | None = None) -> dict:
    """True time-weighted return by chain-linking subperiod returns (valuations at each external cash flow).

    Also links any series of period returns, e.g. monthly returns or subperiod money-weighted returns (LIRR).

    Args:
        market_values: values at the start, at each cash-flow date (just AFTER the flow) and at the end.
        cash_flows: external flow on each of those dates (same length; first and last usually 0).
        subperiod_returns: alternatively, the subperiod returns to link.
        years: optional length of the whole period, to annualise (only if >= 1 under GIPS).

    Returns:
        twr, subperiod_returns, annualized (if years given).

    Example:
        gips_time_weighted_return(market_values=[1000000, 1045000, 1060000, 1080000], cash_flows=[0, 30000, 20000, 0])
        -> subperiods 0.0150, -0.0048, 0.0189; twr 0.0292
    """
    if subperiod_returns is None:
        if market_values is None:
            raise ValueError("give market_values (+ cash_flows) or subperiod_returns")
        mv = _vec(market_values, "market_values")
        cf = np.zeros(mv.size) if cash_flows is None else _vec(cash_flows, "cash_flows")
        if cf.size != mv.size or mv.size < 2:
            raise ValueError("market_values (>= 2) and cash_flows must have the same length")
        if np.any(mv[:-1] <= 0):
            raise ValueError("beginning values must be > 0")
        r = (mv[1:] - cf[1:] - mv[:-1]) / mv[:-1]
    else:
        r = _vec(subperiod_returns, "subperiod_returns")
    twr = float(np.prod(1 + r) - 1)
    out = {"twr": twr, "subperiod_returns": _f(r)}
    if years is not None:
        out["annualized"] = (1 + twr) ** (1 / years) - 1
    return out


@formula("portfolio", formula="MV1 = MV0(1+R)^m + sum CF_i (1+R)^(m - t_i); MWR = (1+R)^m - 1", curriculum=EPP + "; " + GIPS + " (IRR)")
def gips_money_weighted_return(begin_value: float, end_value: float, period_length: float, cash_flows: list | None = None,
                               flow_times: list | None = None, periods_per_year: float | None = None) -> dict:
    """Money-weighted return (internal rate of return) of a portfolio with external cash flows.

    Args:
        begin_value: value at t = 0.
        end_value: value at t = period_length.
        period_length: length of the period in time units (days, quarters, years...).
        cash_flows: external flows into the portfolio (contributions +, withdrawals/distributions -); None or [] = none.
        flow_times: time of each flow in the same units, measured from the start.
        periods_per_year: time units per year, to annualise (e.g. 4 for quarters, 365 for days).

    Returns:
        rate_per_unit (R), period_return ((1+R)^m - 1), annualized (if periods_per_year given).

    Example:
        gips_money_weighted_return(begin_value=1000000, end_value=1080000, period_length=30,
                                   cash_flows=[30000, 20000], flow_times=[5, 16]) -> R 0.0009536/day, MWR 0.0290
    """
    cf, t = _opt_vec(cash_flows, "cash_flows"), _opt_vec(flow_times, "flow_times")
    if cf.size != t.size:
        raise ValueError("cash_flows and flow_times must have the same length")
    if np.any((t < 0) | (t > period_length)) or period_length <= 0:
        raise ValueError("flow_times must lie within [0, period_length] and period_length must be > 0")
    m = period_length

    def f(R):
        return begin_value * (1 + R) ** m + float(cf @ (1 + R) ** (m - t)) - end_value

    lo, hi = -0.999999, 1.0
    while f(hi) < 0 and hi < 1e6:
        hi *= 2
    if f(lo) * f(hi) > 0:
        raise ValueError("no money-weighted return found; check the signs of the values and flows")
    R = brentq(f, lo, hi, xtol=1e-15, rtol=1e-15, maxiter=500)
    out = {"rate_per_unit": R, "period_return": (1 + R) ** m - 1}
    if periods_per_year is not None:
        out["annualized"] = (1 + R) ** periods_per_year - 1
    return out


@formula("portfolio", formula="r = (V1 - V0 - CF)/(V0 + sum CF_i w_i); w_i = (CD - D_i)/CD (end-of-day flows), "
                              "(CD - D_i + 1)/CD (start-of-day), 0.5 (original Dietz)", curriculum=GIPS)
def gips_modified_dietz(begin_value: float, end_value: float, cash_flows: list | None = None,
                        flow_days: list | None = None, period_days: float | None = None,
                        flow_timing: str = "end_of_day", weights: list | None = None) -> dict:
    """Modified (day-weighted) or original (mid-point) Dietz return approximating the time-weighted return.

    Args:
        begin_value, end_value: V0 and V1.
        cash_flows: external flows (+ in, - out); None or [] = no external flows.
        flow_days: day of each flow counted from the start of the period (D_i).
        period_days: calendar days in the period (CD).
        flow_timing: "end_of_day" (GIPS default), "start_of_day" or "midpoint" (original Dietz, w = 0.5).
        weights: explicit weights w_i instead of days.

    Returns:
        return_, weights, weighted_flows, average_capital (denominator), gain.

    Example:
        gips_modified_dietz(begin_value=100000, end_value=110550, cash_flows=[10000], flow_days=[5], period_days=30)
        -> return_ 0.0051
    """
    cf = _opt_vec(cash_flows, "cash_flows")
    if weights is not None:
        w = _opt_vec(weights, "weights")
    elif flow_timing == "midpoint":
        w = np.full(cf.size, 0.5)
    else:
        if cf.size and (flow_days is None or period_days is None):
            raise ValueError("give flow_days and period_days (or weights, or flow_timing='midpoint')")
        d = _opt_vec(flow_days, "flow_days")
        if flow_timing not in ("end_of_day", "start_of_day"):
            raise ValueError("flow_timing must be end_of_day, start_of_day or midpoint")
        w = (period_days - d + (1 if flow_timing == "start_of_day" else 0)) / period_days if cf.size else np.array([])
    if w.size != cf.size:
        raise ValueError("one weight (or day) per cash flow")
    gain = end_value - begin_value - float(cf.sum())
    denom = begin_value + float(cf @ w)
    if denom <= 0:
        raise ValueError("average invested capital must be > 0")
    return {"return_": gain / denom, "weights": _f(w), "weighted_flows": float(cf @ w), "average_capital": denom,
            "gain": gain}


@formula("portfolio", formula="R_composite = sum (V0_i or V0_i + weighted CF_i) x r_i / sum(...)", curriculum=GIPS + " (composite returns)")
def gips_composite_return(portfolio_returns: list, beginning_values: list, weighted_flows: list | None = None) -> dict:
    """Asset-weighted composite return from portfolio returns, weighted by beginning values (or beginning values
    plus day-weighted flows).

    Returns:
        composite_return, weights.

    Example:
        Exhibit 10: returns [0.0051, 0.0028, 0.0032, 0.0136], beginning [100, 97.40, 112.94, 124.47] -> 0.0065
    """
    r = _vec(portfolio_returns, "portfolio_returns")
    b = _vec(beginning_values, "beginning_values")
    if weighted_flows is not None:
        b = b + _vec(weighted_flows, "weighted_flows")
    if r.size != b.size:
        raise ValueError("portfolio_returns and beginning_values must have the same length")
    w = b / b.sum()
    return {"composite_return": float(w @ r), "weights": _f(w)}


@formula("portfolio", formula="annualized = [prod(1 + r_t)]^(periods_per_year/n) - 1, or (1 + R)^(1/years) - 1",
         curriculum=EPP + "; " + GIPS + " (no annualising periods < 1 year)")
def gips_annualized_return(period_returns: list | None = None, periods_per_year: float = 1.0,
                           cumulative_return: float | None = None, years: float | None = None) -> dict:
    """Geometric (compound) annualised return from period returns or a cumulative return.

    Returns:
        annualized, cumulative_return, years, gips_may_annualize (False for periods under one year).

    Example:
        gips_annualized_return(period_returns=[0.02, 0.095, -0.047]) -> annualized 0.021
    """
    if period_returns is not None:
        r = _vec(period_returns, "period_returns")
        cum = float(np.prod(1 + r) - 1)
        yrs = r.size / periods_per_year
    elif cumulative_return is not None and years is not None:
        cum, yrs = cumulative_return, years
    else:
        raise ValueError("give period_returns, or cumulative_return and years")
    if yrs <= 0 or cum <= -1:
        raise ValueError("years must be > 0 and cumulative return > -100%")
    return {"annualized": (1 + cum) ** (1 / yrs) - 1, "cumulative_return": cum, "years": yrs,
            "gips_may_annualize": yrs >= 1}


@formula("portfolio", formula="equal-weighted sd = sqrt(sum(r - rbar)^2/(n-1)); asset-weighted sd = sqrt(sum w_i (r_i - r_aw)^2); "
                              "quartiles at L = (n+1) y/100 in the descending array", curriculum=GIPS + " (internal dispersion)")
def gips_composite_dispersion(returns: list, beginning_values: list | None = None) -> dict:
    """Internal dispersion of annual returns of portfolios in a composite (GIPS measures).

    Args:
        returns: annual returns of portfolios held the full year.
        beginning_values: beginning-of-year values, for asset-weighted measures.

    Returns:
        equal_weighted_std (n-1), equal_weighted_std_population (n), mean, high, low, range_high_low,
        upper_quartile (25th percentile from the top), median, lower_quartile, interquartile_range,
        asset_weighted_mean and asset_weighted_std (if beginning_values).

    Example:
        Exhibit 13 (14 portfolios) -> equal-weighted sd 0.23%, asset-weighted sd 0.21%, range 0.73%, IQR ~0.36%
    """
    r = _vec(returns, "returns")
    n = r.size
    if n < 2:
        raise ValueError("need at least two portfolio returns")
    desc = np.sort(r)[::-1]

    def pct(y):
        L = (n + 1) * y / 100
        k = int(math.floor(L))
        if k < 1:
            return float(desc[0])
        if k >= n:
            return float(desc[-1])
        return float(desc[k - 1] - (L - k) * (desc[k - 1] - desc[k]))

    q1, med, q3 = pct(25), pct(50), pct(75)
    out = {"equal_weighted_std": float(np.std(r, ddof=1)), "equal_weighted_std_population": float(np.std(r)),
           "mean": float(r.mean()), "high": float(desc[0]), "low": float(desc[-1]),
           "range_high_low": float(desc[0] - desc[-1]), "upper_quartile": q1, "median": med, "lower_quartile": q3,
           "interquartile_range": q1 - q3, "n": n}
    if beginning_values is not None:
        b = _vec(beginning_values, "beginning_values")
        if b.size != n:
            raise ValueError("beginning_values must match returns")
        w = b / b.sum()
        m = float(w @ r)
        out.update(asset_weighted_mean=m, asset_weighted_std=math.sqrt(float(w @ (r - m) ** 2)))
    return out


@formula("portfolio", formula="net = gross - fee per period (arithmetic) or (1 + gross)/(1 + fee)^(1/ppy) - 1 (geometric)",
         curriculum=GIPS + " (gross-of-fees vs net-of-fees)")
def gips_net_of_fees_returns(gross_returns: list, annual_fee_rate: float, periods_per_year: float = 1.0,
                             method: str = "arithmetic") -> dict:
    """Net-of-fees returns from gross-of-fees returns and an annual investment management fee.

    Args:
        gross_returns: gross-of-fees period returns (already net of trading expenses).
        annual_fee_rate: management fee as a fraction of assets per year.
        periods_per_year: frequency of the returns (12 monthly).
        method: "arithmetic" (fee/ppy deducted each period) or "geometric" (compounded fee).

    Returns:
        net_returns, cumulative_gross, cumulative_net, fee_drag.

    Example:
        gips_net_of_fees_returns(gross_returns=[0.10], annual_fee_rate=0.01) -> net_returns [0.09]
    """
    g = _vec(gross_returns, "gross_returns")
    if method == "arithmetic":
        net = g - annual_fee_rate / periods_per_year
    elif method == "geometric":
        net = (1 + g) / (1 + annual_fee_rate) ** (1 / periods_per_year) - 1
    else:
        raise ValueError("method must be 'arithmetic' or 'geometric'")
    cg, cn = float(np.prod(1 + g) - 1), float(np.prod(1 + net) - 1)
    return {"net_returns": _f(net), "cumulative_gross": cg, "cumulative_net": cn, "fee_drag": cg - cn}


# ---------------------------------------------------------------- benchmarks
@formula("portfolio", formula="P = M + S + A; S = B - M (style); A = P - B (active management)", curriculum=EPP + " (benchmarks)")
def manager_style_active_decomposition(portfolio_return: float, benchmark_return: float, market_return: float) -> dict:
    """Split a manager's return into market, style (benchmark minus market) and active management.

    Returns:
        style_return (S), active_return (A), excess_over_market (P - M).

    Example:
        manager_style_active_decomposition(portfolio_return=0.036, benchmark_return=0.038, market_return=0.028)
        -> style 0.010, active -0.002
    """
    return {"style_return": benchmark_return - market_return, "active_return": portfolio_return - benchmark_return,
            "excess_over_market": portfolio_return - market_return}


@formula("portfolio", formula="E(R) = a + beta x factor return; skill = actual - E(R_benchmark) - [(a_P - a_B) + (beta_P - beta_B) F]",
         curriculum=EPP + " (factor-model benchmarks)")
def factor_model_benchmark_skill(portfolio_actual_return: float, portfolio_zero_factor: float, portfolio_beta: float,
                                 benchmark_zero_factor: float, benchmark_beta: float, factor_return: float) -> dict:
    """One-factor benchmark: expected returns, the part of the differential explained by exposures, and skill.

    Returns:
        portfolio_expected, benchmark_expected, incremental_expected (exposure differences), total_differential
        (actual - benchmark expected), skill.

    Example:
        factor_model_benchmark_skill(portfolio_actual_return=0.0813, portfolio_zero_factor=0.005, portfolio_beta=1.15,
                                     benchmark_zero_factor=0.015, benchmark_beta=0.95, factor_return=0.059)
        -> incremental 0.0018, skill 0.0084
    """
    pe = portfolio_zero_factor + portfolio_beta * factor_return
    be = benchmark_zero_factor + benchmark_beta * factor_return
    inc = pe - be
    tot = portfolio_actual_return - be
    return {"portfolio_expected": pe, "benchmark_expected": be, "incremental_expected": inc,
            "total_differential": tot, "skill": tot - inc}


# ---------------------------------------------------------------- attribution
@formula("portfolio", formula="allocation = sum (w_p - w_b)(R_b,i - R_b) [Brinson-Fachler] or (w_p - w_b) R_b,i [BHB]; "
                              "selection = sum w_b (R_p,i - R_b,i); interaction = sum (w_p - w_b)(R_p,i - R_b,i)",
         curriculum=EPP + " (micro attribution)")
def brinson_attribution(portfolio_weights: list, benchmark_weights: list, portfolio_returns: list,
                        benchmark_returns: list, method: str = "brinson_fachler", names: list | None = None,
                        interaction_in_selection: bool = False) -> dict:
    """Brinson sector attribution: allocation, selection and interaction effects by segment and in total.

    Args:
        portfolio_weights, benchmark_weights: segment weights (each summing to 1).
        portfolio_returns, benchmark_returns: segment returns. For a segment absent from the benchmark (e.g. cash),
            pass its return as the benchmark return so selection/interaction are zero (curriculum practice).
        method: "brinson_fachler" (curriculum: pure sector allocation vs the total benchmark) or "bhb".
        interaction_in_selection: fold interaction into selection (selection = sum w_p (R_p,i - R_b,i)).

    Returns:
        allocation, selection, interaction, total (lists by segment), totals for each, portfolio_return,
        benchmark_return, active_return.

    Example:
        Consumer cyclical (8.69% vs 5.97% weight, benchmark sector 4.62%, benchmark 1.82%) -> allocation 0.076%
    """
    wp, wb = _vec(portfolio_weights, "portfolio_weights"), _vec(benchmark_weights, "benchmark_weights")
    rp, rb = _vec(portfolio_returns, "portfolio_returns"), _vec(benchmark_returns, "benchmark_returns")
    if not (wp.size == wb.size == rp.size == rb.size):
        raise ValueError("all inputs must have one value per segment")
    if method not in ("brinson_fachler", "bhb"):
        raise ValueError("method must be 'brinson_fachler' or 'bhb'")
    Rb = float(wb @ rb)
    alloc = (wp - wb) * (rb - Rb) if method == "brinson_fachler" else (wp - wb) * rb
    sel = wb * (rp - rb)
    inter = (wp - wb) * (rp - rb)
    if interaction_in_selection:
        sel, inter = sel + inter, np.zeros_like(inter)
    Rp = float(wp @ rp)
    out = {"allocation": _f(alloc), "selection": _f(sel), "interaction": _f(inter), "total": _f(alloc + sel + inter),
           "allocation_total": float(alloc.sum()), "selection_total": float(sel.sum()),
           "interaction_total": float(inter.sum()), "portfolio_return": Rp, "benchmark_return": Rb,
           "active_return": Rp - Rb}
    if names is not None:
        out["names"] = list(names)
    return out


@formula("portfolio", formula="risk-free = r_f; asset category = sum w_i (R_Ci - r_f); benchmarks = sum_i sum_j w_i w_ij (R_Bij - R_Ci); "
                              "managers = sum_i sum_j w_i w_ij (R_Aij - R_Bij); allocation effects = actual - sum of layers",
         curriculum=EPP + " (macro attribution)")
def macro_attribution(policy_weights: list, asset_category_returns: list, manager_weights: list,
                      manager_benchmark_returns: list, manager_returns: list, risk_free_rate: float,
                      total_fund_return: float) -> dict:
    """Fund-sponsor macro attribution by decision level (returns), as in the curriculum's layered table.

    Args:
        policy_weights: policy weight w_i of each asset category.
        asset_category_returns: benchmark return R_Ci of each asset category.
        manager_weights: per category, the policy weights w_ij of its managers (list of lists).
        manager_benchmark_returns: per category, each manager's benchmark return R_Bij.
        manager_returns: per category, each manager's actual return R_Aij.
        risk_free_rate: r_f for the period.
        total_fund_return: actual total fund return (the allocation-effects layer is the residual).

    Returns:
        incremental (net_contributions 0, risk_free_asset, asset_category, benchmarks, investment_managers,
        allocation_effects), cumulative (same keys), total_fund_return.

    Example:
        Curriculum MEPA fund (June): risk-free 0.31%, asset category 3.36%, benchmarks 0.27%, managers 0.01%,
        allocation 0.04%, total 3.99%.
    """
    w = _vec(policy_weights, "policy_weights")
    rc = _vec(asset_category_returns, "asset_category_returns")
    if not (len(manager_weights) == len(manager_benchmark_returns) == len(manager_returns) == w.size == rc.size):
        raise ValueError("one entry per asset category in every input")
    ac = float(w @ (rc - risk_free_rate))
    bench = mgr = 0.0
    for i in range(w.size):
        wij = _vec(manager_weights[i], "manager_weights")
        rbij = _vec(manager_benchmark_returns[i], "manager_benchmark_returns")
        raij = _vec(manager_returns[i], "manager_returns")
        if not (wij.size == rbij.size == raij.size):
            raise ValueError(f"category {i}: manager weights, benchmark returns and returns must match")
        bench += float(w[i]) * float(wij @ (rbij - rc[i]))
        mgr += float(w[i]) * float(wij @ (raij - rbij))
    layers = {"net_contributions": 0.0, "risk_free_asset": risk_free_rate, "asset_category": ac, "benchmarks": bench,
              "investment_managers": mgr}
    layers["allocation_effects"] = total_fund_return - sum(layers.values())
    cum, run = {}, 0.0
    for k, v in layers.items():
        run += v
        cum[k] = run
    return {"incremental": layers, "cumulative": cum, "total_fund_return": total_fund_return}


@formula("portfolio", formula="r = sum w*_j I_j0 + sum (w_j - w*_j) I_j + sum (w_j c_j - w*_j C_j) + sum w_j d_j + sum w_j (p_j - I_j); "
                              "c_j = s_j (1 + p_j + d_j), C_j = s_j (1 + I_j)", curriculum=GPE + " (international attribution)")
def international_performance_attribution(portfolio_weights: list, benchmark_weights: list,
                                          portfolio_local_returns: list, index_local_returns: list,
                                          currency_returns: list, yields: list | None = None) -> dict:
    """International attribution (Solnik): benchmark return, market allocation, currency allocation, yield, selection.

    Args:
        portfolio_weights, benchmark_weights: country weights.
        portfolio_local_returns: capital gain of the portfolio's holdings in local currency (p_j).
        index_local_returns: country index return in local currency (I_j).
        currency_returns: % change in the base-currency value of each local currency (s_j; 0 for the base).
        yields: income return per country (d_j), default 0.

    Returns:
        portfolio_return, benchmark_return, market_allocation, currency_allocation, yield_contribution,
        security_selection, active_return, currency_contribution (portfolio), local_return (portfolio).

    Example:
        Curriculum Q2 (US/Japan/Europe): benchmark 0.735%, market -0.500%, currency +0.330%, selection 3.500%,
        portfolio 4.065%.
    """
    w, ws = _vec(portfolio_weights, "portfolio_weights"), _vec(benchmark_weights, "benchmark_weights")
    p, I = _vec(portfolio_local_returns, "portfolio_local_returns"), _vec(index_local_returns, "index_local_returns")
    s = _vec(currency_returns, "currency_returns")
    d = np.zeros_like(p) if yields is None else _vec(yields, "yields")
    if not (w.size == ws.size == p.size == I.size == s.size == d.size):
        raise ValueError("all inputs need one value per country")
    c = s * (1 + p + d)
    C = s * (1 + I)
    bench = float(ws @ (I + C))
    mkt = float((w - ws) @ I)
    cur = float(w @ c - ws @ C)
    yld = float(w @ d)
    sel = float(w @ (p - I))
    port = float(w @ (p + d + c))
    return {"portfolio_return": port, "benchmark_return": bench, "market_allocation": mkt, "currency_allocation": cur,
            "yield_contribution": yld, "security_selection": sel, "active_return": port - bench,
            "currency_contribution": float(w @ c), "local_return": float(w @ (p + d))}


# ---------------------------------------------------------------- appraisal
@formula("portfolio", formula="Sharpe = (Rp - rf)/sd_p; Treynor = (Rp - rf)/beta; alpha = Rp - [rf + beta(Rm - rf)]; "
                              "M^2 = rf + Sharpe x sd_m; IR = (Rp - Rb)/TE; appraisal = alpha/sd_e",
         curriculum="L1 PM: Portfolio Risk and Return Part II; " + EPP)
def risk_adjusted_performance(portfolio_return: float, risk_free_rate: float, portfolio_std_dev: float | None = None,
                              beta: float | None = None, market_return: float | None = None,
                              market_std_dev: float | None = None, benchmark_return: float | None = None,
                              tracking_error: float | None = None, residual_std_dev: float | None = None) -> dict:
    """Sharpe, Treynor, Jensen's alpha, M-squared, information ratio and appraisal ratio (whatever inputs allow).

    Returns:
        any of sharpe_ratio, treynor_ratio, jensen_alpha, capm_expected_return, m_squared (a return: rf + Sharpe x
        sd_m), m_squared_excess (M^2 - Rm, the L1 form), information_ratio, active_return, appraisal_ratio.

    Example:
        risk_adjusted_performance(portfolio_return=0.10, risk_free_rate=0.03, portfolio_std_dev=0.20, beta=1.1,
                                  market_return=0.09, market_std_dev=0.19)
        -> sharpe 0.35, treynor 0.0636, alpha 0.004, m_squared_excess 0.0065
    """
    ex = portfolio_return - risk_free_rate
    out = {}
    if portfolio_std_dev:
        out["sharpe_ratio"] = ex / portfolio_std_dev
        if market_std_dev is not None:
            out["m_squared"] = risk_free_rate + out["sharpe_ratio"] * market_std_dev
            if market_return is not None:
                out["m_squared_excess"] = out["m_squared"] - market_return
    if beta:
        out["treynor_ratio"] = ex / beta
    if beta is not None and market_return is not None:
        er = risk_free_rate + beta * (market_return - risk_free_rate)
        out["capm_expected_return"] = er
        out["jensen_alpha"] = portfolio_return - er
        if residual_std_dev:
            out["appraisal_ratio"] = out["jensen_alpha"] / residual_std_dev
    if benchmark_return is not None:
        out["active_return"] = portfolio_return - benchmark_return
        if tracking_error:
            out["information_ratio"] = out["active_return"] / tracking_error
    if not out:
        raise ValueError("not enough inputs: give portfolio_std_dev, beta (+ market_return) or benchmark_return")
    return out


@formula("portfolio", formula="TE = sd(Rp - Rb) (n-1); IR = mean(Rp - Rb)/TE; annualised x ppy and x sqrt(ppy)", curriculum=EPP + "; " + GPE)
def active_return_tracking_error(portfolio_returns: list, benchmark_returns: list,
                                 periods_per_year: float | None = None) -> dict:
    """Mean active return, tracking error (sample sd of active returns) and information ratio from return series.

    Returns:
        mean_active_return, tracking_error, information_ratio, active_returns, and annualized_* if periods_per_year.

    Example:
        Portfolio [12,14,20,14,16]%, benchmark [14,10,12,16,13]% -> mean 2.2%, TE 4.27%
    """
    p, b = _vec(portfolio_returns, "portfolio_returns"), _vec(benchmark_returns, "benchmark_returns")
    if p.size != b.size or p.size < 2:
        raise ValueError("need two equal-length series with at least two returns")
    a = p - b
    m, te = float(a.mean()), float(np.std(a, ddof=1))
    out = {"mean_active_return": m, "tracking_error": te, "information_ratio": m / te if te else float("nan"),
           "active_returns": _f(a)}
    if periods_per_year:
        out.update(annualized_active_return=m * periods_per_year,
                   annualized_tracking_error=te * math.sqrt(periods_per_year),
                   annualized_information_ratio=(m / te) * math.sqrt(periods_per_year) if te else float("nan"))
    return out


@formula("portfolio", formula="Sortino = (mean R - MAR)/DD; DD = sqrt(sum min(r_t - MAR, 0)^2/(n - 1))",
         curriculum="L3 PM: Risk Management; L3 Alternative Investments Portfolio Management")
def sortino_ratio_from_returns(returns: list, minimum_acceptable_return: float = 0.0,
                               periods_per_year: float | None = None, ddof: int = 1) -> dict:
    """Sortino ratio and downside deviation (target semideviation) from a return series.

    Args:
        returns: period returns.
        minimum_acceptable_return: MAR per period (often the risk-free rate).
        periods_per_year: annualise (mean x ppy, deviation x sqrt(ppy)).
        ddof: 1 uses n - 1 as in the curriculum; 0 uses n.

    Returns:
        sortino_ratio, downside_deviation, mean_return (annualised if periods_per_year).

    Example:
        sortino_ratio_from_returns(returns=[0.02, -0.01, 0.03, -0.02], minimum_acceptable_return=0.0)
    """
    r = _vec(returns, "returns")
    if r.size <= ddof:
        raise ValueError("not enough returns")
    dd = math.sqrt(float(np.sum(np.minimum(r - minimum_acceptable_return, 0) ** 2)) / (r.size - ddof))
    mean, mar = float(r.mean()), minimum_acceptable_return
    if periods_per_year:
        mean, mar, dd = mean * periods_per_year, mar * periods_per_year, dd * math.sqrt(periods_per_year)
    return {"sortino_ratio": (mean - mar) / dd if dd else float("inf"), "downside_deviation": dd, "mean_return": mean}


@formula("portfolio", formula="max drawdown = max peak-to-trough decline of cumulative wealth; RoMAD = average annual return/MDD; "
                              "Calmar = annualised compound return/MDD", curriculum="L3 PM: Risk Management (RoMAD)")
def drawdown_ratios(returns: list | None = None, periods_per_year: float = 1.0,
                    average_annual_return: float | None = None, max_drawdown: float | None = None) -> dict:
    """Maximum drawdown, return over maximum drawdown (RoMAD) and Calmar ratio.

    Give a return series, or `average_annual_return` and `max_drawdown` directly.

    Returns:
        max_drawdown, romad, calmar_ratio (series only), annualized_return (series only).

    Example:
        drawdown_ratios(average_annual_return=0.15, max_drawdown=0.10) -> romad 1.5
    """
    out = {}
    if returns is not None:
        r = _vec(returns, "returns")
        wealth = np.concatenate([[1.0], np.cumprod(1 + r)])
        mdd = float(np.max(1 - wealth / np.maximum.accumulate(wealth)))
        ann = float(wealth[-1] ** (periods_per_year / r.size) - 1)
        out.update(max_drawdown=mdd, annualized_return=ann, average_annual_return=float(r.mean() * periods_per_year))
        out["calmar_ratio"] = ann / mdd if mdd else float("inf")
        average_annual_return = out["average_annual_return"] if average_annual_return is None else average_annual_return
        max_drawdown = mdd if max_drawdown is None else max_drawdown
    if average_annual_return is None or max_drawdown is None:
        raise ValueError("give returns, or average_annual_return and max_drawdown")
    if max_drawdown <= 0:
        raise ValueError("max_drawdown must be > 0")
    out.setdefault("max_drawdown", max_drawdown)
    out["romad"] = average_annual_return / max_drawdown
    return out
