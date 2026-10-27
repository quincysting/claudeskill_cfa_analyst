"""Risk management: VaR (parametric, historical, Monte Carlo), expected shortfall, component VaR, stress tests,
risk budgeting and derivative credit exposure (CFA L3 Risk Management). Topic ``portfolio``.

Curriculum conventions (L3 Risk Management): VaR = -(mu_h - z sigma_h) x value with the mean INCLUDED, mu_h = mu x h and
sigma_h = sigma x sqrt(h) where h = horizon / input period (e.g. 1/250 for daily from annual); z = 1.65 (95%) and
2.33 (99%) unless `z` is given. sqrt(t) scaling of a VaR figure is only valid with a zero mean.
"""
from __future__ import annotations

import math

import numpy as np
from scipy.stats import norm

from ._registry import formula
from .portfolio import _cov_matrix, _f, _vec

RM = "L3 PM: Risk Management"
_CURRICULUM_Z = {0.90: 1.28, 0.95: 1.65, 0.975: 1.96, 0.99: 2.33}


def _z(confidence: float, z: float | None) -> float:
    if not 0.5 < confidence < 1:
        raise ValueError("confidence must be between 0.5 and 1 (e.g. 0.95)")
    if z is not None:
        return z
    return _CURRICULUM_Z.get(round(confidence, 6), float(norm.ppf(confidence)))


def _mu_sigma(expected_return, std_dev, weights, expected_returns, cov, sds, corr):
    if weights is not None:
        w = _vec(weights, "weights")
        c = _cov_matrix(w.size, cov, sds, corr)
        mu = 0.0 if expected_returns is None else float(w @ _vec(expected_returns, "expected_returns"))
        return mu, math.sqrt(float(w @ c @ w))
    if std_dev is None:
        raise ValueError("give std_dev (and expected_return), or weights with cov / sds + corr")
    return (expected_return or 0.0), std_dev


def _order_stat_var(r: np.ndarray, confidence: float) -> dict:
    """k-th worst return with k = p x n rounded half up (min 1), as in the curriculum's examples."""
    n = r.size
    p = 1 - confidence
    srt = np.sort(r)
    pn = p * n
    k = max(1, int(math.floor(pn + 0.5)))
    lo, hi = max(1, int(math.floor(pn))), max(1, int(math.ceil(pn)))
    return {"k": k, "var_return": float(srt[k - 1]), "avg_adjacent_return": float((srt[lo - 1] + srt[hi - 1]) / 2),
            "tail_mean_return": float(srt[:k].mean()), "n": n}


@formula("portfolio", formula="VaR = -(mu_h - z sigma_h) x V; mu_h = mu x h, sigma_h = sigma sqrt(h); "
                              "ES (normal) = (-mu_h + sigma_h phi(z_c)/(1 - c)) x V, z_c = exact quantile N^-1(c)",
         curriculum=RM + " (analytical / variance-covariance VaR)")
def var_parametric(portfolio_value: float, confidence: float = 0.95, expected_return: float | None = None,
                   std_dev: float | None = None, weights: list | None = None, expected_returns: list | None = None,
                   cov: list | None = None, sds: list | None = None, corr=None, input_periods_per_year: float = 1.0,
                   horizon_periods_per_year: float = 1.0, z: float | None = None, include_mean: bool = True) -> dict:
    """Parametric (analytical, variance-covariance) VaR for an asset or a portfolio, any horizon and confidence.

    Args:
        portfolio_value: money value.
        confidence: e.g. 0.95 (5% VaR) or 0.99 (1% VaR).
        expected_return, std_dev: for a single asset/portfolio, per input period (annual by default).
        weights, expected_returns, cov | sds + corr: alternative portfolio inputs.
        input_periods_per_year: frequency of the inputs (1 annual, 12 monthly).
        horizon_periods_per_year: VaR horizon (1 annual, 12 monthly, 52 weekly, 250 daily).
        z: override the VaR critical value; default curriculum values 1.65 (95%), 2.33 (99%), 1.96, 1.28,
            else the exact normal quantile. VaR uses this z; expected shortfall always uses the exact quantile
            norm.ppf(confidence) (1.6449 at 95%), because ES = -mu + sigma phi(z)/(1 - c) holds only there.
        include_mean: False sets the expected return to zero (then VaR scales with sqrt(t)).

    Returns:
        var (money, positive = loss), var_pct, expected_shortfall (money), expected_shortfall_pct, z (VaR),
        z_expected_shortfall (exact quantile used for ES), horizon_mean, horizon_std_dev.

    Example:
        var_parametric(portfolio_value=50e6, weights=[0.75, 0.25], expected_returns=[0.12, 0.18], sds=[0.20, 0.40],
                       corr=0.90, confidence=0.95) -> var ~13.4 million (book: $13.4 million)
    """
    if portfolio_value <= 0:
        raise ValueError("portfolio_value must be > 0")
    zz = _z(confidence, z)
    mu, sd = _mu_sigma(expected_return, std_dev, weights, expected_returns, cov, sds, corr)
    if sd < 0:
        raise ValueError("std_dev must be >= 0")
    h = input_periods_per_year / horizon_periods_per_year
    mu_h = mu * h if include_mean else 0.0
    sd_h = sd * math.sqrt(h)
    var_pct = -(mu_h - zz * sd_h)
    z_es = float(norm.ppf(confidence))  # phi(z)/(1 - c) is the normal tail mean only at the exact quantile
    es_pct = -mu_h + sd_h * float(norm.pdf(z_es)) / (1 - confidence)
    return {"var": var_pct * portfolio_value, "var_pct": var_pct, "expected_shortfall": es_pct * portfolio_value,
            "expected_shortfall_pct": es_pct, "z": zz, "z_expected_shortfall": z_es, "horizon_mean": mu_h,
            "horizon_std_dev": sd_h}


@formula("portfolio", formula="VaR_T = VaR_1 x sqrt(T) (zero mean, independent returns)", curriculum=RM)
def var_sqrt_time_scaling(var: float, periods: float) -> dict:
    """Scale a one-period VaR to `periods` periods (e.g. daily -> annual with 250). Valid only with zero mean.

    Returns:
        scaled_var, factor.

    Example:
        var_sqrt_time_scaling(var=100000, periods=250) -> scaled_var 1,581,139
    """
    if periods <= 0:
        raise ValueError("periods must be > 0")
    f = math.sqrt(periods)
    return {"scaled_var": var * f, "factor": f}


@formula("portfolio", formula="VaR = -R_(k) x V, R_(k) the k-th worst return, k = (1 - c) n rounded; ES = mean of the k worst",
         curriculum=RM + " (historical simulation)")
def var_historical(returns: list, portfolio_value: float = 1.0, confidence: float = 0.95) -> dict:
    """Historical-simulation VaR and expected shortfall from a return history.

    The k-th worst return is used with k = (1 - confidence) x n rounded half up (minimum 1), matching the
    curriculum (5% of 240 -> 12th worst; 1% of 240 = 2.4 -> 2nd worst; 1% of 60 = 0.6 -> worst).

    Returns:
        var, var_pct, k, n, var_avg_adjacent (averaging the order statistics either side of (1-c)n, the
        curriculum's alternative), expected_shortfall, expected_shortfall_pct.

    Example:
        240 monthly returns, 12th worst -0.11553, value 100,000 -> var 11,553
    """
    r = _vec(returns, "returns")
    o = _order_stat_var(r, confidence)
    return {"var": -o["var_return"] * portfolio_value, "var_pct": -o["var_return"], "k": o["k"], "n": o["n"],
            "var_avg_adjacent": -o["avg_adjacent_return"] * portfolio_value,
            "expected_shortfall": -o["tail_mean_return"] * portfolio_value,
            "expected_shortfall_pct": -o["tail_mean_return"]}


@formula("portfolio", formula="simulate R ~ N(mu_h, sigma_h); VaR = k-th worst simulated loss; ES = mean of the k worst",
         curriculum=RM + " (Monte Carlo simulation)")
def var_monte_carlo(portfolio_value: float, confidence: float = 0.95, expected_return: float | None = None,
                    std_dev: float | None = None, weights: list | None = None, expected_returns: list | None = None,
                    cov: list | None = None, sds: list | None = None, corr=None, input_periods_per_year: float = 1.0,
                    horizon_periods_per_year: float = 1.0, n_sims: int = 100_000, seed: int = 42) -> dict:
    """Monte Carlo VaR under normally distributed returns (seeded, deterministic).

    Inputs as in var_parametric. Converges to the parametric VaR (exact z) as n_sims grows.

    Returns:
        var, var_pct, expected_shortfall, expected_shortfall_pct, n_sims, seed.

    Example:
        var_monte_carlo(portfolio_value=50e6, expected_return=0.135, std_dev=0.244) -> var ~ 13.5 million
        (book: $13.4 million, the parametric answer with z = 1.65; the simulation uses the exact quantile, seed 42)
    """
    if n_sims < 20:
        raise ValueError("n_sims must be >= 20")
    mu, sd = _mu_sigma(expected_return, std_dev, weights, expected_returns, cov, sds, corr)
    h = input_periods_per_year / horizon_periods_per_year
    sim = np.random.default_rng(seed).normal(mu * h, sd * math.sqrt(h), size=n_sims)
    o = _order_stat_var(sim, confidence)
    return {"var": -o["var_return"] * portfolio_value, "var_pct": -o["var_return"],
            "expected_shortfall": -o["tail_mean_return"] * portfolio_value,
            "expected_shortfall_pct": -o["tail_mean_return"], "n_sims": n_sims, "seed": seed}


@formula("portfolio", formula="marginal VaR_i = z (Cov v)_i / sd_P; component VaR_i = v_i x marginal_i (sums to VaR); "
                              "incremental VaR_i = VaR - VaR(without i); zero mean", curriculum=RM + " (incremental VaR)")
def var_component_parametric(position_values: list, cov: list | None = None, sds: list | None = None, corr=None,
                             confidence: float = 0.95, z: float | None = None) -> dict:
    """Decompose zero-mean parametric VaR into component, marginal and incremental VaR by position.

    Args:
        position_values: money held in each position.
        cov | sds + corr: return covariance for the VaR horizon.

    Returns:
        var, component_var (sums to var), pct_contribution, marginal_var (per unit of money), incremental_var
        (VaR minus VaR with the position removed), diversification_benefit (sum of stand-alone VaRs - var).

    Example:
        Two positions of 100 with sd 0.10 and 0.20, corr 0 -> var 36.89 at 95% (z 1.65)
    """
    v = _vec(position_values, "position_values")
    c = _cov_matrix(v.size, cov, sds, corr)
    zz = _z(confidence, z)
    sd_p = math.sqrt(float(v @ c @ v))
    var = zz * sd_p
    marg = zz * (c @ v) / sd_p if sd_p > 0 else np.zeros_like(v)
    comp = v * marg
    inc = []
    for i in range(v.size):
        vi = v.copy()
        vi[i] = 0.0
        inc.append(var - zz * math.sqrt(float(vi @ c @ vi)))
    alone = zz * np.abs(v) * np.sqrt(np.diag(c))
    return {"var": var, "component_var": _f(comp), "pct_contribution": _f(comp / var if var else comp),
            "marginal_var": _f(marg), "incremental_var": inc, "standalone_var": _f(alone),
            "diversification_benefit": float(alone.sum() - var)}


@formula("portfolio", formula="P&L_s = sum_i value_i x shock_(s,i)", curriculum=RM + " (stress testing, scenario analysis)")
def stress_test_scenarios(position_values: list, scenarios: dict) -> dict:
    """Revalue a portfolio under named shock scenarios (returns per position).

    Args:
        position_values: money held in each position.
        scenarios: {"name": [shock per position, decimal return]}.

    Returns:
        pnl (per scenario), pnl_pct, worst_scenario, worst_pnl.

    Example:
        stress_test_scenarios(position_values=[60, 40], scenarios={"equity -10%": [-0.10, 0.0]}) -> pnl -6
    """
    v = _vec(position_values, "position_values")
    if not scenarios:
        raise ValueError("give at least one scenario")
    pnl = {}
    for name, shocks in scenarios.items():
        s = _vec(shocks, f"scenario {name}")
        if s.size != v.size:
            raise ValueError(f"scenario {name!r} needs one shock per position")
        pnl[name] = float(v @ s)
    worst = min(pnl, key=pnl.get)
    return {"pnl": pnl, "pnl_pct": {k: x / float(v.sum()) for k, x in pnl.items()}, "worst_scenario": worst,
            "worst_pnl": pnl[worst]}


@formula("portfolio", formula="return on capital = profit/capital; return on VaR = profit/VaR; "
                              "combined VaR = sqrt(VaR' R VaR)", curriculum=RM + " (risk budgeting, capital allocation)")
def risk_budget_allocation(capital: list, var_budgets: list, profits: list, correlation: list | None = None,
                           names: list | None = None) -> dict:
    """Evaluate business units (desks, managers) against their capital and VaR budgets.

    Args:
        capital: capital allocated to each unit.
        var_budgets: VaR budget (or VaR used) of each unit.
        profits: profit of each unit over the evaluation period.
        correlation: optional correlation matrix of unit results, for the combined (diversified) VaR.

    Returns:
        return_on_capital, return_on_var, combined_var (perfect correlation unless `correlation` given),
        undiversified_var, total_profit, best_by_return_on_capital.

    Example:
        risk_budget_allocation(capital=[100e6, 200e6], var_budgets=[5e6, 5e6], profits=[20e6, 25e6])
        -> return_on_capital [0.20, 0.125], combined_var 10e6
    """
    k = _vec(capital, "capital")
    v = _vec(var_budgets, "var_budgets")
    p = _vec(profits, "profits")
    if not (k.size == v.size == p.size):
        raise ValueError("capital, var_budgets and profits must have the same length")
    R = np.ones((v.size, v.size)) if correlation is None else np.asarray(correlation, dtype=float)
    roc = p / k
    labels = list(names) if names is not None else list(range(k.size))
    return {"return_on_capital": _f(roc), "return_on_var": _f(p / v), "combined_var": math.sqrt(float(v @ R @ v)),
            "undiversified_var": float(v.sum()), "total_profit": float(p.sum()),
            "best_by_return_on_capital": labels[int(np.argmax(roc))]}


@formula("portfolio", formula="IR_P = sqrt(sum IR_i^2); optimal active risk from manager i = TE_P x IR_i/IR_P (uncorrelated)",
         curriculum="L3 PM: Equity Portfolio Management (core-satellite, risk budgeting)")
def active_risk_budget_managers(information_ratios: list, total_active_risk: float,
                                manager_tracking_errors: list | None = None) -> dict:
    """Optimal split of an active-risk (tracking error) budget across managers with uncorrelated active returns.

    Returns:
        portfolio_information_ratio, active_risk_allocations (contribution of each manager), expected_active_return,
        manager_weights (allocation to each manager = allocation / its own tracking error, if given; the rest is
        indexed).

    Example:
        active_risk_budget_managers(information_ratios=[0.3, 0.4], total_active_risk=0.02)
        -> IR_P 0.5, active_risk_allocations [0.012, 0.016]
    """
    ir = _vec(information_ratios, "information_ratios")
    irp = math.sqrt(float(ir @ ir))
    if irp == 0:
        raise ValueError("at least one information ratio must be non-zero")
    alloc = total_active_risk * ir / irp
    out = {"portfolio_information_ratio": irp, "active_risk_allocations": _f(alloc),
           "expected_active_return": irp * total_active_risk}
    if manager_tracking_errors is not None:
        te = _vec(manager_tracking_errors, "manager_tracking_errors")
        w = alloc / te
        out["manager_weights"] = _f(w)
        out["passive_weight"] = 1 - float(w.sum())
    return out


@formula("portfolio", formula="value to long = S/(1 + r_f)^T - F/(1 + r)^T; credit risk borne by the party with positive value",
         curriculum=RM + " (credit risk of forward contracts)")
def credit_risk_forward(spot: float, forward_price: float, rate: float, time_remaining: float,
                        foreign_rate: float = 0.0, payment_due_now: float = 0.0) -> dict:
    """Current and potential credit exposure of a forward (equity or currency) during its life.

    Args:
        spot: current spot price (domestic per unit of foreign currency for FX forwards).
        forward_price: contract forward price.
        rate: domestic risk-free rate (annual, discrete).
        time_remaining: years to expiry.
        foreign_rate: foreign risk-free rate (FX forwards) or yield; 0 for a non-income asset.
        payment_due_now: any amount currently due (current credit risk); 0 mid-life.

    Returns:
        value_long, value_short, bearer ("long"/"short"/"none"), potential_credit_exposure, current_credit_exposure.

    Example:
        credit_risk_forward(spot=102, forward_price=105, rate=0.05, time_remaining=0.75) -> value_long 0.7728
    """
    vl = spot / (1 + foreign_rate) ** time_remaining - forward_price / (1 + rate) ** time_remaining
    bearer = "long" if vl > 0 else "short" if vl < 0 else "none"
    return {"value_long": vl, "value_short": -vl, "bearer": bearer, "potential_credit_exposure": abs(vl),
            "current_credit_exposure": payment_due_now}
