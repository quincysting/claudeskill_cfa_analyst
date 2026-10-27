"""Capital market expectations, strategic asset allocation and rebalancing (CFA L3).

Topic ``portfolio``. Decimals throughout. Optimisers use numpy closed forms when unconstrained and scipy SLSQP
(fixed start, fixed tolerances) when long-only, so results are deterministic.
"""
from __future__ import annotations

import math

import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm

from ._registry import formula
from .portfolio import _cov_matrix, _f, _vec

CME = "L3 PM: Capital Market Expectations"
AA = "L3 PM: Asset Allocation"
REB = "L3 PM: Monitoring and Rebalancing"


# ---------------------------------------------------------------- capital market expectations
@formula("portfolio", formula="E(Re) = D/P + (i + g) - %dS + %d(P/E)", curriculum=CME + " (Grinold-Kroner)")
def cme_grinold_kroner(dividend_yield: float, inflation: float, real_earnings_growth: float,
                       change_in_shares: float = 0.0, change_in_pe: float = 0.0) -> dict:
    """Grinold-Kroner expected equity return.

    Args:
        dividend_yield: expected D1/P0, decimal.
        inflation: expected inflation i.
        real_earnings_growth: expected real total earnings growth g.
        change_in_shares: expected % change in shares outstanding (negative = net buybacks; repurchase yield = -this).
        change_in_pe: expected % change in the P/E per year (repricing return).

    Returns:
        expected_return, income_return (D/P - %dS), nominal_earnings_growth (i + g), repricing_return.

    Example:
        cme_grinold_kroner(dividend_yield=0.02, inflation=0.025, real_earnings_growth=0.03,
                           change_in_shares=-0.01, change_in_pe=0.0) -> expected_return 0.085
    """
    income = dividend_yield - change_in_shares
    growth = inflation + real_earnings_growth
    return {"expected_return": income + growth + change_in_pe, "income_return": income,
            "nominal_earnings_growth": growth, "repricing_return": change_in_pe}


@formula("portfolio", formula="RP_int = sd_i x rho_i,GIM x (RP_GIM/sd_GIM); RP_seg = sd_i x Sharpe_GIM; "
                              "RP = phi RP_int + (1 - phi) RP_seg + illiquidity premium",
         curriculum=CME + " (Singer-Terhaar)")
def cme_singer_terhaar(std_dev: float, correlation_with_global: float, global_sharpe: float,
                       integration_degree: float = 1.0, illiquidity_premium: float = 0.0,
                       risk_free_rate: float | None = None, segmented_sharpe: float | None = None) -> dict:
    """Singer-Terhaar risk premium: blend of fully integrated (CAPM vs the global market) and fully segmented.

    Args:
        std_dev: the asset class's standard deviation.
        correlation_with_global: its correlation with the global investable market (GIM).
        global_sharpe: Sharpe ratio of the GIM (market price of risk).
        integration_degree: phi, 1 = fully integrated, 0 = fully segmented.
        illiquidity_premium: added to the blended premium.
        risk_free_rate: optional, to report the expected return.
        segmented_sharpe: Sharpe ratio used for the segmented case (defaults to global_sharpe, as in the curriculum).

    Returns:
        risk_premium, integrated_premium, segmented_premium, expected_return (if risk_free_rate given).

    Example:
        cme_singer_terhaar(std_dev=0.15, correlation_with_global=0.7, global_sharpe=0.3, integration_degree=0.8)
        -> integrated 0.0315, segmented 0.045, risk_premium 0.0342
    """
    if not 0 <= integration_degree <= 1:
        raise ValueError("integration_degree must be between 0 and 1")
    seg_sr = global_sharpe if segmented_sharpe is None else segmented_sharpe
    integ = std_dev * correlation_with_global * global_sharpe
    seg = std_dev * seg_sr
    rp = integration_degree * integ + (1 - integration_degree) * seg + illiquidity_premium
    out = {"risk_premium": rp, "integrated_premium": integ, "segmented_premium": seg}
    if risk_free_rate is not None:
        out["expected_return"] = risk_free_rate + rp
    return out


@formula("portfolio", formula="ERP = D1/P0 + g - r_LT (long-term government bond yield)", curriculum=CME + " (Gordon growth model)")
def cme_gordon_growth_erp(dividend_yield: float, growth: float, bond_yield: float) -> dict:
    """Equity risk premium from the Gordon (constant) growth model.

    Args:
        dividend_yield: forward D1/P0.
        growth: long-term dividend (= earnings) growth.
        bond_yield: long-term government bond yield.

    Returns:
        equity_risk_premium, expected_equity_return.

    Example:
        cme_gordon_growth_erp(dividend_yield=0.02, growth=0.05, bond_yield=0.04) -> equity_risk_premium 0.03
    """
    er = dividend_yield + growth
    return {"equity_risk_premium": er - bond_yield, "expected_equity_return": er}


@formula("portfolio", formula="E(R_B) = real rf + inflation premium + default + illiquidity + maturity + tax premium",
         curriculum=CME + " (risk premium approach)")
def cme_bond_risk_premium_buildup(real_risk_free_rate: float, inflation_premium: float, default_premium: float = 0.0,
                                  illiquidity_premium: float = 0.0, maturity_premium: float = 0.0,
                                  tax_premium: float = 0.0) -> dict:
    """Build-up (risk premium) estimate of a bond's expected return.

    Returns:
        expected_return, nominal_risk_free_rate (real + inflation), total_risk_premium.

    Example:
        cme_bond_risk_premium_buildup(real_risk_free_rate=0.012, inflation_premium=0.025, default_premium=0.01,
                                      maturity_premium=0.008) -> expected_return 0.055
    """
    nominal = real_risk_free_rate + inflation_premium
    prem = default_premium + illiquidity_premium + maturity_premium + tax_premium
    return {"expected_return": nominal + prem, "nominal_risk_free_rate": nominal, "total_risk_premium": prem}


@formula("portfolio", formula="Cov = B F B' + diag(residual variances); Cov_ij = sum_k sum_l b_ik b_jl Cov(F_k, F_l)",
         curriculum=CME + " (multifactor models)")
def cme_multifactor_covariance(factor_loadings: list, factor_covariance: list,
                               residual_variances: list | None = None) -> dict:
    """Asset covariance matrix implied by a multifactor model.

    Args:
        factor_loadings: one row of factor sensitivities per asset (N x K).
        factor_covariance: K x K factor covariance matrix (a K-list of variances is accepted for uncorrelated factors).
        residual_variances: specific (residual) variance per asset, added on the diagonal.

    Returns:
        covariance, std_devs, correlation.

    Example:
        Two assets, one factor of variance 0.04, loadings 1.0 and 0.5, residual variances 0.01 each ->
        cov[0][1] = 0.02, var[0] = 0.05
    """
    B = np.atleast_2d(np.asarray(factor_loadings, dtype=float))
    F = np.asarray(factor_covariance, dtype=float)
    if F.ndim == 1:
        F = np.diag(F)
    if F.shape != (B.shape[1], B.shape[1]):
        raise ValueError("factor_covariance must be K x K where K = number of loadings per asset")
    C = B @ F @ B.T
    if residual_variances is not None:
        e = _vec(residual_variances, "residual_variances")
        if e.size != B.shape[0]:
            raise ValueError("one residual variance per asset")
        C = C + np.diag(e)
    sd = np.sqrt(np.diag(C))
    return {"covariance": _f(C), "std_devs": _f(sd), "correlation": _f(C / np.outer(sd, sd))}


@formula("portfolio", formula="shrinkage estimate = w x target + (1 - w) x sample", curriculum=CME + " (shrinkage estimators)")
def cme_shrinkage_estimate(sample, target, shrinkage_weight: float) -> dict:
    """Shrinkage estimator of a mean vector or covariance matrix toward a target (prior).

    Args:
        sample: historical estimate (number, list or matrix).
        target: target/prior estimate, same shape (e.g. factor-model covariance).
        shrinkage_weight: weight on the target, 0..1.

    Returns:
        estimate.

    Example:
        cme_shrinkage_estimate(sample=0.08, target=0.06, shrinkage_weight=0.3) -> estimate 0.074
    """
    if not 0 <= shrinkage_weight <= 1:
        raise ValueError("shrinkage_weight must be between 0 and 1")
    s, t = np.asarray(sample, dtype=float), np.asarray(target, dtype=float)
    if s.shape != t.shape:
        raise ValueError("sample and target must have the same shape")
    est = shrinkage_weight * t + (1 - shrinkage_weight) * s
    return {"estimate": float(est) if est.ndim == 0 else _f(est)}


@formula("portfolio", formula="r_t = (1 - lambda) R_t + lambda r_(t-1)  =>  R_t = (r_t - lambda r_(t-1))/(1 - lambda); "
                              "sd(R) = sd(r) x sqrt((1 + lambda)/(1 - lambda))",
         curriculum=CME + " (smoothed data, appraisal-based returns)")
def cme_unsmoothed_volatility(smoothing_lambda: float, smoothed_std_dev: float | None = None,
                              smoothed_returns: list | None = None) -> dict:
    """Undo appraisal smoothing: true volatility from smoothed volatility, and/or the unsmoothed return series.

    Args:
        smoothing_lambda: weight on the prior period's smoothed return, 0 <= lambda < 1.
        smoothed_std_dev: sd of the smoothed (reported) returns.
        smoothed_returns: reported return series; returns the unsmoothed series (one shorter).

    Returns:
        unsmoothed_std_dev and/or unsmoothed_returns (+ their sample sd), adjustment_factor.

    Example:
        cme_unsmoothed_volatility(smoothing_lambda=0.6, smoothed_std_dev=0.10) -> unsmoothed_std_dev 0.2
    """
    lam = smoothing_lambda
    if not 0 <= lam < 1:
        raise ValueError("smoothing_lambda must be in [0, 1)")
    factor = math.sqrt((1 + lam) / (1 - lam))
    out = {"adjustment_factor": factor}
    if smoothed_std_dev is not None:
        out["unsmoothed_std_dev"] = smoothed_std_dev * factor
    if smoothed_returns is not None:
        r = _vec(smoothed_returns, "smoothed_returns")
        if r.size < 2:
            raise ValueError("need at least two smoothed returns")
        R = (r[1:] - lam * r[:-1]) / (1 - lam)
        out["unsmoothed_returns"] = _f(R)
        if R.size > 1:
            out["unsmoothed_returns_sample_std_dev"] = float(np.std(R, ddof=1))
    if len(out) == 1:
        raise ValueError("give smoothed_std_dev and/or smoothed_returns")
    return out


# ---------------------------------------------------------------- mean-variance optimisation
def _solve_mvo(mu, C, risk_aversion, target_return, long_only, lo=None, hi=None):
    n = mu.size
    if not long_only and lo is None and hi is None:
        ones = np.ones(n)
        if risk_aversion is not None:  # max w'mu - 0.5 A w'Cw s.t. 1'w = 1
            K = np.block([[risk_aversion * C, ones[:, None]], [ones[None, :], np.zeros((1, 1))]])
            sol = np.linalg.solve(K, np.concatenate([mu, [1.0]]))
            return sol[:n]
        A = [ones] + ([mu] if target_return is not None else [])
        A = np.array(A)
        b = np.array([1.0] + ([target_return] if target_return is not None else []))
        K = np.block([[2 * C, A.T], [A, np.zeros((A.shape[0], A.shape[0]))]])
        sol = np.linalg.solve(K, np.concatenate([np.zeros(n), b]))
        return sol[:n]
    lo = np.zeros(n) if lo is None else lo
    hi = np.ones(n) if hi is None else hi
    if not long_only:
        lo = np.where(np.isfinite(lo), lo, -10.0)
    cons = [{"type": "eq", "fun": lambda w: w.sum() - 1.0, "jac": lambda w: np.ones(n)}]
    if risk_aversion is not None:
        fun = lambda w: -(w @ mu - 0.5 * risk_aversion * w @ C @ w)  # noqa: E731
        jac = lambda w: -(mu - risk_aversion * C @ w)  # noqa: E731
    else:
        fun = lambda w: w @ C @ w  # noqa: E731
        jac = lambda w: 2 * C @ w  # noqa: E731
        if target_return is not None:
            cons.append({"type": "eq", "fun": lambda w: w @ mu - target_return, "jac": lambda w: mu})
    res = minimize(fun, np.full(n, 1.0 / n), jac=jac, method="SLSQP", bounds=list(zip(lo, hi)), constraints=cons,
                   options={"ftol": 1e-15, "maxiter": 1000})
    if not res.success:
        raise ValueError(f"optimisation failed: {res.message} (is the target return feasible?)")
    w = np.clip(res.x, lo, hi)
    return w / w.sum()


@formula("portfolio", formula="max w'mu - 0.5 A w'Cov w, or min w'Cov w s.t. w'mu = target; 1'w = 1; long-only: w >= 0",
         curriculum=AA + " (mean-variance optimisation); " + "L2 PM: Portfolio Concepts")
def mvo_optimize(expected_returns: list, cov: list | None = None, sds: list | None = None, corr=None,
                 risk_aversion: float | None = None, target_return: float | None = None, long_only: bool = True,
                 min_weights: list | None = None, max_weights: list | None = None,
                 risk_free_rate: float | None = None) -> dict:
    """Mean-variance optimal asset allocation, unconstrained (closed form) or long-only/bounded (SLSQP).

    Modes: `risk_aversion` maximises U = E(R) - 0.5 A var; `target_return` minimises variance at that return;
    neither gives the minimum-variance portfolio.

    Args:
        expected_returns: decimals. cov, or sds + corr.
        long_only: True forbids short sales (weights in [0, 1]); False allows them.
        min_weights, max_weights: optional per-asset bounds (override long_only defaults).
        risk_free_rate: optional, to report the Sharpe ratio.

    Returns:
        weights, expected_return, std_dev, utility (if risk_aversion), sharpe_ratio (if risk_free_rate).

    Example:
        mvo_optimize(expected_returns=[0.10, 0.06], sds=[0.20, 0.10], corr=0.5, risk_aversion=4, long_only=False)
        -> weights [0.3333, 0.6667]
    """
    if risk_aversion is not None and target_return is not None:
        raise ValueError("give risk_aversion or target_return, not both")
    if risk_aversion is not None and risk_aversion <= 0:
        raise ValueError("risk_aversion must be > 0")
    mu = _vec(expected_returns, "expected_returns")
    C = _cov_matrix(mu.size, cov, sds, corr)
    lo = None if min_weights is None else _vec(min_weights, "min_weights")
    hi = None if max_weights is None else _vec(max_weights, "max_weights")
    if not long_only and lo is not None and hi is None:
        hi = np.full(mu.size, 10.0)
    if not long_only and hi is not None and lo is None:
        lo = np.full(mu.size, -10.0)
    w = _solve_mvo(mu, C, risk_aversion, target_return, long_only, lo, hi)
    er, var = float(w @ mu), float(w @ C @ w)
    out = {"weights": _f(w), "expected_return": er, "std_dev": math.sqrt(max(var, 0.0))}
    if risk_aversion is not None:
        out["utility"] = er - 0.5 * risk_aversion * var
    if risk_free_rate is not None and var > 0:
        out["sharpe_ratio"] = (er - risk_free_rate) / math.sqrt(var)
    return out


@formula("portfolio", formula="average over simulated (mu_hat, Cov_hat) of the MVO weights", curriculum=AA + " (resampled efficient frontier)")
def mvo_resampled_weights(expected_returns: list, risk_aversion: float, cov: list | None = None,
                          sds: list | None = None, corr=None, n_samples: int = 200, n_observations: int = 60,
                          long_only: bool = True, seed: int = 42) -> dict:
    """Michaud-style resampled allocation: re-estimate inputs from simulated histories, optimise, average weights.

    Args:
        expected_returns, cov (or sds + corr): the point estimates (per period of the simulated observations).
        risk_aversion: A used for each optimisation (utility mode).
        n_samples: number of simulated histories.
        n_observations: length of each simulated history.
        seed: random seed (results are deterministic for a given seed).

    Returns:
        weights (average), weights_std (dispersion across samples), point_estimate_weights, expected_return, std_dev.

    Example:
        mvo_resampled_weights(expected_returns=[0.08, 0.05], sds=[0.18, 0.06], corr=0.2, risk_aversion=4)
    """
    mu = _vec(expected_returns, "expected_returns")
    C = _cov_matrix(mu.size, cov, sds, corr)
    if n_samples < 1 or n_observations < 2:
        raise ValueError("n_samples >= 1 and n_observations >= 2")
    rng = np.random.default_rng(seed)
    ws = []
    for _ in range(n_samples):
        x = rng.multivariate_normal(mu, C, size=n_observations)
        ws.append(_solve_mvo(x.mean(axis=0), np.cov(x, rowvar=False).reshape(mu.size, mu.size), risk_aversion,
                             None, long_only))
    W = np.array(ws)
    w = W.mean(axis=0)
    return {"weights": _f(w), "weights_std": _f(W.std(axis=0)),
            "point_estimate_weights": _f(_solve_mvo(mu, C, risk_aversion, None, long_only)),
            "expected_return": float(w @ mu), "std_dev": math.sqrt(float(w @ C @ w))}


@formula("portfolio", formula="w = (E_target - E_low)/(E_high - E_low) on the higher-return corner; "
                              "sd <= w sd_high + (1 - w) sd_low", curriculum=AA + " (corner portfolios)")
def corner_portfolio_combination(target_return: float, corner_returns: list, corner_std_devs: list,
                                 corner_weights: list | None = None) -> dict:
    """Efficient portfolio for a target return as a mix of the two adjacent corner portfolios.

    Args:
        target_return: required expected return.
        corner_returns: expected return of each corner portfolio (any order).
        corner_std_devs: their standard deviations.
        corner_weights: optional asset weights of each corner portfolio (one list per corner).

    Returns:
        high_corner, low_corner (indices), weight_high_corner, weight_low_corner, std_dev_approx
        (linear interpolation, an upper bound), asset_weights (if corner_weights given).

    Example:
        corner_portfolio_combination(target_return=0.08, corner_returns=[0.10, 0.07], corner_std_devs=[0.15, 0.10])
        -> weight_high_corner 0.3333, std_dev_approx 0.1167
    """
    r = _vec(corner_returns, "corner_returns")
    s = _vec(corner_std_devs, "corner_std_devs")
    if r.size != s.size or r.size < 2:
        raise ValueError("need at least two corner portfolios with returns and std devs")
    above = [i for i in range(r.size) if r[i] >= target_return]
    below = [i for i in range(r.size) if r[i] <= target_return]
    if not above or not below:
        raise ValueError("target_return is outside the range of the corner portfolios")
    hi = min(above, key=lambda i: r[i])
    lo = max(below, key=lambda i: r[i])
    wh = 1.0 if r[hi] == r[lo] else (target_return - r[lo]) / (r[hi] - r[lo])
    wh = float(wh)
    out = {"high_corner": hi, "low_corner": lo, "weight_high_corner": wh, "weight_low_corner": 1 - wh,
           "std_dev_approx": float(wh * s[hi] + (1 - wh) * s[lo])}
    if corner_weights is not None:
        cw = np.asarray(corner_weights, dtype=float)
        if cw.shape[0] != r.size:
            raise ValueError("one weight list per corner portfolio")
        out["asset_weights"] = _f(wh * cw[hi] + (1 - wh) * cw[lo])
    return out


@formula("portfolio", formula="SFRatio = (E(Rp) - R_L)/sd_p; choose max; P(R < R_L) = N(-SFRatio)",
         curriculum=AA + " (Roy's safety-first criterion)")
def asset_allocation_safety_first(expected_returns: list, std_devs: list, threshold_return: float,
                                  names: list | None = None) -> dict:
    """Roy's safety-first selection among candidate allocations given a shortfall (threshold) return.

    Returns:
        sf_ratios, shortfall_probabilities (normal), best, ranking.

    Example:
        asset_allocation_safety_first(expected_returns=[0.08, 0.10], std_devs=[0.10, 0.16], threshold_return=0.03)
        -> sf_ratios [0.5, 0.4375], best 0
    """
    mu = _vec(expected_returns, "expected_returns")
    sd = _vec(std_devs, "std_devs")
    if mu.size != sd.size or np.any(sd <= 0):
        raise ValueError("expected_returns and std_devs must match in length and std_devs must be > 0")
    sf = (mu - threshold_return) / sd
    labels = list(names) if names is not None else list(range(mu.size))
    order = sorted(range(mu.size), key=lambda i: -sf[i])
    return {"sf_ratios": _f(sf), "shortfall_probabilities": _f(norm.cdf(-sf)), "best": labels[order[0]],
            "ranking": [labels[i] for i in order]}


@formula("portfolio", formula="R_S = R_A - (L0/A0) R_L; var_S = sd_A^2 + (L0/A0)^2 sd_L^2 - 2 (L0/A0) rho sd_A sd_L; "
                              "U = E(R_S) - 0.5 A var_S", curriculum=AA + " (asset/liability management, surplus optimisation)")
def surplus_return_risk(assets: float, liabilities: float, asset_return: float, liability_return: float,
                        asset_std_dev: float = 0.0, liability_std_dev: float = 0.0, correlation: float = 0.0,
                        risk_aversion: float | None = None) -> dict:
    """Surplus return (change in surplus as a fraction of assets) and surplus volatility.

    Args:
        assets, liabilities: current values (A0, L0).
        asset_return, liability_return: expected (or realised) growth rates.
        asset_std_dev, liability_std_dev, correlation: for surplus volatility.
        risk_aversion: optional, for surplus utility.

    Returns:
        surplus_return, surplus_std_dev, surplus_change, beginning_surplus, ending_surplus, funded_ratio_begin,
        funded_ratio_end, surplus_utility (if risk_aversion).

    Example:
        surplus_return_risk(assets=100, liabilities=80, asset_return=0.08, liability_return=0.05)
        -> surplus_return 0.04
    """
    if assets <= 0 or liabilities < 0:
        raise ValueError("assets must be > 0 and liabilities >= 0")
    k = liabilities / assets
    rs = asset_return - k * liability_return
    var = asset_std_dev ** 2 + k ** 2 * liability_std_dev ** 2 - 2 * k * correlation * asset_std_dev * liability_std_dev
    a1, l1 = assets * (1 + asset_return), liabilities * (1 + liability_return)
    out = {"surplus_return": rs, "surplus_std_dev": math.sqrt(max(var, 0.0)), "surplus_change": rs * assets,
           "beginning_surplus": assets - liabilities, "ending_surplus": a1 - l1,
           "funded_ratio_begin": assets / liabilities if liabilities else float("inf"),
           "funded_ratio_end": a1 / l1 if l1 else float("inf")}
    if risk_aversion is not None:
        out["surplus_utility"] = rs - 0.5 * risk_aversion * var
    return out


@formula("portfolio", formula="Pi = lambda Cov w_mkt (+ rf for total returns)", curriculum=AA + " (Black-Litterman, reverse optimisation)")
def black_litterman_implied_returns(market_weights: list, risk_aversion: float, cov: list | None = None,
                                    sds: list | None = None, corr=None, risk_free_rate: float = 0.0) -> dict:
    """Equilibrium (implied) excess returns by reverse optimisation of market-cap weights.

    Args:
        market_weights: market capitalisation weights.
        risk_aversion: lambda; (E(Rm) - rf)/var_m if unknown.
        risk_free_rate: added to report total implied returns.

    Returns:
        implied_excess_returns, implied_total_returns, market_excess_return, market_std_dev.

    Example:
        black_litterman_implied_returns(market_weights=[0.6, 0.4], risk_aversion=3, sds=[0.2, 0.1], corr=0.5)
        -> implied_excess_returns [0.084, 0.030]
    """
    w = _vec(market_weights, "market_weights")
    C = _cov_matrix(w.size, cov, sds, corr)
    pi = risk_aversion * C @ w
    return {"implied_excess_returns": _f(pi), "implied_total_returns": _f(pi + risk_free_rate),
            "market_excess_return": float(w @ pi), "market_std_dev": math.sqrt(float(w @ C @ w))}


@formula("portfolio", formula="mu_BL = [(tau Cov)^-1 + P' Omega^-1 P]^-1 [(tau Cov)^-1 Pi + P' Omega^-1 Q]; "
                              "Omega default diag(P tau Cov P')", curriculum=AA + " (Black-Litterman)")
def black_litterman_posterior(implied_excess_returns: list, view_matrix: list, view_returns: list, tau: float = 0.05,
                              cov: list | None = None, sds: list | None = None, corr=None,
                              view_uncertainty: list | None = None, risk_aversion: float | None = None) -> dict:
    """Black-Litterman posterior expected returns combining equilibrium returns with investor views.

    Args:
        implied_excess_returns: Pi from black_litterman_implied_returns.
        view_matrix: P, one row per view (absolute view = unit vector; relative view = +1/-1).
        view_returns: Q, expected return of each view portfolio.
        tau: scalar on the prior covariance.
        view_uncertainty: Omega (K x K); a K-list is read as its diagonal; default diag(P tau Cov P').
        risk_aversion: optional, to report unconstrained optimal weights Cov^-1 mu / lambda.

    Returns:
        posterior_excess_returns, posterior_covariance (of the mean), optimal_weights (if risk_aversion).

    Example:
        One view that asset 1 returns 0.10 with Omega tiny pulls its posterior to ~0.10.
    """
    pi = _vec(implied_excess_returns, "implied_excess_returns")
    C = _cov_matrix(pi.size, cov, sds, corr)
    P = np.atleast_2d(np.asarray(view_matrix, dtype=float))
    Q = _vec(view_returns, "view_returns")
    if P.shape != (Q.size, pi.size):
        raise ValueError("view_matrix must be K x N with K = len(view_returns), N = number of assets")
    tC = tau * C
    if view_uncertainty is None:
        Om = np.diag(np.diag(P @ tC @ P.T))
    else:
        Om = np.asarray(view_uncertainty, dtype=float)
        Om = np.diag(Om) if Om.ndim == 1 else Om
    tCi = np.linalg.inv(tC)
    Omi = np.linalg.inv(Om)
    M = np.linalg.inv(tCi + P.T @ Omi @ P)
    mu = M @ (tCi @ pi + P.T @ Omi @ Q)
    out = {"posterior_excess_returns": _f(mu), "posterior_covariance": _f(M)}
    if risk_aversion is not None:
        out["optimal_weights"] = _f(np.linalg.solve(risk_aversion * C, mu))
    return out


# ---------------------------------------------------------------- rebalancing
@formula("portfolio", formula="stock exposure = m (A - F); cushion = A - F", curriculum=REB + " (CPPI)")
def cppi_exposure(multiplier: float, assets: float, floor: float) -> dict:
    """Constant-proportion portfolio insurance: equity exposure from the cushion.

    Returns:
        risky_exposure (floored at 0), safe_exposure, cushion, leverage_used (True if exposure > assets).

    Example:
        cppi_exposure(multiplier=2, assets=100, floor=75) -> risky_exposure 50
    """
    if multiplier < 0:
        raise ValueError("multiplier must be >= 0")
    cushion = assets - floor
    e = max(0.0, multiplier * cushion)
    return {"risky_exposure": e, "safe_exposure": assets - e, "cushion": cushion, "leverage_used": e > assets}


@formula("portfolio", formula="buy-and-hold: no trades; constant mix: rebalance to w each period; CPPI: stock = m(A - F)",
         curriculum=REB + " (Perold-Sharpe rebalancing strategies)")
def rebalancing_strategy_paths(stock_prices: list, initial_value: float = 100.0, stock_weight: float = 0.6,
                               cppi_multiplier: float = 2.0, cppi_floor: float | None = None,
                               bill_return: float = 0.0) -> dict:
    """Value paths of buy-and-hold, constant-mix and CPPI strategies over a stock price path.

    Args:
        stock_prices: stock price (or index level) at each rebalancing date, first = start.
        initial_value: starting portfolio value.
        stock_weight: initial (and constant-mix target) stock weight.
        cppi_multiplier, cppi_floor: CPPI parameters (floor defaults to initial_value x (1 - stock_weight/m),
            so that CPPI starts at the same stock exposure).
        bill_return: return on the safe asset per period.

    Returns:
        buy_and_hold, constant_mix, cppi (value lists), plus each strategy's stock holdings in money.

    Example:
        rebalancing_strategy_paths(stock_prices=[100, 90, 100], initial_value=100, stock_weight=0.6)
        -> buy_and_hold ends at 100, constant_mix 100.27 (concave: wins in reversals), cppi 99.33 (convex)
    """
    p = _vec(stock_prices, "stock_prices")
    if p.size < 2 or np.any(p <= 0):
        raise ValueError("need at least two positive stock prices")
    floor = initial_value * (1 - stock_weight / cppi_multiplier) if cppi_floor is None else cppi_floor
    bh_s, bh_b = initial_value * stock_weight, initial_value * (1 - stock_weight)
    cm = initial_value
    cp_s = max(0.0, cppi_multiplier * (initial_value - floor))
    cp_b = initial_value - cp_s
    out = {k: [initial_value] for k in ("buy_and_hold", "constant_mix", "cppi")}
    out.update(buy_and_hold_stock=[bh_s], constant_mix_stock=[cm * stock_weight], cppi_stock=[cp_s])
    for t in range(1, p.size):
        g = p[t] / p[t - 1]
        bh_s, bh_b = bh_s * g, bh_b * (1 + bill_return)
        cm = cm * stock_weight * g + cm * (1 - stock_weight) * (1 + bill_return)
        a = cp_s * g + cp_b * (1 + bill_return)
        cp_s = max(0.0, cppi_multiplier * (a - floor))
        cp_b = a - cp_s
        out["buy_and_hold"].append(float(bh_s + bh_b))
        out["constant_mix"].append(float(cm))
        out["cppi"].append(float(a))
        out["buy_and_hold_stock"].append(float(bh_s))
        out["constant_mix_stock"].append(float(cm * stock_weight))
        out["cppi_stock"].append(float(cp_s))
    out["cppi_floor"] = floor
    return out


@formula("portfolio", formula="rebalance if |w_actual - w_target| > corridor half-width; ad hoc corridor = target x P%",
         curriculum=REB + " (corridor rebalancing)")
def rebalancing_corridor_check(target_weights: list, current_values: list, corridor_widths: list | None = None,
                               corridor_pct_of_target: float | None = None, rebalance_to: str = "target") -> dict:
    """Check each asset class against its rebalancing corridor and size the trades.

    Args:
        target_weights: strategic weights.
        current_values: current market values per asset class.
        corridor_widths: half-widths, e.g. 0.05 for target +/- 5 percentage points; or
        corridor_pct_of_target: ad hoc rule, half-width = target x P (P = 0.10: 45% +/- 4.5%).
        rebalance_to: "target" (any breach -> all classes back to target, as in the curriculum), "edge"
            (breached classes to the corridor edge) or "halfway" (breached classes halfway between target and edge);
            for edge/halfway the residual goes to unbreached classes pro rata to target.

    Returns:
        current_weights, deviations, corridor_widths, breached, new_weights, trades (money; + buy, - sell).

    Example:
        rebalancing_corridor_check(target_weights=[0.45, 0.15, 0.40], current_values=[50, 14, 36],
                                   corridor_pct_of_target=0.10) -> only the first class breaches (49.5% bound)
    """
    tw = _vec(target_weights, "target_weights")
    v = _vec(current_values, "current_values")
    if corridor_widths is not None:
        cw = _vec(corridor_widths, "corridor_widths")
    elif corridor_pct_of_target is not None:
        cw = tw * corridor_pct_of_target
    else:
        raise ValueError("give corridor_widths or corridor_pct_of_target")
    if not (tw.size == v.size == cw.size):
        raise ValueError("target_weights, current_values and corridor widths must have the same length")
    if rebalance_to not in ("target", "edge", "halfway"):
        raise ValueError("rebalance_to must be 'target', 'edge' or 'halfway'")
    total = v.sum()
    w = v / total
    dev = w - tw
    breached = np.abs(dev) > cw + 1e-12
    new_w = w.copy()
    if breached.any():
        if rebalance_to == "target":
            new_w = tw.copy()
        else:
            frac = 1.0 if rebalance_to == "edge" else 0.5
            new_w[breached] = tw[breached] + np.sign(dev[breached]) * cw[breached] * frac
            free = ~breached
            if free.any():
                new_w[free] += (1 - new_w.sum()) * tw[free] / tw[free].sum()
            else:
                new_w = new_w / new_w.sum()
    return {"current_weights": _f(w), "deviations": _f(dev), "corridor_widths": _f(cw), "breached": _f(breached),
            "new_weights": _f(new_w), "trades": _f((new_w - w) * total)}
