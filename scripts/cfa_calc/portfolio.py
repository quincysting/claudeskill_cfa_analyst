"""Portfolio math, CAPM and factor models, active management (CFA L1-L2 Portfolio Management).

Topic ``portfolio``. Inputs are decimals (5% = 0.05). Matrices are nested lists, row-major.
Private helpers here (``_cov_matrix``, ``_vec``) are reused by the other ``portfolio_*`` modules.
"""
from __future__ import annotations

import math

import numpy as np

from ._registry import formula

L1_RR = "L1 PM: Portfolio Risk and Return Part I/II"
L2_PC = "L2 PM: Portfolio Concepts"


# ---------------------------------------------------------------- helpers
def _vec(x, name: str) -> np.ndarray:
    a = np.atleast_1d(np.asarray(x, dtype=float))
    if a.ndim != 1 or a.size == 0:
        raise ValueError(f"{name} must be a non-empty list of numbers")
    return a


def _cov_matrix(n: int, cov=None, sds=None, corr=None) -> np.ndarray:
    """Covariance matrix from `cov`, or from `sds` plus `corr` (a matrix, or a scalar for two assets)."""
    if cov is not None:
        c = np.atleast_2d(np.asarray(cov, dtype=float))
    else:
        if sds is None:
            raise ValueError("give cov, or sds (with corr when more than one asset)")
        s = _vec(sds, "sds")
        if s.size == 1:
            c = np.array([[s[0] ** 2]])
        else:
            if corr is None:
                raise ValueError("corr is required with sds for more than one asset")
            r = np.asarray(corr, dtype=float)
            if r.ndim == 0:
                if s.size != 2:
                    raise ValueError("a scalar corr only works for two assets; pass a correlation matrix")
                r = np.array([[1.0, float(r)], [float(r), 1.0]])
            c = np.outer(s, s) * r
    if c.shape != (n, n):
        raise ValueError(f"covariance matrix must be {n}x{n}, got {c.shape}")
    if not np.allclose(c, c.T, atol=1e-12):
        raise ValueError("covariance/correlation matrix must be symmetric")
    return c


def _f(x):
    """numpy -> plain Python for JSON output."""
    return x.tolist() if hasattr(x, "tolist") else x


# ---------------------------------------------------------------- portfolio math
@formula("portfolio", formula="E(Rp) = sum w_i E(R_i); var_p = w' Cov w; risk contribution_i = w_i (Cov w)_i / sd_p",
         curriculum=L1_RR)
def portfolio_return_risk(weights: list, expected_returns: list | None = None, cov: list | None = None,
                          sds: list | None = None, corr=None) -> dict:
    """Expected return, variance and standard deviation of an N-asset portfolio (two assets included).

    Args:
        weights: asset weights, decimals (normally summing to 1).
        expected_returns: asset expected returns, decimals (optional; omit for risk only).
        cov: covariance matrix (N x N). Or give `sds` and `corr` instead.
        sds: asset standard deviations, decimals.
        corr: correlation matrix, or a single number for two assets.

    Returns:
        expected_return, variance, std_dev, plus marginal_contribution (d sd/d w_i), risk_contribution
        (w_i x marginal, sums to std_dev) and pct_risk_contribution (sums to 1).

    Example:
        portfolio_return_risk(weights=[0.6, 0.4], expected_returns=[0.10, 0.06], sds=[0.20, 0.10], corr=0.5)
        -> expected_return 0.084, std_dev 0.1442
    """
    w = _vec(weights, "weights")
    c = _cov_matrix(w.size, cov, sds, corr)
    var = float(w @ c @ w)
    sd = math.sqrt(max(var, 0.0))
    out = {}
    if expected_returns is not None:
        mu = _vec(expected_returns, "expected_returns")
        if mu.size != w.size:
            raise ValueError("weights and expected_returns must have the same length")
        out["expected_return"] = float(w @ mu)
    marginal = (c @ w) / sd if sd > 0 else np.zeros_like(w)
    out.update(variance=var, std_dev=sd, marginal_contribution=_f(marginal),
               risk_contribution=_f(w * marginal),
               pct_risk_contribution=_f(w * marginal / sd if sd > 0 else np.zeros_like(w)))
    return out


@formula("portfolio", formula="w_gmv = Cov^-1 1 / (1' Cov^-1 1); two assets: w1 = (s2^2 - s12)/(s1^2 + s2^2 - 2 s12)",
         curriculum=L1_RR + "; " + L2_PC)
def global_minimum_variance_portfolio(cov: list | None = None, sds: list | None = None, corr=None,
                                      expected_returns: list | None = None) -> dict:
    """Unconstrained (short sales allowed) global minimum-variance weights; covers the two-asset case.

    Args:
        cov: covariance matrix, or `sds` + `corr` (scalar corr for two assets).
        expected_returns: optional, to report the GMV expected return.

    Returns:
        weights, std_dev, variance, expected_return (if expected_returns given).

    Example:
        global_minimum_variance_portfolio(sds=[0.20, 0.10], corr=0.5) -> weights [0.0, 1.0]
    """
    n = len(sds) if cov is None else len(cov)
    c = _cov_matrix(n, cov, sds, corr)
    ones = np.ones(n)
    x = np.linalg.solve(c, ones)
    w = x / x.sum()
    var = float(w @ c @ w)
    out = {"weights": _f(w), "std_dev": math.sqrt(var), "variance": var}
    if expected_returns is not None:
        out["expected_return"] = float(w @ _vec(expected_returns, "expected_returns"))
    return out


@formula("portfolio", formula="w_tan = Cov^-1 (mu - rf) / 1' Cov^-1 (mu - rf); Sharpe = (E(Rt) - rf)/sd_t",
         curriculum=L1_RR + "; " + L2_PC)
def tangency_portfolio(expected_returns: list, risk_free_rate: float, cov: list | None = None,
                       sds: list | None = None, corr=None) -> dict:
    """Unconstrained tangency (maximum Sharpe ratio) portfolio of risky assets given a risk-free rate.

    Args:
        expected_returns: asset expected returns, decimals.
        risk_free_rate: decimal.
        cov: covariance matrix, or `sds` + `corr`.

    Returns:
        weights, expected_return, std_dev, sharpe_ratio.

    Example:
        tangency_portfolio(expected_returns=[0.10, 0.06], risk_free_rate=0.02, sds=[0.20, 0.10], corr=0.5)
        -> weights [0.3333, 0.6667], sharpe 0.4619
    """
    mu = _vec(expected_returns, "expected_returns")
    c = _cov_matrix(mu.size, cov, sds, corr)
    x = np.linalg.solve(c, mu - risk_free_rate)
    if abs(x.sum()) < 1e-15:
        raise ValueError("tangency portfolio undefined: excess returns are orthogonal to the GMV direction")
    w = x / x.sum()
    er = float(w @ mu)
    sd = math.sqrt(float(w @ c @ w))
    return {"weights": _f(w), "expected_return": er, "std_dev": sd, "sharpe_ratio": (er - risk_free_rate) / sd}


@formula("portfolio", formula="U = E(R) - 0.5 A sd^2 (decimals; = E(R%) - 0.005 A sd%^2 in percent)",
         curriculum=L1_RR + "; L3 Asset Allocation")
def utility_ranking(expected_returns: list, std_devs: list, risk_aversion: float, names: list | None = None) -> dict:
    """Mean-variance utility of one or more portfolios and the one with the highest utility.

    Args:
        expected_returns: decimals, one per portfolio (a single number is fine).
        std_devs: decimals, one per portfolio.
        risk_aversion: A (the curriculum's lambda / R_A), e.g. 4.
        names: optional labels.

    Returns:
        utilities, best (label or index), best_utility, ranking (best first).

    Example:
        utility_ranking(expected_returns=[0.10], std_devs=[0.20], risk_aversion=4) -> utilities [0.02]
    """
    mu = _vec(expected_returns, "expected_returns")
    sd = _vec(std_devs, "std_devs")
    if mu.size != sd.size:
        raise ValueError("expected_returns and std_devs must have the same length")
    if risk_aversion < 0:
        raise ValueError("risk_aversion must be >= 0")
    u = mu - 0.5 * risk_aversion * sd ** 2
    labels = list(names) if names is not None else list(range(mu.size))
    order = sorted(range(mu.size), key=lambda i: -u[i])
    return {"utilities": _f(u), "best": labels[order[0]], "best_utility": float(u[order[0]]),
            "ranking": [labels[i] for i in order]}


@formula("portfolio", formula="y* = (E(Rp) - rf)/(A sd_p^2); E(Rc) = rf + y*(E(Rp) - rf); sd_c = y* sd_p",
         curriculum=L1_RR)
def optimal_risky_allocation(expected_return_risky: float, risk_free_rate: float, std_dev_risky: float,
                             risk_aversion: float) -> dict:
    """Utility-maximising weight in the risky portfolio on the capital allocation line.

    Returns:
        weight_risky (y*), weight_risk_free, expected_return, std_dev, utility.

    Example:
        optimal_risky_allocation(expected_return_risky=0.10, risk_free_rate=0.04, std_dev_risky=0.20,
                                 risk_aversion=3) -> weight_risky 0.5
    """
    if std_dev_risky <= 0 or risk_aversion <= 0:
        raise ValueError("std_dev_risky and risk_aversion must be > 0")
    y = (expected_return_risky - risk_free_rate) / (risk_aversion * std_dev_risky ** 2)
    er = risk_free_rate + y * (expected_return_risky - risk_free_rate)
    sd = abs(y) * std_dev_risky
    return {"weight_risky": y, "weight_risk_free": 1 - y, "expected_return": er, "std_dev": sd,
            "utility": er - 0.5 * risk_aversion * sd ** 2}


@formula("portfolio", formula="E(Rc) = rf + (E(Rp) - rf)/sd_p x sd_c; sd_c = w sd_p (CML when p = market)",
         curriculum=L1_RR + " (CAL / CML)")
def capital_allocation_line(risk_free_rate: float, expected_return_risky: float, std_dev_risky: float,
                            weight_risky: float | None = None, target_std_dev: float | None = None) -> dict:
    """Point on the capital allocation line (capital market line if the risky portfolio is the market).

    Give either `weight_risky` (w > 1 means borrowing) or `target_std_dev`.

    Returns:
        expected_return, std_dev, weight_risky, slope (Sharpe ratio of the risky portfolio).

    Example:
        capital_allocation_line(risk_free_rate=0.04, expected_return_risky=0.10, std_dev_risky=0.20,
                                weight_risky=0.5) -> expected_return 0.07, std_dev 0.10
    """
    if std_dev_risky <= 0:
        raise ValueError("std_dev_risky must be > 0")
    if (weight_risky is None) == (target_std_dev is None):
        raise ValueError("give exactly one of weight_risky or target_std_dev")
    w = weight_risky if weight_risky is not None else target_std_dev / std_dev_risky
    slope = (expected_return_risky - risk_free_rate) / std_dev_risky
    return {"expected_return": risk_free_rate + w * (expected_return_risky - risk_free_rate),
            "std_dev": abs(w) * std_dev_risky, "weight_risky": w, "slope": slope}


# ---------------------------------------------------------------- CAPM and factor models
@formula("portfolio", formula="E(Ri) = rf + beta (E(Rm) - rf); alpha = forecast - E(Ri)", curriculum=L1_RR + " (CAPM, SML)")
def capm_sml_expected_return(risk_free_rate: float, beta: float, market_return: float,
                             forecast_return: float | None = None) -> dict:
    """CAPM required (expected) return from the security market line, and alpha versus a forecast.

    Args:
        market_return: expected market return E(Rm), decimal.
        forecast_return: analyst's forecast (or realised) return; alpha > 0 means undervalued (plots above SML).

    Returns:
        expected_return, market_risk_premium, alpha and verdict (if forecast_return given).

    Example:
        capm_sml_expected_return(risk_free_rate=0.05, beta=1.2, market_return=0.11) -> expected_return 0.122
    """
    er = risk_free_rate + beta * (market_return - risk_free_rate)
    out = {"expected_return": er, "market_risk_premium": market_return - risk_free_rate}
    if forecast_return is not None:
        a = forecast_return - er
        out["alpha"] = a
        out["verdict"] = "undervalued (above SML)" if a > 1e-12 else "overvalued (below SML)" if a < -1e-12 else "fairly valued"
    return out


@formula("portfolio", formula="beta = Cov(i,m)/Var(m) = rho(i,m) sd_i / sd_m; adjusted beta = 1/3 + 2/3 x beta",
         curriculum=L1_RR + "; " + L2_PC + " (adjusted beta)")
def capm_beta_from_moments(cov_with_market: float | None = None, market_variance: float | None = None,
                           correlation: float | None = None, std_dev_asset: float | None = None,
                           std_dev_market: float | None = None, historical_beta: float | None = None) -> dict:
    """Beta from covariance and market variance, or from correlation and standard deviations; plus adjusted beta.

    Give `historical_beta` alone to get only the (Blume) adjusted beta 0.333 + 0.667 x beta.

    Returns:
        beta, adjusted_beta, cov_with_market (when computed).

    Example:
        capm_beta_from_moments(correlation=0.5, std_dev_asset=0.30, std_dev_market=0.20) -> beta 0.75
    """
    if historical_beta is not None:
        return {"beta": historical_beta, "adjusted_beta": 1 / 3 + 2 / 3 * historical_beta}
    if cov_with_market is not None:
        if market_variance is None and std_dev_market is None:
            raise ValueError("give market_variance or std_dev_market with cov_with_market")
        mv = market_variance if market_variance is not None else std_dev_market ** 2
        cov = cov_with_market
    elif None not in (correlation, std_dev_asset, std_dev_market):
        mv = std_dev_market ** 2
        cov = correlation * std_dev_asset * std_dev_market
    else:
        raise ValueError("give cov_with_market + market_variance, or correlation + std_dev_asset + std_dev_market")
    if mv <= 0:
        raise ValueError("market variance must be > 0")
    b = cov / mv
    return {"beta": b, "adjusted_beta": 1 / 3 + 2 / 3 * b, "cov_with_market": cov}


@formula("portfolio", formula="sd_i^2 = beta^2 sd_m^2 + sd_e^2; R^2 = systematic / total", curriculum=L1_RR)
def systematic_risk_decomposition(beta: float, std_dev_market: float, total_std_dev: float | None = None,
                                  residual_std_dev: float | None = None) -> dict:
    """Split total variance into systematic (market) and nonsystematic (residual) parts.

    Give `total_std_dev` or `residual_std_dev`.

    Returns:
        systematic_variance, nonsystematic_variance, total_variance, total_std_dev, systematic_std_dev,
        r_squared (share of variance explained by the market).

    Example:
        systematic_risk_decomposition(beta=1.2, std_dev_market=0.20, total_std_dev=0.30) -> r_squared 0.64
    """
    sysv = beta ** 2 * std_dev_market ** 2
    if total_std_dev is not None:
        tot = total_std_dev ** 2
        res = tot - sysv
        if res < -1e-12:
            raise ValueError("systematic variance exceeds total variance; check inputs")
    elif residual_std_dev is not None:
        res = residual_std_dev ** 2
        tot = sysv + res
    else:
        raise ValueError("give total_std_dev or residual_std_dev")
    return {"systematic_variance": sysv, "nonsystematic_variance": res, "total_variance": tot,
            "total_std_dev": math.sqrt(tot), "systematic_std_dev": math.sqrt(sysv),
            "r_squared": sysv / tot if tot > 0 else float("nan")}


@formula("portfolio", formula="E(R) = rf + sum b_k lambda_k", curriculum=L2_PC + " (APT / multifactor models)")
def apt_expected_return(risk_free_rate: float, factor_sensitivities: list, factor_risk_premiums: list) -> dict:
    """Arbitrage pricing theory / multifactor expected return.

    Args:
        factor_sensitivities: b_k for each factor.
        factor_risk_premiums: lambda_k, the risk premium per unit of each factor, decimals.

    Returns:
        expected_return, factor_contributions.

    Example:
        apt_expected_return(risk_free_rate=0.04, factor_sensitivities=[1.2, 0.5],
                            factor_risk_premiums=[0.05, 0.02]) -> expected_return 0.11
    """
    b = _vec(factor_sensitivities, "factor_sensitivities")
    lam = _vec(factor_risk_premiums, "factor_risk_premiums")
    if b.size != lam.size:
        raise ValueError("factor_sensitivities and factor_risk_premiums must have the same length")
    contrib = b * lam
    return {"expected_return": risk_free_rate + float(contrib.sum()), "factor_contributions": _f(contrib)}


@formula("portfolio", formula="choose w (sum w = 1) so that sum w_j b_j = b_target; profit = E(R_target) - sum w_j E(R_j)",
         curriculum=L2_PC + " (arbitrage opportunities)")
def apt_arbitrage_check(target_expected_return: float, target_sensitivities: list,
                        replicating_expected_returns: list, replicating_sensitivities: list) -> dict:
    """Replicate a portfolio's factor exposures with other portfolios and measure the arbitrage profit.

    Args:
        target_expected_return: E(R) of the portfolio being tested.
        target_sensitivities: its factor sensitivities, one per factor.
        replicating_expected_returns: E(R) of each replicating portfolio (factors + 1 portfolios for an exact fit).
        replicating_sensitivities: one list of factor sensitivities per replicating portfolio.

    Returns:
        weights (of the replicating portfolios), replicating_expected_return, arbitrage_return
        (target minus replica, per unit invested), action, residual_exposure.

    Example:
        One factor: A (E 0.10, b 1.0), B (E 0.15, b 2.0); C (E 0.12, b 1.5) -> replica 0.125, buy replica, sell C.
    """
    mu = _vec(replicating_expected_returns, "replicating_expected_returns")
    B = np.atleast_2d(np.asarray(replicating_sensitivities, dtype=float))
    if B.shape[0] != mu.size:
        raise ValueError("need one sensitivity list per replicating portfolio")
    bt = _vec(target_sensitivities, "target_sensitivities")
    if B.shape[1] != bt.size:
        raise ValueError("sensitivity lists must match the number of factors in target_sensitivities")
    A = np.vstack([np.ones(mu.size), B.T])          # rows: budget, then each factor
    rhs = np.concatenate([[1.0], bt])
    w, *_ = np.linalg.lstsq(A, rhs, rcond=None)
    rep = float(w @ mu)
    profit = target_expected_return - rep
    action = ("no arbitrage" if abs(profit) < 1e-12 else
              "buy target, sell replica" if profit > 0 else "sell target, buy replica")
    return {"weights": _f(w), "replicating_expected_return": rep, "arbitrage_return": profit, "action": action,
            "residual_exposure": _f(A @ w - rhs)}


@formula("portfolio", formula="solve E(R_j) = rf + sum_k lambda_k b_jk for rf and lambda (K + 1 portfolios)",
         curriculum=L2_PC + " (APT: estimating factor risk premiums)")
def apt_solve_factor_premiums(expected_returns: list, sensitivities: list) -> dict:
    """Back out the risk-free rate and factor risk premiums implied by well-diversified portfolios under APT.

    Args:
        expected_returns: E(R) of K + 1 portfolios (more portfolios: least-squares fit).
        sensitivities: one list of K factor sensitivities per portfolio.

    Returns:
        risk_free_rate, factor_risk_premiums, residuals (non-zero residuals flag arbitrage opportunities).

    Example:
        Portfolios J (0.14; 1.0, 1.5), K (0.12; 0.5, 1.0), L (0.11; 1.3, 1.1) -> rf 0.07, premiums [-0.02, 0.06]
    """
    mu = _vec(expected_returns, "expected_returns")
    B = np.atleast_2d(np.asarray(sensitivities, dtype=float))
    if B.shape[0] != mu.size:
        raise ValueError("one sensitivity list per portfolio")
    X = np.hstack([np.ones((mu.size, 1)), B])
    if mu.size < X.shape[1]:
        raise ValueError("need at least (number of factors + 1) portfolios")
    coef, *_ = np.linalg.lstsq(X, mu, rcond=None)
    return {"risk_free_rate": float(coef[0]), "factor_risk_premiums": _f(coef[1:]), "residuals": _f(mu - X @ coef)}


@formula("portfolio", formula="active return = sum (b_P - b_B)_k F_k + asset selection; active risk^2 = b_a' Cov_F b_a + "
                              "active specific variance; FMCAR_k = b_a,k (Cov_F b_a)_k / active risk^2",
         curriculum=L2_PC + " (multifactor models: active return and active risk)")
def factor_active_risk_return(portfolio_sensitivities: list, benchmark_sensitivities: list,
                              factor_returns: list | None = None, active_return: float | None = None,
                              factor_covariance: list | None = None,
                              active_specific_variance: float | None = None) -> dict:
    """Decompose active return into factor tilts and asset selection, and active risk into factor and specific parts.

    Args:
        portfolio_sensitivities, benchmark_sensitivities: factor exposures of portfolio and benchmark.
        factor_returns: realised factor returns (for the return attribution).
        active_return: portfolio minus benchmark return (asset selection = the unexplained remainder).
        factor_covariance: K x K factor covariance (a K-list = uncorrelated factor variances).
        active_specific_variance: active specific risk squared.

    Returns:
        active_exposures; factor_contributions, factor_tilt_return, asset_selection (return part);
        active_factor_variance, active_risk, active_risk_squared, factor_marginal_contributions (FMCAR),
        specific_share (risk part). Units follow the inputs (decimals or percent^2).

    Example:
        Active exposures [0.50, 0.15], factor cov [[225, 12], [12, 144]], specific 53.71 -> active risk^2 115,
        FMCAR [0.497, 0.036]
    """
    bp = _vec(portfolio_sensitivities, "portfolio_sensitivities")
    bb = _vec(benchmark_sensitivities, "benchmark_sensitivities")
    if bp.size != bb.size:
        raise ValueError("portfolio and benchmark need the same number of factors")
    ba = bp - bb
    out = {"active_exposures": _f(ba)}
    if factor_returns is not None:
        f = _vec(factor_returns, "factor_returns")
        if f.size != ba.size:
            raise ValueError("one factor return per factor")
        c = ba * f
        out.update(factor_contributions=_f(c), factor_tilt_return=float(c.sum()))
        if active_return is not None:
            out["asset_selection"] = active_return - float(c.sum())
    if factor_covariance is not None:
        F = np.asarray(factor_covariance, dtype=float)
        F = np.diag(F) if F.ndim == 1 else F
        if F.shape != (ba.size, ba.size):
            raise ValueError("factor_covariance must be K x K")
        fv = float(ba @ F @ ba)
        tot = fv + (active_specific_variance or 0.0)
        out.update(active_factor_variance=fv, active_risk_squared=tot, active_risk=math.sqrt(tot),
                   factor_marginal_contributions=_f(ba * (F @ ba) / tot) if tot else _f(np.zeros_like(ba)),
                   specific_share=(active_specific_variance or 0.0) / tot if tot else 0.0)
    return out


# ---------------------------------------------------------------- active management
@formula("portfolio", formula="IR = TC x IC x sqrt(BR); sd_A* = IR/SR_B x sd_B; SR_P^2 = SR_B^2 + IR^2; "
                              "omega* = IR/(2 lambda_R); VA* = IR^2/(4 lambda_R)",
         curriculum="L2 PM: The Fundamental Law of Active Management; Residual Risk and Return: The Information Ratio")
def fundamental_law_active_management(information_coefficient: float | None = None, breadth: float | None = None,
                                      transfer_coefficient: float = 1.0, information_ratio: float | None = None,
                                      benchmark_sharpe: float | None = None, benchmark_std_dev: float | None = None,
                                      active_risk: float | None = None,
                                      residual_risk_aversion: float | None = None) -> dict:
    """Grinold's fundamental law and optimal active (residual) risk.

    Args:
        information_coefficient, breadth: IC and BR (independent bets per year); or give information_ratio directly.
        transfer_coefficient: TC (1 = unconstrained).
        benchmark_sharpe, benchmark_std_dev: for optimal active risk IR/SR_B x sd_B and the combined Sharpe ratio.
        active_risk: tracking error, to convert IR into expected active return (alpha = IR x omega).
        residual_risk_aversion: lambda_R in the curriculum's percent units (e.g. 0.10); gives
            omega* = IR/(2 lambda_R) and VA* = IR^2/(4 lambda_R), returned as decimals.

    Returns:
        information_ratio, plus max_portfolio_sharpe, optimal_active_risk, expected_active_return,
        optimal_residual_risk, optimal_value_added when inputs allow.

    Example:
        fundamental_law_active_management(information_coefficient=0.0577, breadth=400) -> information_ratio 1.154
    """
    if information_ratio is None:
        if information_coefficient is None or breadth is None:
            raise ValueError("give information_coefficient and breadth, or information_ratio")
        if breadth < 0:
            raise ValueError("breadth must be >= 0")
        information_ratio = transfer_coefficient * information_coefficient * math.sqrt(breadth)
    ir = information_ratio
    out = {"information_ratio": ir}
    if benchmark_sharpe is not None:
        out["max_portfolio_sharpe"] = math.sqrt(benchmark_sharpe ** 2 + ir ** 2)
        if benchmark_std_dev is not None and benchmark_sharpe != 0:
            out["optimal_active_risk"] = ir / benchmark_sharpe * benchmark_std_dev
    if active_risk is not None:
        out["expected_active_return"] = ir * active_risk
    if residual_risk_aversion is not None:
        if residual_risk_aversion <= 0:
            raise ValueError("residual_risk_aversion must be > 0")
        out["optimal_residual_risk"] = ir / (2 * residual_risk_aversion) / 100
        out["optimal_value_added"] = ir ** 2 / (4 * residual_risk_aversion) / 100
    return out


@formula("portfolio", formula="w_i ∝ alpha_i/sd_e_i^2; w0 = (alpha_A/sd_eA^2)/((E(Rm)-rf)/sd_m^2); "
                              "w* = w0/(1 + (1 - beta_A) w0); SR_P^2 = SR_M^2 + (alpha_A/sd_eA)^2",
         curriculum="L2 PM 2013: The Theory of Active Portfolio Management (Treynor-Black)")
def treynor_black_active_weights(alphas: list, residual_std_devs: list, betas: list, market_risk_premium: float,
                                 market_std_dev: float) -> dict:
    """Treynor-Black model: active portfolio weights, its weight in the risky portfolio, and the Sharpe gain.

    Args:
        alphas: forecast alphas of the analysed securities, decimals.
        residual_std_devs: their residual (nonsystematic) standard deviations.
        betas: their betas.
        market_risk_premium: E(Rm) - rf.
        market_std_dev: sd of the market (passive) portfolio.

    Returns:
        active_weights, active_alpha, active_beta, active_residual_std_dev, appraisal_ratio,
        initial_active_weight (w0), active_weight (w*, beta-adjusted), passive_weight, market_sharpe, optimal_sharpe.

    Example:
        Single stock alpha 0.02, residual sd 0.30, beta 1, market premium 0.08, market sd 0.20 -> w* = w0 = 0.1111
    """
    a = _vec(alphas, "alphas")
    se = _vec(residual_std_devs, "residual_std_devs")
    b = _vec(betas, "betas")
    if not (a.size == se.size == b.size):
        raise ValueError("alphas, residual_std_devs and betas must have the same length")
    if np.any(se <= 0) or market_std_dev <= 0:
        raise ValueError("standard deviations must be > 0")
    raw = a / se ** 2
    wa = raw / raw.sum()
    alpha_a = float(wa @ a)
    beta_a = float(wa @ b)
    var_e = float(wa ** 2 @ se ** 2)
    w0 = (alpha_a / var_e) / (market_risk_premium / market_std_dev ** 2)
    wstar = w0 / (1 + (1 - beta_a) * w0)
    sr_m = market_risk_premium / market_std_dev
    ar = alpha_a / math.sqrt(var_e)
    return {"active_weights": _f(wa), "active_alpha": alpha_a, "active_beta": beta_a,
            "active_residual_std_dev": math.sqrt(var_e), "appraisal_ratio": ar, "initial_active_weight": w0,
            "active_weight": wstar, "passive_weight": 1 - wstar, "market_sharpe": sr_m,
            "optimal_sharpe": math.sqrt(sr_m ** 2 + ar ** 2)}


@formula("portfolio", formula="add asset if (E(R_new) - rf)/sd_new > Sharpe_p x corr(new, p)",
         curriculum=L2_PC + "; L3 Asset Allocation")
def add_asset_class_test(sharpe_new: float, sharpe_portfolio: float, correlation: float) -> dict:
    """Does adding a small position in a new asset raise the portfolio Sharpe ratio?

    Returns:
        add (bool), hurdle_sharpe (= Sharpe_p x corr), margin.

    Example:
        add_asset_class_test(sharpe_new=0.25, sharpe_portfolio=0.40, correlation=0.5) -> add True (hurdle 0.20)
    """
    hurdle = sharpe_portfolio * correlation
    return {"add": sharpe_new > hurdle, "hurdle_sharpe": hurdle, "margin": sharpe_new - hurdle}
