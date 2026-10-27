"""Distributions, estimation, hypothesis tests and seeded Monte Carlo (L1 Quant R9-R11, L2 Quant R11).

Every test returns the statistic, degrees of freedom, critical value(s), p-value and decision.
`tail` is "two" (H_a: parameter != value), "upper" (H_a: >) or "lower" (H_a: <).
Distribution functions come from scipy.stats, so critical values are exact rather than table-rounded.
"""
from __future__ import annotations

import math

import numpy as np
from scipy import stats as sps

from ._registry import formula

_R9 = "L1 Quant R9: Common Probability Distributions"
_R10 = "L1 Quant R10: Sampling and Estimation"
_R11 = "L1 Quant R11: Hypothesis Testing"
_TAILS = ("two", "upper", "lower")


def _num(name: str, x) -> float:
    try:
        v = float(x)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be a number, got {x!r}") from None
    if not math.isfinite(v):
        raise ValueError(f"{name} must be finite, got {x!r}")
    return v


def _pos(name: str, x) -> float:
    v = _num(name, x)
    if v <= 0:
        raise ValueError(f"{name} must be positive, got {v}")
    return v


def _n(name: str, n, minimum: int = 2) -> int:
    if int(n) != n or n < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}, got {n!r}")
    return int(n)


def _vec(name: str, xs, min_len: int = 1) -> np.ndarray:
    a = np.array([_num(name, x) for x in xs], dtype=float)
    if a.size < min_len:
        raise ValueError(f"{name} needs at least {min_len} values")
    return a


def _decide(stat: float, dist, alpha: float, tail: str, df=None, **extra) -> dict:
    alpha = _num("alpha", alpha)
    if not 0 < alpha < 1:
        raise ValueError("alpha must be between 0 and 1")
    if tail not in _TAILS:
        raise ValueError(f"tail must be one of {_TAILS}")
    if tail == "two":
        crit = [float(dist.ppf(alpha / 2)), float(dist.ppf(1 - alpha / 2))]
        p = min(1.0, 2 * min(float(dist.cdf(stat)), float(dist.sf(stat))))
        reject = stat < crit[0] or stat > crit[1]
    elif tail == "upper":
        crit = [float(dist.ppf(1 - alpha))]
        p, reject = float(dist.sf(stat)), stat > crit[0]
    else:
        crit = [float(dist.ppf(alpha))]
        p, reject = float(dist.cdf(stat)), stat < crit[0]
    out = {"statistic": float(stat), "df": df, "critical_values": crit, "p_value": p, "reject_null": bool(reject),
           "decision": f"{'reject' if reject else 'fail to reject'} H0 at the {alpha:g} level ({tail}-tailed)"}
    out.update(extra)
    return out


def _variance(var, sd, label: str) -> float:
    if (var is None) == (sd is None):
        raise ValueError(f"give exactly one of {label}_variance or {label}_std_dev")
    return _pos(f"{label}_variance", var) if var is not None else _pos(f"{label}_std_dev", sd) ** 2


# ---------------------------------------------------------------- distributions

@formula("quant", formula="z = (x - mu)/sigma; P(a < X < b) = N(z_b) - N(z_a)", curriculum=_R9)
def normal_probability(mean: float = 0.0, std_dev: float = 1.0, lower: float | None = None,
                       upper: float | None = None) -> dict:
    """Probability that a normal variable falls between lower and upper (either may be omitted = infinite).

    Args:
        mean, std_dev: parameters of the normal distribution.
        lower: lower bound (None = minus infinity).
        upper: upper bound (None = plus infinity).

    Returns:
        probability, z_lower, z_upper.

    Example:
        normal_probability(mean=0.12, std_dev=0.22, lower=0.20) -> 0.3581 (CFA L1 R9 Example 8; table 0.3594 at z = 0.36)
    """
    m, s = _num("mean", mean), _pos("std_dev", std_dev)
    zl = None if lower is None else (_num("lower", lower) - m) / s
    zu = None if upper is None else (_num("upper", upper) - m) / s
    p = (1.0 if zu is None else float(sps.norm.cdf(zu))) - (0.0 if zl is None else float(sps.norm.cdf(zl)))
    return {"probability": max(p, 0.0), "z_lower": zl, "z_upper": zu}


@formula("quant", formula="x = mu + z_p sigma, z_p = N^-1(p)", curriculum=_R9)
def normal_quantile(p: float, mean: float = 0.0, std_dev: float = 1.0) -> dict:
    """Inverse normal: the value with probability p at or below it.

    Args:
        p: cumulative probability, decimal in (0, 1).
        mean, std_dev: parameters of the normal distribution.

    Returns:
        value, z.

    Example:
        normal_quantile(0.95) -> z 1.645; normal_quantile(0.99) -> 2.326 (CFA L1 R9 Section 3.2)
    """
    p = _num("p", p)
    if not 0 < p < 1:
        raise ValueError("p must be strictly between 0 and 1")
    z = float(sps.norm.ppf(p))
    return {"value": _num("mean", mean) + z * _pos("std_dev", std_dev), "z": z}


@formula("quant", formula="mean_L = exp(mu + sigma^2/2); var_L = exp(2mu + sigma^2)[exp(sigma^2) - 1]",
         curriculum=_R9)
def lognormal_moments(mu: float, sigma: float) -> dict:
    """Mean, variance, median and mode of a lognormal variable Y = exp(X), X ~ N(mu, sigma^2).

    Args:
        mu: mean of ln Y (e.g. expected continuously compounded return times horizon).
        sigma: standard deviation of ln Y.

    Returns:
        mean, variance, std_dev, median, mode.

    Example:
        lognormal_moments(mu=0, sigma=1) -> mean e^0.5 = 1.6487
    """
    m, s = _num("mu", mu), _pos("sigma", sigma)
    var = math.exp(2 * m + s * s) * math.expm1(s * s)
    return {"mean": math.exp(m + s * s / 2), "variance": var, "std_dev": math.sqrt(var), "median": math.exp(m),
            "mode": math.exp(m - s * s)}


@formula("quant", formula="r = ln(1 + R) = ln(S1/S0); R = e^r - 1", curriculum=_R9)
def continuously_compounded_return(holding_period_return: float | None = None, cc_return: float | None = None,
                                   begin_price: float | None = None, end_price: float | None = None) -> dict:
    """Convert between holding-period and continuously compounded returns.

    Give one of holding_period_return, cc_return, or begin_price with end_price.

    Returns:
        cc_return, holding_period_return.

    Example:
        continuously_compounded_return(begin_price=30, end_price=34.5) -> cc_return 0.139762 (CFA L1 R9 Section 3.4)
    """
    given = [holding_period_return is not None, cc_return is not None, begin_price is not None or end_price is not None]
    if sum(given) != 1:
        raise ValueError("give exactly one of holding_period_return, cc_return, or begin_price and end_price")
    if cc_return is not None:
        r = _num("cc_return", cc_return)
        return {"cc_return": r, "holding_period_return": math.expm1(r)}
    if holding_period_return is not None:
        hpr = _num("holding_period_return", holding_period_return)
    else:
        hpr = _pos("end_price", end_price) / _pos("begin_price", begin_price) - 1
    if hpr <= -1:
        raise ValueError("holding period return must be greater than -100%")
    return {"cc_return": math.log1p(hpr), "holding_period_return": hpr}


@formula("quant", formula="r_t = ln(S_t/S_(t-1)); sigma_annual = s(r_t) sqrt(periods per year)", curriculum=_R9)
def volatility_from_prices(prices: list, periods_per_year: float = 250) -> dict:
    """Historical volatility: sample standard deviation of continuously compounded returns, annualized.

    Args:
        prices: price series in time order (at least 3 prices).
        periods_per_year: e.g. 250 trading days, 52 weeks, 12 months.

    Returns:
        annualized_volatility, period_std_dev, period_mean, annualized_mean, cc_returns.

    Example:
        volatility_from_prices([25.20, 25.21, 25.52, 26.10, 26.14]) -> period_std_dev 0.010354,
        annualized 0.163711 (CFA L1 R9 Example 10)
    """
    p = _vec("prices", prices, 3)
    if np.any(p <= 0):
        raise ValueError("prices must be positive")
    r = np.diff(np.log(p))
    s, m = float(r.std(ddof=1)), float(r.mean())
    k = _pos("periods_per_year", periods_per_year)
    return {"annualized_volatility": s * math.sqrt(k), "period_std_dev": s, "period_mean": m,
            "annualized_mean": m * k, "cc_returns": r.tolist()}


# ---------------------------------------------------------------- estimation

@formula("quant", formula="SE = s / sqrt(n)", curriculum=_R10)
def standard_error_of_mean(std_dev: float, n: int) -> dict:
    """Standard error of the sample mean (central limit theorem).

    Args:
        std_dev: population or sample standard deviation.
        n: sample size.

    Returns:
        standard_error.

    Example:
        standard_error_of_mean(std_dev=0.30, n=100) -> 0.03 (CFA L1 R10 Example 4)
    """
    return {"standard_error": _pos("std_dev", std_dev) / math.sqrt(_n("n", n, 1))}


@formula("quant", formula="Xbar +/- z_(a/2) sigma/sqrt(n)  or  Xbar +/- t_(a/2, n-1) s/sqrt(n)", curriculum=_R10)
def confidence_interval_mean(sample_mean: float, std_dev: float, n: int, confidence: float = 0.95,
                             distribution: str = "t") -> dict:
    """Confidence interval for a population mean.

    Use "z" when the population variance is known (or n is large and you follow the z convention),
    "t" when it is estimated from the sample (the theoretically correct choice).

    Args:
        sample_mean: point estimate.
        std_dev: population (z) or sample (t) standard deviation.
        n: sample size.
        confidence: e.g. 0.90, 0.95, 0.99.
        distribution: "t" or "z".

    Returns:
        lower, upper, margin, reliability_factor, standard_error, df.

    Example:
        confidence_interval_mean(sample_mean=5.5, std_dev=10, n=10) -> -1.65 to 12.65 (CFA L1 R10 Example 6)
    """
    m, s, n = _num("sample_mean", sample_mean), _pos("std_dev", std_dev), _n("n", n)
    c = _num("confidence", confidence)
    if not 0 < c < 1:
        raise ValueError("confidence must be between 0 and 1")
    if distribution not in ("t", "z"):
        raise ValueError('distribution must be "t" or "z"')
    df = n - 1 if distribution == "t" else None
    rf = float(sps.t.ppf((1 + c) / 2, df)) if df else float(sps.norm.ppf((1 + c) / 2))
    se = s / math.sqrt(n)
    return {"lower": m - rf * se, "upper": m + rf * se, "margin": rf * se, "reliability_factor": rf,
            "standard_error": se, "df": df}


# ---------------------------------------------------------------- hypothesis tests

@formula("quant", formula="z = (Xbar - mu0)/(sigma/sqrt(n))", curriculum=_R11)
def z_test_mean(sample_mean: float, hypothesized_mean: float, std_dev: float | None = None, n: int | None = None,
                standard_error: float | None = None, alpha: float = 0.05, tail: str = "two") -> dict:
    """z-test of a single mean (known variance, or large sample using s).

    Give std_dev and n, or the standard_error directly.

    Returns:
        statistic, df (None), critical_values, p_value, reject_null, decision, standard_error.

    Example:
        z_test_mean(sample_mean=0.39, hypothesized_mean=0, standard_error=0.1336) -> z 2.92 (CFA L1 R11 Example 3)
    """
    if standard_error is None:
        if std_dev is None or n is None:
            raise ValueError("give std_dev and n, or standard_error")
        se = _pos("std_dev", std_dev) / math.sqrt(_n("n", n, 1))
    else:
        se = _pos("standard_error", standard_error)
    z = (_num("sample_mean", sample_mean) - _num("hypothesized_mean", hypothesized_mean)) / se
    return _decide(z, sps.norm, alpha, tail, standard_error=se)


@formula("quant", formula="t_(n-1) = (Xbar - mu0)/(s/sqrt(n))", curriculum=_R11)
def t_test_mean(sample_mean: float, hypothesized_mean: float, std_dev: float, n: int, alpha: float = 0.05,
                tail: str = "two") -> dict:
    """t-test of a single population mean (variance unknown, normal population or large sample).

    Returns:
        statistic, df, critical_values, p_value, reject_null, decision, standard_error.

    Example:
        t_test_mean(sample_mean=1.50, hypothesized_mean=1.10, std_dev=3.60, n=24, alpha=0.10) -> t 0.5443,
        critical +/-1.714, fail to reject (CFA L1 R11 Example 1)
    """
    n = _n("n", n)
    se = _pos("std_dev", std_dev) / math.sqrt(n)
    t = (_num("sample_mean", sample_mean) - _num("hypothesized_mean", hypothesized_mean)) / se
    return _decide(t, sps.t(n - 1), alpha, tail, df=n - 1, standard_error=se)


@formula("quant", formula="s_p^2 = [(n1-1)s1^2 + (n2-1)s2^2]/(n1+n2-2); t = (X1 - X2 - d0)/sqrt(s_p^2/n1 + s_p^2/n2)",
         curriculum=_R11)
def t_test_two_means_pooled(mean1: float, std_dev1: float, n1: int, mean2: float, std_dev2: float, n2: int,
                            hypothesized_difference: float = 0.0, alpha: float = 0.05, tail: str = "two") -> dict:
    """Difference in means, independent normal samples, equal (unknown) variances: pooled t-test.

    Returns:
        statistic, df (n1 + n2 - 2), critical_values, p_value, reject_null, decision, pooled_variance, standard_error.

    Example:
        t_test_two_means_pooled(0.580, 4.598, 120, 1.470, 4.738, 120) -> t -1.477 (CFA L1 R11 Example 4)
    """
    n1, n2 = _n("n1", n1), _n("n2", n2)
    s1, s2 = _pos("std_dev1", std_dev1), _pos("std_dev2", std_dev2)
    sp2 = ((n1 - 1) * s1 ** 2 + (n2 - 1) * s2 ** 2) / (n1 + n2 - 2)
    se = math.sqrt(sp2 / n1 + sp2 / n2)
    t = (_num("mean1", mean1) - _num("mean2", mean2) - _num("hypothesized_difference", hypothesized_difference)) / se
    df = n1 + n2 - 2
    return _decide(t, sps.t(df), alpha, tail, df=df, pooled_variance=sp2, standard_error=se)


@formula("quant", formula="t = (X1 - X2 - d0)/sqrt(s1^2/n1 + s2^2/n2); df = (s1^2/n1 + s2^2/n2)^2 / "
                          "[(s1^2/n1)^2/n1 + (s2^2/n2)^2/n2]", curriculum=_R11)
def t_test_two_means_unequal(mean1: float, std_dev1: float, n1: int, mean2: float, std_dev2: float, n2: int,
                             hypothesized_difference: float = 0.0, alpha: float = 0.05, tail: str = "two") -> dict:
    """Difference in means, independent normal samples, unequal variances.

    Degrees of freedom follow the CFA curriculum formula (Eq. 9, n_i in the denominators). The textbook
    Welch-Satterthwaite df (n_i - 1 in the denominators, slightly smaller) is also returned as df_welch.

    Returns:
        statistic, df, critical_values, p_value, reject_null, decision, df_welch, standard_error.

    Example:
        t_test_two_means_unequal(64.42, 14.03, 21, 55.75, 25.17, 64, alpha=0.10) -> t 1.975, df 64.99,
        reject (CFA L1 R11 Example 5)
    """
    n1, n2 = _n("n1", n1), _n("n2", n2)
    v1, v2 = _pos("std_dev1", std_dev1) ** 2 / n1, _pos("std_dev2", std_dev2) ** 2 / n2
    se = math.sqrt(v1 + v2)
    t = (_num("mean1", mean1) - _num("mean2", mean2) - _num("hypothesized_difference", hypothesized_difference)) / se
    df = (v1 + v2) ** 2 / (v1 ** 2 / n1 + v2 ** 2 / n2)
    welch = (v1 + v2) ** 2 / (v1 ** 2 / (n1 - 1) + v2 ** 2 / (n2 - 1))
    return _decide(t, sps.t(df), alpha, tail, df=df, df_welch=welch, standard_error=se)


@formula("quant", formula="t_(n-1) = (dbar - mu_d0)/(s_d/sqrt(n))", curriculum=_R11)
def t_test_paired(mean_difference: float | None = None, std_dev_difference: float | None = None,
                  n: int | None = None, differences: list | None = None, hypothesized_difference: float = 0.0,
                  alpha: float = 0.05, tail: str = "two") -> dict:
    """Paired-comparisons t-test (dependent samples), from summary statistics or the raw differences.

    Returns:
        statistic, df, critical_values, p_value, reject_null, decision, mean_difference, standard_error.

    Example:
        t_test_paired(mean_difference=3.06, std_dev_difference=6.62, n=50, alpha=0.01) -> t 3.2685, reject
        (CFA L1 R11 Example 6)
    """
    if differences is not None:
        d = _vec("differences", differences, 2)
        dbar, sd, n = float(d.mean()), float(d.std(ddof=1)), d.size
    else:
        if None in (mean_difference, std_dev_difference, n):
            raise ValueError("give differences, or mean_difference, std_dev_difference and n")
        dbar, sd, n = _num("mean_difference", mean_difference), _pos("std_dev_difference", std_dev_difference), _n("n", n)
    se = sd / math.sqrt(n)
    t = (dbar - _num("hypothesized_difference", hypothesized_difference)) / se
    return _decide(t, sps.t(n - 1), alpha, tail, df=n - 1, mean_difference=dbar, standard_error=se)


@formula("quant", formula="chi^2_(n-1) = (n - 1) s^2 / sigma0^2", curriculum=_R11)
def chi_square_test_variance(n: int, sample_variance: float | None = None, hypothesized_variance: float | None = None,
                             sample_std_dev: float | None = None, hypothesized_std_dev: float | None = None,
                             alpha: float = 0.05, tail: str = "two") -> dict:
    """Chi-square test of a single population variance (normal population). Give variances or std devs.

    Returns:
        statistic, df, critical_values, p_value, reject_null, decision.

    Example:
        chi_square_test_variance(n=24, sample_std_dev=3.60, hypothesized_std_dev=4, tail="lower") -> 18.63,
        critical 13.091, fail to reject (CFA L1 R11 Example 7)
    """
    n = _n("n", n)
    s2 = _variance(sample_variance, sample_std_dev, "sample")
    v0 = _variance(hypothesized_variance, hypothesized_std_dev, "hypothesized")
    return _decide((n - 1) * s2 / v0, sps.chi2(n - 1), alpha, tail, df=n - 1)


@formula("quant", formula="F = s1^2 / s2^2 with (n1 - 1, n2 - 1) df", curriculum=_R11)
def f_test_variances(n1: int, n2: int, variance1: float | None = None, variance2: float | None = None,
                     std_dev1: float | None = None, std_dev2: float | None = None, alpha: float = 0.05,
                     tail: str = "two") -> dict:
    """F-test of equality of two population variances (independent normal samples). Give variances or std devs.

    Convention: put the larger sample variance in the numerator for a two-tailed test.

    Returns:
        statistic, df ([n1-1, n2-1]), critical_values, p_value, reject_null, decision.

    Example:
        f_test_variances(n1=12, n2=115, std_dev1=1.178, std_dev2=0.786, tail="upper") -> F 2.25, reject
        (CFA L1 R11 Example 9)
    """
    n1, n2 = _n("n1", n1), _n("n2", n2)
    f = _variance(variance1, std_dev1, "1") / _variance(variance2, std_dev2, "2")
    return _decide(f, sps.f(n1 - 1, n2 - 1), alpha, tail, df=[n1 - 1, n2 - 1])


@formula("quant", formula="t_(n-2) = r sqrt(n - 2) / sqrt(1 - r^2)",
         curriculum="L2 Quant R11: Correlation and Regression; L1 Quant R11")
def correlation_t_test(r: float, n: int, alpha: float = 0.05, tail: str = "two") -> dict:
    """t-test that a population correlation is zero.

    Returns:
        statistic, df, critical_values, p_value, reject_null, decision.

    Example:
        correlation_t_test(r=0.8702, n=6) -> t 3.532 (CFA L2 R11 Example 7)
    """
    r, n = _num("r", r), _n("n", n, 3)
    if not -1 < r < 1:
        raise ValueError("r must be strictly between -1 and 1")
    t = r * math.sqrt(n - 2) / math.sqrt(1 - r * r)
    return _decide(t, sps.t(n - 2), alpha, tail, df=n - 2)


@formula("quant", formula="r_s = 1 - 6 sum d_i^2 / [n(n^2 - 1)]; t = r_s sqrt(n-2)/sqrt(1 - r_s^2)", curriculum=_R11)
def spearman_rank_correlation(x: list, y: list, alpha: float = 0.05, tail: str = "two") -> dict:
    """Spearman rank correlation (nonparametric), ties get average ranks, with the large-sample t-test.

    For n <= 30 compare r_s with the curriculum's Spearman table rather than the t-test.

    Returns:
        statistic (t), df, critical_values, p_value, reject_null, decision, spearman_r, sum_d_squared.

    Example:
        spearman_rank_correlation(sharpe_ratios, expense_ratios) -> r_s 0.2545 (CFA L1 R11 Table 10)
    """
    a, b = _vec("x", x, 3), _vec("y", y, 3)
    if a.size != b.size:
        raise ValueError("x and y must have the same length")
    n = a.size
    d2 = float(np.sum((sps.rankdata(-a) - sps.rankdata(-b)) ** 2))
    rs = 1 - 6 * d2 / (n * (n * n - 1))
    t = rs * math.sqrt(n - 2) / math.sqrt(1 - rs * rs) if abs(rs) < 1 else math.copysign(math.inf, rs)
    return _decide(t, sps.t(n - 2), alpha, tail, df=n - 2, spearman_r=rs, sum_d_squared=d2)


# ---------------------------------------------------------------- Monte Carlo

@formula("quant", formula="V_T = V_0 prod(1 + R_t) (normal) or V_0 exp(sum r_t) (lognormal), seeded draws",
         curriculum="L1 Quant R9: Monte Carlo Simulation")
def monte_carlo_terminal_wealth(initial_value: float, mean_return: float, std_dev: float, periods: int,
                                trials: int = 10000, seed: int = 42, model: str = "lognormal",
                                threshold: float | None = None) -> dict:
    """Seeded Monte Carlo of terminal portfolio value under i.i.d. returns.

    Args:
        initial_value: starting value V_0.
        mean_return: per-period mean; of continuously compounded returns if model="lognormal",
            of simple returns if model="normal".
        std_dev: per-period standard deviation of those returns.
        periods: number of periods simulated.
        trials: number of paths.
        seed: random seed (same seed -> same answer).
        model: "lognormal" (log returns normal) or "normal" (simple returns normal, floored at -100%).
        threshold: optional terminal value; returns the probability of ending below it.

    Returns:
        mean_terminal_value, median_terminal_value, std_terminal_value, percentiles (5, 25, 50, 75, 95),
        prob_below_threshold, trials, seed.

    Example:
        monte_carlo_terminal_wealth(initial_value=100, mean_return=0.005, std_dev=0.04, periods=12) -> mean ~ 107.2
    """
    v0, mu, s = _pos("initial_value", initial_value), _num("mean_return", mean_return), _pos("std_dev", std_dev)
    n, k = _n("periods", periods, 1), _n("trials", trials, 1)
    if model not in ("lognormal", "normal"):
        raise ValueError('model must be "lognormal" or "normal"')
    draws = np.random.default_rng(int(seed)).normal(mu, s, size=(k, n))
    if model == "lognormal":
        vt = v0 * np.exp(draws.sum(axis=1))
    else:
        vt = v0 * np.prod(np.maximum(1 + draws, 0.0), axis=1)
    pct = np.percentile(vt, [5, 25, 50, 75, 95])
    return {"mean_terminal_value": float(vt.mean()), "median_terminal_value": float(np.median(vt)),
            "std_terminal_value": float(vt.std(ddof=1)) if k > 1 else 0.0,
            "percentiles": dict(zip(["p5", "p25", "p50", "p75", "p95"], pct.tolist())),
            "prob_below_threshold": None if threshold is None else float(np.mean(vt < _num("threshold", threshold))),
            "trials": k, "seed": int(seed)}
