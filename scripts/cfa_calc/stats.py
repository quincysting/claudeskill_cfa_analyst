"""Descriptive statistics, return means, risk ratios, probability and counting (L1 Quant R7-R9)."""
from __future__ import annotations

import math
from collections import Counter

import numpy as np
from scipy import stats as sps

from ._registry import formula

_R7 = "L1 Quant R7: Statistical Concepts and Market Returns"
_R8 = "L1 Quant R8: Probability Concepts"
_R9 = "L1 Quant R9: Common Probability Distributions"


def _num(name: str, x) -> float:
    try:
        v = float(x)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be a number, got {x!r}") from None
    if not math.isfinite(v):
        raise ValueError(f"{name} must be finite, got {x!r}")
    return v


def _vec(name: str, xs, min_len: int = 1) -> np.ndarray:
    if isinstance(xs, (int, float)):
        xs = [xs]
    a = np.array([_num(name, x) for x in xs], dtype=float)
    if a.size < min_len:
        raise ValueError(f"{name} needs at least {min_len} values")
    return a


def _prob(name: str, p) -> float:
    p = _num(name, p)
    if not 0 <= p <= 1:
        raise ValueError(f"{name} must be a probability in [0, 1], got {p}")
    return p


def _probs(name: str, ps, tol: float = 1e-6) -> np.ndarray:
    a = _vec(name, ps)
    if np.any(a < 0) or abs(a.sum() - 1) > tol:
        raise ValueError(f"{name} must be non-negative and sum to 1 (sum = {a.sum():.8g})")
    return a


# ---------------------------------------------------------------- descriptive statistics

@formula("quant", formula="s^2 = sum(X-Xbar)^2/(n-1); S_K = n/((n-1)(n-2)) sum(X-Xbar)^3/s^3; "
                          "K_E = n(n+1)/((n-1)(n-2)(n-3)) sum(X-Xbar)^4/s^4 - 3(n-1)^2/((n-2)(n-3))",
         curriculum=_R7)
def descriptive_statistics(data: list) -> dict:
    """Central tendency, dispersion and shape of a data set, using the CFA curriculum formulas.

    Args:
        data: observations (e.g. returns as decimals or percents; outputs keep the same units).

    Returns:
        n, mean, median, modes (empty if no value repeats), min, max, range, mean_absolute_deviation,
        sample_variance, sample_std_dev, population_variance, population_std_dev,
        coefficient_of_variation (sample s / mean), skewness (sample, Eq. 17), excess_kurtosis (sample, Eq. 18).
        skewness needs n >= 3 and excess_kurtosis n >= 4 (None otherwise).

    Example:
        descriptive_statistics([16.2, 20.3, 9.3, -11.1, -17.0]) -> mean 3.54, MAD 14.07,
        sample_variance 277.643, sample_std_dev 16.66 (CFA L1 R7 Examples 7, 10, 12)
    """
    x = _vec("data", data)
    n = x.size
    mean = float(x.mean())
    dev = x - mean
    ss = float(np.sum(dev ** 2))
    s2 = ss / (n - 1) if n > 1 else None
    s = math.sqrt(s2) if s2 is not None else None
    counts = Counter(x.tolist())
    top = max(counts.values())
    modes = sorted(v for v, c in counts.items() if c == top) if top > 1 else []
    skew = kurt = None
    if s:
        if n >= 3:
            skew = n / ((n - 1) * (n - 2)) * float(np.sum(dev ** 3)) / s ** 3
        if n >= 4:
            kurt = (n * (n + 1) / ((n - 1) * (n - 2) * (n - 3)) * float(np.sum(dev ** 4)) / s ** 4
                    - 3 * (n - 1) ** 2 / ((n - 2) * (n - 3)))
    return {"n": n, "mean": mean, "median": float(np.median(x)), "modes": modes, "min": float(x.min()),
            "max": float(x.max()), "range": float(x.max() - x.min()),
            "mean_absolute_deviation": float(np.mean(np.abs(dev))),
            "sample_variance": s2, "sample_std_dev": s, "population_variance": ss / n,
            "population_std_dev": math.sqrt(ss / n),
            "coefficient_of_variation": s / mean if s is not None and mean != 0 else None,
            "skewness": skew, "excess_kurtosis": kurt}


@formula("quant", formula="arithmetic = sum X/n; geometric = (prod X)^(1/n); harmonic = n/sum(1/X); "
                          "weighted = sum w X", curriculum=_R7)
def mean_measures(values: list, weights: list | None = None) -> dict:
    """Arithmetic, geometric, harmonic and (optionally) weighted means of a set of values.

    For returns use geometric_mean_return instead (it works on 1 + R).

    Args:
        values: observations; geometric and harmonic means need all values > 0.
        weights: optional weights for the weighted mean; normalized to sum to 1 (weights_sum reports the
            raw sum so a mistaken input is visible).

    Returns:
        arithmetic_mean, geometric_mean, harmonic_mean, weighted_mean, weights_sum.

    Example:
        mean_measures([10, 15]) -> harmonic_mean 12 (cost averaging, CFA L1 R7 Section 5.4.3)
    """
    x = _vec("values", values)
    pos = bool(np.all(x > 0))
    out = {"arithmetic_mean": float(x.mean()),
           "geometric_mean": float(np.exp(np.mean(np.log(x)))) if pos else None,
           "harmonic_mean": float(x.size / np.sum(1 / x)) if pos else None,
           "weighted_mean": None, "weights_sum": None}
    if weights is not None:
        w = _vec("weights", weights)
        if w.size != x.size or w.sum() == 0:
            raise ValueError("weights must have the same length as values and a non-zero sum")
        out["weighted_mean"] = float(np.sum(w * x) / w.sum())
        out["weights_sum"] = float(w.sum())
    return out


@formula("quant", formula="R_G = [prod(1 + R_t)]^(1/T) - 1", curriculum=_R7)
def geometric_mean_return(returns: list) -> dict:
    """Geometric (compound) mean return of a return series, with the arithmetic mean for comparison.

    Args:
        returns: periodic returns as decimals (each > -1).

    Returns:
        geometric_mean_return, arithmetic_mean_return, cumulative_return, ending_wealth_per_unit, periods.

    Example:
        geometric_mean_return([0.162, 0.203, 0.093, -0.111, -0.170]) -> 0.024270 (CFA L1 R7 Example 7)
    """
    r = _vec("returns", returns)
    if np.any(r <= -1):
        raise ValueError("returns must be greater than -100%")
    w = float(np.prod(1 + r))
    return {"geometric_mean_return": w ** (1 / r.size) - 1, "arithmetic_mean_return": float(r.mean()),
            "cumulative_return": w - 1, "ending_wealth_per_unit": w, "periods": r.size}


@formula("quant", formula="L_y = (n + 1) y/100; P_y = X_floor(L) + (L - floor L)(X_ceil(L) - X_floor(L))",
         curriculum=_R7)
def quantile_curriculum(data: list, p: float) -> dict:
    """Percentile/quartile/quintile/decile by the CFA curriculum method (position (n+1)p, linear interpolation).

    Args:
        data: observations (any order).
        p: the fraction of data at or below the quantile, decimal (0.25 = first quartile, 0.9 = 90th percentile).

    Returns:
        quantile, location (1-based position in the sorted data), lower_value, upper_value.
        Locations below 1 or above n return the minimum or maximum.

    Example:
        quantile_curriculum(dividend_yields_50_stocks, 0.10) -> 0.343 (CFA L1 R7 Example 9)
    """
    x = np.sort(_vec("data", data))
    p = _num("p", p)
    if not 0 < p < 1:
        raise ValueError("p must be strictly between 0 and 1")
    n = x.size
    loc = (n + 1) * p
    if loc <= 1:
        return {"quantile": float(x[0]), "location": loc, "lower_value": float(x[0]), "upper_value": float(x[0])}
    if loc >= n:
        return {"quantile": float(x[-1]), "location": loc, "lower_value": float(x[-1]), "upper_value": float(x[-1])}
    k = int(math.floor(loc))
    lo, hi = float(x[k - 1]), float(x[k])
    return {"quantile": lo + (loc - k) * (hi - lo), "location": loc, "lower_value": lo, "upper_value": hi}


@formula("quant", formula="semivariance = sum_{X_i <= B} (X_i - B)^2/(n - 1); B = mean or target",
         curriculum=_R7)
def semivariance(data: list, target: float | None = None) -> dict:
    """Downside dispersion: (target) semivariance and semideviation, curriculum n - 1 divisor.

    Args:
        data: observations (e.g. returns).
        target: threshold B; None uses the sample mean (plain semivariance).

    Returns:
        semivariance, semideviation, target, observations_below.

    Example:
        semivariance([16.2, 20.3, 9.3, -11.1, -17.0]) -> 159.0553, semideviation 12.61 (CFA L1 R7 Section 7.5)
    """
    x = _vec("data", data, 2)
    b = float(x.mean()) if target is None else _num("target", target)
    below = x[x <= b]
    sv = float(np.sum((below - b) ** 2) / (x.size - 1))
    return {"semivariance": sv, "semideviation": math.sqrt(sv), "target": b, "observations_below": int(below.size)}


@formula("quant", formula="P(|X - mean| <= k s) >= 1 - 1/k^2, k > 1", curriculum=_R7)
def chebyshev_inequality(k: float, mean: float | None = None, std_dev: float | None = None) -> dict:
    """Minimum share of any distribution within k standard deviations of the mean (Chebyshev).

    Args:
        k: number of standard deviations, > 1.
        mean, std_dev: optional, to also return the interval endpoints.

    Returns:
        min_proportion_within, max_proportion_outside, lower, upper.

    Example:
        chebyshev_inequality(k=2, mean=0.97, std_dev=5.65) -> 0.75, interval -10.33 to 12.27 (CFA L1 R7 Example 13)
    """
    k = _num("k", k)
    if k <= 1:
        raise ValueError("k must be greater than 1")
    lo = hi = None
    if mean is not None and std_dev is not None:
        m, s = _num("mean", mean), _num("std_dev", std_dev)
        lo, hi = m - k * s, m + k * s
    return {"min_proportion_within": 1 - 1 / k ** 2, "max_proportion_outside": 1 / k ** 2, "lower": lo, "upper": hi}


@formula("quant", formula="CV = s / Xbar", curriculum=_R7)
def coefficient_of_variation(std_dev: float, mean: float) -> dict:
    """Coefficient of variation: risk per unit of mean (scale-free relative dispersion).

    Args:
        std_dev: standard deviation.
        mean: mean, same units, non-zero.

    Returns:
        coefficient_of_variation, inverse (mean per unit of risk).

    Example:
        coefficient_of_variation(std_dev=21.9, mean=12.3) -> 1.780 (CFA L1 R7 Example 14)
    """
    s, m = _num("std_dev", std_dev), _num("mean", mean)
    if m == 0 or s == 0:
        raise ValueError("mean and std_dev must be non-zero")
    return {"coefficient_of_variation": s / m, "inverse": m / s}


@formula("quant", formula="S_h = (Rbar_p - Rbar_F) / s_p", curriculum=_R7)
def sharpe_ratio_historical(mean_return: float, risk_free_rate: float, std_dev: float) -> dict:
    """Historical Sharpe ratio: mean excess return per unit of standard deviation.

    Args:
        mean_return: mean portfolio return over the period, decimal (or percent, consistently).
        risk_free_rate: mean risk-free return over the same period.
        std_dev: standard deviation of portfolio returns.

    Returns:
        sharpe_ratio, excess_return. Caution: negative Sharpe ratios cannot be ranked (more risk raises them).

    Example:
        sharpe_ratio_historical(mean_return=12.3, risk_free_rate=3.8, std_dev=21.9) -> 0.39 (CFA L1 R7 Section 7.8)
    """
    m, rf, s = _num("mean_return", mean_return), _num("risk_free_rate", risk_free_rate), _num("std_dev", std_dev)
    if s <= 0:
        raise ValueError("std_dev must be positive")
    return {"sharpe_ratio": (m - rf) / s, "excess_return": m - rf}


@formula("quant", formula="SFRatio = [E(R_p) - R_L] / sigma_p; P(R_p < R_L) = N(-SFRatio)", curriculum=_R9)
def roy_safety_first_ratio(expected_returns: list | float, std_devs: list | float, threshold_return: float) -> dict:
    """Roy's safety-first ratio and shortfall probability (normal returns); picks the best portfolio.

    Args:
        expected_returns: expected return of one portfolio or a list of portfolios, decimal.
        std_devs: matching standard deviations.
        threshold_return: minimum acceptable return R_L.

    Returns:
        sf_ratios, shortfall_probabilities, best_index (highest ratio = lowest shortfall probability).

    Example:
        roy_safety_first_ratio([0.25, 0.11, 0.14], [0.27, 0.08, 0.20], 0.0375) -> best_index 1,
        ratio 0.90625, shortfall 0.1824 (CFA L1 R9 Example 9)
    """
    mu, sd = _vec("expected_returns", expected_returns), _vec("std_devs", std_devs)
    if mu.size != sd.size or np.any(sd <= 0):
        raise ValueError("expected_returns and std_devs must match in length; std_devs must be positive")
    sfr = (mu - _num("threshold_return", threshold_return)) / sd
    return {"sf_ratios": sfr.tolist(), "shortfall_probabilities": sps.norm.cdf(-sfr).tolist(),
            "best_index": int(np.argmax(sfr))}


# ---------------------------------------------------------------- probability

@formula("quant", formula="P(A or B) = P(A) + P(B) - P(AB); P(AB) = P(A|B)P(B); P(A|B) = P(AB)/P(B)",
         curriculum=_R8)
def probability_rules(p_a: float, p_b: float, p_a_and_b: float | None = None, p_a_given_b: float | None = None,
                      p_b_given_a: float | None = None, independent: bool = False,
                      mutually_exclusive: bool = False) -> dict:
    """Joint, union and conditional probabilities of two events (addition and multiplication rules).

    Give P(A), P(B) and one way to get the joint probability: p_a_and_b, p_a_given_b, p_b_given_a,
    independent=True or mutually_exclusive=True.

    Returns:
        p_a_or_b, p_a_and_b, p_a_given_b, p_b_given_a, are_independent, odds_for_a, odds_against_a.

    Example:
        probability_rules(p_a=0.35, p_b=0.25, p_a_and_b=0.25) -> p_a_or_b 0.35, p_b_given_a 0.714
        (CFA L1 R8 Example 3)
    """
    pa, pb = _prob("p_a", p_a), _prob("p_b", p_b)
    if p_a_and_b is not None:
        j = _prob("p_a_and_b", p_a_and_b)
    elif p_a_given_b is not None:
        j = _prob("p_a_given_b", p_a_given_b) * pb
    elif p_b_given_a is not None:
        j = _prob("p_b_given_a", p_b_given_a) * pa
    elif independent:
        j = pa * pb
    elif mutually_exclusive:
        j = 0.0
    else:
        raise ValueError("give p_a_and_b, p_a_given_b, p_b_given_a, independent=True or mutually_exclusive=True")
    if j > min(pa, pb) + 1e-12 or j < pa + pb - 1 - 1e-12:
        raise ValueError(f"joint probability {j} is inconsistent with P(A)={pa}, P(B)={pb}")
    return {"p_a_or_b": pa + pb - j, "p_a_and_b": j, "p_a_given_b": j / pb if pb else None,
            "p_b_given_a": j / pa if pa else None, "are_independent": abs(j - pa * pb) < 1e-12,
            "odds_for_a": pa / (1 - pa) if pa < 1 else None, "odds_against_a": (1 - pa) / pa if pa else None}


@formula("quant", formula="P(A) = sum_i P(A|S_i) P(S_i); E(X) = sum_i E(X|S_i) P(S_i)", curriculum=_R8)
def total_probability(conditional_values: list, scenario_probabilities: list) -> dict:
    """Total probability rule (or total expectation rule) over mutually exclusive, exhaustive scenarios.

    Args:
        conditional_values: P(A | S_i) for each scenario, or E(X | S_i) for the expectation version.
        scenario_probabilities: P(S_i), must sum to 1.

    Returns:
        total, weighted_terms.

    Example:
        total_probability([0.75, 0.20, 0.05], [0.45, 0.30, 0.25]) -> 0.41 (CFA L1 R8 Bayes, DriveMed)
    """
    c = _vec("conditional_values", conditional_values)
    p = _probs("scenario_probabilities", scenario_probabilities)
    if c.size != p.size:
        raise ValueError("conditional_values and scenario_probabilities must have the same length")
    terms = c * p
    return {"total": float(terms.sum()), "weighted_terms": terms.tolist()}


@formula("quant", formula="P(Event|Info) = P(Info|Event) P(Event) / P(Info)", curriculum=_R8)
def bayes_update(priors: list, likelihoods: list) -> dict:
    """Bayes' formula: update prior probabilities of mutually exclusive, exhaustive events given new information.

    Args:
        priors: P(Event_i), must sum to 1.
        likelihoods: P(Information | Event_i).

    Returns:
        posteriors, p_information (unconditional probability of the information).

    Example:
        bayes_update([0.45, 0.30, 0.25], [0.75, 0.20, 0.05]) -> posteriors [0.8232, 0.1463, 0.0305]
        (CFA L1 R8 Example 13)
    """
    pr = _probs("priors", priors)
    lk = _vec("likelihoods", likelihoods)
    if lk.size != pr.size or np.any(lk < 0) or np.any(lk > 1):
        raise ValueError("likelihoods must be probabilities, one per prior")
    pi = float(np.sum(pr * lk))
    if pi == 0:
        raise ValueError("the information has zero probability under these priors")
    return {"posteriors": (pr * lk / pi).tolist(), "p_information": pi}


@formula("quant", formula="E(X) = sum P(x_i) x_i; Var(X) = sum P(x_i)[x_i - E(X)]^2", curriculum=_R8)
def expected_value_variance(outcomes: list, probabilities: list) -> dict:
    """Expected value, variance and standard deviation of a discrete random variable.

    Args:
        outcomes: possible values.
        probabilities: their probabilities, summing to 1.

    Returns:
        expected_value, variance, std_dev.

    Example:
        expected_value_variance([2.60, 2.45, 2.20, 2.00], [0.15, 0.45, 0.24, 0.16]) -> E 2.3405,
        variance 0.038785 (CFA L1 R8 Examples 8-9)
    """
    x = _vec("outcomes", outcomes)
    p = _probs("probabilities", probabilities)
    if x.size != p.size:
        raise ValueError("outcomes and probabilities must have the same length")
    ev = float(np.sum(p * x))
    var = float(np.sum(p * (x - ev) ** 2))
    return {"expected_value": ev, "variance": var, "std_dev": math.sqrt(var)}


@formula("quant", formula="Cov(X,Y) = sum_i sum_j P(x_i, y_j)(x_i - E X)(y_j - E Y); rho = Cov/(sd_X sd_Y)",
         curriculum=_R8)
def joint_probability_covariance(x_values: list, y_values: list, joint_probabilities: list) -> dict:
    """Covariance and correlation of two discrete random variables from their joint probability table.

    Args:
        x_values: possible values of X (rows of the table).
        y_values: possible values of Y (columns).
        joint_probabilities: matrix P(X = x_i, Y = y_j), rows = x_values, summing to 1.

    Returns:
        covariance, correlation, expected_x, expected_y, std_dev_x, std_dev_y.

    Example:
        joint_probability_covariance([25, 12, 10], [20, 16, 10],
        [[0.2, 0, 0], [0, 0.5, 0], [0, 0, 0.3]]) -> covariance 16 (CFA L1 R8 Tables 12-13)
    """
    x, y = _vec("x_values", x_values), _vec("y_values", y_values)
    p = np.array([[_num("joint_probabilities", v) for v in row] for row in joint_probabilities], dtype=float)
    if p.shape != (x.size, y.size):
        raise ValueError(f"joint_probabilities must be {x.size} x {y.size}")
    if np.any(p < 0) or abs(p.sum() - 1) > 1e-6:
        raise ValueError("joint_probabilities must be non-negative and sum to 1")
    px, py = p.sum(axis=1), p.sum(axis=0)
    ex, ey = float(px @ x), float(py @ y)
    cov = float(np.sum(p * np.outer(x - ex, y - ey)))
    sx, sy = math.sqrt(float(px @ (x - ex) ** 2)), math.sqrt(float(py @ (y - ey) ** 2))
    return {"covariance": cov, "correlation": cov / (sx * sy) if sx and sy else None, "expected_x": ex,
            "expected_y": ey, "std_dev_x": sx, "std_dev_y": sy}


@formula("quant", formula="E(R_p) = sum w_i E(R_i); Var(R_p) = sum_i sum_j w_i w_j Cov(R_i, R_j)", curriculum=_R8)
def portfolio_expected_return_variance(weights: list, expected_returns: list, covariance_matrix: list) -> dict:
    """Portfolio expected return, variance and standard deviation from weights and a covariance matrix.

    Args:
        weights: portfolio weights (should sum to 1; weights_sum is reported).
        expected_returns: expected return of each asset.
        covariance_matrix: n x n covariance matrix, same units squared (e.g. percent^2 with percent returns).

    Returns:
        expected_return, variance, std_dev, correlation_matrix, variance_terms (sum w_i^2 var_i),
        covariance_terms (the rest), weights_sum.

    Example:
        portfolio_expected_return_variance([0.5, 0.25, 0.25], [13, 6, 15],
        [[400, 45, 189], [45, 81, 38], [189, 38, 441]]) -> 11.75, variance 195.875, std 14 (CFA L1 R8 Section 3)
    """
    w, mu = _vec("weights", weights), _vec("expected_returns", expected_returns)
    c = np.array([[_num("covariance_matrix", v) for v in row] for row in covariance_matrix], dtype=float)
    if c.shape != (w.size, w.size) or mu.size != w.size:
        raise ValueError("weights, expected_returns and covariance_matrix dimensions must agree")
    if not np.allclose(c, c.T):
        raise ValueError("covariance_matrix must be symmetric")
    var = float(w @ c @ w)
    if var < -1e-12:
        raise ValueError("covariance_matrix is not positive semi-definite (negative portfolio variance)")
    d = np.sqrt(np.diag(c))
    corr = (c / np.outer(d, d)).tolist() if np.all(d > 0) else None
    vt = float(np.sum(w ** 2 * np.diag(c)))
    return {"expected_return": float(w @ mu), "variance": var, "std_dev": math.sqrt(max(var, 0.0)),
            "correlation_matrix": corr, "variance_terms": vt, "covariance_terms": var - vt,
            "weights_sum": float(w.sum())}


@formula("quant", formula="n!; nCr = n!/((n-r)! r!); nPr = n!/(n-r)!; multinomial = n!/(n1! n2! ... nk!)",
         curriculum=_R8)
def counting_rules(n: int, r: int | None = None, group_sizes: list | None = None) -> dict:
    """Counting: factorial, combinations, permutations and the multinomial (labeling) formula.

    Args:
        n: number of objects.
        r: number chosen (for combinations and permutations).
        group_sizes: sizes of each label group for the multinomial formula; must sum to n.

    Returns:
        factorial, combinations, permutations, multinomial (exact integers; None when not requested).

    Example:
        counting_rules(n=5, r=3) -> combinations 10, permutations 60 (CFA L1 R8 Section 4.2)
    """
    if int(n) != n or n < 0:
        raise ValueError("n must be a non-negative integer")
    n = int(n)
    out = {"factorial": math.factorial(n), "combinations": None, "permutations": None, "multinomial": None}
    if r is not None:
        if int(r) != r or not 0 <= r <= n:
            raise ValueError("r must be an integer between 0 and n")
        out["combinations"], out["permutations"] = math.comb(n, int(r)), math.perm(n, int(r))
    if group_sizes is not None:
        g = [int(k) for k in group_sizes]
        if any(k < 0 for k in g) or sum(g) != n or any(int(k) != k for k in group_sizes):
            raise ValueError("group_sizes must be non-negative integers summing to n")
        m = math.factorial(n)
        for k in g:
            m //= math.factorial(k)
        out["multinomial"] = m
    return out


@formula("quant", formula="p(x) = nCx p^x (1-p)^(n-x); E = np; Var = np(1-p)", curriculum=_R9)
def binomial_probability(n: int, p: float, k: int) -> dict:
    """Binomial distribution: probability of exactly, at most and at least k successes in n trials.

    Args:
        n: number of independent Bernoulli trials.
        p: probability of success on each trial.
        k: number of successes of interest.

    Returns:
        p_exactly_k, p_at_most_k (cdf), p_at_least_k, mean, variance, std_dev.

    Example:
        binomial_probability(n=12, p=0.5, k=3) -> p_at_most_k 0.072998 (CFA L1 R9 Example 4)
    """
    if int(n) != n or n < 0 or int(k) != k or not 0 <= k <= n:
        raise ValueError("n must be a non-negative integer and k an integer in [0, n]")
    n, k, p = int(n), int(k), _prob("p", p)
    d = sps.binom(n, p)
    return {"p_exactly_k": float(d.pmf(k)), "p_at_most_k": float(d.cdf(k)), "p_at_least_k": float(d.sf(k - 1)),
            "mean": n * p, "variance": n * p * (1 - p), "std_dev": math.sqrt(n * p * (1 - p))}


@formula("quant", formula="S_T = S_0 u^j d^(n-j) with probability nCj p^j (1-p)^(n-j)", curriculum=_R9)
def binomial_tree(s0: float, up_factor: float, down_factor: float, p_up: float, periods: int) -> dict:
    """Binomial price tree: node values per period and the distribution of the terminal price.

    Args:
        s0: starting price.
        up_factor: u = 1 + return on an up move.
        down_factor: d = 1 + return on a down move (d < u).
        p_up: probability of an up move each period.
        periods: number of steps n.

    Returns:
        terminal_prices (from n up moves down to 0), terminal_probabilities, paths_to_node,
        expected_terminal_price, tree (node prices per period, highest first).

    Example:
        binomial_tree(s0=100, up_factor=1.1, down_factor=0.9, p_up=0.5, periods=3) -> P(uud) = 3p^2(1-p)
        (CFA L1 R9 Section 2.2)
    """
    s, u, d = _num("s0", s0), _num("up_factor", up_factor), _num("down_factor", down_factor)
    p = _prob("p_up", p_up)
    if int(periods) != periods or periods < 1:
        raise ValueError("periods must be a positive integer")
    if not 0 < d < u:
        raise ValueError("need 0 < down_factor < up_factor")
    n = int(periods)
    tree = [[s * u ** j * d ** (t - j) for j in range(t, -1, -1)] for t in range(n + 1)]
    ups = np.arange(n, -1, -1)
    probs = sps.binom.pmf(ups, n, p)
    prices = np.array(tree[-1])
    return {"terminal_prices": prices.tolist(), "terminal_probabilities": probs.tolist(),
            "paths_to_node": [math.comb(n, int(j)) for j in ups],
            "expected_terminal_price": float(prices @ probs), "tree": tree}


@formula("quant", formula="(1 + R)[(1 - P_d) + P_d * recovery] = 1 + R_F; premium = R - R_F", curriculum=_R8)
def one_period_default_risk_premium(risk_free_rate: float, default_probability: float,
                                    recovery_rate: float = 0.0) -> dict:
    """Minimum promised return (and default risk premium) on a one-period risky bond for a risk-neutral investor.

    Args:
        risk_free_rate: one-period risk-free return, decimal.
        default_probability: probability the bond defaults.
        recovery_rate: fraction of the promised payment (1 + R) recovered on default.

    Returns:
        promised_return, default_risk_premium.

    Example:
        one_period_default_risk_premium(risk_free_rate=0.058, default_probability=0.06) -> promised 0.12553,
        premium 0.06753 (CFA L1 R8 Example 11)
    """
    rf = _num("risk_free_rate", risk_free_rate)
    pd_, rec = _prob("default_probability", default_probability), _prob("recovery_rate", recovery_rate)
    expected_share = (1 - pd_) + pd_ * rec
    if expected_share == 0:
        raise ValueError("certain default with zero recovery: no promised return compensates")
    r = (1 + rf) / expected_share - 1
    return {"promised_return": r, "default_risk_premium": r - rf}
