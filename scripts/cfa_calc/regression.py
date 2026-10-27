"""Correlation, simple and multiple regression, diagnostics and time-series models (L2 Quant R11-R13).

Regressors `x` may be a dict {name: column}, a flat list (one regressor) or a list of rows
(one row per observation). An intercept is added unless add_intercept=False.
"""
from __future__ import annotations

import math

import numpy as np
from scipy import stats as sps

from ._registry import formula
from .inference import _decide, _n

_L2R11 = "L2 Quant R11: Correlation and Regression"
_L2R12 = "L2 Quant R12: Multiple Regression and Issues in Regression Analysis"
_L2R13 = "L2 Quant R13: Time-Series Analysis"


def _num(name: str, x) -> float:
    try:
        v = float(x)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be a number, got {x!r}") from None
    if not math.isfinite(v):
        raise ValueError(f"{name} must be finite, got {x!r}")
    return v


def _vec(name: str, xs, min_len: int = 1) -> np.ndarray:
    a = np.array([_num(name, v) for v in xs], dtype=float)
    if a.size < min_len:
        raise ValueError(f"{name} needs at least {min_len} values")
    return a


def _regressors(x, names=None) -> tuple[np.ndarray, list[str]]:
    """Return (n x k matrix, names) from a dict of columns, a flat list or a list of rows."""
    if isinstance(x, dict):
        names = list(x)
        cols = [_vec(f"x[{k}]", v) for k, v in x.items()]
        if len({c.size for c in cols}) != 1:
            raise ValueError("all regressor columns must have the same length")
        m = np.column_stack(cols)
    else:
        rows = list(x)
        if rows and not isinstance(rows[0], (list, tuple, np.ndarray)):
            m = _vec("x", rows).reshape(-1, 1)
        else:
            m = np.array([[_num("x", v) for v in r] for r in rows], dtype=float)
            if m.ndim != 2:
                raise ValueError("x rows must all have the same number of regressors")
        names = list(names) if names else [f"x{j + 1}" for j in range(m.shape[1])]
    if len(names) != m.shape[1]:
        raise ValueError("names must match the number of regressors")
    return m, names


def _fit(y: np.ndarray, X: np.ndarray) -> dict:
    n, p = X.shape
    if n <= p:
        raise ValueError(f"need more observations ({n}) than parameters ({p})")
    if np.linalg.matrix_rank(X) < p:
        raise ValueError("regressors are perfectly collinear (e.g. the dummy-variable trap: use n - 1 dummies)")
    xtx_inv = np.linalg.inv(X.T @ X)
    b = xtx_inv @ X.T @ y
    e = y - X @ b
    return {"b": b, "e": e, "xtx_inv": xtx_inv, "sse": float(e @ e), "df": n - p}


# ---------------------------------------------------------------- correlation and OLS

@formula("quant", formula="r = Cov(X,Y)/(s_X s_Y); t = r sqrt(n-2)/sqrt(1-r^2)", curriculum=_L2R11)
def sample_correlation(x: list, y: list, alpha: float = 0.05) -> dict:
    """Sample correlation and covariance of two series, with the t-test of zero correlation.

    Returns:
        correlation, covariance, std_dev_x, std_dev_y, t_statistic, p_value, reject_null, n.

    Example:
        sample_correlation(money_growth, inflation) -> r 0.8702, t 3.532 (CFA L2 R11 Tables 1-2, Example 7)
    """
    a, b = _vec("x", x, 3), _vec("y", y, 3)
    if a.size != b.size:
        raise ValueError("x and y must have the same length")
    cov = float(np.cov(a, b, ddof=1)[0, 1])
    sx, sy = float(a.std(ddof=1)), float(b.std(ddof=1))
    if sx == 0 or sy == 0:
        raise ValueError("a series with zero variance has no correlation")
    r = cov / (sx * sy)
    n = a.size
    t = r * math.sqrt(n - 2) / math.sqrt(1 - r * r) if abs(r) < 1 else math.copysign(math.inf, r)
    test = _decide(t, sps.t(n - 2), alpha, "two", df=n - 2)
    return {"correlation": r, "covariance": cov, "std_dev_x": sx, "std_dev_y": sy, "t_statistic": t,
            "p_value": test["p_value"], "reject_null": test["reject_null"], "n": n}


@formula("quant", formula="b = (X'X)^-1 X'y; R^2 = RSS/SST; adj R^2 = 1 - (n-1)/(n-k-1)(1-R^2); "
                          "F = (RSS/k)/(SSE/(n-k-1)); SEE = sqrt(SSE/(n-k-1))",
         curriculum=f"{_L2R11}; {_L2R12}")
def ols_regression(y: list, x, names: list | None = None, add_intercept: bool = True, alpha: float = 0.05,
                   x_new=None) -> dict:
    """Ordinary least squares (simple or multiple) with the full curriculum output and ANOVA table.

    Args:
        y: dependent variable.
        x: regressors: dict {name: column}, flat list (one regressor) or list of rows.
        names: regressor names when x is not a dict.
        add_intercept: include b0 (default True). With add_intercept=False (regression through the origin)
            the ANOVA is uncentred: SST = sum y^2, RSS = SST - SSE, R^2 = RSS/SST (uncentred R^2, not comparable
            with an intercept model's R^2), F = (RSS/k)/(SSE/(n - k)) with (k, n - k) df and
            adjusted R^2 = 1 - n/(n - k) (1 - R^2).
        alpha: significance level for coefficient intervals and tests.
        x_new: optional regressor values (list in regressor order, or dict) to forecast Y; returns the
            point forecast, the confidence interval for the mean and the prediction interval.

    Returns:
        coefficients, std_errors, t_stats, p_values, confidence_intervals (all keyed by name),
        r_squared, adjusted_r_squared, multiple_r, f_statistic, f_p_value, standard_error_of_estimate,
        anova (rss, sse, sst, df_regression, df_residual, msr, mse), n, k, t_critical, durbin_watson,
        forecast (or None), fitted, residuals.

    Example:
        ols_regression(y=inflation, x=money_growth) -> b0 0.0084, b1 0.5545, R^2 0.7572, SEE 0.0098
        (CFA L2 R11 Examples 10-13)
    """
    yv = _vec("y", y, 3)
    m, names = _regressors(x, names)
    if m.shape[0] != yv.size:
        raise ValueError("y and x must have the same number of observations")
    X = np.column_stack([np.ones(yv.size), m]) if add_intercept else m
    labels = (["intercept"] if add_intercept else []) + names
    f = _fit(yv, X)
    n, k, df = yv.size, m.shape[1], f["df"]
    mse = f["sse"] / df
    se = np.sqrt(np.diag(f["xtx_inv"]) * mse)
    tstat = f["b"] / se
    tc = float(sps.t.ppf(1 - _num("alpha", alpha) / 2, df))
    # without an intercept the fit is not centred on y-bar: SST = sum y^2 (uncentred), so R^2 = 1 - SSE/sum y^2
    sst = float(np.sum((yv - yv.mean()) ** 2)) if add_intercept else float(yv @ yv)
    rss = sst - f["sse"]
    r2 = rss / sst if sst else None
    e = f["e"]
    fstat = (rss / k) / mse if k and mse > 0 else None
    forecast = None
    if x_new is not None:
        xn = np.array([_num("x_new", x_new[nm]) for nm in names] if isinstance(x_new, dict)
                      else [_num("x_new", v) for v in np.atleast_1d(x_new)])
        if xn.size != k:
            raise ValueError(f"x_new needs {k} values")
        x0 = np.concatenate([[1.0], xn]) if add_intercept else xn
        yhat = float(x0 @ f["b"])
        se_mean = math.sqrt(mse * float(x0 @ f["xtx_inv"] @ x0))
        se_pred = math.sqrt(mse + se_mean ** 2)
        forecast = {"prediction": yhat, "std_error_forecast": se_pred, "std_error_mean": se_mean,
                    "prediction_interval": [yhat - tc * se_pred, yhat + tc * se_pred],
                    "confidence_interval_mean": [yhat - tc * se_mean, yhat + tc * se_mean]}
    key = lambda arr: dict(zip(labels, (float(v) for v in arr)))  # noqa: E731
    return {"coefficients": key(f["b"]), "std_errors": key(se), "t_stats": key(tstat),
            "p_values": key(2 * sps.t.sf(np.abs(tstat), df)),
            "confidence_intervals": {lb: [float(b - tc * s), float(b + tc * s)] for lb, b, s in zip(labels, f["b"], se)},
            "r_squared": r2,
            "adjusted_r_squared": None if r2 is None else 1 - (n - int(add_intercept)) / df * (1 - r2),
            "multiple_r": None if r2 is None else math.sqrt(max(r2, 0.0)), "f_statistic": fstat,
            "f_p_value": None if fstat is None else float(sps.f.sf(fstat, k, df)),
            "standard_error_of_estimate": math.sqrt(mse),
            "anova": {"rss": rss, "sse": f["sse"], "sst": sst, "df_regression": k, "df_residual": df,
                      "msr": rss / k if k else None, "mse": mse},
            "n": n, "k": k, "t_critical": tc, "durbin_watson": float(np.sum(np.diff(e) ** 2) / f["sse"]) if f["sse"] else None,
            "forecast": forecast, "fitted": (X @ f["b"]).tolist(), "residuals": e.tolist()}


@formula("quant", formula="F = (RSS/k)/(SSE/(n-k-1)); R^2 = RSS/SST; adj R^2 = 1 - (n-1)/(n-k-1)(1-R^2); "
                          "SEE = sqrt(MSE)", curriculum=f"{_L2R11}; {_L2R12}")
def regression_anova_from_summary(n: int, k: int, rss: float | None = None, sse: float | None = None,
                                  sst: float | None = None, r_squared: float | None = None,
                                  alpha: float = 0.05) -> dict:
    """Build the ANOVA table and F-test from summary sums of squares, as the curriculum does.

    Give any two of rss (regression/explained SS), sse (residual SS), sst (total SS), or r_squared with one of them.

    Args:
        n: observations.
        k: number of slope coefficients (independent variables).

    Returns:
        f_statistic, f_p_value, f_critical, reject_null, r_squared, adjusted_r_squared,
        standard_error_of_estimate, multiple_r, table.

    Example:
        regression_anova_from_summary(n=1819, k=2, rss=2681.6482, sse=2236.2820) -> F 1088.83
        (CFA L2 R12 Section 2.3)
    """
    if int(n) != n or int(k) != k or k < 1 or n <= k + 1:
        raise ValueError("need integers k >= 1 and n > k + 1")
    n, k = int(n), int(k)
    vals = {nm: None if v is None else _num(nm, v) for nm, v in
            (("rss", rss), ("sse", sse), ("sst", sst), ("r_squared", r_squared))}
    rss, sse, sst, r2 = vals["rss"], vals["sse"], vals["sst"], vals["r_squared"]
    if r2 is not None:
        if sst is None and rss is not None:
            sst = rss / r2
        elif sst is None and sse is not None:
            sst = sse / (1 - r2)
        if sst is not None:
            rss = r2 * sst if rss is None else rss
    if sst is None and rss is not None and sse is not None:
        sst = rss + sse
    if sst is not None and rss is None and sse is not None:
        rss = sst - sse
    if sst is not None and sse is None and rss is not None:
        sse = sst - rss
    if None in (rss, sse, sst) or sse <= 0 or sst <= 0:
        raise ValueError("give two of rss, sse, sst (or r_squared with one of them); sums of squares must be positive")
    dfr = n - k - 1
    msr, mse = rss / k, sse / dfr
    fstat = msr / mse
    r2 = rss / sst
    fc = float(sps.f.ppf(1 - _num("alpha", alpha), k, dfr))
    return {"f_statistic": fstat, "f_p_value": float(sps.f.sf(fstat, k, dfr)), "f_critical": fc,
            "reject_null": fstat > fc, "r_squared": r2, "adjusted_r_squared": 1 - (n - 1) / dfr * (1 - r2),
            "standard_error_of_estimate": math.sqrt(mse), "multiple_r": math.sqrt(max(r2, 0.0)),
            "table": {"regression": {"df": k, "ss": rss, "ms": msr}, "residual": {"df": dfr, "ss": sse, "ms": mse},
                      "total": {"df": n - 1, "ss": sst}}}


@formula("quant", formula="t = (b_hat - b)/s_b_hat, df = n - k - 1; CI = b_hat +/- t_c s_b_hat", curriculum=_L2R11)
def regression_coefficient_t_test(coefficient: float, std_error: float, df: int, hypothesized_value: float = 0.0,
                                  alpha: float = 0.05, tail: str = "two") -> dict:
    """t-test and confidence interval for one regression coefficient from reported output.

    Args:
        coefficient: estimated coefficient.
        std_error: its standard error.
        df: residual degrees of freedom, n - k - 1.
        hypothesized_value: value under H0 (e.g. 1 for "beta = 1").

    Returns:
        statistic, df, critical_values, p_value, reject_null, decision, confidence_interval.

    Example:
        regression_coefficient_t_test(coefficient=1.1958, std_error=0.2354, df=58, hypothesized_value=1)
        -> t 0.8318, fail to reject (CFA L2 R11 Example 14)
    """
    b, s = _num("coefficient", coefficient), _num("std_error", std_error)
    if s <= 0 or int(df) != df or df < 1:
        raise ValueError("std_error must be positive and df a positive integer")
    tc = float(sps.t.ppf(1 - _num("alpha", alpha) / 2, int(df)))
    return _decide((b - _num("hypothesized_value", hypothesized_value)) / s, sps.t(int(df)), alpha, tail,
                   df=int(df), confidence_interval=[b - tc * s, b + tc * s])


@formula("quant", formula="s_f^2 = s^2 [1 + 1/n + (X - Xbar)^2/((n-1) s_x^2)]; Y_hat +/- t_c s_f", curriculum=_L2R11)
def simple_regression_prediction_interval(intercept: float, slope: float, x: float, standard_error_of_estimate: float,
                                          n: int, x_mean: float, x_variance: float, confidence: float = 0.95) -> dict:
    """Prediction interval for Y in a one-variable regression, from summary statistics.

    Args:
        intercept, slope: estimated b0 and b1.
        x: value of X to forecast at.
        standard_error_of_estimate: s (SEE).
        n: observations used in the regression.
        x_mean: sample mean of X.
        x_variance: sample variance of X.
        confidence: e.g. 0.95.

    Returns:
        prediction, lower, upper, std_error_forecast, forecast_variance, t_critical, df.

    Example:
        simple_regression_prediction_interval(1.3478, 30.0169, 0.10, 0.7422, 9, 0.0647, 0.004641)
        -> 4.3495, interval 2.4715 to 6.2275 (CFA L2 R11 Example 18)
    """
    n = _n("n", n, 3)
    s, xv = _num("standard_error_of_estimate", standard_error_of_estimate), _num("x_variance", x_variance)
    if s <= 0 or xv <= 0:
        raise ValueError("standard_error_of_estimate and x_variance must be positive")
    xx = _num("x", x)
    yhat = _num("intercept", intercept) + _num("slope", slope) * xx
    var = s * s * (1 + 1 / n + (xx - _num("x_mean", x_mean)) ** 2 / ((n - 1) * xv))
    tc = float(sps.t.ppf((1 + _num("confidence", confidence)) / 2, n - 2))
    sf = math.sqrt(var)
    return {"prediction": yhat, "lower": yhat - tc * sf, "upper": yhat + tc * sf, "std_error_forecast": sf,
            "forecast_variance": var, "t_critical": tc, "df": n - 2}


# ---------------------------------------------------------------- diagnostics

@formula("quant", formula="DW = sum_{t=2..T}(e_t - e_(t-1))^2 / sum e_t^2 ~ 2(1 - r)", curriculum=_L2R12)
def durbin_watson_statistic(residuals: list) -> dict:
    """Durbin-Watson statistic for first-order serial correlation of regression residuals.

    Compare with tabled d_l and d_u: DW < d_l rejects no positive serial correlation; between them is
    inconclusive. Not valid when a lagged dependent variable is a regressor (use autocorrelation_t_test).

    Returns:
        durbin_watson, implied_autocorrelation (1 - DW/2).

    Example:
        durbin_watson_statistic([1, -1, 1, -1]) -> 3.0
    """
    e = _vec("residuals", residuals, 2)
    ss = float(e @ e)
    if ss == 0:
        raise ValueError("residuals are all zero")
    dw = float(np.sum(np.diff(e) ** 2) / ss)
    return {"durbin_watson": dw, "implied_autocorrelation": 1 - dw / 2}


@formula("quant", formula="BP = n R^2 (squared residuals on X) ~ chi^2(k)", curriculum=_L2R12)
def breusch_pagan_test(residuals: list | None = None, x=None, n: int | None = None, r_squared_aux: float | None = None,
                       k: int | None = None, alpha: float = 0.05) -> dict:
    """Breusch-Pagan test for conditional heteroskedasticity (one-tailed chi-square).

    Give residuals with the original regressors x, or the summary n, r_squared_aux (R^2 of squared
    residuals on the regressors) and k (number of regressors).

    Returns:
        statistic (n R^2), df, critical_values, p_value, reject_null, decision, r_squared_aux.

    Example:
        breusch_pagan_test(n=137, r_squared_aux=0.1651, k=1) -> 22.619 > 3.841, reject (CFA L2 R12 Example 8)
    """
    if residuals is not None:
        if x is None:
            raise ValueError("give the regressors x with the residuals")
        e2 = _vec("residuals", residuals, 3) ** 2
        aux = ols_regression(e2.tolist(), x)
        n, k, r2 = e2.size, aux["k"], aux["r_squared"]
    else:
        if None in (n, r_squared_aux, k):
            raise ValueError("give residuals and x, or n, r_squared_aux and k")
        n, k, r2 = _n("n", n, 1), _n("k", k, 1), _num("r_squared_aux", r_squared_aux)
    return _decide(n * r2, sps.chi2(k), alpha, "upper", df=k, r_squared_aux=r2)


@formula("quant", formula="VIF_j = 1/(1 - R_j^2), R_j^2 from regressing X_j on the other regressors",
         curriculum=f"{_L2R12} (multicollinearity)")
def variance_inflation_factors(x, names: list | None = None) -> dict:
    """Variance inflation factor of each regressor (multicollinearity diagnostic; VIF > 5-10 is a warning).

    Args:
        x: regressors (dict of columns, or list of rows with names), at least two.

    Returns:
        vif, r_squared (auxiliary R^2 of each regressor on the others), max_vif.

    Example:
        variance_inflation_factors({"a": [1, 2, 3, 4, 5], "b": [2, 1, 4, 3, 5]}) -> VIF = 1/(1 - r^2) each
    """
    m, names = _regressors(x, names)
    if m.shape[1] < 2:
        raise ValueError("VIF needs at least two regressors")
    vif, r2s = {}, {}
    for j, nm in enumerate(names):
        others = {names[i]: m[:, i].tolist() for i in range(m.shape[1]) if i != j}
        r2 = ols_regression(m[:, j].tolist(), others)["r_squared"]
        r2s[nm], vif[nm] = r2, math.inf if r2 >= 1 else 1 / (1 - r2)
    return {"vif": vif, "r_squared": r2s, "max_vif": max(vif.values())}


@formula("quant", formula="n categories -> n - 1 dummies (0/1); omitted category = intercept", curriculum=_L2R12)
def make_dummy_variables(categories: list, base=None) -> dict:
    """Turn a categorical variable into n - 1 dummy columns (avoids the dummy-variable trap).

    Args:
        categories: category label of each observation (e.g. month names).
        base: the omitted category (its mean is the intercept); default the first label seen.

    Returns:
        dummies ({label: 0/1 column}, ready for ols_regression's dict x), base, levels.

    Example:
        make_dummy_variables(["Jan", "Feb", "Dec"], base="Dec") -> dummies for Jan and Feb
    """
    labels = [str(c) for c in categories]
    if not labels:
        raise ValueError("categories is empty")
    levels = list(dict.fromkeys(labels))
    base = levels[0] if base is None else str(base)
    if base not in levels:
        raise ValueError(f"base {base!r} is not one of the categories")
    return {"dummies": {lv: [1.0 if c == lv else 0.0 for c in labels] for lv in levels if lv != base},
            "base": base, "levels": levels}


# ---------------------------------------------------------------- time series

@formula("quant", formula="linear: y_t = b0 + b1 t; log-linear: ln y_t = b0 + b1 t, growth = e^b1 - 1",
         curriculum=_L2R13)
def trend_regression(y: list, log_linear: bool = False, forecast_t: float | None = None) -> dict:
    """Fit a linear or log-linear time trend (t = 1, 2, ..., T) and optionally forecast period forecast_t.

    Returns:
        intercept, slope, growth_rate_per_period (log-linear only), forecast, t_stats, r_squared,
        standard_error_of_estimate, durbin_watson, residuals.

    Example:
        trend_regression(y=quarterly_sales, log_linear=True, forecast_t=61)
    """
    yv = _vec("y", y, 3)
    if log_linear:
        if np.any(yv <= 0):
            raise ValueError("log-linear trend needs positive y")
        yv = np.log(yv)
    t = np.arange(1, yv.size + 1, dtype=float)
    res = ols_regression(yv.tolist(), {"t": t.tolist()})
    b0, b1 = res["coefficients"]["intercept"], res["coefficients"]["t"]
    fc = None
    if forecast_t is not None:
        fc = trend_forecast(b0, b1, forecast_t, log_linear)["forecast"]
    return {"intercept": b0, "slope": b1, "growth_rate_per_period": math.expm1(b1) if log_linear else None,
            "forecast": fc, "t_stats": res["t_stats"], "r_squared": res["r_squared"],
            "standard_error_of_estimate": res["standard_error_of_estimate"], "durbin_watson": res["durbin_watson"],
            "residuals": res["residuals"]}


@formula("quant", formula="y_hat = b0 + b1 t; log-linear: y_hat = exp(b0 + b1 t)", curriculum=_L2R13)
def trend_forecast(intercept: float, slope: float, t: float, log_linear: bool = False) -> dict:
    """Forecast from estimated trend coefficients.

    Returns:
        forecast, fitted_index (b0 + b1 t, i.e. ln y for a log-linear model).

    Example:
        trend_forecast(intercept=5.5529, slope=0.0609, t=61, log_linear=True) -> 10,585.6 (CFA L2 R13 Example 3)
    """
    v = _num("intercept", intercept) + _num("slope", slope) * _num("t", t)
    return {"forecast": math.exp(v) if log_linear else v, "fitted_index": v}


@formula("quant", formula="r_k = sum (e_t - ebar)(e_(t-k) - ebar)/sum (e_t - ebar)^2; SE = 1/sqrt(T); t = r_k/SE",
         curriculum=_L2R13)
def autocorrelation_t_test(residuals: list | None = None, max_lag: int = 4, autocorrelations: list | None = None,
                           n: int | None = None, n_parameters: int = 2, alpha: float = 0.05) -> dict:
    """t-tests of residual autocorrelations (model specification check for AR models).

    Give the residuals (autocorrelations up to max_lag are computed), or reported autocorrelations with n.

    Args:
        n_parameters: parameters in the fitted model; critical value uses df = n - n_parameters.

    Returns:
        lags (list of {lag, autocorrelation, std_error, t_stat, significant}), std_error, t_critical,
        any_significant.

    Example:
        autocorrelation_t_test(autocorrelations=[...0.8739 at lag 12], n=347) -> t 16.28 at lag 12
        (CFA L2 R13 Example 16)
    """
    if residuals is not None:
        e = _vec("residuals", residuals, 3)
        n = e.size
        d = e - e.mean()
        den = float(d @ d)
        if den == 0:
            raise ValueError("residuals have zero variance")
        L = _n("max_lag", max_lag, 1)
        if not 1 <= L < n:
            raise ValueError("max_lag must be between 1 and n - 1")
        rho = [float(d[k:] @ d[:-k] / den) for k in range(1, L + 1)]
    else:
        if autocorrelations is None or n is None:
            raise ValueError("give residuals, or autocorrelations and n")
        rho, n = [float(_num("autocorrelations", r)) for r in autocorrelations], _n("n", n, 2)
    se = 1 / math.sqrt(n)
    dfree = n - _n("n_parameters", n_parameters, 0)
    if dfree < 1:
        raise ValueError("n must exceed n_parameters")
    tc = float(sps.t.ppf(1 - _num("alpha", alpha) / 2, dfree))
    rows = [{"lag": k + 1, "autocorrelation": r, "std_error": se, "t_stat": r / se, "significant": abs(r / se) > tc}
            for k, r in enumerate(rho)]
    return {"lags": rows, "std_error": se, "t_critical": tc, "any_significant": any(r["significant"] for r in rows)}


@formula("quant", formula="x_t = b0 + sum_j b_j x_(t - lag_j) + e_t; mean-reverting level = b0/(1 - sum b_j)",
         curriculum=_L2R13)
def autoregressive_model_fit(series: list, lags: list | None = None, autocorrelation_lags: int | None = None,
                             alpha: float = 0.05) -> dict:
    """Fit an autoregressive model by OLS (AR(1) by default; e.g. lags=[1, 4] adds a quarterly seasonal lag).

    Returns:
        intercept, coefficients ({"lag_k": b}), std_errors, t_stats, mean_reverting_level, r_squared,
        standard_error_of_estimate, residual_autocorrelations (autocorrelation_t_test output), observations.

    Example:
        autoregressive_model_fit(series=gross_margins, lags=[1]) -> b0, b1 and b0/(1 - b1)
    """
    x = _vec("series", series, 4)
    lags = sorted({_n("lags", v, 1) for v in (lags or [1])})
    p = lags[-1]
    if x.size - p < len(lags) + 2:
        raise ValueError("series too short for these lags")
    cols = {f"lag_{L}": x[p - L:x.size - L].tolist() for L in lags}
    res = ols_regression(x[p:].tolist(), cols, alpha=alpha)
    b0 = res["coefficients"]["intercept"]
    bsum = sum(res["coefficients"][c] for c in cols)
    ac = autocorrelation_t_test(res["residuals"], max_lag=autocorrelation_lags or max(4, p),
                                n_parameters=len(lags) + 1, alpha=alpha)
    return {"intercept": b0, "coefficients": {c: res["coefficients"][c] for c in cols},
            "std_errors": res["std_errors"], "t_stats": res["t_stats"],
            "mean_reverting_level": b0 / (1 - bsum) if bsum != 1 else None, "r_squared": res["r_squared"],
            "standard_error_of_estimate": res["standard_error_of_estimate"], "residual_autocorrelations": ac,
            "observations": len(res["residuals"])}


@formula("quant", formula="x_hat_(t+1) = b0 + sum_j b_j x_(t+1-lag_j); chain rule for later steps", curriculum=_L2R13)
def autoregressive_forecast(intercept: float, coefficients: list, history: list, lags: list | None = None,
                            steps: int = 1) -> dict:
    """Multi-step AR forecasts by the chain rule of forecasting.

    Args:
        intercept: b0.
        coefficients: b_j for each lag, same order as lags.
        history: past values in time order, most recent last; at least max(lags) values.
        lags: lag of each coefficient (default 1, 2, ..., p); e.g. [1, 4] for AR(1) plus a quarterly seasonal lag.
        steps: number of periods ahead.

    Returns:
        forecasts (one per step), mean_reverting_level.

    Example:
        autoregressive_forecast(intercept=0.0834, coefficients=[0.8665], history=[0.65], steps=2)
        -> [0.6467, 0.6438] (CFA L2 R13 Section 4.4)
    """
    b0 = _num("intercept", intercept)
    b = _vec("coefficients", coefficients)
    lags = [_n("lags", v, 1) for v in (lags or range(1, b.size + 1))]
    if len(lags) != b.size:
        raise ValueError("one positive lag per coefficient")
    h = _vec("history", history).tolist()
    if len(h) < max(lags):
        raise ValueError(f"history needs at least {max(lags)} values")
    if int(steps) != steps or steps < 1:
        raise ValueError("steps must be a positive integer")
    out = []
    for _ in range(int(steps)):
        nxt = float(b0 + sum(bj * h[-L] for bj, L in zip(b, lags)))
        out.append(nxt)
        h.append(nxt)
    s = float(b.sum())
    return {"forecasts": out, "mean_reverting_level": b0 / (1 - s) if s != 1 else None}


@formula("quant", formula="RMSE = sqrt(mean((actual - forecast)^2))", curriculum=_L2R13)
def forecast_rmse(actual: list, forecast: list) -> dict:
    """Root mean squared error of out-of-sample forecasts (lower RMSE = more accurate model).

    Returns:
        rmse, mean_squared_error, mean_error, errors (actual - forecast).

    Example:
        forecast_rmse([0.1308, 0.0345, -0.3557, -0.0954], [0.1357, 0.1299, 0.1271, 0.1259]) -> 0.2698
        (CFA L2 R13 Problem 9)
    """
    a, f = _vec("actual", actual), _vec("forecast", forecast)
    if a.size != f.size:
        raise ValueError("actual and forecast must have the same length")
    e = a - f
    mse = float(np.mean(e ** 2))
    return {"rmse": math.sqrt(mse), "mean_squared_error": mse, "mean_error": float(e.mean()), "errors": e.tolist()}


@formula("quant", formula="e_t^2 = a0 + a1 e_(t-1)^2 + u_t; ARCH(1) if a1 significant; "
                          "sigma^2_(t+1) = a0 + a1 e_t^2", curriculum=_L2R13)
def arch1_test(residuals: list, alpha: float = 0.05) -> dict:
    """ARCH(1) test: regress squared residuals on their first lag; a significant a1 means ARCH errors.

    Returns:
        statistic (t on a1), df, critical_values, p_value, reject_null, decision, a0, a1,
        next_period_variance (a0 + a1 e_T^2).

    Example:
        arch1_test(residuals=inflation_model_residuals)
    """
    e2 = _vec("residuals", residuals, 5) ** 2
    res = ols_regression(e2[1:].tolist(), {"lag_1": e2[:-1].tolist()})
    a0, a1 = res["coefficients"]["intercept"], res["coefficients"]["lag_1"]
    df = res["anova"]["df_residual"]
    return _decide(res["t_stats"]["lag_1"], sps.t(df), alpha, "two", df=df, a0=a0, a1=a1,
                   next_period_variance=a0 + a1 * float(e2[-1]))
