"""Tests for cfa_calc.regression. Golden values from the CFA curriculum (KB = E:/CLAUDE/CFA/MD)."""
import math

import numpy as np
import pytest

from cfa_calc import regression as rg

L2R11 = "KB Level-2/02-Quantitative-Methods/Curriculum-2014/SS03-R11-correlation-and-regression.md"
L2R12 = "KB Level-2/02-Quantitative-Methods/Curriculum-2014/SS03-R12-multiple-regression-and-issues-in-regression-analysis.md"
L2R13 = "KB Level-2/02-Quantitative-Methods/Curriculum-2014/SS03-R13-time-series-analysis.md"
approx = pytest.approx

# L2R11 p295 Table 1: long-term money supply growth (X) and inflation (Y), six countries.
# The text works with intermediate values rounded to 6 decimals, so tolerances are ~1e-3.
MONEY = [0.1166, 0.0915, 0.1060, 0.0575, 0.1258, 0.0634]
INFL = [0.0676, 0.0519, 0.0815, 0.0339, 0.0758, 0.0509]


def test_sample_correlation_golden():
    out = rg.sample_correlation(MONEY, INFL)  # L2R11 p300, p311
    assert out["correlation"] == approx(0.8702, abs=5e-4)
    assert out["covariance"] == approx(0.000437, abs=1e-6)
    assert (out["std_dev_x"], out["std_dev_y"]) == (approx(0.028071, abs=1e-5), approx(0.017889, abs=1e-5))
    assert out["t_statistic"] == approx(3.532, abs=0.01) and out["reject_null"]


def test_simple_ols_golden():
    r = rg.ols_regression(INFL, MONEY, names=["money"])  # L2R11 p319-323 Ex 10-13
    assert r["coefficients"]["intercept"] == approx(0.0084, abs=1e-4)
    assert r["coefficients"]["money"] == approx(0.5545, abs=1e-4)
    assert r["standard_error_of_estimate"] == approx(0.0098, abs=1e-4)  # "about 0.98 percent"
    assert r["r_squared"] == approx(0.7572, abs=1e-3)
    assert r["anova"]["sse"] == approx(0.000386, abs=2e-6)
    # consistency: R^2 = r^2, F = t^2, multiple R = |r|
    corr = rg.sample_correlation(MONEY, INFL)
    assert r["r_squared"] == approx(corr["correlation"] ** 2)
    assert r["f_statistic"] == approx(r["t_stats"]["money"] ** 2)
    assert r["t_stats"]["money"] == approx(corr["t_statistic"])


def test_ols_forecast_matches_summary_formula():
    r = rg.ols_regression(INFL, MONEY, x_new=[0.10])
    b0, b1 = r["coefficients"]["intercept"], r["coefficients"]["x1"]
    s = rg.simple_regression_prediction_interval(b0, b1, 0.10, r["standard_error_of_estimate"], 6,
                                                 float(np.mean(MONEY)), float(np.var(MONEY, ddof=1)))
    assert r["forecast"]["prediction"] == approx(s["prediction"])
    assert r["forecast"]["prediction_interval"] == [approx(s["lower"]), approx(s["upper"])]
    ci, pi = r["forecast"]["confidence_interval_mean"], r["forecast"]["prediction_interval"]
    assert pi[0] < ci[0] < ci[1] < pi[1]


def test_prediction_interval_golden():
    out = rg.simple_regression_prediction_interval(1.3478, 30.0169, 0.10, 0.7422, 9, 0.0647, 0.004641)  # L2R11 p337 Ex 18
    assert out["prediction"] == approx(4.3495, abs=1e-4)
    assert out["forecast_variance"] == approx(0.630556, abs=1e-5)
    assert out["t_critical"] == approx(2.365, abs=5e-4)
    assert (out["lower"], out["upper"]) == (approx(2.4715, abs=5e-4), approx(6.2275, abs=5e-4))


def test_multiple_regression_matches_numpy():
    rng = np.random.default_rng(3)
    X = rng.normal(size=(40, 3))
    y = 1.0 + X @ np.array([0.5, -2.0, 0.0]) + rng.normal(scale=0.3, size=40)
    r = rg.ols_regression(y.tolist(), {"a": X[:, 0].tolist(), "b": X[:, 1].tolist(), "c": X[:, 2].tolist()})
    beta = np.linalg.lstsq(np.column_stack([np.ones(40), X]), y, rcond=None)[0]
    assert [r["coefficients"][k] for k in ("intercept", "a", "b", "c")] == [approx(v) for v in beta]
    n, k = 40, 3
    assert r["adjusted_r_squared"] == approx(1 - (n - 1) / (n - k - 1) * (1 - r["r_squared"]))
    a = r["anova"]
    assert a["rss"] + a["sse"] == approx(a["sst"])
    assert r["f_statistic"] == approx((a["rss"] / k) / (a["sse"] / (n - k - 1)))
    rows = rg.ols_regression(y.tolist(), X.tolist())  # list-of-rows input gives the same fit
    assert rows["coefficients"]["x2"] == approx(r["coefficients"]["b"])


def test_anova_from_summary_golden():
    bid_ask = rg.regression_anova_from_summary(1819, 2, rss=2681.6482, sse=2236.2820)  # L2R12 p376
    assert bid_ask["f_statistic"] == approx(1088.8325, abs=1e-3) and bid_ask["reject_null"]
    assert bid_ask["table"]["residual"]["ms"] == approx(1.2314, abs=1e-4)
    dreyfus = rg.regression_anova_from_summary(60, 1, rss=0.1093, sse=0.0176)  # L2R11 p335 Ex 17
    assert dreyfus["f_statistic"] == approx(360.19, abs=0.005)
    months = rg.regression_anova_from_summary(288, 11, rss=0.0543, sse=0.8924)  # L2R12 p379 Table 4
    assert months["f_statistic"] == approx(1.53, abs=0.005)
    assert months["f_p_value"] == approx(0.1213, abs=5e-4) and not months["reject_null"]
    alt = rg.regression_anova_from_summary(1819, 2, rss=2681.6482, r_squared=2681.6482 / 4917.9302)
    assert alt["f_statistic"] == approx(bid_ask["f_statistic"])
    with pytest.raises(ValueError):
        rg.regression_anova_from_summary(100, 2, rss=10)


def test_coefficient_t_test_golden():
    gm = rg.regression_coefficient_t_test(1.1958, 0.2354, 58, hypothesized_value=1)  # L2R11 p328 Ex 14
    assert gm["statistic"] == approx(0.8318, abs=1e-4) and not gm["reject_null"]
    assert gm["critical_values"][1] == approx(2.00, abs=0.005)
    assert rg.regression_coefficient_t_test(1.1958, 0.2354, 58)["statistic"] == approx(5.0798, abs=1e-3)


def test_durbin_watson_and_breusch_pagan():
    assert rg.durbin_watson_statistic([1, -1, 1, -1])["durbin_watson"] == approx(3.0)
    assert rg.durbin_watson_statistic([1, 1, 1, 1])["durbin_watson"] == approx(0.0)
    bp = rg.breusch_pagan_test(n=137, r_squared_aux=0.1651, k=1)  # L2R12 p386 Ex 8
    assert bp["statistic"] == approx(22.619, abs=1e-3)
    assert bp["critical_values"][0] == approx(3.84, abs=0.005) and bp["reject_null"]
    rng = np.random.default_rng(11)
    x = rng.uniform(1, 10, 300)
    e = rng.normal(scale=x)  # variance rises with x
    raw = rg.breusch_pagan_test(residuals=e.tolist(), x=x.tolist())
    assert raw["reject_null"] and raw["df"] == 1


def test_vif():
    out = rg.variance_inflation_factors({"a": [1, 2, 3, 4, 5], "b": [2, 1, 4, 3, 5]})
    r = np.corrcoef([1, 2, 3, 4, 5], [2, 1, 4, 3, 5])[0, 1]
    assert out["vif"]["a"] == approx(1 / (1 - r ** 2)) and out["vif"]["b"] == approx(out["vif"]["a"])


def test_dummy_variables_recover_group_means():
    cats = ["Q1", "Q2", "Q3", "Q4"] * 5
    y = [1.0 + {"Q1": 0, "Q2": 0.5, "Q3": -0.2, "Q4": 0.3}[c] + 0.01 * i for i, c in enumerate(cats)]
    d = rg.make_dummy_variables(cats, base="Q4")
    assert set(d["dummies"]) == {"Q1", "Q2", "Q3"}
    r = rg.ols_regression(y, d["dummies"])
    means = {c: np.mean([v for v, cc in zip(y, cats) if cc == c]) for c in set(cats)}
    assert r["coefficients"]["intercept"] == approx(means["Q4"])  # intercept = omitted-category mean
    assert r["coefficients"]["Q2"] == approx(means["Q2"] - means["Q4"])
    with pytest.raises(ValueError):  # dummy-variable trap: all four dummies plus an intercept
        rg.ols_regression(y, {c: [1.0 if cc == c else 0.0 for cc in cats] for c in ("Q1", "Q2", "Q3", "Q4")})


# ---- time series (L2R13)
def test_trend_forecast_golden():
    # L2R13 p454: text gives ln y = 9.2673 and $10,585.63m from unrounded coefficients; the printed
    # coefficients give 9.2678 -> 10,591.4, within 0.1%.
    assert rg.trend_forecast(5.5529, 0.0609, 61, log_linear=True)["forecast"] == approx(10_585.63, rel=1e-3)
    assert rg.trend_forecast(-1318.7729, 132.4005, 61)["forecast"] == approx(6_757.66, abs=0.01)
    assert rg.trend_forecast(4.1342, -0.0095, 192)["forecast"] == approx(2.3102, abs=1e-4)  # L2R13 p453


def test_trend_regression_recovers_known_trends():
    t = np.arange(1, 21)
    exp_series = np.exp(5 + 0.06 * t)
    out = rg.trend_regression(exp_series.tolist(), log_linear=True, forecast_t=21)
    assert out["intercept"] == approx(5) and out["slope"] == approx(0.06)
    assert out["growth_rate_per_period"] == approx(math.exp(0.06) - 1)
    assert out["forecast"] == approx(math.exp(5 + 0.06 * 21))
    lin = rg.trend_regression((3 + 2 * t + np.sin(t)).tolist(), forecast_t=25)
    assert lin["slope"] == approx(2, abs=0.05)


def test_ar_forecast_golden():
    out = rg.autoregressive_forecast(0.0834, [0.8665], [0.65], steps=2)  # L2R13 p460-461
    assert out["forecasts"] == [approx(0.6467, abs=1e-4), approx(0.6438, abs=1e-4)]  # text rounds 0.646625 up
    assert out["mean_reverting_level"] == approx(0.6250, abs=5e-4)
    seasonal = rg.autoregressive_forecast(0.0403, [-0.2955, 0.3896], [0.02, 0.0, 0.0, 0.01], lags=[1, 4])  # p484
    assert seasonal["forecasts"][0] == approx(0.0451, abs=5e-5)
    retail = rg.autoregressive_forecast(0.1516, [-0.0490, 0.8857], [0.10] + [0.0] * 10 + [0.05], lags=[1, 12])  # p486
    assert retail["forecasts"][0] == approx(0.2377, abs=5e-5)
    far = rg.autoregressive_forecast(0.0834, [0.8665], [0.65], steps=200)["forecasts"][-1]
    assert far == approx(out["mean_reverting_level"], abs=1e-9)  # chain rule converges to b0/(1 - b1)


def test_autoregressive_fit_recovers_ar1():
    rng = np.random.default_rng(5)
    x = [0.5]
    for _ in range(3000):
        x.append(0.2 + 0.6 * x[-1] + rng.normal(scale=0.1))
    fit = rg.autoregressive_model_fit(x)
    assert fit["coefficients"]["lag_1"] == approx(0.6, abs=0.03)
    assert fit["mean_reverting_level"] == approx(0.5, abs=0.02)
    assert not fit["residual_autocorrelations"]["lags"][0]["significant"]
    seas = rg.autoregressive_model_fit(x, lags=[1, 4])
    assert set(seas["coefficients"]) == {"lag_1", "lag_4"}


def test_autocorrelation_t_test_golden():
    rows = rg.autocorrelation_t_test(autocorrelations=[-0.0552, -0.1536], n=347)["lags"]  # L2R13 p485 Table 15
    assert rows[0]["std_error"] == approx(0.0537, abs=5e-5)
    assert rows[0]["t_stat"] == approx(-1.0288, abs=2e-3) and not rows[0]["significant"]
    assert rows[1]["t_stat"] == approx(-2.8619, abs=5e-3) and rows[1]["significant"]
    lag12 = rg.autocorrelation_t_test(autocorrelations=[0.8739], n=347)["lags"][0]
    assert lag12["t_stat"] == approx(16.28, abs=0.01)  # p484
    assert rg.autocorrelation_t_test(autocorrelations=[0.1], n=359)["std_error"] == approx(0.0528, abs=5e-5)  # p462
    alt = rg.autocorrelation_t_test(residuals=[1, -1, 1, -1, 1, -1, 1, -1], max_lag=1)["lags"][0]
    assert alt["autocorrelation"] == approx(-7 / 8)


def test_forecast_rmse_golden():
    out = rg.forecast_rmse([0.1308, 0.0345, -0.3557, -0.0954], [0.1357, 0.1299, 0.1271, 0.1259])  # L2R13 p503/p511
    assert out["rmse"] == approx(0.27, abs=0.005)


def test_arch1():
    rng = np.random.default_rng(9)
    e, prev = [], 0.1
    for _ in range(2000):
        prev = rng.normal() * math.sqrt(0.2 + 0.7 * prev ** 2)
        e.append(prev)
    assert rg.arch1_test(e)["reject_null"]
    assert not rg.arch1_test(rng.normal(size=2000).tolist())["reject_null"]


# ---------------------------------------------------------------- regressions found by independent verification

def test_ols_without_intercept_uses_uncentred_sst():
    # No intercept: SST = sum y^2, RSS = SST - SSE, F(k, n - k), uncentred R^2. Checked with numpy lstsq:
    # b = sum xy / sum x^2 = 119/91, SSE 113.4946, SST 269.11, R^2 0.5783, F(1,5) 6.856, p 0.0472.
    y = [9.0, 7.9, 7.1, 5.8, 4.9, 4.2]
    r = rg.ols_regression(y, [1, 2, 3, 4, 5, 6], add_intercept=False)
    assert r["coefficients"]["x1"] == approx(119 / 91)
    assert r["anova"]["sst"] == approx(269.11) and r["anova"]["sse"] == approx(113.494615, abs=1e-6)
    assert r["r_squared"] == approx(0.5783, abs=5e-5)
    assert r["f_statistic"] == approx(6.856, abs=5e-4) and r["anova"]["df_residual"] == 5
    assert r["f_p_value"] == approx(0.0472, abs=5e-5)
    assert r["adjusted_r_squared"] == approx(1 - 6 / 5 * (1 - r["r_squared"]))  # 1 - n/(n - k) (1 - R^2)
    assert r["f_statistic"] == approx(r["t_stats"]["x1"] ** 2)  # one regressor: F = t^2 holds again


@pytest.mark.parametrize("call", [
    lambda: rg.simple_regression_prediction_interval(1.3478, 30.0169, 0.10, 0.7422, 9.5, 0.0647, 0.004641),
    lambda: rg.breusch_pagan_test(n=137.5, r_squared_aux=0.1651, k=1),
    lambda: rg.autocorrelation_t_test(autocorrelations=[0.1], n=100.5),
    lambda: rg.autocorrelation_t_test(residuals=[0.1, -0.2, 0.3, -0.1, 0.2, -0.3], max_lag=2.5),
    lambda: rg.autoregressive_forecast(intercept=0.08, coefficients=[0.87], history=[0.65, 0.6], lags=[1.5]),
])
def test_fractional_counts_rejected(call):
    with pytest.raises(ValueError):
        call()
