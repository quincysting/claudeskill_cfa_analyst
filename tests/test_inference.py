"""Tests for cfa_calc.inference. Golden values from the CFA curriculum (KB = E:/CLAUDE/CFA/MD).

Curriculum critical values come from printed tables (2-3 decimals); scipy gives exact values, so those
comparisons use table-rounding tolerances.
"""
import math

import pytest

from cfa_calc import inference as inf

R9 = "KB Level-1/02-Quantitative-Methods/Curriculum-2014/SS03-R09-common-probability-distributions.md"
R10 = "KB Level-1/02-Quantitative-Methods/Curriculum-2014/SS03-R10-sampling-and-estimation.md"
R11 = "KB Level-1/02-Quantitative-Methods/Curriculum-2014/SS03-R11-hypothesis-testing.md"
L2R11 = "KB Level-2/02-Quantitative-Methods/Curriculum-2014/SS03-R11-correlation-and-regression.md"
approx = pytest.approx


# ---- distributions (R9)
def test_normal_probability_example_8():
    up = inf.normal_probability(0.12, 0.22, lower=0.20)  # R9 p528 Ex 8: z = 0.363636, ~36% (table at z=0.36: 0.3594)
    assert up["z_lower"] == approx(0.363636, abs=1e-6) and up["probability"] == approx(0.3594, abs=0.002)
    assert inf.normal_probability(0.12, 0.22, lower=0.12, upper=0.20)["probability"] == approx(0.1406, abs=0.002)
    lo = inf.normal_probability(0.12, 0.22, upper=0.055)  # z = -0.295455
    assert lo["z_upper"] == approx(-0.295455, abs=1e-6) and lo["probability"] == approx(0.3838, abs=5e-4)


def test_normal_quantile():
    assert inf.normal_quantile(0.90)["z"] == approx(1.282, abs=5e-4)  # R9 p527
    assert inf.normal_quantile(0.99)["z"] == approx(2.326, abs=1e-3)
    assert inf.normal_quantile(0.975, mean=0.1, std_dev=0.2)["value"] == approx(0.1 + 1.959964 * 0.2, abs=1e-6)
    z = inf.normal_quantile(0.3)["z"]
    assert inf.normal_probability(upper=z)["probability"] == approx(0.3)


def test_lognormal_moments():
    m = inf.lognormal_moments(0.05, 0.2)
    assert m["mean"] == approx(math.exp(0.05 + 0.02))  # R9 p533: exp(mu + sigma^2/2)
    assert m["variance"] == approx(math.exp(0.1 + 0.04) * (math.exp(0.04) - 1))
    assert m["mode"] < m["median"] < m["mean"]


def test_continuously_compounded_return():
    assert inf.continuously_compounded_return(begin_price=30, end_price=34.5)["cc_return"] == approx(0.139762, abs=1e-6)  # R9 p534
    assert inf.continuously_compounded_return(cc_return=0.139762)["holding_period_return"] == approx(0.15, abs=1e-6)
    with pytest.raises(ValueError):
        inf.continuously_compounded_return()


def test_volatility_from_prices_michelin():
    out = inf.volatility_from_prices([25.20, 25.21, 25.52, 26.10, 26.14])  # R9 p536 Ex 10
    assert out["period_std_dev"] == approx(0.010354, abs=1e-6)
    assert out["period_mean"] == approx(0.009156, abs=1e-6)
    assert out["annualized_volatility"] == approx(0.163711, abs=2e-5)  # text multiplies the rounded s by sqrt(250)


# ---- estimation (R10)
def test_confidence_intervals():
    z = inf.confidence_interval_mean(0.45, 0.30, 100, 0.90, "z")  # R10 p572 Ex 4 (table z = 1.65)
    assert (z["lower"], z["upper"]) == (approx(0.4005, abs=2e-4), approx(0.4995, abs=2e-4))
    t = inf.confidence_interval_mean(0.45, 0.30, 100, 0.90, "t")  # R10 p574 Ex 5 (table t = 1.66)
    assert t["reliability_factor"] == approx(1.66, abs=0.005)
    m = inf.confidence_interval_mean(5.5, 10, 10)  # R10 p576 Ex 6
    assert m["reliability_factor"] == approx(2.262, abs=5e-4)
    assert (m["lower"], m["upper"]) == (approx(-1.65, abs=0.005), approx(12.65, abs=0.005))
    assert inf.standard_error_of_mean(0.30, 100)["standard_error"] == approx(0.03)


# ---- hypothesis tests (R11)
def test_t_test_single_mean_sendar():
    out = inf.t_test_mean(1.50, 1.10, 3.60, 24, alpha=0.10)  # R11 p608 Ex 1
    assert out["statistic"] == approx(0.544331, abs=1e-6)
    assert out["critical_values"][1] == approx(1.714, abs=5e-4)
    assert out["reject_null"] is False and out["df"] == 23


def test_z_tests():
    rp = inf.z_test_mean(5.1, 0, std_dev=18.6, n=103, tail="upper")  # R11 p605: z = 2.78 > 1.645
    assert rp["statistic"] == approx(2.78, abs=0.005) and rp["reject_null"]
    cp = inf.z_test_mean(0.39, 0, standard_error=0.1336, alpha=0.01)  # R11 p613 Ex 3
    assert cp["statistic"] == approx(2.92, abs=0.005) and cp["reject_null"]
    assert cp["critical_values"][1] == approx(2.575, abs=0.001)
    inferior = inf.z_test_mean(-0.57, 0, standard_error=0.38)
    assert inferior["statistic"] == approx(-1.50) and not inferior["reject_null"]


def test_two_sample_tests():
    pooled = inf.t_test_two_means_pooled(0.580, 4.598, 120, 1.470, 4.738, 120, alpha=0.10)  # R11 p616 Ex 4
    assert pooled["pooled_variance"] == approx(21.795124, abs=1e-6)
    assert pooled["statistic"] == approx(-1.477, abs=5e-4) and not pooled["reject_null"]
    une = inf.t_test_two_means_unequal(64.42, 14.03, 21, 55.75, 25.17, 64, alpha=0.10)  # R11 p617-618 Ex 5
    assert une["statistic"] == approx(1.975, abs=5e-4)
    assert une["df"] == approx(64.99, abs=0.005)
    assert une["reject_null"] and une["df_welch"] < une["df"]


def test_paired_t_test():
    out = inf.t_test_paired(3.06, 6.62, 50, alpha=0.01)  # R11 p621 Ex 6
    assert out["statistic"] == approx(3.2685, abs=1e-4) and out["reject_null"]
    assert out["critical_values"][1] == approx(2.68, abs=0.005)
    small = inf.t_test_paired(-0.65, 6.71, 24)  # R11 p620: t = -0.475
    assert small["statistic"] == approx(-0.475, abs=5e-4) and not small["reject_null"]
    raw = inf.t_test_paired(differences=[1.0, 2.0, 3.0, 4.0])
    assert raw["statistic"] == approx(2.5 / (math.sqrt(5 / 3) / 2))


def test_chi_square_and_f_tests():
    chi = inf.chi_square_test_variance(24, sample_std_dev=3.60, hypothesized_std_dev=4, tail="lower")  # R11 p624 Ex 7
    assert chi["statistic"] == approx(18.63) and chi["critical_values"][0] == approx(13.091, abs=5e-4)
    assert not chi["reject_null"]
    f1 = inf.f_test_variances(120, 120, variance1=18.776, variance2=13.097, alpha=0.01)  # R11 p626 Ex 8
    assert f1["statistic"] == approx(1.434, abs=5e-4) and not f1["reject_null"]
    assert f1["critical_values"][1] == approx(1.61, abs=0.005)
    f2 = inf.f_test_variances(12, 115, std_dev1=1.178, std_dev2=0.786, tail="upper")  # R11 p627 Ex 9
    assert f2["statistic"] == approx(2.25, abs=0.005) and f2["reject_null"]
    with pytest.raises(ValueError):
        inf.f_test_variances(10, 10, variance1=1)


def test_correlation_tests():
    assert inf.correlation_t_test(0.8702, 6)["statistic"] == approx(3.532, abs=5e-4)  # L2R11 p311 Ex 7
    assert inf.correlation_t_test(0.2860, 120)["statistic"] == approx(3.242, abs=5e-4)  # L2R11 p312 Ex 8
    big = inf.correlation_t_test(0.1119, 924, alpha=0.01)  # L2R11 p312 Ex 9
    assert big["statistic"] == approx(3.4193, abs=1e-4) and big["reject_null"]


def test_spearman_rank_correlation():
    x = [-1.08, -0.96, -1.13, -1.16, -0.91, -1.08, -1.18, -1.00, -1.06, -1.00]  # R11 p630 Table 10
    y = [1.34, 0.92, 1.02, 1.45, 1.35, 0.50, 1.00, 1.50, 1.45, 1.50]
    out = inf.spearman_rank_correlation(x, y)
    assert out["sum_d_squared"] == approx(123) and out["spearman_r"] == approx(0.2545, abs=5e-5)


def test_decision_tails_consistent():
    two = inf.t_test_mean(1.0, 0, 2.0, 30)
    up = inf.t_test_mean(1.0, 0, 2.0, 30, tail="upper")
    assert two["p_value"] == approx(2 * up["p_value"])
    assert inf.t_test_mean(1.0, 0, 2.0, 30, tail="lower")["p_value"] == approx(1 - up["p_value"])
    with pytest.raises(ValueError):
        inf.t_test_mean(1.0, 0, 2.0, 30, tail="left")


# ---- Monte Carlo
def test_monte_carlo_seeded_and_converges():
    a = inf.monte_carlo_terminal_wealth(100, 0.005, 0.04, 12, trials=20000, seed=7)
    b = inf.monte_carlo_terminal_wealth(100, 0.005, 0.04, 12, trials=20000, seed=7)
    assert a == b
    analytic = 100 * math.exp(12 * (0.005 + 0.04 ** 2 / 2))  # lognormal mean
    assert a["mean_terminal_value"] == approx(analytic, rel=0.005)
    n = inf.monte_carlo_terminal_wealth(100, 0.01, 0.05, 1, trials=50000, model="normal", threshold=100)
    assert n["mean_terminal_value"] == approx(101, rel=0.002)
    assert n["prob_below_threshold"] == approx(inf.normal_probability(0.01, 0.05, upper=0)["probability"], abs=0.01)
