"""Tests for cfa_calc.stats. Golden values from the CFA curriculum (KB = E:/CLAUDE/CFA/MD)."""
import math

import pytest

from cfa_calc import stats as st

R7 = "KB Level-1/02-Quantitative-Methods/Curriculum-2014/SS02-R07-statistical-concepts-and-market-returns.md"
R8 = "KB Level-1/02-Quantitative-Methods/Curriculum-2014/SS02-R08-probability-concepts.md"
R9 = "KB Level-1/02-Quantitative-Methods/Curriculum-2014/SS03-R09-common-probability-distributions.md"
approx = pytest.approx

SLASX = [16.2, 20.3, 9.3, -11.1, -17.0]  # R7 p384 Table 15
PRFDX5 = [9.2, 3.8, 13.1, 1.6, -13.0]
PRFDX10 = [14.8, 4.5, 33.3, 20.3, 28.8, 9.2, 3.8, 13.1, 1.6, -13.0]  # R7 p415 Table 28
# R7 p389-390 Table 17: dividend yields (%) of the DJ EuroSTOXX 50, ascending
DIV_YIELDS = [0, 0, 0, 0, 0.26, 1.09, 1.27, 1.33, 1.36, 1.39, 1.41, 1.51, 1.75, 1.81, 1.92, 2.01, 2.16, 2.27, 2.27,
              2.39, 2.49, 2.55, 2.60, 2.65, 2.65, 2.65, 2.95, 3.11, 3.31, 3.34, 3.53, 3.59, 3.66, 3.67, 3.68, 3.78,
              3.87, 3.88, 4.06, 4.27, 4.28, 4.45, 4.68, 5.13, 5.15, 5.66, 6.16, 6.43, 7.68, 8.14]


# ---- descriptive statistics (R7)
def test_descriptive_statistics_mutual_funds():
    a = st.descriptive_statistics(SLASX)
    assert a["mean"] == approx(3.54)  # R7 p385 Ex 7
    assert a["range"] == approx(37.3)  # R7 p396 Ex 10
    assert a["mean_absolute_deviation"] == approx(14.072)  # R7 p396 Ex 10 (14.1%)
    assert a["sample_variance"] == approx(277.643)  # R7 p401 Ex 12
    assert a["sample_std_dev"] == approx(16.66, abs=0.005)
    b = st.descriptive_statistics(PRFDX5)
    assert b["mean_absolute_deviation"] == approx(6.912)  # R7 p396 (6.9%)
    assert b["sample_variance"] == approx(99.758)  # R7 p402
    assert b["sample_std_dev"] == approx(9.99, abs=0.005)


def test_population_variance_and_pe_example():
    t = st.descriptive_statistics([23, 8, 11, 18, 56, 29, 23, 13, 16, 10])  # R7 p399 Ex 11
    assert t["mean"] == approx(20.7)
    assert t["population_variance"] == approx(178.41)
    assert t["population_std_dev"] == approx(13.357, abs=5e-4)
    pe = st.descriptive_statistics([16.73, 22.02, 29.30])  # R7 p397 (text rounds the mean to 22.68)
    assert pe["population_variance"] == approx(26.5542, abs=0.001)


def test_skewness_and_kurtosis_curriculum_formulas():
    d = st.descriptive_statistics(PRFDX10)  # R7 p417 Ex 16 (-0.05), p421 Ex 17 (0.05)
    assert d["sample_std_dev"] == approx(13.65, abs=0.005)
    assert d["skewness"] == approx(-0.05, abs=0.005)
    assert d["excess_kurtosis"] == approx(0.05, abs=0.01)
    sym = st.descriptive_statistics([1, 2, 3, 4, 5, 6, 7, 8, 9])  # R7 p414: symmetric data -> skew 0
    assert sym["skewness"] == approx(0, abs=1e-12)


def test_modes_and_median():
    d = st.descriptive_statistics([1, 2, 2, 3, 3, 4])
    assert d["modes"] == [2.0, 3.0] and d["median"] == 2.5
    assert st.descriptive_statistics([1, 2, 3])["modes"] == []


def test_mean_measures():
    # R7 p382 Ex 6: weighted portfolio return 1.5% (weights as percents are normalized)
    w = st.mean_measures([0.6, -9.3, -10.5, 6.0, 9.0, 10.2, 4.2], [34.6, 10.8, 6.4, 34.0, 1.3, 4.5, 8.4])
    assert w["weighted_mean"] == approx(1.5, abs=0.001)
    h = st.mean_measures([10, 15])  # R7 p387: cost averaging, harmonic mean 12
    assert h["harmonic_mean"] == approx(12)
    assert h["harmonic_mean"] < h["geometric_mean"] < h["arithmetic_mean"]


def test_geometric_mean_return():
    a = st.geometric_mean_return([r / 100 for r in SLASX])
    assert a["geometric_mean_return"] == approx(0.024270, abs=1e-6)  # R7 p385 Ex 7
    assert a["arithmetic_mean_return"] == approx(0.0354)
    b = st.geometric_mean_return([r / 100 for r in PRFDX5])
    assert b["geometric_mean_return"] == approx(0.025319, abs=1e-6)
    c = st.geometric_mean_return([1.0, -0.5])  # R7 p386 Ex 8: arithmetic 25%, geometric 0%
    assert c["geometric_mean_return"] == approx(0, abs=1e-12) and c["arithmetic_mean_return"] == approx(0.25)


def test_quantiles_curriculum_method():
    q = lambda p: st.quantile_curriculum(DIV_YIELDS, p)["quantile"]  # noqa: E731  R7 p390-392 Ex 9
    assert q(0.10) == approx(0.343, abs=1e-9)
    assert q(0.90) == approx(5.609, abs=1e-9)
    assert q(0.25) == approx(1.69, abs=1e-9)
    assert q(0.50) == approx(2.65)
    assert q(0.75) == approx(3.925, abs=1e-9)
    assert q(0.20) == approx(1.394, abs=1e-9)
    assert st.quantile_curriculum([1, 2, 3], 0.01)["quantile"] == 1


def test_semivariance_and_target_semivariance():
    s = st.semivariance(SLASX)  # R7 p403
    assert s["semivariance"] == approx(159.0553) and s["semideviation"] == approx(12.6, abs=0.05)
    t = st.semivariance(SLASX, target=10)  # R7 p404
    assert t["semivariance"] == approx(293.675) and t["semideviation"] == approx(17.14, abs=0.005)


def test_chebyshev():
    assert st.chebyshev_inequality(1.25)["min_proportion_within"] == approx(0.36)  # R7 p405 Table 23
    c = st.chebyshev_inequality(2, mean=0.97, std_dev=5.65)  # R7 p405 Ex 13
    assert c["min_proportion_within"] == approx(0.75)
    assert (c["lower"], c["upper"]) == (approx(-10.33), approx(12.27))
    with pytest.raises(ValueError):
        st.chebyshev_inequality(1)


def test_cv_and_sharpe():
    assert st.coefficient_of_variation(21.9, 12.3)["coefficient_of_variation"] == approx(1.780, abs=5e-4)  # R7 p408
    assert st.coefficient_of_variation(0.9, 3.8)["coefficient_of_variation"] == approx(0.237, abs=5e-4)
    assert st.sharpe_ratio_historical(12.3, 3.8, 21.9)["sharpe_ratio"] == approx(0.39, abs=0.005)  # R7 p409
    rf = sum([2.90, 3.90, 5.60, 5.21, 5.26, 4.86, 4.68, 5.89, 3.83, 1.65]) / 10  # R7 p411-412 Ex 15
    assert st.sharpe_ratio_historical(12.58, rf, 19.44)["sharpe_ratio"] == approx(0.42, abs=0.005)
    assert st.sharpe_ratio_historical(11.64, rf, 13.65)["sharpe_ratio"] == approx(0.53, abs=0.005)


def test_roy_safety_first():
    out = st.roy_safety_first_ratio([0.25, 0.11, 0.14], [0.27, 0.08, 0.20], 0.0375)  # R9 p530-531 Ex 9
    assert out["best_index"] == 1
    assert out["sf_ratios"] == [approx(0.787037, abs=1e-6), approx(0.90625), approx(0.5125)]
    assert out["shortfall_probabilities"][1] == approx(0.1824, abs=5e-5)
    single = st.roy_safety_first_ratio(0.14, 0.16, 0.02)  # R9 p530: portfolio 2 ratio 0.75, N(-0.75) = 0.227
    assert single["sf_ratios"][0] == approx(0.75) and single["shortfall_probabilities"][0] == approx(0.2266, abs=5e-4)


# ---- probability (R8)
def test_probability_rules_limit_orders():
    out = st.probability_rules(0.35, 0.25, p_a_and_b=0.25)  # R8 p458-459 Ex 3
    assert out["p_a_or_b"] == approx(0.35) and out["p_b_given_a"] == approx(5 / 7)
    assert st.probability_rules(0.45, 0.45, independent=True)["p_a_and_b"] == approx(0.2025)  # R8 p460 Ex 4
    with pytest.raises(ValueError):
        st.probability_rules(0.3, 0.3, p_a_and_b=0.5)


def test_total_probability_and_expectation():
    assert st.total_probability([0.75, 0.20, 0.05], [0.45, 0.30, 0.25])["total"] == approx(0.41)  # R8 p482
    # R8 p463 Ex 7: 0.55 = P(A|S)(0.55) + 0.40(0.45) -> P(A|S) = 0.672727
    assert st.total_probability([0.672727, 0.40], [0.55, 0.45])["total"] == approx(0.55, abs=1e-6)
    # R8 p467-468 Ex 9: E(EPS) = 2.4875(0.6) + 2.12(0.4) = 2.3405 (total expectation rule)
    assert st.total_probability([2.4875, 2.12], [0.60, 0.40])["total"] == approx(2.3405)


def test_bayes_drivemed():
    out = st.bayes_update([0.45, 0.30, 0.25], [0.75, 0.20, 0.05])  # R8 p482-484 Ex 13
    assert out["p_information"] == approx(0.41)
    assert out["posteriors"] == [approx(0.8232, abs=5e-5), approx(0.146341, abs=1e-6), approx(0.030488, abs=1e-6)]
    assert sum(out["posteriors"]) == approx(1)
    assert st.bayes_update([1 / 3] * 3, [0.75, 0.20, 0.05])["posteriors"][0] == approx(0.75)  # p484 Part 4


def test_expected_value_variance_eps():
    out = st.expected_value_variance([2.60, 2.45, 2.20, 2.00], [0.15, 0.45, 0.24, 0.16])  # R8 p464-466 Ex 8-9
    assert out["expected_value"] == approx(2.3405)
    assert out["variance"] == approx(0.038785, abs=1e-6)
    assert out["std_dev"] == approx(0.196939, abs=1e-5)
    with pytest.raises(ValueError):
        st.expected_value_variance([1, 2], [0.5, 0.6])


def test_joint_probability_covariance():
    out = st.joint_probability_covariance([25, 12, 10], [20, 16, 10],
                                          [[0.2, 0, 0], [0, 0.5, 0], [0, 0, 0.3]])  # R8 p479 Tables 12-13
    assert out["covariance"] == approx(16)
    assert (out["expected_x"], out["expected_y"]) == (approx(14), approx(15))
    assert out["correlation"] == approx(16 / math.sqrt(31 * 13))


def test_portfolio_expected_return_variance():
    out = st.portfolio_expected_return_variance([0.5, 0.25, 0.25], [13, 6, 15],
                                                [[400, 45, 189], [45, 81, 38], [189, 38, 441]])  # R8 p471-477
    assert out["expected_return"] == approx(11.75)
    assert out["variance"] == approx(195.875) and out["std_dev"] == approx(14, abs=0.005)
    assert out["variance_terms"] == approx(132.625)
    assert out["correlation_matrix"][1][2] == approx(0.201, abs=5e-4)
    two = st.portfolio_expected_return_variance([0.75, 0.25], [20, 12], [[625, 120], [120, 196]])  # R8 p478-479 Ex 12
    assert two["expected_return"] == approx(18) and two["variance"] == approx(408.8125)
    assert two["std_dev"] == approx(20.22, abs=0.005) and two["correlation_matrix"][0][1] == approx(0.342857, abs=1e-6)


def test_counting_rules():
    out = st.counting_rules(5, 3)  # R8 p486-487
    assert (out["combinations"], out["permutations"], out["factorial"]) == (10, 60, 120)
    m = st.counting_rules(18, group_sizes=[4, 4, 3, 4, 3])["multinomial"]  # R8 p486: "close to 13 billion"
    assert m == math.factorial(18) // (24 * 24 * 6 * 24 * 6) and 12e9 < m < 13e9
    assert st.counting_rules(3)["factorial"] == 6  # three analysts to three industries
    with pytest.raises(ValueError):
        st.counting_rules(5, group_sizes=[2, 2])


# ---- binomial (R9)
def test_binomial_probability_golden():
    assert st.binomial_probability(12, 0.5, 3)["p_at_most_k"] == approx(0.072998, abs=1e-6)  # R9 p515 Ex 4
    assert st.binomial_probability(8, 0.5, 5)["p_at_least_k"] == approx(0.363281, abs=1e-6)  # R9 p515
    assert st.binomial_probability(8, 0.9, 6)["p_at_most_k"] == approx(0.186895, abs=1e-6)  # R9 p516 Ex 5
    d = st.binomial_probability(25, 0.107, 0)  # R9 p518 Ex 6
    assert d["mean"] == approx(2.675) and d["variance"] == approx(2.388775)
    assert st.binomial_probability(5, 0.1, 2)["p_at_most_k"] == approx(0.9914, abs=5e-5)  # R9 p514


def test_binomial_tree():
    p = 0.6
    t = st.binomial_tree(100, 1.1, 0.9, p, 3)
    assert t["terminal_probabilities"][1] == approx(3 * p ** 2 * (1 - p))  # P(uudS), R9 p518-519
    assert t["paths_to_node"] == [1, 3, 3, 1]
    assert t["terminal_prices"][1] == approx(100 * 1.1 ** 2 * 0.9)
    assert t["expected_terminal_price"] == approx(100 * (p * 1.1 + (1 - p) * 0.9) ** 3)
    with pytest.raises(ValueError):
        st.binomial_tree(100, 0.9, 1.1, 0.5, 2)


def test_default_risk_premium():
    out = st.one_period_default_risk_premium(0.058, 0.06)  # R8 p471 Ex 11
    assert out["promised_return"] == approx(0.12553, abs=1e-5)
    assert out["default_risk_premium"] == approx(0.0675, abs=1e-4)
    rec = st.one_period_default_risk_premium(0.058, 0.06, recovery_rate=0.35)
    assert (1 + rec["promised_return"]) * (0.94 + 0.06 * 0.35) == approx(1.058)
