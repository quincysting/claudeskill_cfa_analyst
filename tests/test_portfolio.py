"""Tests for cfa_calc.portfolio (L1-L2 portfolio math, CAPM, factor models, active management).

Golden values (paths relative to E:/CLAUDE/CFA/MD):
R43 = Level-1/06-Portfolio-Management/Curriculum-2014/SS12-R43-portfolio-risk-and-return-part-i.md
R44 = Level-1/06-Portfolio-Management/Curriculum-2014/SS12-R44-portfolio-risk-and-return-part-ii.md
K44 = Level-1/06-Portfolio-Management/Kaplan-Notes-2014/SS12-R44-portfolio-risk-and-return-part-ii.md
R57 = Level-2/06-Portfolio-Management/Curriculum-2014/SS18-R57-portfolio-concepts.md
R58 = Level-2/06-Portfolio-Management/Curriculum-2014/SS18-R58-residual-risk-and-return-the-information-ratio.md
R59 = Level-2/06-Portfolio-Management/Curriculum-2014/SS18-R59-the-fundamental-law-of-active-management.md
R55 = Level-2/06-Portfolio-Management/Curriculum-2013/SS18-R55-the-theory-of-active-portfolio-management.md
"""
import numpy as np
import pytest

from cfa_calc import portfolio as pm

A = pytest.approx


# ---------------------------------------------------------------- portfolio math
def test_two_asset_covariance_form():  # R43 Example 4 p.306: 11.58%, variance 0.02281, sd 15.10%
    cov = [[0.1621 ** 2, 0.0050], [0.0050, 0.3311 ** 2]]
    r = pm.portfolio_return_risk(weights=[0.80, 0.20], expected_returns=[0.0993, 0.1820], cov=cov)
    assert r["expected_return"] == A(0.1158, abs=5e-5)
    assert r["variance"] == A(0.02281, abs=1e-5)  # book sums rounded terms
    assert r["std_dev"] == A(0.15103, abs=5e-5)
    assert sum(r["risk_contribution"]) == A(r["std_dev"])


def test_two_asset_correlation_cases():  # R43 Example 7 p.326-327: rho +1 -> 20%, 0 -> 14.14%, -1 -> 0
    kw = dict(weights=[0.5, 0.5], expected_returns=[0.10, 0.10], sds=[0.20, 0.20])
    assert pm.portfolio_return_risk(corr=1.0, **kw)["std_dev"] == A(0.20)
    assert pm.portfolio_return_risk(corr=0.0, **kw)["std_dev"] == A(0.1414, abs=5e-5)
    assert pm.portfolio_return_risk(corr=-1.0, **kw)["std_dev"] == A(0.0, abs=1e-9)


def test_two_asset_practice_problems():  # R43 practice Q7 (12.11%), Q27 (13.04%), Ex.11 Q1 (15.50%, 30.12%)
    assert pm.portfolio_return_risk(weights=[0.3, 0.7], sds=[0.20, 0.12], corr=0.40)["std_dev"] == A(0.1211, abs=1e-4)  # sqrt(1.4688%) = 12.119%
    assert pm.portfolio_return_risk(weights=[0.5, 0.5], sds=[0.20, 0.20], corr=-0.15)["std_dev"] == A(0.1304, abs=5e-5)
    r = pm.portfolio_return_risk(weights=[0.10, 0.90], expected_returns=[0.20, 0.15], sds=[0.50, 0.33], corr=0.0)
    assert (r["expected_return"], r["std_dev"]) == A((0.155, 0.3012), abs=5e-5)


R57_CORR = [[1, 0.5, 0.8], [0.5, 1, 0.5], [0.8, 0.5, 1]]


def test_three_asset_r57_table6():  # R57 Table 6 p.405: weights (0.10, 0.80, 0.10) -> E 7%, var 96.10 (%^2), sd 9.80%
    r = pm.portfolio_return_risk(weights=[0.10, 0.80, 0.10], expected_returns=[0.15, 0.05, 0.15],
                                 sds=[0.15, 0.10, 0.15], corr=R57_CORR)
    assert r["expected_return"] == A(0.07)
    assert r["variance"] == A(0.009610, abs=5e-7)
    assert r["std_dev"] == A(0.0980, abs=5e-5)
    assert pm.portfolio_return_risk(weights=[0.05, 0.90, 0.05], sds=[0.15, 0.10, 0.15],
                                    corr=R57_CORR)["variance"] == A(0.009653, abs=5e-7)


def test_global_minimum_variance():  # R57 p.397: 14.3% large-cap / 85.7% bonds, variance 96.43 (%^2), E 6.43%
    r = pm.global_minimum_variance_portfolio(sds=[0.15, 0.10], corr=0.5, expected_returns=[0.15, 0.05])
    assert r["weights"] == A([1 / 7, 6 / 7])
    assert r["variance"] == A(0.009643, abs=5e-7)
    assert r["expected_return"] == A(0.0643, abs=5e-5)


def test_tangency_portfolio_two_asset_formula():  # R55 Eq.4 two-asset closed form agrees with the matrix solution
    mu, rf, s, rho = np.array([0.12, 0.07]), 0.03, np.array([0.25, 0.10]), 0.3
    r = pm.tangency_portfolio(expected_returns=list(mu), risk_free_rate=rf, sds=list(s), corr=rho)
    e1, e2, c = mu[0] - rf, mu[1] - rf, rho * s[0] * s[1]
    w1 = (e1 * s[1] ** 2 - e2 * c) / (e1 * s[1] ** 2 + e2 * s[0] ** 2 - (e1 + e2) * c)
    assert r["weights"][0] == A(w1)
    gmv = pm.global_minimum_variance_portfolio(sds=list(s), corr=rho, expected_returns=list(mu))
    assert r["sharpe_ratio"] > (gmv["expected_return"] - rf) / gmv["std_dev"]


def test_utility_l1_example6():  # R43 Example 6 p.320-321: A=4 -> Inv 1; A=2 -> Inv 3
    er, sd = [0.12, 0.15, 0.21, 0.24], [0.30, 0.35, 0.40, 0.45]
    r4 = pm.utility_ranking(expected_returns=er, std_devs=sd, risk_aversion=4, names=["1", "2", "3", "4"])
    assert r4["utilities"] == A([-0.06, -0.095, -0.11, -0.165]) and r4["best"] == "1"
    r2 = pm.utility_ranking(expected_returns=er, std_devs=sd, risk_aversion=2, names=["1", "2", "3", "4"])
    assert r2["utilities"] == A([0.03, 0.0275, 0.05, 0.0375]) and r2["best"] == "3"
    assert pm.utility_ranking(expected_returns=0.10, std_devs=0.20, risk_aversion=3)["best_utility"] == A(0.04)  # Ex.5


def test_optimal_risky_allocation():  # R55 practice 2 p.498: y = 8.18/(0.01 x 2.8 x 535.54) = 0.5455
    r = pm.optimal_risky_allocation(expected_return_risky=0.08 + 0.0818, risk_free_rate=0.08,
                                    std_dev_risky=0.053554 ** 0.5, risk_aversion=2.8)
    assert r["weight_risky"] == A(0.5455, abs=5e-5)


def test_cal():  # R57 Example 5B p.420-421
    kw = dict(risk_free_rate=0.05, expected_return_risky=0.15, std_dev_risky=0.25)
    r = pm.capital_allocation_line(target_std_dev=0.10, **kw)
    assert (r["weight_risky"], r["expected_return"], r["slope"]) == A((0.4, 0.09, 0.4))
    assert pm.capital_allocation_line(weight_risky=0.6, **kw)["expected_return"] == A(0.11)
    r = pm.capital_allocation_line(target_std_dev=0.35, **kw)
    assert (r["weight_risky"], r["expected_return"]) == A((1.4, 0.19))
    # R44 Example 2 p.371-372: borrow 25% at rf -> 17.5%, sd 25%
    r = pm.capital_allocation_line(risk_free_rate=0.05, expected_return_risky=0.15, std_dev_risky=0.20, weight_risky=1.25)
    assert (r["expected_return"], r["std_dev"]) == A((0.175, 0.25))


# ---------------------------------------------------------------- CAPM
def test_capm_and_beta():  # R44 Example 6 p.382-383; Example 7 p.386-387; K44 p.169
    b = pm.capm_beta_from_moments(correlation=0.95, std_dev_asset=0.50, std_dev_market=0.25)["beta"]
    assert b == A(1.90)
    assert pm.capm_sml_expected_return(risk_free_rate=0.03, beta=b, market_return=0.10)["expected_return"] == A(0.163)
    b = pm.capm_beta_from_moments(correlation=0.65, std_dev_asset=0.50, std_dev_market=0.23)["beta"]
    assert b == A(1.41, abs=5e-3)
    assert pm.capm_sml_expected_return(risk_free_rate=0.03, beta=1.41, market_return=0.13)["expected_return"] == A(0.171)
    assert pm.capm_beta_from_moments(cov_with_market=0.048, std_dev_market=0.20)["beta"] == A(1.2)


def test_adjusted_beta():  # R57 p.432: 1.5 -> 1.333; 0.5 -> 0.667
    assert pm.capm_beta_from_moments(historical_beta=1.5)["adjusted_beta"] == A(1.333, abs=5e-4)
    assert pm.capm_beta_from_moments(historical_beta=0.5)["adjusted_beta"] == A(0.667, abs=5e-4)


def test_sml_alpha():  # R44 Example 11 p.398-399 (alphas P -2%, Q 4%, R 3%); K44 p.176 stock B undervalued
    alphas = [pm.capm_sml_expected_return(risk_free_rate=0.02, beta=b, market_return=0.12, forecast_return=f)["alpha"]
              for b, f in ((1.5, 0.15), (1.2, 0.18), (1.1, 0.16))]
    assert alphas == A([-0.02, 0.04, 0.03])
    r = pm.capm_sml_expected_return(risk_free_rate=0.07, beta=0.8, market_return=0.15, forecast_return=0.175)
    assert r["expected_return"] == A(0.134) and r["verdict"].startswith("undervalued")


def test_systematic_risk_decomposition():  # R44 Example 11: nonsystematic variance P 0.0171, Q 0.0158, R 0.0137
    got = [pm.systematic_risk_decomposition(beta=b, std_dev_market=0.18, total_std_dev=s)["nonsystematic_variance"]
           for b, s in ((1.5, 0.30), (1.2, 0.25), (1.1, 0.23))]
    assert got == A([0.0171, 0.0158, 0.0137], abs=5e-5)


# ---------------------------------------------------------------- APT / multifactor
def test_apt_premiums_one_and_two_factor():  # R57 Example 11 p.443 (rf 5%, lambda 5%); Example 13 p.445-446
    r = pm.apt_solve_factor_premiums(expected_returns=[0.075, 0.15], sensitivities=[[0.5], [2.0]])
    assert (r["risk_free_rate"], r["factor_risk_premiums"][0]) == A((0.05, 0.05))
    r = pm.apt_solve_factor_premiums(expected_returns=[0.14, 0.12, 0.11], sensitivities=[[1.0, 1.5], [0.5, 1.0], [1.3, 1.1]])
    assert r["risk_free_rate"] == A(0.07) and r["factor_risk_premiums"] == A([-0.02, 0.06])
    assert pm.apt_expected_return(risk_free_rate=0.05, factor_sensitivities=[0.4],
                                  factor_risk_premiums=[0.05])["expected_return"] == A(0.07)


def test_apt_arbitrage():  # R57 Example 12 p.444-445: replica of D = 0.5A + 0.5C, E 7.25% vs 8% -> $75 on $10,000
    r = pm.apt_arbitrage_check(target_expected_return=0.08, target_sensitivities=[0.45],
                               replicating_expected_returns=[0.075, 0.07], replicating_sensitivities=[[0.5], [0.4]])
    assert r["weights"] == A([0.5, 0.5]) and r["replicating_expected_return"] == A(0.0725)
    assert r["arbitrage_return"] * 10_000 == A(75) and r["action"] == "buy target, sell replica"


def test_tracking_portfolio():  # R57 Example 23 p.469-470: weights J 2, K -0.75, L -0.25; E(R) 16.25%
    r = pm.apt_arbitrage_check(target_expected_return=0.1625, target_sensitivities=[1.3, 1.975],
                               replicating_expected_returns=[0.14, 0.12, 0.11],
                               replicating_sensitivities=[[1.0, 1.5], [0.5, 1.0], [1.3, 1.1]])
    assert r["weights"] == A([2.0, -0.75, -0.25])
    assert r["replicating_expected_return"] == A(0.1625)


def test_active_return_decomposition():  # R57 Example 18 Table 20 p.457-458 (percent units)
    r = pm.factor_active_risk_return(portfolio_sensitivities=[0.85, 0.05, 1.40, 0.08],
                                     benchmark_sensitivities=[0.90, 0.10, 1.00, 0.06],
                                     factor_returns=[5.52, -3.35, 5.10, 9.63], active_return=2.0741)
    assert r["factor_contributions"] == A([-0.2760, 0.1675, 2.0400, 0.1926])
    assert r["factor_tilt_return"] == A(2.1241) and r["asset_selection"] == A(-0.05)


def test_active_risk_decomposition():  # R57 p.463-464 (FMCAR 0.497, 0.036); Example 21 p.464-466 (factor 30.0594 of 60)
    r = pm.factor_active_risk_return(portfolio_sensitivities=[0.50, 0.15], benchmark_sensitivities=[0, 0],
                                     factor_covariance=[[225, 12], [12, 144]], active_specific_variance=53.71)
    assert r["active_factor_variance"] == A(61.29) and r["active_risk_squared"] == A(115)
    assert r["factor_marginal_contributions"] == A([0.497, 0.036], abs=5e-4)
    r = pm.factor_active_risk_return(
        portfolio_sensitivities=[0.05, -0.05, -0.25, 0.25, 0.01, 0.03, -0.25, -0.10],
        benchmark_sensitivities=[0.25, 0.05, -0.02, 0.10, 0.00, 0.03, 0.02, 0.03],
        factor_covariance=[225, 144, 100, 196, 169, 81, 121, 64], active_specific_variance=29.9406)
    assert r["active_factor_variance"] == A(30.0594) and r["active_risk"] == A(60 ** 0.5)


def test_market_model_correlation_via_factor_covariance():  # R57 Example 7 p.431: corr(Cisco, Microsoft) 0.4552
    from cfa_calc import portfolio_allocation as pa
    r = pa.cme_multifactor_covariance(factor_loadings=[[2.09], [1.75]], factor_covariance=[[29.8]],
                                      residual_variances=[132.71, 126.79])
    assert r["correlation"][0][1] == A(0.4552, abs=5e-5)


# ---------------------------------------------------------------- active management
def test_fundamental_law():  # R59 p.515 (IC .0577, BR 400 -> 1.154); p.512 (IC .25, BR 4 -> 0.5)
    assert pm.fundamental_law_active_management(information_coefficient=0.0577, breadth=400)["information_ratio"] \
        == A(1.154)
    assert pm.fundamental_law_active_management(information_coefficient=0.25, breadth=4)["information_ratio"] == A(0.5)


def test_optimal_residual_risk_and_value_added():  # R58 Tables 3-4 p.502: IR .75, lambda .10 -> 3.75%, 1.41%
    r = pm.fundamental_law_active_management(information_ratio=0.75, residual_risk_aversion=0.10)
    assert r["optimal_residual_risk"] == A(0.0375) and r["optimal_value_added"] == A(0.0141, abs=5e-5)
    r = pm.fundamental_law_active_management(information_ratio=1.0, residual_risk_aversion=0.05)
    assert (r["optimal_residual_risk"], r["optimal_value_added"]) == A((0.10, 0.05))


def test_optimal_active_risk_sharpe_form():  # SR_P^2 = SR_B^2 + IR^2 at sd_A* = IR/SR_B x sd_B (Grinold-Kahn)
    r = pm.fundamental_law_active_management(information_ratio=0.5, benchmark_sharpe=0.4, benchmark_std_dev=0.2)
    assert r["optimal_active_risk"] == A(0.25) and r["max_portfolio_sharpe"] == A((0.16 + 0.25) ** 0.5)


def test_treynor_black_drex():  # R55 p.481-484: weights 1.1477/-1.6212/1.4735; alpha .2056; w0 .1506; w* .1495; S .4711
    r = pm.treynor_black_active_weights(alphas=[0.07, -0.05, 0.03], residual_std_devs=[0.45, 0.32, 0.26],
                                        betas=[1.6, 1.0, 0.5], market_risk_premium=0.08, market_std_dev=0.20)
    assert r["active_weights"] == A([1.1477, -1.6212, 1.4735], abs=5e-4)
    assert r["active_alpha"] == A(0.2056, abs=5e-4) and r["active_beta"] == A(0.9519, abs=5e-4)
    assert r["active_residual_std_dev"] == A(0.8262, abs=5e-4)
    assert r["initial_active_weight"] == A(0.1506, abs=5e-4) and r["active_weight"] == A(0.1495, abs=5e-4)
    assert r["optimal_sharpe"] == A(0.4711, abs=5e-4)


def test_treynor_black_practice1():  # R55 practice 1 p.495-496: alpha -16.90%, w0 -0.05124, w* -0.0486, S 0.3662
    r = pm.treynor_black_active_weights(alphas=[0.016, -0.044, 0.034, -0.040], residual_std_devs=[0.58, 0.71, 0.60, 0.55],
                                        betas=[1.3, 1.8, 0.7, 1.0], market_risk_premium=0.08, market_std_dev=0.23)
    assert r["active_weights"] == A([-0.6142, 1.1265, -1.2181, 1.7058], abs=1e-3)  # book uses rounded alpha/var ratios
    assert r["active_alpha"] == A(-0.1690, abs=5e-4)
    assert r["initial_active_weight"] == A(-0.05124, abs=5e-5) and r["active_weight"] == A(-0.0486, abs=5e-4)
    assert r["optimal_sharpe"] == A(0.3662, abs=5e-4)


def test_add_asset_class():  # R57 Example 6 p.425-426: Eurobonds no (0.10 < 0.105); EAFE yes (0.30 > 0.1675)
    assert pm.add_asset_class_test(sharpe_new=0.10, sharpe_portfolio=0.25, correlation=0.42)["add"] is False
    r = pm.add_asset_class_test(sharpe_new=0.30, sharpe_portfolio=0.25, correlation=0.67)
    assert r["add"] is True and r["hurdle_sharpe"] == A(0.1675)


def test_validation():
    with pytest.raises(ValueError):
        pm.portfolio_return_risk(weights=[0.5, 0.5, 0.0], sds=[0.1, 0.2, 0.3], corr=0.5)
    with pytest.raises(ValueError):
        pm.capital_allocation_line(risk_free_rate=0.02, expected_return_risky=0.1, std_dev_risky=0.2)
