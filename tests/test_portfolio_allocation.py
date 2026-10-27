"""Tests for cfa_calc.portfolio_allocation (CME, asset allocation, rebalancing).

Golden values (paths relative to E:/CLAUDE/CFA/MD):
R17 = Level-3/05-Economics-and-Capital-Market-Expectations/Curriculum-2014/SS06-R17-capital-market-expectations.md
R18 = Level-3/05-Economics-and-Capital-Market-Expectations/Curriculum-2014/SS07-R18-equity-market-valuation.md
R19 = Level-3/06-Asset-Allocation/Curriculum-2014/SS08-R19-asset-allocation.md
R33 = Level-3/12-Trading-Monitoring-Rebalancing/Curriculum-2014/SS16-R33-monitoring-and-rebalancing.md
Q09 = Level-1/02-Quantitative-Methods/Curriculum-2014/SS03-R09-common-probability-distributions.md
"""
import numpy as np
import pytest

from cfa_calc import portfolio as pm
from cfa_calc import portfolio_allocation as pa

A = pytest.approx


def _corr(lower):
    n = len(lower)
    c = np.eye(n)
    for i, row in enumerate(lower):
        for j, v in enumerate(row):
            c[i, j] = c[j, i] = v
    return c.tolist()


# ---------------------------------------------------------------- CME
def test_grinold_kroner():  # R17 Ex.14 p.48-49 (9.5%); Ex.13 p.47-48 (7.5%, ERP 2.5% vs 5% bond)
    r = pa.cme_grinold_kroner(dividend_yield=0.0225, change_in_shares=-0.01, inflation=0.02, real_earnings_growth=0.04,
                              change_in_pe=0.0025)
    assert r["expected_return"] == A(0.095)
    r = pa.cme_grinold_kroner(dividend_yield=0.0175, change_in_shares=-0.005, inflation=0.025,
                              real_earnings_growth=0.035, change_in_pe=-0.0075)
    assert (r["income_return"], r["nominal_earnings_growth"], r["expected_return"]) == A((0.0225, 0.06, 0.075))


def test_singer_terhaar_canada():  # R17 p.55-56: bonds 1.24% / 5.24%; equities 3.62% / 7.62%
    b = pa.cme_singer_terhaar(std_dev=0.07, correlation_with_global=0.54, global_sharpe=0.28, integration_degree=0.8,
                              risk_free_rate=0.04)
    assert (b["integrated_premium"], b["segmented_premium"]) == A((0.0106, 0.0196), abs=5e-5)
    assert (b["risk_premium"], b["expected_return"]) == A((0.0124, 0.0524), abs=5e-5)
    e = pa.cme_singer_terhaar(std_dev=0.17, correlation_with_global=0.70, global_sharpe=0.28, integration_degree=0.8,
                              risk_free_rate=0.04)
    assert (e["risk_premium"], e["expected_return"]) == A((0.0362, 0.0762), abs=5e-5)


def test_singer_terhaar_example19():  # R17 Ex.19 p.60-61: E(R) US eq 6.87%, US FI 3.85%, non-US eq 6.67%,
    # non-US FI 4.93%, US RE 5.39% (RE: illiquidity 0.30%, integration 0.7)
    cases = [(0.157, 0.85, 0.8, 0.0, 0.0687), (0.038, 0.75, 0.8, 0.0, 0.0385), (0.156, 0.80, 0.8, 0.0, 0.0667),
             (0.091, 0.70, 0.8, 0.0, 0.0493), (0.115, 0.50, 0.7, 0.003, 0.0539)]
    for sd, rho, phi, ill, er in cases:
        r = pa.cme_singer_terhaar(std_dev=sd, correlation_with_global=rho, global_sharpe=0.28, integration_degree=phi,
                                  illiquidity_premium=ill, risk_free_rate=0.03)
        assert r["expected_return"] == A(er, abs=1e-4)  # book blends premiums rounded to 0.01%


def test_gordon_and_buildup():  # R18 Ex.3 p.155-156: 125/2450 + 2.2% = 7.3%; R17 p.50-51 build-up 5%
    r = pa.cme_gordon_growth_erp(dividend_yield=125 / 2450, growth=0.022, bond_yield=0.05)
    assert r["expected_equity_return"] == A(0.073, abs=5e-4)
    b = pa.cme_bond_risk_premium_buildup(real_risk_free_rate=0.015, inflation_premium=0.025, maturity_premium=0.01)
    assert (b["nominal_risk_free_rate"], b["expected_return"]) == A((0.04, 0.05))


def test_multifactor_covariance():  # R17 Exhibits 4-5 p.42-43: M_AB = 0.0226
    F = [[0.0196, 0.0017], [0.0017, 0.0016]]
    r = pa.cme_multifactor_covariance(factor_loadings=[[1.10, 0], [1.05, 0], [0, 1.03]], factor_covariance=F,
                                      residual_variances=[0.10 ** 2, 0.08 ** 2, 0.012 ** 2])
    assert r["covariance"][0][1] == A(0.0226, abs=5e-5)
    assert r["covariance"][0][2] == A(1.10 * 1.03 * 0.0017)
    assert r["covariance"][0][0] == A(1.1 ** 2 * 0.0196 + 0.01)


def test_shrinkage():  # R17 p.39-40: 0.75(48) + 0.25(80) = 56; Ex.12: 0.30(230) + 0.70(190) = 202; means 6.4%
    assert pa.cme_shrinkage_estimate(sample=80, target=48, shrinkage_weight=0.75)["estimate"] == A(56)
    assert pa.cme_shrinkage_estimate(sample=230, target=190, shrinkage_weight=0.70)["estimate"] == A(202)
    m = pa.cme_shrinkage_estimate(sample=[0.04, 0.06, 0.07, 0.08, 0.10], target=[0.07] * 5, shrinkage_weight=0.8)
    assert m["estimate"][0] == A(0.064)


def test_unsmoothing_round_trip():  # standard AR(1) appraisal-smoothing model (Geltner); no curriculum example
    rng = np.random.default_rng(0)
    true = rng.normal(0.02, 0.05, 2000)
    lam = 0.6
    sm = [true[0]]
    for x in true[1:]:
        sm.append((1 - lam) * x + lam * sm[-1])
    r = pa.cme_unsmoothed_volatility(smoothing_lambda=lam, smoothed_returns=sm, smoothed_std_dev=float(np.std(sm, ddof=1)))
    assert r["unsmoothed_returns"] == A(list(true[1:]))
    assert r["unsmoothed_std_dev"] == A(np.std(true, ddof=1), rel=0.05)
    assert pa.cme_unsmoothed_volatility(smoothing_lambda=0.6, smoothed_std_dev=0.10)["unsmoothed_std_dev"] == A(0.2)


# ---------------------------------------------------------------- asset allocation
def test_utility_l3():  # R19 p.206 (5.2 vs 5.0); Ex.4 p.208 (6.0/6.0/5.0); Prob 2 p.314-315 (3.4/3.1/3.5 -> C)
    assert pm.utility_ranking(expected_returns=[0.097, 0.07], std_devs=[0.15, 0.10], risk_aversion=4)["utilities"] == \
        A([0.052, 0.050])
    assert pm.utility_ranking(expected_returns=[0.10, 0.07, 0.0525], std_devs=[0.20, 0.10, 0.05],
                              risk_aversion=2)["utilities"] == A([0.06, 0.06, 0.05])
    r = pm.utility_ranking(expected_returns=[0.115, 0.08, 0.06], std_devs=[0.18, 0.14, 0.10], risk_aversion=5,
                           names=["A", "B", "C"])
    assert r["utilities"] == A([0.034, 0.031, 0.035]) and r["best"] == "C"


def test_utility_corner_portfolios_ex9():  # R19 Ex.9 Part 4 p.243-244, R_A = 3
    er = [11.00, 10.09, 9.67, 7.92, 6.47, 5.73, 5.43, 4.83]
    sd = [20.00, 16.84, 15.57, 11.08, 8.21, 7.04, 6.66, 6.33]
    r = pm.utility_ranking(expected_returns=[x / 100 for x in er], std_devs=[x / 100 for x in sd], risk_aversion=3)
    assert r["utilities"] == A([0.050, 0.058, 0.060, 0.061, 0.055, 0.050, 0.048, 0.042], abs=5e-4)
    assert r["best"] == 3


def test_safety_first():  # R19 Ex.4 p.207 (0.25/0.20/0.05); Prob 3 p.315; Q09 Ex.9 p.530-531 (B, 18.24%)
    assert pa.asset_allocation_safety_first(expected_returns=[0.10, 0.07, 0.0525], std_devs=[0.20, 0.10, 0.05],
                                            threshold_return=0.05)["sf_ratios"] == A([0.25, 0.20, 0.05])
    assert pa.asset_allocation_safety_first(expected_returns=[0.115, 0.08, 0.06], std_devs=[0.18, 0.14, 0.10],
                                            threshold_return=0.035)["sf_ratios"] == A([0.444, 0.321, 0.25], abs=5e-4)
    r = pa.asset_allocation_safety_first(expected_returns=[0.25, 0.11, 0.14], std_devs=[0.27, 0.08, 0.20],
                                         threshold_return=0.0375, names=["A", "B", "C"])
    assert r["sf_ratios"] == A([0.787037, 0.90625, 0.5125], abs=1e-6) and r["best"] == "B"
    assert r["shortfall_probabilities"][1] == A(0.1824, abs=5e-4)


def test_surplus():  # R19 p.262: 250/250, 7.8% vs 5% -> surplus +7.0m, funding ratio 1.027
    r = pa.surplus_return_risk(assets=250, liabilities=250, asset_return=0.078, liability_return=0.05)
    assert r["surplus_change"] == A(7.0) and r["ending_surplus"] == A(7.0)
    assert r["funded_ratio_end"] == A(1.027, abs=5e-4)
    # surplus utility on surplus statistics: R19 p.266 (R_A 6: 0.68 vs -1.42); Prob 13A p.317 (1.60, 1.50, -0.10)
    assert pm.utility_ranking(expected_returns=[0.05, 0.005], std_devs=[0.12, 0.08], risk_aversion=6)["utilities"] == \
        A([0.0068, -0.0142])
    assert pm.utility_ranking(expected_returns=[0.065, 0.04, 0.0], std_devs=[0.14, 0.10, 0.02],
                              risk_aversion=5)["utilities"] == A([0.016, 0.015, -0.001])


def test_surplus_volatility_matches_long_short_portfolio():
    r = pa.surplus_return_risk(assets=100, liabilities=80, asset_return=0.07, liability_return=0.05, asset_std_dev=0.12,
                               liability_std_dev=0.08, correlation=0.4)
    p = pm.portfolio_return_risk(weights=[1.0, -0.8], sds=[0.12, 0.08], corr=0.4)
    assert r["surplus_std_dev"] == A(p["std_dev"]) and r["surplus_return"] == A(0.07 - 0.8 * 0.05)


EX11_MU = [0.10, 0.08, 0.04, 0.045, 0.05, 0.07]
EX11_SD = [0.15, 0.12, 0.07, 0.08, 0.09, 0.10]
EX11_C = _corr([[], [0.76], [0.35, 0.04], [0.50, 0.30, 0.87], [0.24, 0.36, 0.62, 0.52], [0.30, 0.25, -0.05, -0.02, 0.20]])
EX13_R = [10.00, 8.86, 8.35, 7.94, 7.30, 6.13, 5.33]
EX13_SD = [15.00, 11.04, 9.8, 8.99, 7.82, 5.94, 5.37]
EX13_W = [[100, 0, 0, 0, 0, 0], [61.90, 0, 0, 0, 0, 38.10], [40.31, 13.85, 0, 0, 0, 45.83],
          [32.53, 14.30, 0, 0, 8.74, 44.44], [19.93, 21.09, 16.85, 0, 0, 42.13], [0, 26.61, 37.81, 0, 0, 35.58],
          [0, 13.01, 59.94, 0, 0, 27.06]]


def test_corner_portfolio_combination():  # R19 p.234 (8%: w .146, sd <= 9.11); Ex.8 p.235 (7%: w .744)
    r = pa.corner_portfolio_combination(target_return=8.0, corner_returns=EX13_R, corner_std_devs=EX13_SD,
                                        corner_weights=EX13_W)
    assert r["weight_high_corner"] == A(0.146, abs=5e-4) and r["std_dev_approx"] == A(9.11, abs=0.005)
    assert r["asset_weights"] == A([33.67, 14.23, 0, 0, 7.46, 44.64], abs=0.02)
    r = pa.corner_portfolio_combination(target_return=7.0, corner_returns=EX13_R, corner_std_devs=EX13_SD,
                                        corner_weights=EX13_W)
    assert r["weight_high_corner"] == A(0.744, abs=5e-4)
    assert r["asset_weights"] == A([14.83, 22.50, 22.22, 0, 0, 40.45], abs=0.02)


def test_corner_with_risk_free():  # R19 Ex.10 p.247: 6.50 = 7.24w + 2.3(1-w), w .85, sd 9.90
    r = pa.corner_portfolio_combination(target_return=6.50, corner_returns=[7.24, 2.3], corner_std_devs=[11.65, 0.0],
                                        corner_weights=[[49.46, 16.55, 23.05, 0, 10.93, 0], [0, 0, 0, 0, 0, 100]])
    assert r["weight_high_corner"] == A(0.85, abs=5e-4) and r["std_dev_approx"] == A(9.90, abs=0.005)
    assert r["asset_weights"] == A([42.04, 14.07, 19.59, 0, 9.29, 15.0], abs=0.03)  # book rounds w to 0.85


def test_mvo_reproduces_corner_frontier():  # R19 Exhibits 11 and 13 p.232-234: long-only efficient portfolios
    r = pa.mvo_optimize(expected_returns=EX11_MU, sds=EX11_SD, corr=EX11_C, target_return=0.0835)
    assert [x * 100 for x in r["weights"]] == A(EX13_W[2], abs=0.2) and r["std_dev"] == A(0.098, abs=2e-4)
    r = pa.mvo_optimize(expected_returns=EX11_MU, sds=EX11_SD, corr=EX11_C, target_return=0.08)
    assert [x * 100 for x in r["weights"]] == A([33.67, 14.23, 0, 0, 7.46, 44.64], abs=0.15)
    assert r["std_dev"] < 0.0911  # true sd is below the linear bound


def test_mvo_example9_frontier():  # R19 Ex.9 Exhibits 16-17 p.241-242: 8.5% -> (45.25, 4.78, 0, 0, 20.39, 29.58)
    mu = [0.11, 0.09, 0.04, 0.0475, 0.05, 0.07]
    sd = [0.20, 0.18, 0.07, 0.08, 0.095, 0.14]
    c = _corr([[], [0.76], [0.35, 0.04], [0.50, 0.30, 0.87], [0.24, 0.36, 0.62, 0.52], [0.35, 0.25, 0.11, 0.07, 0.12]])
    r = pa.mvo_optimize(expected_returns=mu, sds=sd, corr=c, target_return=0.085)
    assert [x * 100 for x in r["weights"]] == A([45.25, 4.78, 0, 0, 20.39, 29.58], abs=0.1)
    assert r["std_dev"] < 0.1257


def test_mvo_unconstrained_matches_closed_forms():
    mu, sd, rho = [0.10, 0.06], [0.20, 0.10], 0.5
    gmv = pm.global_minimum_variance_portfolio(sds=sd, corr=rho)
    assert pa.mvo_optimize(expected_returns=mu, sds=sd, corr=rho, long_only=False)["weights"] == A(gmv["weights"])
    # utility optimum = GMV + (1/A) Cov^-1 (mu - g 1) direction; check first-order condition instead
    r = pa.mvo_optimize(expected_returns=mu, sds=sd, corr=rho, risk_aversion=3, long_only=False)
    C = np.array([[0.04, 0.01], [0.01, 0.01]])
    grad = np.array(mu) - 3 * C @ np.array(r["weights"])
    assert grad[0] == A(grad[1])
    lo = pa.mvo_optimize(expected_returns=mu, sds=sd, corr=rho, risk_aversion=3, long_only=True)
    assert min(lo["weights"]) >= -1e-12 and sum(lo["weights"]) == A(1)


def test_resampled_weights_deterministic_and_sensible():
    kw = dict(expected_returns=[0.08, 0.05], sds=[0.18, 0.06], corr=0.2, risk_aversion=4, n_samples=30)
    a, b = pa.mvo_resampled_weights(**kw), pa.mvo_resampled_weights(**kw)
    assert a["weights"] == b["weights"] and sum(a["weights"]) == A(1)
    big = pa.mvo_resampled_weights(n_observations=20000, **kw)
    assert big["weights"] == A(big["point_estimate_weights"], abs=0.03)


def test_black_litterman_implied_returns_fit_book():  # R19 Exhibits 14, 22, 23 p.236-252
    # Book equilibrium returns are rf + delta x Cov w; with delta 1.7724 and rf 3.625% (fitted, 2 parameters for
    # 5 values) every printed return is reproduced to the book's 2-decimal rounding.
    sd = [0.1883, 0.3703, 0.2158, 0.3366, 0.1741]
    c = _corr([[], [0.641], [0.747, 0.584], [0.393, 0.444, 0.369], [0.778, 0.599, 0.768, 0.351]])
    w = np.array([12362002, 1269324, 3703025, 2238885, 2161903], float)
    r = pa.black_litterman_implied_returns(market_weights=list(w / w.sum()), risk_aversion=1.77235, sds=sd, corr=c,
                                           risk_free_rate=0.036248)
    assert r["implied_total_returns"] == A([0.0949, 0.1266, 0.0958, 0.0991, 0.0839], abs=5e-4)


def test_black_litterman_posterior_limits():
    pi, sd, rho = [0.05, 0.03], [0.2, 0.1], 0.5
    exact = pa.black_litterman_posterior(implied_excess_returns=pi, view_matrix=[[1, -1]], view_returns=[0.04],
                                         sds=sd, corr=rho, view_uncertainty=[1e-12])["posterior_excess_returns"]
    assert exact[0] - exact[1] == A(0.04, abs=1e-8)
    vague = pa.black_litterman_posterior(implied_excess_returns=pi, view_matrix=[[1, -1]], view_returns=[0.04],
                                         sds=sd, corr=rho, view_uncertainty=[1e6])["posterior_excess_returns"]
    assert vague == A(pi, abs=1e-8)
    # with the equilibrium returns and no informative view, optimal weights are the market weights
    mw = [0.6, 0.4]
    imp = pa.black_litterman_implied_returns(market_weights=mw, risk_aversion=3, sds=sd, corr=rho)
    post = pa.black_litterman_posterior(implied_excess_returns=imp["implied_excess_returns"], view_matrix=[[1, 0]],
                                        view_returns=[0.5], sds=sd, corr=rho, view_uncertainty=[1e9], risk_aversion=3)
    assert post["optimal_weights"] == A(mw, abs=1e-6)


def test_add_asset_class_l3():  # R19 Ex.7 p.216 (0.18 > 0.105, add); Prob 6 p.315 (0.12 < 0.14, do not add)
    assert pm.add_asset_class_test(sharpe_new=0.18, sharpe_portfolio=0.15, correlation=0.7)["add"] is True
    assert pm.add_asset_class_test(sharpe_new=0.12, sharpe_portfolio=0.40, correlation=0.35)["add"] is False


# ---------------------------------------------------------------- rebalancing
def test_cppi_and_buy_and_hold():  # R33 p.108-110: 60/40 on 100: stocks +10% -> 106; +100% -> 160 at 75% stock
    assert pa.cppi_exposure(multiplier=2, assets=100, floor=75)["risky_exposure"] == A(50)
    assert pa.cppi_exposure(multiplier=2, assets=70, floor=75)["risky_exposure"] == 0
    r = pa.rebalancing_strategy_paths(stock_prices=[100, 110], stock_weight=0.6)
    assert r["buy_and_hold"][-1] == A(106)
    r = pa.rebalancing_strategy_paths(stock_prices=[100, 200], stock_weight=0.6)
    assert r["buy_and_hold"][-1] == A(160) and r["buy_and_hold_stock"][-1] / 160 == A(0.75)


def test_strategy_payoff_shapes():  # R33 Exhibit 9 p.111: CM concave (wins in reversals), CPPI convex (wins in trends)
    osc = pa.rebalancing_strategy_paths(stock_prices=[100, 90, 100, 90, 100], stock_weight=0.6, cppi_multiplier=2)
    assert osc["constant_mix"][-1] > osc["buy_and_hold"][-1] > osc["cppi"][-1]
    up = pa.rebalancing_strategy_paths(stock_prices=[100, 110, 121, 133.1], stock_weight=0.6, cppi_multiplier=2)
    assert up["cppi"][-1] > up["buy_and_hold"][-1] > up["constant_mix"][-1]


def test_corridors():  # R33 p.104 (45/15/40 +/- 10% of target; 50/14/36 breaches 49.5%); p.107 halfway 57% -> 52.5%
    r = pa.rebalancing_corridor_check(target_weights=[0.45, 0.15, 0.40], current_values=[50, 14, 36],
                                      corridor_pct_of_target=0.10)
    assert r["corridor_widths"] == A([0.045, 0.015, 0.04])
    assert r["breached"] == [True, False, False] and r["new_weights"] == A([0.45, 0.15, 0.40])
    assert sum(r["trades"]) == A(0, abs=1e-9)
    h = pa.rebalancing_corridor_check(target_weights=[0.5, 0.5], current_values=[57, 43], corridor_widths=[0.05, 0.05],
                                      rebalance_to="halfway")
    assert h["new_weights"][0] == A(0.525)
    none = pa.rebalancing_corridor_check(target_weights=[0.5, 0.5], current_values=[52, 48], corridor_widths=[0.05, 0.05])
    assert none["trades"] == A([0, 0])


def test_black_litterman_docstring_example_matches_engine():
    r = pa.black_litterman_implied_returns(market_weights=[0.6, 0.4], risk_aversion=3, sds=[0.2, 0.1], corr=0.5)
    assert r["implied_excess_returns"] == A([0.084, 0.030])
    assert "[0.084, 0.030]" in pa.black_litterman_implied_returns.__doc__
