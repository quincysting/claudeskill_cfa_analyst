"""Tests for cfa_calc.portfolio_performance. Golden values (paths relative to E:/CLAUDE/CFA/MD):
R34 = Level-3/13-Performance-Evaluation/Curriculum-2014/SS17-R34-evaluating-portfolio-performance.md
R35 = Level-3/14-GIPS/Curriculum-2014/SS18-R35-overview-of-the-global-investment-performance-standards.md
R42 = Level-3/13-Performance-Evaluation/Curriculum-2013/SS17-R42-global-performance-evaluation.md
R44 = Level-1/06-Portfolio-Management/Curriculum-2014/SS12-R44-portfolio-risk-and-return-part-ii.md
RM  = Level-3/10-Risk-Management/Curriculum-2014/SS14-R27-risk-management.md
"""
import pytest

from cfa_calc import portfolio_performance as pf

A = pytest.approx


def test_twr_mientkiewicz():  # R34 Ex.3-4 p.143-144: subperiods 0.0150, -0.0048, 0.0189; TWR 2.92%
    r = pf.gips_time_weighted_return(market_values=[1_000_000, 1_045_000, 1_060_000, 1_080_000],
                                     cash_flows=[0, 30_000, 20_000, 0])
    assert r["subperiod_returns"] == A([0.0150, -0.0048, 0.0189], abs=5e-5)
    assert r["twr"] == A(0.0292, abs=5e-5)


def test_twr_denker_and_large_flows():  # R34 Q20 p.218: 1.922%; R35 Q6 p.323-324: Q1 16.36%
    r = pf.gips_time_weighted_return(market_values=[25e6, 25.8e6, 25.6e6, 26e6], cash_flows=[0, 1e6, -0.5e6, 0])
    assert r["subperiod_returns"] == A([-0.008, 0.01163, 0.01563], abs=1e-5)
    assert r["twr"] == A(0.01922, abs=5e-6)
    q = pf.gips_time_weighted_return(market_values=[200000, 208000, 257000, 263000, 240000, 245000],
                                     cash_flows=[0, 0, 40000, 0, -30000, 0])
    assert q["twr"] == A(0.1636, abs=5e-5)


def test_twr_links_period_returns():  # R34 Q4 p.212: 4.05%; Q6 LIRR p.213: 8.98%; Ex.7 LIRR 2.65%
    assert pf.gips_time_weighted_return(subperiod_returns=[0.0125, 0.0347, -0.0236, 0.0189, -0.0267, 0.0257])["twr"] \
        == A(0.0405, abs=5e-5)
    assert pf.gips_time_weighted_return(subperiod_returns=[0.0535, -0.0234, 0.0462, 0.0125])["twr"] == A(0.0898, abs=5e-5)
    assert pf.gips_time_weighted_return(subperiod_returns=[0.021, 0.0016, -0.014, 0.018])["twr"] == A(0.0265, abs=5e-5)


def test_mwr():  # R34 Ex.5 p.145 (R 0.0009536/day, 2.90%); Ex.6 p.146 (0.020896, 0.859709); R35 SI-IRR p.279-280
    r = pf.gips_money_weighted_return(begin_value=1_000_000, end_value=1_080_000, period_length=30,
                                      cash_flows=[30_000, 20_000], flow_times=[5, 16])
    assert r["rate_per_unit"] == A(0.0009536, abs=5e-8)
    assert r["period_return"] == A(0.0290, abs=5e-5)
    c = pf.gips_money_weighted_return(begin_value=800_000, end_value=3_000_000, period_length=30,
                                      cash_flows=[1_000_000], flow_times=[10])
    assert c["rate_per_unit"] == A(0.020896, abs=1e-6)
    assert c["period_return"] == A(0.859709, abs=1e-4)
    re = pf.gips_money_weighted_return(begin_value=150000, end_value=274300, period_length=8,
                                       cash_flows=[100000, -12665, -11130], flow_times=[2, 5, 7], periods_per_year=4)
    assert re["rate_per_unit"] == A(0.0253, abs=5e-5)
    assert re["annualized"] == A(0.1051, abs=5e-5)


def test_mwr_equals_twr_without_flows():
    r = pf.gips_money_weighted_return(begin_value=100, end_value=110, period_length=1)
    assert r["period_return"] == A(0.10)


def test_modified_dietz():  # R35 p.241-243: Original Dietz 0.52%, Modified Dietz 0.51%; Q7 p.324: 3.65%, 1.92%, 2.58%
    md = pf.gips_modified_dietz(begin_value=100000, end_value=110550, cash_flows=[10000], flow_days=[5], period_days=30)
    assert md["return_"] == A(0.0051, abs=5e-5)
    assert md["weights"] == A([25 / 30])
    od = pf.gips_modified_dietz(begin_value=100000, end_value=110550, cash_flows=[10000], flow_timing="midpoint")
    assert od["return_"] == A(0.0052, abs=5e-5)
    assert pf.gips_modified_dietz(begin_value=74.9, end_value=85.3, cash_flows=[7.5], flow_days=[12],
                                  period_days=31)["return_"] == A(0.0365, abs=5e-5)
    assert pf.gips_modified_dietz(begin_value=127.6, end_value=109.8, cash_flows=[-15, -5], weights=[0.742, 0.387]
                                  )["return_"] == A(0.0192, abs=5e-5)
    assert pf.gips_modified_dietz(begin_value=110.4, end_value=128.4, cash_flows=[15], weights=[0.387]
                                  )["return_"] == A(0.0258, abs=5e-5)


def test_composite_return():  # R35 Exhibit 10-11 p.249-251: 0.65% (beginning values), 0.62% (with weighted flows)
    rets = [0.0051, 0.0028, 0.0032, 0.0136]
    assert pf.gips_composite_return(portfolio_returns=rets, beginning_values=[100.00, 97.40, 112.94, 124.47]
                                    )["composite_return"] == A(0.0065, abs=5e-5)
    assert pf.gips_composite_return(portfolio_returns=rets, beginning_values=[108.30, 107.63, 112.94, 112.10]
                                    )["composite_return"] == A(0.0062, abs=5e-5)


def test_annualized():  # R34 Ex.8 p.148 (2.1%); Q7 p.213 (-2.09%, 3.27%); R35 Exhibit 15 (cum 40%, ann 3.42%)
    assert pf.gips_annualized_return(period_returns=[0.02, 0.095, -0.047])["annualized"] == A(0.021, abs=5e-4)
    assert pf.gips_annualized_return(period_returns=[0.275, -0.189, 0.146, -0.324, 0.123])["annualized"] == \
        A(-0.0209, abs=5e-5)
    assert pf.gips_annualized_return(period_returns=[0.057, 0.049, 0.078, -0.067, 0.053])["annualized"] == \
        A(0.0327, abs=5e-5)
    g = pf.gips_annualized_return(period_returns=[-0.105, 0.163, 0.075, 0.018, 0.112, 0.061, -0.213, 0.165, 0.106, 0.027])
    assert g["cumulative_return"] == A(0.40, abs=0.001) and g["annualized"] == A(0.0342, abs=5e-5)
    assert pf.gips_annualized_return(cumulative_return=0.05, years=0.5)["gips_may_annualize"] is False


EX13_R = [2.66, 2.64, 2.53, 2.49, 2.47, 2.42, 2.38, 2.33, 2.28, 2.21, 2.17, 2.07, 1.96, 1.93]
EX13_V = [118493, 79854, 121562, 86973, 105491, 112075, 98667, 92518, 107768, 96572, 75400, 77384, 31264, 84535]


def test_composite_dispersion():  # R35 Exhibit 13-14 p.268-271
    d = pf.gips_composite_dispersion(returns=EX13_R, beginning_values=EX13_V)
    assert d["equal_weighted_std"] == A(0.23, abs=0.005)
    assert d["equal_weighted_std_population"] == A(0.22, abs=0.005)
    assert d["asset_weighted_std"] == A(0.21, abs=0.005)
    assert d["range_high_low"] == A(0.73) and d["mean"] == A(2.32, abs=0.005)
    assert (d["upper_quartile"], d["median"], d["lower_quartile"]) == A((2.50, 2.355, 2.145))
    assert d["interquartile_range"] == A(0.36, abs=0.006)


def test_net_of_fees():
    assert pf.gips_net_of_fees_returns(gross_returns=[0.10], annual_fee_rate=0.01)["net_returns"] == A([0.09])
    m = pf.gips_net_of_fees_returns(gross_returns=[0.01] * 12, annual_fee_rate=0.012, periods_per_year=12,
                                    method="geometric")
    assert (1 + m["cumulative_gross"]) / (1 + m["cumulative_net"]) == A(1.012)


def test_manager_continuum():  # R34 Ex.9 p.151 (S 1%, A -0.2%); Q8 p.213 (S -3.5%, A -2.8%)
    r = pf.manager_style_active_decomposition(portfolio_return=0.036, benchmark_return=0.038, market_return=0.028)
    assert (r["style_return"], r["active_return"]) == A((0.010, -0.002))
    r = pf.manager_style_active_decomposition(portfolio_return=0.189, benchmark_return=0.217, market_return=0.252)
    assert (r["style_return"], r["active_return"]) == A((-0.035, -0.028))


def test_factor_model_benchmark():  # R34 Ex.12 p.172-173 (1.8%, skill 1.5%); Q13 p.215 (7.29, 7.11, 0.18, 1.02, 0.84)
    r = pf.factor_model_benchmark_skill(portfolio_actual_return=0.109, portfolio_zero_factor=0.01, portfolio_beta=1.2,
                                        benchmark_zero_factor=0.02, benchmark_beta=0.8, factor_return=0.07)
    assert (r["incremental_expected"], r["skill"]) == A((0.018, 0.015))
    r = pf.factor_model_benchmark_skill(portfolio_actual_return=0.0813, portfolio_zero_factor=0.005, portfolio_beta=1.15,
                                        benchmark_zero_factor=0.015, benchmark_beta=0.95, factor_return=0.059)
    assert (r["portfolio_expected"], r["benchmark_expected"]) == A((0.07285, 0.07105))
    assert (r["incremental_expected"], r["total_differential"], r["skill"]) == A((0.0018, 0.01025, 0.00845))


EX8 = dict(  # R34 Exhibit 8 p.173-174 (percent); cash benchmark return set to the cash return
    wp=[5.97, 7.82, 2.90, 31.78, 7.15, 22.47, 12.14, 8.64, 1.13], wb=[5.54, 7.99, 2.38, 34.75, 6.01, 20.91, 16.02, 6.40, 0],
    rp=[-0.79, -3.60, 0.46, 1.92, 0.37, 2.92, 2.00, 0.46, 0.14], rb=[-0.67, -3.95, -0.21, 1.97, 0.14, 2.05, -0.30, -0.37, 0.14])


def _pct(x):
    return [v / 100 for v in x]


def test_brinson_exhibit8():  # R34 p.173-176: totals alloc -0.02, interaction -0.05, selection 0.64, value added 0.57
    r = pf.brinson_attribution(portfolio_weights=_pct(EX8["wp"]), benchmark_weights=_pct(EX8["wb"]),
                               portfolio_returns=_pct(EX8["rp"]), benchmark_returns=_pct(EX8["rb"]))
    assert r["benchmark_return"] == A(0.0069, abs=5e-5) and r["portfolio_return"] == A(0.0126, abs=5e-5)
    assert r["allocation_total"] == A(-0.0002, abs=5e-5)
    assert r["interaction_total"] == A(-0.0005, abs=5e-5)
    assert r["selection_total"] == A(0.0064, abs=5e-5)
    assert r["active_return"] == A(0.0057, abs=5e-5)
    assert r["allocation"][3] == A(-0.0004, abs=5e-5)    # Ex.13 consumer nondurables
    assert r["selection"][6] == A(0.0037, abs=5e-5)      # Ex.14 technology
    assert r["interaction"][6] == A(-0.0009, abs=5e-5)   # Ex.15 technology
    assert r["total"] == A(_pct([-0.01, 0.04, 0.01, -0.05, 0.01, 0.22, 0.32, 0.05, -0.01]), abs=6e-5)


def test_brinson_practice_q23_q24():  # R34 p.218: Consumer Cyclical allocation 0.076%; Financial selection 0.253%
    # remaining sectors lumped into a third bucket whose return makes the benchmark total 1.82%
    rb_total = 0.0597 * 0.0462 + 0.1836 * -0.0143
    r = pf.brinson_attribution(portfolio_weights=[0.0869, 0.1452, 0.7679], benchmark_weights=[0.0597, 0.1836, 0.7567],
                               portfolio_returns=[0.0410, -0.0005, 0.02],
                               benchmark_returns=[0.0462, -0.0143, (0.0182 - rb_total) / 0.7567])
    assert r["benchmark_return"] == A(0.0182)
    assert r["allocation"][0] == A(0.00076, abs=2e-6)  # (2.72%)(2.80%) = 0.0762%
    assert r["selection"][1] == A(0.00253, abs=5e-6)  # (18.36%)(1.38%) = 0.2534%


def test_brinson_bhb_and_identity():
    kw = dict(portfolio_weights=[0.6, 0.4], benchmark_weights=[0.5, 0.5], portfolio_returns=[0.10, 0.02],
              benchmark_returns=[0.08, 0.03])
    for m in ("brinson_fachler", "bhb"):
        r = pf.brinson_attribution(method=m, **kw)
        assert r["allocation_total"] + r["selection_total"] + r["interaction_total"] == A(r["active_return"])


def test_macro_attribution_mepa():  # R34 Exhibits 5-6 p.166-171
    r = pf.macro_attribution(policy_weights=[0.75, 0.25], asset_category_returns=[0.0404, 0.0256],
                             manager_weights=[[0.65, 0.35], [0.55, 0.45]],
                             manager_benchmark_returns=[[0.0461, 0.0431], [0.0199, 0.0255]],
                             manager_returns=[[0.0476, 0.0413], [0.0160, 0.0291]], risk_free_rate=0.0031,
                             total_fund_return=0.0399)
    inc = r["incremental"]
    assert inc["asset_category"] == A(0.0336, abs=5e-5)
    assert inc["benchmarks"] == A(0.0027, abs=5e-5)
    assert inc["investment_managers"] == A(0.0001, abs=5e-5)
    assert inc["allocation_effects"] == A(0.0004, abs=5e-5)
    assert r["cumulative"]["benchmarks"] == A(0.0394, abs=5e-5)  # aggregate manager benchmark 3.94%


def test_international_attribution():  # R42 Q2 p.245-246; Ex.4 p.217-218
    r = pf.international_performance_attribution(
        portfolio_weights=[0.5, 0.2, 0.3], benchmark_weights=[0.5, 0.25, 0.25],
        portfolio_local_returns=[0.05, 0.10, 0.0], index_local_returns=[0.03, 0.05, -0.05],
        currency_returns=[0.0, 100 / 105 - 1, 1 / 0.98 - 1])
    assert r["benchmark_return"] == A(0.00735, abs=5e-6)
    assert r["market_allocation"] == A(-0.005)
    assert r["currency_allocation"] == A(0.0033, abs=5e-6)
    assert r["security_selection"] == A(0.035)
    assert r["portfolio_return"] == A(0.04065, abs=5e-6)
    e = pf.international_performance_attribution(
        portfolio_weights=[0.4, 0.6], benchmark_weights=[0.5, 0.5], portfolio_local_returns=[0.10, 0.0],
        index_local_returns=[0.05, -0.05], currency_returns=[100 / 105 - 1, 1 / 0.98 - 1])
    assert (e["benchmark_return"], e["market_allocation"], e["currency_allocation"], e["portfolio_return"]) == \
        A((-0.0153, -0.01, 0.0066, 0.0313), abs=5e-5)


def test_risk_adjusted_performance_l1():  # R44 Example 10 p.392-393 (Exhibit 8)
    X = pf.risk_adjusted_performance(portfolio_return=0.10, risk_free_rate=0.03, portfolio_std_dev=0.20, beta=1.1,
                                     market_return=0.09, market_std_dev=0.19)
    assert X["sharpe_ratio"] == A(0.35) and X["treynor_ratio"] == A(0.064, abs=5e-4)
    assert X["m_squared_excess"] == A(0.0065) and X["jensen_alpha"] == A(0.004)
    Y = pf.risk_adjusted_performance(portfolio_return=0.11, risk_free_rate=0.03, portfolio_std_dev=0.10, beta=0.7,
                                     market_return=0.09, market_std_dev=0.19)
    assert (Y["sharpe_ratio"], Y["treynor_ratio"], Y["m_squared_excess"], Y["jensen_alpha"]) == \
        A((0.80, 0.114, 0.092, 0.038), abs=5e-4)
    Z = pf.risk_adjusted_performance(portfolio_return=0.12, risk_free_rate=0.03, portfolio_std_dev=0.25, beta=0.6,
                                     market_return=0.09, market_std_dev=0.19)
    assert (Z["sharpe_ratio"], Z["treynor_ratio"], Z["m_squared_excess"], Z["jensen_alpha"]) == \
        A((0.36, 0.150, 0.0084, 0.054), abs=5e-5)


def test_risk_adjusted_performance_l3():  # R34 Q22 p.218 (Sharpe 1.636/1.462, Treynor 2.250/2.375); R42 Q9 p.252
    d = pf.risk_adjusted_performance(portfolio_return=2.0, risk_free_rate=0.2, portfolio_std_dev=1.1, beta=0.8)
    assert (d["sharpe_ratio"], d["treynor_ratio"]) == A((1.636, 2.250), abs=5e-4)
    b = pf.risk_adjusted_performance(portfolio_return=2.1, risk_free_rate=0.2, portfolio_std_dev=1.3, beta=0.8)
    assert (b["sharpe_ratio"], b["treynor_ratio"]) == A((1.462, 2.375), abs=5e-4)
    p3 = pf.risk_adjusted_performance(portfolio_return=0.17, risk_free_rate=0.05, portfolio_std_dev=0.23,
                                      benchmark_return=0.13, tracking_error=0.07)
    assert (p3["sharpe_ratio"], p3["information_ratio"]) == A((0.5217, 0.57), abs=5e-3)
    # R55 (L2 2013) p.484: M^2 return form = rf + S x sd_m: S 0.4711, sd_m 0.20 -> 0.0942 above rf 0
    assert pf.risk_adjusted_performance(portfolio_return=0.4711, risk_free_rate=0.0, portfolio_std_dev=1.0,
                                        market_std_dev=0.20)["m_squared"] == A(0.0942, abs=5e-5)


def test_tracking_error_series():  # R42 Q8 p.252: mean active 2.2%, TE 4.27%
    r = pf.active_return_tracking_error(portfolio_returns=[0.12, 0.14, 0.20, 0.14, 0.16],
                                        benchmark_returns=[0.14, 0.10, 0.12, 0.16, 0.13])
    assert r["mean_active_return"] == A(0.022) and r["tracking_error"] == A(0.0427, abs=5e-5)


def test_sortino_and_drawdown():  # RM p.207-208: RoMAD 15%/10% = 1.5, 10%/40% = 0.25
    assert pf.drawdown_ratios(average_annual_return=0.15, max_drawdown=0.10)["romad"] == A(1.5)
    assert pf.drawdown_ratios(average_annual_return=0.10, max_drawdown=0.40)["romad"] == A(0.25)
    d = pf.drawdown_ratios(returns=[0.10, -0.20, 0.05, 0.10])
    assert d["max_drawdown"] == A(0.20)
    s = pf.sortino_ratio_from_returns(returns=[0.02, -0.01, 0.03, -0.02])
    assert s["downside_deviation"] == A(((0.01 ** 2 + 0.02 ** 2) / 3) ** 0.5)
    assert s["sortino_ratio"] == A(0.005 / s["downside_deviation"])


# ---------------------------------------------------------------- regressions found by independent verification

def test_gips_returns_accept_empty_flow_lists():
    md = pf.gips_modified_dietz(begin_value=100, end_value=105, cash_flows=[], flow_days=[], period_days=30)
    assert md["return_"] == A(0.05) == pf.gips_modified_dietz(begin_value=100, end_value=105)["return_"]
    mw = pf.gips_money_weighted_return(begin_value=100, end_value=110, period_length=1, cash_flows=[], flow_times=[])
    assert mw["period_return"] == A(0.10)
