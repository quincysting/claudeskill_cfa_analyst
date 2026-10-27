"""Tests for cfa_calc.portfolio_risk. Golden values: CFA L3 2014 Risk Management
RM = E:/CLAUDE/CFA/MD/Level-3/10-Risk-Management/Curriculum-2014/SS14-R27-risk-management.md
The book rounds sigma_p (e.g. 0.244) before applying z, so money VaRs are checked to ~0.5%.
"""
import pytest

from cfa_calc import portfolio_risk as rk

A = pytest.approx
TWO = dict(weights=[0.75, 0.25], expected_returns=[0.12, 0.18], sds=[0.20, 0.40], corr=0.90)


def test_var_parametric_two_asset_in_text():  # RM p.174-175: 5% annual $13.4m, daily $1.245m, 1% annual $21.7m, daily $1.75m
    assert rk.var_parametric(portfolio_value=50e6, **TWO)["var"] == A(13.4e6, rel=0.005)
    r = rk.var_parametric(portfolio_value=50e6, horizon_periods_per_year=250, **TWO)
    assert r["horizon_mean"] == A(0.00054) and r["horizon_std_dev"] == A(0.01543, abs=5e-6)
    assert r["var"] == A(1.245e6, rel=0.002)
    assert rk.var_parametric(portfolio_value=50e6, confidence=0.99, **TWO)["var"] == A(21.7e6, rel=0.005)
    assert rk.var_parametric(portfolio_value=50e6, confidence=0.99, horizon_periods_per_year=250,
                             **TWO)["var"] == A(1.75e6, rel=0.015)  # book rounds -0.0354 to -0.035


def test_var_parametric_example5():  # RM p.176 Example 5: $22.485m, $37.65m (annual); $4.827m, $6.93m (weekly)
    kw = dict(portfolio_value=150e6, weights=[0.65, 0.35], expected_returns=[0.12, 0.05], sds=[0.22, 0.07], corr=0.15)
    assert rk.var_parametric(**kw)["var"] == A(22.485e6, rel=0.003)
    assert rk.var_parametric(confidence=0.99, **kw)["var"] == A(37.65e6, rel=0.003)
    assert rk.var_parametric(horizon_periods_per_year=52, **kw)["var"] == A(4.827e6, rel=0.003)
    assert rk.var_parametric(confidence=0.99, horizon_periods_per_year=52, **kw)["var"] == A(6.93e6, rel=0.003)


def test_var_parametric_monthly_inputs_practice10():  # RM p.217/224 Q10: monthly $4.91m/$7.30m, weekly $2.58m/$3.73m
    kw = dict(portfolio_value=100e6, weights=[0.5, 0.5], expected_returns=[0.0085, 0.0095], sds=[0.0320, 0.0526],
              corr=0.35, input_periods_per_year=12)
    assert rk.var_parametric(horizon_periods_per_year=12, **kw)["var"] == A(4.91e6, rel=0.003)
    assert rk.var_parametric(horizon_periods_per_year=12, confidence=0.99, **kw)["var"] == A(7.30e6, rel=0.003)
    assert rk.var_parametric(horizon_periods_per_year=52, **kw)["var"] == A(2.58e6, rel=0.004)
    assert rk.var_parametric(horizon_periods_per_year=52, confidence=0.99, **kw)["var"] == A(3.73e6, rel=0.004)


def test_var_parametric_single_asset_practice19():  # RM p.226 Q19: EUR50m, 9.6%/18% annual -> monthly EUR3.8867m
    r = rk.var_parametric(portfolio_value=50e6, expected_return=0.096, std_dev=0.18, horizon_periods_per_year=12)
    assert r["var"] == A(3.8867e6, rel=1e-4)
    assert rk.var_parametric(portfolio_value=50e6, expected_return=0.096, std_dev=0.18)["var"] == A(10.05e6)


def test_var_sqrt_time():  # RM p.175: daily $100,000 -> annual $1,581,139
    assert rk.var_sqrt_time_scaling(var=100000, periods=250)["scaled_var"] == A(1581139, abs=1)
    # consistency: zero-mean parametric VaR scales exactly with sqrt(t)
    d = rk.var_parametric(portfolio_value=1e6, expected_return=0.1, std_dev=0.2, horizon_periods_per_year=250,
                          include_mean=False)["var"]
    y = rk.var_parametric(portfolio_value=1e6, expected_return=0.1, std_dev=0.2, include_mean=False)["var"]
    assert rk.var_sqrt_time_scaling(var=d, periods=250)["scaled_var"] == A(y)


IBM_WORST = [0.26190, 0.22645, 0.20511, 0.19462, 0.18802, 0.17183, 0.16415, 0.14834, 0.14773, 0.12444,
             0.11692, 0.11553, 0.10838, 0.10805, 0.10687, 0.10503, 0.09873, 0.09550, 0.09276, 0.09091]


def test_var_historical_ibm_example6():  # RM p.179-180 Example 6: 240 months; 5% -> $11,553; 1% -> $22,645 (alt $21,578)
    rets = [-x for x in IBM_WORST] + [0.01] * 220
    r5 = rk.var_historical(returns=rets, portfolio_value=100000, confidence=0.95)
    assert r5["k"] == 12 and r5["var"] == A(11553)
    r1 = rk.var_historical(returns=rets, portfolio_value=100000, confidence=0.99)
    assert r1["k"] == 2 and r1["var"] == A(22645)
    assert r1["var_avg_adjacent"] == A(21578)


DELL = [-0.0214, -0.0106, 0.0262, -0.1196, -0.0313, -0.0362, -0.1137, 0.0401, 0.0129, 0.0652, 0.1196, -0.0789,
        -0.0347, -0.0566, 0.0158, 0.0862, 0.0675, 0.0609, -0.0203, 0.0100, -0.0230, 0.1087, -0.1980, -0.0012,
        -0.1824, -0.0070, 0.0010, -0.0648, 0.2378, -0.0512, 0.1229, -0.1156, -0.2416, -0.2597, -0.0844, -0.0833,
        -0.0723, -0.1021, 0.1114, 0.2257, -0.0043, 0.1867, -0.0255, 0.1831, -0.0360, -0.0531, -0.0228, 0.0170,
        -0.1017, 0.0264, 0.1344, 0.0786, -0.1772, -0.0953, 0.0978, -0.1110, 0.1020, 0.1099, -0.0816, 0.0250]


def test_var_historical_dell_practice11():  # RM p.217/224 Q11: 60 months, $25,000 -> 5% $4,950; 1% $6,492.50
    assert len(DELL) == 60
    assert rk.var_historical(returns=DELL, portfolio_value=25000)["var"] == A(4950)
    assert rk.var_historical(returns=DELL, portfolio_value=25000, confidence=0.99)["var"] == A(6492.5)


MC_WORST = [-0.400, -0.398, -0.397, -0.390, -0.355, -0.350, -0.347, -0.344, -0.343, -0.333,
            -0.320, -0.316, -0.314, -0.310, -0.303, -0.301, -0.301, -0.300, -0.298, -0.296,
            -0.295, -0.282, -0.277, -0.273, -0.273, -0.261, -0.259, -0.253, -0.251, -0.248,
            -0.247, -0.233, -0.229, -0.226, -0.223, -0.222, -0.218, -0.216, -0.215, -0.211]


def test_var_from_simulated_outcomes_practice12():  # RM p.218/224 Q12: 700 simulations -> 5% $2.23m, 1% $3.47m
    sims = MC_WORST + [0.10] * 660
    assert rk.var_historical(returns=sims, portfolio_value=10e6)["var"] == A(2.23e6)
    assert rk.var_historical(returns=sims, portfolio_value=10e6, confidence=0.99)["var"] == A(3.47e6)


def test_var_monte_carlo_converges_to_parametric():
    mc = rk.var_monte_carlo(portfolio_value=50e6, expected_return=0.135, std_dev=0.244, n_sims=400_000, seed=7)
    exact = rk.var_parametric(portfolio_value=50e6, expected_return=0.135, std_dev=0.244, z=1.6448536)
    assert mc["var"] == A(exact["var"], rel=0.01)
    assert mc["expected_shortfall"] == A(exact["expected_shortfall"], rel=0.01)
    assert rk.var_monte_carlo(portfolio_value=1, expected_return=0.1, std_dev=0.2, seed=1)["var"] == \
        rk.var_monte_carlo(portfolio_value=1, expected_return=0.1, std_dev=0.2, seed=1)["var"]  # deterministic


def test_component_var_consistency():
    r = rk.var_component_parametric(position_values=[100, 100], sds=[0.10, 0.20], corr=0.0)
    assert r["var"] == A(1.65 * (0.01 * 1e4 + 0.04 * 1e4) ** 0.5)
    assert sum(r["component_var"]) == A(r["var"])
    assert r["incremental_var"][1] == A(r["var"] - 1.65 * 10)
    assert r["diversification_benefit"] > 0


def test_stress_test():
    r = rk.stress_test_scenarios(position_values=[60, 40], scenarios={"eq -10%": [-0.10, 0.0], "rates +1%": [0, -0.05]})
    assert r["pnl"]["eq -10%"] == A(-6) and r["worst_scenario"] == "eq -10%"


def test_risk_budgeting_desks():  # RM p.200-201: FX ROC 20%, FI 12.5%; perfect correlation combined VaR EUR10m
    r = rk.risk_budget_allocation(capital=[100e6, 200e6], var_budgets=[5e6, 5e6], profits=[20e6, 25e6])
    assert r["return_on_capital"] == A([0.20, 0.125])
    assert r["combined_var"] == A(10e6) and r["total_profit"] == A(45e6)
    assert rk.risk_budget_allocation(capital=[1, 1], var_budgets=[5e6, 5e6], profits=[0, 0],
                                     correlation=[[1, 0], [0, 1]])["combined_var"] == A(5e6 * 2 ** 0.5)


def test_active_risk_budget():
    r = rk.active_risk_budget_managers(information_ratios=[0.3, 0.4], total_active_risk=0.02)
    assert r["portfolio_information_ratio"] == A(0.5)
    assert r["active_risk_allocations"] == A([0.012, 0.016])
    assert sum(x ** 2 for x in r["active_risk_allocations"]) ** 0.5 == A(0.02)


def test_credit_risk_forward():  # RM p.194 (0.7728), Example 11 p.204 (5.873), Example 8 p.197 (FX -0.0235)
    r = rk.credit_risk_forward(spot=102, forward_price=105, rate=0.05, time_remaining=0.75)
    assert r["value_long"] == A(0.7728, abs=2e-4) and r["bearer"] == "long"
    assert rk.credit_risk_forward(spot=108, forward_price=105, rate=0.0425,
                                  time_remaining=8 / 12)["value_long"] == A(5.873, abs=5e-4)
    fx = rk.credit_risk_forward(spot=0.862, forward_price=0.90, rate=0.06, foreign_rate=0.05, time_remaining=1.5)
    assert fx["value_long"] == A(-0.0235, abs=5e-5) and fx["bearer"] == "short"


def test_var_validation():
    with pytest.raises(ValueError):
        rk.var_parametric(portfolio_value=1, std_dev=0.1, confidence=1.2)
    with pytest.raises(ValueError):
        rk.var_parametric(portfolio_value=1)


# ---------------------------------------------------------------- regressions found by independent verification

def test_parametric_es_uses_exact_quantile_and_var_keeps_curriculum_z():
    from scipy.stats import norm
    exact_es = norm.pdf(norm.ppf(0.95)) / 0.05  # 2.0627: phi(z)/(1 - c) needs z = the exact 95% quantile
    r = rk.var_parametric(portfolio_value=1, expected_return=0, std_dev=1)
    assert r["var_pct"] == A(1.65) and r["z"] == 1.65
    assert r["expected_shortfall_pct"] == A(exact_es, rel=1e-12)
    assert rk.var_parametric(portfolio_value=1, expected_return=0, std_dev=1, z=1.6448536269514722)[
        "expected_shortfall_pct"] == A(exact_es, rel=1e-12)


def test_var_monte_carlo_docstring_matches_engine():
    v = rk.var_monte_carlo(portfolio_value=50e6, expected_return=0.135, std_dev=0.244)["var"]
    assert round(v / 1e6, 1) == 13.5 and "13.5 million" in rk.var_monte_carlo.__doc__
