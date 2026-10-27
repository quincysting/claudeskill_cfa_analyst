"""Tests for the IPS return requirement (cfa_calc.pwm) and cfa_calc.pwm_planning.

Golden values (paths relative to E:/CLAUDE/CFA/MD/Level-3):
R10 = 03-Private-Wealth-Management/Curriculum-2014/SS04-R10-managing-individual-investor-portfolios.md
R13 = 03-Private-Wealth-Management/Curriculum-2014/SS04-R13-concentrated-single-asset-positions.md
R14 = 03-Private-Wealth-Management/Curriculum-2014/SS04-R14-lifetime-financial-advice-human-capital-asset-allocation-and-insurance.md
R15 = 04-Institutional-Investors/Curriculum-2014/SS05-R15-managing-institutional-investor-portfolios.md
R19 = 06-Asset-Allocation/Curriculum-2014/SS08-R19-asset-allocation.md
"""
import pytest

from cfa_calc import pwm
from cfa_calc import pwm_planning as pp

A = pytest.approx


# ---------------------------------------------------------------- IPS required return
def test_ips_inger():  # R10 p.190/202: 493,949 / 42,340,437 = 1.17% + 3.00% inflation = 4.17% (additive, after tax)
    r = pwm.ips_required_return(spending_need=493949, portfolio_value=42340437, inflation=0.03)
    assert r["real_spending_rate"] == A(0.0117, abs=5e-5)
    assert r["required_return_after_tax"] == A(0.0417, abs=5e-5)


def test_ips_fairfax_gross_up():  # R10 p.205: 3% real after tax + 4% inflation, 35% tax -> 10.8% pretax
    r = pwm.ips_required_return(spending_rate=0.03, inflation=0.04, tax_rate=0.35)
    assert r["required_return_after_tax"] == A(0.07) and r["required_return_pretax"] == A(0.108, abs=5e-4)


def test_ips_smith_net_need():  # R10 p.238-239: (150,000 - 65,000 pension - 40,000 gift fund)/1,000,000 = 4.5% + 3% = 7.5%
    r = pwm.ips_required_return(spending_need=150000, other_income=105000, portfolio_value=1_000_000, inflation=0.03)
    assert r["net_need"] == A(45000) and r["required_return_after_tax"] == A(0.075)


def test_ips_taylor_reserved_outflow():  # R10 p.233: 48,000/900,000 = 5.3%; base 800,000 if 100,000 goes to the son
    assert pwm.ips_required_return(spending_need=48000, portfolio_value=900000)["real_spending_rate"] == \
        A(0.053, abs=5e-4)
    assert pwm.ips_required_return(spending_need=48000, portfolio_value=900000,
                                   reserved_outflows=100000)["investable_base"] == A(800000)


def test_foundation_required_return():  # R15 p.473: additive 7.3%, multiplicative 7.42%; R19 p.204 6.5%, p.301 8.47%
    assert pwm.ips_required_return(spending_rate=0.05, fees=0.003, inflation=0.02)["required_return_pretax"] == A(0.073)
    assert pwm.ips_required_return(spending_rate=0.05, fees=0.003, inflation=0.02,
                                   method="multiplicative")["required_return_pretax"] == A(0.0742, abs=5e-5)
    assert pwm.ips_required_return(spending_rate=0.04, inflation=0.02, fees=0.004,
                                   method="multiplicative")["required_return_pretax"] == A(0.065, abs=5e-4)
    assert pwm.ips_required_return(spending_rate=0.05, inflation=0.03, fees=0.003,
                                   method="multiplicative")["required_return_pretax"] == A(0.0847, abs=5e-5)


def test_goal_based_required_return():  # R10 Maclins p.239-240 (4.427% after tax, 7.38% pretax); Mueller p.236 (5.4%)
    r = pp.ips_required_return_for_goal(portfolio_value=1_235_000, target_value=2_000_000, years=18,
                                        annual_cash_flow=-26000, tax_rate=0.40)
    assert r["required_return_after_tax"] == A(0.04427, abs=5e-6)
    assert r["required_return_pretax"] == A(0.0738, abs=5e-5)
    assert pp.ips_required_return_for_goal(portfolio_value=2.0e6, target_value=2.6e6,
                                           years=5)["required_return_after_tax"] == A(0.054, abs=5e-4)


# ---------------------------------------------------------------- human capital and retirement
def test_human_capital():  # R14 Eq.1 p.409: HC = sum p_t E[h_t]/(1 + r + v)^t (hand-computed)
    r = pp.human_capital_value(annual_income=100000, years_to_retirement=2, risk_free_rate=0.04,
                               occupational_risk_premium=0.02, income_growth=0.03, financial_capital=50000)
    hc = 100000 * 1.03 / 1.06 + 100000 * 1.03 ** 2 / 1.06 ** 2
    assert r["human_capital"] == A(hc) and r["economic_net_worth"] == A(hc + 50000)
    s = pp.human_capital_value(annual_income=100000, years_to_retirement=2, risk_free_rate=0.04,
                               survival_probs=[0.99, 0.98])
    assert s["human_capital"] == A(0.99 * 100000 / 1.04 + 0.98 * 100000 / 1.04 ** 2)


def test_retirement_annuity():
    r = pp.retirement_annuity(rate=0.05, years=25, payment=50000)
    assert r["present_value"] == A(704697.23, abs=0.01)
    assert pp.retirement_annuity(rate=0.05, years=25, present_value=r["present_value"])["payment"] == A(50000)
    assert pp.retirement_annuity(rate=0.05, years=25, payment=50000, in_advance=True)["present_value"] == \
        A(r["present_value"] * 1.05)
    g = pp.retirement_annuity(rate=0.05, years=3, payment=100, payment_growth=0.03)
    assert g["present_value"] == A(100 / 1.05 + 103 / 1.05 ** 2 + 106.09 / 1.05 ** 3)


def test_retirement_monte_carlo():  # R14 p.429-430 (loose): $1m, $50k rising 3%, 6.14% compound, 13% sd ->
    # book ~10% depleted by age 82, 25% by 88, 50% by 95 (its simulation details are not given)
    r = pp.retirement_monte_carlo(initial_wealth=1e6, annual_spending=50000, mean_return=0.0614 + 0.13 ** 2 / 2,
                                  std_dev=0.13, years=30, spending_growth=0.03, spending_at_start=False)
    ruin = r["probability_of_ruin_by_year"]
    assert 0.04 < ruin[16] < 0.15 and 0.18 < ruin[22] < 0.35 and 0.38 < ruin[29] < 0.62
    assert pp.retirement_monte_carlo(initial_wealth=1e6, annual_spending=50000, mean_return=0.06, std_dev=0.13,
                                     years=30)["probability_of_success"] == \
        pp.retirement_monte_carlo(initial_wealth=1e6, annual_spending=50000, mean_return=0.06, std_dev=0.13,
                                  years=30)["probability_of_success"]
    d = pp.retirement_monte_carlo(initial_wealth=100, annual_spending=10, mean_return=0, std_dev=0, years=10, n_sims=100)
    assert d["probability_of_ruin_by_year"][8] == 0 and d["probability_of_ruin_by_year"][9] == 1


# ---------------------------------------------------------------- concentrated positions
def test_collar_lemesurier():  # R13 Ex.4 p.366-367: 100,000 sh at $40, put 36 / call 48 at $3 each; expiry at $42
    r = pp.concentrated_position_strategies(shares=100000, price=40, put_strike=36, call_strike=48, put_premium=3,
                                            call_premium=3, terminal_prices=[30, 42, 60])
    assert r["net_premium"] == 0 and (r["floor_value"], r["cap_value"]) == A((3.6e6, 4.8e6))
    assert r["hedged_values_at_terminal"] == A([3.6e6, 4.2e6, 4.8e6])


def test_zero_cost_collar_book():  # R13 p.364: $100 stock, $90 put and $120 call at $5 each -> floor 90, cap 120
    r = pp.concentrated_position_strategies(shares=1, price=100, put_strike=90, call_strike=120, put_premium=5,
                                            call_premium=5, terminal_prices=[80, 100, 130])
    assert r["hedged_values_at_terminal"] == A([90, 100, 120])


def test_monetization_and_sale():  # R13 Ex.7 p.379-381 (225m cash sale; 262.5m swap+sell; 346.5m box, 1.04m/yr);
    # Ex.6 p.377-379 (Garcia: 45m x 0.80 x (1 - 0.15) = 30.6m)
    r = pp.concentrated_position_strategies(shares=1, price=350e6, loan_to_value=0.99, annual_cost_rate=0.003,
                                            tax_rate=0.25)
    assert r["monetization_proceeds"] == A(346.5e6) and r["annual_carry_cost"] == A(1.04e6, rel=1e-3)
    assert r["after_tax_sale_proceeds"] == A(262.5e6)
    assert pp.concentrated_position_strategies(shares=1, price=300e6, tax_rate=0.25)["after_tax_sale_proceeds"] == A(225e6)
    assert pp.concentrated_position_strategies(shares=1, price=45e6, tax_rate=0.15,
                                               fraction_sold=0.80)["after_tax_sale_proceeds"] == A(30.6e6)
    pvf = pp.concentrated_position_strategies(shares=1, price=100, put_strike=95, call_strike=110, loan_to_value=0.88)
    assert pvf["monetization_proceeds"] == A(88)  # R13 p.365 prepaid variable forward


# ---------------------------------------------------------------- institutions
def test_pension_funded_status():  # R15 Practice 4 p.525 (750m assets, 200m underfunded -> PBO 950m); Apex p.454-458
    r = pp.pension_funded_status(plan_assets=750e6, liabilities=950e6)
    assert r["surplus"] == A(-200e6) and r["status"] == "underfunded"
    a = pp.pension_funded_status(plan_assets=100e6, liabilities=92e6)
    assert a["surplus"] == A(8e6) and a["funded_ratio"] == A(100 / 92)
    assert pp.pension_funded_status(plan_assets=200, benefit_payments=[100, 100],
                                    discount_rate=0.10)["liabilities"] == A(100 / 1.1 + 100 / 1.21)


def test_spending_rules():  # R15 p.478 formulas; Ex.13 p.483-485 (CAS 70/30 smoothing at 4.5%, no inflation term)
    assert pp.endowment_spending_rule(spending_rate=0.05, market_values=[90, 100, 110])["spending"] == A(5.0)
    assert pp.endowment_spending_rule(spending_rate=0.05, market_values=[90, 100, 110], rule="simple")["spending"] == \
        A(5.5)
    g = pp.endowment_spending_rule(spending_rate=0.045, market_values=[30e6], rule="geometric", smoothing_rate=0.7,
                                   prior_spending=1.2e6)
    assert g["spending"] == A(0.7 * 1.2e6 + 0.3 * 0.045 * 30e6)
    gi = pp.endowment_spending_rule(spending_rate=0.05, market_values=[100], rule="geometric", smoothing_rate=0.8,
                                    prior_spending=4.0, inflation=0.03)
    assert gi["spending"] == A(0.8 * 4.0 * 1.03 + 0.2 * 5.0)


def test_bank_duration_gap():  # R15 p.508: LADG = D_A - (L/A) D_L; dE ~ -LADG x A x dy
    r = pp.bank_leverage_adjusted_duration_gap(asset_duration=5, liability_duration=2, assets=100, liabilities=90,
                                               rate_change=0.01)
    assert (r["duration_gap"], r["equity_value_change"]) == A((3.2, -3.2))


def test_validation():
    with pytest.raises(ValueError):
        pp.endowment_spending_rule(spending_rate=0.05, market_values=[100], rule="geometric")
    with pytest.raises(ValueError):
        pp.retirement_annuity(rate=0.05, years=10)


# ---------------------------------------------------------------- regressions found by independent verification

@pytest.mark.parametrize("call", [
    lambda: pp.retirement_annuity(rate=0.05, years=2.5, payment=100),
    lambda: pp.human_capital_value(annual_income=100000, years_to_retirement=2.5, risk_free_rate=0.04),
    lambda: pp.retirement_monte_carlo(initial_wealth=1e6, annual_spending=50000, mean_return=0.06, std_dev=0.13,
                                      years=30.5),
    lambda: pp.ips_required_return_for_goal(portfolio_value=1e6, target_value=2e6, years=10.5,
                                            annual_cash_flow=-10000),
])
def test_fractional_years_rejected_not_truncated(call):
    with pytest.raises(ValueError):
        call()


def test_whole_float_years_and_fractional_years_without_flows_still_work():
    assert pp.retirement_annuity(rate=0.05, years=2.0, payment=100)["present_value"] == \
        A(pp.retirement_annuity(rate=0.05, years=2, payment=100)["present_value"])
    # no annual flows: a fractional horizon is fine, (target/PV)^(1/n) - 1
    assert pp.ips_required_return_for_goal(portfolio_value=1, target_value=2, years=2.5)["required_return_after_tax"] \
        == A(2 ** (1 / 2.5) - 1)
