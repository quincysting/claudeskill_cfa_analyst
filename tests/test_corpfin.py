"""Tests for cfa_calc.corpfin. Expected values come from CFA curriculum worked examples in the knowledge base
(E:/CLAUDE/CFA/MD); each test cites file and page. KB paths are relative to MD/."""
import pytest
from pytest import approx

from cfa_calc import REGISTRY
from cfa_calc import corpfin as cf

L1 = "Level-1/05-Corporate-Finance/Curriculum-2014/"
L2 = "Level-2/05-Corporate-Finance/Curriculum-2014/"


# ----------------------------------------------------------------------------- capital budgeting

def test_capital_project_metrics_gerhardt():
    # L1 SS11-R36-capital-budgeting.md p26-28, p33 (Gerhardt: NPV 13.136, IRR 19.5197%, PI 1.26)
    r = cf.capital_project_metrics([-50, 16, 16, 16, 16, 20], 0.10)
    assert r["npv"] == approx(13.136, abs=1e-3)
    assert r["irr"] == approx(0.195197, abs=1e-6)
    assert r["profitability_index"] == approx(1.26, abs=0.005)
    assert _npv_at(r["irr"], [-50, 16, 16, 16, 16, 20]) == approx(0, abs=1e-9)  # IRR round trip


def _npv_at(rate, cfs):
    return cf.capital_project_metrics(cfs, rate)["npv"]


def test_payback_and_discounted_payback():
    # L1 SS11-R36 p29 Table 2 (payback 3 2/3 years) and p31 Table 4 (payback 3.33, discounted payback 4.26)
    assert cf.capital_project_metrics([-10000, 2500, 2500, 3000, 3000, 3000], 0.10)["payback_period"] == approx(3 + 2 / 3)
    r = cf.capital_project_metrics([-5000] + [1500] * 5, 0.10)
    assert r["payback_period"] == approx(3.3333, abs=1e-4)
    assert r["discounted_payback_period"] == approx(4.26, abs=0.005)
    assert cf.capital_project_metrics([-1000, 100], 0.10)["payback_period"] is None


def test_payback_drawbacks_table():
    # L1 SS11-R36 p29 Example 1 Table 3: Project A payback 1.0, NPV -90.91; Project F NPV 7380.92
    a = cf.capital_project_metrics([-1000, 1000], 0.10)
    f = cf.capital_project_metrics([-1000, 500, 500, 10000], 0.10)
    assert (a["payback_period"], a["npv"]) == (approx(1.0), approx(-90.91, abs=0.01))
    assert f["payback_period"] == approx(2.0) and f["npv"] == approx(7380.92, abs=0.01)


def test_mirr_textbook_definition():
    # Standard textbook MIRR (Brigham & Ehrhardt): FV of inflows at r / PV of outflows, n-th root.
    # Project A of L1 SS11-R36 p35: FV inflows = 80 x 4.641 = 371.28 -> MIRR = (371.28/200)^(1/4) - 1
    r = cf.capital_project_metrics([-200, 80, 80, 80, 80], 0.10)
    assert r["mirr"] == approx((371.28 / 200) ** 0.25 - 1, abs=1e-9)
    assert r["irr"] == approx(0.2186, abs=1e-4)  # L1 SS11-R36 p35 Table 7


def test_npv_per_share():
    # L1 SS11-R36 Example 6 logic: share price effect = NPV / shares
    assert cf.capital_project_metrics([-50, 16, 16, 16, 16, 20], 0.10, shares_outstanding=10)["npv_per_share"] == approx(1.3136, abs=1e-4)


def test_average_accounting_rate_of_return():
    # L1 SS11-R36 p31-32 Table 5: AAR = 18,000 / 100,000 = 18%
    r = cf.average_accounting_rate_of_return([6000, 24000, 48000, 18000, -6000], 200000)
    assert r["aar"] == approx(0.18)


def test_npv_crossover_rate():
    # L1 SS11-R36 p35-36 Example 4: crossover 15.09%, NPV 27.98
    r = cf.npv_crossover_rate([-200, 80, 80, 80, 80], [-200, 0, 0, 0, 400])
    assert r["crossover_rate"] == approx(0.1509, abs=1e-4)
    assert r["npv_at_crossover"] == approx(27.98, abs=0.01)


def test_project_cash_flows_expansion():
    # L2 SS08-R25-capital-budgeting.md p45: outlay 230,000, CF 92,000, TNOCF 70,000, NPV 162,217;
    # p48 Table 15/18: PV of SL depreciation tax savings 53,071
    r = cf.project_cash_flows(200000, 30000, 220000, 90000, 35000, 0.40, 5, 50000, 25000, rate=0.10)
    assert r["outlay"] == approx(230000)
    assert r["operating_cash_flows"] == [approx(92000)] * 5
    assert r["terminal_nonoperating_cash_flow"] == approx(70000)
    assert r["npv"] == approx(162217, abs=1)
    assert r["pv_depreciation_tax_savings"] == approx(53071, abs=1)


def test_project_cash_flows_macrs():
    # L2 SS08-R25 p47-49 Table 17/18: MACRS 3-year on 175,000 -> NPV 167,403, PV tax savings 58,257 (+5,186)
    dep = cf.macrs_depreciation(175000, 3)["depreciation"]
    assert dep == [approx(58327.5), approx(77787.5), approx(25917.5), approx(12967.5)]
    r = cf.project_cash_flows(200000, 30000, 220000, 90000, dep + [0.0], 0.40, 5, 50000, 25000, rate=0.10)
    assert r["npv"] == approx(167403, abs=1)
    assert r["pv_depreciation_tax_savings"] == approx(58257, abs=1)


def test_project_cash_flows_replacement():
    # L2 SS08-R25 p49-50: outlay 540,000, incremental CF 102,000, TNOCF 150,000, NPV 213,907, IRR 15.40%
    r = cf.project_cash_flows(1000000, 80000, 150000, 30000, 60000, 0.30, 10, salvage_terminal=100000,
                              book_value_terminal=0, salvage_initial=600000, book_value_initial=400000, rate=0.08)
    assert r["outlay"] == approx(540000)
    assert r["operating_cash_flows"][0] == approx(102000)
    assert r["terminal_nonoperating_cash_flow"] == approx(150000)
    assert r["npv"] == approx(213907, abs=1)
    assert r["irr"] == approx(0.1540, abs=1e-4)


def test_macrs_rates_sum_to_one():
    for cls in (3, 5, 7, 10, 15, 20):
        assert sum(cf.macrs_depreciation(1.0, cls)["rates"]) == approx(1.0, abs=0.0015)
    with pytest.raises(ValueError):
        cf.macrs_depreciation(100, 4)


def test_unequal_lives_eaa_and_chain():
    # L2 SS08-R25 p53-55: NPV S 28.93, L 35.66; 6-year chains 72.59 / 62.45; EAA S 16.66, L 14.34
    r = cf.unequal_lives_comparison({"S": [-100, 60, 90], "L": [-140, 80, 70, 60]}, 0.10)
    s, l_ = r["projects"]["S"], r["projects"]["L"]
    assert (s["npv"], l_["npv"]) == (approx(28.93, abs=0.005), approx(35.66, abs=0.005))
    assert (s["chain_npv"], l_["chain_npv"]) == (approx(72.59, abs=0.005), approx(62.45, abs=0.005))
    assert (s["eaa"], l_["eaa"]) == (approx(16.66, abs=0.01), approx(14.34, abs=0.005))
    assert r["best_by_eaa"] == "S" and r["lcm_horizon"] == 6


def test_capital_rationing():
    # L2 SS08-R25 p56 Table 22: budget 1,000 -> Projects 9 and 12, NPV 400
    projects = [{"name": 9, "outlay": 600, "npv": 300}, {"name": 10, "outlay": 600, "npv": 270},
                {"name": 11, "outlay": 200, "npv": 80}, {"name": 12, "outlay": 400, "npv": 100}]
    r = cf.capital_rationing(projects, 1000)
    assert sorted(r["selected"]) == [9, 12] and r["total_npv"] == 400


def test_project_real_nominal_rate_round_trip():
    # L2 SS08-R25 p52: (1 + nominal) = (1 + real)(1 + inflation)
    nom = cf.project_real_nominal_rate(0.03, real_rate=0.05)["nominal_rate"]
    assert nom == approx(0.0815)
    assert cf.project_real_nominal_rate(0.03, nominal_rate=nom)["real_rate"] == approx(0.05)


def test_real_option_adjusted_npv():
    # L2 SS08-R25 p68 Example 10 (Sackley): -0.40 - 0.30 + 1.20 = 0.50
    assert cf.real_option_adjusted_npv(-0.40, 1.20, 0.30)["npv"] == approx(0.50)


def test_abandonment_option_npv():
    # L2 SS08-R25 p68-69 Example 11 (Nyberg): without -9,808; with option 13,158; option adds 22,966
    r = cf.abandonment_option_npv(200000, [{"probability": 0.5, "cash_flows": [40000] * 4},
                                           {"probability": 0.5, "cash_flows": [80000] * 4}], 150000, 1, 0.10)
    assert r["npv_without_option"] == approx(-9808, abs=1)
    assert r["npv_with_option"] == approx(13158, abs=1)
    assert r["option_value"] == approx(22966, abs=1)


# ----------------------------------------------------------------------------- cost of capital

def test_wacc_weights():
    # L1 SS11-R37-cost-of-capital.md p55 Example 1: WACC 11.44%
    r = cf.weighted_average_cost_of_capital(0.08, 0.15, 0.40, weight_debt=0.3, weight_preferred=0.1,
                                            cost_of_preferred=0.10)
    assert r["wacc"] == approx(0.1144)
    # market values give the same answer
    r2 = cf.weighted_average_cost_of_capital(0.08, 0.15, 0.40, debt_value=30, equity_value=60, preferred_value=10,
                                             cost_of_preferred=0.10)
    assert r2["wacc"] == approx(0.1144)


def test_cost_of_debt_ytm_and_rating():
    # L1 SS11-R37 p61 Example 4: rd 4.684%, after-tax 3.045%; debt-rating approach 4% x 0.65 = 2.6%
    r = cf.cost_of_debt_after_tax(0.35, price=1025, face=1000, coupon_rate=0.05, years=10, freq=2)
    assert r["before_tax_cost_of_debt"] == approx(0.04684, abs=1e-5)
    assert r["after_tax_cost_of_debt"] == approx(0.03045, abs=1e-5)
    assert cf.cost_of_debt_after_tax(0.35, comparable_yield=0.04)["after_tax_cost_of_debt"] == approx(0.026)


def test_cost_of_preferred():
    # L1 SS11-R37 p64 Example 5 (Alcoa): 3.75 / 72 = 5.21%
    assert cf.cost_of_preferred_stock(3.75, 72)["cost_of_preferred"] == approx(0.0521, abs=5e-5)


def test_cost_of_equity_capm_and_country_risk():
    # L1 SS11-R37 p65 Example 7: 5% + 1.5(7%) = 15.5%; p77: 0.04 + 1.2(0.045 + 0.03) = 13%
    assert cf.cost_of_equity_capm(0.05, 1.5, 0.07)["cost_of_equity"] == approx(0.155)
    assert cf.cost_of_equity_capm(0.04, 1.2, 0.045, 0.03)["cost_of_equity"] == approx(0.13)


def test_cost_of_equity_ddm_flotation():
    # L1 SS11-R37 p82: D0 2, P 40, g 5% -> 10.25%; 4% flotation -> 10.469%
    assert cf.cost_of_equity_ddm(40, growth_rate=0.05, dividend_current=2)["cost_of_equity"] == approx(0.1025)
    r = cf.cost_of_equity_ddm(40, growth_rate=0.05, dividend_current=2, flotation_cost_pct=0.04)
    assert r["cost_of_equity"] == approx(0.10469, abs=1e-5)
    # sustainable growth g = b x ROE
    assert cf.cost_of_equity_ddm(40, dividend_next=2, retention_rate=0.6, roe=0.1)["growth_rate"] == approx(0.06)


def test_cost_of_equity_bond_yield_plus():
    # L1 SS11-R37 p70: 4.95% + 3.5% = 8.45%
    assert cf.cost_of_equity_bond_yield_plus(0.0495, 0.035)["cost_of_equity"] == approx(0.0845)


def test_country_risk_premium():
    # L1 SS11-R37 p78 Example 12 (Argentina): 0.05 x 0.40/0.28 = 7.14%
    assert cf.country_risk_premium(0.095, 0.045, 0.40, 0.28)["country_risk_premium"] == approx(0.0714, abs=5e-5)


def test_beta_pure_play():
    # L1 SS11-R37 p74: 1.2 unlevered at D/E 0.125, t 35% -> 1.1098; relevered at 0.4 -> 1.3983
    r = cf.beta_pure_play(0.4, 0.35, comparable_beta=1.2, comparable_de_ratio=0.125, comparable_tax_rate=0.35)
    assert r["asset_beta"] == approx(1.1098, abs=1e-4) and r["levered_beta"] == approx(1.3983, abs=1e-4)
    # p74 Example 9: 1.3 / [1 + 0.6 x 720/540] = 0.72
    ex9 = cf.beta_pure_play(0, 0, comparable_beta=1.3, comparable_de_ratio=720 / 540, comparable_tax_rate=0.4)
    assert ex9["asset_beta"] == approx(0.72, abs=0.005)
    # p75 Example 10 (Aerotechnique): 1.0 x [1 + 0.66 x 1.4] = 1.924
    assert cf.beta_pure_play(1.4, 0.34, asset_beta=1.0)["levered_beta"] == approx(1.924)


def test_bayern_wacc_with_comparables():
    # L1 SS11-R37 p75-77 Example 11 (Bayern Chemicals): asset beta 0.60, beta 0.86, re 9.4%, WACC 7.26%
    comps = [{"beta": 1.45, "de_ratio": 1.33, "tax_rate": 0.30}, {"beta": 0.75, "de_ratio": 0.94, "tax_rate": 0.303},
             {"beta": 1.05, "de_ratio": 1.13, "tax_rate": 0.305}]
    b = cf.beta_pure_play(0.7, 0.38, comparables=comps)
    assert b["asset_beta"] == approx(0.60, abs=0.005) and b["levered_beta"] == approx(0.86, abs=0.005)
    re = cf.cost_of_equity_capm(0.045, b["levered_beta"], 0.057)["cost_of_equity"]
    assert re == approx(0.094, abs=0.001)
    w = cf.weighted_average_cost_of_capital(0.0675, re, 0.38, de_ratio=0.7)
    assert w["weight_debt"] == approx(0.41, abs=0.005)
    assert w["wacc"] == approx(0.0726, abs=0.0003)  # book rounds weights to 0.41/0.59


def test_marginal_cost_of_capital_schedule():
    # L1 SS11-R37 p79-80 Table 3: break points 5, 10, 12.5, 13.3 (EUR millions)
    r = cf.marginal_cost_of_capital_schedule(
        {"debt": 0.4, "equity": 0.6},
        {"debt": [[2, 0.02], [5, 0.025], [None, 0.03]], "equity": [[6, 0.05], [8, 0.07], [None, 0.09]]})
    assert r["break_points"] == [approx(5), approx(10), approx(12.5), approx(13.333, abs=1e-3)]
    assert r["schedule"][0]["wacc"] == approx(0.4 * 0.02 + 0.6 * 0.05)
    assert r["schedule"][-1]["wacc"] == approx(0.4 * 0.03 + 0.6 * 0.09)


def test_npv_with_flotation_costs():
    # L1 SS11-R37 p83: NPV 9,591 before; 7,791 with 1,800 flotation; 8,511 if tax deductible (t 40%)
    cfs = [-60000] + [10000] * 10
    assert cf.npv_with_flotation_costs(cfs, 0.072, 0)["npv"] == approx(9591, abs=1)
    assert cf.npv_with_flotation_costs(cfs, 0.072, 1800)["npv"] == approx(7791, abs=1)
    assert cf.npv_with_flotation_costs(cfs, 0.072, 1800, 0.40, True)["npv"] == approx(8511, abs=1)


# ----------------------------------------------------------------------------- leverage

def test_degree_of_leverage_global_auto():
    # L1 SS11-R38-measures-of-leverage.md p108-119 Examples 1, 2, 4, 5: DOL 1.333, DFL 9, DTL 12,
    # Q_OBE 1.5m cars (EUR 36bn), Q_BE 5.5m cars (EUR 132bn)
    r = cf.degree_of_leverage(6e6, 24000, 14000, 15e9, 40e9)
    assert r["dol"] == approx(1.3333, abs=1e-4) and r["dfl"] == approx(9.0) and r["dtl"] == approx(12.0)
    assert r["operating_breakeven_units"] == approx(1.5e6) and r["breakeven_units"] == approx(5.5e6)
    assert r["operating_breakeven_revenue"] == approx(36e9) and r["breakeven_revenue"] == approx(132e9)


# ----------------------------------------------------------------------------- capital structure

def test_mm_firm_value_leverkin():
    # L2 SS08-R26-capital-structure.md p113-116 (Leverkin): no tax V 50,000, re 12.143%, WACC 10%;
    # 25% tax: VU 37,500, VL 41,250, equity 26,250, re 12.143%, WACC 9.091%
    r0 = cf.mm_firm_value(5000, 0.10, 15000, 0.05)
    assert (r0["levered_value"], r0["cost_of_equity"], r0["wacc"]) == (approx(50000), approx(0.12143, abs=1e-5),
                                                                       approx(0.10))
    r = cf.mm_firm_value(5000, 0.10, 15000, 0.05, 0.25)
    assert r["unlevered_value"] == approx(37500) and r["levered_value"] == approx(41250)
    assert r["equity_value"] == approx(26250) and r["cost_of_equity"] == approx(0.12143, abs=1e-5)
    assert r["wacc"] == approx(0.09091, abs=1e-5)
    # consistency: VL = EBIT(1-t)/WACC
    assert 5000 * 0.75 / r["wacc"] == approx(r["levered_value"])


def test_static_tradeoff():
    # static trade-off: VL = VU + tD - PV(costs of financial distress)
    r = cf.mm_firm_value(5000, 0.10, 15000, 0.05, 0.25, pv_financial_distress_costs=1000)
    assert r["levered_value"] == approx(40250)


def test_mm_cost_of_equity_hostales():
    # L2 SS08-R26 p118 Example 2: 15% without taxes, 14.36% with 32% tax
    assert cf.mm_cost_of_equity(0.13, 0.09, 0.5)["cost_of_equity"] == approx(0.15)
    assert cf.mm_cost_of_equity(0.13, 0.09, 0.5, 0.32)["cost_of_equity"] == approx(0.1436)


# ----------------------------------------------------------------------------- payout

def test_dividend_effective_tax_rate():
    # L2 SS08-R27 p160-161 Exhibits 3-5: double taxation 60.7% / 44.8%; imputation 15% / 47%; split-rate 48%
    assert cf.dividend_effective_tax_rate("double_taxation", 0.35, 0.396)["effective_tax_rate"] == approx(0.607, abs=5e-4)
    assert cf.dividend_effective_tax_rate("double_taxation", 0.35, 0.15)["effective_tax_rate"] == approx(0.4475)
    assert cf.dividend_effective_tax_rate("imputation", 0.30, 0.47)["effective_tax_rate"] == approx(0.47)
    assert cf.dividend_effective_tax_rate("split_rate", 0.35, 0.35, 0.20)["effective_tax_rate"] == approx(0.48)
    with pytest.raises(ValueError):
        cf.dividend_effective_tax_rate("split_rate", 0.35, 0.35)


def test_share_repurchase_effects():
    # L1 SS11-R39 p145-146 Example 7 (Jensen): EPS 3.00 at 5% after-tax cost, 2.94 at 6%
    kw = dict(shares_outstanding=2.2e6, net_income=6.6e6, share_price=60, repurchase_amount=12e6)
    assert cf.share_repurchase_effects(after_tax_cost_of_funds=0.05, **kw)["eps_after"] == approx(3.00)
    assert cf.share_repurchase_effects(after_tax_cost_of_funds=0.06, **kw)["eps_after"] == approx(2.94)
    # p147 Example 8: BVPS 10 -> 9.74 (P > B) and 30 -> 30.26 (P < B)
    a = cf.share_repurchase_effects(10e6, 1, 20, repurchase_amount=5e6, book_value_equity=100e6)
    b = cf.share_repurchase_effects(10e6, 1, 20, repurchase_amount=5e6, book_value_equity=300e6)
    assert a["bvps_after"] == approx(9.74, abs=0.005) and b["bvps_after"] == approx(30.26, abs=0.005)


def test_lintner_target_dividend():
    # L2 SS08-R27 p167 Example 9 (Luna): 0.40 + 0.80 x 0.3 x 0.2 = 0.45 (0.448)
    r = cf.lintner_target_dividend(0.40, 2.00, 2.80, 0.30, adjustment_years=5)
    assert r["expected_dividend"] == approx(0.448)
    assert r["expected_dividend_lintner"] == approx(0.40 + 0.2 * (0.3 * 2.8 - 0.40))


def test_dividend_coverage():
    # L2 SS08-R27 p178 Example 15: payout 40%, coverage 2.5x
    r = cf.dividend_coverage(100, 40)
    assert r["payout_ratio"] == approx(0.40) and r["dividend_coverage"] == approx(2.5)
    # p180 Example 16 (Harley Davidson 2006): FCFE = 762 - 220 + 493 = 1,035; coverage 1,035/1,149 = 0.90
    h = cf.dividend_coverage(1043, 213, cfo=762, fixed_capital_investment=220, net_borrowing=493,
                             share_repurchases=936)
    assert h["fcfe"] == approx(1035) and h["fcfe_coverage"] == approx(0.90, abs=0.005)


# ----------------------------------------------------------------------------- working capital

def test_cost_of_trade_credit():
    # L1 SS11-R40-working-capital-management.md p192 Example 6: 1/10 net 30 -> 44.32% (day 20), 20.13% (day 30)
    assert cf.cost_of_trade_credit(0.01, 10, 20)["cost_of_trade_credit"] == approx(0.4432, abs=5e-5)
    assert cf.cost_of_trade_credit(0.01, 10, 30)["cost_of_trade_credit"] == approx(0.2013, abs=5e-5)


def test_short_term_borrowing_cost():
    # L1 SS11-R40 p197-198 Example 7: line of credit 7.00%, banker's acceptance 6.79%, commercial paper 6.56%
    m = 1 / 12
    assert cf.short_term_borrowing_cost(5e6, 0.065, m, commitment_fee=0.005)["effective_annual_cost"] == approx(0.07)
    assert cf.short_term_borrowing_cost(5e6, 0.0675, m, True)["effective_annual_cost"] == approx(0.0679, abs=5e-5)
    cp = cf.short_term_borrowing_cost(5e6, 0.0615, m, True, dealer_fee=0.00125, backup_fee=0.0025)
    assert cp["effective_annual_cost"] == approx(0.0656, abs=5e-5)


def test_short_term_investment_yields():
    # L1 SS11-R40 p174 Example 2: 91-day T-bill at 7.91% discount -> price 98,000.53, MMY 8.07%, BEY 8.18%
    r = cf.short_term_investment_yields(100000, 91, discount_rate=0.0791)
    assert r["purchase_price"] == approx(98000.53, abs=0.01)
    assert r["money_market_yield"] == approx(0.0807, abs=5e-5)
    assert r["bond_equivalent_yield"] == approx(0.0818, abs=5e-5)
    assert r["discount_basis_yield"] == approx(0.0791)  # round trip
    # p174: one-month 5% discount security -> yield 5.0209% (monthly x 12)
    one = cf.short_term_investment_yields(1e6, 30, purchase_price=995833.33)
    assert one["holding_period_yield"] * 12 == approx(0.050209, abs=1e-6)


# ----------------------------------------------------------------------------- M&A

def test_hhi_merger_analysis():
    # L2 SS09-R29-mergers-and-acquisitions.md p269-270 Example 7: pre 1,450; 2+3 -> 1,850 (+400); 9+10 -> 1,500 (+50)
    shares = [0.25, 0.20, 0.10, 0.10, 0.10, 0.05, 0.05, 0.05, 0.05, 0.05]
    a = cf.hhi_merger_analysis(shares, [1, 2])
    assert a["pre_merger_hhi"] == approx(1450) and a["post_merger_hhi"] == approx(1850)
    assert a["change_in_hhi"] == approx(400) and a["likely_action"] == "challenge"
    b = cf.hhi_merger_analysis(shares, [8, 9])
    assert b["post_merger_hhi"] == approx(1500) and b["change_in_hhi"] == approx(50)
    assert b["concentration"] == "moderately concentrated" and b["likely_action"] == "no action"


def test_merger_bid_evaluation_adagio():
    # L2 SS09-R29 p284-286 Example 10 (Adagio/Tantalus, EUR millions, synergies 90)
    cash = cf.merger_bid_evaluation(1125, 75, 300, 30, 90, cash_per_target_share=12)
    assert cash["target_gain"] == approx(60) and cash["acquirer_gain"] == approx(30)
    assert cash["post_merger_value"] == approx(1155)
    stock = cf.merger_bid_evaluation(1125, 75, 300, 30, 90, exchange_ratio=0.80)
    assert stock["price_paid"] == approx(367, abs=0.5) and stock["target_gain"] == approx(67, abs=0.5)
    assert stock["acquirer_gain"] == approx(23, abs=0.5) and stock["target_ownership_of_combined"] == approx(24 / 99)
    mixed = cf.merger_bid_evaluation(1125, 75, 300, 30, 90, cash_per_target_share=6, exchange_ratio=0.40)
    assert mixed["post_merger_share_price"] == approx(1335 / 87)  # book prints 15.35 (= 15.3448)
    assert mixed["price_paid"] == approx(364, abs=0.5) and mixed["acquirer_gain"] == approx(26, abs=0.5)
    for r in (cash, stock, mixed):  # gains always split the synergies
        assert r["target_gain"] + r["acquirer_gain"] == approx(90)


def test_merger_bootstrap_eps():
    # L2 SS09-R29 p252-253 Example 3: 25,000 shares issued, EPS 4.20, efficient P/E 23.8, 105 at P/E 25
    r = cf.merger_bootstrap_eps(100, 4.00, 100000, 50, 2.50, 50000)
    assert r["shares_issued"] == approx(25000) and r["combined_eps"] == approx(4.20)
    assert r["efficient_market_pe"] == approx(23.8, abs=0.05) and r["price_at_acquirer_pe"] == approx(105)
    # at an acquirer price of 80 (equal P/Es) there is no bootstrap: EPS 4.00
    assert cf.merger_bootstrap_eps(80, 4.00, 100000, 50, 2.50, 50000)["combined_eps"] == approx(4.00)


def test_ma_comparable_company_analysis():
    # L2 SS09-R29 p278-280 Example 8 (New Life Books): stock value 33.47, premium 22.9%, takeover price 41.14
    # (book rounds the multiples; unrounded multiples give 33.49 / 41.17)
    comps = [{"price": 20, "eps": 1.00, "cfps": 2.55, "bvps": 6.87, "sps": 12.62},
             {"price": 32, "eps": 1.82, "cfps": 3.90, "bvps": 12.80, "sps": 18.82},
             {"price": 16, "eps": 0.93, "cfps": 2.25, "bvps": 5.35, "sps": 7.62}]
    prem = [[23, 28.5], [17.25, 22.65], [86.75, 102], [45, 53.75], [36.75, 45]]
    r = cf.ma_comparables_valuation({"eps": 1.95, "cfps": 4.12, "bvps": 12.15, "sps": 18.11}, comparables=comps,
                                    takeover_premiums=prem)
    assert r["multiples"]["eps"] == approx(18.26, abs=0.005)
    assert r["estimated_value"] == approx(33.47, abs=0.05)
    assert r["takeover_premium"] == approx(0.229, abs=5e-4)
    assert r["takeover_value"] == approx(41.14, abs=0.05)


def test_ma_comparable_transaction_analysis():
    # L2 SS09-R29 p281-282 Example 9 (Medical Services): weighted takeover value 47.65
    r = cf.ma_comparables_valuation({"eps": 2.62, "cfps": 4.33, "bvps": 12.65, "sps": 22.98},
                                    multiples={"eps": 18.3, "cfps": 10.2, "bvps": 3.7, "sps": 2.4},
                                    weights={"eps": 0.2, "cfps": 0.4, "bvps": 0.2, "sps": 0.2})
    assert r["takeover_value"] == approx(47.65, abs=0.005)


def test_ma_dcf_target_value():
    # L2 SS09-R29 p275-277: PV FCF 5,802; TV 55,922; PV TV 35,670; value 41,471 (thousands)
    r = cf.ma_dcf_target_value([1367, 1351, 1486, 1636, 1799], 0.0941, terminal_growth=0.06)
    assert r["pv_free_cash_flows"] == approx(5802, abs=1)
    assert r["terminal_value"] == approx(55922, abs=1)
    assert r["pv_terminal_value"] == approx(35670, abs=1)
    assert r["firm_value"] == approx(41471, abs=1)
    # p276: market-multiple terminal value 20 x 1,799 = 35,980
    assert cf.ma_dcf_target_value([1367, 1351, 1486, 1636, 1799], 0.0941, terminal_multiple=20)["terminal_value"] == approx(35980)


# ----------------------------------------------------------------------------- registry / validation

def test_all_corporate_finance_functions_registered():
    names = {n for n, f in REGISTRY.items() if f.topic == "corporate_finance"}
    assert len(names) >= 30
    for n in names:
        assert REGISTRY[n].curriculum and REGISTRY[n].formula


def test_input_validation():
    with pytest.raises(ValueError):
        cf.capital_project_metrics([-100], 0.1)
    with pytest.raises(ValueError):
        cf.weighted_average_cost_of_capital(0.05, 0.1, 0.3)
    with pytest.raises(ValueError):
        cf.cost_of_trade_credit(0.02, 10, 5)
    with pytest.raises(ValueError):
        cf.hhi_merger_analysis([0.6, 0.6], [0, 1])
    with pytest.raises(ValueError):
        cf.ma_dcf_target_value([100], 0.08, terminal_growth=0.09)


# ---------------------------------------------------------------- regressions found by independent verification

def test_capital_project_irr_agrees_with_quant_irr_above_1000pct():
    from cfa_calc import tvm
    for cfs in ([-1, 20], [-100, 5000, 100], [-50, 16, 16, 16, 16, 20]):
        assert cf.capital_project_metrics(cfs, 0.1)["irr"] == approx(tvm.irr(cfs)["irr"], rel=1e-10)
    assert cf.capital_project_metrics([-1, 20], 0.1)["irr"] == approx(19.0)


def test_cost_of_debt_rejects_fractional_periods():
    with pytest.raises(ValueError):
        cf.cost_of_debt_after_tax(0.35, price=1025, face=1000, coupon_rate=0.05, years=10.25, freq=2)
