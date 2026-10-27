"""Tests for cfa_calc.econ. Golden values from the CFA 2014 curriculum (paths relative to E:/CLAUDE/CFA/MD).

L1 = Level-1/03-Economics/Curriculum-2014/
  R13 = SS04-R13-demand-and-supply-analysis-introduction.md      R15 = SS04-R15-demand-and-supply-analysis-the-firm.md
  R16 = SS04-R16-the-firm-and-market-structures.md                R17 = SS05-R17-aggregate-output-prices-and-economic-growth.md
  R18 = SS05-R18-understanding-business-cycles.md                 R19 = SS05-R19-monetary-and-fiscal-policy.md
  R20 = SS06-R20-international-trade-and-capital-flows.md         R21 = SS06-R21-currency-exchange-rates.md
K19 = Level-1/03-Economics/Kaplan-Notes-2014/SS05-R19-monetary-and-fiscal-policy.md
L2FX = Level-2/03-Economics/Curriculum-2014/SS04-R14-currency-exchange-rates-determination-and-forecasting.md
L2G  = Level-2/03-Economics/Curriculum-2014/SS04-R15-economic-growth-and-the-investment-decision.md
L3CME = Level-3/05-Economics-and-Capital-Market-Expectations/Curriculum-2014/SS06-R17-capital-market-expectations.md
L3EMV = Level-3/05-Economics-and-Capital-Market-Expectations/Curriculum-2014/SS07-R18-equity-market-valuation.md
"""
import math

import pytest

from cfa_calc import REGISTRY
from cfa_calc import econ as ec

A = pytest.approx


def test_all_public_functions_registered_under_economics():
    names = [n for n, f in REGISTRY.items() if f.func.__module__ == ec.__name__]
    assert len(names) >= 55
    assert all(REGISTRY[n].topic == "economics" and REGISTRY[n].curriculum for n in names)


# ------------------------------------------------------------------------------------- elasticity (R13)
def test_point_elasticity_text():  # R13 p.61: Q = 11,200 - 400P at P=3 -> -0.12; P=4 -> -0.167; P=15 -> -1.154
    assert ec.elasticity_point(dq_dx=-400, x=3, q=10000)["elasticity"] == A(-0.12)
    assert ec.elasticity_point(dq_dx=-400, x=4, q=11200 - 1600)["elasticity"] == A(-0.167, abs=5e-4)
    r = ec.elasticity_point(dq_dx=-400, x=15, q=11200 - 6000)
    assert r["elasticity"] == A(-1.154, abs=5e-4) and r["responsiveness"] == "elastic"


def test_arc_elasticity_text():  # R13 p.62: P 5 -> 6, Q 9,200 -> 8,800 -> -0.244
    r = ec.elasticity_arc(q0=9200, q1=8800, x0=5, x1=6)
    assert r["elasticity"] == A(-0.244, abs=5e-4)
    assert ec.elasticity_arc(q0=8800, q1=9200, x0=6, x1=5)["elasticity"] == A(r["elasticity"])  # symmetric


def test_demand_function_elasticities_gasoline():  # R13 p.68-69 Eq. 27-30: -0.12, 0.30, -0.02
    r = ec.demand_function_elasticities(8400, -400, 3, 60, 50, -10, 20)
    assert r["quantity"] == A(10000)
    assert (r["own_price_elasticity"], r["income_elasticity"], r["cross_price_elasticity"]) == (
        A(-0.12), A(0.30), A(-0.02))
    assert r["good_type"] == "normal" and r["relation_to_other_good"] == "complements"
    assert r["price_rise_effect_on_total_expenditure"] == "rises"


def test_demand_function_elasticities_example13():  # R13 p.69-70 Ex.13: Q 2.088; -2.046, 0.551, 1.537
    r = ec.demand_function_elasticities(2, -0.4, 10.68, 0.0005, 2300, 0.15, 21.40)
    assert r["quantity"] == A(2.088)
    assert r["own_price_elasticity"] == A(-2.046, abs=5e-4)
    assert r["income_elasticity"] == A(0.551, abs=5e-4)
    assert r["cross_price_elasticity"] == A(1.537, abs=5e-4)
    assert r["relation_to_other_good"] == "substitutes" and r["own_price_demand"] == "elastic"


def test_total_revenue_test_consistent_with_arc():  # R13 p.62/65: inelastic demand -> price and TR move together
    r = ec.total_revenue_test(p0=5, p1=6, q0=9200, q1=8800)
    assert r["tr0"] == 46000 and r["tr1"] == 52800 and r["tr_change"] > 0 and r["demand"] == "inelastic"


# ------------------------------------------------------------------------------ markets and surplus (R13)
def test_equilibrium_example6():  # R13 p.38 Ex.6: P* 10.68, Q* 2,088
    r = ec.supply_demand_equilibrium(6360, -400, -1116, 300)
    assert r["equilibrium_price"] == A(10.68) and r["equilibrium_quantity"] == A(2088)


def test_excess_demand_example7():  # R13 p.40-41 Ex.7: P=12 excess supply 924; P=8 excess demand 1,876
    assert ec.supply_demand_equilibrium(6360, -400, -1116, 300, price=12)["excess_demand"] == A(-924)
    assert ec.supply_demand_equilibrium(6360, -400, -1116, 300, price=8)["excess_demand"] == A(1876)


def test_surplus_examples_9_10():  # R13 p.48 Ex.9 CS 625; p.50 Ex.10 PS 1,250 (both at P=65)
    r = ec.supply_demand_equilibrium(180, -2, -15, 1)
    assert r["equilibrium_price"] == A(65) and r["equilibrium_quantity"] == A(50)
    assert r["consumer_surplus"] == A(625) and r["producer_surplus"] == A(1250)
    assert ec.consumer_surplus_linear(180, -2, 65)["consumer_surplus"] == A(625)


def test_consumer_surplus_two_part_tariff():  # R16 p.207-208 Ex.3: Q = 20 - 4P at P=2 -> 12 visits, CS 18
    r = ec.consumer_surplus_linear(20, -4, 2)
    assert r["quantity_demanded"] == A(12) and r["consumer_surplus"] == A(18)


def test_price_floor_example11():  # R13 p.54-55 Ex.11: floor 72 -> Q 36, DWL 147
    r = ec.price_control_effects(180, -2, -15, 1, control_price=72, control_type="floor")
    assert r["binding"] and r["quantity_traded"] == A(36) and r["deadweight_loss"] == A(147)
    eq = ec.supply_demand_equilibrium(180, -2, -15, 1)
    assert r["consumer_surplus_after"] + r["producer_surplus_after"] + r["deadweight_loss"] == A(eq["total_surplus"])


def test_price_ceiling_and_nonbinding():
    r = ec.price_control_effects(180, -2, -15, 1, control_price=60, control_type="ceiling")
    assert r["quantity_traded"] == A(45) and r["shortage"] == A(60 - 45)  # Qd(60)=60, Qs(60)=45
    assert r["deadweight_loss"] == A(0.5 * (67.5 - 60) * 5)
    assert not ec.price_control_effects(180, -2, -15, 1, 60, "floor")["binding"]


def test_per_unit_tax_example12():  # R13 p.58-59 Ex.12: EUR2 tax -> Q 48.667; buyers 65.667, sellers 63.667
    r = ec.per_unit_tax_effects(180, -2, -15, 1, tax=2)
    assert r["quantity_after_tax"] == A(48.667, abs=1e-3)
    assert r["price_paid_by_buyers"] == A(65.667, abs=1e-3) and r["price_received_by_sellers"] == A(63.667, abs=1e-3)
    assert r["buyer_burden_per_unit"] + r["seller_burden_per_unit"] == A(2)
    assert r["deadweight_loss"] == A(0.5 * 2 * (50 - 48.6667), abs=1e-3)


# ------------------------------------------------------------------------------------ the firm (R15)
def test_revenue_schedule_example2():  # R15 p.122-123 Ex.2 Exhibit 7
    q = list(range(11))
    p = [100 - x for x in q]
    r = ec.revenue_schedule(q, p)
    assert r["total_revenue"][10] == 900 and r["average_revenue"][5] == 95
    assert r["marginal_revenue"][1:4] == [99, 97, 95] and r["marginal_revenue"][10] == 81


EX16_Q = list(range(11))
EX16_TVC = [0, 2000, 3800, 5400, 8000, 11000, 15000, 21000, 28800, 38700, 51000]


def test_cost_schedule_example4():  # R15 p.131 Ex.4 Exhibit 16: min AVC 1,800 & min MC 1,600 at 3; min ATC 3,200 at 5
    r = ec.cost_schedule(EX16_Q, 5000, EX16_TVC)
    assert (r["min_avc"], r["min_avc_quantity"]) == (A(1800), 3)
    assert (r["min_mc"], r["min_mc_quantity"]) == (A(1600), 3)
    assert (r["min_atc"], r["min_atc_quantity"]) == (A(3200), 5)
    assert r["average_total_cost"][3] == A(3466.7, abs=0.05) and r["marginal_cost"][10] == A(12300)


def test_profit_schedule_example5():  # R15 p.134-135 Ex.5: breakeven 40 and 80, max profit 30,000 at 60
    q = list(range(0, 101, 10))
    p = [10000 - 25 * x for x in q]
    tc = [100000, 170000, 240000, 300000, 360000, 420000, 480000, 550000, 640000, 710000, 800000]
    r = ec.profit_schedule(q, [a * b for a, b in zip(q, p)], tc)
    assert [b["quantity"] for b in r["breakeven_points"]] == [40, 80]
    assert (r["max_profit"], r["max_profit_quantity"]) == (30000, 60)


def test_profit_schedule_example7():  # R15 p.139-140 Ex.7: breakeven between 1 and 2; max 1,275 at 5 (MR 955 > MC 800)
    q = list(range(11))
    tr = [0, 995, 1980, 2955, 3920, 4875, 5820, 6755, 7680, 8595, 9500]
    tc = [550, 1000, 1500, 2100, 2800, 3600, 4600, 5800, 7200, 8800, 10800]
    r = ec.profit_schedule(q, tr, tc)
    assert r["breakeven_points"][0]["between"] == [1, 2]
    assert (r["max_profit"], r["max_profit_quantity"]) == (1275, 5)
    assert r["marginal_revenue"][5] == 955 and r["marginal_cost"][5] == 800 and r["marginal_cost"][6] == 1000


def test_shutdown_example6():  # R15 p.136 Ex.6: operate short run, exit long run; at TR 1.3m shut down
    r = ec.shutdown_decision(2.0e6, 1.0e6, 1.5e6)
    assert r["short_run"].startswith("operate") and r["long_run"].startswith("exit")
    assert r["loss_if_operating"] == A(0.5e6) and r["loss_if_shut_down"] == A(1.0e6)
    r = ec.shutdown_decision(1.3e6, 1.0e6, 1.5e6)
    assert r["short_run"].startswith("shut down") and r["loss_if_operating"] == A(1.2e6)


def test_product_schedule_example10():  # R15 p.153-154 Ex.10: MP peaks 2,000 at 3; AP peaks 1,600 at 4; MP(7) = -500
    r = ec.product_schedule(list(range(8)), [0, 1000, 2500, 4500, 6400, 7400, 7500, 7000])
    assert (r["max_marginal_product"], r["max_marginal_product_input"]) == (2000, 3)
    assert (r["max_average_product"], r["max_average_product_input"]) == (1600, 4)
    assert r["marginal_product"][7] == -500 and r["negative_mp_from_input"] == 7


def test_input_ratios_examples_11_12():  # R15 p.156-158 Ex.11-12: MP/P 2,4,5; MRP/P 1,2,2.5; skilled adds 300
    r = ec.input_productivity_ratios([200, 500, 1000], [100, 125, 200], product_price=0.5)
    assert r["mp_per_price"] == [2, 4, 5] and r["best_input_index"] == 2
    assert r["marginal_revenue_product"] == [100, 250, 500] and r["mrp_per_price"] == [1, 2, 2.5]
    assert r["profit_contribution"][2] == 300


# ------------------------------------------------------------------------------ market structure (R16)
def test_monopoly_markup():  # R16 p.205: MC 75, E 1.5 -> 225; p.218 practice Q4: MC 40 -> 120
    assert ec.monopoly_price_from_elasticity(75, -1.5)["price"] == A(225)
    r = ec.monopoly_price_from_elasticity(40, 1.5)
    assert r["price"] == A(120) and r["lerner_index"] == A(1 / 1.5)
    with pytest.raises(ValueError):
        ec.monopoly_price_from_elasticity(40, 0.8)


def test_cournot_monopoly_competition():  # R16 p.194-195: Q_D = 450 - P, MC 30 -> Cournot 280 @170; comp 420 @30; mono 210 @240
    r = ec.linear_market_structure_outcomes(450, 1, 30, n_firms=2)
    assert (r["cournot_quantity"], r["cournot_price"], r["cournot_quantity_per_firm"]) == (A(280), A(170), A(140))
    assert (r["competitive_quantity"], r["competitive_price"]) == (A(420), A(30))
    assert (r["monopoly_quantity"], r["monopoly_price"]) == (A(210), A(240))
    assert ec.linear_market_structure_outcomes(450, 1, 30, n_firms=1)["cournot_price"] == A(240)  # n=1 is monopoly


def test_monopoly_quadratic_cost():  # R16 p.204-205: P = 800 - 2Q, TC = 20,000 + 50Q + 3Q^2 -> Q 75, P 650, profit 8,125
    r = ec.monopoly_optimum_quadratic_cost(800, 2, 20000, 50, 3)
    assert (r["quantity"], r["price"], r["profit"]) == (A(75), A(650), A(8125))
    assert abs(r["point_price_elasticity"]) > 1  # monopolist prices on the elastic part of demand


def test_concentration_and_hhi():  # R16 p.211-212: CR3 80% -> 90%, CR4 90% -> 92.5%; HHI3 0.225 -> 0.41
    shares = [0.35, 0.25, 0.20, 0.10, 0.025, 0.025, 0.025, 0.025]
    r3 = ec.concentration_measures(shares, top_n=3, merging_firms=[0, 1])
    assert (r3["concentration_ratio"], r3["hhi"]) == (A(0.80), A(0.225))
    assert (r3["post_merger_concentration_ratio"], r3["post_merger_hhi"]) == (A(0.90), A(0.41))
    r4 = ec.concentration_measures(shares, top_n=4, merging_firms=[0, 1])
    assert (r4["concentration_ratio"], r4["post_merger_concentration_ratio"]) == (A(0.90), A(0.925))


def test_hhi_example5():  # R16 p.212 Ex.5: ten firms at 10% -> CR4 40%, HHI4 0.04
    r = ec.concentration_measures([0.10] * 10, top_n=4)
    assert r["concentration_ratio"] == A(0.40) and r["hhi"] == A(0.04) and r["hhi_all_firms"] == A(0.10)


# ------------------------------------------------------------------------------ national accounts (R17)
def test_gdp_expenditure_and_income_example3():  # R17 p.241-242 Ex.3: GDP 1,527,258 both ways
    r = ec.gdp_expenditure_approach(898728, 269394, 333942 + 59078, 438553, 464722, -8180, 465)
    assert r["gdp"] == A(1527258)
    assert ec.gdp_income_approach(1308939, 218785, -466)["gdp"] == A(1527258)


def test_personal_income_example3():  # R17 p.241-242 Ex.3: PI 1,228,407; taxes 262,779; saving 44,048; 4.6%
    r = ec.personal_income_measures(1308939, 70604 + 93030, 34319, 56969, 174390, 965628, 898728, 18115, 4737)
    assert r["personal_income"] == A(1228407) and r["personal_taxes"] == A(262779)
    assert r["household_saving"] == A(44048) and r["saving_rate"] == A(0.046, abs=5e-4)


def test_gdp_deflator_example2_and_text():  # R17 p.232-233: car economy deflator 107; Canada 2008 121.4, 2009 118.8, -2.1%
    assert ec.gdp_deflator(6199312500, 5793750000)["deflator"][0] == A(107)
    r = ec.gdp_deflator([1599608, 1527258], [1318054, 1285604])
    assert r["deflator"][0] == A(121.4, abs=0.05) and r["deflator"][1] == A(118.8, abs=0.05)
    assert r["inflation"][1] == A(-0.021, abs=5e-4)


def test_sectoral_balances_example4():  # R17 p.244 Ex.4: S 319,802, I 261,214, G-T 82,249 -> X-M -23,661
    r = ec.sectoral_balances(private_saving=319802, investment=261214, fiscal_deficit=82249)
    assert r["trade_balance"] == A(-23661) and r["solved_for"] == "trade_balance"
    back = ec.sectoral_balances(investment=261214, fiscal_deficit=82249, trade_balance=-23661)
    assert back["private_saving"] == A(319802)


def test_is_curve_example5():  # R17 p.248-249 Ex.5: Y = 12,292.7 - 73.2r; r=4 -> Y ~12,000, C 8,020, I 2,680, NX -200
    r = ec.is_curve_linear(2000, 0.7, 400, 0.2, -30, 1500, 1000, -0.1, -200, 0.3, real_rate=4)
    assert r["is_intercept"] == A(12292.7, abs=0.05) and r["is_slope"] == A(-73.2, abs=0.05)
    assert r["output"] == A(11999.9, abs=0.2) and r["taxes"] == A(3399.9, abs=0.2)
    assert r["consumption"] == A(8020, abs=0.2) and r["investment"] == A(2680, abs=0.2)
    assert r["net_exports"] == A(-200, abs=0.2) and r["spending_multiplier"] == A(2.44, abs=5e-3)
    g2000 = ec.is_curve_linear(2000, 0.7, 400, 0.2, -30, 2000, 1000, -0.1, -200, 0.3)
    assert g2000["is_intercept"] == A(13512.2, abs=0.05)


def test_unit_labor_cost_example9():  # R17 p.264 Ex.9: EUR22 / 110 phones = 0.20 (unchanged)
    assert ec.unit_labor_cost(22, 110)["unit_labor_cost"] == A(0.20)


# ------------------------------------------------------------------------ inflation, labor (R18)
def test_price_indices_text():  # R18 p.335-336: Laspeyres 112.45, Paasche 116.03, Fisher 114.23
    r = ec.price_indices([50, 70], [3, 4.4], [70, 60], [4, 4.5])
    assert r["laspeyres"] == A(112.45, abs=0.005) and r["paasche"] == A(116.03, abs=0.005)
    assert r["fisher"] == A(114.23, abs=0.005) and r["laspeyres_inflation"] == A(0.1245, abs=5e-5)


def test_labor_market_definitions():  # R18 p.331-332 definitions: u = U / LF; participation = LF / working-age pop
    r = ec.labor_market_rates(employed=90, unemployed=10, working_age_population=125)
    assert r["labor_force"] == 100 and r["unemployment_rate"] == A(0.10) and r["participation_ratio"] == A(0.80)


# ---------------------------------------------------------------------- money and policy (R19, L2, L3)
def test_money_multiplier():  # R19 p.367-369: EUR100 at 10% -> 1,000; Ex.2 Q2: 5% -> 2,000; K19: 25% -> 4
    r = ec.money_multiplier(0.10, 100)
    assert r["money_multiplier"] == A(10) and r["total_deposits_created"] == A(1000)
    assert ec.money_multiplier(0.05, 100)["total_deposits_created"] == A(2000)
    assert ec.money_multiplier(0.25)["money_multiplier"] == A(4)


def test_quantity_theory():  # K19 p.205/207 Q6: nominal GDP 7tn, M 1tn -> V 7; R19 p.370 MV = PY
    assert ec.quantity_theory_of_money(money_supply=1e12, nominal_gdp=7e12)["velocity"] == A(7)
    r = ec.quantity_theory_of_money(money_supply=100, velocity=5, real_output=250)
    assert r["price_level"] == A(2) and r["nominal_gdp"] == A(500)
    with pytest.raises(ValueError):
        ec.quantity_theory_of_money(money_supply=100)


def test_quantity_theory_growth():  # K19 p.~199: constant V and Y, 5% money growth -> 5% inflation
    r = ec.quantity_theory_growth(money_growth=0.05)
    assert r["inflation"] == A(0.05) and r["inflation_approx"] == A(0.05)
    assert ec.quantity_theory_growth(0.06, 0, 0.02)["inflation"] == A(1.06 / 1.02 - 1)


def test_fisher_relation():  # R19 p.374-375: 2% real + 2% inflation -> 4%; R18 p.343: 3.5% nominal - 1.5% real -> 2%
    r = ec.fisher_relation(real_rate=0.02, inflation=0.02)
    assert r["approx"] == A(0.04) and r["nominal_rate"] == A(1.02 * 1.02 - 1)
    assert ec.fisher_relation(real_rate=0.02, inflation=0.03)["approx"] == A(0.05)
    b = ec.fisher_relation(nominal_rate=0.035, real_rate=0.015)
    assert b["approx"] == A(0.02) and b["inflation"] == A(1.035 / 1.015 - 1)
    assert ec.fisher_relation(nominal_rate=b["nominal_rate"], inflation=b["inflation"])["real_rate"] == A(0.015)


def test_fiscal_multiplier():  # R19 p.415: c 0.9, t 0.2 -> 3.57; c 0.9 no tax -> 10; K19: c 0.8, t 0.25 -> 2.5, $100 -> $250
    assert ec.fiscal_multiplier(0.9, 0.2)["multiplier"] == A(3.57, abs=5e-3)
    assert ec.fiscal_multiplier(0.9)["multiplier"] == A(10)
    r = ec.fiscal_multiplier(0.8, 0.25, change_in_spending=100)
    assert r["multiplier"] == A(2.5) and r["output_change"] == A(250) and r["balanced_budget_multiplier"] == 1


def test_neutral_rate():  # R19 p.396 Eq.4: 2% + 2.5% = 4.5%
    r = ec.neutral_policy_rate(0.025, 0.02, policy_rate=0.05)
    assert r["neutral_rate"] == A(0.045) and r["stance"] == "contractionary"


def test_taylor_rule_l3_example26():  # L3CME p.78 Ex.26: 3.5% + 0.5(1 - 3) + 0.5(4 - 2.5) = 3.25%
    r = ec.taylor_rule_growth_form(0.035, 0.01, 0.03, 0.04, 0.025)
    assert r["optimal_rate"] == A(0.0325) and r["growth_component"] == A(-0.01)


def test_taylor_rule_l2_form():  # L2FX p.582 Eq.15 (Taylor 1993 textbook case: r_n 2%, pi 4%, target 2%, gap 1% -> 7.5%)
    r = ec.taylor_rule_rate(0.02, 0.04, 0.02, 0.01)
    assert r["policy_rate"] == A(0.075) and r["real_policy_rate"] == A(0.035)
    on_target = ec.taylor_rule_rate(0.02, 0.02, 0.02, 0.0)
    assert on_target["policy_rate"] == A(on_target["neutral_nominal_rate"])  # Eq.16 footnote 22


# ---------------------------------------------------------------------------- growth (R17, L2G, L3EMV)
def test_growth_accounting_spain():  # L2G p.684 Ex.15: 0.383x5.92% + 0.617x1.23% - 0.73% = 2.30%; capital 2.27, labor 0.76
    r = ec.growth_accounting(0.383, 0.0592, 0.0123, tfp_growth=-0.0073)
    assert r["output_growth"] == A(0.0230, abs=5e-5)
    assert r["capital_contribution"] == A(0.0227, abs=5e-5) and r["labor_contribution"] == A(0.0076, abs=5e-5)


def test_growth_accounting_l3():  # L3EMV p.147: China 2.5% + 0.5x12% + 0.5x1.5% = 9.25%; p.156 Ex.3: Eurozone 2.2%
    assert ec.growth_accounting(0.5, 0.12, 0.015, tfp_growth=0.025)["output_growth"] == A(0.0925)
    assert ec.growth_accounting(0.4, 0.03, 0.0, tfp_growth=0.01)["output_growth"] == A(0.022)


def test_solow_residual_china():  # L3EMV p.144 Exhibit 1: China 1995-2007 Y 9.25%, K 12.81%, L 2.78%, a 0.5 -> TFP ~1.45%
    r = ec.growth_accounting(0.5, 0.1281, 0.0278, output_growth=0.0925)
    assert r["tfp_growth"] == A(0.0145, abs=1e-4)
    assert ec.growth_accounting(0.5, 0.1281, 0.0278, tfp_growth=r["tfp_growth"])["output_growth"] == A(0.0925)


def test_potential_gdp_growth():  # R17 p.290 Ex.17: US 0.9% + 1.9% = 2.8%; L2G p.684 Ex.15: Spain 1.23% + 1.07% = 2.3%
    assert ec.potential_gdp_growth(0.009, 0.019)["potential_growth"] == A(0.028)
    assert ec.potential_gdp_growth(0.0123, 0.0107)["potential_growth"] == A(0.023, abs=5e-4)
    assert ec.potential_gdp_growth(0.009, 0.019, actual_gdp_growth=0.038)["growth_gap"] == A(0.01)  # Ex.18 p.291


def test_cobb_douglas_marginal_products():  # L2G p.634 fn.9: MPK = aY/K; check by finite difference and CRS
    base = ec.cobb_douglas_output(1.5, 200, 100, 0.35)
    h = 1e-4
    up = ec.cobb_douglas_output(1.5, 200 + h, 100, 0.35)["output"]
    assert (up - base["output"]) / h == A(base["marginal_product_capital"], rel=1e-5)
    assert ec.cobb_douglas_output(1.5, 400, 200, 0.35)["output"] == A(2 * base["output"])  # constant returns
    assert base["output_per_worker"] == A(1.5 * 2 ** 0.35)  # y = A k^a, Eq. 3


def test_neoclassical_steady_state_example10():  # L2G p.661-662 Ex.10: China 6.58%, Japan 0.35%, Ireland 1.41%
    assert ec.neoclassical_steady_state(0.025, 1 - 0.465, 0.012)["output_growth"] == A(0.0658, abs=5e-5)
    assert ec.neoclassical_steady_state(0.002, 1 - 0.573, 0.0)["output_growth"] == A(0.0035, abs=5e-5)
    assert ec.neoclassical_steady_state(0.008, 1 - 0.567, 0.0)["output_growth"] == A(0.0141, abs=5e-5)
    assert ec.neoclassical_steady_state(-0.0073, 0.383, 0.0123)["output_growth"] == A(0.0005, abs=5e-5)  # Ex.15 p.686


def test_steady_state_output_capital_example11():  # L2G p.669 Ex.11: g 3.08%; Y/K 0.7040 (s 20%), 0.4693 (s 30%)
    old = ec.neoclassical_steady_state(0.02, 0.35, 0.01, saving_rate=0.20, depreciation_rate=0.10)
    new = ec.neoclassical_steady_state(0.02, 0.35, 0.01, saving_rate=0.30, depreciation_rate=0.10)
    assert old["per_capita_growth"] == A(0.0308, abs=5e-5)
    # the reading rounds theta/(1 - a) to 0.0308 before dividing by s (0.1408/0.2 = 0.7040); exact is 0.70385
    assert old["output_capital_ratio"] == A(0.7040, abs=2e-4) and new["output_capital_ratio"] == A(0.4693, abs=2e-4)


def test_transition_growth_example11():  # L2G p.669-670 Ex.11: 5.54% at t=0, 4.39% at 5y; k 1.2964, y 1.0951
    r0 = ec.neoclassical_transition_growth(0.02, 0.35, 0.30, 0.7040, 0.4693)
    assert r0["per_capita_income_growth"] == A(0.0554, abs=5e-5)
    r5 = ec.neoclassical_transition_growth(0.02, 0.35, 0.30, 0.5947, 0.4693, reference_output_capital_ratio=0.7040)
    assert r5["per_capita_income_growth"] == A(0.0439, abs=5e-5)
    assert r5["capital_per_worker_ratio"] == A(1.2964, abs=5e-5) and r5["income_per_worker_ratio"] == A(1.0951, abs=5e-5)
    ss = ec.neoclassical_transition_growth(0.02, 0.35, 0.30, 0.4693, 0.4693)
    assert ss["per_capita_income_growth"] == A(0.02 / 0.65)  # converges to theta/(1 - a)


def test_capital_stock_path_example15():  # L2G p.683 Ex.15: K2000 2,324.1; K2001 2,476.3
    r = ec.capital_stock_path(2177.2, [1156.4, 1198.6], [0.263, 0.264], [0.136, 0.137])
    assert r["capital_stock"][0] == A(2324.1, abs=0.05) and r["capital_stock"][1] == A(2476.3, abs=0.05)


def test_gdp_compound_growth_example1():  # L2G p.627 Ex.1: Japan 4.14%; Argentina at 4.14% -> $70,294
    assert ec.gdp_compound_growth(3048, 34828, 60)["growth_rate"] == A(0.0414, abs=5e-5)
    assert ec.gdp_compound_growth(1185, 30079, 60)["growth_rate"] == A(0.0554, abs=5e-5)
    r = ec.gdp_compound_growth(6164, 13468, 60, alternative_growth=0.0414)
    assert r["growth_rate"] == A(0.0131, abs=5e-5) and r["end_value_at_alternative"] == A(70294, rel=1e-4)


def test_equity_growth_decomposition_exhibit3():  # L2G p.630 Exhibit 3: price 7.15%, total 10.82%
    r = ec.equity_growth_decomposition(0.0301, 0.0394, -0.0012, 0.0032, 0.0367)
    assert r["price_return"] == A(0.0715) and r["total_return"] == A(0.1082)


# ------------------------------------------------------------------------------------ trade (R20-R21)
def test_comparative_advantage_example3():  # R20 p.448-449 Ex.3: UK machine costs 2 cloth, India 8; ToT range 2-8
    r = ec.comparative_advantage(4, 8, 2, 16)
    assert r["opportunity_cost_good1_a"] == A(2) and r["opportunity_cost_good1_b"] == A(8)
    assert r["comparative_advantage_good1"] == "A" and r["comparative_advantage_good2"] == "B"
    assert r["absolute_advantage_good1"] == "A" and r["absolute_advantage_good2"] == "B"
    assert r["terms_of_trade_range_good1"] == [A(2), A(8)]
    assert ec.comparative_advantage(4, 8, 2, 6)["comparative_advantage_good2"] == "B"  # p.449: India 6 cloth still CA


def test_tariff_example5():  # R20 p.458-459 Ex.5: CS -185,000; PS +120,000; revenue 40,000; DWL 25,000
    r = ec.tariff_welfare_effects(5, 6, 110000, 130000, 200000, 170000)
    assert r["consumer_surplus_loss"] == A(185000) and r["producer_surplus_gain"] == A(120000)
    assert r["tariff_revenue"] == A(40000) and r["deadweight_loss"] == A(25000)


def test_marshall_lerner():  # R21 p.546-547: 0.4x0.75 + 0.6(0.65 - 1) = 0.09; p.551-552 Ex.8: -0.03, threshold 4/7
    assert ec.marshall_lerner_condition(400, 600, 0.75, 0.65)["ml_value"] == A(0.09)
    r = ec.marshall_lerner_condition(200, 180, 0.3, 0.6)
    assert r["ml_value"] == A(-0.03, abs=0.005) and not r["satisfied"]
    assert r["min_export_weight"] == A(4 / 7)


# ------------------------------------------------------------------------------------- spot FX
def test_cross_rates_l2_text():  # L2FX p.526-527: JPY/EUR 104.61/104.65; GBP/EUR 0.8724/0.8726 (inverting USD/GBP)
    r = ec.fx_cross_rate("JPY/USD", 76.64, "USD/EUR", 1.3649, "JPY/EUR", ask_1=76.66, ask_2=1.3651)
    assert r["bid"] == A(104.61, abs=0.005) and r["ask"] == A(104.65, abs=0.005)
    g = ec.fx_cross_rate("USD/GBP", 1.5644, "USD/EUR", 1.3649, "GBP/EUR", ask_1=1.5646, ask_2=1.3651)
    assert g["price_leg"] == [A(0.63914, abs=5e-6), A(0.63922, abs=5e-6)]
    assert g["bid"] == A(0.8724, abs=5e-5) and g["ask"] == A(0.8726, abs=5e-5)


def test_cross_rates_l2_example1():  # L2FX p.528-529 Ex.1: SEK/EUR 10.0077/10.0094; JPY/CAD 85.76/85.80
    r = ec.fx_cross_rate("SEK/USD", 6.8739, "USD/EUR", 1.4559, "SEK/EUR", ask_1=6.8741, ask_2=1.4561)
    assert r["bid"] == A(10.0077, abs=5e-5) and r["ask"] == A(10.0094, abs=5e-5)
    j = ec.fx_cross_rate("JPY/USD", 81.87, "CAD/USD", 0.9544, "JPY/CAD", ask_1=81.89, ask_2=0.9546)
    assert j["base_leg"] == [A(1.04756, abs=5e-6), A(1.04778, abs=5e-6)]
    assert j["bid"] == A(85.76, abs=0.005) and j["ask"] == A(85.80, abs=0.005)
    assert j["bid"] < j["ask"]


def test_cross_rates_mid_l1_example5():  # R21 p.524 Ex.5: CHF/EUR 1.3381; GBP/EUR 0.8808
    assert ec.fx_cross_rate("USD/EUR", 1.3960, "CHF/USD", 0.9585, "CHF/EUR")["mid"] == A(1.3381, abs=5e-5)
    assert ec.fx_cross_rate("USD/EUR", 1.3960, "USD/GBP", 1.5850, "GBP/EUR")["mid"] == A(0.8808, abs=5e-5)
    with pytest.raises(ValueError):
        ec.fx_cross_rate("USD/EUR", 1.39, "JPY/GBP", 130, "CHF/EUR")


def test_triangular_arbitrage_example1():  # L2FX p.528-529 Ex.1 Q3-Q4: 85.73/85.75 -> buy CAD from dealer, ~JPY 0.01; 85.74/85.81 none
    r = ec.fx_triangular_arbitrage("JPY/CAD", 85.73, 85.75, "JPY/USD", 81.87, 81.89, "CAD/USD", 0.9544, 0.9546)
    assert r["arbitrage"] and round(r["profit_per_base_unit"], 2) == 0.01
    assert r["action"].startswith("buy CAD from the dealer")
    r = ec.fx_triangular_arbitrage("JPY/CAD", 85.74, 85.81, "JPY/USD", 81.87, 81.89, "CAD/USD", 0.9544, 0.9546)
    assert not r["arbitrage"] and r["profit_per_base_unit"] == 0


def test_fx_percent_change_example5():  # R21 p.524 Ex.5 Q3-Q4: EUR -0.7% vs USD; USD +0.4% vs GBP
    assert ec.fx_percent_change(1.3960, 1.3863)["base_currency_change"] == A(-0.007, abs=5e-4)
    assert ec.fx_percent_change(1.5850, 1.5794)["price_currency_change"] == A(0.004, abs=5e-4)


def test_bid_offer_spread():  # L2FX p.524: interbank USD/EUR 1.3649/1.3651 is two pips wide
    r = ec.fx_bid_offer_spread(1.3649, 1.3651)
    assert r["pips"] == A(2) and r["mid"] == A(1.3650)


# ------------------------------------------------------------------------------------- forwards
def test_forward_premium_l2_example2():  # L2FX p.532 Ex.2: CAD/AUD 1.0145, CAD 1.41%, AUD 4.87%, 270d -> -0.0254
    r = ec.fx_forward_cip(spot=1.0145, rate_price_ccy=0.0141, rate_base_ccy=0.0487, days=270)
    assert r["forward_premium"] == A(-0.0254, abs=5e-5)
    tau = 270 / 360  # L2FX p.531 premium formula: F - S = S (tau / (1 + i_B tau)) (i_P - i_B)
    assert r["forward_premium"] == A(1.0145 * tau / (1 + 0.0487 * tau) * (0.0141 - 0.0487))


def test_forward_points_l1_example6():  # R21 p.531-532 Ex.6 Q5: GBP/EUR 0.8489, GBP 1.325%, EUR 1.370%, 270d -> 0.84862, -2.8 pts
    r = ec.fx_forward_cip(0.8489, 0.01325, 0.01370, 270)
    assert r["forward"] == A(0.84862, abs=5e-6) and r["forward_points"] == A(-2.8, abs=0.05)


def test_forward_jpy_points_and_uip():  # L2FX p.547-548 Ex.4: JPY/AUD 79.25, 1% vs 5% -> -301.9 points (JPY scale 100)
    r = ec.fx_forward_cip(79.25, 0.01, 0.05, 360, points_scale=100)
    assert r["forward_points"] == A(-301.9, abs=0.05)
    u = ec.uip_expected_spot(79.25, 0.01, 0.05)
    assert u["expected_spot"] == A(r["forward"])  # forward = expected spot when CIP and UIP both hold
    assert u["expected_change_approx"] == A(-0.04)  # Q1: AUD expected to fall ~4%


def test_uip_example4_q7():  # L2FX p.548 Ex.4 Q7: JPY/GBP 129.67 x 1.001 / 1.03 = 126.02
    assert ec.uip_expected_spot(129.67, 0.001, 0.03)["expected_spot"] == A(126.02, abs=0.005)


def test_cip_arbitrage_direction():  # consistency: a market forward above CIP means borrow price ccy, buy base spot
    r = ec.fx_forward_cip(1.0, 0.02, 0.05, 360, market_forward=1.0)
    assert r["mispricing"] > 0 and r["arbitrage_action"].startswith("market forward too high")
    assert ec.fx_forward_cip(1.0, 0.02, 0.05, 360, market_forward=0.9)["mispricing"] < 0


def test_forward_outright_from_points():  # L2FX p.533: 1.3549 - 15.9 pts -> 1.35331; R21 p.531 Ex.6 Q1: 0.87506
    assert ec.fx_forward_outright(1.3549, -15.9)["forward_bid"] == A(1.35331)
    assert ec.fx_forward_outright(0.8752, -1.4)["forward_bid"] == A(0.87506)


def test_forward_outright_example3_and_pct():  # L2FX p.536-537 Ex.3 Q1 1.49136, Q3 swap 1.48860; R21 p.532 Q4 0.87257
    r = ec.fx_forward_outright(1.4939, -25.4, 1.4941, -24.6)
    assert r["forward_bid"] == A(1.49136)
    assert ec.fx_forward_outright(1.4939, -56.5, 1.4941, -54.0, use_mid_spot=True)["forward_offer"] == A(1.48860)
    assert ec.fx_forward_outright(0.8752, premium_pct=-0.003)["forward_bid"] == A(0.87257, abs=5e-6)


def test_forward_mtm_text():  # L2FX p.534-535: long GBP 10m at 1.6100; close 1.6340 -> +AUD 240,000, PV 237,154
    r = ec.fx_forward_mark_to_market(1.6100, 10e6, "base", "long", 1.6210, 1.6215, 130, 140, 90, 0.048)
    assert r["closing_rate"] == A(1.6340) and r["settlement_cash_flow"] == A(240000)
    assert r["mark_to_market"] == A(237154, abs=1)


def test_forward_mtm_example3_q4_q5():  # L2FX p.537-538 Ex.3: short NZD 10m -> +USD 79,938; short USD 10m -> -NZD 139,959
    r = ec.fx_forward_mark_to_market(0.7900, 10e6, "base", "short", 0.7825, 0.7830, -12.1, -10.0, 90, 0.0031)
    assert r["closing_rate"] == A(0.7820) and r["mark_to_market"] == A(79938, abs=1)
    r = ec.fx_forward_mark_to_market(0.7900, 10e6, "price", "short", 0.7825, 0.7830, -12.1, -10.0, 90, 0.0331)
    assert r["settlement_cash_flow"] == A(-141117, abs=1) and r["mark_to_market"] == A(-139959, abs=1)


def test_forward_mtm_example3_q2():  # L2FX p.537 Ex.3 Q2: sold CHF 1m at 1.4850 CHF/GBP, rolled at mid 1.4940 -> +GBP 4,056.63
    r = ec.fx_forward_mark_to_market(1.4850, 1e6, "price", "short", 1.4939, 1.4941, use_mid_spot=True)
    assert r["settlement_cash_flow"] == A(4056.63, abs=0.01) and r["cash_flow_currency"] == "base"


# ------------------------------------------------------------------------------- parity conditions
def test_ppp_absolute_and_relative():  # L2FX p.548-550: EUR100 good at 1.40 USD/EUR -> $140; 10% vs 5% inflation -> ~+5%
    assert ec.ppp_absolute(140, 100)["ppp_spot"] == A(1.40)
    assert ec.ppp_absolute(140, 100, actual_spot=1.54)["base_currency_overvaluation"] == A(0.10)
    r = ec.ppp_relative_expected_spot(1.0, 0.10, 0.05)
    assert r["expected_change_approx"] == A(0.05) and r["expected_change_pct"] == A(1.10 / 1.05 - 1)


def test_international_fisher_example5():  # L2FX p.553-554 Ex.5 Q3: equal real rates -> inflation gap = 1% - 5% (4%)
    r = ec.international_fisher_effect(0.01, 0.05, 0.0, 0.04)
    assert r["implied_inflation_differential"] == A(-0.04) and r["real_rate_differential"] == A(0.0)


def test_real_exchange_rate():  # R21 p.503 text: +10% S, +5% Eurozone, +2% UK -> ~13%; p.505 Ex.1: +5/+5/+2 -> ~8%
    r = ec.real_exchange_rate_change(0.10, inflation_price_ccy=0.02, inflation_base_ccy=0.05)
    assert r["real_change_approx"] == A(0.13) and r["real_change"] == A(1.10 * 1.05 / 1.02 - 1)
    assert ec.real_exchange_rate_change(0.05, 0.02, 0.05)["real_change_approx"] == A(0.08)
    q0 = ec.real_exchange_rate(1.0, 100, 100)["real_exchange_rate"]
    q1 = ec.real_exchange_rate(1.05, 102, 105)["real_exchange_rate"]
    assert q1 / q0 - 1 == A(ec.real_exchange_rate_change(0.05, 0.02, 0.05)["real_change"])
    assert ec.real_exchange_rate(1.3, 105, 105)["real_exchange_rate"] == A(1.3)  # L2FX Ex.5 Q2: equal CPI moves cancel


def test_real_fx_rate_model():  # L2FX p.561 Eq.12: higher base real rate / lower base risk premium -> stronger base
    r = ec.real_fx_rate_model(0.0, 0.03, 0.01, risk_premium_base_ccy=0.005)
    assert r["deviation_from_long_run"] == A(0.015)
    assert ec.real_fx_rate_model(1.2, 0.02, 0.02)["real_exchange_rate"] == A(1.2)


def test_carry_trade_example8():  # L2FX p.564-565 Ex.8 Q5: JPY/AUD 87.40 -> 86.42, AUD 4.5%, JPY 0.10% -> 3.33% gross, 3.23% net
    r = ec.carry_trade_return(87.40, 86.42, investment_rate=0.045, funding_rate=0.001)
    assert r["gross_return"] == A(0.0333, abs=5e-4) and r["net_return"] == A(0.0323, abs=5e-4)
    cross0 = ec.fx_cross_rate("JPY/USD", 81.30, "USD/AUD", 1.0750, "JPY/AUD")["mid"]
    assert cross0 == A(87.40, abs=0.005)


def test_input_validation():
    with pytest.raises(ValueError):
        ec.fx_cross_rate("USDEUR", 1.1, "JPY/USD", 150, "JPY/EUR")
    with pytest.raises(ValueError):
        ec.supply_demand_equilibrium(100, 2, 0, 1)
    with pytest.raises(ValueError):
        ec.growth_accounting(0.3, 0.02, 0.01)
    with pytest.raises(ValueError):
        ec.concentration_measures([0.6, 0.6])
    assert math.isfinite(ec.fiscal_multiplier(0.0)["multiplier"])


# ---------------------------------------------------------------- regressions found by independent verification

def test_concentration_top_n_must_be_whole():
    with pytest.raises(ValueError):
        ec.concentration_measures([0.5, 0.3, 0.2], top_n=2.5)


def test_fx_forward_cip_names_its_sibling():
    doc = ec.fx_forward_cip.__doc__
    assert "currency_forward_price" in doc and "act/360" in doc
