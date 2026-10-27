"""Fixed income credit models, binomial trees with embedded options, convertibles. Golden values from the KB.

KB paths are relative to E:/CLAUDE/CFA/MD.
L1C = Level-1/08-Fixed-Income/Curriculum-2014/SS16-R56-fundamentals-of-credit-analysis.md
L2C = Level-2/08-Fixed-Income/Curriculum-2014/SS14-R45-credit-analysis-models.md
L2E = Level-2/08-Fixed-Income/Curriculum-2014/SS14-R47-valuing-bonds-with-embedded-options.md
"""
import pytest

from cfa_calc import fixed_income as fi

A = pytest.approx
PAR = [0.035, 0.042, 0.047, 0.052]  # L2E p.373 issuer on-the-run par curve
CALL100 = [[1, 100], [2, 100], [3, 100]]


@pytest.fixture(scope="module")
def tree10():
    return fi.binomial_tree_calibrate(PAR, 0.10)["tree"]


@pytest.fixture(scope="module")
def tree20():
    return fi.binomial_tree_calibrate(PAR, 0.20)["tree"]


# ------------------------------------------------------------------ credit
def test_credit_expected_loss():
    # L1C p.548 definition: EL = PD x loss severity, loss severity = 1 - recovery
    o = fi.credit_expected_loss(0.02, recovery_rate=0.40, exposure=1_000_000)
    assert o["expected_loss_rate"] == A(0.012) and o["expected_loss"] == A(12_000)
    assert o["approx_credit_spread_bp"] == A(120)
    assert fi.credit_expected_loss(0.02, loss_given_default=0.6)["expected_loss"] == A(1.2)
    with pytest.raises(ValueError):
        fi.credit_expected_loss(0.02)


def test_credit_cva_bond_hand_calc():
    # CVA framework (exposure, recovery, LGD, POD, POS, DF), hand-computed for a 3y 5% bond, flat 3%,
    # hazard 1.25%, recovery 40%
    o = fi.credit_cva_bond(0.05, 3, 0.0125, 0.40, risk_free_rate=0.03)
    vnd = 5 / 1.03 + 5 / 1.03 ** 2 + 105 / 1.03 ** 3
    ex = [5 + 5 / 1.03 + 105 / 1.03 ** 2, 5 + 105 / 1.03, 105.0]
    pod = [0.0125, 0.0125 * 0.9875, 0.0125 * 0.9875 ** 2]
    cva = sum(e * 0.6 * p / 1.03 ** (t + 1) for t, (e, p) in enumerate(zip(ex, pod)))
    assert o["value_default_free"] == A(vnd) and o["cva"] == A(cva) and o["fair_value"] == A(vnd - cva)
    assert [r["exposure"] for r in o["schedule"]] == A(ex)
    assert o["cumulative_pod"] == A(1 - 0.9875 ** 3)
    assert fi.bond_ytm(o["fair_value"], 0.05, 3, 1)["ytm"] == A(o["ytm"])
    assert o["credit_spread"] == A(0.0125 * 0.6, abs=1e-3)  # spread ~ PD x LGD
    # zero hazard -> no CVA, no spread
    z = fi.credit_cva_bond(0.05, 3, 0.0, 0.4, spot_rates=[0.02, 0.025, 0.03])
    assert z["cva"] == 0 and z["credit_spread"] == A(0, abs=1e-12)


def test_credit_reduced_form_constant():
    # L2C Example 5 p.293: K 700, lambda 0.01, gamma 0.4, P 0.96 -> PD 0.995%, EL 2.79, PV(EL) 2.68
    o = fi.credit_reduced_form_constant(700, 0.01, 0.4, 1, riskless_zero_price=0.96)
    assert o["default_probability"] == A(0.00995, abs=5e-6)
    assert o["expected_loss"] == A(2.79, abs=5e-3) and o["pv_expected_loss"] == A(2.68, abs=5e-3)
    assert o["credit_spread"] == A(0.004)
    # L2C Example 7 p.302: average spread 3.588% - 1.164% = 2.424% expected loss per year (lambda x gamma)


def test_credit_structural_merton():
    # L2C Example 4 p.285-286: A 1000, K 700, r 1%, u 3%, sigma 30%, 1y -> PD 12.74%, EL 11.78, PV(EL) 13.28
    o = fi.credit_structural_merton(1000, 700, 0.01, 0.30, 1, 0.03)
    assert o["default_probability"] == A(0.1274, abs=5e-5)
    assert o["expected_loss"] == A(11.78, abs=0.01) and o["pv_expected_loss"] == A(13.28, abs=0.01)
    assert o["equity_value"] + o["debt_value"] == A(1000)
    assert 700 * 2.718281828 ** -0.01 - o["debt_value"] == A(o["pv_expected_loss"], abs=1e-6)


def test_credit_pv_expected_loss_curve():
    # L2C Example 8 p.302-305: Q1 25 EUR -> 24.2191 / 24.5350 / 0.3159; Q4 full bond -> 1126.9305 / 1197.4335
    q1 = fi.credit_pv_expected_loss_curve([25], [3.6384], [0.005160], [0.003561])
    assert q1["credit_adjusted_value"] == A(24.2191, abs=5e-4) and q1["pv_expected_loss"] == A(0.3159, abs=5e-4)
    t = [0.1370, 0.7205, 1.3068, 1.8877, 2.4767, 3.0575, 3.6384, 4.2247, 4.8082, 5.3945, 5.9753, 6.5562, 7.1425]
    rf = [.000134, .000980, .001132, .001760, .002486, .003523, .005160, .007276, .009540, .011730, .013719,
          .015536, .017238]
    sp = [.000696, .001179, .001265, .001623, .002037, .002628, .003561, .004767, .006058, .007306, .008440,
          .009475, .010446]
    q4 = fi.credit_pv_expected_loss_curve([25] * 12 + [1025], t, rf, sp)
    assert q4["credit_adjusted_value"] == A(1126.9305, abs=0.01) and q4["risk_free_value"] == A(1197.4335, abs=0.01)
    assert q4["pv_expected_loss"] == A(70.5030, abs=0.01)


def test_credit_migration_expected_return():
    o = fi.credit_migration_expected_return([0.9, 0.08, 0.02], [0, 0.01, 0.03], 7.2, 62)
    hand = 0.08 * (-0.072 + 0.5 * 62 * 1e-4) + 0.02 * (-0.216 + 0.5 * 62 * 9e-4)
    assert o["expected_return"] == A(hand) and o["state_returns"][0] == 0
    with pytest.raises(ValueError):
        fi.credit_migration_expected_return([0.5, 0.4], [0, 0.01], 5)


def test_credit_ratios():
    # L1C Example 10 p.596-597 (Freescale): 4.9x, 6.9x, 7.7x gross; 6.6x net
    o = fi.credit_ratios(990, 7611, 590, cash=1050, debt_tranches=[4899, 1948, 764])
    assert [round(x, 1) for x in o["cumulative_leverage"]] == [4.9, 6.9, 7.7]
    assert round(o["net_debt_to_ebitda"], 1) == 6.6 and o["ebitda_coverage"] == A(990 / 590)
    assert fi.credit_ratios(100, 300, equity=200, ffo=60)["debt_to_capital"] == A(0.6)


# ------------------------------------------------------------------ trees
def test_binomial_tree_calibrate(tree10):
    # L2E Exhibit 5 / capped floater text p.393-397: year-2 top node 7.0053%; spots reproduce bootstrap
    o = fi.binomial_tree_calibrate(PAR, 0.10)
    assert tree10[1] == A([0.044448, 0.054289], abs=5e-7) and tree10[2][2] == A(0.070053, abs=5e-7)
    assert tree10[2][1] / tree10[2][0] == A(2.718281828 ** 0.2)
    assert o["spot_rates"] == A([0.035, 0.042148, 0.047352, 0.052706], abs=5e-7)


def test_binomial_tree_bond_value_callable_putable(tree10, tree20):
    # L2E: option-free 104.643 (p.381), callable at 100 from year 1: 102.899 (p.382), call schedule
    # 102/101/100: 103.942 (p.384), putable at 100: 105.327 (p.391); 20% vol: callable 102.108 (p.385),
    # putable 106.010 (p.392)
    assert fi.binomial_tree_bond_value(tree10, 0.065, 4)["value"] == A(104.643, abs=5e-4)
    c = fi.binomial_tree_bond_value(tree10, 0.065, 4, call_schedule=CALL100)
    assert c["value"] == A(102.899, abs=5e-4) and c["call_option_value"] == A(104.643 - 102.899, abs=1e-3)
    assert fi.binomial_tree_bond_value(tree10, 0.065, 4, call_schedule=[[1, 102], [2, 101], [3, 100]])["value"] == \
        A(103.942, abs=5e-4)
    p = fi.binomial_tree_bond_value(tree10, 0.065, 4, put_schedule=CALL100)
    assert p["value"] == A(105.327, abs=5e-4) and p["put_option_value"] == A(0.684, abs=1e-3)
    assert fi.binomial_tree_bond_value(tree20, 0.065, 4)["value"] == A(104.643, abs=5e-4)
    assert fi.binomial_tree_bond_value(tree20, 0.065, 4, call_schedule=CALL100)["value"] == A(102.108, abs=5e-4)
    assert fi.binomial_tree_bond_value(tree20, 0.065, 4, put_schedule=CALL100)["value"] == A(106.010, abs=5e-4)


def test_binomial_tree_bond_value_step_up(tree10):
    # L2E p.393: single step-up 4.25%/7.5% noncallable 102.082, callable at par yrs 2-3 100.031;
    # multiple step-up 4.2/5/6/7%: noncallable 101.012, callable at par from year 1 99.996
    single = [0.0425, 0.0425, 0.075, 0.075]
    assert fi.binomial_tree_bond_value(tree10, coupons=single)["value"] == A(102.082, abs=5e-4)
    assert fi.binomial_tree_bond_value(tree10, coupons=single, call_schedule=[[2, 100], [3, 100]])["value"] == \
        A(100.031, abs=5e-4)
    multi = [0.042, 0.05, 0.06, 0.07]
    assert fi.binomial_tree_bond_value(tree10, coupons=multi)["value"] == A(101.012, abs=5e-4)
    assert fi.binomial_tree_bond_value(tree10, coupons=multi, call_schedule=CALL100)["value"] == A(99.996, abs=5e-4)


def test_binomial_tree_oas(tree10, tree20):
    # L2E p.386: price 102.218 -> OAS 35 bp at 10% vol, about -6 bp at 20% vol
    assert fi.binomial_tree_oas(102.218, tree10, 0.065, 4, call_schedule=CALL100)["oas_bp"] == A(35, abs=0.05)
    assert fi.binomial_tree_oas(102.218, tree20, 0.065, 4, call_schedule=CALL100)["oas_bp"] == A(-6, abs=0.5)


def test_binomial_tree_effective_duration():
    # L2E p.388-389: OAS 35 bp, +/-25 bp par-curve shift -> V+ 101.621, V- 102.765 (V0 102.218)
    o = fi.binomial_tree_effective_duration(PAR, 0.10, 0.065, 4, oas=0.0035, shift=0.0025, call_schedule=CALL100)
    assert o["value_up"] == A(101.621, abs=1e-3) and o["value_down"] == A(102.765, abs=1e-3)
    assert o["value"] == A(102.218, abs=5e-4)
    assert o["effective_duration"] == A((102.765 - 101.621) / (2 * 102.218 * 0.0025), abs=2e-3)
    assert o["effective_convexity"] < 0  # callable near the call price: negative convexity
    straight = fi.binomial_tree_effective_duration(PAR, 0.10, 0.065, 4)
    assert straight["effective_convexity"] > 0


def test_binomial_tree_capped_floater(tree10):
    # L2E p.397: reference-rate floater = 100; cap 7.25% -> 99.724; cap 7.75% -> 99.858
    o = fi.binomial_tree_capped_floater(tree10, cap=0.0725)
    assert o["value_uncapped"] == A(100) and o["value"] == A(99.724, abs=5e-4)
    assert o["cap_value"] == A(100 - 99.724, abs=5e-4)
    assert fi.binomial_tree_capped_floater(tree10, cap=0.0775)["value"] == A(99.858, abs=5e-4)
    assert fi.binomial_tree_capped_floater(tree10, floor=0.04)["floor_value"] > 0


def test_binomial_tree_pathwise_value(tree10):
    # pathwise valuation equals backward induction for an option-free bond (104.643)
    o = fi.binomial_tree_pathwise_value(tree10, 0.065)
    assert o["paths"] == 8 and o["value"] == A(104.643, abs=5e-4)


def test_convertible_bond_metrics():
    # L2E p.399-402 (ADC, Exhibit 23): conversion value 835.56, MCP 42.06, premium 9.06 / 27.5%, income differential 1.37,
    # payback 6.6 years, premium over straight value 8.5%, conversion price at issue 39.49
    o = fi.convertible_bond_metrics(1065, 25.32, 33, 981.9, 0.0575, 1000, 0.90, issue_price=1000)
    assert o["conversion_value"] == A(835.56) and o["minimum_value"] == A(981.9)
    assert o["market_conversion_price"] == A(42.06, abs=5e-3)
    assert o["conversion_premium_per_share"] == A(9.06, abs=5e-3)
    assert o["conversion_premium_ratio"] == A(0.275, abs=5e-4)
    assert o["favorable_income_differential_per_share"] == A(1.37, abs=5e-3)
    assert o["premium_payback_years"] == A(6.6, abs=0.05)
    assert o["premium_over_straight_value"] == A(0.085, abs=5e-4)
    assert o["conversion_price_at_issue"] == A(39.49, abs=5e-3)


# ---------------------------------------------------------------- regressions found by independent verification

@pytest.mark.parametrize("call", [
    lambda t: fi.binomial_tree_bond_value(t, 0.05, 1.5),
    lambda t: fi.binomial_tree_capped_floater(t, years=1.5),
    lambda t: fi.binomial_tree_pathwise_value(t, 0.05, years=1.5),
])
def test_tree_fractional_years_rejected_not_truncated(call):
    with pytest.raises(ValueError):
        call([[0.03], [0.035, 0.045]])
