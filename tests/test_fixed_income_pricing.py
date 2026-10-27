"""Fixed income pricing and yield measures. Golden values from the CFA curriculum knowledge base.

KB paths are relative to E:/CLAUDE/CFA/MD.
L1V = Level-1/08-Fixed-Income/Curriculum-2014/SS15-R54-introduction-to-fixed-income-valuation.md
L1R = Level-1/08-Fixed-Income/Curriculum-2014/SS16-R55-understanding-fixed-income-risk-and-return.md
"""
import pytest

from cfa_calc import fixed_income as fi

A = pytest.approx


def test_bond_price_conventions_example():
    # conventions.md example
    assert fi.bond_price(face=100, coupon_rate=0.05, ytm=0.06, years=10, freq=2)["price"] == A(92.5613, abs=1e-4)


@pytest.mark.parametrize("cpn, n, r, expected", [(2, 6, 0.03, 94.583), (6, 4, 0.04, 107.260),
                                                 (5, 5, 0.05, 100.000), (0, 10, 0.02, 82.035)])
def test_bond_price_example1(cpn, n, r, expected):
    # L1V Example 1, p.418-419 (per-period coupon/rate => freq=1)
    assert fi.bond_price(100, cpn / 100, r, n, 1)["price"] == A(expected, abs=5e-4)


def test_zero_coupon_fractional_years_and_errors():
    # zero priced at 80 for 5y yields 4.5130% sa (L1V p.434) -> price back to 80
    assert fi.bond_price(100, 0, 0.045130, 5, 2)["price"] == A(80, abs=1e-3)
    assert fi.bond_price(100, 0, 0.05, 2.5, 1)["price"] == A(100 / 1.05 ** 2.5)
    with pytest.raises(ValueError):
        fi.bond_price(100, 0.05, 0.05, 2.25, 2)


def test_bond_price_spot_curve():
    # L1V §2.4 p.425-426: spots 2/3/4% -> 102.960, YTM 3.935%; Example 4 p.426-427 -> 98.104, 3.516%
    out = fi.bond_price_spot_curve([0.02, 0.03, 0.04], 0.05)
    assert out["price"] == A(102.960, abs=5e-4) and out["ytm"] == A(0.03935, abs=5e-6)
    for spots in ([0.0039, 0.014, 0.025, 0.036], [0.0408, 0.0401, 0.037, 0.035]):
        o = fi.bond_price_spot_curve(spots, 0.03)
        assert o["price"] == A(98.104, abs=5e-4) and o["ytm"] == A(0.03516, abs=5e-6)


def test_bond_price_full_flat():
    # L1V p.428-429: 5% sa, N=18, 88/181, ytm 4.80% -> 101.447790, 102.624323, 1.215470, 101.408853
    o = fi.bond_price_full_flat(0.05, 0.048, 18, 88 / 181)
    assert o["pv_at_last_coupon"] == A(101.447790, abs=1e-6)
    assert o["full_price"] == A(102.624323, abs=1e-6)
    assert o["accrued_interest"] == A(1.215470, abs=1e-6)
    assert o["flat_price"] == A(101.408853, abs=1e-6)


@pytest.mark.parametrize("ytm, full, flat", [(0.058, 103.108770, 101.625437), (0.060, 101.472251, 99.988918),
                                             (0.062, 99.868805, 98.385472)])
def test_bond_price_settlement_30_360(ytm, full, flat):
    # L1V Example 5, p.429-430: 6% German corporate, settle 18 Jun 2015, maturity 19 Sep 2026, 30/360 (89/180)
    o = fi.bond_price_settlement("2015-06-18", "2026-09-19", 0.06, ytm, 2, "30/360")
    assert o["periods"] == 23 and o["fraction"] == A(89 / 180)
    assert o["full_price"] == A(full, abs=1e-6) and o["flat_price"] == A(flat, abs=1e-6)
    assert o["accrued_interest"] == A(1.483333, abs=1e-6)


def test_bond_price_settlement_actual_actual():
    # L1V p.428-429: settle 14 May 2015, maturity 15 Feb 2024 -> 88/181, full 102.624323
    o = fi.bond_price_settlement("2015-05-14", "2024-02-15", 0.05, 0.048)
    assert (o["last_coupon"], o["next_coupon"], o["periods"]) == ("2015-02-15", "2015-08-15", 18)
    assert o["full_price"] == A(102.624323, abs=1e-6)


def test_bond_accrued_interest_day_counts():
    # L1V p.428: 4.375% sa, 15 May -> 27 Jun: act/act 43/184 -> 0.511209; 30/360 42/180 -> 0.510417
    aa = fi.bond_accrued_interest("2014-05-15", "2014-06-27", "2014-11-15", 0.04375, 2, "actual/actual")
    th = fi.bond_accrued_interest("2014-05-15", "2014-06-27", "2014-11-15", 0.04375, 2, "30/360")
    assert (aa["days_accrued"], aa["days_in_period"]) == (43, 184)
    assert aa["accrued_interest"] == A(0.511209, abs=1e-6)
    assert th["days_accrued"] == 42 and th["accrued_interest"] == A(0.510417, abs=1e-6)
    act360 = fi.bond_accrued_interest("2014-05-15", "2014-06-27", "2014-11-15", 0.04, 4, "actual/360")
    assert act360["accrued_interest"] == A(0.04 * 100 * 43 / 360)
    with pytest.raises(ValueError):
        fi.bond_accrued_interest("2014-06-27", "2014-05-15", "2014-11-15", 0.04)


def test_bond_matrix_pricing():
    # L1V §3.2 p.431-432 (Exhibit 4): interpolated 3.9318%, price 100.191
    o = fi.bond_matrix_pricing(0.04, 3, [[2, 0.03, 98.5], [2, 0.05, 102.25], [5, 0.02, 90.25], [5, 0.04, 99.125]], 2)
    assert o["estimated_ytm"] == A(0.039318, abs=2e-6) and o["price"] == A(100.191, abs=1e-3)
    # L1V Example 6 p.433: 104.991 (curriculum rounds intermediate yields to 3.1525%)
    o6 = fi.bond_matrix_pricing(0.045, 4, [[3, 0.055, 107.5], [5, 0.045, 104.75]], 1)
    assert o6["price"] == A(104.991, abs=2e-3)


def test_bond_horizon_return_examples():
    # L1R Examples 1-6 p.489-494 (10y 8% at 85.503075, ytm 10.40%)
    ex2 = fi.bond_horizon_return(85.503075, 0.08, 10, 4, 0.104, 0.104, 1)
    assert ex2["fv_coupons"] == A(37.347111, abs=1e-6) and ex2["sale_price"] == A(89.668770, abs=1e-6)
    assert ex2["horizon_yield"] == A(0.1040, abs=1e-6) and ex2["capital_gain"] == A(0, abs=1e-5)
    ex3 = fi.bond_horizon_return(85.503075, 0.08, 10, 10, 0.114, 0.114, 1)
    assert ex3["fv_coupons"] == A(136.380195, abs=1e-6)
    assert ex3["horizon_yield"] == A(0.1070, abs=5e-5)
    ex4 = fi.bond_horizon_return(85.503075, 0.08, 10, 4, 0.114, 0.114, 1)
    assert ex4["sale_price"] == A(85.780408, abs=1e-6) and ex4["horizon_yield"] == A(0.0967, abs=5e-5)
    assert ex4["capital_gain"] == A(-3.888362, abs=2e-6)
    ex6 = fi.bond_horizon_return(85.503075, 0.08, 10, 4, 0.094, 0.094, 1)
    assert ex6["total_return_amount"] == A(130.595309, abs=1e-6) and ex6["horizon_yield"] == A(0.1117, abs=5e-5)
    # Example 7 p.494-495 and the 7-year (= Macaulay duration) horizon p.527-528
    assert fi.bond_horizon_return(117.729753, 0.10, 4, 2, 0.03, 0.03, 1)["horizon_yield"] == A(0.065647, abs=1e-6)
    assert fi.bond_horizon_return(85.503075, 0.08, 10, 7, 0.094, 0.094, 1)["horizon_yield"] == A(0.10408, abs=5e-6)
    assert fi.bond_horizon_return(85.503075, 0.08, 10, 7, 0.114, 0.114, 1)["horizon_yield"] == A(0.10407, abs=5e-6)


def test_bond_ytm():
    # L1V §2.2 p.420; Example 2 p.420-421; Example 7 p.435-436; Example 8 p.436-437
    assert fi.bond_ytm(105, 0.05, 4, 1)["ytm"] == A(0.03634, abs=5e-6)
    assert fi.bond_ytm(98.175677, 0.025, 3, 2)["ytm"] == A(0.03142, abs=5e-6)
    assert fi.bond_ytm(103.75, 0.035, 4, 1)["ytm"] == A(0.02503, abs=5e-6)
    assert fi.bond_ytm(96.50, 0.0225, 6, 1)["ytm"] == A(0.02894, abs=5e-6)
    assert fi.bond_ytm(22.375, 0, 60, 1)["ytm"] == A(0.02527, abs=5e-6)
    assert fi.bond_ytm(98, 0.045, 5, 2)["ytm"] == A(0.0496, abs=5e-5)
    assert fi.bond_ytm(90, 0.08, 5, 2)["ytm"] == A(0.10630, abs=5e-6)
    assert fi.bond_ytm(105, 0.12, 5, 4)["ytm"] == A(0.10696, abs=5e-6)


def test_bond_ytm_round_trip_between_coupons():
    full = fi.bond_price_full_flat(0.05, 0.048, 18, 88 / 181)["full_price"]
    assert fi.bond_ytm(full, 0.05, 9, 2, fraction=88 / 181)["ytm"] == A(0.048, abs=1e-10)


def test_bond_yield_to_call_put_and_worst():
    # L1V §3.3 p.438: 7y 8% at 105; YTC1 6.975%, YTC2 6.956%, YTC3 6.953%, YTM 7.070%, YTW = 6.953%
    assert fi.bond_yield_to_call_put(105, 0.08, 4, 102, 1)["yield"] == A(0.06975, abs=5e-6)
    o = fi.bond_yield_to_worst(105, 0.08, 7, [[4, 102], [5, 101], [6, 100]], 1)
    assert [r[2] for r in o["yields"]] == A([0.06975, 0.06956, 0.06953, 0.07070], abs=5e-6)
    assert o["yield_to_worst"] == A(0.06953, abs=5e-6) and o["worst_date_years"] == 6


def test_bond_current_and_simple_yield():
    # L1V p.436 (2/95 = 2.105%) and Example 8 p.437 (8/90, 12/105)
    assert fi.bond_current_and_simple_yield(0.02, 95)["current_yield"] == A(0.02105, abs=5e-6)
    assert fi.bond_current_and_simple_yield(0.08, 90)["current_yield"] == A(0.08889, abs=5e-6)
    assert fi.bond_current_and_simple_yield(0.12, 105)["current_yield"] == A(0.11429, abs=5e-6)
    # simple yield (L1V p.436 definition): (2 + 5/10)/95
    assert fi.bond_current_and_simple_yield(0.02, 95, years=10)["simple_yield"] == A(2.5 / 95)


def test_yield_periodicity_convert():
    # L1V p.435: 3.582% sa -> 3.566% q, 3.556% m; Example 7: 4.96% -> 4.93% q, 5.02% annual
    assert fi.yield_periodicity_convert(0.03582, 2, 4)["rate"] == A(0.03566, abs=5e-6)
    assert fi.yield_periodicity_convert(0.03582, 2, 12)["rate"] == A(0.03556, abs=5e-6)
    assert fi.yield_periodicity_convert(0.0496, 2, 1)["rate"] == A(0.0502, abs=5e-5)
    # Example 8 p.437: 10.630% sa -> 10.492% q; 10.696% q -> 10.839% sa
    assert fi.yield_periodicity_convert(0.10630, 2, 4)["rate"] == A(0.10492, abs=5e-6)
    assert fi.yield_periodicity_convert(0.10696, 4, 2)["rate"] == A(0.10839, abs=5e-6)
    # money-market periodicity 365/90 -> 2 (p.446): 10% -> 10.127%; zero priced 80 (p.434): 4.5130% sa = 4.5640% EAR
    assert fi.yield_periodicity_convert(0.10, 365 / 90, 2)["rate"] == A(0.10127, abs=5e-6)
    assert fi.yield_periodicity_convert(0.045130, 2, 1)["rate"] == A(0.045640, abs=5e-6)
    cont = fi.yield_periodicity_convert(0.05, 1, 0)["rate"]
    assert fi.yield_periodicity_convert(cont, 0, 1)["rate"] == A(0.05)


def test_cash_flow_yield_portfolio_and_street_vs_true():
    # L1R §3.4 p.512: portfolio of 1y and 30y zeros, cash flow yield 7.8611%
    assert fi.cash_flow_yield(19.6e6, [10e6, 100e6], [1, 30], 1)["yield"] == A(0.078611, abs=5e-7)
    # street vs true: delaying a payment (weekend) can only lower the yield
    street = fi.cash_flow_yield(98.5, [3, 3, 103], [0.5, 1.0, 1.5], 2)["yield"]
    true = fi.cash_flow_yield(98.5, [3, 3, 103], [0.5 + 2 / 365, 1.0, 1.5 + 1 / 365], 2)["yield"]
    assert true < street


def test_money_market_price_and_rate():
    # L1V p.443: T-bill USD10m 91 days 2.25% discount -> 9,943,125
    assert fi.money_market_price(91, 0.0225, "discount", 360, fv=1e7)["price"] == A(9_943_125)
    # p.443-444: BA CAD10m 180d add-on 4.38%/365 -> 10,216,000; resold 135d at 4.17% -> 10,060,829
    assert fi.money_market_price(180, 0.0438, "add_on", 365, principal=1e7)["redemption"] == A(10_216_000)
    assert fi.money_market_price(135, 0.0417, "add_on", 365, fv=10_216_000)["price"] == A(10_060_829, abs=0.5)
    assert fi.money_market_rate(1e7, 10_060_829, 45, "add_on", 365)["rate"] == A(0.04934, abs=5e-6)
    assert fi.money_market_rate(9_943_125, 1e7, 91, "discount", 360)["rate"] == A(0.0225)


@pytest.mark.parametrize("rate, days, basis, year, bey", [
    (0.0433, 180, "discount", 360, 0.04487), (0.0436, 180, "discount", 365, 0.04456),
    (0.0435, 180, "add_on", 360, 0.04410), (0.0445, 180, "add_on", 365, 0.0445),
    (0.0576, 90, "discount", 360, 0.05925)])
def test_money_market_bey(rate, days, basis, year, bey):
    # L1V Example 10 p.445-446 and the commercial paper example p.444-445
    assert fi.money_market_bey(rate, days, basis, year)["bond_equivalent_yield"] == A(bey, abs=5e-6)


def test_frn_price_and_discount_margin():
    # L1V p.440-441: 2y, Libor 1.25% + 0.50%, DM 0.40% -> 100.196
    assert fi.frn_price(0.0125, 0.005, 0.004, 2, 2)["price"] == A(100.196, abs=5e-4)
    # p.441: 5y 3m Libor 1.10% + 0.75% at 95.50 -> DM 1.718%; Example 9 p.441-442: 98 -> 1.791%
    assert fi.frn_discount_margin(95.5, 0.011, 0.0075, 5, 4)["discount_margin"] == A(0.01718, abs=5e-6)
    assert fi.frn_discount_margin(98, 0.02, 0.0125, 4, 4)["discount_margin"] == A(0.01791, abs=5e-6)
    # round trip
    assert fi.frn_price(0.02, 0.0125, 0.01791231, 4, 4)["price"] == A(98, abs=1e-5)


def test_frn_duration():
    # FRN at par on a reset date: index duration = time to next reset; spread duration = fixed-bond duration
    o = fi.frn_duration(5, 0.02, 0.005, 0.005, 4)
    assert o["full_price"] == A(100) and o["index_macaulay_duration"] == A(0.25)
    fixed = fi.bond_duration_convexity(0.025, 0.025, 5, 4)["modified_duration"]
    assert o["spread_duration"] == A(fixed, rel=1e-6)
    assert fi.frn_duration(5, 0.02, 0.005, 0.005, 4, fraction=0.5)["index_macaulay_duration"] == A(0.125)
