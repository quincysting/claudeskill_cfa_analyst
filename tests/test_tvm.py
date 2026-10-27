"""Tests for cfa_calc.tvm. Golden values from the CFA curriculum (KB = E:/CLAUDE/CFA/MD)."""
import pytest

from cfa_calc import tvm

R5 = "KB Level-1/02-Quantitative-Methods/Curriculum-2014/SS02-R05-the-time-value-of-money.md"
R6 = "KB Level-1/02-Quantitative-Methods/Curriculum-2014/SS02-R06-discounted-cash-flow-applications.md"
approx = pytest.approx


# ---- lump sums: R5 Examples 1, 2, 4, 5, 6, 8, 10
def test_fv_lump_sum_golden():
    assert tvm.fv_lump_sum(5_000_000, 0.07, 5)["fv"] == approx(7_012_758.65, abs=0.01)  # R5 p272 Ex 1
    assert tvm.fv_lump_sum(2_500_000, 0.08, 6)["fv"] == approx(3_967_186, abs=1)  # R5 p273 Ex 2
    assert tvm.fv_lump_sum(10_000, 0.08, 2, freq=4)["fv"] == approx(11_716.59, abs=0.01)  # R5 p276 Ex 4
    assert tvm.fv_lump_sum(1_000_000, 0.06, 1, freq=12)["fv"] == approx(1_061_677.81, abs=0.01)  # R5 p276 Ex 5
    assert tvm.fv_lump_sum(10_000, 0.08, 2, continuous=True)["fv"] == approx(11_735.11, abs=0.01)  # R5 p277 Ex 6


def test_pv_lump_sum_golden_and_round_trip():
    assert tvm.pv_lump_sum(100_000, 0.08, 6)["pv"] == approx(63_016.96, abs=0.01)  # R5 p282 Ex 8
    assert tvm.pv_lump_sum(5_000_000, 0.06, 10, freq=12)["pv"] == approx(2_748_163.67, abs=0.01)  # R5 p285 Ex 10
    pv = tvm.pv_lump_sum(1234.5, 0.07, 3.5, freq=4)["pv"]
    assert tvm.fv_lump_sum(pv, 0.07, 3.5, freq=4)["fv"] == approx(1234.5)


# ---- annuities: R5 Examples 7, 11, 12 and the annuity-due illustration
def test_annuity_golden():
    assert tvm.annuity_fv(20_000, 0.09, 30)["fv"] == approx(2_726_150.77, abs=0.01)  # R5 p281 Ex 7
    assert tvm.annuity_pv(1000, 0.12, 5)["pv"] == approx(3_604.78, abs=0.01)  # R5 p286 Ex 11
    assert tvm.annuity_pv(200_000, 0.07, 20, due=True)["pv"] == approx(2_267_119.05, abs=0.01)  # R5 p288 Ex 12
    assert tvm.annuity_pv(100, 0.12, 4, due=True)["pv"] == approx(340.18, abs=0.01)  # R5 p287
    assert tvm.annuity_fv(1000, 0.05, 5)["fv"] == approx(5_525.63, abs=0.01)  # R5 p280 (FV annuity factor 5.525631)


def test_projected_annuity_example_13():
    # R5 p289 Ex 13: 30 payments of 1m from t=10, 5% -> 15,372,451.03 at t=9, 9,909,219.00 today
    at9 = tvm.annuity_pv(1_000_000, 0.05, 30)["pv"]
    assert at9 == approx(15_372_451.03, abs=0.01)
    assert tvm.pv_lump_sum(at9, 0.05, 9)["pv"] == approx(9_909_219.00, abs=0.01)


def test_annuity_due_equals_ordinary_times_one_plus_r():
    o, d = tvm.annuity_pv(50, 0.06, 10)["pv"], tvm.annuity_pv(50, 0.06, 10, due=True)["pv"]
    assert d == approx(o * 1.06)
    assert tvm.annuity_pv(10, 0.0, 5)["pv"] == approx(50)


# ---- tvm_solve: payment, periods, rate
def test_tvm_solve_mortgage_payment():
    out = tvm.tvm_solve("pmt", periods=360, rate=0.08, pv=100_000, freq=12)  # R5 p297 Ex 20
    assert out["pmt"] == approx(-733.76, abs=0.005)


def test_tvm_solve_periods_and_growth_rates():
    assert tvm.tvm_solve("periods", rate=0.07, pv=-10e6, fv=20e6)["periods"] == approx(10.24, abs=0.005)  # R5 p296 Ex 19
    assert tvm.tvm_solve("rate", periods=4, pv=-8436, fv=8445)["rate"] == approx(0.0003, abs=0.00005)  # R5 p294 Ex 17
    assert tvm.tvm_solve("rate", periods=4, pv=-5382, fv=4036)["rate"] == approx(-0.069, abs=0.0005)  # R5 p294 Ex 18


def test_tvm_solve_retirement_example_21():
    # R5 p298-299 Ex 21 (Jill Grant), 8%: saved 54,304.23 at t=15; needs 143,362.53; saves 8,342.87/yr t=16..40
    saved = tvm.annuity_fv(2000, 0.08, 15)["fv"]
    assert saved == approx(54_304.23, abs=0.01)
    need40 = tvm.annuity_pv(100_000, 0.08, 20)["pv"]
    assert need40 == approx(981_814.74, abs=0.01)
    need15 = tvm.pv_lump_sum(need40, 0.08, 25)["pv"]
    assert need15 == approx(143_362.53, abs=0.01)
    pmt = tvm.tvm_solve("pmt", periods=25, rate=0.08, pv=-(need15 - saved))["pmt"]
    assert pmt == approx(8_342.87, abs=0.01)


def test_tvm_solve_round_trips():
    fv = tvm.tvm_solve("fv", periods=24, rate=0.06, pv=-5000, pmt=-100, freq=12)["fv"]
    assert tvm.tvm_solve("rate", periods=24, pv=-5000, pmt=-100, fv=fv, freq=12)["rate"] == approx(0.06)
    assert tvm.tvm_solve("periods", rate=0.06, pv=-5000, pmt=-100, fv=fv, freq=12)["periods"] == approx(24)
    assert tvm.tvm_solve("pv", periods=24, rate=0.06, pmt=-100, fv=fv, freq=12)["pv"] == approx(-5000)
    due = tvm.tvm_solve("pmt", periods=10, rate=0.05, pv=1000, due=True)["pmt"]
    assert -due == approx(1000 / tvm.annuity_pv(1, 0.05, 10, due=True)["pv"])


def test_tvm_solve_rejects_bad_input():
    with pytest.raises(ValueError):
        tvm.tvm_solve("xyz", periods=1, rate=0.1)
    with pytest.raises(ValueError):
        tvm.tvm_solve("rate", periods=5, pv=100, fv=100)  # same signs: no rate


# ---- perpetuities, growing annuities, uneven flows
def test_perpetuity_golden():
    assert tvm.perpetuity_pv(100, 0.05)["pv"] == approx(2000)  # R5 p290 Ex 14
    assert tvm.perpetuity_pv(100, 0.05, first_payment_period=5)["pv"] == approx(1_645.40, abs=0.01)  # R5 p291 Ex 15
    with pytest.raises(ValueError):
        tvm.perpetuity_pv(100, 0.05, growth=0.06)


def test_annuity_as_difference_of_perpetuities():
    # R5 p292 Ex 16: 4-year annuity = perpetuity from t=1 minus perpetuity from t=5
    diff = tvm.perpetuity_pv(100, 0.05)["pv"] - tvm.perpetuity_pv(100, 0.05, first_payment_period=5)["pv"]
    assert diff == approx(tvm.annuity_pv(100, 0.05, 4)["pv"])


def test_growing_annuity_consistency():
    g = tvm.growing_annuity_value(100, 0.10, 0.05, 3)
    direct = 100 / 1.1 + 105 / 1.1 ** 2 + 110.25 / 1.1 ** 3
    assert g["pv"] == approx(direct)
    assert tvm.growing_annuity_value(100, 0.08, 0.0, 7)["pv"] == approx(tvm.annuity_pv(100, 0.08, 7)["pv"])
    assert tvm.growing_annuity_value(100, 0.05, 0.05, 4)["pv"] == approx(4 * 100 / 1.05)


def test_uneven_cash_flows_golden():
    out = tvm.uneven_cash_flows_value(0.05, [0, 1000, 2000, 4000, 5000, 6000])
    assert out["pv"] == approx(15_036.46, abs=0.01)  # R5 p292 Table 3
    assert out["fv"] == approx(19_190.76, abs=0.01)  # R5 p281 Table 2


# ---- NPV / IRR / MIRR (R6)
def test_npv_and_irr_golden():
    assert tvm.npv(0.10, [-1_000_000] + [150_000] * 400)["npv"] == approx(500_000, rel=1e-3)  # R6 p326 Ex 1 (perpetuity)
    cf = [-1000] + [294.8] * 5
    assert tvm.npv(0.11, cf)["npv"] == approx(89.55, abs=0.01)  # R6 p328 Ex 3
    assert tvm.irr(cf)["irr"] == approx(0.145012, abs=1e-6)  # R6 p329 Ex 3
    assert tvm.irr([-10_000, 0, 0, 21_220])["irr"] == approx(0.285, abs=0.0005)  # R6 p331 Project D
    assert tvm.npv(0.08, [-10_000, 0, 0, 21_220])["npv"] == approx(6_845.12, abs=0.01)  # R6 p330 Table 3
    assert tvm.irr([-5_500_000, 1_000_000, 0, 0, 1_500_000, 7_000_000])["irr"] == approx(0.1351, abs=5e-5)  # R6 p345 Prob 2


def test_irr_multiple_and_none():
    multi = tvm.irr([-100, 230, -132])
    assert multi["irr"] is None and multi["irrs"] == [approx(0.10), approx(0.20)]
    none = tvm.irr([1, -3, 3])
    assert none["irr"] is None and none["irrs"] == [] and "no IRR" in none["note"]
    assert tvm.irr([100, 50])["sign_changes"] == 0


def test_irr_npv_zero_consistency_long_series():
    cf = [-100_000] + [733.76] * 360
    r = tvm.irr(cf)["irr"]
    assert tvm.npv(r, cf)["npv"] == approx(0, abs=1e-6)
    assert r * 12 == approx(0.08, abs=1e-5)


def test_mirr():
    # Microsoft Excel MIRR documentation example (displays 13%); hand formula gives 0.126094
    assert tvm.mirr([-120000, 39000, 30000, 21000, 37000, 46000], 0.10, 0.12)["mirr"] == approx(0.126094, abs=1e-6)
    cf = [-1000] + [294.8] * 5
    r = tvm.irr(cf)["irr"]
    assert tvm.mirr(cf, r, r)["mirr"] == approx(r)  # reinvest at IRR -> MIRR = IRR (conventional flows)


# ---- amortization and rate conversions
def test_loan_amortization():
    out = tvm.loan_amortization_schedule(100_000, 0.08, 30)
    assert out["payment"] == approx(733.76, abs=0.005)  # R5 p297 Ex 20
    assert out["schedule"][-1]["ending_balance"] == approx(0, abs=1e-6)
    assert out["total_interest"] == approx(out["payment"] * 360 - 100_000)
    t6 = tvm.loan_amortization_schedule(4329.48, 0.05, 5, freq=1)  # R5 p300-301 Table 6
    assert t6["payment"] == approx(1000, abs=0.01)
    assert t6["schedule"][0]["interest"] == approx(216.47, abs=0.01)


def test_effective_and_stated_rates():
    assert tvm.effective_annual_rate(0.08, 2)["effective_annual_rate"] == approx(0.0816)  # R5 p278
    assert tvm.effective_annual_rate(0.08, continuous=True)["effective_annual_rate"] == approx(0.0833, abs=5e-5)
    assert tvm.stated_annual_rate(0.0816, 2)["stated_rate"] == approx(0.08)  # R5 p279
    assert tvm.stated_annual_rate(0.083287, continuous=True)["stated_rate"] == approx(0.08, abs=1e-6)
    ear = tvm.effective_annual_rate(0.05, 365)["effective_annual_rate"]
    assert tvm.stated_annual_rate(ear, 365)["stated_rate"] == approx(0.05)


# ---- returns (R6)
def test_holding_period_and_mwr_twr():
    assert tvm.holding_period_return(200, 225, 5)["holding_period_return"] == approx(0.15)  # R6 p333
    assert tvm.money_weighted_return([-200, -220, 480])["money_weighted_return"] == approx(0.0939, abs=5e-5)  # R6 p333
    assert tvm.time_weighted_return([0.15, (470 + 10) / 450 - 1], years=2)["annualized"] == approx(0.1076, abs=1e-4)  # p334 (10.76%)
    assert tvm.time_weighted_return([0.20, 0.05, 0.12, -0.10])["time_weighted_return"] == approx(0.27, abs=0.001)  # p335
    assert tvm.time_weighted_return([0.10, 0.02, 0.08, 0.04])["time_weighted_return"] == approx(0.26, abs=0.001)  # p335
    vals = tvm.time_weighted_return(begin_values=[5e6, 5.5e6, 6e6, 6.12e6], end_values=[6e6, 5.775e6, 6.72e6, 5.508e6])
    assert vals["subperiod_returns"] == [approx(0.20), approx(0.05), approx(0.12), approx(-0.10)]  # R6 p335 Table 5


def test_walbright_example_5():
    # R6 p336-338 Ex 5: TWR 21.03%; MWR four-month IRR 6.28% -> 20% annual
    assert tvm.time_weighted_return([0.12, 0.0806])["time_weighted_return"] == approx(0.2103, abs=5e-5)
    mwr = tvm.money_weighted_return([-100, -20, 0, 142.64], freq=3)
    assert mwr["money_weighted_return"] == approx(0.0628, abs=5e-5)
    assert mwr["annualized"] == approx(0.20, abs=0.001)
    assert tvm.annualize_return(0.0628, periods_per_year=3)["annualized"] == approx(0.2004, abs=1e-4)  # p338


def test_mwr_undefined_raises():
    with pytest.raises(ValueError):
        tvm.money_weighted_return([-100, 230, -132])


# ---- money market yields (R6 Examples 6-7)
def test_money_market_yields_golden():
    y = tvm.money_market_yields(days=150, face=100_000, price=98_000)
    assert y["bank_discount_yield"] == approx(0.048)  # R6 p339 Ex 6
    assert y["holding_period_yield"] == approx(0.020408, abs=1e-6)  # R6 p342 Table 8
    assert y["effective_annual_yield"] == approx(0.050388, abs=1e-6)
    assert y["money_market_yield"] == approx(0.04898, abs=1e-5)  # R6 p341


def test_money_market_yield_round_trips():
    ref = tvm.money_market_yields(days=150, face=100_000, price=98_000)
    for k in ("bank_discount_yield", "holding_period_yield", "effective_annual_yield", "money_market_yield"):
        assert tvm.money_market_yields(days=150, face=100_000, **{k: ref[k]})["price"] == approx(98_000)
    # r_MM = 360 r_BD / (360 - t r_BD), R6 p341 Eq. 6
    assert ref["money_market_yield"] == approx(360 * 0.048 / (360 - 150 * 0.048))
    with pytest.raises(ValueError):
        tvm.money_market_yields(days=150, price=98, bank_discount_yield=0.05)
