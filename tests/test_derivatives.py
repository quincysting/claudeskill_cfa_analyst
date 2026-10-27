"""Golden tests for the derivatives modules (L1-L3).

Expected values come from the CFA curriculum worked examples in the knowledge base (path relative to
E:/CLAUDE/CFA/MD, page = the `<!-- page N -->` marker) or from standard textbook examples (Hull).
Where the curriculum rounds intermediate values or reads N(d) from a two-decimal table, the tolerance says so.
"""
import math
import subprocess
import sys
from pathlib import Path

import pytest
from pytest import approx

from cfa_calc import REGISTRY
from cfa_calc import derivatives as dv
from cfa_calc import derivatives_options as do
from cfa_calc import derivatives_risk as dr

R51 = "Level-2/09-Derivatives/Curriculum-2014/SS16-R51-forward-markets-and-contracts.md"
R52 = "Level-2/09-Derivatives/Curriculum-2014/SS16-R52-futures-markets-and-contracts.md"
R53 = "Level-2/09-Derivatives/Curriculum-2014/SS17-R53-option-markets-and-contracts.md"
R54 = "Level-2/09-Derivatives/Curriculum-2014/SS17-R54-swap-markets-and-contracts.md"
R55 = "Level-2/09-Derivatives/Curriculum-2014/SS17-R55-interest-rate-derivative-instruments.md"
R29 = "Level-3/11-Risk-Management-Applications-of-Derivatives/Curriculum-2014/SS15-R29-...forward-and-futures-strategies.md"
R30 = "Level-3/11-Risk-Management-Applications-of-Derivatives/Curriculum-2014/SS15-R30-...option-strategies.md"
R31 = "Level-3/11-Risk-Management-Applications-of-Derivatives/Curriculum-2014/SS15-R31-...swap-strategies.md"


# ============================================================ forwards
def test_forward_price_no_income():
    # R51 p.37: F = 72.50 (1.0825)^5 = 107.76
    assert dv.forward_price(spot=72.5, rate=0.0825, years=5)["forward_price"] == approx(107.76, abs=0.005)
    # R51 Example 1A p.40: F = 125.72 (1.05625)^0.75 = 130.99
    assert dv.forward_price(spot=125.72, rate=0.05625, years=0.75)["forward_price"] == approx(130.99, abs=0.005)


def test_forward_value_during_life_and_expiration():
    # R51 p.38: V = 102 - 105/1.05^0.75 = 0.7728; p.39: 71.19 - 62.25/1.07^0.5 = 11.01
    assert dv.forward_value(spot=102, forward_price_initial=105, rate=0.05, years_remaining=0.75)["value"] == \
        approx(0.7728, abs=1e-4)
    assert dv.forward_value(spot=71.19, forward_price_initial=62.25, rate=0.07, years_remaining=0.5)["value"] == \
        approx(11.01, abs=0.005)
    # R51 Example 1C/1D p.41: short position worth +8.0 two months in, +7.49 at expiration
    r = dv.forward_value(spot=118.875, forward_price_initial=130.99, rate=0.05625, years_remaining=7 / 12,
                         position="short")
    assert r["value"] == approx(8.0, abs=0.05)
    assert r["value_long_per_unit"] == approx(-8.0, abs=0.05)
    assert dv.forward_value(spot=123.50, forward_price_initial=130.99, rate=0.05625, years_remaining=0,
                            position="short")["value"] == approx(7.49, abs=1e-9)


def test_forward_value_zero_at_inception():
    f = dv.forward_price(spot=62.5, rate=0.042, years=200 / 365, income_cash_flows=[[0.75, 50 / 365], [0.75, 140 / 365]])
    v = dv.forward_value(spot=62.5, forward_price_initial=f["forward_price"], rate=0.042, years_remaining=200 / 365,
                         income_cash_flows=[[0.75, 50 / 365], [0.75, 140 / 365]])
    assert v["value"] == approx(0, abs=1e-12)


def test_equity_forward_discrete_dividends():
    # R51 Example 2 p.45-46: PV(D) 1.48, F 62.41, V(75 days) -6.53, V_T -3.91
    divs = [[0.75, 50 / 365], [0.75, 140 / 365]]
    f = dv.forward_price(spot=62.50, rate=0.042, years=200 / 365, income_cash_flows=divs)
    assert f["pv_income_total"] == approx(1.48, abs=0.005)
    assert f["forward_price"] == approx(62.41, abs=0.005)
    v = dv.forward_value(spot=55.75, forward_price_initial=62.41, rate=0.042, years_remaining=125 / 365,
                         income_cash_flows=[[0.75, 65 / 365]])
    assert v["value"] == approx(-6.53, abs=0.005)
    assert dv.forward_value(spot=58.50, forward_price_initial=62.41, rate=0.042, years_remaining=0)["value"] == \
        approx(-3.91, abs=1e-9)


def test_equity_forward_fv_of_dividends_and_continuous_yield():
    # R51 p.43: PV method and FV method give 34.32 (S=35, r=4%, four 0.45 dividends, T=300/365)
    divs = [[0.45, t / 365] for t in (10, 102, 193, 283)]
    pv = dv.forward_price(spot=35, rate=0.04, years=300 / 365, income_cash_flows=divs)["forward_price"]
    fv_divs = sum(0.45 * 1.04 ** ((300 - t) / 365) for t in (10, 102, 193, 283))
    fv = dv.forward_price(spot=35, rate=0.04, years=300 / 365, fv_income=fv_divs)["forward_price"]
    assert pv == approx(34.32, abs=0.005) and fv == approx(pv, rel=1e-12)
    # R51 p.44: CAC 40 at 5475, delta_c 1.5%, r_c 4.625%, T=2 -> 5828.11
    assert dv.forward_price(spot=5475, rate=0.04625, years=2, income_yield=0.015,
                            compounding="continuous")["forward_price"] == approx(5828.11, abs=0.005)


def test_bond_forward_with_coupons():
    # R51 p.47: full price 984.45, coupons 20 at 37 and 219 days, r 5.75%, T 310 days -> PV 39.23, F 991.18
    f = dv.forward_price(spot=984.45, rate=0.0575, years=310 / 365, income_cash_flows=[[20, 37 / 365], [20, 219 / 365]])
    assert f["pv_income_total"] == approx(39.23, abs=0.005)
    assert f["forward_price"] == approx(991.18, abs=0.01)
    # 15 days later at 6.75%: V = -6.28
    v = dv.forward_value(spot=973.14, forward_price_initial=991.18, rate=0.0675, years_remaining=295 / 365,
                         income_cash_flows=[[20, 22 / 365], [20, 204 / 365]])
    assert v["value"] == approx(-6.28, abs=0.01)  # curriculum discounts 991.19 instead of 991.18
    # R51 Example 3 p.48-49: F = 929.76; counterparty (long) value 34.36 a year later
    cpns = [[50, t / 365] for t in (31, 215, 397, 580)]
    assert dv.forward_price(spot=1010.25, rate=0.08, years=581 / 365, income_cash_flows=cpns)["forward_price"] == \
        approx(929.76, abs=0.01)
    v = dv.forward_value(spot=1025.375, forward_price_initial=929.76, rate=0.07, years_remaining=216 / 365,
                         income_cash_flows=[[50, 32 / 365], [50, 215 / 365]])
    assert v["value"] == approx(34.36, abs=0.01)


def test_bond_futures_conversion_factor_and_accrued_interest():
    # R52 p.125: (0.9594 x 1.065^1.25 - 0.0722)/1.0505 = 0.9193
    r = dv.bond_forward_price(bond_price=0.9594, rate=0.065, years=1.25, fv_coupons=0.0722, conversion_factor=1.0505)
    assert r["quoted_futures_price"] == approx(0.9193, abs=5e-5)
    # R52 Example 4 p.125: FV coupons 0.0815; single deliverable 1.0026; with CF 1.0372 -> 0.9667
    assert dv.bond_forward_price(bond_price=1.0132, rate=0.07, years=1, fv_coupons=0.0815)["quoted_futures_price"] \
        == approx(1.0026, abs=5e-5)
    assert dv.bond_forward_price(bond_price=1.0132, rate=0.07, years=1, fv_coupons=0.0815,
                                 conversion_factor=1.0372)["quoted_futures_price"] == approx(0.9667, abs=5e-5)
    # Accrued interest: clean + AI0 equals the full price; AI_T is stripped from the quoted price
    a = dv.bond_forward_price(bond_price=98, accrued_interest_0=2, accrued_interest_T=1, rate=0.05, years=0.5)
    assert a["full_forward_price"] == approx(100 * 1.05 ** 0.5)
    assert a["clean_forward_price"] == approx(100 * 1.05 ** 0.5 - 1)


def test_bond_futures_invoice_and_implied_repo():
    # R55 p.331: 100,000 x 1.055 x 1.2000 = 126,600 + AI; Practice Q1: 97-24, CF 1.17, AI 3,800
    assert dv.bond_futures_invoice_price(futures_price=105.5, conversion_factor=1.2)["converted_price"] == \
        approx(126_600)
    assert dv.bond_futures_invoice_price(futures_price=97.75, conversion_factor=1.17,
                                         accrued_interest=3800)["invoice_price"] == approx(118_167.5)
    # R55 p.333-334: implied repo rate 7.71%
    r = dv.bond_futures_implied_repo_rate(bond_price=107, accrued_interest_paid=3.8904, futures_price=96,
                                          conversion_factor=1.1111, accrued_interest_at_delivery=1.1507,
                                          days_to_delivery=82, interim_coupon=5, days_coupon_reinvested=42,
                                          reinvestment_rate=0.038)
    assert r["proceeds"] == approx(112.8385, abs=1e-4)
    assert r["implied_repo_rate"] == approx(0.0771, abs=5e-5)


def test_tbill_futures():
    # R52 Example 3 p.119: f = 0.9821/0.9931 = 0.9889 (curriculum rounds the bill prices to 4 dp)
    r = dv.tbill_futures_price(discount_rate_short=0.05, days_short=50, discount_rate_long=0.046, days_long=140)
    assert r["futures_price"] == approx(0.9889, abs=1e-4)
    assert r["implied_discount_rate"] == approx((1 - r["futures_price"]) * 360 / 90)


def test_futures_pricing_cost_of_carry():
    # R52 Example 2 p.110-111: S 50, r 8%, 45 days -> 50.48; +2.25 storage 52.73; -0.75 income 49.73; +3.55 54.03
    kw = dict(spot=50, rate=0.08, years=45 / 365)
    assert dv.forward_price(**kw)["forward_price"] == approx(50.48, abs=0.005)
    assert dv.forward_price(fv_costs=2.25, **kw)["forward_price"] == approx(52.73, abs=0.005)
    assert dv.forward_price(fv_income=0.75, **kw)["forward_price"] == approx(49.73, abs=0.005)
    a = dv.forward_arbitrage(market_forward_price=60, fv_costs=3.55, **kw)
    assert a["theoretical_price"] == approx(54.03, abs=0.005)
    assert a["profit_at_expiration"] == approx(5.97, abs=0.005)
    assert a["strategy"].startswith("cash-and-carry")


def test_stock_index_futures():
    # R52 Example 5 p.130-131: 755.42, r 6.25%, 57 days, FV dividends 3.94 -> 758.67 (PV 3.90)
    T = 57 / 365
    a = dv.forward_price(spot=755.42, rate=0.0625, years=T, fv_income=3.94)
    b = dv.forward_price(spot=755.42, rate=0.0625, years=T, pv_income=3.94 / 1.0625 ** T)
    assert a["forward_price"] == approx(758.67, abs=0.005) and b["forward_price"] == approx(a["forward_price"])
    # Part C: continuous r_c = ln(1.0625), delta_c = ln(1.0052)/T -> 758.66
    dc = math.log(1 / (1 - 3.94 / (755.42 * 1.0625 ** T))) / T
    c = dv.forward_price(spot=755.42, rate=math.log(1.0625), years=T, income_yield=dc, compounding="continuous")
    assert c["forward_price"] == approx(758.66, abs=0.01)


def test_currency_forward_and_value():
    # R51 p.55: S 0.5987, r 5.5%, rf 4.75%, 180 days -> 0.6008; 40 days later at 0.65 -> 0.0499
    f = dv.currency_forward_price(spot=0.5987, domestic_rate=0.055, foreign_rate=0.0475, years=180 / 365)
    assert f["forward_price"] == approx(0.6008, abs=5e-5)
    v = dv.currency_forward_value(spot=0.65, forward_price_initial=0.6008, domestic_rate=0.055, foreign_rate=0.0475,
                                  years_remaining=140 / 365, notional_foreign=10e6)
    assert v["value_long_per_unit"] == approx(0.0499, abs=5e-5)
    assert v["value"] == approx(499_000, abs=600)  # p.58 credit exposure 10m x 0.0499
    # Continuous rates r_c = ln(1+r) give the same price (R51 p.55)
    fc = dv.currency_forward_price(spot=0.5987, domestic_rate=math.log(1.055), foreign_rate=math.log(1.0475),
                                   years=180 / 365, compounding="continuous")
    assert fc["forward_price"] == approx(f["forward_price"], rel=1e-12)
    # R51 Example 5 p.56-57: F = 1.7418 vs 1.75 quote -> 5.59% riskless; long value after a month -0.0443
    assert dv.currency_forward_price(spot=1.76, domestic_rate=0.051, foreign_rate=0.062, years=1)["forward_price"] == \
        approx(1.7418, abs=5e-5)
    arb = dv.forward_arbitrage(spot=1.76, market_forward_price=1.75, rate=0.051, years=1, income_yield=0.062)
    assert arb["implied_return"] == approx(0.0559, abs=1e-4)  # curriculum uses the rounded outlay 1.6573
    assert dv.currency_forward_value(spot=1.72, forward_price_initial=1.75, domestic_rate=0.051, foreign_rate=0.062,
                                     years_remaining=11 / 12, position="short")["value"] == approx(0.0443, abs=5e-5)
    # Mark to market and reprice: new rate 0.6518 (R51 p.59)
    assert dv.currency_forward_price(spot=0.65, domestic_rate=0.055, foreign_rate=0.0475,
                                     years=140 / 365)["forward_price"] == approx(0.6518, abs=5e-5)


def test_currency_futures_and_libor_style_parity():
    # R52 Example 6 p.133-134: 0.6012; arbitrage at 0.62 returns 1.0441 per dollar over 78 days
    T = 78 / 365
    assert dv.currency_forward_price(spot=0.60, domestic_rate=0.06, foreign_rate=0.05, years=T)["forward_price"] == \
        approx(0.6012, abs=5e-5)
    arb = dv.forward_arbitrage(spot=0.60, market_forward_price=0.62, rate=0.06, years=T, income_yield=0.05)
    assert 1 + arb["implied_return"] == approx(1.0441, abs=1e-4)
    # R29 Example 9 p.365: LIBOR-style rates, 6 months -> forward 0.7537
    f = dv.currency_forward_price(spot=0.75, domestic_rate=0.06, foreign_rate=0.05, years=180 / 360,
                                  compounding="simple")
    assert f["forward_price"] == approx(0.7537, abs=5e-5)


def test_forward_arbitrage_annualized_return():
    # R51 Example 1B p.41: sell forward at 140 vs 130.99 -> 11.36% for nine months, 15.43% annualized
    a = dv.forward_arbitrage(spot=125.72, market_forward_price=140, rate=0.05625, years=0.75)
    assert a["implied_return"] == approx(0.1136, abs=5e-5)
    assert a["implied_return_annual"] == approx(0.1543, abs=1e-4)  # curriculum annualizes the rounded 11.36%
    assert a["basis"] == approx(125.72 - 140)
    rev = dv.forward_arbitrage(spot=125.72, market_forward_price=125, rate=0.05625, years=0.75)
    assert rev["strategy"].startswith("reverse") and rev["profit_at_expiration"] == approx(130.99 - 125, abs=0.005)


def test_futures_margin_account():
    # R52 Exhibit 1 p.84-85 (long and short 10 contracts, IM 5, MM 3)
    prices = [100, 99.20, 96, 101, 103.50, 103, 104]
    lng = dv.futures_margin_account(settlement_prices=prices, contracts=10, initial_margin=5, maintenance_margin=3)
    assert [r["deposit"] for r in lng["rows"]] == approx([50, 0, 0, 40, 0, 0, 0])
    assert [r["ending_balance"] for r in lng["rows"]] == approx([50, 42, 10, 100, 125, 120, 130])
    assert lng["margin_call_trigger_price"] == approx(98)
    sht = dv.futures_margin_account(settlement_prices=prices, contracts=10, initial_margin=5, maintenance_margin=3,
                                    position="short")
    assert [r["ending_balance"] for r in sht["rows"]] == approx([50, 58, 90, 40, 15, 55, 45])
    assert sht["total_deposits"] == approx(85) and sht["margin_call_trigger_price"] == approx(102)
    # R52 Example 1 p.86-87: 20 contracts, IM 5, MM 2
    ex = dv.futures_margin_account(settlement_prices=[82, 84, 78, 73, 79, 82, 84], contracts=20, initial_margin=5,
                                   maintenance_margin=2)
    assert [r["deposit"] for r in ex["rows"]] == approx([100, 0, 0, 80, 100, 0, 0])
    assert [r["ending_balance"] for r in ex["rows"]] == approx([100, 140, 20, 0, 220, 280, 320])
    assert ex["margin_call_trigger_price"] == approx(79)


# ============================================================ FRAs
def test_fra_rate_and_value():
    # R51 p.50-51: 3x9 FRA, L(90) 5.6%, L(270) 6% -> 6.11%; 25 days later value 0.0026 per $1
    r = dv.fra_rate(rate_short=0.056, days_short=90, rate_long=0.06, days_long=270)
    assert r["fra_rate"] == approx(0.0611, abs=5e-5) and r["notation"] == "3x9"
    v = dv.fra_value(fra_rate=0.0611, rate_to_expiry=0.059, days_to_expiry=65, rate_to_maturity=0.065,
                     days_to_maturity=245)
    assert v["value"] == approx(0.0026, abs=5e-5)


def test_fra_example_4():
    # R51 Example 4 p.51-52: 1x7 FRA 6.19%; value -0.0011 per EUR (EUR -22,000 on 20m); settlement -0.0023 (-46,000)
    r = dv.fra_rate(rate_short=0.0575, days_short=30, rate_long=0.0615, days_long=210)
    assert r["fra_rate"] == approx(0.0619, abs=5e-5) and r["notation"] == "1x7"
    v = dv.fra_value(fra_rate=0.0619, rate_to_expiry=0.0545, days_to_expiry=10, rate_to_maturity=0.0595,
                     days_to_maturity=190, notional=20e6)
    assert v["value_per_unit_long"] == approx(-0.0011, abs=5e-5)
    s = dv.fra_settlement(notional=20e6, fra_rate=0.0619, underlying_rate=0.0572, days_underlying=180)
    assert s["settlement"] / 20e6 == approx(-0.0023, abs=5e-5)  # curriculum rounds to 4 dp before scaling


def test_fra_value_zero_at_inception():
    r = dv.fra_rate(rate_short=0.056, days_short=90, rate_long=0.06, days_long=270)["fra_rate"]
    v = dv.fra_value(fra_rate=r, rate_to_expiry=0.056, days_to_expiry=90, rate_to_maturity=0.06, days_to_maturity=270)
    assert v["value"] == approx(0, abs=1e-14)


# ============================================================ swaps
def test_swap_fixed_rate_and_fra_consistency():
    # R54 p.275: quarterly swap, LIBOR 3.45/3.58/3.70/3.75% -> 0.0092 per quarter, 3.68% (curriculum rounds B to 4 dp)
    r = dv.swap_fixed_rate(spot_rates=[0.0345, 0.0358, 0.0370, 0.0375], days=[90, 180, 270, 360])
    assert r["discount_factors"] == approx([0.9914, 0.9824, 0.9730, 0.9639], abs=5e-5)
    assert r["fixed_rate_periodic"] == approx(0.0092, abs=5e-5)
    assert r["fixed_rate_annual"] == approx(0.0368, abs=3e-4)
    # FRA view: the floating leg PV from implied forwards equals 1 - B_n (swap = portfolio of FRAs)
    assert r["pv_floating_via_forwards"] == approx(1 - r["discount_factors"][-1], rel=1e-12)
    # par-bond view: a bond paying the swap rate is worth par
    assert r["fixed_rate_periodic"] * r["sum_discount_factors"] + r["discount_factors"][-1] == approx(1.0)
    # R54 Example 4A p.277: 0.0392, 7.84%; Example 5A p.282: EUR 0.0324, 6.48%; SF 0.0139, 5.56% (p.279-280)
    e4 = dv.swap_fixed_rate(spot_rates=[0.072, 0.08], days=[180, 360])
    assert e4["fixed_rate_periodic"] == approx(0.0392, abs=5e-5) and e4["fixed_rate_annual"] == approx(0.0784, abs=1e-4)
    e5 = dv.swap_fixed_rate(spot_rates=[0.06, 0.066], days=[180, 360])
    assert e5["fixed_rate_annual"] == approx(0.0648, abs=1e-4)
    sf = dv.swap_fixed_rate(spot_rates=[0.052, 0.054, 0.0555, 0.057], days=[90, 180, 270, 360])
    assert sf["fixed_rate_periodic"] == approx(0.0139, abs=5e-5)


def test_interest_rate_swap_value():
    # R54 p.275-276: 60 days in, pay-fixed value 0.0047 per $1 (PV fixed 1.0004, PV floating 1.0051)
    r = dv.interest_rate_swap_value(fixed_rate=0.0368, floating_rate_set=0.0345, period_fraction=0.25,
                                    spot_rates=[0.0425, 0.0432, 0.0437, 0.0444], days=[30, 120, 210, 300])
    assert r["pv_fixed_leg"] == approx(1.0004, abs=5e-5)
    assert r["pv_floating_leg"] == approx(1.0051, abs=5e-5)
    assert r["value"] == approx(0.0047, abs=5e-5)
    # R54 Example 4B p.277: receive-fixed value 0.0051 per $1, $76,500 on $15m
    e = dv.interest_rate_swap_value(fixed_rate=0.0784, floating_rate_set=0.072, period_fraction=0.5,
                                    spot_rates=[0.071, 0.074], days=[90, 270], notional=15e6, position="receive_fixed")
    # curriculum uses the rounded 0.0392 payment and 4-dp discount factors
    assert e["pv_fixed_leg"] == approx(1.0231, abs=1e-4) and e["pv_floating_leg"] == approx(1.0180, abs=1e-4)
    assert e["value"] == approx(76_500, abs=1_500)  # curriculum rounds to 0.0051 per $1 before scaling


def test_swap_value_zero_at_inception():
    rates, days = [0.0345, 0.0358, 0.0370, 0.0375], [90, 180, 270, 360]
    fs = dv.swap_fixed_rate(spot_rates=rates, days=days)["fixed_rate_annual"]
    v = dv.interest_rate_swap_value(fixed_rate=fs, floating_rate_set=0.0, period_fraction=0.25,
                                    spot_rates=[0.0] + rates[:-1], days=[0, 90, 180, 270])
    # at inception (reset date) the floating leg is worth exactly par
    r0 = dv.interest_rate_swap_value(fixed_rate=fs, floating_rate_set=0.0345, period_fraction=0.25,
                                     spot_rates=rates, days=days)
    assert r0["value"] == approx(0, abs=1e-12)
    assert v["pv_floating_leg"] == approx(1.0)


def test_currency_swap_value():
    # R54 Example 5B p.282: all four swaps paying $ and receiving EUR, S0 0.75 -> St 0.70
    r = dv.currency_swap_value(spot_rate_initial=0.75, spot_rate_now=0.70, period_fraction=0.5,
                               domestic_fixed_rate=0.0784, foreign_fixed_rate=0.0648,
                               domestic_floating_rate_set=0.072, foreign_floating_rate_set=0.06,
                               domestic_spot_rates=[0.071, 0.074], foreign_spot_rates=[0.055, 0.060], days=[90, 270])
    c = r["all_combinations"]
    assert c["pay_domestic_fixed_receive_foreign_fixed"] == approx(-0.0712, abs=3e-4)
    assert c["pay_domestic_fixed_receive_foreign_floating"] == approx(-0.0749, abs=3e-4)
    assert c["pay_domestic_floating_receive_foreign_fixed"] == approx(-0.0661, abs=3e-4)
    assert c["pay_domestic_floating_receive_foreign_floating"] == approx(-0.0698, abs=3e-4)
    assert r["notional_foreign"] == approx(1.3333, abs=5e-5)
    assert r["value"] == approx(c["pay_domestic_fixed_receive_foreign_fixed"])
    # R54 p.281: $/SF swaps, S 0.80 -> 0.82, values +0.0271, +0.0327, +0.0224, +0.0280 (rounded inputs)
    s = dv.currency_swap_value(spot_rate_initial=0.80, spot_rate_now=0.82, period_fraction=0.25,
                               domestic_fixed_rate=0.0368, foreign_fixed_rate=0.0556,
                               domestic_floating_rate_set=0.0345, foreign_floating_rate_set=0.052,
                               domestic_spot_rates=[0.0425, 0.0432, 0.0437, 0.0444],
                               foreign_spot_rates=[0.06, 0.0615, 0.0635, 0.0653], days=[30, 120, 210, 300])["all_combinations"]
    assert s["pay_domestic_fixed_receive_foreign_fixed"] == approx(0.0271, abs=3e-4)
    assert s["pay_domestic_floating_receive_foreign_floating"] == approx(0.0280, abs=3e-4)


def test_equity_swap_value():
    # R54 p.285-286: index 1405.72 -> 1436.59; pay fixed 0.0216, pay floating 0.0169, pay index 2: 0.0161
    kw = dict(index_last_reset=1405.72, index_now=1436.59, spot_rates=[0.0425, 0.0432, 0.0437, 0.0444],
              days=[30, 120, 210, 300], period_fraction=0.25)
    assert dv.equity_swap_value(pay_leg="fixed", fixed_rate=0.0368, **kw)["value"] == approx(0.0216, abs=1e-4)
    assert dv.equity_swap_value(pay_leg="floating", floating_rate_set=0.0345, **kw)["value"] == approx(0.0169, abs=1e-4)
    assert dv.equity_swap_value(index_last_reset=1405.72, index_now=1436.59, pay_leg="equity",
                                other_index_last_reset=5255.18, other_index_now=5285.73)["value"] == \
        approx(0.0161, abs=5e-5)
    # R54 Example 6 p.287: -0.0309 (fixed), -0.0258 (floating), -0.0104 (DJTA) per $1
    k6 = dict(index_last_reset=10033.27, index_now=9955.14, spot_rates=[0.071, 0.074], days=[90, 270],
              period_fraction=0.5)
    assert dv.equity_swap_value(pay_leg="fixed", fixed_rate=0.0784, **k6)["value"] == approx(-0.0309, abs=1e-4)
    assert dv.equity_swap_value(pay_leg="floating", floating_rate_set=0.072, **k6)["value"] == approx(-0.0258, abs=1e-4)
    d = dv.equity_swap_value(index_last_reset=10033.27, index_now=9955.14, pay_leg="equity",
                             other_index_last_reset=2835.17, other_index_now=2842.44, notional=60e6)
    assert d["value"] / 60e6 == approx(-0.0104, abs=5e-5)
    assert dv.equity_swap_value(index_last_reset=10033.27, index_now=9955.14, pay_leg="equity",
                                other_index_last_reset=2835.17, other_index_now=2842.44,
                                position="pay_equity")["value"] == approx(0.0104, abs=5e-5)


def test_swaption_payoff():
    # R54 p.293: payer swaption at 3.60% vs 3.68% market, $20m quarterly -> $4,000 a quarter, PV $15,643
    dfs = [0.9914, 0.9824, 0.9730, 0.9639]
    r = dv.swaption_payoff(exercise_rate=0.036, period_fraction=0.25, discount_factors=dfs, market_swap_rate=0.0368,
                           notional=20e6)
    assert r["payment_per_period"] == approx(4_000)
    assert r["payoff"] == approx(15_643, abs=1)
    # R54 Example 7 p.294-295: receiver at 8% vs 7.84% -> about 0.0015 per $1 ($37,500 on $25m after rounding)
    e = dv.swaption_payoff(exercise_rate=0.08, period_fraction=0.5, spot_rates=[0.072, 0.08], days=[180, 360],
                           market_swap_rate=0.0784, notional=25e6, swaption_type="receiver")
    assert e["payoff"] / 25e6 == approx(0.0015, abs=2e-5)
    # Equivalence: receiver swaption = call on a bond with coupon x (par 1); payer = put on it
    ex = dv.swaption_payoff(exercise_rate=0.08, period_fraction=0.5, spot_rates=[0.072, 0.08], days=[180, 360],
                            swaption_type="receiver")
    assert ex["equivalent_bond_value"] == approx(1.0015, abs=1e-4)  # curriculum rounds 1.00157 down
    assert ex["payoff"] == approx(max(0, ex["equivalent_bond_value"] - 1), rel=1e-12)
    pay = dv.swaption_payoff(exercise_rate=0.09, period_fraction=0.5, spot_rates=[0.072, 0.08], days=[180, 360])
    assert pay["payoff"] == approx(0)
    pay2 = dv.swaption_payoff(exercise_rate=0.07, period_fraction=0.5, spot_rates=[0.072, 0.08], days=[180, 360])
    assert pay2["payoff"] == approx(max(0, 1 - pay2["equivalent_bond_value"]), rel=1e-12)


# ============================================================ option basics, bounds, parity
def test_option_bounds():
    # R53 Example 2 p.179: S 72, r 4.5%, 42 days
    kw = dict(spot=72, rate=0.045, years=42 / 365)
    b70, b75 = do.option_bounds(strike=70, **kw), do.option_bounds(strike=75, **kw)
    assert b70["european_call_lower"] == approx(2.35, abs=0.005) and b75["european_call_lower"] == 0
    assert b70["european_put_lower"] == 0 and b75["european_put_lower"] == approx(2.62, abs=0.005)
    assert b70["american_call_lower"] == approx(2.35, abs=0.005) and b75["american_put_lower"] == approx(3)
    # R53 p.178: X 50, r 5%, T 0.5, S 45 -> European put lower bound 3.80
    assert do.option_bounds(spot=45, strike=50, rate=0.05, years=0.5)["european_put_lower"] == approx(3.80, abs=0.005)


def test_intrinsic_and_time_value():
    r = do.option_intrinsic_time_value(spot=98, strike=100, option_type="call", option_price=7)
    assert r["moneyness"] == "out-of-the-money" and r["time_value"] == 7
    r = do.option_intrinsic_time_value(spot=55, strike=60, option_type="put", option_price=6)
    assert r["moneyness"] == "in-the-money" and r["intrinsic_value"] == 5 and r["time_value"] == approx(1)
    assert do.option_intrinsic_time_value(spot=50, strike=50)["moneyness"] == "at-the-money"


def test_put_call_parity():
    # R53 Example 3 p.187: synthetic call 7.37 vs 8.00 -> sell call, gain 0.63
    r = do.put_call_parity(strike=45, rate=0.045, years=115 / 365, call=8.00, put=3.75, spot=48)
    assert r["synthetic_call"] == approx(7.37, abs=0.005)
    assert r["mispricing"] == approx(0.63, abs=0.005) and r["strategy"].startswith("call rich")
    # solve for each leg and round trip
    c = do.put_call_parity(strike=45, rate=0.045, years=115 / 365, put=3.75, spot=48)["call"]
    assert do.put_call_parity(strike=45, rate=0.045, years=115 / 365, call=c, spot=48)["put"] == approx(3.75)
    assert do.put_call_parity(strike=45, rate=0.045, years=115 / 365, call=c, put=3.75)["spot"] == approx(48)


def test_put_call_forward_parity():
    # R53 p.226-227: c 5.50, p 10.50, F 90.72, X 95, r 6%, 50 days -> -0.7540
    r = do.put_call_forward_parity(strike=95, rate=0.06, years=50 / 365, call=5.50, put=10.50, forward_price=90.72)
    assert r["synthetic_forward_value"] == approx(-0.7540, abs=5e-5)
    # R53 Example 10 p.227: synthetic put 4.88 vs 3.00
    e = do.put_call_forward_parity(strike=90, rate=0.05, years=2, call=15.25, put=3.00, forward_price=101.43)
    assert e["synthetic_put"] == approx(4.88, abs=0.005)
    f = do.put_call_forward_parity(strike=90, rate=0.05, years=2, call=15.25, put=e["synthetic_put"])
    assert f["forward_price"] == approx(101.43)


def test_parity_holds_for_bsm_prices():
    b = do.bsm_option_price(spot=52.75, strike=50, rate=0.0488, volatility=0.35, years=0.75)
    p = do.put_call_parity(strike=50, rate=0.0488, years=0.75, call=b["call_price"], spot=52.75,
                           compounding="continuous")
    assert p["put"] == approx(b["put_price"], rel=1e-12)


# ============================================================ binomial
def test_binomial_one_period():
    # R53 p.193-194: S 50, u 1.25, d 0.80, r 7%, X 50 -> pi 0.6, c 7.01, n 0.556
    r = do.binomial_option_price(spot=50, strike=50, up=1.25, down=0.80, rate_per_period=0.07)
    assert r["risk_neutral_prob"] == approx(0.6) and r["price"] == approx(7.01, abs=0.005)
    assert r["hedge_ratio"] == approx(0.556, abs=5e-4)
    # replication: n S - B reproduces the price and the up/down payoffs
    assert r["hedge_ratio"] * 50 - r["bond_position"] == approx(r["price"])
    assert r["hedge_ratio"] * 62.5 - r["bond_position"] * 1.07 == approx(12.5)
    # R53 Example 4 p.194-195: S 65, u 1.30, d 0.78, r 8%, X 70 -> 7.75, pi 0.5769, n 0.4290
    e = do.binomial_option_price(spot=65, strike=70, up=1.30, down=0.78, rate_per_period=0.08)
    assert e["price"] == approx(7.75, abs=0.005) and e["risk_neutral_prob"] == approx(0.5769, abs=5e-5)
    assert e["hedge_ratio"] == approx(0.4290, abs=5e-5)
    # R53 Example 6 p.200: put 7.56
    p = do.binomial_option_price(spot=65, strike=70, up=1.30, down=0.78, rate_per_period=0.08, option_type="put")
    assert p["price"] == approx(7.56, abs=0.005) and p["hedge_ratio"] < 0


def test_binomial_two_period():
    # R53 p.197-198: u 1.118, d 0.8944, r 3.44% -> 4.58
    r = do.binomial_option_price(spot=50, strike=50, up=1.118, down=0.8944, rate_per_period=0.0344, periods=2)
    assert r["price"] == approx(4.58, abs=0.005)
    # R53 Example 5 p.198-199: S 30, u 1.14, d 0.89, r 3% -> 2.86; n 0.6453, n+ 1.00, n- 0.0659 (rounded inputs)
    e = do.binomial_option_price(spot=30, strike=30, up=1.14, down=0.89, rate_per_period=0.03, periods=2)
    assert e["price"] == approx(2.86, abs=0.005)
    assert e["hedge_ratio_tree"][0][0] == approx(0.6453, abs=1e-3)
    assert e["hedge_ratio_tree"][1] == approx([1.00, 0.0659], abs=1e-3)
    assert e["option_tree"][1] == approx([5.08, 0.24], abs=0.01)


def test_binomial_american_early_exercise():
    # Hull, Options, Futures and Other Derivatives (two-step American put): S 50, X 52, u 1.2, d 0.8, r 5% cont.
    kw = dict(spot=50, strike=52, up=1.2, down=0.8, rate_per_period=math.exp(0.05) - 1, periods=2, option_type="put")
    eu = do.binomial_option_price(**kw)
    am = do.binomial_option_price(american=True, **kw)
    assert eu["price"] == approx(4.1923, abs=5e-4)
    assert am["price"] == approx(5.0894, abs=5e-4)
    assert am["early_exercise_nodes"] == [[1, 1]]  # exercise at the down node after one step
    # American call on a non-dividend asset is never exercised early
    c = do.binomial_option_price(spot=50, strike=52, up=1.2, down=0.8, rate_per_period=0.05, periods=3, american=True)
    assert c["early_exercise_nodes"] == []
    assert c["price"] == approx(do.binomial_option_price(spot=50, strike=52, up=1.2, down=0.8, rate_per_period=0.05,
                                                         periods=3)["price"])


def test_binomial_crr_convergence():
    # R53 Exhibit 23 p.207: 1 step 10.0259, 2 -> 8.4782, 5 -> 8.8305, 10 -> 8.6983, 1000 -> 8.6190
    kw = dict(spot=52.75, strike=50, rate=0.0488, volatility=0.35, years=0.75)
    for n, expected in ((1, 10.0259), (2, 8.4782), (5, 8.8305), (10, 8.6983), (1000, 8.6190)):
        assert do.binomial_crr_option_price(steps=n, **kw)["price"] == approx(expected, abs=1e-4)
    bsm = do.bsm_option_price(**kw)["call_price"]
    assert do.binomial_crr_option_price(steps=1000, **kw)["price"] == approx(bsm, abs=1e-3)
    # Hull American put example: S 50, X 50, r 10%, sigma 40%, 5 months, 5 steps -> 4.49
    am = do.binomial_crr_option_price(spot=50, strike=50, rate=0.10, volatility=0.40, years=5 / 12, steps=5,
                                      option_type="put", american=True)
    assert am["price"] == approx(4.49, abs=0.005) and am["early_exercise_node_count"] > 0
    # dividend yield: CRR European converges to BSM with q
    q = dict(spot=2000, strike=1950, rate=0.02, volatility=0.35, years=1 / 12, dividend_yield=0.005)
    assert do.binomial_crr_option_price(steps=2000, option_type="put", **q)["price"] == \
        approx(do.bsm_option_price(**q)["put_price"], abs=0.01)


def test_binomial_interest_rate_options():
    # R53 p.201-204: zero bond call 0.0119, coupon bond call 0.0352, two-period caplet 0.0138, one-period 0.0102
    tree = [[0.1051], [0.1304, 0.1025], [0.1578, 0.1180, 0.0795]]
    assert do.binomial_interest_rate_option(rate_tree=tree, exercise=0.80, underlying="price",
                                            underlying_values=[0.7417, 0.7979, 0.8583])["price"] == approx(0.0119, abs=1e-4)
    assert do.binomial_interest_rate_option(rate_tree=tree, exercise=0.95, underlying="price",
                                            underlying_values=[0.9183, 0.9840, 1.0546])["price"] == approx(0.0352, abs=5e-5)
    two = do.binomial_interest_rate_option(rate_tree=tree, exercise=0.105)["price"]
    one = do.binomial_interest_rate_option(rate_tree=tree[:2], exercise=0.105)["price"]
    assert two == approx(0.0138, abs=5e-5) and one == approx(0.0102, abs=5e-5)
    assert two + one == approx(0.0240, abs=1e-4)  # cap = sum of caplets
    # R53 Example 7 p.205-206: put on 6% coupon bond 0.0116; put on the rate 0.0064
    t7 = [[0.0513], [0.064, 0.0429], [0.0858, 0.0590, 0.0328]]
    assert do.binomial_interest_rate_option(rate_tree=t7, exercise=1.01, option_type="put", underlying="price",
                                            underlying_values=[0.9763, 1.0010, 1.0263])["price"] == approx(0.0116, abs=5e-5)
    assert do.binomial_interest_rate_option(rate_tree=t7, exercise=0.06, option_type="put")["price"] == \
        approx(0.0064, abs=5e-5)


# ============================================================ BSM, Greeks, implied vol, Black
def test_bsm_hull_and_curriculum():
    # Hull BSM example: S 42, X 40, r 10%, sigma 20%, T 0.5 -> c 4.76, p 0.81
    h = do.bsm_option_price(spot=42, strike=40, rate=0.10, volatility=0.20, years=0.5)
    assert h["call_price"] == approx(4.76, abs=0.005) and h["put_price"] == approx(0.81, abs=0.005)
    # R53 p.212-213 (computer values): 8.6186 / 4.0717; S 53.75 -> 9.3030 / 3.7560; S 62.75 -> 16.3026 / 1.7557
    kw = dict(strike=50, rate=0.0488, volatility=0.35, years=0.75)
    for s, c, p in ((52.75, 8.6186, 4.0717), (53.75, 9.3030, 3.7560), (62.75, 16.3026, 1.7557)):
        r = do.bsm_option_price(spot=s, **kw)
        assert r["call_price"] == approx(c, abs=1e-4) and r["put_price"] == approx(p, abs=1e-4)
    assert do.bsm_option_price(spot=52.75, strike=50, rate=0.0488, volatility=0.40, years=0.75)["call_price"] == \
        approx(9.446, abs=5e-4)  # R53 p.218
    # R53 Example 8 p.211: d1 0.4135, d2 0.2049; c 7.95, p 3.67 with table N(d) (exact N gives 7.91 / 3.63)
    e = do.bsm_option_price(spot=68.5, strike=65, rate=0.04, volatility=0.38, years=110 / 365)
    assert e["d1"] == approx(0.4135, abs=5e-4) and e["d2"] == approx(0.2049, abs=5e-4)
    assert e["call_price"] == approx(7.95, abs=0.05) and e["put_price"] == approx(3.67, abs=0.05)


def test_bsm_cash_flows_and_currency():
    # R53 Example 9 p.220: PV(CF) 19.72 -> 41.28; dividend yield 2.7% -> 43.06 (table N(d))
    kw = dict(spot=225, strike=200, rate=0.0525, volatility=0.15, years=3)
    a = do.bsm_option_price(pv_income=19.72, **kw)
    assert a["d1"] == approx(0.8364, abs=5e-4) and a["call_price"] == approx(41.28, abs=0.05)
    b = do.bsm_option_price(dividend_yield=0.027, **kw)
    assert b["adjusted_spot"] == approx(207.49, abs=0.005) and b["call_price"] == approx(43.06, abs=0.05)
    # R53 p.219-220 currency option: d1 -1.7590, d2 -1.7912, c about 0.0003 (table N(d))
    fx = do.bsm_option_price(spot=0.8475, strike=0.90, rate=0.051, volatility=0.055, years=125 / 365,
                             dividend_yield=0.0425)
    # curriculum plugs the rounded adjusted spot 0.8353 into d1, shifting d1/d2 by about 0.002
    assert fx["d1"] == approx(-1.7590, abs=3e-3) and fx["d2"] == approx(-1.7912, abs=3e-3)
    assert fx["call_price"] == approx(0.0003, abs=2e-4)


def test_bsm_dividend_yield_nasdaq_table():
    # R30 p.381 + footnote 17 p.409: S 2000, T 1/12, sigma 0.35, r 0.02, q 0.005
    kw = dict(spot=2000, rate=0.02, volatility=0.35, years=1 / 12, dividend_yield=0.005)
    for x, c, p in ((1950, 108.43, 56.01), (2000, 81.75, 79.25), (2050, 59.98, 107.39)):
        r = do.bsm_option_price(strike=x, **kw)
        assert r["call_price"] == approx(c, abs=0.01) and r["put_price"] == approx(p, abs=0.01)
    assert do.bsm_option_price(strike=2061, **kw)["call_price"] == approx(55.82, abs=0.01)


def test_bsm_delta_hedge_table():
    # R30 p.440-443: S 1210, X 1200, r 2.75%, sigma 20%; prices and deltas through the delta-hedge example
    kw = dict(strike=1200, rate=0.0275, volatility=0.20)
    for s, days, c, delta in ((1210, 120, 65.88, 0.5826), (1200, 120, 60.19, 0.5542), (1240, 120, 84.59, 0.6635),
                              (1215, 119, 68.55, 0.5966), (1198, 118, 58.54, 0.5479), (1192, 117, 55.04, 0.5300)):
        g = do.bsm_greeks(spot=s, years=days / 365, **kw)
        assert g["price"] == approx(c, abs=0.005) and g["delta"] == approx(delta, abs=2e-4)


def test_bsm_greeks_hull_and_numeric():
    # Hull Greek letters example: S 49, X 50, r 5%, sigma 20%, T 20 weeks
    g = do.bsm_greeks(spot=49, strike=50, rate=0.05, volatility=0.20, years=20 / 52)
    assert g["delta"] == approx(0.522, abs=5e-4) and g["gamma"] == approx(0.066, abs=5e-4)
    assert g["theta"] == approx(-4.31, abs=0.005) and g["theta_per_day"] == approx(-0.0118, abs=5e-5)
    assert g["vega"] == approx(12.1, abs=0.05) and g["rho"] == approx(8.91, abs=0.005)
    # finite-difference checks, both option types, with a dividend yield
    base = dict(spot=100, strike=95, rate=0.04, volatility=0.25, years=0.8, dividend_yield=0.02)
    h = 1e-4

    def px(t, **ch):
        return do.bsm_option_price(**{**base, **ch})[f"{t}_price"]

    for t in ("call", "put"):
        gk = do.bsm_greeks(option_type=t, **base)
        assert gk["delta"] == approx((px(t, spot=100 + h) - px(t, spot=100 - h)) / (2 * h), abs=1e-6)
        assert gk["gamma"] == approx((px(t, spot=100 + 1e-2) - 2 * px(t) + px(t, spot=100 - 1e-2)) / 1e-4, abs=1e-5)
        assert gk["vega"] == approx((px(t, volatility=0.25 + h) - px(t, volatility=0.25 - h)) / (2 * h), abs=1e-5)
        assert gk["rho"] == approx((px(t, rate=0.04 + h) - px(t, rate=0.04 - h)) / (2 * h), abs=1e-5)
        assert gk["theta"] == approx(-(px(t, years=0.8 + h) - px(t, years=0.8 - h)) / (2 * h), abs=1e-4)
    assert do.bsm_greeks(option_type="call", **base)["delta"] - do.bsm_greeks(option_type="put", **base)["delta"] == \
        approx(math.exp(-0.02 * 0.8))


def test_implied_volatility():
    # R53 p.222: price 9.25 -> 0.3882
    r = do.bsm_implied_volatility(option_price=9.25, spot=52.75, strike=50, rate=0.0488, years=0.75)
    assert r["implied_volatility"] == approx(0.3882, abs=5e-5)
    # round trip for a put with a dividend yield
    p = do.bsm_option_price(spot=100, strike=110, rate=0.03, volatility=0.27, years=1.5, dividend_yield=0.01)["put_price"]
    assert do.bsm_implied_volatility(option_price=p, spot=100, strike=110, rate=0.03, years=1.5, dividend_yield=0.01,
                                     option_type="put")["implied_volatility"] == approx(0.27, abs=1e-9)
    with pytest.raises(ValueError):
        do.bsm_implied_volatility(option_price=60, spot=52.75, strike=50, rate=0.0488, years=0.75)


def test_black_futures_option():
    # R53 Example 11 p.229-230: d1 0.9916, d2 0.8765; c 15.06 with table N(d) (exact N gives 15.23)
    r = do.black_futures_option_price(futures_price=139.19, strike=125, rate=0.0425, volatility=0.15, years=215 / 365)
    assert r["d1"] == approx(0.9916, abs=5e-4) and r["d2"] == approx(0.8765, abs=5e-4)
    assert r["call_price"] == approx(15.06, abs=0.2)
    # consistency: Black on F equals BSM on S = F e^(-rT)
    T = 215 / 365
    s = 139.19 * math.exp(-0.0425 * T)
    bsm = do.bsm_option_price(spot=s, strike=125, rate=0.0425, volatility=0.15, years=T)
    assert r["call_price"] == approx(bsm["call_price"], rel=1e-12) and r["put_price"] == approx(bsm["put_price"], rel=1e-12)


def test_black_interest_rate_options_and_cap():
    # R53 p.231-232: caplet, F 9.25%, X 9%, sigma 0.03, r 9.6%, 90 days: d1 1.8466, d2 1.8317, lag factor 0.97745;
    # $607.47 per $1m with table N(d) (exact N gives about $601)
    r = do.black_interest_rate_option(forward_rate=0.0925, strike_rate=0.09, rate=0.096, volatility=0.03,
                                      years=90 / 365, days_underlying=90, notional=1e6)
    assert r["d1"] == approx(1.8466, abs=5e-4) and r["d2"] == approx(1.8317, abs=5e-4)
    assert r["lag_discount"] == approx(0.00242988 / 0.00248594, abs=1e-5)
    assert r["price"] == approx(607.47, rel=0.015)
    # R53 Example 12 p.232-233: floorlet on $10m -> $9,084 with table N(d) (exact N: about $9,330)
    p = do.black_interest_rate_option(forward_rate=0.068, strike_rate=0.07, rate=0.0625, volatility=0.02,
                                      years=280 / 365, days_underlying=180, notional=10e6, option_type="put")
    assert p["d1"] == approx(-1.6461, abs=5e-4) and p["d2"] == approx(-1.6636, abs=5e-4)
    assert p["price"] == approx(9_084, rel=0.03)
    cap = do.black_cap_floor_price(forward_rates=[0.0925, 0.095], expiries=[90 / 365, 180 / 365],
                                   discount_rates=[0.096, 0.097], volatilities=0.03, strike_rate=0.09,
                                   days_underlying=90, notional=1e6)
    assert cap["caplet_prices"][0] == approx(r["price"]) and cap["price"] == approx(sum(cap["caplet_prices"]))


def test_interest_rate_option_payoff():
    # R53 p.164: $10m, LIBOR 6% vs 5.5%, 180-day -> $25,000 paid 180 days later
    assert do.interest_rate_option_payoff(notional=10e6, underlying_rate=0.06, exercise_rate=0.055,
                                          days_underlying=180)["payoff"] == approx(25_000)
    assert do.interest_rate_option_payoff(notional=10e6, underlying_rate=0.06, exercise_rate=0.055,
                                          days_underlying=180, option_type="put")["payoff"] == 0


# ============================================================ strategies
def test_single_option_payoffs():
    # R30 Example 1 p.385: call X 100, c 7
    c = do.option_payoff_profit(option_type="call", strike=100, premium=7, spot_at_expiry=102)
    assert (c["value_at_expiration"], c["profit"], c["max_loss"], c["breakevens"]) == (2, -5, 7, [107])
    assert c["max_gain"] == math.inf
    s = do.option_payoff_profit(option_type="call", strike=100, premium=7, spot_at_expiry=91, position="short")
    assert s["profit"] == 7 and s["max_gain"] == 7 and s["max_loss"] == math.inf
    # R30 Example 2 p.388: put X 60, p 4
    p = do.option_payoff_profit(option_type="put", strike=60, premium=4, spot_at_expiry=55)
    assert p["profit"] == 1 and p["max_gain"] == 56 and p["breakevens"] == [56]
    sp = do.option_payoff_profit(option_type="put", strike=60, premium=4, spot_at_expiry=51, position="short")
    assert sp["profit"] == -5 and sp["max_loss"] == 56


def test_covered_call_and_protective_put():
    # R30 Example 3 p.393
    a = do.covered_call(stock_price=98, call_strike=105, call_premium=8, spot_at_expiry=110)
    assert (a["value_at_expiration"], a["profit"], a["max_gain"], a["max_loss"], a["breakevens"]) == (105, 15, 15, 90, [90])
    assert do.covered_call(stock_price=98, call_strike=105, call_premium=8, spot_at_expiry=88)["profit"] == -2
    # R30 Example 4 p.396
    b = do.protective_put(stock_price=0.875, put_strike=0.90, put_premium=0.075, spot_at_expiry=0.75)
    assert b["value_at_expiration"] == approx(0.90) and b["profit"] == approx(-0.05)
    assert b["max_loss"] == approx(0.05) and b["max_gain"] == math.inf and b["breakevens"] == approx([0.95])


def test_spreads():
    # R30 Example 5 p.399: bull call spread 65/75
    bull = [do.bull_spread(low_strike=65, low_premium=9, high_strike=75, high_premium=4, spot_at_expiry=s)["profit"]
            for s in (78, 69, 62)]
    assert bull == approx([5, -1, -5])
    b = do.bull_spread(low_strike=65, low_premium=9, high_strike=75, high_premium=4, spot_at_expiry=70)
    assert (b["max_gain"], b["max_loss"], b["breakevens"]) == (5, 5, [70])
    # R30 Example 6 p.402: bear put spread 85/95
    bear = [do.bear_spread(low_strike=85, low_premium=3, high_strike=95, high_premium=11, spot_at_expiry=s)["profit"]
            for s in (98, 91, 82)]
    assert bear == approx([-8, -4, 2])
    e = do.bear_spread(low_strike=85, low_premium=3, high_strike=95, high_premium=11, spot_at_expiry=90)
    assert (e["max_gain"], e["max_loss"], e["breakevens"]) == (2, 8, [87])
    # puts bull spread and calls bear spread are the credit versions (same max gain/loss structure)
    bp = do.bull_spread(low_strike=1950, low_premium=56.01, high_strike=2050, high_premium=107.39, spot_at_expiry=2100,
                        option_type="put")
    assert bp["initial_cost"] == approx(56.01 - 107.39) and bp["profit"] == approx(51.38)
    bc = do.bear_spread(low_strike=1950, low_premium=108.43, high_strike=2050, high_premium=59.98, spot_at_expiry=1900,
                        option_type="call")
    assert bc["profit"] == approx(108.43 - 59.98) and bc["max_loss"] == approx(100 - 48.45)


def test_butterfly():
    # R30 Example 7 p.406: put butterfly 1.30/1.40/1.50
    kw = dict(low_strike=1.30, low_premium=0.08, mid_strike=1.40, mid_premium=0.125, high_strike=1.50,
              high_premium=0.18, option_type="put")
    prof = [do.butterfly_spread(spot_at_expiry=s, **kw)["profit"] for s in (1.26, 1.35, 1.47, 1.59)]
    assert prof == approx([-0.01, 0.04, 0.02, -0.01])
    r = do.butterfly_spread(spot_at_expiry=1.40, **kw)
    assert r["max_gain"] == approx(0.09) and r["max_loss"] == approx(0.01)
    assert r["breakevens"] == approx([1.31, 1.49])
    sb = do.butterfly_spread(spot_at_expiry=1.40, position="short", **kw)
    assert sb["profit"] == approx(-0.09)


def test_collar_and_zero_cost_strike():
    # R30 Example 8 p.410: put 40 at 5.32, call 50.59 at 5.32
    kw = dict(stock_price=42, put_strike=40, put_premium=5.32, call_strike=50.59, call_premium=5.32)
    assert [do.collar(spot_at_expiry=s, **kw)["value_at_expiration"] for s in (55, 48, 35)] == approx([50.59, 48, 40])
    r = do.collar(spot_at_expiry=55, **kw)
    assert r["profit"] == approx(8.59) and r["max_gain"] == approx(8.59) and r["max_loss"] == approx(2)
    assert r["breakevens"] == approx([42])
    # R30 p.408-409 footnote 17: zero-cost call strike between 2060 and 2061 for the 1950 put
    z = do.zero_cost_collar_call_strike(spot=2000, put_strike=1950, rate=0.02, volatility=0.35, years=1 / 12,
                                        dividend_yield=0.005)
    assert 2060 < z["call_strike"] < 2061 and z["premium"] == approx(56.01, abs=0.01)


def test_straddle_strangle_box():
    # R30 Example 9 p.413
    a = do.straddle(strike=50, call_premium=6.25, put_premium=5.875, spot_at_expiry=61)
    assert a["profit"] == approx(-1.125) and a["max_loss"] == approx(12.125) and a["max_gain"] == math.inf
    assert a["breakevens"] == approx([37.875, 62.125])
    assert do.straddle(strike=50, call_premium=6.25, put_premium=5.875, spot_at_expiry=37)["profit"] == approx(0.875)
    s = do.straddle(strike=50, call_premium=6.25, put_premium=5.875, spot_at_expiry=61, position="short")
    assert s["profit"] == approx(1.125) and s["max_loss"] == math.inf
    st = do.strangle(put_strike=1950, put_premium=56.01, call_strike=2050, call_premium=59.98, spot_at_expiry=2200)
    assert st["profit"] == approx(150 - 115.99) and st["breakevens"] == approx([1834.01, 2165.99])
    # R30 p.414: box 1950/2050 always worth 100, profit 0.17; Example 10 p.415: cost 9.20, PV 9.75, gain 0.55
    b = do.box_spread(low_strike=1950, high_strike=2050, call_low_premium=108.43, call_high_premium=59.98,
                      put_low_premium=56.01, put_high_premium=107.39, rate=0.02, years=1 / 12, compounding="continuous")
    assert b["value_at_expiration"] == 100 and b["profit"] == approx(0.17)
    e = do.box_spread(low_strike=75, high_strike=85, call_low_premium=16.02, call_high_premium=12.28,
                      put_low_premium=9.72, put_high_premium=15.18, rate=0.0513, years=0.5)
    assert e["initial_cost"] == approx(9.20) and e["profit"] == approx(0.80)
    assert e["pv_payoff"] == approx(9.75, abs=0.005) and e["arbitrage_gain"] == approx(0.55, abs=0.005)
    assert e["strategy"].startswith("buy")


def test_generic_strategy_engine():
    legs = [{"type": "call", "strike": 65, "premium": 9, "quantity": 1},
            {"type": "call", "strike": 75, "premium": 4, "quantity": -1}]
    assert do.option_strategy_expiration(legs=legs, spot_at_expiry=78) == \
        do.bull_spread(low_strike=65, low_premium=9, high_strike=75, high_premium=4, spot_at_expiry=78)
    with pytest.raises(ValueError):
        do.option_strategy_expiration(legs=[{"type": "swap", "strike": 1}], spot_at_expiry=1)


# ============================================================ L3: futures risk management
def test_equity_beta_change():
    # R29 Example 3 p.340: 74.96 -> buy 75; outcome effective beta 1.025
    r = dr.equity_futures_contracts_beta(portfolio_value=65e6, current_beta=0.85, target_beta=1.05,
                                         futures_price=188_500, futures_beta=0.92)
    assert r["contracts"] == approx(74.96, abs=0.005) and r["contracts_rounded"] == 75 and r["action"] == "buy"
    value = 65e6 * (1 - 0.0165) + 75 * (185_000 - 188_500)
    assert value == approx(63_665_000) and value / 65e6 - 1 == approx(-0.0205, abs=1e-4)
    # curriculum: effective beta = 0.0205 / 0.02 = 1.025 from the rounded return
    assert (1 - value / 65e6) / 0.02 == approx(1.025, abs=3e-3)
    # multiplier: S&P quoted 1225 x $250 (R29 p.337)
    m = dr.equity_futures_contracts_beta(portfolio_value=306_250, current_beta=1, target_beta=0, futures_price=1225,
                                         multiplier=250)
    assert m["contracts"] == approx(-1) and m["futures_contract_value"] == approx(306_250)


def test_bond_duration_change():
    # R29 Exhibit 2 p.331-333: 150.47 (increase), -731.19 (full hedge)
    kw = dict(portfolio_value=75e6, current_mdur=6.22, futures_price=82_500, futures_mdur=8.12, yield_beta=1.05)
    assert dr.bond_futures_contracts_duration(target_mdur=7.50, **kw)["contracts"] == approx(150.47, abs=0.005)
    assert dr.bond_futures_contracts_duration(target_mdur=0.0, **kw)["contracts"] == approx(-731.19, abs=0.005)
    # R29 Example 2 p.333: -337.33 -> sell 337
    e = dr.bond_futures_contracts_duration(portfolio_value=100e6, current_mdur=7.2, target_mdur=5.0,
                                           futures_price=120_000, futures_mdur=6.25, yield_beta=1.15)
    assert e["contracts"] == approx(-337.33, abs=0.005) and e["contracts_rounded"] == -337
    # R29 p.352 (FIMA): -104.28 and 33.37
    assert dr.bond_futures_contracts_duration(portfolio_value=10e6, current_mdur=6.5, target_mdur=0.25,
                                              futures_price=87_500, futures_mdur=6.85)["contracts"] == approx(-104.28, abs=0.005)
    assert dr.bond_futures_contracts_duration(portfolio_value=20e6, current_mdur=6.5, target_mdur=7.5,
                                              futures_price=87_500, futures_mdur=6.85)["contracts"] == approx(33.37, abs=0.005)


def test_bond_duration_change_ctd_form():
    # L3 FI 2014 R22 Example 11 p.130 (Level-3/07-Fixed-Income-Portfolio-Management/...R22...): -131.17
    r = dr.bond_futures_contracts_ctd(portfolio_value=50e6, current_duration=9.52, target_duration=7.5,
                                      ctd_price=100_000, ctd_duration=8.47, conversion_factor=1.1)
    assert r["contracts"] == approx(-131.17, abs=0.005) and r["action"] == "sell"
    # BPV form gives the same count: (BPV_T - BPV_P) / BPV_futures
    assert (r["target_bpv"] - r["portfolio_bpv"]) / r["futures_bpv"] == approx(r["contracts"])


def test_synthetic_index_and_cash():
    # R29 Exhibit 4 p.343: 2530.68 -> 2531, invest 100,012,622, bonds grow to 101,240,000, 25,154.24 units
    r = dr.synthetic_equity_cash_futures(amount=100e6, futures_price=4000, multiplier=10, rate=0.05, years=0.25,
                                         dividend_yield=0.025)
    assert r["contracts"] == approx(2530.68, abs=0.005) and r["contracts_rounded"] == 2531
    assert r["effective_amount"] == approx(100_012_622, abs=1) and r["bond_value_at_expiration"] == approx(101_240_000)
    assert r["units_at_start"] == approx(25_154.24, abs=0.005) and r["units_at_expiration"] == 25_310
    # R29 Example 4 p.344-345: equitize $500m
    e = dr.synthetic_equity_cash_futures(amount=500e6, futures_price=1500, multiplier=100, rate=0.03, years=0.5,
                                         dividend_yield=0.01)
    assert e["contracts"] == approx(3382.96, abs=0.005) and e["effective_amount"] == approx(500_005_342, abs=1)
    assert e["bond_value_at_expiration"] == approx(507_450_000) and e["units_at_start"] == approx(336_621.08, abs=0.005)
    # R29 Exhibit 5 p.346: synthetic cash -339.32 -> -339; 49,952,173; 50,332,008; 33,857.81 units
    c = dr.synthetic_equity_cash_futures(amount=50e6, futures_price=1484.72, multiplier=100, rate=0.0465,
                                         years=2 / 12, dividend_yield=0.0075, direction="to_cash")
    assert c["contracts"] == approx(-339.32, abs=0.005) and c["contracts_rounded"] == -339
    assert c["effective_amount"] == approx(49_952_173, abs=1) and c["bond_value_at_expiration"] == approx(50_332_008)
    assert c["units_at_start"] == approx(33_857.81, abs=0.005)
    # R29 Example 5 p.347: -310.35 -> sell 310, 99,887,229, 100,750,000, 308.47 units
    d = dr.synthetic_equity_cash_futures(amount=100e6, futures_price=325_000, rate=0.035, years=0.25,
                                         dividend_yield=0.02, direction="to_cash")
    assert d["contracts"] == approx(-310.35, abs=0.005) and d["effective_amount"] == approx(99_887_229, abs=1)
    assert d["bond_value_at_expiration"] == approx(100_750_000) and d["units_at_start"] == approx(308.47, abs=0.005)


def test_asset_allocation_with_futures():
    # R29 Exhibit 6 p.350: sell 515.63 stock futures, buy 771.97 bond futures
    r = dr.futures_asset_allocation(portfolio_value=300e6, stock_weight=0.8, target_stock_weight=0.5, stock_beta=1.10,
                                    stock_futures_price=200_000, stock_futures_beta=0.96, bond_mdur=6.5,
                                    bond_futures_price=105_250, bond_futures_mdur=7.2)
    assert r["stock_futures"] == approx(-515.63, abs=0.005) and r["bond_futures"] == approx(771.97, abs=0.005)
    assert (r["stock_futures_rounded"], r["bond_futures_rounded"]) == (-516, 772)
    # R29 Example 6 p.354: components -38.27, 67.69, -10.93, -24.62 -> sell 49, buy 43
    e = dr.futures_asset_allocation(portfolio_value=100e6, stock_weight=0.7, target_stock_weight=0.6, stock_beta=1.05,
                                    stock_futures_price=280_000, stock_futures_beta=0.98, bond_mdur=5.5,
                                    bond_futures_price=125_000, bond_futures_mdur=6.5, target_stock_beta=1.0,
                                    target_bond_mdur=5.0)
    c = e["components"]
    assert [c["stock_to_cash"], c["cash_to_bonds"], c["stock_beta_adjustment"], c["bond_duration_adjustment"]] == \
        approx([-38.27, 67.69, -10.93, -24.62], abs=0.005)
    assert (e["stock_futures_rounded"], e["bond_futures_rounded"]) == (-49, 43)
    # moving bonds to stock flips the signs
    up = dr.futures_asset_allocation(portfolio_value=300e6, stock_weight=0.5, target_stock_weight=0.8, stock_beta=1.10,
                                     stock_futures_price=200_000, stock_futures_beta=0.96, bond_mdur=6.5,
                                     bond_futures_price=105_250, bond_futures_mdur=7.2)
    assert up["stock_futures"] == approx(515.63, abs=0.005) and up["bond_futures"] == approx(-771.97, abs=0.005)


def test_preinvesting():
    # R29 Exhibit 9 p.356: 31.74 and 29.99; Example 7 p.357: 15.86 and 28.18
    a = dr.preinvest_futures_contracts(stock_amount=6e6, target_beta=1.08, stock_futures_price=210_500,
                                       stock_futures_beta=0.97, bond_amount=4e6, target_mdur=5.25,
                                       bond_futures_price=115_750, bond_futures_mdur=6.05)
    assert a["stock_futures"] == approx(31.74, abs=0.005) and a["bond_futures"] == approx(29.99, abs=0.005)
    b = dr.preinvest_futures_contracts(stock_amount=3e6, target_beta=1.0, stock_futures_price=195_000,
                                       stock_futures_beta=0.97, bond_amount=3e6, target_mdur=6.2,
                                       bond_futures_price=110_000, bond_futures_mdur=6.0)
    assert b["stock_futures"] == approx(15.86, abs=0.005) and b["bond_futures"] == approx(28.18, abs=0.005)
    assert (b["stock_futures_rounded"], b["bond_futures_rounded"]) == (16, 28)


def test_foreign_equity_and_currency_hedge():
    # R29 Exhibit 12 p.363-364: sell 96.49 futures; forward on EUR 10.4m; values 7,457,500 / 8,165,884 / 8,477,884
    r = dr.foreign_equity_currency_hedge(portfolio_value_foreign=10e6, portfolio_beta=1.10, index_futures_price=120_000,
                                         index_futures_beta=0.95, foreign_rate=0.04, years=1, spot_rate=0.80,
                                         forward_rate=0.815, ending_portfolio_value_foreign=9.5e6,
                                         ending_futures_price=110_600, ending_spot_rate=0.785)
    assert r["index_futures"] == approx(-96.49, abs=0.005) and r["forward_notional_foreign"] == approx(10.4e6)
    assert r["value_unhedged"] == approx(7_457_500) and r["value_market_hedged"] == approx(8_165_884)
    assert r["value_fully_hedged"] == approx(8_477_884)
    assert r["return_fully_hedged"] == approx(0.0597, abs=5e-5)  # about the domestic risk-free rate (6%)
    # R29 Example 9 p.365: SF10m, beta 1, futures beta 0.9 -> -37.04; LIBOR-style 5% for 180 days -> SF10.25m; $7,725,425
    e = dr.foreign_equity_currency_hedge(portfolio_value_foreign=10e6, portfolio_beta=1.0, index_futures_price=300_000,
                                         index_futures_beta=0.90, foreign_rate=0.05, years=0.5, spot_rate=0.75,
                                         forward_rate=0.7537, compounding="simple")
    assert e["index_futures"] == approx(-37.04, abs=0.005) and e["forward_notional_foreign"] == approx(10.25e6)
    assert e["locked_value_domestic"] == approx(7_725_425)
    p = dr.foreign_equity_currency_hedge(portfolio_value_foreign=10e6, portfolio_beta=1.0, index_futures_price=300_000,
                                         index_futures_beta=0.90, foreign_rate=0.05, years=0.5, spot_rate=0.75,
                                         forward_rate=0.7537, hedge_basis="principal")
    assert p["forward_notional_foreign"] == approx(10e6)


def test_minimum_variance_hedge_and_hedged_return():
    # Hull jet-fuel example: rho 0.928, sigma_S 0.0263, sigma_F 0.0313 -> h 0.78, 37.13 contracts of 42,000 gallons
    r = dr.minimum_variance_hedge_ratio(correlation=0.928, asset_sd=0.0263, hedge_sd=0.0313, exposure=2e6,
                                        contract_size=42_000)
    assert r["hedge_ratio"] == approx(0.7798, abs=5e-5) and r["contracts"] == approx(37.13, abs=0.005)
    assert r["r_squared"] == approx(0.928 ** 2)
    # series form equals the regression slope; perfectly linear series -> slope 2, R^2 1
    s = dr.minimum_variance_hedge_ratio(asset_returns=[0.02, -0.04, 0.06, 0.0], hedge_returns=[0.01, -0.02, 0.03, 0.0])
    assert s["hedge_ratio"] == approx(2.0) and s["r_squared"] == approx(1.0)
    # L3 2013 R35 Example 2 p.288 (Level-3/10-Risk-Management/Curriculum-2013/SS14-R35...): -4.05% -> -1.55%
    assert dr.currency_hedged_return(unhedged_return=-0.0405, hedge_ratio=0.5, futures_return=-0.05)["hedged_return"] \
        == approx(-0.0155)


# ============================================================ L3: loans, FRAs, options, caps/floors
def test_fra_hedged_loan():
    # R29 Exhibit 1 p.324-325: FRA payoff 145,631; borrow 39,854,369; repay 41,448,544; 7.24%
    r = dr.fra_hedged_loan_rate(notional=40e6, fra_rate=0.0525, libor_at_expiry=0.06, spread=0.02, days_underlying=180)
    assert r["fra_payoff"] == approx(145_631, abs=1) and r["amount_borrowed"] == approx(39_854_369, abs=1)
    assert r["repayment"] == approx(41_448_544, abs=1) and r["effective_rate"] == approx(0.0724, abs=5e-5)
    table = {0.02: 0.0728, 0.04: 0.0726, 0.08: 0.0722, 0.10: 0.0720}
    for lib, eff in table.items():
        assert dr.fra_hedged_loan_rate(notional=40e6, fra_rate=0.0525, libor_at_expiry=lib, spread=0.02,
                                       days_underlying=180)["effective_rate"] == approx(eff, abs=5e-5)
    # R29 Example 1 p.325-326: 36,810; 9,963,190; 10,174,908; about 7%
    e = dr.fra_hedged_loan_rate(notional=10e6, fra_rate=0.06, libor_at_expiry=0.075, spread=0.01, days_underlying=90)
    assert e["fra_payoff"] == approx(36_810, abs=1) and e["repayment"] == approx(10_174_908, abs=1)
    assert e["effective_rate"] == approx(0.07, abs=1e-4)


def test_interest_rate_option_loans():
    # R30 Exhibit 13 p.417-420: call premium FV 102,667; effective rate 7.79% for LIBOR >= 5%; table below 5%
    kw = dict(notional=40e6, exercise_rate=0.05, premium=100_000, spread=0.02, days_to_expiry=128, loan_days=180,
              libor_now=0.055)
    r = dr.interest_rate_option_loan(libor_at_expiry=0.08, **kw)
    assert r["premium_future_value"] == approx(102_667, abs=1) and r["option_payoff"] == approx(600_000)
    assert r["effective_annual_rate"] == approx(0.0779, abs=5e-5)
    for lib, eff in ((0.01, 0.0360), (0.03, 0.0568), (0.05, 0.0779), (0.09, 0.0779)):
        assert dr.interest_rate_option_loan(libor_at_expiry=lib, **kw)["effective_annual_rate"] == approx(eff, abs=5e-5)
    # R30 Example 11 p.421-422: premium FV 10,083; LIBOR 4% -> 8.17%
    e = dr.interest_rate_option_loan(notional=5e6, exercise_rate=0.05, premium=10_000, libor_at_expiry=0.04,
                                     spread=0.03, days_to_expiry=36, loan_days=90, libor_now=0.0525)
    assert e["premium_future_value"] == approx(10_083, abs=1) and e["effective_annual_rate"] == approx(0.0817, abs=5e-5)
    # R30 Example 12 p.426-427 (lender, put): premium FV 481,968; 9.34% and 7.24%
    kp = dict(notional=100e6, exercise_rate=0.07, premium=475_000, spread=0.01, days_to_expiry=65, loan_days=182,
              libor_now=0.07125, option_type="put")
    a = dr.interest_rate_option_loan(libor_at_expiry=0.09, **kp)
    assert a["premium_future_value"] == approx(481_968, abs=1) and a["effective_annual_rate"] == approx(0.0934, abs=5e-5)
    b = dr.interest_rate_option_loan(libor_at_expiry=0.05, **kp)
    assert b["option_payoff"] == approx(1_011_111, abs=1) and b["effective_annual_rate"] == approx(0.0724, abs=5e-5)


def test_floating_rate_loan_cap_floor_collar():
    # R30 Example 13 p.430-431 (cap): effective interest 543,750 / 613,542 / 611,979 / 593,750
    cap = dr.floating_rate_loan_cap_floor(notional=25e6, spread=0.025, libor_resets=[0.065, 0.0725, 0.06875, 0.07125],
                                          days=[87, 93, 94, 90], cap_rate=0.07)
    rows = cap["rows"]
    assert [r["interest"] for r in rows] == approx([543_750, 629_688, 611_979, 601_563], abs=1)
    assert [r["caplet_payoff"] for r in rows] == approx([0, 16_146, 0, 7_813], abs=1)
    assert [r["effective_interest"] for r in rows] == approx([543_750, 613_542, 611_979, 593_750], abs=1)
    # R30 Example 14 p.433-434 (floor, lender): 1,328,889 / 1,310,417 / 1,200,000 / 1,339,722
    flo = dr.floating_rate_loan_cap_floor(notional=40e6, spread=0.015, libor_resets=[0.05, 0.04875, 0.0425, 0.05125],
                                          days=[184, 185, 180, 182], floor_rate=0.045)
    assert [r["effective_interest"] for r in flo["rows"]] == approx([1_328_889, 1_310_417, 1_200_000, 1_339_722], abs=1)
    # R30 Example 15 p.437 (collar, borrower): 752,014 / 813,750 / 735,486 / 700,001
    col = dr.floating_rate_loan_cap_floor(notional=35e6, spread=0.02, libor_resets=[0.065, 0.0725, 0.065, 0.05875],
                                          days=[91, 93, 89, 90], cap_rate=0.07, floor_rate=0.06)
    assert [r["effective_interest"] for r in col["rows"]] == approx([752_014, 813_750, 735_486, 700_001], abs=1.5)
    assert col["rows"][1]["caplet_payoff"] == approx(22_604, abs=1)
    assert col["rows"][3]["floorlet_payoff"] == approx(10_938, abs=1)


def test_bond_futures_protective_put():
    r = dr.bond_futures_protective_put(portfolio_value=75e6, portfolio_mdur=6.22, futures_price=82_500,
                                       futures_mdur=8.12, put_strike=80_000, put_premium=1_000, yield_beta=1.05,
                                       ending_portfolio_value=70e6, ending_futures_price=76_000)
    # number of puts = number of futures for a full hedge (R29 Exhibit 2: 731.19)
    assert r["puts"] == approx(731.19, abs=0.005) and r["total_premium"] == approx(731_000)
    assert r["put_payoff"] == approx(731 * 4_000) and r["hedged_value"] == approx(70e6 + 731 * 4_000 - 731_000)
    dy = (82_500 - 80_000) / (82_500 * 8.12)
    assert r["approximate_floor_value"] == approx(75e6 * (1 - 6.22 * 1.05 * dy) - 731_000)


# ============================================================ L3: swaps
def test_swap_duration_and_notional():
    # R31 p.468-469: QAM $500m from 6.75 to 3.50: 1-yr semiannual swap -0.50 -> $3.25bn; 5-yr -3.50 -> $464.29m
    assert dr.swap_duration_approx(maturity_years=1, payments_per_year=2)["swap_duration"] == approx(-0.50)
    assert dr.swap_duration_approx(maturity_years=1, payments_per_year=4)["swap_duration"] == approx(-0.625)
    assert dr.swap_duration_approx(maturity_years=5, payments_per_year=2)["swap_duration"] == approx(-3.50)
    assert dr.swap_notional_for_duration(portfolio_value=500e6, current_mdur=6.75, target_mdur=3.50,
                                         swap_mdur=-0.50)["notional"] == approx(3.25e9)
    n = dr.swap_notional_for_duration(portfolio_value=500e6, current_mdur=6.75, target_mdur=3.50, swap_mdur=-3.50)
    assert n["notional"] == approx(464_290_000, abs=5_000) and n["swap_position"] == "pay_fixed"
    # R31 Example 2 p.469: monthly 1-yr swap -0.708; 2-yr semiannual -1.25 -> $200m
    assert dr.swap_duration_approx(maturity_years=1, payments_per_year=12)["swap_duration"] == approx(-0.708, abs=5e-4)
    d = dr.swap_duration_approx(maturity_years=2, payments_per_year=2)["swap_duration"]
    assert dr.swap_notional_for_duration(portfolio_value=250e6, current_mdur=5.5, target_mdur=4.5,
                                         swap_mdur=d)["notional"] == approx(200e6)
    up = dr.swap_notional_for_duration(portfolio_value=250e6, current_mdur=4.5, target_mdur=5.5, swap_mdur=d)
    assert up["swap_position"] == "receive_fixed"
    assert dr.swap_duration_approx(maturity_years=2, payments_per_year=2, position="receive_fixed")["swap_duration"] \
        == approx(1.25)


def test_swap_converted_rate():
    # R31 p.464-465: floating at LIBOR + 3% swapped at 6.27% -> 9.27% fixed; Example 1 p.467: 6% loan, 7% swap -> L - 1%
    assert dr.swap_converted_rate(swap_fixed_rate=0.0627, loan_spread=0.03)["synthetic_fixed_rate"] == approx(0.0927)
    assert dr.swap_converted_rate(swap_fixed_rate=0.07, fixed_rate=0.06)["spread_over_libor"] == approx(-0.01)


# ============================================================ L3: delta and gamma hedging
def test_delta_hedge():
    # R30 Exhibit 22 p.443: 1,000 short calls, delta 0.5826 -> 583 shares; day 1 delta 0.5966 -> 597, buy 14 for 17,010
    d0 = dr.delta_hedge(option_delta=0.5826, spot=1210, option_position=-1000)
    assert d0["shares_required"] == 583 and d0["cash_flow"] == approx(-583 * 1210)
    d1 = dr.delta_hedge(option_delta=0.5966, spot=1215, option_position=-1000, current_shares=583)
    assert (d1["shares_required"], d1["shares_to_trade"]) == (597, 14) and d1["cash_flow"] == approx(-17_010)
    d2 = dr.delta_hedge(option_delta=0.5479, spot=1198, option_position=-1000, current_shares=597)
    assert d2["shares_to_trade"] == -49 and d2["cash_flow"] == approx(58_702)
    # R30 Example 16 p.444-445: 500 calls, delta 0.5176 -> 259 shares, sell 69 at 122.75
    e = dr.delta_hedge(option_delta=0.5176, spot=122.75, option_position=-500, current_shares=328)
    assert (e["shares_required"], e["shares_to_trade"]) == (259, -69) and e["cash_flow"] == approx(8_469.75)
    # options needed to hedge 1,000 shares with puts of delta -0.4 -> buy 2,500 puts
    assert dr.delta_hedge(option_delta=-0.4, spot=50, share_position=1000)["options_required"] == 2500


def test_delta_gamma_neutral():
    # Hull gamma example: gamma -3,000, option delta 0.62 gamma 1.50 -> buy 2,000 options, sell 1,240 shares
    r = dr.delta_gamma_neutral_hedge(portfolio_delta=0, portfolio_gamma=-3000, option1_delta=0.62, option1_gamma=1.5)
    assert r["option1_units"] == approx(2000) and r["underlying_units"] == approx(-1240)
    # two options, no underlying: both Greeks end at zero
    t = dr.delta_gamma_neutral_hedge(portfolio_delta=250, portfolio_gamma=-3000, option1_delta=0.62,
                                     option1_gamma=1.5, option2_delta=-0.35, option2_gamma=0.9)
    assert t["resulting_delta"] == approx(0, abs=1e-8) and t["resulting_gamma"] == approx(0, abs=1e-8)
    assert t["underlying_units"] == 0


# ============================================================ validation, registry, CLI
def test_input_validation():
    with pytest.raises(ValueError):
        dv.forward_price(spot=-1, rate=0.05, years=1)
    with pytest.raises(ValueError):
        dv.forward_price(spot=100, rate=0.05, years=1, compounding="annual")
    with pytest.raises(ValueError):
        do.binomial_option_price(spot=50, strike=50, up=1.05, down=0.9, rate_per_period=0.07)
    with pytest.raises(ValueError):
        do.put_call_parity(strike=50, rate=0.05, years=1, call=5)
    with pytest.raises(ValueError):
        dv.futures_margin_account(settlement_prices=[100, 99], contracts=1, initial_margin=3, maintenance_margin=5)
    with pytest.raises(ValueError):
        dv.currency_swap_value(spot_rate_initial=1, spot_rate_now=1, period_fraction=0.5,
                               domestic_discount_factors=[0.97], foreign_discount_factors=[0.98],
                               pay_leg="domestic_fixed", receive_leg="domestic_floating")


def test_registry_topic_and_docs():
    mods = {"cfa_calc.derivatives", "cfa_calc.derivatives_options", "cfa_calc.derivatives_risk"}
    ours = {n: f for n, f in REGISTRY.items() if f.func.__module__ in mods}
    assert len(ours) >= 60
    for name, f in ours.items():
        assert f.topic == "derivatives", name
        assert f.formula and f.curriculum, name
        assert f.func.__doc__ and "Example" in f.func.__doc__, name


def test_cli_list_and_call():
    root = Path(__file__).resolve().parent.parent
    out = subprocess.run([sys.executable, str(root / "scripts" / "cfa.py"), "list", "derivatives"],
                         capture_output=True, text=True, check=True).stdout
    assert "bsm_option_price" in out and "futures_margin_account" in out and "swap_notional_for_duration" in out
    res = subprocess.run([sys.executable, str(root / "scripts" / "cfa.py"), "bsm_option_price", "spot=42", "strike=40",
                          "rate=0.10", "volatility=0.20", "years=0.5", "--round", "2"],
                         capture_output=True, text=True, check=True).stdout
    assert '"call_price": 4.76' in res and '"put_price": 0.81' in res


# ============================================================ regressions found by independent verification
@pytest.mark.parametrize("call", [
    lambda: dv.forward_price(spot=100, rate=0.05, years=0.5, income_cash_flows=[[2.0, 0.75]]),
    lambda: dv.forward_value(spot=100, forward_price_initial=101, rate=0.05, years_remaining=0.5,
                             income_cash_flows=[[2.0, 0.75]]),
    lambda: dv.bond_forward_price(bond_price=100, rate=0.05, years=0.5, coupons=[[2.0, 0.75]]),
])
def test_forward_income_after_expiry_rejected(call):
    with pytest.raises(ValueError, match="expir"):
        call()


def test_forward_income_on_expiry_date_is_allowed():
    f = dv.forward_price(spot=100, rate=0.05, years=0.5, income_cash_flows=[[2.0, 0.5]])["forward_price"]
    assert f == approx(100 * 1.05 ** 0.5 - 2.0)


def test_swap_fixed_rate_stub_period_uses_accruals():
    # Hand check, first period 60 days then 90-day periods: B_i = 1/(1 + L_i d_i/360)
    # B = 1/1.005, 1/1.0129167, 1/1.0213333, 1/1.03025 = 0.995025, 0.987248, 0.979112, 0.970638
    # FS = (1 - B4) / (B1 x 60/360 + (B2 + B3 + B4) x 90/360) = 0.029362 / 0.900094 = 0.032621
    r = dv.swap_fixed_rate(spot_rates=[0.03, 0.031, 0.032, 0.033], days=[60, 150, 240, 330])
    assert r["fixed_rate_annual"] == approx(0.032621, abs=5e-7)
    assert r["accrual_fractions"] == approx([60 / 360, 0.25, 0.25, 0.25])
    # a fixed leg paying FS x accrual_i plus par at the end is worth par
    assert sum(r["fixed_rate_annual"] * a * b for a, b in zip(r["accrual_fractions"], r["discount_factors"])) \
        + r["discount_factors"][-1] == approx(1.0)


def test_black_docstrings_state_exact_and_curriculum_values():
    b = do.black_futures_option_price(futures_price=139.19, strike=125, rate=0.0425, volatility=0.15, years=215 / 365)
    assert b["call_price"] == approx(15.23, abs=5e-3)
    assert "15.23" in do.black_futures_option_price.__doc__ and "15.06" in do.black_futures_option_price.__doc__
    p = do.black_interest_rate_option(forward_rate=0.068, strike_rate=0.07, rate=0.0625, volatility=0.02,
                                      years=280 / 365, days_underlying=180, notional=10e6, option_type="put")
    assert p["price"] == approx(9331, abs=0.5)
    assert "9,331" in do.black_interest_rate_option.__doc__ and "9,084" in do.black_interest_rate_option.__doc__
    c = do.black_cap_floor_price(forward_rates=[0.0925], expiries=[90 / 365], discount_rates=[0.096],
                                 volatilities=0.03, strike_rate=0.09, days_underlying=90, notional=1e6)
    assert c["price"] == approx(600.81, abs=5e-3)
    assert "600.81" in do.black_cap_floor_price.__doc__ and "607" in do.black_cap_floor_price.__doc__


def test_currency_forward_price_names_its_sibling():
    doc = dv.currency_forward_price.__doc__
    assert "fx_forward_cip" in doc and "annual compounding" in doc
