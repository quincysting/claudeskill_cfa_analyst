"""Tests for cfa_calc.pwm. Golden values from the CFA L3 2014 curriculum (paths relative to E:/CLAUDE/CFA/MD).

R11 = Level-3/03-Private-Wealth-Management/Curriculum-2014/SS04-R11-taxes-and-private-wealth-management-in-a-global-context.md
R12 = Level-3/03-Private-Wealth-Management/Curriculum-2014/SS04-R12-estate-planning-in-a-global-context.md
"""
import pytest

from cfa_calc import pwm

A = pytest.approx


# ---------------------------------------------------------------- tax regimes (R11)
def test_accrual_in_text():  # R11 p.250: EUR100, 6%, 10y, 30% -> 150.90; untaxed 179.08; drag 35.6%
    r = pwm.tax_future_value(initial=100, pretax_return=0.06, years=10, regime="accrual", tax_rate=0.30)
    assert r["future_value"] == A(150.90, abs=0.005)
    assert r["future_value_no_tax"] == A(179.08, abs=0.005)
    assert r["tax_drag_pct"] == A(0.356, abs=0.0005)


def test_accrual_example2():  # R11 p.251 Example 2: EUR100,000, 7%, 20y, 20% -> 297,357; drag 31.23%
    r = pwm.tax_future_value(initial=100000, pretax_return=0.07, years=20, regime="accrual", tax_rate=0.20)
    assert r["future_value"] == A(297357, abs=1)
    assert r["tax_drag_amount"] == A(89611, abs=1)
    assert r["tax_drag_pct"] == A(0.3123, abs=0.00005)


def test_accrual_practice_q2():  # R11 p.280/284 Q2: EUR250,000, 6.5%, 10%, 15y -> 586,547
    r = pwm.tax_future_value(initial=250000, pretax_return=0.065, years=15, regime="accrual", tax_rate=0.10)
    assert r["future_value"] == A(586547, abs=1)
    assert r["accrual_equivalent_return"] == A(0.0585, abs=1e-9)  # Q3: accrual equivalent return 5.85%


def test_deferred_gains():  # R11 p.252-255: 155.36; basis 80% -> 149.36; Example 3 329,575 (drag 20%); Example 4 325,575
    assert pwm.tax_future_value(initial=100, pretax_return=0.06, years=10, regime="deferred",
                                tax_rate=0.30)["future_value"] == A(155.36, abs=0.005)
    assert pwm.tax_future_value(initial=100, pretax_return=0.06, years=10, regime="deferred", tax_rate=0.30,
                                cost_basis_ratio=0.80)["future_value"] == A(149.36, abs=0.005)
    r = pwm.tax_future_value(initial=100000, pretax_return=0.07, years=20, regime="deferred", tax_rate=0.20)
    assert r["future_value"] == A(329575, abs=1)
    assert r["tax_drag_pct"] == A(0.20)  # deferral drag equals t_cg exactly
    assert pwm.tax_future_value(initial=100000, pretax_return=0.07, years=20, regime="deferred", tax_rate=0.20,
                                cost_basis_ratio=0.80)["future_value"] == A(325575, abs=1)


def test_deferred_practice_q4_q5():  # R11 p.280/284: 690,747 (B=1) and 683,247 (B=0.70)
    kw = dict(initial=250000, pretax_return=0.075, years=15, regime="deferred", tax_rate=0.10)
    assert pwm.tax_future_value(**kw)["future_value"] == A(690747, abs=1)
    assert pwm.tax_future_value(cost_basis_ratio=0.70, **kw)["future_value"] == A(683247, abs=1)


def test_accrual_equivalent_example9():  # R11 p.262 Example 9: R_AE 6.1446%, T_AE 12.22%
    r = pwm.tax_future_value(initial=100000, pretax_return=0.07, years=20, regime="deferred", tax_rate=0.20)
    assert r["accrual_equivalent_return"] == A(0.061446, abs=2e-6)
    assert r["accrual_equivalent_tax_rate"] == A(0.1222, abs=5e-5)


def test_wealth_tax():  # R11 p.255: 146.33; Example 5 p.256: 647,844, drag 21.65%; Q12: 1,200,100
    assert pwm.tax_future_value(initial=100, pretax_return=0.06, years=10, regime="wealth",
                                tax_rate=0.02)["future_value"] == A(146.33, abs=0.005)
    r = pwm.tax_future_value(initial=400000, pretax_return=0.06, years=10, regime="wealth", tax_rate=0.01)
    assert r["future_value"] == A(647844, abs=1)
    assert r["tax_drag_pct"] == A(0.2165, abs=0.00005)
    assert pwm.tax_future_value(initial=500000, pretax_return=0.05, years=20, regime="wealth",
                                tax_rate=0.005)["future_value"] == A(1200100, abs=1)


BLEND = dict(p_interest=0.05, p_dividend=0.25, p_realized_gain=0.45, t_interest=0.35, t_dividend=0.15,
             t_capital_gain=0.15)


def test_blended_example8():  # R11 p.258-261 Examples 7-8: r* 7.02%, T* 4.27%, FV 138,662, R_AE 6.756%, T_AE 15.55%
    r = pwm.tax_future_value(initial=100000, pretax_return=0.08, years=5, regime="blended", **BLEND)
    assert r["r_star"] == A(0.0702)
    assert r["T_star"] == A(0.0427, abs=5e-5)
    assert r["future_value"] == A(138662, abs=2)  # book rounds T* to 4.27% before compounding
    assert r["accrual_equivalent_return"] == A(0.06756, abs=5e-6)
    assert r["accrual_equivalent_tax_rate"] == A(0.1555, abs=1e-4)  # book derives it from rounded R_AE


def test_blended_practice_q6_q8_q9():  # R11 p.280-285: Q6 678,158; Q8 1,962,776 (r* .0471, T* .0318); Q9 (B=.70) 1,887,776
    q6 = pwm.tax_future_value(initial=250000, pretax_return=0.075, years=15, regime="blended", p_realized_gain=0.5,
                              t_capital_gain=0.10)
    assert q6["future_value"] == A(678158, abs=1)
    kw = dict(initial=1_000_000, pretax_return=0.06, years=15, regime="blended", p_interest=0.20, p_dividend=0.30,
              p_realized_gain=0.40, t_interest=0.35, t_dividend=0.15, t_capital_gain=0.25)
    q8 = pwm.tax_future_value(**kw)
    assert q8["r_star"] == A(0.0471, abs=5e-5) and q8["T_star"] == A(0.0318, abs=5e-5)
    assert q8["future_value"] == A(1962776, abs=2)
    assert pwm.tax_future_value(cost_basis_ratio=0.70, **kw)["future_value"] == A(1887776, abs=2)


def test_blended_special_cases_match_simple_regimes():  # R11 p.259: p_i = 1 -> accrual; all deferred -> deferred
    acc = pwm.tax_future_value(pretax_return=0.07, years=12, regime="accrual", tax_rate=0.3)["future_value"]
    assert pwm.tax_future_value(pretax_return=0.07, years=12, regime="blended", p_interest=1,
                                t_interest=0.3)["future_value"] == A(acc)
    dfr = pwm.tax_future_value(pretax_return=0.07, years=12, regime="deferred", tax_rate=0.2)["future_value"]
    assert pwm.tax_future_value(pretax_return=0.07, years=12, regime="blended",
                                t_capital_gain=0.2)["future_value"] == A(dfr)


def test_tda_vs_tax_exempt():  # R11 p.264 Example 10; p.266 Example 11; Q7 p.281
    assert pwm.tax_future_value(initial=100000, pretax_return=0.07, years=20, regime="tax_deferred_account",
                                tax_rate=0.20)["future_value"] == A(309575, abs=1)
    assert pwm.tax_future_value(initial=100000, pretax_return=0.07, years=20,
                                regime="tax_exempt")["future_value"] == A(386968, abs=1)
    assert pwm.tax_future_value(initial=4000, pretax_return=0.06, years=30, regime="tax_deferred_account",
                                tax_rate=0.25)["future_value"] == A(17230, abs=1)
    assert pwm.tax_future_value(initial=3000, pretax_return=0.06, years=30,
                                regime="tax_exempt")["future_value"] == A(17230, abs=1)
    assert pwm.tax_future_value(initial=3000, pretax_return=0.06, years=30, regime="accrual",
                                tax_rate=0.25)["future_value"] == A(11236, abs=1)
    assert pwm.tax_future_value(initial=10000, pretax_return=0.075, years=15, regime="tax_deferred_account",
                                tax_rate=0.20)["future_value"] == A(23671, abs=1)


def test_tax_loss_harvesting():  # R11 p.272-275 Examples 12 and 14
    assert pwm.tax_loss_harvesting(market_value=40000, cost_basis=100000, tax_rate=0.20,
                                   other_realized_gains=100000)["tax_savings_now"] == A(12000)
    r = pwm.tax_loss_harvesting(market_value=110000, cost_basis=170000, tax_rate=0.20, growth_multiple=1.8181)
    assert r["tax_savings_now"] == A(12000)
    assert r["harvest_value"] == A(221808, abs=1)
    assert r["harvest_tax"] == A(19962, abs=1)
    assert r["harvest_after_tax"] == A(201846, abs=1)
    assert r["hold_after_tax"] == A(194000, rel=1e-4)  # book rounds 110,000 x 1.8181 to 200,000


def test_after_tax_asset_allocation():  # R11 p.265 Exhibit 6; Q16 p.282/286 (60/40)
    r = pwm.after_tax_asset_allocation(pretax_values=[1_500_000, 500_000], tax_rates=[0.40, 0.0])
    assert r["after_tax_values"][0] == A(900000)
    assert r["after_tax_weights"] == A([0.643, 0.357], abs=0.0005)
    assert pwm.after_tax_asset_allocation(pretax_values=[200000, 80000],
                                          tax_rates=[0.40, 0.0])["after_tax_weights"] == A([0.6, 0.4])


def test_return_conversions():  # R11 p.276: 8% after tax at 40% needs 13.33% pretax; Example 15: 7%/(1-0.5) = 14%
    assert pwm.return_conversions(after_tax_return=0.08, tax_rate=0.40)["pretax_return"] == A(0.13333, abs=5e-6)
    assert pwm.return_conversions(after_tax_return=0.07, tax_rate=0.50)["pretax_return"] == A(0.14)
    assert pwm.return_conversions(pretax_return=0.12, tax_rate=0.50)["after_tax_return"] == A(0.06)


# ---------------------------------------------------------------- estate planning (R12)
def test_core_capital_pearsall():  # R12 practice Q2, p.330/332-333: total 328,599
    r = pwm.estate_core_capital(spending=75000, discount_rate=1.10 / 1.06 - 1,
                                survival_probs=[0.9660, 0.9371, 0.9152, 0.8883, 0.8544],
                                survival_probs_2=[0.8235, 0.7996, 0.7727, 0.7208, 0.6919], total_assets=1_000_000)
    assert r["joint_survival_probs"] == A([0.9940, 0.9874, 0.9807, 0.9688, 0.9551], abs=5e-5)
    assert r["discounted"] == A([71839, 68767, 65817, 62654, 59522], abs=4)
    assert r["core_capital"] == A(328599, abs=5)
    assert r["excess_capital"] == A(1_000_000 - r["core_capital"])


def test_core_capital_webster_first_years():  # R12 p.298-299: year-1 joint 0.9989, expected 499,457, discounted 489,664
    # (book table uses unrounded mortality probabilities; printed 4-decimal inputs give values ~2 lower)
    r = pwm.estate_core_capital(spending=500000, spending_growth=0.03, discount_rate=0.02,
                                survival_probs=[0.9355], survival_probs_2=[0.9831])
    assert r["joint_survival_probs"][0] == A(0.9989, abs=5e-5)
    assert r["expected_spending"][0] == A(499457, abs=3)
    assert r["discounted"][0] == A(489664, abs=3)


def test_gift_vs_bequest():  # R12 p.312-316 Examples 6-7; Japan p.309; Halls Q3 p.334
    base = dict(years=20, estate_tax_rate=0.60, gift_return=0.06, gift_investment_tax=0.20, bequest_return=0.06,
                bequest_investment_tax=0.48)
    assert pwm.gift_vs_bequest_relative_value(**base)["relative_value"] == A(3.45, abs=0.005)
    assert pwm.gift_vs_bequest_relative_value(gift_type="recipient_pays", gift_tax_rate=0.30,
                                              **base)["relative_value"] == A(2.42, abs=0.005)
    ch = dict(base, gift_investment_tax=0.0)
    assert pwm.gift_vs_bequest_relative_value(gift_type="charitable", donor_income_tax_rate=0.48,
                                              **ch)["relative_value"] == A(4.82, abs=0.005)
    assert pwm.gift_vs_bequest_relative_value(years=10, estate_tax_rate=0.5, gift_return=0.08, gift_investment_tax=0.2,
                                              bequest_investment_tax=0.5, gift_type="recipient_pays",
                                              gift_tax_rate=0.5)["relative_value"] == A(1.256, abs=0.0005)
    assert pwm.gift_vs_bequest_relative_value(years=5, estate_tax_rate=0.30, gift_return=0.10, gift_investment_tax=0.25,
                                              bequest_investment_tax=0.30, gift_type="recipient_pays",
                                              gift_tax_rate=0.30)["relative_value"] == A(1.02, abs=0.005)
    # equal returns and taxes: tax-free gift RV = 1/(1 - Te) = 1.67 at 40% (R12 p.307)
    assert pwm.gift_vs_bequest_relative_value(years=7, estate_tax_rate=0.4, gift_return=0.05,
                                              gift_investment_tax=0.3)["relative_value"] == A(1 / 0.6)


def test_donor_pays_credit():  # R12 Eq.7 p.311: numerator factor (1 - Tg + Tg Te)
    r = pwm.gift_vs_bequest_relative_value(years=1, estate_tax_rate=0.45, gift_return=0.0, gift_investment_tax=0.0,
                                           gift_type="donor_pays", gift_tax_rate=0.45)
    assert r["relative_value"] == A((1 - 0.45 + 0.45 * 0.45) / (1 - 0.45))


def test_generation_skipping():  # R12 p.313 Kawaguchi: 275,800,768 vs 551,601,537
    r = pwm.generation_skipping_value(amount=200_000_000, real_return=0.05, years_to_first_transfer=10,
                                      years_to_second_transfer=35, tax_rate_first=0.50)
    assert r["value_two_transfers"] == A(275_800_768, abs=1)
    assert r["value_skipped"] == A(551_601_537, abs=1)
    assert r["relative_value"] == A(2.0)


def test_double_taxation():  # R12 p.324-327: 50%/40% -> 50/40/70; Yankevich deduction 0.65
    r = pwm.double_taxation_rate(residence_rate=0.50, source_rate=0.40)
    assert (r["credit_method"], r["exemption_method"], r["deduction_method"]) == A((0.50, 0.40, 0.70))
    assert pwm.double_taxation_rate(residence_rate=0.50, source_rate=0.30)["deduction_method"] == A(0.65)


def test_validation():
    with pytest.raises(ValueError):
        pwm.tax_future_value(pretax_return=0.05, years=10, regime="bogus")
    with pytest.raises(ValueError):
        pwm.ips_required_return(spending_need=1)
