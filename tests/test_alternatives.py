"""Golden tests for cfa_calc.alternatives. Expected values are the printed answers of CFA curriculum worked examples;
paths are relative to E:/CLAUDE/CFA/MD."""
import pytest

from cfa_calc import REGISTRY
from cfa_calc import alternatives as a
from cfa_calc import equity as e

R40 = "Level-2/10-Alternative-Investments/Curriculum-2014/SS13-R40-private-real-estate-investments.md"
R41 = "Level-2/10-Alternative-Investments/Curriculum-2014/SS13-R41-publicly-traded-real-estate-securities.md"
R42 = "Level-2/10-Alternative-Investments/Curriculum-2014/SS13-R42-private-equity-valuation.md"
R44 = "Level-2/10-Alternative-Investments/Curriculum-2014/SS13-R44-a-primer-on-commodity-investing.md"
L1_R66 = "Level-1/10-Alternative-Investments/Curriculum-2013/SS18-R66-introduction-to-alternative-investments.md"
L1_R63 = "Level-1/10-Alternative-Investments/Curriculum-2014/SS18-R63-introduction-to-alternative-investments.md"
L3_R26 = "Level-3/09-Alternative-Investments/Curriculum-2014/SS13-R26-alternative-investments-portfolio-management.md"


def test_all_registered_under_alternatives():
    names = [n for n, f in REGISTRY.items() if f.topic == "alternatives"]
    assert len(names) >= 20
    assert all(REGISTRY[n].func.__module__ == "cfa_calc.alternatives" for n in names)


# ------------------------------------------------------------------ real estate
def test_noi():
    # R40 p.47 Example 12: NOI 281,250
    r = a.real_estate_noi(50 * 1000 * 12, other_income=25_000, vacancy_rate=0.10, opex_pct_of_egi=0.40,
                          management_fee_pct_of_egi=0.10)
    assert r["effective_gross_income"] == pytest.approx(562_500)
    assert r["noi"] == pytest.approx(281_250)


def test_direct_capitalization():
    # R40 p.49 Example 13: 5,000,000; p.52 Example 15: 1,000,000; p.54 Example 17: 5,000,000
    assert a.real_estate_direct_capitalization(250_000, cap_rate=0.05)["value"] == pytest.approx(5_000_000)
    assert a.real_estate_direct_capitalization(100_000, discount_rate=0.12, growth=0.02)["value"] == pytest.approx(1_000_000)
    assert a.real_estate_direct_capitalization(600_000, discount_rate=0.12)["value"] == pytest.approx(5_000_000)


def test_dcf_with_terminal_cap():
    # R40 p.54 Example 16: 1,041,390, implied going-in cap 9.60%
    r = a.real_estate_dcf_value([100_000] * 5, 0.12, terminal_noi=120_000, terminal_growth=0.02)
    assert r["resale_value"] == pytest.approx(1_200_000)
    assert r["value"] == pytest.approx(1_041_390, abs=1)
    assert r["going_in_cap_rate"] == pytest.approx(0.0960, abs=5e-5)
    # consistency: level growth at g with terminal cap r - g reproduces direct capitalization
    g = a.real_estate_dcf_value([100_000 * 1.02 ** t for t in range(5)], 0.12, terminal_growth=0.02)
    assert g["value"] == pytest.approx(1_000_000)


def test_uk_term_reversion_and_layer():
    # R40 p.56 Example 18: 8,917,703; p.58 Example 19: 8,741,664
    r = a.real_estate_term_and_reversion(400_000, 2, 0.04, 450_000, 0.05)
    assert r["term_value"] == pytest.approx(754_438, abs=1)
    assert r["value"] == pytest.approx(8_917_703, abs=1)
    assert a.real_estate_layer_method(400_000, 450_000, 2, 0.05, 0.06)["value"] == pytest.approx(8_741_664, abs=1)


def test_gim():
    r = a.real_estate_gross_income_multiplier(1_000_000, 125_000, subject_gross_income=100_000)
    assert r["gim"] == pytest.approx(8) and r["implied_value"] == pytest.approx(800_000)


def test_loan_sizing():
    # R40 p.80 Example 30: 4,000,000 (LTV and DSCR give the same amount)
    r = a.real_estate_loan_sizing(5_000_000, 400_000, 0.80, 1.25, 0.08)
    assert r["max_loan"] == pytest.approx(4_000_000)
    assert r["loan_by_dscr"] == pytest.approx(4_000_000)
    # R40 p.94 practice problem 9: DSCR binds at 32,183,652
    r = a.real_estate_loan_sizing(48_000_000, 2_775_840, 0.75, 1.5, 0.0575)
    assert r["max_loan"] == pytest.approx(32_183_652, abs=1)
    assert r["binding_constraint"] == "dscr"
    # amortizing loan: the debt service on the max loan meets the DSCR exactly
    r = a.real_estate_loan_sizing(10_000_000, 900_000, 1.0, 1.2, 0.06, amortization_years=25, payments_per_year=12)
    assert r["binding_constraint"] == "dscr"
    assert 900_000 / r["annual_debt_service"] == pytest.approx(1.2)


def test_leveraged_and_unleveraged_returns():
    # R40 p.81-82 Examples 31-33: cash-on-cash 8%, leveraged IRR 21.14%, unleveraged 11.20%
    r = a.real_estate_investment_returns(5_000_000, [400_000] * 5, 6_000_000, loan=4_000_000, interest_rate=0.08)
    assert r["equity_dividend_rate"] == pytest.approx(0.08)
    assert r["leveraged_irr"] == pytest.approx(0.2114, abs=5e-5)
    assert r["unleveraged_irr"] == pytest.approx(0.1120, abs=5e-5)
    assert r["dscr"] == pytest.approx(1.25) and r["ltv"] == pytest.approx(0.8)


# ------------------------------------------------------------------ REITs
def test_reit_nav():
    # R41 p.121 Exhibit 7 (Office Equity REIT): NAVPS 52.33
    r = a.reit_nav_per_share(270_432, 0.07, 55_689, noi_growth=0.015, noncash_rent=7_667, acquisition_adjustment=4_534,
                             other_assets=65_554 + 34_566 + 45_667 + 23_456, debt=1_010_988, other_liabilities=119_886)
    assert r["next_12m_cash_noi"] == pytest.approx(271_308, abs=1)
    assert r["navps"] == pytest.approx(52.33, abs=0.005)
    # R41 p.127 Example 6: 11.38
    assert a.reit_nav_per_share(115, 0.07, 100, debt=505)["navps"] == pytest.approx(11.38, abs=0.005)


def test_reit_ffo_affo():
    # R41 p.126 Exhibit 8: FFO 261,738 (4.70/share), AFFO 184,870 (3.32/share)
    r = a.reit_ffo_affo(160_638, 101_100, noncash_rent=21_103, maintenance_capex_and_leasing=55_765, shares=55_689)
    assert r["ffo"] == pytest.approx(261_738) and r["affo"] == pytest.approx(184_870)
    assert r["ffo_per_share"] == pytest.approx(4.70, abs=0.005)
    assert r["affo_per_share"] == pytest.approx(3.32, abs=0.005)
    # R41 p.135 Exhibit 11 (CRE 2010): FFO/share 3.95, AFFO/share 3.54, payout on AFFO 79.1%
    r = a.reit_ffo_affo(124_454, 115_110, noncash_rent=5_112, maintenance_capex_and_leasing=20_006, shares=60_600,
                        dividends_per_share=2.80)
    assert r["ffo_per_share"] == pytest.approx(3.95, abs=0.005)
    assert r["affo_per_share"] == pytest.approx(3.54, abs=0.005)
    assert r["payout_on_affo"] == pytest.approx(0.791, abs=5e-4)
    # R41 p.128 Example 7: FFO back-solved from AFFO -> 4.08 per share (1m shares)
    ffo = 3_320_000 + 207_430 + 550_750
    assert ffo / 1_000_000 == pytest.approx(4.08, abs=0.005)


def test_reit_relative_value_and_ddm():
    # R41 p.140: 14.2 x 4.23 = 60.07; 16.5 x 3.76 = 62.04
    r = a.reit_relative_value(4.23, 14.2, 3.76, 16.5, price=69.85)
    assert r["value_from_p_ffo"] == pytest.approx(60.07, abs=0.005)
    assert r["value_from_p_affo"] == pytest.approx(62.04, abs=0.005)
    assert r["current_p_ffo"] == pytest.approx(16.5, abs=0.05)
    # R41 p.142 Exhibit 17: two-step DDM, terminal 3.57/(0.08 - 0.05) = 119.00, value 102.71
    r = e.ddm_multistage_value(0.08, [2.98, 3.25, 3.40], terminal_growth=0.05)
    assert r["terminal_value"] == pytest.approx(119.0)
    assert r["value"] == pytest.approx(102.71, abs=0.005)


def test_unsmoothing_round_trip():
    # Geltner (1993) first-order smoothing: R*_t = (1 - a) R_t + a R*_{t-1}; unsmoothing recovers R_t
    true = [0.02, -0.04, 0.06, 0.01, -0.03, 0.05, 0.00, 0.03]
    alpha, obs = 0.6, [0.01]
    for x in true:
        obs.append((1 - alpha) * x + alpha * obs[-1])
    r = a.unsmooth_appraisal_returns(obs, smoothing_weight=alpha)
    assert r["unsmoothed_returns"] == pytest.approx(true)
    assert r["std_unsmoothed"] > r["std_observed"]
    # auto-estimated weight (lag-1 autocorrelation) on a persistent series
    assert 0 < a.unsmooth_appraisal_returns([0.01, 0.02, 0.03, 0.04, 0.05, 0.06])["smoothing_weight"] < 1
    with pytest.raises(ValueError):  # negative autocorrelation: no evidence of smoothing
        a.unsmooth_appraisal_returns([0.05, -0.05, 0.05, -0.05, 0.05])


# ------------------------------------------------------------------ private equity
def test_waterfall_hurdle_blocks_carry():
    # R42 p.176 Example 2: 40 -> 45 after 2 years, IRR 6.1% < 8% hurdle -> no carry
    r = a.pe_distribution_waterfall([40], [45], [2], hurdle_rate=0.08)
    assert r["deals"][0]["irr"] == pytest.approx(0.061, abs=5e-4)
    assert r["gp_carry"] == 0.0


def test_waterfall_methods_and_clawback():
    # R42 p.178 Example 3: deal-by-deal GP 3; total-return alt. 1 (committed 300 first) GP 0; alt. 2 GP 3
    assert a.pe_distribution_waterfall([30], [45])["gp_carry"] == pytest.approx(3)
    assert a.pe_distribution_waterfall([30], [45], method="whole_of_fund", committed_capital=300)["gp_carry"] == 0
    assert a.pe_distribution_waterfall([30], [45], method="whole_of_fund",
                                       whole_fund_basis="threshold")["gp_carry"] == pytest.approx(3)
    # second deal 25 -> 20 at the annual true-up: GP pays back 1
    r = a.pe_distribution_waterfall([30, 25], [45, 20])
    assert r["gp_carry"] == pytest.approx(3) and r["whole_fund_carry"] == pytest.approx(2)
    assert r["clawback"] == pytest.approx(1)


def test_waterfall_catch_up_tiers():
    # textbook tiers: 100 -> 150 in 1 year, 8% pref, full catch-up -> GP 20% of all profit (10), LP 140
    r = a.pe_distribution_waterfall([100], [150], [1], hurdle_rate=0.08, catch_up=1.0)
    assert r["gp_carry"] == pytest.approx(10.0)
    # no catch-up (hard hurdle): GP 20% of profit above the 8 preferred return
    r = a.pe_distribution_waterfall([100], [150], [1], hurdle_rate=0.08, catch_up=0.0)
    assert r["gp_carry"] == pytest.approx(0.2 * 42)
    # partial: proceeds 110 cover pref (8) and 2 of catch-up -> GP 2 (= 20% of 10)
    assert a.pe_distribution_waterfall([100], [110], [1], hurdle_rate=0.08)["gp_carry"] == pytest.approx(2.0)
    # 80% catch-up: GP gets 80% of the tier until it holds 20% of profit; total still 20% when profit is large
    assert a.pe_distribution_waterfall([100], [200], [1], hurdle_rate=0.08, catch_up=0.8)["gp_carry"] == pytest.approx(20)


EVP = dict(called_down=[50, 15, 10, 25, 10, 5], operating_results=[-5, -15, 25, 45, 55, 105],
           distributions=[0, 0, 0, 25, 45, 75], committed_capital=125)


def test_pe_fund_nav_table():
    # R42 p.186-187 (EVP I): fees, NAV, carry, DPI 1.26, RVPI 1.29, TVPI 2.55, gross IRR 16.1%, net IRR 11.3%
    r = a.pe_fund_nav_table(**EVP)
    rows = r["rows"]
    assert [x["management_fee"] for x in rows] == pytest.approx([1.0, 1.3, 1.5, 2.0, 2.2, 2.3])
    assert [x["nav_before_distributions"] for x in rows] == pytest.approx([44.0, 42.7, 76.2, 144.2, 178.2, 234.1], abs=0.05)
    assert [x["carried_interest"] for x in rows] == pytest.approx([0, 0, 0, 3.8, 6.8, 11.2], abs=0.05)
    assert rows[-1]["nav_after_distributions"] == pytest.approx(147.9, abs=0.05)
    assert r["dpi"] == pytest.approx(1.26, abs=0.005)
    assert r["rvpi"] == pytest.approx(1.29, abs=0.005)
    assert r["tvpi"] == pytest.approx(2.55, abs=0.005)
    assert r["gross_irr"] == pytest.approx(0.161, abs=5e-4)
    assert r["net_irr"] == pytest.approx(0.113, abs=5e-4)
    assert r["gross_cash_flows"] == pytest.approx([-50, -20, -25, 0, 35, 50, 105])


def test_pe_multiples_and_irr():
    # R42 p.183 Example 4: DPI 0.07 + RVPI 0.62 = TVPI 0.69
    r = a.pe_fund_performance_multiples(100, 7, 62, committed_capital=125)
    assert r["tvpi"] == pytest.approx(0.69) and r["pic"] == pytest.approx(0.8)
    # R42 p.187: net IRR of the printed cash flows 11.3%
    assert a.pe_fund_irr([-50, -21.0, -26.3, -1.5, 29.2, 41.0, 91.5])["irr"] == pytest.approx(0.113, abs=5e-4)
    # quarterly annualization
    q = a.pe_fund_irr([-100, 0, 0, 0, 110], periods_per_year=4)
    assert q["irr"] == pytest.approx(0.10)


def test_vc_method_single_round():
    # R42 p.190 Appendix (SpiffyCalc): POST 4,938,272, PRE 1,938,272, F 60.75%, 1,547,771 shares, price 1.94
    r = a.vc_method_single_round(3e6, 4, 0.50, 1e6, exit_value=25e6)
    assert r["post_money"] == pytest.approx(4_938_272, abs=1)
    assert r["pre_money"] == pytest.approx(1_938_272, abs=1)
    assert r["ownership_fraction"] == pytest.approx(0.6075)
    assert r["new_shares"] == pytest.approx(1_547_771, abs=1)
    assert r["price_per_share"] == pytest.approx(1.94, abs=0.005)
    # R42 p.191 sensitivities: r 60% -> POST 3,814,697; t 4.4 -> 4,198,928; exit 22.5m -> 4,444,444
    assert a.vc_method_single_round(3e6, 4, 0.60, 1e6, exit_value=25e6)["post_money"] == pytest.approx(3_814_697, abs=1)
    assert a.vc_method_single_round(3e6, 4.4, 0.50, 1e6, exit_value=25e6)["post_money"] == pytest.approx(4_198_928, abs=1)
    assert a.vc_method_single_round(3e6, 4, 0.50, 1e6, exit_metric=2.25e6, exit_multiple=10)["post_money"] == pytest.approx(4_444_444, abs=1)
    # R42 p.194: 20% rate with 20% annual failure probability -> same 4,938,272 (adjusted rate 50%)
    r = a.vc_method_single_round(3e6, 4, 0.20, 1e6, exit_value=25e6, prob_failure=0.20)
    assert r["adjusted_discount_rate"] == pytest.approx(0.50)
    assert r["post_money"] == pytest.approx(4_938_272, abs=1)
    # R42 p.195: scenario-weighted exit value 10m at 20% -> POST 4,822,531, F 62.21%
    r = a.vc_method_single_round(3e6, 4, 0.20, 1e6, exit_value=10e6)
    assert r["post_money"] == pytest.approx(4_822_531, abs=1)
    assert r["ownership_fraction"] == pytest.approx(0.6221, abs=5e-5)
    # negative-NPV deal (R42 p.195: 5m raise needs 101.25%)
    with pytest.raises(ValueError):
        a.vc_method_single_round(5e6, 4, 0.50, 1e6, exit_value=25e6)


def test_vc_method_multiple_rounds_and_retention():
    # R42 p.197 two-round table
    r = a.vc_method_multiple_rounds(25e6, [3e6, 2e6], [2.25, 2.25], 1e6)
    r1, r2 = r["rounds"]
    assert r1["post_money"] == pytest.approx(4_049_383, abs=1) and r1["pre_money"] == pytest.approx(1_049_383, abs=1)
    assert r1["ownership_fraction"] == pytest.approx(0.7409, abs=5e-5)
    assert r1["new_shares"] == pytest.approx(2_858_824, abs=1) and r1["price_per_share"] == pytest.approx(1.05, abs=0.005)
    assert r2["post_money"] == pytest.approx(11_111_111, abs=1) and r2["ownership_fraction"] == pytest.approx(0.18)
    assert r2["new_shares"] == pytest.approx(847_059, abs=1) and r2["price_per_share"] == pytest.approx(2.36, abs=0.005)
    assert r1["final_ownership"] == pytest.approx(0.6075) and r["founders_final_ownership"] == pytest.approx(0.2125)
    # retention-ratio form of round 1 (expected 18% dilution) gives the same answer
    s = a.vc_method_single_round(3e6, 4, 0.50, 1e6, exit_value=25e6, retention_ratio=0.82)
    assert s["ownership_fraction"] == pytest.approx(r1["ownership_fraction"])
    assert s["post_money"] == pytest.approx(r1["post_money"])


# ------------------------------------------------------------------ hedge funds
def test_hedge_fund_fees_awj():
    # L1_R66 p.201-202 Example 2 (AWJ Capital)
    y1 = a.hedge_fund_fees(100, 130)  # Q1: independent fees
    assert y1["total_fees"] == pytest.approx(8.6) and y1["net_return"] == pytest.approx(0.2140)
    q2 = a.hedge_fund_fees(100, 130, incentive_net_of_management_fee=True)
    assert q2["total_fees"] == pytest.approx(8.08) and q2["net_return"] == pytest.approx(0.2192)
    q3 = a.hedge_fund_fees(100, 130, incentive_net_of_management_fee=True, hurdle_rate=0.05)
    assert q3["incentive_fee"] == pytest.approx(4.48) and q3["net_return"] == pytest.approx(0.2292)
    y2 = a.hedge_fund_fees(y1["ending_value_net"], 110, high_water_mark=y1["high_water_mark_end"])
    assert y2["total_fees"] == pytest.approx(2.2) and y2["net_return"] == pytest.approx(-0.1120, abs=5e-5)
    y3 = a.hedge_fund_fees(y2["ending_value_net"], 128, high_water_mark=y2["high_water_mark_end"])
    assert y3["incentive_fee"] == pytest.approx(1.32) and y3["net_return"] == pytest.approx(0.1514, abs=5e-5)
    assert y3["high_water_mark_end"] == pytest.approx(124.12)
    # Q6: geometric mean 7.47%, total fees 14.68
    geo = ((1 + y1["net_return"]) * (1 + y2["net_return"]) * (1 + y3["net_return"])) ** (1 / 3) - 1
    assert geo == pytest.approx(0.0747, abs=5e-5)
    assert y1["total_fees"] + y2["total_fees"] + y3["total_fees"] == pytest.approx(14.68)


def test_soft_hurdle():
    # soft hurdle: once beaten, incentive on the whole gain; hard hurdle: only above it
    soft = a.hedge_fund_fees(100, 110, management_fee_rate=0, hurdle_rate=0.05, hurdle_type="soft")
    hard = a.hedge_fund_fees(100, 110, management_fee_rate=0, hurdle_rate=0.05)
    assert soft["incentive_fee"] == pytest.approx(2.0) and hard["incentive_fee"] == pytest.approx(1.0)
    assert a.hedge_fund_fees(100, 104, management_fee_rate=0, hurdle_rate=0.05, hurdle_type="soft")["incentive_fee"] == 0


def test_fund_of_funds():
    # L1_R63 p.242 Example 3: direct 14%, through the FoF 11.6%
    r = a.fund_of_hedge_funds_net_return(0.20)
    assert r["direct_hf_net_return"] == pytest.approx(0.14)
    assert r["fof_net_return"] == pytest.approx(0.116)


# ------------------------------------------------------------------ commodities
def test_commodity_return_components():
    # L3_R26 p.67 Exhibit 17: GSCI 2000 = 8.6% + 14.2% + 26.9% = 49.7%
    r = a.commodity_futures_total_return(0.269, 0.142, 0.086)
    assert r["total_return"] == pytest.approx(0.497) and r["excess_return"] == pytest.approx(0.411)


@pytest.mark.parametrize("f1,f0,ds,roll", [
    (40.58, 39.10, 0.40, 1.08), (39.67, 38.70, 0.40, 0.57), (38.45, 37.65, 0.40, 0.40),  # L3_R26 p.66 Exhibit 16
    (28.90, 27.90, 0.35, 0.65), (28.55, 27.65, 0.35, 0.55), (27.88, 27.01, 0.35, 0.52),  # L3_R26 p.70 Example 9
])
def test_roll_return_monetary(f1, f0, ds, roll):
    assert a.commodity_roll_return_monetary(f1, f0, ds)["roll_return"] == pytest.approx(roll, abs=0.005)


def test_term_structure_roll_yield():
    # sign convention vs R44 p.256 Eq. 6: roll yield = -(F_new - F_old)/F_old
    b = a.commodity_term_structure(100, 95, spot_price=101)
    assert b["roll_yield"] == pytest.approx(0.05) and b["term_structure"] == "backwardation"
    assert b["basis"] == pytest.approx(1)
    c = a.commodity_term_structure(100, 104)
    assert c["roll_yield"] == pytest.approx(-0.04) and c["term_structure"] == "contango"


# --- dated whole-of-fund waterfall (independent reference: cfa-analyst-workspace/refs2.py) --------------------
def test_pe_waterfall_cash_flows_hurdle_stops_when_capital_returned():
    out = a.pe_waterfall_cash_flows([[0, 100]], [[2, 100], [5, 30]], hurdle_rate=0.08, carry_rate=0.2,
                                      catch_up=1.0, deal_by_deal_gp_carry=10.0)
    assert out["pref_paid"] == pytest.approx(100 * 1.08 ** 2 * 1.08 ** 3 - 100 * 1.08 ** 3, rel=1e-9)  # 16.64 x 1.08^3
    assert out["gp_carry"] == pytest.approx(6.0)          # full catch-up: GP ends with 20% of the 30 profit
    assert out["events"][1]["lp"] == pytest.approx(24.0)
    assert out["lp_net_irr"] == pytest.approx(0.0889680434, abs=1e-8)
    assert out["clawback"] == pytest.approx(4.0)


def test_pe_waterfall_cash_flows_no_hurdle_is_straight_carry():
    out = a.pe_waterfall_cash_flows([[0, 100]], [[3, 160]], hurdle_rate=0.0, carry_rate=0.2, catch_up=0)
    assert out["gp_carry"] == pytest.approx(12.0) and out["lp_distributions"] == pytest.approx(148.0)


def test_pe_waterfall_cash_flows_partial_catch_up():
    # 80% catch-up: GP reaches 20% of profit after the catch-up tier, then 80/20
    out = a.pe_waterfall_cash_flows([[0, 100]], [[1, 200]], hurdle_rate=0.08, carry_rate=0.2, catch_up=0.8)
    assert out["gp_carry"] == pytest.approx(0.2 * 100)


# ---------------------------------------------------------------- regressions found by independent verification

@pytest.mark.parametrize("cu", [0.1, 0.19, 0.2])
def test_waterfall_rejects_catch_up_between_zero_and_carry(cu):
    # 0 < catch_up <= carry gave a non-monotone carry (18.4 at 0, 9.2 at 0.1); reject like pe_waterfall_cash_flows
    with pytest.raises(ValueError, match="catch_up"):
        a.pe_distribution_waterfall([100], [200], [1], hurdle_rate=0.08, catch_up=cu)
    assert a.pe_distribution_waterfall([100], [200], [1], hurdle_rate=0.08, catch_up=0.0)["gp_carry"] == \
        pytest.approx(18.4)


@pytest.mark.parametrize("call", [
    lambda: a.real_estate_term_and_reversion(term_rent=400000, years_to_review=2.5, term_rate=0.04,
                                             estimated_rental_value=450000, reversion_cap_rate=0.05),
    lambda: a.real_estate_loan_sizing(appraised_value=5e6, noi=400000, max_ltv=0.8, min_dscr=1.25,
                                      interest_rate=0.08, amortization_years=25.5),
])
def test_fractional_periods_rejected_not_truncated(call):
    with pytest.raises(ValueError):
        call()
