"""Fixed income: mortgage-backed securities (L2) and fixed-income portfolio management (L3). Golden values from KB.

KB paths are relative to E:/CLAUDE/CFA/MD.
L2M = Level-2/08-Fixed-Income/Curriculum-2014/SS15-R48-mortgage-backed-sector-of-the-bond-market.md
L3A = Level-3/07-Fixed-Income-Portfolio-Management/Curriculum-2014/SS09-R20-fixed-income-portfolio-management-part-i.md
L3B = Level-3/07-Fixed-Income-Portfolio-Management/Curriculum-2014/SS10-R22-fixed-income-portfolio-management-part-ii.md
"""
import pytest

from cfa_calc import fixed_income as fi

A = pytest.approx


# ------------------------------------------------------------------ MBS
def test_mbs_smm_cpr():
    # L2M p.442-443: SMM 0.005143 <-> CPR 6%
    assert fi.mbs_smm_cpr(cpr=0.06)["smm"] == A(0.005143, abs=5e-7)
    assert fi.mbs_smm_cpr(smm=0.005143)["cpr"] == A(0.06, abs=5e-6)
    with pytest.raises(ValueError):
        fi.mbs_smm_cpr()


@pytest.mark.parametrize("month, psa, cpr, smm", [(5, 100, 0.01, 0.000837), (20, 100, 0.04, 0.003396),
                                                  (31, 100, 0.06, 0.005143), (5, 165, 0.0165, 0.001386),
                                                  (20, 165, 0.066, 0.005674), (31, 165, 0.099, 0.008650)])
def test_mbs_psa_speed(month, psa, cpr, smm):
    # L2M p.443-445
    o = fi.mbs_psa_speed(month, psa)
    assert o["cpr"] == A(cpr) and o["smm"] == A(smm, abs=5e-7)


def test_mbs_mortgage_payment():
    # L2M Exhibit 3 p.446: 400m, WAC 8.125%, 357 months -> 2,975,868; scheduled principal 267,535
    o = fi.mbs_mortgage_payment(400e6, 0.08125, 357)
    assert o["payment"] == A(2_975_868, abs=0.5) and o["principal_first_month"] == A(267_535, abs=0.5)


def test_mbs_cash_flows_exhibit3():
    # L2M Exhibit 3 p.446-447: 400m pass-through, 7.5% net, WAC 8.125%, WAM 357, seasoned 3 months, 100 PSA
    o = fi.mbs_cash_flows(400e6, 0.08125, 0.075, 357, 100, seasoning_months=3)
    s = o["schedule"]
    m1, m27, m100, m357 = s[0], s[26], s[99], s[356]
    assert (m1["net_interest"], m1["scheduled_principal"], m1["prepayment"], m1["cash_flow"]) == \
        A((2_500_000, 267_535, 267_470, 3_035_005), abs=0.5)
    assert m27["beginning_balance"] == A(364_808_016, abs=0.5) and m27["prepayment"] == A(1_874_688, abs=0.5)
    assert m100["beginning_balance"] == A(231_249_776, abs=0.5) and m100["prepayment"] == A(1_187_608, abs=0.5)
    assert m357["beginning_balance"] == A(501_199, abs=0.5) and m357["prepayment"] == A(0, abs=1e-6)
    assert m357["cash_flow"] == A(504_331, abs=0.5) and o["total_principal"] == A(400e6)
    # L2M p.449 average life table (11.66 at 100 PSA, 8.76 at 165 PSA, 2.78 at 700 PSA): the monthly cash flows
    # above match the exhibit to the dollar; the printed WAL table is ~0.01-0.02 years lower (rounding/source)
    for psa, wal in [(100, 11.66), (165, 8.76), (300, 5.63), (700, 2.78)]:
        w = fi.mbs_cash_flows(400e6, 0.08125, 0.075, 357, psa, seasoning_months=3, include_schedule=False)
        assert w["weighted_average_life"] == A(wal, abs=0.03)


def test_mbs_cash_flows_constant_cpr_and_wal_consistency():
    o = fi.mbs_cash_flows(1e6, 0.06, 0.055, 120, cpr=0.10)
    prin = [r["total_principal"] for r in o["schedule"]]
    assert fi.mbs_weighted_average_life(prin)["weighted_average_life"] == A(o["weighted_average_life"])
    assert all(r["smm"] == A(fi.mbs_smm_cpr(cpr=0.10)["smm"]) for r in o["schedule"])


def test_mbs_weighted_average_life():
    assert fi.mbs_weighted_average_life([0] * 11 + [50] + [0] * 23 + [50])["weighted_average_life"] == A(2.0)


def test_mbs_pool_wac_wam():
    # L2M p.439-440: five-loan pool -> WAC 7.28%, WAM 279 months
    o = fi.mbs_pool_wac_wam([125_000, 85_000, 175_000, 110_000, 70_000], [0.075, 0.072, 0.07, 0.078, 0.069],
                            [275, 260, 290, 285, 270])
    assert o["wac"] == A(0.0728, abs=5e-5) and round(o["wam"]) == 279


# ------------------------------------------------------------------ L3 portfolio management
def test_immunization_single_liability():
    # L3A Example 5 p.41: GIC 7.5% sa for 5 years: 9,642,899 grows to 13,934,413
    o = fi.immunization_single_liability(13_934_413, 5, 0.075, 2)
    assert o["required_pv"] == A(9_642_899, abs=1) and o["target_macaulay_duration"] == A(5)
    assert o["target_modified_duration"] == A(5 / 1.0375)
    assert o["liability_convexity"] == A(10 * 11 / 1.0375 ** 2 / 4)  # zero-coupon convexity (L1R eq. 15)


def test_immunization_multiple_liabilities():
    y = 0.05
    liab_t, liab = [3, 7], [50, 50]
    # build a 1y/10y zero barbell with matched PV and Macaulay duration
    L = fi.immunization_multiple_liabilities(liab, liab_t, [1], [1], y, 1)["liabilities"]
    w = fi.barbell_bullet_duration_match(L["macaulay_duration"], 1, 10, 0, 0)["weight_short"]
    a1, a10 = w * L["pv"] * 1.05, (1 - w) * L["pv"] * 1.05 ** 10
    o = fi.immunization_multiple_liabilities(liab, liab_t, [a1, a10], [1, 10], y, 1)
    assert o["pv_condition"] and o["duration_condition"] and o["convexity_condition"] and o["immunized"]
    assert o["surplus"] == A(0, abs=1e-9) and o["money_duration_gap"] == A(0, abs=1e-6)
    # a bullet at 5y cannot bracket the liabilities
    bullet = fi.immunization_multiple_liabilities(liab, liab_t, [L["pv"] * 1.05 ** 5], [5], y, 1)
    assert not bullet["convexity_condition"] and not bullet["immunized"]


def test_contingent_immunization():
    # L3A p.50-51: 500m, safety net 3%, 3 years, immunizable at 4.75% -> 546.72 / 474.90 / cushion 25.10
    o = fi.contingent_immunization(500, 0.03, 3, 0.0475)
    assert o["required_terminal_value"] == A(546.72, abs=5e-3) and o["required_assets_now"] == A(474.90, abs=5e-3)
    assert o["dollar_safety_margin"] == A(25.10, abs=5e-3) and not o["immunize_now"]
    # rates fall to 3.75%: 10y 4.75% notes worth 541.36, required 489.06, margin 52.3
    v = 500 * fi.bond_price(100, 0.0475, 0.0375, 10, 2)["price"] / 100
    d = fi.contingent_immunization(500, 0.03, 3, 0.0375, current_value=v)
    assert v == A(541.36, abs=0.02) and d["required_assets_now"] == A(489.06, abs=5e-3)
    assert d["dollar_safety_margin"] == A(52.3, abs=0.05)
    # rates rise to 5.80%: value 460.55 vs required 460.52 -> margin ~0: immunize
    v2 = 500 * fi.bond_price(100, 0.0475, 0.058, 10, 2)["price"] / 100
    u = fi.contingent_immunization(500, 0.03, 3, 0.058, current_value=v2)
    assert u["required_assets_now"] == A(460.52, abs=0.03) and u["dollar_safety_margin"] == A(0, abs=0.05)
    assert fi.contingent_immunization(500, 0.03, 3, 0.058, current_value=v2 - 1)["immunize_now"]


def test_barbell_bullet_duration_match():
    # 2y and 10y zeros duration-matched to a 6y zero bullet: barbell is more convex (L1R p.522, L3A p.53)
    y = 0.05
    d = {n: fi.bond_duration_convexity(0, y, n, 1) for n in (2, 6, 10)}
    o = fi.barbell_bullet_duration_match(d[6]["modified_duration"], d[2]["modified_duration"],
                                         d[10]["modified_duration"], d[2]["convexity"], d[10]["convexity"],
                                         d[6]["convexity"])
    assert o["weight_short"] == A(0.5) and o["convexity_advantage"] > 0


def test_futures_duration_hedge_contracts():
    # L3B Example 11 p.129-130: GBP50m, 9.52 -> 7.5, CTD D 8.47 at 100,000, CF 1.1 -> sell 131.17
    o = fi.futures_duration_hedge_contracts(50e6, 9.52, 7.5, 8.47, 100_000, 1.1)
    assert o["contracts"] == A(-131.17, abs=5e-3) and o["action"] == "sell" and o["contracts_rounded"] == -131
    # full hedge with yield beta doubles with beta 2
    assert fi.futures_duration_hedge_contracts(1e6, 5, 0, 8, 100_000, 1, 2)["contracts"] == A(-12.5)


def test_bond_expected_return_decomposition():
    o = fi.bond_expected_return_decomposition(100, 4, 0.005, 4.5, 25, 0.002, 0.003, -0.01)
    rc = -4.5 * 0.002 + 0.5 * 25 * 0.002 ** 2
    assert o["rate_change_effect"] == A(rc) and o["rolling_yield"] == A(0.045)
    assert o["total"] == A(0.04 + 0.005 + rc - 0.003 - 0.01)


def test_breakeven_yield_spread_change():
    # L3B Example 20 p.148-149: 300 bp advantage, quarterly, duration 7 -> 10.71 bp
    assert fi.breakeven_yield_spread_change(0.03, 0.25, 7)["breakeven_change_bp"] == A(10.71, abs=5e-3)


def test_leveraged_bond_return():
    # L3B Example 9 p.119-120: 5.75%; Exhibit 20: 3.50% asset return -> 2.25%; p.120: 6.75%
    assert fi.leveraged_bond_return(40e6, 100e6, 0.045, 0.04)["portfolio_return"] == A(0.0575)
    assert fi.leveraged_bond_return(40e6, 100e6, 0.035, 0.04)["portfolio_return"] == A(0.0225)
    assert fi.leveraged_bond_return(40e6, 140e6, 0.065, 0.04)["portfolio_return"] == A(0.1525)
    assert fi.leveraged_bond_return(100, 50, 0.065, 0.06)["portfolio_return"] == A(0.0675)


def test_bond_repo_interest():
    # L3B p.121: 10m overnight at 3% -> repurchase 10,000,833
    o = fi.bond_repo_interest(1e7, 0.03, 1)
    assert o["interest"] == A(833.33, abs=5e-3) and o["repurchase_price"] == A(10_000_833, abs=0.5)
    assert fi.bond_repo_interest(1e7, 0.03, 30, haircut=0.02)["cash_borrowed"] == A(9.8e6)


def test_currency_hedged_bond_return():
    # L3B p.144: euro investor, i_d 3%, i_f 4.5%, spot 0.8000 -> forward 0.7880 (approx IRP)
    assert fi.currency_hedged_bond_return(0, 0.03, 0.045, spot_rate=0.8)["forward_rate_approx"] == A(0.788)
    # Example 17 p.146: local risk premia JGB 0.91% vs Canada 1.86%
    assert fi.currency_hedged_bond_return(0.0216, 0.0, 0.0125)["local_risk_premium"] == A(0.0091)
    assert fi.currency_hedged_bond_return(0.0340, 0.0, 0.0154)["local_risk_premium"] == A(0.0186)
    # Example 18 p.147: US investor in euro bonds: i_d 4%, i_f 3.2%, euro expected +0.6% -> hedge
    o = fi.currency_hedged_bond_return(0.03, 0.04, 0.032, 0.006)
    assert o["forward_premium"] == A(0.008) and o["hedge"] is True


def test_bond_index_tracking_risk():
    # L3A Example 2 p.31-32: tracking risk 0.30350%
    p = [12.8, 6.8, 0.8, -4.6, 4.0, 3.3, 5.4, 5.4, 5.1, 3.7]
    b = [12.6, 6.5, 1.2, -5.0, 4.1, 3.2, 5.1, 5.7, 4.6, 3.8]
    o = fi.bond_index_tracking_risk([x / 100 for x in p], [x / 100 for x in b])
    assert o["tracking_risk"] == A(0.0030350, abs=5e-8) and o["mean_active_return"] == A(0.0009)
