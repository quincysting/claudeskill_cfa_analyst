"""Fixed income term structure, spreads and interest-rate risk. Golden values from the CFA curriculum KB.

KB paths are relative to E:/CLAUDE/CFA/MD.
L1V = Level-1/08-Fixed-Income/Curriculum-2014/SS15-R54-introduction-to-fixed-income-valuation.md
L1R = Level-1/08-Fixed-Income/Curriculum-2014/SS16-R55-understanding-fixed-income-risk-and-return.md
L1C = Level-1/08-Fixed-Income/Curriculum-2014/SS16-R56-fundamentals-of-credit-analysis.md
L2T = Level-2/08-Fixed-Income/Curriculum-2014/SS14-R46-term-structure-and-volatility-of-interest-rates.md
L2E = Level-2/08-Fixed-Income/Curriculum-2014/SS14-R47-valuing-bonds-with-embedded-options.md
L3A = Level-3/07-Fixed-Income-Portfolio-Management/Curriculum-2014/SS09-R20-fixed-income-portfolio-management-part-i.md
L3B = Level-3/07-Fixed-Income-Portfolio-Management/Curriculum-2014/SS10-R22-fixed-income-portfolio-management-part-ii.md
"""
import pytest

from cfa_calc import fixed_income as fi

A = pytest.approx


def test_fi_spot_bootstrap_from_par():
    # L2E p.373-374: par 3.5/4.2/4.7/5.2% -> spots 3.5000/4.2148/4.7352/5.2706%, forwards 3.5/4.935/5.784/6.893%
    o = fi.fi_spot_bootstrap([0.035, 0.042, 0.047, 0.052])
    assert o["spot_rates"] == A([0.035, 0.042148, 0.047352, 0.052706], abs=5e-7)
    assert o["forward_rates"] == A([0.035, 0.04935, 0.05784, 0.06893], abs=5e-6)


def test_fi_spot_bootstrap_from_prices_round_trip():
    spots = [0.02, 0.025, 0.03]
    prices = [fi.bond_price_spot_curve(spots[:k], c)["price"] for k, c in [(1, 0.01), (2, 0.04), (3, 0.06)]]
    assert fi.fi_spot_bootstrap([0.01, 0.04, 0.06], prices)["spot_rates"] == A(spots, abs=1e-12)


def test_fi_par_curve():
    # L1V p.450: spots 5.263/5.616/6.359/7.008% -> par 5.263/5.606/6.306/6.899%
    o = fi.fi_par_curve([0.05263, 0.05616, 0.06359, 0.07008])
    assert o["par_rates"] == A([0.05263, 0.05606, 0.06306, 0.06899], abs=5e-6)
    # par curve -> bootstrap -> spots round trip
    assert fi.fi_spot_bootstrap(o["par_rates"])["spot_rates"] == A([0.05263, 0.05616, 0.06359, 0.07008])


def test_fi_forward_rate():
    # L1V p.451: 3y1y = 5.778%; Example 11 p.452-453: 1y1y 3.419%, 2y1y 2.707% (semiannual bond basis)
    assert fi.fi_forward_rate(0.0365, 3, 0.0418, 4, 2)["forward_rate"] == A(0.05778, abs=5e-6)
    assert fi.fi_forward_rate(0.02548, 1, 0.02983, 2, 2)["forward_rate"] == A(0.03419, abs=5e-6)
    assert fi.fi_forward_rate(0.02983, 2, 0.02891, 3, 2)["forward_rate"] == A(0.02707, abs=5e-6)
    # L1V p.450: 5y zero at 81 -> 4.2592% (curriculum rounds r per period); 2y5y forward priced 75 -> 5.8372%
    assert fi.bond_ytm(81, 0, 5, 2)["ytm"] == A(0.042592, abs=2e-6)
    assert fi.bond_ytm(75, 0, 5, 2)["ytm"] == A(0.058372, abs=2e-6)


def test_fi_forward_curve_and_spot_from_forwards():
    assert fi.fi_forward_curve([0.035, 0.042148, 0.047352, 0.052706])["forward_rates"] == \
        A([0.035, 0.04935, 0.05784, 0.06893], abs=5e-6)
    # L1V p.453-454: forwards 1.88/2.77/3.54/4.12% -> spots 2.3240/2.7278/3.0741%, bond 3.75% -> 102.637
    s = fi.fi_spot_from_forwards([0.0188, 0.0277, 0.0354, 0.0412])["spot_rates"]
    assert s[1:] == A([0.023240, 0.027278, 0.030741], abs=5e-7)
    assert fi.bond_price_spot_curve(s, 0.0375)["price"] == A(102.637, abs=5e-4)


def test_bond_g_i_spread():
    # L1V Example 12 p.458: G-spread 5.932% - 3.605% = 232.7 bp
    ycorp = fi.bond_ytm(100.125, 0.06, 2, 1)["ytm"]
    ygov = fi.bond_ytm(100.75, 0.04, 2, 1)["ytm"]
    assert fi.bond_g_i_spread(ycorp, ygov)["spread_bp"] == A(232.7, abs=0.1)
    # L1V p.432: 2.36% vs interpolated 3y 0.75% / 5y 1.45% govt -> 126 bp
    o = fi.bond_g_i_spread(0.0236, bond_years=4, benchmark_years=[3, 5], benchmark_yields=[0.0075, 0.0145])
    assert o["benchmark_yield"] == A(0.011) and o["spread_bp"] == A(126)


def test_bond_z_spread():
    # L1V Example 12 p.458-459: Z = 234.22 bp
    assert fi.bond_z_spread(100.125, [0.021, 0.03635], 0.06)["z_spread_bp"] == A(234.22, abs=5e-3)


def test_swap_ted_ois_spreads():
    # L2T p.325: 7y swap 6% vs Treasury 5.4% -> 60 bp; euro swap 6% vs Bund 5% -> 100 bp
    assert fi.swap_ted_ois_spreads(swap_rate=0.06, treasury_yield=0.054)["swap_spread_bp"] == A(60)
    o = fi.swap_ted_ois_spreads(libor=0.0050, tbill_yield=0.0010, ois_rate=0.0035)
    assert o["ted_spread_bp"] == A(40) and o["libor_ois_spread_bp"] == A(15) and o["swap_spread"] is None
    with pytest.raises(ValueError):
        fi.swap_ted_ois_spreads()


def test_bond_rolldown_return():
    # consistency: unchanged yield -> rolldown = pull-to-par only; rolled-down yield prices with bond_price
    o = fi.bond_rolldown_return(0.03, 5, 0.03, 1, 0.025, 1)
    p1 = fi.bond_price(100, 0.03, 0.025, 4, 1)["price"]
    assert o["price_begin"] == A(100) and o["price_end"] == A(p1)
    assert o["rolldown_return"] == A((p1 - 100) / 100) and o["holding_period_return"] == A((p1 + 3 - 100) / 100)
    assert o["rolling_yield"] == A(0.03 + (p1 - 100) / 100)


def test_yield_volatility_historical():
    # L2T Exhibit 8 p.340-342: daily sd 0.6360493%, annualized 10.06% (250), 10.26% (260), 12.15% (365)
    y = [6.6945, 6.699, 6.710, 6.675, 6.555, 6.583, 6.569, 6.583, 6.555, 6.593, 6.620, 6.568, 6.575, 6.646,
         6.607, 6.612, 6.575, 6.552, 6.515, 6.533, 6.543, 6.559, 6.500, 6.546, 6.589, 6.539]
    o = fi.yield_volatility_historical(y)
    assert o["daily_std"] == A(0.006360493, abs=5e-8) and o["annual_std"] == A(0.1006, abs=5e-5)
    assert fi.yield_volatility_historical(y, 365)["annual_std"] == A(0.1215, abs=5e-5)


def test_bond_duration_convexity_annual():
    # L1R Exhibit 2 p.497-500: 10y 8% annual at 10.40% -> MacDur 7.0029, ModDur 6.3432
    o = fi.bond_duration_convexity(0.08, 0.104, 10, 1)
    assert o["full_price"] == A(85.503075, abs=1e-6)
    assert o["macaulay_duration"] == A(7.0029, abs=5e-5) and o["modified_duration"] == A(6.3432, abs=5e-5)
    assert o["approx_modified_duration"] == A(o["modified_duration"], rel=1e-6)
    assert o["approx_convexity"] == A(o["convexity"], rel=1e-5)


def test_bond_duration_convexity_between_coupons():
    # L1R Exhibit 3 p.498-500 and §3.1 p.501-502: 6% sa, 57/180 into period, 16 periods, ytm 6%
    o = fi.bond_duration_convexity(0.06, 0.06, 8, 2, fraction=57 / 180, dy=0.0005)
    assert o["full_price"] == A(100.940423, abs=1e-6)
    assert o["macaulay_duration"] == A(6.310634, abs=1e-6) and o["modified_duration"] == A(6.126829, abs=1e-6)
    assert o["pv_plus"] == A(100.631781, abs=1e-6) and o["pv_minus"] == A(101.250227, abs=1e-6)
    assert o["approx_modified_duration"] == A(6.126842, abs=5e-6)
    # money duration p.514-515: HKD100m position -> 618,441,784 (with ModDur rounded to 6.1268)
    assert o["money_duration"] * 1e6 == A(618_441_784, rel=1e-5)


def test_bond_duration_convexity_zero_and_examples():
    # L1R p.518: long zero, N=60, t/T=24/184, ytm 2.961% -> ModDur 29.498, convexity 884.7
    z = fi.bond_duration_convexity(0, 0.02961, 30, 2, fraction=24 / 184)
    assert z["modified_duration"] == A(29.498, abs=5e-4) and z["convexity"] == A(884.7, abs=0.05)
    assert z["full_price"] == A(41.483617, abs=1e-6)
    # Example 8 p.502-503: 3.75% UST, 54 periods, 61/184, 5.14% -> ApproxModDur 15.368, ApproxMacDur 15.763
    t = fi.bond_duration_convexity(0.0375, 0.0514, 27, 2, fraction=61 / 184, dy=0.0005)
    assert t["full_price"] == A(80.501507, abs=1e-6)
    assert t["approx_modified_duration"] == A(15.368, abs=5e-4) and t["approx_macaulay_duration"] == A(15.763, abs=5e-4)
    # Example 13 p.520-521: 7.25% annual, 15y, 83/360, 7.44% -> full 99.956780, ApproxModDur 8.6907
    e = fi.bond_duration_convexity(0.0725, 0.0744, 15, 1, fraction=83 / 360)
    assert e["full_price"] == A(99.956780, abs=1e-6) and e["approx_modified_duration"] == A(8.6907, abs=5e-5)
    # Example 14 p.522-523: 30y and 100y 4% sa at par, 5 bp -> 17.381 / 420.80 and 24.527 / 1,132.88
    b30 = fi.bond_duration_convexity(0.04, 0.04, 30, 2, dy=0.0005)
    b100 = fi.bond_duration_convexity(0.04, 0.04, 100, 2, dy=0.0005)
    assert b30["approx_modified_duration"] == A(17.381, abs=5e-4) and b30["approx_convexity"] == A(420.80, abs=0.05)
    assert b100["approx_modified_duration"] == A(24.527, abs=5e-4) and b100["approx_convexity"] == A(1132.88, abs=0.05)
    # PVBP p.515: 0.625% UST, 10 periods, 22/183, 0.723368% -> 0.04831
    p = fi.bond_duration_convexity(0.00625, 0.00723368, 5, 2, fraction=22 / 183)
    assert p["pvbp"] == A(0.04831, abs=5e-6)


def test_fi_approx_duration_convexity():
    # L1R p.503 and p.524: callable bond -> EffDur 7.6006, EffCon -285.17
    o = fi.fi_approx_duration_convexity(101.060489, 102.890738, 99.050120, 0.0025)
    assert o["duration"] == A(7.6006, abs=5e-5) and o["convexity"] == A(-285.17, abs=5e-3)
    # printed (6-dp) prices reproduce the curriculum's approximate convexities
    assert fi.fi_approx_duration_convexity(100.940423, 101.250227, 100.631781, 0.0005)["convexity"] == A(46.047, abs=1e-3)
    assert fi.fi_approx_duration_convexity(99.956780, 100.043703, 99.869964, 0.0001)["convexity"] == A(107.046, abs=0.01)
    assert fi.fi_approx_duration_convexity(41.483617, 41.606169, 41.361431, 0.0001)["convexity"] == A(882.3, abs=0.05)


def test_bond_price_change_estimate():
    # L1R p.518-519: -10 bp on the long zero -> +2.9940%
    assert fi.bond_price_change_estimate(29.498, -0.001, 884.7)["pct_change"] == A(0.029940, abs=5e-7)
    # Example 13 p.521: +100 bp -> -8.1555%; Example 15 p.526: A -0.9262%, B -0.8669%, C -1.2311%
    assert fi.bond_price_change_estimate(8.6907, 0.01, 107.046)["pct_change"] == A(-0.081555, abs=5e-7)
    for d, c, dy, exp in [(3.72, 12.1, 0.0025, -0.009262), (5.81, 40.7, 0.0015, -0.008669),
                          (12.39, 158.0, 0.0010, -0.012311)]:
        assert fi.bond_price_change_estimate(d, dy, c)["pct_change"] == A(exp, abs=5e-7)
    # money convexity p.521-522: HKD100,940,423 position, +100 bp -> -5,952,018
    assert fi.bond_price_change_estimate(6.1268, 0.01, 46.047, 100_940_423)["money_change"] == A(-5_952_018, abs=1)
    # L1C Exhibit 17 p.590 (BT 2020s): D 6.084, C 47.4: -100 bp -> +6.32%, +75 bp -> -4.43%
    # (the text example "-4.765%" on p.589-590 drops the 1/2; the exhibit applies it)
    assert fi.bond_price_change_estimate(6.084, -0.01, 47.4)["pct_change"] == A(0.0632, abs=5e-5)
    assert fi.bond_price_change_estimate(6.084, 0.0075, 47.4)["pct_change"] == A(-0.0443, abs=5e-5)


def test_bond_portfolio_duration():
    # L1R p.513: 1y and 30y zeros 50/50 -> average ModDur 14.3725; Example 11 p.513-514 -> 6.0495
    assert fi.bond_portfolio_duration([0.980, 27.765], weights=[0.5, 0.5])["duration"] == A(14.3725)
    mods = [4.761 / (1 + 0.0910 / 2), 5.633 / (1 + 0.0938 / 2), 7.652 / (1 + 0.0962 / 2)]
    o = fi.bond_portfolio_duration(mods, [24_886_343, 27_243_887, 44_306_787])
    assert o["duration"] == A(6.0495, abs=5e-5)
    assert fi.bond_price_change_estimate(o["duration"], 0.002)["pct_change"] == A(-0.0121, abs=5e-5)
    # L3A Exhibit 5 p.33: portfolio spread duration 3.43 (Treasuries 0)
    w = [0.2260, 0.0680, 0.0620, 0.2006, 0.0552, 0.0661, 0.3221, 0, 0]
    sd = [0, 6.45, 2.87, 11.04, 2.20, 1.92, 1.10, 0, 0]
    assert fi.bond_portfolio_duration(sd, weights=w)["duration"] == A(3.43, abs=5e-3)


def test_bond_money_duration():
    # L1R Example 12 p.516: ModDur x 99.650 = 242.62; per bp 0.024262; USD10m position -> 2,426.20
    md = 2.4988 / (1 + 0.052617 / 2)
    assert fi.bond_money_duration(md, 99.650)["money_duration"] == A(242.62, abs=5e-3)
    assert fi.bond_money_duration(md, 9_965_000)["pvbp"] == A(2426.20, abs=0.05)
    # L3A Exhibit 9 p.44: 5.025 x 1,065,613 x 0.01 = 53,548
    assert fi.bond_money_duration(5.025, 1_065_613)["dollar_duration_100bp"] == A(53_548, abs=1)


def test_dollar_duration_rebalance_and_swap():
    # L3A Example 7 p.45: 111,945 / 82,579 = 1.356; cash ~1,079,012 (curriculum uses the rounded 0.356)
    o = fi.dollar_duration_rebalance(111_945, 82_579, 1_023_704 + 1_004_770 + 1_002_458)
    assert round(o["rebalancing_ratio"], 3) == 1.356 and o["cash_required"] == A(1_079_012, rel=2e-3)
    # L3B Example 10 p.124: sell 5.5m at D=4; buy D=5 at 90 -> MV 4.4m, par 4.889m
    s = fi.duration_neutral_swap(5.5e6, 4, 5, 90)
    assert s["new_market_value"] == A(4.4e6) and s["new_par_value"] == A(4.889e6, abs=500)


def test_duration_gap_horizon():
    # L1R Example 16 p.530-531: ApproxMacDur 8.1390, 10-year horizon -> gap -1.8610 (risk: lower rates)
    o = fi.duration_gap_horizon(7.5361 * 1.08, 10)
    assert o["duration_gap"] == A(-1.861, abs=5e-4) and "lower rates" in o["risk"]


def test_asset_liability_duration_gap():
    # L3B p.121: A 140 (D 4), L 100 (D 1) -> equity duration 11.50
    assert fi.asset_liability_duration_gap(140, 100, 4, 1)["equity_duration"] == A(11.5)
    # L3A Exhibits 15-16 p.51-52: A 500 (D 4), L 400 (D 7): +100 bp -> 480 / 372 / 108; -100 bp -> 520 / 428 / 92
    up = fi.asset_liability_duration_gap(500, 400, 4, 7, 0.01)
    dn = fi.asset_liability_duration_gap(500, 400, 4, 7, -0.01)
    assert (up["assets_new"], up["liabilities_new"], up["surplus_new"]) == A((480, 372, 108))
    assert (dn["assets_new"], dn["liabilities_new"], dn["surplus_new"]) == A((520, 428, 92))


def test_key_rate_durations_spot():
    spots = [0.03, 0.035, 0.04, 0.045]
    z = fi.key_rate_durations_spot(spots, cash_flows=[0, 0, 0, 100])
    assert z["key_rate_durations"][:3] == A([0, 0, 0]) and z["key_rate_durations"][3] == A(4 / 1.045, rel=1e-6)
    c = fi.key_rate_durations_spot(spots, 0.05)
    up = fi.bond_price_spot_curve([s + 1e-4 for s in spots], 0.05)["price"]
    dn = fi.bond_price_spot_curve([s - 1e-4 for s in spots], 0.05)["price"]
    assert c["effective_duration"] == A((dn - up) / (2e-4 * c["price"]), rel=1e-6)


def test_key_rate_price_change():
    # L2T p.336-337 (Ho's example): Portfolio I (50% 2y, 50% 30y): scenario 1 +1.6%, scenario 2 +1.4%,
    # scenario 3 -1.4%; Portfolio II (16y) scenario 2 0%
    krd = [[2, 0, 0], [0, 0, 30]]
    assert fi.key_rate_price_change(krd, [-0.001] * 3, [0.5, 0.5])["pct_change"] == A(0.016)
    assert fi.key_rate_price_change(krd, [0.001, 0, -0.001], [0.5, 0.5])["pct_change"] == A(0.014)
    assert fi.key_rate_price_change(krd, [-0.001, 0, 0.001], [0.5, 0.5])["pct_change"] == A(-0.014)
    o = fi.key_rate_price_change([0, 16, 0], [0.001, 0, -0.001])
    assert o["pct_change"] == A(0) and o["effective_duration"] == A(16)


def test_foreign_bond_duration():
    # L3B Example 16 p.143: D 6 x beta 0.42 = 2.52
    assert fi.foreign_bond_duration(6, 0.42)["adjusted_duration"] == A(2.52)
