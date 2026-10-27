"""Golden tests for cfa_calc.equity. Expected values are the printed answers of CFA curriculum worked examples
(2014 curriculum unless noted); paths are relative to E:/CLAUDE/CFA/MD."""
import math

import pytest

from cfa_calc import REGISTRY
from cfa_calc import equity as e

L1_R46 = "Level-1/07-Equity/Curriculum-2014/SS13-R46-market-organization-and-structure.md"
L1_R47 = "Level-1/07-Equity/Curriculum-2014/SS13-R47-security-market-indices.md"
L1_R51 = "Level-1/07-Equity/Curriculum-2014/SS14-R51-equity-valuation-concepts-and-basic-tools.md"
R31 = "Level-2/07-Equity/Curriculum-2014/SS10-R31-return-concepts.md"
R35 = "Level-2/07-Equity/Curriculum-2014/SS11-R35-discounted-dividend-valuation.md"
R36 = "Level-2/07-Equity/Curriculum-2014/SS12-R36-free-cash-flow-valuation.md"
R37 = "Level-2/07-Equity/Curriculum-2014/SS12-R37-market-based-valuation-price-and-enterprise-value-multiples.md"
R38 = "Level-2/07-Equity/Curriculum-2014/SS12-R38-residual-income-valuation.md"
R39 = "Level-2/07-Equity/Curriculum-2014/SS12-R39-private-company-valuation.md"
L3_R23 = "Level-3/08-Equity-Portfolio-Management/Curriculum-2014/SS11-R23-equity-portfolio-management.md"


def test_all_registered_under_equity():
    names = [n for n, f in REGISTRY.items() if f.topic == "equity"]
    assert len(names) >= 45
    assert all(REGISTRY[n].func.__module__ == "cfa_calc.equity" for n in names)


# ------------------------------------------------------------------ L1 markets and indices
def test_margin_purchase_return_curriculum():
    # L1_R46 p.61, Example 19: -68.9%
    r = e.equity_margin_purchase_return(purchase_price=20, sale_price=15, shares=1000, call_money_rate=0.05,
                                        leverage_ratio=2.5, dividend_per_share=0.10, commission_per_share=0.01)
    assert r["initial_investment"] == pytest.approx(8010)
    assert r["remaining_equity"] == pytest.approx(2490)
    assert r["total_return"] == pytest.approx(-0.689, abs=5e-4)
    assert r["leveraged_price_return"] == pytest.approx(-0.625)


def test_margin_purchase_return_kaplan():
    # Level-1/07-Equity/Kaplan-Notes-2014/SS01-R46-market-organization-and-structure.md p.211: 23.72%
    r = e.equity_margin_purchase_return(purchase_price=100, sale_price=110, shares=1000, call_money_rate=0.04,
                                        initial_margin=0.40, dividend_per_share=2.0, commission_per_share=0.05)
    assert r["leverage_ratio"] == pytest.approx(2.5)
    assert r["total_return"] == pytest.approx(0.2372, abs=5e-5)


def test_margin_call_price():
    # L1_R46 p.62, Example 20: P = 16
    assert e.equity_margin_call_price(20, 0.40, 0.25)["margin_call_price"] == pytest.approx(16.0)
    # Schweser-Notes-2013 SS13-R46 p.211: 40 x 0.5/0.75 = 26.67
    assert e.equity_margin_call_price(40, 0.50, 0.25)["margin_call_price"] == pytest.approx(26.67, abs=0.005)
    # short: equity/value = (P0(1+IM) - P)/P = MM at the call price
    p = e.equity_margin_call_price(50, 0.5, 0.3, position="short")["margin_call_price"]
    assert (50 * 1.5 - p) / p == pytest.approx(0.3)


P0, P1, DIV = [50, 25, 12.5, 10, 4], [55, 22, 8, 14, 6], [0.75, 0.10, 0, 0.05, 0]


def test_price_weighted_index():
    # L1_R47 p.104, Exhibit 1: 20.30 -> 21.00, PR 3.45%, TR 4.33%, TR index 21.18
    r = e.equity_index_return(P0, P1, DIV, weighting="price")
    assert r["index_value_begin"] == pytest.approx(20.30)
    assert r["price_index_end"] == pytest.approx(21.00)
    assert r["price_return"] == pytest.approx(0.0345, abs=5e-5)
    assert r["total_return"] == pytest.approx(0.0433, abs=5e-5)
    assert r["total_return_index_end"] == pytest.approx(21.18, abs=0.005)
    assert r["weights_begin"][0] == pytest.approx(0.4926, abs=5e-5)


def test_equal_weighted_index():
    # L1_R47 p.106, Exhibit 3: PR 10.40%, TR 10.88%, 1,000 -> 1,104
    r = e.equity_index_return(P0, P1, DIV, weighting="equal")
    assert r["price_return"] == pytest.approx(0.1040, abs=5e-5)
    assert r["total_return"] == pytest.approx(0.1088, abs=5e-5)
    assert r["price_index_end"] == pytest.approx(1104.0)


def test_cap_and_float_weighted_index():
    shares = [3000, 10000, 5000, 8000, 7000]
    # L1_R47 p.108, Exhibit 4: PR 1.49%, TR 2.13%, divisor 570.50, index 1,014.90
    r = e.equity_index_return(P0, P1, DIV, weighting="market_cap", shares=shares)
    assert r["price_return"] == pytest.approx(0.0149, abs=5e-5)
    assert r["total_return"] == pytest.approx(0.0213, abs=5e-5)
    assert r["divisor"] == pytest.approx(570.50)
    assert r["price_index_end"] == pytest.approx(1014.90, abs=0.005)
    # L1_R47 p.110, Exhibit 5: PR -1.66%, TR -0.94%, 983.36
    r = e.equity_index_return(P0, P1, DIV, weighting="float_adjusted", shares=shares,
                              float_fraction=[1.0, 0.7, 0.9, 0.25, 0.8])
    assert r["price_return"] == pytest.approx(-0.0166, abs=5e-5)
    assert r["total_return"] == pytest.approx(-0.0094, abs=5e-5)
    assert r["price_index_end"] == pytest.approx(983.36, abs=0.005)


def test_divisor_adjustment_split():
    # L1_R47 p.105, Exhibit 2: divisor 5 -> 3.69 after 2-for-1 split, index stays 21.00
    r = e.price_weighted_divisor_adjustment(P1, [27.5, 22, 8, 14, 6], 5)
    assert r["new_divisor"] == pytest.approx(3.69, abs=0.005)
    assert sum([27.5, 22, 8, 14, 6]) / r["new_divisor"] == pytest.approx(21.0)


# ------------------------------------------------------------------ required return / ERP
def test_capm_and_multifactor():
    # R31 p.87, Example 4: 8.23%
    assert e.equity_required_return_capm(0.049, 0.74, 0.045)["required_return"] == pytest.approx(0.0823)
    # R31 p.92, FFM illustration: 0.151
    r = e.equity_required_return_multifactor(0.041, [1.2, 0.5, 0.8], [0.055, 0.02, 0.043])
    assert r["required_return"] == pytest.approx(0.151, abs=5e-4)
    # R31 p.94, Example 7 (Microsoft): FFM 8.96%
    r = e.equity_required_return_multifactor(0.047, [1.14, -0.222, -0.328], [0.055, 0.027, 0.043])
    assert r["required_return"] == pytest.approx(0.0896, abs=5e-5)


def test_build_up_and_bond_yield_plus():
    # R31 p.99: 13.85% and modified CAPM 14.75%
    assert e.equity_required_return_build_up(0.05, 0.045, 0.0435)["required_return"] == pytest.approx(0.1385)
    assert e.equity_required_return_build_up(0.05, 0.045, 0.0435, beta=1.2)["required_return"] == pytest.approx(0.1475)
    # R31 p.100, Example 9: 9.132%
    assert e.equity_required_return_bond_yield_plus(0.05632, 0.035)["required_return"] == pytest.approx(0.09132)


def test_equity_risk_premium():
    # R31 p.80: 1.9% + 7% - 5% = 3.9%
    assert e.equity_risk_premium_gordon(0.019, 0.07, 0.05)["equity_risk_premium"] == pytest.approx(0.039)
    # R31 p.82: supply side 2.8%
    r = e.equity_risk_premium_supply_side(0.025, 0.03, 0.0, 0.022, 0.05)
    assert r["equity_risk_premium"] == pytest.approx(0.028, abs=5e-4)
    # hand check of the definition
    ar = e.equity_risk_premium_historical([0.10, -0.05, 0.20], [0.03] * 3, method="arithmetic")
    assert ar["equity_risk_premium"] == pytest.approx(0.25 / 3 - 0.03)
    g = e.equity_risk_premium_historical([0.10, -0.05, 0.20], [0.03] * 3)
    assert g["mean_equity_return"] == pytest.approx((1.1 * 0.95 * 1.2) ** (1 / 3) - 1)


def test_wacc():
    # R36 p.314 Example 1: 10.2%; R36 p.344 Example 13 (with preferred): 9.04%
    assert e.equity_valuation_wacc(0.2, 0.057, 0.3333, 0.118)["wacc"] == pytest.approx(0.102, abs=5e-5)
    assert e.equity_valuation_wacc(0.4, 0.08, 0.30, 0.12, 0.1, 0.08)["wacc"] == pytest.approx(0.0904)


# ------------------------------------------------------------------ DDM
def test_gordon_growth():
    # L1_R51 p.276: 5.00(1.04)/0.04 = 130
    assert e.ddm_gordon_value(0.08, 0.04, d0=5.0)["value"] == pytest.approx(130.0)
    # R35 p.253 Example 5 (Sonoco): 26.89
    assert e.ddm_gordon_value(0.101, 0.06, d0=1.04)["value"] == pytest.approx(26.89, abs=0.005)
    # R35 p.257 Example 9 (Afton, negative growth): 19.32
    assert e.ddm_gordon_value(0.12, -0.10, d1=4.25)["value"] == pytest.approx(19.32, abs=0.005)
    with pytest.raises(ValueError):
        e.ddm_gordon_value(0.05, 0.06, d0=1.0)
    with pytest.raises(ValueError):
        e.ddm_gordon_value(0.10, 0.05, d0=1.0, d1=1.05)


def test_gordon_sensitivity_table():
    # R35 p.256 Example 7, Exhibit 3
    t = e.equity_gordon_sensitivity_table(0.68, [0.09, 0.0925, 0.095], [0.0575, 0.06, 0.0625])["table"]
    expected = [[22.13, 24.03, 26.27], [20.55, 22.18, 24.08], [19.18, 20.59, 22.23]]
    for row, exp in zip(t, expected):
        assert row == pytest.approx(exp, abs=0.006)


def test_multistage_ddm():
    # L1_R51 p.271 Example 3: 20.23 (terminal price given)
    r = e.ddm_multistage_value(0.10, [2.00, 2.10, 2.20], terminal_value=20.00)
    assert r["value"] == pytest.approx(20.23, abs=0.005)
    # L1_R51 p.280 Example 7 (first dividend at t=5): 68.30
    assert e.ddm_multistage_value(0.10, [0, 0, 0, 0, 4.0], terminal_growth=0.06)["value"] == pytest.approx(68.30, abs=0.005)
    # R35 p.269 Example 14 (DuPont, P/E terminal): 56.72
    d = [1.48 * 1.10 ** t for t in range(1, 5)]
    assert e.ddm_multistage_value(0.105, d, terminal_value=14 * d[-1] / 0.40)["value"] == pytest.approx(56.72, abs=0.005)
    # R35 p.277 Example 19 (Yang Co.): 399.48
    r = e.ddm_multistage_value(0.12, [21.00, 18.90, 17.01, 15.309, 60, 40, 40], terminal_growth=0.05)
    assert r["terminal_value"] == pytest.approx(600.0)
    assert r["value"] == pytest.approx(399.48, abs=0.02)  # printed total adds rounded PVs; exact 399.468


def test_two_and_three_stage_ddm():
    # L1_R51 p.281 Example 8: 59.68
    assert e.ddm_two_stage_value(5.00, 0.10, 3, 0.05, 0.15)["value"] == pytest.approx(59.68, abs=0.005)
    # R35 p.267 Example 13 (Meditec): 16.0882, terminal 35.9817
    r = e.ddm_two_stage_value(0.14, 0.15, 10, 0.08, 0.097)
    assert r["value"] == pytest.approx(16.0882, abs=5e-4)
    assert r["terminal_value"] == pytest.approx(35.9817, abs=5e-4)
    # R35 p.273 Example 17 (IBM): 113.06
    assert e.ddm_three_stage_value(1.60, 0.14, 2, 0.12, 5, 0.102, 0.12)["value"] == pytest.approx(113.06, abs=0.005)


def test_h_model_and_declining_three_stage():
    # R35 p.271 Example 16 (Vinci): 36.31 + 36.99 = 73.30
    r = e.ddm_h_model_value(1.37, 0.24, 0.06, 0.10, 12)
    assert r["value_mature_growth"] == pytest.approx(36.31, abs=0.006)  # exact 36.305, printed half-up
    assert r["value"] == pytest.approx(73.30, abs=0.006)  # exact 73.295 = 36.305 + 36.99
    # R35 p.275 Example 18 (Energen): 28.6044, stage-1 share 8.5%
    r = e.ddm_three_stage_declining_value(0.46, 0.12, 5, 0.075, 10, 0.10)
    # the book rounds D5 to 0.8107 (exact 0.810679), so allow 0.002
    assert r["h_model_value_at_n1"] == pytest.approx(42.1564, abs=2e-3)
    assert r["value"] == pytest.approx(28.6044, abs=2e-3)
    assert r["stage1_share"] == pytest.approx(0.085, abs=5e-4)
    # consistency: H-model with gS = gL collapses to Gordon
    assert e.ddm_h_model_value(1.0, 0.05, 0.05, 0.10, 10)["value"] == pytest.approx(e.ddm_gordon_value(0.10, 0.05, d0=1.0)["value"])


def test_implied_returns_and_growth():
    # R35 p.264 Example 12 (FPL): 9.67%
    assert e.ddm_gordon_implied_return(56.60, 0.055, d0=2.24)["required_return"] == pytest.approx(0.0967, abs=1e-4)  # book rounds D1/P0 to 4.17%
    # R35 p.279: H-model expected return 12.3%
    assert e.ddm_h_model_implied_return(20, 1, 0.10, 0.06, 10)["required_return"] == pytest.approx(0.123)
    # R35 p.280 Example 21 (JNJ): IRR 9.98%
    d = [1.66 * 1.09 ** t for t in range(1, 7)]
    assert e.ddm_multistage_implied_return(66.19, d, 0.07)["required_return"] == pytest.approx(0.0998, abs=5e-5)
    # R35 p.260 Example 10: implied growth 6.86%
    assert e.ddm_implied_growth(40, 0.122, d0=2.00)["implied_growth"] == pytest.approx(0.0686, abs=5e-5)
    # round trip: value -> implied return
    v = e.ddm_gordon_value(0.09, 0.04, d1=2.0)["value"]
    assert e.ddm_gordon_implied_return(v, 0.04, d1=2.0)["required_return"] == pytest.approx(0.09)


def test_preferred_stock():
    # L1_R51 p.275 Example 5: 63.33; R35 p.257 Example 8: 11.11
    assert e.equity_preferred_stock_value(4.75, 0.075)["value"] == pytest.approx(63.33, abs=0.005)
    assert e.equity_preferred_stock_value(1.00, 0.09)["value"] == pytest.approx(11.11, abs=0.005)
    # L1_R51 p.276 Example 5 Case 2 (YPG retractable, 34 quarters at 3.875%): 12.71
    r = e.equity_preferred_stock_value(1.25, 0.155, years=8.5, par=25, freq=4)
    assert r["periods"] == 34
    assert r["value"] == pytest.approx(12.71, abs=0.005)


def test_pvgo_and_sustainable_growth():
    # R35 p.261 (Middlesex Water): no-growth 8.54, PVGO 9.85, P/E growth component 12.5
    r = e.equity_pvgo_decomposition(18.39, 0.79, 0.0925)
    assert r["no_growth_value"] == pytest.approx(8.54, abs=0.005)
    assert r["pvgo"] == pytest.approx(9.85, abs=0.005)
    assert r["growth_pe_component"] == pytest.approx(12.5, abs=0.05)
    # R35 p.278 Example 20: 0.85 x 12% = 10.2%
    assert e.ddm_sustainable_growth(retention_rate=0.85, roe=0.12)["sustainable_growth"] == pytest.approx(0.102)
    # R35 p.283 Example 23: Baggai 3.75%, Mondale 13.33%
    assert e.ddm_sustainable_growth(retention_rate=0.30, roa=0.10, equity_multiplier=1.25)["sustainable_growth"] == pytest.approx(0.0375)
    assert e.ddm_sustainable_growth(retention_rate=2 / 3, roa=0.10, equity_multiplier=2.0)["sustainable_growth"] == pytest.approx(0.1333, abs=5e-5)
    # PRAT: 5% margin x 2.0 turnover x 1.5 leverage, payout 40% -> 9% (R35 p.284)
    r = e.ddm_sustainable_growth(payout_ratio=0.40, net_profit_margin=0.05, asset_turnover=2.0, equity_multiplier=1.5)
    assert r["roe"] == pytest.approx(0.15) and r["sustainable_growth"] == pytest.approx(0.09)


# ------------------------------------------------------------------ free cash flow
PITTS = dict(tax_rate=0.40, noncash_charges=300, interest_expense=100, fixed_capital_investment=400,
             working_capital_investment=45, net_borrowing=75)


@pytest.mark.parametrize("start,kw", [("net_income", {"net_income": 240}), ("ebit", {"ebit": 500}),
                                      ("ebitda", {"ebitda": 800}), ("cfo", {"cfo": 495})])
def test_fcff_fcfe_all_starts_pitts(start, kw):
    # R36 p.330-333, Examples 6-7 (Pitts): FCFF 155, FCFE 170 from every starting point
    r = e.equity_fcff_fcfe(start=start, **kw, **PITTS)
    assert r["fcff"] == pytest.approx(155)
    assert r["fcfe"] == pytest.approx(170)


def test_fcff_cane_and_preferred():
    # R36 p.318 Example 2 (Cane 2008): FCFF 97.50
    r = e.equity_fcff_fcfe(tax_rate=0.30, net_income=97.52, noncash_charges=45, interest_expense=15.68,
                           working_capital_investment=56)
    assert r["fcff"] == pytest.approx(97.50, abs=0.005)
    # R36 p.345 Example 13 (Welch, preferred dividends): FCFF 90.4, FCFE 85
    r = e.equity_fcff_fcfe(tax_rate=0.30, net_income=110, noncash_charges=40, interest_expense=32,
                           fixed_capital_investment=70, working_capital_investment=20, net_borrowing=25,
                           preferred_dividends=8)
    assert r["fcff"] == pytest.approx(90.4)
    assert r["fcfe"] == pytest.approx(85)


def test_fcf_from_sales():
    # R36 p.338 Example 9: FCFF 185; p.340 Example 11: FCFE 191.50
    r = e.equity_fcf_from_sales_forecast(3000, 0.10, 0.40, 1 / 3, 0.15, ebit_margin=500 / 3000,
                                         net_profit_margin=0.08, debt_ratio=0.5)
    assert r["fcff"] == pytest.approx(185)
    assert r["net_borrowing"] == pytest.approx(72.5)
    assert r["fcfe"] == pytest.approx(191.5)


def test_fcff_fcfe_valuation():
    # R36 p.314 Example 1 (Cagiati): firm 14,134.6
    r = e.equity_fcff_valuation(0.102, 0.05, fcff0=700, debt=2200, shares=200)
    assert r["firm_value"] == pytest.approx(14134.6, abs=0.05)
    assert r["equity_value"] == pytest.approx(14134.6 - 2200, abs=0.05)
    # R36 p.345 Example 13 (Welch): equity 1,365.40 (FCFF) and 1,357.42 (FCFE)
    assert e.equity_fcff_valuation(0.0904, 0.04, fcff0=90.4, debt=400, preferred=100)["equity_value"] == pytest.approx(1365.40, abs=0.005)
    assert e.equity_fcfe_valuation(0.12, 0.054, fcfe0=85)["equity_value"] == pytest.approx(1357.42, abs=0.005)
    # R36 p.348 Example 15 (Petrobras) and Exhibit 14 sensitivities
    assert e.equity_fcfe_valuation(0.155, 0.073, fcfe0=6.15)["equity_value"] == pytest.approx(80.48, abs=0.006)  # exact 80.475
    assert e.equity_fcfe_valuation(0.10 + 0.75 * 0.055, 0.073, fcfe0=6.15)["equity_value"] == pytest.approx(96.69, abs=0.005)
    assert e.equity_fcfe_valuation(0.155, 0.05, fcfe0=6.15)["equity_value"] == pytest.approx(61.50, abs=0.005)
    # R36 p.351 Example 16 (TechnoSchaft): 40.98
    r = e.equity_fcfe_valuation(0.124, 0.06, forecast_fcfe=[0.900, 1.080, 1.296], terminal_fcfe=3.491)
    assert r["terminal_value"] == pytest.approx(54.55, abs=0.005)
    assert r["equity_value"] == pytest.approx(40.98, abs=0.005)
    # R36 p.353 Example 17 (Sindhuh): 78.73
    r = e.equity_fcfe_valuation(0.104, 0.07, forecast_fcfe=[-0.030, 1.057, 2.023, 2.919], terminal_fcfe=3.759)
    assert r["equity_value"] == pytest.approx(78.73, abs=0.005)
    # consistency: single-stage equals a one-period forecast of FCF1
    a = e.equity_fcfe_valuation(0.1, 0.03, fcfe0=2.0)["equity_value"]
    b = e.equity_fcfe_valuation(0.1, 0.03, forecast_fcfe=[2.06])["equity_value"]
    assert a == pytest.approx(b)


# ------------------------------------------------------------------ multiples
def test_justified_multiples():
    # R35 p.263 Example 11 (Carrefour): leading 16.0, trailing 17.1
    r = e.equity_justified_pe(0.32, 0.09, 0.07)
    assert r["leading_pe"] == pytest.approx(16.0) and r["trailing_pe"] == pytest.approx(17.1, abs=0.05)
    # L1_R51 p.288 Example 11 (Nestle): 14.1; R37 p.406 Example 7 (BP): 24.3
    assert e.equity_justified_pe(0.45, 0.12, 0.55 * 0.16)["leading_pe"] == pytest.approx(14.1, abs=0.05)
    assert e.equity_justified_pe(0.85, 0.095, 0.06)["leading_pe"] == pytest.approx(24.3, abs=0.05)
    # R37 p.432: P/B 1.67; R37 p.439 Example 26 (Getinge): P/S 2.7
    assert e.equity_justified_pb(0.12, 0.10, 0.07)["justified_pb"] == pytest.approx(1.67, abs=0.005)
    assert e.equity_justified_ps(0.082, 0.30, 0.11, 0.10)["trailing_ps"] == pytest.approx(2.7, abs=0.01)
    # R37 p.444 Example 30 (Western Digital): 41.35, P/CF 7.6, P/FCFE 30.9
    r = e.equity_justified_price_to_cash_flow(1.34, 5.43, 0.115, 0.08)
    assert r["value"] == pytest.approx(41.35, abs=0.005)
    assert r["justified_p_cf"] == pytest.approx(7.6, abs=0.05)
    assert r["justified_p_fcfe"] == pytest.approx(30.9, abs=0.05)
    # consistency: justified P/B equals single-stage RI value / book
    ri = e.equity_residual_income_single_stage(10.0, 0.12, 0.10, 0.07)
    assert ri["justified_pb"] == pytest.approx(e.equity_justified_pb(0.12, 0.10, 0.07)["justified_pb"])


def test_peg_ev_harmonic_sue():
    # R37 p.413 Example 11: T forward P/E 10.2, growth 13% -> PEG 0.78
    assert e.equity_peg_ratio(10.2, 0.13)["peg"] == pytest.approx(0.78, abs=0.005)
    # L1_R51 p.295 Example 15 (Cameco): EV 9,567,892, EV/EBITDA 8.9
    r = e.equity_enterprise_value(market_cap=7_349_229, debt=2_487_839, cash_and_investments=269_176, ebitda=1_078_606)
    assert r["enterprise_value"] == pytest.approx(9_567_892)
    assert r["ev_ebitda"] == pytest.approx(8.9, abs=0.05)
    # R37 p.466 Exhibit 28: arithmetic 15, weighted 14.5, harmonic 13.33, weighted harmonic 12.90
    r = e.equity_multiple_harmonic_mean([10, 20], [715, 585])
    assert r["arithmetic_mean"] == pytest.approx(15) and r["weighted_mean"] == pytest.approx(14.5)
    assert r["harmonic_mean"] == pytest.approx(13.33, abs=0.005)
    assert r["weighted_harmonic_mean"] == pytest.approx(1300 / 100.75)
    # R37 p.460 Example 37: (2.40 - 1.53)/0.28 = 3.11; p.461 Example 38 (Sony): SUE 4.225
    assert e.equity_earnings_surprise_sue(2.40, 1.53, 0.28)["sue"] == pytest.approx(3.11, abs=0.005)
    assert e.equity_earnings_surprise_sue(28.950, -6.300, 8.344)["sue"] == pytest.approx(4.225, abs=5e-4)


# ------------------------------------------------------------------ residual income
def test_residual_income_and_eva():
    # R38 p.493-494 Example 1 (Axis): RI -29,000 both ways
    assert e.equity_residual_income(91_000, 1_000_000, 0.12)["residual_income"] == pytest.approx(-29_000)
    wacc = 0.5 * 0.07 * 0.7 + 0.5 * 0.12
    r = e.equity_economic_value_added(140_000, 2_000_000, wacc, market_value=758_333 + 1_000_000)
    assert r["eva"] == pytest.approx(-29_000)
    assert r["mva"] == pytest.approx(-241_667)


def test_residual_income_valuation_bugg():
    # R38 p.500 Example 3-4 (Bugg): 11.15; equals PV of dividends
    r = e.equity_residual_income_valuation(6.00, 0.10, eps=[2.00, 2.50, 4.00], dividends=[1.00, 1.25, 12.25])
    assert r["residual_income"] == pytest.approx([1.40, 1.80, 3.175])
    assert r["value"] == pytest.approx(11.15, abs=0.005)
    ddm = e.ddm_multistage_value(0.10, [1.00, 1.25, 12.25])["value"]
    assert r["value"] == pytest.approx(ddm)


TSMC_ROE = [None, None] + [0.25] * 8 + [0.20] * 11


@pytest.mark.parametrize("terminal,kw,n,expected,src", [
    ("none", {}, 20, 44.38, "p.509 Example 8"),
    ("perpetuity", {}, 20, 49.94, "p.510 Example 9"),
    ("persistence", {"persistence": 0.60}, 21, 46.11, "p.511 Example 10"),
])
def test_residual_income_multistage_tsmc(terminal, kw, n, expected, src):
    # R38 (Taiwan Semiconductor) - see `src` for the page
    r = e.equity_residual_income_valuation(19.59, 0.15, eps=[4.256, 5.556], roe=TSMC_ROE[:n],
                                           dividends=[3.000, 3.284], payout_ratio=0.30, terminal=terminal, **kw)
    assert r["value"] == pytest.approx(expected, abs=0.006)


def test_single_stage_ri_and_implied_growth():
    # R38 p.506 Examples 6-7 (Canon): 50.16 and implied g 8.16%
    r = e.equity_residual_income_single_stage(18.81, 0.16, 0.11, growth=0.08, price=51.90)
    assert r["value"] == pytest.approx(50.16, abs=0.005)
    assert r["implied_growth"] == pytest.approx(0.0816, abs=5e-5)


# ------------------------------------------------------------------ private companies
def test_private_company_methods():
    # R39 p.572 Example 4 (Able): equity 90,661,386 and 84,655,556
    assert e.private_company_capitalized_cash_flow(0.131, 0.03, cash_flow_next=9_358_800, debt=2_000_000)["equity_value"] == pytest.approx(90_661_386, abs=1)
    assert e.private_company_capitalized_cash_flow(0.138, 0.03, cash_flow_next=9_358_800, debt=2_000_000)["equity_value"] == pytest.approx(84_655_556, abs=1)
    # R39 p.602 practice problem 3: FCFE 1.8m x 1.055 / 12.5% = 15.19m
    assert e.private_company_capitalized_cash_flow(0.18, 0.055, cash_flow_current=1.8)["value"] == pytest.approx(15.19, abs=0.005)
    # R39 p.574 Section 4.2.4: intangibles 22,889, value 1,022,889
    r = e.private_company_excess_earnings(100_000, 200_000, 800_000, 0.05, 0.11, 0.12, 0.03)
    assert r["intangibles_value"] == pytest.approx(22_889, abs=1)
    assert r["business_value"] == pytest.approx(1_022_889, abs=1)
    # R39 p.602 practice problem 4: intangibles 4,830,000, value 12,330,000
    r = e.private_company_excess_earnings(1_000_000, 2_000_000, 5_500_000, 0.05, 0.08, 0.15, 0.05)
    assert r["intangibles_value"] == pytest.approx(4_830_000) and r["business_value"] == pytest.approx(12_330_000)


def test_valuation_discounts():
    # R39 p.587: 10% and 20% -> 28%; p.585: control premium 20% -> DLOC 16.7%
    assert e.private_company_valuation_discounts(dlom=0.20, dloc=0.10)["total_discount"] == pytest.approx(0.28)
    assert e.private_company_valuation_discounts(control_premium=0.20)["dloc"] == pytest.approx(0.167, abs=5e-4)
    # R39 p.603 practice problem 5: CP 15%, DLOM 25% -> 34.8%
    assert e.private_company_valuation_discounts(dlom=0.25, control_premium=0.15)["total_discount"] == pytest.approx(0.348, abs=5e-4)
    # R39 p.589 Example 8: 9.6m less 5% DLOM -> 9,120,000
    assert e.private_company_valuation_discounts(dlom=0.05, dloc=0.0, pro_rata_value=9_600_000)["value_after_discounts"] == pytest.approx(9_120_000)
    # L3 Alternatives R26 p.55 Example 7: 50 x 0.8 x 0.75 = 30
    assert e.private_company_valuation_discounts(dlom=0.25, dloc=0.20, pro_rata_value=50)["value_after_discounts"] == pytest.approx(30)


# ------------------------------------------------------------------ L3 equity portfolio management
def test_active_return_risk():
    # hand check: active [1%, -1%, 3%] -> mean 1%, sample std 2%, IR 0.5
    r = e.equity_active_return_risk([0.05, 0.02, 0.08], [0.04, 0.03, 0.05])
    assert r["mean_active_return"] == pytest.approx(0.01)
    assert r["tracking_risk"] == pytest.approx(0.02)
    assert r["information_ratio"] == pytest.approx(0.5)
    m = e.equity_active_return_risk([0.05, 0.02, 0.08], [0.04, 0.03, 0.05], periods_per_year=12)
    assert m["tracking_risk"] == pytest.approx(0.02 * math.sqrt(12))


def test_fundamental_law():
    # L3_R23 p.226 Example 12: A 0.67, B 0.60
    assert e.equity_fundamental_law(0.03, 500)["information_ratio"] == pytest.approx(0.67, abs=0.005)
    assert e.equity_fundamental_law(0.06, 100)["information_ratio"] == pytest.approx(0.60)
    r = e.equity_fundamental_law(0.06, 100, transfer_coefficient=0.5, active_risk=0.04)
    assert r["information_ratio"] == pytest.approx(0.30) and r["expected_active_return"] == pytest.approx(0.012)


def test_manager_portfolio_and_true_misfit():
    # L3_R23 p.232 Example 14: alpha 1.43%, TE 1.34%, IR 1.07
    r = e.equity_manager_portfolio_active_risk([4 / 7, 1 / 7, 1 / 7, 1 / 7], [0, 0.02, 0.04, 0.04], [0, 0.04, 0.06, 0.06])
    assert r["portfolio_alpha"] == pytest.approx(0.0143, abs=5e-5)
    assert r["portfolio_active_risk"] == pytest.approx(0.0134, abs=5e-5)
    assert r["information_ratio"] == pytest.approx(1.07, abs=0.005)
    # L3_R23 p.233: true -3%, misfit 5%, true risk 3.775%, true IR -0.7947
    r = e.equity_manager_true_misfit(0.12, 0.15, 0.10, 0.055, 0.04)
    assert r["true_active_return"] == pytest.approx(-0.03) and r["misfit_active_return"] == pytest.approx(0.05)
    assert r["true_active_risk"] == pytest.approx(0.03775, abs=5e-6)
    assert r["true_information_ratio"] == pytest.approx(-0.7947, abs=5e-4)
