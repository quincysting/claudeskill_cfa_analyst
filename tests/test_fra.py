"""Tests for cfa_calc.fra. Expected values come from CFA curriculum worked examples (and Kaplan notes) in the
knowledge base (E:/CLAUDE/CFA/MD); each test cites file and page. KB paths are relative to MD/."""
import pytest
from pytest import approx

from cfa_calc import REGISTRY
from cfa_calc import fra

L1 = "Level-1/04-Financial-Reporting/Curriculum-2014/"
L2 = "Level-2/04-Financial-Reporting/Curriculum-2014/"
KAPLAN = "Level-1/04-Financial-Reporting/Kaplan-Notes-2014/"


# ----------------------------------------------------------------------------- ratios

KAPLAN_CO = dict(revenue=4000, cogs=3000, ebit=350, interest_expense=50, pretax_income=300,
                 net_income=200, cash=105, receivables=205, inventory=310, payables=110, current_assets=620,
                 current_liabilities=325, total_assets=2060, total_debt=825, total_equity=1020,
                 receivables_begin=195, inventory_begin=290, payables_begin=90, current_assets_begin=580,
                 current_liabilities_begin=275, total_assets_begin=1940, total_debt_begin=875,
                 total_equity_begin=880)


def test_financial_ratios_kaplan_sample_company():
    # Kaplan-Notes-2014/SS08-R28-financial-analysis-techniques.md p158-162 "Using ratios to evaluate a company":
    # current 1.9, quick 0.95, DSO 18.25, inventory turnover 10.0, asset turnover 2.0, WC turnover 13.3,
    # GPM 25%, NPM 5%, return on total capital 19.4%, ROE 21.1%, D/E 80.9%, interest coverage 7.0
    r = fra.financial_ratios(**KAPLAN_CO)
    assert r["current_ratio"] == approx(1.908, abs=1e-3)
    assert r["quick_ratio"] == approx(0.954, abs=1e-3)
    assert r["days_sales_outstanding"] == approx(18.25)
    assert r["inventory_turnover"] == approx(10.0)
    assert r["total_asset_turnover"] == approx(2.0)
    assert r["working_capital_turnover"] == approx(13.33, abs=0.005)
    assert r["gross_profit_margin"] == approx(0.25) and r["net_profit_margin"] == approx(0.05)
    assert r["return_on_total_capital"] == approx(0.194, abs=5e-4)
    assert r["roe"] == approx(0.211, abs=5e-4)
    assert r["debt_to_equity"] == approx(0.809, abs=5e-4)
    assert r["interest_coverage"] == approx(7.0)
    # consistency: CCC = DOH + DSO - DPO; DuPont ROE = NPM x TAT x leverage (same averages)
    assert r["cash_conversion_cycle"] == approx(r["days_inventory_on_hand"] + r["days_sales_outstanding"]
                                                - r["days_payables"])
    assert r["roe"] == approx(r["net_profit_margin"] * r["total_asset_turnover"] * r["financial_leverage"])


def test_financial_ratios_lenovo_receivables():
    # L1 SS08-R28-financial-analysis-techniques.md p360 Example 6 (Lenovo): turnover 22.09, DSO 16.5 days
    r = fra.financial_ratios(revenue=16604815, receivables=1021062, receivables_begin=482086)
    assert r["receivables_turnover"] == approx(22.0934, abs=1e-4)
    assert r["days_sales_outstanding"] == approx(16.5, abs=0.05)


def test_financial_ratios_skips_missing_and_errors_when_empty():
    assert set(fra.financial_ratios(current_assets=2, current_liabilities=1)) == {"current_ratio"}
    with pytest.raises(ValueError):
        fra.financial_ratios()


def test_dupont_three_and_five_way():
    # Kaplan-Notes-2014/SS08-R28 p166 "Extended DuPont analysis", Company A: tax burden 66.7%, interest burden
    # 85.7%, EBIT margin 7%, asset turnover 2.0, leverage 1.67, ROE 13.4%
    r = fra.dupont_roe(20, 500, 250, 150, pretax_income=30, ebit=35)
    assert r["tax_burden"] == approx(0.667, abs=5e-4) and r["interest_burden"] == approx(0.857, abs=5e-4)
    assert r["ebit_margin"] == approx(0.07) and r["asset_turnover"] == approx(2.0)
    assert r["financial_leverage"] == approx(1.667, abs=5e-4)
    assert r["roe"] == approx(r["tax_burden"] * r["interest_burden"] * r["ebit_margin"] * r["asset_turnover"]
                              * r["financial_leverage"])
    # Kaplan p165: NPM 4% x turnover 2.0 x leverage 2.5 = 20%
    assert fra.dupont_roe(4, 100, 50, 20)["roe"] == approx(0.20)


def test_common_size_statement():
    r = fra.common_size_statement({"revenue": 4000, "cogs": 3000, "net_income": 200}, "revenue")
    assert r["common_size"] == {"revenue": approx(1.0), "cogs": approx(0.75), "net_income": approx(0.05)}


def test_cash_conversion_cycle():
    # L1 Corporate Finance SS11-R40-working-capital-management.md p167 Example 1: 41.0 + 39.0 - 34.3 = 45.7 days
    r = fra.cash_conversion_cycle(41.0, 39.0, 34.3)
    assert r["cash_conversion_cycle"] == approx(45.7) and r["operating_cycle"] == approx(80.0)


# ----------------------------------------------------------------------------- inventories

def test_inventory_cost_flow_gsi():
    # L1 SS09-R29-inventories.md p416-418 Example 2 (GSI soap, thousands of yuan): COGS FIFO 50,800 / LIFO 49,200 /
    # WAC 50,267; ending inventory 7,200 / 8,800 / 7,733; gross profit 74,000 / 75,600 / 74,533
    r = fra.inventory_cost_flow([[100000, 110], [200000, 100], [300000, 90]], 520000, sale_price=240)
    assert r["fifo_cogs"] == approx(50.8e6) and r["fifo_ending_inventory"] == approx(7.2e6)
    assert r["lifo_cogs"] == approx(49.2e6) and r["lifo_ending_inventory"] == approx(8.8e6)
    assert r["weighted_average_cogs"] == approx(50.267e6, abs=1e3)
    assert r["weighted_average_ending_inventory"] == approx(7.733e6, abs=1e3)
    assert r["lifo_gross_profit"] == approx(75.6e6)
    one = fra.inventory_cost_flow([[100000, 110], [200000, 100], [300000, 90]], 520000, method="lifo")
    assert one["ending_layers"] == [[80000, 110.0]]
    with pytest.raises(ValueError):
        fra.inventory_cost_flow([[10, 1]], 11)


def test_lifo_to_fifo_caterpillar():
    # L2 SS05-R17-inventories-implications...md p29-32 Example 2 (CAT 2008, 20% tax): FIFO inventory 11,964;
    # FIFO COGS 37,849; FIFO NI 4,010; extra 2008 tax 113
    r = fra.lifo_to_fifo(8781, 3183, 2617, lifo_cogs=38415, net_income=3557, tax_rate=0.20)
    assert r["fifo_inventory"] == approx(11964) and r["fifo_cogs"] == approx(37849)
    assert r["fifo_net_income"] == approx(4010, abs=0.5)
    assert r["current_year_tax_increase"] == approx(113, abs=0.5)
    assert r["equity_increase"] + r["cumulative_tax_on_reserve"] == approx(3183)
    assert not r["lifo_liquidation_indicated"]
    # p37 Example 3 (Sturm Ruger): reserve fell 57,555 -> 46,890 -> liquidation indicated
    assert fra.lifo_to_fifo(13297, 46890, 57555)["lifo_liquidation_indicated"]


def test_lifo_liquidation_profit():
    # L2 SS05-R17 p37-38 Example 4 (Reliable Fans): 1,000 units of the $100 layer at $115 replacement -> $15,000
    assert fra.lifo_liquidation_profit([[1000, 100]], 115)["pretax_liquidation_profit"] == approx(15000)


def test_inventory_writedown():
    # L1 SS09-R29 p421 Example 4 (Acme): IFRS write-down 0.3 to NRV 4.9; US GAAP (LCM) 0.5 to replacement cost 4.7
    assert fra.inventory_writedown(5.2, 4.9)["writedown"] == approx(0.3)
    r = fra.inventory_writedown(5.2, 4.9, "us_gaap_lcm", replacement_cost=4.7, normal_profit=0.4)
    assert r["writedown"] == approx(0.5) and r["carrying_amount"] == approx(4.7)
    # market is bounded below by NRV - normal profit and above by NRV
    assert fra.inventory_writedown(5.2, 4.9, "us_gaap_lcm", 4.0, 0.4)["carrying_amount"] == approx(4.5)
    assert fra.inventory_writedown(5.2, 4.9, "us_gaap_lcm", 5.0, 0.4)["carrying_amount"] == approx(4.9)


# ----------------------------------------------------------------------------- long-lived assets

def test_depreciation_methods():
    # L1 SS09-R30-long-lived-assets.md p454-457 Example 4 ($2,300 cost, $100 residual, 4 years, 800 units):
    # straight line 550/yr; DDB 1,150 / 575 / 288 / 187; units of production 550 / 825 / 550 / 275
    assert fra.depreciation_schedule(2300, 100, 4)["depreciation"] == [approx(550)] * 4
    ddb = fra.depreciation_schedule(2300, 100, 4, "declining_balance")
    assert ddb["depreciation"] == [approx(1150), approx(575), approx(287.5), approx(187.5)]
    assert ddb["schedule"][-1]["ending_book_value"] == approx(100)
    uop = fra.depreciation_schedule(2300, 100, method="units_of_production", units_per_period=[200, 300, 200, 100],
                                    total_units=800)
    assert uop["depreciation"] == [approx(550), approx(825), approx(550), approx(275)]
    syd = fra.depreciation_schedule(2300, 100, 4, "sum_of_years_digits")
    assert syd["depreciation"] == [approx(880), approx(660), approx(440), approx(220)]  # 2,200 x 4/10 ...
    for r in (ddb, uop, syd):
        assert r["total_depreciation"] == approx(2200)


def test_ddb_switches_to_straight_line():
    # 10-year DDB of 1,000 to zero: SL on remaining value exceeds DDB from year 7 (Kaplan/curriculum switch rule)
    dep = fra.depreciation_schedule(1000, 0, 10, "declining_balance")["depreciation"]
    assert sum(dep) == approx(1000)
    assert dep[6] == approx(dep[9]) and dep[6] > 1000 * 0.8 ** 6 * 0.2


def test_capitalized_borrowing_costs():
    # L1 SS09-R30 p448 Example 2 (BILDA): IFRS 180,000 (net of 20,000 investment income); US GAAP 200,000
    r = fra.capitalized_borrowing_costs(1e6, 0.10, 2, 20000)
    assert r["ifrs_capitalized"] == approx(180000) and r["us_gaap_capitalized"] == approx(200000)


def test_capitalized_interest_coverage():
    # L2 SS05-R18-long-lived-assets...md p74-75 Example 3 (MTR Gaming 2006): 15.77 reported, 11.68 adjusted
    r = fra.capitalized_interest_coverage(268800, 17047, 6000, 365)
    assert r["reported_interest_coverage"] == approx(15.77, abs=0.005)
    assert r["adjusted_interest_coverage"] == approx(11.68, abs=0.005)


def test_capitalize_vs_expense():
    # L2 SS05-R18 p69-71 Example 1 (CAP vs NOW; base pre-expenditure NI 700, CFO 700 each year):
    # CAP NI 490/yr, CFO 790/yr, CFI -900; NOW NI 70 then 700, CFO 70 then 700; totals equal (1,470)
    r = fra.capitalize_vs_expense(900, 3, 0.30)
    assert [700 + x for x in r["capitalize_net_income_effect"]] == [approx(490)] * 3
    assert [700 + x for x in r["capitalize_cfo_effect"]] == [approx(790)] * 3
    assert r["capitalize_cfi_effect"][0] == approx(-900)
    assert [700 + x for x in r["expense_net_income_effect"]] == [approx(70), approx(700), approx(700)]
    assert r["capitalize_asset_carrying"] == [approx(600), approx(300), approx(0)]
    assert sum(r["capitalize_net_income_effect"]) == approx(sum(r["expense_net_income_effect"]))


def test_fixed_asset_age():
    # L1 SS10-R35-financial-statement-analysis-applications.md p663 problem 11: age 1.6/0.4 = 4, life 2.8/0.4 = 7
    r = fra.fixed_asset_age(2.8, 1.6, 0.4)
    assert r["average_age"] == approx(4) and r["total_useful_life"] == approx(7)
    assert r["remaining_useful_life"] == approx(3)


def test_asset_impairment():
    # L1 SS09-R30 p466-467 Example 9 (Sussex): IFRS write down to 16,000 (loss 2,000); US GAAP recoverable, no loss
    ifrs = fra.asset_impairment(18000, "ifrs", fair_value_less_costs_to_sell=17000 - 2000, value_in_use=16000)
    assert ifrs["impairment_loss"] == approx(2000) and ifrs["new_carrying_amount"] == approx(16000)
    us = fra.asset_impairment(18000, "us_gaap", undiscounted_cash_flows=19000, fair_value=17000)
    assert us["impairment_loss"] == 0
    # US GAAP loss is measured against fair value once the recoverability test fails
    assert fra.asset_impairment(18000, "us_gaap", undiscounted_cash_flows=17500, fair_value=17000)["impairment_loss"] == approx(1000)


def test_goodwill_impairment():
    # L2 SS06-R19-intercorporate-investments.md p168 Examples 10-11: IFRS loss 100,000 (600,000 if recoverable
    # 800,000: 300,000 goodwill + 300,000 other assets); US GAAP two-step loss 200,000
    assert fra.goodwill_impairment(1.4e6, 3e5, "ifrs", recoverable_amount=1.3e6)["impairment_loss"] == approx(1e5)
    big = fra.goodwill_impairment(1.4e6, 3e5, "ifrs", recoverable_amount=8e5)
    assert big["goodwill_impairment"] == approx(3e5) and big["other_asset_impairment"] == approx(3e5)
    us = fra.goodwill_impairment(1.4e6, 3e5, "us_gaap", fair_value=1.3e6, fair_value_identifiable_net_assets=1.2e6)
    assert us["impairment_loss"] == approx(2e5) and us["implied_goodwill"] == approx(1e5)


# ----------------------------------------------------------------------------- income taxes

def test_deferred_tax_items_and_expense_reston():
    # L1 SS09-R31-income-taxes.md p495-497 Example 1 (Reston, 30%): DTL 257 (2004), 514 (2005);
    # 2005 tax expense = payable 4,327 + 257 = 4,584
    d04 = fra.deferred_tax_items([{"name": "equipment", "kind": "asset", "carrying_amount": 18000,
                                   "tax_base": 17143}], 0.30)
    d05 = fra.deferred_tax_items([{"name": "equipment", "kind": "asset", "carrying_amount": 16000,
                                   "tax_base": 14286}], 0.30)
    assert d04["deferred_tax_liability"] == approx(257, abs=0.5) and d05["deferred_tax_liability"] == approx(514, abs=0.5)
    r = fra.income_tax_expense(0.30 * 14423, d04["net_deferred_tax_liability"], d05["net_deferred_tax_liability"],
                               pretax_income=15280)
    assert r["income_tax_expense"] == approx(4584, abs=0.5)
    assert r["effective_tax_rate"] == approx(0.30, abs=1e-4)  # temporary differences only -> ETR = statutory


def test_deferred_tax_items_classification():
    # L1 SS09-R31 p504-505 Example 4: development costs (asset CA > TB) -> DTL; research costs (asset CA < TB) -> DTA;
    # rent received in advance (liability CA > TB) -> DTA
    r = fra.deferred_tax_items([
        {"name": "development", "kind": "asset", "carrying_amount": 2.5e6, "tax_base": 2.25e6},
        {"name": "research", "kind": "asset", "carrying_amount": 0, "tax_base": 375000},
        {"name": "rent_in_advance", "kind": "liability", "carrying_amount": 10e6, "tax_base": 0}], 0.30)
    dev, res, rent = r["items"]
    assert dev["deferred_tax_liability"] == approx(75000) and res["deferred_tax_asset"] == approx(112500)
    assert rent["deferred_tax_asset"] == approx(3e6)
    assert r["net_deferred_tax_liability"] == approx(75000 - 112500 - 3e6)


def test_tax_rate_change_effect():
    # L1 SS09-R31 p502: rate cut 30% -> 25%: 2006 DTL 643 (not 771); expense reduced 85 on the opening DTL;
    # current-year accelerated depreciation benefit 214 instead of 257
    r = fra.tax_rate_change_effect(1714, 2571, 0.30, 0.25)
    assert r["deferred_tax_end"] == approx(643, abs=0.5)
    assert r["revaluation_effect_on_expense"] == approx(-85.7, abs=0.05)
    assert r["deferred_tax_expense_on_current_year_difference"] == approx(214, abs=0.5)
    assert r["deferred_tax_expense_total"] == approx(r["deferred_tax_end"] - r["deferred_tax_begin_old_rate"])


def test_effective_tax_rate_reconciliation():
    # L1 SS09-R31 p527 problem 13: ETR = 56,860 / 189,167 = 30.1%
    assert fra.effective_tax_rate_reconciliation(189167, 0.35, reported_tax_expense=56860)["effective_tax_rate"] == approx(0.301, abs=5e-4)
    r = fra.effective_tax_rate_reconciliation(1000, 0.35, {"foreign rate differential": -40, "non-deductible": 10})
    assert r["effective_tax_rate"] == approx(0.32) and r["rate_effects"]["foreign rate differential"] == approx(-0.04)
    assert r["unexplained"] == approx(0)


# ----------------------------------------------------------------------------- long-term liabilities, leases

def test_bond_effective_interest_discount():
    # L1 SS09-R32-non-current-long-term-liabilities.md p535 Example 3 (Debond): proceeds 957,876; 2010 interest
    # 57,473, carrying 965,349; 2011 carrying 973,270; straight-line interest 58,424.80
    r = fra.bond_effective_interest_schedule(1e6, 0.05, 0.06, 5)
    s = r["schedule"]
    assert r["issue_price"] == approx(957876, abs=1)
    assert s[0]["interest_expense"] == approx(57473, abs=1) and s[0]["ending_carrying_amount"] == approx(965349, abs=1)
    assert s[1]["ending_carrying_amount"] == approx(973270, abs=1)
    assert s[-1]["ending_carrying_amount"] == approx(1e6)
    assert r["straight_line_interest_expense"] == approx(58424.8, abs=0.5)


def test_bond_effective_interest_premium():
    # L1 SS09-R32 p536 Example 4 (Prembond): proceeds 1,044,518; interest 41,781 / 41,452 / 41,110 / 40,754 / 40,385
    r = fra.bond_effective_interest_schedule(1e6, 0.05, 0.04, 5)
    assert r["issue_price"] == approx(1044518, abs=1)
    assert [round(x["interest_expense"]) for x in r["schedule"]] == [41781, 41452, 41110, 40754, 40385]


def test_finance_lease_lessee_capbs():
    # L1 SS09-R32 p550-552 Example 10 / L2 SS05-R18 p101-104 Example 11 (CAPBS): liability 100,000; interest
    # 7,132 / 4,977 / 2,607 / 0; year-1 total expense 32,132 vs operating lease 28,679
    r = fra.finance_lease_lessee(28679, 0.10, 4)
    s = r["schedule"]
    assert r["initial_liability"] == approx(100000, abs=1)
    assert [x["interest_expense"] for x in s] == [approx(7132, abs=1), approx(4977, abs=1), approx(2607, abs=1),
                                                    approx(0, abs=1)]
    assert s[0]["closing_liability"] == approx(71321, abs=1) and s[0]["total_expense"] == approx(32132, abs=1)
    assert sum(x["total_expense"] for x in s) == approx(4 * 28679, abs=1)  # same total expense as operating lease


def test_finance_lease_arrears_round_trip():
    r = fra.finance_lease_lessee(100, 0.08, 5, payments_in_advance=False)
    assert r["schedule"][-1]["closing_liability"] == approx(0, abs=1e-9)
    assert r["schedule"][0]["interest_expense"] == approx(0.08 * r["initial_liability"])


def test_lease_implicit_rate_cec():
    # L2 SS05-R18 p107-108 Example 12 (CEC): implicit rate on capital leases 7.245%
    pays = [1683, 1683, 1683, 1600] + [1586] * 7 + [454]
    assert fra.lease_implicit_rate(12208, pays)["implicit_rate"] == approx(0.07245, abs=5e-5)


def test_operating_lease_capitalization_cec():
    # L2 SS05-R18 p109: PV of operating leases 520,256 at 7.245%; D/E 4.73 -> 8.78
    pays = [66849, 66396, 66558, 65478, 63872] + [63872] * 7 + [27650]
    r = fra.operating_lease_capitalization(pays, 0.07245, total_liabilities=608854, total_equity=128586)
    assert r["lease_liability_added"] == approx(520256, abs=1)
    assert r["debt_to_equity_reported"] == approx(4.73, abs=0.005)
    assert r["debt_to_equity_adjusted"] == approx(8.78, abs=0.005)


# ----------------------------------------------------------------------------- pensions

def test_db_pension_funded_status():
    # L2 SS06-R20-employee-compensation...md p198 Example 1: ABC liability 1,843; DEF asset limited to ceiling 326
    abc = fra.db_pension_funded_status(6723, 4880)
    assert abc["net_pension_liability"] == approx(1843) and abc["funded_status"] == approx(-1843)
    assert fra.db_pension_funded_status(5485, 5998, asset_ceiling=326)["net_pension_asset"] == approx(326)


def test_pension_periodic_cost_xyz():
    # L2 SS06-R20 p229-234 problems 8-12 (XYZ SA): benefits paid 4,000; total periodic cost 1,020; IFRS P&L 530;
    # IFRS OCI 490; US GAAP P&L 20
    r = fra.pension_periodic_cost(42000, 39000, 0.07, 200, past_service_cost=120, actuarial_loss=460,
                                  actual_return=2700, expected_return_rate=0.08, employer_contributions=1000,
                                  obligation_end=41720)
    assert r["benefits_paid"] == approx(4000)
    assert r["plan_assets_end"] == approx(38700)
    assert r["total_periodic_cost"] == approx(1020)
    assert r["ifrs_pl_cost"] == approx(530) and r["ifrs_oci_remeasurements"] == approx(490)
    assert r["us_gaap_pl_cost"] == approx(20)
    # p214: periodic cost = change in funded status adjusted for contributions
    assert r["total_periodic_cost"] == approx((39000 - 42000) - r["funded_status_end"] + 1000)


def test_pension_corridor_amortization_formula():
    # L2 SS06-R20 p199-200 corridor: (610,000 - 10% x 5,000,000)/10 = 11,000 amortization -> US GAAP P&L adds it
    base = fra.pension_periodic_cost(5e6, 4.85e6, 0.05, 100, expected_return_rate=0.05)
    amort = fra.pension_periodic_cost(5e6, 4.85e6, 0.05, 100, expected_return_rate=0.05,
                                      amortization_actuarial_loss=(610000 - 500000) / 10)
    assert amort["us_gaap_pl_cost"] - base["us_gaap_pl_cost"] == approx(11000)


def test_pension_obligation_unit_credit():
    # L2 SS06-R20 p202-203 Example 2 scenario 1: final salary 60,198.56; lump sum 4,514.89; unit credit 902.98;
    # service cost 715.24 / 758.16 / 803.65 / 851.87 / 902.97; closing 715.24 / 1,516.31 / 2,410.94 / 3,407.47
    r = fra.pension_obligation_unit_credit(50000, 5, 0.0475, 0.06, 0.015)
    assert r["final_salary"] == approx(60198.56, abs=0.01) and r["lump_sum_benefit"] == approx(4514.89, abs=0.01)
    assert r["annual_unit_credit"] == approx(902.98, abs=0.01)
    s = r["schedule"]
    assert [x["current_service_cost"] for x in s] == [approx(v, abs=0.02) for v in (715.24, 758.16, 803.65, 851.87,
                                                                                     902.97)]
    assert [x["closing_obligation"] for x in s[:4]] == [approx(v, abs=0.02) for v in (715.24, 1516.31, 2410.94,
                                                                                      3407.47)]
    assert s[1]["interest_cost"] == approx(42.91, abs=0.01)


def test_pension_income_adjustment_sabmiller():
    # L2 SS06-R20 p215-217 Example 5 (SABMiller): operating profit 2,644; interest expense 918; interest income 363;
    # pretax 2,962
    r = fra.pension_income_adjustment(2619, 879, 316, 2929, total_pension_expense=36, current_service_cost=11,
                                      interest_cost=39, actual_return=47)
    assert r["adjusted_operating_profit"] == approx(2644) and r["adjusted_interest_expense"] == approx(918)
    assert r["adjusted_interest_income"] == approx(363) and r["adjusted_pretax_income"] == approx(2962)


def test_pension_cash_flow_adjustment_georace():
    # L2 SS06-R20 p217-218 Example 6 (GeoRace): excess 67 (48 after tax); CFO 6,161 -> 6,209; CFF -1,741 -> -1,789
    r = fra.pension_cash_flow_adjustment(504, 437, 0.287, cfo=6161, cff=-1741)
    assert r["excess_contribution_after_tax"] == approx(48, abs=0.5)
    assert r["adjusted_cfo"] == approx(6209, abs=0.5) and r["adjusted_cff"] == approx(-1789, abs=0.5)


# ----------------------------------------------------------------------------- EPS

def test_weighted_average_shares():
    # L1 SS08-R25-understanding-income-statements.md p204-205 Examples 13-14 (Angler): 1,125,000; 2-for-1 split
    # -> 2,250,000; basic EPS 2.04 and 1.02
    w = fra.weighted_average_shares(1e6, [[3, 200000], [9, -100000]])["weighted_average_shares"]
    assert w == approx(1125000)
    assert fra.eps_basic_diluted(2.5e6, w, preferred_dividends=200000)["basic_eps"] == approx(2.04, abs=0.005)
    w2 = fra.weighted_average_shares(1e6, [[3, 200000], [9, -100000]], stock_split_factor=2)["weighted_average_shares"]
    assert fra.eps_basic_diluted(2.5e6, w2, preferred_dividends=200000)["basic_eps"] == approx(1.02, abs=0.005)


def test_eps_if_converted_preferred():
    # L1 SS08-R25 p207 Example 15 (Bright-Warm): basic 3.10, diluted 2.92
    r = fra.eps_basic_diluted(1.75e6, 500000, 200000, convertible_preferred=[{"dividends": 200000, "shares": 100000}])
    assert r["basic_eps"] == approx(3.10) and r["diluted_eps"] == approx(2.92, abs=0.005)


def test_eps_if_converted_debt():
    # L1 SS08-R25 p209 Example 16 (Oppnox): basic 1.09, diluted 1.07 (numerator 752,100 / 700,000)
    r = fra.eps_basic_diluted(750000, 690000, convertible_debt=[{"interest": 3000, "shares": 10000}], tax_rate=0.30)
    assert r["basic_eps"] == approx(1.09, abs=0.005) and r["diluted_eps"] == approx(1.07, abs=0.005)
    assert r["diluted_numerator"] == approx(752100) and r["diluted_shares"] == approx(700000)


def test_eps_treasury_stock_method():
    # L1 SS08-R25 p210 Examples 17-18 (Hihotech): 10,909 incremental shares; basic 2.88, diluted 2.84
    r = fra.eps_basic_diluted(2.3e6, 800000, options=[{"count": 30000, "exercise_price": 35}], average_market_price=55)
    assert r["diluted_shares"] - 800000 == approx(10909, abs=1)
    assert r["basic_eps"] == approx(2.88, abs=0.005) and r["diluted_eps"] == approx(2.84, abs=0.005)
    # out-of-the-money options are antidilutive
    assert fra.eps_basic_diluted(2.3e6, 800000, options=[{"count": 30000, "exercise_price": 60}],
                                 average_market_price=55)["antidilutive_excluded"] == ["options_0"]


def test_eps_antidilutive_security():
    # L1 SS08-R25 p211 Example 19 (Dim-Cool): if-converted EPS 3.13 > basic 3.10 -> excluded, diluted = 3.10
    r = fra.eps_basic_diluted(1.75e6, 500000, 200000, convertible_preferred=[{"dividends": 200000, "shares": 60000}])
    assert r["diluted_eps"] == approx(3.10) and r["antidilutive_excluded"] == ["convertible_preferred_0"]


# ----------------------------------------------------------------------------- cash flows

def test_cfo_indirect_method():
    # L1 SS08-R27-understanding-cash-flow-statements.md p311 Example 7 (Pinkerly): net adjustments +5 -> CFO 35
    r = fra.cfo_indirect_method(30, noncash_charges=7, increase_operating_assets=15 - 3,
                                increase_operating_liabilities=10)
    assert r["total_adjustments"] == approx(5) and r["cfo"] == approx(35)


def test_free_cash_flow_acme():
    # L1 SS08-R27 p322: Acme FCFF = 2,606 + 258(1 - 0.34) - 538 = 2,238; FCFE = 2,606 - 538 - 500 = 1,568
    r = fra.free_cash_flow_fcff_fcfe(0.34, 538, interest_expense=258, net_borrowing=-500, cfo=2606)
    assert r["fcff"] == approx(2238, abs=0.5) and r["fcfe"] == approx(1568)


def test_free_cash_flow_starting_points_agree():
    # FCFF from NI, EBIT and EBITDA agree when the inputs are consistent (L2 Equity FCF identities):
    # EBIT = (NI + Int(1-t))/(1-t) with no other items; EBITDA = EBIT + Dep
    t, ni, i, dep, fc, wc = 0.30, 200.0, 50.0, 60.0, 100.0, 20.0
    ebit = (ni + i * (1 - t)) / (1 - t)
    r = fra.free_cash_flow_fcff_fcfe(t, fc, wc, interest_expense=i, net_income=ni, noncash_charges=dep, ebit=ebit,
                                     depreciation=dep, ebitda=ebit + dep, net_borrowing=10)
    assert r["fcff_from_net_income"] == approx(r["fcff_from_ebit"]) == approx(r["fcff_from_ebitda"]) == approx(175)
    assert r["fcfe"] == approx(175 - 35 + 10)


def test_cash_flow_ratios():
    # L1 SS08-R27 p323 Exhibit 15 definitions
    r = fra.cash_flow_ratios(500, revenue=2000, average_total_assets=2500, interest_paid=50, taxes_paid=100,
                             capex=250, dividends_paid=100, common_shares=100, preferred_dividends=20)
    assert r["cash_flow_to_revenue"] == approx(0.25) and r["cash_return_on_assets"] == approx(0.2)
    assert r["interest_coverage"] == approx(13.0) and r["reinvestment"] == approx(2.0)
    assert r["dividend_payment"] == approx(5.0) and r["cash_flow_per_share"] == approx(4.8)


# ----------------------------------------------------------------------------- earnings quality

def test_accruals_ratios_coca_cola():
    # L2 SS07-R23-evaluating-financial-reporting-quality.md p348-351 Example 1 (Coca-Cola 2006): NOA 18,912 and
    # 17,288; BS accruals 1,624, ratio 8.97%; CF accruals 823, ratio 4.55%
    r = fra.accruals_ratios(29963, 2590, 13043, 1314 + 3268, 29427, 4767, 13072, 1154 + 4546,
                            net_income=5080, cfo=5957, cfi=-1700)
    assert (r["noa"], r["noa_prior"]) == (approx(18912), approx(17288))
    assert r["aggregate_accruals_balance_sheet"] == approx(1624)
    assert r["accruals_ratio_balance_sheet"] == approx(0.0897, abs=5e-5)
    assert r["aggregate_accruals_cash_flow"] == approx(823)
    assert r["accruals_ratio_cash_flow"] == approx(0.0455, abs=5e-5)


def test_beneish_m_score():
    # Beneish (1999) FAJ coefficients: with every index 1 and TATA 0, M = -4.84 + .920 + .528 + .404 + .892 + .115
    # - .172 - .327 = -2.48 (below -1.78 -> not flagged)
    r = fra.beneish_m_score(indices=dict(dsri=1, gmi=1, aqi=1, sgi=1, depi=1, sgai=1, tata=0, lvgi=1))
    assert r["m_score"] == approx(-2.48) and not r["likely_manipulator"]
    base = dict(receivables=10, sales=100, cogs=60, current_assets=40, ppe=50, total_assets=100, depreciation=5,
                sga=10, net_income=8, cfo=8, current_liabilities=20, long_term_debt=30)
    assert fra.beneish_m_score(base, base)["m_score"] == approx(-2.48)
    # receivables doubling relative to sales (DSRI 2) adds 0.920
    hot = dict(base, receivables=20)
    r2 = fra.beneish_m_score(hot, base)
    assert r2["indices"]["dsri"] == approx(2.0) and r2["m_score"] == approx(-2.48 + 0.92)
    assert r2["likely_manipulator"]


def test_altman_z_score():
    # L1 SS08-R28 p393 Altman (1968) Z formula; cut-off 1.81
    r = fra.altman_z_score(10, 20, 15, 60, 50, 120, 100)
    assert r["z_score"] == approx(1.2 * 0.1 + 1.4 * 0.2 + 3.3 * 0.15 + 0.6 * 1.2 + 1.0 * 1.2)
    assert r["zone"] == "grey"
    assert fra.altman_z_score(-10, -20, -5, 10, 80, 50, 100)["zone"] == "distress"


# ----------------------------------------------------------------------------- intercorporate investments

def test_equity_method_branch():
    # L2 SS06-R19-intercorporate-investments.md p144 Example 2 (Branch/Williams, 20%): balance 2012 = 310,000
    r = fra.equity_method_investment(200000, 0.20, [200000, 300000, 400000], [50000, 100000, 200000])
    assert r["carrying_amount"] == approx(310000)
    assert [y["equity_income"] for y in r["years"]] == [approx(40000), approx(60000), approx(80000)]


def test_equity_method_parker_goodwill():
    # L2 SS06-R19 p149-150 Example 4 (Parker/Prince, 30%): goodwill 50,000; amortization 9,000; balance 506,000
    r = fra.equity_method_investment(500000, 0.30, [100000], [50000], investee_book_value=1.2e6,
                                     fair_value_adjustments=[{"amount": 300000, "useful_life": 10}])
    assert r["goodwill"] == approx(50000) and r["annual_amortization"] == approx(9000)
    assert r["carrying_amount"] == approx(506000) and r["years"][0]["equity_income"] == approx(21000)


def test_equity_method_unrealized_profit():
    # L2 SS06-R19 p153-154 Example 6 (Jones/Jason downstream sale): equity income 188,000 (2011), 201,000 (2012)
    r = fra.equity_method_investment(1e6, 0.25, [800000, 820000],
                                     fair_value_adjustments=[{"amount": 320000, "useful_life": 10}],  # 8,000/yr
                                     unrealized_profit=[16000, 0])
    assert [y["equity_income"] for y in r["years"]] == [approx(188000), approx(201000)]


def test_acquisition_goodwill_full_partial():
    # L2 SS06-R19 p164-165 Example 9: full goodwill 40,000 / NCI 20,000; partial goodwill 36,000 / NCI 16,000
    r = fra.acquisition_goodwill(180000, 0.90, 160000, fair_value_of_subsidiary=200000)
    assert r["full_goodwill"] == approx(40000) and r["full_goodwill_nci"] == approx(20000)
    assert r["partial_goodwill"] == approx(36000) and r["partial_goodwill_nci"] == approx(16000)


# ----------------------------------------------------------------------------- multinational operations

def test_fx_transaction_gain_loss():
    # L2 SS06-R21-multinational-operations.md p238-239 Example 1 (FinnCo): MXN 100,000 payable, EUR loss 190
    r = fra.fx_transaction_gain_loss(100000, 0.0684, 0.0703, "payable")
    assert r["gain_loss"] == approx(-190) and r["home_amount_at_settlement"] == approx(7030)
    split = fra.fx_transaction_gain_loss(100000, 0.0684, 0.0703, "receivable", rate_at_balance_sheet=0.0690)
    assert split["gain_loss_to_balance_sheet_date"] + split["gain_loss_after_balance_sheet_date"] == approx(190)


def test_fx_translation_current_rate_cta():
    # L2 SS06-R21 p274-276 Example 6 (Canadaco, C$ strengthens): translation adjustment +202,000
    # = 1,500,000 x 0.10 + 1,180,000 x (0.80 - 0.75) - 350,000 x (0.80 - 0.78)
    r = fra.fx_translation_adjustment(1500000, 0.70, 0.80, [[1180000, 0.75], [-350000, 0.78]])
    assert r["translation_gain_loss"] == approx(202000)
    weak = fra.fx_translation_adjustment(1500000, 0.70, 0.60, [[1180000, 0.65], [-350000, 0.62]])
    assert weak["translation_gain_loss"] == approx(-202000)


def test_fx_translation_temporal_remeasurement():
    # L2 SS06-R21 p270-273 Example 5: temporal method loss 245,000 (Case A, net monetary liabilities) and gain
    # 55,000 (Case B, net monetary assets); difference 300,000 = note payable 3,000,000 x 0.10
    flows = [[12e6, 0.75], [-10.2e6, 0.74], [-(750000 + 270000 + 500000), 0.75], [-350000, 0.78]]
    a = fra.fx_translation_adjustment(1.5e6 - 3e6, 0.70, 0.80, flows)
    b = fra.fx_translation_adjustment(1.5e6, 0.70, 0.80, flows)
    assert a["translation_gain_loss"] == approx(-245000) and b["translation_gain_loss"] == approx(55000)
    assert a["ending_exposure"] == approx(980000 + 900000 - 450000 - 3e6)


def test_fx_translation_exposure_sign_rules():
    # L2 SS06-R21 3.4: current rate exposure = net assets (gain in OCI when FC appreciates); temporal exposure =
    # net monetary position (Canadaco net monetary liability -> loss in NI when C$ appreciates)
    cr = fra.fx_translation_exposure("current_rate", "appreciates", total_assets=5.78e6, total_liabilities=3.45e6)
    assert cr["translation_result"] == "gain" and cr["reported_in"].startswith("OCI")
    tm = fra.fx_translation_exposure("temporal", "appreciates", monetary_assets=1.88e6, monetary_liabilities=3.45e6)
    assert tm["position"] == "net liability" and tm["translation_result"] == "loss"
    assert fra.fx_translation_exposure("temporal", "depreciates", monetary_assets=1.88e6,
                                       monetary_liabilities=3.45e6)["translation_result"] == "gain"


# ----------------------------------------------------------------------------- registry / validation

def test_all_fra_functions_registered():
    names = {n for n, f in REGISTRY.items() if f.topic == "fra"}
    assert len(names) >= 40
    for n in names:
        assert REGISTRY[n].curriculum and REGISTRY[n].formula


def test_input_validation():
    with pytest.raises(ValueError):
        fra.depreciation_schedule(100, 200, 5)
    with pytest.raises(ValueError):
        fra.depreciation_schedule(100, 0, 5, "magic")
    with pytest.raises(ValueError):
        fra.asset_impairment(100, "us_gaap", undiscounted_cash_flows=50)
    with pytest.raises(ValueError):
        fra.eps_basic_diluted(100, 0)
    with pytest.raises(ValueError):
        fra.deferred_tax_items([{"kind": "equity", "carrying_amount": 1, "tax_base": 0}], 0.3)
    with pytest.raises(ValueError):
        fra.inventory_writedown(5, 4, "gaap")


def test_beneish_zero_denominator_is_neutral():
    # balance sheet with only current assets and PP&E: the asset-quality index has a zero denominator
    base = dict(receivables=100, sales=1000, cogs=600, current_assets=500, ppe=500, total_assets=1000,
                depreciation=50, sga=100, net_income=80, cfo=90, current_liabilities=200, long_term_debt=300)
    cur = dict(base, receivables=130, sales=1100)
    out = fra.beneish_m_score(current=cur, prior=base)
    assert out["indices"]["aqi"] == 1.0 and any("aqi" in n for n in out["notes"])


# ---------------------------------------------------------------- regressions found by independent verification

@pytest.mark.parametrize("call", [
    lambda: fra.pension_obligation_unit_credit(50000, 5.5, 0.0475, 0.06, 0.015),
    lambda: fra.depreciation_schedule(2300, 100, 4.5),
    lambda: fra.bond_effective_interest_schedule(1e6, 0.05, 0.06, 5.5),
])
def test_fractional_periods_rejected_not_truncated(call):
    with pytest.raises(ValueError):
        call()


def test_payables_turnover_reports_its_basis():
    # without purchases the engine falls back to COGS; the result must say so
    assert fra.financial_ratios(cogs=3000, payables=300)["payables_turnover_basis"] == "cogs"
    r = fra.financial_ratios(cogs=3000, purchases=3300, payables=300)
    assert r["payables_turnover"] == approx(11.0) and r["payables_turnover_basis"] == "purchases"
    assert "payables_turnover_basis" not in fra.financial_ratios(current_assets=2, current_liabilities=1)
