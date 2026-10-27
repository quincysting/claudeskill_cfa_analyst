"""Financial reporting and analysis (CFA Level I-II): ratios, inventories, long-lived assets, income taxes,
long-term liabilities and leases, pensions, EPS, cash flows, earnings quality, intercorporate investments and
multinational operations.

Amounts are in any one currency unit; rates and ratios are decimals. Inputs are positive magnitudes unless a
parameter says otherwise.
"""
from __future__ import annotations

from ._registry import formula
from .corpfin import _irr, _npv

_T = "fra"
L1RA = "L1 FRA R28: Financial Analysis Techniques"
L1INV, L2INV = "L1 FRA R29: Inventories", "L2 FRA R17: Inventories: Implications for Financial Statements and Ratios"
L1LLA, L2LLA = "L1 FRA R30: Long-Lived Assets", "L2 FRA R18: Long-Lived Assets: Implications for FS and Ratios"
L1TAX = "L1 FRA R31: Income Taxes"
L1LTL = "L1 FRA R32: Non-Current (Long-Term) Liabilities"
L2PEN = "L2 FRA R20: Employee Compensation: Post-Employment and Share-Based"
L1EPS = "L1 FRA R25: Understanding Income Statements (EPS)"
L1CF = "L1 FRA R27: Understanding Cash Flow Statements"
L2EQ = "L2 FRA R23: Evaluating Financial Reporting Quality"
L2ICI = "L2 FRA R19: Intercorporate Investments"
L2FX = "L2 FRA R21: Multinational Operations"


def _div(a, b):
    return None if a is None or b is None or b == 0 else a / b


def _avg(end, begin):
    if end is None:
        return None
    return end if begin is None else (end + begin) / 2


def _put(d: dict, key: str, value) -> None:
    if value is not None:
        d[key] = value


# ----------------------------------------------------------------------------- ratio analysis

@formula(_T, formula="turnovers = flow / average stock; days = days in period / turnover; CCC = DOH + DSO - DPO; "
                     "liquidity, solvency, coverage and margin ratios per CFA definitions", curriculum=L1RA)
def financial_ratios(revenue: float | None = None, cogs: float | None = None, ebit: float | None = None,
                     interest_expense: float | None = None, pretax_income: float | None = None,
                     net_income: float | None = None, preferred_dividends: float = 0.0,
                     lease_payments: float | None = None, purchases: float | None = None,
                     cash: float | None = None, marketable_securities: float = 0.0, receivables: float | None = None,
                     inventory: float | None = None, payables: float | None = None,
                     current_assets: float | None = None, current_liabilities: float | None = None,
                     ppe: float | None = None, total_assets: float | None = None,
                     total_debt: float | None = None, total_equity: float | None = None,
                     receivables_begin: float | None = None, inventory_begin: float | None = None,
                     payables_begin: float | None = None, current_assets_begin: float | None = None,
                     current_liabilities_begin: float | None = None, ppe_begin: float | None = None,
                     total_assets_begin: float | None = None, total_debt_begin: float | None = None,
                     total_equity_begin: float | None = None, daily_cash_expenditures: float | None = None,
                     days_in_period: float = 365.0) -> dict:
    """Activity, liquidity, solvency, coverage and profitability ratios from statement line items.

    Every ratio whose inputs are present is returned; the rest are skipped. When a `*_begin` balance is given, the
    ratio uses the average of beginning and ending balances, otherwise the ending balance. total_debt means
    interest-bearing debt; ebit is operating income; ppe is net PP&E. Names match the canonical items of
    `xlsx_tool.py extract`. Quick ratio = (cash + marketable securities + receivables)/CL.

    Payables turnover = purchases / payables. When `purchases` is not given it uses COGS as a proxy (the usual
    shortcut when purchases are not disclosed; purchases = COGS + ending inventory - beginning inventory), and the
    result's `payables_turnover_basis` says which was used: "purchases" or "cogs".

    Example (Kaplan L1 sample company):
        financial_ratios(revenue=4000, cogs=3000, ebit=350, interest_expense=50, net_income=200,
                         current_assets=620, current_liabilities=325, ...) -> current_ratio 1.91, coverage 7.0
    """
    o: dict = {}
    days = days_in_period
    inv, rec, pay = _avg(inventory, inventory_begin), _avg(receivables, receivables_begin), _avg(payables,
                                                                                                 payables_begin)
    ta, eq, fa = _avg(total_assets, total_assets_begin), _avg(total_equity, total_equity_begin), _avg(
        ppe, ppe_begin)
    wc_end = None if current_assets is None or current_liabilities is None else current_assets - current_liabilities
    wc_begin = None if current_assets_begin is None or current_liabilities_begin is None else \
        current_assets_begin - current_liabilities_begin
    wc = _avg(wc_end, wc_begin)
    # activity
    _put(o, "inventory_turnover", _div(cogs, inv))
    _put(o, "days_inventory_on_hand", _div(days, o.get("inventory_turnover")))
    _put(o, "receivables_turnover", _div(revenue, rec))
    _put(o, "days_sales_outstanding", _div(days, o.get("receivables_turnover")))
    _put(o, "payables_turnover", _div(purchases if purchases is not None else cogs, pay))
    if "payables_turnover" in o:
        o["payables_turnover_basis"] = "purchases" if purchases is not None else "cogs"
    _put(o, "days_payables", _div(days, o.get("payables_turnover")))
    _put(o, "working_capital_turnover", _div(revenue, wc))
    _put(o, "fixed_asset_turnover", _div(revenue, fa))
    _put(o, "total_asset_turnover", _div(revenue, ta))
    if all(k in o for k in ("days_inventory_on_hand", "days_sales_outstanding", "days_payables")):
        o["cash_conversion_cycle"] = o["days_inventory_on_hand"] + o["days_sales_outstanding"] - o["days_payables"]
    # liquidity
    _put(o, "current_ratio", _div(current_assets, current_liabilities))
    if cash is not None:
        _put(o, "cash_ratio", _div(cash + marketable_securities, current_liabilities))
        if receivables is not None:
            _put(o, "quick_ratio", _div(cash + marketable_securities + receivables, current_liabilities))
            _put(o, "defensive_interval_days", _div(cash + marketable_securities + receivables,
                                                     daily_cash_expenditures))
    # solvency and coverage
    _put(o, "debt_to_assets", _div(total_debt, total_assets))
    if total_debt is not None and total_equity is not None:
        _put(o, "debt_to_capital", _div(total_debt, total_debt + total_equity))
    _put(o, "debt_to_equity", _div(total_debt, total_equity))
    _put(o, "financial_leverage", _div(ta, eq))
    _put(o, "interest_coverage", _div(ebit, interest_expense))
    if lease_payments is not None and ebit is not None and interest_expense is not None:
        _put(o, "fixed_charge_coverage", _div(ebit + lease_payments, interest_expense + lease_payments))
    # profitability
    if revenue and cogs is not None:
        o["gross_profit_margin"] = (revenue - cogs) / revenue
    _put(o, "operating_profit_margin", _div(ebit, revenue))
    _put(o, "pretax_margin", _div(pretax_income, revenue))
    _put(o, "net_profit_margin", _div(net_income, revenue))
    _put(o, "roa", _div(net_income, ta))
    _put(o, "operating_roa", _div(ebit, ta))
    if total_debt is not None and total_equity is not None:
        cap_begin = None if total_debt_begin is None or total_equity_begin is None else \
            total_debt_begin + total_equity_begin
        _put(o, "return_on_total_capital", _div(ebit, _avg(total_debt + total_equity, cap_begin)))
    _put(o, "roe", _div(net_income, eq))
    if net_income is not None:
        _put(o, "return_on_common_equity", _div(net_income - preferred_dividends, eq))
    if not o:
        raise ValueError("not enough inputs to compute any ratio")
    return o


@formula(_T, formula="ROE = NI/Rev x Rev/TA x TA/E = NI/EBT x EBT/EBIT x EBIT/Rev x Rev/TA x TA/E", curriculum=L1RA)
def dupont_roe(net_income: float, revenue: float, total_assets: float, total_equity: float,
               pretax_income: float | None = None, ebit: float | None = None) -> dict:
    """Three-way DuPont decomposition of ROE, and five-way when pretax income and EBIT are given.

    Pass average assets and equity if you want average-based ratios.

    Example:
        dupont_roe(20, 500, 250, 150, pretax_income=30, ebit=35) -> roe 0.1333, tax_burden 0.667,
        interest_burden 0.857, ebit_margin 0.07, asset_turnover 2.0, leverage 1.667
    """
    if min(revenue, total_assets, total_equity) <= 0:
        raise ValueError("revenue, total_assets and total_equity must be positive")
    o = {"roe": net_income / total_equity, "net_profit_margin": net_income / revenue,
         "asset_turnover": revenue / total_assets, "financial_leverage": total_assets / total_equity,
         "roa": net_income / total_assets}
    if pretax_income is not None and ebit is not None:
        if pretax_income == 0 or ebit == 0:
            raise ValueError("pretax_income and ebit must be non-zero for the five-way decomposition")
        o.update(tax_burden=net_income / pretax_income, interest_burden=pretax_income / ebit,
                 ebit_margin=ebit / revenue)
    return o


@formula(_T, formula="common-size item = item / base (revenue or total assets)", curriculum=L1RA)
def common_size_statement(items: dict, base) -> dict:
    """Vertical common-size statement: every line item divided by a base (a key of items, or a number).

    Example:
        common_size_statement({"revenue": 4000, "cogs": 3000, "net_income": 200}, "revenue") -> cogs 0.75
    """
    b = items[base] if isinstance(base, str) else base
    if not b:
        raise ValueError("base must be non-zero")
    return {"common_size": {k: v / b for k, v in items.items()}, "base": b}


@formula(_T, formula="operating cycle = DOH + DSO; cash conversion cycle = DOH + DSO - DPO",
         curriculum="L1 Corporate Finance R40: Working Capital Management (2.2); " + L1RA)
def cash_conversion_cycle(days_inventory: float, days_receivables: float, days_payables: float) -> dict:
    """Cash conversion (net operating) cycle and operating cycle from the three day counts.

    Example:
        cash_conversion_cycle(41, 39, 34.3) -> 45.7
    """
    return {"cash_conversion_cycle": days_inventory + days_receivables - days_payables,
            "operating_cycle": days_inventory + days_receivables}


# ----------------------------------------------------------------------------- inventories

def _consume(layers, units, from_end):
    seq = list(reversed(layers)) if from_end else list(layers)
    cogs, left, remaining = 0.0, units, []
    for q, c in seq:
        take = min(q, left)
        cogs += take * c
        left -= take
        if q - take > 0:
            remaining.append([q - take, c])
    return cogs, (list(reversed(remaining)) if from_end else remaining)


@formula(_T, formula="periodic FIFO / LIFO / weighted average: COGS + ending inventory = goods available",
         curriculum=L1INV + " (3.1-3.5)")
def inventory_cost_flow(purchases: list, units_sold: float, beginning_inventory: list | None = None,
                        sale_price: float | None = None, method: str = "all") -> dict:
    """COGS, ending inventory and gross profit from cost layers under a periodic inventory system.

    Args:
        purchases: [[units, unit_cost], ...] in purchase order.
        beginning_inventory: optional [[units, unit_cost], ...] layers, oldest first.
        method: "fifo", "lifo", "weighted_average" or "all" (keys prefixed by method).

    Example:
        inventory_cost_flow([[100000, 110], [200000, 100], [300000, 90]], 520000)
        -> fifo_cogs 50.8m, lifo_cogs 49.2m, weighted_average_cogs 50.27m
    """
    layers = [list(map(float, x)) for x in (beginning_inventory or []) + list(purchases)]
    units = sum(q for q, _ in layers)
    available = sum(q * c for q, c in layers)
    if units_sold < 0 or units_sold > units + 1e-9:
        raise ValueError(f"units_sold must be between 0 and units available ({units})")
    methods = ["fifo", "lifo", "weighted_average"] if method == "all" else [method]
    o: dict = {}
    for m in methods:
        if m == "weighted_average":
            cogs, end_layers = units_sold * available / units, [[units - units_sold, available / units]]
        elif m in ("fifo", "lifo"):
            cogs, end_layers = _consume(layers, units_sold, from_end=(m == "lifo"))
        else:
            raise ValueError("method must be fifo, lifo, weighted_average or all")
        p = "" if method != "all" else m + "_"
        o[p + "cogs"] = cogs
        o[p + "ending_inventory"] = available - cogs
        if sale_price is not None:
            o[p + "gross_profit"] = units_sold * sale_price - cogs
        if method != "all":
            o["ending_layers"] = end_layers
    o["goods_available_for_sale"] = available
    return o


@formula(_T, formula="FIFO inventory = LIFO inventory + LIFO reserve; FIFO COGS = LIFO COGS - change in reserve; "
                     "FIFO NI = LIFO NI + change in reserve x (1 - t)", curriculum=L2INV + " (3.1)")
def lifo_to_fifo(lifo_inventory: float, lifo_reserve_end: float, lifo_reserve_begin: float | None = None,
                 lifo_cogs: float | None = None, net_income: float | None = None, tax_rate: float = 0.0) -> dict:
    """Restate LIFO statements to FIFO with the LIFO reserve.

    Returns fifo_inventory, change_in_lifo_reserve, fifo_cogs, fifo_net_income, current-year extra taxes,
    cumulative taxes on the reserve (deferred tax / cash taxes on a switch) and the equity (retained earnings)
    increase. A falling reserve flags a possible LIFO liquidation.

    Example:
        lifo_to_fifo(8781, 3183, 2617, lifo_cogs=38415, net_income=3557, tax_rate=0.20)
        -> fifo_inventory 11964, fifo_cogs 37849, fifo_net_income 4009.8
    """
    o = {"fifo_inventory": lifo_inventory + lifo_reserve_end,
         "cumulative_tax_on_reserve": lifo_reserve_end * tax_rate,
         "equity_increase": lifo_reserve_end * (1 - tax_rate)}
    if lifo_reserve_begin is not None:
        chg = lifo_reserve_end - lifo_reserve_begin
        o.update(change_in_lifo_reserve=chg, current_year_tax_increase=chg * tax_rate,
                 lifo_liquidation_indicated=chg < 0)
        if lifo_cogs is not None:
            o["fifo_cogs"] = lifo_cogs - chg
        if net_income is not None:
            o["fifo_net_income"] = net_income + chg * (1 - tax_rate)
    return o


@formula(_T, formula="liquidation profit = sum units liquidated x (current cost - old layer cost)",
         curriculum=L2INV + " (3.2)")
def lifo_liquidation_profit(liquidated_layers: list, replacement_unit_cost: float, tax_rate: float = 0.0) -> dict:
    """Gross profit created when old, cheaper LIFO layers flow into COGS (an unsustainable earnings boost).

    Args:
        liquidated_layers: [[units, old_unit_cost], ...] drawn from beginning inventory.

    Example:
        lifo_liquidation_profit([[1000, 100]], 115) -> pretax 15000
    """
    pre = sum(q * (replacement_unit_cost - c) for q, c in liquidated_layers)
    return {"pretax_liquidation_profit": pre, "after_tax_liquidation_profit": pre * (1 - tax_rate)}


@formula(_T, formula="IFRS: min(cost, NRV); US GAAP LCM: min(cost, market), market = median(RC, NRV, NRV - normal "
                     "profit)", curriculum=L1INV + " (4); " + L2INV + " (5)")
def inventory_writedown(cost: float, net_realisable_value: float, standard: str = "ifrs",
                        replacement_cost: float | None = None, normal_profit: float | None = None) -> dict:
    """Inventory carrying amount and write-down under IFRS (lower of cost and NRV) or US GAAP.

    Args:
        standard: "ifrs" or "us_gaap_lcnrv" (lower of cost and NRV), or "us_gaap_lcm" (lower of cost or market,
            used for LIFO and retail inventory and in the 2013-2014 curriculum).
        replacement_cost, normal_profit: needed for "us_gaap_lcm" (normal_profit is an amount).

    IFRS allows reversal of past write-downs (limited to the original write-down); US GAAP does not.

    Example:
        inventory_writedown(5.2, 4.9) -> 0.3; inventory_writedown(5.2, 4.9, "us_gaap_lcm", 4.7, 0.5) -> 0.5
    """
    if standard in ("ifrs", "us_gaap_lcnrv"):
        value = net_realisable_value
    elif standard == "us_gaap_lcm":
        if replacement_cost is None:
            raise ValueError("us_gaap_lcm needs replacement_cost")
        floor = net_realisable_value - (normal_profit if normal_profit is not None else float("inf"))
        value = min(max(replacement_cost, floor), net_realisable_value)
    else:
        raise ValueError("standard must be ifrs, us_gaap_lcnrv or us_gaap_lcm")
    carrying = min(cost, value)
    return {"writedown": cost - carrying, "carrying_amount": carrying, "measurement_value": value}


# ----------------------------------------------------------------------------- long-lived assets

@formula(_T, formula="SL: (C - S)/n; DDB: BV x k/n (switch to SL when larger); SYD: (C - S) x remaining/SYD; "
                     "UOP: (C - S)/total units x units", curriculum=L1LLA + " (3.1)")
def depreciation_schedule(cost: float, salvage_value: float = 0.0, useful_life: int | None = None,
                          method: str = "straight_line", declining_rate: float = 2.0,
                          switch_to_straight_line: bool = True, units_per_period: list | None = None,
                          total_units: float | None = None) -> dict:
    """Depreciation schedule: straight line, declining balance (DDB by default), sum-of-years' digits, or UOP.

    Declining balance never takes book value below salvage; it switches to straight line when that is larger,
    and in any case the final year takes book value to salvage (curriculum convention).

    Example:
        depreciation_schedule(2300, 100, 4, "declining_balance") -> [1150, 575, 287.5, 187.5]
    """
    base = cost - salvage_value
    if base < 0:
        raise ValueError("salvage_value exceeds cost")
    if method == "units_of_production":
        if not units_per_period or not total_units:
            raise ValueError("units_of_production needs units_per_period and total_units")
        dep = [base * u / total_units for u in units_per_period]
    else:
        if not useful_life or useful_life < 1 or int(useful_life) != useful_life:
            raise ValueError("useful_life must be a positive integer")
        n = int(useful_life)
        if method == "straight_line":
            dep = [base / n] * n
        elif method == "sum_of_years_digits":
            syd = n * (n + 1) / 2
            dep = [base * (n - t) / syd for t in range(n)]
        elif method == "declining_balance":
            dep, bv = [], cost
            for t in range(n):
                d = bv * declining_rate / n
                sl = (bv - salvage_value) / (n - t)
                if t == n - 1 or (switch_to_straight_line and sl > d):
                    d = sl
                d = min(d, bv - salvage_value)
                dep.append(d)
                bv -= d
        else:
            raise ValueError("method must be straight_line, declining_balance, sum_of_years_digits or "
                             "units_of_production")
    rows, acc = [], 0.0
    for t, d in enumerate(dep, 1):
        rows.append({"period": t, "beginning_book_value": cost - acc, "depreciation": d,
                     "accumulated_depreciation": acc + d, "ending_book_value": cost - acc - d})
        acc += d
    return {"depreciation": dep, "schedule": rows, "total_depreciation": acc}


@formula(_T, formula="IFRS: capitalized = interest during construction - income on temporarily invested borrowings; "
                     "US GAAP: no offset", curriculum=L1LLA + " (2.1)")
def capitalized_borrowing_costs(borrowing: float, interest_rate: float, construction_years: float,
                                investment_income: float = 0.0) -> dict:
    """Borrowing costs capitalized into a self-constructed asset under IFRS and US GAAP.

    Example:
        capitalized_borrowing_costs(1e6, 0.10, 2, 20000) -> ifrs 180000, us_gaap 200000
    """
    interest = borrowing * interest_rate * construction_years
    return {"ifrs_capitalized": interest - investment_income, "us_gaap_capitalized": interest,
            "interest_during_construction": interest}


@formula(_T, formula="adjusted coverage = (EBIT + depreciation of capitalized interest) / "
                     "(interest expense + capitalized interest)", curriculum=L2LLA + " (2.1)")
def capitalized_interest_coverage(ebit: float, interest_expense: float, capitalized_interest: float = 0.0,
                                  depreciation_of_capitalized_interest: float = 0.0) -> dict:
    """Interest coverage as reported and adjusted for capitalized interest (rating-agency approach).

    Example:
        capitalized_interest_coverage(268800, 17047, 6000, 365) -> reported 15.77, adjusted 11.68
    """
    if interest_expense + capitalized_interest <= 0:
        raise ValueError("interest must be positive")
    return {"adjusted_interest_coverage": (ebit + depreciation_of_capitalized_interest)
            / (interest_expense + capitalized_interest),
            "reported_interest_coverage": ebit / interest_expense if interest_expense else None}


@formula(_T, formula="capitalize: expense = depreciation, outflow in CFI; expense: full amount hits NI and CFO",
         curriculum=L2LLA + " (2)")
def capitalize_vs_expense(amount: float, useful_life: int, tax_rate: float = 0.0, salvage_value: float = 0.0) -> dict:
    """Year-by-year effect of capitalizing (straight line) versus expensing one expenditure, relative to not spending.

    Assumes the same treatment for tax and books. Returns per-year lists of net income, CFO and CFI effects and the
    asset carrying amount for both choices. Totals over the life are equal; timing and cash-flow classification are not.

    Example:
        capitalize_vs_expense(900, 3, 0.30) -> capitalize NI effect -210/yr, CFO +90/yr, CFI -900 in year 1;
        expense NI -630 in year 1
    """
    if useful_life < 1:
        raise ValueError("useful_life must be at least 1")
    dep = (amount - salvage_value) / useful_life
    n = useful_life
    return {
        "capitalize_net_income_effect": [-dep * (1 - tax_rate)] * n,
        "capitalize_cfo_effect": [dep * tax_rate] * n,
        "capitalize_cfi_effect": [-amount] + [0.0] * (n - 1),
        "capitalize_asset_carrying": [amount - dep * t for t in range(1, n + 1)],
        "expense_net_income_effect": [-amount * (1 - tax_rate)] + [0.0] * (n - 1),
        "expense_cfo_effect": [-amount * (1 - tax_rate)] + [0.0] * (n - 1),
        "depreciation": dep,
    }


@formula(_T, formula="average age = accumulated dep / dep expense; total life = gross PPE / dep expense; "
                     "remaining life = net PPE / dep expense", curriculum=L2LLA + " (4); L1 FRA R35")
def fixed_asset_age(gross_ppe: float, accumulated_depreciation: float, depreciation_expense: float) -> dict:
    """Average age, average depreciable life and remaining useful life of the fixed asset base (straight line).

    Example:
        fixed_asset_age(2.8, 1.6, 0.4) -> average_age 4, total_useful_life 7, remaining_useful_life 3
    """
    if depreciation_expense <= 0:
        raise ValueError("depreciation_expense must be positive")
    return {"average_age": accumulated_depreciation / depreciation_expense,
            "total_useful_life": gross_ppe / depreciation_expense,
            "remaining_useful_life": (gross_ppe - accumulated_depreciation) / depreciation_expense,
            "percent_depreciated": accumulated_depreciation / gross_ppe}


@formula(_T, formula="IFRS: loss = CA - max(FV less costs to sell, value in use); US GAAP: if CA > undiscounted CF, "
                     "loss = CA - FV", curriculum=L1LLA + " (5)")
def asset_impairment(carrying_amount: float, standard: str = "ifrs", fair_value_less_costs_to_sell: float | None = None,
                     value_in_use: float | None = None, undiscounted_cash_flows: float | None = None,
                     fair_value: float | None = None) -> dict:
    """Impairment of property, plant and equipment held for use (IFRS one-step, US GAAP recoverability test).

    Example:
        asset_impairment(18000, "ifrs", 15000, 16000) -> loss 2000;
        asset_impairment(18000, "us_gaap", undiscounted_cash_flows=19000, fair_value=17000) -> loss 0
    """
    if standard == "ifrs":
        cands = [x for x in (fair_value_less_costs_to_sell, value_in_use) if x is not None]
        if not cands:
            raise ValueError("ifrs needs fair_value_less_costs_to_sell and/or value_in_use")
        recoverable = max(cands)
        loss = max(0.0, carrying_amount - recoverable)
        return {"impairment_loss": loss, "new_carrying_amount": carrying_amount - loss,
                "recoverable_amount": recoverable}
    if standard == "us_gaap":
        if undiscounted_cash_flows is None:
            raise ValueError("us_gaap needs undiscounted_cash_flows (and fair_value if impaired)")
        impaired = carrying_amount > undiscounted_cash_flows
        if impaired and fair_value is None:
            raise ValueError("asset fails the recoverability test; fair_value is needed to measure the loss")
        loss = carrying_amount - fair_value if impaired else 0.0
        return {"impairment_loss": loss, "new_carrying_amount": carrying_amount - loss, "fails_recoverability": impaired}
    raise ValueError("standard must be ifrs or us_gaap")


@formula(_T, formula="IFRS: loss = CA(unit) - recoverable, goodwill first; US GAAP (two-step): loss = goodwill - "
                     "(FV unit - FV identifiable net assets)", curriculum=L2ICI + " (6.4.4)")
def goodwill_impairment(carrying_amount: float, goodwill: float, standard: str = "ifrs",
                        recoverable_amount: float | None = None, fair_value: float | None = None,
                        fair_value_identifiable_net_assets: float | None = None) -> dict:
    """Goodwill impairment for a cash-generating unit (IFRS) or reporting unit (US GAAP two-step test).

    Example:
        goodwill_impairment(1.4e6, 3e5, "ifrs", recoverable_amount=1.3e6) -> 100000;
        goodwill_impairment(1.4e6, 3e5, "us_gaap", fair_value=1.3e6, fair_value_identifiable_net_assets=1.2e6)
        -> 200000
    """
    if standard == "ifrs":
        if recoverable_amount is None:
            raise ValueError("ifrs needs recoverable_amount")
        loss = max(0.0, carrying_amount - recoverable_amount)
        gw = min(loss, goodwill)
        return {"impairment_loss": loss, "goodwill_impairment": gw, "other_asset_impairment": loss - gw}
    if standard == "us_gaap":
        if fair_value is None or fair_value_identifiable_net_assets is None:
            raise ValueError("us_gaap needs fair_value and fair_value_identifiable_net_assets")
        if fair_value >= carrying_amount:
            return {"impairment_loss": 0.0, "goodwill_impairment": 0.0, "implied_goodwill": None}
        implied = max(0.0, fair_value - fair_value_identifiable_net_assets)
        loss = max(0.0, goodwill - implied)
        return {"impairment_loss": loss, "goodwill_impairment": loss, "implied_goodwill": implied}
    raise ValueError("standard must be ifrs or us_gaap")


# ----------------------------------------------------------------------------- income taxes

@formula(_T, formula="asset: (CA - TB) x t > 0 -> DTL, < 0 -> DTA; liability: (TB - CA) x t > 0 -> DTL, < 0 -> DTA",
         curriculum=L1TAX + " (3-4)")
def deferred_tax_items(items: list, tax_rate: float) -> dict:
    """Deferred tax assets and liabilities from carrying amounts versus tax bases.

    Args:
        items: list of {"name", "kind": "asset"|"liability", "carrying_amount", "tax_base"}.
            Permanent differences (e.g. non-deductible items) should be left out or given equal amounts.

    Example:
        [{"name": "PPE", "kind": "asset", "carrying_amount": 16000, "tax_base": 14286}], 0.30 -> DTL 514.2
    """
    rows, dtl, dta = [], 0.0, 0.0
    for it in items:
        diff = it["carrying_amount"] - it["tax_base"]
        if it["kind"] == "asset":
            dt = diff * tax_rate
        elif it["kind"] == "liability":
            dt = -diff * tax_rate
        else:
            raise ValueError(f"item {it.get('name')!r}: kind must be asset or liability")
        dtl += max(dt, 0.0)
        dta += max(-dt, 0.0)
        rows.append({"name": it.get("name"), "temporary_difference": diff,
                     "deferred_tax_liability": max(dt, 0.0), "deferred_tax_asset": max(-dt, 0.0)})
    return {"net_deferred_tax_liability": dtl - dta, "deferred_tax_liability": dtl, "deferred_tax_asset": dta,
            "items": rows}


@formula(_T, formula="income tax expense = taxes payable + change in DTL - change in DTA", curriculum=L1TAX + " (2)")
def income_tax_expense(taxes_payable: float, dtl_begin: float = 0.0, dtl_end: float = 0.0, dta_begin: float = 0.0,
                       dta_end: float = 0.0, pretax_income: float | None = None) -> dict:
    """Income tax expense from current taxes payable and the change in deferred tax balances.

    Example:
        income_tax_expense(4326.9, 257.1, 514.2) -> 4584.0
    """
    deferred = (dtl_end - dtl_begin) - (dta_end - dta_begin)
    o = {"income_tax_expense": taxes_payable + deferred, "deferred_tax_expense": deferred,
         "current_tax_expense": taxes_payable}
    if pretax_income:
        o["effective_tax_rate"] = o["income_tax_expense"] / pretax_income
    return o


@formula(_T, formula="revaluation effect = beginning temporary difference x (new rate - old rate)",
         curriculum=L1TAX + " (3.3)")
def tax_rate_change_effect(temporary_difference_begin: float, temporary_difference_end: float, old_rate: float,
                           new_rate: float) -> dict:
    """Effect of an enacted tax rate change on deferred taxes and on income tax expense in the year of change.

    Temporary differences are net taxable amounts (positive -> DTL, negative -> DTA).

    Example:
        tax_rate_change_effect(1714, 2571, 0.30, 0.25) -> deferred tax end 642.75, revaluation -85.7
    """
    begin_old = temporary_difference_begin * old_rate
    end_new = temporary_difference_end * new_rate
    return {"deferred_tax_end": end_new, "deferred_tax_begin_old_rate": begin_old,
            "deferred_tax_begin_new_rate": temporary_difference_begin * new_rate,
            "revaluation_effect_on_expense": temporary_difference_begin * (new_rate - old_rate),
            "deferred_tax_expense_on_current_year_difference":
                (temporary_difference_end - temporary_difference_begin) * new_rate,
            "deferred_tax_expense_total": end_new - begin_old}


@formula(_T, formula="ETR = income tax expense / pretax income = statutory rate + sum(reconciling item / pretax)",
         curriculum=L1TAX + " (6)")
def effective_tax_rate_reconciliation(pretax_income: float, statutory_rate: float,
                                      reconciling_items: dict | None = None,
                                      reported_tax_expense: float | None = None) -> dict:
    """Reconcile the statutory rate to the effective tax rate.

    Args:
        reconciling_items: {name: tax amount} (e.g. foreign rate differential, non-deductible expenses, credits);
            positive increases tax.
        reported_tax_expense: if given, the ETR is the reported one and any unexplained gap is shown.

    Example:
        effective_tax_rate_reconciliation(189167, 0.35, reported_tax_expense=56860) -> etr 0.3006
    """
    if not pretax_income:
        raise ValueError("pretax_income must be non-zero")
    items = reconciling_items or {}
    statutory_tax = pretax_income * statutory_rate
    explained = statutory_tax + sum(items.values())
    tax = reported_tax_expense if reported_tax_expense is not None else explained
    return {"effective_tax_rate": tax / pretax_income, "income_tax_expense": tax,
            "tax_at_statutory_rate": statutory_tax,
            "rate_effects": {k: v / pretax_income for k, v in items.items()},
            "unexplained": tax - explained}


# ----------------------------------------------------------------------------- long-term liabilities and leases

@formula(_T, formula="interest expense = carrying amount x market rate at issue; amortization = expense - coupon",
         curriculum=L1LTL + " (2.2)")
def bond_effective_interest_schedule(face: float, coupon_rate: float, market_rate: float, years: float,
                                     freq: int = 1, issue_price: float | None = None) -> dict:
    """Issuer's amortization schedule for a bond issued at a premium or discount (effective interest method).

    Example:
        bond_effective_interest_schedule(1e6, 0.05, 0.06, 5) -> issue price 957876, year-1 interest 57473
    """
    n, r, c = int(round(years * freq)), market_rate / freq, face * coupon_rate / freq
    if n < 1 or abs(years * freq - n) > 1e-9:
        raise ValueError(f"years x freq must be a whole number of periods >= 1, got {years * freq}")
    price = issue_price if issue_price is not None else \
        sum(c / (1 + r) ** t for t in range(1, n + 1)) + face / (1 + r) ** n
    rows, ca = [], price
    for t in range(1, n + 1):
        interest = ca * r
        end = ca + interest - c
        rows.append({"period": t, "beginning_carrying_amount": ca, "interest_expense": interest,
                     "coupon_payment": c, "amortization": interest - c, "ending_carrying_amount": end})
        ca = end
    return {"issue_price": price, "discount_or_premium": price - face, "schedule": rows,
            "straight_line_interest_expense": c + (face - price) / n}


@formula(_T, formula="liability = PV(payments); interest = r x outstanding balance; expense = depreciation + interest",
         curriculum=L1LTL + " (3.1); " + L2LLA + " (6.2)")
def finance_lease_lessee(payment: float, rate: float, periods: int, payments_in_advance: bool = True,
                         useful_life: int | None = None, salvage_value: float = 0.0) -> dict:
    """Lessee finance (capital) lease: initial liability, liability schedule, and expense split versus an operating lease.

    With payments in advance, interest accrued in a year is paid with the next year's first-day payment.

    Example:
        finance_lease_lessee(28679, 0.10, 4) -> liability 100000, year-1 interest 7132, total expense 32132
    """
    if periods < 1 or rate <= -1:
        raise ValueError("periods must be >= 1 and rate > -1")
    pv = sum(payment / (1 + rate) ** (t if not payments_in_advance else t - 1) for t in range(1, periods + 1))
    dep = (pv - salvage_value) / (useful_life or periods)
    rows, bal = [], pv
    for t in range(1, periods + 1):
        opening = bal
        if payments_in_advance:
            bal -= payment
            interest = bal * rate
            closing_excl = bal
            bal += interest
        else:
            interest = bal * rate
            bal = bal + interest - payment
            closing_excl = bal
        rows.append({"period": t, "opening_liability": opening, "payment": payment, "interest_expense": interest,
                     "closing_liability": closing_excl, "accrued_interest": interest if payments_in_advance else 0.0,
                     "depreciation": dep, "total_expense": dep + interest,
                     "operating_lease_expense": payment})
    return {"initial_liability": pv, "schedule": rows, "annual_depreciation": dep}


@formula(_T, formula="implicit rate: PV = sum payment_t/(1+r)^t", curriculum=L2LLA + " (6.2.1)")
def lease_implicit_rate(present_value: float, payments: list, payments_in_advance: bool = False) -> dict:
    """Interest rate implicit in a lease from its present value and payment schedule.

    Example:
        lease_implicit_rate(12208, [1683, 1683, 1683, 1600] + [1586]*7 + [454]) -> 0.07245
    """
    cfs = [-present_value] + list(payments) if not payments_in_advance else \
        [payments[0] - present_value] + list(payments[1:])
    r = _irr(cfs)
    if r is None:
        raise ValueError("no unique implicit rate for these payments")
    return {"implicit_rate": r}


@formula(_T, formula="debt adjustment = PV(operating lease payments); adjusted D/E = (L + PV)/E",
         curriculum=L2LLA + " (6.2.1)")
def operating_lease_capitalization(payments: list, discount_rate: float, total_liabilities: float | None = None,
                                   total_equity: float | None = None, payments_in_advance: bool = False,
                                   useful_life: int | None = None) -> dict:
    """Analyst adjustment treating operating leases as finance leases: added debt, adjusted leverage, first-year
    interest and depreciation that replace rent expense.

    Example:
        CEC payments [66849, 66396, 66558, 65478, 63872] + [63872]*7 + [27650] at 7.245% -> pv 520256,
        D/E 4.73 -> 8.78 with liabilities 608854 and equity 128586
    """
    pv = _npv(discount_rate, ([payments[0]] + list(payments[1:])) if payments_in_advance else [0.0] + list(payments))
    life = useful_life or len(payments)
    o = {"lease_liability_added": pv, "first_year_interest": (pv - (payments[0] if payments_in_advance else 0.0))
         * discount_rate, "annual_depreciation": pv / life, "rent_expense_replaced": payments[0]}
    if total_liabilities is not None and total_equity:
        o.update(debt_to_equity_reported=total_liabilities / total_equity,
                 debt_to_equity_adjusted=(total_liabilities + pv) / total_equity,
                 adjusted_total_liabilities=total_liabilities + pv)
    return o


# ----------------------------------------------------------------------------- pensions

@formula(_T, formula="funded status = plan assets - benefit obligation; net asset capped at the asset ceiling",
         curriculum=L2PEN + " (2.3.2)")
def db_pension_funded_status(benefit_obligation: float, plan_assets: float, asset_ceiling: float | None = None) -> dict:
    """Defined benefit plan funded status and the net pension liability or (ceiling-limited) asset reported.

    Example:
        pension_funded_status(6723, 4880) -> net liability 1843;
        pension_funded_status(5485, 5998, asset_ceiling=326) -> net asset 326
    """
    fs = plan_assets - benefit_obligation
    asset = max(fs, 0.0)
    if asset_ceiling is not None:
        asset = min(asset, asset_ceiling)
    return {"funded_status": fs, "net_pension_liability": max(-fs, 0.0), "net_pension_asset": asset}


@formula(_T, formula="IFRS P&L = service cost + r x net liability; OCI = actuarial loss + (r x assets - actual return); "
                     "US GAAP P&L = current service + r x PBO - expected return + amortization; total = sum of all",
         curriculum=L2PEN + " (2.3.2, 2.4.3)")
def pension_periodic_cost(obligation_begin: float, plan_assets_begin: float, discount_rate: float,
                          current_service_cost: float, past_service_cost: float = 0.0, actuarial_loss: float = 0.0,
                          actual_return: float = 0.0, expected_return_rate: float | None = None,
                          employer_contributions: float = 0.0, benefits_paid: float | None = None,
                          obligation_end: float | None = None, plan_assets_end: float | None = None,
                          amortization_past_service_cost: float = 0.0,
                          amortization_actuarial_loss: float = 0.0) -> dict:
    """Defined benefit periodic pension cost: total, IFRS split (P&L vs OCI) and US GAAP P&L expense.

    actuarial_loss is negative for a gain. benefits_paid is inferred from obligation_end or plan_assets_end when not
    given. US GAAP P&L uses the expected return (expected_return_rate x beginning assets) and any amortization of
    past service cost and actuarial losses you pass.

    Example:
        pension_periodic_cost(42000, 39000, 0.07, 200, 120, 460, 2700, 0.08, 1000, obligation_end=41720)
        -> total 1020, ifrs P&L 530, ifrs OCI 490, us_gaap P&L 20, benefits_paid 4000
    """
    interest = discount_rate * obligation_begin
    if benefits_paid is None:
        if obligation_end is not None:
            benefits_paid = (obligation_begin + current_service_cost + past_service_cost + interest + actuarial_loss
                             - obligation_end)
        elif plan_assets_end is not None:
            benefits_paid = plan_assets_begin + employer_contributions + actual_return - plan_assets_end
        else:
            benefits_paid = 0.0
    obl_end = obligation_begin + current_service_cost + past_service_cost + interest + actuarial_loss - benefits_paid
    assets_end = plan_assets_begin + employer_contributions + actual_return - benefits_paid
    total = current_service_cost + past_service_cost + interest + actuarial_loss - actual_return
    ifrs_pl = current_service_cost + past_service_cost + discount_rate * (obligation_begin - plan_assets_begin)
    o = {"total_periodic_cost": total, "ifrs_pl_cost": ifrs_pl, "ifrs_oci_remeasurements": total - ifrs_pl,
         "interest_cost": interest, "benefits_paid": benefits_paid, "obligation_end": obl_end,
         "plan_assets_end": assets_end, "funded_status_end": assets_end - obl_end,
         "excess_contribution_over_cost": employer_contributions - total}
    if expected_return_rate is not None:
        expected = expected_return_rate * plan_assets_begin
        us_pl = (current_service_cost + interest - expected + amortization_past_service_cost
                 + amortization_actuarial_loss)
        o.update(us_gaap_pl_cost=us_pl, us_gaap_oci=total - us_pl, expected_return=expected)
    return o


@formula(_T, formula="unit credit = final salary x benefit rate x service years / service years; "
                     "closing obligation_t = credit x t / (1+r)^(n-t)", curriculum=L2PEN + " (2.3.3, Example 2)")
def pension_obligation_unit_credit(annual_salary: float, years_until_retirement: int, salary_growth: float,
                                   discount_rate: float, benefit_rate: float) -> dict:
    """Projected unit credit build-up of one employee's lump-sum benefit obligation, service cost and interest cost.

    annual_salary is the coming year's salary; service runs from now to retirement; the lump sum is paid at
    retirement and equals final salary x benefit_rate x years of service.

    Example:
        pension_obligation_unit_credit(50000, 5, 0.0475, 0.06, 0.015) -> lump sum 4514.89, year-1 service cost 715.24
    """
    n = int(years_until_retirement)
    if n < 1 or n != years_until_retirement:
        raise ValueError("years_until_retirement must be a whole number of years, at least 1")
    final = annual_salary * (1 + salary_growth) ** (n - 1)
    lump = final * benefit_rate * n
    credit = lump / n
    rows, opening = [], 0.0
    for t in range(1, n + 1):
        closing = credit * t / (1 + discount_rate) ** (n - t)
        interest = opening * discount_rate
        rows.append({"year": t, "opening_obligation": opening, "interest_cost": interest,
                     "current_service_cost": closing - opening - interest, "closing_obligation": closing})
        opening = closing
    return {"lump_sum_benefit": lump, "final_salary": final, "annual_unit_credit": credit, "schedule": rows}


@formula(_T, formula="operating profit + pension expense - service cost; interest expense + interest cost; "
                     "interest income + actual return", curriculum=L2PEN + " (2.4.5)")
def pension_income_adjustment(operating_profit: float, interest_expense: float, interest_income: float,
                              pretax_income: float, total_pension_expense: float, current_service_cost: float,
                              interest_cost: float, actual_return: float) -> dict:
    """Reclassify pension expense: keep only service cost in operating profit, interest cost in interest expense,
    and the actual (not expected) return in investment income. Expenses are positive magnitudes.

    Example:
        pension_income_adjustment(2619, 879, 316, 2929, 36, 11, 39, 47) -> operating profit 2644, pretax 2962
    """
    return {"adjusted_operating_profit": operating_profit + total_pension_expense - current_service_cost,
            "adjusted_interest_expense": interest_expense + interest_cost,
            "adjusted_interest_income": interest_income + actual_return,
            "adjusted_pretax_income": pretax_income + total_pension_expense - current_service_cost - interest_cost
            + actual_return}


@formula(_T, formula="excess contribution after tax = (contribution - total pension cost)(1 - t): move from CFO to CFF",
         curriculum=L2PEN + " (2.4.6)")
def pension_cash_flow_adjustment(employer_contributions: float, total_pension_cost: float, tax_rate: float,
                                 cfo: float | None = None, cff: float | None = None) -> dict:
    """Treat contributions above (below) total pension cost as debt repayment (borrowing): reclassify CFO and CFF.

    cff is signed (negative = net outflow).

    Example:
        pension_cash_flow_adjustment(504, 437, 0.287, 6161, -1741) -> cfo 6208.8, cff -1788.8
    """
    excess = employer_contributions - total_pension_cost
    after = excess * (1 - tax_rate)
    o = {"excess_contribution_after_tax": after, "excess_contribution_pretax": excess}
    if cfo is not None:
        o["adjusted_cfo"] = cfo + after
    if cff is not None:
        o["adjusted_cff"] = cff - after
    return o


# ----------------------------------------------------------------------------- EPS

@formula(_T, formula="weighted shares = sum shares outstanding x fraction of period; splits restated retroactively",
         curriculum=L1EPS + " (6.2)")
def weighted_average_shares(beginning_shares: float, share_changes: list | None = None,
                            stock_split_factor: float = 1.0, period_months: float = 12.0) -> dict:
    """Weighted average number of common shares (issuances positive, buybacks negative).

    Args:
        share_changes: [[months_elapsed_at_change, change_in_shares], ...].
        stock_split_factor: e.g. 2 for a 2-for-1 split or 100% stock dividend during or after the period.

    Example:
        weighted_average_shares(1e6, [[3, 200000], [9, -100000]]) -> 1125000
    """
    total = beginning_shares * period_months
    for month, chg in share_changes or []:
        if not 0 <= month <= period_months:
            raise ValueError("months_elapsed must lie within the period")
        total += chg * (period_months - month)
    return {"weighted_average_shares": total / period_months * stock_split_factor}


@formula(_T, formula="basic = (NI - pref div)/W; diluted adds if-converted and treasury-stock shares, excluding "
                     "antidilutive securities (ranked by incremental EPS)", curriculum=L1EPS + " (6.3)")
def eps_basic_diluted(net_income: float, weighted_average_shares: float, preferred_dividends: float = 0.0,
                      convertible_preferred: list | None = None, convertible_debt: list | None = None,
                      options: list | None = None, average_market_price: float | None = None,
                      tax_rate: float = 0.0) -> dict:
    """Basic and diluted EPS with the if-converted and treasury stock methods and an antidilution check.

    Args:
        convertible_preferred: [{"dividends", "shares"}] (dividends forgone, common shares on conversion).
        convertible_debt: [{"interest", "shares"}] (pre-tax interest saved, common shares on conversion).
        options: [{"count", "exercise_price", "fraction_of_year" (default 1)}] options or warrants.

    Example:
        eps_basic_diluted(750000, 690000, convertible_debt=[{"interest": 3000, "shares": 10000}], tax_rate=0.30)
        -> basic 1.087, diluted 1.074
    """
    if weighted_average_shares <= 0:
        raise ValueError("weighted_average_shares must be positive")
    num, den = net_income - preferred_dividends, float(weighted_average_shares)
    basic = num / den
    cands = []
    for i, p in enumerate(convertible_preferred or []):
        cands.append((f"convertible_preferred_{i}", p["dividends"], p["shares"]))
    for i, d in enumerate(convertible_debt or []):
        cands.append((f"convertible_debt_{i}", d["interest"] * (1 - tax_rate), d["shares"]))
    for i, op in enumerate(options or []):
        if average_market_price is None:
            raise ValueError("options need average_market_price")
        inc = op["count"] * max(0.0, 1 - op["exercise_price"] / average_market_price) * op.get("fraction_of_year", 1)
        cands.append((f"options_{i}", 0.0, inc))
    cands.sort(key=lambda c: c[1] / c[2] if c[2] else float("inf"))
    included, anti = [], []
    for name, dn, dd in cands:
        if dd > 0 and (num + dn) / (den + dd) < num / den:
            num, den = num + dn, den + dd
            included.append(name)
        else:
            anti.append(name)
    return {"diluted_eps": num / den, "basic_eps": basic, "diluted_numerator": num, "diluted_shares": den,
            "included": included, "antidilutive_excluded": anti}


# ----------------------------------------------------------------------------- cash flows

@formula(_T, formula="CFO = NI + non-cash charges - gains + losses - increase in operating current assets "
                     "+ increase in operating current liabilities", curriculum=L1CF + " (3.2)")
def cfo_indirect_method(net_income: float, noncash_charges: float = 0.0, gains_on_disposal: float = 0.0,
                        losses_on_disposal: float = 0.0, increase_operating_assets: float = 0.0,
                        increase_operating_liabilities: float = 0.0) -> dict:
    """Cash flow from operations by the indirect method (signed changes: a decrease is negative).

    Example:
        cfo_indirect_method(30, 7, increase_operating_assets=15 - 3, increase_operating_liabilities=10) -> cfo 35
    """
    adj = (noncash_charges - gains_on_disposal + losses_on_disposal - increase_operating_assets
           + increase_operating_liabilities)
    return {"cfo": net_income + adj, "total_adjustments": adj}


@formula(_T, formula="FCFF = NI + NCC + Int(1-t) - FCInv - WCInv = CFO + Int(1-t) - FCInv = EBIT(1-t) + Dep - FCInv "
                     "- WCInv = EBITDA(1-t) + Dep t - FCInv - WCInv; FCFE = FCFF - Int(1-t) + net borrowing",
         curriculum=L1CF + " (4.3); L2 Equity: Free Cash Flow Valuation")
def free_cash_flow_fcff_fcfe(tax_rate: float, fixed_capital_investment: float, working_capital_investment: float = 0.0,
                             interest_expense: float = 0.0, net_borrowing: float = 0.0, cfo: float | None = None,
                             net_income: float | None = None, noncash_charges: float = 0.0,
                             ebit: float | None = None, ebitda: float | None = None,
                             depreciation: float = 0.0) -> dict:
    """FCFF and FCFE from whichever starting point is given (CFO, net income, EBIT or EBITDA).

    fixed_capital_investment is capex net of asset sale proceeds; net_borrowing is negative for net repayment;
    CFO is assumed to include interest paid (US GAAP). The headline uses the first available of CFO, NI, EBIT, EBITDA.

    Example:
        free_cash_flow_fcff_fcfe(0.34, 538, interest_expense=258, net_borrowing=-500, cfo=2606)
        -> fcff 2238.28, fcfe 1568
    """
    ati = interest_expense * (1 - tax_rate)
    versions = {}
    if cfo is not None:
        versions["fcff_from_cfo"] = cfo + ati - fixed_capital_investment
    if net_income is not None:
        versions["fcff_from_net_income"] = (net_income + noncash_charges + ati - fixed_capital_investment
                                            - working_capital_investment)
    if ebit is not None:
        versions["fcff_from_ebit"] = ebit * (1 - tax_rate) + depreciation - fixed_capital_investment \
            - working_capital_investment
    if ebitda is not None:
        versions["fcff_from_ebitda"] = ebitda * (1 - tax_rate) + depreciation * tax_rate \
            - fixed_capital_investment - working_capital_investment
    if not versions:
        raise ValueError("give cfo, net_income, ebit or ebitda")
    fcff = next(iter(versions.values()))
    return {"fcff": fcff, "fcfe": fcff - ati + net_borrowing, "after_tax_interest": ati, **versions}


@formula(_T, formula="CFO / revenue, assets, equity, operating income, debt; (CFO + interest + taxes)/interest; "
                     "CFO / capex, debt repayment, dividends", curriculum=L1CF + " (4.4, Exhibit 15)")
def cash_flow_ratios(cfo: float, revenue: float | None = None, average_total_assets: float | None = None,
                     average_equity: float | None = None, operating_income: float | None = None,
                     common_shares: float | None = None, preferred_dividends: float = 0.0,
                     total_debt: float | None = None, interest_paid: float | None = None,
                     taxes_paid: float | None = None, capex: float | None = None,
                     debt_repayment: float | None = None, dividends_paid: float | None = None,
                     investing_financing_outflows: float | None = None) -> dict:
    """Cash flow performance and coverage ratios; ratios with missing inputs are skipped.

    Example:
        cash_flow_ratios(2606, revenue=23803) -> cash_flow_to_revenue 0.1095
    """
    o: dict = {}
    _put(o, "cash_flow_to_revenue", _div(cfo, revenue))
    _put(o, "cash_return_on_assets", _div(cfo, average_total_assets))
    _put(o, "cash_return_on_equity", _div(cfo, average_equity))
    _put(o, "cash_to_income", _div(cfo, operating_income))
    _put(o, "cash_flow_per_share", _div(cfo - preferred_dividends, common_shares))
    _put(o, "debt_coverage", _div(cfo, total_debt))
    if interest_paid:
        o["interest_coverage"] = (cfo + interest_paid + (taxes_paid or 0.0)) / interest_paid
    _put(o, "reinvestment", _div(cfo, capex))
    _put(o, "debt_payment", _div(cfo, debt_repayment))
    _put(o, "dividend_payment", _div(cfo, dividends_paid))
    _put(o, "investing_and_financing", _div(cfo, investing_financing_outflows))
    return o


# ----------------------------------------------------------------------------- earnings quality

@formula(_T, formula="NOA = (TA - cash) - (TL - debt); BS accruals ratio = dNOA / avg NOA; "
                     "CF accruals ratio = (NI - CFO - CFI) / avg NOA", curriculum=L2EQ + " (3.2)")
def accruals_ratios(total_assets: float, cash: float, total_liabilities: float, total_debt: float,
                    total_assets_prior: float, cash_prior: float, total_liabilities_prior: float,
                    total_debt_prior: float, net_income: float | None = None, cfo: float | None = None,
                    cfi: float | None = None) -> dict:
    """Balance-sheet and cash-flow-statement accruals ratios (higher = more accruals = lower earnings quality).

    cash is cash plus short-term investments; total_debt is short-term plus long-term debt; cfi is signed.

    Example (Coca-Cola 2006): -> noa 18912, accruals_ratio_balance_sheet 0.0897, accruals_ratio_cash_flow 0.0455
    """
    noa = (total_assets - cash) - (total_liabilities - total_debt)
    noa_p = (total_assets_prior - cash_prior) - (total_liabilities_prior - total_debt_prior)
    avg = (noa + noa_p) / 2
    if avg == 0:
        raise ValueError("average NOA is zero")
    o = {"accruals_ratio_balance_sheet": (noa - noa_p) / avg, "noa": noa, "noa_prior": noa_p,
         "aggregate_accruals_balance_sheet": noa - noa_p}
    if None not in (net_income, cfo, cfi):
        agg = net_income - (cfo + cfi)
        o.update(aggregate_accruals_cash_flow=agg, accruals_ratio_cash_flow=agg / avg)
    return o


def _beneish_ratios(d: dict) -> dict:
    return {"rec": d["receivables"] / d["sales"], "gm": (d["sales"] - d["cogs"]) / d["sales"],
            "aq": 1 - (d["current_assets"] + d["ppe"] + d.get("securities", 0.0)) / d["total_assets"],
            "dep": d["depreciation"] / (d["depreciation"] + d["ppe"]), "sga": d["sga"] / d["sales"],
            "lev": (d["current_liabilities"] + d["long_term_debt"]) / d["total_assets"]}


def _beneish_indices(c: dict, p: dict) -> tuple[dict, list[str]]:
    """The eight indices; an index whose denominator is zero (e.g. AQI when the balance sheet has nothing besides
    current assets and PP&E) is set to 1.0, neutral, and noted."""
    rc, rp = _beneish_ratios(c), _beneish_ratios(p)
    notes: list[str] = []

    def ratio(name: str, num: float, den: float) -> float:
        if den == 0:
            notes.append(f"{name} set to 1.0 (neutral): its denominator is zero")
            return 1.0
        return num / den

    idx = {"dsri": ratio("dsri", rc["rec"], rp["rec"]), "gmi": ratio("gmi", rp["gm"], rc["gm"]),
           "aqi": ratio("aqi", rc["aq"], rp["aq"]), "sgi": ratio("sgi", c["sales"], p["sales"]),
           "depi": ratio("depi", rp["dep"], rc["dep"]), "sgai": ratio("sgai", rc["sga"], rp["sga"]),
           "lvgi": ratio("lvgi", rc["lev"], rp["lev"]),
           "tata": ratio("tata", c["net_income"] - c["cfo"], c["total_assets"])}
    return idx, notes


@formula(_T, formula="M = -4.84 + .920 DSRI + .528 GMI + .404 AQI + .892 SGI + .115 DEPI - .172 SGAI + 4.679 TATA "
                     "- .327 LVGI", curriculum="Beneish (1999), Financial Analysts Journal 55(5); CFA L2 FRA "
                                               "financial reporting quality (post-2014 curricula)")
def beneish_m_score(current: dict | None = None, prior: dict | None = None, indices: dict | None = None,
                    threshold: float = -1.78) -> dict:
    """Beneish eight-variable M-score for earnings manipulation (above the threshold = likely manipulator).

    Args:
        current, prior: statement items with keys receivables, sales, cogs, current_assets, ppe (net), securities
            (optional), total_assets, depreciation, sga, net_income (continuing ops), cfo, current_liabilities,
            long_term_debt.
        indices: or the eight indices directly: dsri, gmi, aqi, sgi, depi, sgai, tata, lvgi.
        threshold: -1.78 (Beneish 1999, used by CFA); -2.22 is a common stricter cut-off.

    Example:
        all indices 1 and tata 0 -> m_score -2.48
    """
    notes: list[str] = []
    if indices:
        idx = indices
    elif current and prior:
        idx, notes = _beneish_indices(current, prior)
    else:
        raise ValueError("give current and prior statement dicts, or indices")
    w = {"dsri": 0.920, "gmi": 0.528, "aqi": 0.404, "sgi": 0.892, "depi": 0.115, "sgai": -0.172, "tata": 4.679,
         "lvgi": -0.327}
    m = -4.84 + sum(w[k] * idx[k] for k in w)
    return {"m_score": m, "likely_manipulator": m > threshold, "indices": idx, "notes": notes}


@formula(_T, formula="Z = 1.2 WC/TA + 1.4 RE/TA + 3.3 EBIT/TA + 0.6 MVE/TL + 1.0 Sales/TA",
         curriculum=L1RA + " (7.2 credit analysis, Altman 1968)")
def altman_z_score(working_capital: float, retained_earnings: float, ebit: float, market_value_equity: float,
                   total_liabilities: float, sales: float, total_assets: float) -> dict:
    """Altman (1968) Z-score for public manufacturers: below 1.81 distress, above 2.99 safe, grey zone between.

    Example:
        altman_z_score(10, 20, 15, 60, 50, 120, 100) -> 2.815 (grey zone)
    """
    if total_assets <= 0 or total_liabilities <= 0:
        raise ValueError("total_assets and total_liabilities must be positive")
    parts = {"x1_working_capital": 1.2 * working_capital / total_assets,
             "x2_retained_earnings": 1.4 * retained_earnings / total_assets,
             "x3_ebit": 3.3 * ebit / total_assets,
             "x4_market_equity_to_liabilities": 0.6 * market_value_equity / total_liabilities,
             "x5_sales": 1.0 * sales / total_assets}
    z = sum(parts.values())
    zone = "distress" if z < 1.81 else ("safe" if z > 2.99 else "grey")
    return {"z_score": z, "zone": zone, "components": parts}


# ----------------------------------------------------------------------------- intercorporate investments

@formula(_T, formula="carrying = cost + share of NI - dividends - amortization of excess - deferred unrealized profit; "
                     "goodwill = cost - share of FV of identifiable net assets", curriculum=L2ICI + " (5)")
def equity_method_investment(purchase_price: float, ownership: float, investee_net_income: list,
                             investee_dividends: list | None = None, investee_book_value: float | None = None,
                             fair_value_adjustments: list | None = None,
                             unrealized_profit: list | None = None) -> dict:
    """Equity-method income and carrying amount of an associate, year by year.

    Args:
        ownership: decimal stake (0.30).
        investee_net_income, investee_dividends: per-year lists (100% of the investee's amounts).
        investee_book_value: book value of the investee's net assets at acquisition (for goodwill).
        fair_value_adjustments: [{"amount": FV - BV at 100%, "useful_life": years or None}] amortized straight line.
        unrealized_profit: per-year intercompany profit left in ending inventory (100%, upstream or downstream);
            the investor's share is deferred and recognized when the goods are sold.

    Example:
        equity_method_investment(500000, 0.30, [100000], [50000], 1200000, [{"amount": 300000, "useful_life": 10}])
        -> goodwill 50000, equity income 21000, carrying 506000
    """
    if not 0 < ownership <= 1:
        raise ValueError("ownership must be in (0, 1]")
    adjs = fair_value_adjustments or []
    amort = sum(ownership * a["amount"] / a["useful_life"] for a in adjs if a.get("useful_life"))
    divs = investee_dividends or [0.0] * len(investee_net_income)
    unreal = unrealized_profit or [0.0] * len(investee_net_income)
    rows, carrying, prev_unreal = [], purchase_price, 0.0
    for t, ni in enumerate(investee_net_income):
        income = ownership * ni - amort - ownership * (unreal[t] - prev_unreal)
        carrying += income - ownership * divs[t]
        prev_unreal = unreal[t]
        rows.append({"year": t + 1, "equity_income": income, "dividends_received": ownership * divs[t],
                     "carrying_amount": carrying})
    o = {"carrying_amount": carrying, "annual_amortization": amort, "years": rows}
    if investee_book_value is not None:
        excess = purchase_price - ownership * investee_book_value
        o.update(excess_purchase_price=excess, goodwill=excess - ownership * sum(a["amount"] for a in adjs))
    return o


@formula(_T, formula="full goodwill = FV of subsidiary - FV identifiable net assets; partial goodwill = price - "
                     "ownership x FV net assets; NCI = (1 - ownership) x (FV subsidiary or FV net assets)",
         curriculum=L2ICI + " (6.4.2)")
def acquisition_goodwill(purchase_price: float, ownership: float, fair_value_identifiable_net_assets: float,
                         fair_value_of_subsidiary: float | None = None) -> dict:
    """Goodwill and non-controlling interest under the full (US GAAP, IFRS option) and partial (IFRS) methods.

    Example:
        acquisition_goodwill(180000, 0.90, 160000, 200000) -> full goodwill 40000, NCI 20000;
        partial goodwill 36000, NCI 16000
    """
    if not 0 < ownership <= 1:
        raise ValueError("ownership must be in (0, 1]")
    fv_sub = fair_value_of_subsidiary if fair_value_of_subsidiary is not None else purchase_price / ownership
    return {"full_goodwill": fv_sub - fair_value_identifiable_net_assets,
            "full_goodwill_nci": (1 - ownership) * fv_sub,
            "partial_goodwill": purchase_price - ownership * fair_value_identifiable_net_assets,
            "partial_goodwill_nci": (1 - ownership) * fair_value_identifiable_net_assets,
            "bargain_purchase_gain": max(0.0, ownership * fair_value_identifiable_net_assets - purchase_price)}


# ----------------------------------------------------------------------------- multinational operations

@formula(_T, formula="gain = foreign amount x (rate settle - rate initial) for a receivable (negative for a payable)",
         curriculum=L2FX + " (2.1)")
def fx_transaction_gain_loss(foreign_amount: float, rate_at_transaction: float, rate_at_settlement: float,
                             position: str = "payable", rate_at_balance_sheet: float | None = None) -> dict:
    """Foreign currency transaction gain/loss (rates in home currency per unit of foreign currency).

    With rate_at_balance_sheet, splits the result into the unrealized part (reporting period 1) and the part
    realized at settlement (period 2). Both go to net income.

    Example:
        fx_transaction_gain_loss(100000, 0.0684, 0.0703, "payable") -> -190
    """
    sign = {"payable": -1.0, "receivable": 1.0}.get(position)
    if sign is None:
        raise ValueError("position must be payable or receivable")
    o = {"gain_loss": sign * foreign_amount * (rate_at_settlement - rate_at_transaction),
         "home_amount_at_transaction": foreign_amount * rate_at_transaction,
         "home_amount_at_settlement": foreign_amount * rate_at_settlement}
    if rate_at_balance_sheet is not None:
        o["gain_loss_to_balance_sheet_date"] = sign * foreign_amount * (rate_at_balance_sheet - rate_at_transaction)
        o["gain_loss_after_balance_sheet_date"] = sign * foreign_amount * (rate_at_settlement - rate_at_balance_sheet)
    return o


@formula(_T, formula="translation gain = exposure_begin x (S_end - S_begin) + sum flow_i x (S_end - S_i)",
         curriculum=L2FX + " (3.3-3.4)")
def fx_translation_adjustment(beginning_exposure: float, rate_begin: float, rate_end: float,
                              flows: list | None = None) -> dict:
    """Translation gain/loss on an exposed position (net assets for current rate, net monetary assets for temporal).

    Args:
        beginning_exposure: foreign-currency amount at the start (negative for a net liability position).
        flows: [[foreign_amount, rate_when_it_occurred], ...] changes in the exposed position during the period
            (e.g. net income at the average rate, dividends at the declaration rate; for temporal: sales,
            inventory purchases, cash expenses, dividends).

    Current rate method: result is the cumulative translation adjustment in OCI. Temporal method: result is a
    remeasurement gain/loss in net income.

    Example:
        fx_translation_adjustment(1500000, 0.70, 0.80, [[1180000, 0.75], [-350000, 0.78]]) -> 202000
    """
    flows = flows or []
    end_exposure = beginning_exposure + sum(a for a, _ in flows)
    gain = beginning_exposure * (rate_end - rate_begin) + sum(a * (rate_end - r) for a, r in flows)
    return {"translation_gain_loss": gain, "ending_exposure": end_exposure}


@formula(_T, formula="current rate exposure = assets - liabilities; temporal exposure = monetary assets - monetary "
                     "liabilities; gain if exposure x change in foreign currency value > 0", curriculum=L2FX + " (3.4)")
def fx_translation_exposure(method: str, foreign_currency_change: str, total_assets: float | None = None,
                            total_liabilities: float | None = None, monetary_assets: float | None = None,
                            monetary_liabilities: float | None = None) -> dict:
    """Balance sheet translation exposure and the sign of the translation result for a currency move.

    Args:
        method: "current_rate" (foreign currency is functional) or "temporal" (parent currency is functional,
            or highly inflationary under US GAAP).
        foreign_currency_change: "appreciates" or "depreciates" against the parent's presentation currency.

    Example:
        fx_translation_exposure("temporal", "appreciates", monetary_assets=1880, monetary_liabilities=3450)
        -> net monetary liability exposure, loss, reported in net income
    """
    if method == "current_rate":
        if total_assets is None or total_liabilities is None:
            raise ValueError("current_rate needs total_assets and total_liabilities")
        exposure, where = total_assets - total_liabilities, "OCI (cumulative translation adjustment)"
    elif method == "temporal":
        if monetary_assets is None or monetary_liabilities is None:
            raise ValueError("temporal needs monetary_assets and monetary_liabilities")
        exposure, where = monetary_assets - monetary_liabilities, "net income (remeasurement gain/loss)"
    else:
        raise ValueError("method must be current_rate or temporal")
    direction = {"appreciates": 1, "depreciates": -1}.get(foreign_currency_change)
    if direction is None:
        raise ValueError("foreign_currency_change must be appreciates or depreciates")
    sign = exposure * direction
    return {"exposure": exposure, "position": "net asset" if exposure > 0 else ("net liability" if exposure < 0
                                                                               else "balanced"),
            "translation_result": "gain" if sign > 0 else ("loss" if sign < 0 else "none"), "reported_in": where}
