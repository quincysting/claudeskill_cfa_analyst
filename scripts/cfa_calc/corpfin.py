"""Corporate finance (CFA Level I-II): capital budgeting, cost of capital, leverage, capital structure, payout,
working capital, and M&A.

Rates are decimals, cash flows are lists with the first element at t = 0 (outflows negative).
"""
from __future__ import annotations

import itertools
import math

import numpy as np

from ._registry import formula
from .tvm import _irr_roots

_T = "corporate_finance"
L1CB = "L1 Corporate Finance R36: Capital Budgeting"
L2CB = "L2 Corporate Finance R25: Capital Budgeting"
L1CC = "L1 Corporate Finance R37: Cost of Capital"


# ----------------------------------------------------------------------------- private helpers

def _npv(rate: float, cash_flows) -> float:
    return sum(cf / (1.0 + rate) ** t for t, cf in enumerate(cash_flows))


def _irr_all(cash_flows) -> list[float]:
    """Every IRR in (-99.9%, 1,000,000%): the quant `irr` root finder, so both functions agree."""
    cfs = np.array([float(c) for c in cash_flows])
    if not (cfs < 0).any() or not (cfs > 0).any():
        return []
    return sorted(set(round(x, 12) for x in _irr_roots(cfs)))


def _irr(cash_flows) -> float | None:
    roots = _irr_all(cash_flows)
    return roots[0] if len(roots) == 1 else (min(roots, key=abs) if roots else None)


def _check_rate(name: str, value: float) -> None:
    if value is None or value <= -1:
        raise ValueError(f"{name} must be a rate greater than -1 (decimal), got {value!r}")


def _as_list(x, n: int, name: str) -> list[float]:
    if isinstance(x, (list, tuple)):
        if len(x) != n:
            raise ValueError(f"{name} has {len(x)} values but life is {n}")
        return [float(v) for v in x]
    return [float(x)] * n


def _payback(cash_flows) -> float | None:
    cum = cash_flows[0]
    if cum >= 0:
        return 0.0
    for t in range(1, len(cash_flows)):
        if cum + cash_flows[t] >= 0:
            return t - 1 + (-cum) / cash_flows[t]
        cum += cash_flows[t]
    return None


# ----------------------------------------------------------------------------- capital budgeting

@formula(_T, formula="NPV = sum CF_t/(1+r)^t; IRR: NPV=0; PI = PV(CF_1..n)/|CF_0|; MIRR = (FV inflows @ reinvest "
                     "/ PV outflows @ r)^(1/n) - 1", curriculum=L1CB)
def capital_project_metrics(cash_flows: list, rate: float, reinvestment_rate: float | None = None,
                            shares_outstanding: float | None = None) -> dict:
    """NPV, IRR, MIRR, profitability index, payback and discounted payback for one project.

    Args:
        cash_flows: after-tax cash flows, CF_0 first (initial outlay negative).
        rate: required rate of return (decimal).
        reinvestment_rate: rate at which inflows are reinvested for MIRR (default = rate).
        shares_outstanding: if given, also returns NPV per share (the expected stock price effect).

    Returns:
        npv, irr (None when the cash flows have no IRR or more than one; irr_all then lists every root found),
        irr_all (all IRRs between -99.9% and 1,000,000%, the same search as quant `irr`), mirr,
        profitability_index, payback_period,
        discounted_payback_period (None if never recovered), pv_future_cash_flows.

    Example:
        capital_project_metrics([-50, 16, 16, 16, 16, 20], 0.10) -> npv 13.136, irr 0.195197, PI 1.263
    """
    cfs = [float(c) for c in cash_flows]
    if len(cfs) < 2:
        raise ValueError("cash_flows needs at least CF_0 and one future cash flow")
    _check_rate("rate", rate)
    rr = rate if reinvestment_rate is None else reinvestment_rate
    n = len(cfs) - 1
    npv = _npv(rate, cfs)
    pv_future = npv - cfs[0]
    fv_in = sum(cf * (1 + rr) ** (n - t) for t, cf in enumerate(cfs) if cf > 0)
    pv_out = -sum(cf / (1 + rate) ** t for t, cf in enumerate(cfs) if cf < 0)
    roots = _irr_all(cfs)
    out = {
        "npv": npv,
        "irr": roots[0] if len(roots) == 1 else None,
        "irr_all": roots,
        "mirr": (fv_in / pv_out) ** (1 / n) - 1 if pv_out > 0 and fv_in > 0 else None,
        "profitability_index": pv_future / -cfs[0] if cfs[0] < 0 else None,
        "payback_period": _payback(cfs),
        "discounted_payback_period": _payback([cf / (1 + rate) ** t for t, cf in enumerate(cfs)]),
        "pv_future_cash_flows": pv_future,
    }
    if shares_outstanding:
        out["npv_per_share"] = npv / shares_outstanding
    return out


@formula(_T, formula="AAR = average net income / average book value", curriculum=L1CB)
def average_accounting_rate_of_return(net_incomes: list, initial_investment: float,
                                      salvage_value: float = 0.0) -> dict:
    """Average accounting rate of return (straight-line book value from investment to salvage).

    Example:
        average_accounting_rate_of_return([6000, 24000, 48000, 18000, -6000], 200000) -> aar 0.18
    """
    if not net_incomes:
        raise ValueError("net_incomes is empty")
    avg_ni = sum(net_incomes) / len(net_incomes)
    avg_bv = (initial_investment + salvage_value) / 2
    return {"aar": avg_ni / avg_bv, "average_net_income": avg_ni, "average_book_value": avg_bv}


@formula(_T, formula="crossover rate = IRR of (CF_A - CF_B)", curriculum=L1CB)
def npv_crossover_rate(cash_flows_a: list, cash_flows_b: list) -> dict:
    """Discount rate at which two projects' NPV profiles cross (the NPV/IRR ranking conflict point).

    Example:
        npv_crossover_rate([-200, 80, 80, 80, 80], [-200, 0, 0, 0, 400]) -> crossover_rate 0.1509, npv 27.98
    """
    if len(cash_flows_a) != len(cash_flows_b):
        raise ValueError("pad the shorter cash flow list with zeros so both have the same length")
    diff = [a - b for a, b in zip(cash_flows_a, cash_flows_b)]
    roots = _irr_all(diff)
    if not roots:
        raise ValueError("the NPV profiles do not cross between -99% and 1000%")
    r = min(roots, key=lambda x: abs(x - 0.1))
    return {"crossover_rate": r, "npv_at_crossover": _npv(r, cash_flows_a), "all_crossover_rates": roots}


@formula(_T, formula="Outlay = FCInv + NWCInv - Sal0 + T(Sal0 - B0); CF = (S - C - D)(1 - T) + D; "
                     "TNOCF = SalT + NWCInv - T(SalT - BT)", curriculum=L2CB)
def project_cash_flows(fixed_capital_investment: float, nwc_investment: float, sales, cash_operating_expenses,
                       depreciation, tax_rate: float, life: int, salvage_terminal: float = 0.0,
                       book_value_terminal: float = 0.0, salvage_initial: float = 0.0,
                       book_value_initial: float = 0.0, rate: float | None = None) -> dict:
    """Expansion or replacement project cash flows (initial outlay, annual after-tax OCF, terminal cash flow).

    For a replacement project pass incremental sales, cash expenses and depreciation (new minus old), the old
    asset's sale price and book value as salvage_initial / book_value_initial, and incremental terminal salvage.

    Args:
        sales, cash_operating_expenses, depreciation: one number per year or a list of `life` numbers.
        rate: if given, NPV, IRR and the PV of the depreciation tax savings are added.

    Returns:
        outlay, operating_cash_flows, terminal_nonoperating_cash_flow, cash_flows (t = 0..life), and npv, irr,
        pv_depreciation_tax_savings when rate is given.

    Example:
        project_cash_flows(200000, 30000, 220000, 90000, 35000, 0.40, 5, 50000, 25000, rate=0.10) -> npv 162217
    """
    if life < 1:
        raise ValueError("life must be at least one year")
    s, c, d = (_as_list(x, life, nm) for x, nm in ((sales, "sales"), (cash_operating_expenses, "cash expenses"),
                                                     (depreciation, "depreciation")))
    outlay = fixed_capital_investment + nwc_investment - salvage_initial \
        + tax_rate * (salvage_initial - book_value_initial)
    ocf = [(s[i] - c[i] - d[i]) * (1 - tax_rate) + d[i] for i in range(life)]
    tnocf = salvage_terminal + nwc_investment - tax_rate * (salvage_terminal - book_value_terminal)
    cfs = [-outlay] + ocf[:-1] + [ocf[-1] + tnocf]
    out = {"outlay": outlay, "operating_cash_flows": ocf, "terminal_nonoperating_cash_flow": tnocf,
           "cash_flows": cfs}
    if rate is not None:
        _check_rate("rate", rate)
        out["npv"] = _npv(rate, cfs)
        out["irr"] = _irr(cfs)
        out["pv_depreciation_tax_savings"] = sum(tax_rate * d[i] / (1 + rate) ** (i + 1) for i in range(life))
    return out


_MACRS = {  # U.S. MACRS half-year rates as printed in L2 R25 Table 16 (percent)
    3: [33.33, 44.45, 14.81, 7.41],
    5: [20.00, 32.00, 19.20, 11.52, 11.52, 5.76],
    7: [14.29, 24.49, 17.49, 12.49, 8.93, 8.93, 8.93, 4.45],
    10: [10.00, 18.00, 14.40, 11.52, 9.22, 7.37, 6.55, 6.55, 6.55, 6.55, 3.29],
    15: [5.00, 9.50, 8.55, 7.70, 6.93, 6.23] + [5.90] * 9 + [2.99],
    20: [3.75, 7.22, 6.68, 6.18, 5.71, 5.28, 4.89, 4.52] + [4.46] * 12 + [2.25],
}


@formula(_T, formula="D_t = MACRS rate_t x depreciable base", curriculum=L2CB + " (Table 16)")
def macrs_depreciation(depreciable_base: float, recovery_class: int) -> dict:
    """U.S. MACRS accelerated depreciation schedule (3, 5, 7, 10, 15 or 20-year class, half-year convention).

    Example:
        macrs_depreciation(175000, 3) -> depreciation [58327.5, 77787.5, 25917.5, 12967.5]
    """
    if recovery_class not in _MACRS:
        raise ValueError(f"recovery_class must be one of {sorted(_MACRS)}")
    rates = [r / 100 for r in _MACRS[recovery_class]]
    return {"depreciation": [depreciable_base * r for r in rates], "rates": rates}


@formula(_T, formula="EAA = NPV x r / (1 - (1+r)^-n); LCM chain NPV = sum NPV/(1+r)^(k n)", curriculum=L2CB)
def unequal_lives_comparison(projects: dict, rate: float) -> dict:
    """Compare mutually exclusive projects with unequal lives: equivalent annual annuity and LCM replacement chain.

    Args:
        projects: {name: cash_flows (CF_0 first)}.

    Returns:
        per-project npv, eaa, chain_npv over the least common multiple of lives, plus best_by_eaa.

    Example:
        unequal_lives_comparison({"S": [-100, 60, 90], "L": [-140, 80, 70, 60]}, 0.10) -> S eaa 16.67, L 14.34
    """
    _check_rate("rate", rate)
    if not projects:
        raise ValueError("projects is empty")
    lives = {k: len(v) - 1 for k, v in projects.items()}
    horizon = math.lcm(*lives.values())
    res = {}
    for name, cfs in projects.items():
        n, npv = lives[name], _npv(rate, cfs)
        annuity = (1 - (1 + rate) ** -n) / rate if rate else n
        chain = sum(npv / (1 + rate) ** (k * n) for k in range(horizon // n))
        res[name] = {"life": n, "npv": npv, "eaa": npv / annuity, "chain_npv": chain}
    return {"best_by_eaa": max(res, key=lambda k: res[k]["eaa"]), "lcm_horizon": horizon, "projects": res}


@formula(_T, formula="max sum NPV s.t. sum outlay <= budget (indivisible projects)", curriculum=L2CB)
def capital_rationing(projects: list, budget: float) -> dict:
    """Best set of indivisible projects under a fixed capital budget (exhaustive search; PI ranking for reference).

    Args:
        projects: list of {"name", "outlay", "npv"}; at most 20 projects.

    Example:
        projects 9:(600,300), 10:(600,270), 11:(200,80), 12:(400,100), budget 1000 -> choose 9 and 12, NPV 400
    """
    if len(projects) > 20:
        raise ValueError("at most 20 projects (exhaustive search)")
    best, best_npv = (), 0.0
    for k in range(1, len(projects) + 1):
        for combo in itertools.combinations(projects, k):
            if sum(p["outlay"] for p in combo) <= budget:
                v = sum(p["npv"] for p in combo)
                if v > best_npv:
                    best, best_npv = combo, v
    pi = sorted(((p["name"], 1 + p["npv"] / p["outlay"]) for p in projects), key=lambda x: -x[1])
    return {"selected": [p["name"] for p in best], "total_npv": best_npv,
            "total_outlay": sum(p["outlay"] for p in best), "pi_ranking": pi}


@formula(_T, formula="(1 + nominal) = (1 + real)(1 + inflation)", curriculum=L2CB + " (6.4 inflation)")
def project_real_nominal_rate(inflation_rate: float, nominal_rate: float | None = None,
                              real_rate: float | None = None) -> dict:
    """Convert a project discount rate between nominal and real terms (discount nominal CFs at nominal rates).

    Example:
        project_real_nominal_rate(0.03, real_rate=0.05) -> nominal_rate 0.0815
    """
    if (nominal_rate is None) == (real_rate is None):
        raise ValueError("give exactly one of nominal_rate or real_rate")
    if real_rate is None:
        real_rate = (1 + nominal_rate) / (1 + inflation_rate) - 1
    else:
        nominal_rate = (1 + real_rate) * (1 + inflation_rate) - 1
    return {"nominal_rate": nominal_rate, "real_rate": real_rate, "inflation_rate": inflation_rate}


@formula(_T, formula="Project NPV = NPV(DCF) - cost of options + value of options", curriculum=L2CB + " (real options)")
def real_option_adjusted_npv(npv_without_options: float, option_value: float, option_cost: float = 0.0) -> dict:
    """NPV including a real option (timing, sizing, flexibility, abandonment).

    Example:
        real_option_adjusted_npv(-0.40, 1.20, 0.30) -> npv 0.50
    """
    npv = npv_without_options - option_cost + option_value
    return {"npv": npv, "accept": npv > 0, "net_option_value": option_value - option_cost}


@formula(_T, formula="abandon at year k if salvage > PV(remaining CF); E[NPV] over scenarios",
         curriculum=L2CB + " (abandonment option)")
def abandonment_option_npv(initial_outlay: float, scenarios: list, abandonment_value: float,
                           abandonment_year: int, rate: float) -> dict:
    """Expected NPV with and without an option to abandon at a given year.

    Args:
        initial_outlay: positive amount invested at t = 0.
        scenarios: list of {"probability", "cash_flows" (years 1..n)}; probabilities sum to 1.
        abandonment_value: cash received if abandoned (after that year's cash flow).

    Example:
        Nyberg: outlay 200000, CF 40000 or 80000 x 4 yrs (50/50), abandon for 150000 at year 1, 10% ->
        npv_without_option -9808, npv_with_option 13158
    """
    _check_rate("rate", rate)
    if abs(sum(s["probability"] for s in scenarios) - 1) > 1e-9:
        raise ValueError("scenario probabilities must sum to 1")
    k, without, with_opt, detail = abandonment_year, 0.0, 0.0, []
    for s in scenarios:
        cfs = [-initial_outlay] + list(s["cash_flows"])
        if not 1 <= k < len(cfs) - 1:
            raise ValueError("abandonment_year must be before the final year")
        npv_keep = _npv(rate, cfs)
        pv_remaining_at_k = sum(cf / (1 + rate) ** (t - k) for t, cf in enumerate(cfs) if t > k)
        abandon = abandonment_value > pv_remaining_at_k
        npv_opt = _npv(rate, cfs[:k] + [cfs[k] + abandonment_value]) if abandon else npv_keep
        without += s["probability"] * npv_keep
        with_opt += s["probability"] * npv_opt
        detail.append({"abandon": abandon, "npv_continue": npv_keep, "npv_with_option": npv_opt})
    return {"npv_with_option": with_opt, "npv_without_option": without, "option_value": with_opt - without,
            "scenarios": detail}


# ----------------------------------------------------------------------------- cost of capital

@formula(_T, formula="WACC = wd rd (1 - t) + wp rp + we re", curriculum=L1CC)
def weighted_average_cost_of_capital(cost_of_debt: float, cost_of_equity: float, tax_rate: float,
                                     weight_debt: float | None = None, weight_preferred: float = 0.0,
                                     cost_of_preferred: float = 0.0, debt_value: float | None = None,
                                     equity_value: float | None = None, preferred_value: float = 0.0,
                                     de_ratio: float | None = None) -> dict:
    """WACC from weights, market values, or a debt-to-equity ratio (before-tax cost of debt).

    Give exactly one of: weight_debt (with weight_preferred), debt_value + equity_value (+ preferred_value),
    or de_ratio (no preferred).

    Example:
        weighted_average_cost_of_capital(0.08, 0.15, 0.40, weight_debt=0.3, weight_preferred=0.1,
                                         cost_of_preferred=0.10) -> wacc 0.1144
    """
    if debt_value is not None and equity_value is not None:
        v = debt_value + equity_value + preferred_value
        wd, wp = debt_value / v, preferred_value / v
    elif de_ratio is not None:
        wd, wp = de_ratio / (1 + de_ratio), 0.0
    elif weight_debt is not None:
        wd, wp = weight_debt, weight_preferred
    else:
        raise ValueError("give weight_debt, or debt_value and equity_value, or de_ratio")
    we = 1 - wd - wp
    if min(wd, wp, we) < 0:
        raise ValueError("weights must be non-negative and sum to 1")
    after_tax_rd = cost_of_debt * (1 - tax_rate)
    return {"wacc": wd * after_tax_rd + wp * cost_of_preferred + we * cost_of_equity,
            "weight_debt": wd, "weight_preferred": wp, "weight_equity": we, "after_tax_cost_of_debt": after_tax_rd}


@formula(_T, formula="rd = YTM (periodic yield x freq) or comparable-rated yield; after-tax = rd(1 - t)",
         curriculum=L1CC + " (3.1 cost of debt)")
def cost_of_debt_after_tax(tax_rate: float, price: float | None = None, face: float = 1000.0,
                           coupon_rate: float | None = None, years: float | None = None, freq: int = 2,
                           comparable_yield: float | None = None) -> dict:
    """Before- and after-tax cost of debt by the YTM approach (price given) or the debt-rating approach.

    Example:
        cost_of_debt_after_tax(0.35, price=1025, face=1000, coupon_rate=0.05, years=10, freq=2)
        -> before_tax 0.04684, after_tax 0.03045
    """
    if comparable_yield is not None:
        rd = comparable_yield
    else:
        if price is None or coupon_rate is None or years is None or price <= 0:
            raise ValueError("give price, coupon_rate and years (YTM approach) or comparable_yield")
        n, c = int(round(years * freq)), face * coupon_rate / freq
        if n < 1 or abs(years * freq - n) > 1e-9:
            raise ValueError(f"years x freq must be a whole number of coupon periods >= 1, got {years * freq}")
        rd = _irr([-price] + [c] * (n - 1) + [c + face]) * freq  # periodic YTM x freq (bond-equivalent)
    return {"after_tax_cost_of_debt": rd * (1 - tax_rate), "before_tax_cost_of_debt": rd}


@formula(_T, formula="rp = Dp / (Pp (1 - f))", curriculum=L1CC + " (3.2 preferred)")
def cost_of_preferred_stock(dividend: float, price: float, flotation_cost_pct: float = 0.0) -> dict:
    """Cost of (non-convertible, non-callable) preferred stock.

    Example:
        cost_of_preferred_stock(3.75, 72) -> 0.0521
    """
    if price <= 0:
        raise ValueError("price must be positive")
    return {"cost_of_preferred": dividend / (price * (1 - flotation_cost_pct))}


@formula(_T, formula="re = Rf + beta (ERP + CRP)", curriculum=L1CC + " (3.3.1 CAPM, 4.2 country risk)")
def cost_of_equity_capm(risk_free_rate: float, beta: float, equity_risk_premium: float,
                        country_risk_premium: float = 0.0) -> dict:
    """Cost of equity by CAPM, optionally adding a country risk premium to the equity risk premium.

    Example:
        cost_of_equity_capm(0.05, 1.5, 0.07) -> 0.155; cost_of_equity_capm(0.04, 1.2, 0.045, 0.03) -> 0.13
    """
    return {"cost_of_equity": risk_free_rate + beta * (equity_risk_premium + country_risk_premium)}


@formula(_T, formula="re = D1 / (P0 (1 - f) - F) + g; g = b x ROE", curriculum=L1CC + " (3.3.2 DDM, 4.4 flotation)")
def cost_of_equity_ddm(price: float, growth_rate: float | None = None, dividend_next: float | None = None,
                       dividend_current: float | None = None, retention_rate: float | None = None,
                       roe: float | None = None, flotation_cost_pct: float = 0.0,
                       flotation_cost_per_share: float = 0.0) -> dict:
    """Cost of equity from the constant-growth DDM, with optional flotation costs (cost of external equity).

    Example:
        cost_of_equity_ddm(40, growth_rate=0.05, dividend_current=2) -> 0.1025;
        with flotation_cost_pct=0.04 -> 0.10469
    """
    g = growth_rate if growth_rate is not None else (retention_rate * roe if retention_rate is not None
                                                     and roe is not None else None)
    if g is None:
        raise ValueError("give growth_rate, or retention_rate and roe")
    d1 = dividend_next if dividend_next is not None else (dividend_current * (1 + g) if dividend_current
                                                         is not None else None)
    if d1 is None:
        raise ValueError("give dividend_next or dividend_current")
    net_price = price * (1 - flotation_cost_pct) - flotation_cost_per_share
    if net_price <= 0:
        raise ValueError("price net of flotation costs must be positive")
    return {"cost_of_equity": d1 / net_price + g, "dividend_yield": d1 / net_price, "growth_rate": g}


@formula(_T, formula="re = rd + risk premium", curriculum=L1CC + " (3.3.3 bond yield plus risk premium)")
def cost_of_equity_bond_yield_plus(bond_yield: float, risk_premium: float) -> dict:
    """Cost of equity by the bond-yield-plus-risk-premium method.

    Example:
        cost_of_equity_bond_yield_plus(0.0495, 0.035) -> 0.0845
    """
    return {"cost_of_equity": bond_yield + risk_premium}


@formula(_T, formula="CRP = sovereign yield spread x (sigma equity index / sigma sovereign bond)",
         curriculum=L1CC + " (4.2 country risk)")
def country_risk_premium(sovereign_yield: float, developed_yield: float, equity_index_volatility: float,
                         sovereign_bond_volatility: float) -> dict:
    """Country risk premium for a developing market (sovereign spread scaled by relative volatility).

    Example:
        country_risk_premium(0.095, 0.045, 0.40, 0.28) -> 0.0714
    """
    if sovereign_bond_volatility <= 0:
        raise ValueError("sovereign_bond_volatility must be positive")
    spread = sovereign_yield - developed_yield
    return {"country_risk_premium": spread * equity_index_volatility / sovereign_bond_volatility,
            "sovereign_yield_spread": spread}


@formula(_T, formula="beta_A = beta_E / (1 + (1 - t) D/E); beta_E,project = beta_A (1 + (1 - t) D/E)",
         curriculum=L1CC + " (4.1 pure-play, Hamada)")
def beta_pure_play(target_de_ratio: float, target_tax_rate: float, comparable_beta: float | None = None,
                   comparable_de_ratio: float | None = None, comparable_tax_rate: float | None = None,
                   comparables: list | None = None, asset_beta: float | None = None) -> dict:
    """Unlever comparable equity betas (Hamada) and relever at the project's capital structure.

    Give one comparable (comparable_beta, comparable_de_ratio, comparable_tax_rate), a list `comparables` of
    {"beta", "de_ratio", "tax_rate"} (asset betas are averaged), or an asset_beta directly.

    Example:
        beta_pure_play(0.4, 0.35, comparable_beta=1.2, comparable_de_ratio=0.125, comparable_tax_rate=0.35)
        -> asset_beta 1.1098, levered_beta 1.3983
    """
    def unlever(b, de, t):
        return b / (1 + (1 - t) * de)

    if asset_beta is None:
        if comparables:
            betas = [unlever(c["beta"], c["de_ratio"], c["tax_rate"]) for c in comparables]
            asset_beta = sum(betas) / len(betas)
        elif comparable_beta is not None and comparable_de_ratio is not None and comparable_tax_rate is not None:
            asset_beta = unlever(comparable_beta, comparable_de_ratio, comparable_tax_rate)
        else:
            raise ValueError("give comparable_beta/de_ratio/tax_rate, comparables, or asset_beta")
    return {"levered_beta": asset_beta * (1 + (1 - target_tax_rate) * target_de_ratio), "asset_beta": asset_beta}


@formula(_T, formula="break point = amount at which a source's cost changes / weight of that source",
         curriculum=L1CC + " (4.3 marginal cost of capital)")
def marginal_cost_of_capital_schedule(weights: dict, tiers: dict) -> dict:
    """Marginal cost of capital break points and the WACC in each financing interval.

    Args:
        weights: {source: target weight}, summing to 1.
        tiers: {source: [[upper_amount, after_tax_cost], ..., [None, cost]]}; amounts are of that source alone,
            ascending; the last tier has upper_amount None. Use after-tax cost for debt.

    Example:
        weights {"debt": .4, "equity": .6}; debt tiers [[2, .02], [5, .025], [None, .03]], equity
        [[6, .05], [8, .07], [None, .09]] -> break points 5, 10, 12.5, 13.33
    """
    if abs(sum(weights.values()) - 1) > 1e-9 or set(weights) != set(tiers):
        raise ValueError("weights must sum to 1 and name the same sources as tiers")
    bps = sorted({t[0] / weights[s] for s, ts in tiers.items() for t in ts if t[0] is not None})
    edges = [0.0] + bps

    def cost_at(src, total):
        amt = total * weights[src]
        return next(c for up, c in tiers[src] if up is None or amt < up - 1e-12)

    schedule = []
    for i, lo in enumerate(edges):
        hi = edges[i + 1] if i + 1 < len(edges) else None
        probe = lo + 1e-9 * (1 + lo)
        schedule.append({"from": lo, "to": hi,
                         "wacc": sum(weights[s] * cost_at(s, probe) for s in weights)})
    return {"break_points": bps, "schedule": schedule}


@formula(_T, formula="NPV = PV(CF) - outlay - flotation cost x (1 - t if deductible)",
         curriculum=L1CC + " (4.4 flotation costs)")
def npv_with_flotation_costs(cash_flows: list, rate: float, flotation_cost: float, tax_rate: float = 0.0,
                             tax_deductible: bool = False) -> dict:
    """Project NPV with flotation costs treated as an initial cash outflow (the recommended approach).

    Example:
        npv_with_flotation_costs([-60000] + [10000]*10, 0.072, 1800) -> npv 7791 (8511 if deductible at 40%)
    """
    _check_rate("rate", rate)
    base = _npv(rate, cash_flows)
    cost = flotation_cost * (1 - tax_rate) if tax_deductible else flotation_cost
    return {"npv": base - cost, "npv_before_flotation": base, "flotation_cost_after_tax": cost}


# ----------------------------------------------------------------------------- leverage

@formula(_T, formula="DOL = Q(P-V)/[Q(P-V)-F]; DFL = EBIT/(EBIT-I); DTL = DOL x DFL; Q_OBE = F/(P-V); "
                     "Q_BE = (F+I)/(P-V)", curriculum="L1 Corporate Finance R38: Measures of Leverage")
def degree_of_leverage(units: float, price: float, variable_cost: float, fixed_cost: float,
                       interest: float = 0.0) -> dict:
    """Degree of operating, financial and total leverage, and operating / total breakeven quantities.

    Example:
        degree_of_leverage(6e6, 24000, 14000, 15e9, 40e9) -> DOL 1.333, DFL 9.0, DTL 12.0, Q_BE 5.5e6
    """
    cm = price - variable_cost
    if cm <= 0:
        raise ValueError("price must exceed variable cost per unit")
    contribution = units * cm
    ebit = contribution - fixed_cost
    ebt = ebit - interest
    dol = contribution / ebit if ebit else math.inf
    dfl = ebit / ebt if ebt else math.inf
    return {"dol": dol, "dfl": dfl, "dtl": dol * dfl, "ebit": ebit, "ebt": ebt, "contribution": contribution,
            "operating_breakeven_units": fixed_cost / cm, "breakeven_units": (fixed_cost + interest) / cm,
            "operating_breakeven_revenue": fixed_cost / cm * price,
            "breakeven_revenue": (fixed_cost + interest) / cm * price}


# ----------------------------------------------------------------------------- capital structure

@formula(_T, formula="VU = EBIT(1-t)/r0; VL = VU + tD - PV(distress); re = r0 + (r0 - rd)(1 - t) D/E",
         curriculum="L2 Corporate Finance R26: Capital Structure (MM I and II, static trade-off)")
def mm_firm_value(ebit: float, unlevered_cost_of_capital: float, debt: float, cost_of_debt: float,
                  tax_rate: float = 0.0, pv_financial_distress_costs: float = 0.0) -> dict:
    """Modigliani-Miller value, cost of equity and WACC (no-tax or corporate-tax case; perpetual EBIT).

    pv_financial_distress_costs > 0 gives the static trade-off value VL = VU + tD - PV(costs of financial distress).

    Example:
        mm_firm_value(5000, 0.10, 15000, 0.05, 0.25) -> VU 37500, VL 41250, re 0.12143, wacc 0.09091
    """
    r0 = unlevered_cost_of_capital
    if r0 <= 0:
        raise ValueError("unlevered_cost_of_capital must be positive")
    vu = ebit * (1 - tax_rate) / r0
    tax_shield = tax_rate * debt
    vl = vu + tax_shield - pv_financial_distress_costs
    equity = vl - debt
    if equity <= 0:
        raise ValueError("debt exceeds firm value")
    re = r0 + (r0 - cost_of_debt) * (1 - tax_rate) * debt / equity
    wacc = debt / vl * cost_of_debt * (1 - tax_rate) + equity / vl * re
    return {"levered_value": vl, "unlevered_value": vu, "equity_value": equity, "cost_of_equity": re,
            "wacc": wacc, "value_of_tax_shield": tax_shield}


@formula(_T, formula="re = r0 + (r0 - rd)(1 - t) D/E", curriculum="L2 Corporate Finance R26: Capital Structure")
def mm_cost_of_equity(unlevered_cost_of_capital: float, cost_of_debt: float, de_ratio: float,
                      tax_rate: float = 0.0) -> dict:
    """MM Proposition II cost of levered equity (set tax_rate 0 for the no-tax case).

    Example:
        mm_cost_of_equity(0.13, 0.09, 0.5) -> 0.15; with tax_rate 0.32 -> 0.1436
    """
    r0 = unlevered_cost_of_capital
    return {"cost_of_equity": r0 + (r0 - cost_of_debt) * (1 - tax_rate) * de_ratio}


# ----------------------------------------------------------------------------- payout

@formula(_T, formula="double: 1-(1-tc)(1-ti); split-rate: 1-(1-tc_div)(1-ti); imputation: ti",
         curriculum="L2 Corporate Finance R27: Dividends and Share Repurchases: Analysis (3.4.1)")
def dividend_effective_tax_rate(system: str, corporate_tax_rate: float, individual_tax_rate: float,
                                distributed_corporate_tax_rate: float | None = None) -> dict:
    """Effective tax rate on a unit of corporate earnings paid out as dividends.

    Args:
        system: "double_taxation", "split_rate" (needs distributed_corporate_tax_rate) or "imputation".

    Example:
        dividend_effective_tax_rate("double_taxation", 0.35, 0.396) -> 0.6074
    """
    if system == "double_taxation":
        eff = 1 - (1 - corporate_tax_rate) * (1 - individual_tax_rate)
    elif system == "split_rate":
        if distributed_corporate_tax_rate is None:
            raise ValueError("split_rate needs distributed_corporate_tax_rate")
        eff = 1 - (1 - distributed_corporate_tax_rate) * (1 - individual_tax_rate)
    elif system == "imputation":
        eff = individual_tax_rate
    else:
        raise ValueError("system must be double_taxation, split_rate or imputation")
    return {"effective_tax_rate": eff, "net_to_shareholder_per_unit": 1 - eff}


@formula(_T, formula="EPS after = (NI - amount x after-tax cost of funds) / (shares - repurchased); "
                     "BVPS after = (BV - amount)/(shares - repurchased)",
         curriculum="L1 Corporate Finance R39: Dividends and Share Repurchases: Basics (4.2)")
def share_repurchase_effects(shares_outstanding: float, net_income: float, share_price: float,
                             repurchase_amount: float | None = None, shares_repurchased: float | None = None,
                             after_tax_cost_of_funds: float = 0.0, book_value_equity: float | None = None) -> dict:
    """Effect of a buyback on EPS and BVPS (debt-financed: after-tax borrowing rate; cash: forgone after-tax yield).

    Example:
        share_repurchase_effects(2.2e6, 6.6e6, 60, repurchase_amount=12e6, after_tax_cost_of_funds=0.06)
        -> eps_after 2.94
    """
    if repurchase_amount is None and shares_repurchased is None:
        raise ValueError("give repurchase_amount or shares_repurchased")
    if repurchase_amount is None:
        repurchase_amount = shares_repurchased * share_price
    if shares_repurchased is None:
        shares_repurchased = repurchase_amount / share_price
    remaining = shares_outstanding - shares_repurchased
    if remaining <= 0:
        raise ValueError("cannot repurchase all shares")
    out = {"eps_after": (net_income - repurchase_amount * after_tax_cost_of_funds) / remaining,
           "eps_before": net_income / shares_outstanding, "shares_repurchased": shares_repurchased,
           "shares_after": remaining, "earnings_yield": net_income / shares_outstanding / share_price}
    if book_value_equity is not None:
        out["bvps_before"] = book_value_equity / shares_outstanding
        out["bvps_after"] = (book_value_equity - repurchase_amount) / remaining
    return out


@formula(_T, formula="E(D1) = D0 + (E1 - E0) x target payout x adjustment factor; classic Lintner: "
                     "D0 + adj x (target x E1 - D0)",
         curriculum="L2 Corporate Finance R27 (4.1.1 target payout adjustment model)")
def lintner_target_dividend(last_dividend: float, last_earnings: float, expected_earnings: float,
                            target_payout_ratio: float, adjustment_factor: float | None = None,
                            adjustment_years: float | None = None) -> dict:
    """Expected dividend under a target payout adjustment model (curriculum form and classic Lintner form).

    Example:
        lintner_target_dividend(0.40, 2.00, 2.80, 0.30, adjustment_years=5) -> expected_dividend 0.448
    """
    adj = adjustment_factor if adjustment_factor is not None else (1 / adjustment_years if adjustment_years
                                                                    else None)
    if adj is None or not 0 < adj <= 1:
        raise ValueError("give adjustment_factor in (0, 1] or adjustment_years >= 1")
    return {"expected_dividend": last_dividend + (expected_earnings - last_earnings) * target_payout_ratio * adj,
            "expected_dividend_lintner": last_dividend + adj * (target_payout_ratio * expected_earnings
                                                                - last_dividend),
            "adjustment_factor": adj}


@formula(_T, formula="payout = D/NI; coverage = NI/D; FCFE = CFO - FCInv + net borrowing; "
                     "FCFE coverage = FCFE/(D + repurchases)", curriculum="L2 Corporate Finance R27 (5 dividend safety)")
def dividend_coverage(net_income: float, dividends: float, cfo: float | None = None,
                      fixed_capital_investment: float = 0.0, net_borrowing: float = 0.0,
                      share_repurchases: float = 0.0) -> dict:
    """Dividend safety: payout ratio, earnings coverage, and FCFE coverage of dividends plus buybacks.

    Example:
        dividend_coverage(1043, 213, cfo=762, fixed_capital_investment=220, net_borrowing=493,
                          share_repurchases=936) -> fcfe 1035, fcfe_coverage 0.90
    """
    if dividends <= 0:
        raise ValueError("dividends must be positive")
    out = {"payout_ratio": dividends / net_income, "dividend_coverage": net_income / dividends}
    if cfo is not None:
        fcfe = cfo - fixed_capital_investment + net_borrowing
        out.update(fcfe=fcfe, fcfe_coverage=fcfe / (dividends + share_repurchases))
    return out


# ----------------------------------------------------------------------------- working capital

@formula(_T, formula="cost = (1 + d/(1-d))^(365/(payment day - discount days)) - 1",
         curriculum="L1 Corporate Finance R40: Working Capital Management (7.1)")
def cost_of_trade_credit(discount_rate: float, discount_period_days: float, payment_day: float,
                         days_in_year: float = 365.0) -> dict:
    """Effective annual cost of forgoing a trade discount (e.g. 2/10 net 30 paid on day 30).

    Example:
        cost_of_trade_credit(0.01, 10, 20) -> 0.4432; paid on day 30 -> 0.2013
    """
    extra = payment_day - discount_period_days
    if extra <= 0:
        raise ValueError("payment_day must be after the discount period")
    return {"cost_of_trade_credit": (1 + discount_rate / (1 - discount_rate)) ** (days_in_year / extra) - 1}


@formula(_T, formula="cost = (interest + fees) / usable proceeds, annualized; discount basis: proceeds = loan - interest",
         curriculum="L1 Corporate Finance R40 (8.4 computing the costs of borrowing)")
def short_term_borrowing_cost(amount: float, rate: float, term_fraction_of_year: float,
                              discount_basis: bool = False, commitment_fee: float = 0.0, dealer_fee: float = 0.0,
                              backup_fee: float = 0.0) -> dict:
    """Effective annual cost of a line of credit, banker's acceptance or commercial paper.

    Args:
        rate, commitment_fee, dealer_fee, backup_fee: annual rates applied to `amount` and prorated by the term.
        discount_basis: True for all-inclusive / discount instruments (BA, CP): proceeds = amount - interest.

    Example:
        CP: short_term_borrowing_cost(5e6, 0.0615, 1/12, True, dealer_fee=0.00125, backup_fee=0.0025) -> 0.0656
    """
    if not 0 < term_fraction_of_year <= 1:
        raise ValueError("term_fraction_of_year must be in (0, 1]")
    f = term_fraction_of_year
    interest = rate * amount * f
    fees = (commitment_fee + dealer_fee + backup_fee) * amount * f
    proceeds = amount - interest if discount_basis else amount
    return {"effective_annual_cost": (interest + fees) / proceeds / f, "interest": interest, "fees": fees,
            "net_proceeds": proceeds}


@formula(_T, formula="MMY = (F-P)/P x 360/days; BEY = (F-P)/P x 365/days; discount yield = (F-P)/F x 360/days",
         curriculum="L1 Corporate Finance R40 (4.1.1 yields on short-term investments)")
def short_term_investment_yields(face_value: float, days: float, purchase_price: float | None = None,
                                 discount_rate: float | None = None) -> dict:
    """Money market, bond-equivalent and discount-basis yields of a short-term discount instrument.

    Example:
        short_term_investment_yields(100000, 91, discount_rate=0.0791) -> mmy 0.0807, bey 0.0818
    """
    if purchase_price is None:
        if discount_rate is None:
            raise ValueError("give purchase_price or discount_rate")
        purchase_price = face_value * (1 - discount_rate * days / 360)
    hpr = (face_value - purchase_price) / purchase_price
    return {"money_market_yield": hpr * 360 / days, "bond_equivalent_yield": hpr * 365 / days,
            "discount_basis_yield": (face_value - purchase_price) / face_value * 360 / days,
            "holding_period_yield": hpr, "purchase_price": purchase_price}


# ----------------------------------------------------------------------------- mergers and acquisitions

_L2MA = "L2 Corporate Finance R29: Mergers and Acquisitions"


@formula(_T, formula="HHI = sum (share x 100)^2", curriculum=_L2MA + " (6.1 antitrust)")
def hhi_merger_analysis(market_shares: list, merging: list) -> dict:
    """Pre- and post-merger Herfindahl-Hirschman Index and the likely antitrust response (1992 guidelines).

    Args:
        market_shares: decimals (0.25 for 25%).
        merging: 0-based indices of the firms that merge.

    Example:
        shares [.25,.20,.10,.10,.10,.05,.05,.05,.05,.05], merging [1, 2] -> 1450 -> 1850, change 400, challenge
    """
    if sum(market_shares) > 1 + 1e-9 or min(market_shares) < 0:
        raise ValueError("market_shares are non-negative decimals (0.25 for 25%) that sum to at most 1")
    if len(set(merging)) < 2:
        raise ValueError("merging needs at least two firm indices")
    pre = sum((s * 100) ** 2 for s in market_shares)
    merged = sum(market_shares[i] for i in merging)
    post = sum((s * 100) ** 2 for i, s in enumerate(market_shares) if i not in merging) + (merged * 100) ** 2
    change = post - pre
    if post < 1000:
        level, action = "not concentrated", "no action"
    elif post <= 1800:
        level, action = "moderately concentrated", "possible challenge" if change >= 100 else "no action"
    else:
        level, action = "highly concentrated", "challenge" if change >= 50 else "no action"
    return {"post_merger_hhi": post, "pre_merger_hhi": pre, "change_in_hhi": change, "concentration": level,
            "likely_action": action}


@formula(_T, formula="Target gain = PT - VT; Acquirer gain = S - (PT - VT); VA* = VA + VT + S - C",
         curriculum=_L2MA + " (7.2 bid evaluation)")
def merger_bid_evaluation(acquirer_value: float, acquirer_shares: float, target_value: float,
                          target_shares: float, synergies: float, cash_per_target_share: float = 0.0,
                          exchange_ratio: float = 0.0) -> dict:
    """Gains to target and acquirer shareholders for a cash, stock or mixed offer (stock dilution included).

    Args:
        acquirer_value, target_value: pre-merger market values of equity.
        exchange_ratio: acquirer shares given per target share.

    Example:
        Adagio stock offer: merger_bid_evaluation(1125, 75, 300, 30, 90, exchange_ratio=0.8)
        -> price_paid 367.27, target_gain 67.27, acquirer_gain 22.73
    """
    cash = cash_per_target_share * target_shares
    new_shares = exchange_ratio * target_shares
    combined = acquirer_value + target_value + synergies - cash
    post_price = combined / (acquirer_shares + new_shares)
    price_paid = cash + new_shares * post_price
    premium = price_paid - target_value
    return {"target_gain": premium, "acquirer_gain": synergies - premium, "price_paid": price_paid,
            "post_merger_value": combined, "post_merger_share_price": post_price, "new_shares_issued": new_shares,
            "target_ownership_of_combined": new_shares / (acquirer_shares + new_shares)}


@formula(_T, formula="shares issued = offer value / acquirer price; combined EPS = (EA + ET)/(NA + issued)",
         curriculum=_L2MA + " (3.6 bootstrapping earnings)")
def merger_bootstrap_eps(acquirer_price: float, acquirer_eps: float, acquirer_shares: float, target_price: float,
                         target_eps: float, target_shares: float, offer_premium: float = 0.0) -> dict:
    """Post-merger EPS of an all-stock deal, exchange ratio, and the bootstrapping effect.

    Example:
        merger_bootstrap_eps(100, 4.00, 100000, 50, 2.50, 50000) -> combined_eps 4.20, shares_issued 25000
    """
    offer = target_price * (1 + offer_premium) * target_shares
    issued = offer / acquirer_price
    earnings = acquirer_eps * acquirer_shares + target_eps * target_shares
    eps = earnings / (acquirer_shares + issued)
    combined_mv = acquirer_price * acquirer_shares + target_price * target_shares
    return {"combined_eps": eps, "shares_issued": issued, "exchange_ratio": issued / target_shares,
            "eps_accretion": eps - acquirer_eps, "efficient_market_pe": combined_mv / earnings,
            "price_at_acquirer_pe": eps * acquirer_price / acquirer_eps}


@formula(_T, formula="value = sum w_i x multiple_i x target metric_i; takeover price = value x (1 + mean premium)",
         curriculum=_L2MA + " (7.1.2-7.1.3 comparable company and transaction analysis)")
def ma_comparables_valuation(target_metrics: dict, comparables: list | None = None,
                             multiples: dict | None = None, weights: dict | None = None,
                             takeover_premiums: list | None = None, takeover_premium: float | None = None) -> dict:
    """Target value from comparable company or comparable transaction multiples.

    Args:
        target_metrics: {metric: target value per share}, e.g. {"eps": 1.95, "cfps": 4.12}.
        comparables: list of {"price", <metric>: value}; mean price/metric multiples are computed.
        multiples: {metric: multiple} to use directly instead of comparables.
        weights: {metric: weight} (default equal weights).
        takeover_premiums: list of [pre-deal price, deal price] for recent deals (comparable company analysis);
            or takeover_premium as a decimal. Omit for comparable transactions (premium already in the price).

    Example:
        multiples {"eps": 18.3, "cfps": 10.2, "bvps": 3.7, "sps": 2.4}, weights .2/.4/.2/.2 -> 47.65
    """
    if multiples is None:
        if not comparables:
            raise ValueError("give comparables or multiples")
        multiples = {m: sum(c["price"] / c[m] for c in comparables) / len(comparables) for m in target_metrics}
    w = weights or {m: 1 / len(target_metrics) for m in target_metrics}
    if abs(sum(w.values()) - 1) > 1e-9:
        raise ValueError("weights must sum to 1")
    values = {m: multiples[m] * target_metrics[m] for m in target_metrics}
    value = sum(w[m] * values[m] for m in values)
    if takeover_premiums:
        takeover_premium = sum(deal / pre - 1 for pre, deal in takeover_premiums) / len(takeover_premiums)
    prem = takeover_premium or 0.0
    return {"takeover_value": value * (1 + prem), "estimated_value": value, "takeover_premium": prem,
            "multiples": multiples, "values_by_metric": values}


@formula(_T, formula="V = sum FCF_t/(1+WACC)^t + TV/(1+WACC)^T; TV = FCF_T(1+g)/(WACC-g) or multiple x FCF_T",
         curriculum=_L2MA + " (7.1.1 discounted cash flow analysis)")
def ma_dcf_target_value(free_cash_flows: list, wacc: float, terminal_growth: float | None = None,
                        terminal_multiple: float | None = None, net_debt: float = 0.0,
                        shares: float | None = None) -> dict:
    """Two-stage DCF value of an acquisition target (first-stage FCFs plus a terminal value).

    Example:
        ma_dcf_target_value([1367, 1351, 1486, 1636, 1799], 0.0941, terminal_growth=0.06) -> firm value 41471
    """
    _check_rate("wacc", wacc)
    if (terminal_growth is None) == (terminal_multiple is None):
        raise ValueError("give exactly one of terminal_growth or terminal_multiple")
    if terminal_growth is not None and terminal_growth >= wacc:
        raise ValueError("terminal_growth must be below wacc")
    n = len(free_cash_flows)
    pv_fcf = sum(f / (1 + wacc) ** (t + 1) for t, f in enumerate(free_cash_flows))
    tv = (free_cash_flows[-1] * (1 + terminal_growth) / (wacc - terminal_growth) if terminal_growth is not None
          else terminal_multiple * free_cash_flows[-1])
    pv_tv = tv / (1 + wacc) ** n
    firm = pv_fcf + pv_tv
    out = {"firm_value": firm, "pv_free_cash_flows": pv_fcf, "terminal_value": tv, "pv_terminal_value": pv_tv,
           "equity_value": firm - net_debt}
    if shares:
        out["value_per_share"] = (firm - net_debt) / shares
    return out
