"""Economics: microeconomics, macroeconomics, growth, trade and currency exchange rates (L1-L3).

Conventions (see references/conventions.md):
- Rates, growth rates, shares, elasticities-as-percentages: decimals (5% = 0.05).
- Currency quotes are always PRICE/BASE: "USD/EUR" = 1.10 means 1 EUR (base) costs 1.10 USD (price).
  A rise in a P/B quote is an appreciation of the base currency. Pass pair names as "PRICE/BASE" strings.
- Linear demand and supply are written as quantity functions, Q = intercept + slope * P, unless a docstring says
  the function takes an inverse demand curve P = a - bQ (market-structure functions do, as the reading does).
"""
from __future__ import annotations

import math

from ._registry import formula

_R13 = "L1 Econ R13: Demand and Supply Analysis: Introduction"
_R15 = "L1 Econ R15: Demand and Supply Analysis: The Firm"
_R16 = "L1 Econ R16: The Firm and Market Structures"
_R17 = "L1 Econ R17: Aggregate Output, Prices, and Economic Growth"
_R18 = "L1 Econ R18: Understanding Business Cycles"
_R19 = "L1 Econ R19: Monetary and Fiscal Policy"
_R20 = "L1 Econ R20: International Trade and Capital Flows"
_R21 = "L1 Econ R21: Currency Exchange Rates"
_L2FX = "L2 Econ R14: Currency Exchange Rates: Determination and Forecasting"
_L2G = "L2 Econ R15: Economic Growth and the Investment Decision"
_L3CME = "L3 R17: Capital Market Expectations"
_L3EMV = "L3 R18: Equity Market Valuation"


# ----------------------------------------------------------------------------------------------- helpers
def _num(name: str, x) -> float:
    try:
        v = float(x)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be a number, got {x!r}") from None
    if not math.isfinite(v):
        raise ValueError(f"{name} must be finite, got {x!r}")
    return v


def _pos(name: str, x) -> float:
    v = _num(name, x)
    if v <= 0:
        raise ValueError(f"{name} must be > 0, got {x!r}")
    return v


def _vec(name: str, xs, n: int | None = None) -> list[float]:
    if isinstance(xs, (int, float)):
        xs = [xs]
    try:
        out = [_num(name, x) for x in xs]
    except TypeError:
        raise ValueError(f"{name} must be a list of numbers") from None
    if not out:
        raise ValueError(f"{name} must not be empty")
    if n is not None and len(out) != n:
        raise ValueError(f"{name} has {len(out)} values; expected {n}")
    return out


def _ascending(name: str, q: list[float]) -> None:
    if any(b <= a for a, b in zip(q, q[1:])):
        raise ValueError(f"{name} must be strictly increasing")


def _marginal(q: list[float], y: list[float]) -> list[float | None]:
    return [None] + [(y[i] - y[i - 1]) / (q[i] - q[i - 1]) for i in range(1, len(q))]


def _responsiveness(e: float) -> str:
    a = abs(e)
    if math.isclose(a, 1.0, abs_tol=1e-12):
        return "unit elastic"
    return "elastic" if a > 1 else "inelastic"


def _sign_label(x: float, pos: str, neg: str, zero: str) -> str:
    return pos if x > 0 else neg if x < 0 else zero


def _pair(quote: str) -> tuple[str, str]:
    parts = [p.strip().upper() for p in str(quote).split("/")]
    if len(parts) != 2 or not all(parts) or parts[0] == parts[1]:
        raise ValueError(f"currency pair must look like 'PRICE/BASE' (e.g. 'USD/EUR'), got {quote!r}")
    return parts[0], parts[1]


# ------------------------------------------------------------------------------- elasticity (L1 R13)
@formula("economics", formula="E = (dQ/dX)(X/Q)", curriculum=_R13 + " (Sec 4)")
def elasticity_point(dq_dx: float, x: float, q: float) -> dict:
    """Point elasticity of quantity with respect to one driver X (own price, income or another good's price).

    Args:
        dq_dx: slope of quantity with respect to X (for a linear demand function, the coefficient on X).
        x: level of X where the elasticity is measured.
        q: quantity at that point (non-zero).

    Returns:
        elasticity, abs_elasticity, responsiveness ("elastic" if |E| > 1, "inelastic" if < 1, "unit elastic").

    Example:
        elasticity_point(dq_dx=-400, x=3, q=10000) -> elasticity -0.12 (inelastic)
    """
    dq_dx, x, q = _num("dq_dx", dq_dx), _num("x", x), _num("q", q)
    if q == 0:
        raise ValueError("q must be non-zero")
    e = dq_dx * x / q
    return {"elasticity": e, "abs_elasticity": abs(e), "responsiveness": _responsiveness(e)}


@formula("economics", formula="E_arc = (dQ / Q_avg) / (dX / X_avg)", curriculum=_R13 + " (Sec 4.1)")
def elasticity_arc(q0: float, q1: float, x0: float, x1: float) -> dict:
    """Arc (midpoint) elasticity between two observed points, using average quantity and average X as bases.

    Args:
        q0, q1: quantity before and after.
        x0, x1: driver (price, income, other price) before and after; must differ.

    Returns:
        elasticity, pct_change_q (on the average base), pct_change_x, responsiveness.

    Example:
        elasticity_arc(q0=9200, q1=8800, x0=5, x1=6) -> elasticity -0.244
    """
    q0, q1, x0, x1 = (_num(n, v) for n, v in (("q0", q0), ("q1", q1), ("x0", x0), ("x1", x1)))
    if x0 == x1:
        raise ValueError("x0 and x1 must differ")
    if q0 + q1 == 0 or x0 + x1 == 0:
        raise ValueError("average quantity and average x must be non-zero")
    pq = (q1 - q0) / ((q0 + q1) / 2)
    px = (x1 - x0) / ((x0 + x1) / 2)
    e = pq / px
    return {"elasticity": e, "pct_change_q": pq, "pct_change_x": px, "responsiveness": _responsiveness(e)}


@formula("economics", formula="Q = a + b_P*P + b_I*I + b_Y*P_Y;  E_k = b_k * X_k / Q",
         curriculum=_R13 + " (Sec 4.5)")
def demand_function_elasticities(intercept: float, own_price_coef: float, own_price: float,
                                 income_coef: float = 0.0, income: float = 0.0,
                                 cross_price_coef: float = 0.0, other_price: float = 0.0) -> dict:
    """Own-price, income and cross-price elasticities from a linear demand function at a given point.

    Demand: Q = intercept + own_price_coef*P + income_coef*I + cross_price_coef*P_other.

    Returns:
        quantity, own_price_elasticity, income_elasticity, cross_price_elasticity, own_price_demand
        (elastic/inelastic), good_type (normal/inferior), relation_to_other_good (substitutes/complements),
        price_rise_effect_on_total_expenditure (rises if demand is inelastic).

    Example:
        demand_function_elasticities(intercept=8400, own_price_coef=-400, own_price=3, income_coef=60, income=50,
                                     cross_price_coef=-10, other_price=20) -> Q 10000, -0.12, 0.30, -0.02
    """
    a, bp, p = _num("intercept", intercept), _num("own_price_coef", own_price_coef), _num("own_price", own_price)
    bi, i = _num("income_coef", income_coef), _num("income", income)
    by, py = _num("cross_price_coef", cross_price_coef), _num("other_price", other_price)
    q = a + bp * p + bi * i + by * py
    if q <= 0:
        raise ValueError(f"demand function gives non-positive quantity {q} at these inputs")
    own, inc, cross = bp * p / q, bi * i / q, by * py / q
    return {
        "quantity": q,
        "own_price_elasticity": own,
        "income_elasticity": inc,
        "cross_price_elasticity": cross,
        "own_price_demand": _responsiveness(own),
        "good_type": _sign_label(inc, "normal", "inferior", "income-insensitive"),
        "relation_to_other_good": _sign_label(cross, "substitutes", "complements", "independent"),
        "price_rise_effect_on_total_expenditure": ("rises" if abs(own) < 1 else "falls" if abs(own) > 1
                                                   else "unchanged"),
    }


@formula("economics", formula="TR = P*Q; TR moves with P if |E| < 1, against P if |E| > 1",
         curriculum=_R13 + " (Sec 4.2)")
def total_revenue_test(p0: float, p1: float, q0: float, q1: float) -> dict:
    """Total revenue (expenditure) test of own-price elasticity between two price-quantity points.

    Returns:
        tr_change, tr0, tr1, arc_elasticity, demand ("elastic", "inelastic", "unit elastic").

    Example:
        total_revenue_test(p0=5, p1=6, q0=9200, q1=8800) -> TR 46,000 -> 52,800, inelastic
    """
    arc = elasticity_arc(q0=q0, q1=q1, x0=p0, x1=p1)
    tr0, tr1 = float(p0) * float(q0), float(p1) * float(q1)
    return {"tr_change": tr1 - tr0, "tr0": tr0, "tr1": tr1, "arc_elasticity": arc["elasticity"],
            "demand": arc["responsiveness"]}


# ------------------------------------------------------------------- markets and surplus (L1 R13/R16)
def _market(a_d: float, b_d: float, a_s: float, b_s: float) -> tuple[float, float]:
    if b_d >= 0:
        raise ValueError("demand_slope must be negative (Q_d = a + b*P with b < 0)")
    if b_s <= 0:
        raise ValueError("supply_slope must be positive (Q_s = c + d*P with d > 0)")
    p = (a_d - a_s) / (b_s - b_d)
    q = a_d + b_d * p
    if q <= 0 or p <= 0:
        raise ValueError(f"no positive equilibrium (P={p}, Q={q}); check the curves")
    return p, q


def _cs(a_d: float, b_d: float, p: float) -> float:
    pc = -a_d / b_d  # choke price
    return 0.0 if p >= pc else a_d * (pc - p) + b_d / 2 * (pc ** 2 - p ** 2)


def _ps(a_s: float, b_s: float, p: float) -> float:
    pl = max(0.0, -a_s / b_s)  # lowest price at which anything is supplied
    return 0.0 if p <= pl else a_s * (p - pl) + b_s / 2 * (p ** 2 - pl ** 2)


@formula("economics", formula="Q_d = a + bP, Q_s = c + dP; P* = (a - c)/(d - b); CS, PS = areas",
         curriculum=_R13 + " (Sec 3.6-3.11)")
def supply_demand_equilibrium(demand_intercept: float, demand_slope: float, supply_intercept: float,
                              supply_slope: float, price: float | None = None) -> dict:
    """Equilibrium of linear demand and supply, consumer/producer surplus, and excess demand at any price.

    Args:
        demand_intercept, demand_slope: Q_d = demand_intercept + demand_slope*P (slope < 0).
        supply_intercept, supply_slope: Q_s = supply_intercept + supply_slope*P (slope > 0).
        price: optional non-equilibrium price at which to report excess demand (negative = excess supply).

    Returns:
        equilibrium_price, equilibrium_quantity, consumer_surplus, producer_surplus, total_surplus,
        and with `price`: quantity_demanded, quantity_supplied, excess_demand.

    Example:
        supply_demand_equilibrium(demand_intercept=180, demand_slope=-2, supply_intercept=-15, supply_slope=1)
        -> P* 65, Q* 50, CS 625, PS 1250
    """
    a_d, b_d = _num("demand_intercept", demand_intercept), _num("demand_slope", demand_slope)
    a_s, b_s = _num("supply_intercept", supply_intercept), _num("supply_slope", supply_slope)
    p, q = _market(a_d, b_d, a_s, b_s)
    cs, ps = _cs(a_d, b_d, p), _ps(a_s, b_s, p)
    out = {"equilibrium_price": p, "equilibrium_quantity": q, "consumer_surplus": cs, "producer_surplus": ps,
           "total_surplus": cs + ps}
    if price is not None:
        pr = _num("price", price)
        qd, qs = max(0.0, a_d + b_d * pr), max(0.0, a_s + b_s * pr)
        out.update(quantity_demanded=qd, quantity_supplied=qs, excess_demand=qd - qs)
    return out


@formula("economics", formula="CS = integral from P to choke price of Q_d(P) dP", curriculum=_R13 + " (Sec 3.9)")
def consumer_surplus_linear(demand_intercept: float, demand_slope: float, price: float) -> dict:
    """Consumer surplus under a linear demand curve Q = demand_intercept + demand_slope*P at a given price.

    Returns:
        consumer_surplus, quantity_demanded, choke_price (price at which Q = 0).

    Example:
        consumer_surplus_linear(demand_intercept=180, demand_slope=-2, price=65) -> 625
    """
    a, b, p = _num("demand_intercept", demand_intercept), _num("demand_slope", demand_slope), _num("price", price)
    if b >= 0:
        raise ValueError("demand_slope must be negative")
    return {"consumer_surplus": _cs(a, b, p), "quantity_demanded": max(0.0, a + b * p), "choke_price": -a / b}


@formula("economics", formula="DWL = 1/2 * (P_demand(Q') - P_supply(Q')) * (Q* - Q')",
         curriculum=_R13 + " (Sec 3.13)")
def price_control_effects(demand_intercept: float, demand_slope: float, supply_intercept: float,
                          supply_slope: float, control_price: float, control_type: str) -> dict:
    """Quantity traded, shortage/surplus and deadweight loss from a price floor or ceiling (linear curves).

    Args:
        demand_*/supply_*: Q = intercept + slope*P for each curve.
        control_price: the legal floor or ceiling.
        control_type: "floor" (binding if above P*) or "ceiling" (binding if below P*).

    Returns:
        deadweight_loss, binding, quantity_traded, excess_supply (floor) or shortage (ceiling),
        equilibrium_price, equilibrium_quantity, consumer_surplus_after, producer_surplus_after.
        Surplus after assumes the units traded go to the highest-value buyers / lowest-cost sellers.

    Example:
        price_control_effects(180, -2, -15, 1, control_price=72, control_type="floor") -> DWL 147
    """
    a_d, b_d = _num("demand_intercept", demand_intercept), _num("demand_slope", demand_slope)
    a_s, b_s = _num("supply_intercept", supply_intercept), _num("supply_slope", supply_slope)
    pc = _num("control_price", control_price)
    kind = str(control_type).lower()
    if kind not in ("floor", "ceiling"):
        raise ValueError("control_type must be 'floor' or 'ceiling'")
    p, q = _market(a_d, b_d, a_s, b_s)
    ts = _cs(a_d, b_d, p) + _ps(a_s, b_s, p)
    qd, qs = max(0.0, a_d + b_d * pc), max(0.0, a_s + b_s * pc)
    binding = pc > p if kind == "floor" else pc < p
    out = {"deadweight_loss": 0.0, "binding": binding, "quantity_traded": q, "equilibrium_price": p,
           "equilibrium_quantity": q}
    if not binding:
        out.update(consumer_surplus_after=_cs(a_d, b_d, p), producer_surplus_after=_ps(a_s, b_s, p))
        return out
    if kind == "floor":
        qt = qd
        dwl = 0.5 * (pc - (qt - a_s) / b_s) * (q - qt)
        cs_after = _cs(a_d, b_d, pc)
        out.update(excess_supply=qs - qd, consumer_surplus_after=cs_after, producer_surplus_after=ts - dwl - cs_after)
    else:
        qt = qs
        dwl = 0.5 * ((qt - a_d) / b_d - pc) * (q - qt)
        ps_after = _ps(a_s, b_s, pc)
        out.update(shortage=qd - qs, producer_surplus_after=ps_after, consumer_surplus_after=ts - dwl - ps_after)
    out.update(deadweight_loss=dwl, quantity_traded=qt)
    return out


@formula("economics", formula="Q_d(P_b) = Q_s(P_b - t); DWL = t*(Q* - Q_t)/2; revenue = t*Q_t",
         curriculum=_R13 + " (Sec 3.13)")
def per_unit_tax_effects(demand_intercept: float, demand_slope: float, supply_intercept: float,
                         supply_slope: float, tax: float) -> dict:
    """Incidence of a per-unit tax with linear curves (identical whether levied on buyers or sellers).

    Returns:
        price_paid_by_buyers, price_received_by_sellers, quantity_after_tax, tax_revenue, deadweight_loss,
        buyer_burden_per_unit, seller_burden_per_unit, equilibrium_price, equilibrium_quantity (pre-tax).

    Example:
        per_unit_tax_effects(180, -2, -15, 1, tax=2) -> buyers pay 65.67, sellers get 63.67, Q 48.67
    """
    a_d, b_d = _num("demand_intercept", demand_intercept), _num("demand_slope", demand_slope)
    a_s, b_s = _num("supply_intercept", supply_intercept), _num("supply_slope", supply_slope)
    t = _num("tax", tax)
    if t < 0:
        raise ValueError("tax must be >= 0")
    p, q = _market(a_d, b_d, a_s, b_s)
    pb = (a_d - a_s + b_s * t) / (b_s - b_d)
    qt = a_d + b_d * pb
    if qt <= 0:
        raise ValueError("tax is large enough to eliminate the market")
    return {"price_paid_by_buyers": pb, "price_received_by_sellers": pb - t, "quantity_after_tax": qt,
            "tax_revenue": t * qt, "deadweight_loss": 0.5 * t * (q - qt), "buyer_burden_per_unit": pb - p,
            "seller_burden_per_unit": p - (pb - t), "equilibrium_price": p, "equilibrium_quantity": q}


# ----------------------------------------------------------------- the firm: revenue, cost (L1 R15)
@formula("economics", formula="TR = P*Q; AR = TR/Q; MR = dTR/dQ", curriculum=_R15 + " (Sec 3.1.1)")
def revenue_schedule(quantities: list, prices: list) -> dict:
    """Total, average and marginal revenue from a price schedule (MR per unit between consecutive rows).

    Returns:
        total_revenue, average_revenue (None at Q = 0), marginal_revenue (None for the first row).

    Example:
        revenue_schedule(quantities=[0, 1, 2], prices=[100, 99, 98]) -> TR [0, 99, 196], MR [None, 99, 97]
    """
    q = _vec("quantities", quantities)
    p = _vec("prices", prices, len(q))
    _ascending("quantities", q)
    tr = [a * b for a, b in zip(q, p)]
    return {"total_revenue": tr, "average_revenue": [t / x if x else None for t, x in zip(tr, q)],
            "marginal_revenue": _marginal(q, tr)}


@formula("economics", formula="TC = TFC + TVC; AFC, AVC, ATC = cost/Q; MC = dTC/dQ",
         curriculum=_R15 + " (Sec 3.1.3)")
def cost_schedule(quantities: list, total_fixed_cost: float, total_variable_costs: list) -> dict:
    """Cost curves from total fixed cost and a total variable cost schedule, with shutdown and breakeven points.

    Under perfect competition the short-run shutdown price is min AVC and the breakeven price is min ATC.

    Returns:
        total_cost, average_fixed_cost, average_variable_cost, average_total_cost, marginal_cost (lists;
        None where undefined), min_avc / min_avc_quantity (shutdown point), min_atc / min_atc_quantity
        (breakeven point), min_mc / min_mc_quantity.

    Example:
        cost_schedule([0, 1, 2, 3, 4, 5], 5000, [0, 2000, 3800, 5400, 8000, 11000]) -> min AVC 1800 at Q=3
    """
    q = _vec("quantities", quantities)
    tvc = _vec("total_variable_costs", total_variable_costs, len(q))
    _ascending("quantities", q)
    if q[0] < 0:
        raise ValueError("quantities must be >= 0")
    tfc = _num("total_fixed_cost", total_fixed_cost)
    tc = [tfc + v for v in tvc]
    afc = [tfc / x if x else None for x in q]
    avc = [v / x if x else None for v, x in zip(tvc, q)]
    atc = [c / x if x else None for c, x in zip(tc, q)]
    mc = _marginal(q, tc)

    def _min(vals):
        pts = [(v, x) for v, x in zip(vals, q) if v is not None]
        return min(pts) if pts else (None, None)

    (mavc, qavc), (matc, qatc), (mmc, qmc) = _min(avc), _min(atc), _min(mc)
    return {"total_cost": tc, "average_fixed_cost": afc, "average_variable_cost": avc, "average_total_cost": atc,
            "marginal_cost": mc, "min_avc": mavc, "min_avc_quantity": qavc, "min_atc": matc,
            "min_atc_quantity": qatc, "min_mc": mmc, "min_mc_quantity": qmc}


@formula("economics", formula="profit = TR - TC; breakeven where TR = TC; max profit where MR = MC",
         curriculum=_R15 + " (Sec 3.1.3-3.1.4)")
def profit_schedule(quantities: list, total_revenues: list, total_costs: list) -> dict:
    """Profit, breakeven points and the profit-maximizing output from revenue and cost schedules.

    Returns:
        max_profit, max_profit_quantity, profit (list), marginal_revenue, marginal_cost, breakeven_points
        (list of {"quantity": q} where profit = 0 exactly, or {"between": [q_lo, q_hi],
        "interpolated_quantity": q} where profit changes sign).

    Example:
        profit_schedule([30, 40, 50, 60], [277500, 360000, 437500, 510000], [300000, 360000, 420000, 480000])
        -> breakeven at 40, max profit 30,000 at 60
    """
    q = _vec("quantities", quantities)
    tr = _vec("total_revenues", total_revenues, len(q))
    tc = _vec("total_costs", total_costs, len(q))
    _ascending("quantities", q)
    pi = [r - c for r, c in zip(tr, tc)]
    be: list[dict] = []
    for i, v in enumerate(pi):
        if v == 0:
            be.append({"quantity": q[i]})
        elif i and pi[i - 1] != 0 and (pi[i - 1] < 0) != (v < 0):
            qi = q[i - 1] + (q[i] - q[i - 1]) * (-pi[i - 1]) / (v - pi[i - 1])
            be.append({"between": [q[i - 1], q[i]], "interpolated_quantity": qi})
    k = max(range(len(pi)), key=lambda j: pi[j])
    return {"max_profit": pi[k], "max_profit_quantity": q[k], "profit": pi, "marginal_revenue": _marginal(q, tr),
            "marginal_cost": _marginal(q, tc), "breakeven_points": be}


@formula("economics", formula="TR >= TC stay; TVC <= TR < TC operate short run, exit long run; TR < TVC shut down",
         curriculum=_R15 + " (Sec 3.1.3, Exhibit 21)")
def shutdown_decision(total_revenue: float, total_fixed_cost: float, total_variable_cost: float) -> dict:
    """Short-run (operate vs shut down) and long-run (stay vs exit) decision from revenue and cost totals.

    Returns:
        short_run, long_run, loss_if_operating (positive = loss), loss_if_shut_down (= fixed cost), profit.

    Example:
        shutdown_decision(total_revenue=2.0e6, total_fixed_cost=1.0e6, total_variable_cost=1.5e6)
        -> operate in the short run (loss 0.5m < 1m), exit in the long run
    """
    tr, tfc, tvc = (_num(n, v) for n, v in (("total_revenue", total_revenue),
                                            ("total_fixed_cost", total_fixed_cost),
                                            ("total_variable_cost", total_variable_cost)))
    if min(tr, tfc, tvc) < 0:
        raise ValueError("inputs must be >= 0")
    profit = tr - tfc - tvc
    if profit >= 0:
        sr, lr = "stay in market (TR >= TC)", "stay in market"
    elif tr >= tvc:
        sr, lr = "operate: TR covers TVC and part of TFC", "exit market unless TR rises to cover TC"
    else:
        sr, lr = "shut down production: TR < TVC", "exit market"
    return {"short_run": sr, "long_run": lr, "loss_if_operating": -profit, "loss_if_shut_down": tfc,
            "profit": profit}


@formula("economics", formula="AP = TP/L; MP = dTP/dL", curriculum=_R15 + " (Sec 3.2)")
def product_schedule(inputs: list, total_products: list) -> dict:
    """Average and marginal product of a variable input, with the onset of diminishing marginal returns.

    Returns:
        average_product, marginal_product (lists, None where undefined), max_marginal_product and
        max_marginal_product_input (diminishing marginal returns set in beyond it), max_average_product and
        max_average_product_input, negative_mp_from_input (first input with MP < 0, or None).

    Example:
        product_schedule([0, 1, 2, 3, 4], [0, 1000, 2500, 4500, 6400]) -> MP peaks at 2,000 with 3 units
    """
    x = _vec("inputs", inputs)
    tp = _vec("total_products", total_products, len(x))
    _ascending("inputs", x)
    ap = [t / v if v else None for t, v in zip(tp, x)]
    mp = _marginal(x, tp)
    mps = [(m, v) for m, v in zip(mp, x) if m is not None]
    aps = [(a, v) for a, v in zip(ap, x) if a is not None]
    neg = next((v for m, v in mps if m < 0), None)
    return {"average_product": ap, "marginal_product": mp,
            "max_marginal_product": max(mps)[0] if mps else None,
            "max_marginal_product_input": max(mps)[1] if mps else None,
            "max_average_product": max(aps)[0] if aps else None,
            "max_average_product_input": max(aps)[1] if aps else None, "negative_mp_from_input": neg}


@formula("economics", formula="least cost: MP_1/P_1 = ... = MP_n/P_n; profit max: MRP_i = MP_i*P = P_i",
         curriculum=_R15 + " (Sec 3.2.2)")
def input_productivity_ratios(marginal_products: list, input_prices: list, product_price: float | None = None) -> dict:
    """Output per currency unit of each input (MP/P) and, given the product price, MRP and MRP/P.

    Hire first the input with the highest ratio; at the optimum all MP/P ratios are equal and every MRP/P = 1.

    Returns:
        mp_per_price, best_input_index (0-based, highest MP/P), and with product_price: marginal_revenue_product,
        mrp_per_price, profit_contribution (MRP - input price).

    Example:
        input_productivity_ratios([200, 500, 1000], [100, 125, 200], product_price=0.5)
        -> MP/P [2, 4, 5]; MRP [100, 250, 500]; MRP/P [1, 2, 2.5]
    """
    mp = _vec("marginal_products", marginal_products)
    pr = _vec("input_prices", input_prices, len(mp))
    if any(p <= 0 for p in pr):
        raise ValueError("input_prices must be > 0")
    ratio = [m / p for m, p in zip(mp, pr)]
    out = {"mp_per_price": ratio, "best_input_index": max(range(len(ratio)), key=lambda i: ratio[i])}
    if product_price is not None:
        pp = _pos("product_price", product_price)
        mrp = [m * pp for m in mp]
        out.update(marginal_revenue_product=mrp, mrp_per_price=[m / p for m, p in zip(mrp, pr)],
                   profit_contribution=[m - p for m, p in zip(mrp, pr)])
    return out


# ------------------------------------------------------------------------ market structure (L1 R16)
@formula("economics", formula="MR = P(1 - 1/|E|) = MC  ->  P = MC / (1 - 1/|E|)", curriculum=_R16 + " (Sec 6.3)")
def monopoly_price_from_elasticity(marginal_cost: float, price_elasticity: float) -> dict:
    """Profit-maximizing price from marginal cost and the price elasticity of demand (|E| must exceed 1).

    Returns:
        price, markup_over_mc (P/MC - 1), lerner_index ((P - MC)/P = 1/|E|).

    Example:
        monopoly_price_from_elasticity(marginal_cost=75, price_elasticity=-1.5) -> 225
    """
    mc = _pos("marginal_cost", marginal_cost)
    e = abs(_num("price_elasticity", price_elasticity))
    if e <= 1:
        raise ValueError("|price_elasticity| must exceed 1: a profit maximizer never prices on inelastic demand")
    p = mc / (1 - 1 / e)
    return {"price": p, "markup_over_mc": p / mc - 1, "lerner_index": (p - mc) / p}


@formula("economics", formula="P = a - bQ, MC = c: competitive Q=(a-c)/b; monopoly Q=(a-c)/2b; "
                              "Cournot (n firms) Q = n(a-c)/((n+1)b)", curriculum=_R16 + " (Sec 5.3)")
def linear_market_structure_outcomes(demand_intercept: float, demand_slope: float, marginal_cost: float,
                                     n_firms: int = 2) -> dict:
    """Price and output under perfect competition, Cournot oligopoly (n identical firms) and monopoly.

    Args:
        demand_intercept, demand_slope: INVERSE market demand P = demand_intercept - demand_slope*Q (slope > 0).
        marginal_cost: constant marginal cost (no fixed cost).
        n_firms: number of Cournot competitors (2 = duopoly).

    Returns:
        competitive_price, competitive_quantity, cournot_price, cournot_quantity, cournot_quantity_per_firm,
        monopoly_price, monopoly_quantity, monopoly_profit.

    Example:
        linear_market_structure_outcomes(450, 1, 30, n_firms=2) -> Cournot P 170, Q 280; monopoly P 240, Q 210
    """
    a, b, c = _num("demand_intercept", demand_intercept), _pos("demand_slope", demand_slope), \
        _num("marginal_cost", marginal_cost)
    if int(n_firms) != n_firms or n_firms < 1:
        raise ValueError("n_firms must be a positive whole number")
    if a <= c:
        raise ValueError("demand_intercept must exceed marginal_cost")
    n = int(n_firms)
    qc, qm, qn = (a - c) / b, (a - c) / (2 * b), n * (a - c) / ((n + 1) * b)
    pm = a - b * qm
    return {"competitive_price": c, "competitive_quantity": qc, "cournot_price": a - b * qn,
            "cournot_quantity": qn, "cournot_quantity_per_firm": qn / n, "monopoly_price": pm,
            "monopoly_quantity": qm, "monopoly_profit": (pm - c) * qm}


@formula("economics", formula="P = a - bQ, TC = F + cQ + dQ^2; MR = a - 2bQ = MC = c + 2dQ",
         curriculum=_R16 + " (Sec 6.3)")
def monopoly_optimum_quadratic_cost(demand_intercept: float, demand_slope: float, fixed_cost: float = 0.0,
                                    cost_linear: float = 0.0, cost_quadratic: float = 0.0) -> dict:
    """Profit-maximizing monopoly output, price and profit with linear inverse demand and quadratic total cost.

    Args:
        demand_intercept, demand_slope: inverse demand P = a - b*Q (b > 0).
        fixed_cost, cost_linear, cost_quadratic: TC = F + c*Q + d*Q^2.

    Returns:
        quantity, price, profit, marginal_revenue (= marginal cost at the optimum), total_revenue, total_cost,
        point_price_elasticity at the optimum.

    Example:
        monopoly_optimum_quadratic_cost(800, 2, fixed_cost=20000, cost_linear=50, cost_quadratic=3)
        -> Q 75, P 650, profit 8,125
    """
    a, b = _num("demand_intercept", demand_intercept), _pos("demand_slope", demand_slope)
    f, c, d = _num("fixed_cost", fixed_cost), _num("cost_linear", cost_linear), _num("cost_quadratic", cost_quadratic)
    if 2 * b + 2 * d <= 0:
        raise ValueError("need 2*demand_slope + 2*cost_quadratic > 0 for a maximum")
    q = (a - c) / (2 * b + 2 * d)
    if q <= 0:
        raise ValueError("no positive output: demand_intercept must exceed cost_linear")
    p = a - b * q
    tr, tc = p * q, f + c * q + d * q * q
    return {"quantity": q, "price": p, "profit": tr - tc, "marginal_revenue": a - 2 * b * q,
            "total_revenue": tr, "total_cost": tc, "point_price_elasticity": -(1 / b) * p / q}


@formula("economics", formula="CR_N = sum of top-N shares; HHI_N = sum of squared top-N shares",
         curriculum=_R16 + " (Sec 7)")
def concentration_measures(market_shares: list, top_n: int | None = None, merging_firms: list | None = None) -> dict:
    """N-firm concentration ratio and Herfindahl-Hirschman index, before and after an optional merger.

    Args:
        market_shares: shares as decimals (0.35 = 35%), in any order.
        top_n: N for CR_N and HHI_N (default: all firms).
        merging_firms: 0-based indices (into market_shares) of firms that merge; their shares are combined.

    Returns:
        concentration_ratio, hhi (top N, decimal scale 0-1), hhi_points (x 10,000), hhi_all_firms, and with
        merging_firms: post_merger_concentration_ratio, post_merger_hhi, hhi_change, post_merger_hhi_all_firms.

    Example:
        concentration_measures([0.35, 0.25, 0.20, 0.10, 0.025, 0.025, 0.025, 0.025], top_n=3, merging_firms=[0, 1])
        -> CR3 0.80 -> 0.90; HHI3 0.225 -> 0.41
    """
    s = _vec("market_shares", market_shares)
    if any(x < 0 or x > 1 for x in s):
        raise ValueError("market_shares must be decimals between 0 and 1")
    if sum(s) > 1 + 1e-9:
        raise ValueError(f"market_shares sum to {sum(s)}; they must not exceed 1")
    n = len(s) if top_n is None else int(top_n)
    if n < 1 or (top_n is not None and n != top_n):
        raise ValueError(f"top_n must be a whole number >= 1, got {top_n!r}")

    def _measures(shares):
        top = sorted(shares, reverse=True)[:n]
        return sum(top), sum(x * x for x in top), sum(x * x for x in shares)

    cr, hhi, hhi_all = _measures(s)
    out = {"concentration_ratio": cr, "hhi": hhi, "hhi_points": hhi * 1e4, "hhi_all_firms": hhi_all}
    if merging_firms:
        idx = {int(i) for i in merging_firms}
        if any(i < 0 or i >= len(s) for i in idx) or len(idx) < 2:
            raise ValueError("merging_firms must list at least two valid indices into market_shares")
        post = [x for i, x in enumerate(s) if i not in idx] + [sum(s[i] for i in idx)]
        cr2, hhi2, hhi2_all = _measures(post)
        out.update(post_merger_concentration_ratio=cr2, post_merger_hhi=hhi2, hhi_change=hhi2 - hhi,
                   post_merger_hhi_all_firms=hhi2_all)
    return out


# ---------------------------------------------------------------- national accounts (L1 R17-R18)
@formula("economics", formula="GDP = C + I + G + (X - M) + change in inventories + statistical discrepancy",
         curriculum=_R17 + " (Sec 2.2-2.3)")
def gdp_expenditure_approach(consumption: float, investment: float, government_spending: float, exports: float,
                             imports: float, inventory_change: float = 0.0,
                             statistical_discrepancy: float = 0.0) -> dict:
    """Nominal GDP by the expenditure approach.

    Args:
        investment: business gross fixed investment (add government investment to government_spending or here,
            following the source's accounts).
        inventory_change: change in inventories (if not already inside investment).

    Returns:
        gdp, net_exports.

    Example:
        gdp_expenditure_approach(898728, 269394, 333942 + 59078, 438553, 464722, -8180, 465) -> 1,527,258
    """
    vals = [_num(n, v) for n, v in (("consumption", consumption), ("investment", investment),
                                    ("government_spending", government_spending), ("exports", exports),
                                    ("imports", imports), ("inventory_change", inventory_change),
                                    ("statistical_discrepancy", statistical_discrepancy))]
    c, i, g, x, m, inv, sd = vals
    return {"gdp": c + i + g + x - m + inv + sd, "net_exports": x - m}


@formula("economics", formula="GDP = national income + capital consumption allowance + statistical discrepancy",
         curriculum=_R17 + " (Sec 2.3)")
def gdp_income_approach(national_income: float, capital_consumption_allowance: float,
                        statistical_discrepancy: float = 0.0) -> dict:
    """Nominal GDP by the income approach.

    Args:
        national_income: compensation + profits before tax + interest + proprietors' income + rent
            + inventory valuation adjustment + indirect business taxes - subsidies.

    Returns:
        gdp.

    Example:
        gdp_income_approach(1308939, 218785, -466) -> 1,527,258
    """
    return {"gdp": _num("national_income", national_income)
            + _num("capital_consumption_allowance", capital_consumption_allowance)
            + _num("statistical_discrepancy", statistical_discrepancy)}


@formula("economics", formula="PI = NI - indirect taxes - corporate taxes - undistributed profits + transfers; "
                              "S_household = PDI - C - interest to business - transfers to foreigners",
         curriculum=_R17 + " (Sec 2.3)")
def personal_income_measures(national_income: float, indirect_business_taxes: float, corporate_income_taxes: float,
                             undistributed_corporate_profits: float, transfer_payments: float,
                             personal_disposable_income: float | None = None, consumption: float | None = None,
                             interest_paid_to_business: float = 0.0, transfers_to_foreigners: float = 0.0) -> dict:
    """Personal income from national income, and (optionally) personal taxes, household saving and saving rate.

    Returns:
        personal_income, and with personal_disposable_income: personal_taxes (PI - PDI), and with consumption too:
        household_saving, saving_rate (saving / PDI).

    Example:
        personal_income_measures(1308939, 163634, 34319, 56969, 174390, 965628, 898728, 18115, 4737)
        -> PI 1,228,407; saving 44,048; saving rate 4.56%
    """
    pi = (_num("national_income", national_income) - _num("indirect_business_taxes", indirect_business_taxes)
          - _num("corporate_income_taxes", corporate_income_taxes)
          - _num("undistributed_corporate_profits", undistributed_corporate_profits)
          + _num("transfer_payments", transfer_payments))
    out = {"personal_income": pi}
    if personal_disposable_income is not None:
        pdi = _pos("personal_disposable_income", personal_disposable_income)
        out["personal_taxes"] = pi - pdi
        if consumption is not None:
            s = pdi - _num("consumption", consumption) - _num("interest_paid_to_business", interest_paid_to_business) \
                - _num("transfers_to_foreigners", transfers_to_foreigners)
            out.update(household_saving=s, saving_rate=s / pdi)
    return out


@formula("economics", formula="GDP deflator = nominal GDP / real GDP x 100; inflation = D_t / D_t-1 - 1",
         curriculum=_R17 + " (Sec 2.1.2)")
def gdp_deflator(nominal_gdp: list, real_gdp: list) -> dict:
    """Implicit GDP price deflator for one or more years and the inflation rate it implies.

    Args:
        nominal_gdp, real_gdp: values (or equal-length lists, oldest first) in the same currency units.

    Returns:
        deflator (list, base year = 100), inflation (list; None for the first year), real_growth (list).

    Example:
        gdp_deflator([1599608, 1527258], [1318054, 1285604]) -> deflators 121.4, 118.8; 2009 inflation -2.1%
    """
    n = _vec("nominal_gdp", nominal_gdp)
    r = _vec("real_gdp", real_gdp, len(n))
    if any(x <= 0 for x in r):
        raise ValueError("real_gdp must be > 0")
    d = [a / b * 100 for a, b in zip(n, r)]
    return {"deflator": d, "inflation": [None] + [d[i] / d[i - 1] - 1 for i in range(1, len(d))],
            "real_growth": [None] + [r[i] / r[i - 1] - 1 for i in range(1, len(r))]}


@formula("economics", formula="S = I + (G - T) + (X - M)", curriculum=_R17 + " (Sec 3.1.1)")
def sectoral_balances(private_saving: float | None = None, investment: float | None = None,
                      fiscal_deficit: float | None = None, trade_balance: float | None = None) -> dict:
    """Solve the saving-investment identity S = I + (G - T) + (X - M) for whichever term is omitted.

    Args:
        private_saving: S (household + business saving).
        investment: I.
        fiscal_deficit: G - T (positive = deficit).
        trade_balance: X - M (negative = deficit, financed by foreign saving).
        Exactly one must be omitted.

    Returns:
        all four terms, solved_for.

    Example:
        sectoral_balances(private_saving=319802, investment=261214, fiscal_deficit=82249) -> trade balance -23,661
    """
    vals = {"private_saving": private_saving, "investment": investment, "fiscal_deficit": fiscal_deficit,
            "trade_balance": trade_balance}
    missing = [k for k, v in vals.items() if v is None]
    if len(missing) != 1:
        raise ValueError("give exactly three of private_saving, investment, fiscal_deficit, trade_balance")
    v = {k: _num(k, x) for k, x in vals.items() if x is not None}
    k = missing[0]
    if k == "private_saving":
        v[k] = v["investment"] + v["fiscal_deficit"] + v["trade_balance"]
    elif k == "investment":
        v[k] = v["private_saving"] - v["fiscal_deficit"] - v["trade_balance"]
    elif k == "fiscal_deficit":
        v[k] = v["private_saving"] - v["investment"] - v["trade_balance"]
    else:
        v[k] = v["private_saving"] - v["investment"] - v["fiscal_deficit"]
    return {**{key: v[key] for key in vals}, "solved_for": k}


@formula("economics", formula="Laspeyres = sum p1 q0 / sum p0 q0; Paasche = sum p1 q1 / sum p0 q1; "
                              "Fisher = sqrt(L x P)", curriculum=_R18 + " (Sec 4.2.2)")
def price_indices(base_quantities: list, base_prices: list, current_quantities: list, current_prices: list) -> dict:
    """Laspeyres, Paasche and Fisher price indices (base period = 100) and the inflation each implies.

    Returns:
        laspeyres, paasche, fisher (index levels), and *_inflation (index / 100 - 1).

    Example:
        price_indices([50, 70], [3, 4.4], [70, 60], [4, 4.5]) -> Laspeyres 112.45, Paasche 116.03, Fisher 114.23
    """
    q0 = _vec("base_quantities", base_quantities)
    n = len(q0)
    p0 = _vec("base_prices", base_prices, n)
    q1 = _vec("current_quantities", current_quantities, n)
    p1 = _vec("current_prices", current_prices, n)
    dot = lambda a, b: sum(x * y for x, y in zip(a, b))  # noqa: E731
    if dot(p0, q0) <= 0 or dot(p0, q1) <= 0:
        raise ValueError("base-price basket values must be > 0")
    lasp, paas = dot(p1, q0) / dot(p0, q0) * 100, dot(p1, q1) / dot(p0, q1) * 100
    fish = math.sqrt(lasp * paas)
    return {"laspeyres": lasp, "paasche": paas, "fisher": fish, "laspeyres_inflation": lasp / 100 - 1,
            "paasche_inflation": paas / 100 - 1, "fisher_inflation": fish / 100 - 1}


@formula("economics", formula="labor force = employed + unemployed; u = unemployed / labor force; "
                              "participation = labor force / working-age population", curriculum=_R18 + " (Sec 4.1)")
def labor_market_rates(employed: float, unemployed: float, working_age_population: float | None = None) -> dict:
    """Unemployment rate and participation (activity) ratio.

    Returns:
        unemployment_rate, labor_force, and with working_age_population: participation_ratio.

    Example:
        labor_market_rates(employed=90, unemployed=10, working_age_population=125) -> u 10%, participation 80%
    """
    e, u = _num("employed", employed), _num("unemployed", unemployed)
    if e < 0 or u < 0 or e + u == 0:
        raise ValueError("employed and unemployed must be >= 0 and not both zero")
    lf = e + u
    out = {"unemployment_rate": u / lf, "labor_force": lf}
    if working_age_population is not None:
        out["participation_ratio"] = lf / _pos("working_age_population", working_age_population)
    return out


@formula("economics", formula="ULC = hourly compensation / output per hour", curriculum=_R17 + " (Sec 3.3.2)")
def unit_labor_cost(hourly_compensation: float, output_per_hour: float) -> dict:
    """Unit labor cost; it rises (shifting SRAS left) when wages grow faster than productivity.

    Example:
        unit_labor_cost(hourly_compensation=22, output_per_hour=110) -> 0.20
    """
    return {"unit_labor_cost": _num("hourly_compensation", hourly_compensation)
            / _pos("output_per_hour", output_per_hour)}


@formula("economics", formula="Y = C + I + G + NX with C = c0 + mpc(Y - T), T = t0 + t1 Y, I = i0 + i1 Y + i2 r, "
                              "NX = x0 + x1 Y  ->  Y = [c0 - mpc t0 + i0 + G + x0 + i2 r] / [1 - mpc(1 - t1) - i1 - x1]",
         curriculum=_R17 + " (Sec 3.1.1, Example 5)")
def is_curve_linear(autonomous_consumption: float, mpc: float, autonomous_investment: float,
                    investment_income_coef: float, investment_rate_coef: float, government_spending: float,
                    autonomous_net_exports: float, net_exports_income_coef: float, autonomous_taxes: float,
                    tax_income_coef: float, real_rate: float | None = None) -> dict:
    """IS curve (goods-market equilibrium) for a linear economy, and the expenditure components at a given rate.

    Coefficients carry their own signs as in the behavioural equations (e.g. investment_rate_coef = -30,
    net_exports_income_coef = -0.1, autonomous_taxes = -200). real_rate must be in the SAME units the
    investment_rate_coef assumes: the curriculum example writes I = 400 + 0.2Y - 30r with r = 4 meaning 4%.

    Returns:
        is_intercept, is_slope (dY/dr), spending_multiplier (dY/dG at a given r), and with real_rate:
        output, consumption, investment, net_exports, taxes.

    Example:
        is_curve_linear(2000, 0.7, 400, 0.2, -30, 1500, 1000, -0.1, -200, 0.3, real_rate=4)
        -> Y = 12,292.7 - 73.2r; Y(4) ~ 12,000; multiplier 2.44
    """
    c0, mpc = _num("autonomous_consumption", autonomous_consumption), _num("mpc", mpc)
    i0, i1, i2 = (_num("autonomous_investment", autonomous_investment), _num("investment_income_coef",
                  investment_income_coef), _num("investment_rate_coef", investment_rate_coef))
    g = _num("government_spending", government_spending)
    x0, x1 = _num("autonomous_net_exports", autonomous_net_exports), _num("net_exports_income_coef",
                                                                          net_exports_income_coef)
    t0, t1 = _num("autonomous_taxes", autonomous_taxes), _num("tax_income_coef", tax_income_coef)
    denom = 1 - mpc * (1 - t1) - i1 - x1
    if denom <= 0:
        raise ValueError("1 - mpc(1 - t) - investment and net-export income coefficients must be > 0 (stable model)")
    out = {"is_intercept": (c0 - mpc * t0 + i0 + g + x0) / denom, "is_slope": i2 / denom,
           "spending_multiplier": 1 / denom}
    if real_rate is not None:
        r = _num("real_rate", real_rate)
        y = out["is_intercept"] + out["is_slope"] * r
        t = t0 + t1 * y
        out.update(output=y, taxes=t, consumption=c0 + mpc * (y - t), investment=i0 + i1 * y + i2 * r,
                   net_exports=x0 + x1 * y)
    return out


# ------------------------------------------------------------------ money and policy (L1 R19, L2, L3)
@formula("economics", formula="money multiplier = 1 / reserve requirement; new money = deposit / reserve requirement",
         curriculum=_R19 + " (Sec 2.1.2)")
def money_multiplier(reserve_requirement: float, new_deposit: float = 0.0) -> dict:
    """Simple (maximum) money multiplier and the total deposits a new deposit can support.

    Returns:
        money_multiplier, total_deposits_created (new_deposit / reserve_requirement).

    Example:
        money_multiplier(reserve_requirement=0.10, new_deposit=100) -> multiplier 10, total 1,000
    """
    rr = _pos("reserve_requirement", reserve_requirement)
    if rr > 1:
        raise ValueError("reserve_requirement is a decimal between 0 and 1")
    return {"money_multiplier": 1 / rr, "total_deposits_created": _num("new_deposit", new_deposit) / rr}


@formula("economics", formula="M x V = P x Y", curriculum=_R19 + " (Sec 2.1.4)")
def quantity_theory_of_money(money_supply: float | None = None, velocity: float | None = None,
                             price_level: float | None = None, real_output: float | None = None,
                             nominal_gdp: float | None = None) -> dict:
    """Solve the equation of exchange MV = PY for the one missing quantity.

    Args:
        money_supply (M), velocity (V), price_level (P), real_output (Y); or nominal_gdp in place of P and Y.
        Leave out exactly one of M, V, P, Y (or of M, V when nominal_gdp is given).

    Returns:
        money_supply, velocity, nominal_gdp, and price_level / real_output when determinable.

    Example:
        quantity_theory_of_money(money_supply=1e12, nominal_gdp=7e12) -> velocity 7
    """
    m = None if money_supply is None else _pos("money_supply", money_supply)
    v = None if velocity is None else _pos("velocity", velocity)
    p = None if price_level is None else _pos("price_level", price_level)
    y = None if real_output is None else _pos("real_output", real_output)
    py = None if nominal_gdp is None else _pos("nominal_gdp", nominal_gdp)
    if py is None and p is not None and y is not None:
        py = p * y
    if py is not None and [m, v].count(None) == 1:
        m, v = (py / v, v) if m is None else (m, py / m)
    elif m is not None and v is not None and py is None and [p, y].count(None) == 1:
        py = m * v
        p, y = (py / y, y) if p is None else (p, py / p)
    else:
        raise ValueError("leave out exactly one of money_supply, velocity, price_level, real_output "
                         "(nominal_gdp may replace price_level and real_output)")
    return {"money_supply": m, "velocity": v, "nominal_gdp": py, "price_level": p, "real_output": y}


@formula("economics", formula="(1 + m)(1 + v) = (1 + inflation)(1 + g);  inflation ~ m + v - g",
         curriculum=_R19 + " (Sec 2.1.4)")
def quantity_theory_growth(money_growth: float, velocity_growth: float = 0.0,
                           real_output_growth: float = 0.0) -> dict:
    """Inflation implied by the quantity equation in growth-rate form.

    Returns:
        inflation (exact), inflation_approx (money growth + velocity growth - real growth), nominal_gdp_growth.

    Example:
        quantity_theory_growth(money_growth=0.05) -> inflation 5% with constant velocity and output
    """
    m, v, g = _num("money_growth", money_growth), _num("velocity_growth", velocity_growth), \
        _num("real_output_growth", real_output_growth)
    if g <= -1:
        raise ValueError("real_output_growth must exceed -100%")
    nominal = (1 + m) * (1 + v) - 1
    return {"inflation": (1 + nominal) / (1 + g) - 1, "inflation_approx": m + v - g, "nominal_gdp_growth": nominal}


@formula("economics", formula="(1 + i) = (1 + r)(1 + inflation);  i ~ r + inflation", curriculum=_R19 + " (Sec 2.1.7)")
def fisher_relation(nominal_rate: float | None = None, real_rate: float | None = None,
                    inflation: float | None = None) -> dict:
    """Fisher effect: solve for whichever of nominal rate, real rate or (expected) inflation is omitted.

    Also gives breakeven inflation from a nominal and an inflation-linked yield (omit inflation).

    Returns:
        nominal_rate, real_rate, inflation (exact multiplicative solution), approx (additive solution for the
        omitted term), solved_for.

    Example:
        fisher_relation(nominal_rate=0.035, real_rate=0.015) -> inflation approx 2.0%, exact 1.97%
    """
    vals = {"nominal_rate": nominal_rate, "real_rate": real_rate, "inflation": inflation}
    missing = [k for k, v in vals.items() if v is None]
    if len(missing) != 1:
        raise ValueError("give exactly two of nominal_rate, real_rate, inflation")
    v = {k: _num(k, x) for k, x in vals.items() if x is not None}
    k = missing[0]
    if k == "nominal_rate":
        v[k], approx = (1 + v["real_rate"]) * (1 + v["inflation"]) - 1, v["real_rate"] + v["inflation"]
    elif k == "real_rate":
        v[k], approx = (1 + v["nominal_rate"]) / (1 + v["inflation"]) - 1, v["nominal_rate"] - v["inflation"]
    else:
        v[k], approx = (1 + v["nominal_rate"]) / (1 + v["real_rate"]) - 1, v["nominal_rate"] - v["real_rate"]
    return {**{key: v[key] for key in vals}, "approx": approx, "solved_for": k}


@formula("economics", formula="fiscal multiplier = 1 / [1 - MPC(1 - t)]; balanced-budget multiplier = 1",
         curriculum=_R19 + " (Sec 3.2.2-3.2.3)")
def fiscal_multiplier(mpc: float, tax_rate: float = 0.0, change_in_spending: float = 0.0) -> dict:
    """Government spending multiplier with a proportional net tax rate, and the implied change in output.

    Returns:
        multiplier, output_change (multiplier x change_in_spending), mpc_after_tax (MPC(1 - t)), mps,
        balanced_budget_multiplier (1).

    Example:
        fiscal_multiplier(mpc=0.9, tax_rate=0.2) -> 3.57
    """
    c, t = _num("mpc", mpc), _num("tax_rate", tax_rate)
    if not 0 <= c < 1:
        raise ValueError("mpc must be in [0, 1)")
    if not 0 <= t < 1:
        raise ValueError("tax_rate must be in [0, 1)")
    k = 1 / (1 - c * (1 - t))
    return {"multiplier": k, "output_change": k * _num("change_in_spending", change_in_spending),
            "mpc_after_tax": c * (1 - t), "mps": 1 - c, "balanced_budget_multiplier": 1.0}


@formula("economics", formula="neutral rate = trend real growth + inflation target", curriculum=_R19 + " (Sec 2.4)")
def neutral_policy_rate(trend_real_growth: float, inflation_target: float, policy_rate: float | None = None) -> dict:
    """Neutral nominal policy rate and, given the actual policy rate, the monetary stance.

    Returns:
        neutral_rate, and with policy_rate: stance ("contractionary" above neutral, "expansionary" below),
        gap (policy - neutral).

    Example:
        neutral_policy_rate(trend_real_growth=0.025, inflation_target=0.02) -> 4.5%
    """
    n = _num("trend_real_growth", trend_real_growth) + _num("inflation_target", inflation_target)
    out = {"neutral_rate": n}
    if policy_rate is not None:
        gap = _num("policy_rate", policy_rate) - n
        out.update(gap=gap, stance=_sign_label(gap, "contractionary", "expansionary", "neutral"))
    return out


@formula("economics", formula="i = r_n + inflation + a(inflation - target) + b(output gap); real r = i - inflation",
         curriculum=_L2FX + " (Sec 6.3, Eq. 15-16)")
def taylor_rule_rate(neutral_real_rate: float, inflation: float, target_inflation: float, output_gap: float,
                     inflation_weight: float = 0.5, output_weight: float = 0.5) -> dict:
    """Taylor-rule nominal and real policy rate (Taylor 1993 weights 0.5/0.5 by default).

    Args:
        neutral_real_rate: r_n, the neutral real policy rate.
        inflation, target_inflation: current and target inflation.
        output_gap: (y - y*) as a decimal percentage deviation of output from potential.

    Returns:
        policy_rate, real_policy_rate, neutral_nominal_rate (r_n + target inflation).

    Example:
        taylor_rule_rate(0.02, 0.04, 0.02, 0.01) -> 7.5% (Taylor 1993: 2 + 4 + 0.5x2 + 0.5x1)
    """
    rn, pi, ps = _num("neutral_real_rate", neutral_real_rate), _num("inflation", inflation), \
        _num("target_inflation", target_inflation)
    gap, a, b = _num("output_gap", output_gap), _num("inflation_weight", inflation_weight), \
        _num("output_weight", output_weight)
    real = rn + a * (pi - ps) + b * gap
    return {"policy_rate": real + pi, "real_policy_rate": real, "neutral_nominal_rate": rn + ps}


@formula("economics", formula="R = R_neutral + 0.5(GDPg_forecast - GDPg_trend) + 0.5(I_forecast - I_target)",
         curriculum=_L3CME + " (Sec 4.1.6, Eq. 12)")
def taylor_rule_growth_form(neutral_rate: float, gdp_growth_forecast: float, gdp_growth_trend: float,
                            inflation_forecast: float, inflation_target: float, growth_weight: float = 0.5,
                            inflation_weight: float = 0.5) -> dict:
    """L3 capital-market-expectations form of the Taylor rule, using growth and inflation forecasts.

    Returns:
        optimal_rate, change_vs_neutral, growth_component, inflation_component.

    Example:
        taylor_rule_growth_form(0.035, 0.01, 0.03, 0.04, 0.025) -> 3.25%
    """
    rn = _num("neutral_rate", neutral_rate)
    g = _num("growth_weight", growth_weight) * (_num("gdp_growth_forecast", gdp_growth_forecast)
                                                - _num("gdp_growth_trend", gdp_growth_trend))
    i = _num("inflation_weight", inflation_weight) * (_num("inflation_forecast", inflation_forecast)
                                                      - _num("inflation_target", inflation_target))
    return {"optimal_rate": rn + g + i, "change_vs_neutral": g + i, "growth_component": g, "inflation_component": i}


# ------------------------------------------------------------------------------ growth (L1 R17, L2 R15, L3 R18)
@formula("economics", formula="dY/Y = dA/A + a dK/K + (1 - a) dL/L", curriculum=_L2G + " (Sec 4.3, Eq. 4); "
                                                                            + _L3EMV + " (Eq. 3)")
def growth_accounting(capital_share: float, capital_growth: float, labor_growth: float,
                      tfp_growth: float | None = None, output_growth: float | None = None) -> dict:
    """Solow growth accounting: output growth from inputs and TFP, or TFP (Solow residual) from output growth.

    Args:
        capital_share: alpha, capital's share of income (1 - labor share).
        capital_growth, labor_growth: growth of capital stock and labor input (e.g. hours worked).
        tfp_growth / output_growth: give exactly one; the other is solved.

    Returns:
        output_growth, tfp_growth, capital_contribution, labor_contribution, labor_productivity_growth
        (output growth - labor growth), capital_deepening (capital growth - labor growth).

    Example:
        growth_accounting(capital_share=0.383, capital_growth=0.0592, labor_growth=0.0123, tfp_growth=-0.0073)
        -> 2.30%
    """
    a = _num("capital_share", capital_share)
    if not 0 < a < 1:
        raise ValueError("capital_share must be between 0 and 1")
    gk, gl = _num("capital_growth", capital_growth), _num("labor_growth", labor_growth)
    if (tfp_growth is None) == (output_growth is None):
        raise ValueError("give exactly one of tfp_growth and output_growth")
    ck, cl = a * gk, (1 - a) * gl
    if tfp_growth is not None:
        ga = _num("tfp_growth", tfp_growth)
        gy = ga + ck + cl
    else:
        gy = _num("output_growth", output_growth)
        ga = gy - ck - cl
    return {"output_growth": gy, "tfp_growth": ga, "capital_contribution": ck, "labor_contribution": cl,
            "labor_productivity_growth": gy - gl, "capital_deepening": gk - gl}


@formula("economics", formula="potential GDP growth = labor force growth + labor productivity growth",
         curriculum=_R17 + " (Sec 4.3); " + _L2G + " (Eq. 5)")
def potential_gdp_growth(labor_force_growth: float, labor_productivity_growth: float,
                         actual_gdp_growth: float | None = None) -> dict:
    """Potential GDP growth by the labor-productivity method, and the growth gap if actual growth is given.

    Returns:
        potential_growth, and with actual_gdp_growth: growth_gap (actual - potential; positive suggests rising
        inflation pressure and tighter policy, negative suggests slack).

    Example:
        potential_gdp_growth(labor_force_growth=0.009, labor_productivity_growth=0.019) -> 2.8%
    """
    pot = _num("labor_force_growth", labor_force_growth) + _num("labor_productivity_growth", labor_productivity_growth)
    out = {"potential_growth": pot}
    if actual_gdp_growth is not None:
        out["growth_gap"] = _num("actual_gdp_growth", actual_gdp_growth) - pot
    return out


@formula("economics", formula="Y = A K^a L^(1-a); y = A k^a; MPK = aY/K; MPL = (1-a)Y/L",
         curriculum=_L2G + " (Sec 4.1, Eq. 2-3)")
def cobb_douglas_output(tfp: float, capital: float, labor: float, capital_share: float) -> dict:
    """Constant-returns Cobb-Douglas output, per-worker output and the marginal products of capital and labor.

    Returns:
        output, output_per_worker, capital_per_worker, marginal_product_capital, marginal_product_labor.

    Example:
        cobb_douglas_output(tfp=1, capital=100, labor=100, capital_share=0.3) -> output 100, MPK 0.3, MPL 0.7
    """
    A, K, L = _pos("tfp", tfp), _pos("capital", capital), _pos("labor", labor)
    a = _num("capital_share", capital_share)
    if not 0 < a < 1:
        raise ValueError("capital_share must be between 0 and 1")
    y = A * K ** a * L ** (1 - a)
    return {"output": y, "output_per_worker": y / L, "capital_per_worker": K / L,
            "marginal_product_capital": a * y / K, "marginal_product_labor": (1 - a) * y / L}


@formula("economics", formula="g_y* = theta/(1 - a); g_Y* = theta/(1 - a) + n; Y/K* = [theta/(1 - a) + delta + n]/s",
         curriculum=_L2G + " (Sec 5.1, Eq. 8-9)")
def neoclassical_steady_state(tfp_growth: float, capital_share: float, labor_force_growth: float = 0.0,
                              saving_rate: float | None = None, depreciation_rate: float | None = None) -> dict:
    """Neoclassical (Solow) steady state: sustainable per-capita and total growth, and the output-capital ratio.

    Args:
        tfp_growth: theta. capital_share: alpha (1 - labor share of income). labor_force_growth: n.
        saving_rate (s) and depreciation_rate (delta): give both to get the steady-state Y/K.

    Returns:
        per_capita_growth, output_growth, and with s and delta: output_capital_ratio (Psi),
        marginal_product_capital (alpha x Psi, the real interest rate in steady state).

    Example:
        neoclassical_steady_state(tfp_growth=0.025, capital_share=0.535, labor_force_growth=0.012) -> 6.58%
    """
    th, a, n = _num("tfp_growth", tfp_growth), _num("capital_share", capital_share), \
        _num("labor_force_growth", labor_force_growth)
    if not 0 < a < 1:
        raise ValueError("capital_share must be between 0 and 1")
    g = th / (1 - a)
    out = {"per_capita_growth": g, "output_growth": g + n}
    if (saving_rate is None) != (depreciation_rate is None):
        raise ValueError("give both saving_rate and depreciation_rate, or neither")
    if saving_rate is not None:
        s, d = _pos("saving_rate", saving_rate), _num("depreciation_rate", depreciation_rate)
        psi = (g + d + n) / s
        out.update(output_capital_ratio=psi, marginal_product_capital=a * psi)
    return out


@formula("economics", formula="dy/y = theta/(1 - a) + a s (Y/K - Psi); dk/k = theta/(1 - a) + s (Y/K - Psi); "
                              "k_new/k_old = [(Y/K)_new/(Y/K)_old]^(1/(a - 1)); y_new/y_old = (k_new/k_old)^a",
         curriculum=_L2G + " (Sec 5.1, Eq. 10, Example 11)")
def neoclassical_transition_growth(tfp_growth: float, capital_share: float, saving_rate: float,
                                   output_capital_ratio: float, steady_state_output_capital_ratio: float,
                                   reference_output_capital_ratio: float | None = None) -> dict:
    """Growth of per-capita income off the steady state, and level effects of a change such as a higher saving rate.

    Args:
        output_capital_ratio: actual Y/K at the date of interest.
        steady_state_output_capital_ratio: Psi for the (new) parameters (see neoclassical_steady_state).
        reference_output_capital_ratio: Y/K on the comparison (old) path at the same date, to get the
            proportional effect on capital per worker and income per worker.

    Returns:
        per_capita_income_growth, capital_per_worker_growth, and with the reference ratio:
        capital_per_worker_ratio (new/old), income_per_worker_ratio (new/old).

    Example:
        neoclassical_transition_growth(0.02, 0.35, 0.30, output_capital_ratio=0.7040,
                                       steady_state_output_capital_ratio=0.4693) -> 5.54%
    """
    th, a, s = _num("tfp_growth", tfp_growth), _num("capital_share", capital_share), _pos("saving_rate", saving_rate)
    yk, psi = _pos("output_capital_ratio", output_capital_ratio), _pos("steady_state_output_capital_ratio",
                                                                       steady_state_output_capital_ratio)
    if not 0 < a < 1:
        raise ValueError("capital_share must be between 0 and 1")
    g = th / (1 - a)
    out = {"per_capita_income_growth": g + a * s * (yk - psi), "capital_per_worker_growth": g + s * (yk - psi)}
    if reference_output_capital_ratio is not None:
        kr = (yk / _pos("reference_output_capital_ratio", reference_output_capital_ratio)) ** (1 / (a - 1))
        out.update(capital_per_worker_ratio=kr, income_per_worker_ratio=kr ** a)
    return out


@formula("economics", formula="K_t = K_t-1 + GDP_t (I/GDP - D/GDP)", curriculum=_L2G + " (Example 15)")
def capital_stock_path(initial_capital: float, gdp: list, investment_to_gdp: list, depreciation_to_gdp: list) -> dict:
    """Perpetual-inventory capital stock from GDP, gross investment share and depreciation share.

    Returns:
        capital_stock (one value per GDP year), capital_growth (compound annual growth from the first to the last
        computed stock, None for a single year).

    Example:
        capital_stock_path(2177.2, [1156.4, 1198.6], [0.263, 0.264], [0.136, 0.137]) -> [2324.1, 2476.3]
    """
    k = _pos("initial_capital", initial_capital)
    y = _vec("gdp", gdp)
    inv = _vec("investment_to_gdp", investment_to_gdp, len(y))
    dep = _vec("depreciation_to_gdp", depreciation_to_gdp, len(y))
    path = []
    for g, i, d in zip(y, inv, dep):
        k = k + g * (i - d)
        path.append(k)
    growth = (path[-1] / path[0]) ** (1 / (len(path) - 1)) - 1 if len(path) > 1 else None
    return {"capital_stock": path, "capital_growth": growth}


@formula("economics", formula="g = (end/start)^(1/years) - 1", curriculum=_L2G + " (Sec 2, Example 1)")
def gdp_compound_growth(start_value: float, end_value: float, years: float,
                        alternative_growth: float | None = None) -> dict:
    """Compound annual growth of GDP (or GDP per capita) and the end level under an alternative growth rate.

    Returns:
        growth_rate, and with alternative_growth: end_value_at_alternative, ratio_to_actual.

    Example:
        gdp_compound_growth(3048, 34828, 60) -> 4.14% a year
    """
    s, e, t = _pos("start_value", start_value), _pos("end_value", end_value), _pos("years", years)
    out = {"growth_rate": (e / s) ** (1 / t) - 1}
    if alternative_growth is not None:
        alt = s * (1 + _num("alternative_growth", alternative_growth)) ** t
        out.update(end_value_at_alternative=alt, ratio_to_actual=alt / e)
    return out


@formula("economics", formula="%dP = %dGDP + %d(E/GDP) + %d(P/E); total return = price return + dividend yield",
         curriculum=_L2G + " (Sec 2.2)")
def equity_growth_decomposition(real_gdp_growth: float, inflation: float, earnings_to_gdp_growth: float = 0.0,
                                pe_growth: float = 0.0, dividend_yield: float = 0.0) -> dict:
    """Decompose equity-market appreciation into nominal GDP growth, profit-share change and P/E change.

    Uses (log) growth rates, which add. Over long horizons the profit-share and P/E terms should be near zero,
    so long-run equity price growth converges on nominal GDP growth.

    Returns:
        total_return, price_return, nominal_gdp_growth.

    Example:
        equity_growth_decomposition(0.0301, 0.0394, -0.0012, 0.0032, 0.0367) -> price 7.15%, total 10.82%
    """
    ng = _num("real_gdp_growth", real_gdp_growth) + _num("inflation", inflation)
    pr = ng + _num("earnings_to_gdp_growth", earnings_to_gdp_growth) + _num("pe_growth", pe_growth)
    return {"total_return": pr + _num("dividend_yield", dividend_yield), "price_return": pr, "nominal_gdp_growth": ng}


# ---------------------------------------------------------------------------------- trade (L1 R20-R21)
@formula("economics", formula="opportunity cost of good 1 = output of good 2 / output of good 1 (per worker)",
         curriculum=_R20 + " (Sec 2.4.1)")
def comparative_advantage(output_a_good1: float, output_a_good2: float, output_b_good1: float,
                          output_b_good2: float) -> dict:
    """Absolute and comparative advantage for two countries (A, B) and two goods from output per worker.

    Returns:
        opportunity_cost_good1_a, opportunity_cost_good1_b (units of good 2 given up per unit of good 1),
        opportunity_cost_good2_a, opportunity_cost_good2_b, comparative_advantage_good1 ("A"/"B"/"none"),
        comparative_advantage_good2, absolute_advantage_good1, absolute_advantage_good2,
        terms_of_trade_range_good1 ([low, high] price of good 1 in units of good 2 at which both gain).

    Example:
        comparative_advantage(4, 8, 2, 16) -> good 1 (machinery) costs 2 cloth in A, 8 cloth in B; A exports good 1
    """
    a1, a2, b1, b2 = (_pos(n, v) for n, v in (("output_a_good1", output_a_good1), ("output_a_good2", output_a_good2),
                                             ("output_b_good1", output_b_good1), ("output_b_good2", output_b_good2)))
    oa1, ob1 = a2 / a1, b2 / b1

    def _who(x, y, lower_wins=True):
        if math.isclose(x, y):
            return "none"
        return ("A" if x < y else "B") if lower_wins else ("A" if x > y else "B")

    return {"opportunity_cost_good1_a": oa1, "opportunity_cost_good1_b": ob1,
            "opportunity_cost_good2_a": 1 / oa1, "opportunity_cost_good2_b": 1 / ob1,
            "comparative_advantage_good1": _who(oa1, ob1), "comparative_advantage_good2": _who(1 / oa1, 1 / ob1),
            "absolute_advantage_good1": _who(a1, b1, lower_wins=False),
            "absolute_advantage_good2": _who(a2, b2, lower_wins=False),
            "terms_of_trade_range_good1": sorted([oa1, ob1])}


@formula("economics", formula="dCS = -[dP Qd1 + 1/2 dP (Qd0 - Qd1)]; dPS = dP Qs0 + 1/2 dP (Qs1 - Qs0); "
                              "revenue = dP (Qd1 - Qs1); DWL = -dCS - dPS - revenue", curriculum=_R20 + " (Sec 3.1)")
def tariff_welfare_effects(world_price: float, domestic_price_with_tariff: float, supply_free_trade: float,
                           supply_with_tariff: float, demand_free_trade: float, demand_with_tariff: float) -> dict:
    """Welfare effects of an import tariff for a small country (linear curves between the two points).

    For an import quota with the same price effect, the tariff_revenue area is the quota rent instead.

    Returns:
        deadweight_loss, consumer_surplus_loss, producer_surplus_gain, tariff_revenue, imports_free_trade,
        imports_with_tariff.

    Example:
        tariff_welfare_effects(5, 6, 110000, 130000, 200000, 170000) -> CS -185,000, PS +120,000, revenue 40,000,
        DWL 25,000
    """
    pw, pt = _pos("world_price", world_price), _pos("domestic_price_with_tariff", domestic_price_with_tariff)
    s0, s1 = _num("supply_free_trade", supply_free_trade), _num("supply_with_tariff", supply_with_tariff)
    d0, d1 = _num("demand_free_trade", demand_free_trade), _num("demand_with_tariff", demand_with_tariff)
    if pt <= pw:
        raise ValueError("domestic_price_with_tariff must exceed world_price")
    if d1 < s1:
        raise ValueError("with the tariff, demand must still exceed domestic supply (imports >= 0)")
    dp = pt - pw
    cs = dp * d1 + 0.5 * dp * (d0 - d1)
    ps = dp * s0 + 0.5 * dp * (s1 - s0)
    rev = dp * (d1 - s1)
    return {"deadweight_loss": cs - ps - rev, "consumer_surplus_loss": cs, "producer_surplus_gain": ps,
            "tariff_revenue": rev, "imports_free_trade": d0 - s0, "imports_with_tariff": d1 - s1}


@formula("economics", formula="w_X e_X + w_M (e_M - 1) > 0", curriculum=_R21 + " (Sec 5.1)")
def marshall_lerner_condition(exports: float, imports: float, export_elasticity: float,
                              import_elasticity: float) -> dict:
    """Generalized Marshall-Lerner condition: does a depreciation improve the trade balance?

    Args:
        exports, imports: trade values in the same currency (weights are shares of total trade).
        export_elasticity, import_elasticity: price elasticities of demand, entered as positive magnitudes.

    Returns:
        ml_value, satisfied (True if > 0: depreciation improves, appreciation worsens the balance),
        export_weight, import_weight, min_export_weight (export share above which the condition holds, or None).

    Example:
        marshall_lerner_condition(400, 600, 0.75, 0.65) -> 0.09 (satisfied)
    """
    x, m = _pos("exports", exports), _pos("imports", imports)
    ex, em = abs(_num("export_elasticity", export_elasticity)), abs(_num("import_elasticity", import_elasticity))
    wx = x / (x + m)
    val = wx * ex + (1 - wx) * (em - 1)
    den = ex - em + 1
    return {"ml_value": val, "satisfied": val > 0, "export_weight": wx, "import_weight": 1 - wx,
            "min_export_weight": (1 - em) / den if den > 0 and em < 1 else None}


# ------------------------------------------------------------------------- spot FX (L1 R21, L2 R14)
def _leg(pair: tuple[str, str], bid: float, ask: float, want: tuple[str, str]) -> tuple[float, float]:
    """Return (bid, ask) for the `want` pair from a quote on `pair` (same pair or its inverse)."""
    if pair == want:
        return bid, ask
    if pair == (want[1], want[0]):
        return 1 / ask, 1 / bid
    raise ValueError(f"quote {pair[0]}/{pair[1]} cannot give {want[0]}/{want[1]}")


@formula("economics", formula="(P/B) = (P/C) x (C/B); bid = bid x bid, offer = offer x offer after inverting "
                              "(1/offer, 1/bid) any leg quoted the other way round", curriculum=_R21 + " (Sec 2.3); "
                                                                                                  + _L2FX + " (Sec 2.1)")
def fx_cross_rate(quote_1: str, bid_1: float, quote_2: str, bid_2: float, target: str,
                  ask_1: float | None = None, ask_2: float | None = None) -> dict:
    """Cross exchange rate (bid/offer or mid) for `target` from two quotes that share one currency.

    All pairs are PRICE/BASE strings: "USD/EUR" 1.3649 means 1 EUR costs 1.3649 USD. Inversion is handled:
    a leg quoted the other way is inverted with bid and offer swapped (bid' = 1/offer, offer' = 1/bid).
    Omit ask_1/ask_2 to work with mid rates.

    Returns:
        bid, ask, mid, pair, price_leg ([bid, ask] of PRICE/COMMON), base_leg ([bid, ask] of COMMON/BASE),
        common_currency.

    Example:
        fx_cross_rate("JPY/USD", 76.64, "USD/EUR", 1.3649, "JPY/EUR", ask_1=76.66, ask_2=1.3651) -> 104.61/104.65
    """
    p1, p2, (P, B) = _pair(quote_1), _pair(quote_2), _pair(target)
    b1, b2 = _pos("bid_1", bid_1), _pos("bid_2", bid_2)
    a1 = b1 if ask_1 is None else _pos("ask_1", ask_1)
    a2 = b2 if ask_2 is None else _pos("ask_2", ask_2)
    if a1 < b1 or a2 < b2:
        raise ValueError("ask must be >= bid")
    common = (set(p1) & set(p2)) - {P, B}
    if len(common) != 1 or set(p1) | set(p2) != {P, B} | common:
        raise ValueError(f"{quote_1} and {quote_2} must share exactly one currency outside {target} "
                         f"and together contain both {P} and {B}")
    C = common.pop()
    legs = {p1: (b1, a1), p2: (b2, a2)}
    price_quote = p1 if P in p1 else p2
    base_quote = p2 if price_quote == p1 else p1
    if B not in base_quote:
        raise ValueError(f"{target}: price and base currencies must come from different quotes")
    pb, pa = _leg(price_quote, *legs[price_quote], (P, C))
    bb, ba = _leg(base_quote, *legs[base_quote], (C, B))
    bid, ask = pb * bb, pa * ba
    return {"bid": bid, "ask": ask, "mid": (bid + ask) / 2, "pair": f"{P}/{B}", "price_leg": [pb, pa],
            "base_leg": [bb, ba], "common_currency": C}


@formula("economics", formula="arbitrage if dealer offer < implied bid (buy base from dealer) or dealer bid > "
                              "implied offer (sell base to dealer)", curriculum=_L2FX + " (Sec 2.1)")
def fx_triangular_arbitrage(dealer_quote: str, dealer_bid: float, dealer_ask: float, quote_1: str, bid_1: float,
                            ask_1: float, quote_2: str, bid_2: float, ask_2: float) -> dict:
    """Check a dealer's cross-rate quote against the interbank-implied cross and size the arbitrage profit.

    All pairs are PRICE/BASE strings. Profit is per unit of the dealer quote's BASE currency, in its PRICE currency.

    Returns:
        arbitrage (bool), profit_per_base_unit, action, implied_bid, implied_ask.

    Example:
        fx_triangular_arbitrage("JPY/CAD", 85.73, 85.75, "JPY/USD", 81.87, 81.89, "CAD/USD", 0.9544, 0.9546)
        -> buy CAD from the dealer at 85.75, sell in the interbank market at 85.76: ~JPY 0.01 per CAD
    """
    imp = fx_cross_rate(quote_1, bid_1, quote_2, bid_2, dealer_quote, ask_1=ask_1, ask_2=ask_2)
    db, da = _pos("dealer_bid", dealer_bid), _pos("dealer_ask", dealer_ask)
    if da < db:
        raise ValueError("dealer_ask must be >= dealer_bid")
    P, B = _pair(dealer_quote)
    if da < imp["bid"]:
        profit, act = imp["bid"] - da, f"buy {B} from the dealer at {da}, sell {B} in the interbank market"
    elif db > imp["ask"]:
        profit, act = db - imp["ask"], f"buy {B} in the interbank market, sell {B} to the dealer at {db}"
    else:
        profit, act = 0.0, "no arbitrage: dealer quote lies within the implied interbank bid-offer"
    return {"arbitrage": profit > 0, "profit_per_base_unit": profit, "action": act, "implied_bid": imp["bid"],
            "implied_ask": imp["ask"]}


@formula("economics", formula="base currency change = S1/S0 - 1; price currency change = S0/S1 - 1",
         curriculum=_R21 + " (Sec 2.3)")
def fx_percent_change(old_rate: float, new_rate: float) -> dict:
    """Percentage appreciation/depreciation of each currency from a move in a PRICE/BASE quote.

    A rise in USD/EUR (price USD, base EUR) is a EUR appreciation; the USD change is not the negative of it.

    Returns:
        base_currency_change, price_currency_change.

    Example:
        fx_percent_change(old_rate=1.3960, new_rate=1.3863) -> base (EUR) -0.69%, price (USD) +0.70%
    """
    s0, s1 = _pos("old_rate", old_rate), _pos("new_rate", new_rate)
    return {"base_currency_change": s1 / s0 - 1, "price_currency_change": s0 / s1 - 1}


@formula("economics", formula="spread = offer - bid; pips = spread / pip size; % spread = spread / mid",
         curriculum=_L2FX + " (Sec 2.1)")
def fx_bid_offer_spread(bid: float, ask: float, pip_size: float = 0.0001) -> dict:
    """Bid-offer spread of a PRICE/BASE quote in price-currency units, pips and percent of mid.

    Args:
        pip_size: 0.0001 for most pairs, 0.01 for JPY pairs.

    Returns:
        spread, pips, pct_of_mid, mid.

    Example:
        fx_bid_offer_spread(1.3649, 1.3651) -> 2 pips
    """
    b, a, pip = _pos("bid", bid), _pos("ask", ask), _pos("pip_size", pip_size)
    if a < b:
        raise ValueError("ask must be >= bid")
    mid = (a + b) / 2
    return {"spread": a - b, "pips": (a - b) / pip, "pct_of_mid": (a - b) / mid, "mid": mid}


# ------------------------------------------------------------------------------- forwards (L1 R21, L2 R14)
@formula("economics", formula="F_P/B = S_P/B x (1 + i_P x days/360) / (1 + i_B x days/360)",
         curriculum=_R21 + " (Sec 3.3); " + _L2FX + " (Sec 2.2, Eq. 1)")
def fx_forward_cip(spot: float, rate_price_ccy: float, rate_base_ccy: float, days: float,
                   days_in_year_price: float = 360, days_in_year_base: float = 360, points_scale: float = 10000,
                   market_forward: float | None = None) -> dict:
    """Forward rate from covered interest parity for a PRICE/BASE quote, forward points, and an arbitrage check.

    Uses simple (money-market, act/360 by default) interest: F = S (1 + i_P d/360) / (1 + i_B d/360). Use it for
    LIBOR-style deposit rates and forward points (economics/FX readings). For annually compounded rates quoted with
    a horizon in years (derivatives readings: F = S (1 + r_dom)^T / (1 + r_for)^T, or continuous), use
    derivatives' `currency_forward_price`, whose default is annual compounding.

    Args:
        spot: PRICE/BASE spot rate (e.g. CAD/AUD 1.0145 = CAD per 1 AUD).
        rate_price_ccy, rate_base_ccy: annualized money-market (LIBOR) rates of the price and base currencies.
        days: actual days to settlement. days_in_year_*: 360 for most currencies, 365 for GBP.
        points_scale: 10,000 for 4-decimal pairs, 100 for JPY pairs.
        market_forward: optional quoted forward; returns the covered interest arbitrage if it differs.

    Returns:
        forward, forward_premium (F - S; > 0 = base currency at a premium), forward_points, forward_premium_pct
        (F/S - 1, not annualized), and with market_forward: mispricing (market - CIP), arbitrage_action.

    Example:
        fx_forward_cip(spot=1.0145, rate_price_ccy=0.0141, rate_base_ccy=0.0487, days=270) -> premium -0.0254
    """
    s = _pos("spot", spot)
    ip, ib, d = _num("rate_price_ccy", rate_price_ccy), _num("rate_base_ccy", rate_base_ccy), _num("days", days)
    if d < 0:
        raise ValueError("days must be >= 0")
    tp, tb = d / _pos("days_in_year_price", days_in_year_price), d / _pos("days_in_year_base", days_in_year_base)
    if 1 + ib * tb <= 0 or 1 + ip * tp <= 0:
        raise ValueError("rates imply a non-positive growth factor")
    f = s * (1 + ip * tp) / (1 + ib * tb)
    out = {"forward": f, "forward_premium": f - s, "forward_points": (f - s) * _pos("points_scale", points_scale),
           "forward_premium_pct": f / s - 1}
    if market_forward is not None:
        mf = _pos("market_forward", market_forward)
        mis = mf - f
        if mis > 0:
            act = ("market forward too high: borrow price currency, buy base spot, invest in base, "
                   "sell base forward; profit = market - CIP per base unit at settlement")
        elif mis < 0:
            act = ("market forward too low: borrow base currency, sell it spot, invest in price currency, "
                   "buy base forward; profit = CIP - market per base unit at settlement")
        else:
            act = "none: market forward equals the CIP forward"
        out.update(mispricing=mis, arbitrage_action=act)
    return out


@formula("economics", formula="all-in forward = spot + points / scale; sell base -> bid side, buy base -> offer side",
         curriculum=_R21 + " (Sec 3.3); " + _L2FX + " (Sec 2.2)")
def fx_forward_outright(spot_bid: float, points_bid: float = 0.0, spot_ask: float | None = None,
                        points_ask: float | None = None, points_scale: float = 10000, use_mid_spot: bool = False,
                        premium_pct: float | None = None) -> dict:
    """All-in (outright) forward bid and offer from a PRICE/BASE spot quote and forward points.

    A client SELLING the base currency forward deals on the bid (spot bid + bid points); BUYING the base deals on
    the offer. For an FX swap the spot leg is at mid (use_mid_spot=True) while points stay on the relevant side.
    premium_pct (e.g. -0.003 for a 0.30% discount) instead converts a percentage quote: forward = spot x (1 + pct).

    Returns:
        forward_bid, forward_offer, forward_mid, spot_used_bid, spot_used_offer.

    Example:
        fx_forward_outright(spot_bid=1.3549, points_bid=-15.9) -> 1.35331
    """
    sb = _pos("spot_bid", spot_bid)
    sa = sb if spot_ask is None else _pos("spot_ask", spot_ask)
    if sa < sb:
        raise ValueError("spot_ask must be >= spot_bid")
    if premium_pct is not None:
        k = 1 + _num("premium_pct", premium_pct)
        fb, fa, ub, ua = sb * k, sa * k, sb, sa
    else:
        pb = _num("points_bid", points_bid)
        pa = pb if points_ask is None else _num("points_ask", points_ask)
        sc = _pos("points_scale", points_scale)
        ub, ua = ((sb + sa) / 2,) * 2 if use_mid_spot else (sb, sa)
        fb, fa = ub + pb / sc, ua + pa / sc
    return {"forward_bid": fb, "forward_offer": fa, "forward_mid": (fb + fa) / 2, "spot_used_bid": ub,
            "spot_used_offer": ua}


@formula("economics", formula="MTM = notional x (closing all-in forward - contract forward) / (1 + r x days/360)",
         curriculum=_L2FX + " (Sec 2.2, Example 3)")
def fx_forward_mark_to_market(contract_rate: float, notional: float, notional_currency: str, position: str,
                              spot_bid: float, spot_ask: float, points_bid: float = 0.0, points_ask: float = 0.0,
                              days_to_settlement: float = 0.0, discount_rate: float = 0.0,
                              days_in_year: float = 360, points_scale: float = 10000,
                              use_mid_spot: bool = False) -> dict:
    """Mark-to-market of an existing FX forward on a PRICE/BASE pair by pricing the offsetting forward.

    Args:
        contract_rate: original all-in forward rate (PRICE/BASE).
        notional: contract size in `notional_currency`.
        notional_currency: "base" or "price" (which side of the PRICE/BASE quote the notional is in).
        position: "long" or "short" the notional currency.
        spot_bid/spot_ask, points_bid/points_ask: current market for the remaining term.
        days_to_settlement, discount_rate: discount at the rate of the CASH-FLOW currency (price-currency rate for a
            base-currency notional, base-currency rate for a price-currency notional), Actual/360 by default.
        use_mid_spot: True for an FX swap roll (mid spot, sided points).

    Returns:
        mark_to_market (present value, + = gain), settlement_cash_flow, cash_flow_currency ("price"/"base"),
        closing_rate, closing_side ("bid" if the offset sells the base currency, else "offer").

    Example:
        fx_forward_mark_to_market(1.6100, 10e6, "base", "long", 1.6210, 1.6215, 130, 140, 90, 0.048)
        -> settlement +AUD 240,000, MTM +AUD 237,154
    """
    k, n = _pos("contract_rate", contract_rate), _pos("notional", notional)
    ccy, pos = str(notional_currency).lower(), str(position).lower()
    if ccy not in ("base", "price") or pos not in ("long", "short"):
        raise ValueError("notional_currency must be 'base' or 'price'; position must be 'long' or 'short'")
    sells_base = (ccy, pos) in (("base", "long"), ("price", "short"))
    fwd = fx_forward_outright(spot_bid, points_bid, spot_ask, points_ask, points_scale, use_mid_spot)
    close = fwd["forward_bid"] if sells_base else fwd["forward_offer"]
    if ccy == "base":
        cf = n * (close - k) if pos == "long" else n * (k - close)
    else:
        cf = n / close - n / k if pos == "long" else n / k - n / close
    d, r = _num("days_to_settlement", days_to_settlement), _num("discount_rate", discount_rate)
    if d < 0:
        raise ValueError("days_to_settlement must be >= 0")
    pv = cf / (1 + r * d / _pos("days_in_year", days_in_year))
    return {"mark_to_market": pv, "settlement_cash_flow": cf, "cash_flow_currency": "price" if ccy == "base" else "base",
            "closing_rate": close, "closing_side": "bid" if sells_base else "offer"}


# ------------------------------------------------------------------------- parity conditions (L2 R14)
@formula("economics", formula="E[S_T] = S x [(1 + i_P)/(1 + i_B)]^T;  %dS ~ i_P - i_B",
         curriculum=_L2FX + " (Sec 3.1.2, Eq. 2)")
def uip_expected_spot(spot: float, rate_price_ccy: float, rate_base_ccy: float, years: float = 1.0) -> dict:
    """Expected future spot rate under uncovered interest parity for a PRICE/BASE quote.

    The higher-yielding currency is expected to depreciate by the rate differential. UIP is not enforced by
    arbitrage (it assumes risk neutrality); when it fails, the carry trade earns the differential.

    Returns:
        expected_spot, expected_change_pct (base currency, exact), expected_change_approx (i_P - i_B) x T.

    Example:
        uip_expected_spot(spot=129.67, rate_price_ccy=0.001, rate_base_ccy=0.03) -> 126.02 JPY/GBP
    """
    s, ip, ib, t = _pos("spot", spot), _num("rate_price_ccy", rate_price_ccy), _num("rate_base_ccy", rate_base_ccy), \
        _num("years", years)
    if ip <= -1 or ib <= -1 or t < 0:
        raise ValueError("rates must exceed -100% and years must be >= 0")
    e = s * ((1 + ip) / (1 + ib)) ** t
    return {"expected_spot": e, "expected_change_pct": e / s - 1, "expected_change_approx": (ip - ib) * t}


@formula("economics", formula="S_P/B = P_price / P_base (absolute PPP)", curriculum=_L2FX + " (Sec 3.1.4)")
def ppp_absolute(price_level_price_ccy: float, price_level_base_ccy: float, actual_spot: float | None = None) -> dict:
    """Absolute-PPP exchange rate (PRICE/BASE) from the two price levels, and base-currency misvaluation.

    Returns:
        ppp_spot, and with actual_spot: base_currency_overvaluation (actual / PPP - 1; > 0 = overvalued).

    Example:
        ppp_absolute(price_level_price_ccy=140, price_level_base_ccy=100) -> 1.40 USD/EUR
    """
    ppp = _pos("price_level_price_ccy", price_level_price_ccy) / _pos("price_level_base_ccy", price_level_base_ccy)
    out = {"ppp_spot": ppp}
    if actual_spot is not None:
        out["base_currency_overvaluation"] = _pos("actual_spot", actual_spot) / ppp - 1
    return out


@formula("economics", formula="E[S_T] = S x [(1 + pi_P)/(1 + pi_B)]^T;  %dS ~ pi_P - pi_B",
         curriculum=_L2FX + " (Sec 3.1.4, Eq. 3-4)")
def ppp_relative_expected_spot(spot: float, inflation_price_ccy: float, inflation_base_ccy: float,
                               years: float = 1.0) -> dict:
    """Relative / ex ante PPP: the high-inflation currency depreciates by the inflation differential.

    Returns:
        expected_spot (PRICE/BASE), expected_change_pct (exact), expected_change_approx (pi_P - pi_B) x T.

    Example:
        ppp_relative_expected_spot(spot=1.0, inflation_price_ccy=0.10, inflation_base_ccy=0.05) -> about +5%
    """
    s, pp, pb, t = _pos("spot", spot), _num("inflation_price_ccy", inflation_price_ccy), \
        _num("inflation_base_ccy", inflation_base_ccy), _num("years", years)
    if pp <= -1 or pb <= -1 or t < 0:
        raise ValueError("inflation must exceed -100% and years must be >= 0")
    e = s * ((1 + pp) / (1 + pb)) ** t
    return {"expected_spot": e, "expected_change_pct": e / s - 1, "expected_change_approx": (pp - pb) * t}


@formula("economics", formula="i_P - i_B = (r_P - r_B) + (pi_P - pi_B); with real rate parity i_P - i_B = pi_P - pi_B",
         curriculum=_L2FX + " (Sec 3.1.5)")
def international_fisher_effect(nominal_rate_price_ccy: float, nominal_rate_base_ccy: float,
                                expected_inflation_price_ccy: float | None = None,
                                expected_inflation_base_ccy: float | None = None) -> dict:
    """International Fisher effect: nominal rate gap = expected inflation gap when real rates are equal.

    Returns:
        implied_inflation_differential (i_P - i_B), and with both expected inflation rates:
        real_rate_price_ccy, real_rate_base_ccy, real_rate_differential (r_P - r_B; 0 under real rate parity).

    Example:
        international_fisher_effect(nominal_rate_price_ccy=0.01, nominal_rate_base_ccy=0.05) -> -4%
    """
    ip, ib = _num("nominal_rate_price_ccy", nominal_rate_price_ccy), _num("nominal_rate_base_ccy",
                                                                          nominal_rate_base_ccy)
    out = {"implied_inflation_differential": ip - ib}
    if (expected_inflation_price_ccy is None) != (expected_inflation_base_ccy is None):
        raise ValueError("give both expected inflation rates or neither")
    if expected_inflation_price_ccy is not None:
        rp = ip - _num("expected_inflation_price_ccy", expected_inflation_price_ccy)
        rb = ib - _num("expected_inflation_base_ccy", expected_inflation_base_ccy)
        out.update(real_rate_price_ccy=rp, real_rate_base_ccy=rb, real_rate_differential=rp - rb)
    return out


@formula("economics", formula="q_P/B = S_P/B x CPI_base / CPI_price", curriculum=_R21 + " (Sec 2.1); "
                                                                                 + _L2FX + " (Sec 3.1.4, Eq. 5)")
def real_exchange_rate(spot: float, cpi_price_ccy: float, cpi_base_ccy: float) -> dict:
    """Real exchange rate for a PRICE/BASE quote: the price of base-country goods in units of price-country goods.

    A higher q is a real appreciation of the base currency: for a resident of the price-currency country, the
    base country's goods have become dearer (lower relative purchasing power abroad).

    Returns:
        real_exchange_rate.

    Example:
        real_exchange_rate(spot=1.05, cpi_price_ccy=102, cpi_base_ccy=105) -> 1.0809
    """
    return {"real_exchange_rate": _pos("spot", spot) * _pos("cpi_base_ccy", cpi_base_ccy)
            / _pos("cpi_price_ccy", cpi_price_ccy)}


@formula("economics", formula="%dq = (1 + %dS)(1 + pi_base)/(1 + pi_price) - 1 ~ %dS + pi_base - pi_price",
         curriculum=_R21 + " (Sec 2.1)")
def real_exchange_rate_change(spot_change: float, inflation_price_ccy: float, inflation_base_ccy: float) -> dict:
    """Percentage change in the real PRICE/BASE exchange rate from the nominal change and the two inflation rates.

    Returns:
        real_change (exact), real_change_approx.

    Example:
        real_exchange_rate_change(spot_change=0.10, inflation_price_ccy=0.02, inflation_base_ccy=0.05) -> ~+13%
    """
    ds, pp, pb = _num("spot_change", spot_change), _num("inflation_price_ccy", inflation_price_ccy), \
        _num("inflation_base_ccy", inflation_base_ccy)
    if pp <= -1:
        raise ValueError("inflation_price_ccy must exceed -100%")
    return {"real_change": (1 + ds) * (1 + pb) / (1 + pp) - 1, "real_change_approx": ds + pb - pp}


@formula("economics", formula="q_P/B = q_bar + (r_B - r_P) - (phi_B - phi_P)", curriculum=_L2FX + " (Sec 3.3, Eq. 12)")
def real_fx_rate_model(long_run_real_rate: float, real_rate_base_ccy: float, real_rate_price_ccy: float,
                       risk_premium_base_ccy: float = 0.0, risk_premium_price_ccy: float = 0.0) -> dict:
    """Stylized model of the real exchange rate (PRICE/BASE, in log/percentage terms) around its long-run value.

    The base currency is stronger in real terms when its real rate is higher and its risk premium lower.
    Combine with taylor_rule_rate to map policy gaps into real-rate differentials (L2 Eq. 17).

    Returns:
        real_exchange_rate, deviation_from_long_run.

    Example:
        real_fx_rate_model(long_run_real_rate=0, real_rate_base_ccy=0.03, real_rate_price_ccy=0.01) -> +2%
    """
    dev = (_num("real_rate_base_ccy", real_rate_base_ccy) - _num("real_rate_price_ccy", real_rate_price_ccy)) - (
        _num("risk_premium_base_ccy", risk_premium_base_ccy) - _num("risk_premium_price_ccy", risk_premium_price_ccy))
    return {"real_exchange_rate": _num("long_run_real_rate", long_run_real_rate) + dev,
            "deviation_from_long_run": dev}


@formula("economics", formula="carry return = (S_1/S_0)(1 + i_invest) - 1 - i_funding, S quoted FUNDING/INVEST",
         curriculum=_L2FX + " (Sec 4, Example 8)")
def carry_trade_return(spot_start: float, spot_end: float, investment_rate: float, funding_rate: float,
                       years: float = 1.0) -> dict:
    """Return on a carry trade: borrow the low-yield funding currency, invest in the high-yield currency.

    Args:
        spot_start, spot_end: rate quoted FUNDING/INVESTMENT (price = funding currency, base = investment
            currency), e.g. JPY/AUD for borrowing JPY to invest in AUD.
        investment_rate, funding_rate: annual money-market rates (simple interest over `years`).

    Returns:
        net_return (on the funded amount, in funding currency), gross_return, currency_return
        (investment currency change), interest_differential.

    Example:
        carry_trade_return(87.40, 86.42, investment_rate=0.045, funding_rate=0.001) -> net ~3.23%
    """
    s0, s1 = _pos("spot_start", spot_start), _pos("spot_end", spot_end)
    ii, if_, t = _num("investment_rate", investment_rate), _num("funding_rate", funding_rate), _num("years", years)
    if t < 0:
        raise ValueError("years must be >= 0")
    gross = (s1 / s0) * (1 + ii * t) - 1
    return {"net_return": gross - if_ * t, "gross_return": gross, "currency_return": s1 / s0 - 1,
            "interest_differential": (ii - if_) * t}
