"""Fixed income: binomial interest-rate trees — calibration, bonds with embedded options, OAS, floaters,
pathwise valuation, and convertible-bond metrics (L2).

Trees are annual and lognormal: at year i the rates are r_i,low x e^(2 sigma k), k = 0..i (lowest first),
each move has probability 1/2, as in the CFA curriculum.
"""
from __future__ import annotations

import itertools
import math

from ._registry import formula
from .fi_pricing import _solve

L2_EO = "L2 FI: Valuing Bonds with Embedded Options (2014 R47) / Arbitrage-Free Valuation Framework"


def _check_tree(tree: list) -> None:
    if not tree or any(len(level) != i + 1 for i, level in enumerate(tree)):
        raise ValueError("tree must be a list of levels with 1, 2, 3, ... rates (lowest rate first)")


def _schedule(sched, years: int) -> dict:
    out = {}
    for y, px in sched or []:
        if not (1 <= int(y) < years and int(y) == y):
            raise ValueError(f"exercise years must be whole numbers in [1, {years - 1}], got {y}")
        out[int(y)] = float(px)
    return out


def _years(years) -> int:
    """Tree maturity: a whole number of one-year periods (fractions raise instead of being truncated)."""
    if int(years) != years or years < 1:
        raise ValueError(f"years must be a positive whole number of annual tree periods, got {years!r}")
    return int(years)


def _coupons(coupon_rate, coupons, years, face) -> list:
    if coupons is not None:
        return [face * c for c in coupons]
    if coupon_rate is None or years is None:
        raise ValueError("give coupons (rates per year) or coupon_rate with years")
    return [face * coupon_rate] * _years(years)


def _value(tree: list, cpns: list, face: float, calls: dict, puts: dict, oas: float) -> tuple:
    n = len(cpns)
    if n > len(tree):
        raise ValueError(f"tree has {len(tree)} levels but the bond needs {n}")
    vals = [face] * (n + 1)  # ex-coupon values at year n
    levels = []
    for i in range(n - 1, -1, -1):
        c = cpns[i]  # coupon paid at year i+1
        new = []
        for j in range(i + 1):
            r = tree[i][j] + oas
            if r <= -1:
                raise ValueError("rate + OAS must exceed -100%")
            v = 0.5 * ((vals[j] + c) + (vals[j + 1] + c)) / (1 + r)
            if i in calls and v > calls[i]:
                v = calls[i]
            elif i in puts and v < puts[i]:
                v = puts[i]
            new.append(v)
        vals = new
        levels.append(new)
    return vals[0], levels[::-1]


@formula("fixed_income", formula="choose r_i,low so the i+1-year par bond prices at par; r_i,k = r_i,low e^(2 sigma k)",
         curriculum=L2_EO)
def binomial_tree_calibrate(par_rates: list, volatility: float) -> dict:
    """Calibrate an annual lognormal binomial interest-rate tree to a par curve (iterative, level by level).

    Args:
        par_rates: annual-pay par yields for 1, 2, ..., N years (the issuer's or benchmark on-the-run curve).
        volatility: annual volatility of the one-year rate, decimal (adjacent nodes differ by e^(2 sigma)).

    Returns:
        tree (list of levels, each lowest rate first), spot_rates, discount_factors.

    Example:
        par 3.5%, 4.2%, 4.7%, 5.2%, sigma 10% -> year-1 rates 4.4448%/5.4289%, year-2 top 7.0053%
    """
    if volatility < 0:
        raise ValueError("volatility must be >= 0")
    tree = [[par_rates[0]]]
    m = math.exp(2 * volatility)
    for i in range(1, len(par_rates)):
        c = par_rates[i]

        def err(r_low, i=i, c=c):
            trial = tree + [[r_low * m ** k for k in range(i + 1)]]
            return _value(trial, [100 * c] * (i + 1), 100.0, {}, {}, 0.0)[0] - 100.0

        r_low = _solve(err, lo=1e-9, hi=2.0, what=f"lowest rate at year {i}")
        tree.append([r_low * m ** k for k in range(i + 1)])
    dfs = [_value(tree, [0.0] * t, 1.0, {}, {}, 0.0)[0] for t in range(1, len(par_rates) + 1)]
    spots = [df ** (-1 / t) - 1 for t, df in enumerate(dfs, 1)]
    return {"tree": tree, "spot_rates": spots, "discount_factors": dfs}


@formula("fixed_income", formula="V = 1/2[(V_H + C) + (V_L + C)]/(1 + r + OAS); callable: min(V, call); "
         "putable: max(V, put)", curriculum=L2_EO)
def binomial_tree_bond_value(tree: list, coupon_rate: float | None = None, years: int | None = None,
                             coupons: list | None = None, face: float = 100.0, call_schedule: list | None = None,
                             put_schedule: list | None = None, oas: float = 0.0) -> dict:
    """Backward-induction value of an option-free, callable, putable or step-up bond on a binomial tree.

    Args:
        tree: levels of one-year rates (from binomial_tree_calibrate or given).
        coupon_rate + years: level annual coupon and maturity; or coupons: coupon rate for each year (step-ups).
        face: par. call_schedule / put_schedule: list of [year, price] the option can be exercised at
            (years 1..N-1). The issuer calls if the node value exceeds the call price; the investor puts if
            it is below the put price.
        oas: constant spread added to every tree rate.

    Returns:
        value, value_option_free, call_option_value (straight - callable), put_option_value (putable - straight),
        node_values (ex-coupon values by year, lowest-rate node first).

    Example:
        6.5% 4y on the 10%-vol tree: option-free 104.643, callable at 100 from year 1 -> 102.899
    """
    _check_tree(tree)
    if years is None and coupons is None:
        years = len(tree)
    cpns = _coupons(coupon_rate, coupons, years, face)
    n = len(cpns)
    calls, puts = _schedule(call_schedule, n), _schedule(put_schedule, n)
    v, nodes = _value(tree, cpns, face, calls, puts, oas)
    straight = _value(tree, cpns, face, {}, {}, oas)[0] if (calls or puts) else v
    return {"value": v, "value_option_free": straight,
            "call_option_value": straight - v if calls and not puts else None,
            "put_option_value": v - straight if puts and not calls else None,
            "net_option_value_to_investor": v - straight, "node_values": nodes}


@formula("fixed_income", formula="solve tree value(rates + OAS) = market price", curriculum=L2_EO)
def binomial_tree_oas(price: float, tree: list, coupon_rate: float | None = None, years: int | None = None,
                      coupons: list | None = None, face: float = 100.0, call_schedule: list | None = None,
                      put_schedule: list | None = None) -> dict:
    """Option-adjusted spread: the constant spread added to every tree rate that matches the market price.

    Args:
        price: market price. tree, coupon_rate/years/coupons, face, call_schedule, put_schedule:
            as in binomial_tree_bond_value.

    Returns:
        oas, oas_bp, model_value_at_zero_oas.

    Example:
        6.5% 4y callable at 100, price 102.218, 10% vol tree -> OAS 35 bp
    """
    def f(s):
        return binomial_tree_bond_value(tree, coupon_rate, years, coupons, face, call_schedule, put_schedule,
                                        s)["value"] - price

    lo = -min(min(level) for level in tree) + 1e-6
    s = _solve(f, lo=max(lo, -0.5), hi=1.0, what="OAS")
    return {"oas": s, "oas_bp": s * 1e4, "model_value_at_zero_oas": f(0.0) + price}


@formula("fixed_income", formula="shift par curve +/- dy, recalibrate, add OAS, revalue; EffDur = (V- - V+)/(2 V0 dy) ; "
         "EffCon = (V- + V+ - 2V0)/(V0 dy^2)", curriculum=L2_EO)
def binomial_tree_effective_duration(par_rates: list, volatility: float, coupon_rate: float | None = None,
                                     years: int | None = None, coupons: list | None = None, oas: float = 0.0,
                                     shift: float = 0.0025, face: float = 100.0, call_schedule: list | None = None,
                                     put_schedule: list | None = None) -> dict:
    """Effective duration and convexity of a bond (with or without options) from shifted, recalibrated trees.

    OAS is held constant. Convexity uses the L1 scaling (no factor 2): pair it with 1/2 x C x dy^2.

    Args:
        par_rates: par curve the tree is calibrated to. volatility: rate volatility.
        coupon_rate/years/coupons, face, call_schedule, put_schedule: bond terms.
        oas: option-adjusted spread to keep. shift: parallel par-curve shift, decimal.

    Returns:
        effective_duration, effective_convexity, value, value_up (V+), value_down (V-).

    Example:
        6.5% 4y callable at 100, OAS 35 bp, 10% vol, +/-25 bp -> V+ 101.621, V- 102.765
    """
    vals = []
    for d in (0.0, shift, -shift):
        tree = binomial_tree_calibrate([p + d for p in par_rates], volatility)["tree"]
        vals.append(binomial_tree_bond_value(tree, coupon_rate, years, coupons, face, call_schedule,
                                             put_schedule, oas)["value"])
    v0, vp, vm = vals
    return {"effective_duration": (vm - vp) / (2 * v0 * shift),
            "effective_convexity": (vm + vp - 2 * v0) / (v0 * shift ** 2), "value": v0, "value_up": vp,
            "value_down": vm}


@formula("fixed_income", formula="coupon at node = min(max(r + margin, floor), cap) x F, set at the node, paid in arrears",
         curriculum=L2_EO)
def binomial_tree_capped_floater(tree: list, years: int | None = None, margin: float = 0.0,
                                 cap: float | None = None, floor: float | None = None, face: float = 100.0) -> dict:
    """Value of a floating-rate note (one-year reference rate + margin) with an optional cap and/or floor.

    Args:
        tree: one-year rate tree (the reference rate). years: maturity (default: tree length).
        margin: quoted margin over the reference rate. cap / floor: coupon-rate limits. face: par.

    Returns:
        value, value_uncapped, cap_value (uncapped - capped, cost to the investor), floor_value.

    Example:
        reference-rate floater on the 10%-vol tree with a 7.25% cap -> 99.724
    """
    _check_tree(tree)
    n = len(tree) if years is None else _years(years)
    if n > len(tree):
        raise ValueError("tree too short for the maturity")

    def val(cp, fl):
        vals = [face] * (n + 1)
        for i in range(n - 1, -1, -1):
            new = []
            for j in range(i + 1):
                r = tree[i][j]
                c = r + margin
                if cp is not None:
                    c = min(c, cp)
                if fl is not None:
                    c = max(c, fl)
                new.append((0.5 * (vals[j] + vals[j + 1]) + face * c) / (1 + r))
            vals = new
        return vals[0]

    v, plain = val(cap, floor), val(None, None)
    return {"value": v, "value_uncapped": plain,
            "cap_value": plain - val(cap, None) if cap is not None else None,
            "floor_value": val(None, floor) - plain if floor is not None else None}


@formula("fixed_income", formula="V = average over all 2^(N-1) rate paths of sum CF_t / prod_{s<t}(1 + r_s)",
         curriculum=L2_EO)
def binomial_tree_pathwise_value(tree: list, coupon_rate: float, years: int | None = None,
                                 face: float = 100.0) -> dict:
    """Pathwise valuation: discount the option-free bond's cash flows along every equally likely rate path.

    Args:
        tree: one-year rate tree. coupon_rate: annual coupon. years: maturity (<= 20; default tree length).
        face: par.

    Returns:
        value, paths (count), path_values (only when <= 64 paths).

    Example:
        equals backward induction for an option-free bond (6.5% 4y -> 104.643 on the 10%-vol tree)
    """
    _check_tree(tree)
    n = len(tree) if years is None else _years(years)
    if not 1 <= n <= min(len(tree), 20):
        raise ValueError("years must be 1..min(len(tree), 20)")
    cfs = [face * coupon_rate] * n
    cfs[-1] += face
    vals = []
    for moves in itertools.product((0, 1), repeat=n - 1):
        j, disc, pv = 0, 1.0, 0.0
        for t in range(n):
            if t:
                j += moves[t - 1]
            disc /= 1 + tree[t][j]
            pv += cfs[t] * disc
        vals.append(pv)
    return {"value": sum(vals) / len(vals), "paths": len(vals), "path_values": vals if len(vals) <= 64 else None}


@formula("fixed_income", formula="Conversion value = stock x ratio ; min value = max(conversion, straight) ; "
         "market conversion price = price/ratio ; premium = MCP - stock", curriculum=L2_EO)
def convertible_bond_metrics(convertible_price: float, conversion_ratio: float, stock_price: float,
                             straight_value: float | None = None, coupon_rate: float | None = None,
                             par: float = 1000.0, dividend_per_share: float = 0.0,
                             issue_price: float | None = None) -> dict:
    """Traditional convertible-bond analysis: conversion value, floors, conversion premium and payback.

    Args:
        convertible_price: market price per `par` of bonds. conversion_ratio: shares per `par`.
        stock_price: current share price. straight_value: value without conversion, per `par`.
        coupon_rate: annual coupon (for income differential). par: par amount the ratio refers to.
        dividend_per_share: annual dividend. issue_price: price at issuance per `par` (conversion price).

    Returns:
        conversion_value, minimum_value, market_conversion_price, conversion_premium_per_share,
        conversion_premium_ratio, favorable_income_differential_per_share, premium_payback_years,
        premium_over_straight_value, conversion_price_at_issue.

    Example:
        ADC: price 1,065, ratio 25.32, stock $33, straight 981.9 -> MCP 42.06, premium 27.5%, payback 6.6y
    """
    if conversion_ratio <= 0 or stock_price <= 0 or convertible_price <= 0:
        raise ValueError("price, ratio and stock price must be positive")
    conv = stock_price * conversion_ratio
    mcp = convertible_price / conversion_ratio
    prem = mcp - stock_price
    fid = None if coupon_rate is None else (coupon_rate * par - conversion_ratio * dividend_per_share) / conversion_ratio
    return {"conversion_value": conv,
            "minimum_value": conv if straight_value is None else max(conv, straight_value),
            "market_conversion_price": mcp, "conversion_premium_per_share": prem,
            "conversion_premium_ratio": prem / stock_price,
            "favorable_income_differential_per_share": fid,
            "premium_payback_years": None if not fid or fid <= 0 else prem / fid,
            "premium_over_straight_value": None if straight_value is None else convertible_price / straight_value - 1,
            "conversion_price_at_issue": None if issue_price is None else issue_price / conversion_ratio}
