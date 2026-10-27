"""Equity investments: markets and indices (L1), required return, DDM, free cash flow, multiples,
residual income, private company valuation (L2) and equity portfolio management (L3).

All rates are decimals, all functions return dicts of unrounded values (see references/conventions.md).
"""
from __future__ import annotations

import math
from typing import Sequence

from ._registry import formula

T = "equity"


# ----------------------------------------------------------------------------------------------- helpers
def _need(value, name: str):
    if value is None:
        raise ValueError(f"{name} is required")
    return value


def _spread(r: float, g: float, rname: str = "required_return", gname: str = "growth") -> float:
    if r <= g:
        raise ValueError(f"{rname} ({r}) must exceed {gname} ({g}); a growing perpetuity has no finite value")
    return r - g


def _bisect(f, lo: float, hi: float, tol: float = 1e-12, max_iter: int = 1000) -> float:
    flo, fhi = f(lo), f(hi)
    if flo == 0:
        return lo
    if fhi == 0:
        return hi
    if flo * fhi > 0:
        raise ValueError("no solution in the search interval (the price may be inconsistent with the cash flows)")
    for _ in range(max_iter):
        mid = (lo + hi) / 2
        fm = f(mid)
        if fm == 0 or (hi - lo) / 2 < tol:
            return mid
        if flo * fm < 0:
            hi = mid
        else:
            lo, flo = mid, fm
    return (lo + hi) / 2


def _same_len(*seqs: Sequence, names: str) -> int:
    n = len(seqs[0])
    if n == 0 or any(len(s) != n for s in seqs):
        raise ValueError(f"{names} must be non-empty and the same length")
    return n


def _pv_stream(rate: float, flows: Sequence[float]) -> float:
    return sum(cf / (1 + rate) ** t for t, cf in enumerate(flows, start=1))


def _mean(xs: Sequence[float]) -> float:
    return sum(xs) / len(xs)


def _stdev(xs: Sequence[float]) -> float:
    if len(xs) < 2:
        raise ValueError("need at least two observations for a standard deviation")
    m = _mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))


# ------------------------------------------------------------------------------- L1: markets and indices
@formula(T, formula="IM = 1/leverage; equity0 = IM*P0*N + comm; equity1 = P1*N - loan - loan*i*t + div*N - comm; "
                    "R = equity1/equity0 - 1",
         curriculum="L1 Equity R46 Market Organization and Structure (leveraged positions)")
def equity_margin_purchase_return(purchase_price: float, sale_price: float, shares: float, call_money_rate: float,
                                  leverage_ratio: float | None = None, initial_margin: float | None = None,
                                  dividend_per_share: float = 0.0, commission_per_share: float = 0.0,
                                  years: float = 1.0) -> dict:
    """Total return on equity for a stock bought on margin (long position), incl. interest, dividends, commissions.

    Args:
        purchase_price, sale_price: per share.
        shares: number of shares bought.
        call_money_rate: annual interest rate on the margin loan (simple interest over `years`).
        leverage_ratio: position value / equity (give this or initial_margin; leverage = 1/initial margin).
        initial_margin: equity fraction of the purchase, decimal.
        dividend_per_share: dividends received over the holding period, per share.
        commission_per_share: commission paid on the purchase and again on the sale.
        years: holding period; interest = loan * call_money_rate * years.

    Returns:
        total_return, initial_investment, remaining_equity, loan, margin_interest, dividends, leverage_ratio,
        price_return, leveraged_price_return (leverage * price return, ignoring costs).

    Example:
        purchase 20, sale 15, 1,000 shares, leverage 2.5, call money 5%, dividend 0.10, commission 0.01
        -> total_return -0.689 (CFA L1 R46 Example 19).
    """
    if (leverage_ratio is None) == (initial_margin is None):
        raise ValueError("give exactly one of leverage_ratio or initial_margin")
    im = 1 / leverage_ratio if leverage_ratio is not None else initial_margin
    if not 0 < im <= 1:
        raise ValueError("initial margin must be in (0, 1] (leverage ratio >= 1)")
    if purchase_price <= 0 or shares <= 0:
        raise ValueError("purchase_price and shares must be positive")
    value = purchase_price * shares
    equity = im * value
    loan = value - equity
    commission = commission_per_share * shares
    initial = equity + commission
    interest = loan * call_money_rate * years
    dividends = dividend_per_share * shares
    remaining = sale_price * shares - loan - interest + dividends - commission
    price_ret = sale_price / purchase_price - 1
    return {"total_return": remaining / initial - 1, "initial_investment": initial, "remaining_equity": remaining,
            "loan": loan, "margin_interest": interest, "dividends": dividends, "commissions_total": 2 * commission,
            "leverage_ratio": 1 / im, "price_return": price_ret, "leveraged_price_return": price_ret / im}


@formula(T, formula="long: P = P0(1 - IM)/(1 - MM); short: P = P0(1 + IM)/(1 + MM)",
         curriculum="L1 Equity R46 Market Organization and Structure (margin call price)")
def equity_margin_call_price(purchase_price: float, initial_margin: float, maintenance_margin: float,
                             position: str = "long") -> dict:
    """Price that triggers a margin call for a margin purchase (long) or a short sale.

    Args:
        purchase_price: initial price (sale price for a short).
        initial_margin: equity posted as a fraction of the position value, decimal.
        maintenance_margin: minimum equity / position value, decimal.
        position: "long" (margin purchase: call when price falls below) or "short" (call when price rises above).

    Returns:
        margin_call_price, leverage_ratio (1/initial margin), initial_equity_per_share.

    Example:
        purchase 20, initial margin 40%, maintenance 25%, long -> 16.00 (CFA L1 R46 Example 20).
    """
    if not 0 < initial_margin <= 1 or not 0 <= maintenance_margin < 1:
        raise ValueError("margins must be decimals: 0 < initial_margin <= 1, 0 <= maintenance_margin < 1")
    if position == "long":
        price = purchase_price * (1 - initial_margin) / (1 - maintenance_margin)
    elif position == "short":
        price = purchase_price * (1 + initial_margin) / (1 + maintenance_margin)
    else:
        raise ValueError("position must be 'long' or 'short'")
    return {"margin_call_price": price, "leverage_ratio": 1 / initial_margin,
            "initial_equity_per_share": initial_margin * purchase_price, "position": position}


@formula(T, formula="PR = sum(h_i P1_i)/sum(h_i P0_i) - 1; TR = PR + sum(h_i D_i)/sum(h_i P0_i); holdings h: price=1, "
                    "equal=1/P0, cap=shares, float=shares*float",
         curriculum="L1 Equity R47 Security Market Indices (index weighting, price and total return)")
def equity_index_return(prices_begin: list, prices_end: list, dividends: list | None = None, weighting: str = "price",
                        shares: list | None = None, float_fraction: list | None = None,
                        divisor: float | None = None, base_value: float = 1000.0) -> dict:
    """Single-period price and total return, index values and weights for a price-, equal-, cap- or float-weighted index.

    Args:
        prices_begin, prices_end: constituent prices at the beginning and end of the period.
        dividends: income per share received during the period (default zero).
        weighting: "price", "equal", "market_cap" or "float_adjusted".
        shares: shares outstanding (required for market_cap and float_adjusted).
        float_fraction: fraction of shares in the market float, decimal (float_adjusted only).
        divisor: price-weighted divisor (default = number of constituents).
        base_value: starting index value for the non-price-weighted versions (default 1000).

    Returns:
        price_return, total_return, income_return, index_value_begin, price_index_end, total_return_index_end,
        weights_begin, weights_end, divisor.

    Example:
        prices 50/25/12.5/10/4 -> 55/22/8/14/6, dividends 0.75/0.10/0/0.05/0, price weighted
        -> price return 3.45%, total return 4.33% (CFA L1 R47 Exhibit 1).
    """
    n = _same_len(prices_begin, prices_end, names="prices_begin and prices_end")
    dividends = dividends or [0.0] * n
    _same_len(prices_begin, dividends, names="prices and dividends")
    if any(p <= 0 for p in prices_begin):
        raise ValueError("beginning prices must be positive")
    if weighting == "price":
        h = [1.0] * n
    elif weighting == "equal":
        h = [1.0 / p for p in prices_begin]
    elif weighting in ("market_cap", "float_adjusted"):
        h = list(_need(shares, "shares"))
        _same_len(prices_begin, h, names="prices and shares")
        if weighting == "float_adjusted":
            fl = _need(float_fraction, "float_fraction")
            _same_len(h, fl, names="shares and float_fraction")
            h = [s * f for s, f in zip(h, fl)]
    else:
        raise ValueError("weighting must be 'price', 'equal', 'market_cap' or 'float_adjusted'")
    v0 = sum(x * p for x, p in zip(h, prices_begin))
    v1 = sum(x * p for x, p in zip(h, prices_end))
    inc = sum(x * d for x, d in zip(h, dividends))
    pr, ir = v1 / v0 - 1, inc / v0
    if weighting == "price":
        div = divisor if divisor is not None else float(n)
        idx0 = v0 / div
    else:
        div = v0 / base_value
        idx0 = base_value
    return {"price_return": pr, "total_return": pr + ir, "income_return": ir, "index_value_begin": idx0,
            "price_index_end": idx0 * (1 + pr), "total_return_index_end": idx0 * (1 + pr + ir),
            "weights_begin": [x * p / v0 for x, p in zip(h, prices_begin)],
            "weights_end": [x * p / v1 for x, p in zip(h, prices_end)], "divisor": div}


@formula(T, formula="D_new = sum(P_after) / (sum(P_before) / D_old)",
         curriculum="L1 Equity R47 Security Market Indices (divisor adjustment for splits and constituent changes)")
def price_weighted_divisor_adjustment(prices_before: list, prices_after: list, divisor: float) -> dict:
    """New divisor that keeps a price-weighted index unchanged across a split, stock dividend or constituent change.

    Args:
        prices_before: constituent prices (old constituents) just before the change.
        prices_after: constituent prices (new constituents/split-adjusted) just after the change.
        divisor: current divisor.

    Returns:
        new_divisor, index_value (unchanged by construction), old_divisor.

    Example:
        prices 55/22/8/14/6 (divisor 5), 2-for-1 split of the first -> 27.5/22/8/14/6 -> divisor 3.69
        (CFA L1 R47 Exhibit 2).
    """
    if divisor <= 0 or not prices_before or not prices_after:
        raise ValueError("divisor must be positive and price lists non-empty")
    idx = sum(prices_before) / divisor
    return {"new_divisor": sum(prices_after) / idx, "index_value": idx, "old_divisor": divisor}


# ------------------------------------------------------------------------------------ required return / ERP
@formula(T, formula="r = Rf + beta * ERP", curriculum="L2 Equity R31 Return Concepts (CAPM)")
def equity_required_return_capm(risk_free: float, beta: float, equity_risk_premium: float) -> dict:
    """Required return on equity from the CAPM.

    Args:
        risk_free: risk-free rate, decimal.
        beta: equity beta.
        equity_risk_premium: E(Rm) - Rf, decimal.

    Returns:
        required_return, risk_premium (beta * ERP).

    Example:
        Rf 4.9%, beta 0.74, ERP 4.5% -> 8.23% (CFA L2 R31 Example 4, Exxon Mobil).
    """
    return {"required_return": risk_free + beta * equity_risk_premium, "risk_premium": beta * equity_risk_premium}


@formula(T, formula="r = Rf + sum(beta_k * premium_k)",
         curriculum="L2 Equity R31 Return Concepts (Fama-French, Pastor-Stambaugh, macro factor models)")
def equity_required_return_multifactor(risk_free: float, factor_betas: list, factor_premiums: list,
                                       factor_names: list | None = None) -> dict:
    """Required return from a multifactor model (Fama-French RMRF/SMB/HML, Pastor-Stambaugh +LIQ, BIRR, ...).

    Args:
        risk_free: risk-free rate (FFM uses a short-term rate), decimal.
        factor_betas: sensitivities, e.g. [market, size, value] for Fama-French.
        factor_premiums: factor risk premia, decimals, same order.
        factor_names: optional labels for the contributions.

    Returns:
        required_return, contributions (beta_k * premium_k per factor).

    Example:
        Rf 4.1%, betas [1.2, 0.5, 0.8], premia [5.5%, 2%, 4.3%] -> 15.1% (CFA L2 R31, Fama-French illustration).
    """
    n = _same_len(factor_betas, factor_premiums, names="factor_betas and factor_premiums")
    names = factor_names or [f"factor_{i + 1}" for i in range(n)]
    contrib = {nm: b * p for nm, b, p in zip(names, factor_betas, factor_premiums)}
    return {"required_return": risk_free + sum(contrib.values()), "contributions": contrib}


@formula(T, formula="r = Rf + [beta*]ERP + size premium + specific-company premium + other premia",
         curriculum="L2 Equity R31 Return Concepts (build-up method; modified CAPM)")
def equity_required_return_build_up(risk_free: float, equity_risk_premium: float, size_premium: float = 0.0,
                                    specific_company_premium: float = 0.0, other_premia: float = 0.0,
                                    beta: float | None = None) -> dict:
    """Build-up (private company) required return; with `beta` it becomes the modified CAPM.

    Args:
        risk_free: risk-free rate, decimal.
        equity_risk_premium: decimal.
        size_premium: beta-adjusted size premium, decimal.
        specific_company_premium: company-specific premium, decimal (can be negative).
        other_premia: any further premia (+) or discounts (-), decimal.
        beta: if given, ERP is multiplied by beta (modified CAPM); default treats beta as 1.

    Returns:
        required_return, equity_premium_component.

    Example:
        Rf 5%, ERP 4.5%, size premium 4.35% -> 13.85%; with beta 1.2 -> 14.75% (CFA L2 R31 Section 4.3.1).
    """
    eq = (1.0 if beta is None else beta) * equity_risk_premium
    return {"required_return": risk_free + eq + size_premium + specific_company_premium + other_premia,
            "equity_premium_component": eq}


@formula(T, formula="r = YTM on the company's long-term debt + risk premium",
         curriculum="L2 Equity R31 Return Concepts (bond yield plus risk premium)")
def equity_required_return_bond_yield_plus(bond_ytm: float, risk_premium: float) -> dict:
    """Bond-yield-plus-risk-premium cost of equity (typical U.S. premium 3-4%).

    Args:
        bond_ytm: YTM on the company's long-term debt, decimal.
        risk_premium: equity premium over the company's own debt, decimal.

    Returns:
        required_return.

    Example:
        IBM bond YTM 5.632% + 3.5% -> 9.132% (CFA L2 R31 Example 9).
    """
    return {"required_return": bond_ytm + risk_premium}


@formula(T, formula="ERP = D1/P0 (index) + long-term earnings growth - long-term government bond yield",
         curriculum="L2 Equity R31 Return Concepts (Gordon growth model equity risk premium)")
def equity_risk_premium_gordon(forward_dividend_yield: float, long_term_growth: float,
                               government_bond_yield: float) -> dict:
    """Forward-looking (Gordon growth model) equity risk premium for a market index.

    Args:
        forward_dividend_yield: year-ahead aggregate dividends / index level, decimal.
        long_term_growth: consensus long-term earnings (dividend) growth, decimal.
        government_bond_yield: current long-term government bond yield, decimal.

    Returns:
        equity_risk_premium, expected_market_return (yield + growth).

    Example:
        1.9% + 7.0% - 5.0% -> 3.9% (CFA L2 R31, S&P 500 September 2007).
    """
    er = forward_dividend_yield + long_term_growth
    return {"equity_risk_premium": er - government_bond_yield, "expected_market_return": er}


@formula(T, formula="ERP = [(1+EINFL)(1+EGREPS)(1+EGPE) - 1 + EINC] - Rf",
         curriculum="L2 Equity R31 Return Concepts (Ibbotson-Chen supply-side estimate)")
def equity_risk_premium_supply_side(expected_inflation: float, real_eps_growth: float, pe_growth: float,
                                    income_return: float, risk_free: float) -> dict:
    """Supply-side (macroeconomic) equity risk premium in the Ibbotson-Chen format.

    Args:
        expected_inflation: EINFL, decimal.
        real_eps_growth: expected real EPS growth (~ real GDP growth), decimal.
        pe_growth: expected growth in the P/E (0 under efficient markets), decimal.
        income_return: expected income component incl. reinvestment, decimal.
        risk_free: expected risk-free return (long-term government bond yield), decimal.

    Returns:
        equity_risk_premium, expected_equity_return.

    Example:
        2.5%, 3%, 0, 2.2%, Rf 5% -> 2.8% (CFA L2 R31 Section 3.2.3).
    """
    er = (1 + expected_inflation) * (1 + real_eps_growth) * (1 + pe_growth) - 1 + income_return
    return {"equity_risk_premium": er - risk_free, "expected_equity_return": er}


@formula(T, formula="ERP = mean(R_equity) - mean(R_f), both arithmetic or both geometric",
         curriculum="L2 Equity R31 Return Concepts (historical equity risk premium)")
def equity_risk_premium_historical(equity_returns: list, risk_free_returns: list, method: str = "geometric") -> dict:
    """Historical equity risk premium as the difference of mean equity and mean risk-free returns.

    Args:
        equity_returns: periodic equity index returns, decimals.
        risk_free_returns: matching risk-free (bill or bond) returns, decimals.
        method: "geometric" (curriculum's growing preference) or "arithmetic".

    Returns:
        equity_risk_premium, mean_equity_return, mean_risk_free_return.

    Example:
        equity [0.10, -0.05, 0.20], bills [0.03, 0.03, 0.03], arithmetic -> 0.0533.
    """
    n = _same_len(equity_returns, risk_free_returns, names="equity_returns and risk_free_returns")
    if method == "arithmetic":
        me, mf = _mean(equity_returns), _mean(risk_free_returns)
    elif method == "geometric":
        me = math.prod(1 + r for r in equity_returns) ** (1 / n) - 1
        mf = math.prod(1 + r for r in risk_free_returns) ** (1 / n) - 1
    else:
        raise ValueError("method must be 'geometric' or 'arithmetic'")
    return {"equity_risk_premium": me - mf, "mean_equity_return": me, "mean_risk_free_return": mf}


@formula(T, formula="WACC = wd*rd*(1 - t) + wp*rp + we*re, we = 1 - wd - wp",
         curriculum="L2 Equity R36 Free Cash Flow Valuation (discount rate for FCFF)")
def equity_valuation_wacc(debt_weight: float, cost_of_debt: float, tax_rate: float, cost_of_equity: float,
                          preferred_weight: float = 0.0, cost_of_preferred: float = 0.0) -> dict:
    """Weighted average cost of capital (market-value or target weights) for discounting FCFF.

    Args:
        debt_weight: D/(D+P+E), decimal.
        cost_of_debt: before-tax cost of debt, decimal.
        tax_rate: marginal tax rate, decimal.
        cost_of_equity: required return on equity, decimal.
        preferred_weight: P/(D+P+E), decimal.
        cost_of_preferred: required return on preferred, decimal.

    Returns:
        wacc, equity_weight, after_tax_cost_of_debt.

    Example:
        20% debt at 5.7%, tax 33.33%, equity 11.8% -> 10.2% (CFA L2 R36 Example 1, Cagiati).
    """
    we = 1 - debt_weight - preferred_weight
    if we <= 0 or debt_weight < 0 or preferred_weight < 0:
        raise ValueError("weights must be non-negative and leave a positive equity weight")
    atd = cost_of_debt * (1 - tax_rate)
    return {"wacc": debt_weight * atd + preferred_weight * cost_of_preferred + we * cost_of_equity,
            "equity_weight": we, "after_tax_cost_of_debt": atd}


# ----------------------------------------------------------------------------------------------- DDM
@formula(T, formula="V0 = D1/(r - g) = D0(1 + g)/(r - g)",
         curriculum="L1 R51 / L2 R35 Discounted Dividend Valuation (Gordon growth model)")
def ddm_gordon_value(required_return: float, growth: float, d0: float | None = None, d1: float | None = None) -> dict:
    """Gordon (constant-growth) DDM value; also values a perpetuity (g = 0) or declining dividends (g < 0).

    Args:
        required_return: required return on equity, decimal.
        growth: constant dividend growth rate, decimal (must be below required_return).
        d0: most recent (current) annual dividend; d1 = d0 * (1 + g).
        d1: next year's dividend (give d0 or d1, not both).

    Returns:
        value, d1, dividend_yield (d1/value), capital_gains_yield (= g).

    Example:
        D0 1.04, g 6%, r 10.1% -> 26.89 (CFA L2 R35 Example 5, Sonoco).
    """
    if (d0 is None) == (d1 is None):
        raise ValueError("give exactly one of d0 or d1 (watch the D0 vs D1 trap)")
    d1 = d1 if d1 is not None else d0 * (1 + growth)
    v = d1 / _spread(required_return, growth)
    return {"value": v, "d1": d1, "dividend_yield": d1 / v, "capital_gains_yield": growth}


@formula(T, formula="V0 = sum D_t/(1+r)^t + V_n/(1+r)^n,  V_n = D_n(1+g)/(r - g) or a given terminal value",
         curriculum="L2 Equity R35 Discounted Dividend Valuation (multistage / spreadsheet DDM)")
def ddm_multistage_value(required_return: float, dividends: list, terminal_growth: float | None = None,
                         terminal_value: float | None = None, terminal_dividend: float | None = None) -> dict:
    """General multistage DDM: explicit dividends D1..Dn plus a terminal value at t = n.

    Args:
        required_return: decimal.
        dividends: forecast dividends for t = 1..n (zeros allowed for non-payers).
        terminal_growth: stable growth after n; terminal value = D_{n+1}/(r - g).
        terminal_value: alternative terminal value at t = n (e.g. justified P/E x EPS_n); overrides growth.
        terminal_dividend: D_{n+1} if it is not D_n * (1 + g) (e.g. a payout change).

    Returns:
        value, pv_dividends, terminal_value, pv_terminal_value, terminal_value_share.

    Example:
        dividends [21, 18.9, 17.01, 15.309, 60, 40, 40], g 5%, r 12% -> 399.48 (CFA L2 R35 Example 19, Yang Co.).
    """
    if not dividends:
        raise ValueError("dividends must hold at least one forecast dividend")
    n, r = len(dividends), required_return
    if terminal_value is not None:
        tv = terminal_value
    elif terminal_growth is not None:
        dn1 = terminal_dividend if terminal_dividend is not None else dividends[-1] * (1 + terminal_growth)
        tv = dn1 / _spread(r, terminal_growth, gname="terminal_growth")
    else:
        tv = 0.0
    pvd = _pv_stream(r, dividends)
    pvt = tv / (1 + r) ** n
    return {"value": pvd + pvt, "pv_dividends": pvd, "terminal_value": tv, "pv_terminal_value": pvt,
            "terminal_value_share": pvt / (pvd + pvt) if pvd + pvt else float("nan")}


@formula(T, formula="D_t = D0(1+gS)^t, t<=n; V_n = D_n(1+gL)/(r - gL); V0 = sum PV(D_t) + PV(V_n)",
         curriculum="L1 R51 / L2 R35 Discounted Dividend Valuation (two-stage DDM)")
def ddm_two_stage_value(d0: float, g_short: float, years_short: int, g_long: float, required_return: float) -> dict:
    """Two-stage DDM: growth g_short for `years_short` years, then g_long forever.

    Args:
        d0: current dividend.
        g_short: first-stage growth, decimal.
        years_short: length of the first stage in years (integer).
        g_long: mature growth, decimal (< required_return).
        required_return: decimal.

    Returns:
        value, pv_dividends, terminal_value, pv_terminal_value, terminal_value_share, dividends.

    Example:
        D0 0.14, 15% for 10 years then 8%, r 9.7% -> 16.09 (CFA L2 R35 Example 13, Carl Zeiss Meditec).
    """
    if years_short < 1 or int(years_short) != years_short:
        raise ValueError("years_short must be a positive integer")
    divs = [d0 * (1 + g_short) ** t for t in range(1, int(years_short) + 1)]
    out = ddm_multistage_value(required_return, divs, terminal_growth=g_long)
    return {**out, "dividends": divs}


@formula(T, formula="growth g1 for n1 years, g2 for n2 years, then g3 forever; V0 = PV(dividends) + PV(V_{n1+n2})",
         curriculum="L2 Equity R35 Discounted Dividend Valuation (three-stage DDM, distinct stages)")
def ddm_three_stage_value(d0: float, g1: float, years1: int, g2: float, years2: int, g3: float,
                          required_return: float) -> dict:
    """Three-stage DDM with three distinct constant growth rates.

    Args:
        d0: current dividend.
        g1, years1: stage-1 growth and length (years).
        g2, years2: stage-2 growth and length (years).
        g3: perpetual growth, decimal (< required_return).
        required_return: decimal.

    Returns:
        value, pv_dividends, terminal_value, pv_terminal_value, terminal_value_share, dividends.

    Example:
        D0 1.60, 14% x2, 12% x5, 10.2% after, r 12% -> 113.06 (CFA L2 R35 Example 17, IBM).
    """
    divs, d = [], d0
    for g, n in ((g1, years1), (g2, years2)):
        if n < 0 or int(n) != n:
            raise ValueError("stage lengths must be non-negative integers")
        for _ in range(int(n)):
            d *= 1 + g
            divs.append(d)
    out = ddm_multistage_value(required_return, divs, terminal_growth=g3)
    return {**out, "dividends": divs}


@formula(T, formula="V0 = D0(1 + gL)/(r - gL) + D0*H*(gS - gL)/(r - gL),  H = transition years / 2",
         curriculum="L2 Equity R35 Discounted Dividend Valuation (H-model)")
def ddm_h_model_value(d0: float, g_short: float, g_long: float, required_return: float,
                      transition_years: float) -> dict:
    """H-model: growth declines linearly from g_short to g_long over `transition_years`, then g_long forever.

    Args:
        d0: current dividend.
        g_short: initial growth rate, decimal.
        g_long: long-run growth rate, decimal (< required_return).
        required_return: decimal.
        transition_years: years for growth to fall from g_short to g_long (H = half of it).

    Returns:
        value, value_mature_growth (first term), value_extraordinary_growth (second term), half_life_h.

    Example:
        D0 1.37, gS 24% -> gL 6% over 12 years, r 10% -> 73.30 (CFA L2 R35 Example 16, Vinci).
    """
    h = transition_years / 2
    s = _spread(required_return, g_long, gname="g_long")
    a, b = d0 * (1 + g_long) / s, d0 * h * (g_short - g_long) / s
    return {"value": a + b, "value_mature_growth": a, "value_extraordinary_growth": b, "half_life_h": h}


@formula(T, formula="stage 1: D_t = D0(1+gS)^t, t<=n1; at n1 apply the H-model with D_n1; V0 = PV(D) + PV(V_n1)",
         curriculum="L2 Equity R35 Discounted Dividend Valuation (three-stage DDM with declining stage-2 growth)")
def ddm_three_stage_declining_value(d0: float, g_short: float, years_short: int, g_long: float,
                                    transition_years: float, required_return: float) -> dict:
    """Three-stage DDM: constant g_short for `years_short`, linear decline to g_long over `transition_years` (H-model).

    Args:
        d0: current dividend.
        g_short: stage-1 growth, decimal.
        years_short: stage-1 length, integer years.
        g_long: mature growth, decimal.
        transition_years: length of the linearly declining stage 2 (H = transition_years/2).
        required_return: decimal.

    Returns:
        value, pv_stage1_dividends, h_model_value_at_n1, pv_h_model_value, stage1_share, dividends.

    Example:
        D0 0.46, 12% x5, decline to 7.5% over 10 years, r 10% -> 28.60 (CFA L2 R35 Example 18, Energen).
    """
    if years_short < 0 or int(years_short) != years_short:
        raise ValueError("years_short must be a non-negative integer")
    n, r = int(years_short), required_return
    divs = [d0 * (1 + g_short) ** t for t in range(1, n + 1)]
    dn = d0 * (1 + g_short) ** n
    vn = ddm_h_model_value(dn, g_short, g_long, r, transition_years)["value"]
    pvd, pvn = _pv_stream(r, divs), vn / (1 + r) ** n
    return {"value": pvd + pvn, "pv_stage1_dividends": pvd, "h_model_value_at_n1": vn, "pv_h_model_value": pvn,
            "stage1_share": pvd / (pvd + pvn), "dividends": divs}


@formula(T, formula="r = D1/P0 + g", curriculum="L2 Equity R35 / R31 (Gordon-implied required/expected return)")
def ddm_gordon_implied_return(price: float, growth: float, d0: float | None = None, d1: float | None = None) -> dict:
    """Market-implied required (expected) return from the Gordon growth model.

    Args:
        price: current share price.
        growth: constant dividend growth, decimal.
        d0: current dividend (d1 = d0(1+g)); or give d1 directly.
        d1: next year's dividend.

    Returns:
        required_return, dividend_yield (D1/P0), capital_gains_yield (g).

    Example:
        D0 2.24, g 5.5%, P0 56.60 -> 9.67% (CFA L2 R35 Example 12, FPL Group).
    """
    if (d0 is None) == (d1 is None):
        raise ValueError("give exactly one of d0 or d1")
    if price <= 0:
        raise ValueError("price must be positive")
    d1 = d1 if d1 is not None else d0 * (1 + growth)
    return {"required_return": d1 / price + growth, "dividend_yield": d1 / price, "capital_gains_yield": growth}


@formula(T, formula="r = (D0/P0)[(1 + gL) + H(gS - gL)] + gL",
         curriculum="L2 Equity R35 Discounted Dividend Valuation (H-model expected return)")
def ddm_h_model_implied_return(price: float, d0: float, g_short: float, g_long: float,
                               transition_years: float) -> dict:
    """Expected (IRR) return implied by the H-model at the current price.

    Args:
        price: current share price.
        d0: current dividend.
        g_short, g_long: initial and long-run growth, decimals.
        transition_years: years over which growth declines (H = half).

    Returns:
        required_return, half_life_h.

    Example:
        D0 1, P0 20, 10% declining to 6% over 10 years -> 12.3% (CFA L2 R35 Section 5.6).
    """
    h = transition_years / 2
    return {"required_return": d0 / price * ((1 + g_long) + h * (g_short - g_long)) + g_long, "half_life_h": h}


@formula(T, formula="solve r: P0 = sum D_t/(1+r)^t + [D_n(1+g)/(r - g)]/(1+r)^n",
         curriculum="L2 Equity R35 Discounted Dividend Valuation (IRR of a multistage DDM)")
def ddm_multistage_implied_return(price: float, dividends: list, terminal_growth: float) -> dict:
    """Expected return (IRR) that equates a multistage dividend forecast to the market price.

    Args:
        price: current share price.
        dividends: forecast dividends D1..Dn.
        terminal_growth: growth after n (terminal value by Gordon on D_n(1+g)).

    Returns:
        required_return, check_value (model value at the solved rate).

    Example:
        D0 1.66 growing 9% for 6 years then 7%, P0 66.19 -> 9.98% (CFA L2 R35 Example 21, Johnson & Johnson).
    """
    if price <= 0 or not dividends:
        raise ValueError("price must be positive and dividends non-empty")
    f = lambda r: ddm_multistage_value(r, dividends, terminal_growth=terminal_growth)["value"] - price  # noqa: E731
    r = _bisect(f, terminal_growth + 1e-9, 10.0)
    return {"required_return": r, "check_value": f(r) + price}


@formula(T, formula="with D0: g = (P0 r - D0)/(P0 + D0); with D1: g = r - D1/P0",
         curriculum="L2 Equity R35 Discounted Dividend Valuation (implied dividend growth rate)")
def ddm_implied_growth(price: float, required_return: float, d0: float | None = None, d1: float | None = None) -> dict:
    """Constant dividend growth rate implied by the market price (Gordon growth model solved for g).

    Args:
        price: current share price.
        required_return: decimal.
        d0: current dividend (growth applies to it), or
        d1: next year's dividend.

    Returns:
        implied_growth.

    Example:
        P0 40, D0 2.00, r 12.2% -> 6.86% (CFA L2 R35 Example 10).
    """
    if (d0 is None) == (d1 is None):
        raise ValueError("give exactly one of d0 or d1")
    if price <= 0:
        raise ValueError("price must be positive")
    g = (price * required_return - d0) / (price + d0) if d0 is not None else required_return - d1 / price
    return {"implied_growth": g}


@formula(T, formula="perpetual: V = D/r; with maturity/retraction: V = sum (D/m)/(1+r/m)^t + Par/(1+r/m)^N",
         curriculum="L1 R51 / L2 R35 (preferred stock valuation)")
def equity_preferred_stock_value(annual_dividend: float, required_return: float, years: float | None = None,
                                 par: float | None = None, freq: int = 1) -> dict:
    """Value of a non-callable preferred share: perpetual, or with a maturity/retraction date paying par.

    Args:
        annual_dividend: dividend per year (paid in `freq` equal installments).
        required_return: annual required return (stated rate compounded `freq` times), decimal.
        years: years to maturity/retraction (None = perpetual).
        par: amount received at maturity/retraction (required with `years`).
        freq: dividend payments per year.

    Returns:
        value, pv_dividends, pv_par, periods.

    Example:
        4.75 at 7.5%, perpetual -> 63.33 (CFA L1 R51 Example 5, Union Electric).
    """
    if required_return <= 0 or freq < 1:
        raise ValueError("required_return must be positive and freq >= 1")
    if years is None:
        return {"value": annual_dividend / required_return, "pv_dividends": annual_dividend / required_return,
                "pv_par": 0.0, "periods": math.inf}
    n = years * freq
    if abs(n - round(n)) > 1e-9:
        raise ValueError("years * freq must be a whole number of periods")
    n, i = int(round(n)), required_return / freq
    pvd = annual_dividend / freq * (1 - (1 + i) ** -n) / i
    pvp = _need(par, "par") / (1 + i) ** n
    return {"value": pvd + pvp, "pv_dividends": pvd, "pv_par": pvp, "periods": n}


@formula(T, formula="V0 = E1/r + PVGO;  P/E1 = 1/r + PVGO/E1",
         curriculum="L2 Equity R35 Discounted Dividend Valuation (present value of growth opportunities)")
def equity_pvgo_decomposition(price: float, eps_next: float, required_return: float) -> dict:
    """Split price into no-growth value (E1/r) and the present value of growth opportunities.

    Args:
        price: current price (or intrinsic value).
        eps_next: next year's EPS (E1), the no-growth perpetual earnings level.
        required_return: decimal.

    Returns:
        pvgo, no_growth_value, pvgo_share_of_price, leading_pe, no_growth_pe, growth_pe_component.

    Example:
        P 18.39, E1 0.79, r 9.25% -> no-growth value 8.54, PVGO 9.85 (CFA L2 R35 Section 4.5, Middlesex Water).
    """
    if required_return <= 0 or price <= 0 or eps_next == 0:
        raise ValueError("price and required_return must be positive and eps_next non-zero")
    ng = eps_next / required_return
    pv = price - ng
    return {"pvgo": pv, "no_growth_value": ng, "pvgo_share_of_price": pv / price, "leading_pe": price / eps_next,
            "no_growth_pe": 1 / required_return, "growth_pe_component": pv / eps_next}


@formula(T, formula="g = b x ROE;  ROE = (NI/Sales)(Sales/Assets)(Assets/Equity)  (PRAT)",
         curriculum="L2 Equity R35 Discounted Dividend Valuation (sustainable growth, DuPont/PRAT)")
def ddm_sustainable_growth(retention_rate: float | None = None, payout_ratio: float | None = None,
                           roe: float | None = None, net_profit_margin: float | None = None,
                           asset_turnover: float | None = None, equity_multiplier: float | None = None,
                           roa: float | None = None) -> dict:
    """Sustainable growth g = b x ROE, with ROE given directly, as ROA x leverage, or from the 3-way DuPont.

    Args:
        retention_rate: b, decimal (or give payout_ratio = 1 - b).
        payout_ratio: dividend payout, decimal.
        roe: return on equity, decimal; else
        roa: return on assets with equity_multiplier; else
        net_profit_margin, asset_turnover, equity_multiplier: DuPont components.

    Returns:
        sustainable_growth, roe, retention_rate.

    Example:
        b 0.30, ROA 10%, equity multiplier 1.25 -> 3.75% (CFA L2 R35 Example 23, Baggai).
    """
    if (retention_rate is None) == (payout_ratio is None):
        raise ValueError("give exactly one of retention_rate or payout_ratio")
    b = retention_rate if retention_rate is not None else 1 - payout_ratio
    if roe is None:
        em = _need(equity_multiplier, "roe or equity_multiplier")
        if roa is None:
            roa = _need(net_profit_margin, "net_profit_margin") * _need(asset_turnover, "asset_turnover")
        roe = roa * em
    return {"sustainable_growth": b * roe, "roe": roe, "retention_rate": b}


@formula(T, formula="V = CF0(1 + g)/(r - g) over a grid of r and g",
         curriculum="L2 Equity R35 / R36 (sensitivity of single-stage DDM and FCF values)")
def equity_gordon_sensitivity_table(cash_flow_0: float, required_returns: list, growth_rates: list) -> dict:
    """Sensitivity grid of single-stage values CF0(1+g)/(r-g) (dividends, FCFE or FCFF) for ranges of r and g.

    Args:
        cash_flow_0: current-period cash flow (D0, FCFE0 per share or FCFF0).
        required_returns: discount rates (rows), decimals.
        growth_rates: growth rates (columns), decimals.

    Returns:
        table (list of rows; NaN where r <= g), required_returns, growth_rates.

    Example:
        D0 0.68, r 9.25%, g 6% -> 22.18 at the centre of the grid (CFA L2 R35 Example 7, Exhibit 3).
    """
    if not required_returns or not growth_rates:
        raise ValueError("required_returns and growth_rates must be non-empty")
    table = [[cash_flow_0 * (1 + g) / (r - g) if r > g else float("nan") for g in growth_rates]
             for r in required_returns]
    return {"table": table, "required_returns": list(required_returns), "growth_rates": list(growth_rates)}


# ------------------------------------------------------------------------------------------- free cash flow
@formula(T, formula="NI: FCFF = NI + NCC + Int(1-t) [+ PrefDiv] - FCInv - WCInv; CFO: FCFF = CFO + Int(1-t) - FCInv; "
                    "EBIT: FCFF = EBIT(1-t) + Dep - FCInv - WCInv; EBITDA: FCFF = EBITDA(1-t) + Dep*t - FCInv - WCInv; "
                    "FCFE = FCFF - Int(1-t) [- PrefDiv] + Net borrowing",
         curriculum="L2 Equity R36 Free Cash Flow Valuation (computing FCFF and FCFE)")
def equity_fcff_fcfe(tax_rate: float, start: str = "net_income", net_income: float | None = None,
                     cfo: float | None = None, ebit: float | None = None, ebitda: float | None = None,
                     noncash_charges: float = 0.0, interest_expense: float = 0.0,
                     fixed_capital_investment: float = 0.0, working_capital_investment: float = 0.0,
                     net_borrowing: float = 0.0, preferred_dividends: float = 0.0) -> dict:
    """FCFF and FCFE from net income, CFO (U.S. GAAP interest in CFO), EBIT or EBITDA.

    Args:
        tax_rate: decimal.
        start: "net_income", "cfo", "ebit" or "ebitda".
        net_income: net income available to common (start="net_income").
        cfo: cash flow from operations, interest paid included (start="cfo").
        ebit, ebitda: operating earnings (start="ebit"/"ebitda").
        noncash_charges: net noncash charges (depreciation for EBIT/EBITDA starts; not used with CFO).
        interest_expense: before-tax interest expense.
        fixed_capital_investment: FCInv (capex net of asset sales).
        working_capital_investment: WCInv (increase in non-cash working capital excl. short-term debt).
        net_borrowing: debt issued minus repaid.
        preferred_dividends: added back to FCFF and deducted for FCFE.

    Returns:
        fcff, fcfe, after_tax_interest, start.

    Example:
        NI 240, NCC 300, interest 100, t 40%, FCInv 400, WCInv 45, net borrowing 75 -> FCFF 155, FCFE 170
        (CFA L2 R36 Example 6, Pitts Corporation).
    """
    ati = interest_expense * (1 - tax_rate)
    if start == "net_income":
        fcff = (_need(net_income, "net_income") + noncash_charges + ati + preferred_dividends
                - fixed_capital_investment - working_capital_investment)
    elif start == "cfo":
        fcff = _need(cfo, "cfo") + ati + preferred_dividends - fixed_capital_investment
    elif start == "ebit":
        fcff = (_need(ebit, "ebit") * (1 - tax_rate) + noncash_charges - fixed_capital_investment
                - working_capital_investment)
    elif start == "ebitda":
        fcff = (_need(ebitda, "ebitda") * (1 - tax_rate) + noncash_charges * tax_rate - fixed_capital_investment
                - working_capital_investment)
    else:
        raise ValueError("start must be 'net_income', 'cfo', 'ebit' or 'ebitda'")
    fcfe = fcff - ati - preferred_dividends + net_borrowing
    return {"fcff": fcff, "fcfe": fcfe, "after_tax_interest": ati, "start": start}


@formula(T, formula="FCFF = EBIT(1-t) - fc_rate*dSales - wc_rate*dSales; "
                    "FCFE = NI - (1-DR)(fc_rate*dSales) - (1-DR)(wc_rate*dSales)",
         curriculum="L2 Equity R36 Free Cash Flow Valuation (forecasting free cash flow from sales)")
def equity_fcf_from_sales_forecast(prior_sales: float, sales_growth: float, tax_rate: float,
                                   fixed_capital_rate: float, working_capital_rate: float,
                                   ebit_margin: float | None = None, net_profit_margin: float | None = None,
                                   debt_ratio: float = 0.0) -> dict:
    """One-year FCFF and/or FCFE forecast with investment tied to the sales increase.

    Args:
        prior_sales: last year's sales.
        sales_growth: forecast sales growth, decimal.
        tax_rate: decimal (FCFF only).
        fixed_capital_rate: (capex - depreciation) / increase in sales, decimal.
        working_capital_rate: increase in working capital / increase in sales, decimal.
        ebit_margin: EBIT / sales (gives FCFF).
        net_profit_margin: net income / sales (gives FCFE).
        debt_ratio: target fraction of net new investment financed with debt (FCFE).

    Returns:
        sales, sales_increase, fixed_capital_investment_net, working_capital_investment, fcff (if ebit_margin),
        net_income, net_borrowing, fcfe (if net_profit_margin).

    Example:
        sales 3,000 +10%, EBIT margin 16.67%, t 40%, FC 33.33%, WC 15% -> FCFF 185 (CFA L2 R36 Example 9, Pitts).
    """
    if ebit_margin is None and net_profit_margin is None:
        raise ValueError("give ebit_margin (FCFF) and/or net_profit_margin (FCFE)")
    sales = prior_sales * (1 + sales_growth)
    ds = sales - prior_sales
    fc, wc = fixed_capital_rate * ds, working_capital_rate * ds
    out = {"sales": sales, "sales_increase": ds, "fixed_capital_investment_net": fc, "working_capital_investment": wc}
    if ebit_margin is not None:
        out["fcff"] = ebit_margin * sales * (1 - tax_rate) - fc - wc
    if net_profit_margin is not None:
        ni, nb = net_profit_margin * sales, debt_ratio * (fc + wc)
        out.update(net_income=ni, net_borrowing=nb, fcfe=ni - fc - wc + nb)
    return out


def _dcf(rate: float, g: float, cf0, forecast, terminal_cf) -> tuple[float, float, float, float]:
    """Return (value, pv_explicit, terminal_value, pv_terminal) for single- or multi-stage growing cash flows."""
    s = _spread(rate, g, rname="discount rate", gname="growth")
    if forecast is None:
        cf1 = _need(cf0, "cash flow at t=0 (or a forecast list)") * (1 + g)
        return cf1 / s, 0.0, cf1 / s, cf1 / s
    if not forecast:
        raise ValueError("forecast list must not be empty")
    n = len(forecast)
    tcf = terminal_cf if terminal_cf is not None else forecast[-1] * (1 + g)
    tv = tcf / s
    pve, pvt = _pv_stream(rate, forecast), tv / (1 + rate) ** n
    return pve + pvt, pve, tv, pvt


@formula(T, formula="Firm = sum FCFF_t/(1+WACC)^t + [FCFF_{n+1}/(WACC - g)]/(1+WACC)^n; "
                    "Equity = Firm - debt - preferred + nonoperating assets",
         curriculum="L2 Equity R36 Free Cash Flow Valuation (single-stage and multistage FCFF)")
def equity_fcff_valuation(wacc: float, growth: float, fcff0: float | None = None, forecast_fcff: list | None = None,
                          terminal_fcff: float | None = None, debt: float = 0.0, preferred: float = 0.0,
                          nonoperating_assets: float = 0.0, shares: float | None = None) -> dict:
    """Firm and equity value from FCFF: single-stage (fcff0) or explicit forecast plus Gordon terminal value.

    Args:
        wacc: discount rate, decimal.
        growth: stable growth after the forecast (or forever in the single-stage case), decimal.
        fcff0: current FCFF (single-stage: FCFF1 = FCFF0(1+g)).
        forecast_fcff: FCFF for t = 1..n (multistage).
        terminal_fcff: FCFF_{n+1} if it is not forecast[-1](1+g).
        debt: market value of debt; preferred: value of preferred stock.
        nonoperating_assets: cash/investments not in FCFF, added to equity.
        shares: shares outstanding for a per-share value.

    Returns:
        firm_value, equity_value, value_per_share (if shares), pv_forecast, terminal_value, pv_terminal_value.

    Example:
        FCFF0 700, g 5%, WACC 10.2%, debt 2,200, 200 shares -> firm 14,134.6, per share 59.67
        (CFA L2 R36 Example 1, Cagiati).
    """
    firm, pve, tv, pvt = _dcf(wacc, growth, fcff0, forecast_fcff, terminal_fcff)
    eq = firm - debt - preferred + nonoperating_assets
    out = {"firm_value": firm, "equity_value": eq, "pv_forecast": pve, "terminal_value": tv, "pv_terminal_value": pvt}
    if shares:
        out["value_per_share"] = eq / shares
    return out


@formula(T, formula="Equity = sum FCFE_t/(1+r)^t + [FCFE_{n+1}/(r - g)]/(1+r)^n",
         curriculum="L2 Equity R36 Free Cash Flow Valuation (single-stage and two-stage FCFE)")
def equity_fcfe_valuation(required_return: float, growth: float, fcfe0: float | None = None,
                          forecast_fcfe: list | None = None, terminal_fcfe: float | None = None,
                          shares: float | None = None) -> dict:
    """Equity value from FCFE: single-stage (fcfe0) or explicit forecast plus Gordon terminal value.

    Args:
        required_return: required return on equity, decimal.
        growth: stable FCFE growth, decimal.
        fcfe0: current FCFE (total or per share).
        forecast_fcfe: FCFE for t = 1..n.
        terminal_fcfe: FCFE_{n+1} when it differs from forecast[-1](1+g) (e.g. capex falls when growth slows).
        shares: shares outstanding (if FCFE is a total).

    Returns:
        equity_value, value_per_share (if shares), pv_forecast, terminal_value, pv_terminal_value.

    Example:
        FCFE per share [0.90, 1.08, 1.296], FCFE4 3.491, g 6%, r 12.4% -> 40.98 (CFA L2 R36 Example 16, TechnoSchaft).
    """
    eq, pve, tv, pvt = _dcf(required_return, growth, fcfe0, forecast_fcfe, terminal_fcfe)
    out = {"equity_value": eq, "pv_forecast": pve, "terminal_value": tv, "pv_terminal_value": pvt}
    if shares:
        out["value_per_share"] = eq / shares
    return out


# ----------------------------------------------------------------------------------------------- multiples
@formula(T, formula="leading P0/E1 = (1 - b)/(r - g); trailing P0/E0 = (1 - b)(1 + g)/(r - g)",
         curriculum="L1 R51 / L2 R35, R37 (justified P/E from fundamentals)")
def equity_justified_pe(payout_ratio: float, required_return: float, growth: float) -> dict:
    """Justified leading and trailing P/E from the Gordon growth model.

    Args:
        payout_ratio: dividend payout 1 - b, decimal.
        required_return: decimal.
        growth: sustainable growth, decimal.

    Returns:
        leading_pe, trailing_pe.

    Example:
        payout 0.32, r 9%, g 7% -> leading 16.0, trailing 17.1 (CFA L2 R35 Example 11, Carrefour).
    """
    s = _spread(required_return, growth)
    return {"leading_pe": payout_ratio / s, "trailing_pe": payout_ratio * (1 + growth) / s}


@formula(T, formula="P0/B0 = (ROE - g)/(r - g) = 1 + (ROE - r)/(r - g)",
         curriculum="L2 Equity R37 Market-Based Valuation / R38 Residual Income (justified P/B)")
def equity_justified_pb(roe: float, required_return: float, growth: float) -> dict:
    """Justified price-to-book from ROE, required return and sustainable growth.

    Args:
        roe: return on equity, decimal.
        required_return: decimal.
        growth: sustainable growth, decimal.

    Returns:
        justified_pb.

    Example:
        ROE 12%, r 10%, g 7% -> 1.67 (CFA L2 R37 Section 3.2.3).
    """
    return {"justified_pb": (roe - growth) / _spread(required_return, growth)}


@formula(T, formula="trailing P0/S0 = PM0 (1 - b)(1 + g)/(r - g); leading P0/S1 = PM1 (1 - b)/(r - g)",
         curriculum="L2 Equity R37 Market-Based Valuation (justified P/S)")
def equity_justified_ps(net_profit_margin: float, payout_ratio: float, required_return: float, growth: float) -> dict:
    """Justified price-to-sales from profit margin, payout, required return and growth.

    Args:
        net_profit_margin: E/S, decimal.
        payout_ratio: 1 - b, decimal.
        required_return: decimal.
        growth: decimal.

    Returns:
        trailing_ps, leading_ps.

    Example:
        margin 8.2%, payout 30%, g 10%, r 11% -> 2.7 (CFA L2 R37 Example 26, Getinge).
    """
    s = _spread(required_return, growth)
    return {"trailing_ps": net_profit_margin * payout_ratio * (1 + growth) / s,
            "leading_ps": net_profit_margin * payout_ratio / s}


@formula(T, formula="V0 = FCFE0(1 + g)/(r - g); justified P/CF = V0/CF0; P/FCFE = V0/FCFE0",
         curriculum="L2 Equity R37 Market-Based Valuation (justified price to cash flow)")
def equity_justified_price_to_cash_flow(fcfe0: float, cash_flow0: float, required_return: float,
                                        growth: float) -> dict:
    """Justified P/CF and P/FCFE from a constant-growth FCFE value.

    Args:
        fcfe0: trailing FCFE per share.
        cash_flow0: trailing cash flow per share (e.g. earnings plus noncash charges).
        required_return, growth: decimals.

    Returns:
        value, justified_p_cf, justified_p_fcfe.

    Example:
        FCFE 1.34, CF 5.43, r 11.5%, g 8% -> value 41.35, P/CF 7.6, P/FCFE 30.9 (CFA L2 R37 Example 30).
    """
    v = fcfe0 * (1 + growth) / _spread(required_return, growth)
    return {"value": v, "justified_p_cf": v / cash_flow0, "justified_p_fcfe": v / fcfe0}


@formula(T, formula="PEG = (P/E) / (expected EPS growth in percent)",
         curriculum="L2 Equity R37 Market-Based Valuation (P/E-to-growth)")
def equity_peg_ratio(pe: float, growth: float) -> dict:
    """P/E-to-growth ratio; growth is a decimal and is converted to percentage points (0.13 -> 13).

    Args:
        pe: P/E (state whether forward or trailing).
        growth: expected EPS growth, decimal.

    Returns:
        peg.

    Example:
        forward P/E 10.2, growth 13% -> 0.78 (CFA L2 R37 Example 11).
    """
    if growth <= 0:
        raise ValueError("PEG is not meaningful for zero or negative growth")
    return {"peg": pe / (growth * 100)}


@formula(T, formula="EV = market cap + market value of debt + preferred + minority interest - cash & investments",
         curriculum="L1 R51 / L2 R37 (enterprise value and EV multiples)")
def equity_enterprise_value(market_cap: float | None = None, share_price: float | None = None,
                            shares: float | None = None, debt: float = 0.0, preferred: float = 0.0,
                            minority_interest: float = 0.0, cash_and_investments: float = 0.0,
                            ebitda: float | None = None, ebit: float | None = None,
                            sales: float | None = None) -> dict:
    """Enterprise value and EV/EBITDA, EV/EBIT, EV/Sales.

    Args:
        market_cap: equity market value (or share_price x shares).
        debt: market value of debt; preferred; minority_interest (noncontrolling interest).
        cash_and_investments: cash, equivalents and short-term investments.
        ebitda, ebit, sales: optional denominators.

    Returns:
        enterprise_value, ev_ebitda, ev_ebit, ev_sales (when denominators given).

    Example:
        cap 7,349,229, debt 2,487,839, cash 269,176, EBITDA 1,078,606 -> EV 9,567,892, EV/EBITDA 8.9
        (CFA L1 R51 Example 15, Cameco).
    """
    if market_cap is None:
        market_cap = _need(share_price, "market_cap or share_price") * _need(shares, "shares")
    ev = market_cap + debt + preferred + minority_interest - cash_and_investments
    out = {"enterprise_value": ev}
    for k, d in (("ev_ebitda", ebitda), ("ev_ebit", ebit), ("ev_sales", sales)):
        if d:
            out[k] = ev / d
    return out


@formula(T, formula="harmonic = n / sum(1/X_i); weighted harmonic = 1 / sum(w_i/X_i)",
         curriculum="L2 Equity R37 Market-Based Valuation (averaging multiples)")
def equity_multiple_harmonic_mean(multiples: list, weights: list | None = None) -> dict:
    """Arithmetic, weighted, harmonic and weighted harmonic means of price multiples (portfolio P/E).

    Args:
        multiples: positive multiples X_i (e.g. P/Es).
        weights: portfolio value weights (normalized to sum to 1); default equal.

    Returns:
        weighted_harmonic_mean, harmonic_mean, arithmetic_mean, weighted_mean, median.

    Example:
        P/Es 10 and 20, weights 55%/45% -> weighted harmonic 12.90 = portfolio P/E (CFA L2 R37 Exhibit 28).
    """
    if not multiples or any(x <= 0 for x in multiples):
        raise ValueError("multiples must be non-empty and positive")
    n = len(multiples)
    w = weights or [1.0] * n
    _same_len(multiples, w, names="multiples and weights")
    tot = sum(w)
    w = [x / tot for x in w]
    s = sorted(multiples)
    med = s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2
    return {"weighted_harmonic_mean": 1 / sum(wi / x for wi, x in zip(w, multiples)),
            "harmonic_mean": n / sum(1 / x for x in multiples), "arithmetic_mean": _mean(multiples),
            "weighted_mean": sum(wi * x for wi, x in zip(w, multiples)), "median": med}


@formula(T, formula="SUE = (EPS - E[EPS]) / sigma(EPS - E[EPS])",
         curriculum="L2 Equity R37 Market-Based Valuation (momentum: earnings surprise, SUE)")
def equity_earnings_surprise_sue(actual_eps: float, expected_eps: float, std_dev: float) -> dict:
    """Standardized unexpected earnings (or scaled surprise using the dispersion of analyst forecasts).

    Args:
        actual_eps: reported EPS.
        expected_eps: consensus/expected EPS.
        std_dev: std. dev. of past surprises (SUE) or of analysts' forecasts (scaled surprise).

    Returns:
        sue, unexpected_earnings, percent_surprise (relative to the forecast; sign flips if forecast < 0).

    Example:
        actual 28.95, consensus -6.30, std 8.344 -> SUE 4.225 (CFA L2 R37 Example 38, Sony).
    """
    if std_dev <= 0:
        raise ValueError("std_dev must be positive")
    ue = actual_eps - expected_eps
    return {"sue": ue / std_dev, "unexpected_earnings": ue,
            "percent_surprise": ue / expected_eps if expected_eps else float("nan")}


# ----------------------------------------------------------------------------------------- residual income
@formula(T, formula="RI = NI - r_e x B_{t-1} = (ROE - r_e) B_{t-1}",
         curriculum="L2 Equity R38 Residual Income Valuation (residual income)")
def equity_residual_income(net_income: float, beginning_book_equity: float, cost_of_equity: float) -> dict:
    """Residual income for one period: net income less the equity charge on beginning book equity.

    Args:
        net_income: net income (or EPS) for the period.
        beginning_book_equity: book value of equity (per share) at the start of the period.
        cost_of_equity: required return on equity, decimal.

    Returns:
        residual_income, equity_charge, roe.

    Example:
        NI 91,000, equity 1,000,000, r 12% -> RI -29,000 (CFA L2 R38 Example 1, Axis Manufacturing).
    """
    if beginning_book_equity == 0:
        raise ValueError("beginning_book_equity must be non-zero")
    ch = cost_of_equity * beginning_book_equity
    return {"residual_income": net_income - ch, "equity_charge": ch, "roe": net_income / beginning_book_equity}


@formula(T, formula="EVA = NOPAT - WACC x total capital; MVA = market value of the company - book value of capital",
         curriculum="L2 Equity R38 Residual Income Valuation (EVA and MVA)")
def equity_economic_value_added(nopat: float, total_capital: float, wacc: float,
                                market_value: float | None = None) -> dict:
    """Economic value added (capital-charge approach to residual income) and market value added.

    Args:
        nopat: net operating profit after taxes (EBIT(1 - t), plus EVA adjustments).
        total_capital: book (adjusted) total capital, debt + equity.
        wacc: cost of capital, decimal (after-tax debt cost).
        market_value: market value of debt and equity, for MVA.

    Returns:
        eva, capital_charge, return_on_capital, mva (if market_value).

    Example:
        NOPAT 140,000, capital 2,000,000, WACC 8.45% -> EVA -29,000 (CFA L2 R38 Example 1, capital approach).
    """
    ch = wacc * total_capital
    out = {"eva": nopat - ch, "capital_charge": ch, "return_on_capital": nopat / total_capital}
    if market_value is not None:
        out["mva"] = market_value - total_capital
    return out


@formula(T, formula="V0 = B0 + sum (E_t - r B_{t-1})/(1+r)^t + terminal; B_t = B_{t-1} + E_t - D_t (clean surplus); "
                    "persistence: last RI_T/((1 + r - w)(1+r)^{T-1})",
         curriculum="L2 Equity R38 Residual Income Valuation (general and multistage RI model)")
def equity_residual_income_valuation(book_value: float, required_return: float, eps: list | None = None,
                                     roe: list | None = None, dividends: list | None = None,
                                     payout_ratio: float | list | None = None, terminal: str = "none",
                                     persistence: float | None = None,
                                     terminal_premium: float | None = None) -> dict:
    """Multistage residual income value with clean-surplus book values and a choice of terminal assumption.

    Args:
        book_value: current book value per share B0.
        required_return: cost of equity, decimal.
        eps: EPS forecasts for t = 1..T (entries may be None where `roe` supplies the period).
        roe: ROE on beginning book value per period, used where eps is missing/None.
        dividends: dividends per period (entries may be None); otherwise payout_ratio x EPS.
        payout_ratio: scalar or per-period payout used where dividends are missing.
        terminal: "none" (RI = 0 after T), "perpetuity" (RI_T / r, no growth), "persistence" (Eq. 8: the
            last forecast RI fades at factor w) or "premium" (P_T - B_T given in terminal_premium).
        persistence: w in [0, 1] for terminal="persistence".
        terminal_premium: P_T - B_T for terminal="premium".

    Returns:
        value, book_value, pv_residual_income, pv_terminal, residual_income, book_values, eps.

    Example:
        B0 6.00, EPS [2.00, 2.50, 4.00], dividends [1.00, 1.25, 12.25], r 10% -> 11.15
        (CFA L2 R38 Example 3, Bugg Properties).
    """
    T_ = max(len(eps or []), len(roe or []))
    if T_ == 0:
        raise ValueError("give eps and/or roe forecasts")
    r = required_return
    bv, rows_e, rows_b, ri = book_value, [], [book_value], []
    for t in range(T_):
        e = eps[t] if eps and t < len(eps) and eps[t] is not None else None
        if e is None:
            if not roe or t >= len(roe) or roe[t] is None:
                raise ValueError(f"period {t + 1}: need eps or roe")
            e = roe[t] * bv
        d = dividends[t] if dividends and t < len(dividends) and dividends[t] is not None else None
        if d is None:
            if payout_ratio is None:
                raise ValueError(f"period {t + 1}: need a dividend or payout_ratio")
            p = payout_ratio[t] if isinstance(payout_ratio, (list, tuple)) else payout_ratio
            d = p * e
        ri.append(e - r * bv)
        bv = bv + e - d
        rows_e.append(e)
        rows_b.append(bv)
    pv = [x / (1 + r) ** t for t, x in enumerate(ri, start=1)]
    pv_term = 0.0
    if terminal == "perpetuity":
        pv_term = ri[-1] / r / (1 + r) ** T_
    elif terminal == "persistence":
        w = _need(persistence, "persistence")
        if not 0 <= w <= 1:
            raise ValueError("persistence must be between 0 and 1")
        pv_term = ri[-1] / ((1 + r - w) * (1 + r) ** (T_ - 1)) - pv[-1]  # replaces the last term (Eq. 8)
    elif terminal == "premium":
        pv_term = _need(terminal_premium, "terminal_premium") / (1 + r) ** T_
    elif terminal != "none":
        raise ValueError("terminal must be 'none', 'perpetuity', 'persistence' or 'premium'")
    return {"value": book_value + sum(pv) + pv_term, "book_value": book_value, "pv_residual_income": sum(pv),
            "pv_terminal": pv_term, "residual_income": ri, "book_values": rows_b, "eps": rows_e}


@formula(T, formula="V0 = B0 + (ROE - r) B0/(r - g); implied g = r - (ROE - r) B0/(P0 - B0)",
         curriculum="L2 Equity R38 Residual Income Valuation (single-stage RI, implied growth, justified P/B)")
def equity_residual_income_single_stage(book_value: float, roe: float, required_return: float,
                                        growth: float | None = None, price: float | None = None) -> dict:
    """Single-stage residual income value (constant ROE and growth) and/or the market-implied RI growth rate.

    Args:
        book_value: current book value per share.
        roe: long-run ROE, decimal.
        required_return: cost of equity, decimal.
        growth: constant growth of RI, decimal (needed for the value).
        price: market price (gives the implied growth rate).

    Returns:
        value, justified_pb (when growth given); implied_growth, market_pb (when price given).

    Example:
        B0 18.81, ROE 16%, r 11%, g 8% -> 50.16; at price 51.90 implied g 8.16% (CFA L2 R38 Examples 6-7, Canon).
    """
    out = {}
    if growth is not None:
        v = book_value + (roe - required_return) * book_value / _spread(required_return, growth)
        out.update(value=v, justified_pb=v / book_value)
    if price is not None:
        if price == book_value:
            raise ValueError("price equals book value: implied growth is undefined")
        out.update(implied_growth=required_return - (roe - required_return) * book_value / (price - book_value),
                   market_pb=price / book_value)
    if not out:
        raise ValueError("give growth (for value) and/or price (for implied growth)")
    return out


# ------------------------------------------------------------------------------------ private companies
@formula(T, formula="V = CF1/(r - g) = CF0(1 + g)/(r - g); equity = V - debt",
         curriculum="L2 Equity R39 Private Company Valuation (capitalized cash flow method)")
def private_company_capitalized_cash_flow(discount_rate: float, growth: float, cash_flow_next: float | None = None,
                                          cash_flow_current: float | None = None, debt: float = 0.0) -> dict:
    """Capitalized cash flow (single-stage) value of a private firm (FCFF at WACC) or its equity (FCFE at r).

    Args:
        discount_rate: WACC for FCFF or required return for FCFE, decimal.
        growth: long-term growth, decimal.
        cash_flow_next: normalized next-year cash flow; or
        cash_flow_current: normalized current cash flow (grown one year).
        debt: debt to subtract when capitalizing FCFF.

    Returns:
        value, capitalization_rate, equity_value.

    Example:
        FCFF 9,358,800, WACC 13.1%, g 3%, debt 2,000,000 -> equity 90,661,386 (CFA L2 R39 Example 4, Able).
    """
    if (cash_flow_next is None) == (cash_flow_current is None):
        raise ValueError("give exactly one of cash_flow_next or cash_flow_current")
    cf1 = cash_flow_next if cash_flow_next is not None else cash_flow_current * (1 + growth)
    cap = _spread(discount_rate, growth, rname="discount_rate")
    v = cf1 / cap
    return {"value": v, "capitalization_rate": cap, "equity_value": v - debt}


@formula(T, formula="RI = NE - r_wc WC - r_fa FA; intangibles = RI(1 + g)/(r_i - g); value = WC + FA + intangibles",
         curriculum="L2 Equity R39 Private Company Valuation (excess earnings method)")
def private_company_excess_earnings(normalized_earnings: float, working_capital: float, fixed_assets: float,
                                    return_on_working_capital: float, return_on_fixed_assets: float,
                                    intangibles_discount_rate: float, growth: float) -> dict:
    """Excess earnings method: capitalize earnings above the required returns on tangible assets.

    Args:
        normalized_earnings: normalized earnings for the year just ended.
        working_capital, fixed_assets: fair values.
        return_on_working_capital, return_on_fixed_assets: required returns, decimals.
        intangibles_discount_rate: required return on intangibles, decimal.
        growth: growth of the residual (excess) earnings, decimal.

    Returns:
        business_value, intangibles_value, excess_earnings, required_return_on_tangibles.

    Example:
        NE 100,000, WC 200,000 at 5%, FA 800,000 at 11%, r 12%, g 3% -> intangibles 22,889, value 1,022,889
        (CFA L2 R39 Section 4.2.4).
    """
    req = return_on_working_capital * working_capital + return_on_fixed_assets * fixed_assets
    ri = normalized_earnings - req
    intang = ri * (1 + growth) / _spread(intangibles_discount_rate, growth, rname="intangibles_discount_rate")
    return {"business_value": working_capital + fixed_assets + intang, "intangibles_value": intang,
            "excess_earnings": ri, "required_return_on_tangibles": req}


@formula(T, formula="DLOC = 1 - 1/(1 + control premium); total discount = 1 - (1 - DLOC)(1 - DLOM)",
         curriculum="L2 Equity R39 Private Company Valuation (valuation discounts and premiums)")
def private_company_valuation_discounts(dlom: float = 0.0, dloc: float | None = None,
                                        control_premium: float | None = None,
                                        pro_rata_value: float | None = None) -> dict:
    """Combine lack-of-control and lack-of-marketability discounts multiplicatively; DLOC from a control premium.

    Args:
        dlom: discount for lack of marketability, decimal.
        dloc: discount for lack of control, decimal (or derive it from control_premium).
        control_premium: control premium, decimal (DLOC = 1 - 1/(1 + CP)).
        pro_rata_value: pro rata controlling, marketable value of the interest (optional).

    Returns:
        total_discount, dloc, dlom, value_after_discounts (if pro_rata_value), marketable_minority_value.

    Example:
        control premium 15%, DLOM 25% -> DLOC 13.0%, total 34.8% (CFA L2 R39 practice problem 5).
    """
    if dloc is not None and control_premium is not None:
        raise ValueError("give dloc or control_premium, not both")
    if control_premium is not None:
        dloc = 1 - 1 / (1 + control_premium)
    dloc = dloc or 0.0
    if not 0 <= dloc < 1 or not 0 <= dlom < 1:
        raise ValueError("discounts must be decimals in [0, 1)")
    out = {"total_discount": 1 - (1 - dloc) * (1 - dlom), "dloc": dloc, "dlom": dlom}
    if pro_rata_value is not None:
        out.update(marketable_minority_value=pro_rata_value * (1 - dloc),
                   value_after_discounts=pro_rata_value * (1 - dloc) * (1 - dlom))
    return out


# --------------------------------------------------------------------------------- L3 equity portfolio mgmt
@formula(T, formula="active return = R_p - R_b; tracking risk = stdev(R_p - R_b); IR = mean active / tracking risk",
         curriculum="L3 Equity Portfolio Management R23 (active return, tracking risk, information ratio)")
def equity_active_return_risk(portfolio_returns: list, benchmark_returns: list, periods_per_year: int = 1) -> dict:
    """Mean active return, tracking risk (sample std of active returns) and information ratio, annualized.

    Args:
        portfolio_returns, benchmark_returns: periodic returns, decimals.
        periods_per_year: 12 for monthly data, etc. (mean x ppy, std x sqrt(ppy)).

    Returns:
        mean_active_return, tracking_risk, information_ratio, active_returns.

    Example:
        portfolio [0.05, 0.02, 0.08], benchmark [0.04, 0.03, 0.05] -> active [1%, -1%, 3%], TE 2%, IR 0.5.
    """
    _same_len(portfolio_returns, benchmark_returns, names="portfolio_returns and benchmark_returns")
    act = [p - b for p, b in zip(portfolio_returns, benchmark_returns)]
    m, s = _mean(act) * periods_per_year, _stdev(act) * math.sqrt(periods_per_year)
    return {"mean_active_return": m, "tracking_risk": s, "information_ratio": m / s if s else float("nan"),
            "active_returns": act}


@formula(T, formula="IR = TC x IC x sqrt(BR); expected active return = IR x active risk",
         curriculum="L3 Equity Portfolio Management R23 (Grinold-Kahn fundamental law of active management)")
def equity_fundamental_law(information_coefficient: float, breadth: float, transfer_coefficient: float = 1.0,
                           active_risk: float | None = None) -> dict:
    """Fundamental law of active management, optionally with a transfer coefficient (TC = 1 is the 2014 form).

    Args:
        information_coefficient: IC, correlation of forecasts with outcomes.
        breadth: number of independent active decisions per year.
        transfer_coefficient: TC in [0, 1] (constraint efficiency), default 1.
        active_risk: tracking risk, decimal, to translate IR into expected active return.

    Returns:
        information_ratio, expected_active_return (if active_risk).

    Example:
        IC 0.03, breadth 500 -> IR 0.67 (CFA L3 R23 Example 12, Manager A).
    """
    if breadth < 0:
        raise ValueError("breadth must be non-negative")
    ir = transfer_coefficient * information_coefficient * math.sqrt(breadth)
    out = {"information_ratio": ir}
    if active_risk is not None:
        out["expected_active_return"] = ir * active_risk
    return out


@formula(T, formula="alpha_P = sum w_i alpha_i; TE_P = sqrt(sum_i sum_j w_i w_j rho_ij TE_i TE_j); IR = alpha_P/TE_P",
         curriculum="L3 Equity Portfolio Management R23 (core-satellite / portfolio of managers)")
def equity_manager_portfolio_active_risk(weights: list, alphas: list, active_risks: list,
                                         correlations: list | None = None) -> dict:
    """Expected alpha, tracking risk and IR of a portfolio of managers (alphas uncorrelated unless given).

    Args:
        weights: allocation to each manager, decimals.
        alphas: expected active returns, decimals.
        active_risks: tracking risks, decimals.
        correlations: optional correlation matrix of active returns (default identity).

    Returns:
        portfolio_alpha, portfolio_active_risk, information_ratio.

    Example:
        weights 4/7,1/7,1/7,1/7, alphas 0/2/4/4%, TE 0/4/6/6% -> alpha 1.43%, TE 1.34%, IR 1.07
        (CFA L3 R23 Example 14).
    """
    n = _same_len(weights, alphas, active_risks, names="weights, alphas and active_risks")
    rho = correlations or [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]
    if len(rho) != n or any(len(row) != n for row in rho):
        raise ValueError("correlations must be an n x n matrix")
    a = sum(w * x for w, x in zip(weights, alphas))
    var = sum(weights[i] * weights[j] * rho[i][j] * active_risks[i] * active_risks[j]
              for i in range(n) for j in range(n))
    te = math.sqrt(var)
    return {"portfolio_alpha": a, "portfolio_active_risk": te, "information_ratio": a / te if te else float("nan")}


@formula(T, formula="true active = R_manager - R_normal; misfit = R_normal - R_investor; "
                    "total active risk^2 = true risk^2 + misfit risk^2; true IR = true active / true risk",
         curriculum="L3 Equity Portfolio Management R23 (true and misfit active return)")
def equity_manager_true_misfit(manager_return: float, normal_benchmark_return: float,
                               investor_benchmark_return: float, total_active_risk: float | None = None,
                               misfit_active_risk: float | None = None) -> dict:
    """Split a manager's total active return into 'true' and 'misfit' parts; true active risk and IR.

    Args:
        manager_return: manager's return, decimal.
        normal_benchmark_return: return on the manager's normal (style) benchmark.
        investor_benchmark_return: return on the investor's asset-class benchmark.
        total_active_risk: total active risk vs the investor's benchmark (optional).
        misfit_active_risk: std of misfit active return (optional, with total_active_risk).

    Returns:
        true_active_return, misfit_active_return, total_active_return, true_active_risk, true_information_ratio.

    Example:
        manager 12%, normal 15%, investor 10%, total risk 5.5%, misfit 4% -> true -3%, true risk 3.775%,
        true IR -0.7947 (CFA L3 R23 Section 7.1).
    """
    t, m = manager_return - normal_benchmark_return, normal_benchmark_return - investor_benchmark_return
    out = {"true_active_return": t, "misfit_active_return": m, "total_active_return": t + m}
    if total_active_risk is not None and misfit_active_risk is not None:
        if misfit_active_risk > total_active_risk:
            raise ValueError("misfit risk cannot exceed total active risk")
        tr = math.sqrt(total_active_risk ** 2 - misfit_active_risk ** 2)
        out.update(true_active_risk=tr, true_information_ratio=t / tr if tr else float("nan"))
    return out
