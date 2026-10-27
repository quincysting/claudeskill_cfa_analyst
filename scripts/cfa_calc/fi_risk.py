"""Fixed income: interest-rate risk — duration, convexity, money duration, key rates, duration gaps (L1-L3)."""
from __future__ import annotations

from ._registry import formula
from .fi_pricing import L1_RISK, _level_cfs, _periods, _pv, _rate_ok

L3_PM = "L3 FI: Fixed-Income Portfolio Management Parts I-II (2014 R20, R22)"


@formula("fixed_income", formula="MacDur = sum t w_t ; ModDur = MacDur/(1+r) ; Conv = sum t(t+1)PV_t/((1+r)^2 P) "
         "; ApproxModDur = (PV- - PV+)/(2 dy PV0) ; ApproxCon = (PV- + PV+ - 2PV0)/(dy^2 PV0)", curriculum=L1_RISK)
def bond_duration_convexity(coupon_rate: float, ytm: float, years: float, freq: int = 2, fraction: float = 0.0,
                            face: float = 100.0, dy: float = 0.0001) -> dict:
    """Macaulay, modified, approximate and money duration, PVBP and convexity of an option-free bond.

    All outputs are annualized (durations in years, convexity in years^2) and based on the full price.

    Args:
        coupon_rate: annual coupon. ytm: annual yield (periodicity freq).
        years: years to maturity from the last coupon date (N = years*freq periods).
        freq: coupons per year. fraction: t/T elapsed in the current period. face: par.
        dy: yield bump for the approximate (finite-difference) measures, decimal (0.0005 = 5 bp).

    Returns:
        macaulay_duration, modified_duration, convexity, approx_modified_duration, approx_macaulay_duration,
        approx_convexity, money_duration, money_convexity, pvbp, full_price, pv_plus, pv_minus.

    Example:
        10y 8% annual at 10.40% -> Macaulay 7.0029, modified 6.3432
    """
    if not 0 <= fraction < 1:
        raise ValueError("fraction must be in [0, 1)")
    if dy <= 0:
        raise ValueError("dy must be positive")
    n = _periods(years, freq)
    cfs = _level_cfs(face, coupon_rate, n, freq)
    r = ytm / freq
    _rate_ok(r)
    pvs = [cf / (1 + r) ** (k - fraction) for k, cf in enumerate(cfs, 1)]
    p = sum(pvs)
    ts = [k - fraction for k in range(1, n + 1)]
    mac = sum(t * v for t, v in zip(ts, pvs)) / p / freq
    mod = mac / (1 + r)
    conv = sum(t * (t + 1) * v for t, v in zip(ts, pvs)) / ((1 + r) ** 2 * p) / freq ** 2
    pp, pm = _pv(cfs, (ytm + dy) / freq, fraction), _pv(cfs, (ytm - dy) / freq, fraction)
    amod = (pm - pp) / (2 * dy * p)
    bp = 0.0001
    pvbp = (_pv(cfs, (ytm - bp) / freq, fraction) - _pv(cfs, (ytm + bp) / freq, fraction)) / 2
    return {"macaulay_duration": mac, "modified_duration": mod, "convexity": conv,
            "approx_modified_duration": amod, "approx_macaulay_duration": amod * (1 + r),
            "approx_convexity": (pm + pp - 2 * p) / (dy ** 2 * p), "money_duration": mod * p,
            "money_convexity": conv * p, "pvbp": pvbp, "full_price": p, "pv_plus": pp, "pv_minus": pm}


@formula("fixed_income", formula="Dur = (PV- - PV+)/(2 dy PV0) ; Con = (PV- + PV+ - 2 PV0)/(dy^2 PV0)",
         curriculum=L1_RISK)
def fi_approx_duration_convexity(pv0: float, pv_minus: float, pv_plus: float, delta: float) -> dict:
    """Approximate (yield) or effective (curve) duration and convexity from three prices.

    Use the bond's own yield shift for approximate modified duration/convexity; use a benchmark-curve
    shift from a pricing model (tree, OAS model) for effective duration/convexity of bonds with options.

    Args:
        pv0: initial full price. pv_minus: price when yield/curve falls by delta. pv_plus: when it rises.
        delta: the shift, decimal (0.0025 = 25 bp).

    Returns:
        duration, convexity, price_value_of_shift (average absolute price change).

    Example:
        callable bond PV0 101.060489, PV+ 99.050120, PV- 102.890738, 25 bp -> EffDur 7.6006, EffCon -285.17
    """
    if pv0 <= 0 or delta <= 0:
        raise ValueError("pv0 and delta must be positive")
    return {"duration": (pv_minus - pv_plus) / (2 * delta * pv0),
            "convexity": (pv_minus + pv_plus - 2 * pv0) / (delta ** 2 * pv0),
            "price_value_of_shift": (pv_minus - pv_plus) / 2}


@formula("fixed_income", formula="%dP ~ -ModDur x dy + 1/2 x Convexity x dy^2 ; money: x full price",
         curriculum=L1_RISK)
def bond_price_change_estimate(modified_duration: float, yield_change: float, convexity: float = 0.0,
                               price: float | None = None) -> dict:
    """Estimated percentage (and money) price change from duration and the convexity adjustment.

    Works for yield, benchmark-curve or spread changes (use effective or spread duration accordingly).

    Args:
        modified_duration: annual modified (or effective/spread) duration.
        yield_change: change in annual yield, decimal (+0.01 = +100 bp).
        convexity: annual convexity (curriculum scaling, used with 1/2; Bloomberg's /100 figure must be x100).
        price: optional full price or position value for the money change.

    Returns:
        pct_change, duration_effect, convexity_effect, money_change (None if no price).

    Example:
        ModDur 29.498, convexity 884.7, yield -10 bp -> +2.9940%
    """
    d = -modified_duration * yield_change
    c = 0.5 * convexity * yield_change ** 2
    return {"pct_change": d + c, "duration_effect": d, "convexity_effect": c,
            "money_change": None if price is None else (d + c) * price}


def _weights(n: int, market_values, weights) -> list:
    if weights is None:
        if market_values is None or len(market_values) != n:
            raise ValueError("give market_values (or weights) matching durations")
        tot = sum(market_values)
        if tot == 0:
            raise ValueError("total market value is zero")
        return [v / tot for v in market_values]
    if len(weights) != n:
        raise ValueError("weights must match durations")
    return list(weights)


@formula("fixed_income", formula="D_p = sum w_i D_i ; C_p = sum w_i C_i (w = market-value weights)", curriculum=L1_RISK)
def bond_portfolio_duration(durations: list, market_values: list | None = None, weights: list | None = None,
                            convexities: list | None = None) -> dict:
    """Market-value-weighted portfolio duration (modified, effective, spread or key-rate) and convexity.

    Spread duration: give Treasury holdings a spread duration of 0.

    Args:
        durations: duration of each holding. market_values or weights: position sizes.
        convexities: optional convexity of each holding.

    Returns:
        duration, convexity (None if not given), weights, contributions.

    Example:
        1y and 30y zeros 50/50, ModDur 0.980 / 27.765 -> 14.3725
    """
    w = _weights(len(durations), market_values, weights)
    contrib = [wi * d for wi, d in zip(w, durations)]
    conv = None if convexities is None else sum(wi * c for wi, c in zip(w, convexities))
    return {"duration": sum(contrib), "convexity": conv, "weights": w, "contributions": contrib}


@formula("fixed_income", formula="MoneyDur = D x V ; DD(100bp) = D x V x 0.01 ; PVBP = D x V x 0.0001",
         curriculum=L1_RISK + "; " + L3_PM)
def bond_money_duration(duration: float, value: float) -> dict:
    """Money (dollar) duration, 100 bp dollar duration and PVBP/DV01 of a position.

    Args:
        duration: annual modified (or effective) duration. value: full market value of the position.

    Returns:
        money_duration (per 1.00 yield change), dollar_duration_100bp, pvbp.

    Example:
        Mac 2.4988, ytm 5.2617% sa, full 99.650 -> ModDur 2.4339, money duration 242.62, PVBP 0.024262
    """
    md = duration * value
    return {"money_duration": md, "dollar_duration_100bp": md * 0.01, "pvbp": md * 0.0001}


@formula("fixed_income", formula="ratio = target DD / current DD ; cash = (ratio - 1) x market value",
         curriculum=L3_PM)
def dollar_duration_rebalance(target_dollar_duration: float, current_dollar_duration: float,
                              market_value: float) -> dict:
    """Rebalance a portfolio pro rata to restore its target dollar duration.

    Args:
        target_dollar_duration: desired dollar duration. current_dollar_duration: after the yield shift/time.
        market_value: current portfolio market value.

    Returns:
        rebalancing_ratio, pct_change_each_position, cash_required (negative = cash released).

    Example:
        111,945 / 82,579 -> ratio 1.356, add ~35.6% to each position
    """
    if current_dollar_duration == 0:
        raise ValueError("current_dollar_duration must be nonzero")
    ratio = target_dollar_duration / current_dollar_duration
    return {"rebalancing_ratio": ratio, "pct_change_each_position": ratio - 1,
            "cash_required": (ratio - 1) * market_value}


@formula("fixed_income", formula="new MV = (D_old x MV_old) / D_new ; par = new MV / (price/100)", curriculum=L3_PM)
def duration_neutral_swap(old_market_value: float, old_duration: float, new_duration: float,
                          new_price: float) -> dict:
    """Size a bond swap so that the portfolio's dollar duration is unchanged.

    Args:
        old_market_value: value of the position sold. old_duration: its duration.
        new_duration: duration of the bond bought. new_price: its price per 100 par.

    Returns:
        new_market_value, new_par_value, dollar_duration (per 100 bp, = D x MV / 100).

    Example:
        sell 5.5m at D=4, buy D=5 at 90 -> buy MV 4.4m, par 4.889m
    """
    if new_duration <= 0 or new_price <= 0:
        raise ValueError("new_duration and new_price must be positive")
    dd = old_market_value * old_duration / 100
    mv = dd / new_duration * 100
    return {"new_market_value": mv, "new_par_value": mv / (new_price / 100), "dollar_duration": dd}


@formula("fixed_income", formula="Duration gap = Macaulay duration - investment horizon", curriculum=L1_RISK)
def duration_gap_horizon(macaulay_duration: float, horizon_years: float) -> dict:
    """Bond duration gap versus the investor's horizon and which rate move hurts.

    Args:
        macaulay_duration: annual Macaulay duration. horizon_years: investment horizon.

    Returns:
        duration_gap, risk (text: which rate move is the risk).

    Example:
        MacDur 8.1390, horizon 10 -> gap -1.861 (reinvestment risk dominates: lower rates hurt)
    """
    gap = macaulay_duration - horizon_years
    risk = ("market price risk dominates: higher rates hurt" if gap > 0 else
            "reinvestment risk dominates: lower rates hurt" if gap < 0 else "immunized: risks offset")
    return {"duration_gap": gap, "risk": risk}


@formula("fixed_income", formula="D_E = (D_A A - D_L L)/E ; dSurplus ~ -(D_A A - D_L L) dy ; gap = D_A - D_L L/A",
         curriculum=L3_PM)
def asset_liability_duration_gap(assets: float, liabilities: float, duration_assets: float,
                                 duration_liabilities: float, yield_change: float = 0.0) -> dict:
    """Equity (surplus) duration, duration gap and surplus change for an asset-liability or leveraged position.

    Args:
        assets, liabilities: market/present values. duration_assets, duration_liabilities: modified durations.
        yield_change: parallel rate change for the surplus estimate, decimal.

    Returns:
        surplus, equity_duration, duration_gap, assets_new, liabilities_new, surplus_new, surplus_change.

    Example:
        A 140 (D 4), L 100 (D 1) -> equity duration 11.5
    """
    e = assets - liabilities
    if e == 0:
        raise ValueError("assets equal liabilities: equity duration undefined")
    da, dl = -duration_assets * assets * yield_change, -duration_liabilities * liabilities * yield_change
    return {"surplus": e, "equity_duration": (duration_assets * assets - duration_liabilities * liabilities) / e,
            "duration_gap": duration_assets - duration_liabilities * liabilities / assets,
            "assets_new": assets + da, "liabilities_new": liabilities + dl, "surplus_new": e + da - dl,
            "surplus_change": da - dl}


@formula("fixed_income", formula="KRD_k = -(P(z_k+dz) - P(z_k-dz))/(2 dz P) ; sum KRD = effective (spot) duration",
         curriculum="L2/L3 FI: key rate durations (2014 L2 R46 §8)")
def key_rate_durations_spot(spot_rates: list, coupon_rate: float = 0.0, freq: int = 1, face: float = 100.0,
                            shift: float = 0.0001, cash_flows: list | None = None) -> dict:
    """Key rate (partial) durations of a bond to each point of its spot curve.

    Args:
        spot_rates: spot rates for periods 1..N (each period is a key rate). coupon_rate: annual coupon.
        freq: periods per year. face: par. shift: bump size, decimal. cash_flows: optional explicit cash flows.

    Returns:
        key_rate_durations (one per period), effective_duration (their sum), price.

    Example:
        a zero-coupon bond has all its key rate duration at its maturity point
    """
    cfs = list(cash_flows) if cash_flows is not None else _level_cfs(face, coupon_rate, len(spot_rates), freq)
    if len(cfs) != len(spot_rates):
        raise ValueError("cash_flows and spot_rates must have the same length")

    def price(zs) -> float:
        return sum(cf / (1 + z / freq) ** k for k, (cf, z) in enumerate(zip(cfs, zs), 1))

    p0 = price(spot_rates)
    krd = []
    for i in range(len(spot_rates)):
        up = [z + shift if j == i else z for j, z in enumerate(spot_rates)]
        dn = [z - shift if j == i else z for j, z in enumerate(spot_rates)]
        krd.append((price(dn) - price(up)) / (2 * shift * p0))
    return {"key_rate_durations": krd, "effective_duration": sum(krd), "price": p0}


@formula("fixed_income", formula="%dP ~ -sum_k KRD_k x dy_k ; portfolio KRD = sum w_i KRD_ik", curriculum=L3_PM)
def key_rate_price_change(key_rate_durations: list, rate_changes: list, weights: list | None = None) -> dict:
    """Portfolio key rate durations and the estimated % price change for a non-parallel curve shift.

    Args:
        key_rate_durations: list of KRD vectors, one per holding (or a single vector for one bond/portfolio).
        rate_changes: change at each key rate, decimals (same length as each KRD vector).
        weights: market-value weights of the holdings (required when more than one vector).

    Returns:
        portfolio_key_rate_durations, effective_duration, pct_change, contributions.

    Example:
        50% 2y zero + 50% 30y zero (KRDs 2 and 30), 2y +10 bp, 30y -10 bp -> +1.4%
    """
    vecs = key_rate_durations if key_rate_durations and isinstance(key_rate_durations[0], list) \
        else [key_rate_durations]
    w = [1.0] if len(vecs) == 1 and weights is None else _weights(len(vecs), None, weights)
    k = len(rate_changes)
    if any(len(v) != k for v in vecs):
        raise ValueError("each KRD vector must have one entry per rate change")
    port = [sum(wi * v[j] for wi, v in zip(w, vecs)) for j in range(k)]
    contrib = [-d * dy for d, dy in zip(port, rate_changes)]
    return {"portfolio_key_rate_durations": port, "effective_duration": sum(port), "pct_change": sum(contrib),
            "contributions": contrib}


@formula("fixed_income", formula="Duration contribution = local duration x country beta x weight", curriculum=L3_PM)
def foreign_bond_duration(local_duration: float, country_beta: float, weight: float = 1.0) -> dict:
    """Duration of a foreign bond measured against domestic rate changes (country beta adjustment).

    Args:
        local_duration: duration in local-yield terms. country_beta: d(foreign yield)/d(domestic yield).
        weight: portfolio weight of the bond.

    Returns:
        adjusted_duration, portfolio_contribution.

    Example:
        German bond D=6, beta 0.42 -> 2.52 for a UK portfolio
    """
    adj = local_duration * country_beta
    return {"adjusted_duration": adj, "portfolio_contribution": adj * weight}
