"""Trading cost measurement (CFA L3 Execution of Portfolio Decisions). Topic ``portfolio``."""
from __future__ import annotations

import numpy as np

from ._registry import formula
from .portfolio import _f, _vec

EXEC = "L3 PM: Execution of Portfolio Decisions"


def _side_sign(side: str) -> int:
    if side not in ("buy", "sell"):
        raise ValueError("side must be 'buy' or 'sell'")
    return 1 if side == "buy" else -1


@formula("portfolio", formula="IS = paper return - actual return = explicit + realized P/L + delay + missed trade; "
                              "delay = filled x (arrival - decision); realized = filled x (exec - arrival); "
                              "missed = unfilled x (cancel price - decision); all / (shares ordered x decision price)",
         curriculum=EXEC + " (implementation shortfall)")
def implementation_shortfall(decision_price: float, shares_ordered: float, fill_shares: list, fill_prices: list,
                             final_price: float, commissions: float = 0.0, fill_arrival_prices: list | None = None,
                             side: str = "buy", market_return: float | None = None, beta: float = 1.0) -> dict:
    """Implementation shortfall and its components versus a paper portfolio traded at the decision price.

    Args:
        decision_price: price when the decision was made (benchmark of the paper portfolio).
        shares_ordered: size of the intended trade.
        fill_shares: shares executed in each fill.
        fill_prices: execution price of each fill.
        final_price: price at the end of the measurement period (when the rest of the order is cancelled).
        commissions: total explicit costs (commissions, fees, taxes) in money.
        fill_arrival_prices: price when the order (or that day's order) was released to the market, per fill
            (e.g. prior close). Defaults to the decision price (no delay cost).
        side: "buy" or "sell".
        market_return: optional market return over the period, for the market-adjusted shortfall.
        beta: beta of the security for the market adjustment.

    Returns:
        shortfall (fraction of paper investment), shortfall_amount, explicit_cost, realized_pl, delay_cost,
        missed_trade_cost (all as fractions; *_amount in money), paper_gain, actual_gain,
        market_adjusted_shortfall (if market_return given).

    Example:
        Buy 1,000 decided at 10.00; 700 filled at 10.07 after the price moved to 10.05; close 10.08; commission 14:
        shortfall 0.0087 = explicit 0.0014 + realized 0.0014 + delay 0.0035 + missed 0.0024.
    """
    s = _side_sign(side)
    q = _vec(fill_shares, "fill_shares")
    px = _vec(fill_prices, "fill_prices")
    arr = np.full(q.size, decision_price) if fill_arrival_prices is None else _vec(fill_arrival_prices, "fill_arrival_prices")
    if not (q.size == px.size == arr.size):
        raise ValueError("fill_shares, fill_prices and fill_arrival_prices must have the same length")
    filled = float(q.sum())
    if filled > shares_ordered + 1e-9 or shares_ordered <= 0 or decision_price <= 0:
        raise ValueError("need shares_ordered > 0, decision_price > 0 and fills not exceeding the order")
    unfilled = shares_ordered - filled
    paper = shares_ordered * decision_price
    delay = s * float(q @ (arr - decision_price))
    realized = s * float(q @ (px - arr))
    missed = s * unfilled * (final_price - decision_price)
    total = commissions + delay + realized + missed
    paper_gain = s * shares_ordered * (final_price - decision_price)
    actual_gain = s * float(q @ (final_price - px)) - commissions
    out = {"shortfall": total / paper, "shortfall_amount": total, "explicit_cost": commissions / paper,
           "realized_pl": realized / paper, "delay_cost": delay / paper, "missed_trade_cost": missed / paper,
           "explicit_cost_amount": commissions, "realized_pl_amount": realized, "delay_cost_amount": delay,
           "missed_trade_cost_amount": missed, "paper_gain": paper_gain, "actual_gain": actual_gain,
           "filled_shares": filled, "unfilled_shares": unfilled}
    if market_return is not None:
        out["market_adjusted_shortfall"] = total / paper - s * beta * market_return
    return out


@formula("portfolio", formula="effective spread = 2 x (trade price - midpoint) for buys, 2 x (midpoint - trade price) for sells; "
                              "quoted spread = ask - bid", curriculum=EXEC + " (bid-ask spread)")
def effective_spread(trade_prices: list, bids: list, asks: list, sides: list, trade_sizes: list | None = None) -> dict:
    """Effective and quoted spreads for one or more trades, with simple and size-weighted averages.

    Args:
        trade_prices: execution prices.
        bids, asks: prevailing quotes at each trade.
        sides: "buy" or "sell" per trade.
        trade_sizes: optional, for the weighted average.

    Returns:
        effective_spreads, quoted_spreads, midpoints, average_effective_spread, weighted_average_effective_spread,
        average_quoted_spread, effective_spreads_pct (of midpoint).

    Example:
        effective_spread(trade_prices=[10.03], bids=[10.00], asks=[10.04], sides=["buy"]) -> effective 0.02, quoted 0.04
    """
    p = _vec(trade_prices, "trade_prices")
    b = _vec(bids, "bids")
    a = _vec(asks, "asks")
    if isinstance(sides, str):
        sides = [sides]
    sg = np.array([_side_sign(x) for x in sides], dtype=float)
    if not (p.size == b.size == a.size == sg.size):
        raise ValueError("trade_prices, bids, asks and sides must have the same length")
    mid = (a + b) / 2
    eff = 2 * sg * (p - mid)
    out = {"effective_spreads": _f(eff), "quoted_spreads": _f(a - b), "midpoints": _f(mid),
           "effective_spreads_pct": _f(eff / mid), "average_effective_spread": float(eff.mean()),
           "average_quoted_spread": float((a - b).mean())}
    if trade_sizes is not None:
        n = _vec(trade_sizes, "trade_sizes")
        if n.size != p.size:
            raise ValueError("trade_sizes must match trade_prices")
        out["weighted_average_effective_spread"] = float(n @ eff / n.sum())
    return out


@formula("portfolio", formula="VWAP = sum(P_i Q_i)/sum Q_i; cost vs VWAP = (avg exec price - VWAP)/VWAP for buys (sign flips for sells)",
         curriculum=EXEC + " (VWAP benchmark)")
def vwap_comparison(trade_prices: list, trade_sizes: list, side: str = "buy", market_vwap: float | None = None,
                    market_prices: list | None = None, market_volumes: list | None = None) -> dict:
    """Compare a trader's average execution price with the market's volume-weighted average price.

    Give `market_vwap` directly, or the day's `market_prices` and `market_volumes`.

    Returns:
        average_execution_price, vwap, cost (fraction of VWAP, positive = worse than VWAP), cost_amount.

    Example:
        vwap_comparison(trade_prices=[20.10, 20.20], trade_sizes=[100, 100], market_vwap=20.00) -> cost 0.0075
    """
    s = _side_sign(side)
    p = _vec(trade_prices, "trade_prices")
    q = _vec(trade_sizes, "trade_sizes")
    if p.size != q.size:
        raise ValueError("trade_prices and trade_sizes must have the same length")
    if market_vwap is None:
        if market_prices is None or market_volumes is None:
            raise ValueError("give market_vwap or market_prices + market_volumes")
        mp, mv = _vec(market_prices, "market_prices"), _vec(market_volumes, "market_volumes")
        market_vwap = float(mp @ mv / mv.sum())
    avg = float(p @ q / q.sum())
    return {"average_execution_price": avg, "vwap": market_vwap, "cost": s * (avg - market_vwap) / market_vwap,
            "cost_amount": s * (avg - market_vwap) * float(q.sum())}
