"""Derivatives II: option pricing (bounds, parity, binomial, BSM, Greeks, Black) and option strategies.

Discrete-time results (bounds, parity, one/two-period binomial) follow the 2014 curriculum and use
(1+r)^T by default; Black-Scholes-Merton and Black use continuously compounded rates.
"""
from __future__ import annotations

import math
from statistics import NormalDist

from scipy.optimize import brentq

from ._registry import formula
from .derivatives import _growth, _nonneg, _pos, _sign

R53 = "L2 Derivatives: Option Markets and Contracts (2014 R53; L1 2014 R60)"
R30 = "L3 Risk Management Applications of Option Strategies (2014 R30; L1 2014 R62)"

_N = NormalDist()
OPTION_TYPES = ("call", "put")


def _otype(option_type: str) -> str:
    if option_type not in OPTION_TYPES:
        raise ValueError(f"option_type must be 'call' or 'put', got {option_type!r}")
    return option_type


def _payoff(option_type: str, s: float, x: float) -> float:
    return max(0.0, s - x) if option_type == "call" else max(0.0, x - s)


# ---------------------------------------------------------------- basics, bounds, parity
@formula("derivatives", formula="Intrinsic: call max(0, S-X), put max(0, X-S); time value = price - intrinsic",
         curriculum=R53)
def option_intrinsic_time_value(spot: float, strike: float, option_type: str = "call",
                                option_price: float | None = None) -> dict:
    """Moneyness, intrinsic (exercise) value and time value of an option.

    Args:
        spot: underlying price. strike: exercise price. option_type: "call" or "put".
        option_price: market price; if given, time value = price - intrinsic.

    Returns:
        moneyness ("in-the-money", "at-the-money", "out-of-the-money"), intrinsic_value, time_value.

    Example:
        option_intrinsic_time_value(spot=62, strike=60, option_type="put", option_price=4)
        -> out-of-the-money, intrinsic 0, time value 4 (L3 R30 Example 2)
    """
    _otype(option_type)
    iv = _payoff(option_type, spot, strike)
    if abs(spot - strike) <= 1e-12 * max(1.0, abs(strike)):
        m = "at-the-money"
    else:
        m = "in-the-money" if iv > 0 else "out-of-the-money"
    return {"moneyness": m, "intrinsic_value": iv,
            "time_value": None if option_price is None else option_price - iv}


@formula("derivatives",
         formula="c, C >= max(0, S - PV(X)); c, C <= S; p >= max(0, PV(X) - S), p <= PV(X); P >= max(0, X - S), P <= X",
         curriculum=R53)
def option_bounds(spot: float, strike: float, rate: float, years: float, pv_income: float = 0.0,
                  compounding: str = "discrete") -> dict:
    """Lower and upper no-arbitrage bounds for European and American calls and puts.

    Args:
        spot: underlying price. strike: exercise price. rate: risk-free rate. years: time to expiration.
        pv_income: PV of cash flows on the underlying over the option's life (subtracted from spot).
        compounding: "discrete" (curriculum, (1+r)^T) or "continuous".

    Returns:
        european_call_lower/upper, american_call_lower/upper, european_put_lower/upper,
        american_put_lower/upper, pv_strike.

    Example:
        option_bounds(spot=72, strike=70, rate=0.045, years=42/365) -> european_call_lower 2.35
        (L2 R53 Example 2)
    """
    _pos("spot", spot)
    _pos("strike", strike)
    _nonneg("years", years)
    pvx = strike / _growth(rate, years, compounding)
    s_adj = spot - pv_income
    return {"european_call_lower": max(0.0, s_adj - pvx), "european_call_upper": spot,
            "american_call_lower": max(0.0, s_adj - pvx, spot - strike), "american_call_upper": spot,
            "european_put_lower": max(0.0, pvx - s_adj), "european_put_upper": pvx,
            "american_put_lower": max(0.0, pvx - s_adj, strike - spot), "american_put_upper": strike,
            "pv_strike": pvx}


@formula("derivatives", formula="c + X/(1+r)^T = p + S - PV(income)  (fiduciary call = protective put)",
         curriculum=R53)
def put_call_parity(strike: float, rate: float, years: float, call: float | None = None, put: float | None = None,
                    spot: float | None = None, pv_income: float = 0.0, compounding: str = "discrete") -> dict:
    """Put-call parity for European options: solve for the missing leg, or flag a mispricing.

    Leave exactly one of call, put, spot as None to solve for it. If all three are given, the actual call
    is compared with the synthetic call (long put + long underlying + borrow PV(X)).

    Args:
        strike, rate, years: option terms. call, put, spot: prices (one may be None).
        pv_income: PV of dividends on the underlying during the option life.
        compounding: "discrete" (curriculum) or "continuous".

    Returns:
        call, put, spot, synthetic_call, synthetic_put, pv_strike, and when all were given: mispricing
        (actual call - synthetic call) and strategy.

    Example:
        put_call_parity(strike=45, rate=0.045, years=115/365, call=8.00, put=3.75, spot=48)
        -> synthetic_call 7.37, mispricing 0.63 (L2 R53 Example 3)
    """
    missing = [k for k, v in (("call", call), ("put", put), ("spot", spot)) if v is None]
    if len(missing) > 1:
        raise ValueError("give at least two of call, put, spot")
    pvx = strike / _growth(rate, years, compounding)
    if call is None:
        call = put + spot - pv_income - pvx
    elif put is None:
        put = call + pvx - spot + pv_income
    elif spot is None:
        spot = call + pvx - put + pv_income
    syn_call = put + spot - pv_income - pvx
    syn_put = call + pvx - spot + pv_income
    out = {"call": call, "put": put, "spot": spot, "synthetic_call": syn_call, "synthetic_put": syn_put,
           "pv_strike": pvx}
    if not missing:
        diff = call - syn_call
        if abs(diff) < 1e-12:
            strat = "none: parity holds"
        elif diff > 0:
            strat = "call rich: sell call; buy put, buy underlying, borrow PV(X)"
        else:
            strat = "call cheap: buy call; sell put, short underlying, lend PV(X)"
        out.update({"mispricing": diff, "strategy": strat})
    return out


@formula("derivatives", formula="c - p = (F - X)/(1+r)^T  (put-call-forward parity)", curriculum=R53)
def put_call_forward_parity(strike: float, rate: float, years: float, call: float | None = None,
                            put: float | None = None, forward_price: float | None = None,
                            compounding: str = "discrete") -> dict:
    """Put-call-forward parity: solve for the missing call, put or forward price, or flag a mispricing.

    Args:
        strike, rate, years: terms (options and forward expire together). call, put, forward_price: one may be None.
        compounding: "discrete" (curriculum) or "continuous".

    Returns:
        call, put, forward_price, synthetic_forward_value (c - p + (X - F)/(1+r)^T; zero when priced right),
        synthetic_put (c + (X - F)/(1+r)^T), pv_strike_minus_forward.

    Example:
        put_call_forward_parity(strike=90, rate=0.05, years=2, call=15.25, put=3.00, forward_price=101.43)
        -> synthetic_put 4.88 (L2 R53 Example 10)
    """
    missing = [k for k, v in (("call", call), ("put", put), ("forward_price", forward_price)) if v is None]
    if len(missing) > 1:
        raise ValueError("give at least two of call, put, forward_price")
    g = _growth(rate, years, compounding)
    if call is None:
        call = put + (forward_price - strike) / g
    elif put is None:
        put = call - (forward_price - strike) / g
    elif forward_price is None:
        forward_price = strike + (call - put) * g
    pv_diff = (strike - forward_price) / g
    return {"call": call, "put": put, "forward_price": forward_price,
            "synthetic_forward_value": call - put + pv_diff, "synthetic_put": call + pv_diff,
            "pv_strike_minus_forward": pv_diff}


# ---------------------------------------------------------------- binomial
def _binomial_engine(spot: float, strike: float, up: float, down: float, prob: float, disc: float, n: int,
                     option_type: str, american: bool) -> dict:
    """Recombining tree; node j at step i has j down moves. Returns trees (top node first)."""
    stock = [[spot * up ** (i - j) * down ** j for j in range(i + 1)] for i in range(n + 1)]
    value = [[0.0] * (i + 1) for i in range(n + 1)]
    value[n] = [_payoff(option_type, s, strike) for s in stock[n]]
    early = []
    for i in range(n - 1, -1, -1):
        for j in range(i + 1):
            cont = disc * (prob * value[i + 1][j] + (1 - prob) * value[i + 1][j + 1])
            ex = _payoff(option_type, stock[i][j], strike)
            if american and ex > cont + 1e-12:
                value[i][j] = ex
                early.append([i, j])
            else:
                value[i][j] = cont
    hedge = [[(value[i + 1][j] - value[i + 1][j + 1]) / (stock[i + 1][j] - stock[i + 1][j + 1])
              for j in range(i + 1)] for i in range(n)]
    return {"stock": stock, "value": value, "hedge": hedge, "early": sorted(early)}


@formula("derivatives",
         formula="pi = (1 + r - d)/(u - d); c = [pi c+ + (1-pi) c-]/(1+r); n = (c+ - c-)/(S+ - S-); c = nS - B",
         curriculum=R53)
def binomial_option_price(spot: float, strike: float, up: float, down: float, rate_per_period: float,
                          periods: int = 1, option_type: str = "call", american: bool = False) -> dict:
    """Binomial option price with given up/down factors (one- or multi-period), hedge ratios and replication.

    Args:
        spot: underlying price. strike: exercise price.
        up, down: gross factors (u = 1.25 for +25%, d = 0.80 for -20%); need d < 1 + r < u.
        rate_per_period: risk-free rate per binomial period (discrete).
        periods: number of periods. option_type: "call" or "put".
        american: True checks early exercise at every node (value = max(continuation, exercise)).

    Returns:
        price, risk_neutral_prob, hedge_ratio (units of underlying per option at t=0; negative for puts),
        bond_position (B in price = n S - B: positive = borrow, negative = lend), underlying_tree,
        option_tree, hedge_ratio_tree, early_exercise_nodes ([period, down_moves]).

    Example:
        binomial_option_price(spot=65, strike=70, up=1.30, down=0.78, rate_per_period=0.08)
        -> price 7.75, pi 0.5769, hedge_ratio 0.4290 (L2 R53 Example 4)
    """
    _otype(option_type)
    _pos("spot", spot)
    if not isinstance(periods, int) or periods < 1:
        raise ValueError("periods must be a positive integer")
    if not 0 < down < 1 + rate_per_period < up:
        raise ValueError("need 0 < down < 1 + rate_per_period < up (otherwise arbitrage)")
    pi = (1 + rate_per_period - down) / (up - down)
    t = _binomial_engine(spot, strike, up, down, pi, 1 / (1 + rate_per_period), periods, option_type, american)
    price = t["value"][0][0]
    n0 = t["hedge"][0][0]
    return {"price": price, "risk_neutral_prob": pi, "hedge_ratio": n0, "bond_position": n0 * spot - price,
            "underlying_tree": t["stock"], "option_tree": t["value"], "hedge_ratio_tree": t["hedge"],
            "early_exercise_nodes": t["early"]}


@formula("derivatives",
         formula="CRR: u = e^(sigma sqrt(dt)), d = 1/u, p = (e^((r-q)dt) - d)/(u - d), discount e^(-r dt)",
         curriculum=R53 + " (Section 6.6, Exhibit 23); Hull, Options, Futures and Other Derivatives ch. 13/21")
def binomial_crr_option_price(spot: float, strike: float, rate: float, volatility: float, years: float,
                              steps: int, option_type: str = "call", american: bool = False,
                              dividend_yield: float = 0.0) -> dict:
    """Cox-Ross-Rubinstein binomial price for N steps (continuous rates, optional dividend yield).

    Converges to Black-Scholes-Merton for European options; use american=True for early exercise
    (American puts, or calls on dividend-paying assets).

    Args:
        spot, strike: prices. rate: continuously compounded risk-free rate. volatility: annual sigma.
        years: time to expiration. steps: number of binomial steps. option_type: "call" or "put".
        american: early exercise allowed. dividend_yield: continuous dividend (or foreign-rate) yield.

    Returns:
        price, up, down, risk_neutral_prob, hedge_ratio (t=0 delta), early_exercise_node_count.

    Example:
        binomial_crr_option_price(spot=52.75, strike=50, rate=0.0488, volatility=0.35, years=0.75, steps=1)
        -> 10.0259; steps=1000 -> 8.6190 (L2 R53 Exhibit 23)
    """
    _otype(option_type)
    _pos("volatility", volatility)
    _pos("years", years)
    if not isinstance(steps, int) or steps < 1:
        raise ValueError("steps must be a positive integer")
    dt = years / steps
    u = math.exp(volatility * math.sqrt(dt))
    d = 1 / u
    p = (math.exp((rate - dividend_yield) * dt) - d) / (u - d)
    if not 0 < p < 1:
        raise ValueError("risk-neutral probability outside (0, 1); use more steps")
    t = _binomial_engine(spot, strike, u, d, p, math.exp(-rate * dt), steps, option_type, american)
    return {"price": t["value"][0][0], "up": u, "down": d, "risk_neutral_prob": p,
            "hedge_ratio": t["hedge"][0][0], "early_exercise_node_count": len(t["early"])}


@formula("derivatives",
         formula="Backward induction on a one-period rate tree, prob 0.5: V = [0.5 V_up + 0.5 V_down]/(1 + r_node); rate payoff max(0, r - X)/(1 + r) paid one period later",
         curriculum=R53 + " (Section 6.4, Example 7)")
def binomial_interest_rate_option(rate_tree: list, exercise: float, option_type: str = "call",
                                  underlying: str = "rate", underlying_values: list | None = None,
                                  notional: float = 1.0, prob_up: float = 0.5) -> dict:
    """European option on an interest rate (caplet/floorlet) or on a bond, priced on a binomial rate tree.

    The option expires at the last level of rate_tree. For underlying="rate" the payoff
    max(0, r - X) (call) or max(0, X - r) (put) is paid one period after expiration, so it is discounted at
    the node's one-period rate. For underlying="price" pass the bond prices at expiration in
    underlying_values. Sum caplets (or floorlets) of different expirations to price a cap (floor).

    Args:
        rate_tree: one-period rates by level, [[r0], [r_up, r_down], [r_uu, r_ud, r_dd], ...].
        exercise: exercise rate (underlying="rate") or exercise price (underlying="price").
        option_type: "call" or "put". underlying: "rate" or "price".
        underlying_values: bond prices at expiration nodes (top first) for underlying="price".
        notional: multiplies the result. prob_up: risk-neutral probability (0.5 in the curriculum trees).

    Returns:
        price, value_tree (per 1 notional).

    Example:
        binomial_interest_rate_option(rate_tree=[[0.1051], [0.1304, 0.1025], [0.1578, 0.1180, 0.0795]],
                                      exercise=0.105) -> 0.0138 (L2 R53 Exhibit 22, two-period caplet)
    """
    _otype(option_type)
    if underlying not in ("rate", "price"):
        raise ValueError("underlying must be 'rate' or 'price'")
    n = len(rate_tree) - 1
    if n < 0 or any(len(level) != i + 1 for i, level in enumerate(rate_tree)):
        raise ValueError("rate_tree level i must hold i + 1 rates")
    last = rate_tree[n]
    if underlying == "rate":
        values = [_payoff(option_type, r, exercise) / (1 + r) for r in last]
    else:
        if underlying_values is None or len(underlying_values) != n + 1:
            raise ValueError(f"underlying_values needs {n + 1} prices at expiration")
        values = [_payoff(option_type, v, exercise) for v in underlying_values]
    tree = [values]
    for i in range(n - 1, -1, -1):
        nxt = tree[0]
        tree.insert(0, [(prob_up * nxt[j] + (1 - prob_up) * nxt[j + 1]) / (1 + rate_tree[i][j])
                        for j in range(i + 1)])
    return {"price": tree[0][0] * notional, "value_tree": tree}


# ---------------------------------------------------------------- Black-Scholes-Merton, Greeks, Black
def _bsm_d(s_adj: float, strike: float, rate: float, vol: float, years: float) -> tuple:
    _pos("spot (adjusted)", s_adj)
    _pos("strike", strike)
    _pos("volatility", vol)
    _pos("years", years)
    srt = vol * math.sqrt(years)
    d1 = (math.log(s_adj / strike) + (rate + vol * vol / 2) * years) / srt
    return d1, d1 - srt


@formula("derivatives",
         formula="c = S e^(-qT) N(d1) - X e^(-rT) N(d2); p = X e^(-rT) N(-d2) - S e^(-qT) N(-d1); d1 = [ln(S e^(-qT)/X) + (r + s^2/2)T]/(s sqrt T)",
         curriculum=R53)
def bsm_option_price(spot: float, strike: float, rate: float, volatility: float, years: float,
                     dividend_yield: float = 0.0, pv_income: float = 0.0) -> dict:
    """Black-Scholes-Merton European call and put prices (with dividend yield or PV of cash flows).

    For currency options set dividend_yield to the continuously compounded foreign rate.

    Args:
        spot: underlying price. strike: exercise price.
        rate: continuously compounded risk-free rate (r^c = ln(1 + r) from an annual rate).
        volatility: annual standard deviation of continuously compounded returns. years: time to expiration.
        dividend_yield: continuous dividend yield (or foreign rate). pv_income: PV of discrete cash flows.

    Returns:
        call_price, put_price, d1, d2, n_d1, n_d2 (N(d1), N(d2)), adjusted_spot, pv_strike.

    Example:
        bsm_option_price(spot=42, strike=40, rate=0.10, volatility=0.20, years=0.5) -> call 4.76, put 0.81
        (Hull, Options, Futures and Other Derivatives, BSM example)
    """
    s_adj = (spot - pv_income) * math.exp(-dividend_yield * years)
    d1, d2 = _bsm_d(s_adj, strike, rate, volatility, years)
    pvx = strike * math.exp(-rate * years)
    return {"call_price": s_adj * _N.cdf(d1) - pvx * _N.cdf(d2),
            "put_price": pvx * _N.cdf(-d2) - s_adj * _N.cdf(-d1),
            "d1": d1, "d2": d2, "n_d1": _N.cdf(d1), "n_d2": _N.cdf(d2), "adjusted_spot": s_adj, "pv_strike": pvx}


@formula("derivatives",
         formula="delta_c = e^(-qT)N(d1), delta_p = e^(-qT)(N(d1)-1); gamma = e^(-qT)n(d1)/(S s sqrt T); vega = S e^(-qT) n(d1) sqrt T; theta, rho (Hull)",
         curriculum=R53 + "; L3 2014 R30 Section 4; Hull ch. 19")
def bsm_greeks(spot: float, strike: float, rate: float, volatility: float, years: float,
               dividend_yield: float = 0.0, option_type: str = "call") -> dict:
    """BSM Greeks: delta, gamma, theta, vega, rho (per unit of the input, plus per-day/per-1% versions).

    Args:
        spot, strike, rate (continuous), volatility, years, dividend_yield: as in bsm_option_price.
        option_type: "call" or "put".

    Returns:
        delta, gamma, theta (per year), theta_per_day (/365), vega (per 1.00 of vol), vega_per_1pct,
        rho (per 1.00 of rate), rho_per_1pct, price.

    Example:
        bsm_greeks(spot=49, strike=50, rate=0.05, volatility=0.20, years=0.3846) -> delta 0.522, gamma 0.066,
        theta -4.31, vega 12.1, rho 8.91 (Hull, Greek letters example)
    """
    _otype(option_type)
    q = dividend_yield
    eq = math.exp(-q * years)
    d1, d2 = _bsm_d(spot * eq, strike, rate, volatility, years)
    pdf = _N.pdf(d1)
    er = math.exp(-rate * years)
    gamma = eq * pdf / (spot * volatility * math.sqrt(years))
    vega = spot * eq * pdf * math.sqrt(years)
    decay = -spot * pdf * volatility * eq / (2 * math.sqrt(years))
    if option_type == "call":
        delta = eq * _N.cdf(d1)
        theta = decay + q * spot * _N.cdf(d1) * eq - rate * strike * er * _N.cdf(d2)
        rho = strike * years * er * _N.cdf(d2)
        price = spot * eq * _N.cdf(d1) - strike * er * _N.cdf(d2)
    else:
        delta = eq * (_N.cdf(d1) - 1)
        theta = decay - q * spot * _N.cdf(-d1) * eq + rate * strike * er * _N.cdf(-d2)
        rho = -strike * years * er * _N.cdf(-d2)
        price = strike * er * _N.cdf(-d2) - spot * eq * _N.cdf(-d1)
    return {"delta": delta, "gamma": gamma, "theta": theta, "theta_per_day": theta / 365, "vega": vega,
            "vega_per_1pct": vega / 100, "rho": rho, "rho_per_1pct": rho / 100, "price": price}


@formula("derivatives", formula="Solve BSM(sigma) = market price for sigma (root finding)", curriculum=R53)
def bsm_implied_volatility(option_price: float, spot: float, strike: float, rate: float, years: float,
                           dividend_yield: float = 0.0, option_type: str = "call") -> dict:
    """Implied volatility: the sigma that makes the BSM price equal the market price.

    Args:
        option_price: observed European option price. spot, strike, rate (continuous), years, dividend_yield:
            as in bsm_option_price. option_type: "call" or "put".

    Returns:
        implied_volatility, model_price (at that volatility).

    Example:
        bsm_implied_volatility(option_price=9.25, spot=52.75, strike=50, rate=0.0488, years=0.75) -> 0.3882
        (L2 R53 Section 7.5.2)
    """
    key = f"{_otype(option_type)}_price"

    def f(sig: float) -> float:
        return bsm_option_price(spot=spot, strike=strike, rate=rate, volatility=sig, years=years,
                                dividend_yield=dividend_yield)[key] - option_price

    lo, hi = 1e-6, 10.0
    if f(lo) > 0 or f(hi) < 0:
        raise ValueError("option_price is outside the BSM no-arbitrage range for these inputs")
    sig = brentq(f, lo, hi, xtol=1e-12)
    return {"implied_volatility": sig, "model_price": f(sig) + option_price}


@formula("derivatives",
         formula="c = e^(-rT)[F N(d1) - X N(d2)]; p = e^(-rT)[X N(-d2) - F N(-d1)]; d1 = [ln(F/X) + s^2 T/2]/(s sqrt T)",
         curriculum=R53 + " (Section 8.3)")
def black_futures_option_price(futures_price: float, strike: float, rate: float, volatility: float,
                               years: float) -> dict:
    """Black model for European options on futures or forwards.

    Args:
        futures_price: futures (or forward) price. strike: exercise price.
        rate: continuously compounded risk-free rate. volatility: volatility of the futures price. years: T.

    Returns:
        call_price, put_price, d1, d2, n_d1, n_d2.

    Example:
        black_futures_option_price(futures_price=139.19, strike=125, rate=0.0425, volatility=0.15,
                                   years=215/365) -> call 15.23
        (curriculum prints 15.06 using two-decimal N(d) tables; exact 15.23. L2 R53 Example 11)
    """
    _pos("futures_price", futures_price)
    d1, d2 = _bsm_d(futures_price, strike, 0.0, volatility, years)
    disc = math.exp(-rate * years)
    return {"call_price": disc * (futures_price * _N.cdf(d1) - strike * _N.cdf(d2)),
            "put_price": disc * (strike * _N.cdf(-d2) - futures_price * _N.cdf(-d1)),
            "d1": d1, "d2": d2, "n_d1": _N.cdf(d1), "n_d2": _N.cdf(d2)}


@formula("derivatives",
         formula="Price = Black(F_rate, X, r, s, T) x e^(-F m/365) x (m/360) x notional  (caplet/floorlet, payoff paid m days after expiry)",
         curriculum=R53 + " (Section 8.4)")
def black_interest_rate_option(forward_rate: float, strike_rate: float, rate: float, volatility: float,
                               years: float, days_underlying: float, notional: float = 1.0,
                               option_type: str = "call", day_basis: float = 360,
                               days_in_year: float = 365) -> dict:
    """Black model for an interest rate call (caplet) or put (floorlet), 2014 curriculum procedure.

    Steps: Black value with the forward rate as the underlying; discount for the payment lag at the forward
    rate (e^(-F m/365)); convert the annual rate to a periodic price (x m/360); scale by notional.

    Args:
        forward_rate: forward rate for the underlying period. strike_rate: exercise rate.
        rate: continuously compounded risk-free rate to option expiration. volatility: forward-rate volatility.
        years: time to option expiration. days_underlying: m (days in the underlying rate).
        notional: notional principal. option_type: "call" (caplet) or "put" (floorlet).
        day_basis: 360 for the rate accrual. days_in_year: 365 for the lag discount.

    Returns:
        price (times notional), price_per_unit, black_value (annual-rate units), lag_discount, d1, d2.

    Example:
        black_interest_rate_option(forward_rate=0.068, strike_rate=0.07, rate=0.0625, volatility=0.02,
            years=280/365, days_underlying=180, notional=10e6, option_type="put") -> 9,331
        (curriculum prints about 9,084 using table-rounded N(d); exact 9,331. L2 R53 Example 12)
    """
    _otype(option_type)
    b = black_futures_option_price(futures_price=forward_rate, strike=strike_rate, rate=rate,
                                   volatility=volatility, years=years)
    base = b[f"{option_type}_price"]
    lag = math.exp(-forward_rate * days_underlying / days_in_year)
    unit = base * lag * days_underlying / day_basis
    return {"price": unit * notional, "price_per_unit": unit, "black_value": base, "lag_discount": lag,
            "d1": b["d1"], "d2": b["d2"]}


@formula("derivatives", formula="Cap = sum of caplets; floor = sum of floorlets (each priced with the Black model)",
         curriculum=R53 + " (Section 8.4)")
def black_cap_floor_price(forward_rates: list, expiries: list, discount_rates: list, volatilities,
                          strike_rate: float, days_underlying: float, notional: float = 1.0,
                          cap_or_floor: str = "cap") -> dict:
    """Interest rate cap or floor as the sum of Black-model caplets or floorlets.

    Args:
        forward_rates: forward rate for each caplet period. expiries: years to each caplet expiration.
        discount_rates: continuously compounded risk-free rate to each expiration.
        volatilities: one volatility for all caplets, or a list.
        strike_rate: cap or floor rate. days_underlying: days in each period (m).
        notional: notional principal. cap_or_floor: "cap" or "floor".

    Returns:
        price, caplet_prices (or floorlet prices).

    Example:
        black_cap_floor_price(forward_rates=[0.0925], expiries=[90/365], discount_rates=[0.096],
                              volatilities=0.03, strike_rate=0.09, days_underlying=90, notional=1e6)
        -> 600.81 (curriculum prints about 607 using table-rounded N(d); exact 600.81. L2 R53 Section 8.4
        first caplet)
    """
    if cap_or_floor not in ("cap", "floor"):
        raise ValueError("cap_or_floor must be 'cap' or 'floor'")
    n = len(forward_rates)
    if not n or len(expiries) != n or len(discount_rates) != n:
        raise ValueError("forward_rates, expiries and discount_rates need the same non-zero length")
    vols = list(volatilities) if isinstance(volatilities, (list, tuple)) else [volatilities] * n
    if len(vols) != n:
        raise ValueError("volatilities must be one number or one per caplet")
    otype = "call" if cap_or_floor == "cap" else "put"
    parts = [black_interest_rate_option(forward_rate=f, strike_rate=strike_rate, rate=r, volatility=v, years=t,
                                        days_underlying=days_underlying, notional=notional,
                                        option_type=otype)["price"]
             for f, t, r, v in zip(forward_rates, expiries, discount_rates, vols)]
    return {"price": sum(parts), "caplet_prices": parts}


@formula("derivatives",
         formula="Call payoff = NP max(0, L - X)(days/360); put = NP max(0, X - L)(days/360); paid days after expiry",
         curriculum=R53 + " (Section 4.1.4)")
def interest_rate_option_payoff(notional: float, underlying_rate: float, exercise_rate: float,
                                days_underlying: float, option_type: str = "call", day_basis: float = 360) -> dict:
    """Payoff of an interest rate call or put (paid in arrears, at the end of the underlying period).

    Args:
        notional: notional principal. underlying_rate: LIBOR at expiration. exercise_rate: strike rate.
        days_underlying: days in the underlying rate. option_type: "call" or "put". day_basis: 360.

    Returns:
        payoff (paid days_underlying days after expiration), pv_at_expiration (discounted at underlying_rate).

    Example:
        interest_rate_option_payoff(notional=10e6, underlying_rate=0.06, exercise_rate=0.055,
                                    days_underlying=180) -> 25,000 (L2 R53 Section 4.1.4)
    """
    _otype(option_type)
    acc = days_underlying / day_basis
    pay = notional * _payoff(option_type, underlying_rate, exercise_rate) * acc
    return {"payoff": pay, "pv_at_expiration": pay / (1 + underlying_rate * acc)}


# ---------------------------------------------------------------- strategies at expiration
def _strategy(legs: list, s_t: float, qty_u: float = 0.0, s_0: float = 0.0) -> dict:
    """Piecewise-linear payoff engine: value, profit, max gain/loss and breakevens at expiration."""
    clean = []
    for leg in legs:
        t = _otype(leg.get("type", ""))
        clean.append((t, float(leg["strike"]), float(leg.get("premium", 0.0)), float(leg.get("quantity", 1.0))))
    cost = qty_u * s_0 + sum(q * p for _, _, p, q in clean)

    def profit(s: float) -> float:
        return qty_u * s + sum(q * _payoff(t, s, x) for t, x, _, q in clean) - cost

    kinks = sorted({0.0, *(x for _, x, _, _ in clean)})
    slope_up = qty_u + sum(q for t, _, _, q in clean if t == "call")
    vals = [profit(k) for k in kinks]
    max_gain = math.inf if slope_up > 1e-12 else max(vals)
    max_loss = math.inf if slope_up < -1e-12 else -min(vals)
    roots = []
    for (a, pa), (b, pb) in zip(zip(kinks, vals), zip(kinks[1:], vals[1:])):
        if abs(pa) < 1e-12 and abs(pb) < 1e-12:
            continue  # flat at zero
        if abs(pa) < 1e-12:
            roots.append(a)
        elif pa * pb < 0:
            roots.append(a + (-pa) * (b - a) / (pb - pa))
    k, pk = kinks[-1], vals[-1]
    if abs(pk) < 1e-12 and abs(slope_up) > 1e-12:
        roots.append(k)
    elif abs(slope_up) > 1e-12 and k - pk / slope_up > k:
        roots.append(k - pk / slope_up)
    breakevens = []
    for r in sorted(roots):
        if not breakevens or abs(r - breakevens[-1]) > 1e-9:
            breakevens.append(r)
    v_t = profit(s_t) + cost
    return {"value_at_expiration": v_t, "profit": v_t - cost, "initial_cost": cost, "max_gain": max_gain,
            "max_loss": max_loss, "breakevens": breakevens}


@formula("derivatives",
         formula="Value at T = q_u S_T + sum q_i payoff_i(S_T); profit = value - initial cost; max/min over kinks; breakevens = roots",
         curriculum=R30)
def option_strategy_expiration(legs: list, spot_at_expiry: float, underlying_quantity: float = 0.0,
                               underlying_price: float = 0.0) -> dict:
    """Any option strategy at expiration: value, profit, maximum gain and loss, breakeven prices.

    Args:
        legs: [{"type": "call"|"put", "strike": X, "premium": c0, "quantity": +1 long / -1 short}, ...].
        spot_at_expiry: underlying price at expiration, S_T.
        underlying_quantity: units of the underlying held (+) or shorted (-). underlying_price: S0 paid.

    Returns:
        value_at_expiration, profit, initial_cost (net outlay; negative = net credit), max_gain,
        max_loss (positive = loss; inf = unlimited), breakevens.

    Example:
        option_strategy_expiration(legs=[{"type": "call", "strike": 50, "premium": 6.25},
            {"type": "put", "strike": 50, "premium": 5.875}], spot_at_expiry=61) -> profit -1.125,
        breakevens [37.875, 62.125] (L3 R30 Example 9, straddle)
    """
    if not legs and not underlying_quantity:
        raise ValueError("give at least one option leg or an underlying position")
    return _strategy(legs, spot_at_expiry, underlying_quantity, underlying_price)


@formula("derivatives", formula="Long call: max(0, S_T - X) - c0; long put: max(0, X - S_T) - p0; short = negative",
         curriculum=R30 + "; L1 2014 R60")
def option_payoff_profit(option_type: str, strike: float, premium: float, spot_at_expiry: float,
                         position: str = "long") -> dict:
    """Payoff and profit of a single long or short call or put at expiration.

    Args:
        option_type: "call" or "put". strike: exercise price. premium: price paid/received.
        spot_at_expiry: S_T. position: "long" or "short".

    Returns:
        value_at_expiration, profit, initial_cost (negative = net credit), max_gain,
        max_loss (positive = loss; inf = unlimited), breakevens.

    Example:
        option_payoff_profit(option_type="call", strike=100, premium=7, spot_at_expiry=102) -> value 2,
        profit -5, breakeven 107 (L3 R30 Example 1)
    """
    q = _sign(position)
    return _strategy([{"type": option_type, "strike": strike, "premium": premium, "quantity": q}], spot_at_expiry)


@formula("derivatives", formula="Covered call = long S + short call: V_T = S_T - max(0, S_T - X); max gain X - S0 + c0",
         curriculum=R30)
def covered_call(stock_price: float, call_strike: float, call_premium: float, spot_at_expiry: float) -> dict:
    """Covered call (long underlying, short call) at expiration.

    Args:
        stock_price: S0 paid for the underlying. call_strike, call_premium: call written. spot_at_expiry: S_T.

    Returns:
        value_at_expiration, profit, initial_cost (negative = net credit), max_gain,
        max_loss (positive = loss; inf = unlimited), breakevens.

    Example:
        covered_call(stock_price=98, call_strike=105, call_premium=8, spot_at_expiry=110) -> profit 15,
        max loss 90, breakeven 90 (L3 R30 Example 3)
    """
    return _strategy([{"type": "call", "strike": call_strike, "premium": call_premium, "quantity": -1}],
                     spot_at_expiry, 1.0, stock_price)


@formula("derivatives", formula="Protective put = long S + long put: V_T = S_T + max(0, X - S_T); max loss S0 + p0 - X",
         curriculum=R30)
def protective_put(stock_price: float, put_strike: float, put_premium: float, spot_at_expiry: float) -> dict:
    """Protective put (long underlying, long put) at expiration.

    Args:
        stock_price: S0. put_strike, put_premium: put bought. spot_at_expiry: S_T.

    Returns:
        value_at_expiration, profit, initial_cost (negative = net credit), max_gain,
        max_loss (positive = loss; inf = unlimited), breakevens.

    Example:
        protective_put(stock_price=0.875, put_strike=0.90, put_premium=0.075, spot_at_expiry=0.75)
        -> profit -0.05, max loss 0.05, breakeven 0.95 (L3 R30 Example 4)
    """
    return _strategy([{"type": "put", "strike": put_strike, "premium": put_premium, "quantity": 1}],
                     spot_at_expiry, 1.0, stock_price)


@formula("derivatives",
         formula="Bull spread: long X1, short X2 (X1 < X2), calls or puts; max gain (calls) X2 - X1 - (c1 - c2)",
         curriculum=R30)
def bull_spread(low_strike: float, low_premium: float, high_strike: float, high_premium: float,
                spot_at_expiry: float, option_type: str = "call") -> dict:
    """Bull spread: buy the low-strike option, sell the high-strike option (calls: debit; puts: credit).

    Args:
        low_strike, low_premium: option bought. high_strike, high_premium: option sold.
        spot_at_expiry: S_T. option_type: "call" or "put".

    Returns:
        value_at_expiration, profit, initial_cost (negative = net credit), max_gain,
        max_loss (positive = loss; inf = unlimited), breakevens.

    Example:
        bull_spread(low_strike=65, low_premium=9, high_strike=75, high_premium=4, spot_at_expiry=78)
        -> profit 5, max loss 5, breakeven 70 (L3 R30 Example 5)
    """
    if not low_strike < high_strike:
        raise ValueError("need low_strike < high_strike")
    return _strategy([{"type": option_type, "strike": low_strike, "premium": low_premium, "quantity": 1},
                      {"type": option_type, "strike": high_strike, "premium": high_premium, "quantity": -1}],
                     spot_at_expiry)


@formula("derivatives",
         formula="Bear spread: long X2, short X1 (X1 < X2), puts or calls; max gain (puts) X2 - X1 - (p2 - p1)",
         curriculum=R30)
def bear_spread(low_strike: float, low_premium: float, high_strike: float, high_premium: float,
                spot_at_expiry: float, option_type: str = "put") -> dict:
    """Bear spread: buy the high-strike option, sell the low-strike option (puts: debit; calls: credit).

    Args:
        low_strike, low_premium: option sold. high_strike, high_premium: option bought.
        spot_at_expiry: S_T. option_type: "put" or "call".

    Returns:
        value_at_expiration, profit, initial_cost (negative = net credit), max_gain,
        max_loss (positive = loss; inf = unlimited), breakevens.

    Example:
        bear_spread(low_strike=85, low_premium=3, high_strike=95, high_premium=11, spot_at_expiry=82)
        -> profit 2, max loss 8, breakeven 87 (L3 R30 Example 6)
    """
    if not low_strike < high_strike:
        raise ValueError("need low_strike < high_strike")
    return _strategy([{"type": option_type, "strike": high_strike, "premium": high_premium, "quantity": 1},
                      {"type": option_type, "strike": low_strike, "premium": low_premium, "quantity": -1}],
                     spot_at_expiry)


@formula("derivatives", formula="Butterfly: long X1, short 2 X2, long X3 (calls or puts); max gain X2 - X1 - net premium",
         curriculum=R30)
def butterfly_spread(low_strike: float, low_premium: float, mid_strike: float, mid_premium: float,
                     high_strike: float, high_premium: float, spot_at_expiry: float, option_type: str = "call",
                     position: str = "long") -> dict:
    """Butterfly spread: long one low, short two middle, long one high strike (short = the reverse).

    Args:
        low/mid/high strikes and premiums (X1 < X2 < X3). spot_at_expiry: S_T.
        option_type: "call" or "put". position: "long" or "short".

    Returns:
        value_at_expiration, profit, initial_cost (negative = net credit), max_gain,
        max_loss (positive = loss; inf = unlimited), breakevens.

    Example:
        butterfly_spread(low_strike=1.30, low_premium=0.08, mid_strike=1.40, mid_premium=0.125,
                         high_strike=1.50, high_premium=0.18, spot_at_expiry=1.35, option_type="put")
        -> profit 0.04, max gain 0.09, breakevens 1.31 and 1.49 (L3 R30 Example 7)
    """
    if not low_strike < mid_strike < high_strike:
        raise ValueError("need low_strike < mid_strike < high_strike")
    q = _sign(position)
    return _strategy([{"type": option_type, "strike": low_strike, "premium": low_premium, "quantity": q},
                      {"type": option_type, "strike": mid_strike, "premium": mid_premium, "quantity": -2 * q},
                      {"type": option_type, "strike": high_strike, "premium": high_premium, "quantity": q}],
                     spot_at_expiry)


@formula("derivatives", formula="Collar = long S + long put X1 + short call X2: V_T = S_T + max(0, X1 - S_T) - max(0, S_T - X2)",
         curriculum=R30)
def collar(stock_price: float, put_strike: float, put_premium: float, call_strike: float, call_premium: float,
           spot_at_expiry: float) -> dict:
    """Equity collar: long underlying, long put (floor), short call (cap); zero-cost when premiums match.

    Args:
        stock_price: S0. put_strike, put_premium: put bought. call_strike, call_premium: call sold.
        spot_at_expiry: S_T.

    Returns:
        value_at_expiration, profit, initial_cost (negative = net credit), max_gain,
        max_loss (positive = loss; inf = unlimited), breakevens.

    Example:
        collar(stock_price=42, put_strike=40, put_premium=5.32, call_strike=50.59, call_premium=5.32,
               spot_at_expiry=55) -> value 50.59, profit 8.59, max loss 2, breakeven 42 (L3 R30 Example 8)
    """
    if not put_strike < call_strike:
        raise ValueError("need put_strike < call_strike")
    return _strategy([{"type": "put", "strike": put_strike, "premium": put_premium, "quantity": 1},
                      {"type": "call", "strike": call_strike, "premium": call_premium, "quantity": -1}],
                     spot_at_expiry, 1.0, stock_price)


@formula("derivatives", formula="Straddle = call + put at the same X; breakevens X +/- (c0 + p0)", curriculum=R30)
def straddle(strike: float, call_premium: float, put_premium: float, spot_at_expiry: float,
             position: str = "long") -> dict:
    """Long (or short) straddle: call and put with the same strike and expiration.

    Args:
        strike: common exercise price. call_premium, put_premium: prices. spot_at_expiry: S_T.
        position: "long" (bet on volatility) or "short".

    Returns:
        value_at_expiration, profit, initial_cost (negative = net credit), max_gain,
        max_loss (positive = loss; inf = unlimited), breakevens.

    Example:
        straddle(strike=50, call_premium=6.25, put_premium=5.875, spot_at_expiry=37) -> profit 0.875,
        max loss 12.125, breakevens 37.875 and 62.125 (L3 R30 Example 9)
    """
    q = _sign(position)
    return _strategy([{"type": "call", "strike": strike, "premium": call_premium, "quantity": q},
                      {"type": "put", "strike": strike, "premium": put_premium, "quantity": q}], spot_at_expiry)


@formula("derivatives", formula="Strangle = put X1 + call X2 (X1 < X2); breakevens X1 - premiums, X2 + premiums",
         curriculum=R30)
def strangle(put_strike: float, put_premium: float, call_strike: float, call_premium: float,
             spot_at_expiry: float, position: str = "long") -> dict:
    """Long (or short) strangle: out-of-the-money put and call with different strikes.

    Args:
        put_strike, put_premium, call_strike, call_premium: the two options (put_strike < call_strike).
        spot_at_expiry: S_T. position: "long" or "short".

    Returns:
        value_at_expiration, profit, initial_cost (negative = net credit), max_gain,
        max_loss (positive = loss; inf = unlimited), breakevens.

    Example:
        strangle(put_strike=1950, put_premium=56.01, call_strike=2050, call_premium=59.98, spot_at_expiry=2200)
        -> profit 34.01, breakevens 1834.01 and 2165.99 (premiums from L3 R30 Section 2)
    """
    if not put_strike < call_strike:
        raise ValueError("need put_strike < call_strike")
    q = _sign(position)
    return _strategy([{"type": "put", "strike": put_strike, "premium": put_premium, "quantity": q},
                      {"type": "call", "strike": call_strike, "premium": call_premium, "quantity": q}],
                     spot_at_expiry)


@formula("derivatives",
         formula="Box = bull call spread + bear put spread: value at T = X2 - X1; fair cost = (X2 - X1)/(1+r)^T",
         curriculum=R30)
def box_spread(low_strike: float, high_strike: float, call_low_premium: float, call_high_premium: float,
               put_low_premium: float, put_high_premium: float, rate: float, years: float,
               compounding: str = "discrete") -> dict:
    """Box spread: riskless payoff X2 - X1; compares its cost with the PV of the payoff (arbitrage check).

    Args:
        low_strike, high_strike: X1 < X2. call_low/high_premium, put_low/high_premium: option prices.
        rate: risk-free rate. years: time to expiration. compounding: "discrete" or "continuous".

    Returns:
        value_at_expiration (X2 - X1), initial_cost (c1 - c2 + p2 - p1), profit, pv_payoff,
        arbitrage_gain (pv_payoff - cost; > 0 buy the box, < 0 sell it), strategy.

    Example:
        box_spread(low_strike=75, high_strike=85, call_low_premium=16.02, call_high_premium=12.28,
                   put_low_premium=9.72, put_high_premium=15.18, rate=0.0513, years=0.5)
        -> cost 9.20, pv 9.75, gain 0.55 (L3 R30 Example 10)
    """
    if not low_strike < high_strike:
        raise ValueError("need low_strike < high_strike")
    cost = call_low_premium - call_high_premium + put_high_premium - put_low_premium
    width = high_strike - low_strike
    pv = width / _growth(rate, years, compounding)
    gain = pv - cost
    strat = "buy the box (underpriced)" if gain > 1e-12 else ("sell the box (overpriced)" if gain < -1e-12 else
                                                              "fairly priced")
    return {"value_at_expiration": width, "initial_cost": cost, "profit": width - cost, "pv_payoff": pv,
            "arbitrage_gain": gain, "strategy": strat}


@formula("derivatives", formula="Find X2 such that BSM call(X2) = BSM put(X1)  (zero-cost collar)", curriculum=R30)
def zero_cost_collar_call_strike(spot: float, put_strike: float, rate: float, volatility: float, years: float,
                                 dividend_yield: float = 0.0) -> dict:
    """Call strike that makes a collar zero-cost: the call premium equals the BSM put premium.

    Args:
        spot: underlying price. put_strike: strike of the protective put.
        rate (continuous), volatility, years, dividend_yield: BSM inputs.

    Returns:
        call_strike, premium (common put and call price), put_price.

    Example:
        zero_cost_collar_call_strike(spot=2000, put_strike=1950, rate=0.02, volatility=0.35, years=1/12,
                                     dividend_yield=0.005) -> about 2060 (L3 R30 Section 2.4.1)
    """
    bs = dict(spot=spot, rate=rate, volatility=volatility, years=years, dividend_yield=dividend_yield)
    put = bsm_option_price(strike=put_strike, **bs)["put_price"]

    def f(x: float) -> float:
        return bsm_option_price(strike=x, **bs)["call_price"] - put

    lo, hi = spot * 1e-6, spot * 100
    if f(lo) < 0:
        raise ValueError("put premium exceeds any call premium; no zero-cost collar")
    x2 = brentq(f, lo, hi, xtol=1e-10)
    return {"call_strike": x2, "premium": put, "put_price": put}
