"""Fixed income portfolio management (L3): immunization, contingent immunization, futures duration management,
expected-return decomposition, breakeven spreads, leverage and repo, currency hedging, tracking risk."""
from __future__ import annotations

import statistics

from ._registry import formula
from .fi_pricing import _pv_dur_conv, _rate_ok

L3_PM = "L3 FI: Fixed-Income Portfolio Management Parts I-II (2014 R20, R22)"
L3_RET = "L3 FI: Overview of Fixed-Income Portfolio Management (expected return decomposition)"


@formula("fixed_income", formula="PV = L/(1+y/m)^(mH) ; immunize: PV_A = PV_L, MacDur_A = H, convexity_A minimal >= C_L",
         curriculum=L3_PM)
def immunization_single_liability(liability_amount: float, horizon_years: float, ytm: float,
                                  freq: int = 2) -> dict:
    """Targets for immunizing a single liability: required assets, duration and the liability's convexity.

    Args:
        liability_amount: amount due at the horizon. horizon_years: years to the liability.
        ytm: portfolio yield (annual, periodicity freq). freq: compounding periods per year.

    Returns:
        required_pv, target_macaulay_duration, target_modified_duration, liability_convexity,
        money_duration, bpv (money duration x 0.0001).

    Example:
        GIC paying 13,934,413 in 5y at 7.5% sa -> required 9,642,899
    """
    if horizon_years <= 0:
        raise ValueError("horizon_years must be positive")
    m = _pv_dur_conv([liability_amount], [horizon_years], ytm, freq)
    md = m["modified_duration"] * m["pv"]
    return {"required_pv": m["pv"], "target_macaulay_duration": m["macaulay_duration"],
            "target_modified_duration": m["modified_duration"], "liability_convexity": m["convexity"],
            "money_duration": md, "bpv": md * 1e-4}


@formula("fixed_income", formula="conditions: PV_A >= PV_L ; D_A = D_L ; dispersion/convexity_A > liabilities'",
         curriculum=L3_PM)
def immunization_multiple_liabilities(liability_flows: list, liability_times: list, asset_flows: list,
                                      asset_times: list, ytm: float, freq: int = 2,
                                      duration_tolerance: float = 0.05) -> dict:
    """Check the multiple-liability immunization conditions for an asset cash-flow portfolio.

    Args:
        liability_flows / liability_times: liability amounts and times (years).
        asset_flows / asset_times: portfolio cash flows and times (years).
        ytm: discount yield (annual, periodicity freq). freq: periodicity.
        duration_tolerance: |D_A - D_L| allowed, years.

    Returns:
        assets / liabilities (pv, macaulay_duration, modified_duration, convexity, dispersion), surplus,
        pv_condition, duration_condition, convexity_condition, immunized, money_duration_gap.

    Example:
        liabilities 50 at 3y and 50 at 7y vs a 1y/10y barbell with matched PV and duration -> immunized
    """
    a = _pv_dur_conv(asset_flows, asset_times, ytm, freq)
    lia = _pv_dur_conv(liability_flows, liability_times, ytm, freq)
    pv_ok = a["pv"] >= lia["pv"] - 1e-9 * abs(lia["pv"])
    dur_ok = abs(a["macaulay_duration"] - lia["macaulay_duration"]) <= duration_tolerance
    conv_ok = a["dispersion"] > lia["dispersion"] and min(asset_times) <= min(liability_times) \
        and max(asset_times) >= max(liability_times)
    return {"assets": a, "liabilities": lia, "surplus": a["pv"] - lia["pv"], "pv_condition": pv_ok,
            "duration_condition": dur_ok, "convexity_condition": conv_ok, "immunized": pv_ok and dur_ok and conv_ok,
            "money_duration_gap": a["modified_duration"] * a["pv"] - lia["modified_duration"] * lia["pv"]}


@formula("fixed_income", formula="Required terminal = V0(1+s/m)^(mT) ; required assets = terminal/(1+i/m)^(m T_rem) ; "
         "safety margin = current value - required assets", curriculum=L3_PM)
def contingent_immunization(initial_value: float, safety_net_return: float, horizon_years: float,
                            current_yield: float, current_value: float | None = None,
                            years_remaining: float | None = None, freq: int = 2) -> dict:
    """Contingent immunization: required terminal value, dollar safety margin and the immunization trigger.

    Args:
        initial_value: portfolio value at the start. safety_net_return: minimum acceptable annual return.
        horizon_years: original horizon. current_yield: rate at which the portfolio can be immunized now.
        current_value: portfolio value now (default initial_value). years_remaining: horizon left (default all).
        freq: compounding periods per year.

    Returns:
        required_terminal_value, required_assets_now, dollar_safety_margin, cushion_spread, immunize_now.

    Example:
        500m, safety net 3%, 3y, immunizable at 4.75% sa -> terminal 546.72, required 474.90, cushion 25.10
    """
    cv = initial_value if current_value is None else current_value
    tr = horizon_years if years_remaining is None else years_remaining
    _rate_ok(safety_net_return / freq)
    _rate_ok(current_yield / freq)
    term = initial_value * (1 + safety_net_return / freq) ** (freq * horizon_years)
    req = term / (1 + current_yield / freq) ** (freq * tr)
    margin = cv - req
    return {"required_terminal_value": term, "required_assets_now": req, "dollar_safety_margin": margin,
            "cushion_spread": current_yield - safety_net_return, "immunize_now": margin <= 0}


@formula("fixed_income", formula="w_s D_s + (1-w_s) D_l = D_target ; C_barbell = w_s C_s + (1-w_s) C_l",
         curriculum=L3_PM)
def barbell_bullet_duration_match(target_duration: float, duration_short: float, duration_long: float,
                                  convexity_short: float, convexity_long: float,
                                  bullet_convexity: float | None = None) -> dict:
    """Duration-matched barbell weights and its convexity versus a bullet (or an immunization target).

    Args:
        target_duration: bullet / liability duration. duration_short, duration_long: barbell legs' durations.
        convexity_short, convexity_long: legs' convexities. bullet_convexity: optional bullet convexity.

    Returns:
        weight_short, weight_long, barbell_convexity, convexity_advantage (barbell - bullet; None without bullet).

    Example:
        2y and 10y zeros matched to a 6y bullet: the barbell has the higher convexity
    """
    if duration_long == duration_short:
        raise ValueError("leg durations must differ")
    ws = (duration_long - target_duration) / (duration_long - duration_short)
    cb = ws * convexity_short + (1 - ws) * convexity_long
    return {"weight_short": ws, "weight_long": 1 - ws, "barbell_convexity": cb,
            "convexity_advantage": None if bullet_convexity is None else cb - bullet_convexity}


@formula("fixed_income", formula="N = (D_T - D_I) P_I / (D_CTD P_CTD) x CF x yield beta "
         "= (BPV_target - BPV_portfolio)/BPV_futures", curriculum=L3_PM)
def futures_duration_hedge_contracts(portfolio_value: float, current_duration: float, target_duration: float,
                                     ctd_duration: float, ctd_price: float, conversion_factor: float = 1.0,
                                     yield_beta: float = 1.0) -> dict:
    """Number of bond futures contracts to move a portfolio to a target duration (negative = sell).

    To fully hedge a position set target_duration = 0 (hedge ratio form: D_H P_H / (D_CTD P_CTD) x CF).

    Args:
        portfolio_value: market value. current_duration / target_duration: modified durations.
        ctd_duration: duration of the cheapest-to-deliver bond. ctd_price: its price per contract
            (face of one contract x price). conversion_factor: CTD conversion factor. yield_beta: hedge-item
            yield change per 1 bp change in the CTD yield.

    Returns:
        contracts (signed), contracts_rounded, action, portfolio_bpv, target_bpv, futures_bpv.

    Example:
        GBP 50m, D 9.52 -> 7.5, CTD D 8.47 priced 100,000, CF 1.1 -> sell 131.17 contracts
    """
    if ctd_duration <= 0 or ctd_price <= 0 or conversion_factor <= 0:
        raise ValueError("ctd_duration, ctd_price and conversion_factor must be positive")
    bpv_p = current_duration * portfolio_value * 1e-4
    bpv_t = target_duration * portfolio_value * 1e-4
    bpv_f = ctd_duration * ctd_price * 1e-4 / conversion_factor
    n = (bpv_t - bpv_p) / bpv_f * yield_beta
    return {"contracts": n, "contracts_rounded": round(n), "action": "buy" if n > 0 else "sell" if n < 0 else "none",
            "portfolio_bpv": bpv_p, "target_bpv": bpv_t, "futures_bpv": bpv_f}


@formula("fixed_income", formula="E(R) ~ coupon/P + rolldown + (-D dy + 1/2 C dy^2) - E(credit loss) + E(currency)",
         curriculum=L3_RET)
def bond_expected_return_decomposition(price: float, annual_coupon: float, rolldown_return: float = 0.0,
                                       modified_duration: float = 0.0, convexity: float = 0.0,
                                       yield_change: float = 0.0, expected_credit_loss: float = 0.0,
                                       currency_return: float = 0.0, horizon_years: float = 1.0) -> dict:
    """Decompose a bond's expected horizon return into yield income, rolldown, rate/spread change, credit, FX.

    Args:
        price: current price (per 100). annual_coupon: coupon amount per year (per 100).
        rolldown_return: price return from rolling down an unchanged curve (see bond_rolldown_return).
        modified_duration, convexity: for the expected yield/spread change. yield_change: expected change, decimal.
        expected_credit_loss: PD x LGD over the horizon, decimal. currency_return: expected FX return, decimal.
        horizon_years: holding period (scales yield income).

    Returns:
        yield_income, rolldown_return, rolling_yield, rate_change_effect, credit_loss, currency_return, total.

    Example:
        coupon 4 on price 100, rolldown 0.5%, D 4.5, dy +0.2%, EL 0.3%, FX -1% -> total about 2.31%
    """
    if price <= 0:
        raise ValueError("price must be positive")
    yi = annual_coupon / price * horizon_years
    rc = -modified_duration * yield_change + 0.5 * convexity * yield_change ** 2
    total = yi + rolldown_return + rc - expected_credit_loss + currency_return
    return {"yield_income": yi, "rolldown_return": rolldown_return, "rolling_yield": yi + rolldown_return,
            "rate_change_effect": rc, "credit_loss": -expected_credit_loss, "currency_return": currency_return,
            "total": total}


@formula("fixed_income", formula="Breakeven dy = yield advantage x horizon / duration", curriculum=L3_PM)
def breakeven_yield_spread_change(yield_advantage: float, horizon_years: float, duration: float) -> dict:
    """Yield (spread) change that wipes out a yield advantage over a horizon (use the longer duration).

    Args:
        yield_advantage: annual yield pickup, decimal. horizon_years: horizon. duration: (spread) duration of
            the bond whose price change offsets the advantage (the higher of the two).

    Returns:
        breakeven_change, breakeven_change_bp, horizon_yield_advantage.

    Example:
        300 bp advantage, 3 months, duration 7 -> 10.71 bp
    """
    if duration <= 0:
        raise ValueError("duration must be positive")
    adv = yield_advantage * horizon_years
    return {"breakeven_change": adv / duration, "breakeven_change_bp": adv / duration * 1e4,
            "horizon_yield_advantage": adv}


@formula("fixed_income", formula="R_p = r_F + (B/E)(r_F - k)", curriculum=L3_PM)
def leveraged_bond_return(equity: float, borrowed: float, return_on_assets: float, borrowing_rate: float) -> dict:
    """Return on equity of a leveraged bond portfolio (repo or other borrowing).

    Args:
        equity: own funds. borrowed: borrowed funds. return_on_assets: return on all invested funds, decimal.
        borrowing_rate: cost of borrowing, decimal.

    Returns:
        portfolio_return, profit_on_borrowed, profit_on_equity, leverage_ratio (B/E).

    Example:
        E 40m, B 100m at 4%, assets earn 4.5% -> 5.75%
    """
    if equity <= 0:
        raise ValueError("equity must be positive")
    pb, pe = borrowed * (return_on_assets - borrowing_rate), equity * return_on_assets
    return {"portfolio_return": (pb + pe) / equity, "profit_on_borrowed": pb, "profit_on_equity": pe,
            "leverage_ratio": borrowed / equity}


@formula("fixed_income", formula="cash lent = collateral x (1 - haircut) ; interest = cash x repo rate x days/360",
         curriculum=L3_PM)
def bond_repo_interest(collateral_value: float, repo_rate: float, days: float, haircut: float = 0.0,
                       days_in_year: float = 360) -> dict:
    """Repo (repurchase agreement) financing: cash borrowed, interest and repurchase price.

    Args:
        collateral_value: market value of securities sold. repo_rate: annual repo rate.
        days: term. haircut: margin as a fraction of collateral. days_in_year: 360 (US) or 365.

    Returns:
        cash_borrowed, interest, repurchase_price.

    Example:
        10m overnight at 3% -> interest 833, repurchase 10,000,833
    """
    if not 0 <= haircut < 1 or days <= 0:
        raise ValueError("haircut must be in [0, 1) and days > 0")
    cash = collateral_value * (1 - haircut)
    i = cash * repo_rate * days / days_in_year
    return {"cash_borrowed": cash, "interest": i, "repurchase_price": cash + i}


@formula("fixed_income", formula="f = (F - S)/S ~ i_d - i_f ; hedged ~ r_l + f ; unhedged ~ r_l + e",
         curriculum=L3_PM)
def currency_hedged_bond_return(local_return: float, domestic_rate: float, foreign_rate: float,
                                expected_currency_return: float | None = None,
                                spot_rate: float | None = None) -> dict:
    """Fully hedged vs unhedged foreign bond return and the forward premium from interest rate parity.

    Args:
        local_return: bond return in local currency. domestic_rate / foreign_rate: short-term rates.
        expected_currency_return: expected change in the foreign currency (domestic per foreign).
        spot_rate: optional spot (domestic per foreign) to report the IRP forward.

    Returns:
        forward_premium (approx i_d - i_f), forward_premium_exact, hedged_return, unhedged_return,
        local_risk_premium, forward_rate_approx, forward_rate_exact, hedge (True if hedging is expected
        to beat staying unhedged).

    Example:
        US investor in euro bonds: i_d 4%, i_f 3.2%, euro expected +0.6% -> hedge (0.8% > 0.6%)
    """
    f = domestic_rate - foreign_rate
    fx = (1 + domestic_rate) / (1 + foreign_rate) - 1
    unh = None if expected_currency_return is None else local_return + expected_currency_return
    return {"forward_premium": f, "forward_premium_exact": fx, "hedged_return": local_return + f,
            "unhedged_return": unh, "local_risk_premium": local_return - foreign_rate,
            "forward_rate_approx": None if spot_rate is None else spot_rate * (1 + f),
            "forward_rate_exact": None if spot_rate is None else spot_rate * (1 + fx),
            "hedge": None if expected_currency_return is None else f > expected_currency_return}


@formula("fixed_income", formula="Tracking risk = sample stdev(portfolio return - benchmark return)",
         curriculum=L3_PM)
def bond_index_tracking_risk(portfolio_returns: list, benchmark_returns: list) -> dict:
    """Tracking risk (tracking error) of a bond portfolio against its index.

    Args:
        portfolio_returns, benchmark_returns: periodic returns, decimals, same length (>= 2).

    Returns:
        tracking_risk, mean_active_return, active_returns.

    Example:
        10 periods (L3 R20 Exhibit 4) -> 0.30350%
    """
    if len(portfolio_returns) != len(benchmark_returns) or len(portfolio_returns) < 2:
        raise ValueError("need two equal-length lists with at least 2 returns")
    act = [p - b for p, b in zip(portfolio_returns, benchmark_returns)]
    return {"tracking_risk": statistics.stdev(act), "mean_active_return": statistics.fmean(act),
            "active_returns": act}
