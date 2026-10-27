"""Derivatives III: L3 risk-management applications of forwards, futures, options and swaps.

Futures contract counts are returned unrounded (`contracts`) and rounded to the nearest integer
(`contracts_rounded`), as the curriculum rounds before computing outcomes. A positive count means buy,
negative means sell.
"""
from __future__ import annotations

import math
from statistics import fmean

from ._registry import formula
from .derivatives import _growth, _nonneg, _pos, fra_settlement

R29 = "L3 Risk Management Applications of Forward and Futures Strategies (2014 R29)"
R30 = "L3 Risk Management Applications of Option Strategies (2014 R30)"
R31 = "L3 Risk Management Applications of Swap Strategies (2014 R31)"
FI_L3 = "L3 Fixed-Income Portfolio Management Part II (2014 R22)"
CUR = "L3 Currency Risk Management (2013 R35; 2014 R28)"


def _round_half_up(x: float) -> int:
    return int(math.floor(abs(x) + 0.5)) * (1 if x >= 0 else -1)


def _action(n: float) -> str:
    return "buy" if n > 0 else ("sell" if n < 0 else "none")


# ---------------------------------------------------------------- beta, duration and asset allocation
@formula("derivatives", formula="N_f = ((beta_T - beta_S)/beta_f) x (S/f)", curriculum=R29)
def equity_futures_contracts_beta(portfolio_value: float, current_beta: float, target_beta: float,
                                  futures_price: float, futures_beta: float = 1.0, multiplier: float = 1.0) -> dict:
    """Number of stock index futures to move a stock portfolio's beta to a target (0 = full hedge).

    Args:
        portfolio_value: market value S of the stock position.
        current_beta, target_beta: beta_S and beta_T. futures_price: quoted index futures price.
        futures_beta: beta of the futures (often about 1). multiplier: currency per index point
            (f = quoted price x multiplier; use 1 if the price is already per contract).

    Returns:
        contracts (unrounded; + buy, - sell), contracts_rounded, action, futures_contract_value.

    Example:
        equity_futures_contracts_beta(portfolio_value=65e6, current_beta=0.85, target_beta=1.05,
                                      futures_price=188_500, futures_beta=0.92) -> 74.96 (L3 R29 Example 3)
    """
    _pos("futures_price", futures_price)
    _pos("multiplier", multiplier)
    if futures_beta == 0:
        raise ValueError("futures_beta must be non-zero")
    f = futures_price * multiplier
    n = (target_beta - current_beta) / futures_beta * portfolio_value / f
    return {"contracts": n, "contracts_rounded": _round_half_up(n), "action": _action(n),
            "futures_contract_value": f}


@formula("derivatives", formula="N_f = ((MDUR_T - MDUR_B)/MDUR_f) x (B/f) x yield beta", curriculum=R29)
def bond_futures_contracts_duration(portfolio_value: float, current_mdur: float, target_mdur: float,
                                    futures_price: float, futures_mdur: float, yield_beta: float = 1.0) -> dict:
    """Number of bond futures to change a bond portfolio's modified duration (implied futures duration form).

    Args:
        portfolio_value: market value B of the bonds. current_mdur, target_mdur: MDUR_B and MDUR_T.
        futures_price: futures price per contract (f). futures_mdur: implied modified duration of the futures
            (duration of the cheapest-to-deliver bond at futures expiration).
        yield_beta: change in portfolio yield per unit change in the futures' implied yield.

    Returns:
        contracts (+ buy, - sell), contracts_rounded, action.

    Example:
        bond_futures_contracts_duration(portfolio_value=75e6, current_mdur=6.22, target_mdur=7.50,
            futures_price=82_500, futures_mdur=8.12, yield_beta=1.05) -> 150.47 (L3 R29 Exhibit 2)
    """
    _pos("futures_price", futures_price)
    _pos("futures_mdur", futures_mdur)
    n = (target_mdur - current_mdur) / futures_mdur * portfolio_value / futures_price * yield_beta
    return {"contracts": n, "contracts_rounded": _round_half_up(n), "action": _action(n)}


@formula("derivatives",
         formula="N_f = (D_T - D_I) P_I / (D_CTD P_CTD) x CF x yield beta  (= (BPV_T - BPV_P)/BPV_CTD x CF)",
         curriculum=FI_L3 + "; " + R29)
def bond_futures_contracts_ctd(portfolio_value: float, current_duration: float, target_duration: float,
                               ctd_price: float, ctd_duration: float, conversion_factor: float,
                               yield_beta: float = 1.0) -> dict:
    """Number of bond futures to change duration, sized on the cheapest-to-deliver bond and its conversion factor.

    Args:
        portfolio_value: P_I. current_duration, target_duration: D_I and D_T.
        ctd_price: price of the CTD bond per futures contract (P_CTD). ctd_duration: its duration.
        conversion_factor: CF of the CTD bond. yield_beta: portfolio yield change per CTD yield change.

    Returns:
        contracts (+ buy, - sell), contracts_rounded, action, portfolio_bpv, target_bpv, ctd_bpv
        (basis-point values, D x P x 0.0001), futures_bpv (ctd_bpv / CF).

    Example:
        bond_futures_contracts_ctd(portfolio_value=50e6, current_duration=9.52, target_duration=7.5,
            ctd_price=100_000, ctd_duration=8.47, conversion_factor=1.1) -> -131.17 (L3 FI 2014 R22 Example 11)
    """
    _pos("ctd_price", ctd_price)
    _pos("ctd_duration", ctd_duration)
    _pos("conversion_factor", conversion_factor)
    n = (target_duration - current_duration) * portfolio_value / (ctd_duration * ctd_price) * conversion_factor \
        * yield_beta
    ctd_bpv = ctd_duration * ctd_price * 1e-4
    return {"contracts": n, "contracts_rounded": _round_half_up(n), "action": _action(n),
            "portfolio_bpv": current_duration * portfolio_value * 1e-4,
            "target_bpv": target_duration * portfolio_value * 1e-4, "ctd_bpv": ctd_bpv,
            "futures_bpv": ctd_bpv / conversion_factor}


@formula("derivatives",
         formula="N* = V(1+r)^T/(q f) (equitize, buy) or -V(1+r)^T/(q f) (to cash, sell); V* = |N*| q f/(1+r)^T; units = |N*| q/(1+d)^T",
         curriculum=R29)
def synthetic_equity_cash_futures(amount: float, futures_price: float, rate: float, years: float,
                                  dividend_yield: float = 0.0, multiplier: float = 1.0,
                                  direction: str = "equitize", compounding: str = "discrete") -> dict:
    """Synthetic index fund / equitizing cash (long futures + bonds) or synthetic cash (long stock + short futures).

    Args:
        amount: cash to equitize, or stock value to convert to cash (V).
        futures_price: quoted futures price (f). multiplier: q (1 if f is per contract).
        rate: risk-free rate. years: time to futures expiration.
        dividend_yield: index dividend yield (for the unit count). direction: "equitize" or "to_cash".
        compounding: "discrete" (curriculum) or "continuous".

    Returns:
        contracts (+ buy, - sell, unrounded), contracts_rounded, effective_amount (V* after rounding),
        bond_value_at_expiration (V* grown at r = |N*| q f), units_at_start, units_at_expiration.

    Example:
        synthetic_equity_cash_futures(amount=100e6, futures_price=4000, multiplier=10, rate=0.05,
                                      years=0.25, dividend_yield=0.025) -> 2530.68, rounded 2531,
        effective 100,012,622 (L3 R29 Exhibit 4)
    """
    if direction not in ("equitize", "to_cash"):
        raise ValueError("direction must be 'equitize' or 'to_cash'")
    _pos("amount", amount)
    _pos("futures_price", futures_price)
    _pos("multiplier", multiplier)
    g = _growth(rate, years, compounding)
    sgn = 1 if direction == "equitize" else -1
    n = sgn * amount * g / (multiplier * futures_price)
    n_r = _round_half_up(n)
    fv = abs(n_r) * multiplier * futures_price
    return {"contracts": n, "contracts_rounded": n_r, "effective_amount": fv / g, "bond_value_at_expiration": fv,
            "units_at_start": abs(n_r) * multiplier / _growth(dividend_yield, years, compounding),
            "units_at_expiration": abs(n_r) * multiplier}


@formula("derivatives",
         formula="Shift D from stock to bonds: N_sf = (0 - beta_S)/beta_f x D/f_s, N_bf = (MDUR_B - 0)/MDUR_f x D/f_b; then beta/duration adjustments on the new amounts",
         curriculum=R29)
def futures_asset_allocation(portfolio_value: float, stock_weight: float, target_stock_weight: float,
                             stock_beta: float, stock_futures_price: float, stock_futures_beta: float,
                             bond_mdur: float, bond_futures_price: float, bond_futures_mdur: float,
                             target_stock_beta: float | None = None, target_bond_mdur: float | None = None,
                             yield_beta: float = 1.0) -> dict:
    """Synthetically reallocate between stocks and bonds with index and bond futures (no securities traded).

    Step 1 converts the stock (bond) amount being moved to synthetic cash, step 2 converts that cash into the
    other asset class at its existing beta or duration, step 3 (optional) moves the beta/duration of the new
    stock and bond allocations to targets. Components are rounded individually, as in the curriculum.

    Args:
        portfolio_value: total stock + bond value. stock_weight, target_stock_weight: current and target weights.
        stock_beta: beta of the stock holdings. stock_futures_price, stock_futures_beta: index futures.
        bond_mdur: modified duration of the bond holdings. bond_futures_price, bond_futures_mdur: bond futures.
        target_stock_beta, target_bond_mdur: optional new beta / duration (defaults: unchanged).
        yield_beta: applied to the bond futures counts.

    Returns:
        stock_futures (total, + buy / - sell), bond_futures, stock_futures_rounded, bond_futures_rounded
        (sum of rounded components), components (each step), amount_moved_to_bonds.

    Example:
        futures_asset_allocation(portfolio_value=300e6, stock_weight=0.8, target_stock_weight=0.5,
            stock_beta=1.10, stock_futures_price=200_000, stock_futures_beta=0.96, bond_mdur=6.5,
            bond_futures_price=105_250, bond_futures_mdur=7.2) -> stock -515.63, bond 771.97
        (L3 R29 Exhibit 6)
    """
    for n_, v in (("stock_futures_price", stock_futures_price), ("bond_futures_price", bond_futures_price),
                  ("bond_futures_mdur", bond_futures_mdur)):
        _pos(n_, v)
    if stock_futures_beta == 0:
        raise ValueError("stock_futures_beta must be non-zero")
    if not (0 <= stock_weight <= 1 and 0 <= target_stock_weight <= 1):
        raise ValueError("weights must be between 0 and 1")
    moved = (stock_weight - target_stock_weight) * portfolio_value  # > 0: stock -> bonds
    comps = {
        "stock_to_cash": -stock_beta / stock_futures_beta * moved / stock_futures_price,
        "cash_to_bonds": bond_mdur / bond_futures_mdur * moved / bond_futures_price * yield_beta,
    }
    tb = stock_beta if target_stock_beta is None else target_stock_beta
    td = bond_mdur if target_bond_mdur is None else target_bond_mdur
    comps["stock_beta_adjustment"] = (tb - stock_beta) / stock_futures_beta * target_stock_weight * portfolio_value \
        / stock_futures_price
    comps["bond_duration_adjustment"] = (td - bond_mdur) / bond_futures_mdur * (1 - target_stock_weight) \
        * portfolio_value / bond_futures_price * yield_beta
    s_keys, b_keys = ("stock_to_cash", "stock_beta_adjustment"), ("cash_to_bonds", "bond_duration_adjustment")
    return {"stock_futures": sum(comps[k] for k in s_keys), "bond_futures": sum(comps[k] for k in b_keys),
            "stock_futures_rounded": sum(_round_half_up(comps[k]) for k in s_keys),
            "bond_futures_rounded": sum(_round_half_up(comps[k]) for k in b_keys),
            "components": comps, "amount_moved_to_bonds": moved}


@formula("derivatives",
         formula="N_sf = (beta_T - 0)/beta_f x S/f_s; N_bf = (MDUR_T - 0)/MDUR_f x B/f_b x yield beta",
         curriculum=R29)
def preinvest_futures_contracts(stock_amount: float, target_beta: float, stock_futures_price: float,
                                stock_futures_beta: float, bond_amount: float, target_mdur: float,
                                bond_futures_price: float, bond_futures_mdur: float, yield_beta: float = 1.0) -> dict:
    """Pre-investing expected cash: long stock and bond futures sized at the target beta and duration.

    Args:
        stock_amount: amount to be invested in stock. target_beta: its desired beta.
        stock_futures_price, stock_futures_beta: index futures. bond_amount: amount for bonds.
        target_mdur: desired modified duration. bond_futures_price, bond_futures_mdur: bond futures.
        yield_beta: applied to the bond count.

    Returns:
        stock_futures, bond_futures (unrounded, buy), stock_futures_rounded, bond_futures_rounded.

    Example:
        preinvest_futures_contracts(stock_amount=6e6, target_beta=1.08, stock_futures_price=210_500,
            stock_futures_beta=0.97, bond_amount=4e6, target_mdur=5.25, bond_futures_price=115_750,
            bond_futures_mdur=6.05) -> 31.74 and 29.99 (L3 R29 Exhibit 9)
    """
    s = equity_futures_contracts_beta(portfolio_value=stock_amount, current_beta=0.0, target_beta=target_beta,
                                      futures_price=stock_futures_price, futures_beta=stock_futures_beta)
    b = bond_futures_contracts_duration(portfolio_value=bond_amount, current_mdur=0.0, target_mdur=target_mdur,
                                        futures_price=bond_futures_price, futures_mdur=bond_futures_mdur,
                                        yield_beta=yield_beta)
    return {"stock_futures": s["contracts"], "bond_futures": b["contracts"],
            "stock_futures_rounded": s["contracts_rounded"], "bond_futures_rounded": b["contracts_rounded"]}


@formula("derivatives",
         formula="N_puts = (MDUR_P/MDUR_f)(B/f) x yield beta; floor ~ B(1 - MDUR_P x beta_y x dy*) - premium, dy* = (f - X)/(f MDUR_f)",
         curriculum=FI_L3 + " (protective put buying); " + R29)
def bond_futures_protective_put(portfolio_value: float, portfolio_mdur: float, futures_price: float,
                                futures_mdur: float, put_strike: float, put_premium: float, yield_beta: float = 1.0,
                                ending_portfolio_value: float | None = None,
                                ending_futures_price: float | None = None) -> dict:
    """Protective put on bond futures for a bond portfolio: number of puts, cost, approximate floor, outcome.

    The hedge ratio equals the number of futures that would fully hedge the portfolio. The floor is a
    duration approximation: the portfolio value when rates have risen enough to push the futures price
    down to the put strike, less the premium; below that the puts offset further losses.

    Args:
        portfolio_value: B. portfolio_mdur: modified duration. futures_price: f per contract.
        futures_mdur: implied modified duration of the futures. put_strike, put_premium: per contract.
        yield_beta: portfolio yield change per futures yield change.
        ending_portfolio_value, ending_futures_price: optional outcome to evaluate.

    Returns:
        puts (unrounded), puts_rounded, total_premium, yield_rise_to_strike, approximate_floor_value,
        and with ending values: put_payoff, hedged_value (ending value + payoff - premium).

    Example:
        bond_futures_protective_put(portfolio_value=75e6, portfolio_mdur=6.22, futures_price=82_500,
            futures_mdur=8.12, put_strike=80_000, put_premium=1_000, yield_beta=1.05) -> 731.19 puts
    """
    _pos("put_strike", put_strike)
    n = -bond_futures_contracts_duration(portfolio_value=portfolio_value, current_mdur=portfolio_mdur,
                                         target_mdur=0.0, futures_price=futures_price, futures_mdur=futures_mdur,
                                         yield_beta=yield_beta)["contracts"]
    n_r = _round_half_up(n)
    prem = n_r * put_premium
    dy = max(0.0, (futures_price - put_strike) / (futures_price * futures_mdur))
    out = {"puts": n, "puts_rounded": n_r, "total_premium": prem, "yield_rise_to_strike": dy,
           "approximate_floor_value": portfolio_value * (1 - portfolio_mdur * yield_beta * dy) - prem}
    if ending_portfolio_value is not None and ending_futures_price is not None:
        pay = n_r * max(0.0, put_strike - ending_futures_price)
        out.update({"put_payoff": pay, "hedged_value": ending_portfolio_value + pay - prem})
    return out


# ---------------------------------------------------------------- currency
@formula("derivatives",
         formula="N_f = (0 - beta)/beta_f x V/f; forward notional = V(1+r_f)^T (or V when hedging the principal)",
         curriculum=R29 + "; " + CUR)
def foreign_equity_currency_hedge(portfolio_value_foreign: float, portfolio_beta: float, index_futures_price: float,
                                  index_futures_beta: float, foreign_rate: float, years: float, spot_rate: float,
                                  forward_rate: float, hedge_basis: str = "hedged_value",
                                  compounding: str = "discrete", ending_portfolio_value_foreign: float | None = None,
                                  ending_futures_price: float | None = None,
                                  ending_spot_rate: float | None = None) -> dict:
    """Hedge a foreign equity portfolio's local market risk with index futures and its currency risk with a forward.

    Hedging the local market locks in about the foreign risk-free rate, so the forward notional is
    V(1 + r_f)^T (hedge_basis="hedged_value"); hedge_basis="principal" sells forward the current value V only.
    With both hedges the domestic return is about the domestic risk-free rate.

    Args:
        portfolio_value_foreign: V in foreign currency. portfolio_beta: beta vs the foreign index.
        index_futures_price, index_futures_beta: foreign index futures (price per contract, foreign currency).
        foreign_rate: foreign risk-free rate. years: hedge horizon. spot_rate, forward_rate: domestic per foreign.
        hedge_basis: "hedged_value" or "principal". compounding: "discrete", "continuous" or "simple".
        ending_*: optional outcome (portfolio value, futures price, spot rate at the horizon) to evaluate.

    Returns:
        index_futures (unrounded, sell), index_futures_rounded, forward_notional_foreign, initial_value_domestic,
        locked_value_domestic; with ending values: value_unhedged, value_market_hedged, value_fully_hedged
        (domestic currency) and their returns.

    Example:
        foreign_equity_currency_hedge(portfolio_value_foreign=10e6, portfolio_beta=1.10, index_futures_price=120_000,
            index_futures_beta=0.95, foreign_rate=0.04, years=1, spot_rate=0.80, forward_rate=0.815,
            ending_portfolio_value_foreign=9.5e6, ending_futures_price=110_600, ending_spot_rate=0.785)
        -> -96.49 contracts; values 7,457,500 / 8,165,884 / 8,477,884 (L3 R29 Exhibit 12)
    """
    if hedge_basis not in ("hedged_value", "principal"):
        raise ValueError("hedge_basis must be 'hedged_value' or 'principal'")
    nf = equity_futures_contracts_beta(portfolio_value=portfolio_value_foreign, current_beta=portfolio_beta,
                                       target_beta=0.0, futures_price=index_futures_price,
                                       futures_beta=index_futures_beta)
    notional = portfolio_value_foreign * (_growth(foreign_rate, years, compounding) if hedge_basis == "hedged_value"
                                          else 1.0)
    v0 = portfolio_value_foreign * spot_rate
    out = {"index_futures": nf["contracts"], "index_futures_rounded": nf["contracts_rounded"],
           "forward_notional_foreign": notional, "initial_value_domestic": v0,
           "locked_value_domestic": notional * forward_rate}
    ends = (ending_portfolio_value_foreign, ending_futures_price, ending_spot_rate)
    if all(e is not None for e in ends):
        vf, ft, st = ends
        unhedged = vf * st
        mkt = (vf + nf["contracts_rounded"] * (ft - index_futures_price)) * st
        full = mkt + notional * (forward_rate - st)
        out.update({"value_unhedged": unhedged, "value_market_hedged": mkt, "value_fully_hedged": full,
                    "return_unhedged": unhedged / v0 - 1, "return_market_hedged": mkt / v0 - 1,
                    "return_fully_hedged": full / v0 - 1})
    return out


@formula("derivatives",
         formula="h* = cov(R_asset, R_hedge)/var(R_hedge) = rho sigma_asset/sigma_hedge; contracts = h* x exposure/contract size",
         curriculum=CUR + "; Hull ch. 3 (minimum variance hedge ratio)")
def minimum_variance_hedge_ratio(asset_returns: list | None = None, hedge_returns: list | None = None,
                                 correlation: float | None = None, asset_sd: float | None = None,
                                 hedge_sd: float | None = None, exposure: float | None = None,
                                 contract_size: float | None = None) -> dict:
    """Minimum-variance (regression) hedge ratio, from return series or from correlation and volatilities.

    For currency hedging of a foreign asset, h* = 1 (translation risk) + cov(R, s)/var(s) (economic risk).

    Args:
        asset_returns, hedge_returns: matched return series (domestic-currency asset returns and
            futures/forward returns); or correlation, asset_sd, hedge_sd.
        exposure: size of the position to hedge (units or currency); contract_size: per hedging contract.

    Returns:
        hedge_ratio, r_squared (hedge effectiveness = correlation^2), contracts (if exposure and contract_size).

    Example:
        minimum_variance_hedge_ratio(correlation=0.928, asset_sd=0.0263, hedge_sd=0.0313, exposure=2_000_000,
                                     contract_size=42_000) -> h 0.7798, 37.13 contracts (Hull, jet fuel example)
    """
    if asset_returns is not None and hedge_returns is not None:
        if len(asset_returns) != len(hedge_returns) or len(asset_returns) < 3:
            raise ValueError("return series need equal length of at least 3")
        ma, mh = fmean(asset_returns), fmean(hedge_returns)
        cov = sum((a - ma) * (h - mh) for a, h in zip(asset_returns, hedge_returns))
        vh = sum((h - mh) ** 2 for h in hedge_returns)
        va = sum((a - ma) ** 2 for a in asset_returns)
        _pos("variance of hedge_returns", vh)
        h = cov / vh
        r2 = cov * cov / (va * vh) if va > 0 else 0.0
    elif None not in (correlation, asset_sd, hedge_sd):
        _pos("hedge_sd", hedge_sd)
        _nonneg("asset_sd", asset_sd)
        h = correlation * asset_sd / hedge_sd
        r2 = correlation ** 2
    else:
        raise ValueError("give asset_returns and hedge_returns, or correlation, asset_sd and hedge_sd")
    out = {"hedge_ratio": h, "r_squared": r2}
    if exposure is not None and contract_size:
        out["contracts"] = h * exposure / contract_size
    return out


@formula("derivatives", formula="R_H = R* - h x R_F  (hedged domestic-currency return)", curriculum=CUR)
def currency_hedged_return(unhedged_return: float, hedge_ratio: float, futures_return: float) -> dict:
    """Return on a foreign asset with a fraction h of its value hedged by currency futures/forwards.

    Args:
        unhedged_return: R*, the asset return in domestic currency. hedge_ratio: h (1 = hedge the principal).
        futures_return: R_F, the return on the currency futures/forward position (as a long).

    Returns:
        hedged_return, hedge_contribution (-h x R_F).

    Example:
        currency_hedged_return(unhedged_return=-0.0405, hedge_ratio=0.5, futures_return=-0.05) -> -0.0155
        (L3 2013 R35 Example 2)
    """
    return {"hedged_return": unhedged_return - hedge_ratio * futures_return,
            "hedge_contribution": -hedge_ratio * futures_return}


# ---------------------------------------------------------------- interest rate risk on loans
@formula("derivatives",
         formula="Borrow NP - FRA payoff at L + spread; effective rate = (repayment/NP - 1) x 360/m",
         curriculum=R29)
def fra_hedged_loan_rate(notional: float, fra_rate: float, libor_at_expiry: float, spread: float,
                         days_underlying: float, day_basis: float = 360) -> dict:
    """Effective borrowing rate when a future single-payment loan is hedged with a long FRA.

    The FRA settles (discounted) when the loan starts; the borrower raises notional minus the FRA receipt at
    LIBOR + spread, so the effective rate is close to FRA rate + spread.

    Args:
        notional: loan amount needed. fra_rate: FRA rate. libor_at_expiry: LIBOR when the loan starts.
        spread: loan spread over LIBOR. days_underlying: loan term in days (m). day_basis: 360.

    Returns:
        effective_rate (annualized, add-on), fra_payoff, amount_borrowed, repayment, loan_rate.

    Example:
        fra_hedged_loan_rate(notional=40e6, fra_rate=0.0525, libor_at_expiry=0.06, spread=0.02,
                             days_underlying=180) -> 0.0724, FRA payoff 145,631 (L3 R29 Exhibit 1)
    """
    _pos("notional", notional)
    pay = fra_settlement(notional=notional, fra_rate=fra_rate, underlying_rate=libor_at_expiry,
                         days_underlying=days_underlying, day_basis=day_basis)["settlement"]
    borrowed = notional - pay
    loan_rate = libor_at_expiry + spread
    acc = days_underlying / day_basis
    repay = borrowed * (1 + loan_rate * acc)
    return {"effective_rate": (repay / notional - 1) / acc, "fra_payoff": pay, "amount_borrowed": borrowed,
            "repayment": repay, "loan_rate": loan_rate}


@formula("derivatives",
         formula="Premium FV = prem(1 + (L0 + s) d/360); call (borrower): rate = ((NP + interest - payoff)/(NP - prem FV))^(365/m) - 1; put (lender) analogous",
         curriculum=R30 + " (Section 3.1-3.2)")
def interest_rate_option_loan(notional: float, exercise_rate: float, premium: float, libor_at_expiry: float,
                              spread: float, days_to_expiry: float, loan_days: float, libor_now: float,
                              option_type: str = "call", day_basis: float = 360,
                              days_in_year: float = 365) -> dict:
    """Effective annual rate on a future loan protected by an interest rate call (borrower) or put (lender).

    The premium is compounded to the loan start at today's LIBOR + spread; the option pays
    NP x max(0, L - X) (call) or NP x max(0, X - L) (put) x m/360 at loan repayment.

    Args:
        notional: loan amount. exercise_rate: option strike rate. premium: option cost paid today.
        libor_at_expiry: LIBOR when the loan starts (option expiry). spread: loan spread over LIBOR.
        days_to_expiry: days from purchase to loan start. loan_days: loan term m.
        libor_now: LIBOR when the option is bought (premium financing at libor_now + spread).
        option_type: "call" (borrower ceiling) or "put" (lender floor). day_basis: 360. days_in_year: 365.

    Returns:
        effective_annual_rate, loan_interest, option_payoff, premium_future_value, effective_proceeds
        (borrower: NP - prem FV; lender: outlay NP + prem FV).

    Example:
        interest_rate_option_loan(notional=40e6, exercise_rate=0.05, premium=100_000, libor_at_expiry=0.08,
            spread=0.02, days_to_expiry=128, loan_days=180, libor_now=0.055) -> 0.0779 (L3 R30 Exhibit 13)
    """
    if option_type not in ("call", "put"):
        raise ValueError("option_type must be 'call' (borrower) or 'put' (lender)")
    _pos("notional", notional)
    _pos("loan_days", loan_days)
    prem_fv = premium * (1 + (libor_now + spread) * days_to_expiry / day_basis)
    acc = loan_days / day_basis
    interest = notional * (libor_at_expiry + spread) * acc
    if option_type == "call":
        payoff = notional * max(0.0, libor_at_expiry - exercise_rate) * acc
        proceeds = notional - prem_fv
        rate = ((notional + interest - payoff) / proceeds) ** (days_in_year / loan_days) - 1
    else:
        payoff = notional * max(0.0, exercise_rate - libor_at_expiry) * acc
        proceeds = notional + prem_fv
        rate = ((notional + interest + payoff) / proceeds) ** (days_in_year / loan_days) - 1
    return {"effective_annual_rate": rate, "loan_interest": interest, "option_payoff": payoff,
            "premium_future_value": prem_fv, "effective_proceeds": proceeds}


@formula("derivatives",
         formula="Interest_t = NP(L_(t-1) + s)(days_t/360); caplet = NP max(0, L_(t-1) - cap)(days_t/360); floorlet = NP max(0, floor - L_(t-1))(days_t/360); effective = interest - caplet + floorlet",
         curriculum=R30 + " (Section 3.3-3.5)")
def floating_rate_loan_cap_floor(notional: float, spread: float, libor_resets: list, days: list,
                                 cap_rate: float | None = None, floor_rate: float | None = None,
                                 day_basis: float = 360) -> dict:
    """Effective interest on a floating-rate loan with a cap (borrower), floor (lender) or collar.

    Each period's rate is set at the previous reset; caplets/floorlets expire on the resets and pay at the
    next payment date. No caplet/floorlet covers the first period (its rate is already set).
    effective = interest - caplet payoff + floorlet payoff holds for a borrower (long cap, short floor) and
    for a lender (long floor, short cap) alike.

    Args:
        notional: loan principal. spread: spread over LIBOR.
        libor_resets: LIBOR at the start and at each reset date (one per payment period).
        days: days in each payment period (same length as libor_resets).
        cap_rate, floor_rate: strike rates (None if absent). day_basis: 360.

    Returns:
        total_effective_interest, rows (period, libor_set, loan_rate, days, interest, caplet_payoff,
        floorlet_payoff, effective_interest).

    Example:
        floating_rate_loan_cap_floor(notional=25e6, spread=0.025, libor_resets=[0.065, 0.0725, 0.06875, 0.07125],
                                     days=[87, 93, 94, 90], cap_rate=0.07) -> effective 543,750 / 613,542 /
        611,979 / 593,750 (L3 R30 Example 13)
    """
    if len(libor_resets) != len(days) or not days:
        raise ValueError("libor_resets and days need the same non-zero length")
    rows = []
    for i, (lib, d) in enumerate(zip(libor_resets, days)):
        acc = d / day_basis
        interest = notional * (lib + spread) * acc
        cap_pay = notional * max(0.0, lib - cap_rate) * acc if (cap_rate is not None and i > 0) else 0.0
        floor_pay = notional * max(0.0, floor_rate - lib) * acc if (floor_rate is not None and i > 0) else 0.0
        rows.append({"period": i + 1, "libor_set": lib, "loan_rate": lib + spread, "days": d, "interest": interest,
                     "caplet_payoff": cap_pay, "floorlet_payoff": floor_pay,
                     "effective_interest": interest - cap_pay + floor_pay})
    return {"total_effective_interest": sum(r["effective_interest"] for r in rows), "rows": rows}


# ---------------------------------------------------------------- swaps
@formula("derivatives",
         formula="D(pay-fixed swap) = D(floating) - D(fixed) ~ (period/2) - 0.75 x maturity (curriculum assumption)",
         curriculum=R31)
def swap_duration_approx(maturity_years: float, payments_per_year: float, fixed_leg_duration: float | None = None,
                         position: str = "pay_fixed") -> dict:
    """Approximate duration of an interest rate swap position (curriculum rule of thumb).

    Floating leg duration is about half the payment interval; the fixed leg is taken as 75% of maturity unless
    given. Pay-fixed swap = long floating bond + short fixed bond (negative duration).

    Args:
        maturity_years: swap tenor. payments_per_year: payment frequency.
        fixed_leg_duration: duration of the fixed-rate bond (default 0.75 x maturity).
        position: "pay_fixed" or "receive_fixed".

    Returns:
        swap_duration, floating_leg_duration, fixed_leg_duration.

    Example:
        swap_duration_approx(maturity_years=1, payments_per_year=12) -> -0.708 (L3 R31 Example 2)
    """
    _pos("maturity_years", maturity_years)
    _pos("payments_per_year", payments_per_year)
    if position not in ("pay_fixed", "receive_fixed"):
        raise ValueError("position must be 'pay_fixed' or 'receive_fixed'")
    flt = 0.5 / payments_per_year
    fix = 0.75 * maturity_years if fixed_leg_duration is None else fixed_leg_duration
    d = flt - fix
    return {"swap_duration": d if position == "pay_fixed" else -d, "floating_leg_duration": flt,
            "fixed_leg_duration": fix}


@formula("derivatives", formula="NP = B x (MDUR_T - MDUR_B)/MDUR_S", curriculum=R31)
def swap_notional_for_duration(portfolio_value: float, current_mdur: float, target_mdur: float,
                               swap_mdur: float) -> dict:
    """Swap notional principal needed to move a bond portfolio's duration to a target.

    Args:
        portfolio_value: B. current_mdur, target_mdur: MDUR_B and MDUR_T.
        swap_mdur: modified duration of the swap per unit notional (negative for pay-fixed;
            see swap_duration_approx).

    Returns:
        notional (positive), swap_position ("pay_fixed" if lowering duration with a negative-duration swap).

    Example:
        swap_notional_for_duration(portfolio_value=250e6, current_mdur=5.5, target_mdur=4.5, swap_mdur=-1.25)
        -> 200,000,000 (L3 R31 Example 2)
    """
    if swap_mdur == 0:
        raise ValueError("swap_mdur must be non-zero")
    np_ = portfolio_value * (target_mdur - current_mdur) / swap_mdur
    # a negative-duration swap is pay-fixed; a negative notional means take the other side
    pay_fixed = (swap_mdur < 0) == (np_ >= 0)
    return {"notional": abs(np_), "swap_position": "pay_fixed" if pay_fixed else "receive_fixed"}


@formula("derivatives",
         formula="Floating loan L + s with pay-fixed swap at FS -> fixed FS + s; fixed asset c with pay-fixed swap -> floating L + (c - FS)",
         curriculum=R31)
def swap_converted_rate(swap_fixed_rate: float, loan_spread: float | None = None,
                        fixed_rate: float | None = None) -> dict:
    """Rate after converting a floating-rate loan to fixed, or a fixed-rate loan/asset to floating, with a swap.

    Args:
        swap_fixed_rate: fixed rate on the swap.
        loan_spread: for a floating-rate borrower at LIBOR + spread who pays fixed on the swap.
        fixed_rate: for a fixed-rate position (loan held or bond issued) swapped to floating; the result is
            LIBOR + (fixed_rate - swap_fixed_rate) for a holder paying fixed / receiving LIBOR on the swap
            (an issuer receiving fixed pays LIBOR + (fixed_rate - swap_fixed_rate) too).

    Returns:
        synthetic_fixed_rate (floating -> fixed), or spread_over_libor (fixed -> floating).

    Example:
        swap_converted_rate(swap_fixed_rate=0.07, fixed_rate=0.06) -> LIBOR - 0.01 (L3 R31 Example 1)
    """
    if (loan_spread is None) == (fixed_rate is None):
        raise ValueError("give exactly one of loan_spread (floating -> fixed) or fixed_rate (fixed -> floating)")
    if loan_spread is not None:
        return {"synthetic_fixed_rate": swap_fixed_rate + loan_spread}
    return {"spread_over_libor": fixed_rate - swap_fixed_rate}


# ---------------------------------------------------------------- option portfolio hedging
@formula("derivatives",
         formula="Shares to delta-hedge = -option position x delta; options to hedge shares = -shares / delta; trade = required - held",
         curriculum=R30 + " (Section 4.1); " + "L2 2014 R53 Section 7.3.1")
def delta_hedge(option_delta: float, spot: float, option_position: float | None = None,
                share_position: float | None = None, current_shares: float = 0.0, round_units: bool = True) -> dict:
    """Delta-neutral hedge: shares needed against an option position (or options against shares), and the rebalance.

    Args:
        option_delta: delta per option (call > 0, put < 0).
        spot: underlying price (values the rebalancing trade).
        option_position: options held (+ long, - short), e.g. -1000 for 1,000 calls sold.
        share_position: alternatively, shares to be hedged with options.
        current_shares: shares already held in the hedge (for rebalancing).
        round_units: round the required units to whole numbers (curriculum practice).

    Returns:
        With option_position: shares_required, shares_to_trade (+ buy / - sell), cash_flow (- = pay; finance
        by borrowing, + = invest). With share_position: options_required (+ buy / - sell).

    Example:
        delta_hedge(option_delta=0.5966, spot=1215, option_position=-1000, current_shares=583)
        -> shares_required 597, buy 14, cash -17,010 (L3 R30 Exhibit 22)
    """
    if (option_position is None) == (share_position is None):
        raise ValueError("give exactly one of option_position or share_position")
    if option_position is not None:
        req = -option_position * option_delta
        if round_units:
            req = _round_half_up(req)
        trade = req - current_shares
        return {"shares_required": req, "shares_to_trade": trade, "cash_flow": -trade * spot}
    if option_delta == 0:
        raise ValueError("option_delta must be non-zero")
    req = -share_position / option_delta
    return {"options_required": _round_half_up(req) if round_units else req}


@formula("derivatives",
         formula="One option + underlying: n1 = -Gamma_P/Gamma_1, shares = -(Delta_P + n1 Delta_1); two options: solve n1 D1 + n2 D2 = -Delta_P, n1 G1 + n2 G2 = -Gamma_P",
         curriculum=R30 + " (Section 4.2); Hull ch. 19 (gamma neutrality)")
def delta_gamma_neutral_hedge(portfolio_delta: float, portfolio_gamma: float, option1_delta: float,
                              option1_gamma: float, option2_delta: float | None = None,
                              option2_gamma: float | None = None) -> dict:
    """Make a position delta- and gamma-neutral with one option plus the underlying, or with two options.

    Args:
        portfolio_delta, portfolio_gamma: current position Greeks (in units of the underlying).
        option1_delta, option1_gamma: Greeks of the first hedging option.
        option2_delta, option2_gamma: Greeks of a second option (then no underlying is used).

    Returns:
        option1_units, option2_units (0 if unused), underlying_units (+ buy / - sell),
        resulting_delta, resulting_gamma.

    Example:
        delta_gamma_neutral_hedge(portfolio_delta=0, portfolio_gamma=-3000, option1_delta=0.62,
                                  option1_gamma=1.5) -> buy 2,000 options, sell 1,240 shares (Hull)
    """
    if option2_delta is None or option2_gamma is None:
        if option1_gamma == 0:
            raise ValueError("option1_gamma must be non-zero")
        n1, n2 = -portfolio_gamma / option1_gamma, 0.0
        shares = -(portfolio_delta + n1 * option1_delta)
    else:
        det = option1_delta * option2_gamma - option2_delta * option1_gamma
        if abs(det) < 1e-15:
            raise ValueError("the two options have proportional delta/gamma; cannot neutralize both")
        n1 = (-portfolio_delta * option2_gamma + portfolio_gamma * option2_delta) / det
        n2 = (-portfolio_gamma * option1_delta + portfolio_delta * option1_gamma) / det
        shares = 0.0
    d_out = portfolio_delta + n1 * option1_delta + n2 * (option2_delta or 0.0) + shares
    g_out = portfolio_gamma + n1 * option1_gamma + n2 * (option2_gamma or 0.0)
    return {"option1_units": n1, "option2_units": n2, "underlying_units": shares, "resulting_delta": d_out,
            "resulting_gamma": g_out}
