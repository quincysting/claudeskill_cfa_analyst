"""Fixed income: credit risk — expected loss, CVA with hazard rates, structural and reduced-form models (L1-L2)."""
from __future__ import annotations

import math
from statistics import NormalDist

from ._registry import formula
from .fi_pricing import _pv, _solve

L1_CR = "L1 FI: Fundamentals of Credit Analysis (2014 R56)"
L2_CR = "L2 FI: Credit Analysis Models (2014 R45; CVA/hazard-rate framework of later editions)"
_N = NormalDist().cdf


@formula("fixed_income", formula="EL = PD x LGD x exposure ; LGD = 1 - recovery ; credit spread ~ PD x LGD",
         curriculum=L1_CR)
def credit_expected_loss(default_probability: float, recovery_rate: float | None = None,
                         loss_given_default: float | None = None, exposure: float = 100.0) -> dict:
    """Expected credit loss and the approximate credit spread it implies.

    Args:
        default_probability: annual probability of default, decimal.
        recovery_rate: fraction recovered in default (or give loss_given_default = loss severity).
        exposure: amount at risk.

    Returns:
        expected_loss_rate, expected_loss, loss_given_default, approx_credit_spread, approx_credit_spread_bp.

    Example:
        PD 2%, recovery 40% -> EL 1.2% of exposure, spread ~ 120 bp
    """
    if not 0 <= default_probability <= 1:
        raise ValueError("default_probability must be in [0, 1]")
    if loss_given_default is None:
        if recovery_rate is None:
            raise ValueError("give recovery_rate or loss_given_default")
        loss_given_default = 1 - recovery_rate
    if not 0 <= loss_given_default <= 1:
        raise ValueError("loss given default / recovery must be in [0, 1]")
    el = default_probability * loss_given_default
    return {"expected_loss_rate": el, "expected_loss": el * exposure, "loss_given_default": loss_given_default,
            "approx_credit_spread": el, "approx_credit_spread_bp": el * 1e4}


@formula("fixed_income", formula="CVA = sum_t LGD_t x POD_t x DF_t ; POD_t = h x POS_{t-1} ; POS_t = prod(1-h) ; "
         "fair value = VND - CVA", curriculum=L2_CR)
def credit_cva_bond(coupon_rate: float, years: int, hazard_rate: float | list, recovery_rate: float,
                    risk_free_rate: float | None = None, spot_rates: list | None = None,
                    face: float = 100.0) -> dict:
    """Credit valuation adjustment of an annual-pay bond with period hazard rates (probabilities of default).

    Exposure at date t is the bond's risk-free value at t including the coupon due then; recovery is a
    fraction of exposure; default can occur once per year.

    Args:
        coupon_rate: annual coupon. years: maturity (whole years).
        hazard_rate: conditional annual default probability (float) or a list, one per year.
        recovery_rate: recovery as a fraction of exposure.
        risk_free_rate: flat annual risk-free rate, or give spot_rates (annual, one per year).
        face: par.

    Returns:
        cva, value_default_free, fair_value, ytm, credit_spread, cumulative_pod, schedule (per year: exposure,
        recovery, lgd, pod, pos, expected_loss, discount_factor, pv_expected_loss).

    Example:
        3y 5% bond, flat 3%, hazard 1.25%, recovery 40% -> VND 105.657, CVA ~2.24, fair value ~103.42
    """
    n = int(years)
    if n < 1 or n != years:
        raise ValueError("years must be a positive whole number (annual periods)")
    hz = list(hazard_rate) if isinstance(hazard_rate, (list, tuple)) else [hazard_rate] * n
    if len(hz) != n or any(not 0 <= h <= 1 for h in hz):
        raise ValueError("hazard_rate must be in [0, 1], one value or one per year")
    if not 0 <= recovery_rate <= 1:
        raise ValueError("recovery_rate must be in [0, 1]")
    if spot_rates is None:
        if risk_free_rate is None:
            raise ValueError("give risk_free_rate or spot_rates")
        spot_rates = [risk_free_rate] * n
    if len(spot_rates) != n:
        raise ValueError("spot_rates needs one rate per year")
    dfs = [(1 + z) ** -(t + 1) for t, z in enumerate(spot_rates)]
    cfs = [face * coupon_rate] * n
    cfs[-1] += face
    rows, pos, cva = [], 1.0, 0.0
    for t in range(n):
        exposure = sum(cf * dfs[j] for j, cf in enumerate(cfs) if j >= t) / dfs[t]
        rec = recovery_rate * exposure
        lgd = exposure - rec
        pod = hz[t] * pos
        pos *= 1 - hz[t]
        el = lgd * pod
        cva += el * dfs[t]
        rows.append({"year": t + 1, "exposure": exposure, "recovery": rec, "lgd": lgd, "pod": pod, "pos": pos,
                     "expected_loss": el, "discount_factor": dfs[t], "pv_expected_loss": el * dfs[t]})
    vnd = sum(cf * df for cf, df in zip(cfs, dfs))
    fv = vnd - cva
    ytm = _solve(lambda y: _pv(cfs, y) - fv, what="yield")
    rf_ytm = _solve(lambda y: _pv(cfs, y) - vnd, what="risk-free yield")
    return {"cva": cva, "value_default_free": vnd, "fair_value": fv, "ytm": ytm, "credit_spread": ytm - rf_ytm,
            "cumulative_pod": 1 - pos, "schedule": rows}


@formula("fixed_income", formula="PD = 1 - e^(-lambda T) ; D = K e^(-lambda gamma T) P(T) ; EL = K(1 - e^(-lambda gamma T)) ; "
         "spread = lambda gamma", curriculum=L2_CR)
def credit_reduced_form_constant(face: float, default_intensity: float, loss_given_default: float, years: float,
                                 riskless_zero_price: float | None = None,
                                 riskless_rate: float | None = None) -> dict:
    """Reduced-form credit model with constant default intensity and loss given default (zero-coupon debt).

    Args:
        face: promised payment K. default_intensity: lambda per year. loss_given_default: gamma (fraction).
        years: T - t. riskless_zero_price: P(t,T) per 1 of face, or riskless_rate (continuous) to compute it.

    Returns:
        default_probability, expected_loss, pv_expected_loss, risky_debt_value, credit_spread (continuous).

    Example:
        K 700, lambda 0.01, gamma 0.4, 1y, P 0.96 -> PD 0.995%, EL 2.79, PV of EL 2.68
    """
    if riskless_zero_price is None:
        if riskless_rate is None:
            raise ValueError("give riskless_zero_price or riskless_rate")
        riskless_zero_price = math.exp(-riskless_rate * years)
    k = 1 - math.exp(-default_intensity * loss_given_default * years)
    return {"default_probability": 1 - math.exp(-default_intensity * years), "expected_loss": face * k,
            "pv_expected_loss": face * riskless_zero_price * k,
            "risky_debt_value": face * riskless_zero_price * (1 - k),
            "credit_spread": default_intensity * loss_given_default}


@formula("fixed_income", formula="Equity = A N(d1) - K e^(-rT) N(d2) ; PD = N(-e2) ; EL = K N(-e2) - A e^(uT) N(-e1) ; "
         "PV(EL) = K e^(-rT) N(-d2) - A N(-d1)", curriculum=L2_CR)
def credit_structural_merton(asset_value: float, debt_face: float, risk_free_rate: float, asset_volatility: float,
                             years: float, expected_asset_return: float | None = None) -> dict:
    """Structural (Merton) credit model: equity as a call on assets, debt = riskless debt minus a put.

    Args:
        asset_value: A_t. debt_face: K of the zero-coupon debt. risk_free_rate: r (continuous).
        asset_volatility: sigma. years: T - t. expected_asset_return: u (actual drift; default r).

    Returns:
        equity_value, debt_value, default_probability (actual, N(-e2)), risk_neutral_default_probability,
        expected_loss, pv_expected_loss, credit_spread (continuous yield spread of the debt).

    Example:
        A 1000, K 700, r 1%, u 3%, sigma 30%, 1y -> PD 12.74%, EL 11.78, PV(EL) 13.28
    """
    if min(asset_value, debt_face, asset_volatility, years) <= 0:
        raise ValueError("asset_value, debt_face, asset_volatility and years must be positive")
    u = risk_free_rate if expected_asset_return is None else expected_asset_return
    st = asset_volatility * math.sqrt(years)
    ln = math.log(asset_value / debt_face)
    d1 = (ln + risk_free_rate * years + 0.5 * asset_volatility ** 2 * years) / st
    e1 = (ln + u * years + 0.5 * asset_volatility ** 2 * years) / st
    d2, e2 = d1 - st, e1 - st
    pk = debt_face * math.exp(-risk_free_rate * years)
    equity = asset_value * _N(d1) - pk * _N(d2)
    debt = asset_value - equity
    return {"equity_value": equity, "debt_value": debt, "default_probability": _N(-e2),
            "risk_neutral_default_probability": _N(-d2),
            "expected_loss": debt_face * _N(-e2) - asset_value * math.exp(u * years) * _N(-e1),
            "pv_expected_loss": pk * _N(-d2) - asset_value * _N(-d1),
            "credit_spread": -math.log(debt / debt_face) / years - risk_free_rate}


@formula("fixed_income", formula="PV_risky = CF e^(-(y+s)t) ; PV_riskless = CF e^(-y t) ; PV(EL) = difference",
         curriculum=L2_CR)
def credit_pv_expected_loss_curve(cash_flows: list, times: list, risk_free_yields: list,
                                  credit_spreads: list) -> dict:
    """Present value of expected loss from term structures of risk-free yields and credit spreads (continuous).

    Args:
        cash_flows: promised payments. times: years to each payment.
        risk_free_yields: continuously compounded risk-free zero yields, decimals.
        credit_spreads: continuously compounded credit spreads, decimals.

    Returns:
        credit_adjusted_value, risk_free_value, pv_expected_loss, pv_risky (list), pv_riskless (list).

    Example:
        EUR25 in 3.6384y, rf 0.5160%, spread 0.3561% -> 24.2191 vs 24.5350, PV(EL) 0.3159
    """
    if not len(cash_flows) == len(times) == len(risk_free_yields) == len(credit_spreads):
        raise ValueError("all lists must have the same length")
    risky = [cf * math.exp(-(y + s) * t) for cf, t, y, s in zip(cash_flows, times, risk_free_yields, credit_spreads)]
    safe = [cf * math.exp(-y * t) for cf, t, y in zip(cash_flows, times, risk_free_yields)]
    return {"credit_adjusted_value": sum(risky), "risk_free_value": sum(safe),
            "pv_expected_loss": sum(safe) - sum(risky), "pv_risky": risky, "pv_riskless": safe}


@formula("fixed_income", formula="E[return] = sum_i p_i x (-ModDur x ds_i + 1/2 Conv x ds_i^2)", curriculum=L2_CR)
def credit_migration_expected_return(probabilities: list, spread_changes: list, modified_duration: float,
                                     convexity: float = 0.0) -> dict:
    """Expected instantaneous return impact of credit migration (rating transitions -> spread changes).

    Args:
        probabilities: transition probabilities to each rating (sum to 1, staying put included with ds=0).
        spread_changes: spread change (decimal) for each destination rating.
        modified_duration: (spread) duration of the bond. convexity: its convexity.

    Returns:
        expected_return, state_returns.

    Example:
        p = [0.9, 0.08, 0.02], ds = [0, +0.01, +0.03], D 7.2, C 62 -> about -0.0110
    """
    if len(probabilities) != len(spread_changes):
        raise ValueError("probabilities and spread_changes must match")
    if abs(sum(probabilities) - 1) > 1e-6:
        raise ValueError("probabilities must sum to 1")
    rets = [-modified_duration * ds + 0.5 * convexity * ds ** 2 for ds in spread_changes]
    return {"expected_return": sum(p * r for p, r in zip(probabilities, rets)), "state_returns": rets}


@formula("fixed_income", formula="Coverage = EBITDA/interest ; leverage = debt/EBITDA ; net = (debt - cash)/EBITDA ; "
         "FFO/debt ; debt/capital", curriculum=L1_CR)
def credit_ratios(ebitda: float, total_debt: float, interest_expense: float | None = None, cash: float = 0.0,
                  ffo: float | None = None, equity: float | None = None, ebit: float | None = None,
                  debt_tranches: list | None = None) -> dict:
    """Credit (capacity) ratios: coverage and leverage, including cumulative leverage through each debt tranche.

    Args:
        ebitda: EBITDA. total_debt: total debt. interest_expense: gross interest. cash: cash and equivalents.
        ffo: funds from operations. equity: book equity (for debt/capital). ebit: EBIT (EBIT coverage).
        debt_tranches: amounts by seniority, most senior first (cumulative leverage through each level).

    Returns:
        ebitda_coverage, ebit_coverage, debt_to_ebitda, net_debt_to_ebitda, ffo_to_debt, debt_to_capital,
        cumulative_leverage (list or None). Ratios with missing inputs are None.

    Example:
        Freescale: secured 4,899 + senior 1,948 + sub 764 over EBITDA 990 -> 4.9x, 6.9x, 7.7x; net 6.6x
    """
    if ebitda == 0:
        raise ValueError("ebitda must be nonzero")

    def div(a, b):
        return None if a is None or b in (None, 0) else a / b

    cum, run = None, 0.0
    if debt_tranches:
        cum = []
        for d in debt_tranches:
            run += d
            cum.append(run / ebitda)
    return {"ebitda_coverage": div(ebitda, interest_expense), "ebit_coverage": div(ebit, interest_expense),
            "debt_to_ebitda": total_debt / ebitda, "net_debt_to_ebitda": (total_debt - cash) / ebitda,
            "ffo_to_debt": div(ffo, total_debt),
            "debt_to_capital": None if equity is None else total_debt / (total_debt + equity),
            "cumulative_leverage": cum}
