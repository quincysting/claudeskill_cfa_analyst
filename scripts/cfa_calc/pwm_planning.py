"""PWM planning and institutional investors: goal-based required return, human capital, retirement, concentrated
positions, pensions, endowment spending rules, bank duration gap (CFA L3 SS4-SS5). Topic ``pwm``."""
from __future__ import annotations


import numpy as np
from scipy.optimize import brentq

from ._registry import formula
from .pwm import _arr

IND = "L3 PWM: Managing Individual Investor Portfolios"
LFA = "L3 PWM: Lifetime Financial Advice: Human Capital, Asset Allocation, and Insurance"
CSP = "L3 PWM: Concentrated Single-Asset Positions"
INST = "L3 Institutional: Managing Institutional Investor Portfolios"


def _whole(name: str, x, minimum: int = 1) -> int:
    """A whole number of years >= minimum; fractions raise instead of being truncated."""
    if int(x) != x or x < minimum:
        raise ValueError(f"{name} must be a whole number of years >= {minimum}, got {x!r}")
    return int(x)


def _fv_with_flows(r: float, pv: float, flow: float, n: int, growth: float, due: bool) -> float:
    """Value after n years of pv growing at r with annual flows (growing at `growth`), end- or start-of-year."""
    v = pv
    for t in range(n):
        f = flow * (1 + growth) ** t
        v = (v + f) * (1 + r) if due else v * (1 + r) + f
    return v


@formula("pwm", formula="solve r: PV(1+r)^n + sum CF_t (1+r)^(n-t) = target; pretax = r/(1 - t)",
         curriculum=IND + "; " + INST + " (return objectives)")
def ips_required_return_for_goal(portfolio_value: float, target_value: float, years: float,
                                 annual_cash_flow: float = 0.0, cash_flow_growth: float = 0.0,
                                 flows_at_start: bool = False, tax_rate: float = 0.0) -> dict:
    """Return needed to grow a portfolio to a goal given annual contributions (+) or withdrawals (-).

    Covers goal-based IPS questions (accumulate X by year n while spending Y a year) and pension/trust funding
    targets. Whole years only when there are cash flows.

    Args:
        portfolio_value: investable assets today.
        target_value: value needed at the horizon (e.g. a pledge, a retirement fund, liabilities).
        years: horizon n.
        annual_cash_flow: yearly flow (+ contribution, - withdrawal), first flow at year 1 (or year 0 if
            flows_at_start).
        cash_flow_growth: annual growth of the flow (e.g. inflation).
        tax_rate: gross the after-tax answer up to pre-tax.

    Returns:
        required_return_after_tax, required_return_pretax.

    Example:
        ips_required_return_for_goal(portfolio_value=1235000, target_value=2000000, years=18,
                                     annual_cash_flow=-26000, tax_rate=0.40) -> 4.427% after tax, 7.38% pretax
    """
    if portfolio_value <= 0 or years <= 0:
        raise ValueError("portfolio_value and years must be > 0")
    if not 0 <= tax_rate < 1:
        raise ValueError("tax_rate must be in [0, 1)")
    if annual_cash_flow == 0:
        r = (target_value / portfolio_value) ** (1 / years) - 1
    else:
        n = _whole("years (with annual cash flows)", years)
        f = lambda x: _fv_with_flows(x, portfolio_value, annual_cash_flow, n, cash_flow_growth,  # noqa: E731
                                     flows_at_start) - target_value
        lo, hi = -0.99, 1.0
        if f(lo) * f(hi) > 0:
            raise ValueError("goal cannot be reached for returns between -99% and 100%")
        r = brentq(f, lo, hi, xtol=1e-14)
    return {"required_return_after_tax": r, "required_return_pretax": r / (1 - tax_rate)}


# ---------------------------------------------------------------- human capital and retirement
@formula("pwm", formula="HC = sum_t p_t x income_0 (1+g)^t / (1 + r_f + v)^t; economic net worth = HC + financial capital",
         curriculum=LFA + " (human capital)")
def human_capital_value(annual_income: float, years_to_retirement: int, risk_free_rate: float,
                        occupational_risk_premium: float = 0.0, income_growth: float = 0.0,
                        survival_probs: list | None = None, financial_capital: float = 0.0,
                        other_assets: float = 0.0, liabilities: float = 0.0) -> dict:
    """Present value of future labour income (human capital) and the holistic (economic) balance sheet.

    Args:
        annual_income: current annual (after-tax) earnings; year-1 income = annual_income x (1 + g).
        years_to_retirement: number of future earning years.
        risk_free_rate: nominal (or real, with real growth) risk-free rate.
        occupational_risk_premium: v, higher for volatile/equity-like income.
        income_growth: g.
        survival_probs: probability of being alive to earn in each year (defaults to 1).
        financial_capital, other_assets, liabilities: for economic net worth.

    Returns:
        human_capital, pv_by_year, discount_rate, economic_net_worth, human_capital_share.

    Example:
        human_capital_value(annual_income=100000, years_to_retirement=2, risk_free_rate=0.04,
                            occupational_risk_premium=0.02, income_growth=0.03)
        -> 100000(1.03)/1.06 + 100000(1.03)^2/1.06^2
    """
    n = _whole("years_to_retirement", years_to_retirement, 0)
    t = np.arange(1, n + 1)
    p = np.ones(n) if survival_probs is None else _arr(survival_probs, "survival_probs")
    if p.size != n:
        raise ValueError("one survival probability per earning year")
    k = risk_free_rate + occupational_risk_premium
    pv = p * annual_income * (1 + income_growth) ** t / (1 + k) ** t
    hc = float(pv.sum())
    enw = hc + financial_capital + other_assets - liabilities
    return {"human_capital": hc, "pv_by_year": pv.tolist(), "discount_rate": k, "economic_net_worth": enw,
            "human_capital_share": hc / (hc + financial_capital + other_assets) if hc + financial_capital + other_assets else float("nan")}


@formula("pwm", formula="PV = PMT x [1 - (1+r)^-n]/r (x (1+r) if paid in advance); PMT = PV / annuity factor; "
                        "growing: PMT_t = PMT_1 (1+g)^(t-1)", curriculum=LFA + " (annuities, retirement income)")
def retirement_annuity(rate: float, years: int, payment: float | None = None, present_value: float | None = None,
                       payment_growth: float = 0.0, in_advance: bool = False) -> dict:
    """Capital needed for (or sustainable payment from) a retirement income stream: annuity value or withdrawal.

    Give `payment` to get the capital needed (annuity purchase value), or `present_value` to get the sustainable
    level (first-year) payment.

    Returns:
        present_value, payment, annuity_factor, withdrawal_rate (payment / present value).

    Example:
        retirement_annuity(rate=0.05, years=25, payment=50000) -> present_value 704,697
    """
    if (payment is None) == (present_value is None):
        raise ValueError("give exactly one of payment or present_value")
    n = _whole("years", years)
    t = np.arange(1, n + 1) - (1 if in_advance else 0)
    factor = float(np.sum((1 + payment_growth) ** np.arange(n) / (1 + rate) ** t))
    if payment is None:
        payment = present_value / factor
    else:
        present_value = payment * factor
    return {"present_value": present_value, "payment": payment, "annuity_factor": factor,
            "withdrawal_rate": payment / present_value}


@formula("pwm", formula="W_(t+1) = (W_t - spending_t)(1 + R_t), R_t ~ N(mu, sd); success = W stays > 0 through the horizon",
         curriculum=LFA + "; L3 Estate Planning (Monte Carlo core capital)")
def retirement_monte_carlo(initial_wealth: float, annual_spending: float, mean_return: float, std_dev: float,
                           years: int, spending_growth: float = 0.0, spending_at_start: bool = True,
                           n_sims: int = 10000, seed: int = 42) -> dict:
    """Monte Carlo probability that a retirement portfolio sustains its spending (seeded, deterministic).

    Args:
        initial_wealth: portfolio at retirement.
        annual_spending: first-year withdrawal.
        mean_return, std_dev: annual (arithmetic) return distribution, normal.
        years: horizon (e.g. to age 95).
        spending_growth: growth of withdrawals (inflation) when returns are nominal.
        spending_at_start: withdraw at the start of each year (else at year end).

    Returns:
        probability_of_success, probability_of_ruin_by_year (cumulative), median_terminal_wealth,
        percentile_10_terminal_wealth.

    Example:
        retirement_monte_carlo(initial_wealth=1e6, annual_spending=50000, mean_return=0.0614, std_dev=0.13,
                               years=30, spending_growth=0.03)
    """
    if n_sims < 100:
        raise ValueError("n_sims must be >= 100")
    n = _whole("years", years)
    rng = np.random.default_rng(seed)
    R = rng.normal(mean_return, std_dev, size=(n_sims, n))
    W = np.full(n_sims, float(initial_wealth))
    alive = np.ones(n_sims, dtype=bool)
    ruin = []
    for t in range(n):
        s = annual_spending * (1 + spending_growth) ** t
        W = (W - s) * (1 + R[:, t]) if spending_at_start else W * (1 + R[:, t]) - s
        alive &= W > 0
        W = np.where(alive, W, 0.0)
        ruin.append(float(1 - alive.mean()))
    return {"probability_of_success": float(alive.mean()), "probability_of_ruin_by_year": ruin,
            "median_terminal_wealth": float(np.median(W)), "percentile_10_terminal_wealth": float(np.percentile(W, 10))}


# ---------------------------------------------------------------- concentrated positions
@formula("pwm", formula="floor = shares x (K_put - net premium); cap = shares x (K_call - net premium); "
                        "loan = LTV x hedged value; after-tax sale = V x fraction x (1 - t) + t x basis share",
         curriculum=CSP + " (collars, monetization, outright sale)")
def concentrated_position_strategies(shares: float, price: float, put_strike: float | None = None,
                                     call_strike: float | None = None, put_premium: float = 0.0,
                                     call_premium: float = 0.0, loan_to_value: float | None = None,
                                     tax_rate: float = 0.0, cost_basis: float = 0.0, fraction_sold: float = 1.0,
                                     terminal_prices: list | None = None, annual_cost_rate: float = 0.0) -> dict:
    """Compare hedging/monetizing a concentrated stock position with selling it.

    Args:
        shares, price: position size and current price.
        put_strike, call_strike, put_premium, call_premium: collar terms per share (protective put: no call).
        loan_to_value: cash raised as a fraction of the current position value (e.g. 0.99 short against the box,
            0.88 prepaid variable forward; with a collar lenders often advance about the floor / price).
        tax_rate, cost_basis: capital gains tax and total basis, for the after-tax outright sale.
        fraction_sold: share of the position sold outright.
        terminal_prices: optional prices at expiry to value the collared position.
        annual_cost_rate: carrying cost of the monetization (e.g. 0.003 = 30 bp/yr).

    Returns:
        position_value, net_premium, floor_value, cap_value, monetization_proceeds, annual_carry_cost,
        after_tax_sale_proceeds, sale_tax, hedged_values_at_terminal (if terminal_prices).

    Example:
        concentrated_position_strategies(shares=100000, price=40, put_strike=36, call_strike=48, put_premium=3,
                                         call_premium=3, terminal_prices=[42]) -> zero-cost collar, value 4.2m
    """
    if shares <= 0 or price <= 0:
        raise ValueError("shares and price must be > 0")
    v = shares * price
    net = put_premium - call_premium
    out = {"position_value": v, "net_premium": net * shares}
    floor = None
    if put_strike is not None:
        floor = shares * (put_strike - net)
        out["floor_value"] = floor
    if call_strike is not None:
        out["cap_value"] = shares * (call_strike - net)
    if loan_to_value is not None:
        out["monetization_proceeds"] = loan_to_value * v
        out["annual_carry_cost"] = annual_cost_rate * out["monetization_proceeds"]
    sold = v * fraction_sold
    tax = tax_rate * max(0.0, sold - cost_basis * fraction_sold)
    out.update(after_tax_sale_proceeds=sold - tax, sale_tax=tax)
    if terminal_prices is not None:
        st = _arr(terminal_prices, "terminal_prices")
        lo = put_strike if put_strike is not None else -np.inf
        hi = call_strike if call_strike is not None else np.inf
        out["hedged_values_at_terminal"] = (shares * (np.clip(st, lo, hi) - net)).tolist()
    return out


# ---------------------------------------------------------------- institutional investors
@formula("pwm", formula="PV(L) = sum B_t/(1 + r)^t; surplus = assets - PV(L); funded ratio = assets/PV(L)",
         curriculum=INST + " (defined benefit pension plans)")
def pension_funded_status(plan_assets: float, liabilities: float | None = None, benefit_payments: list | None = None,
                          discount_rate: float | None = None) -> dict:
    """Pension surplus and funded ratio, with liabilities given or valued from projected benefit payments.

    Returns:
        liabilities, surplus, funded_ratio, status ("overfunded"/"fully funded"/"underfunded").

    Example:
        pension_funded_status(plan_assets=750e6, liabilities=950e6) -> surplus -200m, funded_ratio 0.789
    """
    if liabilities is None:
        if benefit_payments is None or discount_rate is None:
            raise ValueError("give liabilities, or benefit_payments and discount_rate")
        b = _arr(benefit_payments, "benefit_payments")
        liabilities = float(np.sum(b / (1 + discount_rate) ** np.arange(1, b.size + 1)))
    if liabilities <= 0:
        raise ValueError("liabilities must be > 0")
    s = plan_assets - liabilities
    status = "overfunded" if s > 1e-9 else "underfunded" if s < -1e-9 else "fully funded"
    return {"liabilities": liabilities, "surplus": s, "funded_ratio": plan_assets / liabilities, "status": status}


@formula("pwm", formula="simple: s x MV_(t-1); rolling: s x avg(MV_(t-1..t-3)); "
                        "geometric: R x S_(t-1)(1 + I_(t-1)) + (1 - R) x s x MV_(t-1)",
         curriculum=INST + " (foundation/endowment spending rules)")
def endowment_spending_rule(spending_rate: float, market_values: list, rule: str = "rolling",
                            smoothing_rate: float = 0.7, prior_spending: float | None = None,
                            inflation: float = 0.0, window: int = 3) -> dict:
    """Spending for the coming year under the simple, rolling-average or geometric-smoothing rule.

    Args:
        spending_rate: s, e.g. 0.05.
        market_values: past market values, oldest first; the last is MV_(t-1).
        rule: "simple", "rolling" (average of the last `window` values) or "geometric".
        smoothing_rate: R, weight on last year's inflation-adjusted spending (geometric rule).
        prior_spending: S_(t-1), required for the geometric rule.
        inflation: I_(t-1), for the geometric rule (0 reproduces rules quoted without inflation).

    Returns:
        spending, spending_rate_effective (spending / latest value).

    Example:
        endowment_spending_rule(spending_rate=0.05, market_values=[90, 100, 110]) -> spending 5.0
    """
    mv = _arr(market_values, "market_values")
    if rule == "simple":
        s = spending_rate * mv[-1]
    elif rule == "rolling":
        if mv.size < window:
            raise ValueError(f"rolling rule needs {window} market values")
        s = spending_rate * float(mv[-window:].mean())
    elif rule == "geometric":
        if prior_spending is None:
            raise ValueError("geometric rule needs prior_spending")
        s = smoothing_rate * prior_spending * (1 + inflation) + (1 - smoothing_rate) * spending_rate * mv[-1]
    else:
        raise ValueError("rule must be 'simple', 'rolling' or 'geometric'")
    return {"spending": float(s), "spending_rate_effective": float(s / mv[-1])}


@formula("pwm", formula="LADG = D_A - (L/A) D_L; dE ~ -LADG x A x dy/(1 + y)", curriculum=INST + " (banks)")
def bank_leverage_adjusted_duration_gap(asset_duration: float, liability_duration: float, assets: float,
                                        liabilities: float, rate_change: float = 0.0, yield_level: float = 0.0) -> dict:
    """Leverage-adjusted duration gap and the approximate change in the market value of equity.

    Args:
        asset_duration, liability_duration: (modified) durations; with Macaulay durations give yield_level.
        assets, liabilities: market values.
        rate_change: parallel shift in rates (0.01 = +100 bp).
        yield_level: y for the Macaulay adjustment dy/(1 + y); leave 0 for modified durations.

    Returns:
        leverage_ratio (L/A), duration_gap, equity_value_change, equity_value.

    Example:
        bank_leverage_adjusted_duration_gap(asset_duration=5, liability_duration=2, assets=100, liabilities=90,
                                            rate_change=0.01) -> gap 3.2, equity change -3.2
    """
    if assets <= 0:
        raise ValueError("assets must be > 0")
    k = liabilities / assets
    gap = asset_duration - k * liability_duration
    return {"leverage_ratio": k, "duration_gap": gap,
            "equity_value_change": -gap * assets * rate_change / (1 + yield_level),
            "equity_value": assets - liabilities}
