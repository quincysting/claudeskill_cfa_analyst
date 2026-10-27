"""Private wealth management: IPS return requirement, taxes, estate planning (CFA L3 SS4).

Topic ``pwm``. Decimals throughout; money in any currency.
"""
from __future__ import annotations


import numpy as np

from ._registry import formula

IND = "L3 PWM: Managing Individual Investor Portfolios"
TAX = "L3 PWM: Taxes and Private Wealth Management in a Global Context"
EST = "L3 PWM: Estate Planning in a Global Context"


def _arr(x, name: str) -> np.ndarray:
    a = np.atleast_1d(np.asarray(x, dtype=float))
    if a.ndim != 1 or a.size == 0:
        raise ValueError(f"{name} must be a non-empty list of numbers")
    return a


# ---------------------------------------------------------------- IPS return requirement
@formula("pwm", formula="real = (spending - other income)/(portfolio - reserved); nominal = real + i + fees (additive) "
                        "or (1+real)(1+i)(1+fees) - 1 (multiplicative); pretax = after-tax/(1 - t)",
         curriculum=IND + "; L3 Institutional: Managing Institutional Investor Portfolios")
def ips_required_return(spending_need: float | None = None, portfolio_value: float | None = None,
                        spending_rate: float | None = None, inflation: float = 0.0, fees: float = 0.0,
                        tax_rate: float = 0.0, other_income: float = 0.0, reserved_outflows: float = 0.0,
                        method: str = "additive") -> dict:
    """Required return for an IPS from spending needs, inflation, costs and taxes.

    Give `spending_need` and `portfolio_value` (money), or `spending_rate` directly.

    Args:
        spending_need: annual (after-tax) spending to be met from the portfolio.
        portfolio_value: investable assets.
        spending_rate: alternative to the two above (e.g. a foundation's 5% payout).
        inflation: expected inflation (added to preserve real value).
        fees: annual investment costs as a fraction of assets.
        tax_rate: rate used to gross an after-tax requirement up to pre-tax (0 for tax-exempt investors).
        other_income: income from outside the portfolio (salary, pension) that reduces the need.
        reserved_outflows: near-term lump sums (house, gift, emergency reserve) removed from the investable base.
        method: "additive" (common for individuals) or "multiplicative" (exact compounding; common for
            endowments/foundations).

    Returns:
        required_return_pretax (headline), required_return_after_tax, real_spending_rate, investable_base, net_need.

    Example:
        ips_required_return(spending_need=120000, portfolio_value=2000000, inflation=0.03, tax_rate=0.30)
        -> after-tax 0.09, pretax 0.128571
    """
    if method not in ("additive", "multiplicative"):
        raise ValueError("method must be 'additive' or 'multiplicative'")
    if not 0 <= tax_rate < 1:
        raise ValueError("tax_rate must be in [0, 1)")
    out = {}
    if spending_rate is None:
        if spending_need is None or portfolio_value is None:
            raise ValueError("give spending_rate, or spending_need and portfolio_value")
        base = portfolio_value - reserved_outflows
        if base <= 0:
            raise ValueError("portfolio_value net of reserved_outflows must be > 0")
        need = spending_need - other_income
        spending_rate = need / base
        out.update(investable_base=base, net_need=need)
    if method == "additive":
        at = spending_rate + inflation + fees
    else:
        at = (1 + spending_rate) * (1 + inflation) * (1 + fees) - 1
    return {"required_return_pretax": at / (1 - tax_rate), "required_return_after_tax": at,
            "real_spending_rate": spending_rate, **out}


@formula("pwm", formula="nominal = (1 + real)(1 + i) - 1 (approx. real + i); pretax = after-tax/(1 - t)", curriculum=IND + "; " + TAX)
def return_conversions(after_tax_return: float | None = None, pretax_return: float | None = None,
                       tax_rate: float = 0.0, inflation: float = 0.0) -> dict:
    """Convert between pre-tax and after-tax, and nominal and real, returns.

    Give `after_tax_return` (grossed up) or `pretax_return` (taxed). Inflation converts the after-tax nominal
    return to real (exact and approximate).

    Returns:
        pretax_return, after_tax_return, real_after_tax_exact, real_after_tax_approx.

    Example:
        return_conversions(after_tax_return=0.06, tax_rate=0.40) -> pretax_return 0.10
    """
    if not 0 <= tax_rate < 1:
        raise ValueError("tax_rate must be in [0, 1)")
    if (after_tax_return is None) == (pretax_return is None):
        raise ValueError("give exactly one of after_tax_return or pretax_return")
    if after_tax_return is None:
        after_tax_return = pretax_return * (1 - tax_rate)
    else:
        pretax_return = after_tax_return / (1 - tax_rate)
    return {"pretax_return": pretax_return, "after_tax_return": after_tax_return,
            "real_after_tax_exact": (1 + after_tax_return) / (1 + inflation) - 1,
            "real_after_tax_approx": after_tax_return - inflation}


# ---------------------------------------------------------------- taxes
_REGIMES = ("accrual", "deferred", "wealth", "blended", "tax_deferred_account", "tax_exempt")


@formula("pwm", formula="accrual [1+r(1-t)]^n; deferred (1+r)^n(1-t) + tB; wealth [(1+r)(1-tw)]^n; "
                        "blended (1+r*)^n(1-T*) + T* - (1-B)t_cg, r* = r(1 - p_i t_i - p_d t_d - p_g t_cg), "
                        "T* = t_cg(1 - p_i - p_d - p_g)/(1 - p_i t_i - p_d t_d - p_g t_cg); TDA (1+r)^n(1-T_n); exempt (1+r)^n",
         curriculum=TAX)
def tax_future_value(pretax_return: float, years: float, regime: str = "accrual", initial: float = 1.0,
                     tax_rate: float = 0.0, cost_basis_ratio: float = 1.0, p_interest: float = 0.0,
                     p_dividend: float = 0.0, p_realized_gain: float = 0.0, t_interest: float = 0.0,
                     t_dividend: float = 0.0, t_capital_gain: float = 0.0, upfront_tax_rate: float = 0.0) -> dict:
    """After-tax future value under the curriculum's tax regimes, with tax drag and accrual equivalents.

    Args:
        pretax_return: r, decimal.
        years: n.
        regime: "accrual" (annual tax on returns), "deferred" (capital gains taxed at the end), "wealth"
            (annual tax on value), "blended" (mix of interest/dividends/realized/deferred gains),
            "tax_deferred_account" (TDA: taxed at withdrawal at tax_rate = T_n), "tax_exempt" (Roth-type).
        initial: starting value.
        tax_rate: t_i (accrual), t_cg (deferred), t_w (wealth) or T_n (TDA).
        cost_basis_ratio: B, cost basis / current market value (deferred and blended).
        p_interest, p_dividend, p_realized_gain: blended: shares of return taxed annually as interest, dividends
            and realised gains (the rest is deferred gain).
        t_interest, t_dividend, t_capital_gain: blended tax rates.
        upfront_tax_rate: tax-exempt account funded with after-tax money: multiplies by (1 - T_0).

    Returns:
        future_value, future_value_no_tax, tax_drag_amount, tax_drag_pct (share of the pre-tax gain lost),
        accrual_equivalent_return, accrual_equivalent_tax_rate, plus r_star and T_star for blended.

    Example:
        tax_future_value(initial=100000, pretax_return=0.07, years=20, regime="accrual", tax_rate=0.20)
        -> future_value 297357 (book: EUR297,357)
    """
    if regime not in _REGIMES:
        raise ValueError(f"regime must be one of {_REGIMES}")
    if years <= 0:
        raise ValueError("years must be > 0")
    r, n, t, B = pretax_return, years, tax_rate, cost_basis_ratio
    out = {}
    if regime == "accrual":
        f = (1 + r * (1 - t)) ** n
    elif regime == "deferred":
        f = (1 + r) ** n * (1 - t) + t * B
    elif regime == "wealth":
        f = ((1 + r) * (1 - t)) ** n
    elif regime == "tax_deferred_account":
        f = (1 + r) ** n * (1 - t)
    elif regime == "tax_exempt":
        f = (1 + r) ** n * (1 - upfront_tax_rate)
    else:
        p_def = 1 - p_interest - p_dividend - p_realized_gain
        if p_def < -1e-12:
            raise ValueError("p_interest + p_dividend + p_realized_gain must not exceed 1")
        k = 1 - p_interest * t_interest - p_dividend * t_dividend - p_realized_gain * t_capital_gain
        r_star = r * k
        T_star = t_capital_gain * p_def / k
        f = (1 + r_star) ** n * (1 - T_star) + T_star - (1 - B) * t_capital_gain
        out.update(r_star=r_star, T_star=T_star)
    fv = initial * f
    fv0 = initial * (1 + r) ** n
    gain0 = fv0 - initial
    rae = f ** (1 / n) - 1
    out = {"future_value": fv, "future_value_no_tax": fv0, "tax_drag_amount": fv0 - fv,
           "tax_drag_pct": (fv0 - fv) / gain0 if gain0 else float("nan"), "accrual_equivalent_return": rae,
           "accrual_equivalent_tax_rate": 1 - rae / r if r else float("nan"), **out}
    return out


@formula("pwm", formula="savings now = t x harvested loss; hold vs harvest after-tax values after growth",
         curriculum=TAX + " (tax-loss harvesting)")
def tax_loss_harvesting(market_value: float, cost_basis: float, tax_rate: float,
                        other_realized_gains: float | None = None, growth_multiple: float | None = None,
                        reinvest_savings: bool = True) -> dict:
    """Value of realising a loss now to offset gains, and the later after-tax comparison (hold vs harvest).

    Args:
        market_value: current value of the position with the unrealised loss.
        cost_basis: its tax basis (> market_value for a loss).
        tax_rate: capital gains tax rate.
        other_realized_gains: gains this year the loss can offset (caps the usable loss if given).
        growth_multiple: optional growth of the position until final sale (e.g. 1.8181).
        reinvest_savings: harvest case reinvests the tax saved in the same asset.

    Returns:
        tax_savings_now, usable_loss; with growth_multiple: hold_after_tax, harvest_after_tax, hold_tax,
        harvest_tax, advantage.

    Example:
        tax_loss_harvesting(market_value=110000, cost_basis=170000, tax_rate=0.20, growth_multiple=1.8181)
        -> savings 12000; hold after-tax ~194,000 vs harvest ~201,846
    """
    if not 0 <= tax_rate < 1:
        raise ValueError("tax_rate must be in [0, 1)")
    loss = max(0.0, cost_basis - market_value)
    usable = loss if other_realized_gains is None else min(loss, max(0.0, other_realized_gains))
    savings = tax_rate * usable
    out = {"tax_savings_now": savings, "usable_loss": usable}
    if growth_multiple is not None:
        hold_v = market_value * growth_multiple
        hold_tax = tax_rate * (hold_v - cost_basis)
        invested = market_value + (savings if reinvest_savings else 0.0)
        harv_v = invested * growth_multiple
        harv_tax = tax_rate * (harv_v - invested)
        out.update(hold_value=hold_v, hold_tax=hold_tax, hold_after_tax=hold_v - hold_tax, harvest_value=harv_v,
                   harvest_tax=harv_tax, harvest_after_tax=harv_v - harv_tax,
                   advantage=(harv_v - harv_tax) - (hold_v - hold_tax))
    return out


@formula("pwm", formula="after-tax value = V (1 - T_n) for TDAs, V - t (V - basis) for taxable, V for tax-exempt",
         curriculum=TAX + " (after-tax asset allocation)")
def after_tax_asset_allocation(pretax_values: list, tax_rates: list, cost_bases: list | None = None,
                               names: list | None = None) -> dict:
    """Restate holdings at after-tax value and compare pre-tax and after-tax asset weights.

    Args:
        pretax_values: market value of each holding/account.
        tax_rates: tax due on liquidation: T_n for a tax-deferred account, t_cg for a taxable holding (with
            cost_bases), 0 for tax-exempt.
        cost_bases: optional basis per holding; when given, tax applies only to the gain (V - basis).

    Returns:
        after_tax_values, pretax_weights, after_tax_weights, total_pretax, total_after_tax.

    Example:
        after_tax_asset_allocation(pretax_values=[1500000, 500000], tax_rates=[0.40, 0.0])
        -> after_tax_weights [0.643, 0.357]
    """
    v = _arr(pretax_values, "pretax_values")
    t = _arr(tax_rates, "tax_rates")
    if v.size != t.size:
        raise ValueError("pretax_values and tax_rates must have the same length")
    if cost_bases is None:
        at = v * (1 - t)
    else:
        b = _arr(cost_bases, "cost_bases")
        if b.size != v.size:
            raise ValueError("cost_bases must match pretax_values")
        at = v - t * (v - b)
    out = {"after_tax_values": at.tolist(), "pretax_weights": (v / v.sum()).tolist(),
           "after_tax_weights": (at / at.sum()).tolist(), "total_pretax": float(v.sum()),
           "total_after_tax": float(at.sum())}
    if names is not None:
        out["names"] = list(names)
    return out


# ---------------------------------------------------------------- estate planning
@formula("pwm", formula="p_joint = p1 + p2 - p1 p2; core capital = sum p_j x spending_j/(1 + r)^j; excess = assets - core",
         curriculum=EST + " (core capital)")
def estate_core_capital(spending: float | list, survival_probs: list, discount_rate: float,
                        survival_probs_2: list | None = None, spending_growth: float = 0.0,
                        total_assets: float | None = None) -> dict:
    """Core capital as the survival-weighted PV of spending needs, and excess capital.

    Args:
        spending: year-1 spending (grown at spending_growth), or a list of spending per year.
        survival_probs: probability the (first) person is alive in each year 1..N.
        discount_rate: real risk-free rate for real spending (nominal for nominal spending);
            real = (1 + nominal)/(1 + inflation) - 1.
        survival_probs_2: optional second person; the need lasts while either is alive (joint survival).
        spending_growth: annual growth of spending when `spending` is a single number.
        total_assets: optional, to report excess capital.

    Returns:
        core_capital, joint_survival_probs, spending_by_year, expected_spending, discounted, excess_capital.

    Example:
        Pearsall (book): real spending 75,000, real rate 3.7736%, five years of joint survival -> core capital 328,599.
    """
    p1 = _arr(survival_probs, "survival_probs")
    if np.any((p1 < 0) | (p1 > 1)):
        raise ValueError("survival probabilities must be between 0 and 1")
    if survival_probs_2 is not None:
        p2 = _arr(survival_probs_2, "survival_probs_2")
        if p2.size != p1.size:
            raise ValueError("survival_probs_2 must match survival_probs in length")
        p = p1 + p2 - p1 * p2
    else:
        p = p1
    n = p.size
    years = np.arange(1, n + 1)
    if np.ndim(spending) == 0:
        s = float(spending) * (1 + spending_growth) ** (years - 1)
    else:
        s = _arr(spending, "spending")
        if s.size != n:
            raise ValueError("spending list must have one value per survival probability")
    exp = p * s
    disc = exp / (1 + discount_rate) ** years
    out = {"core_capital": float(disc.sum()), "joint_survival_probs": p.tolist(), "spending_by_year": s.tolist(),
           "expected_spending": exp.tolist(), "discounted": disc.tolist()}
    if total_assets is not None:
        out["excess_capital"] = total_assets - out["core_capital"]
    return out


@formula("pwm", formula="RV = [1+r_g(1-t_ig)]^n x G / ([1+r_e(1-t_ie)]^n (1-T_e)); G = 1 (tax-free), 1-T_g (recipient pays), "
                        "1-T_g+T_g T_e (donor pays); charitable: [(1+r_g)^n + T_oi[1+r_e(1-t_ie)]^n(1-T_e)]/([1+r_e(1-t_ie)]^n(1-T_e))",
         curriculum=EST + " (lifetime gifts vs bequests)")
def gift_vs_bequest_relative_value(years: float, estate_tax_rate: float, gift_return: float,
                                   gift_investment_tax: float, bequest_return: float | None = None,
                                   bequest_investment_tax: float | None = None, gift_type: str = "tax_free",
                                   gift_tax_rate: float = 0.0, donor_income_tax_rate: float = 0.0) -> dict:
    """Relative after-tax value of a lifetime gift versus a bequest (RV > 1 favours giving now).

    Args:
        years: n until the bequest would occur.
        estate_tax_rate: T_e on the bequest.
        gift_return, gift_investment_tax: r_g and t_ig of the recipient investing the gift.
        bequest_return, bequest_investment_tax: r_e and t_ie of the donor keeping the asset (default = gift values).
        gift_type: "tax_free", "recipient_pays" (gift tax T_g paid by the recipient), "donor_pays" (gift tax paid by
            the donor; credit T_g x T_e because it shrinks the estate), or "charitable" (donor deducts the gift at
            donor_income_tax_rate T_oi; the charity's growth is untaxed).
        gift_tax_rate: T_g.
        donor_income_tax_rate: T_oi for charitable gifts.

    Returns:
        relative_value, fv_gift (per unit), fv_bequest (per unit).

    Example:
        gift_vs_bequest_relative_value(years=20, estate_tax_rate=0.60, gift_return=0.06, gift_investment_tax=0.20,
                                       bequest_return=0.06, bequest_investment_tax=0.48) -> 3.45 (book)
    """
    if gift_type not in ("tax_free", "recipient_pays", "donor_pays", "charitable"):
        raise ValueError("gift_type must be tax_free, recipient_pays, donor_pays or charitable")
    re_ = gift_return if bequest_return is None else bequest_return
    tie = gift_investment_tax if bequest_investment_tax is None else bequest_investment_tax
    fv_beq = (1 + re_ * (1 - tie)) ** years * (1 - estate_tax_rate)
    if gift_type == "charitable":
        fv_gift = (1 + gift_return) ** years + donor_income_tax_rate * fv_beq
    else:
        g = {"tax_free": 1.0, "recipient_pays": 1 - gift_tax_rate,
             "donor_pays": 1 - gift_tax_rate + gift_tax_rate * estate_tax_rate}[gift_type]
        fv_gift = (1 + gift_return * (1 - gift_investment_tax)) ** years * g
    return {"relative_value": fv_gift / fv_beq, "fv_gift": fv_gift, "fv_bequest": fv_beq}


@formula("pwm", formula="twice taxed: A(1+r)^n1(1-T1)(1+r)^n2(1-T2); skipped: A(1+r)^(n1+n2)(1-T); ratio 1/(1-T1) at equal rates",
         curriculum=EST + " (generation-skipping)")
def generation_skipping_value(amount: float, real_return: float, years_to_first_transfer: float,
                              years_to_second_transfer: float, tax_rate_first: float,
                              tax_rate_second: float | None = None, tax_rate_skip: float | None = None) -> dict:
    """Wealth reaching the grandchildren via two taxed transfers versus one generation-skipping transfer.

    Args:
        amount: capital today.
        real_return: growth rate.
        years_to_first_transfer: years until the transfer to the children.
        years_to_second_transfer: years from today until the transfer to the grandchildren.
        tax_rate_first, tax_rate_second: transfer tax rates at each step (second defaults to first).
        tax_rate_skip: rate on the skipping transfer (defaults to tax_rate_second).

    Returns:
        value_two_transfers, value_skipped, relative_value.

    Example:
        generation_skipping_value(amount=200e6, real_return=0.05, years_to_first_transfer=10,
                                  years_to_second_transfer=35, tax_rate_first=0.5) -> 275,800,768 vs 551,601,537
    """
    if years_to_second_transfer < years_to_first_transfer:
        raise ValueError("years_to_second_transfer must be >= years_to_first_transfer")
    t2 = tax_rate_first if tax_rate_second is None else tax_rate_second
    ts = t2 if tax_rate_skip is None else tax_rate_skip
    g = 1 + real_return
    two = amount * g ** years_to_first_transfer * (1 - tax_rate_first) * g ** (
        years_to_second_transfer - years_to_first_transfer) * (1 - t2)
    skip = amount * g ** years_to_second_transfer * (1 - ts)
    return {"value_two_transfers": two, "value_skipped": skip, "relative_value": skip / two}


@formula("pwm", formula="credit: max(T_res, T_src); exemption: T_src; deduction: T_res + T_src - T_res T_src",
         curriculum=EST + " (double taxation relief)")
def double_taxation_rate(residence_rate: float, source_rate: float) -> dict:
    """Effective tax rate under the credit, exemption and deduction methods of double-taxation relief.

    Returns:
        credit_method, exemption_method, deduction_method (total rates), and the split between countries.

    Example:
        double_taxation_rate(residence_rate=0.50, source_rate=0.40) -> credit 0.50, exemption 0.40, deduction 0.70
    """
    ded = residence_rate + source_rate - residence_rate * source_rate
    return {"credit_method": max(residence_rate, source_rate), "exemption_method": source_rate,
            "deduction_method": ded, "credit_to_residence": max(0.0, residence_rate - source_rate),
            "credit_to_source": source_rate, "deduction_to_residence": ded - source_rate,
            "deduction_to_source": source_rate}
