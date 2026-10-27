"""Alternative investments: real estate and REITs, private equity (waterfalls, fund NAV, VC method),
hedge fund fees, commodity futures returns and appraisal unsmoothing (Levels I-III).

All rates are decimals, all functions return dicts of unrounded values (see references/conventions.md).
"""
from __future__ import annotations

import math
from typing import Sequence

from ._registry import formula

T = "alternatives"


# ----------------------------------------------------------------------------------------------- helpers
def _need(value, name: str):
    if value is None:
        raise ValueError(f"{name} is required")
    return value


def _npv(rate: float, flows: Sequence[float]) -> float:
    return sum(cf / (1 + rate) ** t for t, cf in enumerate(flows))


def _irr(flows: Sequence[float]) -> float:
    """IRR of periodic cash flows (first at t = 0): bracket the first sign change on a grid, then bisect."""
    if not any(cf < 0 for cf in flows) or not any(cf > 0 for cf in flows):
        raise ValueError("IRR needs at least one negative and one positive cash flow")
    grid = [-0.99 + 0.01 * k for k in range(100)] + [0.01 * k for k in range(1, 1001)]
    lo, flo = grid[0], _npv(grid[0], flows)
    for hi in grid[1:]:
        fhi = _npv(hi, flows)
        if flo == 0:
            return lo
        if flo * fhi < 0:
            for _ in range(200):
                mid = (lo + hi) / 2
                fm = _npv(mid, flows)
                if flo * fm <= 0:
                    hi = mid
                else:
                    lo, flo = mid, fm
            return (lo + hi) / 2
        lo, flo = hi, fhi
    raise ValueError("no IRR found between -99% and 1000%")


def _annuity_pv(payment: float, rate: float, n: int) -> float:
    return payment * n if rate == 0 else payment * (1 - (1 + rate) ** -n) / rate


# ----------------------------------------------------------------------------------------------- real estate
@formula(T, formula="PGI = rent + other income; EGI = PGI(1 - vacancy); NOI = EGI - operating expenses",
         curriculum="L2 Alternatives R40 Private Real Estate Investments (net operating income)")
def real_estate_noi(potential_rental_income: float, other_income: float = 0.0, vacancy_rate: float = 0.0,
                    operating_expenses: float = 0.0, opex_pct_of_egi: float = 0.0,
                    management_fee_pct_of_egi: float = 0.0) -> dict:
    """Net operating income from potential gross income, vacancy and operating expenses (before debt service).

    Args:
        potential_rental_income: rent at full occupancy for the year.
        other_income: parking, laundry, etc. (included in PGI before vacancy, as in the curriculum).
        vacancy_rate: vacancy and collection loss as a fraction of PGI, decimal.
        operating_expenses: fixed-amount operating expenses (taxes, insurance, maintenance...).
        opex_pct_of_egi: operating expenses expressed as a fraction of EGI, decimal.
        management_fee_pct_of_egi: property management fee as a fraction of EGI, decimal.

    Returns:
        noi, potential_gross_income, vacancy_loss, effective_gross_income, total_operating_expenses.

    Example:
        50 units x 1,000 x 12 + 25,000 other, 10% vacancy, opex 40% + mgmt 10% of EGI -> NOI 281,250
        (CFA L2 R40 Example 12).
    """
    if not 0 <= vacancy_rate < 1:
        raise ValueError("vacancy_rate must be a decimal in [0, 1)")
    pgi = potential_rental_income + other_income
    egi = pgi * (1 - vacancy_rate)
    opex = operating_expenses + (opex_pct_of_egi + management_fee_pct_of_egi) * egi
    return {"noi": egi - opex, "potential_gross_income": pgi, "vacancy_loss": pgi - egi,
            "effective_gross_income": egi, "total_operating_expenses": opex}


@formula(T, formula="Value = NOI1 / cap rate; cap rate = discount rate - growth (constant growth)",
         curriculum="L2 Alternatives R40 Private Real Estate Investments (direct capitalization)")
def real_estate_direct_capitalization(noi: float, cap_rate: float | None = None, discount_rate: float | None = None,
                                      growth: float = 0.0) -> dict:
    """Direct capitalization: value = first-year NOI (or rent for an all-risks yield) / cap rate.

    Args:
        noi: first-year (stabilized) NOI, or rent when cap_rate is an all risks yield.
        cap_rate: going-in cap rate, decimal; if omitted, cap rate = discount_rate - growth.
        discount_rate: required total return (used only when cap_rate is omitted).
        growth: constant growth of NOI and value, decimal.

    Returns:
        value, cap_rate, implied_total_return (cap rate + growth).

    Example:
        NOI 250,000 at a 5% cap rate -> 5,000,000 (CFA L2 R40 Example 13).
    """
    if cap_rate is None:
        cap_rate = _need(discount_rate, "cap_rate or discount_rate") - growth
    if cap_rate <= 0:
        raise ValueError("cap rate must be positive (discount rate must exceed growth)")
    return {"value": noi / cap_rate, "cap_rate": cap_rate, "implied_total_return": cap_rate + growth}


@formula(T, formula="V = sum NOI_t/(1+r)^t + [NOI_{n+1}/terminal cap rate x (1 - selling costs)]/(1+r)^n",
         curriculum="L2 Alternatives R40 Private Real Estate Investments (DCF with terminal cap rate)")
def real_estate_dcf_value(noi_forecast: list, discount_rate: float, terminal_noi: float | None = None,
                          terminal_cap_rate: float | None = None, terminal_growth: float | None = None,
                          selling_cost_rate: float = 0.0, terminal_value: float | None = None) -> dict:
    """Property value as the PV of NOI over the holding period plus the PV of the resale (reversion).

    Args:
        noi_forecast: NOI for years 1..n.
        discount_rate: required return (IRR), decimal.
        terminal_noi: NOI in year n+1 (default NOI_n x (1 + terminal_growth)).
        terminal_cap_rate: residual cap rate; default discount_rate - terminal_growth.
        terminal_growth: growth after year n, decimal.
        selling_cost_rate: selling costs as a fraction of the resale price.
        terminal_value: resale price given directly (overrides the cap-rate calculation).

    Returns:
        value, pv_noi, resale_value, pv_resale, going_in_cap_rate, terminal_cap_rate.

    Example:
        NOI 100,000 x 5 years, year-6 NOI 120,000, r 12%, g 2% (terminal cap 10%) -> 1,041,390
        (CFA L2 R40 Example 16).
    """
    if not noi_forecast:
        raise ValueError("noi_forecast must not be empty")
    n, r = len(noi_forecast), discount_rate
    if terminal_value is None:
        g = terminal_growth or 0.0
        tcap = terminal_cap_rate if terminal_cap_rate is not None else r - g
        if tcap <= 0:
            raise ValueError("terminal cap rate must be positive")
        tnoi = terminal_noi if terminal_noi is not None else noi_forecast[-1] * (1 + g)
        terminal_value = tnoi / tcap
    else:
        tcap = terminal_cap_rate
    resale = terminal_value * (1 - selling_cost_rate)
    pv_noi = sum(x / (1 + r) ** t for t, x in enumerate(noi_forecast, start=1))
    pv_resale = resale / (1 + r) ** n
    v = pv_noi + pv_resale
    return {"value": v, "pv_noi": pv_noi, "resale_value": resale, "pv_resale": pv_resale,
            "going_in_cap_rate": noi_forecast[0] / v, "terminal_cap_rate": tcap}


@formula(T, formula="V = PV(term rent to review at term rate) + [ERV/ARY]/(1 + ARY)^n",
         curriculum="L2 Alternatives R40 Private Real Estate Investments (term and reversion valuation)")
def real_estate_term_and_reversion(term_rent: float, years_to_review: int, term_rate: float,
                                   estimated_rental_value: float, reversion_cap_rate: float,
                                   reversion_discount_rate: float | None = None) -> dict:
    """UK term-and-reversion valuation: contract rent to the next review, then ERV capitalized at the all risks yield.

    Args:
        term_rent: passing (contract) rent per year.
        years_to_review: whole years until the rent review.
        term_rate: discount rate for the (less risky) term rent, decimal.
        estimated_rental_value: ERV (market rent) from the review.
        reversion_cap_rate: all risks yield on comparable fully let properties, decimal.
        reversion_discount_rate: rate to discount the reversion (default = reversion_cap_rate).

    Returns:
        value, term_value, reversion_value_at_review, reversion_value.

    Example:
        rent 400,000 for 2 years at 4%, ERV 450,000 at 5% -> 8,917,703 (CFA L2 R40 Example 18).
    """
    if years_to_review < 0 or int(years_to_review) != years_to_review:
        raise ValueError(f"years_to_review must be a whole number of years, got {years_to_review!r}")
    dr = reversion_cap_rate if reversion_discount_rate is None else reversion_discount_rate
    term = _annuity_pv(term_rent, term_rate, int(years_to_review))
    at_review = estimated_rental_value / reversion_cap_rate
    rev = at_review / (1 + dr) ** years_to_review
    return {"value": term + rev, "term_value": term, "reversion_value_at_review": at_review, "reversion_value": rev}


@formula(T, formula="V = contract rent/cap rate + [(ERV - rent)/top-slice rate]/(1 + top-slice rate)^n",
         curriculum="L2 Alternatives R40 Private Real Estate Investments (layer method)")
def real_estate_layer_method(term_rent: float, estimated_rental_value: float, years_to_review: int,
                             bottom_slice_cap_rate: float, top_slice_cap_rate: float) -> dict:
    """UK layer (hardcore) method: contract rent in perpetuity plus the deferred incremental rent after review.

    Args:
        term_rent: current contract rent (bottom slice), valued in perpetuity.
        estimated_rental_value: ERV after the review; the top slice is ERV - term rent.
        years_to_review: years until the incremental rent starts.
        bottom_slice_cap_rate: cap rate for the secure contract rent, decimal.
        top_slice_cap_rate: cap (and discount) rate for the incremental rent, decimal.

    Returns:
        value, bottom_slice_value, top_slice_value.

    Example:
        rent 400,000 at 5%, ERV 450,000, top slice at 6%, review in 2 years -> 8,741,664 (CFA L2 R40 Example 19).
    """
    bottom = term_rent / bottom_slice_cap_rate
    top = (estimated_rental_value - term_rent) / top_slice_cap_rate / (1 + top_slice_cap_rate) ** years_to_review
    return {"value": bottom + top, "bottom_slice_value": bottom, "top_slice_value": top}


@formula(T, formula="GIM = sale price / gross income; value = GIM x subject gross income",
         curriculum="L2 Alternatives R40 Private Real Estate Investments (gross income multiplier)")
def real_estate_gross_income_multiplier(sale_price: float, gross_income: float,
                                        subject_gross_income: float | None = None) -> dict:
    """Gross income multiplier from a comparable sale and the implied value of a subject property.

    Args:
        sale_price: comparable's sale price.
        gross_income: comparable's first-year gross income.
        subject_gross_income: subject property's gross income (optional).

    Returns:
        gim, implied_value (if subject_gross_income).

    Example:
        price 1,000,000, gross income 125,000 -> GIM 8; subject income 100,000 -> 800,000.
    """
    if gross_income <= 0:
        raise ValueError("gross_income must be positive")
    gim = sale_price / gross_income
    out = {"gim": gim}
    if subject_gross_income is not None:
        out["implied_value"] = gim * subject_gross_income
    return out


@formula(T, formula="loan = min(LTV x value, PV of (NOI/DSCR) debt service); interest-only: (NOI/DSCR)/rate",
         curriculum="L2 Alternatives R40 Private Real Estate Investments (LTV and DSCR loan sizing)")
def real_estate_loan_sizing(appraised_value: float, noi: float, max_ltv: float, min_dscr: float,
                            interest_rate: float, amortization_years: float | None = None,
                            payments_per_year: int = 1) -> dict:
    """Maximum commercial mortgage given LTV and DSCR limits (the lower amount binds).

    Args:
        appraised_value: property value.
        noi: first-year NOI.
        max_ltv: maximum loan-to-value, decimal.
        min_dscr: minimum debt service coverage (NOI / annual debt service).
        interest_rate: annual mortgage rate, decimal.
        amortization_years: None for interest-only; else level-payment amortization term.
        payments_per_year: mortgage payments per year (amortizing loans).

    Returns:
        max_loan, loan_by_ltv, loan_by_dscr, binding_constraint, annual_debt_service, dscr, ltv.

    Example:
        value 5m, NOI 400,000, LTV 80%, DSCR 1.25, 8% interest-only -> 4,000,000 (CFA L2 R40 Example 30).
    """
    if min_dscr <= 0 or not 0 < max_ltv <= 1:
        raise ValueError("min_dscr must be positive and max_ltv in (0, 1]")
    if amortization_years is not None:
        n_pay = amortization_years * payments_per_year
        if n_pay < 1 or abs(n_pay - round(n_pay)) > 1e-9:
            raise ValueError(f"amortization_years x payments_per_year must be a whole number of payments, got {n_pay}")
    by_ltv = max_ltv * appraised_value
    max_ds = noi / min_dscr

    def debt_service(loan: float) -> float:
        if amortization_years is None:
            return loan * interest_rate
        n, i = int(round(amortization_years * payments_per_year)), interest_rate / payments_per_year
        return loan / _annuity_pv(1.0, i, n) * payments_per_year

    by_dscr = max_ds / debt_service(1.0)
    loan = min(by_ltv, by_dscr)
    ds = debt_service(loan)
    return {"max_loan": loan, "loan_by_ltv": by_ltv, "loan_by_dscr": by_dscr,
            "binding_constraint": "ltv" if by_ltv <= by_dscr else "dscr", "annual_debt_service": ds,
            "dscr": noi / ds if ds else math.inf, "ltv": loan / appraised_value}


@formula(T, formula="equity dividend rate = (NOI1 - DS1)/equity; IRR of -equity, NOI_t - DS_t, + sale - loan balance",
         curriculum="L2 Alternatives R40 Private Real Estate Investments (cash-on-cash, leveraged/unleveraged IRR)")
def real_estate_investment_returns(price: float, noi: list, sale_price: float, loan: float = 0.0,
                                   interest_rate: float = 0.0, debt_service: float | None = None,
                                   loan_balance_at_sale: float | None = None, selling_costs: float = 0.0) -> dict:
    """Equity dividend (cash-on-cash) rate, DSCR, LTV and leveraged vs unleveraged IRR for a property purchase.

    Args:
        price: purchase price.
        noi: annual NOI for years 1..n (sale at end of year n).
        sale_price: resale price at the end of year n.
        loan: mortgage amount at purchase.
        interest_rate: mortgage rate (interest-only debt service default = loan x rate).
        debt_service: annual debt service if not interest-only.
        loan_balance_at_sale: balance repaid at sale (default = loan, i.e. interest-only).
        selling_costs: costs deducted from the sale proceeds.

    Returns:
        leveraged_irr, unleveraged_irr, equity_dividend_rate, equity, dscr, ltv, annual_debt_service.

    Example:
        price 5m, NOI 400,000 x 5, sold 6m, 4m loan at 8% IO -> cash-on-cash 8%, leveraged IRR 21.14%,
        unleveraged 11.20% (CFA L2 R40 Examples 31-33).
    """
    if not noi:
        raise ValueError("noi must hold at least one year")
    ds = debt_service if debt_service is not None else loan * interest_rate
    bal = loan if loan_balance_at_sale is None else loan_balance_at_sale
    equity = price - loan
    if equity <= 0:
        raise ValueError("loan must be smaller than the price")
    net_sale = sale_price - selling_costs
    unlev = [-price] + list(noi)
    unlev[-1] += net_sale
    lev = [-equity] + [x - ds for x in noi]
    lev[-1] += net_sale - bal
    return {"leveraged_irr": _irr(lev), "unleveraged_irr": _irr(unlev), "equity_dividend_rate": (noi[0] - ds) / equity,
            "equity": equity, "dscr": noi[0] / ds if ds else math.inf, "ltv": loan / price,
            "annual_debt_service": ds}


@formula(T, formula="NAV = [(NOI - noncash rent + acquisition adj)(1 + g)]/cap rate + other assets - debt - "
                    "other liabilities; NAVPS = NAV/shares",
         curriculum="L2 Alternatives R41 Publicly Traded Real Estate Securities (net asset value per share)")
def reit_nav_per_share(noi: float, cap_rate: float, shares: float, noi_growth: float = 0.0,
                       noncash_rent: float = 0.0, acquisition_adjustment: float = 0.0, other_assets: float = 0.0,
                       debt: float = 0.0, other_liabilities: float = 0.0, price: float | None = None) -> dict:
    """REIT/REOC NAV per share by capitalizing next-12-month cash NOI at a market cap rate.

    Args:
        noi: last-12-month real estate NOI.
        cap_rate: market cap rate from comparable transactions, decimal.
        shares: shares outstanding.
        noi_growth: expected NOI growth for the next 12 months, decimal.
        noncash_rent: straight-line (non-cash) rent to remove.
        acquisition_adjustment: NOI to add for a full year of properties acquired during the year.
        other_assets: cash, receivables, land held at book, prepaid (exclude goodwill/intangibles).
        debt: debt (market value if materially different); other_liabilities: excl. deferred taxes.
        price: share price, for the premium/(discount) to NAV.

    Returns:
        navps, nav, property_value, next_12m_cash_noi, gross_asset_value, price_to_nav (if price).

    Example:
        NOI 115m capitalized at 7%, debt 505m, 100m shares -> NAVPS 11.38 (CFA L2 R41 Example 6).
    """
    if cap_rate <= 0 or shares <= 0:
        raise ValueError("cap_rate and shares must be positive")
    cash_noi = (noi - noncash_rent + acquisition_adjustment) * (1 + noi_growth)
    prop = cash_noi / cap_rate
    gross = prop + other_assets
    nav = gross - debt - other_liabilities
    out = {"navps": nav / shares, "nav": nav, "property_value": prop, "next_12m_cash_noi": cash_noi,
           "gross_asset_value": gross}
    if price is not None:
        out["price_to_nav"] = price / (nav / shares)
    return out


@formula(T, formula="FFO = NI + depreciation & amortization + deferred tax charges - gains + losses on property sales; "
                    "AFFO = FFO - non-cash rent - maintenance capex and leasing costs",
         curriculum="L2 Alternatives R41 Publicly Traded Real Estate Securities (FFO and AFFO)")
def reit_ffo_affo(net_income: float, depreciation_amortization: float, deferred_tax_charges: float = 0.0,
                  gains_on_sales: float = 0.0, losses_on_sales: float = 0.0, noncash_rent: float = 0.0,
                  maintenance_capex_and_leasing: float = 0.0, shares: float | None = None,
                  dividends_per_share: float | None = None) -> dict:
    """Funds from operations and adjusted funds from operations (per share and payout ratios optional).

    Args:
        net_income: net income available to common.
        depreciation_amortization: real estate depreciation and amortization.
        deferred_tax_charges: deferred portion of tax expense (added back).
        gains_on_sales, losses_on_sales: on property sales and debt restructuring (excluded).
        noncash_rent: straight-line rent adjustment (deducted for AFFO).
        maintenance_capex_and_leasing: recurring maintenance capex, leasing commissions, TI allowances.
        shares: weighted average shares, for per-share figures.
        dividends_per_share: for payout ratios on FFO and AFFO.

    Returns:
        ffo, affo, ffo_per_share, affo_per_share, payout_on_ffo, payout_on_affo (when inputs allow).

    Example:
        NI 160,638 + D&A 101,100 -> FFO 261,738; less 21,103 and 55,765 -> AFFO 184,870; per share 4.70 / 3.32
        on 55,689 shares (CFA L2 R41 Exhibit 8).
    """
    ffo = net_income + depreciation_amortization + deferred_tax_charges - gains_on_sales + losses_on_sales
    affo = ffo - noncash_rent - maintenance_capex_and_leasing
    out = {"ffo": ffo, "affo": affo}
    if shares:
        out.update(ffo_per_share=ffo / shares, affo_per_share=affo / shares)
        if dividends_per_share is not None:
            out.update(payout_on_ffo=dividends_per_share / out["ffo_per_share"],
                       payout_on_affo=dividends_per_share / out["affo_per_share"])
    return out


@formula(T, formula="value = P/FFO multiple x forecast FFO/share; value = P/AFFO multiple x forecast AFFO/share",
         curriculum="L2 Alternatives R41 Publicly Traded Real Estate Securities (relative value: P/FFO, P/AFFO)")
def reit_relative_value(ffo_per_share: float | None = None, p_ffo_multiple: float | None = None,
                        affo_per_share: float | None = None, p_affo_multiple: float | None = None,
                        price: float | None = None) -> dict:
    """REIT value from peer P/FFO and P/AFFO multiples; with a price, the REIT's own multiples and gap to value.

    Args:
        ffo_per_share, affo_per_share: forecast (year-ahead) per-share FFO/AFFO.
        p_ffo_multiple, p_affo_multiple: benchmark (peer or sector) multiples.
        price: current share price.

    Returns:
        value_from_p_ffo, value_from_p_affo, current_p_ffo, current_p_affo (as inputs allow).

    Example:
        14.2 x 4.23 -> 60.07; 16.5 x 3.76 -> 62.04 (CFA L2 R41 Section 8.1, Capitol Shopping Center REIT).
    """
    out = {}
    if ffo_per_share is not None and p_ffo_multiple is not None:
        out["value_from_p_ffo"] = ffo_per_share * p_ffo_multiple
    if affo_per_share is not None and p_affo_multiple is not None:
        out["value_from_p_affo"] = affo_per_share * p_affo_multiple
    if price is not None:
        if ffo_per_share:
            out["current_p_ffo"] = price / ffo_per_share
        if affo_per_share:
            out["current_p_affo"] = price / affo_per_share
    if not out:
        raise ValueError("give a per-share FFO/AFFO with a multiple, or with a price")
    return out


@formula(T, formula="unsmoothed R_t = (R*_t - a R*_{t-1}) / (1 - a), a = smoothing weight (lag-1 autocorrelation)",
         curriculum="L3 Alternatives R26 (appraisal smoothing, first-order autoregressive unsmoothing; Geltner 1993)")
def unsmooth_appraisal_returns(observed_returns: list, smoothing_weight: float | None = None) -> dict:
    """Remove first-order appraisal smoothing from a return series (e.g. NCREIF, private equity NAV returns).

    Args:
        observed_returns: appraisal-based returns R*_t in time order, decimals.
        smoothing_weight: a in [0, 1); default = the series' lag-1 autocorrelation.

    Returns:
        unsmoothed_returns (one fewer than observed), smoothing_weight, std_observed, std_unsmoothed.

    Example:
        true returns recovered from R*_t = (1 - a) R_t + a R*_{t-1} (round trip; Geltner 1993).
    """
    x = list(observed_returns)
    if len(x) < 3:
        raise ValueError("need at least three observations")
    if smoothing_weight is None:
        m = sum(x) / len(x)
        num = sum((x[t] - m) * (x[t - 1] - m) for t in range(1, len(x)))
        smoothing_weight = num / sum((v - m) ** 2 for v in x)
    a = smoothing_weight
    if not 0 <= a < 1:
        raise ValueError("smoothing_weight must be in [0, 1)")
    u = [(x[t] - a * x[t - 1]) / (1 - a) for t in range(1, len(x))]

    def sd(v):
        m = sum(v) / len(v)
        return math.sqrt(sum((y - m) ** 2 for y in v) / (len(v) - 1))

    return {"unsmoothed_returns": u, "smoothing_weight": a, "std_observed": sd(x), "std_unsmoothed": sd(u)}


# ----------------------------------------------------------------------------------------------- private equity
def _waterfall(capital: float, proceeds: float, pref: float, carry: float, catch_up: float) -> tuple[float, float]:
    """Split proceeds into (LP, GP): return of capital, preferred return, GP catch-up, then carry split."""
    lp = gp = 0.0
    rem = proceeds
    x = min(rem, capital)
    lp, rem = lp + x, rem - x
    x = min(rem, pref)
    lp, rem = lp + x, rem - x
    if pref > 0 and catch_up > 0 and rem > 0:
        need = carry * pref / (catch_up - carry) if catch_up > carry else math.inf
        x = min(rem, need)
        gp, lp, rem = gp + catch_up * x, lp + (1 - catch_up) * x, rem - x
    return lp + (1 - carry) * rem, gp + carry * rem


@formula(T, formula="tiers: return of capital -> preferred return (1+h)^t - 1 -> GP catch-up -> carry split; "
                    "clawback = deal-by-deal carry - whole-fund carry",
         curriculum="L2 Alternatives R42 Private Equity Valuation (carried interest, distribution waterfalls, clawback)")
def pe_distribution_waterfall(invested: list, proceeds: list, holding_years: list | None = None,
                              carry_rate: float = 0.20, hurdle_rate: float = 0.0, catch_up: float = 1.0,
                              method: str = "deal_by_deal", committed_capital: float | None = None,
                              whole_fund_basis: str = "committed", value_threshold: float = 0.20) -> dict:
    """GP carried interest under a deal-by-deal (American) or whole-of-fund (European) waterfall.

    Args:
        invested: capital invested in each deal.
        proceeds: exit proceeds of each deal (0 for a write-off).
        holding_years: years each deal was held (needed with a hurdle; also gives deal IRRs).
        carry_rate: GP share of profits, decimal.
        hurdle_rate: annual compound preferred return to LPs, decimal (0 = no hurdle).
        catch_up: GP share of distributions in the catch-up tier (1.0 = full catch-up, 0 = none/hard hurdle).
            A partial catch-up must exceed carry_rate; 0 < catch_up <= carry_rate raises ValueError.
        method: "deal_by_deal" (carry per exit, with clawback at fund level) or "whole_of_fund".
        committed_capital: required for whole_fund_basis="committed".
        whole_fund_basis: "committed" (LPs first get all committed capital back - curriculum alternative 1),
            "invested" (LPs first get invested capital back) or "threshold" (carry on all profit once
            portfolio value >= (1 + value_threshold) x invested - curriculum alternative 2).
        value_threshold: threshold for the "threshold" basis, decimal.

    Returns:
        gp_carry, lp_distributions, total_profit, deals (per-deal lp/gp/profit/irr), whole_fund_carry,
        clawback (deal-by-deal: carry paid in excess of the whole-fund entitlement).

    Example:
        exit 30 -> 45 deal by deal, 20% carry -> GP 3; a second deal 25 -> 20 -> clawback 1
        (CFA L2 R42 Example 3).
    """
    n = len(invested)
    if n == 0 or len(proceeds) != n or (holding_years is not None and len(holding_years) != n):
        raise ValueError("invested, proceeds (and holding_years) must be non-empty and the same length")
    if hurdle_rate and holding_years is None:
        raise ValueError("holding_years is required with a hurdle_rate")
    if catch_up and catch_up <= carry_rate:
        raise ValueError("catch_up must exceed carry_rate (or be 0 for no catch-up)")
    if not 0 <= carry_rate < 1 or not 0 <= catch_up <= 1:
        raise ValueError("carry_rate must be in [0, 1) and catch_up in [0, 1]")
    yrs = holding_years or [0.0] * n
    prefs = [c * ((1 + hurdle_rate) ** t - 1) for c, t in zip(invested, yrs)]
    deals = []
    for c, p, t, pr in zip(invested, proceeds, yrs, prefs):
        lp, gp = _waterfall(c, p, pr, carry_rate, catch_up)
        d = {"invested": c, "proceeds": p, "profit": p - c, "lp": lp, "gp": gp}
        if holding_years is not None and t > 0 and c > 0:
            d["irr"] = (p / c) ** (1 / t) - 1 if p > 0 else -1.0
        deals.append(d)
    tot_c, tot_p, tot_pref = sum(invested), sum(proceeds), sum(prefs)
    whole_lp, whole_gp = _waterfall(tot_c, tot_p, tot_pref, carry_rate, catch_up)
    if method == "deal_by_deal":
        gp = sum(d["gp"] for d in deals)
        return {"gp_carry": gp, "lp_distributions": tot_p - gp, "total_profit": tot_p - tot_c, "deals": deals,
                "whole_fund_carry": whole_gp, "clawback": max(0.0, gp - whole_gp)}
    if method != "whole_of_fund":
        raise ValueError("method must be 'deal_by_deal' or 'whole_of_fund'")
    if whole_fund_basis == "committed":
        cc = _need(committed_capital, "committed_capital")
        gp = _waterfall(cc, tot_p, tot_pref, carry_rate, catch_up)[1]
    elif whole_fund_basis == "invested":
        gp = whole_gp
    elif whole_fund_basis == "threshold":
        gp = carry_rate * (tot_p - tot_c) if tot_p >= (1 + value_threshold) * tot_c else 0.0
    else:
        raise ValueError("whole_fund_basis must be 'committed', 'invested' or 'threshold'")
    return {"gp_carry": gp, "lp_distributions": tot_p - gp, "total_profit": tot_p - tot_c, "deals": deals,
            "whole_fund_carry": gp, "clawback": 0.0}


@formula(T, formula="NAV_before = NAV_after(t-1) + called - fees + operating results; carry = c x (NAV_before above "
                    "max(committed, prior peak NAV_before)); NAV_after = NAV_before - carry - distributions; "
                    "DPI = dist/paid-in, RVPI = NAV/paid-in, TVPI = DPI + RVPI",
         curriculum="L2 Alternatives R42 Private Equity Valuation (fund NAV, fees, carried interest, DPI/RVPI/TVPI)")
def pe_fund_nav_table(called_down: list, operating_results: list, distributions: list, committed_capital: float,
                      management_fee_rate: float = 0.02, carry_rate: float = 0.20,
                      fee_basis: str = "paid_in") -> dict:
    """Year-by-year PE fund table: paid-in, management fees, NAV before/after distributions, carry, multiples, IRRs.

    Args:
        called_down: capital called each year.
        operating_results: realized + unrealized results each year.
        distributions: distributions to LPs each year (0 where none).
        committed_capital: total LP commitments (carry starts once NAV before distributions exceeds it).
        management_fee_rate: annual fee, decimal.
        carry_rate: carried interest, decimal.
        fee_basis: "paid_in" (cumulative called capital, as in the curriculum) or "committed".

    Returns:
        rows (year index, called, paid_in, management_fee, operating_result, nav_before_distributions,
        carried_interest, distributions, nav_after_distributions), dpi, rvpi, tvpi, paid_in, gross_irr, net_irr.

    Notes:
        Gross IRR uses -calls (start of year) and operating results (year end); net IRR also deducts fees and
        carry - the curriculum's simplified convention (terminal NAV is not added). Carry after the first
        crossing is charged only on new highs of NAV before distributions (a high-water floor; the curriculum
        only shows rising NAVs).

    Example:
        EVP I calls 50/15/10/25/10/5, results -5/-15/25/45/55/105, distributions 0/0/0/25/45/75, committed 125
        -> NAV after 147.9, DPI 1.26, RVPI 1.29, TVPI 2.55, gross IRR 16.1%, net IRR 11.3% (CFA L2 R42 case).
    """
    n = len(called_down)
    if n == 0 or len(operating_results) != n or len(distributions) != n:
        raise ValueError("called_down, operating_results and distributions must be non-empty and the same length")
    if fee_basis not in ("paid_in", "committed"):
        raise ValueError("fee_basis must be 'paid_in' or 'committed'")
    rows, paid, nav, peak = [], 0.0, 0.0, committed_capital
    for t in range(n):
        paid += called_down[t]
        fee = management_fee_rate * (paid if fee_basis == "paid_in" else committed_capital)
        before = nav + called_down[t] - fee + operating_results[t]
        carry = carry_rate * (before - peak) if before > peak else 0.0
        peak = max(peak, before)
        nav = before - carry - distributions[t]
        rows.append({"year": t + 1, "called": called_down[t], "paid_in": paid, "management_fee": fee,
                     "operating_result": operating_results[t], "nav_before_distributions": before,
                     "carried_interest": carry, "distributions": distributions[t], "nav_after_distributions": nav})
    calls = list(called_down) + [0.0]
    gross = [-calls[0]] + [operating_results[t] - calls[t + 1] for t in range(n)]
    net = [-calls[0]] + [operating_results[t] - calls[t + 1] - rows[t]["management_fee"] - rows[t]["carried_interest"]
                         for t in range(n)]
    dpi, rvpi = sum(distributions) / paid, nav / paid
    out = {"rows": rows, "dpi": dpi, "rvpi": rvpi, "tvpi": dpi + rvpi, "paid_in": paid, "gross_cash_flows": gross,
           "net_cash_flows": net}
    for k, cf in (("gross_irr", gross), ("net_irr", net)):
        try:
            out[k] = _irr(cf)
        except ValueError:
            out[k] = float("nan")
    return out


@formula(T, formula="PIC = paid-in/committed; DPI = distributions/paid-in; RVPI = NAV/paid-in; TVPI = DPI + RVPI",
         curriculum="L2 Alternatives R42 Private Equity Valuation (fund performance multiples)")
def pe_fund_performance_multiples(paid_in: float, cumulative_distributions: float, nav: float,
                                  committed_capital: float | None = None) -> dict:
    """Private equity fund multiples (net of fees and carry when inputs are net).

    Args:
        paid_in: cumulative paid-in (called) capital.
        cumulative_distributions: cumulative distributions to LPs.
        nav: residual value (NAV after distributions).
        committed_capital: optional, for PIC.

    Returns:
        dpi, rvpi, tvpi, pic (if committed_capital).

    Example:
        paid-in 115, distributions 145, NAV 147.9 -> DPI 1.26, RVPI 1.29, TVPI 2.55 (CFA L2 R42 case, EVP I).
    """
    if paid_in <= 0:
        raise ValueError("paid_in must be positive")
    out = {"dpi": cumulative_distributions / paid_in, "rvpi": nav / paid_in,
           "tvpi": (cumulative_distributions + nav) / paid_in}
    if committed_capital:
        out["pic"] = paid_in / committed_capital
    return out


@formula(T, formula="solve sum CF_t/(1 + IRR)^t = 0 (terminal NAV added to the last cash flow)",
         curriculum="L2 Alternatives R42 Private Equity Valuation (IRR of fund cash flows)")
def pe_fund_irr(cash_flows: list, terminal_nav: float = 0.0, periods_per_year: int = 1) -> dict:
    """IRR of fund (or deal) cash flows, LP perspective: calls negative, distributions positive.

    Args:
        cash_flows: periodic net cash flows, first at t = 0.
        terminal_nav: residual NAV treated as a final distribution (since-inception IRR).
        periods_per_year: 4 for quarterly flows; the IRR is annualized by compounding.

    Returns:
        irr (annual), periodic_irr.

    Example:
        [-50, -21, -26.3, -1.5, 29.2, 41, 91.5] -> 11.3% (CFA L2 R42 case, EVP I net IRR).
    """
    cf = list(cash_flows)
    if len(cf) < 2:
        raise ValueError("need at least two cash flows")
    cf[-1] += terminal_nav
    i = _irr(cf)
    return {"irr": (1 + i) ** periods_per_year - 1, "periodic_irr": i}


@formula(T, formula="r~ = (1+r)/(1-q) - 1; POST = I/F, F = [I(1+r~)^t / V] / retention; PRE = POST - I; "
                    "shares y = x F/(1 - F); price = I/y",
         curriculum="L2 Alternatives R42 Private Equity Valuation (venture capital method)")
def vc_method_single_round(investment: float, years: float, discount_rate: float, existing_shares: float,
                           exit_value: float | None = None, exit_metric: float | None = None,
                           exit_multiple: float | None = None, prob_failure: float = 0.0,
                           retention_ratio: float = 1.0) -> dict:
    """Venture capital method for one round: post/pre-money, required ownership, new shares and price per share.

    Args:
        investment: amount invested now (I).
        years: time to exit (t).
        discount_rate: VC's required (target) annual return, decimal.
        existing_shares: shares held by founders/existing owners (x).
        exit_value: terminal value at exit (V); or exit_metric x exit_multiple (e.g. earnings x P/E).
        prob_failure: annual probability of failure q; adjusted rate = (1 + r)/(1 - q) - 1.
        retention_ratio: fraction of ownership the investor keeps after future dilution (1 = none).

    Returns:
        post_money, pre_money, ownership_fraction, final_ownership, new_shares, price_per_share,
        required_terminal_wealth, adjusted_discount_rate, exit_value.

    Example:
        V 25m in 4 years, r 50%, I 3m, 1m founder shares -> POST 4,938,272, F 60.75%, 1,547,771 shares at 1.94
        (CFA L2 R42 Appendix, SpiffyCalc).
    """
    if exit_value is None:
        exit_value = _need(exit_metric, "exit_value or exit_metric") * _need(exit_multiple, "exit_multiple")
    if not 0 <= prob_failure < 1 or not 0 < retention_ratio <= 1:
        raise ValueError("prob_failure must be in [0, 1) and retention_ratio in (0, 1]")
    if investment <= 0 or exit_value <= 0 or existing_shares <= 0:
        raise ValueError("investment, exit value and existing_shares must be positive")
    adj = (1 + discount_rate) / (1 - prob_failure) - 1
    w = investment * (1 + adj) ** years
    f_final = w / exit_value
    f = f_final / retention_ratio
    if f >= 1:
        raise ValueError(f"required ownership {f:.4f} >= 100%: pre-money value is not positive (negative-NPV deal)")
    post = investment / f
    y = existing_shares * f / (1 - f)
    return {"post_money": post, "pre_money": post - investment, "ownership_fraction": f, "final_ownership": f_final,
            "new_shares": y, "price_per_share": investment / y, "required_terminal_wealth": w,
            "adjusted_discount_rate": adj, "exit_value": exit_value}


@formula(T, formula="POST_k = V/(1+R_k); PRE_k = POST_k - I_k; POST_i = PRE_{i+1}/(1+R_i); F_i = I_i/POST_i; "
                    "y_i = x_i F_i/(1 - F_i); x_{i+1} = x_i + y_i",
         curriculum="L2 Alternatives R42 Private Equity Valuation (VC method, multiple financing rounds)")
def vc_method_multiple_rounds(exit_value: float, investments: list, compound_factors: list,
                              existing_shares: float) -> dict:
    """Venture capital method with several rounds: work back from exit to price each round and its dilution.

    Args:
        exit_value: terminal value at exit.
        investments: amount raised in each round, first round first.
        compound_factors: (1 + R_i), the compound discount factor from round i to the next round (last: to exit),
            e.g. 1.5**2 = 2.25 for two years at 50%.
        existing_shares: founders' shares before the first round.

    Returns:
        rounds (post_money, pre_money, ownership_fraction, new_shares, price_per_share, final_ownership,
        existing_shares), founders_final_ownership.

    Example:
        V 25m, rounds 3m then 2m, factors 2.25 and 2.25, 1m founder shares -> POST1 4,049,383, F1 74.09%,
        p1 1.05; POST2 11,111,111, F2 18%, p2 2.36 (CFA L2 R42 Appendix, two rounds).
    """
    k = len(investments)
    if k == 0 or len(compound_factors) != k:
        raise ValueError("investments and compound_factors must be non-empty and the same length")
    post, pre = [0.0] * k, [0.0] * k
    nxt = exit_value
    for i in range(k - 1, -1, -1):
        post[i] = nxt / compound_factors[i]
        pre[i] = post[i] - investments[i]
        if pre[i] <= 0:
            raise ValueError(f"round {i + 1}: pre-money value is not positive")
        nxt = pre[i]
    f = [inv / p for inv, p in zip(investments, post)]
    rounds, x = [], existing_shares
    for i in range(k):
        y = x * f[i] / (1 - f[i])
        keep = math.prod(1 - fj for fj in f[i + 1:])
        rounds.append({"post_money": post[i], "pre_money": pre[i], "ownership_fraction": f[i], "new_shares": y,
                       "price_per_share": investments[i] / y, "final_ownership": f[i] * keep, "existing_shares": x})
        x += y
    return {"rounds": rounds, "founders_final_ownership": math.prod(1 - fi for fi in f), "total_shares": x}


# ----------------------------------------------------------------------------------------------- hedge funds
@formula(T, formula="mgmt = m x AUM(end or beginning); incentive = p x max(0, value - max(HWM, B(1 + h))) "
                    "[value net of mgmt fee if specified]; soft hurdle: on the whole gain once the hurdle is beaten",
         curriculum="L1 Alternatives R63/R66 Introduction to Alternative Investments (hedge fund fees)")
def hedge_fund_fees(beginning_value: float, ending_value: float, management_fee_rate: float = 0.02,
                    incentive_fee_rate: float = 0.20, management_fee_basis: str = "end",
                    incentive_net_of_management_fee: bool = False, hurdle_rate: float = 0.0,
                    hurdle_type: str = "hard", high_water_mark: float | None = None) -> dict:
    """One-period hedge fund management and incentive fees and the investor's net return.

    Args:
        beginning_value: investor capital at the start of the period (net of prior fees).
        ending_value: gross fund value at the end of the period, before this period's fees.
        management_fee_rate, incentive_fee_rate: decimals ("2 and 20").
        management_fee_basis: "end" or "beginning" of period assets.
        incentive_net_of_management_fee: True if the incentive is on gains after the management fee.
        hurdle_rate: hurdle return on beginning capital, decimal.
        hurdle_type: "hard" (incentive only on gains above the hurdle) or "soft" (on the whole gain once beaten).
        high_water_mark: highest prior net value; gains are measured above max(HWM, beginning value).

    Returns:
        management_fee, incentive_fee, total_fees, ending_value_net, net_return, gross_return, high_water_mark_end,
        fee_drag.

    Example:
        100 -> 130, 2% on end AUM, 20% independent -> fees 8.6, net return 21.40%; with a 5% hard hurdle and
        incentive net of the management fee -> fees 7.08, net 22.92% (CFA L1 R66 Example 2, AWJ Capital).
    """
    if beginning_value <= 0:
        raise ValueError("beginning_value must be positive")
    if management_fee_basis not in ("end", "beginning") or hurdle_type not in ("hard", "soft"):
        raise ValueError("management_fee_basis must be 'end'/'beginning' and hurdle_type 'hard'/'soft'")
    mgmt = management_fee_rate * (ending_value if management_fee_basis == "end" else beginning_value)
    val = ending_value - mgmt if incentive_net_of_management_fee else ending_value
    hwm = max(beginning_value, high_water_mark or 0.0)
    if hurdle_type == "hard":
        base = val - max(hwm, beginning_value * (1 + hurdle_rate))
    else:
        base = val - hwm if (val - beginning_value) / beginning_value > hurdle_rate else 0.0
    inc = incentive_fee_rate * max(0.0, base)
    net = ending_value - mgmt - inc
    return {"management_fee": mgmt, "incentive_fee": inc, "total_fees": mgmt + inc, "ending_value_net": net,
            "net_return": net / beginning_value - 1, "gross_return": ending_value / beginning_value - 1,
            "high_water_mark_end": max(hwm, net), "fee_drag": (ending_value - net) / beginning_value}


@formula(T, formula="net HF return after HF fees, then FoF fees applied to the FoF's gains",
         curriculum="L1 Alternatives R63/R66 Introduction to Alternative Investments (fund of hedge funds fees)")
def fund_of_hedge_funds_net_return(gross_return: float, hf_management_fee: float = 0.02,
                                   hf_incentive_fee: float = 0.20, fof_management_fee: float = 0.01,
                                   fof_incentive_fee: float = 0.10, management_fee_basis: str = "beginning",
                                   incentive_net_of_management_fee: bool = False) -> dict:
    """Investor return through a fund of hedge funds: two layers of management and incentive fees.

    Args:
        gross_return: underlying hedge funds' return before all fees, decimal.
        hf_management_fee, hf_incentive_fee: underlying fund fees ("2 and 20").
        fof_management_fee, fof_incentive_fee: fund-of-funds fees ("1 and 10").
        management_fee_basis: "beginning" or "end" of period assets (both layers).
        incentive_net_of_management_fee: incentive on gains after management fee (both layers).

    Returns:
        fof_net_return, direct_hf_net_return, fee_layering_cost (direct net - FoF net).

    Example:
        gross 20%, 2/20 then 1/10, fees on beginning AUM, independent -> direct 14%, through the FoF 11.6%
        (CFA L1 R66 Example 3).
    """
    hf = hedge_fund_fees(1.0, 1 + gross_return, hf_management_fee, hf_incentive_fee, management_fee_basis,
                         incentive_net_of_management_fee)
    fof = hedge_fund_fees(1.0, hf["ending_value_net"], fof_management_fee, fof_incentive_fee, management_fee_basis,
                          incentive_net_of_management_fee)
    return {"fof_net_return": fof["net_return"], "direct_hf_net_return": hf["net_return"],
            "fee_layering_cost": hf["net_return"] - fof["net_return"]}


# ----------------------------------------------------------------------------------------------- commodities
@formula(T, formula="total return = spot return + roll return + collateral return (+ rebalancing return)",
         curriculum="L2 Alternatives R44 Commodity Investing / L3 R26 (commodity index return components)")
def commodity_futures_total_return(spot_return: float, roll_return: float, collateral_return: float,
                                   rebalancing_return: float = 0.0) -> dict:
    """Total return of a fully collateralized commodity futures position or index from its components.

    Args:
        spot_return: change in spot (nearby) price, decimal.
        roll_return: roll yield, decimal (positive in backwardation, negative in contango).
        collateral_return: T-bill yield on the collateral, decimal.
        rebalancing_return: index rebalancing return, decimal.

    Returns:
        total_return, excess_return (spot + roll = futures return).

    Example:
        GSCI 2000: 26.9% + 14.2% + 8.6% -> 49.7% (CFA L3 R26 Exhibit 17).
    """
    ex = spot_return + roll_return
    return {"total_return": ex + collateral_return + rebalancing_return, "excess_return": ex}


@formula(T, formula="roll return = change in futures price - change in spot price",
         curriculum="L3 Alternatives R26 Alternative Investments Portfolio Management (roll return)")
def commodity_roll_return_monetary(futures_price_end: float, futures_price_begin: float,
                                   spot_price_change: float) -> dict:
    """Roll return in price units over a period: futures gain not explained by the spot move.

    Args:
        futures_price_end, futures_price_begin: same contract's price at the end and start of the period.
        spot_price_change: change in the spot price over the period.

    Returns:
        roll_return, futures_price_change, roll_return_pct (relative to the beginning futures price).

    Example:
        June contract 39.10 -> 40.58, spot +0.40 -> roll return 1.08 (CFA L3 R26 Exhibit 16).
    """
    df = futures_price_end - futures_price_begin
    rr = df - spot_price_change
    return {"roll_return": rr, "futures_price_change": df, "roll_return_pct": rr / futures_price_begin}


@formula(T, formula="roll yield = (F_near - F_far)/F_near; basis = spot - futures; F_far < F_near: backwardation",
         curriculum="L2 Alternatives R44 A Primer on Commodity Investing (roll return and term structure)")
def commodity_term_structure(near_futures_price: float, far_futures_price: float,
                             spot_price: float | None = None) -> dict:
    """Roll yield from rolling the expiring (near) contract into the next one, and the term-structure shape.

    Args:
        near_futures_price: price of the expiring/near contract (converges to spot).
        far_futures_price: price of the next contract bought on the roll.
        spot_price: optional, for the basis.

    Returns:
        roll_yield (positive = backwardation), term_structure, basis (spot - near futures, if spot given).

    Notes:
        R44 Eq. 6 writes (F_new - F_old)/F_old, whose negative values mean a positive roll return; this
        function reports the roll yield with the investor's sign (same magnitude).

    Example:
        near 100, next 95 -> roll yield +5%, backwardation.
    """
    if near_futures_price <= 0 or far_futures_price <= 0:
        raise ValueError("futures prices must be positive")
    ry = (near_futures_price - far_futures_price) / near_futures_price
    shape = "backwardation" if ry > 0 else "contango" if ry < 0 else "flat"
    out = {"roll_yield": ry, "term_structure": shape}
    if spot_price is not None:
        out["basis"] = spot_price - near_futures_price
    return out


def _dated_irr(flows: list) -> float | None:
    """IRR of [(t_years, amount)] flows by bisection on [-0.99, 10]; None if there is no sign change."""
    def npv(r: float) -> float:
        return sum(cf / (1 + r) ** t for t, cf in flows)
    lo, hi = -0.99, 10.0
    if npv(lo) * npv(hi) > 0:
        return None
    for _ in range(200):
        mid = (lo + hi) / 2
        if npv(lo) * npv(mid) <= 0:
            hi = mid
        else:
            lo = mid
    return (lo + hi) / 2


@formula(T, formula="per distribution: return unreturned capital -> preferred return (compounds on unreturned capital "
                    "+ unpaid pref between dated cash flows) -> GP catch-up until GP = carry x profit distributed -> "
                    "split (1 - carry)/carry",
         curriculum="L2 Alternatives R42 Private Equity Valuation (whole-of-fund waterfall, hurdle, catch-up, "
                    "clawback); standard LPA mechanics")
def pe_waterfall_cash_flows(contributions: list, distributions: list, hurdle_rate: float = 0.08,
                            carry_rate: float = 0.20, catch_up: float = 1.0, residual_nav: float = 0.0,
                            deal_by_deal_gp_carry: float | None = None) -> dict:
    """Whole-of-fund (European) waterfall driven by dated LP contributions and fund distributions.

    Use this when cash flows have dates: the preferred return accrues only on capital not yet returned (plus unpaid
    pref), so early exits that return capital stop the hurdle clock on that capital. pe_distribution_waterfall
    instead accrues the hurdle per deal over each deal's holding period.

    Args:
        contributions: [[t, amount], ...] LP capital paid in, t in years from the first call.
        distributions: [[t, amount], ...] gross fund distributions available to split between LPs and the GP.
        hurdle_rate: annual compound preferred return, decimal (0 for none).
        carry_rate: GP carried interest share, decimal.
        catch_up: GP share of the catch-up tier (1.0 full catch-up; 0 = no catch-up, carry only above the hurdle).
        residual_nav: unrealized value still in the fund at the last date (for RVPI/TVPI; not run through the waterfall).
        deal_by_deal_gp_carry: carry already paid to the GP under a deal-by-deal waterfall, to size the clawback.

    Returns:
        gp_carry, lp_distributions, pref_paid, events (per distribution: capital, pref, catch-up, split to LP/GP),
        lp_net_irr, fund_gross_irr, dpi, rvpi, tvpi, clawback (if deal_by_deal_gp_carry is given).

    Example:
        contributions [[0, 100]], distributions [[2, 100], [5, 30]], 8% hurdle, full catch-up, 20% carry
        -> pref paid 20.96, GP carry 6.0, LP gets 100 then 24.0, LP net IRR 8.90%
    """
    if not 0 <= carry_rate < 1:
        raise ValueError("carry_rate must be in [0, 1)")
    if catch_up and catch_up <= carry_rate:
        raise ValueError("catch_up must exceed carry_rate (or be 0 for no catch-up)")
    events = sorted([(float(t), "c", float(a)) for t, a in contributions] +
                    [(float(t), "d", float(a)) for t, a in distributions], key=lambda e: (e[0], e[1]))
    cap = pref = 0.0
    pref_paid = profit = gp = paid_in = 0.0
    last_t = events[0][0] if events else 0.0
    lp_flows, fund_flows, out = [], [], []
    for t, kind, amt in events:
        pref = (cap + pref) * (1 + hurdle_rate) ** (t - last_t) - cap
        last_t = t
        if kind == "c":
            cap += amt
            paid_in += amt
            lp_flows.append((t, -amt))
            fund_flows.append((t, -amt))
            continue
        fund_flows.append((t, amt))
        rem, row = amt, {"t": t, "distribution": amt}
        x = min(rem, cap); cap -= x; rem -= x; row["return_of_capital"] = x
        x = min(rem, pref); pref -= x; rem -= x; pref_paid += x; profit += x; row["preferred_return"] = x
        lp_amt = row["return_of_capital"] + x
        gp_amt = 0.0
        if catch_up and rem > 0:
            deficit = carry_rate * profit - gp
            x = min(rem, max(deficit, 0.0) / (catch_up - carry_rate))
            gp_amt += catch_up * x
            lp_amt += (1 - catch_up) * x
            gp += catch_up * x
            profit += x
            rem -= x
            row["catch_up"] = x
        row["split"] = rem
        gp_amt += carry_rate * rem
        lp_amt += (1 - carry_rate) * rem
        gp += carry_rate * rem
        profit += rem
        row["lp"], row["gp"] = lp_amt, gp_amt
        lp_flows.append((t, lp_amt))
        out.append(row)
    lp_dist = sum(r["lp"] for r in out)
    if residual_nav and lp_flows:
        lp_flows = lp_flows + [(last_t, residual_nav)]
    res = {"gp_carry": gp, "lp_distributions": lp_dist, "pref_paid": pref_paid, "events": out,
           "lp_net_irr": _dated_irr(lp_flows), "fund_gross_irr": _dated_irr(fund_flows + ([(last_t, residual_nav)]
                                                                                           if residual_nav else [])),
           "dpi": lp_dist / paid_in if paid_in else None, "rvpi": residual_nav / paid_in if paid_in else None,
           "tvpi": (lp_dist + residual_nav) / paid_in if paid_in else None,
           "unreturned_capital": cap, "unpaid_pref": pref}
    if deal_by_deal_gp_carry is not None:
        res["clawback"] = max(0.0, deal_by_deal_gp_carry - gp)
    return res
