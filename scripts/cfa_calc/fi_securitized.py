"""Fixed income: mortgage-backed securities — prepayment speeds, mortgage cash flows, WAL, WAC/WAM (L2)."""
from __future__ import annotations

from ._registry import formula

L2_MBS = "L2 FI: Mortgage-Backed Sector of the Bond Market (2014 R48)"


@formula("fixed_income", formula="CPR = 1 - (1 - SMM)^12 ; SMM = 1 - (1 - CPR)^(1/12)", curriculum=L2_MBS)
def mbs_smm_cpr(smm: float | None = None, cpr: float | None = None) -> dict:
    """Convert between the single monthly mortality rate (SMM) and the conditional prepayment rate (CPR).

    Args:
        smm: monthly prepayment rate, or cpr: annualized prepayment rate (give exactly one).

    Returns:
        smm, cpr.

    Example:
        CPR 6% <-> SMM 0.5143%
    """
    if (smm is None) == (cpr is None):
        raise ValueError("give exactly one of smm or cpr")
    if smm is not None:
        if not 0 <= smm <= 1:
            raise ValueError("smm must be in [0, 1]")
        return {"smm": smm, "cpr": 1 - (1 - smm) ** 12}
    if not 0 <= cpr <= 1:
        raise ValueError("cpr must be in [0, 1]")
    return {"smm": 1 - (1 - cpr) ** (1 / 12), "cpr": cpr}


@formula("fixed_income", formula="100 PSA: CPR = 6% x min(t, 30)/30 ; x PSA/100 ; SMM = 1 - (1 - CPR)^(1/12)",
         curriculum=L2_MBS)
def mbs_psa_speed(month: int, psa: float = 100.0) -> dict:
    """CPR and SMM for a loan age (months since origination) at a given PSA speed.

    Args:
        month: months since origination (1 = first month). psa: speed in % of the PSA benchmark (100 = 100 PSA).

    Returns:
        cpr, smm.

    Example:
        month 20 at 165 PSA -> CPR 6.6%, SMM 0.5674%
    """
    if month < 1 or psa < 0:
        raise ValueError("month must be >= 1 and psa >= 0")
    cpr = min(0.06 * min(month, 30) / 30 * psa / 100, 1.0)
    return {"cpr": cpr, "smm": 1 - (1 - cpr) ** (1 / 12)}


@formula("fixed_income", formula="PMT = B x i / (1 - (1+i)^-n), i = annual rate / 12", curriculum=L2_MBS)
def mbs_mortgage_payment(balance: float, annual_rate: float, months: int) -> dict:
    """Level monthly payment of a fully amortizing fixed-rate mortgage, split for the first month.

    Args:
        balance: outstanding principal. annual_rate: mortgage rate (WAC for a pool). months: remaining term.

    Returns:
        payment, interest_first_month, principal_first_month.

    Example:
        400m at 8.125% over 357 months -> 2,975,868
    """
    if balance <= 0 or months < 1:
        raise ValueError("balance must be positive and months >= 1")
    i = annual_rate / 12
    pmt = balance / months if i == 0 else balance * i / (1 - (1 + i) ** -months)
    return {"payment": pmt, "interest_first_month": balance * i, "principal_first_month": pmt - balance * i}


@formula("fixed_income", formula="sched principal = PMT - B x WAC/12 ; prepay = SMM x (B - sched) ; "
         "net interest = B x pass-through/12 ; WAL = sum t x principal_t / (12 x total principal)", curriculum=L2_MBS)
def mbs_cash_flows(balance: float, wac: float, pass_through_rate: float, wam_months: int, psa: float = 100.0,
                   cpr: float | None = None, seasoning_months: int = 0, include_schedule: bool = True) -> dict:
    """Monthly pass-through cash-flow schedule under a PSA (or constant CPR) prepayment assumption.

    Args:
        balance: pool balance today. wac: weighted-average coupon of the mortgages. pass_through_rate: net coupon
        paid to investors (WAC minus servicing/guarantee fees). wam_months: remaining term.
        psa: prepayment speed in % PSA (ignored if cpr given). cpr: constant annual prepayment rate.
        seasoning_months: loan age today (month 1 of the schedule uses age seasoning_months + 1).
        include_schedule: return the month-by-month rows.

    Returns:
        weighted_average_life (years), total_principal, total_interest, total_cash_flow, schedule (rows: month,
        age, beginning_balance, smm, mortgage_payment, net_interest, scheduled_principal, prepayment,
        total_principal, cash_flow).

    Example:
        400m, WAC 8.125%, pass-through 7.5%, WAM 357, age 3, 100 PSA -> month-1 cash flow 3,035,005, WAL 11.66y
    """
    if balance <= 0 or wam_months < 1 or seasoning_months < 0:
        raise ValueError("balance must be positive, wam_months >= 1, seasoning_months >= 0")
    smm_const = None if cpr is None else mbs_smm_cpr(cpr=cpr)["smm"]
    bal, rows, wsum, tp, ti = balance, [], 0.0, 0.0, 0.0
    for m in range(1, wam_months + 1):
        if bal <= 1e-9:
            break
        age = seasoning_months + m
        smm = smm_const if smm_const is not None else mbs_psa_speed(age, psa)["smm"]
        pay = mbs_mortgage_payment(bal, wac, wam_months - m + 1)["payment"]
        sched = min(pay - bal * wac / 12, bal)
        prep = smm * (bal - sched)
        prin = sched + prep
        net_int = bal * pass_through_rate / 12
        if include_schedule:
            rows.append({"month": m, "age": age, "beginning_balance": bal, "smm": smm, "mortgage_payment": pay,
                         "net_interest": net_int, "scheduled_principal": sched, "prepayment": prep,
                         "total_principal": prin, "cash_flow": net_int + prin})
        wsum += m * prin
        tp += prin
        ti += net_int
        bal -= prin
    return {"weighted_average_life": wsum / (12 * tp), "total_principal": tp, "total_interest": ti,
            "total_cash_flow": tp + ti, "schedule": rows if include_schedule else None}


@formula("fixed_income", formula="WAL = sum t x principal_t / (periods_per_year x total principal)", curriculum=L2_MBS)
def mbs_weighted_average_life(principal_payments: list, periods_per_year: float = 12) -> dict:
    """Weighted average life of any amortizing security or tranche from its principal payments.

    Args:
        principal_payments: principal received at periods 1, 2, ... (scheduled + prepayments).
        periods_per_year: 12 for monthly.

    Returns:
        weighted_average_life (years), total_principal.

    Example:
        principal 50 at month 12 and 50 at month 36 -> WAL 2.0 years
    """
    tot = sum(principal_payments)
    if tot <= 0:
        raise ValueError("total principal must be positive")
    w = sum(t * p for t, p in enumerate(principal_payments, 1))
    return {"weighted_average_life": w / (periods_per_year * tot), "total_principal": tot}


@formula("fixed_income", formula="WAC = sum w_i rate_i ; WAM = sum w_i months_i (w = balance weights)",
         curriculum=L2_MBS)
def mbs_pool_wac_wam(balances: list, rates: list, months_remaining: list) -> dict:
    """Weighted-average coupon and weighted-average maturity of a mortgage pool.

    Args:
        balances: outstanding balance of each loan. rates: mortgage rates. months_remaining: remaining terms.

    Returns:
        wac, wam, weights.

    Example:
        five loans (L2 R48) -> WAC 7.28%, WAM 279 months
    """
    if not len(balances) == len(rates) == len(months_remaining) or not balances:
        raise ValueError("lists must be non-empty and the same length")
    tot = sum(balances)
    w = [b / tot for b in balances]
    return {"wac": sum(a * r for a, r in zip(w, rates)), "wam": sum(a * m for a, m in zip(w, months_remaining)),
            "weights": w}
