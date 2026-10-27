"""Build the Excel fixtures used by tests and evals. Formulas are written with their cached values (xlsxwriter),
exactly as Excel saves them, so readers see both.

  northwind_dcf.xlsx       a DCF with planted flaws (see FLAWS)
  harbor_financials.xlsx   three years of statements with earnings-quality red flags
"""
from __future__ import annotations

from pathlib import Path

import xlsxwriter

HERE = Path(__file__).resolve().parent

FLAWS = {
    "hardcoded_growth": "DCF!E4 grows revenue with a typed 1.06 instead of the growth input",
    "pasted_ebit": "DCF!F5 is a typed 180 over the EBIT formula row (formula gives 189.37)",
    "inconsistent_fcff": "DCF!E12 leaves out the change in NWC that the rest of the FCFF row subtracts",
    "fcff_uses_ebit": "FCFF row starts from EBIT, not NOPAT = EBIT x (1 - t)",
    "tv_discount_period": "PV of terminal value discounts 4 periods; the TV sits at the end of year 5",
    "terminal_growth_high": "terminal growth 6% exceeds long-run nominal GDP; TV dominates EV",
    "wacc_book_weights": "WACC sheet weights debt and equity at book value; the market cap is 1,500 vs book 400",
    "wacc_not_linked": "Assumptions!B9 WACC is a typed 8% not linked to the WACC sheet",
    "bs_imbalance": "BS 2022A total assets 905 vs total liabilities and equity 900",
}


def northwind(path: Path) -> dict:
    a = {"growth": 0.06, "margin": 0.15, "tax": 0.25, "da": 0.04, "capex": 0.05, "nwc": 0.12, "wacc": 0.08,
         "g": 0.06, "net_debt": 300.0, "shares": 50.0}
    wb = xlsxwriter.Workbook(path)
    bold, blue = wb.add_format({"bold": True}), wb.add_format({"font_color": "blue"})
    pct = wb.add_format({"num_format": "0.0%", "font_color": "blue"})

    ws = wb.add_worksheet("Assumptions")
    ws.write("A1", "Northwind Industrial - DCF assumptions ($ millions)", bold)
    rows = [("Revenue growth", a["growth"], pct), ("EBIT margin", a["margin"], pct), ("Tax rate", a["tax"], pct),
            ("D&A (% of revenue)", a["da"], pct), ("Capex (% of revenue)", a["capex"], pct),
            ("Net working capital (% of revenue)", a["nwc"], pct), ("WACC", a["wacc"], pct),
            ("Terminal growth", a["g"], pct), ("Net debt", a["net_debt"], blue), ("Shares outstanding (m)", a["shares"], blue)]
    for i, (lab, v, fmt) in enumerate(rows, 3):
        ws.write(f"A{i}", lab)
        ws.write(f"B{i}", v, fmt)
    ws.set_column("A:A", 36)
    A = {k: f"Assumptions!$B${i}" for i, k in enumerate(["growth", "margin", "tax", "da", "capex", "nwc", "wacc", "g",
                                                         "net_debt", "shares"], 3)}

    d = wb.add_worksheet("DCF")
    cols = "BCDEFG"
    years = ["2023A", "2024E", "2025E", "2026E", "2027E", "2028E"]
    d.write("A2", "Period")
    d.write("A3", "($ millions)", bold)
    for j, (c, y) in enumerate(zip(cols, years)):
        d.write(f"{c}3", y, bold)
        if j:
            d.write(f"{c}2", j)
    rev = [1000.0]
    for j in range(1, 6):
        rev.append(rev[-1] * (1.06 if j == 3 else 1 + a["growth"]))
    ebit = [r * a["margin"] for r in rev]
    ebit[4] = 180.0
    tax = [e * a["tax"] for e in ebit]
    nopat = [e - t for e, t in zip(ebit, tax)]
    da = [r * a["da"] for r in rev]
    capex = [r * a["capex"] for r in rev]
    nwc = [r * a["nwc"] for r in rev]
    dnwc = [None] + [nwc[j] - nwc[j - 1] for j in range(1, 6)]
    fcff = [None] + [ebit[j] + da[j] - capex[j] - (0 if j == 3 else dnwc[j]) for j in range(1, 6)]
    df = [None] + [1 / (1 + a["wacc"]) ** j for j in range(1, 6)]
    pv = [None] + [fcff[j] * df[j] for j in range(1, 6)]
    labels = {4: "Revenue", 5: "EBIT", 6: "Taxes on EBIT", 7: "NOPAT", 8: "Depreciation & amortization",
              9: "Capital expenditures", 10: "Net working capital", 11: "Change in NWC", 12: "Free cash flow to firm",
              13: "Discount factor", 14: "PV of FCFF"}
    for r, lab in labels.items():
        d.write(f"A{r}", lab)
    d.write("B4", rev[0], blue)
    d.write("B10", nwc[0], blue)
    for j in range(1, 6):
        c, p = cols[j], cols[j - 1]
        d.write_formula(f"{c}4", f"={p}4*1.06" if j == 3 else f"={p}4*(1+{A['growth']})", None, rev[j])
        if j == 4:
            d.write(f"{c}5", 180)
        else:
            d.write_formula(f"{c}5", f"={c}4*{A['margin']}", None, ebit[j])
        d.write_formula(f"{c}6", f"={c}5*{A['tax']}", None, tax[j])
        d.write_formula(f"{c}7", f"={c}5-{c}6", None, nopat[j])
        d.write_formula(f"{c}8", f"={c}4*{A['da']}", None, da[j])
        d.write_formula(f"{c}9", f"={c}4*{A['capex']}", None, capex[j])
        d.write_formula(f"{c}10", f"={c}4*{A['nwc']}", None, nwc[j])
        d.write_formula(f"{c}11", f"={c}10-{p}10", None, dnwc[j])
        d.write_formula(f"{c}12", f"={c}5+{c}8-{c}9" if j == 3 else f"={c}5+{c}8-{c}9-{c}11", None, fcff[j])
        d.write_formula(f"{c}13", f"=1/(1+{A['wacc']})^{c}2", None, df[j])
        d.write_formula(f"{c}14", f"={c}12*{c}13", None, pv[j])
    sum_pv = sum(pv[1:])
    tv = fcff[5] * (1 + a["g"]) / (a["wacc"] - a["g"])
    pv_tv = tv / (1 + a["wacc"]) ** 4
    ev = sum_pv + pv_tv
    eq = ev - a["net_debt"]
    out = [("Sum of PV of FCFF", "=SUM(C14:G14)", sum_pv),
           ("Terminal value", f"=G12*(1+{A['g']})/({A['wacc']}-{A['g']})", tv),
           ("PV of terminal value", f"=B17/(1+{A['wacc']})^4", pv_tv),
           ("Enterprise value", "=B16+B18", ev), ("Net debt", f"={A['net_debt']}", a["net_debt"]),
           ("Equity value", "=B19-B20", eq), ("Value per share ($)", f"=B21/{A['shares']}", eq / a["shares"])]
    for i, (lab, f, v) in enumerate(out, 16):
        d.write(f"A{i}", lab, bold if lab.startswith(("Enterprise", "Equity", "Value")) else None)
        d.write_formula(f"B{i}", f, None, v)
    d.set_column("A:A", 30)
    d.set_column("B:G", 11)

    w = wb.add_worksheet("WACC")
    rf, beta, erp, kd, price = 0.04, 1.2, 0.055, 0.06, 30.0
    ke = rf + beta * erp
    eb, db = 400.0, 350.0
    wacc_book = eb / (eb + db) * ke + db / (eb + db) * kd * (1 - a["tax"])
    winp = [("Risk-free rate", rf), ("Equity beta", beta), ("Equity risk premium", erp), ("Pre-tax cost of debt", kd),
            ("Share price ($)", price), ("Equity (book value)", eb), ("Debt (book value)", db)]
    for i, (lab, v) in enumerate(winp, 2):
        w.write(f"A{i}", lab)
        w.write(f"B{i}", v, blue)
    w.write("A10", "Cost of equity (CAPM)")
    w.write_formula("B10", "=B2+B3*B4", None, ke)
    w.write("A11", "Weight of equity")
    w.write_formula("B11", "=B7/(B7+B8)", None, eb / (eb + db))
    w.write("A12", "Weight of debt")
    w.write_formula("B12", "=B8/(B7+B8)", None, db / (eb + db))
    w.write("A13", "WACC (computed)")
    w.write_formula("B13", f"=B11*B10+B12*B5*(1-{A['tax']})", None, wacc_book)
    w.set_column("A:A", 26)

    b = wb.add_worksheet("BS")
    b.write("A1", "Balance sheet ($ millions)", bold)
    yrs = ["2021A", "2022A", "2023A"]
    bs = {"Cash": [80, 95, 110], "Receivables": [150, 160, 170], "Inventory": [120, 130, 140],
          "PP&E, net": [520, 520, 540]}
    liab = {"Payables": [90, 95, 100], "Long-term debt": [400, 405, 410]}
    eqv = [380, 400, 450]
    b.write_row("A2", ["", *yrs], bold)
    r = 3
    for lab, v in bs.items():
        b.write_row(f"A{r}", [lab, *v])
        r += 1
    ta = [sum(x) for x in zip(*bs.values())]
    b.write("A7", "Total assets", bold)
    for j, c in enumerate("BCD"):
        b.write_formula(f"{c}7", f"=SUM({c}3:{c}6)", None, ta[j])
    b.write_row("A8", ["Payables", *liab["Payables"]])
    b.write_row("A9", ["Long-term debt", *liab["Long-term debt"]])
    tl = [x + y for x, y in zip(liab["Payables"], liab["Long-term debt"])]
    b.write("A10", "Total liabilities")
    for j, c in enumerate("BCD"):
        b.write_formula(f"{c}10", f"={c}8+{c}9", None, tl[j])
    b.write_row("A11", ["Total equity", *eqv])
    b.write("A12", "Total liabilities and equity", bold)
    for j, c in enumerate("BCD"):
        b.write_formula(f"{c}12", f"={c}10+{c}11", None, tl[j] + eqv[j])
    b.set_column("A:A", 30)
    wb.close()
    return {"as_built": {"sum_pv_fcff": sum_pv, "terminal_value": tv, "pv_terminal_value": pv_tv,
                         "enterprise_value": ev, "equity_value": eq, "value_per_share": eq / a["shares"],
                         "wacc_book": wacc_book, "cost_of_equity": ke},
            "balance_sheet_assets": ta, "balance_sheet_l_plus_e": [x + y for x, y in zip(tl, eqv)]}


def harbor(path: Path) -> dict:
    """Retailer whose earnings grow while cash flow falls: receivables and inventory build faster than sales."""
    yrs = ["2021", "2022", "2023"]
    is_ = {"Revenue": [2000, 2200, 2420], "Cost of goods sold": [1200, 1309, 1428],
           "Selling, general and administrative": [500, 540, 580], "Depreciation and amortization": [60, 66, 72]}
    gp = [r - c for r, c in zip(is_["Revenue"], is_["Cost of goods sold"])]
    ebit = [g - s - d for g, s, d in zip(gp, is_["Selling, general and administrative"],
                                          is_["Depreciation and amortization"])]
    interest = [40, 42, 45]
    pretax = [e - i for e, i in zip(ebit, interest)]
    tax = [round(p * 0.25, 2) for p in pretax]
    ni = [p - t for p, t in zip(pretax, tax)]
    bal = {"Cash and cash equivalents": [150, 140, 120], "Accounts receivable": [220, 300, 420],
           "Inventory": [300, 360, 450], "Property, plant and equipment, net": [900, 960, 1010]}
    ca = [sum(x) for x in zip(bal["Cash and cash equivalents"], bal["Accounts receivable"], bal["Inventory"])]
    ta = [c + p for c, p in zip(ca, bal["Property, plant and equipment, net"])]
    ap = [180, 200, 215]
    std = [50, 60, 120]
    cl = [x + y for x, y in zip(ap, std)]
    ltd = [500, 520, 560]
    tl = [c + l for c, l in zip(cl, ltd)]
    eq = [a - l for a, l in zip(ta, tl)]
    capex = [110, 126, 122]
    div = [60, 64, 70]
    wb = xlsxwriter.Workbook(path)
    bold = wb.add_format({"bold": True})
    s = wb.add_worksheet("Financials")
    s.write("A1", "Harbor Retail Co. - financial statements (USD millions, fiscal years)", bold)
    s.write_row("A3", ["Income statement", *yrs], bold)
    lines = [("Revenue", is_["Revenue"]), ("Cost of goods sold", is_["Cost of goods sold"]), ("Gross profit", gp),
             ("Selling, general and administrative", is_["Selling, general and administrative"]),
             ("Depreciation and amortization", is_["Depreciation and amortization"]), ("Operating income (EBIT)", ebit),
             ("Interest expense", interest), ("Income before taxes", pretax), ("Income tax expense", tax),
             ("Net income", ni), ("Dividends paid", div), ("Shares outstanding (millions)", [100, 100, 100])]
    r = 4
    for lab, v in lines:
        s.write_row(f"A{r}", [lab, *v])
        r += 1
    r += 1
    s.write_row(f"A{r}", ["Balance sheet", *yrs], bold)
    r += 1
    bs_lines = [("Cash and cash equivalents", bal["Cash and cash equivalents"]),
                ("Accounts receivable", bal["Accounts receivable"]), ("Inventory", bal["Inventory"]),
                ("Total current assets", ca), ("Property, plant and equipment, net", bal["Property, plant and equipment, net"]),
                ("Total assets", ta), ("Accounts payable", ap), ("Short-term debt", std),
                ("Total current liabilities", cl), ("Long-term debt", ltd), ("Total liabilities", tl),
                ("Total equity", eq), ("Total liabilities and equity", [l + e for l, e in zip(tl, eq)])]
    for lab, v in bs_lines:
        s.write_row(f"A{r}", [lab, *v])
        r += 1
    r += 1
    # cash flow for 2022 and 2023 (needs prior-year balances)
    d_ar = [None] + [bal["Accounts receivable"][i] - bal["Accounts receivable"][i - 1] for i in (1, 2)]
    d_inv = [None] + [bal["Inventory"][i] - bal["Inventory"][i - 1] for i in (1, 2)]
    d_ap = [None] + [ap[i] - ap[i - 1] for i in (1, 2)]
    cfo = [None] + [ni[i] + is_["Depreciation and amortization"][i] - d_ar[i] - d_inv[i] + d_ap[i] for i in (1, 2)]
    s.write_row(f"A{r}", ["Cash flow statement", *yrs], bold)
    r += 1
    cf_lines = [("Net income", ni), ("Depreciation and amortization", is_["Depreciation and amortization"]),
                ("Increase in accounts receivable", [None] + [-x for x in d_ar[1:]]),
                ("Increase in inventory", [None] + [-x for x in d_inv[1:]]),
                ("Increase in accounts payable", d_ap), ("Cash flow from operating activities", cfo),
                ("Capital expenditures", [-x for x in capex])]
    for lab, v in cf_lines:
        s.write_row(f"A{r}", [lab, *["" if x is None else x for x in v]])
        r += 1
    s.set_column("A:A", 40)
    wb.close()
    return {"net_income": ni, "cfo": cfo, "revenue": is_["Revenue"], "receivables": bal["Accounts receivable"],
            "inventory": bal["Inventory"], "total_assets": ta, "total_equity": eq, "ebit": ebit, "tax": tax,
            "pretax": pretax, "interest": interest, "capex": capex, "cogs": is_["Cost of goods sold"], "ap": ap,
            "current_assets": ca, "current_liabilities": cl, "total_debt": [a + b for a, b in zip(std, ltd)]}


if __name__ == "__main__":
    import json
    facts = {"northwind_dcf.xlsx": northwind(HERE / "northwind_dcf.xlsx"),
             "harbor_financials.xlsx": harbor(HERE / "harbor_financials.xlsx"), "northwind_flaws": FLAWS}
    (HERE / "fixture_facts.json").write_text(json.dumps(facts, indent=1), encoding="utf-8")
    print("wrote fixtures:", ", ".join(k for k in facts if k.endswith(".xlsx")))
