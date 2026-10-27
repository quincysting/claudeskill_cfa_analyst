# Calculation conventions

Read this before calling or writing any `cfa_calc` function. The point of the engine is that every number in an
answer is reproducible: same inputs, same output, no mental arithmetic.

## Calling

```bash
python scripts/cfa.py list fixed_income            # what exists (topics: quant, economics, fra, corporate_finance,
                                                   #   equity, fixed_income, derivatives, alternatives, portfolio, pwm)
python scripts/cfa.py search duration              # find by keyword
python scripts/cfa.py describe bond_price          # signature, units, formula, curriculum source
python scripts/cfa.py bond_price face=100 coupon_rate=0.05 ytm=0.06 years=10 freq=2
python scripts/cfa.py npv '{"rate": 0.08, "cash_flows": [-100, 30, 40, 50]}'
```

For a multi-step analysis, write a short Python script that imports the library and prints every intermediate value:

```python
import sys; sys.path.insert(0, r"<skill>/scripts")
import cfa_calc as cfa          # every formula is a package attribute; no need to know its module
p = cfa.bond_price(face=100, coupon_rate=0.05, ytm=0.06, years=10, freq=2)
print(p)
```

## Units and signs

| Quantity | Convention |
|---|---|
| Rates, yields, returns, growth, volatility, tax rates, weights | decimals: 5% is `0.05` |
| Time | years (`years`, `t`), unless the parameter says `periods` or `days` |
| Compounding frequency | `freq` = payments/compounding periods per year (1 annual, 2 semiannual, 4, 12, 365) |
| Cash flows | list, first element at t = 0 unless the parameter says otherwise; outflows negative, inflows positive |
| Prices | per the stated `face`/`par` (default 100) |
| Day counts | money market: actual/360 unless `days_in_year` is given; bond accrued interest: the `day_count` parameter |
| Basis points | inputs and outputs in decimals; a function returning bp says so in the key name (`spread_bp`) |
| Currency quotes | `price/base`, e.g. USD/EUR = 1.10 means 1 EUR costs 1.10 USD (CFA curriculum convention) |

## Results

Every function returns a `dict` of named, unrounded values: the headline result first, then the intermediate values a
reader needs to follow the calculation (e.g. `{"price": 92.56, "pv_coupons": ..., "pv_face": ...}`). Round only when
presenting, and say what you rounded.

## Writing a formula (for maintainers)

```python
from ._registry import formula

@formula("fixed_income", formula="P = sum c/(1+y/m)^t + F/(1+y/m)^N",
         curriculum="L1 FI: Introduction to Fixed-Income Valuation")
def bond_price(face: float = 100.0, coupon_rate: float = 0.0, ytm: float = 0.0, years: float = 1.0,
               freq: int = 2) -> dict:
    """Price of an option-free bond on a coupon date from its yield to maturity.

    Args:
        face: par value.
        coupon_rate: annual coupon rate, decimal.
        ytm: annual yield to maturity (stated rate, compounded `freq` times a year), decimal.
        years: years to maturity; years * freq must be a whole number of periods.
        freq: coupons per year.

    Returns:
        price, pv_coupons, pv_face, periods.

    Example:
        bond_price(face=100, coupon_rate=0.05, ytm=0.06, years=10, freq=2) -> price 92.56
    """
```

- Pure functions: no I/O, no global state. Randomness only with an explicit `seed` argument (default 42).
- Keyword arguments with type hints; validate inputs and raise `ValueError` with a message that says what is wrong.
- Names are global across modules (`python cfa.py <name>`), so make them specific (`bond_price`, not `price`).
- numpy and scipy are available; prefer the standard library when it is just as clear.
- Every function gets at least one test whose expected value comes from a CFA curriculum worked example (cite the
  knowledge-base file and page in a comment) or a standard textbook example (cite it), plus consistency tests where
  they exist (parity relations, numeric vs analytic derivatives, round trips such as price -> yield -> price).
