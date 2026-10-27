"""Tests for cfa_calc.portfolio_trading. Golden values:
R32 = E:/CLAUDE/CFA/MD/Level-3/12-Trading-Monitoring-Rebalancing/Curriculum-2014/SS16-R32-execution-of-portfolio-decisions.md
"""
import pytest

from cfa_calc import portfolio_trading as tr

A = pytest.approx


def test_implementation_shortfall_impulse_robotics():  # R32 p.39-40: 87 bp = 14 + 14 + 35 + 24; market-adjusted -0.13%
    r = tr.implementation_shortfall(decision_price=10.00, shares_ordered=1000, fill_shares=[700], fill_prices=[10.07],
                                    fill_arrival_prices=[10.05], final_price=10.08, commissions=14, market_return=0.01)
    assert r["shortfall"] == A(0.0087)
    assert (r["explicit_cost"], r["realized_pl"], r["delay_cost"], r["missed_trade_cost"]) == \
        A((0.0014, 0.0014, 0.0035, 0.0024))
    assert (r["paper_gain"], r["actual_gain"]) == A((80, -7))
    assert r["market_adjusted_shortfall"] == A(-0.0013)


def test_implementation_shortfall_two_day_acme():  # R32 Prob 11 p.75-76: 67 bp (delay -0.01%, realized 0.21%)
    r = tr.implementation_shortfall(decision_price=10.00, shares_ordered=1000, fill_shares=[600, 100],
                                    fill_prices=[10.02, 10.08], fill_arrival_prices=[10.00, 9.99], final_price=10.05,
                                    commissions=32)
    assert r["shortfall"] == A(0.0067)
    assert (r["delay_cost"], r["realized_pl"], r["missed_trade_cost"], r["explicit_cost"]) == \
        A((-0.0001, 0.0021, 0.0015, 0.0032))


def test_implementation_shortfall_sell_and_prob23():  # R32 Prob 10 p.75 (sell, EUR6,000); Prob 23 p.77-78 ($2,434)
    s = tr.implementation_shortfall(decision_price=53.25, shares_ordered=30000, fill_shares=[10000] * 3,
                                    fill_prices=[53.22, 53.06, 52.87], final_price=52.87, side="sell")
    assert s["shortfall_amount"] == A(6000)
    b = tr.implementation_shortfall(decision_price=17.05, shares_ordered=10000, fill_shares=[3200, 3000],
                                    fill_prices=[17.05, 17.25], final_price=17.50, commissions=124)
    assert b["shortfall_amount"] == A(2434)
    assert (b["paper_gain"], b["actual_gain"]) == A((4500, 2066))


def test_effective_spread_example1():  # R32 Ex.1 p.27-28: 0.06/0.11/0.22; avg 0.13; weighted 0.155; quoted avg 0.1233
    r = tr.effective_spread(trade_prices=[10.15, 10.11, 10.01], bids=[10.12, 10.11, 10.05], asks=[10.24, 10.22, 10.19],
                            sides=["sell"] * 3, trade_sizes=[200, 300, 500])
    assert r["effective_spreads"] == A([0.06, 0.11, 0.22])
    assert r["average_effective_spread"] == A(0.13) and r["weighted_average_effective_spread"] == A(0.155)
    assert r["average_quoted_spread"] == A(0.1233, abs=5e-5)


def test_effective_spread_buys():  # R32 p.27 (20.01 vs 19.97/20.03 -> 0.02); Prob 1 p.73 (0.04); Q20 p.71 ($0.07)
    assert tr.effective_spread(trade_prices=[20.01], bids=[19.97], asks=[20.03], sides="buy")["effective_spreads"] == \
        A([0.02])
    assert tr.effective_spread(trade_prices=[4.74], bids=[4.69], asks=[4.75], sides="buy")["effective_spreads"] == A([0.04])
    assert tr.effective_spread(trade_prices=[17.07], bids=[17.02], asks=[17.05], sides="buy")["effective_spreads"] == \
        A([0.07])


def test_vwap():  # R32 Prob 8 p.74-75 (VWAP 2.7053, sell cost $4.53); p.37 (buy 500 @ 157.25 vs 156.00 -> EUR625)
    r = tr.vwap_comparison(trade_prices=[2.66], trade_sizes=[100], side="sell",
                           market_prices=[2.71, 2.72, 2.76, 2.77, 2.70, 2.66, 2.65],
                           market_volumes=[200, 200, 100, 100, 1100, 100, 100])
    assert r["vwap"] == A(2.7053, abs=5e-5) and r["cost_amount"] == A(4.53, abs=0.005)
    assert tr.vwap_comparison(trade_prices=[157.25], trade_sizes=[500], market_vwap=156.00)["cost_amount"] == A(625)
    assert tr.vwap_comparison(trade_prices=[17.05], trade_sizes=[3200], market_vwap=17.00)["cost_amount"] == A(160)


def test_validation():
    with pytest.raises(ValueError):
        tr.implementation_shortfall(decision_price=10, shares_ordered=100, fill_shares=[200], fill_prices=[10],
                                    final_price=10)
    with pytest.raises(ValueError):
        tr.effective_spread(trade_prices=[1], bids=[1], asks=[1], sides=["hold"])
