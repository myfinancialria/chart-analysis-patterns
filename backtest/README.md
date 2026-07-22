# NIFTY 500 chart-pattern backtest (10 years)

Walk-forward backtests of the live scanner's pattern engine (`detect.py` +
`breakouts.py`) over ~10 years of daily data (Yahoo Finance) on all 500 current
NIFTY 500 stocks. Same detectors and same high-conviction volume filter used on
the dashboard, run on trailing data only (no look-ahead).

## Files

| File | What it is |
|---|---|
| **`nifty500_breakout_trades.csv`** | The full trade log — **1,955 trades** across 393 stocks, both entry styles, position-sized. One row per trade. |
| `backtest_summary_by_type.csv` | Aggregates for Breakout vs Retest vs All. |
| `backtest_summary_by_pattern.csv` | Aggregates split by entry style × pattern. |
| `report_trade_backtest.html` | Visual report of the breakout trade system. |
| `report_pattern_returns.html` | Visual report of the raw pattern-return study. |

Regenerate from the repo root: `python trade_backtest2.py` (trade log + summaries),
`python backtest.py` (raw pattern forward-returns).

## Trade rules

- **Signals:** volume-confirmed **bullish breakouts** only — Resistance breakout,
  Range breakout (up), down-trendline breakout — confidence ≥ 0.60, breakout
  volume ≥ 1.3× average.
- **Two entry styles** (both in the CSV, `entry_type` column):
  - **Breakout** — enter next day's open after the break. Stop 0.3% below the level.
  - **Retest** — wait up to 25 days for price to pull back to the broken level and
    hold, then enter at the level. Stop 3% below it. (Skipped if support breaks
    first or price never pulls back.)
- **Target** = entry + 2R (1:2). **Trail:** after 2R is reached, hold and exit on
  the first daily **close below the 50-day MA**.
- **Position sizing:** `qty = floor(10,000 / (entry − stop))` → worst-case loss
  ≈ **₹10,000** per position. `capital` and `pnl_rupees` are per trade.

## CSV columns

`symbol, sector, pattern, entry_type, signal_date, entry_date, entry, stop,
target, risk_per_share, qty, capital, exit_date, exit, holding_days, ret_pct,
r_multiple, pnl_rupees, outcome`

`outcome` ∈ {`Trailed (50-DMA)` = hit 2R then trailed out, `Stopped out`,
`Open (end of data)`}.

## Headline result

| Entry | Trades | Win % | Avg R | Profit factor | Total ₹ P&L | Avg ₹/trade | Hit 2R |
|---|---|---|---|---|---|---|---|
| Breakout | 1,245 | 31.1 | +0.49 | 1.69 | ₹60.7 L | ₹4,875 | 36% |
| **Retest** | 710 | 30.6 | **+1.12** | **2.55** | **₹79.3 L** | **₹11,166** | **43.5%** |
| All | 1,955 | 30.9 | +0.72 | 2.01 | **₹1.40 Cr** | ₹7,160 | 38.7% |

**Waiting for the retest of the broken level more than doubled the edge per trade**
— a tighter entry means a bigger R-multiple when it works. Win rate is ~31% for
both: the system is carried by a minority of big trend-riders, so it only works
with identical small risk on every trade.

## Caveats

Current-membership NIFTY 500 (**survivorship bias** — flatters results); entries/
exits at modelled prices with **no brokerage, slippage or gap-through-stop cost**;
trades overlap across stocks, so totals are fixed-₹-risk-per-trade, **not a
compounded account**; one decade is a single, broadly-rising market regime.
**Descriptive research — not investment advice, not a forecast.** Prepared by a
person who is not a SEBI-registered Research Analyst or Investment Adviser.
