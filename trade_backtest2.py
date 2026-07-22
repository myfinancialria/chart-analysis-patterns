"""
Full trade backtest -> CSV, two entry styles, position-sized to a fixed rupee risk.

For every volume-confirmed bullish breakout on the NIFTY 500 (10y), across ALL
eligible stocks, it records trades for BOTH entry styles:

  BREAKOUT  enter next day's open after the break.  Stop just below the level.
  RETEST    wait for price to pull back to the broken level (former resistance,
            now support) within RETEST_WINDOW days and hold; enter at the level.
            Stop RETEST_STOP below it. (Skipped if support breaks first or price
            never pulls back.)

Both use target = entry + 2R and, once 2R is hit, trail out on the first close
below the 50-DMA. POSITION SIZING: qty = floor(MAX_LOSS / (entry-stop)) so the
worst-case loss on any position is ~MAX_LOSS (Rs 10,000). Capital and rupee P&L
are recorded per trade.

Outputs: output/nifty500_breakout_trades.csv (full log) and
         output/backtest_summary_by_type.csv (aggregates).
"""
from __future__ import annotations
import time, math
from pathlib import Path
import numpy as np, pandas as pd, yfinance as yf
from breakouts import scan_extra

HERE = Path(__file__).parent
OUT = HERE / "output"; OUT.mkdir(exist_ok=True)
HIGH_CONF, VOL_SURGE_MIN = 0.60, 1.3
STEP, START = 5, 135
BO_STOP = 0.003            # breakout stop: 0.3% below level
RETEST_WINDOW = 25         # days to wait for a pullback
RETEST_STOP = 0.03         # retest stop: 3% below the level
MAX_LOSS = 10_000          # rupees risked per position
TRADE_NAMES = {"Resistance breakout", "Range breakout (up)", "Trendline breakout (down-line)"}


def _size(entry, stop):
    rps = entry - stop
    if rps <= 0:
        return None
    qty = int(math.floor(MAX_LOSS / rps))
    return (round(rps, 2), qty) if qty >= 1 else None


def _run(entry, stop, i0, o, h, l, c, dma, dates, n):
    """Simulate one long trade from entry_idx i0. Returns exit dict or None."""
    if i0 >= n or entry <= 0 or stop >= entry:
        return None
    R = entry - stop
    target = entry + 2 * R
    tgt = False
    for i in range(i0, n):
        if not tgt:
            if l[i] <= stop:
                return _mk(i0, i, entry, stop, stop, target, R, "Stopped out", dates)
            if h[i] >= target:
                tgt = True
        if tgt and not np.isnan(dma[i]) and c[i] < dma[i]:
            return _mk(i0, i, entry, stop, c[i], target, R, "Trailed (50-DMA)", dates)
    return _mk(i0, n - 1, entry, stop, c[n - 1], target, R, "Open (end of data)", dates)


def _mk(i0, ix, entry, stop, exitp, target, R, outcome, dates):
    rps, qty = _size(entry, stop) or (None, None)
    if qty is None:
        return None
    pnl = qty * (exitp - entry)
    return {"entry_date": dates[i0], "entry": round(float(entry), 2), "stop": round(float(stop), 2),
            "target": round(float(target), 2), "risk_per_share": rps, "qty": qty,
            "capital": round(qty * entry, 0), "exit_date": dates[ix], "exit": round(float(exitp), 2),
            "holding_days": int(ix - i0), "ret_pct": round((exitp / entry - 1) * 100, 2),
            "r_multiple": round((exitp - entry) / R, 2), "pnl_rupees": round(pnl, 0), "outcome": outcome}


def find_retest(t, L, o, h, l, c, dates, n):
    """Index price first pulls back to level L (support holds), within the window."""
    for j in range(t + 1, min(t + 1 + RETEST_WINDOW, n)):
        if c[j] < L * (1 - RETEST_STOP):     # support broke before a clean retest
            return None
        if l[j] <= L:                        # pulled back to the level
            return j
    return None


def backtest_symbol(sym, sector, o, h, l, c, v, dma, dates):
    n = len(c); rows = []
    guard = {"Breakout": -1, "Retest": -1}
    t = START
    while t < n:
        best = None
        for s in scan_extra(h[:t + 1], l[:t + 1], c[:t + 1], v[:t + 1], min_conf=0.45):
            if (s["name"] in TRADE_NAMES and s["bias"] == "Bullish" and s["confidence"] >= HIGH_CONF
                    and s.get("vol_surge") is not None and not np.isnan(s.get("vol_surge", np.nan))
                    and s["vol_surge"] >= VOL_SURGE_MIN):
                lvl = s.get("upper_now")
                if lvl and not np.isnan(lvl) and (best is None or s["confidence"] > best[0]):
                    best = (s["confidence"], s["name"], float(lvl), round(float(s["vol_surge"]), 2))
        if best is None:
            t += STEP; continue
        conf, name, L, vs = best
        meta = {"symbol": sym, "sector": sector, "pattern": name, "vol_surge": vs, "signal_date": dates[t]}
        # BREAKOUT entry (next open)
        if t > guard["Breakout"] and t + 1 < n:
            tr = _run(o[t + 1], L * (1 - BO_STOP), t + 1, o, h, l, c, dma, dates, n)
            if tr:
                rows.append({**meta, "entry_type": "Breakout", **tr})
                guard["Breakout"] = dates.index(tr["exit_date"])
        # RETEST entry (pullback to the level)
        if t > guard["Retest"]:
            j = find_retest(t, L, o, h, l, c, dates, n)
            if j is not None:
                tr = _run(L, L * (1 - RETEST_STOP), j, o, h, l, c, dma, dates, n)
                if tr:
                    rows.append({**meta, "entry_type": "Retest", **tr})
                    guard["Retest"] = dates.index(tr["exit_date"])
        t += STEP
    return rows


def main():
    uni = pd.read_csv(HERE / "data" / "nifty500_list.csv")
    sect = dict(zip(uni["Symbol"], uni["Industry"]))
    syms = [s for s in uni["Symbol"].dropna().unique().tolist() if "DUMMY" not in s.upper()]
    print(f"universe: {len(syms)} symbols")
    rows, B = [], 90
    for i in range(0, len(syms), B):
        batch = syms[i:i + B]; tick = {s: f"{s}.NS" for s in batch}
        try:
            data = yf.download(list(tick.values()), period="max", interval="1d",
                               auto_adjust=False, progress=False, threads=True, group_by="column")
        except Exception as e:
            print("  dl fail", e); continue
        for s in batch:
            try:
                sub = pd.DataFrame({k: data[k][tick[s]] for k in ("Open", "High", "Low", "Close", "Volume")}).dropna().tail(2600)
            except Exception:
                continue
            if len(sub) < START + 60:
                continue
            o, h, l, c, vv = (sub[k].values.astype(float) for k in ("Open", "High", "Low", "Close", "Volume"))
            dma = pd.Series(c).rolling(50).mean().values
            dates = [d.strftime("%Y-%m-%d") for d in sub.index]
            rows += backtest_symbol(s, sect.get(s, ""), o, h, l, c, vv, dma, dates)
        print(f"  {min(i+B,len(syms))}/{len(syms)} · {len(rows)} trades")

    cols = ["symbol", "sector", "pattern", "entry_type", "signal_date", "entry_date", "entry",
            "stop", "target", "risk_per_share", "qty", "capital", "exit_date", "exit",
            "holding_days", "ret_pct", "r_multiple", "pnl_rupees", "outcome"]
    df = pd.DataFrame(rows)[cols].sort_values(["symbol", "signal_date", "entry_type"]).reset_index(drop=True)
    df.to_csv(OUT / "nifty500_breakout_trades.csv", index=False)

    def summ(g):
        wins = g[g.pnl_rupees > 0]; gl = -g[g.pnl_rupees <= 0].pnl_rupees.sum()
        return pd.Series({
            "trades": len(g), "stocks": g.symbol.nunique(),
            "win_rate_%": round((g.pnl_rupees > 0).mean() * 100, 1),
            "avg_R": round(g.r_multiple.mean(), 2), "avg_ret_%": round(g.ret_pct.mean(), 2),
            "avg_hold_days": round(g.holding_days.mean(), 0),
            "total_pnl_rupees": round(g.pnl_rupees.sum(), 0),
            "avg_pnl_rupees": round(g.pnl_rupees.mean(), 0),
            "avg_capital": round(g.capital.mean(), 0),
            "profit_factor": round(wins.pnl_rupees.sum() / gl, 2) if gl > 0 else None,
            "pct_hit_2R": round((g.outcome == "Trailed (50-DMA)").mean() * 100, 1)})

    by = df.groupby("entry_type").apply(summ).reset_index()
    byp = df.groupby(["entry_type", "pattern"]).apply(summ).reset_index()
    tot = summ(df); tot["entry_type"] = "ALL"
    out = pd.concat([by, pd.DataFrame([tot])[by.columns]], ignore_index=True)
    out.to_csv(OUT / "backtest_summary_by_type.csv", index=False)
    byp.to_csv(OUT / "backtest_summary_by_pattern.csv", index=False)

    print("\n=== summary by entry type ===")
    print(out.to_string(index=False))
    print(f"\nTotal trades {len(df)} across {df.symbol.nunique()} stocks · "
          f"total P&L Rs {df.pnl_rupees.sum():,.0f} · CSV -> {OUT/'nifty500_breakout_trades.csv'}")


if __name__ == "__main__":
    main()
