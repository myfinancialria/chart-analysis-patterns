"""
Rule-based trade backtest of the BULLISH volume-confirmed breakouts, 10y NIFTY 500.

Strategy (long-only):
  ENTRY   next day's open after a volume-confirmed bullish breakout fires.
  STOP    just below the broken resistance/trendline level (the level price broke).
  TARGET  1:2 risk-reward  -> target = entry + 2R,  R = entry - stop.
  TRAIL   once the 2R target is reached, stop taking profit; instead HOLD and
          trail, exiting on the first daily CLOSE below the 50-day moving average.
  So a trade ends by: stop hit (before 2R) = loss; else 2R reached then rides the
  trend until it closes under the 50-DMA. One position per stock at a time.

Outputs output/trades.csv (full log) + output/trades_summary.json.
"""
from __future__ import annotations
import json, time
from pathlib import Path
import numpy as np, pandas as pd, yfinance as yf

from detect import _atr  # noqa
from breakouts import scan_extra

HERE = Path(__file__).parent
OUT = HERE / "output"; OUT.mkdir(exist_ok=True)
HIGH_CONF, VOL_SURGE_MIN = 0.60, 1.3
STEP, START = 5, 135
SL_BUFFER = 0.003          # stop this far below the broken level
TRADE_NAMES = {"Resistance breakout", "Range breakout (up)", "Trendline breakout (down-line)"}


def sim_trade(i_sig, level, o, h, l, c, dma, dates, n):
    """Simulate one long trade opened the day AFTER signal index i_sig."""
    i0 = i_sig + 1
    if i0 >= n:
        return None
    entry = o[i0]
    sl = level * (1 - SL_BUFFER)
    if not (sl < entry) or entry <= 0:
        return None
    R = entry - sl
    target = entry + 2 * R
    tgt_hit = False
    for i in range(i0, n):
        if not tgt_hit:
            if l[i] <= sl:
                return _rec(i0, i, entry, sl, sl, target, R, "Stopped out", dates)
            if h[i] >= target:
                tgt_hit = True
        if tgt_hit:
            if not np.isnan(dma[i]) and c[i] < dma[i]:
                return _rec(i0, i, entry, sl, c[i], target, R, "Trailed (50-DMA)", dates)
    return _rec(i0, n - 1, entry, sl, c[n - 1], target, R, "Open (end of data)", dates)


def _rec(i0, ix, entry, sl, exitp, target, R, outcome, dates):
    return {
        "entry_date": dates[i0], "entry": round(float(entry), 2),
        "stop": round(float(sl), 2), "target": round(float(target), 2),
        "exit_date": dates[ix], "exit": round(float(exitp), 2),
        "holding_days": int(ix - i0),
        "ret_pct": round((exitp / entry - 1) * 100, 2),
        "r_multiple": round((exitp - entry) / R, 2) if R > 0 else None,
        "outcome": outcome,
    }


def backtest_symbol(sym, sector, o, h, l, c, v, dma, dates):
    n = len(c)
    trades, t, guard = [], START, -1
    while t < n:
        if t <= guard:
            t += STEP; continue
        best = None
        for s in scan_extra(h[:t + 1], l[:t + 1], c[:t + 1], v[:t + 1], min_conf=0.45):
            if (s["name"] in TRADE_NAMES and s["bias"] == "Bullish"
                    and s["confidence"] >= HIGH_CONF
                    and s.get("vol_surge") is not None and not np.isnan(s.get("vol_surge", np.nan))
                    and s["vol_surge"] >= VOL_SURGE_MIN):
                lvl = s.get("upper_now")
                if lvl and not np.isnan(lvl) and (best is None or s["confidence"] > best[0]):
                    best = (s["confidence"], s["name"], float(lvl), round(float(s["vol_surge"]), 2))
        if best is None:
            t += STEP; continue
        tr = sim_trade(t, best[2], o, h, l, c, dma, dates, n)
        if tr:
            tr = {"symbol": sym, "sector": sector, "pattern": best[1],
                  "vol_surge": best[3], "signal_date": dates[t], **tr}
            trades.append(tr)
            guard = dates.index(tr["exit_date"]) if tr["exit_date"] in dates else t + tr["holding_days"] + 1
        t += STEP
    return trades


def main():
    uni = pd.read_csv(HERE / "data" / "nifty500_list.csv")
    sect = dict(zip(uni["Symbol"], uni["Industry"]))
    syms = [s for s in uni["Symbol"].dropna().unique().tolist() if "DUMMY" not in s.upper()]
    print(f"universe: {len(syms)} symbols")

    trades = []
    B = 90
    for i in range(0, len(syms), B):
        batch = syms[i:i + B]
        tick = {s: f"{s}.NS" for s in batch}
        try:
            data = yf.download(list(tick.values()), period="max", interval="1d",
                               auto_adjust=False, progress=False, threads=True, group_by="column")
        except Exception as e:
            print("  download fail:", e); continue
        for s in batch:
            ys = tick[s]
            try:
                sub = pd.DataFrame({k: data[k][ys] for k in ("Open", "High", "Low", "Close", "Volume")}).dropna()
            except Exception:
                continue
            sub = sub.tail(2600)
            if len(sub) < START + 60:
                continue
            o, h, l, c, v = (sub[k].values.astype(float) for k in ("Open", "High", "Low", "Close", "Volume"))
            dma = pd.Series(c).rolling(50).mean().values
            dates = [d.strftime("%Y-%m-%d") for d in sub.index]
            trades += backtest_symbol(s, sect.get(s, ""), o, h, l, c, v, dma, dates)
        print(f"  {min(i + B, len(syms))}/{len(syms)} · {len(trades)} trades")

    df = pd.DataFrame(trades)
    df = df[df["r_multiple"].notna()].reset_index(drop=True)
    df.to_csv(OUT / "trades.csv", index=False)
    n = len(df)
    wins = df[df["ret_pct"] > 0]
    losses = df[df["ret_pct"] <= 0]
    gp = wins["ret_pct"].sum(); gl = -losses["ret_pct"].sum()

    def pstats(g):
        w = g[g.ret_pct > 0]
        return {"trades": len(g), "win_rate": round((g.ret_pct > 0).mean() * 100, 1) if len(g) else 0,
                "avg_ret": round(g.ret_pct.mean(), 2) if len(g) else 0,
                "avg_R": round(g.r_multiple.mean(), 2) if len(g) else 0,
                "avg_hold": round(g.holding_days.mean(), 0) if len(g) else 0}

    # equity curve: sequence trades by exit date, compound 1% risk per trade (R-based)
    seq = df.sort_values("exit_date")
    eq, val = [], 100.0
    RISK = 0.01
    for _, r in seq.iterrows():
        val *= (1 + RISK * (r["r_multiple"] or 0))
        eq.append({"date": r["exit_date"], "eq": round(val, 2)})

    summary = {
        "as_of": time.strftime("%Y-%m-%d"), "universe": len(syms),
        "strategy": {"entry": "next-day open after volume-confirmed bullish breakout",
                     "stop": f"{SL_BUFFER*100:.1f}% below broken level", "target": "1:2 (2R)",
                     "trail": "after 2R, exit on close < 50-DMA", "patterns": sorted(TRADE_NAMES)},
        "trades": n,
        "win_rate": round((df.ret_pct > 0).mean() * 100, 1),
        "avg_ret": round(df.ret_pct.mean(), 2),
        "median_ret": round(df.ret_pct.median(), 2),
        "avg_R": round(df.r_multiple.mean(), 2),
        "avg_hold": round(df.holding_days.mean(), 0),
        "profit_factor": round(gp / gl, 2) if gl > 0 else None,
        "expectancy_R": round(df.r_multiple.mean(), 2),
        "best": round(df.ret_pct.max(), 1), "worst": round(df.ret_pct.min(), 1),
        "pct_hit_2R": round((df.outcome == "Trailed (50-DMA)").mean() * 100, 1),
        "pct_stopped": round((df.outcome == "Stopped out").mean() * 100, 1),
        "by_pattern": {p: pstats(g) for p, g in df.groupby("pattern")},
        "by_outcome": {o: pstats(g) for o, g in df.groupby("outcome")},
        "equity": eq[::max(1, len(eq)//400)],  # thin for the chart
        "equity_final": round(val, 2),
        # samples for the report: biggest winners, some losers, most recent
        "top_winners": df.sort_values("ret_pct", ascending=False).head(12).to_dict("records"),
        "recent": df.sort_values("signal_date", ascending=False).head(18).to_dict("records"),
    }
    (OUT / "trades_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(f"\nTRADES {n} | win {summary['win_rate']}% | avgR {summary['avg_R']} | PF {summary['profit_factor']} "
          f"| avg hold {summary['avg_hold']}d | 2R-reached {summary['pct_hit_2R']}% | equity100->{summary['equity_final']}")
    print("by pattern:", json.dumps(summary["by_pattern"], indent=2))


if __name__ == "__main__":
    main()
