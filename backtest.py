"""
10-year walk-forward backtest of the chart-pattern engine on the NIFTY 500.

For each stock, step weekly through history; at each step run the SAME detectors
(detect.scan_symbol + breakouts.scan_extra) and the SAME high-conviction volume
filter used live, on trailing data only (no look-ahead). Record the forward
return over several horizons, DIRECTION-ADJUSTED (long a Bullish signal, short a
Bearish one), then aggregate by pattern.

Caveats (printed in the report): current-membership NIFTV 500 (survivorship
bias), EOD signals entered at the signal-day close, no costs/slippage, and
overlapping signals de-duplicated per (symbol, pattern) within DEDUP days.
"""
from __future__ import annotations
import json, sys, time
from pathlib import Path
import numpy as np, pandas as pd, yfinance as yf

from detect import scan_symbol
from breakouts import scan_extra

HERE = Path(__file__).parent
OUT = HERE / "output"; OUT.mkdir(exist_ok=True)
HIGH_CONF, VOL_SURGE_MIN, VOL_CONTRACT_MAX = 0.60, 1.3, 0.85
STEP, DEDUP = 5, 20          # scan weekly; one signal per (sym,pattern) per ~month
HORIZONS = (10, 20, 60)      # trading-day forward windows
START_BARS = 135             # need this much trailing history before scanning


def _passes(bias, state, conf, vs, vc) -> bool:
    if bias not in ("Bullish", "Bearish") or conf < HIGH_CONF:
        return False
    if state in ("Breakout", "Breakdown"):
        return vs is not None and not np.isnan(vs) and vs >= VOL_SURGE_MIN
    if state == "Forming":
        return vc is not None and not np.isnan(vc) and vc <= VOL_CONTRACT_MAX
    return False


def backtest_symbol(sym, h, l, c, v) -> list[dict]:
    n = len(c)
    recs, last_fire = [], {}
    for t in range(START_BARS, n, STEP):
        ht, lt, ct, vt = h[:t + 1], l[:t + 1], c[:t + 1], v[:t + 1]
        sigs = []
        p = scan_symbol(ht, lt, ct, vt, min_conf=0.45)
        if p is not None:
            r = p.as_row()
            sigs.append((r["name"], r["bias"], "Forming", r["confidence"],
                         r.get("vol_surge", np.nan), r.get("vol_contraction", np.nan)))
        for s in scan_extra(ht, lt, ct, vt, min_conf=0.45):
            sigs.append((s["name"], s["bias"], s.get("state", "Forming"), s["confidence"],
                         s.get("vol_surge", np.nan), s.get("vol_contraction", np.nan)))
        for name, bias, state, conf, vs, vc in sigs:
            if not _passes(bias, state, conf,
                           float(vs) if vs is not None else np.nan,
                           float(vc) if vc is not None else np.nan):
                continue
            if name in last_fire and t - last_fire[name] < DEDUP:
                continue
            last_fire[name] = t
            entry = c[t]
            rec = {"symbol": sym, "name": name, "bias": bias, "state": state, "conf": round(conf, 3)}
            for hz in HORIZONS:
                if t + hz < n and entry > 0:
                    fwd = c[t + hz] / entry - 1.0
                    rec[f"ret{hz}"] = fwd if bias == "Bullish" else -fwd  # directional
                else:
                    rec[f"ret{hz}"] = np.nan
            recs.append(rec)
    return recs


def main():
    syms = pd.read_csv(HERE / "data" / "nifty500_list.csv")["Symbol"].dropna().unique().tolist()
    syms = [s for s in syms if "DUMMY" not in s.upper()]
    print(f"universe: {len(syms)} symbols")

    all_recs, base_fwd = [], {hz: [] for hz in HORIZONS}
    B = 90
    for i in range(0, len(syms), B):
        batch = syms[i:i + B]
        tick = {s: f"{s}.NS" for s in batch}
        try:
            data = yf.download(list(tick.values()), period="max", interval="1d",
                               auto_adjust=False, progress=False, threads=True, group_by="column")
        except Exception as e:
            print(f"  batch {i} download failed: {e}"); continue
        if data is None or data.empty:
            continue
        for s in batch:
            ys = tick[s]
            try:
                sub = pd.DataFrame({k: data[k][ys] for k in ("High", "Low", "Close", "Volume")}).dropna()
            except Exception:
                continue
            sub = sub.tail(2600)  # ~10y
            if len(sub) < START_BARS + max(HORIZONS) + 20:
                continue
            h, l, c, vv = (sub["High"].values.astype(float), sub["Low"].values.astype(float),
                           sub["Close"].values.astype(float), sub["Volume"].values.astype(float))
            # baseline: forward returns of random long entries (weekly), same horizons
            for t in range(START_BARS, len(c), STEP):
                for hz in HORIZONS:
                    if t + hz < len(c) and c[t] > 0:
                        base_fwd[hz].append(c[t + hz] / c[t] - 1.0)
            all_recs.extend(backtest_symbol(s, h, l, c, vv))
        print(f"  processed {min(i + B, len(syms))}/{len(syms)} symbols · {len(all_recs)} signals so far")

    df = pd.DataFrame(all_recs)
    df.to_csv(OUT / "backtest_signals.csv", index=False)
    print(f"\nTotal signals: {len(df)}")

    # ---- aggregate by pattern ----
    def agg(g):
        row = {"signals": len(g)}
        for hz in HORIZONS:
            col = g[f"ret{hz}"].dropna()
            row[f"n{hz}"] = len(col)
            row[f"mean{hz}"] = round(float(col.mean()) * 100, 2) if len(col) else None
            row[f"median{hz}"] = round(float(col.median()) * 100, 2) if len(col) else None
            row[f"win{hz}"] = round(float((col > 0).mean()) * 100, 1) if len(col) else None
        return pd.Series(row)

    by_pattern = df.groupby(["name", "bias"]).apply(agg).reset_index().sort_values("mean20", ascending=False)
    by_state = df.groupby("state").apply(agg).reset_index()
    baseline = {hz: {"mean": round(float(np.mean(base_fwd[hz])) * 100, 2),
                     "median": round(float(np.median(base_fwd[hz])) * 100, 2),
                     "win": round(float((np.array(base_fwd[hz]) > 0).mean()) * 100, 1),
                     "n": len(base_fwd[hz])} for hz in HORIZONS}

    summary = {
        "as_of": time.strftime("%Y-%m-%d"),
        "universe": len(syms),
        "horizons": list(HORIZONS),
        "filter": {"high_conf": HIGH_CONF, "vol_surge_min": VOL_SURGE_MIN, "vol_contract_max": VOL_CONTRACT_MAX,
                   "step_days": STEP, "dedup_days": DEDUP},
        "total_signals": len(df),
        "baseline_long": baseline,
        "by_pattern": by_pattern.to_dict("records"),
        "by_state": by_state.to_dict("records"),
    }
    (OUT / "backtest_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print("\n=== by pattern (dir-adjusted, 20d) ===")
    print(by_pattern[["name", "bias", "signals", "mean20", "median20", "win20", "mean60", "win60"]].to_string(index=False))
    print("\nbaseline (random long):", baseline)
    print(f"\nWrote {OUT/'backtest_summary.json'} and backtest_signals.csv")


if __name__ == "__main__":
    main()
