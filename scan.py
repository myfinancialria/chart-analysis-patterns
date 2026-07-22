"""
Daily EOD chart-pattern scanner for the NIFTY 500.

Fetches ~1 year of daily candles from Fyers (cached per day), runs the
channel / triangle / wedge detector in detect.py over every stock, writes a
ranked CSV, draws annotated charts for the strongest setups, and (optionally)
posts an internal Slack digest.

Usage:
    python scan.py                       # full run, fetch + detect + charts
    python scan.py --cache               # reuse today's cached candles
    python scan.py --limit 40            # smoke-test on first 40 symbols
    python scan.py --min-conf 0.6        # stricter confidence floor
    python scan.py --no-charts           # skip chart montage
    python scan.py --slack               # also post to Slack (internal channel)

Compliance: output is descriptive ("X is forming a Y") and meant for the
internal Slack workspace only — no buy/sell calls, no public broadcast.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv

from detect import scan_symbol
from breakouts import scan_extra

HERE = Path(__file__).parent
ROOT = HERE.parent
CACHE_DIR = HERE / "cache"
OUT_DIR = HERE / "output"

CACHE_DIR.mkdir(exist_ok=True)
OUT_DIR.mkdir(exist_ok=True)

# Load env from the repo first, then the parent fyers-bot dir (local dev).
for env in (HERE / ".env", ROOT / ".env"):
    if env.exists():
        load_dotenv(env)


def _first_existing(*paths: Path) -> Path:
    """Return the first path that exists, else the first candidate (for error msgs)."""
    for p in paths:
        if p.exists():
            return p
    return paths[0]


# Symbol list + token resolve to the bundled repo copy first, parent dir as fallback.
LIST_PATH = _first_existing(HERE / "data" / "nifty500_list.csv",
                            ROOT / "data" / "nifty500_list.csv")
TOKEN_PATH = _first_existing(HERE / "access_token.txt", ROOT / "access_token.txt")


# ---------------- data ----------------
def load_universe() -> pd.DataFrame:
    if not LIST_PATH.exists():
        print("Symbol list missing — fetching NIFTY 500 list from NSE…")
        import io
        import requests
        r = requests.get(
            "https://archives.nseindia.com/content/indices/ind_nifty500list.csv",
            headers={"User-Agent": "Mozilla/5.0"}, timeout=20)
        r.raise_for_status()
        LIST_PATH.parent.mkdir(parents=True, exist_ok=True)
        pd.read_csv(io.StringIO(r.text)).to_csv(LIST_PATH, index=False)
    n500 = pd.read_csv(LIST_PATH)
    n500.columns = [c.strip() for c in n500.columns]
    return n500[["Symbol", "Industry"]].dropna().drop_duplicates("Symbol")


def fetch_history(symbols: list[str], days: int) -> pd.DataFrame:
    """Pull daily OHLCV for each symbol from Fyers. Long-format DataFrame."""
    from fyers_apiv3 import fyersModel

    client_id = os.getenv("FYERS_CLIENT_ID")
    token = os.getenv("FYERS_ACCESS_TOKEN") or (
        TOKEN_PATH.read_text().strip() if TOKEN_PATH.exists() else "")
    if not token:
        sys.exit("No Fyers token — set FYERS_ACCESS_TOKEN, provide access_token.txt, "
                 "or run auto_login.py first.")
    fyers = fyersModel.FyersModel(client_id=client_id, token=token, log_path="")

    prof = fyers.get_profile()
    if prof.get("s") != "ok":
        sys.exit(f"Fyers token invalid: {prof.get('message')}. Re-run login.py.")

    days = min(days, 365)  # Fyers caps daily-resolution history at ~366 days/request
    end_dt = date.today()
    start_dt = end_dt - timedelta(days=days)
    # Fyers data API limit is ~200 req/min — throttle to ~170/min with margin.
    THROTTLE = 0.36

    def fetch_one(sym: str):
        resp = fyers.history(data={
            "symbol": f"NSE:{sym}-EQ", "resolution": "D", "date_format": "1",
            "range_from": start_dt.strftime("%Y-%m-%d"),
            "range_to": end_dt.strftime("%Y-%m-%d"), "cont_flag": "1",
        })
        if resp.get("s") != "ok" or not resp.get("candles"):
            return None
        df = pd.DataFrame(resp["candles"],
                          columns=["ts", "open", "high", "low", "close", "volume"])
        df["symbol"] = sym
        df["date"] = (pd.to_datetime(df["ts"], unit="s", utc=True)
                      .dt.tz_convert("Asia/Kolkata").dt.strftime("%Y-%m-%d"))
        return df[["date", "symbol", "open", "high", "low", "close", "volume"]]

    def fetch_pass(syms: list[str], label: str):
        rows, failed = [], []
        for i, sym in enumerate(syms, 1):
            try:
                df = fetch_one(sym)
                (rows.append(df) if df is not None else failed.append(sym))
            except Exception:
                failed.append(sym)
            if i % 50 == 0:
                print(f"  {label}: {i}/{len(syms)}")
            time.sleep(THROTTLE)
        return rows, failed

    # skip placeholder/non-tradable tickers
    symbols = [s for s in symbols if "DUMMY" not in s.upper()]
    rows, failed = fetch_pass(symbols, "fetch")
    if failed:
        print(f"  retrying {len(failed)} stragglers after a pause…")
        time.sleep(5)
        more, failed = fetch_pass(failed, "retry")
        rows += more
    if failed:
        print(f"  {len(failed)} still failed: {', '.join(failed[:10])}"
              f"{' …' if len(failed) > 10 else ''}")

    if not rows:
        sys.exit("No history fetched — check token / connectivity.")
    return pd.concat(rows, ignore_index=True)


def get_candles(symbols: list[str], days: int, use_cache: bool) -> pd.DataFrame:
    cache_file = CACHE_DIR / f"ohlcv_{date.today():%Y%m%d}.csv"
    if use_cache and cache_file.exists():
        print(f"Using cached candles: {cache_file.name}")
        df = pd.read_csv(cache_file)
        return df[df["symbol"].isin(symbols)]
    print(f"Fetching {days}d daily candles for {len(symbols)} symbols from Fyers…")
    df = fetch_history(symbols, days)
    df.to_csv(cache_file, index=False)
    print(f"Cached -> {cache_file}")
    return df


# ---------------- scan ----------------
def run_scan(df: pd.DataFrame, sectors: dict, args) -> pd.DataFrame:
    results = []
    for sym, g in df.groupby("symbol", sort=False):
        g = g.sort_values("date")
        if len(g) < 60:
            continue
        hi, lo, cl, vol = (g["high"].values, g["low"].values,
                           g["close"].values, g["volume"].values)
        last_date = g["date"].iloc[-1]
        meta = {"symbol": sym, "sector": sectors.get(sym, "Unknown"),
                "last_date": last_date}

        # two-rail patterns (channels / triangles / wedges / rectangles)
        p = scan_symbol(hi, lo, cl, vol, min_conf=args.min_conf)
        if p is not None:
            results.append({**meta, "category": "Pattern", "state": "Forming",
                            **p.as_row()})

        # single-line signals (S/R boxes, range breakouts, trendline breaks)
        for sig in scan_extra(hi, lo, cl, vol, min_conf=args.min_conf):
            results.append({**meta, **sig})

    if not results:
        return pd.DataFrame()
    out = pd.DataFrame(results)
    # breakouts first (most actionable), then patterns, then levels being tested
    state_rank = {"Breakout": 0, "Breakdown": 0, "Forming": 1, "Testing": 2}
    out["_rank"] = out["state"].map(lambda s: state_rank.get(s, 3))
    out = out.sort_values(["_rank", "confidence"], ascending=[True, False]).drop(columns="_rank")
    return out.reset_index(drop=True)


# ---------------- high-conviction filter ----------------
def filter_high_conviction(out: pd.DataFrame, min_conf: float = 0.60,
                           vol_surge_min: float = 1.3,
                           vol_contract_max: float = 0.85) -> pd.DataFrame:
    """Keep only strong, directional, volume-confirmed setups:
      * bias is clearly Bullish or Bearish (drop Neutral / channel trends),
      * confidence >= min_conf,
      * VOLUME confirms the signal —
          breakouts/breakdowns need a volume surge (vol_surge >= vol_surge_min),
          forming triangles/wedges need volume drying up (vol_contraction <= vol_contract_max).
    Testing/"pressing a level" rows are dropped (no confirmation yet)."""
    if out.empty:
        return out
    d = out[out["bias"].isin(["Bullish", "Bearish"])].copy()
    d = d[pd.to_numeric(d["confidence"], errors="coerce") >= min_conf]

    def _vol_ok(r) -> bool:
        state = r.get("state")
        if state in ("Breakout", "Breakdown"):
            vs = r.get("vol_surge")
            return pd.notna(vs) and float(vs) >= vol_surge_min
        if state == "Forming":
            vc = r.get("vol_contraction")
            return pd.notna(vc) and float(vc) <= vol_contract_max
        return False  # Testing / anything unconfirmed

    d = d[d.apply(_vol_ok, axis=1)]
    return d.reset_index(drop=True)


# ---------------- charts ----------------
def draw_montage(df: pd.DataFrame, candles: pd.DataFrame, top_n: int) -> Path | None:
    # the montage draws two-rail patterns; breakouts/levels live on the dashboard
    df = df[df["category"] == "Pattern"]
    if df.empty:
        return None
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from detect import find_pivots

    picks = df.head(top_n)
    cols = 3
    rows = int(np.ceil(len(picks) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5.2, rows * 3.3))
    axes = np.atleast_1d(axes).ravel()

    for ax, (_, r) in zip(axes, picks.iterrows()):
        g = candles[candles["symbol"] == r["symbol"]].sort_values("date")
        lb = int(r["window"])
        g = g.tail(lb)
        h, l, c = g["high"].values, g["low"].values, g["close"].values
        x = np.arange(len(c))

        # candlesticks
        o = g["open"].values
        for j in range(len(c)):
            up = c[j] >= o[j]
            ax.vlines(j, l[j], h[j], color="#888", linewidth=0.5, zorder=1)
            ax.vlines(j, o[j], c[j], color="#1a9850" if up else "#d73027",
                      linewidth=2.2, zorder=2)

        # re-fit the two boundary lines for drawing
        hi_idx = find_pivots(h, 3, hi=True)
        lo_idx = find_pivots(l, 3, hi=False)
        if len(hi_idx) >= 2 and len(lo_idx) >= 2:
            su, iu = np.polyfit(hi_idx, h[hi_idx], 1)
            sl, il = np.polyfit(lo_idx, l[lo_idx], 1)
            ax.plot(x, su * x + iu, "--", color="#d73027", lw=1.3)
            ax.plot(x, sl * x + il, "--", color="#1a9850", lw=1.3)
            ax.scatter(hi_idx, h[hi_idx], s=14, color="#d73027", zorder=3)
            ax.scatter(lo_idx, l[lo_idx], s=14, color="#1a9850", zorder=3)

        ax.set_title(f"{r['symbol']} · {r['name']}\n"
                     f"{r['bias']} · conf {r['confidence']:.2f} · "
                     f"to-up {r['pct_to_upper']:+.1f}% / to-low {r['pct_to_lower']:+.1f}%",
                     fontsize=8.5)
        ax.set_xticks([])
        ax.tick_params(axis="y", labelsize=7)

    for ax in axes[len(picks):]:
        ax.axis("off")

    fig.suptitle(f"NIFTY 500 — chart patterns (daily EOD) · {date.today():%d %b %Y}",
                 fontsize=13, y=0.997)
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    path = OUT_DIR / "patterns_montage.png"
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return path


# ---------------- main ----------------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=365, help="calendar days of history (Fyers max ~365)")
    ap.add_argument("--limit", type=int, default=0, help="cap symbols (smoke test)")
    ap.add_argument("--min-conf", type=float, default=0.50, help="detection confidence floor")
    ap.add_argument("--cache", action="store_true", help="reuse today's cached candles")
    ap.add_argument("--no-charts", action="store_true")
    ap.add_argument("--charts", type=int, default=12, help="how many setups to chart")
    ap.add_argument("--slack", action="store_true", help="post digest to Slack")
    ap.add_argument("--site", action="store_true", help="build the docs/ dashboard")
    # high-conviction filter: surface only strong, volume-confirmed bull/bear setups
    ap.add_argument("--all-signals", action="store_true",
                    help="keep every detected signal (skip the high-conviction/volume filter)")
    ap.add_argument("--high-conf", type=float, default=0.60, help="confidence floor for surfaced setups")
    ap.add_argument("--vol-surge", type=float, default=1.3, help="min breakout volume vs base")
    ap.add_argument("--vol-contract", type=float, default=0.85, help="max forming-pattern volume ratio")
    ap.add_argument("--writeups", action="store_true", help="generate LLM write-ups for surfaced setups")
    args = ap.parse_args()

    uni = load_universe()
    sectors = dict(zip(uni["Symbol"], uni["Industry"]))
    symbols = uni["Symbol"].tolist()
    if args.limit:
        symbols = symbols[:args.limit]

    candles = get_candles(symbols, args.days, args.cache)
    print(f"Loaded {len(candles):,} candle rows for {candles['symbol'].nunique()} symbols")

    out = run_scan(candles, sectors, args)
    if out.empty:
        print("No patterns found at this confidence floor.")
        return 0

    if not args.all_signals:
        n_all = len(out)
        out = filter_high_conviction(out, args.high_conf, args.vol_surge, args.vol_contract)
        print(f"High-conviction filter: kept {len(out)}/{n_all} strong, volume-confirmed "
              f"bull/bear setups (conf>={args.high_conf}, breakout vol>={args.vol_surge}x, "
              f"forming vol<={args.vol_contract}).")
        if out.empty:
            print("No high-conviction setups today — building an empty dashboard.")

    csv_path = OUT_DIR / f"patterns_{date.today():%Y%m%d}.csv"
    out.to_csv(csv_path, index=False)
    latest = OUT_DIR / "patterns_latest.csv"
    out.to_csv(latest, index=False)

    print(f"\nFound {len(out)} signals "
          f"({(out['state'].isin(['Breakout', 'Breakdown'])).sum()} fresh breakouts):")
    for name, cnt in out["name"].value_counts().items():
        print(f"  {name:<26} {cnt}")
    print(f"\nTop signals:")
    cols = ["symbol", "category", "name", "bias", "state", "confidence",
            "pct_to_upper", "pct_to_lower"]
    print(out[cols].head(15).to_string(index=False))
    print(f"\nCSV -> {csv_path}")

    montage = None
    if not args.no_charts:
        montage = draw_montage(out, candles, args.charts)
        if montage:
            print(f"Charts -> {montage}")

    writeups = {}
    if args.writeups and not out.empty:
        try:
            import generate_writeups
            writeups = generate_writeups.generate(out)
        except Exception as exc:  # never let write-up trouble break the scan
            print(f"write-up generation skipped: {exc}")

    if args.site:
        import build_site
        idx = build_site.build(out, candles, writeups)
        print(f"Site -> {idx}")

    if args.slack:
        import slack_post
        slack_post.post(out, montage)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
