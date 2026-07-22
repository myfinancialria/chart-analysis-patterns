"""
Build the static dashboard (docs/) from a completed scan.

Renders one annotated candlestick chart per detected pattern (white background,
green up-days / red down-days, a daily-volume panel and a dated x-axis) plus a
single self-contained index.html grouped by category, with a plain-language
"can we take a trade?" note under every chart.

Called by scan.py --site, or standalone:
    python build_site.py            # uses output/patterns_latest.csv + latest cache
"""
from __future__ import annotations

import json
import shutil
from datetime import date, datetime, timezone, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from detect import find_pivots

HERE = Path(__file__).parent
OUT_DIR = HERE / "output"
CACHE_DIR = HERE / "cache"
DOCS = HERE / "docs"
CHARTS = DOCS / "charts"
HISTORY = DOCS / "history"
KEEP_DAYS = 90                    # retain this many daily snapshots

SCREENER = "https://www.screener.in/company/{sym}/"   # company / fundamentals / financials

UP, DOWN = "#16a34a", "#dc2626"          # green up-day, red down-day
BIAS_COLOR = {"Bullish": UP, "Bearish": DOWN, "Neutral": "#64748b",
              "Up-trend": UP, "Down-trend": DOWN}

# category -> friendly section title (order controls how sections stack)
CATEGORIES = [
    ("Pattern", "Chart Patterns"),
    ("Level", "Support / Resistance"),
    ("Range", "Ranges"),
    ("Trendline", "Trendlines"),
]

# verdict -> pill colour on the site
VERDICT_COLOR = {
    "Tradeable": "#16a34a",
    "Watch": "#d97706",
    "Avoid longs": "#dc2626",
    "No trade yet": "#64748b",
}


def _nan(v):
    return v is None or (isinstance(v, float) and np.isnan(v))


def _human_vol(v) -> str:
    """Indian-style short volume: 1.2 Cr / 3.4 L / 56.7K."""
    if v is None or _nan(v):
        return "—"
    v = float(v)
    if v >= 1e7:
        return f"{v / 1e7:.2f} Cr"
    if v >= 1e5:
        return f"{v / 1e5:.2f} L"
    if v >= 1e3:
        return f"{v / 1e3:.1f}K"
    return f"{int(v)}"


def _trade_note(row: pd.Series) -> tuple[str, str]:
    """Plain-language read on whether the setup is actionable yet.

    Returns (verdict, sentence). Educational only — descriptive, not advice.
    """
    state = str(row.get("state", "Forming"))
    bias = str(row.get("bias", ""))
    conf = float(row.get("confidence", 0) or 0)
    vs = row.get("vol_surge")
    vs = None if _nan(vs) else float(vs)
    bull = bias in ("Bullish", "Up-trend")
    bear = bias in ("Bearish", "Down-trend")

    if state == "Breakout":
        if conf >= 0.70 and vs and vs >= 1.5:
            return ("Tradeable",
                    f"Fresh breakout confirmed on {vs:.1f}× average volume. A long "
                    "can be considered on a close above the breakout level, with a "
                    "stop just below the base and risk kept to ~1–2% of capital.")
        if vs and vs >= 1.2:
            return ("Watch",
                    "Breakout underway but volume is only moderate — wait for a "
                    "decisive close above the level before committing.")
        return ("Watch",
                "Breakout not yet backed by a clear volume surge — treat as "
                "tentative and wait for confirmation.")

    if state == "Breakdown":
        return ("Avoid longs",
                "Support has given way — no fresh long here. Only aggressive "
                "traders short a weak pullback, with a stop above the broken level.")

    if state == "Testing":
        edge = "resistance" if not bear else "support"
        return ("Watch",
                f"Price is pressing {edge} — no trade yet. Set an alert and act only "
                "on a confirmed break backed by higher volume.")

    # Forming
    if bull:
        return ("No trade yet",
                "Constructive pattern still forming — not actionable. Wait for an "
                "upside breakout with a volume surge before considering a long.")
    if bear:
        return ("No trade yet",
                "Weak pattern still forming — avoid. A breakdown would confirm more "
                "downside; there is no long setup here.")
    return ("No trade yet",
            "Pattern still developing with no clear edge — stand aside until it "
            "resolves into a breakout or breakdown.")


def _render_chart(symbol: str, g: pd.DataFrame, row: pd.Series, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    g = g.sort_values("date").tail(int(row["window"]))
    o, h, l, c = (g[k].values for k in ("open", "high", "low", "close"))
    vol = g["volume"].values if "volume" in g else np.zeros(len(c))
    dates = pd.to_datetime(g["date"]).dt.strftime("%d %b").values
    x = np.arange(len(c))
    category = row.get("category", "Pattern")

    fig, (ax, axv) = plt.subplots(
        2, 1, figsize=(5.8, 3.9), sharex=True, constrained_layout=True,
        gridspec_kw={"height_ratios": [3.1, 1], "hspace": 0.06})

    # ---- price candles ----
    for j in range(len(c)):
        col = UP if c[j] >= o[j] else DOWN
        ax.vlines(j, l[j], h[j], color=col, linewidth=0.6, zorder=1, alpha=0.55)
        ax.vlines(j, o[j], c[j], color=col, linewidth=2.1, zorder=2)

    xe = np.array([0, len(c) - 1])
    if category in ("Level", "Range"):
        res, sup = row.get("res_level"), row.get("sup_level")
        if not _nan(res):
            ax.axhline(res, ls="--", color=DOWN, lw=1.3)
        if not _nan(sup):
            ax.axhline(sup, ls="--", color=UP, lw=1.3)
        if not _nan(res) and not _nan(sup):
            ax.axhspan(sup, res, color="#3b82f6", alpha=0.06)
    elif category == "Trendline" and not _nan(row.get("tl_slope")):
        line = row["tl_slope"] * x + row["tl_intercept"]
        ax.plot(x, line, "--", color="#d97706", lw=1.4)
        ax.scatter([len(c) - 1], [c[-1]], s=44, color="#d97706", marker="*", zorder=4)
    else:
        hi_idx = find_pivots(h, 3, hi=True)
        lo_idx = find_pivots(l, 3, hi=False)
        if len(hi_idx) >= 2 and len(lo_idx) >= 2:
            su, iu = np.polyfit(hi_idx, h[hi_idx], 1)
            sl, il = np.polyfit(lo_idx, l[lo_idx], 1)
            ax.plot(xe, su * xe + iu, "--", color=DOWN, lw=1.2)
            ax.plot(xe, sl * xe + il, "--", color=UP, lw=1.2)
            ax.scatter(hi_idx, h[hi_idx], s=11, color=DOWN, zorder=3)
            ax.scatter(lo_idx, l[lo_idx], s=11, color=UP, zorder=3)

    ax.set_title(f"{symbol} — {row['name']}", fontsize=10, color="#111827",
                 fontweight="bold")
    ax.set_ylabel("Price ₹", fontsize=7.5, color="#374151")
    ax.grid(axis="y", color="#eef2f7", lw=0.8, zorder=0)
    ax.tick_params(axis="y", labelsize=7, colors="#6b7280")
    ax.tick_params(axis="x", length=0)

    # ---- volume panel ----
    vcol = [UP if c[j] >= o[j] else DOWN for j in range(len(c))]
    axv.bar(x, vol, color=vcol, width=0.72, alpha=0.85, zorder=2)
    axv.set_ylabel("Vol", fontsize=7.5, color="#374151")
    axv.grid(axis="y", color="#eef2f7", lw=0.8, zorder=0)
    axv.tick_params(axis="y", labelsize=6, colors="#9ca3af")
    axv.yaxis.set_major_formatter(plt.FuncFormatter(
        lambda y, _: _human_vol(y) if y > 0 else ""))

    # ---- dated x-axis (on the bottom/volume panel) ----
    n = len(x)
    step = max(1, n // 6)
    ticks = list(range(0, n, step))
    if ticks[-1] != n - 1:
        ticks.append(n - 1)
    axv.set_xticks(ticks)
    axv.set_xticklabels([dates[i] for i in ticks], fontsize=6.5, color="#6b7280",
                        rotation=0)
    axv.set_xlim(-0.8, n - 0.2)

    for a in (ax, axv):
        a.set_facecolor("#ffffff")
        for s in a.spines.values():
            s.set_color("#d1d5db")
        a.spines["top"].set_visible(False)
        a.spines["right"].set_visible(False)

    fig.patch.set_facecolor("#ffffff")
    fig.savefig(path, dpi=115, facecolor="#ffffff", bbox_inches="tight")
    plt.close(fig)


def _latest_cache() -> pd.DataFrame:
    files = sorted(CACHE_DIR.glob("ohlcv_*.csv"))
    if not files:
        raise SystemExit("No cached candles found — run scan.py first.")
    return pd.read_csv(files[-1])


def build(df: pd.DataFrame | None = None, candles: pd.DataFrame | None = None,
          writeups: dict | None = None) -> Path:
    if df is None:
        df = pd.read_csv(OUT_DIR / "patterns_latest.csv")
    if candles is None:
        candles = _latest_cache()
    writeups = writeups or {}

    DOCS.mkdir(exist_ok=True)
    CHARTS.mkdir(exist_ok=True)
    for old in CHARTS.glob("*.png"):
        old.unlink()

    records = []
    for _, r in df.iterrows():
        g = candles[candles["symbol"] == r["symbol"]]
        if g.empty:
            continue
        chart = f"charts/{r['symbol']}.png"
        _render_chart(r["symbol"], g, r, DOCS / chart)

        def num(key, nd=1):
            v = r.get(key)
            return None if pd.isna(v) else round(float(v), nd)

        gs = g.sort_values("date")
        vol_last = int(gs["volume"].iloc[-1]) if "volume" in gs else None
        vol_avg = (int(gs["volume"].tail(20).mean())
                   if "volume" in gs and len(gs) else None)
        verdict, note = _trade_note(r)

        records.append({
            "symbol": r["symbol"], "sector": r.get("sector", ""),
            "category": r.get("category", "Pattern"),
            "state": r.get("state", "Forming"),
            "name": r["name"], "bias": r["bias"],
            "confidence": round(float(r["confidence"]), 2),
            "close": float(r["close"]),
            "to_upper": num("pct_to_upper"), "to_lower": num("pct_to_lower"),
            "vol_contraction": num("vol_contraction", 2),
            "vol_surge": num("vol_surge", 2),
            "volume": vol_last, "vol_avg": vol_avg,
            "window": int(r["window"]), "chart": chart,
            "color": BIAS_COLOR.get(r["bias"], "#64748b"),
            "verdict": verdict, "note": note,
            "vcolor": VERDICT_COLOR.get(verdict, "#64748b"),
            "ai_note": writeups.get(f"{r['symbol']}|{r['name']}"),
        })

    ist = timezone(timedelta(hours=5, minutes=30))
    built = datetime.now(timezone.utc).astimezone(ist)
    last_date = df["last_date"].iloc[0] if "last_date" in df and len(df) else str(date.today())
    counts = df["name"].value_counts().to_dict()

    # tag each signal as new-today / new-this-week vs the prior daily snapshots
    _tag_new(records, str(last_date))

    breakouts = sum(1 for r in records if r["state"] in ("Breakout", "Breakdown"))
    tradeable = sum(1 for r in records if r["verdict"] == "Tradeable")
    meta = {
        "as_of": last_date,
        "built": built.strftime("%d %b %Y, %H:%M IST"),
        "total": len(records),
        "breakouts": breakouts,
        "tradeable": tradeable,
        "new_today": sum(1 for r in records if r.get("new_today")),
        "new_week": sum(1 for r in records if r.get("new_week")),
        "counts": counts,
        "rows": records,
    }

    # --- daily archive: snapshot this scan under history/<date>/, keep 90 days ---
    dates = _archive(meta, str(last_date))
    meta["dates"] = dates

    (DOCS / "data.json").write_text(json.dumps(meta, indent=2))
    (DOCS / "index.html").write_text(_html(meta))
    (DOCS / ".nojekyll").write_text("")
    return DOCS / "index.html"


def _tag_new(records: list[dict], today: str) -> None:
    """Flag each signal new_today / new_week / first_seen by comparing (symbol,
    pattern) against the prior daily snapshots in history/. A signal is
    new_today if it wasn't in the most recent earlier scan; new_week if it hadn't
    appeared in any scan more than 7 days ago (i.e. first surfaced this week)."""
    prior: dict[str, set] = {}
    if HISTORY.exists():
        for d in HISTORY.iterdir():
            j = d / "data.json"
            if d.is_dir() and d.name < today and j.exists():
                try:
                    rows = json.loads(j.read_text()).get("rows", [])
                    prior[d.name] = {f"{r.get('symbol')}|{r.get('name')}" for r in rows}
                except Exception:
                    pass
    dates = sorted(prior)
    yset = prior[dates[-1]] if dates else set()
    try:
        cutoff = (date.fromisoformat(today) - timedelta(days=7)).isoformat()
    except ValueError:
        cutoff = today
    older = set().union(*[prior[d] for d in dates if d <= cutoff]) if dates else set()
    first_seen: dict[str, str] = {}
    for d in dates:                       # ascending → earliest wins
        for k in prior[d]:
            first_seen.setdefault(k, d)
    for r in records:
        k = f"{r['symbol']}|{r['name']}"
        r["new_today"] = k not in yset
        r["new_week"] = k not in older
        r["first_seen"] = first_seen.get(k, today)


def _archive(meta: dict, day: str) -> list[str]:
    """Snapshot today's data.json + charts into history/<day>/, prune >KEEP_DAYS.

    Returns the list of available snapshot dates (newest first) for the date filter.
    """
    HISTORY.mkdir(exist_ok=True)
    snap = HISTORY / day
    scharts = snap / "charts"
    scharts.mkdir(parents=True, exist_ok=True)
    for old in scharts.glob("*.png"):
        old.unlink()
    for png in CHARTS.glob("*.png"):
        shutil.copy2(png, scharts / png.name)
    (snap / "data.json").write_text(json.dumps(meta))

    # prune: keep only the newest KEEP_DAYS snapshots (ISO names sort chronologically)
    snaps = sorted((d for d in HISTORY.iterdir() if d.is_dir()), reverse=True)
    for d in snaps[KEEP_DAYS:]:
        shutil.rmtree(d, ignore_errors=True)

    dates = [d.name for d in sorted(HISTORY.iterdir(), reverse=True) if d.is_dir()]
    (HISTORY / "manifest.json").write_text(json.dumps({"dates": dates}))
    return dates


def _html(meta: dict) -> str:
    data_json = json.dumps(meta)
    cats_json = json.dumps(CATEGORIES)
    chips = "".join(
        f'<span class="chip">{n}<b>{c}</b></span>' for n, c in meta["counts"].items())
    return f"""<!DOCTYPE html>
<html lang="en"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>NIFTY 500 Chart Patterns</title>
<style>
:root{{--bg:#ffffff;--card:#ffffff;--line:#e5e7eb;--mut:#6b7280;--fg:#111827;
  --soft:#f9fafb;--up:#16a34a;--down:#dc2626}}
*{{box-sizing:border-box}}
body{{margin:0;background:var(--bg);color:var(--fg);
  font:15px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif}}
header{{padding:24px 20px 8px;max-width:1320px;margin:0 auto}}
h1{{margin:0 0 4px;font-size:23px}}
.sub{{color:var(--mut);font-size:13px}}
.chips{{margin:12px 0;display:flex;flex-wrap:wrap;gap:8px}}
.rlinks{{margin:10px 0 0;display:flex;flex-wrap:wrap;gap:16px;font-size:13px}}
.rlinks a{{color:#2563eb;text-decoration:none;font-weight:600}}
.rlinks a:hover{{text-decoration:underline}}
.ftabs{{margin:14px 0 2px;display:flex;flex-wrap:wrap;gap:8px}}
.ftab{{font:600 13px inherit;color:var(--mut);background:var(--card);border:1px solid var(--line);
  border-radius:999px;padding:7px 14px;cursor:pointer;display:inline-flex;align-items:center;gap:2px}}
.ftab:hover{{border-color:#94a3b8}}
.ftab.active{{background:#111827;color:#fff;border-color:#111827}}
.ftab b{{margin-left:5px;font-weight:800;opacity:.85}}
.badge-new{{font-size:9.5px;font-weight:800;letter-spacing:.03em;color:#fff;background:#e11d48;
  padding:1px 6px;border-radius:999px;margin-left:6px;vertical-align:middle}}
.badge-new.wk{{background:#0e7490}}
.chip{{background:var(--soft);border:1px solid var(--line);border-radius:999px;
  padding:4px 11px;font-size:12px;color:var(--mut)}}
.chip b{{color:var(--fg);margin-left:6px}}
.chip.hot{{background:#fff7ed;border-color:#fdba74;color:#c2410c}}
.chip.go{{background:#f0fdf4;border-color:#86efac;color:#15803d}}
.tag{{font-size:10px;font-weight:700;padding:1px 7px;border-radius:999px;
  background:#eef2f7;color:#475569;margin-left:6px}}
.tag.brk{{background:#f59e0b;color:#fff}}
.bar{{max-width:1320px;margin:0 auto;padding:0 20px;display:flex;flex-wrap:wrap;
  gap:8px;align-items:center}}
input,select{{background:#fff;border:1px solid var(--line);color:var(--fg);
  border-radius:8px;padding:8px 10px;font-size:14px}}
.catwrap{{max-width:1320px;margin:0 auto;padding:0 20px}}
h2.cat{{font-size:15px;margin:26px 0 2px;padding-bottom:6px;
  border-bottom:2px solid var(--line);display:flex;align-items:baseline;gap:8px}}
h2.cat span{{font-size:12px;color:var(--mut);font-weight:500}}
.grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(340px,1fr));
  gap:16px;margin:14px 0 8px}}
.card{{background:var(--card);border:1px solid var(--line);border-radius:14px;
  overflow:hidden;display:flex;flex-direction:column;box-shadow:0 1px 2px rgba(0,0,0,.04)}}
.card.brk{{border-color:#f59e0b}}
.card img{{width:100%;display:block;background:#fff}}
.meta{{padding:11px 13px}}
.row1{{display:flex;justify-content:space-between;align-items:baseline;gap:8px}}
.sym{{font-weight:700;font-size:16px}}
.badge{{font-size:11px;font-weight:700;padding:2px 8px;border-radius:999px;color:#fff}}
.pat{{color:var(--mut);font-size:12.5px;margin:3px 0 8px}}
.stats{{display:flex;gap:14px;font-size:12px;color:var(--mut);flex-wrap:wrap}}
.stats b{{color:var(--fg)}}
.conf{{height:5px;background:var(--line);border-radius:3px;margin:9px 0 0;overflow:hidden}}
.conf>i{{display:block;height:100%;background:#3b82f6}}
.note{{margin:10px 0 0;padding:9px 11px;background:var(--soft);border:1px solid var(--line);
  border-radius:9px;font-size:12.5px;line-height:1.45;color:#374151}}
.ainote{{margin:8px 0 0;padding:9px 11px;background:#f5f8ff;border:1px solid #dbe6ff;
  border-radius:9px;font-size:12.5px;line-height:1.5;color:#1f2937}}
.ainote .ailbl{{display:inline-block;font-size:10px;font-weight:800;letter-spacing:.03em;
  text-transform:uppercase;color:#3b5bdb;margin-bottom:4px}}
.ainote ul{{margin:4px 0 0;padding-left:18px}} .ainote li{{margin:2px 0}}
.ainote b{{color:#111827}}
.verdict{{display:inline-block;font-size:10.5px;font-weight:800;letter-spacing:.02em;
  text-transform:uppercase;padding:2px 8px;border-radius:999px;color:#fff;margin-bottom:5px}}
a.symlink{{color:#15803d;text-decoration:none;font-weight:700;font-size:16px}}
a.symlink:hover{{text-decoration:underline}}
.fund{{margin-top:8px;font-size:12px}}
.fund a{{color:#2563eb;text-decoration:none;font-weight:600}}
.fund a:hover{{text-decoration:underline}}
footer{{max-width:1320px;margin:0 auto;padding:22px 20px 60px;color:var(--mut);
  font-size:11.5px;border-top:1px solid var(--line)}}
.dim{{color:#9ca3af}}
.empty{{color:#9ca3af;padding:8px 0}}
</style></head><body>
<header>
  <h1>NIFTY 500 — Daily Chart Patterns</h1>
  <div class="sub">Daily EOD scan · only <b>strong, volume-confirmed</b> bullish &amp;
    bearish setups · with a plain-English AI read on each · as of <b id="asof"></b>
    <span class="dim">· built <span id="built"></span></span></div>
  <div class="chips">
    <span class="chip go">✅ Tradeable now<b id="tcount"></b></span>
    <span class="chip hot">⚡ Fresh breakouts<b id="bocount"></b></span>{chips}</div>
  <div class="rlinks">
    <a href="backtest/" target="_blank" rel="noopener">🎯 10-year backtest &amp; results ↗</a>
    <a href="backtest/report_trade_backtest.html" target="_blank" rel="noopener">📈 Trade-system report ↗</a>
    <a href="backtest/nifty500_breakout_trades.csv">⬇ Trade log (CSV)</a>
  </div>
  <div class="ftabs" id="ftabs">
    <button class="ftab active" data-nf="">All signals</button>
    <button class="ftab" data-nf="today">⭐ New today<b id="ntcount"></b></button>
    <button class="ftab" data-nf="week">🆕 New this week<b id="nwcount"></b></button>
  </div>
</header>
<div class="bar">
  <input id="q" placeholder="Search symbol / sector…" style="flex:1;min-width:180px">
  <select id="fdate" title="Scan date"></select>
  <select id="fstate"><option value="">Any state</option>
    <option value="Breakout">Breakout</option><option value="Breakdown">Breakdown</option>
    <option value="Forming">Forming</option><option value="Testing">Testing</option></select>
  <select id="fverdict"><option value="">Any verdict</option>
    <option value="Tradeable">Tradeable</option><option value="Watch">Watch</option>
    <option value="Avoid longs">Avoid longs</option><option value="No trade yet">No trade yet</option></select>
  <select id="fpat"><option value="">All names</option></select>
  <select id="fbias"><option value="">All bias</option></select>
  <select id="sort">
    <option value="confidence">Sort: confidence</option>
    <option value="to_upper">Sort: nearest resistance</option>
    <option value="to_lower">Sort: nearest support</option>
    <option value="volume">Sort: volume</option>
    <option value="symbol">Sort: symbol</option>
  </select>
</div>
<div class="catwrap" id="cats"></div>
<footer>
  <b>Educational / informational only.</b> This is a descriptive geometric scan of
  historical price action — not investment advice, not a buy/sell recommendation,
  and not a price forecast. The trade notes (rule-based) and the "AI read" (generated
  by a language model from the scan's own numbers) are automated, descriptive
  observations about setup readiness, not personalised advice. Prepared by a person who is
  <b>not</b> a SEBI-registered Research Analyst or Investment Adviser. Chart patterns
  fail often; always do your own research and manage risk. Source: Fyers daily candles.
</footer>
<script>
const M = {data_json};
const CATS = {cats_json};
const SCREENER = "https://www.screener.in/company/";
document.getElementById('built').textContent = M.built;
const cats = document.getElementById('cats');
const fp = document.getElementById('fpat'), fb = document.getElementById('fbias');
const fd = document.getElementById('fdate');
const pct = v => v==null ? '—' : (v>=0?'+':'')+v+'%';
const vol = v => v==null ? '—' : v>=1e7?(v/1e7).toFixed(2)+' Cr'
  : v>=1e5?(v/1e5).toFixed(2)+' L' : v>=1e3?(v/1e3).toFixed(1)+'K' : v;
const isBrk = r => r.state==='Breakout' || r.state==='Breakdown';

// current view: rows for the selected date + the path prefix for its charts
let CUR = M.rows, BASE = '';

// date picker — newest first; latest === M.as_of uses the inlined data
(M.dates && M.dates.length ? M.dates : [M.as_of]).forEach(d=>
  fd.add(new Option(d===M.as_of ? d+' (latest)' : d, d)));
fd.value = M.as_of;

async function loadDate(d){{
  if(d===M.as_of){{ CUR=M.rows; BASE=''; }}
  else {{
    try{{
      const res=await fetch(`history/${{d}}/data.json`,{{cache:'no-store'}});
      const j=await res.json(); CUR=j.rows||[]; BASE=`history/${{d}}/`;
    }}catch(e){{ CUR=[]; BASE=''; }}
  }}
  refreshFilters();
  document.getElementById('asof').textContent = d;
  document.getElementById('bocount').textContent =
    CUR.filter(r=>r.state==='Breakout'||r.state==='Breakdown').length;
  document.getElementById('tcount').textContent =
    CUR.filter(r=>r.verdict==='Tradeable').length;
  const nt=CUR.filter(r=>r.new_today).length, nw=CUR.filter(r=>r.new_week).length;
  document.getElementById('ntcount').textContent = nt;
  document.getElementById('nwcount').textContent = nw;
  render();
}}
function refreshFilters(){{
  const pv=fp.value, bv=fb.value;
  fp.innerHTML='<option value="">All names</option>';
  fb.innerHTML='<option value="">All bias</option>';
  [...new Set(CUR.map(r=>r.name))].sort().forEach(p=>fp.add(new Option(p,p)));
  [...new Set(CUR.map(r=>r.bias))].sort().forEach(b=>fb.add(new Option(b,b)));
  if([...fp.options].some(o=>o.value===pv)) fp.value=pv;
  if([...fb.options].some(o=>o.value===bv)) fb.value=bv;
}}

function card(r){{
  const tag = r.state!=='Forming'
    ? `<span class="tag ${{isBrk(r)?'brk':''}}">${{isBrk(r)?'⚡ ':''}}${{r.state}}</span>` : '';
  const volx = r.vol_avg ? ` <span class="dim">(avg ${{vol(r.vol_avg)}})</span>` : '';
  const url = SCREENER + encodeURIComponent(r.symbol) + '/';
  const tradeable = r.verdict==='Tradeable';
  // tradeable symbols link out to company / fundamentals / financials
  const sym = tradeable
    ? `<a class="sym symlink" href="${{url}}" target="_blank" rel="noopener">${{r.symbol}} ↗</a>`
    : `<span class="sym">${{r.symbol}}</span>`;
  const fund = tradeable
    ? `<div class="fund"><a href="${{url}}" target="_blank" rel="noopener">📊 Company, fundamentals &amp; financials · industry overview ↗</a></div>`
    : '';
  const ai = r.ai_note
    ? `<div class="ainote"><span class="ailbl">🤖 AI read</span>${{mdLite(r.ai_note)}}</div>` : '';
  const nb = r.new_today ? '<span class="badge-new" title="first appeared today">⭐ NEW</span>'
    : (r.new_week ? '<span class="badge-new wk" title="first appeared this week">🆕 wk</span>' : '');
  return `<div class="card ${{isBrk(r)?'brk':''}}">
    <img loading="lazy" src="${{BASE}}${{r.chart}}" alt="${{r.symbol}} ${{r.name}}">
    <div class="meta">
      <div class="row1">${{sym}}
        <span class="badge" style="background:${{r.color}}">${{r.bias}}</span>${{nb}}</div>
      <div class="pat">${{r.name}}${{tag}} · <span class="dim">${{r.sector}}</span></div>
      <div class="stats">
        <span>₹<b>${{r.close.toLocaleString('en-IN')}}</b></span>
        <span>vol <b>${{vol(r.volume)}}</b>${{volx}}</span>
        <span>resist <b>${{pct(r.to_upper)}}</b></span>
        <span>support <b>${{pct(r.to_lower)}}</b></span>
      </div>
      <div class="conf"><i style="width:${{Math.round(r.confidence*100)}}%"></i></div>
      <div class="note"><span class="verdict" style="background:${{r.vcolor}}">${{r.verdict}}</span><br>${{r.note}}</div>
      ${{ai}}
      ${{fund}}
    </div></div>`;
}}
function mdLite(s){{
  let t=(s||'').replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
  const parts=t.split('**'); t=parts.map((x,i)=>i%2?('<b>'+x+'</b>'):x).join('');
  const lines=t.split('\\n').map(x=>x.trim()).filter(Boolean);
  const isB=x=>x.startsWith('* ')||x.startsWith('- ')||x.startsWith('• ');
  let html='',inul=false;
  for(const ln of lines){{
    if(isB(ln)){{ if(!inul){{html+='<ul>';inul=true;}} html+='<li>'+ln.slice(2)+'</li>'; }}
    else {{ if(inul){{html+='</ul>';inul=false;}} html+='<p style="margin:4px 0 0">'+ln+'</p>'; }}
  }}
  if(inul) html+='</ul>';
  return html;
}}
let NF='';  // '' | 'today' | 'week' — the New-signal filter tab
function render(){{
  const q=document.getElementById('q').value.toLowerCase();
  const p=fp.value, b=fb.value, s=document.getElementById('sort').value;
  const st=document.getElementById('fstate').value;
  const vd=document.getElementById('fverdict').value;
  let rows=CUR.filter(r=>(!p||r.name===p)&&(!b||r.bias===b)&&
    (!st||r.state===st)&&(!vd||r.verdict===vd)&&
    (NF===''||(NF==='today'&&r.new_today)||(NF==='week'&&r.new_week))&&
    (!q||r.symbol.toLowerCase().includes(q)||(r.sector||'').toLowerCase().includes(q)));
  const key=r=> r[s]==null ? -Infinity : r[s];
  rows.sort((a,z)=> s==='symbol' ? a.symbol.localeCompare(z.symbol)
    : (s==='confidence'||s==='volume') ? key(z)-key(a) : key(a)-key(z));
  let html='';
  for(const [cat,title] of CATS){{
    const rs=rows.filter(r=>r.category===cat);
    if(!rs.length) continue;
    html+=`<h2 class="cat">${{title}} <span>${{rs.length}} chart${{rs.length>1?'s':''}}</span></h2>`
      +`<div class="grid">${{rs.map(card).join('')}}</div>`;
  }}
  cats.innerHTML = html || '<p class="empty">No signals match your filters.</p>';
}}
['q','fstate','fverdict','fpat','fbias','sort'].forEach(id=>
  document.getElementById(id).addEventListener('input',render));
document.querySelectorAll('.ftab').forEach(btn=>btn.addEventListener('click',()=>{{
  NF=btn.dataset.nf;
  document.querySelectorAll('.ftab').forEach(b=>b.classList.toggle('active',b===btn));
  render();
}}));
fd.addEventListener('change',()=>loadDate(fd.value));
loadDate(M.as_of);
</script></body></html>"""


if __name__ == "__main__":
    print("Built:", build())
