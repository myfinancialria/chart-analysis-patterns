"""
Build the static dashboard (docs/) from a completed scan.

Renders one annotated candlestick chart per detected pattern and a single
self-contained index.html (data embedded inline, charts referenced relatively)
so it works opened straight from disk or served by any static host.

Called by scan.py --site, or standalone:
    python build_site.py            # uses output/patterns_latest.csv + latest cache
"""
from __future__ import annotations

import json
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

BIAS_COLOR = {"Bullish": "#16a34a", "Bearish": "#dc2626", "Neutral": "#64748b",
              "Up-trend": "#16a34a", "Down-trend": "#dc2626"}


def _nan(v):
    return v is None or (isinstance(v, float) and np.isnan(v))


def _render_chart(symbol: str, g: pd.DataFrame, row: pd.Series, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    g = g.sort_values("date").tail(int(row["window"]))
    o, h, l, c = (g[k].values for k in ("open", "high", "low", "close"))
    x = np.arange(len(c))
    category = row.get("category", "Pattern")

    fig, ax = plt.subplots(figsize=(5.4, 3.0))
    for j in range(len(c)):
        up = c[j] >= o[j]
        col = "#16a34a" if up else "#dc2626"
        ax.vlines(j, l[j], h[j], color="#9aa4b2", linewidth=0.5, zorder=1)
        ax.vlines(j, o[j], c[j], color=col, linewidth=2.0, zorder=2)

    xe = np.array([0, len(c) - 1])
    if category in ("Level", "Range"):
        # horizontal support / resistance boxes
        res, sup = row.get("res_level"), row.get("sup_level")
        if not _nan(res):
            ax.axhline(res, ls="--", color="#dc2626", lw=1.4)
        if not _nan(sup):
            ax.axhline(sup, ls="--", color="#16a34a", lw=1.4)
        if not _nan(res) and not _nan(sup):
            ax.axhspan(sup, res, color="#3b82f6", alpha=0.06)
    elif category == "Trendline" and not _nan(row.get("tl_slope")):
        # single sloping trendline that price has broken
        line = row["tl_slope"] * x + row["tl_intercept"]
        ax.plot(x, line, "--", color="#f59e0b", lw=1.5)
        ax.scatter([len(c) - 1], [c[-1]], s=46, color="#f59e0b",
                   marker="*", zorder=4)  # break point
    else:
        # two-rail patterns: fit upper/lower rails through the pivots
        hi_idx = find_pivots(h, 3, hi=True)
        lo_idx = find_pivots(l, 3, hi=False)
        if len(hi_idx) >= 2 and len(lo_idx) >= 2:
            su, iu = np.polyfit(hi_idx, h[hi_idx], 1)
            sl, il = np.polyfit(lo_idx, l[lo_idx], 1)
            ax.plot(xe, su * xe + iu, "--", color="#dc2626", lw=1.3)
            ax.plot(xe, sl * xe + il, "--", color="#16a34a", lw=1.3)
            ax.scatter(hi_idx, h[hi_idx], s=12, color="#dc2626", zorder=3)
            ax.scatter(lo_idx, l[lo_idx], s=12, color="#16a34a", zorder=3)

    ax.set_title(f"{symbol} — {row['name']}", fontsize=10, color="#e5e7eb")
    ax.set_xticks([])
    ax.tick_params(axis="y", labelsize=7, colors="#9aa4b2")
    for s in ax.spines.values():
        s.set_color("#334155")
    fig.patch.set_facecolor("#0f172a")
    ax.set_facecolor("#0f172a")
    fig.tight_layout()
    fig.savefig(path, dpi=110, facecolor="#0f172a", bbox_inches="tight")
    plt.close(fig)


def _latest_cache() -> pd.DataFrame:
    files = sorted(CACHE_DIR.glob("ohlcv_*.csv"))
    if not files:
        raise SystemExit("No cached candles found — run scan.py first.")
    return pd.read_csv(files[-1])


def build(df: pd.DataFrame | None = None, candles: pd.DataFrame | None = None) -> Path:
    if df is None:
        df = pd.read_csv(OUT_DIR / "patterns_latest.csv")
    if candles is None:
        candles = _latest_cache()

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
            "window": int(r["window"]), "chart": chart,
            "color": BIAS_COLOR.get(r["bias"], "#64748b"),
        })

    ist = timezone(timedelta(hours=5, minutes=30))
    built = datetime.now(timezone.utc).astimezone(ist)
    last_date = df["last_date"].iloc[0] if "last_date" in df and len(df) else str(date.today())
    counts = df["name"].value_counts().to_dict()

    breakouts = sum(1 for r in records if r["state"] in ("Breakout", "Breakdown"))
    meta = {
        "as_of": last_date,
        "built": built.strftime("%d %b %Y, %H:%M IST"),
        "total": len(records),
        "breakouts": breakouts,
        "counts": counts,
        "rows": records,
    }
    (DOCS / "data.json").write_text(json.dumps(meta, indent=2))
    (DOCS / "index.html").write_text(_html(meta))
    (DOCS / ".nojekyll").write_text("")
    return DOCS / "index.html"


def _html(meta: dict) -> str:
    data_json = json.dumps(meta)
    chips = "".join(
        f'<span class="chip">{n}<b>{c}</b></span>' for n, c in meta["counts"].items())
    return f"""<!DOCTYPE html>
<html lang="en"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<meta name="robots" content="noindex,nofollow">
<title>NIFTY 500 Chart Patterns</title>
<style>
:root{{--bg:#0b1220;--card:#0f172a;--line:#1e293b;--mut:#94a3b8;--fg:#e5e7eb}}
*{{box-sizing:border-box}}
body{{margin:0;background:var(--bg);color:var(--fg);
  font:15px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif}}
header{{padding:22px 20px 8px;max-width:1280px;margin:0 auto}}
h1{{margin:0 0 4px;font-size:22px}}
.sub{{color:var(--mut);font-size:13px}}
.chips{{margin:12px 0;display:flex;flex-wrap:wrap;gap:8px}}
.chip{{background:var(--card);border:1px solid var(--line);border-radius:999px;
  padding:4px 11px;font-size:12px;color:var(--mut)}}
.chip b{{color:var(--fg);margin-left:6px}}
.chip.hot{{background:#1e293b;border-color:#f59e0b;color:#fbbf24}}
.card.brk{{border-color:#f59e0b}}
.tag{{font-size:10px;font-weight:700;padding:1px 7px;border-radius:999px;
  background:#334155;color:#cbd5e1;margin-left:6px}}
.tag.brk{{background:#f59e0b;color:#0b1220}}
.bar{{max-width:1280px;margin:0 auto;padding:0 20px;display:flex;flex-wrap:wrap;
  gap:8px;align-items:center}}
input,select{{background:var(--card);border:1px solid var(--line);color:var(--fg);
  border-radius:8px;padding:8px 10px;font-size:14px}}
.grid{{max-width:1280px;margin:14px auto 60px;padding:0 20px;display:grid;
  grid-template-columns:repeat(auto-fill,minmax(330px,1fr));gap:16px}}
.card{{background:var(--card);border:1px solid var(--line);border-radius:14px;
  overflow:hidden;display:flex;flex-direction:column}}
.card img{{width:100%;display:block;background:#0f172a}}
.meta{{padding:11px 13px}}
.row1{{display:flex;justify-content:space-between;align-items:baseline;gap:8px}}
.sym{{font-weight:700;font-size:16px}}
.badge{{font-size:11px;font-weight:700;padding:2px 8px;border-radius:999px;color:#fff}}
.pat{{color:var(--mut);font-size:12.5px;margin:3px 0 8px}}
.stats{{display:flex;gap:14px;font-size:12px;color:var(--mut);flex-wrap:wrap}}
.stats b{{color:var(--fg)}}
.conf{{height:5px;background:var(--line);border-radius:3px;margin-top:9px;overflow:hidden}}
.conf>i{{display:block;height:100%;background:#3b82f6}}
footer{{max-width:1280px;margin:0 auto;padding:18px 20px 50px;color:var(--mut);
  font-size:11.5px;border-top:1px solid var(--line)}}
.dim{{color:#64748b}}
</style></head><body>
<header>
  <h1>NIFTY 500 — Chart Patterns</h1>
  <div class="sub">Daily EOD · patterns, S/R boxes &amp; breakouts · as of <b id="asof"></b>
    <span class="dim">· built <span id="built"></span></span></div>
  <div class="chips"><span class="chip hot">⚡ Fresh breakouts<b id="bocount"></b></span>{chips}</div>
</header>
<div class="bar">
  <input id="q" placeholder="Search symbol / sector…" style="flex:1;min-width:180px">
  <select id="fcat"><option value="">All types</option>
    <option value="Pattern">Patterns</option><option value="Level">S/R boxes</option>
    <option value="Range">Ranges</option><option value="Trendline">Trendlines</option></select>
  <select id="fstate"><option value="">Any state</option>
    <option value="Breakout">Breakout</option><option value="Breakdown">Breakdown</option>
    <option value="Forming">Forming</option><option value="Testing">Testing</option></select>
  <select id="fpat"><option value="">All names</option></select>
  <select id="fbias"><option value="">All bias</option></select>
  <select id="sort">
    <option value="confidence">Sort: confidence</option>
    <option value="to_upper">Sort: nearest resistance</option>
    <option value="to_lower">Sort: nearest support</option>
    <option value="symbol">Sort: symbol</option>
  </select>
</div>
<div class="grid" id="grid"></div>
<footer>
  <b>Educational / internal use only.</b> This is a descriptive geometric scan of
  historical price action — not investment advice, not a buy/sell recommendation,
  and not a price forecast. Prepared by a person who is <b>not</b> a SEBI-registered
  Research Analyst or Investment Adviser. Chart patterns fail often. Source: Fyers daily candles.
</footer>
<script>
const M = {data_json};
document.getElementById('asof').textContent = M.as_of;
document.getElementById('built').textContent = M.built;
document.getElementById('bocount').textContent = M.breakouts;
const grid = document.getElementById('grid');
const pats = [...new Set(M.rows.map(r=>r.name))].sort();
const biases = [...new Set(M.rows.map(r=>r.bias))].sort();
const fp = document.getElementById('fpat'), fb = document.getElementById('fbias');
pats.forEach(p=>fp.add(new Option(p,p)));
biases.forEach(b=>fb.add(new Option(b,b)));
const pct = v => v==null ? '—' : (v>=0?'+':'')+v+'%';
const isBrk = r => r.state==='Breakout' || r.state==='Breakdown';

function card(r){{
  let extra='';
  if(isBrk(r) && r.vol_surge!=null)
    extra=`<span title="breakout volume vs avg">· vol ×${{r.vol_surge}}</span>`;
  else if(r.vol_contraction!=null && r.vol_contraction<0.8)
    extra=`<span title="volume contracting">· vol ↓ ${{r.vol_contraction}}</span>`;
  const tag = r.state!=='Forming'
    ? `<span class="tag ${{isBrk(r)?'brk':''}}">${{isBrk(r)?'⚡ ':''}}${{r.state}}</span>` : '';
  return `<div class="card ${{isBrk(r)?'brk':''}}">
    <img loading="lazy" src="${{r.chart}}" alt="${{r.symbol}} ${{r.name}}">
    <div class="meta">
      <div class="row1"><span class="sym">${{r.symbol}}</span>
        <span class="badge" style="background:${{r.color}}">${{r.bias}}</span></div>
      <div class="pat">${{r.name}}${{tag}} · <span class="dim">${{r.sector}}</span></div>
      <div class="stats">
        <span>₹<b>${{r.close.toLocaleString('en-IN')}}</b></span>
        <span>resist <b>${{pct(r.to_upper)}}</b></span>
        <span>support <b>${{pct(r.to_lower)}}</b></span>
        ${{extra}}
      </div>
      <div class="conf"><i style="width:${{Math.round(r.confidence*100)}}%"></i></div>
    </div></div>`;
}}
function render(){{
  const q=document.getElementById('q').value.toLowerCase();
  const p=fp.value, b=fb.value, s=document.getElementById('sort').value;
  const cat=document.getElementById('fcat').value, st=document.getElementById('fstate').value;
  let rows=M.rows.filter(r=>(!p||r.name===p)&&(!b||r.bias===b)&&
    (!cat||r.category===cat)&&(!st||r.state===st)&&
    (!q||r.symbol.toLowerCase().includes(q)||(r.sector||'').toLowerCase().includes(q)));
  const key=r=> r[s]==null ? Infinity : r[s];
  rows.sort((a,z)=> s==='symbol' ? a.symbol.localeCompare(z.symbol)
    : s==='confidence' ? z.confidence-a.confidence : key(a)-key(z));
  grid.innerHTML = rows.map(card).join('') ||
    '<p class="dim">No signals match.</p>';
}}
['q','fcat','fstate','fpat','fbias','sort'].forEach(id=>
  document.getElementById(id).addEventListener('input',render));
render();
</script></body></html>"""


if __name__ == "__main__":
    print("Built:", build())
