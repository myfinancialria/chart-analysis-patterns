import json, datetime
from pathlib import Path
import pandas as pd, numpy as np

s = json.load(open("output/trades_summary.json"))
df = pd.read_csv("output/trades.csv").sort_values("exit_date").reset_index(drop=True)
df["cumR"] = df["r_multiple"].cumsum()
peak = df["cumR"].cummax(); maxdd = round((df["cumR"] - peak).min(), 1)
totR = round(df["r_multiple"].sum(), 1)
w = df[df.ret_pct > 0]; l = df[df.ret_pct <= 0]

# ---- cumulative-R sparkline (SVG) over the date axis ----
pts = df[["exit_date", "cumR"]].values
idx = np.linspace(0, len(pts) - 1, min(260, len(pts))).astype(int)
xs = np.array([datetime.date.fromisoformat(pts[i][0]).toordinal() for i in idx], float)
ys = np.array([pts[i][1] for i in idx], float)
W, H, PAD = 860, 240, 6
sx = (xs - xs.min()) / (xs.max() - xs.min()) * (W - 2 * PAD) + PAD
sy = H - PAD - (ys - ys.min()) / (ys.max() - ys.min()) * (H - 2 * PAD)
line = " ".join(f"{x:.1f},{y:.1f}" for x, y in zip(sx, sy))
area = f"{sx[0]:.1f},{H-PAD} " + line + f" {sx[-1]:.1f},{H-PAD}"
yr_ticks = ""
for yr in range(2017, 2027, 2):
    o = datetime.date(yr, 1, 1).toordinal()
    if xs.min() <= o <= xs.max():
        x = (o - xs.min()) / (xs.max() - xs.min()) * (W - 2 * PAD) + PAD
        yr_ticks += f'<line x1="{x:.0f}" y1="0" x2="{x:.0f}" y2="{H}" class="grid"/><text x="{x:.0f}" y="{H-4}" class="axl">{yr}</text>'

def money(v): return f"₹{v:,.2f}"

def trrow(r):
    oc = r["outcome"]
    ocls = "ok" if oc.startswith("Trailed") else ("bad" if oc.startswith("Stopped") else "open")
    rcls = "pos" if r["ret_pct"] > 0 else "neg"
    return f"""<tr>
      <td class="tname">{r['symbol']}</td><td class="dim2">{r['pattern'].replace(' (up)','').replace(' (down-line)','')}</td>
      <td class="mono">{r['entry_date']}</td><td class="mono r">{money(r['entry'])}</td>
      <td class="mono r">{money(r['stop'])}</td><td class="mono r">{money(r['target'])}</td>
      <td class="mono">{r['exit_date']}</td><td class="mono r">{money(r['exit'])}</td>
      <td class="mono r">{int(r['holding_days'])}</td>
      <td class="mono r {rcls}">{r['ret_pct']:+.1f}%</td>
      <td class="mono r {rcls}">{r['r_multiple']:+.2f}</td>
      <td><span class="oc {ocls}">{oc.replace(' (50-DMA)','·50DMA').replace('Stopped out','Stopped').replace(' (end of data)','')}</span></td>
    </tr>"""

recent = "".join(trrow(r) for r in s["recent"])
winners = "".join(trrow(r) for r in s["top_winners"])
bp = s["by_pattern"]
def bprow(p, g):
    return f"""<tr><td class="tname">{p}</td><td class="mono r">{g['trades']}</td>
      <td class="mono r">{g['win_rate']:.0f}%</td><td class="mono r pos">{g['avg_ret']:+.2f}%</td>
      <td class="mono r">{g['avg_R']:+.2f}</td><td class="mono r">{int(g['avg_hold'])}d</td></tr>"""
bprows = "".join(bprow(p, g) for p, g in sorted(bp.items(), key=lambda kv: -kv[1]["avg_R"]))
built = datetime.date.today().strftime("%d %b %Y")

html = f"""<style>
:root{{--bg:#eef0f3;--paper:#fff;--ink:#0f172a;--mut:#5b6472;--faint:#8a94a3;--line:#e3e7ec;
  --accent:#0e7490;--bull:#16a34a;--bear:#dc2626;--pos:#15803d;--neg:#b91c1c;--amber:#b45309;
  --bull-bg:#eaf7ef;--bear-bg:#fdeceb;--amber-bg:#fef3e2;--fill:rgba(14,116,137,.13);
  --shadow:0 1px 2px rgba(15,23,42,.05),0 8px 24px -16px rgba(15,23,42,.25);}}
@media(prefers-color-scheme:dark){{:root{{--bg:#080d16;--paper:#111a29;--ink:#e6ebf2;--mut:#9aa6b6;--faint:#6b7889;
  --line:#22314a;--accent:#22d3ee;--bull:#34d399;--bear:#f87171;--pos:#4ade80;--neg:#f87171;--amber:#fbbf24;
  --bull-bg:#0f2a20;--bear-bg:#2a1414;--amber-bg:#2a2110;--fill:rgba(34,211,238,.13);
  --shadow:0 1px 2px rgba(0,0,0,.4),0 12px 30px -18px rgba(0,0,0,.7);}}}}
:root[data-theme="light"]{{--bg:#eef0f3;--paper:#fff;--ink:#0f172a;--mut:#5b6472;--faint:#8a94a3;--line:#e3e7ec;--accent:#0e7490;--bull:#16a34a;--bear:#dc2626;--pos:#15803d;--neg:#b91c1c;--amber:#b45309;--bull-bg:#eaf7ef;--bear-bg:#fdeceb;--amber-bg:#fef3e2;--fill:rgba(14,116,137,.13);--shadow:0 1px 2px rgba(15,23,42,.05),0 8px 24px -16px rgba(15,23,42,.25);}}
:root[data-theme="dark"]{{--bg:#080d16;--paper:#111a29;--ink:#e6ebf2;--mut:#9aa6b6;--faint:#6b7889;--line:#22314a;--accent:#22d3ee;--bull:#34d399;--bear:#f87171;--pos:#4ade80;--neg:#f87171;--amber:#fbbf24;--bull-bg:#0f2a20;--bear-bg:#2a1414;--amber-bg:#2a2110;--fill:rgba(34,211,238,.13);--shadow:0 1px 2px rgba(0,0,0,.4),0 12px 30px -18px rgba(0,0,0,.7);}}
*{{box-sizing:border-box}}
body{{margin:0;background:var(--bg);color:var(--ink);font-family:system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;line-height:1.55;-webkit-font-smoothing:antialiased}}
.mono{{font-family:ui-monospace,"SF Mono","Cascadia Mono",Menlo,monospace;font-variant-numeric:tabular-nums}}
.wrap{{max-width:940px;margin:0 auto;padding:40px 22px 64px}}
.eyebrow{{font-size:12px;font-weight:700;letter-spacing:.14em;text-transform:uppercase;color:var(--accent)}}
h1{{font-family:"Iowan Old Style","Palatino Linotype",Palatino,Georgia,serif;font-size:clamp(28px,4.6vw,44px);line-height:1.09;margin:12px 0 10px;text-wrap:balance;font-weight:600;letter-spacing:-.01em}}
.lede{{font-size:18px;color:var(--mut);max-width:68ch;margin:0}}
h2{{font-family:"Iowan Old Style",Palatino,Georgia,serif;font-size:23px;font-weight:600;margin:42px 0 4px;letter-spacing:-.01em}}
.sub{{color:var(--mut);font-size:14px;margin:0 0 18px}}
.tiles{{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin:24px 0}}
.tile{{background:var(--paper);border:1px solid var(--line);border-radius:13px;padding:15px 16px;box-shadow:var(--shadow)}}
.tile .k{{font-size:10.5px;font-weight:700;letter-spacing:.05em;text-transform:uppercase;color:var(--faint)}}
.tile .v{{font-size:25px;font-weight:750;margin-top:5px;letter-spacing:-.01em}}
.tile .s{{font-size:12px;color:var(--mut);margin-top:2px}}
.rules{{background:var(--paper);border:1px solid var(--line);border-radius:14px;box-shadow:var(--shadow);padding:6px 20px;margin:22px 0}}
.rule-i{{display:grid;grid-template-columns:96px 1fr;gap:14px;padding:12px 0;border-top:1px solid var(--line);font-size:14px}}
.rule-i:first-child{{border-top:0}} .rule-i .rk{{font-weight:700;color:var(--accent);letter-spacing:.02em;text-transform:uppercase;font-size:11.5px;padding-top:2px}}
.card{{background:var(--paper);border:1px solid var(--line);border-radius:14px;box-shadow:var(--shadow);padding:18px 20px}}
svg .grid{{stroke:var(--line);stroke-width:1}} svg .axl{{fill:var(--faint);font:600 10px ui-monospace,monospace}}
svg .ln{{fill:none;stroke:var(--accent);stroke-width:2}} svg .ar{{fill:var(--fill)}}
.tblwrap{{overflow-x:auto;border:1px solid var(--line);border-radius:14px;box-shadow:var(--shadow);background:var(--paper)}}
table{{border-collapse:collapse;width:100%;font-size:13px;min-width:760px}}
th,td{{padding:9px 11px;text-align:left;border-top:1px solid var(--line);white-space:nowrap}}
thead th{{border-top:0;font-size:10.5px;letter-spacing:.03em;text-transform:uppercase;color:var(--faint);font-weight:700;background:var(--bg);position:sticky;top:0}}
td.r,th.r{{text-align:right}} .tname{{font-weight:700}} .dim2{{color:var(--mut);font-size:12px}}
td.pos{{color:var(--pos);font-weight:700}} td.neg{{color:var(--neg);font-weight:700}}
.oc{{font-size:10.5px;font-weight:700;padding:2px 8px;border-radius:999px}}
.oc.ok{{background:var(--bull-bg);color:var(--pos)}} .oc.bad{{background:var(--bear-bg);color:var(--neg)}} .oc.open{{background:var(--amber-bg);color:var(--amber)}}
.prose{{max-width:68ch}} .prose p{{margin:12px 0}}
.note{{font-size:13px;color:var(--mut);background:var(--paper);border:1px solid var(--line);border-radius:12px;padding:16px 18px}}
.warn{{border-left:4px solid var(--amber)}}
.tabs{{display:flex;gap:8px;margin:0 0 12px}} .tb{{font-size:12.5px;font-weight:700;color:var(--mut);cursor:default;
  padding:6px 12px;border:1px solid var(--line);border-radius:999px;background:var(--paper)}}
.foot{{margin-top:40px;font-size:12.5px;color:var(--faint);border-top:1px solid var(--line);padding-top:18px}}
@media(max-width:680px){{.tiles{{grid-template-columns:repeat(2,1fr)}}}}
</style>
<div class="wrap">
  <div class="eyebrow">Trade backtest · NIFTY 500 · 2016–2026 · long-only</div>
  <h1>Breakout system: 1,333 trades, +668R, profit factor 1.9 — but you'd lose 69% of the time</h1>
  <p class="lede">Every volume-confirmed bullish breakout, traded by one fixed rulebook across ten years and 500 stocks: enter the day after the break, stop just under the level, aim for 1:2, and once 2R is reached let it run on the 50-day moving average. It works — because a few big trend-riders pay for a long tail of small, quickly-cut losers.</p>

  <div class="tiles">
    <div class="tile"><div class="k">Trades</div><div class="v mono">{s['trades']:,}</div><div class="s">10 years, one per stock at a time</div></div>
    <div class="tile"><div class="k">Win rate</div><div class="v mono">{s['win_rate']}%</div><div class="s">low by design — tight stops</div></div>
    <div class="tile"><div class="k">Avg per trade</div><div class="v" style="color:var(--pos)">+{s['avg_R']}R</div><div class="s">+{s['avg_ret']}% · median {s['median_ret']}%</div></div>
    <div class="tile"><div class="k">Profit factor</div><div class="v mono">{s['profit_factor']}</div><div class="s">gross win ÷ gross loss</div></div>
  </div>

  <h2>The rulebook tested</h2>
  <div class="rules">
    <div class="rule-i"><div class="rk">Entry</div><div>Next day's <b>open</b> after a bullish breakout fires on ≥1.3× average volume (Resistance / Range / down-trendline breakouts).</div></div>
    <div class="rule-i"><div class="rk">Stop</div><div>Just below the <b>broken level</b> (the resistance / trendline price broke). Risk <b>R</b> = entry − stop.</div></div>
    <div class="rule-i"><div class="rk">Target</div><div>Minimum <b>1:2</b> risk-reward → target = entry + 2R.</div></div>
    <div class="rule-i"><div class="rk">Trail</div><div>Once 2R is hit, <b>stop taking profit</b> — hold and ride, exiting only on the first daily <b>close below the 50-DMA</b>.</div></div>
  </div>

  <h2>Cumulative edge (R captured)</h2>
  <p class="sub">Running sum of R across all {s['trades']:,} trades, ordered by exit date. Rising = the edge compounding; the dips are the losing streaks you must sit through. Total <b>+{totR}R</b>, worst peak-to-trough <b>{maxdd}R</b>.</p>
  <div class="card"><svg viewBox="0 0 {W} {H}" width="100%" preserveAspectRatio="none" style="display:block;height:240px">
    {yr_ticks}
    <polygon class="ar" points="{area}"/><polyline class="ln" points="{line}"/>
    <circle cx="{sx[-1]:.1f}" cy="{sy[-1]:.1f}" r="3.5" fill="var(--accent)"/>
  </svg></div>

  <h2>Why it works: cut fast, ride long</h2>
  <div class="prose"><p>The whole system lives on <b>asymmetry</b>. Losers are cut quickly at the stop — {len(l):,} of them ({len(l)/len(df)*100:.0f}%), averaging <b>−6.3%</b> and <b>−1.0R</b>, out in about <b>20 days</b>. Winners are made to run — only {len(w):,} ({len(w)/len(df)*100:.0f}%), but they average <b>+27%</b> and <b>+3.9R</b> over <b>78 days</b> because the 50-DMA trail keeps you in the trend long after the 2R target. {s['pct_hit_2R']}% of trades reached 2R and switched to trailing; {s['pct_stopped']}% were stopped first. Take away the trail — cash out at 2R — and you cap exactly the trades that make the strategy profitable.</p></div>

  <h2>By pattern</h2>
  <div class="tblwrap"><table>
    <thead><tr><th>Breakout type</th><th class="r">Trades</th><th class="r">Win</th><th class="r">Avg return</th><th class="r">Avg R</th><th class="r">Avg hold</th></tr></thead>
    <tbody>{bprows}</tbody></table></div>

  <h2>Trade log</h2>
  <p class="sub">The exact entry/stop/target/exit for a sample of trades. Full 1,333-row log is in the backtest output (trades.csv).</p>
  <div class="tabs"><span class="tb">Biggest winners</span></div>
  <div class="tblwrap"><table><thead><tr><th>Stock</th><th>Pattern</th><th>Entry date</th><th class="r">Entry</th><th class="r">Stop</th><th class="r">Target</th><th>Exit date</th><th class="r">Exit</th><th class="r">Days</th><th class="r">Return</th><th class="r">R</th><th>Outcome</th></tr></thead><tbody>{winners}</tbody></table></div>
  <div class="tabs" style="margin-top:18px"><span class="tb">Most recent signals</span></div>
  <div class="tblwrap"><table><thead><tr><th>Stock</th><th>Pattern</th><th>Entry date</th><th class="r">Entry</th><th class="r">Stop</th><th class="r">Target</th><th>Exit date</th><th class="r">Exit</th><th class="r">Days</th><th class="r">Return</th><th class="r">R</th><th>Outcome</th></tr></thead><tbody>{recent}</tbody></table></div>

  <h2>Read this before believing it</h2>
  <div class="note warn">
    <b>The 69% loss rate is the catch.</b> Positive expectancy does <em>not</em> feel good in real time — you sit through long strings of small losses ({maxdd}R worst drawdown) waiting for the occasional {s['best']:.0f}%-type winner. Most people abandon a 31%-win system before the winners arrive. It only works if every trade is the same small risk and you never skip one.<br><br>
    <b>Method &amp; limits.</b> 500 <em>current</em> NIFTY 500 names (survivorship bias — delisted losers aren't here, so results flatter). Entry at the next open, exits at the stop/close levels with <b>no brokerage, slippage or gap-through-stop cost</b> — a gap opening below the stop would fill worse than modelled (a few trades already show −30%+ single-trade losses from gap-up entries with wide stops). Trades overlap across stocks, so the +668R is fixed-risk-per-trade, <b>not</b> a compounded account return. One decade is one regime — a broadly rising market that favours long breakouts. <b>Descriptive research, not advice, and not a forecast.</b>
  </div>
  <div class="foot mono">NIFTY 500 breakout trade backtest · {s['trades']:,} trades · 2016–2026 · entry/stop/target/50-DMA-trail · generated {built}. Educational only; prepared by a person who is not a SEBI-registered adviser.</div>
</div>"""
Path("report2.html").write_text(html, encoding="utf-8")
print("wrote report2.html", len(html), "bytes")
