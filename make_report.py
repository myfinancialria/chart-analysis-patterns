import json, datetime
from pathlib import Path

s = json.load(open("output/backtest_summary.json"))
b20 = s["baseline_long"]["20"]["mean"]; b60 = s["baseline_long"]["60"]["mean"]
b20w = s["baseline_long"]["20"]["win"]
pats = s["by_pattern"]
for r in pats:
    d = 1 if r["bias"] == "Bullish" else -1
    r["edge20"] = round(r["mean20"] - d * b20, 2)
    r["edge60"] = round(r["mean60"] - d * b60, 2)
pats = sorted(pats, key=lambda r: r["mean20"], reverse=True)

SCALE = 3.0  # % axis half-range for the 20d bar chart
def bar_row(r):
    v = r["mean20"]
    w = min(abs(v) / SCALE * 50, 50)
    bull = r["bias"] == "Bullish"
    cls = "bull" if bull else "bear"
    # positive grows right from the 50% centre; negative grows left
    style = (f"left:50%;width:{w:.2f}%" if v >= 0 else f"left:{50-w:.2f}%;width:{w:.2f}%")
    # value label sits just PAST the bar's end, on the track background (readable)
    vstyle = (f"left:calc(50% + {w:.2f}% + 6px)" if v >= 0 else f"right:calc(50% + {w:.2f}% + 6px)")
    val = f"{v:+.2f}%"
    badge = "▲" if bull else "▼"
    return f"""<div class="brow">
      <div class="blabel"><span class="bdot {cls}">{badge}</span>{r['name']}</div>
      <div class="btrack"><span class="bzero"></span><span class="bbase" title="market avg +{b20}%"></span>
        <span class="bfill {cls}" style="{style}"></span>
        <span class="bval {cls}" style="{vstyle}">{val}</span></div>
      <div class="bwin mono">{r['win20']:.0f}%</div>
    </div>"""

rows = "\n".join(bar_row(r) for r in pats)

def trow(r):
    e = r["edge20"]; ecls = "pos" if e > 0 else ("neg" if e < 0 else "")
    d = "Long" if r["bias"] == "Bullish" else "Short"
    return f"""<tr>
      <td class="tname">{r['name']}</td>
      <td><span class="pill {'bull' if r['bias']=='Bullish' else 'bear'}">{r['bias']}</span></td>
      <td class="mono r">{int(r['signals'])}</td>
      <td class="mono r">{r['mean20']:+.2f}</td>
      <td class="mono r">{r['median20']:+.2f}</td>
      <td class="mono r">{r['win20']:.0f}%</td>
      <td class="mono r">{r['mean60']:+.2f}</td>
      <td class="mono r">{r['win60']:.0f}%</td>
      <td class="mono r {ecls}">{e:+.2f}</td>
    </tr>"""
table = "\n".join(trow(r) for r in pats)

top = pats[0]
bt = [x for x in s["by_state"]]
st = {x["state"]: x for x in bt}
built = datetime.date.today().strftime("%d %b %Y")

html = f"""<style>
:root{{
  --bg:#eef0f3; --paper:#ffffff; --ink:#0f172a; --mut:#5b6472; --faint:#8a94a3;
  --line:#e3e7ec; --accent:#0e7490; --bull:#16a34a; --bear:#dc2626; --base:#64748b;
  --bull-bg:#eaf7ef; --bear-bg:#fdeceb; --pos:#15803d; --neg:#b91c1c;
  --shadow:0 1px 2px rgba(15,23,42,.05),0 8px 24px -16px rgba(15,23,42,.25);
}}
@media (prefers-color-scheme:dark){{:root{{
  --bg:#080d16; --paper:#111a29; --ink:#e6ebf2; --mut:#9aa6b6; --faint:#6b7889;
  --line:#22314a; --accent:#22d3ee; --bull:#34d399; --bear:#f87171; --base:#8595a9;
  --bull-bg:#0f2a20; --bear-bg:#2a1414; --pos:#4ade80; --neg:#f87171;
  --shadow:0 1px 2px rgba(0,0,0,.4),0 12px 30px -18px rgba(0,0,0,.7);
}}}}
:root[data-theme="light"]{{ --bg:#eef0f3; --paper:#ffffff; --ink:#0f172a; --mut:#5b6472; --faint:#8a94a3; --line:#e3e7ec; --accent:#0e7490; --bull:#16a34a; --bear:#dc2626; --base:#64748b; --bull-bg:#eaf7ef; --bear-bg:#fdeceb; --pos:#15803d; --neg:#b91c1c; --shadow:0 1px 2px rgba(15,23,42,.05),0 8px 24px -16px rgba(15,23,42,.25);}}
:root[data-theme="dark"]{{ --bg:#080d16; --paper:#111a29; --ink:#e6ebf2; --mut:#9aa6b6; --faint:#6b7889; --line:#22314a; --accent:#22d3ee; --bull:#34d399; --bear:#f87171; --base:#8595a9; --bull-bg:#0f2a20; --bear-bg:#2a1414; --pos:#4ade80; --neg:#f87171; --shadow:0 1px 2px rgba(0,0,0,.4),0 12px 30px -18px rgba(0,0,0,.7);}}
*{{box-sizing:border-box}}
body{{margin:0;background:var(--bg);color:var(--ink);
  font-family:system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;line-height:1.55;
  -webkit-font-smoothing:antialiased}}
.mono{{font-family:ui-monospace,"SF Mono","Cascadia Mono",Menlo,monospace;font-variant-numeric:tabular-nums}}
.serif{{font-family:"Iowan Old Style","Palatino Linotype",Palatino,Georgia,serif}}
.wrap{{max-width:900px;margin:0 auto;padding:40px 22px 64px}}
.eyebrow{{font-size:12px;font-weight:700;letter-spacing:.14em;text-transform:uppercase;color:var(--accent)}}
h1{{font-family:"Iowan Old Style","Palatino Linotype",Palatino,Georgia,serif;
  font-size:clamp(30px,5vw,46px);line-height:1.08;margin:12px 0 10px;text-wrap:balance;font-weight:600;letter-spacing:-.01em}}
.lede{{font-size:18px;color:var(--mut);max-width:66ch;margin:0}}
.rule{{height:1px;background:var(--line);margin:30px 0}}
.tiles{{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin:26px 0}}
.tile{{background:var(--paper);border:1px solid var(--line);border-radius:13px;padding:15px 16px;box-shadow:var(--shadow)}}
.tile .k{{font-size:11px;font-weight:700;letter-spacing:.05em;text-transform:uppercase;color:var(--faint)}}
.tile .v{{font-size:26px;font-weight:750;margin-top:5px;letter-spacing:-.01em}}
.tile .s{{font-size:12px;color:var(--mut);margin-top:2px}}
.finding{{background:var(--paper);border:1px solid var(--line);border-left:4px solid var(--accent);
  border-radius:12px;padding:18px 20px;box-shadow:var(--shadow);margin:24px 0}}
.finding b{{color:var(--ink)}}
h2{{font-family:"Iowan Old Style",Palatino,Georgia,serif;font-size:23px;font-weight:600;margin:40px 0 4px;letter-spacing:-.01em}}
.sub{{color:var(--mut);font-size:14px;margin:0 0 18px}}
.card{{background:var(--paper);border:1px solid var(--line);border-radius:14px;box-shadow:var(--shadow);padding:20px 20px 12px}}
.chart-head{{display:flex;justify-content:space-between;font-size:11px;font-weight:700;letter-spacing:.05em;
  text-transform:uppercase;color:var(--faint);padding:0 0 10px}}
.brow{{display:grid;grid-template-columns:210px 1fr 46px;align-items:center;gap:12px;padding:7px 0;border-top:1px solid var(--line)}}
.brow:first-of-type{{border-top:0}}
.blabel{{font-size:13.5px;font-weight:600;display:flex;align-items:center;gap:8px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}}
.bdot{{font-size:10px;width:16px;text-align:center;flex:0 0 auto}}
.bdot.bull{{color:var(--bull)}} .bdot.bear{{color:var(--bear)}}
.btrack{{position:relative;height:24px;background:linear-gradient(90deg,var(--bear-bg),transparent 48%,transparent 52%,var(--bull-bg));border-radius:5px}}
.bzero{{position:absolute;left:50%;top:0;bottom:0;width:1.5px;background:var(--faint);opacity:.6}}
.bbase{{position:absolute;left:{50+b20/SCALE*50:.2f}%;top:-3px;bottom:-3px;width:0;border-left:1.5px dashed var(--base)}}
.bfill{{position:absolute;top:4px;bottom:4px;border-radius:3px}}
.bfill.bull{{background:var(--bull)}} .bfill.bear{{background:var(--bear)}}
.bval{{position:absolute;top:2px;font-size:12px;font-weight:700;font-family:ui-monospace,Menlo,monospace;font-variant-numeric:tabular-nums}}
.bval.vr{{left:calc(50% + 4px)}} .bval.vl{{right:calc(50% + 4px)}}
.bval.bull{{color:var(--pos)}} .bval.bear{{color:var(--neg)}}
.bwin{{font-size:12.5px;color:var(--mut);text-align:right}}
.baxis{{display:flex;justify-content:space-between;font-size:11px;color:var(--faint);padding:8px 0 2px;margin-left:222px}}
.legend{{font-size:12px;color:var(--mut);margin:12px 2px 0;display:flex;gap:16px;flex-wrap:wrap}}
.legend b{{color:var(--base)}}
.tblwrap{{overflow-x:auto;border:1px solid var(--line);border-radius:14px;box-shadow:var(--shadow);background:var(--paper)}}
table{{border-collapse:collapse;width:100%;font-size:13.5px;min-width:640px}}
th,td{{padding:11px 12px;text-align:left;border-top:1px solid var(--line)}}
thead th{{border-top:0;font-size:11px;letter-spacing:.04em;text-transform:uppercase;color:var(--faint);font-weight:700;background:var(--bg)}}
td.r,th.r{{text-align:right}} .tname{{font-weight:600}}
.pill{{font-size:11px;font-weight:700;padding:2px 9px;border-radius:999px}}
.pill.bull{{background:var(--bull-bg);color:var(--pos)}} .pill.bear{{background:var(--bear-bg);color:var(--neg)}}
td.pos{{color:var(--pos);font-weight:700}} td.neg{{color:var(--neg);font-weight:700}}
.prose{{max-width:68ch}} .prose p{{margin:12px 0}} .prose li{{margin:6px 0}}
.note{{font-size:13px;color:var(--mut);background:var(--paper);border:1px solid var(--line);border-radius:12px;padding:16px 18px}}
.foot{{margin-top:40px;font-size:12.5px;color:var(--faint);border-top:1px solid var(--line);padding-top:18px}}
@media(max-width:680px){{.tiles{{grid-template-columns:repeat(2,1fr)}}
  .brow{{grid-template-columns:130px 1fr 40px;gap:8px}} .blabel{{font-size:12px}} .baxis{{margin-left:138px}}}}
</style>

<div class="wrap">
  <div class="eyebrow">10-year backtest · NIFTY 500 · daily EOD</div>
  <h1>Which chart patterns actually pay — and only two clearly beat the market</h1>
  <p class="lede">Every volume-confirmed pattern the live scanner fires was replayed across ten years of history on all 500 stocks — {s['total_signals']:,} signals in all — then scored on the forward return you'd have earned trading it as intended. Volume-confirmed <b>breakouts</b> carry the edge; forming patterns and short setups do not.</p>

  <div class="tiles">
    <div class="tile"><div class="k">Signals tested</div><div class="v mono">{s['total_signals']:,}</div><div class="s">10y · weekly walk-forward</div></div>
    <div class="tile"><div class="k">Best pattern · 20d</div><div class="v" style="color:var(--pos)">+{top['mean20']:.2f}%</div><div class="s">{top['name']}</div></div>
    <div class="tile"><div class="k">Its win rate</div><div class="v mono">{top['win20']:.0f}%</div><div class="s">vs {b20w:.0f}% base rate</div></div>
    <div class="tile"><div class="k">Market baseline</div><div class="v mono">+{b20}%</div><div class="s">random long, 20 days</div></div>
  </div>

  <div class="finding"><b>Bottom line.</b> A <b>Resistance breakout</b> on a real volume surge returned <b>+{top['mean20']:.2f}% over the next 20 trading days</b> ({top['win20']:.0f}% of the time positive) — about <b>+{top['edge20']:.2f}%</b> ahead of simply buying a random stock (+{b20}%), widening to <b>+{top['mean60']:.1f}% at 60 days</b>. <b>Range breakouts</b> are the only other pattern that clearly beats the market. Everything else lands at or below the baseline, and shorting the bearish patterns <b>lost money</b> across a decade the market spent rising.</p></div>

  <h2>Return by pattern, traded as intended</h2>
  <p class="sub">Average forward return over 20 trading days · long on bullish signals (green), short on bearish (red) · dashed line = market baseline (+{b20}%).</p>
  <div class="card">
    <div class="chart-head"><span>Pattern</span><span style="margin-right:56px">20-day return &amp; win-rate →</span></div>
    {rows}
    <div class="baxis"><span>−{SCALE:.0f}%</span><span>0</span><span>+{SCALE:.0f}%</span></div>
  </div>
  <div class="legend"><span><b>— — —</b> market baseline, a random 20-day long (+{b20}%)</span><span>Right column = win rate</span></div>

  <h2>Full results</h2>
  <p class="sub">Direction-adjusted returns (%). Edge = return minus the same-direction market baseline; positive means the pattern added value beyond the market's drift.</p>
  <div class="tblwrap"><table>
    <thead><tr><th>Pattern</th><th>Bias</th><th class="r">Signals</th><th class="r">Mean 20d</th><th class="r">Median 20d</th><th class="r">Win 20d</th><th class="r">Mean 60d</th><th class="r">Win 60d</th><th class="r">Edge 20d</th></tr></thead>
    <tbody>{table}</tbody>
  </table></div>

  <h2>What it means</h2>
  <div class="prose">
    <p><b>The edge lives in completed breakouts, not forming patterns.</b> Grouped by what the signal <em>is</em>: fresh <b>breakouts</b> (n={int(st['Breakout']['signals'])}) returned <b>+{st['Breakout']['mean20']:.2f}%/20d</b> ({st['Breakout']['win20']:.0f}% win, +{st['Breakout']['mean60']:.1f}%/60d); still-<b>forming</b> triangles &amp; wedges (n={int(st['Forming']['signals'])}) were essentially flat at <b>{st['Forming']['mean20']:+.2f}%</b> — they haven't resolved yet, so there's no edge until price actually breaks. The takeaway: a volume-backed break is the tradeable moment; the tidy geometry beforehand is a watch-list, not a trigger.</p>
    <p><b>Volume is the filter that matters.</b> Every signal here already cleared the volume test — breakouts on ≥1.3× average volume, squeezes on drying-up volume. Resistance and Range breakouts, which pair a clean level with a genuine volume surge, are exactly the setups that beat the market.</p>
    <p><b>Shorting didn't work — but the bearish read wasn't useless.</b> Traded as shorts, all five bearish patterns lost money, because 2016–2026 was a broad up-market and shorts fought the tide. Yet several still flagged <em>relative weakness</em>: a stock breaking down or forming a rising wedge tended to underperform a random stock over the next 20 days. Useful as an "avoid / reduce" signal, not as a naked short.</p>
  </div>

  <h2>How this was tested</h2>
  <div class="note">
    <b>Method.</b> All 500 current NIFTY 500 names, ~10 years of daily candles (Yahoo Finance). For each stock the clock is stepped forward one week at a time; at each step the <em>same</em> detectors and the <em>same</em> high-conviction volume filter used on the live dashboard run on trailing data only — no look-ahead. Each qualifying signal (deduplicated to one per pattern per stock per ~month) is entered at that day's close and scored on its forward return, long for bullish and short for bearish, at 10 / 20 / 60 trading days. The baseline is the average forward return of a random long entry over the same windows and dates.<br><br>
    <b>Caveats — read these.</b> The universe is <em>today's</em> NIFTV 500, so it carries survivorship bias (past winners are over-represented, which flatters bullish results). Entries are end-of-day at the close with <em>no</em> brokerage, slippage or impact costs. Signals overlap and market regimes shift, so a single decade is one sample, not a law. This is a descriptive study of historical price behaviour — <b>not</b> investment advice, and not a promise about the future.
  </div>

  <div class="foot mono">NIFTY 500 pattern backtest · {s['total_signals']:,} signals · 10-year walk-forward · generated {built}. Educational / informational only; prepared by a person who is not a SEBI-registered adviser.</div>
</div>"""

Path("report.html").write_text(html, encoding="utf-8")
print("wrote report.html", len(html), "bytes | top:", top["name"], top["mean20"])
