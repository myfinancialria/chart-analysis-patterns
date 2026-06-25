# NIFTY 500 chart-pattern scanner (daily EOD)

Detects **parallel channels, triangles, and wedges** across the NIFTY 500 on the
daily timeframe, ranks them by confidence, charts the strongest setups, and posts
an internal Slack digest.

## How it works

Every pattern in this family is just an **upper line through swing highs** + a
**lower line through swing lows**. The label depends on the two slopes and whether
the lines converge:

| Upper (highs) | Lower (lows) | Converging? | Pattern |
|---|---|---|---|
| flat | flat | no | Rectangle / Range |
| up | up | no | Ascending channel |
| down | down | no | Descending channel |
| flat | up | yes | Ascending triangle (bullish) |
| down | flat | yes | Descending triangle (bearish) |
| down | up | yes | Symmetrical triangle (neutral) |
| up | up | yes | Rising wedge (bearish) |
| down | down | yes | Falling wedge (bullish) |

Pipeline (`detect.py`): **swing pivots** (fractal, ±3 bars) → **least-squares line
fit** through highs and lows (quality measured by normalised RMSE, not R², so flat
lines aren't wrongly rejected) → **classify** by slope + convergence → **score**
(line fit 45% / touch count 30% / shape clarity 25%). Each stock is tried over
60/90/130-bar lookbacks and the best-scoring window wins.

## Run

```bash
# from this directory
../.venv/bin/python scan.py                 # fetch + detect + charts
../.venv/bin/python scan.py --cache         # reuse today's cached candles
../.venv/bin/python scan.py --limit 40      # smoke test on 40 symbols
../.venv/bin/python scan.py --min-conf 0.6  # stricter
../.venv/bin/python scan.py --slack         # also post Slack digest
./run.sh                                     # daily wrapper (fetch + Slack)
```

Self-test the detector with no network/Fyers needed:

```bash
../.venv/bin/python detect.py   # 8/8 synthetic patterns should classify
```

## Inputs / outputs

- **Universe:** `../data/nifty500_list.csv` (Symbol + Industry)
- **Auth:** `../access_token.txt` + `FYERS_CLIENT_ID` (same as the rest of fyers-bot)
- **Candles cache:** `cache/ohlcv_YYYYMMDD.csv` (one Fyers pull per day; `--cache` reuses it)
- **Results:** `output/patterns_YYYYMMDD.csv` + `output/patterns_latest.csv`
- **Charts:** `output/patterns_montage.png`
- **Slack:** `SLACK_BOT_TOKEN` + `SLACK_CHANNEL` (uploads montage) or `SLACK_WEBHOOK_URL` (text)

## Tuning

Knobs live at the top of `detect.py`: `FLAT_SLOPE`, `CONVERGE`, `NRMSE_MAX`,
`MIN_TOUCH`, `MIN_WINDOW`, `MIN_CONF`. Loosen `NRMSE_MAX` / lower `MIN_TOUCH` for
more (noisier) hits; raise `--min-conf` for fewer, cleaner ones.

## Daily website

`build_site.py` turns a completed scan into a self-contained dashboard under
`docs/` — one annotated candlestick chart per detected pattern plus a filterable,
sortable `index.html` (data embedded inline, so it works opened straight from disk).

```bash
../.venv/bin/python scan.py --site          # scan + build docs/
../.venv/bin/python build_site.py           # rebuild docs/ from the latest scan only
open docs/index.html                         # view locally
```

### Cloud refresh (GitHub Actions)

`.github/workflows/daily-scan.yml` runs every weekday at **16:30 IST**: it mints a
fresh Fyers token from repo secrets (via `auto_login.py`), runs the scan, rebuilds
`docs/`, commits it back, and uploads the dashboard as a private build artifact.

Required repo secrets (Settings → Secrets → Actions): `FYERS_CLIENT_ID`,
`FYERS_SECRET`, `FYERS_REDIRECT`, `FYERS_FY_ID`, `FYERS_PIN`, `FYERS_TOTP_KEY`.

**Viewing privately.** The repo is private and the site is marked `noindex`. On a
personal GitHub plan, Pages from a private repo would be *public*, so Pages is left
**disabled** to honour the internal-only requirement. View the dashboard by either
(a) downloading the per-run **artifact** from the Actions tab, or (b) `git pull` and
opening `docs/index.html`. For a private *live URL*, put it behind Cloudflare Access
or Netlify password protection (free) — the build artifact is host-agnostic.

## Compliance

Output is **descriptive** ("X is forming an ascending triangle, N% below
resistance") for the **internal Slack workspace only**. No buy/sell/target calls,
no public broadcast — consistent with not being a SEBI-registered RA/RIA.
