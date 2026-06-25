"""
Single-line signals to complement the two-rail pattern engine (detect.py):

  * Horizontal SUPPORT / RESISTANCE boxes — a flat level tested >= MIN_TOUCH times.
  * RANGE / box breakouts — price escaping a rectangle (flat support + resistance).
  * Single TRENDLINE breakouts — a sloping line (down-line through highs / up-line
    through lows) that price has just broken.

Each detector returns plain dicts with a schema that merges into the scan output
(category + state + the line geometry needed to draw the chart). Pure module —
run `python breakouts.py` for the synthetic self-test.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import linregress

from detect import find_pivots, _atr

MIN_TOUCH = 3
BREAK_BARS = 6        # a breakout/break must have happened within this many bars
NRMSE_MAX = 0.030     # trendline fit quality (RMSE / mean price)
NEAR_PCT = 3.0        # "testing" a level = price within this % of it


def _conf(touches: int, fit_q: float, edge: float) -> float:
    """touches (more = better), fit quality 0..1, edge 0..1 (break margin/volume)."""
    touch_q = min(1.0, touches / 6.0)
    return round(0.40 * fit_q + 0.30 * touch_q + 0.30 * edge, 3)


def _blank() -> dict:
    """Columns shared with the pattern rows so everything merges into one table."""
    return {"res_level": np.nan, "sup_level": np.nan,
            "tl_slope": np.nan, "tl_intercept": np.nan,
            "vol_contraction": np.nan, "vol_surge": np.nan,
            "upper_now": np.nan, "lower_now": np.nan,
            "pct_to_upper": np.nan, "pct_to_lower": np.nan}


# ---------------- horizontal levels + range/box breakouts ----------------
def scan_levels(high, low, close, volume, lookback: int = 130) -> list[dict]:
    high, low, close = map(lambda a: np.asarray(a, float), (high, low, close))
    volume = np.asarray(volume, float) if volume is not None else None
    n = min(lookback, len(close))
    if n < 35:
        return []
    h, l, c = high[-n:], low[-n:], close[-n:]
    v = volume[-n:] if volume is not None else None
    atr = _atr(h, l, c)
    mean = float(c.mean())
    tol = max(0.6 * atr, 0.012 * mean)
    margin = 0.30 * atr
    last = float(c[-1])

    base_n = n - BREAK_BARS
    base_high = float(h[:base_n].max())
    base_low = float(l[:base_n].min())

    hi_idx = np.array(find_pivots(h, 3, hi=True))
    lo_idx = np.array(find_pivots(l, 3, hi=False))
    res_touch = int(np.sum(np.abs(h[hi_idx] - base_high) <= tol)) if len(hi_idx) else 0
    sup_touch = int(np.sum(np.abs(l[lo_idx] - base_low) <= tol)) if len(lo_idx) else 0

    vol_surge = np.nan
    if v is not None and v[:base_n].mean() > 0:
        vol_surge = round(float(v[base_n:].mean() / v[:base_n].mean()), 2)
    vol_edge = 0.0 if np.isnan(vol_surge) else min(1.0, max(0.0, (vol_surge - 1) / 1.5))

    out = []
    box = res_touch >= MIN_TOUCH and sup_touch >= MIN_TOUCH  # a real range exists

    # ---- upside breakout of resistance / range ----
    if res_touch >= MIN_TOUCH and last > base_high + margin:
        edge = min(1.0, (last - base_high) / (3 * atr)) * 0.6 + vol_edge * 0.4
        row = {**_blank(),
               "category": "Range" if box else "Level",
               "name": "Range breakout (up)" if box else "Resistance breakout",
               "bias": "Bullish", "state": "Breakout",
               "confidence": _conf(res_touch, 1.0, edge),
               "window": n, "close": round(last, 2),
               "res_level": round(base_high, 2),
               "sup_level": round(base_low, 2) if box else np.nan,
               "upper_now": round(base_high, 2),
               "lower_now": round(base_low, 2) if box else np.nan,
               "pct_to_upper": round((base_high - last) / last * 100, 2),
               "pct_to_lower": round((last - base_low) / last * 100, 2) if box else np.nan,
               "vol_surge": vol_surge}
        out.append(row)
    # ---- downside breakdown of support / range ----
    elif sup_touch >= MIN_TOUCH and last < base_low - margin:
        edge = min(1.0, (base_low - last) / (3 * atr)) * 0.6 + vol_edge * 0.4
        out.append({**_blank(),
                    "category": "Range" if box else "Level",
                    "name": "Range breakdown (down)" if box else "Support breakdown",
                    "bias": "Bearish", "state": "Breakdown",
                    "confidence": _conf(sup_touch, 1.0, edge),
                    "window": n, "close": round(last, 2),
                    "res_level": round(base_high, 2) if box else np.nan,
                    "sup_level": round(base_low, 2),
                    "upper_now": round(base_high, 2) if box else np.nan,
                    "lower_now": round(base_low, 2),
                    "pct_to_upper": round((base_high - last) / last * 100, 2) if box else np.nan,
                    "pct_to_lower": round((last - base_low) / last * 100, 2),
                    "vol_surge": vol_surge})
    else:
        # ---- still inside: report levels price is currently testing ----
        dist_res = (base_high - last) / last * 100
        if res_touch >= MIN_TOUCH and 0 <= dist_res <= NEAR_PCT:
            out.append({**_blank(), "category": "Level", "name": "Resistance box",
                        "bias": "Neutral", "state": "Testing",
                        "confidence": _conf(res_touch, 1.0, 1 - dist_res / NEAR_PCT),
                        "window": n, "close": round(last, 2),
                        "res_level": round(base_high, 2), "upper_now": round(base_high, 2),
                        "pct_to_upper": round(dist_res, 2)})
        dist_sup = (last - base_low) / last * 100
        if sup_touch >= MIN_TOUCH and 0 <= dist_sup <= NEAR_PCT:
            out.append({**_blank(), "category": "Level", "name": "Support box",
                        "bias": "Neutral", "state": "Testing",
                        "confidence": _conf(sup_touch, 1.0, 1 - dist_sup / NEAR_PCT),
                        "window": n, "close": round(last, 2),
                        "sup_level": round(base_low, 2), "lower_now": round(base_low, 2),
                        "pct_to_lower": round(dist_sup, 2)})
    return out


# ---------------- single trendline breakouts ----------------
def _fit_line(idx: np.ndarray, val: np.ndarray, mean: float, tol: float):
    lr = linregress(idx.astype(float), val)
    resid = val - (lr.slope * idx + lr.intercept)
    nrmse = float(np.sqrt(np.mean(resid ** 2))) / mean
    touches = int(np.sum(np.abs(resid) <= tol))
    return lr.slope, lr.intercept, nrmse, touches


def _last_cross(series_above: np.ndarray) -> int:
    """Index of the most recent False->True transition in a boolean series (-1 if none)."""
    cross = np.where(series_above[1:] & ~series_above[:-1])[0]
    return int(cross[-1] + 1) if len(cross) else -1


def scan_trendline(high, low, close, volume, lookback: int = 90) -> list[dict]:
    high, low, close = map(lambda a: np.asarray(a, float), (high, low, close))
    volume = np.asarray(volume, float) if volume is not None else None
    n = min(lookback, len(close))
    if n < 30:
        return []
    h, l, c = high[-n:], low[-n:], close[-n:]
    v = volume[-n:] if volume is not None else None
    x = np.arange(n)
    atr = _atr(h, l, c)
    mean = float(c.mean())
    tol = max(0.6 * atr, 0.012 * mean)
    margin = 0.30 * atr
    last = float(c[-1])

    def vol_edge() -> tuple[float, float]:
        if v is None or v[:-BREAK_BARS].mean() <= 0:
            return np.nan, 0.0
        surge = round(float(v[-BREAK_BARS:].mean() / v[:-BREAK_BARS].mean()), 2)
        return surge, min(1.0, max(0.0, (surge - 1) / 1.5))

    out = []
    # ---- DOWN-trend line through swing highs; break ABOVE = bullish ----
    hi_idx = np.array(find_pivots(h, 3, hi=True))
    if len(hi_idx) >= MIN_TOUCH:
        s, b, nr, t = _fit_line(hi_idx, h[hi_idx], mean, tol)
        line = s * x + b
        if s < 0 and nr <= NRMSE_MAX and t >= MIN_TOUCH:
            above = c > line
            xc = _last_cross(above)
            if above[-1] and xc >= n - BREAK_BARS and last > line[-1] + margin:
                surge, ve = vol_edge()
                edge = min(1.0, (last - line[-1]) / (3 * atr)) * 0.6 + ve * 0.4
                fit_q = 1 - min(1.0, nr / NRMSE_MAX)
                out.append({**_blank(), "category": "Trendline",
                            "name": "Trendline breakout (down-line)",
                            "bias": "Bullish", "state": "Breakout",
                            "confidence": _conf(t, fit_q, edge),
                            "window": n, "close": round(last, 2),
                            "tl_slope": float(s), "tl_intercept": float(b),
                            "upper_now": round(float(line[-1]), 2),
                            "pct_to_upper": round((line[-1] - last) / last * 100, 2),
                            "vol_surge": surge})
    # ---- UP-trend line through swing lows; break BELOW = bearish ----
    lo_idx = np.array(find_pivots(l, 3, hi=False))
    if len(lo_idx) >= MIN_TOUCH:
        s, b, nr, t = _fit_line(lo_idx, l[lo_idx], mean, tol)
        line = s * x + b
        if s > 0 and nr <= NRMSE_MAX and t >= MIN_TOUCH:
            below = c < line
            xc = _last_cross(below)
            if below[-1] and xc >= n - BREAK_BARS and last < line[-1] - margin:
                surge, ve = vol_edge()
                edge = min(1.0, (line[-1] - last) / (3 * atr)) * 0.6 + ve * 0.4
                fit_q = 1 - min(1.0, nr / NRMSE_MAX)
                out.append({**_blank(), "category": "Trendline",
                            "name": "Trendline breakdown (up-line)",
                            "bias": "Bearish", "state": "Breakdown",
                            "confidence": _conf(t, fit_q, edge),
                            "window": n, "close": round(last, 2),
                            "tl_slope": float(s), "tl_intercept": float(b),
                            "lower_now": round(float(line[-1]), 2),
                            "pct_to_lower": round((last - line[-1]) / last * 100, 2),
                            "vol_surge": surge})
    return out


def scan_extra(high, low, close, volume, min_conf: float = 0.45) -> list[dict]:
    """All single-line signals for one symbol, de-duplicated by name, conf-filtered."""
    sigs = scan_levels(high, low, close, volume) + scan_trendline(high, low, close, volume)
    best: dict[str, dict] = {}
    for s in sigs:
        if s["confidence"] < min_conf:
            continue
        if s["name"] not in best or s["confidence"] > best[s["name"]]["confidence"]:
            best[s["name"]] = s
    return list(best.values())


# ---------------- self-test ----------------
def _synth():
    rng = np.random.default_rng(4)
    P = 6  # swing period

    def saw(n, upper_fn, lower_fn, brk):
        """Oscillate between two envelopes (peaks touch upper, troughs touch lower),
        then override the last len(brk) closes with the breakout path."""
        x = np.arange(n)
        upper, lower = upper_fn(x), lower_fn(x)
        ph = x % P
        tri = np.where(ph <= P / 2, ph / (P / 2), 2 - ph / (P / 2))
        c = lower + tri * (upper - lower) + rng.normal(0, 0.15, n)
        c[-len(brk):] = brk
        h = c + 0.4; l = c - 0.4
        vol = np.r_[np.full(n - len(brk), 5e5), np.full(len(brk), 1.3e6)]
        return h, l, c, vol

    n = 84
    cases = {
        # flat resistance at 100, rising lows (no flat support) -> resistance-only breakout
        "Resistance breakout": saw(
            n, lambda x: np.full_like(x, 100, float), lambda x: 82 + 0.18 * x,
            [101, 103, 105, 107.5]),
        # flat support at 100, falling highs (no flat resistance) -> support-only breakdown
        "Support breakdown": saw(
            n, lambda x: 118 - 0.18 * x, lambda x: np.full_like(x, 100, float),
            [99, 97, 95, 92.5]),
        # descending peaks on a down-line, troughs well below -> down-line breakout
        "Trendline breakout (down-line)": saw(
            n, lambda x: 130 - 0.45 * x, lambda x: 95 - 0.45 * x,
            [130 - 0.45 * (n - 3) + 3, 130 - 0.45 * (n - 2) + 5, 130 - 0.45 * (n - 1) + 8]),
    }

    print(f"{'expected':<32} {'detected?':<9} conf")
    print("-" * 52)
    ok = 0
    for expect, (h, l, c, v) in cases.items():
        sigs = scan_extra(h, l, c, v, min_conf=0.0)
        names = [s["name"] for s in sigs]
        hit = expect in names
        ok += hit
        conf = next((s["confidence"] for s in sigs if s["name"] == expect), 0)
        print(f"{expect:<32} {'Y' if hit else 'n':<9} {conf}")
    print("-" * 52)
    print(f"{ok}/{len(cases)} detected")


if __name__ == "__main__":
    _synth()
