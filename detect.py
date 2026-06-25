"""
Geometric chart-pattern detection: parallel channels, triangles, wedges.

The whole family reduces to one idea: find the swing pivots, fit an UPPER line
through the swing highs and a LOWER line through the swing lows, then classify by
the two slopes + whether the lines converge.

    upper flat   + lower flat              -> Rectangle / Range
    upper ~down  + lower ~down  (parallel) -> Descending channel
    upper ~up    + lower ~up    (parallel) -> Ascending channel
    upper flat   + lower up     (converge) -> Ascending triangle   (bullish)
    upper down   + lower flat    (converge)-> Descending triangle  (bearish)
    upper down   + lower up     (converge) -> Symmetrical triangle (neutral)
    upper up     + lower up      (converge)-> Rising wedge         (bearish)
    upper down   + lower down    (converge)-> Falling wedge        (bullish)

Pure module — no IO, no network. Run `python detect.py` for the synthetic self-test.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
import numpy as np
from scipy.stats import linregress

# ---------------- tunables ----------------
FLAT_SLOPE = 0.05      # |%/bar| below this = a "flat" line
CONVERGE   = 0.70      # width_end/width_start below this = converging (triangle/wedge)
DIVERGE    = 1.45      # above this = broadening (we skip those)
NRMSE_MAX  = 0.030     # max line fit error (RMSE / mean price) to accept a line
MIN_TOUCH  = 3         # min pivots sitting on a line (within tolerance)
MIN_WINDOW = 22        # min bars a pattern must span
MIN_CONF   = 0.45      # default confidence floor for a reportable pattern


@dataclass
class Pattern:
    name: str
    bias: str
    window: int            # bars the pattern spans
    upper_slope_pct: float # %/bar
    lower_slope_pct: float
    upper_touches: int
    lower_touches: int
    upper_nrmse: float
    lower_nrmse: float
    conv_ratio: float      # width_end / width_start  (<1 converging)
    apex_bars: float       # bars until the two lines meet (NaN if parallel)
    upper_now: float       # line price at the last bar
    lower_now: float
    close: float
    pct_to_upper: float    # how far last close sits below upper line
    pct_to_lower: float    # how far last close sits above lower line
    vol_contraction: float # vol(last third)/vol(first third); <1 = drying up
    confidence: float

    def as_row(self) -> dict:
        return asdict(self)


# ---------------- pivots ----------------
def find_pivots(values: np.ndarray, k: int, hi: bool) -> list[int]:
    """Fractal swing points: index i is a pivot if it's the local extreme over +/-k bars.
    Adjacent duplicates within k bars are collapsed to the single most extreme one."""
    n = len(values)
    raw = []
    for i in range(k, n - k):
        seg = values[i - k:i + k + 1]
        c = values[i]
        if hi and c >= seg.max():
            raw.append(i)
        elif (not hi) and c <= seg.min():
            raw.append(i)

    kept: list[int] = []
    for i in raw:
        if kept and i - kept[-1] <= k:
            better = values[i] > values[kept[-1]] if hi else values[i] < values[kept[-1]]
            if better:
                kept[-1] = i
        else:
            kept.append(i)
    return kept


def _fit_line(xs: np.ndarray, ys: np.ndarray, mean_price: float, tol: float):
    """Least-squares line through pivots. Returns slope, intercept, nrmse, touches.
    nrmse (not R^2) is used so a genuinely flat line isn't wrongly rejected."""
    lr = linregress(xs, ys)
    pred = lr.slope * xs + lr.intercept
    resid = ys - pred
    rmse = float(np.sqrt(np.mean(resid ** 2)))
    nrmse = rmse / mean_price
    touches = int(np.sum(np.abs(resid) <= tol))
    return lr.slope, lr.intercept, nrmse, touches


def _classify(su_pct: float, sl_pct: float, conv: float) -> tuple[str, str] | None:
    """Map (upper slope %/bar, lower slope %/bar, convergence ratio) -> (name, bias)."""
    up_flat, lo_flat = abs(su_pct) < FLAT_SLOPE, abs(sl_pct) < FLAT_SLOPE
    up_dn, up_up = su_pct <= -FLAT_SLOPE, su_pct >= FLAT_SLOPE
    lo_dn, lo_up = sl_pct <= -FLAT_SLOPE, sl_pct >= FLAT_SLOPE
    converging = conv < CONVERGE

    if converging:
        if up_flat and lo_up:  return "Ascending triangle",  "Bullish"
        if up_dn and lo_flat:  return "Descending triangle", "Bearish"
        if up_dn and lo_up:    return "Symmetrical triangle", "Neutral"
        if up_up and lo_up:    return "Rising wedge",        "Bearish"
        if up_dn and lo_dn:    return "Falling wedge",       "Bullish"
        return None
    if conv > DIVERGE:
        return None  # broadening formation — out of scope
    # parallel-ish band
    if up_flat and lo_flat:    return "Rectangle / Range",   "Neutral"
    if up_up and lo_up:        return "Ascending channel",   "Up-trend"
    if up_dn and lo_dn:        return "Descending channel",  "Down-trend"
    return None


def _analyze_window(high: np.ndarray, low: np.ndarray, close: np.ndarray,
                    atr: float, k: int) -> Pattern | None:
    """Detect the best pattern within one OHLC window (arrays already sliced)."""
    n = len(close)
    if n < MIN_WINDOW:
        return None

    hi_idx = find_pivots(high, k, hi=True)
    lo_idx = find_pivots(low,  k, hi=False)
    if len(hi_idx) < 2 or len(lo_idx) < 2:
        return None

    mean_price = float(close.mean())
    tol = max(0.6 * atr, 0.012 * mean_price)   # how close a pivot must sit to "touch"

    su, iu, nru, tu = _fit_line(np.array(hi_idx, float), high[hi_idx], mean_price, tol)
    sl, il, nrl, tl = _fit_line(np.array(lo_idx, float), low[lo_idx],  mean_price, tol)

    if nru > NRMSE_MAX or nrl > NRMSE_MAX:
        return None
    if tu < MIN_TOUCH or tl < MIN_TOUCH:
        return None

    x0, x1 = 0.0, float(n - 1)
    upper0, upper1 = su * x0 + iu, su * x1 + iu
    lower0, lower1 = sl * x0 + il, sl * x1 + il
    width0, width1 = upper0 - lower0, upper1 - lower1
    # lines must not cross inside the window and width must stay positive
    if width0 <= 0 or width1 <= 0:
        return None
    conv = width1 / width0

    su_pct = su / mean_price * 100.0
    sl_pct = sl / mean_price * 100.0
    cls = _classify(su_pct, sl_pct, conv)
    if cls is None:
        return None
    name, bias = cls

    # apex (where the lines meet), measured in bars beyond the last bar
    apex_bars = float("nan")
    if abs(su - sl) > 1e-9:
        x_star = (il - iu) / (su - sl)
        apex_bars = x_star - x1
        if name not in ("Rectangle / Range", "Ascending channel", "Descending channel"):
            # a real triangle/wedge apex is ahead of price and not absurdly far
            if not (0 < apex_bars < 3.5 * n):
                return None

    last = float(close[-1])
    pct_to_upper = (upper1 - last) / last * 100.0
    pct_to_lower = (last - lower1) / last * 100.0
    # price must still be inside the pattern (allow a small breakout overshoot);
    # if it has already escaped far beyond a rail, this isn't a forming pattern
    if min(pct_to_upper, pct_to_lower) < -3.0:
        return None

    third = max(1, n // 3)
    # volume isn't passed here; contraction is filled by the caller if available
    vol_contraction = float("nan")

    line_q  = 1.0 - min(1.0, ((nru + nrl) / 2) / NRMSE_MAX)
    touch_q = min(1.0, (tu + tl) / 8.0)
    if name in ("Ascending channel", "Descending channel", "Rectangle / Range"):
        clarity = 1.0 - min(1.0, abs(su_pct - sl_pct) / (abs(su_pct) + abs(sl_pct) + 0.1))
    else:
        clarity = min(1.0, (CONVERGE - conv) / CONVERGE) if conv < CONVERGE else 0.0
    confidence = round(0.45 * line_q + 0.30 * touch_q + 0.25 * clarity, 3)

    return Pattern(
        name=name, bias=bias, window=n,
        upper_slope_pct=round(su_pct, 4), lower_slope_pct=round(sl_pct, 4),
        upper_touches=tu, lower_touches=tl,
        upper_nrmse=round(nru, 4), lower_nrmse=round(nrl, 4),
        conv_ratio=round(conv, 3), apex_bars=round(apex_bars, 1),
        upper_now=round(upper1, 2), lower_now=round(lower1, 2), close=round(last, 2),
        pct_to_upper=round(pct_to_upper, 2), pct_to_lower=round(pct_to_lower, 2),
        vol_contraction=vol_contraction, confidence=confidence,
    )


def _atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> float:
    if len(close) < 2:
        return float(high[-1] - low[-1]) if len(close) else 0.0
    pc = np.roll(close, 1)
    tr = np.maximum(high - low, np.maximum(np.abs(high - pc), np.abs(low - pc)))
    tr[0] = high[0] - low[0]
    return float(np.mean(tr[-period:]))


def scan_symbol(high, low, close, volume=None,
                lookbacks=(60, 90, 130), k: int = 3,
                min_conf: float = MIN_CONF) -> Pattern | None:
    """Try several lookback windows and return the single best-confidence pattern.
    Each input is a 1-D array of daily values (oldest -> newest)."""
    high, low, close = map(lambda a: np.asarray(a, float), (high, low, close))
    best: Pattern | None = None
    for lb in lookbacks:
        if lb > len(close):
            continue
        h, l, c = high[-lb:], low[-lb:], close[-lb:]
        atr = _atr(h, l, c)
        p = _analyze_window(h, l, c, atr, k)
        if p is None:
            continue
        if volume is not None:
            v = np.asarray(volume[-lb:], float)
            third = max(1, len(v) // 3)
            first, last = v[:third].mean(), v[-third:].mean()
            p.vol_contraction = round(float(last / first), 2) if first > 0 else float("nan")
        if best is None or p.confidence > best.confidence:
            best = p
    if best and best.confidence >= min_conf:
        return best
    return None


# ---------------- self-test ----------------
def _synth():
    """Generate clean synthetic patterns and confirm the classifier labels them."""
    rng = np.random.default_rng(7)
    n = 90
    x = np.arange(n)
    P = 6  # swing period

    def make(top0, topS, bot0, botS, noise=0.25):
        upper = top0 + topS * x
        lower = bot0 + botS * x
        phase = x % P
        tri = np.where(phase <= P / 2, phase / (P / 2), 2 - phase / (P / 2))  # 0->1->0
        close = lower + tri * (upper - lower)
        poke = 0.003 * (upper + lower) / 2           # small wick beyond the envelope
        high = close + poke + rng.normal(0, noise, n)
        low = close - poke + rng.normal(0, noise, n)
        return high, low, close

    cases = {
        "Symmetrical triangle": make(120, -0.15, 80, 0.15),
        "Ascending triangle":   make(120, 0.0,  80, 0.30),
        "Descending triangle":  make(120, -0.30, 80, 0.0),
        "Rising wedge":         make(120, 0.18,  80, 0.42),
        "Falling wedge":        make(120, -0.42, 80, -0.18),
        "Ascending channel":    make(120, 0.30,  90, 0.30),
        "Descending channel":   make(120, -0.30, 90, -0.30),
        "Rectangle / Range":    make(120, 0.0,   90, 0.0),
    }
    print(f"{'expected':<22} {'detected':<22} {'conf':>5}  ok")
    print("-" * 60)
    ok = 0
    for expect, (h, l, c) in cases.items():
        p = scan_symbol(h, l, c, lookbacks=(n,), min_conf=0.0)
        got = p.name if p else "—"
        good = got == expect
        ok += good
        print(f"{expect:<22} {got:<22} {p.confidence if p else 0:>5}  {'Y' if good else 'n'}")
    print("-" * 60)
    print(f"{ok}/{len(cases)} classified correctly")


if __name__ == "__main__":
    _synth()
