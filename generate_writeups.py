"""
Plain-English LLM write-ups for the high-conviction chart setups.

Multi-provider (Gemini -> Groq -> OpenRouter), tried in order, each with a
per-day request cap kept UNDER its free tier and per-minute pacing. When a
provider hits its daily cap or quota we fall through to the next. Per-day counts
persist in cache/ai_usage.json (reset each UTC day); write-ups cache in
cache/writeups_cache.json keyed by a hash of the *stable* signal facts, so only
new/changed setups cost a request. Safe no-op if no provider key is set.

Educational, descriptive only — no buy/sell advice.

Env (set the provider keys you have):
  GEMINI_API_KEY / GROQ_API_KEY / OPENROUTER_API_KEY
  AI_PROVIDER_ORDER (gemini,groq,openrouter), *_MODEL, *_DAILY_CAP, *_INTERVAL
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

import pandas as pd

try:
    import requests
except Exception:  # pragma: no cover
    requests = None

HERE = Path(__file__).parent
CACHE_DIR = HERE / "cache"
CACHE_PATH = CACHE_DIR / "writeups_cache.json"
USAGE_PATH = CACHE_DIR / "ai_usage.json"
PROMPT_VERSION = "patterns-v1"
BUDGET_SECS = float(os.getenv("AI_WRITEUP_BUDGET_SECS", "420"))

SYSTEM = (
    "You are a plain-English stock-chart educator writing for a complete beginner "
    "in India. You explain what a chart pattern is and what the trading VOLUME is "
    "telling us, in simple words, with no jargon and no buy/sell advice."
)


def _env(name: str, default: str) -> str:
    v = os.getenv(name)
    return v if v not in (None, "") else default


_PROVIDER_DEFS = {
    "gemini": {
        "base_url": _env("GEMINI_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai"),
        "key": os.getenv("GEMINI_API_KEY", ""),
        # 2.0-flash (not 2.5): no "thinking" tokens that would eat the response
        # budget and truncate the write-up, plus its own fresh daily quota.
        "model": _env("GEMINI_MODEL", "gemini-2.0-flash"),
        "daily_cap": int(_env("GEMINI_DAILY_CAP", "1400")),
        "interval": float(_env("GEMINI_INTERVAL", "4.5")),
    },
    "groq": {
        "base_url": _env("GROQ_BASE_URL", "https://api.groq.com/openai/v1"),
        "key": os.getenv("GROQ_API_KEY", ""),
        "model": _env("GROQ_MODEL", "llama-3.3-70b-versatile"),
        "daily_cap": int(_env("GROQ_DAILY_CAP", "900")),
        "interval": float(_env("GROQ_INTERVAL", "2.2")),
    },
    "openrouter": {
        "base_url": _env("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
        "key": os.getenv("OPENROUTER_API_KEY", ""),
        "model": _env("OPENROUTER_MODEL", "google/gemma-4-31b-it:free"),
        "daily_cap": int(_env("OPENROUTER_DAILY_CAP", "40")),
        "interval": float(_env("OPENROUTER_INTERVAL", "4.0")),
    },
}
PROVIDER_ORDER = [n.strip() for n in _env("AI_PROVIDER_ORDER", "gemini,groq,openrouter").split(",") if n.strip()]
PROVIDERS = [dict(_PROVIDER_DEFS[n], name=n) for n in PROVIDER_ORDER
             if n in _PROVIDER_DEFS and _PROVIDER_DEFS[n]["key"]]


class AllProvidersExhausted(Exception):
    pass


def _load_json(path: Path, default):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def _load_usage() -> dict:
    u = _load_json(USAGE_PATH, {})
    today = time.strftime("%Y-%m-%d", time.gmtime())
    if u.get("date") != today:
        u = {"date": today, "used": {}}
    u.setdefault("used", {})
    return u


def _save(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def _is_daily_quota(resp) -> bool:
    if resp.status_code != 429:
        return False
    t = (resp.text or "").lower()
    return any(k in t for k in ("per day", "/day", "per-day", "daily", "requests per day",
                                "quota", "resource_exhausted", "rpd", "free-models-per-day"))


def _post(provider: dict, system: str, user: str, timeout: int = 90):
    return requests.post(
        f"{provider['base_url'].rstrip('/')}/chat/completions",
        headers={
            "Authorization": f"Bearer {provider['key']}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://myfinancialria.github.io/chart-analysis-patterns/",
            "X-Title": "NIFTY 500 Chart Patterns",
        },
        json={
            "model": provider["model"],
            "temperature": 0.4,
            "max_tokens": 700,  # headroom so any provider's reply isn't truncated
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        },
        timeout=timeout,
    )


def _nz(v):
    return v is not None and not (isinstance(v, float) and pd.isna(v))


def signal_facts(r) -> dict:
    """Stable facts about one setup (no daily price wiggle) for prompt + cache key."""
    f = {
        "symbol": r.get("symbol"),
        "sector": r.get("sector"),
        "pattern": r.get("name"),
        "bias": r.get("bias"),
        "state": r.get("state"),
        "confidence": round(float(r.get("confidence", 0) or 0), 2),
    }
    vs, vc = r.get("vol_surge"), r.get("vol_contraction")
    if _nz(vs):
        f["breakout_volume_vs_average_x"] = round(float(vs), 2)
    if _nz(vc):
        f["forming_volume_now_vs_start_ratio"] = round(float(vc), 2)
    tu, tl = r.get("pct_to_upper"), r.get("pct_to_lower")
    if _nz(tu):
        f["pct_to_resistance_line"] = round(float(tu), 2)
    if _nz(tl):
        f["pct_to_support_line"] = round(float(tl), 2)
    return f


def _prompt(facts: dict) -> str:
    return (
        "Explain this stock's chart setup in very simple words a beginner can follow, "
        "and say clearly what the VOLUME is telling us. Cover: what the pattern is and "
        "whether it leans bullish or bearish; what a break of the key level would mean; "
        "and whether volume is CONFIRMING the move — a breakout on a big volume surge is "
        "far more convincing, while a quiet squeeze where volume dries up often comes "
        "just before a strong move. Write 3-5 short sentences or bullets, then one short "
        "honest risk. No buy/sell advice. Use only these facts:\n\n"
        + json.dumps(facts, ensure_ascii=False, indent=2)
    )


def cache_key(facts: dict) -> str:
    blob = json.dumps({"v": PROMPT_VERSION, "f": facts}, sort_keys=True, ensure_ascii=False)
    return hashlib.sha1(blob.encode("utf-8")).hexdigest()


def generate_one(facts: dict, usage: dict):
    """One write-up via the provider chain, honoring per-day caps + failover."""
    prompt = _prompt(facts)
    used = usage["used"]
    any_available = False
    for p in PROVIDERS:
        if used.get(p["name"], 0) >= p["daily_cap"]:
            continue
        any_available = True
        time.sleep(p["interval"])
        try:
            resp = _post(p, SYSTEM, prompt)
        except Exception as exc:
            print(f"    ({p['name']} error: {exc}); trying next")
            continue
        if resp.status_code == 200:
            used[p["name"]] = used.get(p["name"], 0) + 1
            _save(USAGE_PATH, usage)
            data = resp.json()
            choices = data.get("choices") or []
            text = (choices[0].get("message", {}).get("content") or "").strip() if choices else ""
            if text:
                return text, p["name"]
            continue
        if resp.status_code == 429:
            if _is_daily_quota(resp):
                used[p["name"]] = p["daily_cap"]
                _save(USAGE_PATH, usage)
                print(f"    ({p['name']} daily quota reached — switching provider)")
            else:
                print(f"    ({p['name']} per-minute 429 — switching provider)")
            continue
        print(f"    ({p['name']} HTTP {resp.status_code}: {resp.text[:150]}); trying next")
    if not any_available:
        raise AllProvidersExhausted()
    raise RuntimeError("all providers failed for this item")


def generate(df: pd.DataFrame) -> dict:
    """Return {f'{symbol}|{name}': write-up} for the surfaced setups. Cache +
    per-day usage persist; a wall-clock budget keeps the step bounded."""
    out: dict[str, str] = {}
    if df is None or df.empty:
        return out
    cache = _load_json(CACHE_PATH, {})
    usage = _load_usage()

    work = []
    for _, r in df.iterrows():
        facts = signal_facts(r)
        key = f"{r.get('symbol')}|{r.get('name')}"
        ck = cache_key(facts)
        work.append((key, ck, facts))

    for key, ck, _ in work:
        if ck in cache:
            out[key] = cache[ck]

    todo = [(key, ck, facts) for key, ck, facts in work if ck not in cache]
    if todo and not PROVIDERS:
        print("write-ups: no provider key set (GEMINI_API_KEY / GROQ_API_KEY / OPENROUTER_API_KEY) — skipping.")
        return out
    if not todo:
        print(f"write-ups: all {len(work)} setups already cached.")
        return out

    print(f"write-ups: {len(work)} setups, {len(todo)} to generate via "
          f"{', '.join(p['name'] for p in PROVIDERS)}.")
    deadline = time.monotonic() + BUDGET_SECS
    ok = 0
    for key, ck, facts in todo:
        if time.monotonic() > deadline:
            print(f"  time budget reached — {ok} generated this run.")
            break
        try:
            text, prov = generate_one(facts, usage)
            cache[ck] = text
            out[key] = text
            ok += 1
            _save(CACHE_PATH, cache)
        except AllProvidersExhausted:
            print("  all providers hit their daily cap — stopping; rest resume next run.")
            break
        except Exception as exc:
            print(f"  ! failed: {key} — {exc}")
    _save(CACHE_PATH, cache)
    _save(USAGE_PATH, usage)
    print(f"  write-ups this run: {ok} | usage today (UTC): {usage['used']}")
    return out
