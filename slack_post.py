"""Post the pattern-scan digest to the internal Slack workspace.

Bot-token mode (uploads the montage image): SLACK_BOT_TOKEN + SLACK_CHANNEL
Webhook mode (text only):                    SLACK_WEBHOOK_URL

Internal use only — descriptive pattern notes, no buy/sell calls.
"""
from __future__ import annotations

import json
import os
from datetime import date
from pathlib import Path

import pandas as pd
import requests

BIAS_EMOJI = {"Bullish": ":green_circle:", "Bearish": ":red_circle:",
              "Neutral": ":white_circle:", "Up-trend": ":arrow_upper_right:",
              "Down-trend": ":arrow_lower_right:"}


def _line(r: pd.Series) -> str:
    emo = BIAS_EMOJI.get(r["bias"], ":white_circle:")
    vol = ""
    if pd.notna(r.get("vol_contraction")) and r["vol_contraction"] < 0.8:
        vol = " · vol drying up"
    return (f"{emo} *{r['symbol']}* — {r['name']} ({r['bias']})  "
            f"conf {r['confidence']:.2f} · "
            f"{r['pct_to_upper']:+.1f}% to resistance / "
            f"{r['pct_to_lower']:+.1f}% to support{vol}")


def build_text(df: pd.DataFrame) -> str:
    counts = df["name"].value_counts()
    head = "  ·  ".join(f"{n.split()[0]} {c}" for n, c in counts.items())
    triangles = df[df["name"].str.contains("triangle|wedge", case=False)].head(10)
    channels = df[df["name"].str.contains("channel|Rectangle")].head(6)

    parts = [f"*NIFTY 500 chart patterns — {date.today():%d %b %Y}* (daily EOD)",
             f"_{len(df)} setups: {head}_", ""]
    if not triangles.empty:
        parts.append("*Triangles & wedges (tightening):*")
        parts += [_line(r) for _, r in triangles.iterrows()]
        parts.append("")
    if not channels.empty:
        parts.append("*Channels & ranges:*")
        parts += [_line(r) for _, r in channels.iterrows()]
    parts.append("")
    parts.append("_Educational pattern scan — descriptive only, not investment advice._")
    return "\n".join(parts)


def post_webhook(df: pd.DataFrame) -> None:
    requests.post(os.environ["SLACK_WEBHOOK_URL"],
                  json={"text": build_text(df)}, timeout=15).raise_for_status()
    print("Posted pattern digest to Slack webhook.")


def post_upload(df: pd.DataFrame, image: Path | None) -> None:
    token = os.environ["SLACK_BOT_TOKEN"]
    channel = os.environ["SLACK_CHANNEL"]
    headers = {"Authorization": f"Bearer {token}"}
    comment = build_text(df)

    if not image or not image.exists():
        requests.post("https://slack.com/api/chat.postMessage", headers=headers,
                      json={"channel": channel, "text": comment},
                      timeout=15).raise_for_status()
        print("Posted pattern digest (text) to Slack.")
        return

    size = image.stat().st_size
    j = requests.get("https://slack.com/api/files.getUploadURLExternal", headers=headers,
                     params={"filename": image.name, "length": size}, timeout=15).json()
    if not j.get("ok"):
        raise RuntimeError(f"getUploadURLExternal failed: {j}")
    with image.open("rb") as f:
        requests.post(j["upload_url"], files={"file": f}, timeout=60).raise_for_status()
    r = requests.post("https://slack.com/api/files.completeUploadExternal",
                      headers={**headers, "Content-Type": "application/json"},
                      data=json.dumps({
                          "files": [{"id": j["file_id"], "title": f"Patterns {date.today()}"}],
                          "channel_id": channel, "initial_comment": comment}),
                      timeout=15).json()
    if not r.get("ok"):
        raise RuntimeError(f"completeUploadExternal failed: {r}")
    print(f"Uploaded pattern montage to Slack channel {channel}.")


def post(df: pd.DataFrame, image: Path | None = None) -> None:
    if os.environ.get("SLACK_BOT_TOKEN") and os.environ.get("SLACK_CHANNEL"):
        post_upload(df, image)
    elif os.environ.get("SLACK_WEBHOOK_URL"):
        post_webhook(df)
    else:
        print("No Slack creds set (SLACK_BOT_TOKEN+SLACK_CHANNEL or SLACK_WEBHOOK_URL) — skipping.")
