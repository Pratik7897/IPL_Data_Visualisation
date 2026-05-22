"""
modules/ai/summarizer.py
Phase 4 — Feature 2: AI Match Summary Generator.

Generates structured match summaries from live analytics data.

Providers (priority order):
  1. Gemini (google-generativeai) — if GEMINI_API_KEY is set
  2. OpenAI (openai)              — if OPENAI_API_KEY is set
  3. Template engine              — always available, no API key needed

All providers return the same response schema:
    {
      short_summary:      str,
      detailed_summary:   str,
      key_moments:        list[str],
      fan_reaction:       str,
      top_players:        list[str],
      sentiment_verdict:  str,
      generated_by:       str,   # provider name
    }
"""
import os
import logging
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)

# ── Provider selection ────────────────────────────────────────────────────────
_GEMINI_KEY = os.getenv("GEMINI_API_KEY", "")
_OPENAI_KEY = os.getenv("OPENAI_API_KEY", "")


def _provider() -> str:
    if _GEMINI_KEY:
        return "gemini"
    if _OPENAI_KEY:
        return "openai"
    return "template"


# ── Data gathering ────────────────────────────────────────────────────────────
_IPL_PLAYERS = [
    "Kohli", "Dhoni", "Rohit", "Bumrah", "Jadeja", "Dhawan", "Warner",
    "Babar", "Maxwell", "Stokes", "Rashid", "Nortje", "Siraj", "Shami",
    "Gill", "Pant", "Rahul", "Iyer", "Hardik", "Pollard", "Russell",
    "Karthik", "Samson", "Ruturaj", "Buttler", "Tewatia", "Chahal",
    "Kuldeep", "Boult", "Rabada", "Suryakumar", "Hetmyer", "Stoinis",
]

_PLAYER_RE = re.compile(
    r"\b(" + "|".join(re.escape(p) for p in _IPL_PLAYERS) + r")\b",
    re.IGNORECASE,
)


def _gather_analytics() -> dict:
    """Read-only pull from existing DB functions — no side effects."""
    from modules.database import (
        fetch_recent_tweets_with_sentiment,
        fetch_team_sentiment_counts,
        fetch_match_events,
        get_tweet_volume_per_minute,
    )

    tweets  = fetch_recent_tweets_with_sentiment(limit=2000)
    teams   = fetch_team_sentiment_counts()
    events  = fetch_match_events(minutes=90)
    volume  = get_tweet_volume_per_minute(minutes=60)

    # Overall sentiment distribution
    labels = [t["label"] for t in tweets if t.get("label")]
    label_counts = Counter(labels)
    total = len(labels) or 1

    # Team sentiment map
    team_sent: dict[str, dict] = defaultdict(lambda: {"Positive": 0, "Neutral": 0, "Negative": 0})
    for r in teams:
        t = r.get("team_mention") or ""
        if t:
            team_sent[t][r["label"]] = r["count"]

    # Top players by mention count
    player_mentions: Counter = Counter()
    for tw in tweets:
        text = tw.get("text", "")
        for m in _PLAYER_RE.finditer(text):
            player_mentions[m.group().title()] += 1
    top_players = [p for p, _ in player_mentions.most_common(5)]

    # Peak volume minute
    peak_minute = ""
    peak_count  = 0
    for row in volume:
        if row["count"] > peak_count:
            peak_count  = row["count"]
            peak_minute = row["minute"]

    # Event summary
    event_types = Counter(e["event_type"] for e in events)

    # Dominant team (most positive sentiment)
    dominant_team = ""
    best_score    = -99
    for team, s in team_sent.items():
        tot = sum(s.values()) or 1
        score = (s["Positive"] - s["Negative"]) / tot
        if score > best_score:
            best_score    = score
            dominant_team = team

    return {
        "total_tweets":    len(tweets),
        "label_counts":    dict(label_counts),
        "pos_pct":         round(label_counts.get("Positive", 0) / total * 100, 1),
        "neg_pct":         round(label_counts.get("Negative", 0) / total * 100, 1),
        "neu_pct":         round(label_counts.get("Neutral",  0) / total * 100, 1),
        "team_sent":       dict(team_sent),
        "top_players":     top_players,
        "dominant_team":   dominant_team,
        "peak_minute":     peak_minute,
        "peak_volume":     peak_count,
        "event_types":     dict(event_types),
        "total_events":    len(events),
    }


# ── Template engine ────────────────────────────────────────────────────────────
def _template_summary(data: dict) -> dict:
    dominant  = data["dominant_team"] or "the batting side"
    top_p     = data["top_players"]
    star      = top_p[0] if top_p else "the leading player"
    second    = top_p[1] if len(top_p) > 1 else "their teammates"
    pos_pct   = data["pos_pct"]
    neg_pct   = data["neg_pct"]
    total     = data["total_tweets"]
    events    = data["event_types"]
    wickets   = events.get("wicket", 0)
    sixes     = events.get("six", 0)
    fours     = events.get("four", 0)

    # Sentiment verdict
    if pos_pct > 55:
        verdict = "overwhelmingly positive"
    elif pos_pct > 40:
        verdict = "cautiously optimistic"
    elif neg_pct > 45:
        verdict = "frustrated and critical"
    else:
        verdict = "mixed and uncertain"

    short = (
        f"{dominant} commands fan sentiment with {pos_pct}% positive reactions "
        f"as {star} drives the biggest buzz across {total:,} analysed tweets."
    )

    detailed = (
        f"Across {total:,} tweets analysed in the last 90 minutes, fan mood is {verdict}. "
        f"{dominant} holds the strongest positive sentiment on social media. "
        f"{star} is the most mentioned player, generating significant discussion alongside "
        f"{second}. "
    )
    if wickets:
        detailed += f"The match has seen {wickets} wicket event(s) logged. "
    if sixes:
        detailed += f"{sixes} six(es) have electrified fans this session. "
    if data["peak_minute"]:
        detailed += (
            f"Tweet volume peaked at {data['peak_minute']} UTC "
            f"({data['peak_volume']} tweets in one minute), "
            "likely corresponding to a key on-field moment."
        )

    key_moments = []
    if star:
        key_moments.append(f"{star} generates the highest social media buzz of the match.")
    if dominant:
        key_moments.append(f"{dominant} supporters dominate positive sentiment threads.")
    if wickets:
        key_moments.append(f"{wickets} wicket moment(s) triggered sharp sentiment swings.")
    if sixes:
        key_moments.append(f"{sixes} six event(s) produced viral fan reactions.")
    if data["peak_minute"]:
        key_moments.append(
            f"Engagement spike at {data['peak_minute']} — "
            f"{data['peak_volume']} tweets in one minute."
        )

    fan_reaction = (
        f"Fans are {verdict} about today's match. "
        f"Negative sentiment sits at {neg_pct}%, mainly driven by "
        f"{'umpiring decisions and missed opportunities' if neg_pct > 30 else 'opponent supporters'}. "
        f"Overall engagement is {'high' if total > 500 else 'moderate'} with {total:,} tweets processed."
    )

    return {
        "short_summary":    short,
        "detailed_summary": detailed,
        "key_moments":      key_moments,
        "fan_reaction":     fan_reaction,
        "top_players":      top_p,
        "sentiment_verdict": verdict,
        "generated_by":     "Template Engine (set GEMINI_API_KEY or OPENAI_API_KEY for AI summaries)",
    }


# ── Gemini provider ───────────────────────────────────────────────────────────
def _gemini_summary(data: dict) -> dict:
    import google.generativeai as genai
    genai.configure(api_key=_GEMINI_KEY)
    model = genai.GenerativeModel("gemini-1.5-flash")

    prompt = _build_prompt(data)
    resp   = model.generate_content(prompt)
    return _parse_llm_response(resp.text, "Gemini 1.5 Flash", data)


# ── OpenAI provider ───────────────────────────────────────────────────────────
def _openai_summary(data: dict) -> dict:
    from openai import OpenAI
    client = OpenAI(api_key=_OPENAI_KEY)
    prompt = _build_prompt(data)
    resp   = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are an expert cricket analyst and sports journalist."},
            {"role": "user",   "content": prompt},
        ],
        temperature=0.7,
        max_tokens=800,
    )
    return _parse_llm_response(resp.choices[0].message.content, "GPT-4o-mini", data)


def _build_prompt(data: dict) -> str:
    return f"""You are an expert cricket analyst covering IPL 2026. 
Generate a structured match summary from the following real-time fan sentiment data.

DATA:
- Total tweets analysed: {data['total_tweets']}
- Positive sentiment: {data['pos_pct']}%
- Negative sentiment: {data['neg_pct']}%
- Neutral sentiment: {data['neu_pct']}%
- Dominant team by sentiment: {data['dominant_team']}
- Top mentioned players: {', '.join(data['top_players'][:5])}
- Peak tweet volume: {data['peak_volume']} tweets at {data['peak_minute']} UTC
- Match events: {data['event_types']}

Return JSON with exactly these keys:
{{
  "short_summary": "1 sentence, punchy",
  "detailed_summary": "3-4 sentences, analytical",
  "key_moments": ["moment 1", "moment 2", "moment 3"],
  "fan_reaction": "2 sentences on how fans are reacting",
  "sentiment_verdict": "one phrase describing overall mood"
}}"""


def _parse_llm_response(text: str, backend: str, data: dict) -> dict:
    import json
    try:
        # Extract JSON from possible markdown code fence
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            parsed = json.loads(match.group())
            parsed["top_players"]  = data["top_players"]
            parsed["generated_by"] = backend
            return parsed
    except Exception:
        pass
    # Fallback to template if parsing fails
    result = _template_summary(data)
    result["generated_by"] = f"{backend} (parse error — template fallback)"
    return result


# ── Public entry point ────────────────────────────────────────────────────────
def generate_summary() -> dict:
    """
    Generate an AI match summary from current live data.
    Provider is auto-selected based on available API keys.
    """
    data     = _gather_analytics()
    provider = _provider()

    logger.info(f"[summarizer] Generating summary via {provider} …")
    try:
        if provider == "gemini":
            result = _gemini_summary(data)
        elif provider == "openai":
            result = _openai_summary(data)
        else:
            result = _template_summary(data)
    except Exception as e:
        logger.warning(f"[summarizer] {provider} failed: {e} — falling back to template")
        result = _template_summary(data)
        result["generated_by"] += f" (fallback from {provider} error)"

    result["generated_at"] = datetime.now(timezone.utc).isoformat()
    result["analytics"]    = {
        "total_tweets": data["total_tweets"],
        "pos_pct": data["pos_pct"],
        "neg_pct": data["neg_pct"],
        "neu_pct": data["neu_pct"],
    }
    return result
