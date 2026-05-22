"""
modules/ai/player_index.py
Phase 4 — Feature 3: Player Popularity Index.

Ranks IPL players in real-time purely from existing tweet/sentiment data.
Zero impact on the existing pipeline — only reads from modules.database.

Popularity score formula (0-100):
    score = 40 * normalised_mention_freq
          + 30 * positive_sentiment_ratio
          + 20 * normalised_engagement
          + 10 * trend_velocity          (recent vs. earlier mentions)
"""
import re
import logging
from collections import Counter, defaultdict
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)

# ── Player registry ────────────────────────────────────────────────────────────
# (name, team, role)
PLAYER_REGISTRY = [
    ("Virat Kohli",     "RCB",  "Batsman"),
    ("MS Dhoni",        "CSK",  "WK-Batsman"),
    ("Rohit Sharma",    "MI",   "Batsman"),
    ("Jasprit Bumrah",  "MI",   "Bowler"),
    ("Ravindra Jadeja", "CSK",  "All-rounder"),
    ("Shubman Gill",    "GT",   "Batsman"),
    ("Hardik Pandya",   "MI",   "All-rounder"),
    ("KL Rahul",        "LSG",  "WK-Batsman"),
    ("Shreyas Iyer",    "KKR",  "Batsman"),
    ("Rishabh Pant",    "DC",   "WK-Batsman"),
    ("Glenn Maxwell",   "RCB",  "All-rounder"),
    ("David Warner",    "DC",   "Batsman"),
    ("Pat Cummins",     "SRH",  "Bowler"),
    ("Jos Buttler",     "RR",   "WK-Batsman"),
    ("Andre Russell",   "KKR",  "All-rounder"),
    ("Mohammed Shami",  "GT",   "Bowler"),
    ("Yuzvendra Chahal","RR",   "Bowler"),
    ("Rashid Khan",     "GT",   "Bowler"),
    ("Suryakumar Yadav","MI",   "Batsman"),
    ("Ruturaj Gaikwad", "CSK",  "Batsman"),
    ("Sanju Samson",    "RR",   "WK-Batsman"),
    ("Prithvi Shaw",    "DC",   "Batsman"),
    ("Trent Boult",     "RR",   "Bowler"),
    ("Kagiso Rabada",   "PBKS", "Bowler"),
    ("Kieron Pollard",  "MI",   "All-rounder"),
    ("AB de Villiers",  "RCB",  "Batsman"),
    ("Dinesh Karthik",  "RCB",  "WK-Batsman"),
    ("Faf du Plessis",  "RCB",  "Batsman"),
    ("Shikhar Dhawan",  "PBKS", "Batsman"),
    ("Mayank Agarwal",  "PBKS", "Batsman"),
]

# Build search aliases (surname + full name)
def _make_aliases(name: str) -> list[str]:
    parts = name.split()
    aliases = [name]
    if len(parts) >= 2:
        aliases.append(parts[-1])   # surname
        aliases.append(parts[0])    # first name
    return aliases


_PLAYER_PATTERNS: dict[str, re.Pattern] = {}
for player_name, _, _ in PLAYER_REGISTRY:
    aliases = _make_aliases(player_name)
    pattern = re.compile(
        r"\b(" + "|".join(re.escape(a) for a in aliases) + r")\b",
        re.IGNORECASE,
    )
    _PLAYER_PATTERNS[player_name] = pattern


def _normalise(values: dict, key: str) -> dict:
    """Min-max normalise a dict of floats."""
    mx = max(values.values()) if values else 1
    mn = min(values.values()) if values else 0
    rng = mx - mn or 1
    return {k: (v - mn) / rng for k, v in values.items()}


# ── Core scoring engine ───────────────────────────────────────────────────────
def compute_player_rankings() -> list[dict]:
    """
    Read existing tweet+sentiment data and rank all tracked players.
    Returns a list of dicts sorted by popularity score descending.
    """
    from modules.database import fetch_recent_tweets_with_sentiment

    # Fetch recent tweets split into two time windows for trend velocity
    recent_tweets = fetch_recent_tweets_with_sentiment(limit=2000)
    mid = len(recent_tweets) // 2
    early_tweets  = recent_tweets[mid:]   # older half
    late_tweets   = recent_tweets[:mid]   # newer half

    # ── Count mentions per player ──────────────────────────────────────────────
    def _count_mentions(tweets: list) -> dict[str, dict]:
        counts: dict[str, dict] = {
            name: {"total": 0, "Positive": 0, "Negative": 0, "Neutral": 0,
                   "engagement": 0}
            for name, _, _ in PLAYER_REGISTRY
        }
        for tweet in tweets:
            text       = tweet.get("text", "")
            label      = tweet.get("label", "Neutral") or "Neutral"
            likes      = tweet.get("like_count", 0) or 0
            retweets   = tweet.get("retweet_count", 0) or 0
            engagement = likes + retweets * 3

            for player_name, pattern in _PLAYER_PATTERNS.items():
                if pattern.search(text):
                    counts[player_name]["total"]      += 1
                    counts[player_name][label]         += 1
                    counts[player_name]["engagement"] += engagement
        return counts

    all_counts   = _count_mentions(recent_tweets)
    early_counts = _count_mentions(early_tweets)
    late_counts  = _count_mentions(late_tweets)

    # Raw metrics
    mention_raw    = {n: c["total"]      for n, c in all_counts.items()}
    engagement_raw = {n: c["engagement"] for n, c in all_counts.items()}

    # Normalise
    mention_norm    = _normalise(mention_raw, "mention")
    engagement_norm = _normalise(engagement_raw, "engagement")

    results = []
    for name, team, role in PLAYER_REGISTRY:
        c   = all_counts[name]
        ec  = early_counts[name]
        lc  = late_counts[name]
        tot = c["total"] or 1

        # Positive sentiment ratio
        pos_ratio = c["Positive"] / tot

        # Trend velocity: recent vs. earlier (clamped)
        early_t = ec["total"] or 0.1
        late_t  = lc["total"] or 0
        velocity = min((late_t - early_t) / early_t, 1.0)

        # Composite score (0-100)
        score = (
            40 * mention_norm.get(name, 0)
            + 30 * pos_ratio
            + 20 * engagement_norm.get(name, 0)
            + 10 * max(velocity, 0)
        )
        score = round(score * 100, 1)

        # Trend indicator
        if velocity > 0.3:
            trend = "↑↑ Trending"
        elif velocity > 0.05:
            trend = "↑ Rising"
        elif velocity < -0.3:
            trend = "↓↓ Fading"
        elif velocity < -0.05:
            trend = "↓ Dropping"
        else:
            trend = "→ Stable"

        results.append({
            "name":         name,
            "team":         team,
            "role":         role,
            "popularity":   score,
            "mentions":     c["total"],
            "positive_pct": round(pos_ratio * 100, 1),
            "negative_pct": round(c["Negative"] / tot * 100, 1),
            "engagement":   c["engagement"],
            "trend":        trend,
            "velocity":     round(velocity, 3),
        })

    results.sort(key=lambda x: x["popularity"], reverse=True)
    # Assign rank
    for i, r in enumerate(results, 1):
        r["rank"] = i

    return results


def get_top_trending() -> dict:
    """Return only the single top trending player right now."""
    rankings = compute_player_rankings()
    # Sort by velocity (trend) rather than overall popularity
    by_velocity = sorted(rankings, key=lambda x: x["velocity"], reverse=True)
    top = by_velocity[0] if by_velocity else {}
    return {
        "player": top.get("name", "N/A"),
        "team":   top.get("team", ""),
        "trend":  top.get("trend", ""),
        "velocity": top.get("velocity", 0),
        "popularity": top.get("popularity", 0),
        "mentions":  top.get("mentions", 0),
    }


def get_player_sentiment_history(player_name: str) -> dict:
    """
    Return per-tweet sentiment breakdown for a specific player.
    Uses existing fetch functions — read-only.
    """
    from modules.database import fetch_recent_tweets_with_sentiment

    tweets  = fetch_recent_tweets_with_sentiment(limit=2000)
    pattern = _PLAYER_PATTERNS.get(player_name)

    if not pattern:
        return {"error": f"Player '{player_name}' not found in registry."}

    player_info = next(
        ((n, t, r) for n, t, r in PLAYER_REGISTRY if n == player_name), None
    )

    matched = []
    label_counts = Counter()
    for tw in tweets:
        if pattern.search(tw.get("text", "")):
            label = tw.get("label", "Neutral") or "Neutral"
            label_counts[label] += 1
            matched.append({
                "text":    tw["text"][:120],
                "label":   label,
                "score":   tw.get("score", 0),
                "created": tw.get("created_at", ""),
            })

    total = len(matched) or 1
    return {
        "player":      player_name,
        "team":        player_info[1] if player_info else "",
        "role":        player_info[2] if player_info else "",
        "total_mentions": len(matched),
        "positive_pct":   round(label_counts["Positive"] / total * 100, 1),
        "negative_pct":   round(label_counts["Negative"] / total * 100, 1),
        "neutral_pct":    round(label_counts["Neutral"]  / total * 100, 1),
        "recent_tweets":  matched[:10],
        "generated_at":   datetime.now(timezone.utc).isoformat(),
    }
