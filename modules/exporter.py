"""
Phase 3C — Post-match data export module.

Provides CSV, JSON and Markdown summary exports of the entire match dataset.

Endpoints registered in app.py on the Flask `server` object:
    GET /export/csv      → tweets + sentiment as .csv
    GET /export/json     → same data as .json
    GET /export/events   → match_events as .csv
    GET /export/summary  → Markdown analysis report
    GET /export/health   → simple health-check (used by Nginx + monitoring)

All endpoints stream the response so large datasets don't blow up memory.
"""
import csv
import io
import json
import logging
from collections import Counter
from datetime import datetime, timezone
from typing import Any

from flask import Response, stream_with_context

logger = logging.getLogger(__name__)


# ── Helpers ───────────────────────────────────────────────────────────────────
def _now_tag() -> str:
    """Timestamp string suitable for filenames: 20260522_153000"""
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def _csv_response(rows: list[dict], filename: str) -> Response:
    """Stream a list-of-dicts as a downloadable CSV file."""
    if not rows:
        return Response("No data available yet.\n",
                        mimetype="text/plain", status=204)

    fieldnames = list(rows[0].keys())

    def generate():
        buf = io.StringIO()
        writer = csv.DictWriter(buf, fieldnames=fieldnames,
                                extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: (v if v is not None else "") for k, v in row.items()})
            yield buf.getvalue()
            buf.seek(0)
            buf.truncate()

    headers = {
        "Content-Disposition": f'attachment; filename="{filename}"',
        "Content-Type": "text/csv; charset=utf-8",
        "X-Content-Type-Options": "nosniff",
    }
    return Response(stream_with_context(generate()),
                    headers=headers, mimetype="text/csv")


def _json_response(data: Any, filename: str) -> Response:
    """Return pretty-printed JSON as a downloadable file."""
    payload = json.dumps(data, default=str, ensure_ascii=False, indent=2)
    headers = {
        "Content-Disposition": f'attachment; filename="{filename}"',
        "Content-Type": "application/json; charset=utf-8",
    }
    return Response(payload, headers=headers, mimetype="application/json")


# ── Export builders ───────────────────────────────────────────────────────────
def export_tweets_csv() -> Response:
    """All tweets + sentiment labels as CSV."""
    from modules.database import fetch_recent_tweets_with_sentiment
    rows = fetch_recent_tweets_with_sentiment(limit=10_000)
    logger.info(f"[export] CSV — {len(rows)} tweet rows")
    return _csv_response(rows, f"ipl_tweets_{_now_tag()}.csv")


def export_tweets_json() -> Response:
    """All tweets + sentiment as JSON."""
    from modules.database import fetch_recent_tweets_with_sentiment
    rows = fetch_recent_tweets_with_sentiment(limit=10_000)
    logger.info(f"[export] JSON — {len(rows)} tweet rows")
    return _json_response({
        "exported_at": datetime.now(timezone.utc).isoformat(),
        "total":       len(rows),
        "tweets":      rows,
    }, f"ipl_tweets_{_now_tag()}.json")


def export_events_csv() -> Response:
    """All match events as CSV."""
    from modules.database import fetch_match_events
    rows = fetch_match_events(minutes=1440)   # full 24 h window
    logger.info(f"[export] Events CSV — {len(rows)} rows")
    return _csv_response(rows, f"ipl_events_{_now_tag()}.csv")


def export_summary_md() -> Response:
    """Generate a Markdown post-match analysis report."""
    from modules.database import (
        fetch_recent_tweets_with_sentiment,
        fetch_team_sentiment_counts,
        fetch_match_events,
    )

    tweets  = fetch_recent_tweets_with_sentiment(limit=10_000)
    teams   = fetch_team_sentiment_counts()
    events  = fetch_match_events(minutes=1440)

    total   = len(tweets)
    labels  = [t["label"] for t in tweets if t.get("label")]
    cnt     = Counter(labels)
    pos_pct = round(cnt.get("Positive", 0) / max(total, 1) * 100, 1)
    neg_pct = round(cnt.get("Negative", 0) / max(total, 1) * 100, 1)
    neu_pct = round(cnt.get("Neutral",  0) / max(total, 1) * 100, 1)

    # Overall dominant sentiment
    dominant = cnt.most_common(1)[0][0] if cnt else "N/A"

    # Top teams by total mentions
    team_totals: dict[str, int] = {}
    for r in teams:
        team_totals[r["team_mention"]] = \
            team_totals.get(r["team_mention"], 0) + r["count"]
    top_teams = sorted(team_totals.items(), key=lambda x: x[1], reverse=True)[:5]

    # Team sentiment breakdown table
    team_table: dict[str, dict] = {}
    for r in teams:
        t = r["team_mention"]
        if t not in team_table:
            team_table[t] = {"Positive": 0, "Neutral": 0, "Negative": 0}
        team_table[t][r["label"]] = r["count"]

    # Event summary
    event_cnt = Counter(e["event_type"] for e in events)

    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    filename     = f"ipl_match_summary_{_now_tag()}.md"

    md = f"""# 🏏 IPL Sentiment Match Report
*Generated: {generated_at}*

---

## 📊 Overall Sentiment

| Metric | Value |
|---|---|
| Total Tweets Analysed | **{total:,}** |
| Dominant Sentiment | **{dominant}** |
| Positive | {cnt.get('Positive', 0):,} tweets ({pos_pct}%) |
| Neutral  | {cnt.get('Neutral',  0):,} tweets ({neu_pct}%) |
| Negative | {cnt.get('Negative', 0):,} tweets ({neg_pct}%) |

---

## 🏆 Top Teams by Mentions

| Rank | Team | Total Mentions |
|---|---|---|
"""
    for i, (team, cnt_val) in enumerate(top_teams, 1):
        md += f"| {i} | {team} | {cnt_val:,} |\n"

    md += """
---

## 📋 Team Sentiment Breakdown

| Team | Positive | Neutral | Negative | Total |
|---|---|---|---|---|
"""
    for team, s in sorted(team_table.items(),
                           key=lambda x: sum(x[1].values()), reverse=True):
        tot = sum(s.values())
        md += (f"| {team} | {s['Positive']:,} | {s['Neutral']:,} "
               f"| {s['Negative']:,} | {tot:,} |\n")

    md += """
---

## ⚡ Match Events Summary

| Event Type | Count |
|---|---|
"""
    for ev, count in sorted(event_cnt.items(), key=lambda x: x[1], reverse=True):
        md += f"| {ev.title()} | {count} |\n"

    if not event_cnt:
        md += "| — | No events logged |\n"

    md += f"""
---

## 🔍 Notes

- Sentiment classified via **cardiffnlp/twitter-roberta-base-sentiment** (RoBERTa)
- Hinglish pre-screen applied before model inference (Phase 2C)
- Match events sourced from manual logger and/or CricAPI poller (Phase 2B)
- Data window: last 24 hours

---
*IPL Sentiment Dashboard — [github.com/Pratik7897/IPL_Data_Visualisation](https://github.com/Pratik7897/IPL_Data_Visualisation)*
"""

    logger.info(f"[export] Summary MD — {total} tweets, {len(events)} events")
    headers = {
        "Content-Disposition": f'attachment; filename="{filename}"',
        "Content-Type": "text/markdown; charset=utf-8",
    }
    return Response(md, headers=headers, mimetype="text/markdown")


def export_health() -> Response:
    """Simple JSON health-check — confirms DB is reachable."""
    from modules.database import fetch_recent_tweets_with_sentiment
    try:
        rows = fetch_recent_tweets_with_sentiment(limit=1)
        db_ok = True
    except Exception as e:
        rows = []
        db_ok = False
        logger.warning(f"[export/health] DB check failed: {e}")

    payload = {
        "status":       "ok" if db_ok else "degraded",
        "db":           "reachable" if db_ok else "error",
        "tweets_in_db": len(rows),
        "timestamp":    datetime.now(timezone.utc).isoformat(),
    }
    status_code = 200 if db_ok else 503
    return Response(json.dumps(payload, indent=2),
                    status=status_code,
                    mimetype="application/json")


# ── Route registration ────────────────────────────────────────────────────────
def register_export_routes(flask_server):
    """
    Call this once after `server = app.server` to attach all export
    endpoints to the Flask application.

    Usage in app.py:
        from modules.exporter import register_export_routes
        register_export_routes(server)
    """
    flask_server.add_url_rule("/export/csv",     "export_csv",     export_tweets_csv)
    flask_server.add_url_rule("/export/json",    "export_json",    export_tweets_json)
    flask_server.add_url_rule("/export/events",  "export_events",  export_events_csv)
    flask_server.add_url_rule("/export/summary", "export_summary", export_summary_md)
    flask_server.add_url_rule("/export/health",  "export_health",  export_health)
    logger.info("[exporter] Export routes registered: /export/{csv,json,events,summary,health}")
