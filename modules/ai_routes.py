"""
modules/ai_routes.py
Phase 4+5 — Flask route registration for all AI/ML endpoints.

Phase 4 (8 routes): predict-match, match-momentum, expected-score,
    generate-summary, player-rankings, top-trending-player,
    player-sentiment-history, ai-health

Phase 5 (8 routes): prediction-explanation, team-comparison,
    stream-comments, live-sentiment-feed, replay-match-list,
    replay-match-start, replay-match-next, replay-match-reset
"""
import json
import logging
import time
from datetime import datetime, timezone
from functools import wraps

from flask import request, Response

logger = logging.getLogger(__name__)


# ── Response helpers ──────────────────────────────────────────────────────────
def _ok(data: dict, status: int = 200) -> Response:
    return Response(
        json.dumps(data, default=str, ensure_ascii=False),
        status=status, mimetype="application/json",
    )

def _err(msg: str, status: int = 400) -> Response:
    return _ok({"error": msg, "timestamp": datetime.now(timezone.utc).isoformat()}, status)

def _timed(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        t0 = time.perf_counter()
        resp = fn(*args, **kwargs)
        elapsed = round((time.perf_counter() - t0) * 1000, 1)
        try:
            data = json.loads(resp.get_data())
            data["elapsed_ms"] = elapsed
            return Response(json.dumps(data, default=str),
                            status=resp.status_code, mimetype="application/json")
        except Exception:
            return resp
    return wrapper


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 4 HANDLERS
# ══════════════════════════════════════════════════════════════════════════════
def predict_match_handler():
    try:
        payload = request.get_json(force=True, silent=True) or {}
        from modules.ai.predictor import predict_match
        return _ok(predict_match(payload))
    except Exception as e:
        logger.exception("[ai_routes] /predict-match")
        return _err(str(e), 500)

def match_momentum_handler():
    try:
        payload = request.get_json(force=True, silent=True) or {}
        from modules.ai.predictor import get_match_momentum
        return _ok(get_match_momentum(payload))
    except Exception as e:
        logger.exception("[ai_routes] /match-momentum")
        return _err(str(e), 500)

def expected_score_handler():
    try:
        payload = request.get_json(force=True, silent=True) or {}
        from modules.ai.predictor import expected_score
        return _ok(expected_score(payload))
    except Exception as e:
        logger.exception("[ai_routes] /expected-score")
        return _err(str(e), 500)

def generate_summary_handler():
    try:
        from modules.ai.summarizer import generate_summary
        return _ok(generate_summary())
    except Exception as e:
        logger.exception("[ai_routes] /generate-summary")
        return _err(str(e), 500)

def player_rankings_handler():
    try:
        limit = int(request.args.get("limit", 20))
        from modules.ai.player_index import compute_player_rankings
        rankings = compute_player_rankings()[:limit]
        return _ok({"count": len(rankings), "rankings": rankings})
    except Exception as e:
        logger.exception("[ai_routes] /player-rankings")
        return _err(str(e), 500)

def top_trending_handler():
    try:
        from modules.ai.player_index import get_top_trending
        return _ok(get_top_trending())
    except Exception as e:
        logger.exception("[ai_routes] /top-trending-player")
        return _err(str(e), 500)

def player_history_handler():
    player = request.args.get("player", "Virat Kohli")
    try:
        from modules.ai.player_index import get_player_sentiment_history
        return _ok(get_player_sentiment_history(player))
    except Exception as e:
        logger.exception("[ai_routes] /player-sentiment-history")
        return _err(str(e), 500)

def ai_health_handler():
    checks = {}
    _mods = [
        ("predictor",    "modules.ai.predictor",    "_get_model"),
        ("summarizer",   "modules.ai.summarizer",   "_provider"),
        ("player_index", "modules.ai.player_index", "PLAYER_REGISTRY"),
    ]
    for key, mod, attr in _mods:
        try:
            import importlib
            m = importlib.import_module(mod)
            val = getattr(m, attr)
            if callable(val): val()
            checks[key] = f"ok" if key != "player_index" else f"ok ({len(val)} players)"
        except Exception as e:
            checks[key] = f"error: {e}"
    for key, mod, fn in [
        ("stream_replay", "modules.ai.stream_replay", "get_stream_items"),
        ("match_replay",  "modules.ai.match_replay",  "get_match_list"),
        ("explainer",     "modules.ai.explainer",     "generate_explanation"),
    ]:
        try:
            import importlib
            m = importlib.import_module(mod)
            f = getattr(m, fn)
            r = f(1) if key == "stream_replay" else (f() if key == "match_replay" else None)
            checks[key] = f"ok ({len(r)} matches)" if key == "match_replay" else "ok"
        except Exception as e:
            checks[key] = f"error: {e}"

    all_ok = all("ok" in v for v in checks.values())
    return _ok({
        "status": "ok" if all_ok else "degraded",
        "checks": checks,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }, 200 if all_ok else 503)


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 5 HANDLERS
# ══════════════════════════════════════════════════════════════════════════════

def prediction_explanation_handler():
    """POST /prediction-explanation — prediction + explainability report"""
    try:
        payload = request.get_json(force=True, silent=True) or {}
        from modules.ai.predictor import predict_match
        from modules.ai.explainer import generate_explanation
        pred   = predict_match(payload)
        report = generate_explanation(payload, pred)
        return _ok({**pred, "explanation": report})
    except Exception as e:
        logger.exception("[ai_routes] /prediction-explanation")
        return _err(str(e), 500)

def team_comparison_handler():
    """GET /team-comparison?team_a=MI&team_b=CSK&venue=Wankhede"""
    try:
        team_a = request.args.get("team_a", "MI").upper()
        team_b = request.args.get("team_b", "CSK").upper()
        venue  = request.args.get("venue", "Neutral")
        from modules.ai.predictor import predict_match
        from modules.ai.explainer import (
            HISTORICAL_WIN_RATES, VENUE_STATS,
            TEAM_FULL_NAMES, _get_live_sentiment,
        )
        pred  = predict_match({"team": team_a, "opponent": team_b,
                                "venue": venue, "toss_won": 0})
        sent  = _get_live_sentiment(team_a, team_b)
        vs    = VENUE_STATS.get(venue, {})
        return _ok({
            "team_a": {
                "code": team_a, "name": TEAM_FULL_NAMES.get(team_a, team_a),
                "win_prob": pred["team_win_prob"],
                "historical_wr": HISTORICAL_WIN_RATES.get(team_a, 0.5),
                "sentiment_pos": round(sent["pos_a"] * 100, 1),
                "venue_wr": vs.get(team_a, 0.5),
                "tweet_volume": sent["vol_a"],
            },
            "team_b": {
                "code": team_b, "name": TEAM_FULL_NAMES.get(team_b, team_b),
                "win_prob": pred["opponent_win_prob"],
                "historical_wr": HISTORICAL_WIN_RATES.get(team_b, 0.5),
                "sentiment_pos": round(sent["pos_b"] * 100, 1),
                "venue_wr": vs.get(team_b, 0.5),
                "tweet_volume": sent["vol_b"],
            },
            "venue": venue, "momentum": pred["momentum"],
            "confidence": pred["confidence"],
        })
    except Exception as e:
        logger.exception("[ai_routes] /team-comparison")
        return _err(str(e), 500)

def stream_comments_handler():
    """GET /stream-comments?limit=20"""
    try:
        limit = int(request.args.get("limit", 20))
        from modules.ai.stream_replay import get_stream_feed_api
        return _ok(get_stream_feed_api(limit))
    except Exception as e:
        logger.exception("[ai_routes] /stream-comments")
        return _err(str(e), 500)

def live_sentiment_feed_handler():
    """GET /live-sentiment-feed — merged replay + live tweets"""
    try:
        from modules.ai.stream_replay import get_stream_items
        from modules.database import fetch_recent_tweets_with_sentiment
        real   = fetch_recent_tweets_with_sentiment(limit=5)
        stream = get_stream_items(10)
        combined = []
        for tw in real:
            combined.append({
                "text": tw["text"], "source": "Live Stream",
                "sentiment": tw.get("label", "Neutral") or "Neutral",
                "timestamp": str(tw.get("created_at", ""))[-8:],
                "engagement": (tw.get("like_count") or 0) + (tw.get("retweet_count") or 0),
                "source_color": "#00E5A0",
            })
        combined.extend(stream)
        return _ok({"count": len(combined), "feed": combined[:20]})
    except Exception as e:
        logger.exception("[ai_routes] /live-sentiment-feed")
        return _err(str(e), 500)

def replay_match_list_handler():
    """GET /replay-match-list"""
    try:
        from modules.ai.match_replay import get_match_list
        return _ok({"matches": get_match_list()})
    except Exception as e:
        logger.exception("[ai_routes] /replay-match-list")
        return _err(str(e), 500)

def replay_match_start_handler():
    """POST /replay-match-start  Body: { match_id, speed }"""
    try:
        payload  = request.get_json(force=True, silent=True) or {}
        match_id = payload.get("match_id", "csk_vs_mi_2019_final")
        speed    = float(payload.get("speed", 1.0))
        from modules.ai.match_replay import start_replay
        return _ok(start_replay(match_id, speed))
    except Exception as e:
        logger.exception("[ai_routes] /replay-match-start")
        return _err(str(e), 500)

def replay_match_next_handler():
    """GET /replay-match-next"""
    try:
        from modules.ai.match_replay import get_replay_next_moment
        moment = get_replay_next_moment()
        if moment is None:
            return _ok({"status": "idle", "message": "No replay active."})
        return _ok(moment)
    except Exception as e:
        logger.exception("[ai_routes] /replay-match-next")
        return _err(str(e), 500)

def replay_match_reset_handler():
    """POST /replay-match-reset"""
    try:
        from modules.ai.match_replay import reset_replay
        return _ok(reset_replay())
    except Exception as e:
        logger.exception("[ai_routes] /replay-match-reset")
        return _err(str(e), 500)


# ── Registration ──────────────────────────────────────────────────────────────
def register_ai_routes(flask_server):
    routes = [
        # Phase 4
        ("/predict-match",            "predict_match",   predict_match_handler,          ["POST"]),
        ("/match-momentum",           "match_momentum",  match_momentum_handler,         ["POST"]),
        ("/expected-score",           "expected_score",  expected_score_handler,         ["POST"]),
        ("/generate-summary",         "gen_summary",     generate_summary_handler,       ["GET"]),
        ("/player-rankings",          "player_ranks",    player_rankings_handler,        ["GET"]),
        ("/top-trending-player",      "top_trend",       top_trending_handler,           ["GET"]),
        ("/player-sentiment-history", "player_hist",     player_history_handler,         ["GET"]),
        ("/ai-health",                "ai_health",       ai_health_handler,              ["GET"]),
        # Phase 5
        ("/prediction-explanation",   "pred_explain",    prediction_explanation_handler, ["POST"]),
        ("/team-comparison",          "team_compare",    team_comparison_handler,        ["GET"]),
        ("/stream-comments",          "stream_comments", stream_comments_handler,        ["GET"]),
        ("/live-sentiment-feed",      "live_feed",       live_sentiment_feed_handler,    ["GET"]),
        ("/replay-match-list",        "replay_list",     replay_match_list_handler,      ["GET"]),
        ("/replay-match-start",       "replay_start",    replay_match_start_handler,     ["POST"]),
        ("/replay-match-next",        "replay_next",     replay_match_next_handler,      ["GET"]),
        ("/replay-match-reset",       "replay_reset",    replay_match_reset_handler,     ["POST"]),
    ]
    for path, endpoint, handler, methods in routes:
        flask_server.add_url_rule(path, endpoint, _timed(handler), methods=methods)
    logger.info("[ai_routes] Registered %d endpoints (Phase 4+5)", len(routes))
