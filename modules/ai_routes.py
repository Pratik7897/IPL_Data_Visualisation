"""
modules/ai_routes.py
Phase 4 — Flask route registration for all AI/ML endpoints.

Registers 8 new routes on the existing Flask server (app.server).
No existing routes are modified.

Routes added:
    POST /predict-match         → Feature 1: win probability
    POST /match-momentum        → Feature 1: momentum curve
    POST /expected-score        → Feature 1: score projection
    GET  /generate-summary      → Feature 2: AI match summary
    GET  /player-rankings       → Feature 3: full leaderboard
    GET  /top-trending-player   → Feature 3: single top trending player
    GET  /player-sentiment-history?player=<name>  → Feature 3: player deep dive
    GET  /ai-health             → combined health-check for AI module
"""
import json
import logging
import time
from datetime import datetime, timezone
from functools import wraps

from flask import request, Response, jsonify

logger = logging.getLogger(__name__)


# ── Response helpers ──────────────────────────────────────────────────────────
def _ok(data: dict, status: int = 200) -> Response:
    return Response(
        json.dumps(data, default=str, ensure_ascii=False),
        status=status,
        mimetype="application/json",
    )


def _err(msg: str, status: int = 400) -> Response:
    return _ok({"error": msg, "timestamp": datetime.now(timezone.utc).isoformat()}, status)


def _timed(fn):
    """Decorator: add elapsed_ms to every response."""
    @wraps(fn)
    def wrapper(*args, **kwargs):
        t0 = time.perf_counter()
        resp = fn(*args, **kwargs)
        elapsed = round((time.perf_counter() - t0) * 1000, 1)
        try:
            data = json.loads(resp.get_data())
            data["elapsed_ms"] = elapsed
            return Response(json.dumps(data, default=str),
                            status=resp.status_code,
                            mimetype="application/json")
        except Exception:
            return resp
    return wrapper


# ── Route handlers ────────────────────────────────────────────────────────────

# ── Feature 1: Match Prediction ──────────────────────────────────────────────
def predict_match_handler():
    """POST /predict-match
    Body: { team, opponent, current_score, wickets_fallen, overs_completed,
            target, toss_won, venue_home, ... }
    """
    try:
        payload = request.get_json(force=True, silent=True) or {}
        from modules.ai.predictor import predict_match
        result = predict_match(payload)
        return _ok(result)
    except Exception as e:
        logger.exception("[ai_routes] /predict-match error")
        return _err(str(e), 500)


def match_momentum_handler():
    """POST /match-momentum"""
    try:
        payload = request.get_json(force=True, silent=True) or {}
        from modules.ai.predictor import get_match_momentum
        result = get_match_momentum(payload)
        return _ok(result)
    except Exception as e:
        logger.exception("[ai_routes] /match-momentum error")
        return _err(str(e), 500)


def expected_score_handler():
    """POST /expected-score"""
    try:
        payload = request.get_json(force=True, silent=True) or {}
        from modules.ai.predictor import expected_score
        result = expected_score(payload)
        return _ok(result)
    except Exception as e:
        logger.exception("[ai_routes] /expected-score error")
        return _err(str(e), 500)


# ── Feature 2: AI Summary ────────────────────────────────────────────────────
def generate_summary_handler():
    """GET /generate-summary"""
    try:
        from modules.ai.summarizer import generate_summary
        result = generate_summary()
        return _ok(result)
    except Exception as e:
        logger.exception("[ai_routes] /generate-summary error")
        return _err(str(e), 500)


# ── Feature 3: Player Index ──────────────────────────────────────────────────
def player_rankings_handler():
    """GET /player-rankings?limit=10"""
    try:
        limit = int(request.args.get("limit", 20))
        from modules.ai.player_index import compute_player_rankings
        rankings = compute_player_rankings()[:limit]
        return _ok({"count": len(rankings), "rankings": rankings})
    except Exception as e:
        logger.exception("[ai_routes] /player-rankings error")
        return _err(str(e), 500)


def top_trending_handler():
    """GET /top-trending-player"""
    try:
        from modules.ai.player_index import get_top_trending
        result = get_top_trending()
        return _ok(result)
    except Exception as e:
        logger.exception("[ai_routes] /top-trending-player error")
        return _err(str(e), 500)


def player_history_handler():
    """GET /player-sentiment-history?player=Virat+Kohli"""
    player = request.args.get("player", "Virat Kohli")
    try:
        from modules.ai.player_index import get_player_sentiment_history
        result = get_player_sentiment_history(player)
        return _ok(result)
    except Exception as e:
        logger.exception("[ai_routes] /player-sentiment-history error")
        return _err(str(e), 500)


def ai_health_handler():
    """GET /ai-health — verify all AI modules load correctly."""
    checks = {}
    try:
        from modules.ai.predictor import _get_model
        _get_model()
        checks["predictor"] = "ok"
    except Exception as e:
        checks["predictor"] = f"error: {e}"

    try:
        from modules.ai.summarizer import _provider
        checks["summarizer"] = f"ok (provider={_provider()})"
    except Exception as e:
        checks["summarizer"] = f"error: {e}"

    try:
        from modules.ai.player_index import PLAYER_REGISTRY
        checks["player_index"] = f"ok ({len(PLAYER_REGISTRY)} players registered)"
    except Exception as e:
        checks["player_index"] = f"error: {e}"

    all_ok = all("ok" in v for v in checks.values())
    return _ok({
        "status": "ok" if all_ok else "degraded",
        "checks": checks,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }, 200 if all_ok else 503)


# ── Registration entry point ──────────────────────────────────────────────────
def register_ai_routes(flask_server):
    """
    Attach all AI/ML endpoints to the Flask app.
    Call once after `server = app.server` in app.py.

        from modules.ai_routes import register_ai_routes
        register_ai_routes(server)
    """
    routes = [
        ("/predict-match",              "predict_match",    predict_match_handler,    ["POST"]),
        ("/match-momentum",             "match_momentum",   match_momentum_handler,   ["POST"]),
        ("/expected-score",             "expected_score",   expected_score_handler,   ["POST"]),
        ("/generate-summary",           "generate_summary", generate_summary_handler, ["GET"]),
        ("/player-rankings",            "player_rankings",  player_rankings_handler,  ["GET"]),
        ("/top-trending-player",        "top_trending",     top_trending_handler,     ["GET"]),
        ("/player-sentiment-history",   "player_history",   player_history_handler,   ["GET"]),
        ("/ai-health",                  "ai_health",        ai_health_handler,        ["GET"]),
    ]
    for path, endpoint, handler, methods in routes:
        flask_server.add_url_rule(path, endpoint, _timed(handler), methods=methods)

    logger.info(
        "[ai_routes] Registered %d AI endpoints: %s",
        len(routes),
        ", ".join(r[0] for r in routes),
    )
