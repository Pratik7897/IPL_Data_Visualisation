"""
modules/ai/predictor.py
Phase 4 — Feature 1: Match Outcome Prediction Engine.

Exposes three public functions consumed by ai_routes.py:

    predict_match(payload)       → win probability, confidence, momentum
    get_match_momentum(payload)  → momentum time-series data
    expected_score(payload)      → expected final score projection

ML backend:
    Primary   → RandomForestClassifier (sklearn, always available)
    Optional  → XGBoostClassifier (if xgboost installed, auto-selected)

The model is trained on first call from synthetic data and cached in-process.
It never modifies the existing sentiment pipeline.
"""
import os
import math
import logging
import threading
from collections import defaultdict
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────
TEAMS = ["MI", "CSK", "RCB", "KKR", "SRH", "RR", "DC", "PBKS", "GT", "LSG"]

HISTORICAL_WIN_RATES = {
    "MI": 0.58, "CSK": 0.60, "RCB": 0.48, "KKR": 0.52,
    "SRH": 0.50, "RR":  0.49, "DC":  0.47, "PBKS": 0.44,
    "GT": 0.62, "LSG": 0.51,
}

FEATURE_COLS = [
    "current_score", "wickets_fallen", "overs_completed",
    "current_run_rate", "required_run_rate", "overs_remaining",
    "target", "runs_needed", "toss_won", "venue_home",
    "historical_win_rate", "team_sentiment_score",
    "opponent_sentiment_score", "tweet_volume_ratio", "momentum_score",
]

# ── Model singleton ────────────────────────────────────────────────────────────
_model      = None
_model_lock = threading.Lock()


def _get_model():
    global _model
    if _model is not None:
        return _model
    with _model_lock:
        if _model is not None:
            return _model
        _model = _train_model()
    return _model


def _train_model():
    """Train RandomForest (or XGBoost) on synthetic data."""
    logger.info("[predictor] Training prediction model …")

    from modules.ai.training.generate_sample_data import generate_dataset
    rows = generate_dataset(8000)

    import pandas as pd
    df = pd.DataFrame(rows)
    X = df[FEATURE_COLS].values
    y = df["won"].values

    # Prefer XGBoost if installed
    try:
        from xgboost import XGBClassifier
        clf = XGBClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            use_label_encoder=False, eval_metric="logloss",
            random_state=42, n_jobs=-1,
        )
        backend = "XGBoost"
    except ImportError:
        from sklearn.ensemble import RandomForestClassifier
        clf = RandomForestClassifier(
            n_estimators=400, max_depth=10, min_samples_leaf=5,
            random_state=42, n_jobs=-1,
        )
        backend = "RandomForest"

    clf.fit(X, y)
    logger.info(f"[predictor] Model ready — backend={backend}, samples={len(rows)}")
    return clf


# ── Feature extraction ────────────────────────────────────────────────────────
def _extract_features(p: dict) -> np.ndarray:
    """Convert API payload → feature vector aligned with FEATURE_COLS."""
    team = p.get("team", "").upper()
    overs_done = float(p.get("overs_completed", 10))
    overs_rem  = 20.0 - overs_done
    score      = float(p.get("current_score", 80))
    wickets    = float(p.get("wickets_fallen", 3))
    target     = float(p.get("target", 165))
    runs_req   = target - score
    crr        = score / overs_done if overs_done > 0 else 0
    rrr        = runs_req / overs_rem if overs_rem > 0 else 99.9

    t_sent  = float(p.get("team_sentiment_score", 0.0))
    o_sent  = float(p.get("opponent_sentiment_score", 0.0))
    vol_r   = float(p.get("tweet_volume_ratio", 0.5))
    mom     = float(p.get("momentum_score", 0.0))
    toss    = float(p.get("toss_won", 0))
    v_home  = float(p.get("venue_home", 0))
    hist_wr = HISTORICAL_WIN_RATES.get(team, 0.50)

    return np.array([[
        score, wickets, overs_done, crr, rrr, overs_rem,
        target, runs_req, toss, v_home, hist_wr,
        t_sent, o_sent, vol_r, mom,
    ]])


def _derive_from_live_data(team: str, opponent: str) -> dict:
    """Pull sentiment signals from the live database — read-only."""
    try:
        from modules.database import fetch_team_sentiment_counts, get_tweet_volume_per_minute
        rows = fetch_team_sentiment_counts()

        counts: dict[str, dict] = defaultdict(lambda: {"Positive": 0, "Negative": 0, "Neutral": 0})
        for r in rows:
            t = (r.get("team_mention") or "").upper()
            counts[t][r["label"]] = r["count"]

        def sentiment_score(t):
            c = counts[t]
            total = sum(c.values()) or 1
            return (c["Positive"] - c["Negative"]) / total

        t_tot = sum(counts.get(team, {}).values()) or 1
        o_tot = sum(counts.get(opponent, {}).values()) or 1
        total_vol = t_tot + o_tot or 1

        return {
            "team_sentiment_score":      round(sentiment_score(team), 3),
            "opponent_sentiment_score":  round(sentiment_score(opponent), 3),
            "tweet_volume_ratio":        round(t_tot / total_vol, 3),
        }
    except Exception as e:
        logger.warning(f"[predictor] Could not fetch live sentiment: {e}")
        return {"team_sentiment_score": 0.0,
                "opponent_sentiment_score": 0.0,
                "tweet_volume_ratio": 0.5}


# ── Public API functions ───────────────────────────────────────────────────────
def predict_match(payload: dict) -> dict:
    """
    Predict win probability for a match snapshot.

    Required payload keys: team, opponent
    Optional: current_score, wickets_fallen, overs_completed, target,
              toss_won, venue_home, momentum_score

    Returns:
        {
          team_win_prob: float,       # 0-100
          opponent_win_prob: float,
          confidence: str,            # Low / Medium / High
          momentum: str,              # Building / Stable / Fading
          model_backend: str,
        }
    """
    model = _get_model()

    team     = (payload.get("team", "MI")).upper()
    opponent = (payload.get("opponent", "CSK")).upper()

    # Merge live sentiment signals with payload
    live = _derive_from_live_data(team, opponent)
    merged = {**live, **payload}  # payload overrides live

    # Momentum heuristic from score/rrr gap
    overs_done = float(merged.get("overs_completed", 10))
    overs_rem  = 20.0 - overs_done
    score      = float(merged.get("current_score", 80))
    target     = float(merged.get("target", 165))
    runs_req   = target - score
    crr        = score / overs_done if overs_done > 0 else 0
    rrr        = runs_req / overs_rem if overs_rem > 0 else 99.9
    wickets    = float(merged.get("wickets_fallen", 3))

    mom_raw = ((crr - rrr) / 6) + (1 - wickets / 10) + live["team_sentiment_score"] * 0.3
    mom_raw = max(-1, min(1, mom_raw))
    merged["momentum_score"] = mom_raw

    X = _extract_features(merged)
    proba = model.predict_proba(X)[0]
    team_win = round(float(proba[1]) * 100, 1)
    opp_win  = round(100 - team_win, 1)

    # Confidence from max probability distance from 50%
    gap = abs(team_win - 50)
    if gap < 10:
        confidence = "Low"
    elif gap < 25:
        confidence = "Medium"
    else:
        confidence = "High"

    if mom_raw > 0.2:
        momentum = "Building 📈"
    elif mom_raw < -0.2:
        momentum = "Fading 📉"
    else:
        momentum = "Stable ➡️"

    backend = type(model).__name__

    return {
        "team":             team,
        "opponent":         opponent,
        "team_win_prob":    team_win,
        "opponent_win_prob": opp_win,
        "confidence":       confidence,
        "momentum":         momentum,
        "momentum_raw":     round(mom_raw, 3),
        "model_backend":    backend,
    }


def get_match_momentum(payload: dict) -> dict:
    """
    Return a simulated momentum curve for the current match state.
    Useful for plotting momentum shifts over time on the frontend.

    Returns list of {over: int, team_advantage: float} points.
    """
    model  = _get_model()
    team   = (payload.get("team", "MI")).upper()
    opp    = (payload.get("opponent", "CSK")).upper()
    live   = _derive_from_live_data(team, opp)
    target = float(payload.get("target", 165))

    points = []
    for over in range(1, 21):
        # Build a plausible snapshot at this over
        frac   = over / 20
        score  = int(target * 0.9 * frac * (0.9 + 0.2 * (over / 20) ** 0.5))
        wkts   = min(int(frac * 7), 10)
        snap   = {
            "team": team, "opponent": opp,
            "current_score": score, "wickets_fallen": wkts,
            "overs_completed": over, "target": target,
            "toss_won": payload.get("toss_won", 0),
            "venue_home": payload.get("venue_home", 0),
            **live,
        }
        result = predict_match(snap)
        points.append({"over": over, "team_win_prob": result["team_win_prob"]})

    return {
        "team": team,
        "opponent": opp,
        "target": int(target),
        "momentum_curve": points,
    }


def expected_score(payload: dict) -> dict:
    """
    Project the expected final score based on current run rate + wickets.

    Returns:
        {
          expected_score: int,
          range_low: int,
          range_high: int,
          run_rate_needed: float,
        }
    """
    overs_done  = float(payload.get("overs_completed", 10))
    overs_rem   = 20.0 - overs_done
    score       = float(payload.get("current_score", 80))
    wickets     = float(payload.get("wickets_fallen", 3))

    crr = score / overs_done if overs_done > 0 else 0

    # Adjust expected rate for wickets in hand
    wickets_in_hand = 10 - wickets
    rate_multiplier = 0.7 + 0.35 * (wickets_in_hand / 10)
    projected_rate  = crr * rate_multiplier

    # Death-over boost (last 4 overs)
    if overs_rem <= 4:
        projected_rate *= 1.15

    exp   = int(score + projected_rate * overs_rem)
    lo    = int(exp * 0.88)
    hi    = int(exp * 1.12)

    target = float(payload.get("target", exp + 10))
    rrr    = (target - score) / overs_rem if overs_rem > 0 else 99.9

    return {
        "expected_score":  exp,
        "range_low":       lo,
        "range_high":      hi,
        "current_run_rate": round(crr, 2),
        "run_rate_needed":  round(rrr, 2),
        "wickets_in_hand":  int(wickets_in_hand),
        "overs_remaining":  round(overs_rem, 1),
    }
