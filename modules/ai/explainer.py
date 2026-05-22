"""
modules/ai/explainer.py
Phase 5 — Feature 3: Prediction Explainability Engine.

Given a prediction payload + result, generates:
  - bullet-point "WHY THIS PREDICTION?" reasons
  - confidence tier (Low / Medium / High)
  - data quality score
  - confidence percentage

Zero side effects — purely functional, stateless.
"""
import logging
from collections import defaultdict

logger = logging.getLogger(__name__)

# ── Historical venue win rates (batting first) ────────────────────────────────
VENUE_STATS = {
    "Wankhede":          {"MI": 0.72, "CSK": 0.55, "batting_first_win": 0.46},
    "Chepauk":           {"CSK": 0.75, "MI": 0.48,  "batting_first_win": 0.52},
    "Chinnaswamy":       {"RCB": 0.68, "batting_first_win": 0.44},
    "Eden Gardens":      {"KKR": 0.71, "batting_first_win": 0.50},
    "Narendra Modi":     {"GT": 0.74,  "batting_first_win": 0.48},
    "Kotla":             {"DC": 0.65,  "batting_first_win": 0.51},
    "Sawai Mansingh":    {"RR": 0.69,  "batting_first_win": 0.47},
    "DY Patil":          {"batting_first_win": 0.49},
    "Brabourne":         {"batting_first_win": 0.50},
    "Hyderabad":         {"SRH": 0.70, "batting_first_win": 0.50},
    "Neutral":           {"batting_first_win": 0.49},
}

HISTORICAL_WIN_RATES = {
    "MI": 0.58, "CSK": 0.60, "RCB": 0.48, "KKR": 0.52,
    "SRH": 0.50, "RR":  0.49, "DC":  0.47, "PBKS": 0.44,
    "GT": 0.62, "LSG": 0.51,
}

TEAM_FULL_NAMES = {
    "MI": "Mumbai Indians", "CSK": "Chennai Super Kings",
    "RCB": "Royal Challengers Bengaluru", "KKR": "Kolkata Knight Riders",
    "SRH": "Sunrisers Hyderabad", "RR": "Rajasthan Royals",
    "DC": "Delhi Capitals", "PBKS": "Punjab Kings",
    "GT": "Gujarat Titans", "LSG": "Lucknow Super Giants",
}


def _get_live_sentiment(team_a: str, team_b: str) -> dict:
    """Pull sentiment signals from live DB — read-only."""
    try:
        from modules.database import fetch_team_sentiment_counts
        rows = fetch_team_sentiment_counts()
        counts: dict = defaultdict(lambda: {"Positive": 0, "Negative": 0, "Neutral": 0})
        for r in rows:
            t = (r.get("team_mention") or "").upper()
            counts[t][r["label"]] = r["count"]

        def ratio(t):
            c = counts[t]
            tot = sum(c.values()) or 1
            pos = c["Positive"] / tot
            neg = c["Negative"] / tot
            return pos, neg, tot

        pos_a, neg_a, vol_a = ratio(team_a)
        pos_b, neg_b, vol_b = ratio(team_b)

        return {
            "pos_a": pos_a, "neg_a": neg_a, "vol_a": int(vol_a),
            "pos_b": pos_b, "neg_b": neg_b, "vol_b": int(vol_b),
            "data_quality": "live" if (vol_a + vol_b) > 50 else "sparse",
        }
    except Exception as e:
        logger.warning(f"[explainer] sentiment fetch failed: {e}")
        return {"pos_a": 0.5, "neg_a": 0.25, "vol_a": 0,
                "pos_b": 0.5, "neg_b": 0.25, "vol_b": 0,
                "data_quality": "unavailable"}


def generate_explanation(payload: dict, prediction: dict) -> dict:
    """
    Generate a structured explainability report.

    Args:
        payload:    The original prediction request payload.
        prediction: The result from predictor.predict_match().

    Returns:
        {
          team:              str,
          opponent:          str,
          win_prob:          float,
          confidence:        str,
          confidence_pct:    float,
          data_quality:      str,
          reasons:           list[str],   # bullet-point reasons
          counter_reasons:   list[str],   # risks / counter-arguments
          verdict:           str,         # one-line summary sentence
        }
    """
    team_a   = prediction.get("team", payload.get("team", "MI")).upper()
    team_b   = prediction.get("opponent", payload.get("opponent", "CSK")).upper()
    prob_a   = prediction.get("team_win_prob", 50.0)
    prob_b   = prediction.get("opponent_win_prob", 50.0)
    momentum = prediction.get("momentum", "Stable ➡️")
    conf     = prediction.get("confidence", "Medium")
    venue    = payload.get("venue", "Neutral")
    toss     = int(payload.get("toss_won", 0))

    # Live sentiment
    sent = _get_live_sentiment(team_a, team_b)
    pos_a, pos_b = sent["pos_a"], sent["pos_b"]
    vol_a, vol_b = sent["vol_a"], sent["vol_b"]

    # Historical win rates
    hist_a = HISTORICAL_WIN_RATES.get(team_a, 0.50)
    hist_b = HISTORICAL_WIN_RATES.get(team_b, 0.50)

    # Venue advantage
    venue_stats  = VENUE_STATS.get(venue, VENUE_STATS["Neutral"])
    venue_fav_a  = venue_stats.get(team_a, 0.5)
    venue_fav_b  = venue_stats.get(team_b, 0.5)

    name_a = TEAM_FULL_NAMES.get(team_a, team_a)
    name_b = TEAM_FULL_NAMES.get(team_b, team_b)

    winner  = team_a if prob_a >= prob_b else team_b
    loser   = team_b if prob_a >= prob_b else team_a
    w_prob  = max(prob_a, prob_b)
    w_hist  = hist_a if prob_a >= prob_b else hist_b
    l_hist  = hist_b if prob_a >= prob_b else hist_a
    w_pos   = pos_a  if prob_a >= prob_b else pos_b
    l_pos   = pos_b  if prob_a >= prob_b else pos_a
    w_vol   = vol_a  if prob_a >= prob_b else vol_b
    l_vol   = vol_b  if prob_a >= prob_b else vol_a
    w_vfav  = venue_fav_a if prob_a >= prob_b else venue_fav_b
    w_name  = TEAM_FULL_NAMES.get(winner, winner)
    l_name  = TEAM_FULL_NAMES.get(loser, loser)

    # ── Build REASONS ─────────────────────────────────────────────────────────
    reasons = []

    # Sentiment reason
    sent_diff = abs(w_pos - l_pos) * 100
    if sent_diff > 5:
        direction = "higher" if w_pos > l_pos else "lower"
        reasons.append(
            f"📊 {winner} fan sentiment is {sent_diff:.0f}% {direction} positive "
            f"({w_pos*100:.0f}% vs {l_pos*100:.0f}%)"
        )

    # Historical form
    hist_diff = abs(w_hist - l_hist) * 100
    if hist_diff > 2:
        reasons.append(
            f"📈 Stronger historical IPL win rate: {winner} {w_hist*100:.0f}% "
            f"vs {loser} {l_hist*100:.0f}%"
        )

    # Momentum
    if "Building" in momentum or "↑" in momentum:
        reasons.append(f"⚡ {winner} momentum is building — pressure shifting on the field")
    elif "Stable" in momentum:
        reasons.append(f"➡️ Match momentum is currently stable and balanced")

    # Venue advantage
    if w_vfav > 0.60:
        reasons.append(
            f"🏟️ {venue} historically favours {winner} "
            f"(home win rate: {w_vfav*100:.0f}%)"
        )

    # Toss
    if toss == 1:
        reasons.append(f"🪙 {winner} won the toss — tactical advantage secured")

    # Volume
    if w_vol > l_vol and (w_vol + l_vol) > 20:
        reasons.append(
            f"🔊 {w_vol} fan mentions vs {l_vol} — "
            f"{winner} dominates social conversation"
        )

    # Win probability gap
    gap = abs(prob_a - prob_b)
    if gap > 25:
        reasons.append(f"🎯 High-confidence signal: {w_prob:.0f}% win probability computed from {len(reasons)+1} data inputs")

    if not reasons:
        reasons.append(f"🤖 Model predicts {winner} based on composite IPL performance metrics")

    # ── COUNTER-REASONS (risks) ────────────────────────────────────────────────
    counter = []
    if l_hist > 0.55:
        counter.append(f"⚠️ {loser} has a strong IPL track record ({l_hist*100:.0f}% historical win rate)")
    if l_pos > 0.45:
        counter.append(f"⚠️ {loser} still commands {l_pos*100:.0f}% positive fan sentiment")
    if gap < 15:
        counter.append("⚠️ Model confidence is moderate — match remains competitive")
    if not counter:
        counter.append(f"ℹ️ {loser} could turn the match with a strong powerplay or key wickets")

    # ── Confidence percentage ─────────────────────────────────────────────────
    # Map: Low → 45-64, Medium → 65-79, High → 80-95
    conf_pct = {
        "Low":    45 + gap * 0.3,
        "Medium": 65 + gap * 0.2,
        "High":   80 + gap * 0.15,
    }.get(conf, 65)
    conf_pct = round(min(conf_pct, 96), 1)

    # ── Data quality ──────────────────────────────────────────────────────────
    dq = sent["data_quality"]
    if dq == "live" and (vol_a + vol_b) > 200:
        data_quality = "Strong (live data)"
    elif dq == "live":
        data_quality = "Moderate (live, limited volume)"
    else:
        data_quality = "Estimated (historical baselines)"

    # ── Verdict ───────────────────────────────────────────────────────────────
    verdict = (
        f"{w_name} is favoured to win with {w_prob:.0f}% probability. "
        f"Key drivers: {reasons[0].split('—')[0].strip() if reasons else 'composite model output'}."
    )

    return {
        "team":            winner,
        "opponent":        loser,
        "win_prob":        round(w_prob, 1),
        "confidence":      conf,
        "confidence_pct":  conf_pct,
        "data_quality":    data_quality,
        "reasons":         reasons,
        "counter_reasons": counter,
        "verdict":         verdict,
        "venue":           venue,
    }
