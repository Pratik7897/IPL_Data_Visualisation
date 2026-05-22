"""
modules/ai/training/generate_sample_data.py
Phase 4 — Feature 1: Synthetic IPL training dataset generator.

Generates realistic ball-by-ball IPL match snapshots for training
the match outcome prediction model. No external dataset required.

Run standalone to regenerate:
    python modules/ai/training/generate_sample_data.py

Outputs:
    modules/ai/training/ipl_match_snapshots.csv
"""
import os
import csv
import random
import math
from pathlib import Path

random.seed(42)

OUT_PATH = Path(__file__).parent / "ipl_match_snapshots.csv"

# IPL teams with rough historical win rates (source: cricinfo 2015-2024)
TEAMS = {
    "MI":   0.58, "CSK":  0.60, "RCB":  0.48, "KKR":  0.52,
    "SRH":  0.50, "RR":   0.49, "DC":   0.47, "PBKS": 0.44,
    "GT":   0.62, "LSG":  0.51,
}

VENUES = ["Wankhede", "Chepauk", "Chinnaswamy", "Eden Gardens",
          "Narendra Modi", "Kotla", "Sawai Mansingh", "DY Patil",
          "Brabourne", "Hyderabad"]

FIELDNAMES = [
    "team", "opponent",
    "current_score", "wickets_fallen", "overs_completed",
    "current_run_rate", "required_run_rate", "overs_remaining",
    "target", "runs_needed",
    "toss_won",              # 1 = this team won toss
    "venue_home",            # 1 = home ground
    "historical_win_rate",   # team's overall IPL win rate
    "team_sentiment_score",  # -1..1 (positive → negative)
    "opponent_sentiment_score",
    "tweet_volume_ratio",    # team_tweets / total_tweets
    "momentum_score",        # derived: recent wickets / recent runs composite
    "won",                   # 1 = team won the match (target)
]


def _clamp(v, lo, hi):
    return max(lo, min(hi, v))


def generate_chase_snapshot(team, opponent, won_flag):
    """Simulate a second-innings chase snapshot at a random over."""
    hist_wr  = TEAMS[team]
    opp_wr   = TEAMS[opponent]

    target    = random.randint(140, 210)
    over_done = random.uniform(5, 19)
    balls     = int(over_done * 6)
    over_rem  = 20 - over_done

    # Score trajectory biased by outcome
    if won_flag:
        run_pct = random.uniform(0.45, 0.85)
    else:
        run_pct = random.uniform(0.30, 0.65)

    score    = int((target - 1) * run_pct * (balls / 120))
    score    = _clamp(score, 0, target - 1)
    wickets  = _clamp(int(random.gauss(3.5, 2.0) * (over_done / 20)), 0, 10)

    crr      = score / over_done if over_done > 0 else 0
    runs_req = target - score
    rrr      = runs_req / over_rem if over_rem > 0 else 99.9

    # Sentiment: winning side gets positive boost
    base_sent = 0.3 if won_flag else -0.2
    t_sent    = _clamp(base_sent + random.uniform(-0.4, 0.4), -1, 1)
    o_sent    = _clamp(-base_sent + random.uniform(-0.4, 0.4), -1, 1)

    toss       = random.randint(0, 1)
    venue_home = random.randint(0, 1)
    vol_ratio  = _clamp(random.gauss(0.5, 0.15), 0.1, 0.9)

    # Momentum: negative if many wickets + low run rate gap, else positive
    mom = _clamp((crr - rrr) / 6 + (1 - wickets / 10) + random.uniform(-0.2, 0.2), -1, 1)

    return {
        "team":                      team,
        "opponent":                  opponent,
        "current_score":             score,
        "wickets_fallen":            wickets,
        "overs_completed":           round(over_done, 1),
        "current_run_rate":          round(crr, 2),
        "required_run_rate":         round(rrr, 2),
        "overs_remaining":           round(over_rem, 1),
        "target":                    target,
        "runs_needed":               runs_req,
        "toss_won":                  toss,
        "venue_home":                venue_home,
        "historical_win_rate":       hist_wr,
        "team_sentiment_score":      round(t_sent, 3),
        "opponent_sentiment_score":  round(o_sent, 3),
        "tweet_volume_ratio":        round(vol_ratio, 3),
        "momentum_score":            round(mom, 3),
        "won":                       int(won_flag),
    }


def generate_dataset(n=8000):
    rows = []
    teams = list(TEAMS.keys())
    for _ in range(n):
        t1, t2 = random.sample(teams, 2)
        won = random.random() < (TEAMS[t1] / (TEAMS[t1] + TEAMS[t2]) + random.uniform(-0.1, 0.1))
        rows.append(generate_chase_snapshot(t1, t2, won))
        # Add opponent's perspective too (mirror)
        rows.append(generate_chase_snapshot(t2, t1, not won))
    return rows


if __name__ == "__main__":
    rows = generate_dataset(8000)
    with open(OUT_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Generated {len(rows)} rows → {OUT_PATH}")
