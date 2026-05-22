"""
modules/ai/match_replay.py
Phase 5 — Feature 4: Historical Match Replay Engine.

Replays famous IPL matches as cinematic real-time simulations.
Each match is a sequence of "moments" — key events with:
  - commentary text
  - sentiment impact
  - win probability shift
  - over/ball reference

No external dataset needed — matches are embedded as structured data.
"""
import random
import threading
import logging
from datetime import datetime, timezone
from typing import Optional

logger = logging.getLogger(__name__)

# ── Famous IPL matches dataset ────────────────────────────────────────────────
FAMOUS_MATCHES = {
    "csk_vs_mi_2019_final": {
        "title": "CSK vs MI — IPL 2019 Final",
        "venue": "Wankhede",
        "date": "May 12, 2019",
        "team_a": "CSK", "team_b": "MI",
        "result": "MI won by 1 run",
        "description": "One of the greatest IPL finals ever. MI defended 149 against CSK in a thriller.",
        "moments": [
            {"over": "1.0",  "text": "Sharma gets MI off to a flyer! Boundary through point. 7/0", "sentiment": "Positive", "team": "MI", "prob_mi": 55},
            {"over": "2.3",  "text": "Rohit Sharma edges behind! Huge wicket for CSK. MI wobbling. 14/1", "sentiment": "Negative", "team": "MI", "prob_mi": 48},
            {"over": "4.6",  "text": "SIX! Quinton de Kock launches one over long-on. MI accelerating. 38/1", "sentiment": "Positive", "team": "MI", "prob_mi": 54},
            {"over": "7.2",  "text": "de Kock departs for 29. Spin on both ends causing issues. 52/2", "sentiment": "Negative", "team": "MI", "prob_mi": 50},
            {"over": "10.0", "text": "Halfway mark: MI 71/2. Hardik Pandya and Kieron Pollard at the crease.", "sentiment": "Neutral", "team": "MI", "prob_mi": 52},
            {"over": "14.1", "text": "Pollard hits back-to-back sixes! Death overs are crucial. MI 102/4", "sentiment": "Positive", "team": "MI", "prob_mi": 57},
            {"over": "17.3", "text": "Pandya dismisses Pollard. Shardul stunning. MI 122/5", "sentiment": "Negative", "team": "MI", "prob_mi": 53},
            {"over": "20.0", "text": "MI post 149/8. Hardik's 25* crucial. CSK need 150 to win.", "sentiment": "Neutral", "team": "MI", "prob_mi": 52},
            {"over": "21.0", "text": "CSK innings begins. Shane Watson and Faf du Plessis open.", "sentiment": "Neutral", "team": "CSK", "prob_mi": 50},
            {"over": "22.4", "text": "Watson smashed through point! CSK starting aggressively. 12/0", "sentiment": "Positive", "team": "CSK", "prob_mi": 47},
            {"over": "25.2", "text": "Bumrah strikes! Faf du Plessis goes for 26. CSK 44/1", "sentiment": "Positive", "team": "MI", "prob_mi": 52},
            {"over": "27.0", "text": "Watson hits a SIX off Krunal! CSK fans go wild! 58/1", "sentiment": "Positive", "team": "CSK", "prob_mi": 45},
            {"over": "30.0", "text": "Suresh Raina gone cheaply for 8. CSK 79/2 at halfway", "sentiment": "Negative", "team": "CSK", "prob_mi": 54},
            {"over": "34.3", "text": "Shane Watson reaches FIFTY! CSK back in the hunt! 97/2", "sentiment": "Positive", "team": "CSK", "prob_mi": 48},
            {"over": "36.1", "text": "Watson clean bowled by Bumrah! CRUCIAL! CSK 115/3. 35 needed off 23", "sentiment": "Negative", "team": "CSK", "prob_mi": 58},
            {"over": "37.5", "text": "Dhoni walks in at number 7. CSK need 23 off 12. DHONI TIME!!", "sentiment": "Positive", "team": "CSK", "prob_mi": 56},
            {"over": "38.4", "text": "SIX! Dhoni hits Bumrah over wide long-on! 13 off 7 needed!", "sentiment": "Positive", "team": "CSK", "prob_mi": 50},
            {"over": "39.1", "text": "Malinga yorker! Dhoni misses! CSK need 9 off 5 balls!", "sentiment": "Negative", "team": "CSK", "prob_mi": 56},
            {"over": "39.4", "text": "SIX! Dhoni again! 3 runs needed off 2 balls! UNBELIEVABLE!", "sentiment": "Positive", "team": "CSK", "prob_mi": 50},
            {"over": "39.5", "text": "Malinga full toss, Dhoni hits it straight to fielder. 2 needed off last ball!", "sentiment": "Negative", "team": "CSK", "prob_mi": 52},
            {"over": "39.6", "text": "SHARP SINGLE! But only ONE RUN! MI WIN BY 1 RUN! PANDEMONIUM AT WANKHEDE! 🏆", "sentiment": "Positive", "team": "MI", "prob_mi": 100},
        ]
    },
    "rcb_vs_csk_2015_pe": {
        "title": "RCB vs CSK — 2015 Eliminator",
        "venue": "Chinnaswamy",
        "date": "May 20, 2015",
        "team_a": "RCB", "team_b": "CSK",
        "result": "CSK won by 3 wickets",
        "description": "AB de Villiers' genius vs Dhoni's cool head in a classic Bangalore slugfest.",
        "moments": [
            {"over": "1.0",  "text": "Kohli and Gayle open for RCB. Chinnaswamy is ELECTRIC! 🎉", "sentiment": "Positive", "team": "RCB", "prob_rcb": 50},
            {"over": "3.2",  "text": "GAYLE GONE! 0 off 3 balls. RCB powerplay troubled. 18/1", "sentiment": "Negative", "team": "RCB", "prob_rcb": 45},
            {"over": "6.0",  "text": "End of powerplay: RCB 46/1. Kohli looking dangerous.", "sentiment": "Neutral", "team": "RCB", "prob_rcb": 52},
            {"over": "10.0", "text": "Kohli gone for 44! RCB 78/2. AB de Villiers walks in!", "sentiment": "Negative", "team": "RCB", "prob_rcb": 50},
            {"over": "14.3", "text": "AB de VILLIERS! Six, six, FOUR in the over! The man is inhuman! 🔥", "sentiment": "Positive", "team": "RCB", "prob_rcb": 58},
            {"over": "17.0", "text": "AB hits 100 off just 57 balls! One of the great playoff innings!", "sentiment": "Positive", "team": "RCB", "prob_rcb": 62},
            {"over": "20.0", "text": "RCB finish with 178/5. AB de Villiers 119* off 56 balls! UNREAL!", "sentiment": "Positive", "team": "RCB", "prob_rcb": 60},
            {"over": "22.0", "text": "CSK start the chase. McCullum and Hussey open with intent.", "sentiment": "Neutral", "team": "CSK", "prob_rcb": 58},
            {"over": "25.0", "text": "McCullum hits a 100m six! CSK racing at 12 per over! 50/0 after 4", "sentiment": "Positive", "team": "CSK", "prob_rcb": 52},
            {"over": "29.3", "text": "BOTH openers gone! CSK 88/2. Raina and Bravo at the crease.", "sentiment": "Negative", "team": "CSK", "prob_rcb": 57},
            {"over": "32.0", "text": "Raina gone for 28. CSK 108/3. Dhoni walks in! Tension cranking.", "sentiment": "Negative", "team": "CSK", "prob_rcb": 60},
            {"over": "35.4", "text": "Dhoni hits a straight six off Chahal! Still in the game! 134/3", "sentiment": "Positive", "team": "CSK", "prob_rcb": 54},
            {"over": "38.0", "text": "CSK need 27 off 12 balls. Dhoni and Bravo. Can they do it?", "sentiment": "Neutral", "team": "CSK", "prob_rcb": 52},
            {"over": "39.2", "text": "SIX! Bravo! 14 needed off 6. CSK are ALIVE!", "sentiment": "Positive", "team": "CSK", "prob_rcb": 46},
            {"over": "39.6", "text": "BOUNDARY! CSK WIN BY 3 WICKETS! Bravo finishes in style! 🏆 CSK are through!", "sentiment": "Positive", "team": "CSK", "prob_rcb": 0},
        ]
    },
    "mi_vs_srh_2013_final": {
        "title": "MI vs SRH — IPL 2013 Final",
        "venue": "Eden Gardens",
        "date": "May 26, 2013",
        "team_a": "MI", "team_b": "SRH",
        "result": "MI won by 23 runs",
        "description": "Mumbai Indians' maiden IPL title. Rohit Sharma's first as captain.",
        "moments": [
            {"over": "2.0",  "text": "Rohit and Dinesh Karthik off to a solid start. 16/0", "sentiment": "Positive", "team": "MI", "prob_mi": 52},
            {"over": "5.1",  "text": "Karthik gone! But Pollard is here. 33/1 after powerplay", "sentiment": "Neutral", "team": "MI", "prob_mi": 52},
            {"over": "10.4", "text": "Rohit Sharma 50! Captain leading from the front! 72/1", "sentiment": "Positive", "team": "MI", "prob_mi": 58},
            {"over": "15.0", "text": "Pollard and Rohit putting together a match-winning stand. 102/1", "sentiment": "Positive", "team": "MI", "prob_mi": 62},
            {"over": "18.2", "text": "Rohit out for 65. Momentum shifts slightly. 120/4", "sentiment": "Negative", "team": "MI", "prob_mi": 58},
            {"over": "20.0", "text": "MI total: 148/9. Below par but could be enough on this Eden surface.", "sentiment": "Neutral", "team": "MI", "prob_mi": 54},
            {"over": "22.0", "text": "SRH chase begins. Warner and Dhawan opening. Electric atmosphere!", "sentiment": "Neutral", "team": "SRH", "prob_mi": 52},
            {"over": "25.3", "text": "Warner GONE! Brilliant catch by Malinga. SRH 28/1", "sentiment": "Positive", "team": "MI", "prob_mi": 56},
            {"over": "28.0", "text": "Harbhajan Singh on! Spinning it sharply. SRH 42/2", "sentiment": "Positive", "team": "MI", "prob_mi": 60},
            {"over": "32.4", "text": "Steyn dropped a sitter! That could be costly for SRH.", "sentiment": "Negative", "team": "SRH", "prob_mi": 63},
            {"over": "35.0", "text": "SRH 79/5 at halfway. Required rate climbing. MI on top.", "sentiment": "Positive", "team": "MI", "prob_mi": 68},
            {"over": "38.0", "text": "Malinga takes his 3rd! SRH 95/6. MI smelling the title.", "sentiment": "Positive", "team": "MI", "prob_mi": 78},
            {"over": "40.0", "text": "🏆 MI WIN BY 23 RUNS! ROHIT SHARMA LIFTS THE IPL TROPHY! First title! Wankhede, Eden — this team is CHAMPIONS!", "sentiment": "Positive", "team": "MI", "prob_mi": 100},
        ]
    },
    "kkr_vs_pbks_last_ball_2022": {
        "title": "KKR vs PBKS — Last-Ball Thriller 2022",
        "venue": "DY Patil",
        "date": "March 31, 2022",
        "team_a": "KKR", "team_b": "PBKS",
        "result": "KKR won by 6 wickets (last ball)",
        "description": "An absolute nail-biter. KKR needed 6 off the last over to win.",
        "moments": [
            {"over": "1.0",  "text": "PBKS start aggressively. Shikhar Dhawan drives through covers. 8/0", "sentiment": "Positive", "team": "PBKS", "prob_kkr": 48},
            {"over": "6.0",  "text": "Powerplay: PBKS 54/0! Both openers batting beautifully.", "sentiment": "Positive", "team": "PBKS", "prob_kkr": 42},
            {"over": "9.3",  "text": "Mayank Agarwal out for 31! Varun Chakravarthy turning it! PBKS 70/1", "sentiment": "Negative", "team": "PBKS", "prob_kkr": 47},
            {"over": "13.0", "text": "Liam Livingstone SIX over deep mid-wicket! Crowd erupts. 96/2", "sentiment": "Positive", "team": "PBKS", "prob_kkr": 44},
            {"over": "17.2", "text": "PBKS 120/3. Death overs start. Both teams know what's at stake.", "sentiment": "Neutral", "team": "PBKS", "prob_kkr": 48},
            {"over": "20.0", "text": "PBKS post 137/9. Below expectations but KKR chase won't be easy.", "sentiment": "Neutral", "team": "PBKS", "prob_kkr": 52},
            {"over": "22.0", "text": "Ajinkya Rahane opens KKR chase. Aaron Finch departs early. 8/1", "sentiment": "Negative", "team": "KKR", "prob_kkr": 50},
            {"over": "27.0", "text": "Shreyas Iyer and Nitish Rana building! 62/2 at 7 overs. On track!", "sentiment": "Positive", "team": "KKR", "prob_kkr": 54},
            {"over": "31.3", "text": "Iyer gone for 34! KKR 85/3. 53 needed off 5 overs. Tricky!", "sentiment": "Negative", "team": "KKR", "prob_kkr": 50},
            {"over": "35.0", "text": "Sam Billings smashes Rabada for 14 in the over! KKR 107/3. 31 off 3 overs!", "sentiment": "Positive", "team": "KKR", "prob_kkr": 54},
            {"over": "37.2", "text": "Billings out! KKR 118/4. 20 needed off 16 balls. Rinku Singh walks in.", "sentiment": "Negative", "team": "KKR", "prob_kkr": 49},
            {"over": "39.0", "text": "19th over done. KKR 131/5. 7 RUNS NEEDED OFF LAST OVER. Arshdeep Singh to bowl!", "sentiment": "Neutral", "team": "KKR", "prob_kkr": 48},
            {"over": "39.1", "text": "DOT BALL! Rinku Singh can't get the bat on it. 7 off 5 balls!", "sentiment": "Negative", "team": "KKR", "prob_kkr": 44},
            {"over": "39.2", "text": "WIDE! Down leg. 6 off 5 but Rinku facing. Edge of seats!", "sentiment": "Neutral", "team": "KKR", "prob_kkr": 46},
            {"over": "39.3", "text": "FOUR! Rinku drives through covers! 2 RUNS NEEDED OFF 3 BALLS!", "sentiment": "Positive", "team": "KKR", "prob_kkr": 54},
            {"over": "39.4", "text": "DOT. Wide yorker. 2 needed off 2. HEART IS POUNDING.", "sentiment": "Negative", "team": "KKR", "prob_kkr": 51},
            {"over": "39.5", "text": "DOT! Short ball. KKR need 2 off LAST BALL. THIS IS INSANE!", "sentiment": "Negative", "team": "KKR", "prob_kkr": 48},
            {"over": "39.6", "text": "SIX!!!! RINKU SINGH HITS ARSHDEEP FOR A SIX OFF THE LAST BALL! KKR WIN!!! 🤯🏆 ABSOLUTE SCENES AT DY PATIL!", "sentiment": "Positive", "team": "KKR", "prob_kkr": 100},
        ]
    },
    "gt_vs_rr_2022_final": {
        "title": "GT vs RR — IPL 2022 Final",
        "venue": "Narendra Modi",
        "date": "May 29, 2022",
        "team_a": "GT", "team_b": "RR",
        "result": "GT won by 7 wickets",
        "description": "Gujarat Titans won the IPL title in their debut season. Hardik Pandya's masterclass.",
        "moments": [
            {"over": "1.0",  "text": "Buttler and Devdutt Padikkal open for RR. Motera buzzing! 5/0", "sentiment": "Positive", "team": "RR", "prob_gt": 48},
            {"over": "4.5",  "text": "WICKET! Padikkal gone for 9. Hardik Pandya gets the big one! 23/1", "sentiment": "Positive", "team": "GT", "prob_gt": 53},
            {"over": "8.3",  "text": "Buttler reaches 50 off 37 balls! RR hitting back hard. 65/1", "sentiment": "Positive", "team": "RR", "prob_gt": 48},
            {"over": "12.0", "text": "Buttler caught at deep midwicket for 39. Key wicket for GT! 76/3", "sentiment": "Positive", "team": "GT", "prob_gt": 55},
            {"over": "16.4", "text": "Rashid Khan strikes! Two in two! RR collapsing. 96/5", "sentiment": "Positive", "team": "GT", "prob_gt": 63},
            {"over": "20.0", "text": "RR finish 130/9. Below par on this surface. GT need 131 to win the IPL!", "sentiment": "Neutral", "team": "GT", "prob_gt": 65},
            {"over": "22.0", "text": "GT chase begins. Wriddhiman Saha and Shubman Gill open. The crowd is with them!", "sentiment": "Positive", "team": "GT", "prob_gt": 64},
            {"over": "24.2", "text": "SIX! Gill goes aerial! GT cruising. 28/0 after 4 overs.", "sentiment": "Positive", "team": "GT", "prob_gt": 67},
            {"over": "26.1", "text": "Saha out for 5. GT 34/1. Hardik walks in to a THUNDEROUS reception!", "sentiment": "Neutral", "team": "GT", "prob_gt": 66},
            {"over": "29.0", "text": "Gill and Hardik taking GT to the brink. 68/1 after 9 overs!", "sentiment": "Positive", "team": "GT", "prob_gt": 72},
            {"over": "32.3", "text": "Gill out for 45! GT 81/2. 50 more needed from 45 balls. Well placed!", "sentiment": "Neutral", "team": "GT", "prob_gt": 71},
            {"over": "36.0", "text": "Hardik brings up his fifty! Captain. Leader. Champion. 102/2", "sentiment": "Positive", "team": "GT", "prob_gt": 80},
            {"over": "37.4", "text": "Hardik hits the WINNING RUNS! GT WIN IPL 2022 IN THEIR FIRST SEASON! 🏆🟡🔵", "sentiment": "Positive", "team": "GT", "prob_gt": 100},
        ]
    },
}

# ── Replay state management ───────────────────────────────────────────────────
_replay_state: dict = {
    "active":    False,
    "match_id":  None,
    "current_idx": 0,
    "moments":   [],
    "played":    [],
    "speed":     1.0,   # multiplier
}
_replay_lock = threading.Lock()


def get_match_list() -> list[dict]:
    """Return all available matches for the replay dropdown."""
    return [
        {
            "id":          mid,
            "title":       m["title"],
            "date":        m["date"],
            "venue":       m["venue"],
            "result":      m["result"],
            "description": m["description"],
            "moment_count": len(m["moments"]),
        }
        for mid, m in FAMOUS_MATCHES.items()
    ]


def start_replay(match_id: str, speed: float = 1.0) -> dict:
    """Initialise a replay session."""
    match = FAMOUS_MATCHES.get(match_id)
    if not match:
        return {"error": f"Match '{match_id}' not found."}
    with _replay_lock:
        _replay_state.update({
            "active":      True,
            "match_id":    match_id,
            "current_idx": 0,
            "moments":     list(match["moments"]),
            "played":      [],
            "speed":       float(speed),
            "team_a":      match["team_a"],
            "team_b":      match["team_b"],
            "title":       match["title"],
            "result":      match["result"],
        })
    logger.info(f"[match_replay] Replay started: {match['title']}")
    return {"status": "started", "match": match["title"], "total_moments": len(match["moments"])}


def get_replay_next_moment() -> Optional[dict]:
    """Return the next unplayed moment and advance the cursor."""
    with _replay_lock:
        if not _replay_state["active"]:
            return None
        idx = _replay_state["current_idx"]
        moments = _replay_state["moments"]
        if idx >= len(moments):
            _replay_state["active"] = False
            return {"done": True, "result": _replay_state.get("result", "Match complete")}
        moment = moments[idx]
        _replay_state["current_idx"] = idx + 1
        _replay_state["played"].append(moment)
        return {**moment, "progress": f"{idx+1}/{len(moments)}"}


def get_replay_state() -> dict:
    """Return full current replay state for the dashboard."""
    with _replay_lock:
        state = dict(_replay_state)
    return state


def reset_replay() -> dict:
    """Stop and reset the current replay."""
    with _replay_lock:
        _replay_state.update({
            "active": False, "match_id": None,
            "current_idx": 0, "moments": [], "played": [],
        })
    return {"status": "reset"}
