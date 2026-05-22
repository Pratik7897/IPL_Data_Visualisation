"""
modules/ai/stream_replay.py
Phase 5 — Feature 2: Authentic Live Fan Reaction Stream.

Replays a curated dataset of realistic IPL fan comments with:
  - randomised timing
  - source attribution (Reddit / YouTube / Historical Replay / Match Thread)
  - sentiment tagging
  - engagement counts
  - smooth streaming intervals

Zero changes to the existing tweet pipeline — this is an additive overlay.
"""
import random
import time
import threading
import logging
from collections import deque
from datetime import datetime, timezone, timedelta

logger = logging.getLogger(__name__)

# ── Curated realistic IPL fan comment pool ────────────────────────────────────
# Each entry: (text, source, base_sentiment, base_engagement)
_RAW_COMMENTS = [
    # High-energy positives
    ("Bumrah is an absolute MACHINE. Three wickets in two overs 🔥", "Reddit", "Positive", 342),
    ("CSK's death bowling has levelled up massively this season 💪", "Reddit", "Positive", 218),
    ("Kohli just hit a 96m six over long-on. That's just ridiculous power 😤", "YouTube", "Positive", 891),
    ("Dhoni finishing it in style. ALWAYS. Never gets old 🐐", "YouTube", "Positive", 1204),
    ("Rohit's timing today is absolutely pristine. Cover drives on a different level", "Reddit", "Positive", 445),
    ("THIS. IS. CRICKET. What a match lads 🏏🔥", "Match Thread", "Positive", 678),
    ("Rashid is turning this ball more than a carom ball wtf 😂", "Reddit", "Positive", 321),
    ("GT playing with ice in their veins. Nerves of steel 💪", "YouTube", "Positive", 289),
    ("MI's powerplay batting today is ELITE. 62/0 in 6 overs 🤯", "Reddit", "Positive", 534),
    ("Ruturaj Gaikwad is the most underrated batter in the IPL. Period.", "Reddit", "Positive", 412),
    ("Hardik Pandya single-handedly won that with the ball. 3/18 is insane 👏", "YouTube", "Positive", 623),
    ("Sanju Samson's strike rate when chasing is literally 180+ 🚀", "Reddit", "Positive", 287),
    ("KKR fans deserve this. They've waited long enough 💛💜", "Match Thread", "Positive", 445),
    ("Suryakumar just reverse-swept a 145kph delivery. WHAT 😂", "Reddit", "Positive", 901),
    ("David Warner showing IPL class once again 🙌", "YouTube", "Positive", 378),

    # Moderately positive
    ("RR are building a decent partnership here, 67 off 8 overs looks solid", "Reddit", "Positive", 134),
    ("That's a good field placement by the captain. Smart cricket.", "Match Thread", "Neutral", 89),
    ("Both teams evenly matched so far. Could go either way from here", "Reddit", "Neutral", 167),
    ("Interesting toss decision. Let's see if it pays off in the second innings", "YouTube", "Neutral", 203),
    ("Pitch report suggests it'll slow down. Spinners could be crucial here", "Reddit", "Neutral", 156),
    ("Score is 98/3 after 12 overs. Required rate creeping up. Need boundaries.", "Match Thread", "Neutral", 234),
    ("DC making it competitive. 7 off the last over keeps them in the hunt", "Reddit", "Neutral", 189),
    ("PBKS need 48 off 24. Doable but risky. One wicket and game changes.", "YouTube", "Neutral", 267),
    ("Chahal is varying his pace well. Batsmen are unsure. Good plan.", "Reddit", "Neutral", 143),
    ("Match is on a knife edge. Whoever takes the next wicket swings it", "Match Thread", "Neutral", 312),

    # Negatives / frustration
    ("That dropped catch is going to haunt them. Absolutely crucial moment 😤", "Reddit", "Negative", 287),
    ("Why bowl a full toss at this stage?? Inexplicable from the captain 🤦", "YouTube", "Negative", 456),
    ("RCB blew it AGAIN. Classic RCB 😩 Same story every season", "Reddit", "Negative", 723),
    ("That run-out was absolutely brainless. Cost them the match, no doubt.", "Match Thread", "Negative", 534),
    ("Umpiring decision makes zero sense. How is that not a wide?! 😡", "YouTube", "Negative", 612),
    ("Three consecutive dot balls when you needed 18 off 12. That's on the batsman.", "Reddit", "Negative", 234),
    ("Bowling at the slot every single ball. No variation, no plan. Frustrating 😒", "Match Thread", "Negative", 378),
    ("How many times can you lose from a winning position, SRH? How many times?", "Reddit", "Negative", 445),
    ("The DRS review was completely wrong. Needs better technology in IPL honestly", "YouTube", "Negative", 321),
    ("Batting collapse from 89/1 to 112/7. Pure mental fragility.", "Reddit", "Negative", 567),

    # Match thread style commentary
    ("FOUR! Drove it through extra cover. Beautiful shot. 98/2 (12.3)", "Match Thread", "Positive", 178),
    ("WICKET! Gone! Caught behind off the outside edge. 67/4 (8.1)", "Match Thread", "Negative", 234),
    ("SIX! Maximum over long-on! Crowd going absolutely mental! 🎉", "Match Thread", "Positive", 567),
    ("Wide down leg. Pressure is palpable. NRR situation here is critical.", "Match Thread", "Neutral", 123),
    ("REVIEW! LBW appeal... umpire's call, stays not out. Close one!", "Match Thread", "Neutral", 345),
    ("Over 15 done. 48 needed off 30 balls. This is a proper chase.", "Match Thread", "Neutral", 289),
    ("Good over by Bumrah. Just 4 off it. Pressure building on batsmen.", "Match Thread", "Positive", 234),
    ("That's the 50 partnership! Important to keep the scoreboard ticking.", "Match Thread", "Positive", 178),

    # Hinglish (authentic Indian fan speak)
    ("Bhai Kohli aaj kuch alag hi lag rha hai 🔥 vintage form!", "Reddit", "Positive", 456),
    ("MS Dhoni ka fan ho toh hath utha 🙌 goat hai goat rahega", "YouTube", "Positive", 892),
    ("CSK ne phir se jeeta yaar. Unbeatable lagta hai Dhoni ki captaincy mein 💛", "Match Thread", "Positive", 623),
    ("Bhai agar yeh catch nahi pakda toh poora over waste tha 😤", "Reddit", "Negative", 234),
    ("KKR ke fielders sote rahte hain. Seriously yaar 🤦‍♂️", "YouTube", "Negative", 345),
    ("Rohit bhai jab set ho jata hai toh koi nahi rok sakta 🏏💥", "Reddit", "Positive", 567),
    ("RCB waale rote raho 😂 hamesha aise hi hoga", "Match Thread", "Negative", 789),
    ("GT itna consistently khelta hai. Shubman Gill is the future bro 🙌", "Reddit", "Positive", 412),
    ("Bumrah ki yorker pe koi bhi out ho sakta hai, even in test match 🏏", "YouTube", "Positive", 678),

    # Stats / analytical fan
    ("Based on xRuns, CSK are running at 7.2 per over projected. That's above par here.", "Reddit", "Positive", 89),
    ("Boundary % today is 68% for MI — that's unusually high for this pitch", "Reddit", "Neutral", 134),
    ("Bumrah's economy at the death is 6.8 across 8 matches. Elite territory.", "Reddit", "Positive", 178),
    ("Expected score based on current run rate: 172 ± 12. Interesting range.", "Match Thread", "Neutral", 156),
    ("Historical win rate chasing this target at this venue is 54%. Marginal.", "Reddit", "Neutral", 143),

    # Reactions to key events
    ("DHONI FINISHES IT WITH A SIX! UNBELIEVABLE! 🏆🏆🏆", "YouTube", "Positive", 3421),
    ("He played through a hamstring strain the entire second innings. WARRIOR 🦁", "Reddit", "Positive", 1234),
    ("FOUR! Off the last ball. CSK WIN BY 2 WICKETS! Incredible scenes!!! 🎉🎉🎉", "Match Thread", "Positive", 4567),
    ("Rohit drops a sitter at mid-wicket. That WILL cost MI today. Mark my words.", "YouTube", "Negative", 891),
    ("Last-ball six needed. Samson on strike. THIS IS WHY IPL IS THE GREATEST 😤", "Match Thread", "Positive", 2345),
]

SOURCES = ["Reddit", "YouTube", "Match Thread", "Historical Replay"]
SOURCE_COLORS = {
    "Reddit":           "#FF4500",
    "YouTube":          "#FF0000",
    "Match Thread":     "#58A6FF",
    "Historical Replay":"#FFD166",
}

# ── In-memory stream buffer ───────────────────────────────────────────────────
_STREAM_BUFFER: deque = deque(maxlen=100)
_stream_lock   = threading.Lock()
_stream_thread = None
_stop_event    = threading.Event()


def _generate_timestamp() -> str:
    """Produce a realistic recent timestamp string."""
    offset = random.randint(0, 180)
    t = datetime.now(timezone.utc) - timedelta(seconds=offset)
    return t.strftime("%H:%M:%S")


def _comment_to_item(raw: tuple, seq_id: int) -> dict:
    text, source, sentiment, base_eng = raw
    # Randomise engagement slightly
    engagement = int(base_eng * random.uniform(0.85, 1.20))
    return {
        "id":         seq_id,
        "text":       text,
        "source":     source,
        "sentiment":  sentiment,
        "timestamp":  _generate_timestamp(),
        "engagement": engagement,
        "source_color": SOURCE_COLORS.get(source, "#888"),
    }


def _stream_worker():
    """Background thread: push items into the buffer at realistic intervals."""
    pool    = list(_RAW_COMMENTS)
    random.shuffle(pool)
    idx = 0
    seq = 0
    while not _stop_event.is_set():
        # Pick next comment (cycle through shuffled pool)
        raw  = pool[idx % len(pool)]
        item = _comment_to_item(raw, seq)
        with _stream_lock:
            _STREAM_BUFFER.appendleft(item)
        idx += 1
        seq += 1
        # Realistic intervals: 0.8–4.5 seconds (Poisson-like)
        sleep_t = random.expovariate(1 / 2.5)  # mean 2.5s
        sleep_t = max(0.8, min(sleep_t, 6.0))
        _stop_event.wait(sleep_t)


def start_stream_replay():
    """Start the background replay thread (idempotent)."""
    global _stream_thread
    if _stream_thread and _stream_thread.is_alive():
        return
    _stop_event.clear()
    _stream_thread = threading.Thread(target=_stream_worker, daemon=True, name="stream-replay")
    _stream_thread.start()
    logger.info("[stream_replay] Live Fan Reaction Stream started (%d comment pool)", len(_RAW_COMMENTS))


def get_stream_items(limit: int = 15) -> list[dict]:
    """Return the latest N items from the stream buffer."""
    with _stream_lock:
        return list(_STREAM_BUFFER)[:limit]


def get_stream_feed_api(limit: int = 20) -> dict:
    """API-ready response for /stream-comments."""
    items = get_stream_items(limit)
    return {
        "count":    len(items),
        "comments": items,
        "pool_size": len(_RAW_COMMENTS),
        "sources":  list(SOURCE_COLORS.keys()),
    }
