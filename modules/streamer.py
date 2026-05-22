"""
Tweepy v2 Filtered Stream for IPL tweet ingestion.
Falls back to a demo data generator when no API keys are configured.
"""
import os
import time
import logging
import threading
import random
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

# ── IPL stream rules ───────────────────────────────────────────────────────────
STREAM_RULES = [
    {"value": "#IPL2025 -is:retweet lang:en",   "tag": "ipl_main"},
    {"value": "#MIvsCSK OR #CSKvsMI -is:retweet lang:en", "tag": "mi_csk"},
    {"value": "#RCB OR #KKR -is:retweet lang:en", "tag": "rcb_kkr"},
    {"value": "#Kohli OR #Dhoni OR #Bumrah -is:retweet lang:en", "tag": "players"},
    {"value": "#IPLfinal OR #IPL2025final -is:retweet lang:en", "tag": "final"},
]

TWEET_FIELDS = "id,text,author_id,created_at,lang,public_metrics"


# ── Real Tweepy stream ─────────────────────────────────────────────────────────
class IPLStreamListener:
    """Tweepy v2 StreamingClient subclass."""

    def __init__(self):
        try:
            import tweepy
            self._tweepy = tweepy
        except ImportError:
            logger.error("tweepy not installed. Run: pip install tweepy")
            raise

    def build_client(self, bearer_token: str):
        class _Listener(self._tweepy.StreamingClient):
            def on_tweet(inner_self, tweet):
                _process_tweet(
                    tweet_id=str(tweet.id),
                    text=tweet.text,
                    author=str(tweet.author_id),
                    created_at=str(tweet.created_at),
                    retweet_count=getattr(tweet, "public_metrics", {}).get("retweet_count", 0) if tweet.public_metrics else 0,
                    like_count=getattr(tweet, "public_metrics", {}).get("like_count", 0) if tweet.public_metrics else 0,
                )

            def on_errors(inner_self, errors):
                logger.warning(f"Stream error: {errors}")

            def on_disconnect(inner_self):
                logger.warning("Stream disconnected.")

        client = _Listener(bearer_token)

        # Clear + set rules
        existing = client.get_rules()
        if existing.data:
            ids = [r.id for r in existing.data]
            client.delete_rules(ids)
        client.add_rules([self._tweepy.StreamRule(r["value"]) for r in STREAM_RULES])
        return client

    def start(self, bearer_token: str):
        client = self.build_client(bearer_token)
        logger.info("Starting real Tweepy filtered stream …")
        client.filter(tweet_fields=TWEET_FIELDS)


def _process_tweet(tweet_id, text, author, created_at,
                   retweet_count=0, like_count=0):
    """Shared processing: store + analyse."""
    from modules.database import insert_tweet
    from modules.sentiment import analyze_and_store

    insert_tweet(tweet_id, text, author, created_at,
                 retweet_count=retweet_count, like_count=like_count)
    analyze_and_store(tweet_id, text)
    logger.debug(f"Processed tweet {tweet_id[:8]}…")


# ── Demo data generator ────────────────────────────────────────────────────────
DEMO_TWEETS = [
    # Positive
    ("Bumrah is absolutely unplayable tonight! What a spell! 🔥 #IPL2025 #MIvsCSK", "MI"),
    ("Dhoni finishes off in style! That six was pure class 💥 #Thala #CSK", "CSK"),
    ("Kohli on 🔥 brilliant century coming up! RCB all the way!! #IPL2025", "RCB"),
    ("What a catch by Jadeja!! Absolute genius in the field 🙌 #CSK", "CSK"),
    ("Maxwell going berserk! RCB fans this is your moment!! 🚀", "RCB"),
    ("KL Rahul looking so classy tonight, LSG are going to win this! 💯", "LSG"),
    ("Shubman Gill is just a different level of batter. GT are flying! 🔥", "GT"),
    ("Russell hits it out of the ground AGAIN! KKR KKR KKR!! 💥💥💥", "KKR"),
    ("Sanju Samson keeping it together under pressure. RR looking good!", "RR"),
    ("Suryakumar Yadav 360 degree batting, absolutely incredible scenes! 🌟", "MI"),
    # Negative
    ("Dropped catch again!! How are we losing this match smh 😡 #CSK", "CSK"),
    ("Terrible bowling, three wides in one over is unacceptable 😤 #MI", "MI"),
    ("RCB collapse again... same story every year 💔 disappointed", "RCB"),
    ("That was clearly a no ball, absolutely robbed by the umpires 😡 #KKR", "KKR"),
    ("Another wicket falls, our batting lineup is a disaster today 😢", ""),
    ("Worst DRS decision I've ever seen. Match fixing vibes 😤 #IPL2025", ""),
    ("That runout was so avoidable, horrible running between the wickets 👎", "SRH"),
    ("Losing from a winning position AGAIN. This team has no spine 😡 #PBKS", "PBKS"),
    # Neutral
    ("CSK need 48 off 24 balls, this could go either way #IPL2025", "CSK"),
    ("Interesting field placement by Rohit, let's see if it works #MI", "MI"),
    ("Over 14: SRH 134/4, game is nicely poised right now #IPL2025", "SRH"),
    ("Third umpire checking for no ball on that Bumrah wicket #MIvsCSK", "MI"),
    ("Both teams playing well, whoever wins the next 2 overs wins the match", ""),
    ("Score update: RR 98/3 after 12 overs, reasonable platform set", "RR"),
    ("Strategic timeout coming up, teams will regroup #IPL2025", ""),
    ("Rain interruption, DLS method may come into play if it continues", ""),
    # Event-heavy
    ("WICKET!! Kohli is OUT! Caught at deep mid-wicket! RCB 87/3 😮", "RCB"),
    ("SIX!! Dhoni hits it over long-on, right out of the stadium! 💥🔥", "CSK"),
    ("FOUR!! Rohit drives through covers, beautiful timing! #MI", "MI"),
    ("Wide ball, that was miles down leg side. 3 wides this over #SRH", "SRH"),
    ("NO BALL!! Free hit coming up for Gill 😍 #GT", "GT"),
    ("End of over 15: RCB 122/4, Maxwell on 67* off 38 balls! #IPL2025", "RCB"),
    ("WICKET!! Bumrah strikes again, Ruturaj clean bowled! #MIvsCSK 🔥", "MI"),
    ("That SIX by Russell nearly hit the commentary box! KKR crowd going crazy!", "KKR"),
]


class DemoStreamer:
    """Generates realistic fake tweets for dev/demo without real API keys."""

    def __init__(self, tweets_per_second=0.8):
        self.tps = tweets_per_second
        self._counter = 0
        self._running = False

    def start(self):
        self._running = True
        logger.info("🏏 Demo streamer started — generating synthetic IPL tweets")
        t = threading.Thread(target=self._loop, daemon=True)
        t.start()

    def stop(self):
        self._running = False

    def _loop(self):
        while self._running:
            self._emit()
            time.sleep(1.0 / self.tps + random.uniform(0, 1.5))

    def _emit(self):
        self._counter += 1
        text, _ = random.choice(DEMO_TWEETS)
        # Sprinkle hashtags
        text += f" #IPL2025"
        tweet_id = f"demo_{self._counter:010d}_{int(time.time()*1000) % 100000}"
        author = f"fan_{random.randint(1000, 9999)}"
        created_at = datetime.now(timezone.utc).isoformat()
        _process_tweet(tweet_id, text, author, created_at,
                       retweet_count=random.randint(0, 500),
                       like_count=random.randint(0, 2000))


def start_streamer(bearer_token: str = None):
    """Start real stream if token provided, otherwise demo."""
    if bearer_token:
        try:
            listener = IPLStreamListener()
            t = threading.Thread(
                target=listener.start, args=(bearer_token,), daemon=True)
            t.start()
            logger.info("Real Tweepy stream started in background thread.")
        except Exception as exc:
            logger.warning(f"Real stream failed ({exc}), falling back to demo.")
            DemoStreamer().start()
    else:
        DemoStreamer().start()
