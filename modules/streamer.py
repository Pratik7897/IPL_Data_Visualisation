"""
Tweepy v2 Filtered Stream for IPL tweet ingestion.
Falls back to a demo data generator when no API keys are configured.

Phase 2A — Auto-reconnect:
  • IPLStreamSupervisor wraps the bare stream in a while-True supervisor loop
    with exponential back-off + jitter (cap: 5 min).
  • A threading.Event lets the supervisor be stopped cleanly.
  • StreamStatus is a module-level singleton the dashboard can query to show
    connection state in the header.
"""
import os
import time
import logging
import threading
import random
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

# ── IPL stream rules ────────────────────────────────────────────────────────────
STREAM_RULES = [
    {"value": "#IPL2025 -is:retweet lang:en",            "tag": "ipl_main"},
    {"value": "#MIvsCSK OR #CSKvsMI -is:retweet lang:en","tag": "mi_csk"},
    {"value": "#RCB OR #KKR -is:retweet lang:en",        "tag": "rcb_kkr"},
    {"value": "#Kohli OR #Dhoni OR #Bumrah -is:retweet lang:en", "tag": "players"},
    {"value": "#IPLfinal OR #IPL2025final -is:retweet lang:en",  "tag": "final"},
]

TWEET_FIELDS = "id,text,author_id,created_at,lang,public_metrics"

# Back-off parameters
_BACKOFF_BASE    = 5    # seconds for first retry
_BACKOFF_MAX     = 300  # 5 minutes cap
_BACKOFF_FACTOR  = 2    # multiply each failure
_BACKOFF_JITTER  = 5    # ±seconds of random jitter


# ── Module-level stream status (queryable from the dashboard) ───────────────────
class StreamStatus:
    """Thread-safe singleton tracking connection state."""

    def __init__(self):
        self._lock     = threading.Lock()
        self.mode      = "idle"          # "demo" | "live" | "reconnecting" | "idle"
        self.connected = False
        self.attempts  = 0
        self.last_ok   = None            # datetime of last successful connect
        self.last_err  = None            # last exception string
        self.next_retry_in = 0           # seconds until next reconnect attempt

    def set(self, **kwargs):
        with self._lock:
            for k, v in kwargs.items():
                setattr(self, k, v)

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "mode":          self.mode,
                "connected":     self.connected,
                "attempts":      self.attempts,
                "last_ok":       self.last_ok,
                "last_err":      self.last_err,
                "next_retry_in": self.next_retry_in,
            }


# Singleton — import from here in app.py
stream_status = StreamStatus()


# ── Real Tweepy stream ──────────────────────────────────────────────────────────
class IPLStreamListener:
    """Thin wrapper around tweepy.StreamingClient."""

    def __init__(self):
        try:
            import tweepy
            self._tweepy = tweepy
        except ImportError:
            logger.error("tweepy not installed. Run: pip install tweepy")
            raise

    def build_client(self, bearer_token: str):
        tweepy = self._tweepy

        class _Listener(tweepy.StreamingClient):
            def on_tweet(inner_self, tweet):
                _process_tweet(
                    tweet_id=str(tweet.id),
                    text=tweet.text,
                    author=str(tweet.author_id),
                    created_at=str(tweet.created_at),
                    retweet_count=(tweet.public_metrics or {}).get("retweet_count", 0),
                    like_count=(tweet.public_metrics or {}).get("like_count", 0),
                )

            def on_errors(inner_self, errors):
                logger.warning(f"Stream API error: {errors}")

            def on_disconnect(inner_self):
                logger.warning("Stream disconnected by server.")

        client = _Listener(bearer_token)

        # Sync rules: delete old, add current
        existing = client.get_rules()
        if existing.data:
            client.delete_rules([r.id for r in existing.data])
        client.add_rules([tweepy.StreamRule(r["value"]) for r in STREAM_RULES])
        return client

    def connect_and_filter(self, bearer_token: str):
        """Single connection attempt — raises on failure."""
        client = self.build_client(bearer_token)
        logger.info("Tweepy filtered stream connecting …")
        # filter() blocks until disconnected or an exception is raised
        client.filter(tweet_fields=TWEET_FIELDS)


# ── Auto-reconnect supervisor ───────────────────────────────────────────────────
class IPLStreamSupervisor:
    """
    Runs the Tweepy stream in a daemon thread with automatic reconnect.

    Back-off schedule (each consecutive failure doubles the wait):
        attempt 1 →  5s ± jitter
        attempt 2 → 10s ± jitter
        attempt 3 → 20s ± jitter
        …
        attempt N → 300s (cap) ± jitter
    A successful connection resets the back-off to the base value.
    """

    def __init__(self, bearer_token: str):
        self._token    = bearer_token
        self._stop_evt = threading.Event()
        self._thread   = threading.Thread(
            target=self._supervisor_loop, daemon=True, name="ipl-stream-supervisor"
        )

    def start(self):
        self._thread.start()
        logger.info("IPLStreamSupervisor started.")

    def stop(self):
        """Signal the supervisor to exit cleanly."""
        self._stop_evt.set()
        logger.info("IPLStreamSupervisor stop requested.")

    # ── internal ──────────────────────────────────────────────────────────────
    def _supervisor_loop(self):
        listener = IPLStreamListener()
        backoff  = _BACKOFF_BASE

        while not self._stop_evt.is_set():
            stream_status.set(
                mode="live",
                connected=True,
                last_ok=datetime.now(timezone.utc),
                last_err=None,
                next_retry_in=0,
            )
            try:
                logger.info(
                    f"[supervisor] Connecting to Tweepy stream "
                    f"(attempt #{stream_status.attempts + 1}) …"
                )
                stream_status.set(attempts=stream_status.attempts + 1)
                listener.connect_and_filter(self._token)

                # If we reach here the stream ended without exception (server closed)
                logger.warning("[supervisor] Stream ended cleanly — scheduling reconnect.")

            except Exception as exc:
                err_msg = str(exc)
                logger.warning(f"[supervisor] Stream error: {err_msg}")
                stream_status.set(connected=False, last_err=err_msg)

            if self._stop_evt.is_set():
                break

            # Exponential back-off with jitter
            jitter  = random.uniform(-_BACKOFF_JITTER, _BACKOFF_JITTER)
            wait    = max(1, min(backoff + jitter, _BACKOFF_MAX))
            backoff = min(backoff * _BACKOFF_FACTOR, _BACKOFF_MAX)

            logger.info(
                f"[supervisor] Reconnecting in {wait:.1f}s "
                f"(next cap={backoff}s) …"
            )
            stream_status.set(mode="reconnecting", next_retry_in=round(wait))

            # Sleep in 1-second chunks so stop() is responsive
            for _ in range(int(wait)):
                if self._stop_evt.is_set():
                    break
                time.sleep(1)
                stream_status.set(next_retry_in=max(0, stream_status.next_retry_in - 1))

            # Reset backoff on a long-lived connection (>60 s uptime)
            last_ok = stream_status.last_ok
            if last_ok and (datetime.now(timezone.utc) - last_ok).seconds > 60:
                backoff = _BACKOFF_BASE
                logger.info("[supervisor] Long-lived connection — back-off reset.")

        stream_status.set(mode="idle", connected=False)
        logger.info("[supervisor] Stopped.")


# ── Tweet processing (shared by real stream + demo) ────────────────────────────
def _process_tweet(tweet_id, text, author, created_at,
                   retweet_count=0, like_count=0):
    """Store + analyse a single tweet."""
    from modules.database import insert_tweet
    from modules.sentiment import analyze_and_store

    insert_tweet(tweet_id, text, author, created_at,
                 retweet_count=retweet_count, like_count=like_count)
    analyze_and_store(tweet_id, text)
    logger.debug(f"Processed tweet {tweet_id[:8]}…")


# ── Demo data generator ─────────────────────────────────────────────────────────
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
    # Hinglish tweets (Phase 2C)
    ("Bumrah ekdum mast bowling kar raha hai aaj! 🔥 #MI zabardast!", "MI"),
    ("Yaar CSK ne kya dhamaka kiya! Dhoni bhai thoda aur khelo! #Thala", "CSK"),
    ("KKR ki fielding bilkul bekar hai aaj 😡 yeh kya chal raha hai bhai", "KKR"),
    ("Virat bhai ne century maari! Shandar! RCB waale khush ho jao 💯", "RCB"),
    ("Rohit sharma wah wah! Ek dum solid innings tha yaar! #MI best", "MI"),
    ("SRH ka batting lineup aaj bahut kharab laga yaar, haar jayenge 😢", "SRH"),
    ("Jadeja bhai ne kya catch pakda! Kamaal ka fielder hai yeh 🙌 #CSK", "CSK"),
    ("Dekhte hain aage kya hota hai, abhi match ka pata nahi #IPL2025", ""),
    ("Maxwell ne zabardast chakka maara bhai! RCB fans mast ho jao! 💥", "RCB"),
    ("Yeh toss ke baad team decide karenge, abhi sab shayad ho sakta hai", ""),
    ("KL Rahul ekdum bindaas batting kar raha hai, LSG jeetenge aaj!", "LSG"),
    ("Ruturaj out ho gaya yaar, bekar shot tha! CSK barbad ho rahi hai 😡", "CSK"),
    ("Gill ne lajawaab century maari! GT ke fans josh mein hain! 🏆", "GT"),
    ("Russell bhai khatarnak mode mein hai! KKR ka dhamaka jaari hai! 💥", "KKR"),
    ("Narine ki bowling ekdum toofani! Behtareen spell chal raha hai! 🔥", "KKR"),
]


class DemoStreamer:
    """Generates realistic synthetic tweets for dev/demo — no API key needed."""

    def __init__(self, tweets_per_second: float = 0.8):
        self.tps      = tweets_per_second
        self._counter = 0
        self._running = False

    def start(self):
        self._running = True
        stream_status.set(mode="demo", connected=True)
        logger.info("🏏 Demo streamer started — generating synthetic IPL tweets")
        threading.Thread(target=self._loop, daemon=True, name="demo-streamer").start()

    def stop(self):
        self._running = False
        stream_status.set(mode="idle", connected=False)

    def _loop(self):
        while self._running:
            self._emit()
            time.sleep(1.0 / self.tps + random.uniform(0, 1.5))

    def _emit(self):
        self._counter += 1
        text, _ = random.choice(DEMO_TWEETS)
        text += " #IPL2025"
        tweet_id   = f"demo_{self._counter:010d}_{int(time.time()*1000) % 100000}"
        author     = f"fan_{random.randint(1000, 9999)}"
        created_at = datetime.now(timezone.utc).isoformat()
        _process_tweet(tweet_id, text, author, created_at,
                       retweet_count=random.randint(0, 500),
                       like_count=random.randint(0, 2000))


# ── Public entry point ──────────────────────────────────────────────────────────
def start_streamer(bearer_token: str = None):
    """
    Start the appropriate streamer:
      • bearer_token present → IPLStreamSupervisor (real stream + auto-reconnect)
      • bearer_token absent  → DemoStreamer (synthetic tweets)
    """
    if bearer_token:
        try:
            supervisor = IPLStreamSupervisor(bearer_token)
            supervisor.start()
            logger.info("Real Tweepy stream supervisor started.")
        except Exception as exc:
            logger.warning(f"Real stream init failed ({exc}), falling back to demo.")
            DemoStreamer().start()
    else:
        DemoStreamer().start()
