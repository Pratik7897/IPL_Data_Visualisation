"""
Phase 3A — Dual-mode database layer.

Supports SQLite (dev/local) and PostgreSQL (production) via a single
DATABASE_URL environment variable.

  SQLite (default, zero config):
      DATABASE_URL is unset  →  uses ./data/ipl_sentiment.db

  PostgreSQL (production):
      DATABASE_URL=postgresql://user:pass@host:5432/dbname

The module auto-detects the backend from the URL scheme and uses:
  • sqlite3          for SQLite
  • psycopg2         for PostgreSQL  (install: psycopg2-binary)

All public functions share the same signature so app.py needs zero changes.

Postgres-specific improvements over SQLite:
  • Connection pool via threading.local() — no "database is locked" errors
  • ON CONFLICT DO NOTHING / DO UPDATE replace INSERT OR IGNORE / REPLACE
  • now() replaces datetime('now')
  • %s placeholders replace ? placeholders
  • RETURNING id for generated keys
"""
import os
import sqlite3
import logging
import threading
from pathlib import Path
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

# ── Backend detection ─────────────────────────────────────────────────────────
_DATABASE_URL = os.getenv("DATABASE_URL", "")
_USE_POSTGRES  = _DATABASE_URL.startswith("postgresql") or \
                 _DATABASE_URL.startswith("postgres")

if _USE_POSTGRES:
    try:
        import psycopg2
        import psycopg2.extras
        logger.info(f"[database] Backend: PostgreSQL  ({_DATABASE_URL.split('@')[-1]})")
    except ImportError:
        logger.error(
            "[database] psycopg2 not installed! "
            "Run: pip install psycopg2-binary\n"
            "Falling back to SQLite."
        )
        _USE_POSTGRES = False

if not _USE_POSTGRES:
    _SQLITE_PATH = Path(__file__).parent.parent / "data" / "ipl_sentiment.db"
    logger.info(f"[database] Backend: SQLite  ({_SQLITE_PATH})")


# ── SQLite helpers ─────────────────────────────────────────────────────────────
def _sqlite_conn():
    _SQLITE_PATH.parent.mkdir(exist_ok=True)
    conn = sqlite3.connect(str(_SQLITE_PATH), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


# ── PostgreSQL helpers ─────────────────────────────────────────────────────────
# Use a per-thread connection pool to avoid locking.
_pg_local = threading.local()


def _pg_conn():
    if not hasattr(_pg_local, "conn") or _pg_local.conn.closed:
        _pg_local.conn = psycopg2.connect(
            _DATABASE_URL,
            cursor_factory=psycopg2.extras.RealDictCursor,
        )
        _pg_local.conn.autocommit = False
    return _pg_local.conn


# ── Unified get_connection() ───────────────────────────────────────────────────
def get_connection():
    return _pg_conn() if _USE_POSTGRES else _sqlite_conn()


# ── Placeholder helper ─────────────────────────────────────────────────────────
# SQLite uses ?  PostgreSQL uses %s
_PH = "%s" if _USE_POSTGRES else "?"


def _ph(n: int) -> str:
    """Return n comma-separated placeholders, e.g. _ph(3) → '?, ?, ?'."""
    return ", ".join([_PH] * n)


# ── now() helper ───────────────────────────────────────────────────────────────
def _now_expr() -> str:
    """SQL fragment for current timestamp in the right dialect."""
    return "NOW()" if _USE_POSTGRES else "datetime('now')"


# ──────────────────────────────────────────────────────────────────────────────
# SCHEMA
# ──────────────────────────────────────────────────────────────────────────────
_SCHEMA_SQLITE = """
CREATE TABLE IF NOT EXISTS tweets (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    tweet_id      TEXT UNIQUE,
    text          TEXT NOT NULL,
    author        TEXT,
    created_at    TEXT,
    ingested_at   TEXT DEFAULT (datetime('now')),
    lang          TEXT DEFAULT 'en',
    retweet_count INTEGER DEFAULT 0,
    like_count    INTEGER DEFAULT 0
);

CREATE TABLE IF NOT EXISTS sentiments (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    tweet_id     TEXT UNIQUE,
    label        TEXT,
    score        REAL,
    team_mention TEXT,
    event_tag    TEXT,
    analyzed_at  TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS match_events (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    event_type  TEXT,
    team        TEXT,
    player      TEXT,
    over_ball   TEXT,
    description TEXT,
    event_time  TEXT DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_tweets_created ON tweets(created_at);
CREATE INDEX IF NOT EXISTS idx_tweets_ingested ON tweets(ingested_at);
CREATE INDEX IF NOT EXISTS idx_sent_label      ON sentiments(label);
CREATE INDEX IF NOT EXISTS idx_sent_team       ON sentiments(team_mention);
CREATE INDEX IF NOT EXISTS idx_sent_analyzed   ON sentiments(analyzed_at);
"""

_SCHEMA_POSTGRES = """
CREATE TABLE IF NOT EXISTS tweets (
    id            SERIAL PRIMARY KEY,
    tweet_id      TEXT UNIQUE,
    text          TEXT NOT NULL,
    author        TEXT,
    created_at    TIMESTAMPTZ,
    ingested_at   TIMESTAMPTZ DEFAULT NOW(),
    lang          TEXT DEFAULT 'en',
    retweet_count INTEGER DEFAULT 0,
    like_count    INTEGER DEFAULT 0
);

CREATE TABLE IF NOT EXISTS sentiments (
    id           SERIAL PRIMARY KEY,
    tweet_id     TEXT UNIQUE,
    label        TEXT,
    score        REAL,
    team_mention TEXT,
    event_tag    TEXT,
    analyzed_at  TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS match_events (
    id          SERIAL PRIMARY KEY,
    event_type  TEXT,
    team        TEXT,
    player      TEXT,
    over_ball   TEXT,
    description TEXT,
    event_time  TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_tweets_created  ON tweets(created_at);
CREATE INDEX IF NOT EXISTS idx_tweets_ingested ON tweets(ingested_at);
CREATE INDEX IF NOT EXISTS idx_sent_label      ON sentiments(label);
CREATE INDEX IF NOT EXISTS idx_sent_team       ON sentiments(team_mention);
CREATE INDEX IF NOT EXISTS idx_sent_analyzed   ON sentiments(analyzed_at);
"""


def init_db():
    """Create tables and indexes. Safe to call on every startup."""
    if _USE_POSTGRES:
        conn = _pg_conn()
        cur  = conn.cursor()
        cur.execute(_SCHEMA_POSTGRES)
        conn.commit()
        logger.info("[database] PostgreSQL schema ready.")
    else:
        conn = _sqlite_conn()
        conn.executescript(_SCHEMA_SQLITE)
        conn.commit()
        conn.close()
        logger.info("[database] SQLite schema ready.")


# ──────────────────────────────────────────────────────────────────────────────
# WRITE OPERATIONS
# ──────────────────────────────────────────────────────────────────────────────
def insert_tweet(tweet_id, text, author, created_at, lang="en",
                 retweet_count=0, like_count=0):
    if _USE_POSTGRES:
        conn = _pg_conn()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    f"""INSERT INTO tweets
                           (tweet_id, text, author, created_at, lang,
                            retweet_count, like_count)
                        VALUES ({_ph(7)})
                        ON CONFLICT (tweet_id) DO NOTHING""",
                    (tweet_id, text, author, created_at, lang,
                     retweet_count, like_count),
                )
            conn.commit()
        except Exception as e:
            conn.rollback()
            logger.warning(f"[database] insert_tweet error: {e}")
    else:
        conn = _sqlite_conn()
        try:
            conn.execute(
                """INSERT OR IGNORE INTO tweets
                   (tweet_id, text, author, created_at, lang,
                    retweet_count, like_count)
                   VALUES (?,?,?,?,?,?,?)""",
                (tweet_id, text, author, created_at, lang,
                 retweet_count, like_count),
            )
            conn.commit()
        finally:
            conn.close()


def insert_sentiment(tweet_id, label, score, team_mention=None, event_tag=None):
    if _USE_POSTGRES:
        conn = _pg_conn()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    f"""INSERT INTO sentiments
                           (tweet_id, label, score, team_mention, event_tag)
                        VALUES ({_ph(5)})
                        ON CONFLICT (tweet_id) DO UPDATE
                           SET label=EXCLUDED.label,
                               score=EXCLUDED.score,
                               team_mention=EXCLUDED.team_mention,
                               event_tag=EXCLUDED.event_tag,
                               analyzed_at=NOW()""",
                    (tweet_id, label, score, team_mention, event_tag),
                )
            conn.commit()
        except Exception as e:
            conn.rollback()
            logger.warning(f"[database] insert_sentiment error: {e}")
    else:
        conn = _sqlite_conn()
        try:
            conn.execute(
                """INSERT OR REPLACE INTO sentiments
                   (tweet_id, label, score, team_mention, event_tag)
                   VALUES (?,?,?,?,?)""",
                (tweet_id, label, score, team_mention, event_tag),
            )
            conn.commit()
        finally:
            conn.close()


def insert_event(event_type, team="", player="", over_ball="", description=""):
    if _USE_POSTGRES:
        conn = _pg_conn()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    f"""INSERT INTO match_events
                           (event_type, team, player, over_ball, description)
                        VALUES ({_ph(5)})""",
                    (event_type, team, player, over_ball, description),
                )
            conn.commit()
        except Exception as e:
            conn.rollback()
            logger.warning(f"[database] insert_event error: {e}")
    else:
        conn = _sqlite_conn()
        try:
            conn.execute(
                """INSERT INTO match_events
                   (event_type, team, player, over_ball, description)
                   VALUES (?,?,?,?,?)""",
                (event_type, team, player, over_ball, description),
            )
            conn.commit()
        finally:
            conn.close()


# ──────────────────────────────────────────────────────────────────────────────
# READ OPERATIONS
# ──────────────────────────────────────────────────────────────────────────────
def _fetchall(query: str, params: tuple = ()):
    """Execute a SELECT and return list of dicts — works for both backends."""
    if _USE_POSTGRES:
        conn = _pg_conn()
        with conn.cursor() as cur:
            cur.execute(query, params)
            return [dict(r) for r in cur.fetchall()]
    else:
        conn = _sqlite_conn()
        try:
            rows = conn.execute(query, params).fetchall()
            return [dict(r) for r in rows]
        finally:
            conn.close()


def fetch_recent_tweets_with_sentiment(limit=200):
    q = """SELECT t.tweet_id, t.text, t.author, t.created_at,
                  t.retweet_count, t.like_count,
                  s.label, s.score, s.team_mention, s.event_tag
           FROM tweets t
           LEFT JOIN sentiments s ON t.tweet_id = s.tweet_id
           ORDER BY t.ingested_at DESC
           LIMIT {ph}""".format(ph=_PH)
    return _fetchall(q, (limit,))


def fetch_sentiment_timeseries(minutes=30):
    if _USE_POSTGRES:
        q = """SELECT s.analyzed_at, s.label, s.score, s.team_mention
               FROM sentiments s
               JOIN tweets t ON s.tweet_id = t.tweet_id
               WHERE s.analyzed_at >= NOW() - INTERVAL %s
               ORDER BY s.analyzed_at ASC"""
        return _fetchall(q, (f"{minutes} minutes",))
    else:
        q = """SELECT s.analyzed_at, s.label, s.score, s.team_mention
               FROM sentiments s
               JOIN tweets t ON s.tweet_id = t.tweet_id
               WHERE s.analyzed_at >= datetime('now', ? || ' minutes')
               ORDER BY s.analyzed_at ASC"""
        return _fetchall(q, (f"-{minutes}",))


def fetch_match_events(minutes=60):
    if _USE_POSTGRES:
        q = """SELECT * FROM match_events
               WHERE event_time >= NOW() - INTERVAL %s
               ORDER BY event_time ASC"""
        return _fetchall(q, (f"{minutes} minutes",))
    else:
        q = """SELECT * FROM match_events
               WHERE event_time >= datetime('now', ? || ' minutes')
               ORDER BY event_time ASC"""
        return _fetchall(q, (f"-{minutes}",))


def fetch_team_sentiment_counts():
    q = """SELECT team_mention, label, COUNT(*) as count
           FROM sentiments
           WHERE team_mention IS NOT NULL AND team_mention != ''
           GROUP BY team_mention, label"""
    return _fetchall(q)


def fetch_all_tweet_texts(sentiment_filter=None):
    if sentiment_filter:
        q = f"""SELECT t.text FROM tweets t
                JOIN sentiments s ON t.tweet_id = s.tweet_id
                WHERE s.label = {_PH}"""
        rows = _fetchall(q, (sentiment_filter,))
    else:
        rows = _fetchall("SELECT text FROM tweets")
    return [r["text"] for r in rows]


def get_tweet_volume_per_minute(minutes=30):
    if _USE_POSTGRES:
        q = """SELECT to_char(date_trunc('minute', ingested_at),
                              'YYYY-MM-DD"T"HH24:MI') AS minute,
                      COUNT(*) AS count
               FROM tweets
               WHERE ingested_at >= NOW() - INTERVAL %s
               GROUP BY minute
               ORDER BY minute ASC"""
        return _fetchall(q, (f"{minutes} minutes",))
    else:
        q = """SELECT strftime('%Y-%m-%dT%H:%M', ingested_at) as minute,
                      COUNT(*) as count
               FROM tweets
               WHERE ingested_at >= datetime('now', ? || ' minutes')
               GROUP BY minute
               ORDER BY minute ASC"""
        return _fetchall(q, (f"-{minutes}",))
