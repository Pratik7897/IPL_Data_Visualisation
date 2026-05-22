"""
SQLite database layer for IPL Sentiment Dashboard.
Handles tweet storage, sentiment results, and match events.
"""
import sqlite3
import json
import os
from datetime import datetime
from pathlib import Path

DB_PATH = Path(__file__).parent.parent / "data" / "ipl_sentiment.db"


def get_connection():
    DB_PATH.parent.mkdir(exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_connection()
    cur = conn.cursor()

    cur.executescript("""
        CREATE TABLE IF NOT EXISTS tweets (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            tweet_id    TEXT UNIQUE,
            text        TEXT NOT NULL,
            author      TEXT,
            created_at  TEXT,
            ingested_at TEXT DEFAULT (datetime('now')),
            lang        TEXT DEFAULT 'en',
            retweet_count INTEGER DEFAULT 0,
            like_count  INTEGER DEFAULT 0
        );

        CREATE TABLE IF NOT EXISTS sentiments (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            tweet_id     TEXT UNIQUE,
            label        TEXT,           -- Positive / Neutral / Negative
            score        REAL,           -- confidence
            team_mention TEXT,           -- MI, CSK, RCB, etc.
            event_tag    TEXT,           -- wicket, six, four, etc.
            analyzed_at  TEXT DEFAULT (datetime('now'))
        );

        CREATE TABLE IF NOT EXISTS match_events (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            event_type TEXT,             -- wicket, six, four, wide, no_ball, over
            team       TEXT,
            player     TEXT,
            over_ball  TEXT,
            description TEXT,
            event_time TEXT DEFAULT (datetime('now'))
        );

        CREATE INDEX IF NOT EXISTS idx_tweets_created  ON tweets(created_at);
        CREATE INDEX IF NOT EXISTS idx_sent_label      ON sentiments(label);
        CREATE INDEX IF NOT EXISTS idx_sent_team       ON sentiments(team_mention);
    """)
    conn.commit()
    conn.close()


def insert_tweet(tweet_id, text, author, created_at, lang="en",
                 retweet_count=0, like_count=0):
    conn = get_connection()
    try:
        conn.execute(
            """INSERT OR IGNORE INTO tweets
               (tweet_id, text, author, created_at, lang, retweet_count, like_count)
               VALUES (?,?,?,?,?,?,?)""",
            (tweet_id, text, author, created_at, lang, retweet_count, like_count)
        )
        conn.commit()
    finally:
        conn.close()


def insert_sentiment(tweet_id, label, score, team_mention=None, event_tag=None):
    conn = get_connection()
    try:
        conn.execute(
            """INSERT OR REPLACE INTO sentiments
               (tweet_id, label, score, team_mention, event_tag)
               VALUES (?,?,?,?,?)""",
            (tweet_id, label, score, team_mention, event_tag)
        )
        conn.commit()
    finally:
        conn.close()


def insert_event(event_type, team="", player="", over_ball="", description=""):
    conn = get_connection()
    try:
        conn.execute(
            """INSERT INTO match_events (event_type, team, player, over_ball, description)
               VALUES (?,?,?,?,?)""",
            (event_type, team, player, over_ball, description)
        )
        conn.commit()
    finally:
        conn.close()


def fetch_recent_tweets_with_sentiment(limit=200):
    conn = get_connection()
    try:
        rows = conn.execute(
            """SELECT t.tweet_id, t.text, t.author, t.created_at,
                      t.retweet_count, t.like_count,
                      s.label, s.score, s.team_mention, s.event_tag
               FROM tweets t
               LEFT JOIN sentiments s ON t.tweet_id = s.tweet_id
               ORDER BY t.ingested_at DESC
               LIMIT ?""",
            (limit,)
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def fetch_sentiment_timeseries(minutes=30):
    conn = get_connection()
    try:
        rows = conn.execute(
            """SELECT s.analyzed_at, s.label, s.score, s.team_mention
               FROM sentiments s
               JOIN tweets t ON s.tweet_id = t.tweet_id
               WHERE s.analyzed_at >= datetime('now', ? || ' minutes')
               ORDER BY s.analyzed_at ASC""",
            (f"-{minutes}",)
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def fetch_match_events(minutes=60):
    conn = get_connection()
    try:
        rows = conn.execute(
            """SELECT * FROM match_events
               WHERE event_time >= datetime('now', ? || ' minutes')
               ORDER BY event_time ASC""",
            (f"-{minutes}",)
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def fetch_team_sentiment_counts():
    conn = get_connection()
    try:
        rows = conn.execute(
            """SELECT team_mention, label, COUNT(*) as count
               FROM sentiments
               WHERE team_mention IS NOT NULL AND team_mention != ''
               GROUP BY team_mention, label"""
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def fetch_all_tweet_texts(sentiment_filter=None):
    conn = get_connection()
    try:
        if sentiment_filter:
            rows = conn.execute(
                """SELECT t.text FROM tweets t
                   JOIN sentiments s ON t.tweet_id = s.tweet_id
                   WHERE s.label = ?""",
                (sentiment_filter,)
            ).fetchall()
        else:
            rows = conn.execute("SELECT text FROM tweets").fetchall()
        return [r["text"] for r in rows]
    finally:
        conn.close()


def get_tweet_volume_per_minute(minutes=30):
    conn = get_connection()
    try:
        rows = conn.execute(
            """SELECT strftime('%Y-%m-%dT%H:%M', ingested_at) as minute,
                      COUNT(*) as count
               FROM tweets
               WHERE ingested_at >= datetime('now', ? || ' minutes')
               GROUP BY minute
               ORDER BY minute ASC""",
            (f"-{minutes}",)
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()
