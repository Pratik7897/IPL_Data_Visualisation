#!/usr/bin/env python3
"""
Phase 3A — SQLite → PostgreSQL data migration script.

Run ONCE to move all historical data from the local SQLite file to
a running PostgreSQL instance.

Usage:
    # 1. Make sure PostgreSQL is running (docker compose up postgres -d)
    # 2. Set DATABASE_URL in your shell:
    export DATABASE_URL=postgresql://ipl_user:ipl_pass@localhost:5432/ipl_sentiment

    # 3. Run the migration:
    python scripts/migrate_sqlite_to_pg.py

    # Optional: specify a custom SQLite path
    python scripts/migrate_sqlite_to_pg.py --sqlite data/ipl_sentiment.db
"""
import os
import sys
import sqlite3
import argparse
import logging
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
)
log = logging.getLogger("migrate")

# Add project root to path so we can import database module
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))


def parse_args():
    p = argparse.ArgumentParser(description="Migrate SQLite → PostgreSQL")
    p.add_argument(
        "--sqlite",
        default=str(ROOT / "data" / "ipl_sentiment.db"),
        help="Path to the SQLite database file",
    )
    p.add_argument(
        "--batch", type=int, default=500,
        help="Insert batch size (default: 500)",
    )
    return p.parse_args()


def connect_sqlite(path: str):
    if not Path(path).exists():
        log.error(f"SQLite file not found: {path}")
        sys.exit(1)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    return conn


def connect_postgres(url: str):
    try:
        import psycopg2
        import psycopg2.extras
        conn = psycopg2.connect(url, cursor_factory=psycopg2.extras.RealDictCursor)
        conn.autocommit = False
        return conn
    except ImportError:
        log.error("psycopg2 not installed. Run: pip install psycopg2-binary")
        sys.exit(1)
    except Exception as e:
        log.error(f"Could not connect to PostgreSQL: {e}")
        sys.exit(1)


def migrate_table(src, dst, table: str, insert_sql: str,
                  row_to_tuple, batch_size=500):
    src_cur = src.cursor()
    src_cur.execute(f"SELECT * FROM {table} ORDER BY id ASC")

    total = 0
    batch = []

    import psycopg2.extras as pgextras

    def flush():
        nonlocal total
        if not batch:
            return
        with dst.cursor() as cur:
            pgextras.execute_values(cur, insert_sql, batch, page_size=batch_size)
        dst.commit()
        total += len(batch)
        batch.clear()
        log.info(f"  [{table}] {total} rows inserted …")

    for row in src_cur:
        batch.append(row_to_tuple(row))
        if len(batch) >= batch_size:
            flush()

    flush()  # remaining rows
    log.info(f"  [{table}] ✅  {total} rows migrated.")
    return total


def main():
    args = parse_args()

    db_url = os.getenv("DATABASE_URL", "")
    if not db_url.startswith("postgres"):
        log.error(
            "DATABASE_URL is not set or is not a PostgreSQL URL.\n"
            "Set it before running this script:\n"
            "  export DATABASE_URL=postgresql://user:pass@host:5432/dbname"
        )
        sys.exit(1)

    log.info(f"Source : SQLite   → {args.sqlite}")
    log.info(f"Target : Postgres → {db_url.split('@')[-1]}")
    log.info("─" * 60)

    src = connect_sqlite(args.sqlite)
    dst = connect_postgres(db_url)

    # Ensure schema exists in PG (idempotent)
    from modules.database import _SCHEMA_POSTGRES
    with dst.cursor() as cur:
        cur.execute(_SCHEMA_POSTGRES)
    dst.commit()
    log.info("PostgreSQL schema verified.")

    # ── tweets ──
    log.info("Migrating tweets …")
    migrate_table(
        src, dst, "tweets",
        """INSERT INTO tweets
               (tweet_id, text, author, created_at, ingested_at,
                lang, retweet_count, like_count)
           VALUES %s
           ON CONFLICT (tweet_id) DO NOTHING""",
        lambda r: (
            r["tweet_id"], r["text"], r["author"],
            r["created_at"], r["ingested_at"],
            r["lang"], r["retweet_count"], r["like_count"],
        ),
        batch_size=args.batch,
    )

    # ── sentiments ──
    log.info("Migrating sentiments …")
    migrate_table(
        src, dst, "sentiments",
        """INSERT INTO sentiments
               (tweet_id, label, score, team_mention, event_tag, analyzed_at)
           VALUES %s
           ON CONFLICT (tweet_id) DO NOTHING""",
        lambda r: (
            r["tweet_id"], r["label"], r["score"],
            r["team_mention"], r["event_tag"], r["analyzed_at"],
        ),
        batch_size=args.batch,
    )

    # ── match_events ──
    log.info("Migrating match_events …")
    migrate_table(
        src, dst, "match_events",
        """INSERT INTO match_events
               (event_type, team, player, over_ball, description, event_time)
           VALUES %s""",
        lambda r: (
            r["event_type"], r["team"], r["player"],
            r["over_ball"], r["description"], r["event_time"],
        ),
        batch_size=args.batch,
    )

    src.close()
    dst.close()
    log.info("─" * 60)
    log.info("Migration complete! ✅")


if __name__ == "__main__":
    main()
