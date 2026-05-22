# 🏏 IPL Sentiment Live Dashboard

Real-time sentiment analysis and live dashboard for IPL matches — tracks how Twitter/X reacts ball-by-ball to wickets, sixes, DRS drama, and more.

---

## Architecture

```
X API v2 (Tweepy filtered stream)
        │
        ▼
 modules/streamer.py  ─── keyword filters: #IPL2026, #MIvsCSK, player names
        │
        ▼
 modules/sentiment.py ─── cardiffnlp/twitter-roberta-base-sentiment
        │                  + team/event entity detection
        ▼
 SQLite (data/ipl_sentiment.db)
        │
        ▼
 modules/charts.py    ─── Plotly figure builders
        │
        ▼
 app.py (Plotly Dash) ─── dcc.Interval 5-sec refresh → live dashboard
```

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure (optional — runs in demo mode without keys)

```bash
cp .env.example .env
# Edit .env and add TWITTER_BEARER_TOKEN if you have X API v2 access
```

### 3. Run

```bash
python app.py
# Open http://localhost:8050
```

Demo mode streams synthetic tweets automatically — no API key needed.

---

## Dashboard Panels

| Panel | Description |
|-------|-------------|
| **Sentiment Timeline** | Rolling 1-min & 5-min averages per polarity, annotated with match events |
| **Tweet Volume** | Per-minute histogram coloured by intensity; event spike markers |
| **Team Sentiment Bars** | Grouped bars — Positive/Neutral/Negative split per IPL team |
| **Overall Mood Donut** | Live breakdown of the dominant sentiment |
| **Word Clouds** | Most-used terms per sentiment class (auto-regenerated) |
| **Live Tweet Feed** | 10 most recent tweets with sentiment badge + event tag |
| **Event Logger** | Manually log match events (wicket, six, four …) to annotate charts |

---

## Real X API v2 Setup

1. Apply for a developer account at [developer.twitter.com](https://developer.twitter.com)
2. Create a project + app → copy the **Bearer Token**
3. Add to `.env`:
   ```
   TWITTER_BEARER_TOKEN=AAAAAAAAAAAAAAAAAAAAAxxxxxxxxxx
   ```
4. Restart the app — the real filtered stream will replace the demo generator.

Stream rules (in `modules/streamer.py`) filter for:
- `#IPL2026` main hashtag
- Match-specific tags like `#MIvsCSK`
- Player names: `#Kohli`, `#Bumrah`, `#Dhoni`, …

---

## Optional: Fine-tune the Model

For better accuracy on cricket slang, emojis, and Hinglish:

```bash
# Prepare a CSV:  text,label
# (collect & label ~1000+ IPL tweets manually or via weak supervision)

python finetune.py \
  --dataset data/ipl_tweets.csv \
  --output  models/ipl-roberta \
  --epochs  3

# Use the fine-tuned model:
export MODEL_PATH=models/ipl-roberta
python app.py
```

---

## Docker

```bash
docker-compose up --build
# Dashboard at http://localhost:8050
```

---

## File Structure

```
ipl_sentiment/
├── app.py                  # Dash application + callbacks
├── finetune.py             # Optional model fine-tuning script
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── .env.example
├── modules/
│   ├── database.py         # SQLite schema, queries
│   ├── sentiment.py        # HuggingFace inference + entity detection
│   ├── streamer.py         # Tweepy v2 stream + demo generator
│   └── charts.py           # Plotly figure builders + word clouds
└── data/
    └── ipl_sentiment.db    # Auto-created on first run
```

---

## Extending the System

### Live ball-by-ball event feed
Replace the manual event logger with a Cricbuzz / ESPNcricinfo poller:

```python
# modules/events_poller.py (not included — requires paid API)
import requests, time
from modules.database import insert_event

def poll_cricbuzz(match_id, api_key, interval=10):
    while True:
        r = requests.get(
            f"https://api.cricapi.com/v1/cricScore?apikey={api_key}&id={match_id}"
        )
        data = r.json()
        # parse last ball, map to event_type, call insert_event(...)
        time.sleep(interval)
```

### Redis queue (high-volume)
For matches with 10k+ tweets/min, swap SQLite writes in `streamer.py` for a
Redis list push, and run a separate worker that pops and inserts in batches.

```python
import redis
r = redis.Redis()
r.lpush("tweet_queue", json.dumps(tweet_dict))

# worker.py
while True:
    raw = r.rpop("tweet_queue")
    if raw:
        t = json.loads(raw)
        insert_tweet(**t)
        analyze_and_store(t["tweet_id"], t["text"])
```
