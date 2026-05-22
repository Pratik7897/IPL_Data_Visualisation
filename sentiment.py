"""
Sentiment analysis using cardiffnlp/twitter-roberta-base-sentiment.
Handles batching, caching, and IPL entity detection.
"""
import re
import logging
from typing import Optional

logger = logging.getLogger(__name__)

# ── IPL Team & Player keyword maps ────────────────────────────────────────────
TEAM_KEYWORDS = {
    "MI":  ["mumbai indians", "mumbaiindians", "#mi ", " mi ", "rohit sharma",
            "hardik pandya", "bumrah", "suryakumar"],
    "CSK": ["chennai super kings", "csk", "dhoni", "msd", "thala", "jadeja",
            "ruturaj", "gaikwad"],
    "RCB": ["royal challengers", "rcb", "virat", "kohli", "maxwell", "faf"],
    "KKR": ["kolkata knight riders", "kkr", "russell", "narine", "iyer",
            "venkatesh"],
    "SRH": ["sunrisers hyderabad", "srh", "head", "abhishek sharma",
            "cummins", "klaasen"],
    "RR":  ["rajasthan royals", "rr", "buttler", "sanju samson", "ashwin"],
    "PBKS":["punjab kings", "pbks", "shashank", "prabhsimran"],
    "LSG": ["lucknow super giants", "lsg", "kl rahul", "stoinis"],
    "GT":  ["gujarat titans", "gt", "shubman gill", "mohit sharma"],
    "DC":  ["delhi capitals", "dc", "axar patel", "kuldeep"],
}

EVENT_KEYWORDS = {
    "wicket":  ["wicket", "out!", "bowled", "lbw", "caught", "stumped",
                "runout", "run out", "gone!", "dismissed", "duck"],
    "six":     ["six!", "sixer", "sixxx", "maximum", "gone for six",
                "out of the ground", "💥", "🔥", "🚀"],
    "four":    ["four!", "boundary!", "hits it to the fence", "races away"],
    "wide":    ["wide ball", "wide!", "wiiide"],
    "no_ball": ["no ball", "no-ball", "noball", "free hit"],
    "over":    ["end of over", "over complete", "maiden over"],
}

NOISE_PATTERN = re.compile(
    r"http\S+|www\S+|@\w+|#|RT\s|[^\w\s.,!?'\U0001F300-\U0001FFFF]",
    re.UNICODE,
)


def clean_tweet(text: str) -> str:
    text = NOISE_PATTERN.sub(" ", text)
    return " ".join(text.split())[:512]


def detect_team(text: str) -> Optional[str]:
    lower = text.lower()
    scores = {}
    for team, kws in TEAM_KEYWORDS.items():
        scores[team] = sum(1 for kw in kws if kw in lower)
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else None


def detect_event(text: str) -> Optional[str]:
    lower = text.lower()
    for event, kws in EVENT_KEYWORDS.items():
        if any(kw in lower for kw in kws):
            return event
    return None


# ── Model loading (lazy, singleton) ───────────────────────────────────────────
_pipeline = None


def get_pipeline():
    global _pipeline
    if _pipeline is None:
        try:
            from transformers import pipeline as hf_pipeline
            logger.info("Loading cardiffnlp/twitter-roberta-base-sentiment …")
            _pipeline = hf_pipeline(
                "text-classification",
                model="cardiffnlp/twitter-roberta-base-sentiment",
                tokenizer="cardiffnlp/twitter-roberta-base-sentiment",
                top_k=1,
                truncation=True,
                max_length=128,
            )
            logger.info("Model loaded.")
        except Exception as exc:
            logger.warning(f"Could not load HuggingFace model: {exc}. "
                           "Falling back to keyword-based sentiment.")
            _pipeline = "fallback"
    return _pipeline


# Fallback: simple keyword sentiment
_POS_WORDS = {"great", "amazing", "brilliant", "love", "won", "win",
              "six", "century", "milestone", "superb", "fantastic",
              "beast", "king", "legend", "fire", "🔥", "💯", "❤"}
_NEG_WORDS = {"lost", "loss", "wicket", "out", "terrible", "awful",
              "disappointed", "disaster", "dropped", "injured", "bad",
              "worst", "😢", "😡", "💔", "👎"}

LABEL_MAP = {
    "LABEL_0": "Negative",
    "LABEL_1": "Neutral",
    "LABEL_2": "Positive",
}


def _fallback_sentiment(text: str):
    lower = set(text.lower().split())
    pos = len(lower & _POS_WORDS)
    neg = len(lower & _NEG_WORDS)
    if pos > neg:
        return "Positive", 0.65
    elif neg > pos:
        return "Negative", 0.65
    return "Neutral", 0.60


def analyze(text: str):
    """Return (label, confidence_score) for a single tweet text."""
    pipe = get_pipeline()
    cleaned = clean_tweet(text)

    if pipe == "fallback" or not cleaned:
        return _fallback_sentiment(text)

    try:
        result = pipe(cleaned)[0]
        raw_label = result["label"]
        score = round(result["score"], 4)
        label = LABEL_MAP.get(raw_label, raw_label)
        return label, score
    except Exception as exc:
        logger.debug(f"Inference error: {exc}")
        return _fallback_sentiment(text)


def analyze_and_store(tweet_id: str, text: str):
    """Full pipeline: analyze + detect team/event + persist."""
    from modules.database import insert_sentiment

    label, score = analyze(text)
    team = detect_team(text)
    event = detect_event(text)
    insert_sentiment(tweet_id, label, score, team, event)
    return label, score, team, event
