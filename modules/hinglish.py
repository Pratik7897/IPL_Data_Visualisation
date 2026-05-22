"""
Phase 2C — Hinglish keyword augmentation layer.

Many IPL tweets mix Hindi/Marathi/Tamil words into English sentences (Hinglish).
The RoBERTa model was trained on English tweets and will misclassify these.

This module provides:
  1. A fast keyword-based pre-screen that maps Hinglish slang → sentiment label
  2. A fallback to the RoBERTa model when no keyword matches

Usage:
    The hinglish_override() function is called from modules/sentiment.py
    BEFORE the model inference step.
"""

# ── Positive Hinglish / desi slang ─────────────────────────────────────────────
POSITIVE_KEYWORDS = {
    # General praise
    "ekdum", "mast", "bindaas", "badhiya", "zabardast", "shandar", "kamaal",
    "behtareen", "lajawaab", "wah", "waah", "kya baat", "jai", "jeet",
    "sahi", "solid", "dhamaka", "toofani", "chhaap", "josh", "andar",
    "ufff", "arrey wah", "bhai kya", "boss", "bhai", "yaar", "yar",
    # Cricket specific
    "chapka", "century maar", "chauka", "chakka", "sixer", "chaka",
    "khidmat", "jeetenge", "jeeta", "jeet gaya", "jeet gayi", "hamare",
    "khatarnak", "chaka diya", "shot maar", "dabaake", "maarke",
    # Team chants
    "csk zindabad", "mi zindabad", "rcb zindabad", "kkr zindabad",
    "thala", "thalaiva", "dhoni bhai", "virat bhai", "rohit bhai",
    "bumrah best", "jadeja rocks",
}

# ── Negative Hinglish / desi slang ─────────────────────────────────────────────
NEGATIVE_KEYWORDS = {
    # General frustration
    "bekar", "bakwaas", "ghatiya", "chha gaya", "ch gaya", "kya yaar",
    "kya baat hai nahi", "haaro", "haar gaye", "haar gayi", "haar gaya",
    "loose", "nalayak", "ajeeb", "galat", "sharam", "sharm karo",
    "kharab", "bura", "pagal", "kambakht", "bechaara", "bechara",
    # Cricket frustration
    "drop kiya", "dropped", "chuka", "chook gaya", "bowled out", "run out",
    "wicket gir", "wicket gira", "pakad nahi", "fielding bekar",
    "batting bekar", "bowling bekar", "harega", "haar jayega",
    "kuch nahi hoga", "barbad", "tabahi",
    # Abusive-coded (cleaned, no profanity)
    "ullu", "bewakoof", "nikamme", "nikamma",
}

# ── Neutral / match commentary Hinglish ────────────────────────────────────────
NEUTRAL_KEYWORDS = {
    "abhi", "dekhte hain", "pata nahi", "shayad", "lagta hai",
    "ho sakta hai", "aage dekhenge", "toss", "toss jeet",
    "playing xi", "playing 11", "pitch report", "aaj ka match",
    "khelenge", "overs baaki", "run chahiye", "run rate",
}


def hinglish_override(text: str) -> str | None:
    """
    Scan tweet text for Hinglish keywords.
    Returns 'Positive', 'Negative', or 'Neutral' if a keyword is matched,
    otherwise returns None (fall through to RoBERTa model).

    Scoring: count pos/neg/neutral hits and return majority.
    """
    low = text.lower()

    pos = sum(1 for kw in POSITIVE_KEYWORDS if kw in low)
    neg = sum(1 for kw in NEGATIVE_KEYWORDS if kw in low)
    neu = sum(1 for kw in NEUTRAL_KEYWORDS   if kw in low)

    # Need at least one clear signal to override
    total = pos + neg + neu
    if total == 0:
        return None  # no Hinglish detected, let the model decide

    if pos > neg and pos > neu:
        return "Positive"
    if neg > pos and neg >= neu:
        return "Negative"
    if neu > pos and neu > neg:
        return "Neutral"

    # Tie → let model decide
    return None
