"""
modules/ai/__init__.py
Phase 4 — AI/ML feature pack.

Three isolated, additive modules:
  predictor.py    → Feature 1: Match outcome prediction (RandomForest / XGBoost)
  summarizer.py   → Feature 2: AI match summary generator (Gemini / OpenAI / template)
  player_index.py → Feature 3: Player popularity index

None of these modules import from or modify the existing pipeline.
They only READ from modules.database via the public fetch_* functions.
"""
