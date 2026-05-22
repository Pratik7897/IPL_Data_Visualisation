"""
Phase 2B — Live ball-by-ball event poller using CricAPI.
https://cricapi.com  (free tier: 100 calls/day → poll every 30s during a match)

Environment variables:
  CRICAPI_KEY  – your CricAPI key (if blank, the poller is a no-op)
  MATCH_ID     – CricAPI match UUID for the live match

Usage (called once from app.py main()):
    from modules.events_poller import start_poller
    start_poller()          # reads env vars internally
"""
import os
import time
import logging
import threading
from datetime import datetime, timezone

import requests

logger = logging.getLogger(__name__)

# How often we hit the API (seconds). 30 s → ~120 calls/hour, ~960/day
_POLL_INTERVAL = int(os.getenv("CRICAPI_POLL_INTERVAL", "30"))
_API_BASE      = "https://api.cricapi.com/v1"
_TIMEOUT       = 10  # seconds per request


# ── Poller status (readable from dashboard) ─────────────────────────────────────
class PollerStatus:
    """Thread-safe singleton for poller health info."""

    def __init__(self):
        self._lock    = threading.Lock()
        self.running  = False
        self.calls    = 0            # total API calls made
        self.credits  = None         # remaining credits (from API response)
        self.last_ok  = None         # datetime of last successful call
        self.last_err = None         # last error string
        self.events_logged = 0       # total events auto-logged

    def set(self, **kwargs):
        with self._lock:
            for k, v in kwargs.items():
                setattr(self, k, v)

    def snapshot(self) -> dict:
        with self._lock:
            return {k: v for k, v in vars(self).items() if not k.startswith("_")}


poller_status = PollerStatus()


# ── API helpers ─────────────────────────────────────────────────────────────────
def _get_live_score(api_key: str, match_id: str) -> dict:
    """Fetch live score data for a match. Returns parsed JSON or raises."""
    resp = requests.get(
        f"{_API_BASE}/match_info",
        params={"apikey": api_key, "id": match_id},
        timeout=_TIMEOUT,
    )
    resp.raise_for_status()
    data = resp.json()
    if not data.get("status") == "success":
        raise ValueError(f"API error: {data.get('status')} — {data.get('reason','')}")
    return data


def _extract_score_state(data: dict) -> dict:
    """Normalise API response to a simple comparable state dict."""
    info  = data.get("data", {})
    score = (info.get("score") or [{}])
    # Take batting team's current inning
    current = score[-1] if score else {}
    return {
        "wickets": current.get("wickets", 0),
        "runs":    current.get("r", 0),
        "overs":   current.get("o", 0.0),
        "inning":  current.get("inning", ""),
        "credits": data.get("info", {}).get("credits", None),
    }


def _detect_events(prev: dict, curr: dict) -> list:
    """
    Compare two consecutive state snapshots and return a list of detected events.
    Each event: {"type": str, "description": str}
    """
    events = []

    # New wicket
    prev_w = prev.get("wickets", 0) or 0
    curr_w = curr.get("wickets", 0) or 0
    if curr_w > prev_w:
        diff = curr_w - prev_w
        events.extend([{"type": "wicket", "description": f"Wicket! ({curr_w} total)"}] * diff)

    # New over completed (integer part increased)
    prev_ov = int(float(prev.get("overs", 0) or 0))
    curr_ov = int(float(curr.get("overs", 0) or 0))
    if curr_ov > prev_ov and curr.get("inning") == prev.get("inning"):
        events.append({
            "type": "over",
            "description": f"End of over {curr_ov} — {curr.get('runs',0)}/{curr_w}",
        })

    return events


# ── Main poll loop ──────────────────────────────────────────────────────────────
def _poll_loop(api_key: str, match_id: str, stop_evt: threading.Event):
    from modules.database import insert_event  # lazy import avoids circular dep

    logger.info(
        f"[events_poller] Starting — match_id={match_id}, "
        f"interval={_POLL_INTERVAL}s"
    )
    poller_status.set(running=True)
    prev_state: dict = {}

    while not stop_evt.is_set():
        try:
            data  = _get_live_score(api_key, match_id)
            curr_state = _extract_score_state(data)

            # Update credits in status
            poller_status.set(
                calls=poller_status.calls + 1,
                credits=curr_state.get("credits"),
                last_ok=datetime.now(timezone.utc),
                last_err=None,
            )

            if prev_state:
                detected = _detect_events(prev_state, curr_state)
                for ev in detected:
                    insert_event(ev["type"], player=ev.get("description", ""))
                    poller_status.set(events_logged=poller_status.events_logged + 1)
                    logger.info(f"[events_poller] Auto-logged: {ev}")

            prev_state = curr_state

        except requests.exceptions.HTTPError as e:
            sc = e.response.status_code if e.response else 0
            msg = f"HTTP {sc}: {e}"
            logger.warning(f"[events_poller] {msg}")
            poller_status.set(last_err=msg)

            # Respect rate-limit headers
            if sc == 429:
                retry_after = int(e.response.headers.get("Retry-After", 60))
                logger.warning(f"[events_poller] Rate-limited — sleeping {retry_after}s")
                stop_evt.wait(retry_after)
                continue

        except Exception as e:
            msg = str(e)
            logger.warning(f"[events_poller] Error: {msg}")
            poller_status.set(last_err=msg)

        # Wait for next poll (interruptible)
        stop_evt.wait(_POLL_INTERVAL)

    poller_status.set(running=False)
    logger.info("[events_poller] Stopped.")


# ── Public entry point ──────────────────────────────────────────────────────────
_stop_event: threading.Event = threading.Event()


def start_poller(api_key: str = None, match_id: str = None) -> bool:
    """
    Start the CricAPI poller in a daemon thread.
    Returns True if started, False if skipped (missing key/match_id).

    Called from app.py main() — reads CRICAPI_KEY and MATCH_ID from env by default.
    """
    key      = api_key  or os.getenv("CRICAPI_KEY",  "")
    mid      = match_id or os.getenv("MATCH_ID", "")

    if not key or not mid:
        logger.info(
            "[events_poller] Skipped — set CRICAPI_KEY and MATCH_ID env vars "
            "to enable live ball-by-ball events."
        )
        return False

    _stop_event.clear()
    t = threading.Thread(
        target=_poll_loop,
        args=(key, mid, _stop_event),
        daemon=True,
        name="cricapi-poller",
    )
    t.start()
    return True


def stop_poller():
    """Signal the poller to exit cleanly."""
    _stop_event.set()
