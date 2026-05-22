"""
IPL Sentiment Dashboard — main Dash application entry point.
Run: python app.py
"""
import os
import sys
import logging
from pathlib import Path
from collections import Counter
from datetime import datetime, timezone

# ── Path setup ─────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("ipl_dashboard")

# ── Dash + Bootstrap ───────────────────────────────────────────────────────────
import dash
from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc

# ── Internal modules ───────────────────────────────────────────────────────────
from modules.database import (
    init_db, fetch_recent_tweets_with_sentiment,
    fetch_sentiment_timeseries, fetch_match_events,
    fetch_team_sentiment_counts, fetch_all_tweet_texts,
    get_tweet_volume_per_minute, insert_event,
)
from modules.charts import (
    build_sentiment_timeseries, build_team_sentiment_bars,
    build_volume_histogram, build_sentiment_donut, build_wordcloud_img,
    COLORS,
)
from modules.streamer import start_streamer, stream_status
from modules.events_poller import start_poller, poller_status
from modules.exporter import register_export_routes
from modules.ai_routes import register_ai_routes
from modules.ai.stream_replay import start_stream_replay

# ── App init ───────────────────────────────────────────────────────────────────
app = dash.Dash(
    __name__,
    external_stylesheets=[
        dbc.themes.CYBORG,
        "https://fonts.googleapis.com/css2?family=DM+Mono:wght@300;400;500&family=Syne:wght@700;800&display=swap",
    ],
    title="🏏 IPL Sentiment Live",
    update_title=None,
)
server = app.server
register_export_routes(server)   # Phase 3C — /export/{csv,json,events,summary,health}
register_ai_routes(server)       # Phase 4  — /predict-match, /generate-summary, /player-rankings …

# ──────────────────────────────────────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────────────────────────────────────
CARD_STYLE = {
    "background": COLORS["card"],
    "border": f"1px solid {COLORS['border']}",
    "borderRadius": "12px",
    "padding": "16px",
    "marginBottom": "12px",
}

EVENT_EMOJI = {
    "wicket": "🏏", "six": "💥", "four": "4️⃣",
    "wide": "〰️",  "no_ball": "🚫", "over": "⬛",
}

EVENT_TAG_STYLE = {
    "wicket": {"background": "#FF475720", "color": COLORS["Negative"],
               "border": f"1px solid {COLORS['Negative']}"},
    "six":    {"background": "#00E5A020", "color": COLORS["Positive"],
               "border": f"1px solid {COLORS['Positive']}"},
    "four":   {"background": "#00E5A020", "color": COLORS["Positive"],
               "border": f"1px solid {COLORS['Positive']}"},
    "wide":   {"background": "#FFD16620", "color": COLORS["Neutral"],
               "border": f"1px solid {COLORS['Neutral']}"},
    "no_ball":{"background": "#FFD16620", "color": COLORS["Neutral"],
               "border": f"1px solid {COLORS['Neutral']}"},
    "over":   {"background": "#21262D",   "color": COLORS["muted"],
               "border": f"1px solid {COLORS['border']}"},
}

SENTIMENT_BADGE = {
    "Positive": {"background": "#00E5A020", "color": COLORS["Positive"],
                 "border": f"1px solid {COLORS['Positive']}"},
    "Neutral":  {"background": "#FFD16620", "color": COLORS["Neutral"],
                 "border": f"1px solid {COLORS['Neutral']}"},
    "Negative": {"background": "#FF475720", "color": COLORS["Negative"],
                 "border": f"1px solid {COLORS['Negative']}"},
}

def _fmt_count(n: int) -> str:
    """Format like-count: 1200 → 1.2k, 342 → 342."""
    if n >= 1000:
        return f"{n/1000:.1f}k"
    return str(n)

def _pill(text, style_dict):
    return html.Span(text, style={
        **style_dict,
        "borderRadius": "20px", "padding": "2px 10px",
        "fontSize": "11px", "fontWeight": "700",
    })

def _tweet_card(tweet):
    label  = tweet.get("label") or "Neutral"
    event  = tweet.get("event_tag")
    team   = tweet.get("team_mention")
    likes  = tweet.get("like_count", 0) or 0

    # Row 2: meta pills
    meta = []
    if team:
        meta.append(html.Span(
            f"🏆 {team}",
            style={"color": COLORS.get(team, COLORS["muted"]),
                   "fontSize": "11px", "fontWeight": "600"},
        ))
    if event:
        es = EVENT_TAG_STYLE.get(event, EVENT_TAG_STYLE["over"])
        meta.append(html.Span(
            f"{EVENT_EMOJI.get(event,'•')} {event.upper()}",
            style={**es, "borderRadius": "20px", "padding": "1px 8px",
                   "fontSize": "10px", "fontWeight": "700"},
        ))
    if likes:
        meta.append(html.Span(
            f"♥ {_fmt_count(likes)}",
            style={"color": COLORS["muted"], "fontSize": "11px"},
        ))

    return html.Div([
        # Tweet text + sentiment badge on same line
        html.Div([
            html.Span(tweet.get("text", ""), style={
                "color": COLORS["text"], "fontSize": "13px",
                "lineHeight": "1.6", "flex": "1",
            }),
        ], style={"marginBottom": "6px"}),
        _pill(label, SENTIMENT_BADGE.get(label, SENTIMENT_BADGE["Neutral"])),
        html.Div(meta, style={
            "display": "flex", "gap": "8px", "flexWrap": "wrap",
            "alignItems": "center", "marginTop": "6px",
        }),
    ], style={
        **CARD_STYLE,
        "padding": "12px 14px",
        "marginBottom": "8px",
        "borderLeft": f"3px solid {COLORS.get(label, '#888')}",
        "borderRadius": "8px",
    })


# ──────────────────────────────────────────────────────────────────────────────
# LAYOUT HELPERS
# ──────────────────────────────────────────────────────────────────────────────
def _stat_box(short_label, value_id, colour):
    return html.Div([
        html.Div(short_label, style={
            "color": COLORS["muted"], "fontSize": "10px",
            "textTransform": "uppercase", "letterSpacing": "2px",
            "marginBottom": "4px",
        }),
        html.Div(id=value_id, children="—", style={
            "color": colour, "fontSize": "30px", "fontWeight": "800",
            "fontFamily": "'Syne', sans-serif", "letterSpacing": "-1px",
        }),
    ], style={
        **CARD_STYLE,
        "textAlign": "center", "padding": "16px 8px",
        "flex": "1", "marginBottom": "0",
    })


# ──────────────────────────────────────────────────────────────────────────────
# APP LAYOUT  (matches the screenshot exactly)
# ──────────────────────────────────────────────────────────────────────────────
app.layout = html.Div([

    # ── HEADER ────────────────────────────────────────────────────────────────
    html.Div([
        # Left: logo + title
        html.Div([
            html.Span("🏏", style={"fontSize": "28px"}),
            html.Span(" IPL SENTIMENT", style={
                "fontFamily": "'Syne', sans-serif", "fontWeight": "800",
                "fontSize": "22px", "letterSpacing": "3px", "marginLeft": "8px",
                "color": COLORS["text"],
            }),
            html.Span(" LIVE", style={
                "fontFamily": "'Syne', sans-serif", "fontWeight": "800",
                "fontSize": "22px", "color": COLORS["Positive"],
            }),
        ], style={"display": "flex", "alignItems": "center"}),

        # Right: status line   "● LIVE  |  Updated HH:MM:SS UTC  |  N tweets analysed"
        html.Div([
            html.Span("● LIVE", style={
                "color": COLORS["Positive"], "fontSize": "12px",
                "fontWeight": "700", "marginRight": "10px",
            }),
            html.Span("|", style={"color": COLORS["border"], "marginRight": "10px"}),
            html.Span(id="last-update",
                      style={"color": COLORS["muted"], "fontSize": "12px",
                             "marginRight": "10px"}),
            html.Span("|", style={"color": COLORS["border"], "marginRight": "10px"}),
            html.Span(id="tweet-total",
                      style={"color": COLORS["muted"], "fontSize": "12px"}),
        ], style={"display": "flex", "alignItems": "center"}),
    ], style={
        "display": "flex", "justifyContent": "space-between", "alignItems": "center",
        "padding": "14px 24px",
        "borderBottom": f"1px solid {COLORS['border']}",
        "background": COLORS["card"],
        "marginBottom": "0",
    }),

    # ── STREAM STATUS BAR (thin) ───────────────────────────────────────────────
    html.Div(id="stream-status-bar", style={
        "padding": "3px 24px",
        "fontSize": "10px",
        "background": "#0D1117",
        "borderBottom": f"1px solid {COLORS['border']}",
        "marginBottom": "16px",
    }),

    # ── BODY ──────────────────────────────────────────────────────────────────
    html.Div([

        # ── LEFT COLUMN (300px) ────────────────────────────────────────────────
        html.Div([

            # POS / NEU / NEG stat boxes
            html.Div([
                _stat_box("POS", "stat-pos", COLORS["Positive"]),
                _stat_box("NEU", "stat-neu", COLORS["Neutral"]),
                _stat_box("NEG", "stat-neg", COLORS["Negative"]),
            ], style={"display": "flex", "gap": "8px", "marginBottom": "12px"}),

            # OVERALL MOOD label
            html.Div("OVERALL MOOD", style={
                "color": COLORS["muted"], "fontSize": "10px",
                "letterSpacing": "2px", "textTransform": "uppercase",
                "marginBottom": "4px", "paddingLeft": "4px",
            }),
            dcc.Graph(id="donut-chart", config={"displayModeBar": False},
                      style={"height": "200px", "marginBottom": "12px"}),

            # TEAM SENTIMENT label
            html.Div("TEAM SENTIMENT", style={
                "color": COLORS["muted"], "fontSize": "10px",
                "letterSpacing": "2px", "textTransform": "uppercase",
                "marginBottom": "4px", "paddingLeft": "4px",
            }),
            dcc.Graph(id="team-bars", config={"displayModeBar": False},
                      style={"height": "220px", "marginBottom": "12px"}),

            # LOG MATCH EVENT
            html.Div([
                html.Div("LOG MATCH EVENT", style={
                    "color": COLORS["muted"], "fontSize": "10px",
                    "textTransform": "uppercase", "letterSpacing": "2px",
                    "marginBottom": "10px",
                }),
                dcc.Dropdown(
                    id="event-type",
                    options=[
                        {"label": "🏏 Wicket",   "value": "wicket"},
                        {"label": "💥 Six",      "value": "six"},
                        {"label": "4️⃣ Four",     "value": "four"},
                        {"label": "〰️ Wide",     "value": "wide"},
                        {"label": "🚫 No Ball",  "value": "no_ball"},
                        {"label": "⬛ Over End", "value": "over"},
                    ],
                    placeholder="Event type…",
                    style={"marginBottom": "8px",
                           "background": COLORS["bg"], "color": COLORS["text"]},
                ),
                dcc.Input(
                    id="event-player", type="text",
                    placeholder="Player name (optional)",
                    style={
                        "width": "100%", "background": COLORS["bg"],
                        "color": COLORS["text"],
                        "border": f"1px solid {COLORS['border']}",
                        "borderRadius": "6px", "padding": "6px 10px",
                        "marginBottom": "8px", "fontSize": "13px",
                        "fontFamily": "'DM Mono', monospace",
                    },
                ),
                html.Button(
                    "Log Event", id="log-event-btn", n_clicks=0,
                    style={
                        "width": "100%",
                        "background": COLORS["Positive"],
                        "color": "#0D1117",
                        "border": "none", "borderRadius": "6px",
                        "padding": "10px", "fontWeight": "800",
                        "fontSize": "13px", "cursor": "pointer",
                        "fontFamily": "'DM Mono', monospace",
                        "letterSpacing": "1px",
                    },
                ),
                html.Div(id="event-log-status", style={
                    "marginTop": "8px", "fontSize": "12px",
                    "color": COLORS["Positive"], "fontWeight": "600",
                }),
            ], style=CARD_STYLE),

            # ── EXPORT PANEL (Phase 3C) ────────────────────────────────────
            html.Div([
                html.Div("EXPORT DATA", style={
                    "color": COLORS["muted"], "fontSize": "10px",
                    "textTransform": "uppercase", "letterSpacing": "2px",
                    "marginBottom": "12px",
                }),
                *[
                    html.A(
                        label,
                        href=href,
                        target="_blank",
                        style={
                            "display": "block",
                            "width": "100%",
                            "boxSizing": "border-box",
                            "background": bg,
                            "color": fg,
                            "border": f"1px solid {border}",
                            "borderRadius": "6px",
                            "padding": "9px 14px",
                            "marginBottom": "8px",
                            "fontWeight": "700",
                            "fontSize": "12px",
                            "textDecoration": "none",
                            "letterSpacing": "1px",
                            "fontFamily": "'DM Mono', monospace",
                            "cursor": "pointer",
                            "textAlign": "center",
                        },
                    )
                    for label, href, bg, fg, border in [
                        ("⬇ Download CSV",
                         "/export/csv",
                         "#00E5A015", COLORS["Positive"], COLORS["Positive"]),
                        ("⬇ Download JSON",
                         "/export/json",
                         "#FFD16615", COLORS["Neutral"], COLORS["Neutral"]),
                        ("⬇ Events CSV",
                         "/export/events",
                         "#FF475715", COLORS["Negative"], COLORS["Negative"]),
                        ("📄 Match Summary",
                         "/export/summary",
                         "#58A6FF15", "#58A6FF", "#58A6FF"),
                    ]
                ],
                html.A(
                    "● API Health",
                    href="/export/health",
                    target="_blank",
                    style={
                        "display": "block", "textAlign": "center",
                        "fontSize": "10px", "color": COLORS["muted"],
                        "marginTop": "4px", "textDecoration": "none",
                        "letterSpacing": "1px",
                    },
                ),
            ], style={**CARD_STYLE, "marginTop": "0"}),

        ], style={"width": "300px", "flexShrink": "0"}),

        # ── CENTRE COLUMN (flex) ───────────────────────────────────────────────
        html.Div([

            # SENTIMENT TIMELINE label + chart
            html.Div("SENTIMENT TIMELINE — ROLLING 1-MIN & 5-MIN AVERAGES", style={
                "color": COLORS["muted"], "fontSize": "10px",
                "letterSpacing": "2px", "textTransform": "uppercase",
                "marginBottom": "4px", "paddingLeft": "4px",
            }),
            dcc.Graph(id="timeline-chart", config={"displayModeBar": False},
                      style={"height": "300px", "marginBottom": "16px"}),

            # TWEET VOLUME label + chart
            html.Div("TWEET VOLUME PER MINUTE", style={
                "color": COLORS["muted"], "fontSize": "10px",
                "letterSpacing": "2px", "textTransform": "uppercase",
                "marginBottom": "4px", "paddingLeft": "4px",
            }),
            dcc.Graph(id="volume-chart", config={"displayModeBar": False},
                      style={"height": "200px", "marginBottom": "16px"}),

            # Word clouds row
            html.Div([
                html.Div([
                    html.Div("Positive Cloud", style={
                        "color": COLORS["Positive"], "fontSize": "12px",
                        "fontWeight": "700", "marginBottom": "8px",
                    }),
                    html.Img(id="wc-positive",
                             style={"width": "100%", "borderRadius": "8px",
                                    "minHeight": "80px"}),
                ], style={**CARD_STYLE, "flex": "1", "marginBottom": "0"}),
                html.Div([
                    html.Div("Negative Cloud", style={
                        "color": COLORS["Negative"], "fontSize": "12px",
                        "fontWeight": "700", "marginBottom": "8px",
                    }),
                    html.Img(id="wc-negative",
                             style={"width": "100%", "borderRadius": "8px",
                                    "minHeight": "80px"}),
                ], style={**CARD_STYLE, "flex": "1", "marginBottom": "0"}),
            ], style={"display": "flex", "gap": "12px"}),

        ], style={"flex": "1", "minWidth": "0", "margin": "0 16px"}),

        # ── RIGHT COLUMN — SOCIAL SENTIMENT STREAM (300px) ─────────────────────
        html.Div([
            # Header with pulse
            html.Div([
                html.Span(id="stream-pulse", children="●", style={
                    "color": COLORS["Positive"], "fontSize": "10px",
                    "marginRight": "6px", "animation": "pulse 1.5s infinite",
                }),
                html.Span("SOCIAL SENTIMENT STREAM", style={
                    "color": COLORS["text"], "fontSize": "11px",
                    "fontWeight": "800", "letterSpacing": "2px",
                    "fontFamily": "'Syne', sans-serif",
                }),
            ], style={"display": "flex", "alignItems": "center",
                      "marginBottom": "12px"}),
            # Source legend pills
            html.Div([
                html.Span("Reddit",   style={"background": "#FF450020", "color": "#FF4500",
                    "border": "1px solid #FF4500", "borderRadius": "12px",
                    "padding": "1px 7px", "fontSize": "9px", "fontWeight": "700",
                    "marginRight": "4px"}),
                html.Span("YouTube",  style={"background": "#FF000020", "color": "#FF0000",
                    "border": "1px solid #FF0000", "borderRadius": "12px",
                    "padding": "1px 7px", "fontSize": "9px", "fontWeight": "700",
                    "marginRight": "4px"}),
                html.Span("Thread",   style={"background": "#58A6FF20", "color": "#58A6FF",
                    "border": "1px solid #58A6FF", "borderRadius": "12px",
                    "padding": "1px 7px", "fontSize": "9px", "fontWeight": "700"}),
            ], style={"marginBottom": "10px", "display": "flex", "flexWrap": "wrap", "gap": "4px"}),
            # Live stream feed (updated every 8s)
            html.Div(id="fan-stream-feed", style={
                "overflowY": "auto", "maxHeight": "420px",
                "marginBottom": "12px",
            }),
            # Divider
            html.Div("─── LIVE TWEET FEED ───", style={
                "color": COLORS["muted"], "fontSize": "9px",
                "letterSpacing": "2px", "textAlign": "center",
                "marginBottom": "8px",
            }),
            html.Div(id="tweet-feed", style={
                "overflowY": "auto", "maxHeight": "400px",
            }),
        ], style={"width": "300px", "flexShrink": "0"}),


    ], style={
        "display": "flex", "gap": "0",
        "padding": "16px 20px 20px",
        "alignItems": "flex-start",
    }),

    # ── Intervals ─────────────────────────────────────────────────────────────
    dcc.Interval(id="refresh-interval",       interval=5000,  n_intervals=0),
    dcc.Interval(id="ai-refresh-interval",    interval=30000, n_intervals=0),
    dcc.Interval(id="stream-refresh-interval",interval=8000,  n_intervals=0),
    dcc.Interval(id="replay-interval",        interval=4000,  n_intervals=0, disabled=True),

    # ── PHASE 5 — INTERACTIVE PREDICTION CONTROL PANEL ────────────────────────
    html.Div([
        html.Div("🎯 INTERACTIVE MATCH PREDICTOR", style={
            "color": COLORS["text"], "fontSize": "13px", "fontWeight": "800",
            "letterSpacing": "3px", "fontFamily": "'Syne', sans-serif",
            "marginBottom": "16px",
            "borderBottom": f"1px solid {COLORS['Positive']}40",
            "paddingBottom": "10px",
        }),
        # Control row
        html.Div([
            # Team A
            html.Div([
                html.Div("TEAM 1", style={"color": COLORS["muted"], "fontSize": "9px",
                    "letterSpacing": "2px", "marginBottom": "6px"}),
                dcc.Dropdown(
                    id="ip-team-a",
                    options=[
                        {"label": t, "value": t} for t in
                        ["MI","CSK","RCB","KKR","SRH","RR","DC","PBKS","GT","LSG"]
                    ],
                    value="MI",
                    clearable=False,
                    style={"background": "#161B22", "color": COLORS["text"]},
                ),
            ], style={"flex": "1", "minWidth": "120px"}),
            # VS divider
            html.Div("VS", style={
                "color": COLORS["Positive"], "fontWeight": "800",
                "fontFamily": "'Syne', sans-serif", "fontSize": "18px",
                "alignSelf": "flex-end", "marginBottom": "4px",
                "padding": "0 10px",
            }),
            # Team B
            html.Div([
                html.Div("TEAM 2", style={"color": COLORS["muted"], "fontSize": "9px",
                    "letterSpacing": "2px", "marginBottom": "6px"}),
                dcc.Dropdown(
                    id="ip-team-b",
                    options=[
                        {"label": t, "value": t} for t in
                        ["MI","CSK","RCB","KKR","SRH","RR","DC","PBKS","GT","LSG"]
                    ],
                    value="CSK",
                    clearable=False,
                    style={"background": "#161B22", "color": COLORS["text"]},
                ),
            ], style={"flex": "1", "minWidth": "120px"}),
            # Venue
            html.Div([
                html.Div("VENUE", style={"color": COLORS["muted"], "fontSize": "9px",
                    "letterSpacing": "2px", "marginBottom": "6px"}),
                dcc.Dropdown(
                    id="ip-venue",
                    options=[
                        {"label": v, "value": v} for v in
                        ["Wankhede","Chepauk","Chinnaswamy","Eden Gardens",
                         "Narendra Modi","Kotla","Sawai Mansingh","Hyderabad",
                         "DY Patil","Brabourne","Neutral"]
                    ],
                    value="Neutral",
                    clearable=False,
                    style={"background": "#161B22", "color": COLORS["text"]},
                ),
            ], style={"flex": "1", "minWidth": "150px"}),
            # Toss
            html.Div([
                html.Div("TOSS WINNER", style={"color": COLORS["muted"], "fontSize": "9px",
                    "letterSpacing": "2px", "marginBottom": "6px"}),
                dcc.Dropdown(
                    id="ip-toss",
                    options=[
                        {"label": "None / Unknown", "value": "none"},
                        {"label": "Team 1 won toss",  "value": "a"},
                        {"label": "Team 2 won toss",  "value": "b"},
                    ],
                    value="none",
                    clearable=False,
                    style={"background": "#161B22", "color": COLORS["text"]},
                ),
            ], style={"flex": "1", "minWidth": "160px"}),
            # Predict button
            html.Div([
                html.Div("\u00a0", style={"fontSize": "9px", "marginBottom": "6px"}),
                html.Button(
                    "⚡ PREDICT", id="predict-btn", n_clicks=0,
                    style={
                        "background": f"linear-gradient(135deg, {COLORS['Positive']}, #00b371)",
                        "color": "#0D1117", "border": "none",
                        "borderRadius": "8px", "padding": "10px 22px",
                        "fontWeight": "800", "fontSize": "13px",
                        "cursor": "pointer", "letterSpacing": "1px",
                        "fontFamily": "'Syne', sans-serif",
                        "whiteSpace": "nowrap",
                    },
                ),
            ]),
        ], style={"display": "flex", "gap": "12px", "flexWrap": "wrap",
                  "alignItems": "flex-end", "marginBottom": "16px"}),

        # Results row (2 cols: prob bars + explainability)
        html.Div([
            # Left: win probability output
            html.Div([
                html.Div(id="ip-result-header", children="Select teams and click PREDICT", style={
                    "color": COLORS["muted"], "fontSize": "12px",
                    "marginBottom": "14px", "fontStyle": "italic",
                }),
                html.Div(id="ip-bar-a-label", style={
                    "display": "flex", "justifyContent": "space-between",
                    "marginBottom": "6px"}),
                html.Div(id="ip-bar-a", style={"height": "10px", "borderRadius": "5px",
                    "marginBottom": "14px", "width": "0%",
                    "background": f"linear-gradient(90deg, {COLORS['Positive']}, #00b371)",
                    "transition": "width 1s ease"}),
                html.Div(id="ip-bar-b-label", style={
                    "display": "flex", "justifyContent": "space-between",
                    "marginBottom": "6px"}),
                html.Div(id="ip-bar-b", style={"height": "10px", "borderRadius": "5px",
                    "marginBottom": "14px", "width": "0%",
                    "background": f"linear-gradient(90deg, {COLORS['Negative']}, #b30000)",
                    "transition": "width 1s ease"}),
                html.Div(id="ip-pills", style={"display": "flex", "gap": "8px",
                    "flexWrap": "wrap"}),
            ], style={"flex": "1", "minWidth": "280px", "paddingRight": "20px",
                      "borderRight": f"1px solid {COLORS['border']}"}),

            # Right: WHY THIS PREDICTION?
            html.Div([
                html.Div("💡 WHY THIS PREDICTION?", style={
                    "color": COLORS["Neutral"], "fontSize": "10px",
                    "letterSpacing": "2px", "fontWeight": "700",
                    "marginBottom": "12px",
                }),
                # Confidence meter
                html.Div([
                    html.Span("CONFIDENCE", style={"color": COLORS["muted"], "fontSize": "9px",
                        "letterSpacing": "2px", "marginRight": "10px"}),
                    html.Span(id="ip-conf-pct", children="—", style={
                        "color": "#58A6FF", "fontWeight": "800",
                        "fontSize": "16px", "fontFamily": "'Syne', sans-serif",
                    }),
                    html.Div(id="ip-conf-bar", style={
                        "height": "4px", "borderRadius": "2px", "marginTop": "4px",
                        "background": "linear-gradient(90deg, #58A6FF, #0066FF)",
                        "width": "0%", "transition": "width 1s ease",
                    }),
                ], style={"marginBottom": "12px"}),
                # Data quality
                html.Div(id="ip-data-quality", style={
                    "color": COLORS["muted"], "fontSize": "10px",
                    "marginBottom": "12px", "fontStyle": "italic",
                }),
                # Reasons list
                html.Div("SIGNALS", style={"color": COLORS["muted"], "fontSize": "9px",
                    "letterSpacing": "2px", "marginBottom": "8px"}),
                html.Ul(id="ip-reasons", style={
                    "paddingLeft": "16px", "margin": "0 0 10px 0",
                    "color": COLORS["text"], "fontSize": "12px", "lineHeight": "1.8",
                }),
                html.Div("COUNTER-RISKS", style={"color": COLORS["muted"], "fontSize": "9px",
                    "letterSpacing": "2px", "marginBottom": "6px"}),
                html.Ul(id="ip-counter", style={
                    "paddingLeft": "16px", "margin": "0",
                    "color": COLORS["Neutral"], "fontSize": "11px", "lineHeight": "1.7",
                }),
            ], style={"flex": "1", "minWidth": "280px", "paddingLeft": "20px"}),

        ], style={"display": "flex", "gap": "0", "flexWrap": "wrap"}),

    ], style={
        **CARD_STYLE,
        "margin": "0 20px 16px",
        "background": "#0D1117",
        "border": f"1px solid {COLORS['Positive']}40",
    }),


    # ── PHASE 4 — AI FEATURE ROW ──────────────────────────────────────────────
    html.Div([

        # ── CARD 1: MATCH WIN PROBABILITY ─────────────────────────────────────
        html.Div([
            html.Div("MATCH WIN PROBABILITY", style={
                "color": COLORS["muted"], "fontSize": "10px",
                "textTransform": "uppercase", "letterSpacing": "2px",
                "marginBottom": "14px", "fontWeight": "700",
            }),
            # Team A
            html.Div(style={"display": "flex", "justifyContent": "space-between",
                            "marginBottom": "6px"}, children=[
                html.Span(id="pred-team-a-name",
                          style={"color": COLORS["Positive"], "fontWeight": "700", "fontSize": "14px"}),
                html.Span("—", id="pred-team-a-pct",
                          style={"color": COLORS["Positive"], "fontWeight": "800",
                                 "fontSize": "20px", "fontFamily": "'Syne', sans-serif"}),
            ]),
            html.Div(id="pred-bar-a", style={
                "height": "8px", "borderRadius": "4px", "marginBottom": "14px",
                "background": f"linear-gradient(90deg, {COLORS['Positive']}, #00b371)",
                "width": "50%", "transition": "width 0.8s ease",
            }),
            # Team B
            html.Div(style={"display": "flex", "justifyContent": "space-between",
                            "marginBottom": "6px"}, children=[
                html.Span(id="pred-team-b-name",
                          style={"color": COLORS["Negative"], "fontWeight": "700", "fontSize": "14px"}),
                html.Span("—", id="pred-team-b-pct",
                          style={"color": COLORS["Negative"], "fontWeight": "800",
                                 "fontSize": "20px", "fontFamily": "'Syne', sans-serif"}),
            ]),
            html.Div(id="pred-bar-b", style={
                "height": "8px", "borderRadius": "4px", "marginBottom": "16px",
                "background": f"linear-gradient(90deg, {COLORS['Negative']}, #b30000)",
                "width": "50%", "transition": "width 0.8s ease",
            }),
            # Confidence + Momentum pills
            html.Div(id="pred-meta", style={"display": "flex", "gap": "8px", "flexWrap": "wrap"}),
        ], style={**CARD_STYLE, "flex": "1", "minWidth": "260px",
                  "background": "#0D1117",
                  "border": f"1px solid {COLORS['Positive']}40"}),

        # ── CARD 2: AI MATCH INSIGHTS ─────────────────────────────────────────
        html.Div([
            html.Div("AI MATCH INSIGHTS", style={
                "color": COLORS["muted"], "fontSize": "10px",
                "textTransform": "uppercase", "letterSpacing": "2px",
                "marginBottom": "10px", "fontWeight": "700",
            }),
            html.Div(id="ai-provider-badge", style={
                "fontSize": "10px", "color": "#58A6FF",
                "marginBottom": "10px", "letterSpacing": "1px",
            }),
            html.Div(id="ai-short-summary", style={
                "fontSize": "14px", "color": COLORS["text"],
                "lineHeight": "1.6", "marginBottom": "12px",
                "borderLeft": f"3px solid {COLORS['Positive']}",
                "paddingLeft": "10px", "fontStyle": "italic",
            }),
            html.Div("KEY MOMENTS", style={
                "color": COLORS["muted"], "fontSize": "9px",
                "letterSpacing": "2px", "textTransform": "uppercase",
                "marginBottom": "6px",
            }),
            html.Ul(id="ai-key-moments", style={
                "paddingLeft": "16px", "margin": "0",
                "color": COLORS["text"], "fontSize": "12px", "lineHeight": "1.8",
            }),
            html.Div(id="ai-fan-reaction", style={
                "marginTop": "10px", "fontSize": "12px",
                "color": COLORS["muted"], "lineHeight": "1.5",
            }),
        ], style={**CARD_STYLE, "flex": "2", "minWidth": "320px",
                  "background": "#0D1117",
                  "border": "1px solid #58A6FF40"}),

        # ── CARD 3: PLAYER POPULARITY INDEX ──────────────────────────────────
        html.Div([
            html.Div("PLAYER POPULARITY INDEX", style={
                "color": COLORS["muted"], "fontSize": "10px",
                "textTransform": "uppercase", "letterSpacing": "2px",
                "marginBottom": "14px", "fontWeight": "700",
            }),
            html.Div(id="player-leaderboard"),
            html.Div(id="trending-player-banner", style={
                "marginTop": "10px", "padding": "8px 12px",
                "background": "#FFD16610",
                "border": f"1px solid {COLORS['Neutral']}",
                "borderRadius": "8px", "fontSize": "11px",
                "color": COLORS["Neutral"], "fontWeight": "700",
            }),
        ], style={**CARD_STYLE, "flex": "1", "minWidth": "260px",
                  "background": "#0D1117",
                  "border": f"1px solid {COLORS['Neutral']}40"}),

    ], style={
        "display": "flex", "gap": "16px", "flexWrap": "wrap",
        "padding": "0 20px 24px", "alignItems": "flex-start",
    }),

    # ── PHASE 5 — REPLAY HISTORIC MATCH ──────────────────────────────────────
    html.Div([
        html.Div([
            html.Span("🎬", style={"fontSize": "18px", "marginRight": "8px"}),
            html.Span("REPLAY HISTORIC MATCH", style={
                "color": COLORS["text"], "fontSize": "13px", "fontWeight": "800",
                "letterSpacing": "3px", "fontFamily": "'Syne', sans-serif",
            }),
        ], style={"display": "flex", "alignItems": "center",
                  "marginBottom": "14px", "borderBottom": f"1px solid {COLORS['Neutral']}40",
                  "paddingBottom": "10px"}),

        html.Div([
            # Match selector
            html.Div([
                html.Div("SELECT MATCH", style={"color": COLORS["muted"], "fontSize": "9px",
                    "letterSpacing": "2px", "marginBottom": "6px"}),
                dcc.Dropdown(
                    id="replay-match-select",
                    options=[
                        {"label": "CSK vs MI — IPL 2019 Final",                 "value": "csk_vs_mi_2019_final"},
                        {"label": "RCB vs CSK — 2015 Eliminator",              "value": "rcb_vs_csk_2015_pe"},
                        {"label": "MI vs SRH — IPL 2013 Final",                "value": "mi_vs_srh_2013_final"},
                        {"label": "KKR vs PBKS — Rinku Six Thriller 2022",     "value": "kkr_vs_pbks_last_ball_2022"},
                        {"label": "GT vs RR — IPL 2022 Final (Debut Season)",  "value": "gt_vs_rr_2022_final"},
                    ],
                    value="csk_vs_mi_2019_final",
                    clearable=False,
                    style={"background": "#161B22", "color": COLORS["text"],
                           "minWidth": "340px"},
                ),
            ], style={"flex": "0 0 auto"}),
            # Speed
            html.Div([
                html.Div("REPLAY SPEED", style={"color": COLORS["muted"], "fontSize": "9px",
                    "letterSpacing": "2px", "marginBottom": "6px"}),
                dcc.Dropdown(
                    id="replay-speed",
                    options=[
                        {"label": "0.5× (Cinematic)", "value": 0.5},
                        {"label": "1× (Normal)",      "value": 1.0},
                        {"label": "2× (Fast)",        "value": 2.0},
                    ],
                    value=1.0, clearable=False,
                    style={"background": "#161B22", "color": COLORS["text"],
                           "minWidth": "160px"},
                ),
            ], style={"flex": "0 0 auto"}),
            # Buttons
            html.Div([
                html.Div("\u00a0", style={"fontSize": "9px", "marginBottom": "6px"}),
                html.Div([
                    html.Button(
                        "▶ START REPLAY", id="replay-start-btn", n_clicks=0,
                        style={
                            "background": f"linear-gradient(135deg, {COLORS['Neutral']}, #b88500)",
                            "color": "#0D1117", "border": "none",
                            "borderRadius": "8px", "padding": "10px 18px",
                            "fontWeight": "800", "fontSize": "12px",
                            "cursor": "pointer", "letterSpacing": "1px",
                            "fontFamily": "'Syne', sans-serif",
                            "marginRight": "8px",
                        },
                    ),
                    html.Button(
                        "⏹ STOP", id="replay-stop-btn", n_clicks=0,
                        style={
                            "background": "#FF475720", "color": COLORS["Negative"],
                            "border": f"1px solid {COLORS['Negative']}",
                            "borderRadius": "8px", "padding": "10px 18px",
                            "fontWeight": "700", "fontSize": "12px",
                            "cursor": "pointer", "letterSpacing": "1px",
                        },
                    ),
                ], style={"display": "flex"}),
            ]),
        ], style={"display": "flex", "gap": "16px", "flexWrap": "wrap",
                  "alignItems": "flex-end", "marginBottom": "16px"}),

        # Replay status bar
        html.Div(id="replay-status", children="Select a match and press START REPLAY", style={
            "color": COLORS["muted"], "fontSize": "11px",
            "marginBottom": "12px", "fontStyle": "italic",
        }),

        # Replay feed (cinematic commentary)
        html.Div(id="replay-feed", style={
            "maxHeight": "320px", "overflowY": "auto",
        }),

    ], style={
        **CARD_STYLE,
        "margin": "0 20px 24px",
        "background": "#0D1117",
        "border": f"1px solid {COLORS['Neutral']}40",
    }),

], style={
    "background": COLORS["bg"],
    "minHeight": "100vh",
    "fontFamily": "'DM Mono', monospace",
})


# ──────────────────────────────────────────────────────────────────────────────
# MAIN CALLBACK
# ──────────────────────────────────────────────────────────────────────────────
@app.callback(
    Output("timeline-chart",    "figure"),
    Output("volume-chart",      "figure"),
    Output("team-bars",         "figure"),
    Output("donut-chart",       "figure"),
    Output("tweet-feed",        "children"),
    Output("wc-positive",       "src"),
    Output("wc-negative",       "src"),
    Output("stat-pos",          "children"),
    Output("stat-neu",          "children"),
    Output("stat-neg",          "children"),
    Output("last-update",       "children"),
    Output("tweet-total",       "children"),
    Output("stream-status-bar", "children"),
    Input("refresh-interval",   "n_intervals"),
)
def refresh_dashboard(_n):
    tweets      = fetch_recent_tweets_with_sentiment(limit=500)
    ts_rows     = fetch_sentiment_timeseries(minutes=30)
    events      = fetch_match_events(minutes=60)
    team_counts = fetch_team_sentiment_counts()
    volume      = get_tweet_volume_per_minute(minutes=30)

    # ── Figures ──
    fig_timeline = build_sentiment_timeseries(ts_rows, events)
    fig_volume   = build_volume_histogram(volume, events)
    fig_teams    = build_team_sentiment_bars(team_counts)
    fig_donut    = build_sentiment_donut(tweets)

    # ── Tweet feed (10 most recent) ──
    feed = [_tweet_card(t) for t in tweets[:10]]

    # ── Word clouds ──
    pos_texts = fetch_all_tweet_texts("Positive")
    neg_texts = fetch_all_tweet_texts("Negative")
    wc_pos = build_wordcloud_img(pos_texts, "Positive")
    wc_neg = build_wordcloud_img(neg_texts, "Negative")

    # ── Stats (all fetched tweets) ──
    counts   = Counter(t.get("label") for t in tweets if t.get("label"))
    pos_n    = counts.get("Positive", 0)
    neu_n    = counts.get("Neutral",  0)
    neg_n    = counts.get("Negative", 0)
    total_n  = pos_n + neu_n + neg_n

    stat_pos = f"{pos_n:,}"
    stat_neu = f"{neu_n:,}"
    stat_neg = f"{neg_n:,}"

    now        = datetime.now(timezone.utc).strftime("Updated %H:%M:%S UTC")
    tweet_tot  = f"{total_n:,} tweets analysed"

    # ── Stream status bar ──
    ss = stream_status.snapshot()
    _mode_meta = {
        "demo":         ("#FFD166", "⚙",  "DEMO MODE — no API key"),
        "live":         ("#00E5A0", "●",  "LIVE"),
        "reconnecting": ("#FF9F1C", "↺",  "RECONNECTING"),
        "idle":         ("#888888", "○",  "IDLE"),
    }
    clr, ico, lbl = _mode_meta.get(ss["mode"], ("#888", "?", ss["mode"].upper()))
    bar_parts = [
        html.Span(f"{ico} {lbl}", style={"color": clr, "fontWeight": "700",
                                         "marginRight": "10px"}),
    ]
    if ss["mode"] == "reconnecting" and ss["next_retry_in"] > 0:
        bar_parts.append(html.Span(
            f"retry in {ss['next_retry_in']}s",
            style={"color": "#FF9F1C", "marginRight": "10px"},
        ))
    if ss["last_err"]:
        bar_parts.append(html.Span(
            f"err: {ss['last_err'][:100]}",
            style={"color": COLORS["Negative"]},
        ))
    ps = poller_status.snapshot()
    if ps["running"]:
        bar_parts.append(html.Span(
            f"  ⚡ CricAPI: {ps['events_logged']} events logged",
            style={"color": "#00E5A0", "marginLeft": "12px"},
        ))
    status_bar = html.Div(bar_parts, style={"display": "flex", "alignItems": "center"})

    return (
        fig_timeline, fig_volume, fig_teams, fig_donut,
        feed, wc_pos, wc_neg,
        stat_pos, stat_neu, stat_neg,
        now, tweet_tot, status_bar,
    )


# ──────────────────────────────────────────────────────────────────────────────
# LOG EVENT CALLBACK
# ──────────────────────────────────────────────────────────────────────────────
@app.callback(
    Output("event-log-status", "children"),
    Input("log-event-btn",     "n_clicks"),
    State("event-type",        "value"),
    State("event-player",      "value"),
    prevent_initial_call=True,
)
def log_event(n_clicks, event_type, player):
    if not event_type:
        return "⚠ Select an event type first."
    insert_event(event_type, player=player or "")
    now_str = datetime.now(timezone.utc).strftime("%H:%M:%S")
    emoji   = EVENT_EMOJI.get(event_type, "•")
    return f"💥 {event_type.upper()} logged — {now_str}"


# ──────────────────────────────────────────────────────────────────────────────
# PHASE 4 — AI FEATURES CALLBACK  (30-second refresh)
# ──────────────────────────────────────────────────────────────────────────────
@app.callback(
    Output("pred-team-a-name",       "children"),
    Output("pred-team-a-pct",        "children"),
    Output("pred-bar-a",             "style"),
    Output("pred-team-b-name",       "children"),
    Output("pred-team-b-pct",        "children"),
    Output("pred-bar-b",             "style"),
    Output("pred-meta",              "children"),
    Output("ai-provider-badge",      "children"),
    Output("ai-short-summary",       "children"),
    Output("ai-key-moments",         "children"),
    Output("ai-fan-reaction",        "children"),
    Output("player-leaderboard",     "children"),
    Output("trending-player-banner", "children"),
    Input("ai-refresh-interval",     "n_intervals"),
)
def refresh_ai_panels(_n):
    """Updates all three AI feature cards every 30 seconds."""

    def _pill(text, color, bg="#ffffff10"):
        return html.Span(text, style={
            "background": bg, "color": color,
            "border": f"1px solid {color}", "borderRadius": "20px",
            "padding": "2px 10px", "fontSize": "10px", "fontWeight": "700",
        })

    # ── Feature 1: Match Win Probability ─────────────────────────────────────
    try:
        from modules.ai.predictor import predict_match
        team_data   = fetch_team_sentiment_counts()
        team_totals: dict = {}
        for r in team_data:
            t = r.get("team_mention") or ""
            if t:
                team_totals[t] = team_totals.get(t, 0) + r["count"]
        ranked = sorted(team_totals, key=team_totals.get, reverse=True)
        team_a = ranked[0] if len(ranked) > 0 else "MI"
        team_b = ranked[1] if len(ranked) > 1 else "CSK"

        pred   = predict_match({"team": team_a, "opponent": team_b})
        prob_a = pred["team_win_prob"]
        prob_b = pred["opponent_win_prob"]

        bar_base = {"height": "8px", "borderRadius": "4px",
                    "marginBottom": "14px", "transition": "width 0.8s ease"}
        bar_a_style = {**bar_base, "width": f"{prob_a}%",
                       "background": f"linear-gradient(90deg, {COLORS['Positive']}, #00b371)"}
        bar_b_style = {**bar_base, "width": f"{prob_b}%",
                       "background": f"linear-gradient(90deg, {COLORS['Negative']}, #b30000)"}
        meta_pills  = [
            _pill(f"🎯 Confidence: {pred['confidence']}", "#58A6FF"),
            _pill(f"📊 {pred['momentum']}", COLORS["Neutral"]),
            _pill(f"⚙ {pred['model_backend']}", COLORS["muted"]),
        ]
        pred_a_name = f"🏏 {team_a}"; pred_b_name = f"🏏 {team_b}"
        pred_a_pct  = f"{prob_a}%";  pred_b_pct  = f"{prob_b}%"

    except Exception as exc:
        logger.warning(f"[AI callback] predictor error: {exc}")
        pred_a_name = "Team A"; pred_b_name = "Team B"
        pred_a_pct  = "—";      pred_b_pct  = "—"
        bar_a_style = {"width": "50%"}; bar_b_style = {"width": "50%"}
        meta_pills  = [_pill("⚠ Loading model…", COLORS["muted"])]

    # ── Feature 2: AI Match Summary ───────────────────────────────────────────
    try:
        from modules.ai.summarizer import generate_summary
        summary        = generate_summary()
        provider_badge = f"⚡ {summary.get('generated_by', 'Template Engine')}"
        short_summary  = summary.get("short_summary", "Generating summary…")
        key_moments    = [html.Li(m) for m in summary.get("key_moments", [])]
        fan_reaction   = f"💬 {summary.get('fan_reaction', '')}"
    except Exception as exc:
        logger.warning(f"[AI callback] summarizer error: {exc}")
        provider_badge = "⚠ Summary unavailable"
        short_summary  = "Could not generate summary at this time."
        key_moments    = []; fan_reaction = ""

    # ── Feature 3: Player Popularity Index ───────────────────────────────────
    try:
        from modules.ai.player_index import compute_player_rankings, get_top_trending
        rankings = compute_player_rankings()[:8]
        trending = get_top_trending()

        TREND_COLOR = {
            "↑↑ Trending": COLORS["Positive"],
            "↑ Rising":    "#58A6FF",
            "→ Stable":    COLORS["muted"],
            "↓ Dropping":  COLORS["Neutral"],
            "↓↓ Fading":   COLORS["Negative"],
        }

        leaderboard = []
        for r in rankings:
            tc = TREND_COLOR.get(r["trend"], COLORS["muted"])
            leaderboard.append(html.Div([
                html.Div([
                    html.Span(f"#{r['rank']}", style={
                        "color": COLORS["muted"], "fontSize": "11px",
                        "width": "24px", "display": "inline-block"}),
                    html.Span(r["name"], style={
                        "color": COLORS["text"], "fontSize": "12px",
                        "fontWeight": "700", "marginRight": "6px"}),
                    html.Span(r["team"], style={
                        "color": COLORS.get(r["team"], "#888"), "fontSize": "10px"}),
                ]),
                html.Div([
                    html.Div(style={
                        "height": "4px", "borderRadius": "2px",
                        "background": f"linear-gradient(90deg, {tc}, {tc}60)",
                        "width": f"{r['popularity']}%",
                        "display": "inline-block", "verticalAlign": "middle",
                        "marginRight": "6px", "transition": "width 0.6s ease"}),
                    html.Span(f"{r['popularity']:.0f}", style={
                        "color": tc, "fontSize": "11px",
                        "fontWeight": "700", "fontFamily": "'Syne', sans-serif"}),
                    html.Span(f" {r['trend']}", style={"color": tc, "fontSize": "10px"}),
                ]),
            ], style={"marginBottom": "10px"}))

        trending_banner = (
            f"🔥 NOW TRENDING: {trending.get('player', '—')} "
            f"({trending.get('team', '')}) — {trending.get('trend', '')}"
        )
    except Exception as exc:
        logger.warning(f"[AI callback] player_index error: {exc}")
        leaderboard = [html.Div("Loading player data…", style={"color": COLORS["muted"]})]
        trending_banner = "⚡ Player data loading…"

    return (
        pred_a_name, pred_a_pct, bar_a_style,
        pred_b_name, pred_b_pct, bar_b_style,
        meta_pills,
        provider_badge, short_summary, key_moments, fan_reaction,
        leaderboard, trending_banner,
    )


# ──────────────────────────────────────────────────────────────────────────────
# PHASE 5 — CALLBACK 1: Interactive Prediction + Explainability
# ──────────────────────────────────────────────────────────────────────────────
@app.callback(
    Output("ip-result-header", "children"),
    Output("ip-bar-a-label",   "children"),
    Output("ip-bar-a",         "style"),
    Output("ip-bar-b-label",   "children"),
    Output("ip-bar-b",         "style"),
    Output("ip-pills",         "children"),
    Output("ip-conf-pct",      "children"),
    Output("ip-conf-bar",      "style"),
    Output("ip-data-quality",  "children"),
    Output("ip-reasons",       "children"),
    Output("ip-counter",       "children"),
    Input("predict-btn",       "n_clicks"),
    State("ip-team-a",         "value"),
    State("ip-team-b",         "value"),
    State("ip-venue",          "value"),
    State("ip-toss",           "value"),
    prevent_initial_call=True,
)
def run_interactive_prediction(n_clicks, team_a, team_b, venue, toss):
    if not team_a or not team_b or team_a == team_b:
        return ("⚠ Pick two different teams.",
                [], {}, [], {}, [], "—", {"width": "0%"}, "", [], [])

    toss_val = 1 if toss == "a" else (0 if toss == "b" else 0)
    payload  = {"team": team_a, "opponent": team_b, "venue": venue, "toss_won": toss_val}

    def _pill_s(text, color):
        return html.Span(text, style={
            "background": f"{color}20", "color": color,
            "border": f"1px solid {color}", "borderRadius": "20px",
            "padding": "3px 12px", "fontSize": "11px", "fontWeight": "700",
        })

    try:
        from modules.ai.predictor import predict_match
        from modules.ai.explainer import generate_explanation
        pred    = predict_match(payload)
        explain = generate_explanation(payload, pred)
        prob_a  = pred["team_win_prob"]
        prob_b  = pred["opponent_win_prob"]

        bar_base = {"height": "10px", "borderRadius": "5px",
                    "marginBottom": "14px", "transition": "width 1s ease"}

        header = f"🏏 {team_a} vs {team_b}  ·  {venue}  ·  {pred['momentum']}"
        label_a = [
            html.Span(f"🏏 {team_a}", style={"color": COLORS["Positive"], "fontWeight": "700"}),
            html.Span(f"{prob_a}%",   style={"color": COLORS["Positive"], "fontWeight": "800",
                                              "fontFamily": "'Syne', sans-serif", "fontSize": "18px"}),
        ]
        label_b = [
            html.Span(f"🏏 {team_b}", style={"color": COLORS["Negative"], "fontWeight": "700"}),
            html.Span(f"{prob_b}%",   style={"color": COLORS["Negative"], "fontWeight": "800",
                                              "fontFamily": "'Syne', sans-serif", "fontSize": "18px"}),
        ]
        style_a = {**bar_base, "width": f"{prob_a}%",
                   "background": f"linear-gradient(90deg, {COLORS['Positive']}, #00b371)"}
        style_b = {**bar_base, "width": f"{prob_b}%",
                   "background": f"linear-gradient(90deg, {COLORS['Negative']}, #b30000)"}

        pills = [
            _pill_s(f"🎯 {pred['confidence']} Confidence", "#58A6FF"),
            _pill_s(f"📊 {pred['momentum']}", COLORS["Neutral"]),
            _pill_s(f"⚙ {pred['model_backend']}", COLORS["muted"]),
        ]

        conf_pct  = explain["confidence_pct"]
        conf_bar  = {"height": "4px", "borderRadius": "2px", "marginTop": "4px",
                     "background": "linear-gradient(90deg, #58A6FF, #0066FF)",
                     "width": f"{conf_pct}%", "transition": "width 1s ease"}
        dq        = f"📡 Data quality: {explain['data_quality']}"
        reasons   = [html.Li(r) for r in explain["reasons"]]
        counter   = [html.Li(c) for c in explain["counter_reasons"]]

        return (header, label_a, style_a, label_b, style_b,
                pills, f"{conf_pct}%", conf_bar, dq, reasons, counter)

    except Exception as exc:
        logger.warning(f"[interactive predict] {exc}")
        return (f"⚠ Error: {exc}", [], {}, [], {}, [], "—", {"width": "0%"}, "", [], [])


# ──────────────────────────────────────────────────────────────────────────────
# PHASE 5 — CALLBACK 2: Social Sentiment Stream (8-second refresh)
# ──────────────────────────────────────────────────────────────────────────────
@app.callback(
    Output("fan-stream-feed", "children"),
    Input("stream-refresh-interval", "n_intervals"),
)
def refresh_fan_stream(_n):
    try:
        from modules.ai.stream_replay import get_stream_items
        items = get_stream_items(12)
        if not items:
            return [html.Div("Buffering stream…", style={"color": COLORS["muted"],
                                                          "fontSize": "12px"})]
        cards = []
        SENT_COLOR = {"Positive": COLORS["Positive"],
                      "Negative": COLORS["Negative"],
                      "Neutral":  COLORS["Neutral"]}
        SOURCE_CLR = {"Reddit": "#FF4500", "YouTube": "#FF0000",
                      "Match Thread": "#58A6FF", "Historical Replay": "#FFD166"}
        for item in items:
            sc   = SOURCE_CLR.get(item["source"], "#888")
            sent = item["sentiment"]
            tc   = SENT_COLOR.get(sent, COLORS["muted"])
            eng  = item["engagement"]
            eng_str = f"{eng/1000:.1f}k" if eng >= 1000 else str(eng)
            cards.append(html.Div([
                # Source + timestamp
                html.Div([
                    html.Span(item["source"], style={
                        "background": f"{sc}20", "color": sc,
                        "border": f"1px solid {sc}", "borderRadius": "10px",
                        "padding": "1px 7px", "fontSize": "9px", "fontWeight": "700",
                        "marginRight": "6px",
                    }),
                    html.Span(item["timestamp"], style={
                        "color": COLORS["muted"], "fontSize": "10px",
                    }),
                ], style={"marginBottom": "5px", "display": "flex", "alignItems": "center"}),
                # Comment text
                html.Div(item["text"], style={
                    "color": COLORS["text"], "fontSize": "12px",
                    "lineHeight": "1.5", "marginBottom": "6px",
                }),
                # Sentiment + engagement
                html.Div([
                    html.Span(sent, style={
                        "background": f"{tc}20", "color": tc,
                        "border": f"1px solid {tc}", "borderRadius": "10px",
                        "padding": "1px 7px", "fontSize": "9px", "fontWeight": "700",
                        "marginRight": "6px",
                    }),
                    html.Span(f"♥ {eng_str}", style={
                        "color": COLORS["muted"], "fontSize": "10px",
                    }),
                ], style={"display": "flex", "alignItems": "center"}),
            ], style={
                **CARD_STYLE,
                "padding": "10px 12px", "marginBottom": "8px",
                "borderLeft": f"3px solid {sc}", "borderRadius": "8px",
                "animation": "fadeIn 0.4s ease",
            }))
        return cards
    except Exception as exc:
        logger.warning(f"[stream callback] {exc}")
        return [html.Div("Stream loading…", style={"color": COLORS["muted"]})]


# ──────────────────────────────────────────────────────────────────────────────
# PHASE 5 — CALLBACK 3: Historical Match Replay
# ──────────────────────────────────────────────────────────────────────────────
@app.callback(
    Output("replay-status",    "children"),
    Output("replay-feed",      "children"),
    Output("replay-interval",  "disabled"),
    Input("replay-start-btn",  "n_clicks"),
    Input("replay-stop-btn",   "n_clicks"),
    Input("replay-interval",   "n_intervals"),
    State("replay-match-select","value"),
    State("replay-speed",      "value"),
    State("replay-feed",       "children"),
    prevent_initial_call=True,
)
def handle_replay(start_clicks, stop_clicks, _n_intervals,
                  match_id, speed, current_feed):
    from dash import ctx
    triggered = ctx.triggered_id if ctx.triggered_id else ""
    current_feed = current_feed or []

    from modules.ai.match_replay import (
        start_replay, get_replay_next_moment,
        reset_replay, get_replay_state,
    )

    SENT_COLOR = {"Positive": COLORS["Positive"],
                  "Negative": COLORS["Negative"],
                  "Neutral":  COLORS["Neutral"]}

    # ── STOP ─────────────────────────────────────────────────────────────────
    if triggered == "replay-stop-btn":
        reset_replay()
        return "⏹ Replay stopped.", [], True

    # ── START ─────────────────────────────────────────────────────────────────
    if triggered == "replay-start-btn":
        result = start_replay(match_id, float(speed or 1.0))
        if "error" in result:
            return f"⚠ {result['error']}", [], True
        status = f"▶ Replaying: {result['match']}  ({result['total_moments']} moments)"
        interval_ms = int(4000 / float(speed or 1.0))
        return status, [], False

    # ── TICK (interval fired) ──────────────────────────────────────────────────
    if triggered == "replay-interval":
        state = get_replay_state()
        if not state["active"]:
            return "✅ Replay complete.", current_feed, True

        moment = get_replay_next_moment()
        if moment is None or moment.get("done"):
            result_txt = moment.get("result", "Match complete") if moment else ""
            return f"🏆 {result_txt}", current_feed, True

        tc   = SENT_COLOR.get(moment.get("sentiment", "Neutral"), COLORS["muted"])
        team = moment.get("team", "")
        prog = moment.get("progress", "")
        card = html.Div([
            html.Div([
                html.Span(f"Over {moment['over']}", style={
                    "color": COLORS["muted"], "fontSize": "10px",
                    "marginRight": "10px", "fontFamily": "'DM Mono', monospace",
                }),
                html.Span(team, style={
                    "color": COLORS.get(team, "#888"), "fontSize": "10px",
                    "fontWeight": "700", "marginRight": "10px",
                }),
                html.Span(prog, style={
                    "color": COLORS["muted"], "fontSize": "9px",
                }),
            ], style={"marginBottom": "4px", "display": "flex", "alignItems": "center"}),
            html.Div(moment["text"], style={
                "color": COLORS["text"], "fontSize": "13px",
                "lineHeight": "1.5", "fontWeight": "600",
            }),
        ], style={
            **CARD_STYLE,
            "padding": "10px 14px", "marginBottom": "8px",
            "borderLeft": f"4px solid {tc}", "borderRadius": "8px",
            "animation": "fadeIn 0.5s ease",
        })

        # Prepend new moment at top
        new_feed = [card] + (current_feed[:18] if isinstance(current_feed, list) else [])
        idx  = state.get("current_idx", 0)
        tot  = len(state.get("moments", []))
        pct  = int(idx / max(tot, 1) * 100)
        status = f"▶ Replaying: {state.get('title','...')}  — {pct}% complete"
        return status, new_feed, False

    return "Select a match and press START REPLAY.", current_feed, True


# ──────────────────────────────────────────────────────────────────────────────
# STARTUP
# ──────────────────────────────────────────────────────────────────────────────
def main():
    init_db()
    bearer_token = os.getenv("TWITTER_BEARER_TOKEN")
    start_streamer(bearer_token)
    start_stream_replay()          # Phase 5: fan reaction stream

    started = start_poller()
    if started:
        logger.info("CricAPI event poller started.")
    else:
        logger.info("CricAPI poller inactive (set CRICAPI_KEY + MATCH_ID to enable).")

    port = int(os.getenv("DASH_PORT", 8050))
    logger.info(f"Dashboard starting on http://0.0.0.0:{port}")
    app.run(host="0.0.0.0", port=port, debug=False)


if __name__ == "__main__":
    main()
