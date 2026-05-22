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

        # ── RIGHT COLUMN — LIVE FEED (300px) ──────────────────────────────────
        html.Div([
            html.Div("LIVE FEED", style={
                "color": COLORS["text"], "fontSize": "12px", "fontWeight": "800",
                "textTransform": "uppercase", "letterSpacing": "3px",
                "marginBottom": "12px",
                "fontFamily": "'Syne', sans-serif",
            }),
            html.Div(id="tweet-feed", style={
                "overflowY": "auto", "maxHeight": "860px",
            }),
        ], style={"width": "300px", "flexShrink": "0"}),

    ], style={
        "display": "flex", "gap": "0",
        "padding": "16px 20px 20px",
        "alignItems": "flex-start",
    }),

    # ── Intervals ─────────────────────────────────────────────────────────────
    dcc.Interval(id="refresh-interval",    interval=5000,  n_intervals=0),
    dcc.Interval(id="ai-refresh-interval", interval=30000, n_intervals=0),

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
# STARTUP
# ──────────────────────────────────────────────────────────────────────────────
def main():
    init_db()
    bearer_token = os.getenv("TWITTER_BEARER_TOKEN")
    start_streamer(bearer_token)

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
