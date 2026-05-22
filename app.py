"""
IPL Sentiment Dashboard — main Dash application entry point.
Run: python app.py
"""
import os
import sys
import logging
from pathlib import Path
from datetime import datetime

# ── Path setup ────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("ipl_dashboard")

# ── Dash + Bootstrap ──────────────────────────────────────────────────────────
import dash
from dash import dcc, html, Input, Output, State, callback_context
import dash_bootstrap_components as dbc

# ── Internal modules ──────────────────────────────────────────────────────────
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
from modules.streamer import start_streamer

# ── Bootstrap theme + custom CSS ─────────────────────────────────────────────
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

# ──────────────────────────────────────────────────────────────────────────────
# STYLES
# ──────────────────────────────────────────────────────────────────────────────
CARD_STYLE = {
    "background": COLORS["card"],
    "border": f"1px solid {COLORS['border']}",
    "borderRadius": "12px",
    "padding": "16px",
    "marginBottom": "16px",
}

BADGE_STYLES = {
    "Positive": {"background": "#00E5A020", "color": COLORS["Positive"],
                 "border": f"1px solid {COLORS['Positive']}"},
    "Neutral":  {"background": "#FFD16620", "color": COLORS["Neutral"],
                 "border": f"1px solid {COLORS['Neutral']}"},
    "Negative": {"background": "#FF475720", "color": COLORS["Negative"],
                 "border": f"1px solid {COLORS['Negative']}"},
}

EVENT_EMOJI = {
    "wicket": "🏏", "six": "💥", "four": "4️⃣",
    "wide": "〰️", "no_ball": "🚫", "over": "⬛",
}


def _badge(text, kind="Positive"):
    s = BADGE_STYLES.get(kind, BADGE_STYLES["Neutral"])
    return html.Span(text, style={
        **s, "borderRadius": "6px", "padding": "2px 8px",
        "fontSize": "11px", "fontWeight": "600", "marginLeft": "6px",
    })


def _tweet_card(tweet):
    label = tweet.get("label") or "Neutral"
    event = tweet.get("event_tag")
    team  = tweet.get("team_mention")
    likes = tweet.get("like_count", 0) or 0
    rts   = tweet.get("retweet_count", 0) or 0

    meta_parts = []
    if team:
        meta_parts.append(html.Span(f"🏆 {team}", style={"color": COLORS.get(team, COLORS["muted"]), "fontSize": "11px"}))
    if event:
        meta_parts.append(html.Span(f" {EVENT_EMOJI.get(event,'•')} {event.upper()}", style={"color": COLORS["Neutral"], "fontSize": "11px"}))
    if likes:
        meta_parts.append(html.Span(f"  ❤ {likes:,}", style={"color": COLORS["muted"], "fontSize": "11px"}))
    if rts:
        meta_parts.append(html.Span(f"  🔁 {rts:,}", style={"color": COLORS["muted"], "fontSize": "11px"}))

    return html.Div([
        html.Div([
            html.Span(tweet.get("text", ""), style={
                "color": COLORS["text"], "fontSize": "13px", "lineHeight": "1.5",
            }),
            _badge(label, label),
        ], style={"marginBottom": "4px"}),
        html.Div(meta_parts, style={"display": "flex", "gap": "8px", "flexWrap": "wrap"}),
    ], style={
        **CARD_STYLE,
        "padding": "12px 14px",
        "marginBottom": "8px",
        "borderLeft": f"3px solid {COLORS.get(label, '#888')}",
    })


# ──────────────────────────────────────────────────────────────────────────────
# LAYOUT
# ──────────────────────────────────────────────────────────────────────────────
def _stat_box(label, value_id, colour):
    return html.Div([
        html.Div(label, style={"color": COLORS["muted"], "fontSize": "11px",
                               "textTransform": "uppercase", "letterSpacing": "1px"}),
        html.Div(id=value_id, children="—", style={
            "color": colour, "fontSize": "28px", "fontWeight": "800",
            "fontFamily": "'Syne', sans-serif",
        }),
    ], style={**CARD_STYLE, "textAlign": "center", "padding": "18px 12px"})


app.layout = html.Div([

    # ── Header ──
    html.Div([
        html.Div([
            html.Span("🏏", style={"fontSize": "32px"}),
            html.Span(" IPL SENTIMENT", style={
                "fontFamily": "'Syne', sans-serif", "fontWeight": "800",
                "fontSize": "24px", "letterSpacing": "3px", "marginLeft": "10px",
                "color": COLORS["text"],
            }),
            html.Span(" LIVE", style={
                "fontFamily": "'Syne', sans-serif", "fontWeight": "800",
                "fontSize": "24px", "color": COLORS["Positive"],
            }),
        ], style={"display": "flex", "alignItems": "center"}),

        html.Div([
            html.Span("● LIVE", style={
                "color": COLORS["Positive"], "fontSize": "11px",
                "fontWeight": "700", "animation": "pulse 1.5s infinite",
                "marginRight": "16px",
            }),
            html.Span(id="last-update", style={"color": COLORS["muted"], "fontSize": "11px"}),
        ], style={"display": "flex", "alignItems": "center"}),
    ], style={
        "display": "flex", "justifyContent": "space-between", "alignItems": "center",
        "padding": "16px 24px", "borderBottom": f"1px solid {COLORS['border']}",
        "background": COLORS["card"], "marginBottom": "20px",
    }),

    # ── Body ──
    html.Div([

        # Left column: stats + donut + team bars
        html.Div([
            html.Div([
                _stat_box("Positive", "stat-pos", COLORS["Positive"]),
                _stat_box("Neutral",  "stat-neu", COLORS["Neutral"]),
                _stat_box("Negative", "stat-neg", COLORS["Negative"]),
            ], style={"display": "grid", "gridTemplateColumns": "1fr 1fr 1fr", "gap": "10px"}),

            dcc.Graph(id="donut-chart", config={"displayModeBar": False},
                      style={"height": "220px"}),

            dcc.Graph(id="team-bars", config={"displayModeBar": False},
                      style={"height": "260px"}),

            # Manual event injector
            html.Div([
                html.Div("➕ Log Match Event", style={
                    "color": COLORS["muted"], "fontSize": "11px",
                    "textTransform": "uppercase", "letterSpacing": "1px",
                    "marginBottom": "10px",
                }),
                dcc.Dropdown(
                    id="event-type",
                    options=[
                        {"label": "🏏 Wicket", "value": "wicket"},
                        {"label": "💥 Six",    "value": "six"},
                        {"label": "4️⃣ Four",   "value": "four"},
                        {"label": "〰️ Wide",   "value": "wide"},
                        {"label": "🚫 No Ball", "value": "no_ball"},
                        {"label": "⬛ Over End", "value": "over"},
                    ],
                    placeholder="Event type …",
                    style={"background": COLORS["bg"], "color": COLORS["text"],
                           "marginBottom": "8px"},
                ),
                dcc.Input(id="event-player", type="text",
                          placeholder="Player name (optional)",
                          style={"width": "100%", "background": COLORS["bg"],
                                 "color": COLORS["text"], "border": f"1px solid {COLORS['border']}",
                                 "borderRadius": "6px", "padding": "6px 10px",
                                 "marginBottom": "8px"}),
                html.Button("Log Event", id="log-event-btn", n_clicks=0,
                            style={"width": "100%", "background": COLORS["Positive"],
                                   "color": COLORS["bg"], "border": "none",
                                   "borderRadius": "6px", "padding": "8px",
                                   "fontWeight": "700", "cursor": "pointer"}),
                html.Div(id="event-log-status", style={"marginTop": "8px",
                                                        "fontSize": "12px",
                                                        "color": COLORS["Positive"]}),
            ], style=CARD_STYLE),

        ], style={"width": "300px", "flexShrink": "0"}),

        # Centre column: timeline + volume
        html.Div([
            dcc.Graph(id="timeline-chart", config={"displayModeBar": False},
                      style={"height": "320px", "marginBottom": "16px"}),
            dcc.Graph(id="volume-chart", config={"displayModeBar": False},
                      style={"height": "220px", "marginBottom": "16px"}),

            # Word clouds row
            html.Div([
                html.Div([
                    html.Div("😊 Positive Cloud", style={
                        "color": COLORS["Positive"], "fontSize": "12px",
                        "fontWeight": "600", "marginBottom": "8px",
                    }),
                    html.Img(id="wc-positive", style={"width": "100%", "borderRadius": "8px"}),
                ], style={**CARD_STYLE, "flex": "1"}),
                html.Div([
                    html.Div("😡 Negative Cloud", style={
                        "color": COLORS["Negative"], "fontSize": "12px",
                        "fontWeight": "600", "marginBottom": "8px",
                    }),
                    html.Img(id="wc-negative", style={"width": "100%", "borderRadius": "8px"}),
                ], style={**CARD_STYLE, "flex": "1"}),
            ], style={"display": "flex", "gap": "16px"}),

        ], style={"flex": "1", "minWidth": "0", "margin": "0 16px"}),

        # Right column: live tweet feed
        html.Div([
            html.Div("📡 Live Tweet Feed", style={
                "color": COLORS["text"], "fontSize": "13px", "fontWeight": "700",
                "textTransform": "uppercase", "letterSpacing": "2px",
                "marginBottom": "12px",
            }),
            html.Div(id="tweet-feed", style={
                "overflowY": "auto", "maxHeight": "820px",
            }),
        ], style={"width": "320px", "flexShrink": "0"}),

    ], style={
        "display": "flex",
        "gap": "0",
        "padding": "0 20px 20px",
        "alignItems": "flex-start",
    }),

    # ── Interval ──
    dcc.Interval(id="refresh-interval", interval=5000, n_intervals=0),

    # ── CSS ──
    html.Style("""
        body { background: #0D1117 !important; }
        * { box-sizing: border-box; }
        ::-webkit-scrollbar { width: 4px; }
        ::-webkit-scrollbar-track { background: #0D1117; }
        ::-webkit-scrollbar-thumb { background: #21262D; border-radius: 4px; }
        @keyframes pulse {
            0%,100% { opacity: 1; }
            50%      { opacity: 0.3; }
        }
        .Select-control, .Select-menu-outer {
            background: #161B22 !important;
            border-color: #21262D !important;
            color: #E6EDF3 !important;
        }
        .Select-value-label { color: #E6EDF3 !important; }
    """),

], style={"background": COLORS["bg"], "minHeight": "100vh",
          "fontFamily": "'DM Mono', monospace"})


# ──────────────────────────────────────────────────────────────────────────────
# CALLBACKS
# ──────────────────────────────────────────────────────────────────────────────
@app.callback(
    Output("timeline-chart",  "figure"),
    Output("volume-chart",    "figure"),
    Output("team-bars",       "figure"),
    Output("donut-chart",     "figure"),
    Output("tweet-feed",      "children"),
    Output("wc-positive",     "src"),
    Output("wc-negative",     "src"),
    Output("stat-pos",        "children"),
    Output("stat-neu",        "children"),
    Output("stat-neg",        "children"),
    Output("last-update",     "children"),
    Input("refresh-interval", "n_intervals"),
)
def refresh_dashboard(_n):
    tweets  = fetch_recent_tweets_with_sentiment(limit=200)
    ts_rows = fetch_sentiment_timeseries(minutes=30)
    events  = fetch_match_events(minutes=60)
    team_counts = fetch_team_sentiment_counts()
    volume  = get_tweet_volume_per_minute(minutes=30)

    # Figures
    fig_timeline = build_sentiment_timeseries(ts_rows, events)
    fig_volume   = build_volume_histogram(volume, events)
    fig_teams    = build_team_sentiment_bars(team_counts)
    fig_donut    = build_sentiment_donut(tweets)

    # Tweet feed (10 most recent)
    feed = [_tweet_card(t) for t in tweets[:10]]

    # Word clouds
    pos_texts = fetch_all_tweet_texts("Positive")
    neg_texts = fetch_all_tweet_texts("Negative")
    wc_pos = build_wordcloud_img(pos_texts, "Positive")
    wc_neg = build_wordcloud_img(neg_texts, "Negative")

    # Stats
    from collections import Counter
    counts = Counter(t.get("label") for t in tweets if t.get("label"))
    stat_pos = str(counts.get("Positive", 0))
    stat_neu = str(counts.get("Neutral",  0))
    stat_neg = str(counts.get("Negative", 0))

    now = datetime.utcnow().strftime("Updated %H:%M:%S UTC")

    return (fig_timeline, fig_volume, fig_teams, fig_donut,
            feed, wc_pos, wc_neg,
            stat_pos, stat_neu, stat_neg, now)


@app.callback(
    Output("event-log-status", "children"),
    Input("log-event-btn", "n_clicks"),
    State("event-type",   "value"),
    State("event-player", "value"),
    prevent_initial_call=True,
)
def log_event(n_clicks, event_type, player):
    if not event_type:
        return "⚠ Select an event type first."
    insert_event(event_type, player=player or "")
    emoji = EVENT_EMOJI.get(event_type, "•")
    return f"{emoji} {event_type.upper()} logged!"


# ──────────────────────────────────────────────────────────────────────────────
# STARTUP
# ──────────────────────────────────────────────────────────────────────────────
def main():
    init_db()
    bearer_token = os.getenv("TWITTER_BEARER_TOKEN")
    start_streamer(bearer_token)
    logger.info("Dashboard starting on http://0.0.0.0:8050")
    app.run(host="0.0.0.0", port=8050, debug=False)


if __name__ == "__main__":
    main()
