"""
Plotly figure builders for the IPL Sentiment Dashboard.
"""
import io
import base64
import textwrap
from datetime import datetime
from collections import Counter, defaultdict

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# ── Colour palette ─────────────────────────────────────────────────────────────
COLORS = {
    "Positive": "#00E5A0",
    "Neutral":  "#FFD166",
    "Negative": "#FF4757",
    "bg":       "#0D1117",
    "card":     "#161B22",
    "border":   "#21262D",
    "text":     "#E6EDF3",
    "muted":    "#8B949E",
    "MI":       "#004BA0",
    "CSK":      "#F7D000",
    "RCB":      "#EC1C24",
    "KKR":      "#3A225D",
    "SRH":      "#FF6B00",
    "RR":       "#254AA5",
    "PBKS":     "#ED1F27",
    "LSG":      "#A4C9E3",
    "GT":       "#1C1C1C",
    "DC":       "#00008B",
}

LAYOUT_BASE = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color=COLORS["text"], family="'DM Mono', 'Courier New', monospace"),
    margin=dict(l=40, r=20, t=40, b=40),
    legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor=COLORS["border"]),
)

AXIS_STYLE = dict(
    gridcolor=COLORS["border"],
    zerolinecolor=COLORS["border"],
    tickcolor=COLORS["muted"],
    tickfont=dict(size=10),
)


# ── 1. Sentiment time-series ───────────────────────────────────────────────────
def build_sentiment_timeseries(rows: list, events: list) -> go.Figure:
    if not rows:
        return _empty_fig("No data yet — streaming tweets …")

    df = pd.DataFrame(rows)
    df["ts"] = pd.to_datetime(df["analyzed_at"])
    df = df.set_index("ts").sort_index()

    fig = go.Figure()

    for label in ["Positive", "Neutral", "Negative"]:
        sub = df[df["label"] == label].copy()
        if sub.empty:
            continue
        sub["y"] = 1
        # 1-min rolling count
        r1 = sub["y"].resample("1min").sum().fillna(0)
        r5 = sub["y"].resample("5min").sum().fillna(0)

        col = COLORS[label]
        fig.add_trace(go.Scatter(
            x=r1.index, y=r1.values,
            mode="lines", name=f"{label} (1-min)",
            line=dict(color=col, width=2),
            fill="tozeroy",
            fillcolor=col.replace(")", ",0.08)").replace("rgb(", "rgba(")
                        if col.startswith("rgb") else col + "14",
        ))
        fig.add_trace(go.Scatter(
            x=r5.index, y=r5.values,
            mode="lines", name=f"{label} (5-min)",
            line=dict(color=col, width=1, dash="dot"),
        ))

    # Annotate match events
    for ev in events:
        try:
            et = pd.to_datetime(ev["event_time"])
        except Exception:
            continue
        etype = ev.get("event_type", "")
        symbol = {"wicket": "🏏", "six": "💥", "four": "4️⃣",
                  "wide": "W", "no_ball": "NB", "over": "⬛"}.get(etype, "•")
        fig.add_vline(
            x=et, line_width=1, line_dash="dash",
            line_color=COLORS.get("Negative" if etype == "wicket" else "Positive"),
            annotation_text=symbol,
            annotation_font_size=12,
        )

    fig.update_layout(
        **LAYOUT_BASE,
        title=dict(text="📈 Live Sentiment Timeline", font=dict(size=14)),
        xaxis=dict(**AXIS_STYLE, title="Time"),
        yaxis=dict(**AXIS_STYLE, title="Tweet Count"),
        hovermode="x unified",
    )
    return fig


# ── 2. Team sentiment bars ─────────────────────────────────────────────────────
def build_team_sentiment_bars(counts: list) -> go.Figure:
    if not counts:
        return _empty_fig("Waiting for team mentions …")

    df = pd.DataFrame(counts)
    teams = df["team_mention"].unique()

    fig = go.Figure()
    for label in ["Positive", "Neutral", "Negative"]:
        sub = df[df["label"] == label]
        fig.add_trace(go.Bar(
            x=sub["team_mention"], y=sub["count"],
            name=label,
            marker_color=COLORS[label],
            text=sub["count"], textposition="auto",
        ))

    fig.update_layout(
        **LAYOUT_BASE,
        title=dict(text="🏆 Team Sentiment Breakdown", font=dict(size=14)),
        barmode="group",
        xaxis=dict(**AXIS_STYLE, title="Team"),
        yaxis=dict(**AXIS_STYLE, title="Tweet Count"),
    )
    return fig


# ── 3. Tweet-volume histogram ──────────────────────────────────────────────────
def build_volume_histogram(volume_rows: list, events: list) -> go.Figure:
    if not volume_rows:
        return _empty_fig("Collecting tweet volume data …")

    df = pd.DataFrame(volume_rows)
    df["ts"] = pd.to_datetime(df["minute"])

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=df["ts"], y=df["count"],
        marker=dict(
            color=df["count"],
            colorscale=[[0, COLORS["Neutral"]], [0.5, COLORS["Positive"]], [1, COLORS["Negative"]]],
            showscale=False,
        ),
        name="Tweets/min",
    ))

    # Spike lines for events
    for ev in events:
        try:
            et = pd.to_datetime(ev["event_time"])
        except Exception:
            continue
        etype = ev.get("event_type", "")
        c = COLORS["Negative"] if etype == "wicket" else COLORS["Positive"]
        fig.add_vline(x=et, line_width=1, line_dash="dot",
                      line_color=c)

    fig.update_layout(
        **LAYOUT_BASE,
        title=dict(text="📊 Tweet Volume / Minute", font=dict(size=14)),
        xaxis=dict(**AXIS_STYLE, title="Time"),
        yaxis=dict(**AXIS_STYLE, title="Tweets"),
    )
    return fig


# ── 4. Sentiment donut ─────────────────────────────────────────────────────────
def build_sentiment_donut(rows: list) -> go.Figure:
    if not rows:
        return _empty_fig("")

    labels = [r["label"] for r in rows if r.get("label")]
    if not labels:
        return _empty_fig("")

    counts = Counter(labels)
    fig = go.Figure(go.Pie(
        labels=list(counts.keys()),
        values=list(counts.values()),
        hole=0.62,
        marker=dict(colors=[COLORS.get(l, "#888") for l in counts.keys()]),
        textinfo="label+percent",
        textfont=dict(size=11),
        hoverinfo="label+value",
    ))

    total = sum(counts.values())
    dominant = max(counts, key=counts.get)
    dom_pct = int(counts[dominant] / total * 100)

    fig.update_layout(
        **LAYOUT_BASE,
        title=dict(text="Overall Mood", font=dict(size=13), x=0.5),
        annotations=[dict(
            text=f"<b>{dominant}</b><br>{dom_pct}%",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=14, color=COLORS[dominant]),
        )],
        margin=dict(l=10, r=10, t=40, b=10),
    )
    return fig


# ── 5. Word cloud ──────────────────────────────────────────────────────────────
def build_wordcloud_img(texts: list, sentiment: str) -> str:
    """Returns base64 PNG for embedding in <img>."""
    try:
        from wordcloud import WordCloud, STOPWORDS
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        stop = STOPWORDS | {
            "https", "http", "t", "co", "amp", "ipl", "ipl2025",
            "rt", "the", "a", "is", "in", "of", "to", "and", "for",
        }
        combined = " ".join(texts)
        if not combined.strip():
            return ""

        col = {"Positive": "#00E5A0", "Negative": "#FF4757"}.get(sentiment, "#FFD166")
        wc = WordCloud(
            width=500, height=220,
            background_color="#0D1117",
            colormap=None,
            color_func=lambda *a, **kw: col,
            stopwords=stop,
            max_words=60,
            prefer_horizontal=0.85,
        ).generate(combined)

        buf = io.BytesIO()
        plt.figure(figsize=(5, 2.2), facecolor="#0D1117")
        plt.imshow(wc, interpolation="bilinear")
        plt.axis("off")
        plt.tight_layout(pad=0)
        plt.savefig(buf, format="png", bbox_inches="tight",
                    facecolor="#0D1117", dpi=110)
        plt.close()
        buf.seek(0)
        return "data:image/png;base64," + base64.b64encode(buf.read()).decode()
    except Exception as e:
        return ""


# ── helpers ───────────────────────────────────────────────────────────────────
def _empty_fig(msg: str) -> go.Figure:
    fig = go.Figure()
    fig.update_layout(
        **LAYOUT_BASE,
        annotations=[dict(
            text=msg, x=0.5, y=0.5, showarrow=False,
            font=dict(size=13, color=COLORS["muted"]),
            xref="paper", yref="paper",
        )],
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
    )
    return fig
