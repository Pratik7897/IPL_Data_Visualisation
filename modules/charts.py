"""
Plotly figure builders for the IPL Sentiment Dashboard.
Fixed for Plotly 6.x + Pandas 3.x:
  - fillcolor uses rgba() not 8-digit hex
  - add_vline x= uses ISO string not Timestamp (avoids pandas integer-add bug)
"""
import io
import base64
from datetime import datetime
from collections import Counter

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots  # noqa: imported for future use

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
    # margin and legend are set per-chart to allow overrides
)

_DEFAULT_MARGIN  = dict(l=40, r=20, t=40, b=40)
_DEFAULT_LEGEND  = dict(bgcolor="rgba(0,0,0,0)", bordercolor=COLORS["border"])

AXIS_STYLE = dict(
    gridcolor=COLORS["border"],
    zerolinecolor=COLORS["border"],
    tickcolor=COLORS["muted"],
    tickfont=dict(size=10),
)


def _hex_to_rgba(hex_color: str, alpha: float = 0.08) -> str:
    """Convert #RRGGBB → rgba(r,g,b,alpha). Safe for Plotly 6.x."""
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def _ts_to_str(ts) -> str:
    """Convert Pandas Timestamp (or datetime) to ISO string for add_vline."""
    if hasattr(ts, "isoformat"):
        return ts.isoformat()
    return str(ts)


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
        r1 = sub["y"].resample("1min").sum().fillna(0)
        r5 = sub["y"].resample("5min").sum().fillna(0)

        col      = COLORS[label]
        fill_col = _hex_to_rgba(col, 0.08)

        fig.add_trace(go.Scatter(
            x=r1.index.astype(str), y=r1.values,
            mode="lines", name=f"{label} (1-min)",
            line=dict(color=col, width=2),
            fill="tozeroy",
            fillcolor=fill_col,
        ))
        fig.add_trace(go.Scatter(
            x=r5.index.astype(str), y=r5.values,
            mode="lines", name=f"{label} (5-min)",
            line=dict(color=col, width=1, dash="dot"),
        ))

    # Annotate match events — add_vline is broken in Plotly 6 + Pandas 3;
    # use add_shape + add_annotation instead.
    for ev in events:
        try:
            et_str = _ts_to_str(pd.to_datetime(ev["event_time"]))
        except Exception:
            continue
        etype  = ev.get("event_type", "")
        symbol = {"wicket": "W", "six": "6", "four": "4",
                  "wide": "wd", "no_ball": "NB", "over": "ov"}.get(etype, "•")
        line_color = COLORS["Negative"] if etype == "wicket" else COLORS["Positive"]
        fig.add_shape(
            type="line",
            x0=et_str, x1=et_str, y0=0, y1=1,
            xref="x", yref="paper",
            line=dict(color=line_color, width=1, dash="dash"),
        )
        fig.add_annotation(
            x=et_str, y=1, text=symbol,
            xref="x", yref="paper",
            showarrow=False, font=dict(size=11, color=line_color),
            yanchor="bottom",
        )

    fig.update_layout(
        **LAYOUT_BASE,
        xaxis=dict(**AXIS_STYLE, title=""),
        yaxis=dict(**AXIS_STYLE, title="Count"),
        hovermode="x unified",
        margin=dict(l=40, r=20, t=10, b=40),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=10),
                    orientation="h", yanchor="bottom", y=-0.3,
                    xanchor="left", x=0),
    )
    return fig


# ── 2. Team sentiment bars (HORIZONTAL — matches screenshot) ──────────────────
def build_team_sentiment_bars(counts: list) -> go.Figure:
    if not counts:
        return _empty_fig("Waiting for team mentions …")

    df = pd.DataFrame(counts)
    teams = sorted(df["team_mention"].unique())

    fig = go.Figure()
    for label in ["Positive", "Neutral", "Negative"]:
        sub = df[df["label"] == label].set_index("team_mention")
        vals = [sub.loc[t, "count"] if t in sub.index else 0 for t in teams]
        fig.add_trace(go.Bar(
            y=teams, x=vals,
            name=label,
            orientation="h",
            marker_color=COLORS[label],
            text=[str(v) if v else "" for v in vals],
            textposition="inside",
            textfont=dict(size=9),
        ))

    fig.update_layout(
        **LAYOUT_BASE,
        barmode="stack",
        xaxis=dict(**AXIS_STYLE, title="", showticklabels=False),
        yaxis=dict(
            gridcolor=COLORS["border"], zerolinecolor=COLORS["border"],
            tickcolor=COLORS["muted"], tickfont=dict(size=11),
            title="",
        ),

        legend=dict(
            bgcolor="rgba(0,0,0,0)",
            orientation="h",
            yanchor="bottom", y=-0.25,
            xanchor="left", x=0,
            font=dict(size=10),
        ),
        margin=dict(l=40, r=10, t=10, b=40),
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
        x=df["ts"].astype(str), y=df["count"],
        marker=dict(
            color=df["count"],
            colorscale=[[0, COLORS["Neutral"]], [0.5, COLORS["Positive"]], [1, COLORS["Negative"]]],
            showscale=False,
        ),
        name="Tweets/min",
    ))

    # Spike shapes for events
    for ev in events:
        try:
            et_str = _ts_to_str(pd.to_datetime(ev["event_time"]))
        except Exception:
            continue
        etype = ev.get("event_type", "")
        c = COLORS["Negative"] if etype == "wicket" else COLORS["Positive"]
        fig.add_shape(
            type="line",
            x0=et_str, x1=et_str, y0=0, y1=1,
            xref="x", yref="paper",
            line=dict(color=c, width=1, dash="dot"),
        )

    fig.update_layout(
        **LAYOUT_BASE,
        xaxis=dict(**AXIS_STYLE, title=""),
        yaxis=dict(**AXIS_STYLE, title=""),
        margin=dict(l=40, r=20, t=10, b=40),
    )
    return fig


# ── 4. Sentiment donut ─────────────────────────────────────────────────────────
def build_sentiment_donut(rows: list) -> go.Figure:
    if not rows:
        return _empty_fig("")

    labels = [r["label"] for r in rows if r.get("label")]
    if not labels:
        return _empty_fig("")

    counts   = Counter(labels)
    total    = sum(counts.values())
    dominant = max(counts, key=counts.get)
    dom_pct  = int(counts[dominant] / total * 100)

    fig = go.Figure(go.Pie(
        labels=list(counts.keys()),
        values=list(counts.values()),
        hole=0.62,
        marker=dict(colors=[COLORS.get(l, "#888") for l in counts.keys()]),
        textinfo="label+percent",
        textfont=dict(size=11),
        hoverinfo="label+value",
    ))
    fig.update_layout(
        **LAYOUT_BASE,
        margin=dict(l=10, r=10, t=10, b=10),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
        annotations=[dict(
            text=f"<b>{dominant}</b><br>{dom_pct}%",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=15, color=COLORS[dominant]),
        )],
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
            "https", "http", "t", "co", "amp", "ipl", "ipl2026",
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
    except Exception:
        return ""


# ── helpers ────────────────────────────────────────────────────────────────────
def _empty_fig(msg: str) -> go.Figure:
    fig = go.Figure()
    fig.update_layout(
        **LAYOUT_BASE,
        margin=dict(l=20, r=20, t=20, b=20),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
        annotations=[dict(
            text=msg, x=0.5, y=0.5, showarrow=False,
            font=dict(size=13, color=COLORS["muted"]),
            xref="paper", yref="paper",
        )],
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
    )
    return fig

