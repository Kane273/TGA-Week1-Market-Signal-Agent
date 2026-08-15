"""
Stock Analyzer & AI Agent
- Live market data via yfinance
- AI agent powered locally (no external AI API)
"""

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go
import re
import os
import json
from openai import OpenAI
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings

warnings.filterwarnings("ignore")

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NASDAQ Terminal",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Theme & CSS ───────────────────────────────────────────────────────────────

def _inject_css():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:ital,opsz,wght@0,14..32,300..900;1,14..32,300..900&family=JetBrains+Mono:wght@400;500;600&display=swap');

    /* ─── Reset & base ──────────────────────────────────────── */
    *, *::before, *::after { box-sizing: border-box; }

    html, body,
    .stApp,
    .stAppViewContainer,
    [data-testid="stAppViewContainer"] {
        background: #07091a !important;
        font-family: 'Inter', system-ui, -apple-system, sans-serif !important;
        color: #94a3b8 !important;
    }

    /* Streamlit top toolbar → blend into background */
    [data-testid="stHeader"],
    .stApp > header {
        background: #07091a !important;
        border-bottom: 1px solid rgba(255,255,255,0.04) !important;
        backdrop-filter: none !important;
    }

    /* Main content padding */
    .main .block-container,
    section.main .block-container {
        padding: 2.2rem 2.8rem 3rem !important;
        max-width: 100% !important;
    }

    /* ─── Sidebar ───────────────────────────────────────────── */
    [data-testid="stSidebar"],
    section[data-testid="stSidebar"] {
        background: #070916 !important;
        border-right: 1px solid rgba(255,255,255,0.045) !important;
    }
    [data-testid="stSidebar"] .block-container {
        padding: 0.75rem 1rem 2rem !important;
    }

    /* ─── Typography ────────────────────────────────────────── */
    h1 {
        background: linear-gradient(135deg, #f8faff 0%, #c7d2fe 45%, #a78bfa 100%) !important;
        -webkit-background-clip: text !important;
        -webkit-text-fill-color: transparent !important;
        background-clip: text !important;
        font-weight: 900 !important;
        letter-spacing: -0.045em !important;
        line-height: 1.08 !important;
        margin-bottom: 0.2rem !important;
        font-size: 2.35rem !important;
    }
    h2 {
        color: #dde4f0 !important;
        font-weight: 700 !important;
        letter-spacing: -0.03em !important;
    }
    h3, h4 {
        color: #94a3b8 !important;
        font-weight: 600 !important;
        letter-spacing: -0.02em !important;
        margin-bottom: 0.6rem !important;
    }
    p, li { color: #94a3b8 !important; line-height: 1.65 !important; }
    strong, b { color: #cbd5e1 !important; font-weight: 600 !important; }
    .stCaption, small, [data-testid="stCaptionContainer"] p {
        color: #3a4d68 !important;
        font-size: 0.74rem !important;
    }
    hr {
        border: none !important;
        border-top: 1px solid rgba(255,255,255,0.055) !important;
        margin: 1.6rem 0 !important;
    }

    /* ─── Metric containers ─────────────────────────────────── */
    [data-testid="metric-container"] {
        background: linear-gradient(145deg, #0c1526 0%, #101d38 100%) !important;
        border: 1px solid rgba(255,255,255,0.07) !important;
        border-radius: 20px !important;
        padding: 22px 24px !important;
        position: relative !important;
        overflow: hidden !important;
        box-shadow:
            0 1px 0 rgba(255,255,255,0.06) inset,
            0 8px 40px rgba(0,0,0,0.55) !important;
        transition: border-color 0.25s, box-shadow 0.25s !important;
    }
    [data-testid="metric-container"]::before {
        content: '' !important;
        position: absolute !important;
        top: 0 !important; left: 0 !important; right: 0 !important; height: 1px !important;
        background: linear-gradient(90deg, transparent 0%, rgba(167,139,250,0.35) 50%, transparent 100%) !important;
        pointer-events: none !important;
    }
    [data-testid="metric-container"]:hover {
        border-color: rgba(167,139,250,0.2) !important;
        box-shadow:
            0 1px 0 rgba(255,255,255,0.07) inset,
            0 12px 48px rgba(0,0,0,0.6),
            0 0 28px rgba(99,102,241,0.07) !important;
    }
    [data-testid="stMetricLabel"] > div {
        font-size: 0.64rem !important;
        font-weight: 700 !important;
        text-transform: uppercase !important;
        letter-spacing: 0.15em !important;
        color: #2d3e58 !important;
    }
    [data-testid="stMetricValue"] > div {
        font-size: 1.9rem !important;
        font-weight: 800 !important;
        color: #f0f4ff !important;
        letter-spacing: -0.035em !important;
        line-height: 1.1 !important;
    }
    [data-testid="stMetricDelta"] > div {
        font-size: 0.77rem !important;
        font-weight: 600 !important;
        letter-spacing: 0.01em !important;
    }

    /* ─── Plotly chart card containers ─────────────────────── */
    /* This gives ALL charts a glass-card background automatically */
    [data-testid="stPlotlyChart"] {
        background: linear-gradient(145deg, #0c1526 0%, #101d38 100%) !important;
        border: 1px solid rgba(255,255,255,0.07) !important;
        border-radius: 20px !important;
        padding: 20px 18px 14px !important;
        position: relative !important;
        overflow: hidden !important;
        box-shadow:
            0 1px 0 rgba(255,255,255,0.055) inset,
            0 8px 40px rgba(0,0,0,0.5) !important;
    }
    [data-testid="stPlotlyChart"]::before {
        content: '' !important;
        position: absolute !important;
        top: 0 !important; left: 0 !important; right: 0 !important; height: 1px !important;
        background: linear-gradient(90deg, transparent, rgba(167,139,250,0.2), transparent) !important;
        pointer-events: none !important;
    }

    /* ─── Buttons ───────────────────────────────────────────── */
    .stButton > button {
        background: rgba(11, 16, 34, 0.85) !important;
        color: #55657e !important;
        border: 1px solid rgba(255,255,255,0.07) !important;
        border-radius: 10px !important;
        font-size: 0.78rem !important;
        font-weight: 500 !important;
        transition: all 0.18s cubic-bezier(0.4,0,0.2,1) !important;
        text-align: left !important;
        letter-spacing: 0.01em !important;
        padding: 8px 13px !important;
    }
    .stButton > button:hover {
        background: rgba(99,102,241,0.1) !important;
        color: #c7d2fe !important;
        border-color: rgba(99,102,241,0.3) !important;
        box-shadow: 0 0 20px rgba(99,102,241,0.14), 0 4px 16px rgba(0,0,0,0.3) !important;
        transform: translateX(3px) !important;
    }
    [data-testid="baseButton-primary"] {
        background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%) !important;
        color: #fff !important;
        border: 1px solid rgba(167,139,250,0.3) !important;
        font-weight: 700 !important;
        letter-spacing: 0.025em !important;
        box-shadow:
            0 1px 0 rgba(255,255,255,0.18) inset,
            0 4px 22px rgba(99,102,241,0.42) !important;
        text-align: center !important;
    }
    [data-testid="baseButton-primary"]:hover {
        box-shadow:
            0 1px 0 rgba(255,255,255,0.2) inset,
            0 6px 32px rgba(99,102,241,0.58) !important;
        transform: translateY(-1px) !important;
    }

    /* ─── Inputs / Selects ──────────────────────────────────── */
    .stSelectbox [data-baseweb="select"] > div,
    .stNumberInput > div > div > input,
    .stTextInput > div > div > input {
        background: rgba(8, 12, 26, 0.9) !important;
        border: 1px solid rgba(255,255,255,0.08) !important;
        border-radius: 10px !important;
        color: #dde4f0 !important;
        font-size: 0.84rem !important;
    }
    .stSelectbox [data-baseweb="select"] > div:focus-within,
    .stNumberInput > div > div > input:focus,
    .stTextInput > div > div > input:focus {
        border-color: rgba(99,102,241,0.45) !important;
        box-shadow: 0 0 0 3px rgba(99,102,241,0.1) !important;
        outline: none !important;
    }
    .stSlider [data-baseweb="slider"] [role="slider"] {
        background: #6366f1 !important;
        box-shadow: 0 0 12px rgba(99,102,241,0.55) !important;
    }
    .stSlider [data-baseweb="slider"] [data-testid="stSliderTrackFill"] {
        background: linear-gradient(90deg, #4f46e5, #7c3aed) !important;
    }

    /* ─── Expander ──────────────────────────────────────────── */
    details > summary {
        background: rgba(10, 14, 28, 0.8) !important;
        border: 1px solid rgba(255,255,255,0.07) !important;
        border-radius: 12px !important;
        color: #94a3b8 !important;
        padding: 11px 16px !important;
        font-weight: 600 !important;
        font-size: 0.85rem !important;
        cursor: pointer !important;
        transition: all 0.16s !important;
    }
    details > summary:hover {
        border-color: rgba(99,102,241,0.28) !important;
        color: #c7d2fe !important;
    }
    details[open] > summary {
        border-radius: 12px 12px 0 0 !important;
        border-color: rgba(99,102,241,0.22) !important;
        color: #a5b4fc !important;
    }

    /* ─── Alert boxes ───────────────────────────────────────── */
    [data-testid="stAlert"] {
        background: rgba(10, 14, 28, 0.75) !important;
        border: 1px solid rgba(255,255,255,0.07) !important;
        border-left-width: 3px !important;
        border-radius: 12px !important;
    }

    /* ─── DataFrames ────────────────────────────────────────── */
    .stDataFrame,
    [data-testid="stDataFrameResizable"] {
        background: linear-gradient(145deg, #0c1526, #101d38) !important;
        border: 1px solid rgba(255,255,255,0.07) !important;
        border-radius: 18px !important;
        overflow: hidden !important;
        box-shadow: 0 8px 40px rgba(0,0,0,0.45) !important;
    }

    /* ─── Chat ──────────────────────────────────────────────── */
    [data-testid="stChatMessage"] {
        background: rgba(10, 14, 28, 0.65) !important;
        border: 1px solid rgba(255,255,255,0.06) !important;
        border-radius: 18px !important;
        margin: 6px 0 !important;
    }
    [data-testid="stChatInput"] textarea {
        background: rgba(10, 14, 28, 0.9) !important;
        border: 1px solid rgba(255,255,255,0.1) !important;
        border-radius: 16px !important;
        color: #dde4f0 !important;
        font-size: 0.88rem !important;
    }
    [data-testid="stChatInput"] textarea:focus {
        border-color: rgba(99,102,241,0.5) !important;
        box-shadow: 0 0 0 3px rgba(99,102,241,0.12) !important;
    }

    /* ─── Progress bar ──────────────────────────────────────── */
    [data-testid="stProgress"] > div {
        background: rgba(255,255,255,0.05) !important;
        border-radius: 99px !important;
    }
    [data-testid="stProgress"] > div > div > div > div {
        background: linear-gradient(90deg, #4f46e5, #a78bfa) !important;
        border-radius: 99px !important;
        box-shadow: 0 0 12px rgba(99,102,241,0.45) !important;
    }

    /* ─── Spinner ───────────────────────────────────────────── */
    .stSpinner > div {
        border-color: rgba(99,102,241,0.12) !important;
        border-top-color: #6366f1 !important;
    }

    /* ─── Checkbox ──────────────────────────────────────────── */
    [data-testid="stCheckbox"] input:checked + span {
        background: #6366f1 !important;
        border-color: #6366f1 !important;
        box-shadow: 0 0 10px rgba(99,102,241,0.45) !important;
    }

    /* ─── Radio navigation ──────────────────────────────────── */
    .stRadio > label { display: none !important; }
    .stRadio > div {
        gap: 2px !important;
        flex-direction: column !important;
        display: flex !important;
    }
    .stRadio > div > label {
        background: transparent !important;
        border: 1px solid transparent !important;
        border-radius: 10px !important;
        padding: 9px 13px !important;
        color: #3a4d68 !important;
        font-size: 0.83rem !important;
        font-weight: 500 !important;
        cursor: pointer !important;
        transition: all 0.15s cubic-bezier(0.4,0,0.2,1) !important;
        letter-spacing: 0.01em !important;
    }
    .stRadio > div > label:hover {
        background: rgba(99,102,241,0.08) !important;
        color: #a5b4fc !important;
        border-color: rgba(99,102,241,0.18) !important;
    }
    .stRadio > div > label:has(input:checked) {
        background: rgba(99,102,241,0.13) !important;
        color: #a5b4fc !important;
        border-color: rgba(99,102,241,0.32) !important;
        box-shadow: inset 0 1px 0 rgba(255,255,255,0.04), 0 0 20px rgba(99,102,241,0.08) !important;
    }
    .stRadio > div > label > div:first-child { display: none !important; }
    </style>
    """, unsafe_allow_html=True)


# ── UI helper components ──────────────────────────────────────────────────────

def _stat_card(label: str, value: str, sub: str = "", accent: str = "#6366f1",
               icon: str = "") -> str:
    sub_html = (
        f'<div style="margin-top:8px;font-size:0.76rem;font-weight:600;'
        f'color:{accent};letter-spacing:0.01em;opacity:0.85">{sub}</div>'
    ) if sub else ""
    icon_html = (
        f'<div style="width:32px;height:32px;border-radius:9px;'
        f'background:{accent}18;border:1px solid {accent}30;'
        f'display:flex;align-items:center;justify-content:center;font-size:1rem;flex-shrink:0">{icon}</div>'
    ) if icon else ""
    return f"""
    <div style="
        background:linear-gradient(145deg,#0c1526 0%,#101d38 100%);
        border:1px solid rgba(255,255,255,0.07);
        border-radius:20px; padding:22px 24px;
        position:relative; overflow:hidden;
        box-shadow:0 1px 0 rgba(255,255,255,0.06) inset, 0 8px 40px rgba(0,0,0,0.55);
        height:100%;
    ">
      <div style="
          position:absolute;top:0;left:0;right:0;height:1px;
          background:linear-gradient(90deg,transparent,{accent}38,transparent);
          pointer-events:none
      "></div>
      <div style="position:absolute;top:-20px;right:-20px;width:100px;height:100px;
          background:radial-gradient(circle,{accent}1a 0%,transparent 65%);pointer-events:none"></div>
      <div style="display:flex;align-items:flex-start;justify-content:space-between;margin-bottom:14px">
        <span style="color:#2d3e58;font-size:0.63rem;font-weight:700;text-transform:uppercase;
            letter-spacing:0.15em;padding-top:2px">{label}</span>
        {icon_html}
      </div>
      <div style="color:#f0f4ff;font-size:1.95rem;font-weight:800;letter-spacing:-0.035em;line-height:1.05">{value}</div>
      {sub_html}
    </div>"""


def _badge(text: str, color: str = "#6366f1", bg: str = "") -> str:
    bg = bg or f"{color}15"
    return (f'<span style="display:inline-block;background:{bg};color:{color};'
            f'border:1px solid {color}40;border-radius:7px;padding:3px 10px;'
            f'font-size:0.7rem;font-weight:700;letter-spacing:0.09em;text-transform:uppercase;'
            f'vertical-align:middle">{text}</span>')


def _section_hdr(title: str, sub: str = "") -> str:
    """Elegant section header with gradient accent line."""
    sub_html = f'<div style="color:#3a4d68;font-size:0.75rem;font-weight:500;margin-top:2px">{sub}</div>' if sub else ""
    return f"""
    <div style="margin:2rem 0 1.2rem;display:flex;align-items:center;gap:14px">
      <div style="width:3px;height:28px;border-radius:2px;
          background:linear-gradient(180deg,#a78bfa,#6366f1);flex-shrink:0"></div>
      <div>
        <div style="color:#dde4f0;font-size:1rem;font-weight:700;letter-spacing:-0.02em;line-height:1.2">{title}</div>
        {sub_html}
      </div>
    </div>"""


def _medal_card(rank: int, symbol: str, name: str, price: float,
                pct: float, score: float, sector: str) -> str:
    medals = {
        1: ("#f59e0b", "rgba(120,53,15,0.35)", "🥇", "rgba(245,158,11,0.12)"),
        2: ("#8b9eb0", "rgba(30,41,59,0.4)",   "🥈", "rgba(139,158,176,0.08)"),
        3: ("#c97c3a", "rgba(44,24,16,0.4)",   "🥉", "rgba(201,124,58,0.1)"),
    }
    accent, bg_tint, icon, glow = medals.get(rank, ("#6366f1","rgba(30,27,75,0.4)","✦","rgba(99,102,241,0.1)"))
    pct_color = "#10b981" if pct >= 0 else "#f43f5e"
    pct_sign  = "+" if pct >= 0 else ""
    return f"""
    <div style="
        background:linear-gradient(145deg,#0c1526 0%,{bg_tint} 100%);
        border:1px solid rgba(255,255,255,0.08);
        border-radius:20px; padding:20px 22px;
        box-shadow:0 1px 0 rgba(255,255,255,0.06) inset, 0 8px 40px rgba(0,0,0,0.5), 0 0 30px {glow};
        position:relative; overflow:hidden; height:100%;
    ">
      <div style="position:absolute;top:0;left:0;right:0;height:1px;
          background:linear-gradient(90deg,transparent,{accent}40,transparent);pointer-events:none"></div>
      <div style="position:absolute;top:-25px;right:-25px;width:110px;height:110px;
          background:radial-gradient(circle,{accent}1a 0%,transparent 65%);pointer-events:none"></div>
      <div style="font-size:1.5rem;margin-bottom:10px;line-height:1">{icon}</div>
      <div style="font-family:'JetBrains Mono',monospace;font-size:1.3rem;font-weight:700;
          color:{accent};letter-spacing:0.05em;line-height:1">{symbol}</div>
      <div style="color:#3a4d68;font-size:0.71rem;font-weight:500;margin:5px 0 14px;
          white-space:nowrap;overflow:hidden;text-overflow:ellipsis">{name[:30]}</div>
      <div style="display:flex;align-items:baseline;gap:8px;margin-bottom:14px">
        <span style="color:#f0f4ff;font-size:1.45rem;font-weight:800;letter-spacing:-0.025em">${price:.2f}</span>
        <span style="color:{pct_color};font-size:0.82rem;font-weight:700">{pct_sign}{pct:.2f}%</span>
      </div>
      <div style="display:flex;align-items:center;gap:10px;margin-bottom:6px">
        <div style="flex:1;height:3px;background:rgba(255,255,255,0.06);border-radius:99px;overflow:hidden">
          <div style="height:100%;width:{min(score,100):.0f}%;
              background:linear-gradient(90deg,{accent}cc,{accent});
              border-radius:99px;box-shadow:0 0 8px {accent}80"></div>
        </div>
        <span style="color:{accent};font-size:0.73rem;font-weight:700;letter-spacing:0.05em;
            white-space:nowrap">{score:.0f}/100</span>
      </div>
      <div style="color:#2d3e58;font-size:0.67rem;font-weight:600;text-transform:uppercase;
          letter-spacing:0.08em">{sector}</div>
    </div>"""


def _chart_dark(fig, height: int = 360, colorscale: bool = False):
    """Apply refined dark theme to any Plotly figure."""
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#3a4d68", family="Inter,system-ui,sans-serif", size=11),
        xaxis=dict(
            gridcolor="rgba(255,255,255,0.045)",
            linecolor="rgba(255,255,255,0.06)",
            tickfont=dict(color="#3a4d68", size=10),
            title_font=dict(color="#4a5e7a", size=11),
            zeroline=False,
            showline=False,
        ),
        yaxis=dict(
            gridcolor="rgba(255,255,255,0.045)",
            linecolor="rgba(255,255,255,0.06)",
            tickfont=dict(color="#3a4d68", size=10),
            title_font=dict(color="#4a5e7a", size=11),
            zeroline=False,
            showline=False,
        ),
        legend=dict(
            bgcolor="rgba(10,15,30,0.85)",
            bordercolor="rgba(255,255,255,0.07)",
            borderwidth=1,
            font=dict(color="#94a3b8", size=11),
        ),
        margin=dict(l=4, r=4, t=10, b=4),
        height=height,
        coloraxis_showscale=colorscale,
        hoverlabel=dict(
            bgcolor="rgba(12,18,38,0.95)",
            bordercolor="rgba(167,139,250,0.4)",
            font=dict(color="#e2e8f0", size=12),
        ),
    )
    return fig

CSV_PATH = os.path.join(os.path.dirname(__file__), "nasdaq_screener.csv")

# ═══════════════════════════════════════════════════════════════════════════════
# DATA
# ═══════════════════════════════════════════════════════════════════════════════

@st.cache_data
def load_data() -> pd.DataFrame:
    df = pd.read_csv(CSV_PATH)
    df["Last Sale"] = (
        df["Last Sale"].astype(str).str.replace("$", "", regex=False).str.replace(",", "")
    )
    df["Last Sale"] = pd.to_numeric(df["Last Sale"], errors="coerce")
    df["% Change"] = df["% Change"].astype(str).str.replace("%", "", regex=False)
    df["% Change"] = pd.to_numeric(df["% Change"], errors="coerce").fillna(0)
    df["Market Cap"] = pd.to_numeric(df["Market Cap"], errors="coerce").fillna(0)
    df["Volume"] = pd.to_numeric(df["Volume"], errors="coerce").fillna(0)
    df["Net Change"] = pd.to_numeric(df["Net Change"], errors="coerce").fillna(0)
    df["IPO Year"] = pd.to_numeric(df["IPO Year"], errors="coerce")
    df = df[df["Last Sale"].notna() & (df["Last Sale"] > 0)]
    df = df[df["Symbol"].notna()]
    df["Sector"] = df["Sector"].fillna("Unknown")
    df["Industry"] = df["Industry"].fillna("Unknown")
    df["Country"] = df["Country"].fillna("Unknown")
    return df.reset_index(drop=True)


# ═══════════════════════════════════════════════════════════════════════════════
# SCORING (CSV-based)
# ═══════════════════════════════════════════════════════════════════════════════

def compute_scores(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    p5, p95 = d["% Change"].quantile(0.05), d["% Change"].quantile(0.95)
    d["momentum_score"] = ((d["% Change"] - p5) / (p95 - p5 + 1e-9)).clip(0, 1) * 100

    d["log_vol"] = np.log1p(d["Volume"])
    v10, v90 = d["log_vol"].quantile(0.1), d["log_vol"].quantile(0.9)
    d["volume_score"] = ((d["log_vol"] - v10) / (v90 - v10 + 1e-9)).clip(0, 1) * 100

    d["log_mcap"] = np.log10(d["Market Cap"].clip(1))
    d["mcap_score"] = (100 - np.abs(d["log_mcap"] - 8.7) * 15).clip(0, 100)

    d["price_score"] = (100 - np.abs(np.log10(d["Last Sale"].clip(0.01)) - 1.5) * 20).clip(0, 100)
    d["direction_score"] = (d["Net Change"] > 0).astype(float) * 60 + 20

    d["score"] = (
        d["momentum_score"] * 0.35
        + d["volume_score"] * 0.25
        + d["mcap_score"] * 0.20
        + d["price_score"] * 0.10
        + d["direction_score"] * 0.10
    ).round(1)
    return d


def filter_df(df, min_price=0.0, max_price=1e9, min_cap=0.0, min_vol=0.0,
              sector="All", direction="All") -> pd.DataFrame:
    m = ((df["Last Sale"] >= min_price) & (df["Last Sale"] <= max_price)
         & (df["Market Cap"] >= min_cap) & (df["Volume"] >= min_vol))
    if sector != "All":
        m &= df["Sector"] == sector
    if direction == "Gainers":
        m &= df["% Change"] > 0
    elif direction == "Losers":
        m &= df["% Change"] < 0
    return df[m]


# ═══════════════════════════════════════════════════════════════════════════════
# LIVE DATA (yfinance)
# ═══════════════════════════════════════════════════════════════════════════════

def _calc_rsi(prices: pd.Series, period: int = 14) -> float:
    delta = prices.diff()
    gain = delta.where(delta > 0, 0).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
    rs = gain / (loss + 1e-9)
    rsi = 100 - (100 / (1 + rs))
    return float(rsi.iloc[-1]) if not rsi.empty else 50.0


def _fetch_one(symbol: str) -> dict | None:
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info or {}
        hist = ticker.history(period="3mo")
        if hist.empty:
            return None
        price = (info.get("currentPrice") or info.get("regularMarketPrice")
                 or float(hist["Close"].iloc[-1]))
        if not price:
            return None
        rsi = _calc_rsi(hist["Close"])
        ma20 = float(hist["Close"].rolling(20).mean().iloc[-1]) if len(hist) >= 20 else None
        ma50 = float(hist["Close"].rolling(50).mean().iloc[-1]) if len(hist) >= 50 else None
        w52h = info.get("fiftyTwoWeekHigh", price)
        w52l = info.get("fiftyTwoWeekLow", price)
        target = info.get("targetMeanPrice")
        upside = ((target - price) / price * 100) if target and price else None
        return {
            "symbol": symbol,
            "current_price": price,
            "rsi": rsi,
            "ma20": ma20,
            "ma50": ma50,
            "week52_high": w52h,
            "week52_low": w52l,
            "pct_from_52w_low": ((price - w52l) / (w52l + 1e-9)) * 100 if w52l else 0,
            "analyst_target": target,
            "analyst_upside": upside,
            "pe_ratio": info.get("forwardPE") or info.get("trailingPE"),
            "beta": info.get("beta"),
            "hist": hist,
        }
    except Exception:
        return None


@st.cache_data(ttl=300, show_spinner=False)
def fetch_live_data(symbols: tuple) -> list[dict]:
    results = []
    with ThreadPoolExecutor(max_workers=6) as ex:
        futures = {ex.submit(_fetch_one, s): s for s in symbols}
        for f in as_completed(futures):
            r = f.result()
            if r:
                results.append(r)
    return results


def score_live(live: dict) -> tuple[float, dict]:
    rsi = live.get("rsi", 50)
    rsi_s = 90 if rsi < 35 else (15 if rsi > 75 else (70 if 40 <= rsi <= 60 else 50))

    price = live.get("current_price", 0)
    ma20, ma50 = live.get("ma20"), live.get("ma50")
    if ma20 and ma50 and price:
        ma_s = 85 if price > ma20 > ma50 else (65 if price > ma20 else 35)
    else:
        ma_s = 50

    upside = live.get("analyst_upside")
    analyst_s = (50 if upside is None else
                 95 if upside > 30 else
                 75 if upside > 15 else
                 55 if upside > 0 else 20)

    pct_low = live.get("pct_from_52w_low", 50)
    pos_s = 90 if pct_low < 15 else (70 if pct_low < 30 else (50 if pct_low < 60 else 30))

    composite = rsi_s * 0.25 + ma_s * 0.25 + analyst_s * 0.30 + pos_s * 0.20
    return composite, {"RSI": rsi_s, "MA Trend": ma_s, "Analyst Target": analyst_s, "52w Position": pos_s}


def _fmt_mcap(val: float) -> str:
    if val >= 1e12: return f"${val/1e12:.2f}T"
    if val >= 1e9:  return f"${val/1e9:.2f}B"
    if val >= 1e6:  return f"${val/1e6:.1f}M"
    return f"${val:,.0f}" if val > 0 else "N/A"


# ═══════════════════════════════════════════════════════════════════════════════
# LOCAL AI AGENT
# ═══════════════════════════════════════════════════════════════════════════════

KNOWLEDGE_BASE = {
    "rsi": """**RSI (Relative Strength Index)**

RSI measures price momentum on a 0–100 scale:

- **Below 30** — Oversold territory. Price has dropped sharply; some traders watch this level as a potential reversal zone.
- **30–50** — Recovering range. Downward pressure may be easing.
- **50–70** — Healthy momentum range. Positive trend without being stretched.
- **Above 70** — Overbought territory. Price has risen quickly; a pullback or consolidation is common.

*How it is used:* RSI crossing back above 30 is often watched as a potential stabilization signal. RSI crossing below 70 from above may indicate momentum is fading. Works best combined with broader trend context.

The **Top Picks** page uses CSV data for initial screening. The **Stock Detail** page fetches live RSI via yfinance.""",

    "macd": """**MACD (Moving Average Convergence Divergence)**

Tracks momentum via two exponential moving averages (12-day and 26-day EMA), with a 9-day signal line on top.

Key signals:
- **Bullish crossover**: MACD line crosses above signal line — often viewed as a positive momentum indicator
- **Bearish crossover**: MACD line crosses below signal line — often viewed as a negative momentum indicator
- **Zero-line cross**: MACD crossing above zero may indicate a broader uptrend is forming
- **Histogram**: Widening = strengthening trend; narrowing = weakening momentum

Best used to observe trend direction and momentum shifts. Lagging by nature — most useful for confirming existing trends rather than predicting new ones.""",

    "moving average": """**Moving Averages (MA)**

Smooth price noise to reveal underlying trend direction.

- **20-day MA**: Short-term trend. Price above MA20 = bullish near-term momentum.
- **50-day MA**: Medium-term. Widely watched by institutions.
- **200-day MA**: Long-term. Price above 200d = bull market territory.

Key signals:
- **Golden Cross**: 50d crosses above 200d — strong long-term bullish signal
- **Death Cross**: 50d crosses below 200d — bearish long-term signal
- **Price stacked above MA20 > MA50**: Strong bullish alignment

Live MA20 and MA50 are shown in the **Stock Detail** page chart.""",

    "volume": """**Volume Analysis**

Volume = number of shares traded. It is commonly used to assess the strength behind price moves.

- **High volume + rising price** — Often interpreted as a sign of strong interest, potentially institutional activity.
- **High volume + falling price** — May indicate broad selling pressure.
- **Low volume + rising price** — Some analysts view this as a less convincing move.
- **Volume spike** — Can signal increased market attention; direction matters.

Volume is generally considered alongside price to give context to how significant a move may be.""",

    "market cap": """**Market Capitalization**

Market cap = share price × shares outstanding.

- **Micro-cap** (< $300M): Extremely volatile. High risk, high reward.
- **Small-cap** ($300M–$2B): Growth sweet spot. Enough liquidity, still room to run.
- **Mid-cap** ($2B–$10B): Solid growth with lower volatility.
- **Large-cap** (> $10B): Stable, but slower growth potential.

The **Top Picks** scoring favors mid-cap stocks ($100M–$2B) as the sweet spot for large percentage gains.""",

    "momentum": """**Momentum Investing**

Momentum is the tendency for stocks that have been rising to continue rising over a period of time. Key observations:

- High % change relative to peers and the broader index
- Price near or at new 52-week highs
- Relative strength compared to its sector

How momentum is generally studied:
- Stocks in the top 20% of price performance over 1–6 months are commonly tracked by momentum strategies
- Stocks in established downtrends may not fit a momentum-based framework
- Momentum can reverse quickly, so risk management is considered an important part of any strategy""",

    "short squeeze": """**Short Squeeze**

When heavily shorted stocks rise, short sellers are forced to buy back shares to cover losses — driving the price even higher.

Requirements:
- High short interest (15–20%+ of float)
- A catalyst: earnings beat, news, product announcement
- Low float (fewer shares = faster price moves)

Famous examples: GameStop (GME), AMC, Bed Bath & Beyond.

Risks: Extremely volatile, timing is nearly impossible. Short squeezes often collapse as fast as they rise.""",

    "sector rotation": """**Sector Rotation**

Investment capital moves between sectors as the economic cycle shifts.

Economic cycle → Favored sectors:
- **Early recovery**: Financials, Consumer Discretionary, Technology
- **Mid expansion**: Industrials, Materials, Energy
- **Late cycle**: Energy, Healthcare, Utilities
- **Recession**: Utilities, Consumer Staples, Healthcare

Use the **Market Overview** page to see which sectors are outperforming today.""",

    "pe ratio": """**Price-to-Earnings (P/E) Ratio**

P/E = Stock Price ÷ Earnings Per Share (EPS)

- **Low P/E** (< 15): Potentially undervalued, or slow growth expected
- **Market average**: Historically 15–20 for the S&P 500
- **High P/E** (> 30): Market expects strong future growth (common in tech)
- **Negative P/E**: Company is currently losing money

Always compare P/E within the same sector. 40x P/E is normal for tech but expensive for utilities.""",

    "breakout": """**Breakout Trading**

A breakout occurs when price moves above a key resistance level, often accompanied by increased volume.

What traders commonly observe:
- Price consolidating sideways for an extended period
- A resistance level that has been tested multiple times
- Price closing above resistance on above-average volume (often 2x+)

Some traders watch for a move on the breakout day or a retest of the broken level. False breakouts are common — many analysts wait for a confirmed daily close above the level before drawing conclusions.""",

    "earnings": """**Earnings and Stock Price**

Quarterly earnings reports are among the most significant scheduled events for individual stocks.

- **Beat + raise guidance** — Historically associated with strong price increases (5–20%+)
- **Miss or lower guidance** — Historically associated with sharp price declines (5–15%+)
- **In-line results** — Often a muted reaction; the language around future guidance tends to matter most

A commonly observed pattern: stocks may rise ahead of an earnings report and then decline even after a positive result, if the outcome was already reflected in the price beforehand.

Stocks that have beaten analyst estimates for several consecutive quarters are sometimes studied for trend consistency, though past results do not predict future performance.""",

    "stop loss": """**Stop-Loss Orders**

A stop-loss is an order that automatically exits a position if price falls to a specified level, capping the potential loss on a trade.

Common approaches:
- **Percentage-based**: Exits if the stock drops a set percentage from the entry price (commonly 7–10% in growth-stock frameworks)
- **Technical**: Exits if price closes below a key support level or moving average
- **Trailing**: The exit level moves up as price rises, preserving a portion of any gains

Stop-loss orders are a risk management tool. How and whether to use them depends on individual strategy, time horizon, and risk tolerance.""",

    "diversification": """**Diversification**

Spreads risk across positions to reduce the impact of any single loss.

- Hold 10–20 positions for meaningful diversification
- Diversify across sectors (5 tech stocks is not diversified)
- Mix market caps (large + mid + small)
- Consider geographic diversification

Too many positions (40+) dilutes returns toward market-index levels. Concentrated portfolios outperform when picks are right — but lose more when wrong.""",

    "yfinance": """**Live Data (yfinance)**

This app uses yfinance to fetch real-time and historical stock data from Yahoo Finance.

Available in:
- **Stock Detail** page — live price, RSI, MA20/MA50 chart, 52-week range, analyst targets, and score breakdown
- **Top Picks** page — enable "Fetch Live Data" to layer RSI, analyst targets, and moving average signals on top of the CSV-based screening

yfinance data is cached for 5 minutes to avoid rate limits. If a symbol fails to load, the ticker may be delisted or temporarily unavailable.""",
}


def _stock_analysis_csv(symbol: str, df: pd.DataFrame) -> str:
    rows = df[df["Symbol"].str.upper() == symbol.upper()]
    if rows.empty:
        return f"I don't have **{symbol}** in the NASDAQ screener. Check the ticker or use the Screener page to search."

    row = rows.iloc[0]
    scored = compute_scores(rows)
    score = float(scored["score"].iloc[0])
    price, pct, vol, mcap = row["Last Sale"], row["% Change"], row["Volume"], row["Market Cap"]

    momentum_txt = ("Significant positive price movement today" if pct > 5 else
                    "Mild positive price movement today" if pct > 0 else
                    "Mild negative price movement today" if pct > -5 else
                    "Significant negative price movement today")
    vol_txt = ("Very high liquidity" if vol > 1_000_000 else
               "Good liquidity" if vol > 200_000 else
               "Lower liquidity — use limit orders" if vol > 50_000 else
               "Thin liquidity — exercise caution with position size")
    cap_txt = ("Large-cap — stable but slower growth" if mcap > 10e9 else
               "Mid-cap — solid growth profile" if mcap > 2e9 else
               "Small-cap — higher growth potential, higher volatility" if mcap > 300e6 else
               "Micro-cap — maximum volatility, high risk/reward")
    score_txt = ("Scores highly on momentum, volume, and market-cap metrics" if score >= 70 else
                 "Above average across screening metrics" if score >= 55 else
                 "Average across screening metrics" if score >= 40 else
                 "Below average across screening metrics")

    ipo = row.get("IPO Year")
    ipo_txt = f"\n- **IPO Year**: {int(ipo)}" if ipo and not np.isnan(float(ipo)) else ""

    return f"""**Analysis: {symbol}** — {row['Name']}

**Key Metrics (NASDAQ Screener)**
- **Price**: ${price:.2f}  |  **Daily Change**: {pct:+.2f}% ({row['Net Change']:+.2f})
- **Volume**: {vol:,.0f}
- **Market Cap**: {_fmt_mcap(mcap)}
- **Sector**: {row['Sector']}  |  **Industry**: {row.get('Industry', 'N/A')}
- **Country**: {row['Country']}{ipo_txt}

**Signal Assessment**
- Momentum: {momentum_txt}
- Liquidity: {vol_txt}
- Size: {cap_txt}
- CSV Score: **{score:.1f}/100** — {score_txt}

**Tip:** Go to the **Stock Detail** page and click "Fetch Live Analysis" for live RSI, moving average chart, 52-week range, and analyst price target for {symbol}."""


def _sector_analysis(sector: str, df: pd.DataFrame) -> str:
    sec_df = df[df["Sector"] == sector]
    if sec_df.empty:
        return f"No stocks found for sector: **{sector}**."
    avg_pct = sec_df["% Change"].mean()
    gainers = (sec_df["% Change"] > 0).sum()
    total = len(sec_df)
    sentiment = ("Strongly bullish" if avg_pct > 2 else "Mildly bullish" if avg_pct > 0.2 else
                 "Neutral/mixed" if avg_pct > -0.2 else "Mildly bearish" if avg_pct > -2 else "Strongly bearish")
    top5 = sec_df.nlargest(5, "% Change")
    lines = "\n".join(
        f"  {i+1}. **{r['Symbol']}** — {r['Name'][:30]} | ${r['Last Sale']:.2f} | {r['% Change']:+.2f}%"
        for i, (_, r) in enumerate(top5.iterrows())
    )
    return (f"**Sector: {sector}**\n\n"
            f"- Stocks: {total} | Gainers: {gainers} ({gainers/total*100:.0f}%)\n"
            f"- Average change: {avg_pct:+.2f}% | Sentiment: **{sentiment}**\n\n"
            f"**Top 5 Today**\n{lines}")


def _top_picks_summary(df: pd.DataFrame) -> str:
    scored = compute_scores(
        df[(df["Market Cap"] >= 100e6) & (df["Volume"] >= 100_000) & (df["Last Sale"] >= 1.0)]
    ).nlargest(10, "score")
    lines = "\n".join(
        f"  {i+1}. **{r['Symbol']}** — {r['Name'][:30]} | ${r['Last Sale']:.2f} | {r['% Change']:+.2f}% | Score: {r['score']:.0f}/100"
        for i, (_, r) in enumerate(scored.iterrows())
    )
    return (f"**Top 10 High-Gain Candidates (CSV scoring)**\n\n{lines}\n\n"
            "Go to the **Top Picks** page and enable *Fetch Live Data* to add RSI, analyst targets, and MA signals.")


def _gainers_losers(df: pd.DataFrame, mode: str = "gainers", n: int = 10) -> str:
    f = df[(df["Market Cap"] >= 50e6) & (df["Volume"] >= 50_000)]
    top = f.nlargest(n, "% Change") if mode == "gainers" else f.nsmallest(n, "% Change")
    header = f"**Top {n} {'Gainers' if mode == 'gainers' else 'Losers'} Today**"
    lines = "\n".join(
        f"  {i+1}. **{r['Symbol']}** | {r['Name'][:28]} | ${r['Last Sale']:.2f} | {r['% Change']:+.2f}%"
        for i, (_, r) in enumerate(top.iterrows())
    )
    return f"{header}\n\n{lines}\n\n*(Filtered: $50M+ market cap, 50K+ volume)*"


def _most_active(df: pd.DataFrame, n: int = 10) -> str:
    top = df.nlargest(n, "Volume")
    lines = "\n".join(
        f"  {i+1}. **{r['Symbol']}** | {r['Name'][:28]} | ${r['Last Sale']:.2f} | Vol: {r['Volume']:,.0f} | {r['% Change']:+.2f}%"
        for i, (_, r) in enumerate(top.iterrows())
    )
    return f"**Top {n} Most Active (by Volume)**\n\n{lines}"


DISCLAIMER = "\n\n---\n*All information provided is for informational purposes only and does not constitute financial advice. Always do your own research and consult a licensed financial advisor before making investment decisions.*"


# ── OpenAI client ──────────────────────────────────────────────────────────────

@st.cache_resource
def get_openai_client():
    base_url = os.environ.get("AI_INTEGRATIONS_OPENAI_BASE_URL")
    api_key  = os.environ.get("AI_INTEGRATIONS_OPENAI_API_KEY")
    if not base_url or not api_key:
        return None
    try:
        return OpenAI(base_url=base_url, api_key=api_key)
    except Exception:
        return None


# ── Tool implementations ───────────────────────────────────────────────────────

def _tool_market_overview(df: pd.DataFrame) -> str:
    total = len(df)
    g = int((df["% Change"] > 0).sum())
    l = int((df["% Change"] < 0).sum())
    avg = df["% Change"].mean()
    perf = (df[df["Sector"] != "Unknown"]
            .groupby("Sector")["% Change"].mean()
            .sort_values(ascending=False))
    sector_lines = "\n".join(f"  {s}: {p:+.2f}%" for s, p in perf.items())
    return (f"Total stocks: {total:,} | Gainers: {g:,} ({g/total*100:.1f}%) | "
            f"Losers: {l:,} ({l/total*100:.1f}%) | Unchanged: {total-g-l:,}\n"
            f"Average daily change: {avg:+.2f}%\n"
            f"Best sector: {perf.index[0]} ({perf.iloc[0]:+.2f}%) | "
            f"Worst sector: {perf.index[-1]} ({perf.iloc[-1]:+.2f}%)\n\n"
            f"All sectors:\n{sector_lines}")


def _tool_sector(sector: str, df: pd.DataFrame) -> str:
    sectors = [s for s in df["Sector"].unique() if s and s != "Unknown"]
    match = next((s for s in sectors if s.lower() == sector.lower()), None)
    if not match:
        match = next((s for s in sectors if sector.lower() in s.lower()), None)
    if match:
        return _sector_analysis(match, df)
    return f"Sector '{sector}' not found. Available sectors: {', '.join(sorted(sectors))}"


def _tool_compare(symbols: list, df: pd.DataFrame) -> str:
    results, missing = [], []
    for sym in symbols[:6]:
        rows = df[df["Symbol"].str.upper() == sym.upper()]
        if not rows.empty:
            results.append(compute_scores(rows).iloc[0])
        else:
            missing.append(sym.upper())
    if not results:
        return "None of the specified symbols were found in the dataset."
    lines = [
        f"{r['Symbol']} | {r['Name'][:25]} | ${r['Last Sale']:.2f} | "
        f"{r['% Change']:+.2f}% | Vol: {r['Volume']:,.0f} | "
        f"MCap: {_fmt_mcap(r['Market Cap'])} | Score: {r['score']:.0f}/100"
        for r in results
    ]
    out = "\n".join(lines)
    if missing:
        out += f"\n\nNot found in dataset: {', '.join(missing)}"
    return out


# ── Tool specs (OpenAI function-calling format) ────────────────────────────────

_TOOL_SPECS = [
    {"type": "function", "function": {
        "name": "lookup_stock",
        "description": "Look up price, volume, sector, market cap, and composite score for a specific stock ticker from the NASDAQ dataset.",
        "parameters": {"type": "object",
                       "properties": {"symbol": {"type": "string", "description": "Ticker symbol e.g. AAPL, NVDA, TSLA"}},
                       "required": ["symbol"]},
    }},
    {"type": "function", "function": {
        "name": "get_top_gainers",
        "description": "Get the top gaining stocks today ranked by % price change.",
        "parameters": {"type": "object",
                       "properties": {"n": {"type": "integer", "description": "How many to return (default 10)"}}},
    }},
    {"type": "function", "function": {
        "name": "get_top_losers",
        "description": "Get the biggest losing stocks today ranked by % price change.",
        "parameters": {"type": "object",
                       "properties": {"n": {"type": "integer", "description": "How many to return (default 10)"}}},
    }},
    {"type": "function", "function": {
        "name": "get_most_active",
        "description": "Get the most actively traded stocks by volume today.",
        "parameters": {"type": "object",
                       "properties": {"n": {"type": "integer", "description": "How many to return (default 10)"}}},
    }},
    {"type": "function", "function": {
        "name": "get_top_picks",
        "description": "Get top-scored stocks from the composite screening model (momentum + volume + market cap + price).",
        "parameters": {"type": "object", "properties": {}},
    }},
    {"type": "function", "function": {
        "name": "get_sector_analysis",
        "description": "Get performance breakdown and top stocks for a specific market sector.",
        "parameters": {"type": "object",
                       "properties": {"sector": {"type": "string", "description": "Sector name e.g. Technology, Healthcare, Finance, Energy"}},
                       "required": ["sector"]},
    }},
    {"type": "function", "function": {
        "name": "get_market_overview",
        "description": "Get broad market overview: gainers/losers count, sector performance rankings, average daily change.",
        "parameters": {"type": "object", "properties": {}},
    }},
    {"type": "function", "function": {
        "name": "compare_stocks",
        "description": "Compare multiple stocks side by side on price, change, volume, market cap, and composite score.",
        "parameters": {"type": "object",
                       "properties": {"symbols": {"type": "array", "items": {"type": "string"},
                                                  "description": "Ticker symbols to compare e.g. ['AAPL','MSFT','GOOGL']"}},
                       "required": ["symbols"]},
    }},
]

_SYSTEM_PROMPT = """You are a stock market data analyst assistant embedded in a NASDAQ stock screener app. You have access to live NASDAQ screening data covering 7,000+ stocks with price, volume, sector, market cap, and daily % change.

Use the provided tools to fetch real data before answering questions about specific stocks, sectors, or market conditions. Call multiple tools in parallel when useful.

Rules:
- Always base your answers on real data from the tools, not assumptions or general knowledge about prices
- Format data clearly using markdown tables or numbered lists
- Never make explicit investment recommendations or tell users to buy or sell any asset
- Present analysis objectively and note it is for informational purposes only
- When users ask about live RSI, analyst price targets, or charts, mention the Stock Detail page for live yfinance data
- Be conversational and precise; don't pad responses unnecessarily
- For educational questions (what is RSI, how does MACD work, etc.), explain clearly without excessive jargon"""


def _dispatch_tool(name: str, args: dict, df: pd.DataFrame) -> str:
    try:
        if name == "lookup_stock":
            return _stock_analysis_csv(args.get("symbol", "").upper(), df)
        elif name == "get_top_gainers":
            return _gainers_losers(df, "gainers", int(args.get("n", 10)))
        elif name == "get_top_losers":
            return _gainers_losers(df, "losers", int(args.get("n", 10)))
        elif name == "get_most_active":
            return _most_active(df, int(args.get("n", 10)))
        elif name == "get_top_picks":
            return _top_picks_summary(df)
        elif name == "get_sector_analysis":
            return _tool_sector(args.get("sector", ""), df)
        elif name == "get_market_overview":
            return _tool_market_overview(df)
        elif name == "compare_stocks":
            return _tool_compare(args.get("symbols", []), df)
        return f"Unknown tool: {name}"
    except Exception as e:
        return f"Error in {name}: {e}"


def agent_respond_stream(user_msg: str, df: pd.DataFrame, history: list = None):
    """Streaming generator that yields response text tokens."""
    client = get_openai_client()

    if not client:
        # Fallback: rule-based responses
        yield _agent_respond_inner_fallback(user_msg, df) + DISCLAIMER
        return

    # Build message list with trimmed conversation history
    messages = [{"role": "system", "content": _SYSTEM_PROMPT}]
    if history:
        for m in history[-20:]:
            if m.get("role") in ("user", "assistant") and m.get("content"):
                # Strip the disclaimer from stored assistant messages before sending
                content = m["content"]
                if "---\n*All information" in content:
                    content = content[:content.index("---\n*All information")].rstrip()
                messages.append({"role": m["role"], "content": content})
    messages.append({"role": "user", "content": user_msg})

    try:
        # Step 1: first call handles tool selection (not streamed, needed for tool use)
        # reasoning_effort='none' required for function tools with gpt-5.6 models
        resp = client.chat.completions.create(
            model="gpt-5.6-luna",
            messages=messages,
            tools=_TOOL_SPECS,
            tool_choice="auto",
            max_completion_tokens=2048,
            reasoning_effort="none",
        )
        msg = resp.choices[0].message

        # Step 2: execute any requested tool calls
        if msg.tool_calls:
            messages.append(msg)
            for tc in msg.tool_calls:
                args = json.loads(tc.function.arguments or "{}")
                result = _dispatch_tool(tc.function.name, args, df)
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": result,
                })

            # Step 3: stream the synthesised response
            stream = client.chat.completions.create(
                model="gpt-5.6-luna",
                messages=messages,
                max_completion_tokens=2048,
                stream=True,
            )
            for chunk in stream:
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta.content
                if delta:
                    yield delta
        else:
            # No tools needed — direct answer
            yield msg.content or ""

        yield DISCLAIMER

    except Exception as e:
        try:
            yield _agent_respond_inner_fallback(user_msg, df) + DISCLAIMER
        except Exception:
            yield f"I encountered an error processing your request. Please try again.\n\n*Error: {e}*" + DISCLAIMER


# ── Rule-based fallback (kept for offline / error cases) ─────────────────────

def _agent_respond_inner_fallback(user_msg: str, df: pd.DataFrame) -> str:
    msg = user_msg.strip()
    msg_lower = msg.lower()

    analyze_match = re.search(
        r"\b(?:analyze|analysis|tell me about|what about|look at|check|research|thoughts on|opinion on)\s+([A-Z]{1,5})\b",
        msg, re.IGNORECASE)
    if analyze_match:
        return _stock_analysis_csv(analyze_match.group(1).upper(), df)

    bare = re.match(r"^\s*([A-Z]{1,5})\s*\??$", msg.upper())
    if bare and bare.group(1) not in {"RSI", "PE", "MA", "EMA", "SMA", "ETF", "IPO"}:
        return _stock_analysis_csv(bare.group(1), df)

    vs_match = re.search(r"\b([A-Z]{1,5})\s+vs\.?\s+([A-Z]{1,5})\b", msg.upper())
    if vs_match:
        return _tool_compare([vs_match.group(1), vs_match.group(2)], df)

    if re.search(r"\b(top pick|best stock|high gain|highest score|highest scoring)\b", msg_lower):
        return _top_picks_summary(df)
    if re.search(r"\b(top gainer|biggest gainer|most gained|best performer)\b", msg_lower):
        return _gainers_losers(df, "gainers")
    if re.search(r"\b(top loser|biggest loser|most lost|worst)\b", msg_lower):
        return _gainers_losers(df, "losers")
    if re.search(r"\b(most active|highest volume|most traded)\b", msg_lower):
        return _most_active(df)
    if re.search(r"\bmarket\b", msg_lower):
        return _tool_market_overview(df)

    for sec in [s for s in df["Sector"].unique() if s and s != "Unknown"]:
        if sec.lower() in msg_lower:
            return _sector_analysis(sec, df)

    for key, content in KNOWLEDGE_BASE.items():
        if key in msg_lower:
            return content

    ignore = {"I","A","AN","THE","IS","IT","IN","OF","TO","DO","BE","GO","MY","WE","US","ON",
              "AT","BY","AS","UP","OR","AND","FOR","ARE","WAS","RSI","PE","MA","EMA","SMA",
              "ETF","IPO","CEO","CFO","CTO","EPS","ROI","AI"}
    for t in re.findall(r"\b([A-Z]{1,5})\b", msg.upper()):
        if t not in ignore and not df[df["Symbol"].str.upper() == t].empty:
            return _stock_analysis_csv(t, df)

    return ("Ask me anything about stocks or the market, for example:\n\n"
            "- *How is the market today?*\n"
            "- *Tell me about NVDA*\n"
            "- *Compare AAPL vs MSFT*\n"
            "- *Show top picks*\n"
            "- *How is the tech sector doing?*\n"
            "- *Explain RSI or MACD*")


# ═══════════════════════════════════════════════════════════════════════════════
# PAGES
# ═══════════════════════════════════════════════════════════════════════════════

def page_market_overview(df: pd.DataFrame):
    st.title("Market Overview")
    total = len(df)
    gainers = int((df["% Change"] > 0).sum())
    losers  = int((df["% Change"] < 0).sum())
    unchanged = total - gainers - losers
    avg_chg = df["% Change"].mean()

    # Sentiment pill
    if avg_chg > 0.5:
        sent, scol = "BULLISH", "#22c55e"
    elif avg_chg > -0.5:
        sent, scol = "NEUTRAL", "#f59e0b"
    else:
        sent, scol = "BEARISH", "#ef4444"

    st.markdown(
        f'<div style="display:flex;align-items:center;gap:10px;margin-bottom:1.2rem">'
        f'<span style="color:#475569;font-size:0.8rem">NASDAQ Screener Snapshot · {total:,} stocks</span>'
        f'&nbsp;{_badge(f"⬤  {sent}", scol)}'
        f'&nbsp;<span style="color:{scol};font-size:0.8rem;font-weight:600">Avg {avg_chg:+.2f}%</span>'
        f'</div>',
        unsafe_allow_html=True,
    )

    # Hero stat cards
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(_stat_card("Total Stocks", f"{total:,}", "in NASDAQ dataset", "#6366f1", "📋"), unsafe_allow_html=True)
    with c2:
        st.markdown(_stat_card("Gainers", f"{gainers:,}", f"{gainers/total*100:.1f}% of market", "#22c55e", "▲"), unsafe_allow_html=True)
    with c3:
        st.markdown(_stat_card("Losers", f"{losers:,}", f"{losers/total*100:.1f}% of market", "#ef4444", "▼"), unsafe_allow_html=True)
    with c4:
        st.markdown(_stat_card("Unchanged", f"{unchanged:,}", "flat on the day", "#94a3b8", "◆"), unsafe_allow_html=True)

    st.markdown("<div style='height:1.6rem'></div>", unsafe_allow_html=True)

    sec_df = df[df["Sector"] != "Unknown"]
    sec_breadth = (
        sec_df.groupby("Sector")
        .apply(lambda g: pd.Series({
            "Gainers": int((g["% Change"] > 0).sum()),
            "Losers":  int((g["% Change"] < 0).sum()),
            "Avg % Change": g["% Change"].mean(),
        }))
        .reset_index()
        .sort_values("Avg % Change", ascending=False)
    )

    # Both charts horizontal → no rotated labels, consistent layout
    st.markdown(_section_hdr("Sector Performance"), unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        # Gainers vs Losers as stacked horizontal bars (% of sector)
        sb = sec_breadth.copy()
        sb["Total"] = sb["Gainers"] + sb["Losers"]
        sb["% Gainers"] = sb["Gainers"] / sb["Total"] * 100
        sb["% Losers"]  = sb["Losers"]  / sb["Total"] * 100
        sb_sorted = sb.sort_values("% Gainers", ascending=True)
        fig = go.Figure()
        fig.add_trace(go.Bar(
            y=sb_sorted["Sector"], x=sb_sorted["% Gainers"],
            name="Gainers", orientation="h",
            marker=dict(color="rgba(16,185,129,0.75)", line=dict(width=0)),
        ))
        fig.add_trace(go.Bar(
            y=sb_sorted["Sector"], x=sb_sorted["% Losers"],
            name="Losers", orientation="h",
            marker=dict(color="rgba(244,63,94,0.65)", line=dict(width=0)),
        ))
        fig.update_layout(
            barmode="stack", bargap=0.22,
            xaxis=dict(ticksuffix="%", range=[0, 100]),
            legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="right", x=1),
        )
        st.plotly_chart(_chart_dark(fig, 380), use_container_width=True)

    with c2:
        # Average % change — horizontal, colored by value
        sb2 = sec_breadth.sort_values("Avg % Change", ascending=True)
        colors = ["rgba(244,63,94,0.8)" if v < 0 else "rgba(16,185,129,0.8)"
                  for v in sb2["Avg % Change"]]
        fig2 = go.Figure(go.Bar(
            y=sb2["Sector"], x=sb2["Avg % Change"],
            orientation="h",
            marker=dict(color=colors, line=dict(width=0)),
            text=[f"{v:+.2f}%" for v in sb2["Avg % Change"]],
            textposition="outside",
            textfont=dict(color="#4a5e7a", size=10),
        ))
        fig2.add_vline(x=0, line_color="rgba(255,255,255,0.1)", line_width=1)
        st.plotly_chart(_chart_dark(fig2, 380), use_container_width=True)

    st.markdown(_section_hdr("Return Distribution", f"{len(df):,} stocks · clipped ±20%"), unsafe_allow_html=True)
    fig3 = px.histogram(
        df[(df["% Change"] > -20) & (df["% Change"] < 20)],
        x="% Change", nbins=80,
        color_discrete_sequence=["rgba(99,102,241,0.7)"],
    )
    fig3.add_vline(x=0, line_dash="dot", line_color="rgba(255,255,255,0.2)", line_width=1)
    fig3.update_traces(marker_line_width=0)
    st.plotly_chart(_chart_dark(fig3, 220), use_container_width=True)

    st.markdown(_section_hdr("Movers Today", "$50M+ cap · 50K+ volume"), unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    qualified = df[(df["Market Cap"] >= 50e6) & (df["Volume"] >= 50_000)]
    with c1:
        st.markdown('<div style="color:#10b981;font-size:0.72rem;font-weight:700;text-transform:uppercase;letter-spacing:0.1em;margin-bottom:8px">▲ Top 10 Gainers</div>', unsafe_allow_html=True)
        top_g = qualified.nlargest(10, "% Change")
        st.dataframe(
            top_g[["Symbol","Name","Last Sale","% Change","Volume"]].style
            .format({"Last Sale":"${:.2f}","% Change":"{:+.2f}%","Volume":"{:,.0f}"})
            .map(lambda v: "color:#10b981;font-weight:700" if isinstance(v, (int,float)) and v > 0 else "", subset=["% Change"]),
            use_container_width=True, height=340,
        )
    with c2:
        st.markdown('<div style="color:#f43f5e;font-size:0.72rem;font-weight:700;text-transform:uppercase;letter-spacing:0.1em;margin-bottom:8px">▼ Top 10 Losers</div>', unsafe_allow_html=True)
        top_l = qualified.nsmallest(10, "% Change")
        st.dataframe(
            top_l[["Symbol","Name","Last Sale","% Change","Volume"]].style
            .format({"Last Sale":"${:.2f}","% Change":"{:+.2f}%","Volume":"{:,.0f}"})
            .map(lambda v: "color:#f43f5e;font-weight:700" if isinstance(v, (int,float)) and v < 0 else "", subset=["% Change"]),
            use_container_width=True, height=340,
        )


def page_screener(df: pd.DataFrame):
    st.title("NASDAQ Screener")
    st.caption(f"{len(df):,} stocks · filter by price, cap, volume, sector, and direction")

    with st.expander("⚙️  Filters", expanded=True):
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            min_price = st.number_input("Min Price ($)", 0.0, value=0.5, step=0.5)
            max_price = st.number_input("Max Price ($)", 0.0, value=10000.0, step=10.0)
        with c2:
            cap_opts = {"Any":0,"$50M+":50e6,"$100M+":100e6,"$300M+":300e6,"$1B+":1e9,"$10B+":10e9}
            min_cap = cap_opts[st.selectbox("Min Market Cap", list(cap_opts.keys()))]
        with c3:
            vol_opts = {"Any":0,"10K+":10_000,"100K+":100_000,"500K+":500_000,"1M+":1_000_000}
            min_vol = vol_opts[st.selectbox("Min Volume", list(vol_opts.keys()))]
        with c4:
            sectors = ["All"] + sorted(s for s in df["Sector"].unique() if s != "Unknown")
            sector = st.selectbox("Sector", sectors)
            direction = st.selectbox("Direction", ["All","Gainers","Losers"])

    sort_map = {"% Change ↓":("% Change",False),"% Change ↑":("% Change",True),
                "Volume ↓":("Volume",False),"Market Cap ↓":("Market Cap",False),
                "Price ↓":("Last Sale",False),"Price ↑":("Last Sale",True)}
    sort_col, asc = sort_map[st.selectbox("Sort by", list(sort_map.keys()))]

    filtered = filter_df(df, min_price, max_price, min_cap, min_vol, sector, direction)
    filtered = filtered.sort_values(sort_col, ascending=asc)

    match_pct = len(filtered)/len(df)*100
    c1, c2, c3 = st.columns(3)
    c1.metric("Matching Stocks", f"{len(filtered):,}", f"{match_pct:.1f}% of universe")
    c2.metric("Gainers in set", f"{int((filtered['% Change']>0).sum()):,}")
    c3.metric("Avg Change", f"{filtered['% Change'].mean():+.2f}%" if len(filtered) else "—")

    st.dataframe(
        filtered[["Symbol","Name","Last Sale","Net Change","% Change","Volume","Market Cap","Sector","Industry","Country"]]
        .head(500).style.format({
            "Last Sale":"${:.2f}","Net Change":"{:+.2f}",
            "% Change":"{:+.2f}%","Volume":"{:,.0f}","Market Cap":"${:,.0f}",
        }),
        use_container_width=True, height=540,
    )


def page_top_picks(df: pd.DataFrame):
    st.title("Top Picks")
    st.caption("Composite-scored candidates · enable live data for RSI, MA signals, and analyst targets")

    with st.sidebar:
        st.markdown(
            '<div style="color:#6366f1;font-size:0.7rem;font-weight:700;text-transform:uppercase;'
            'letter-spacing:0.1em;margin-bottom:8px">⚙ Filters</div>',
            unsafe_allow_html=True,
        )
        min_price = st.slider("Min Price ($)", 0.5, 50.0, 1.0, 0.5)
        cap_opts = {"$50M+":50e6,"$100M+":100e6,"$300M+":300e6,"$1B+":1e9}
        min_cap = cap_opts[st.selectbox("Min Market Cap", list(cap_opts.keys()), index=1)]
        vol_opts = {"50K+":50_000,"100K+":100_000,"500K+":500_000,"1M+":1_000_000}
        min_vol = vol_opts[st.selectbox("Min Volume", list(vol_opts.keys()), index=1)]
        sectors = ["All"] + sorted(s for s in df["Sector"].unique() if s != "Unknown")
        sector = st.selectbox("Sector", sectors)
        dir_map = {"All":"All","Gainers Only":"Gainers","Losers Only":"Losers"}
        direction = dir_map[st.selectbox("Direction", list(dir_map.keys()))]
        top_n = st.slider("Results to show", 10, 150, 50)
        st.markdown("---")
        fetch_live = st.checkbox("⚡ Fetch Live Data (yfinance)", value=False,
                                  help="Adds RSI, MA signals, analyst targets. ~30s for top 30.")

    filtered = filter_df(df, min_price, 1e9, min_cap, min_vol, sector, direction)
    if filtered.empty:
        st.warning("No stocks match the current filters.")
        return

    scored = compute_scores(filtered).nlargest(top_n, "score").reset_index(drop=True)

    if fetch_live:
        symbols = tuple(scored["Symbol"].tolist()[:30])
        with st.spinner(f"⚡ Fetching live data for {len(symbols)} candidates…"):
            live_list = fetch_live_data(symbols)
        live_dict = {r["symbol"]: r for r in live_list}

        rows = []
        for _, row in scored.iterrows():
            sym = row["Symbol"]
            if sym not in live_dict:
                continue
            live = live_dict[sym]
            ls, _ = score_live(live)
            rows.append({
                "Symbol": sym, "Name": row["Name"][:35],
                "Price": live["current_price"], "RSI": live["rsi"],
                "vs 52w Low": live["pct_from_52w_low"],
                "Analyst Upside %": live["analyst_upside"],
                "Beta": live["beta"], "Sector": row["Sector"], "Live Score": ls,
            })

        if not rows:
            st.error("Could not retrieve live data — yfinance may be rate-limited. Try again in a moment.")
            return

        result_df = pd.DataFrame(rows).sort_values("Live Score", ascending=False).reset_index(drop=True)

        # Medal cards
        st.markdown("#### 🏆 Top 3 Live-Scored")
        cols = st.columns(3)
        for i, (col, (_, r)) in enumerate(zip(cols, result_df.head(3).iterrows())):
            with col:
                upside = f"+{r['Analyst Upside %']:.1f}%" if r["Analyst Upside %"] else "No target"
                st.markdown(
                    _medal_card(i+1, r["Symbol"], r["Name"], r["Price"],
                                0.0, r["Live Score"], r["Sector"]),
                    unsafe_allow_html=True,
                )
                st.caption(f"RSI: {r['RSI']:.1f}  ·  Analyst upside: {upside}")

        st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
        st.markdown("#### Full Live-Scored List")

        def _score_color(val):
            t = max(0.0, min(1.0, val / 100.0))
            r_ = int(220*(1-t)*2) if t >= 0.5 else 220
            g_ = 220 if t >= 0.5 else int(220*t*2)
            return f"background-color:rgba({r_},{g_},80,0.18);color:{'#4ade80' if t>0.6 else ('#facc15' if t>0.4 else '#f87171')};font-weight:700"

        st.dataframe(
            result_df.style
            .format({
                "Price":"${:.2f}", "RSI":"{:.1f}", "vs 52w Low":"{:+.1f}%",
                "Analyst Upside %": lambda x: f"+{x:.1f}%" if x else "N/A",
                "Beta": lambda x: f"{x:.2f}" if x else "N/A", "Live Score":"{:.1f}",
            })
            .map(_score_color, subset=["Live Score"]),
            use_container_width=True, height=500,
        )

    else:
        st.markdown(
            '<div style="background:rgba(99,102,241,0.08);border:1px solid rgba(99,102,241,0.22);'
            'border-radius:11px;padding:12px 16px;font-size:0.82rem;color:#94a3b8;margin-bottom:1rem">'
            '⚡ Enable <b style="color:#a5b4fc">Fetch Live Data</b> in the sidebar to layer RSI, '
            'analyst targets, and moving-average signals on top of the CSV score.</div>',
            unsafe_allow_html=True,
        )

        # Medal cards for top 3
        st.markdown("#### 🏆 Top Ranked by Score")
        cols = st.columns(3)
        for i, (col, (_, r)) in enumerate(zip(cols, scored.head(3).iterrows())):
            with col:
                st.markdown(
                    _medal_card(i+1, r["Symbol"], r["Name"], r["Last Sale"],
                                r["% Change"], r["score"], r["Sector"]),
                    unsafe_allow_html=True,
                )

        st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)

        disp = scored[["Symbol","Name","Last Sale","% Change","Volume","Market Cap","Sector","score"]].rename(
            columns={"Last Sale":"Price","score":"Score"})

        def _score_color(val):
            t = max(0.0, min(1.0, val / 100.0))
            r_ = int(220*(1-t)*2) if t >= 0.5 else 220
            g_ = 220 if t >= 0.5 else int(220*t*2)
            return f"background-color:rgba({r_},{g_},80,0.18);color:{'#4ade80' if t>0.6 else ('#facc15' if t>0.4 else '#f87171')};font-weight:700"

        def _pct_color(val):
            if isinstance(val, (int, float)):
                return f"color:{'#4ade80' if val>0 else '#f87171'};font-weight:600"
            return ""

        st.dataframe(
            disp.style
            .format({"Price":"${:.2f}","% Change":"{:+.2f}%","Volume":"{:,.0f}",
                     "Market Cap":"${:,.0f}","Score":"{:.1f}"})
            .map(_score_color, subset=["Score"])
            .map(_pct_color, subset=["% Change"]),
            use_container_width=True, height=460,
        )

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("#### Score Distribution")
            fig = px.histogram(scored, x="score", nbins=20, color_discrete_sequence=["#6366f1"])
            fig.update_traces(marker_line_width=0, opacity=0.85)
            st.plotly_chart(_chart_dark(fig, 260), use_container_width=True)
        with c2:
            st.markdown("#### Sector Mix")
            sec_c = scored.groupby("Sector").size().reset_index(name="Count")
            fig2 = px.bar(sec_c.sort_values("Count"), x="Count", y="Sector",
                          orientation="h", color="Count", color_continuous_scale="Purples")
            st.plotly_chart(_chart_dark(fig2, 260), use_container_width=True)


def page_stock_detail(df: pd.DataFrame):
    st.title("Stock Detail")
    st.caption("CSV snapshot + live yfinance data: price chart, RSI, moving averages, analyst targets, and score")

    symbols = sorted(df["Symbol"].dropna().unique().tolist())
    default_idx = symbols.index("AAPL") if "AAPL" in symbols else 0
    selected = st.selectbox("Search ticker", symbols, index=default_idx)

    rows = df[df["Symbol"] == selected]
    if rows.empty:
        st.error("Stock not found.")
        return

    row = rows.iloc[0]
    scored = compute_scores(rows)
    csv_score = float(scored["score"].iloc[0])
    pct_color = "#22c55e" if row["% Change"] >= 0 else "#ef4444"
    pct_sign  = "+" if row["% Change"] >= 0 else ""

    # ── Stock header card ──────────────────────────────────────────────────────
    st.markdown(f"""
    <div style="
        background:linear-gradient(135deg,#0c1525 0%,#0f1d38 100%);
        border:1px solid rgba(99,102,241,0.25); border-radius:18px;
        padding:22px 26px; margin-bottom:1.2rem;
        box-shadow:0 6px 32px rgba(0,0,0,0.5),inset 0 1px 0 rgba(255,255,255,0.04);
    ">
      <div style="display:flex;align-items:flex-start;justify-content:space-between;flex-wrap:wrap;gap:12px">
        <div>
          <div style="display:flex;align-items:center;gap:10px;margin-bottom:6px">
            <span style="font-family:'JetBrains Mono',monospace;font-size:1.9rem;font-weight:800;
                color:#a5b4fc;letter-spacing:0.04em">{selected}</span>
            {_badge(row['Sector'], "#6366f1") if row['Sector'] != 'Unknown' else ''}
          </div>
          <div style="color:#64748b;font-size:0.88rem;font-weight:500;margin-bottom:10px">{row['Name']}</div>
          <div style="display:flex;gap:16px;flex-wrap:wrap">
            <span style="color:#94a3b8;font-size:0.8rem">📍 {row['Country']}</span>
            <span style="color:#94a3b8;font-size:0.8rem">🏭 {row.get('Industry','N/A')}</span>
          </div>
        </div>
        <div style="text-align:right">
          <div style="color:#e2e8f0;font-size:2.4rem;font-weight:900;letter-spacing:-0.035em;line-height:1">
            ${row['Last Sale']:.2f}
          </div>
          <div style="color:{pct_color};font-size:1.1rem;font-weight:700;margin-top:4px">
            {pct_sign}{row['% Change']:.2f}%&nbsp;
            <span style="font-size:0.85rem;opacity:0.8">({row['Net Change']:+.2f})</span>
          </div>
          <div style="color:#475569;font-size:0.75rem;margin-top:6px">
            Vol {row['Volume']:,.0f} · {_fmt_mcap(row['Market Cap'])} cap · CSV score
            <b style="color:#a5b4fc">{csv_score:.0f}/100</b>
          </div>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    if st.button("⚡ Fetch Live Analysis (yfinance)", type="primary"):
        with st.spinner(f"Fetching live data for {selected}…"):
            live_list = fetch_live_data((selected,))

        if not live_list:
            st.error(f"Could not fetch live data for **{selected}**. The symbol may be delisted or yfinance is rate-limited.")
        else:
            live = live_list[0]
            hist = live["hist"]
            price = live["current_price"]

            lc1, lc2, lc3, lc4 = st.columns(4)
            rsi = live["rsi"]
            lc1.metric("Live Price", f"${price:.2f}")
            lc2.metric("RSI (14)", f"{rsi:.1f}",
                       "Oversold" if rsi < 35 else ("Overbought" if rsi > 70 else "Neutral"))
            if live["analyst_upside"] is not None:
                lc3.metric("Analyst Upside", f"{live['analyst_upside']:+.1f}%",
                           f"Target: ${live['analyst_target']:.2f}")
            else:
                lc3.metric("Analyst Upside", "N/A")
            lc4.metric("Beta", f"{live['beta']:.2f}" if live["beta"] else "N/A")

            st.markdown("#### Price Chart — 3 Months")
            fig = go.Figure()
            fig.add_trace(go.Candlestick(
                x=hist.index, open=hist["Open"], high=hist["High"],
                low=hist["Low"], close=hist["Close"], name="OHLC",
                increasing_line_color="#22c55e", decreasing_line_color="#ef4444",
                increasing_fillcolor="rgba(34,197,94,0.15)",
                decreasing_fillcolor="rgba(239,68,68,0.15)",
            ))
            if len(hist) >= 20:
                fig.add_trace(go.Scatter(x=hist.index, y=hist["Close"].rolling(20).mean(),
                                         name="MA20", line=dict(color="#f59e0b", width=1.5)))
            if len(hist) >= 50:
                fig.add_trace(go.Scatter(x=hist.index, y=hist["Close"].rolling(50).mean(),
                                         name="MA50", line=dict(color="#6366f1", width=1.5)))
            fig.update_layout(xaxis_rangeslider_visible=False)
            st.plotly_chart(_chart_dark(fig, 420), use_container_width=True)

            cl, cr = st.columns(2)
            with cl:
                st.markdown("#### 52-Week Range")
                lo, hi = live["week52_low"], live["week52_high"]
                pct_pos = float(np.clip((price - lo) / (hi - lo + 1e-9), 0, 1))
                st.progress(pct_pos)
                st.markdown(
                    f'<div style="display:flex;justify-content:space-between;color:#64748b;font-size:0.78rem;margin-top:4px">'
                    f'<span>Low&nbsp;<b style="color:#e2e8f0">${lo:.2f}</b></span>'
                    f'<span>Now&nbsp;<b style="color:#a5b4fc">${price:.2f}</b></span>'
                    f'<span>High&nbsp;<b style="color:#e2e8f0">${hi:.2f}</b></span></div>',
                    unsafe_allow_html=True,
                )
            with cr:
                st.markdown("#### Volume — Last 30 Days")
                v30 = hist.tail(30)
                vfig = px.bar(v30, x=v30.index, y="Volume", color_discrete_sequence=["#6366f1"])
                vfig.update_traces(opacity=0.85, marker_line_width=0)
                vfig.update_layout(showlegend=False)
                st.plotly_chart(_chart_dark(vfig, 220), use_container_width=True)

            live_score, breakdown = score_live(live)
            score_pct = live_score / 100
            score_col = "#22c55e" if score_pct > 0.65 else ("#f59e0b" if score_pct > 0.4 else "#ef4444")
            st.markdown(
                f'<div style="display:flex;align-items:center;gap:12px;margin:1rem 0 0.4rem">'
                f'<span style="color:#cbd5e1;font-size:1rem;font-weight:700">Live Score</span>'
                f'<span style="color:{score_col};font-size:1.7rem;font-weight:900;letter-spacing:-0.03em">'
                f'{live_score:.1f}<span style="font-size:1rem;opacity:0.6">/100</span></span></div>',
                unsafe_allow_html=True,
            )
            sfig = px.bar(x=list(breakdown.keys()), y=list(breakdown.values()),
                          color=list(breakdown.values()), color_continuous_scale="RdYlGn",
                          range_color=[0, 100])
            sfig.update_traces(marker_line_width=0, opacity=0.9)
            sfig.update_layout(yaxis_range=[0, 100])
            st.plotly_chart(_chart_dark(sfig, 220), use_container_width=True)

    else:
        st.markdown(
            '<div style="background:rgba(99,102,241,0.07);border:1px solid rgba(99,102,241,0.2);'
            'border-radius:11px;padding:12px 16px;font-size:0.82rem;color:#94a3b8;margin-bottom:1rem">'
            '⚡ Click <b style="color:#a5b4fc">Fetch Live Analysis</b> above to load the yfinance chart, '
            'RSI, analyst targets, and live score.</div>',
            unsafe_allow_html=True,
        )
        st.markdown(f"#### CSV Score: {csv_score:.1f} / 100")
        csv_factors = {
            "Momentum": float(scored["momentum_score"].iloc[0]),
            "Volume":   float(scored["volume_score"].iloc[0]),
            "Market Cap": float(scored["mcap_score"].iloc[0]),
            "Price Range": float(scored["price_score"].iloc[0]),
            "Direction": float(scored["direction_score"].iloc[0]),
        }
        sfig = px.bar(x=list(csv_factors.keys()), y=list(csv_factors.values()),
                      color=list(csv_factors.values()), color_continuous_scale="RdYlGn",
                      range_color=[0, 100])
        sfig.update_traces(marker_line_width=0, opacity=0.9)
        sfig.update_layout(yaxis_range=[0, 100])
        st.plotly_chart(_chart_dark(sfig, 220), use_container_width=True)

    st.markdown("---")
    st.markdown(f"#### Top Peers — {row['Sector']}")
    peers = df[(df["Sector"] == row["Sector"]) & (df["Symbol"] != selected)]
    if not peers.empty:
        peer_sc = compute_scores(peers).nlargest(10, "score")
        st.dataframe(
            peer_sc[["Symbol","Name","Last Sale","% Change","Volume","score"]]
            .rename(columns={"Last Sale":"Price","score":"Score"})
            .style.format({"Price":"${:.2f}","% Change":"{:+.2f}%","Volume":"{:,.0f}","Score":"{:.1f}"}),
            use_container_width=True, height=300,
        )


def page_ai_agent(df: pd.DataFrame):
    st.title("AI Agent")
    st.caption("GPT-powered · live NASDAQ data · tool-calling · streaming")

    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "pending_q" not in st.session_state:
        st.session_state.pending_q = None

    quick_qs = [
        "What's the market looking like today?",
        "Show me the top picks",
        "Who are the biggest gainers?",
        "Which stocks are falling the most?",
        "Most active stocks right now",
        "How is the tech sector doing?",
        "Tell me about NVDA",
        "Compare AAPL vs MSFT vs GOOGL",
        "Explain RSI",
        "What is a short squeeze?",
        "How does momentum investing work?",
        "What is market cap?",
    ]

    with st.sidebar:
        st.markdown(
            '<div style="color:#6366f1;font-size:0.7rem;font-weight:700;text-transform:uppercase;'
            'letter-spacing:0.1em;margin-bottom:8px">💡 Quick prompts</div>',
            unsafe_allow_html=True,
        )
        for q in quick_qs:
            if st.button(q, use_container_width=True, key=f"q_{q[:22]}"):
                st.session_state.pending_q = q
                st.rerun()

        st.markdown("---")
        if st.button("🗑  Clear conversation", use_container_width=True):
            st.session_state.messages = []
            st.rerun()
        st.markdown("---")
        st.markdown(
            '<div style="color:#3d4f68;font-size:0.72rem;line-height:1.7">'
            '⚡ <b style="color:#64748b">Tip:</b> Ask naturally<br>'
            '&nbsp;&nbsp;• <i>How is energy doing?</i><br>'
            '&nbsp;&nbsp;• <i>Tell me about TSLA</i><br>'
            '&nbsp;&nbsp;• <i>AAPL vs MSFT vs AMZN</i><br>'
            '&nbsp;&nbsp;• <i>Explain MACD</i><br><br>'
            'For live RSI & charts →<br><b style="color:#6366f1">Stock Detail</b></div>',
            unsafe_allow_html=True,
        )

    # ── Welcome screen (empty state) ──────────────────────────────────────────
    if not st.session_state.messages:
        st.markdown("""
        <div style="text-align:center;padding:2.5rem 1rem 1.5rem">
          <div style="font-size:2.8rem;margin-bottom:0.6rem">⚡</div>
          <div style="color:#e2e8f0;font-size:1.4rem;font-weight:800;letter-spacing:-0.025em;margin-bottom:0.4rem">
            Ask me anything about the market
          </div>
          <div style="color:#475569;font-size:0.85rem;max-width:440px;margin:0 auto">
            I have live access to 7,000+ NASDAQ stocks. I can look up tickers, compare stocks,
            analyze sectors, explain concepts, and more.
          </div>
        </div>
        """, unsafe_allow_html=True)

        # Feature cards
        features = [
            ("📈", "Market & Sectors", "How is tech doing? What's the market breadth today?"),
            ("🔍", "Stock Lookup", "Tell me about NVDA · Analyze TSLA · What is AAPL's score?"),
            ("⚖️", "Compare Stocks", "AAPL vs MSFT vs GOOGL · Who has the best score?"),
            ("🎓", "Learn Concepts", "Explain RSI · What is a short squeeze · How does MACD work?"),
        ]
        cols = st.columns(2)
        for i, (icon, title, desc) in enumerate(features):
            with cols[i % 2]:
                st.markdown(f"""
                <div style="
                    background:linear-gradient(135deg,#0c1525,#0f1d38);
                    border:1px solid rgba(99,102,241,0.2); border-radius:14px;
                    padding:16px 18px; margin-bottom:10px;
                    box-shadow:0 3px 16px rgba(0,0,0,0.35);
                ">
                  <div style="font-size:1.3rem;margin-bottom:6px">{icon}</div>
                  <div style="color:#a5b4fc;font-size:0.85rem;font-weight:700;margin-bottom:4px">{title}</div>
                  <div style="color:#475569;font-size:0.75rem;line-height:1.5">{desc}</div>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)

    # ── Chat history ──────────────────────────────────────────────────────────
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # ── Input ─────────────────────────────────────────────────────────────────
    prompt = st.session_state.pending_q
    if prompt:
        st.session_state.pending_q = None
    else:
        prompt = st.chat_input("Ask about any stock, sector, or market concept…")

    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        history = st.session_state.messages[:-1]
        with st.chat_message("assistant"):
            response_text = st.write_stream(agent_respond_stream(prompt, df, history))

        st.session_state.messages.append({"role": "assistant", "content": response_text})


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    _inject_css()
    df = load_data()

    gainers = int((df["% Change"] > 0).sum())
    losers  = int((df["% Change"] < 0).sum())
    avg_chg = df["% Change"].mean()
    sent_color = "#22c55e" if avg_chg > 0.5 else ("#ef4444" if avg_chg < -0.5 else "#f59e0b")

    # ── Sidebar branding ───────────────────────────────────────────────────────
    st.sidebar.markdown(f"""
    <div style="padding:14px 6px 18px">
      <div style="display:flex;align-items:center;gap:10px;margin-bottom:4px">
        <span style="font-size:1.5rem">⚡</span>
        <div>
          <div style="color:#e2e8f0;font-size:1.05rem;font-weight:800;letter-spacing:-0.02em;line-height:1.2">
            NASDAQ Terminal
          </div>
          <div style="color:#3d4f68;font-size:0.66rem;font-weight:600;letter-spacing:0.1em;text-transform:uppercase">
            AI-Powered Screener
          </div>
        </div>
      </div>
      <div style="
          margin-top:12px;
          background:rgba(8,16,31,0.7);
          border:1px solid rgba(99,102,241,0.18);
          border-radius:10px; padding:10px 12px;
      ">
        <div style="display:flex;justify-content:space-between;margin-bottom:5px">
          <span style="color:#3d4f68;font-size:0.67rem;text-transform:uppercase;letter-spacing:0.1em;font-weight:700">Universe</span>
          <span style="color:#94a3b8;font-size:0.78rem;font-weight:600">{len(df):,} stocks</span>
        </div>
        <div style="display:flex;justify-content:space-between;margin-bottom:5px">
          <span style="color:#22c55e;font-size:0.67rem;text-transform:uppercase;letter-spacing:0.08em;font-weight:700">▲ Gainers</span>
          <span style="color:#22c55e;font-size:0.78rem;font-weight:600">{gainers:,}</span>
        </div>
        <div style="display:flex;justify-content:space-between;margin-bottom:5px">
          <span style="color:#ef4444;font-size:0.67rem;text-transform:uppercase;letter-spacing:0.08em;font-weight:700">▼ Losers</span>
          <span style="color:#ef4444;font-size:0.78rem;font-weight:600">{losers:,}</span>
        </div>
        <div style="display:flex;justify-content:space-between">
          <span style="color:#3d4f68;font-size:0.67rem;text-transform:uppercase;letter-spacing:0.08em;font-weight:700">Avg Chg</span>
          <span style="color:{sent_color};font-size:0.78rem;font-weight:700">{avg_chg:+.2f}%</span>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    nav_labels = ["🌐  Market Overview","📊  Screener","🚀  Top Picks","🔍  Stock Detail","💬  AI Agent"]
    page = st.sidebar.radio("nav", nav_labels, label_visibility="collapsed")

    st.sidebar.markdown(
        '<div style="color:#1e293b;font-size:0.65rem;text-align:center;margin-top:18px;'
        'padding:8px;border-top:1px solid rgba(99,102,241,0.1)">'
        'NASDAQ data · live via yfinance</div>',
        unsafe_allow_html=True,
    )

    pages = {
        "🌐  Market Overview": page_market_overview,
        "📊  Screener":        page_screener,
        "🚀  Top Picks":       page_top_picks,
        "🔍  Stock Detail":    page_stock_detail,
        "💬  AI Agent":        page_ai_agent,
    }
    pages[page](df)


if __name__ == "__main__":
    main()
