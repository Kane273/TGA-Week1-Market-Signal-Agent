import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
from openai import OpenAI
import os
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px
from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings

warnings.filterwarnings("ignore")

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Stock Analyzer & AI Agent",
    page_icon="📈",
    layout="wide",
)

# ── OpenAI client ─────────────────────────────────────────────────────────────
_openai_client = None

def get_openai_client():
    global _openai_client
    if _openai_client is None:
        _openai_client = OpenAI(
            base_url=os.environ.get("AI_INTEGRATIONS_OPENAI_BASE_URL"),
            api_key=os.environ.get("AI_INTEGRATIONS_OPENAI_API_KEY", "dummy"),
        )
    return _openai_client


# ── Data loading ──────────────────────────────────────────────────────────────
CSV_PATH = os.path.join(os.path.dirname(__file__), "nasdaq_screener.csv")


@st.cache_data(ttl=3600)
def load_nasdaq_data() -> pd.DataFrame:
    df = pd.read_csv(CSV_PATH)
    df["Last Sale"] = (
        df["Last Sale"].astype(str).str.replace("$", "", regex=False).str.replace(",", "")
    )
    df["Last Sale"] = pd.to_numeric(df["Last Sale"], errors="coerce")
    df["% Change"] = (
        df["% Change"].astype(str).str.replace("%", "", regex=False)
    )
    df["% Change"] = pd.to_numeric(df["% Change"], errors="coerce").fillna(0)
    df["Market Cap"] = pd.to_numeric(df["Market Cap"], errors="coerce").fillna(0)
    df["Volume"] = pd.to_numeric(df["Volume"], errors="coerce").fillna(0)
    df["Net Change"] = pd.to_numeric(df["Net Change"], errors="coerce").fillna(0)
    df = df[df["Last Sale"].notna() & (df["Last Sale"] > 0)]
    df = df[df["Symbol"].notna()]
    return df.reset_index(drop=True)


# ── Stock scoring (CSV only) ──────────────────────────────────────────────────

def score_stocks_csv(
    df: pd.DataFrame,
    min_price: float = 1.0,
    min_market_cap: float = 50_000_000,
    min_volume: float = 100_000,
    top_n: int = 100,
) -> pd.DataFrame:
    """Score stocks from CSV data for high-gain potential."""
    filtered = df[
        (df["Last Sale"] >= min_price)
        & (df["Market Cap"] >= min_market_cap)
        & (df["Volume"] >= min_volume)
        & df["Sector"].notna()
        & (df["Sector"].str.strip() != "")
    ].copy()

    if filtered.empty:
        return filtered

    # Momentum: % change
    p5, p95 = filtered["% Change"].quantile(0.05), filtered["% Change"].quantile(0.95)
    filtered["momentum_score"] = (
        ((filtered["% Change"] - p5) / (p95 - p5 + 1e-9)).clip(0, 1) * 100
    )

    # Volume (log-normalized)
    filtered["log_vol"] = np.log1p(filtered["Volume"])
    v10, v90 = filtered["log_vol"].quantile(0.1), filtered["log_vol"].quantile(0.9)
    filtered["volume_score"] = (
        ((filtered["log_vol"] - v10) / (v90 - v10 + 1e-9)).clip(0, 1) * 100
    )

    # Market-cap sweet spot: mid-cap ($100M–$2B) → log10 ~ 8–9.3
    filtered["log_mcap"] = np.log10(filtered["Market Cap"].clip(1))
    filtered["mcap_score"] = (100 - np.abs(filtered["log_mcap"] - 8.7) * 15).clip(0, 100)

    filtered["composite_score"] = (
        filtered["momentum_score"] * 0.40
        + filtered["volume_score"] * 0.30
        + filtered["mcap_score"] * 0.30
    )

    return filtered.nlargest(top_n, "composite_score").reset_index(drop=True)


# ── Live data fetch ───────────────────────────────────────────────────────────

def _calculate_rsi(prices: pd.Series, period: int = 14) -> float:
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
        price = info.get("currentPrice") or info.get("regularMarketPrice") or float(hist["Close"].iloc[-1])
        if not price:
            return None

        rsi = _calculate_rsi(hist["Close"])
        ma20 = float(hist["Close"].rolling(20).mean().iloc[-1]) if len(hist) >= 20 else None
        ma50 = float(hist["Close"].rolling(50).mean().iloc[-1]) if len(hist) >= 50 else None

        w52h = info.get("fiftyTwoWeekHigh", price)
        w52l = info.get("fiftyTwoWeekLow", price)
        pct_from_low = ((price - w52l) / (w52l + 1e-9)) * 100 if w52l else 0
        pct_from_high = ((w52h - price) / (w52h + 1e-9)) * 100 if w52h else 0

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
            "pct_from_52w_low": pct_from_low,
            "pct_from_52w_high": pct_from_high,
            "analyst_target": target,
            "analyst_upside": upside,
            "pe_ratio": info.get("forwardPE") or info.get("trailingPE"),
            "short_float": info.get("shortPercentOfFloat", 0),
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


def score_live_stock(live: dict) -> tuple[float, dict]:
    """Score a stock using live data."""
    rsi = live.get("rsi", 50)
    if rsi < 35:
        rsi_s = 90
    elif rsi > 75:
        rsi_s = 15
    elif 40 <= rsi <= 60:
        rsi_s = 70
    else:
        rsi_s = 50

    price = live.get("current_price", 0)
    ma20, ma50 = live.get("ma20"), live.get("ma50")
    if ma20 and ma50 and price:
        ma_s = 85 if price > ma20 > ma50 else (65 if price > ma20 else 35)
    else:
        ma_s = 50

    upside = live.get("analyst_upside")
    if upside is None:
        analyst_s = 50
    elif upside > 30:
        analyst_s = 95
    elif upside > 15:
        analyst_s = 75
    elif upside > 0:
        analyst_s = 55
    else:
        analyst_s = 20

    pct_low = live.get("pct_from_52w_low", 50)
    pos_s = 90 if pct_low < 15 else (70 if pct_low < 30 else (50 if pct_low < 60 else 30))

    composite = rsi_s * 0.25 + ma_s * 0.25 + analyst_s * 0.30 + pos_s * 0.20
    return composite, {"RSI Score": rsi_s, "MA Score": ma_s, "Analyst Score": analyst_s, "Position Score": pos_s}


# ═══════════════════════════════════════════════════════════════════════════════
# PAGES
# ═══════════════════════════════════════════════════════════════════════════════

def page_screener():
    st.title("📊 NASDAQ Stock Screener")
    st.caption("Browse and filter all stocks from the NASDAQ universe")

    df = load_nasdaq_data()

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        min_price = st.number_input("Min Price ($)", min_value=0.0, value=1.0, step=0.5)
    with col2:
        cap_opts = {
            "Any": 0,
            "$50M+": 50_000_000,
            "$300M+": 300_000_000,
            "$1B+": 1_000_000_000,
            "$10B+": 10_000_000_000,
        }
        min_cap_label = st.selectbox("Min Market Cap", list(cap_opts.keys()), index=1)
        min_cap = cap_opts[min_cap_label]
    with col3:
        vol_opts = {"Any": 0, "10K+": 10_000, "100K+": 100_000, "500K+": 500_000, "1M+": 1_000_000}
        min_vol_label = st.selectbox("Min Volume", list(vol_opts.keys()), index=2)
        min_vol = vol_opts[min_vol_label]
    with col4:
        sectors = ["All"] + sorted(df["Sector"].dropna().unique().tolist())
        sector = st.selectbox("Sector", sectors)

    sort_by = st.selectbox("Sort by", ["% Change ↓", "Volume ↓", "Market Cap ↓", "Last Sale ↓"])
    sort_map = {
        "% Change ↓": "% Change",
        "Volume ↓": "Volume",
        "Market Cap ↓": "Market Cap",
        "Last Sale ↓": "Last Sale",
    }

    filtered = df[
        (df["Last Sale"] >= min_price)
        & (df["Market Cap"] >= min_cap)
        & (df["Volume"] >= min_vol)
    ]
    if sector != "All":
        filtered = filtered[filtered["Sector"] == sector]
    filtered = filtered.sort_values(sort_map[sort_by], ascending=False)

    st.metric("Matching Stocks", f"{len(filtered):,}", f"of {len(df):,} total")

    cols = ["Symbol", "Name", "Last Sale", "Net Change", "% Change", "Volume", "Market Cap", "Sector", "Country"]
    st.dataframe(
        filtered[cols].head(500).style.format(
            {
                "Last Sale": "${:.2f}",
                "Net Change": "{:+.2f}",
                "% Change": "{:+.2f}%",
                "Volume": "{:,.0f}",
                "Market Cap": "${:,.0f}",
            }
        ),
        use_container_width=True,
        height=540,
    )


def page_top_picks():
    st.title("🚀 Top High-Gain Picks")
    st.caption("Stocks ranked for high-gain potential using momentum, volume, and live market signals")

    df = load_nasdaq_data()

    with st.sidebar:
        st.subheader("🔧 Screening Filters")
        min_price = st.slider("Min Price ($)", 1.0, 50.0, 2.0, 0.5)
        cap_opts = {"$50M+": 50_000_000, "$100M+": 100_000_000, "$300M+": 300_000_000, "$1B+": 1_000_000_000}
        min_cap = st.selectbox("Min Market Cap", list(cap_opts.keys()), index=1)
        vol_opts = {"50K+": 50_000, "100K+": 100_000, "500K+": 500_000, "1M+": 1_000_000}
        min_vol = st.selectbox("Min Daily Volume", list(vol_opts.keys()), index=1)
        top_n = st.slider("Candidates to evaluate", 25, 150, 60)
        fetch_live = st.checkbox("⚡ Fetch Live Data", value=False,
                                  help="Fetches RSI, analyst targets, and moving averages. ~30 sec for top 30 stocks.")

    with st.spinner("Scoring stocks from NASDAQ data..."):
        candidates = score_stocks_csv(
            df,
            min_price=min_price,
            min_market_cap=cap_opts[min_cap],
            min_volume=vol_opts[min_vol],
            top_n=top_n,
        )

    if candidates.empty:
        st.warning("No stocks match the current filters. Try relaxing the criteria.")
        return

    if fetch_live:
        symbols = tuple(candidates["Symbol"].tolist()[:30])
        with st.spinner(f"Fetching live data for top {len(symbols)} candidates… this takes ~30 seconds"):
            live_list = fetch_live_data(symbols)

        live_dict = {r["symbol"]: r for r in live_list}
        rows = []
        for _, row in candidates.iterrows():
            sym = row["Symbol"]
            if sym not in live_dict:
                continue
            live = live_dict[sym]
            live_score, _ = score_live_stock(live)
            rows.append(
                {
                    "Symbol": sym,
                    "Name": row["Name"][:35],
                    "Price": live["current_price"],
                    "RSI": live["rsi"],
                    "vs 52w Low": live["pct_from_52w_low"],
                    "Analyst Upside %": live["analyst_upside"],
                    "Beta": live["beta"],
                    "Sector": row["Sector"],
                    "Live Score": live_score,
                }
            )

        if not rows:
            st.error("Could not retrieve live data. yfinance may be rate-limited. Try again in a moment.")
            return

        result_df = pd.DataFrame(rows).sort_values("Live Score", ascending=False).reset_index(drop=True)

        st.subheader("⭐ Top 3 Live-Scored Picks")
        top3 = st.columns(3)
        for col, (_, r) in zip(top3, result_df.head(3).iterrows()):
            with col:
                upside_str = f"Analyst +{r['Analyst Upside %']:.1f}%" if r["Analyst Upside %"] else "No target"
                st.metric(r["Symbol"], f"${r['Price']:.2f}", f"Score: {r['Live Score']:.0f}/100")
                st.caption(r["Name"])
                st.write(f"RSI: {r['RSI']:.1f} | {upside_str}")

        st.subheader("All Live-Scored Picks")
        st.dataframe(
            result_df.style.format(
                {
                    "Price": "${:.2f}",
                    "RSI": "{:.1f}",
                    "vs 52w Low": "{:+.1f}%",
                    "Analyst Upside %": lambda x: f"+{x:.1f}%" if x else "N/A",
                    "Beta": lambda x: f"{x:.2f}" if x else "N/A",
                    "Live Score": "{:.1f}",
                }
            ).background_gradient(subset=["Live Score"], cmap="RdYlGn"),
            use_container_width=True,
            height=500,
        )

    else:
        st.info("💡 Enable **Fetch Live Data** in the sidebar for RSI, MA signals, and analyst targets.")

        top5 = st.columns(5)
        for col, (_, r) in zip(top5, candidates.head(5).iterrows()):
            with col:
                st.metric(r["Symbol"], f"${r['Last Sale']:.2f}", f"{r['% Change']:+.2f}%")
                st.caption(f"Score: {r['composite_score']:.1f}/100")

        st.subheader("All Top Picks")
        disp = candidates[
            ["Symbol", "Name", "Last Sale", "% Change", "Volume", "Market Cap", "Sector", "composite_score"]
        ].rename(columns={"Last Sale": "Price", "composite_score": "Score"})

        st.dataframe(
            disp.style.format(
                {
                    "Price": "${:.2f}",
                    "% Change": "{:+.2f}%",
                    "Volume": "{:,.0f}",
                    "Market Cap": "${:,.0f}",
                    "Score": "{:.1f}",
                }
            ).background_gradient(subset=["Score"], cmap="RdYlGn"),
            use_container_width=True,
            height=500,
        )

        st.subheader("Sector Distribution")
        sec_counts = disp.groupby("Sector").size().reset_index(name="Count")
        fig = px.bar(
            sec_counts.sort_values("Count"),
            x="Count", y="Sector", orientation="h",
            color="Count", color_continuous_scale="viridis",
        )
        fig.update_layout(height=400, coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)


def page_stock_detail():
    st.title("🔍 Stock Deep Dive")
    st.caption("Fetch live chart, technicals, and analyst data for any stock")

    df = load_nasdaq_data()
    symbols = sorted(df["Symbol"].dropna().unique().tolist())
    default_idx = symbols.index("AAPL") if "AAPL" in symbols else 0
    selected = st.selectbox("Choose a stock", symbols, index=default_idx)

    if st.button("Fetch Live Analysis", type="primary"):
        with st.spinner(f"Fetching live data for {selected}…"):
            live_list = fetch_live_data((selected,))

        if not live_list:
            st.error(f"Could not fetch data for **{selected}**. The symbol may be invalid or delisted.")
            return

        live = live_list[0]
        hist = live["hist"]
        price = live["current_price"]

        csv_row = df[df["Symbol"] == selected].iloc[0] if selected in df["Symbol"].values else None
        if csv_row is not None:
            st.subheader(f"{selected} — {csv_row['Name']}")
        else:
            st.subheader(selected)

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("Current Price", f"${price:.2f}")
        with c2:
            rsi = live["rsi"]
            rsi_label = "Oversold 🟢" if rsi < 35 else ("Overbought 🔴" if rsi > 70 else "Neutral ⚪")
            st.metric("RSI (14)", f"{rsi:.1f}", rsi_label)
        with c3:
            if live["analyst_upside"] is not None:
                st.metric(
                    "Analyst Upside",
                    f"{live['analyst_upside']:+.1f}%",
                    f"Target: ${live['analyst_target']:.2f}",
                )
            else:
                st.metric("Analyst Upside", "N/A")
        with c4:
            beta = live.get("beta")
            st.metric("Beta", f"{beta:.2f}" if beta else "N/A")

        # Chart
        st.subheader("Price Chart — 3 Months")
        fig = go.Figure()
        fig.add_trace(
            go.Candlestick(
                x=hist.index,
                open=hist["Open"],
                high=hist["High"],
                low=hist["Low"],
                close=hist["Close"],
                name="Price",
            )
        )
        if len(hist) >= 20:
            fig.add_trace(
                go.Scatter(
                    x=hist.index,
                    y=hist["Close"].rolling(20).mean(),
                    name="MA20",
                    line=dict(color="#f59e0b", width=1.5),
                )
            )
        if len(hist) >= 50:
            fig.add_trace(
                go.Scatter(
                    x=hist.index,
                    y=hist["Close"].rolling(50).mean(),
                    name="MA50",
                    line=dict(color="#3b82f6", width=1.5),
                )
            )
        fig.update_layout(xaxis_rangeslider_visible=False, height=420)
        st.plotly_chart(fig, use_container_width=True)

        col_l, col_r = st.columns(2)
        with col_l:
            st.subheader("52-Week Range")
            lo, hi = live["week52_low"], live["week52_high"]
            pct = float(np.clip((price - lo) / (hi - lo + 1e-9), 0, 1))
            st.progress(pct)
            st.caption(f"Low: ${lo:.2f}  ·  Now: ${price:.2f}  ·  High: ${hi:.2f}")

        with col_r:
            st.subheader("Daily Volume (30d)")
            vol_fig = px.bar(hist.tail(30), x=hist.tail(30).index, y="Volume", color_discrete_sequence=["#6366f1"])
            vol_fig.update_layout(height=220, showlegend=False, margin=dict(t=10, b=10))
            st.plotly_chart(vol_fig, use_container_width=True)

        # Score breakdown
        live_score, breakdown = score_live_stock(live)
        st.subheader(f"High-Gain Score: {live_score:.1f} / 100")
        score_fig = px.bar(
            x=list(breakdown.keys()),
            y=list(breakdown.values()),
            color=list(breakdown.values()),
            color_continuous_scale="RdYlGn",
            range_color=[0, 100],
            labels={"x": "Factor", "y": "Score"},
        )
        score_fig.update_layout(height=250, coloraxis_showscale=False)
        st.plotly_chart(score_fig, use_container_width=True)


def page_ai_agent():
    st.title("💬 AI Stock Analysis Agent")
    st.caption(
        "Ask about stock picks, market signals, technical indicators, or load a specific stock for targeted analysis."
    )

    df = load_nasdaq_data()

    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "agent_context" not in st.session_state:
        st.session_state.agent_context = ""
    if "pending_q" not in st.session_state:
        st.session_state.pending_q = None

    with st.sidebar:
        st.subheader("📌 Load Stock Context")
        symbols = ["— None —"] + sorted(df["Symbol"].dropna().unique().tolist())
        ctx_sym = st.selectbox("Stock to analyze", symbols)
        if ctx_sym != "— None —" and st.button("Load into Agent", use_container_width=True):
            row = df[df["Symbol"] == ctx_sym].iloc[0]
            st.session_state.agent_context = (
                f"Symbol: {ctx_sym}\n"
                f"Name: {row['Name']}\n"
                f"Last Price: ${row['Last Sale']:.2f}\n"
                f"Daily Change: {row['% Change']:+.2f}%  ({row['Net Change']:+.2f})\n"
                f"Volume: {row['Volume']:,.0f}\n"
                f"Market Cap: ${row['Market Cap']:,.0f}\n"
                f"Sector: {row['Sector']}\n"
                f"Industry: {row.get('Industry', 'N/A')}\n"
                f"Country: {row['Country']}\n"
                f"IPO Year: {row.get('IPO Year', 'N/A')}\n"
            )
            st.success(f"✅ Loaded {ctx_sym}")

        if st.session_state.agent_context:
            st.info(st.session_state.agent_context)
            if st.button("Clear Context", use_container_width=True):
                st.session_state.agent_context = ""
                st.rerun()

        st.markdown("---")
        st.subheader("💡 Quick Questions")
        quick_qs = [
            "What indicators suggest a stock is about to make a big move?",
            "Explain RSI and how to use it for stock picks",
            "What sectors tend to outperform in a bull market?",
            "How do I evaluate a stock for short-term gains?",
            "What is a short squeeze and what triggers it?",
            "What is the difference between momentum and value investing?",
            "How do earnings reports affect stock price?",
        ]
        for q in quick_qs:
            if st.button(q, use_container_width=True, key=f"quick_{q[:20]}"):
                st.session_state.pending_q = q
                st.rerun()

        st.markdown("---")
        if st.button("🗑 Clear Conversation", use_container_width=True):
            st.session_state.messages = []
            st.session_state.agent_context = ""
            st.rerun()

    # Display history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Input: quick question or chat box
    prompt = st.session_state.pending_q
    if prompt:
        st.session_state.pending_q = None
    else:
        prompt = st.chat_input("Ask about any stock, strategy, or market concept…")

    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        system_msg = (
            "You are an expert stock market analyst and trading strategist. "
            "You have deep expertise in:\n"
            "• Technical analysis: RSI, MACD, Bollinger Bands, candlestick patterns, moving averages\n"
            "• Fundamental analysis: P/E, P/S, EV/EBITDA, earnings growth, sector dynamics\n"
            "• Market sentiment: short interest, options flow, institutional positioning\n"
            "• Momentum and swing trading strategies\n"
            "• Risk management and position sizing\n\n"
            "When analyzing stocks, consider multiple timeframes and both technical and fundamental factors. "
            "Be specific and actionable. Use bullet points for lists. "
            "Always note key risks alongside opportunities. "
            "Remind users that this is educational analysis, not personalized financial advice.\n"
            "Do not use emojis in your responses."
        )

        if st.session_state.agent_context:
            system_msg += f"\n\nThe user has loaded the following stock data for analysis:\n{st.session_state.agent_context}"

        api_msgs = [{"role": "system", "content": system_msg}]
        for m in st.session_state.messages[-12:]:
            api_msgs.append({"role": m["role"], "content": m["content"]})

        with st.chat_message("assistant"):
            placeholder = st.empty()
            full = ""
            try:
                stream = get_openai_client().chat.completions.create(
                    model="gpt-5.6-luna",
                    max_completion_tokens=1200,
                    messages=api_msgs,
                    stream=True,
                )
                for chunk in stream:
                    delta = chunk.choices[0].delta.content
                    if delta:
                        full += delta
                        placeholder.markdown(full + "▌")
                placeholder.markdown(full)
            except Exception as exc:
                full = f"Error connecting to AI: {exc}"
                placeholder.error(full)

        st.session_state.messages.append({"role": "assistant", "content": full})


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    st.sidebar.title("📈 Stock Analyzer")
    st.sidebar.markdown("---")

    page = st.sidebar.radio(
        "Navigate",
        ["📊 Screener", "🚀 Top Picks", "🔍 Stock Detail", "💬 AI Agent"],
    )

    st.sidebar.markdown("---")
    st.sidebar.caption(f"NASDAQ data · yfinance live · OpenAI agent")
    st.sidebar.caption(f"Refreshed: {datetime.now().strftime('%Y-%m-%d %H:%M')}")

    if page == "📊 Screener":
        page_screener()
    elif page == "🚀 Top Picks":
        page_top_picks()
    elif page == "🔍 Stock Detail":
        page_stock_detail()
    elif page == "💬 AI Agent":
        page_ai_agent()


if __name__ == "__main__":
    main()
