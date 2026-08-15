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
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings

warnings.filterwarnings("ignore")

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Stock Analyzer & AI Agent",
    page_icon="📈",
    layout="wide",
)

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

- **Below 30** — Oversold. Price dropped sharply; potential bounce incoming. Possible buy signal.
- **30–50** — Recovering. Selling pressure easing; watch for reversal.
- **50–70** — Healthy bullish range. Positive momentum without excess.
- **Above 70** — Overbought. Price ran up fast; pullback or consolidation likely.

*How traders use it:* Buy when RSI crosses back above 30 (oversold bounce). Tighten stops when RSI crosses below 70. Works best combined with trend context — don't buy an oversold stock in a strong downtrend.

The **Top Picks** page uses CSV data for initial screening. The **Stock Detail** page fetches live RSI via yfinance.""",

    "macd": """**MACD (Moving Average Convergence Divergence)**

Tracks momentum via two exponential moving averages (12-day and 26-day EMA), with a 9-day signal line on top.

Key signals:
- **Bullish crossover**: MACD line crosses above signal line — buy signal
- **Bearish crossover**: MACD line crosses below signal line — sell signal
- **Zero-line cross**: MACD crossing above zero confirms broad uptrend
- **Histogram**: Widening = strengthening trend; narrowing = weakening

Best used to confirm trend direction and spot momentum shifts. Lagging by nature — not ideal for pinpoint entries.""",

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

Volume = number of shares traded. It confirms or denies price moves.

- **High volume + rising price** — Strong bullish signal. Institutions are buying.
- **High volume + falling price** — Strong selling pressure. Exit or wait.
- **Low volume + rising price** — Weak rally. Lacks conviction.
- **Volume spike** — Signals a potential turning point.

For high-gain picks: look for stocks where volume is 2–3x the average alongside a price breakout. Volume is the fuel; price is the fire.""",

    "market cap": """**Market Capitalization**

Market cap = share price × shares outstanding.

- **Micro-cap** (< $300M): Extremely volatile. High risk, high reward.
- **Small-cap** ($300M–$2B): Growth sweet spot. Enough liquidity, still room to run.
- **Mid-cap** ($2B–$10B): Solid growth with lower volatility.
- **Large-cap** (> $10B): Stable, but slower growth potential.

The **Top Picks** scoring favors mid-cap stocks ($100M–$2B) as the sweet spot for large percentage gains.""",

    "momentum": """**Momentum Investing**

Stocks rising tend to continue rising. Key signals:

- High % change vs. peers and index
- Breaking to new 52-week highs
- Relative strength vs. its sector

Strategies:
- Buy top 20% of performers over 1–6 months
- Avoid stocks in clear downtrends even if "cheap"
- Always use stop-losses — momentum can reverse fast""",

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

A breakout occurs when price moves above a key resistance level with increased volume.

What to look for:
- Price consolidating sideways for weeks
- A resistance level tested multiple times
- Price closes above resistance on 2x+ average volume

Entry: Buy the breakout day or retest of the broken level.
Stop-loss: Just below the broken resistance (now support).

False breakouts are common — always wait for a daily close above the level.""",

    "earnings": """**Earnings and Stock Price**

Quarterly earnings reports are the biggest scheduled catalyst for individual stocks.

- **Beat + raise guidance** — Strong surge (5–20%+)
- **Miss or lower guidance** — Sharp drop (5–15%+)
- **In-line** — Muted reaction; focus on guidance language

"Buy the rumor, sell the news": Stocks often rise into earnings, then drop even on a beat if expectations were priced in.

Strategy: Look for stocks that beat estimates 3+ consecutive quarters — they tend to trend higher.""",

    "stop loss": """**Stop-Loss Orders**

An order to automatically sell if price falls to a set level, limiting your loss.

Common strategies:
- **Percentage stop**: Sell if stock drops 7–10% below buy price
- **Technical stop**: Sell if price closes below a key support or moving average
- **Trailing stop**: Stop moves up with the stock, locking in gains

Mental stops don't work. Set real orders. The hardest part of trading is cutting a small loss before it becomes a large one.""",

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

    momentum_txt = ("Strong positive momentum — significant gain today" if pct > 5 else
                    "Mild positive momentum" if pct > 0 else
                    "Negative momentum — down today" if pct > -5 else
                    "Heavy selling pressure — significant decline today")
    vol_txt = ("Very high liquidity" if vol > 1_000_000 else
               "Good liquidity" if vol > 200_000 else
               "Lower liquidity — use limit orders" if vol > 50_000 else
               "Thin liquidity — exercise caution with position size")
    cap_txt = ("Large-cap — stable but slower growth" if mcap > 10e9 else
               "Mid-cap — solid growth profile" if mcap > 2e9 else
               "Small-cap — higher growth potential, higher volatility" if mcap > 300e6 else
               "Micro-cap — maximum volatility, high risk/reward")
    score_txt = ("Strong candidate for high-gain potential" if score >= 70 else
                 "Above-average on high-gain metrics" if score >= 55 else
                 "Average — no standout signals" if score >= 40 else
                 "Below average — weak momentum and volume")

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
        f"  {i+1}. **{r.Symbol}** — {r.Name[:30]} | ${r['Last Sale']:.2f} | {r['% Change']:+.2f}%"
        for i, r in enumerate(top5.itertuples())
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
        f"  {i+1}. **{r.Symbol}** — {r.Name[:30]} | ${r['Last Sale']:.2f} | {r['% Change']:+.2f}% | Score: {r.score:.0f}/100"
        for i, r in enumerate(scored.itertuples())
    )
    return (f"**Top 10 High-Gain Candidates (CSV scoring)**\n\n{lines}\n\n"
            "Go to the **Top Picks** page and enable *Fetch Live Data* to add RSI, analyst targets, and MA signals.")


def _gainers_losers(df: pd.DataFrame, mode: str = "gainers", n: int = 10) -> str:
    f = df[(df["Market Cap"] >= 50e6) & (df["Volume"] >= 50_000)]
    top = f.nlargest(n, "% Change") if mode == "gainers" else f.nsmallest(n, "% Change")
    header = f"**Top {n} {'Gainers' if mode == 'gainers' else 'Losers'} Today**"
    lines = "\n".join(
        f"  {i+1}. **{r.Symbol}** | {r.Name[:28]} | ${r['Last Sale']:.2f} | {r['% Change']:+.2f}%"
        for i, r in enumerate(top.itertuples())
    )
    return f"{header}\n\n{lines}\n\n*(Filtered: $50M+ market cap, 50K+ volume)*"


def _most_active(df: pd.DataFrame, n: int = 10) -> str:
    top = df.nlargest(n, "Volume")
    lines = "\n".join(
        f"  {i+1}. **{r.Symbol}** | {r.Name[:28]} | ${r['Last Sale']:.2f} | Vol: {r['Volume']:,.0f} | {r['% Change']:+.2f}%"
        for i, r in enumerate(top.itertuples())
    )
    return f"**Top {n} Most Active (by Volume)**\n\n{lines}"


DISCLAIMER = "\n\n---\n*All information provided is for informational purposes only and does not constitute financial advice. Always do your own research and consult a licensed financial advisor before making investment decisions.*"


def _agent_respond_inner(user_msg: str, df: pd.DataFrame) -> str:
    msg = user_msg.strip()
    msg_lower = msg.lower()

    if re.match(r"^(hi|hello|hey|howdy|sup|what'?s up|yo)\b", msg_lower):
        return ("Hello! I'm your stock analysis agent. I can help you with:\n\n"
                "- **Analyze a ticker**: type `AAPL` or `analyze NVDA`\n"
                "- **Top picks**: `show top picks`\n"
                "- **Gainers / Losers**: `top gainers` or `biggest losers`\n"
                "- **Most active**: `most active stocks`\n"
                "- **Sector**: `technology sector` or `all sectors`\n"
                "- **Compare**: `AAPL vs MSFT`\n"
                "- **Education**: `explain RSI`, `MACD`, `momentum`, `short squeeze`, `P/E ratio`, `breakout`, etc.\n\n"
                "For live RSI, charts, and analyst targets, use the **Stock Detail** page.")

    # Analyze intent
    analyze_match = re.search(
        r"\b(?:analyze|analysis|tell me about|what about|look at|check|research|thoughts on|opinion on)\s+([A-Z]{1,5})\b",
        msg, re.IGNORECASE)
    if analyze_match:
        return _stock_analysis_csv(analyze_match.group(1).upper(), df)

    # Bare ticker
    bare = re.match(r"^\s*([A-Z]{1,5})\s*\??$", msg.upper())
    if bare and bare.group(1) not in {"RSI", "PE", "MA", "EMA", "SMA", "ETF", "IPO"}:
        return _stock_analysis_csv(bare.group(1), df)

    # Compare
    vs_match = re.search(r"\b([A-Z]{1,5})\s+vs\.?\s+([A-Z]{1,5})\b", msg.upper())
    if vs_match:
        a, b = vs_match.group(1), vs_match.group(2)
        results = []
        for sym in [a, b]:
            rows = df[df["Symbol"].str.upper() == sym]
            if not rows.empty:
                sc = compute_scores(rows).iloc[0]
                results.append(sc)
        if not results:
            return "I couldn't find either symbol in the dataset."
        lines = []
        for r in results:
            lines.append(f"**{r['Symbol']}** — {r['Name'][:28]}\n"
                         f"  ${r['Last Sale']:.2f} | {r['% Change']:+.2f}% | "
                         f"Vol: {r['Volume']:,.0f} | {_fmt_mcap(r['Market Cap'])} | Score: {r['score']:.0f}/100")
        winner = max(results, key=lambda r: r["score"])
        return "**Comparison**\n\n" + "\n\n".join(lines) + f"\n\n**Best by score**: {winner['Symbol']} ({winner['score']:.0f}/100)"

    if re.search(r"\bcompare\b", msg_lower):
        tickers = [t for t in re.findall(r"\b([A-Z]{1,5})\b", msg.upper())
                   if t not in {"VS", "AND", "OR", "TO", "THE", "RSI", "PE", "MA", "COMPARE"}]
        if len(tickers) >= 2:
            results = []
            for sym in tickers[:5]:
                rows = df[df["Symbol"].str.upper() == sym]
                if not rows.empty:
                    results.append(compute_scores(rows).iloc[0])
            if results:
                lines = [f"**{r['Symbol']}** — ${r['Last Sale']:.2f} | {r['% Change']:+.2f}% | Score: {r['score']:.0f}/100"
                         for r in results]
                winner = max(results, key=lambda r: r["score"])
                return "**Comparison**\n\n" + "\n".join(lines) + f"\n\n**Best by score**: {winner['Symbol']}"

    if re.search(r"\b(top pick|best stock|high gain|high.potential|recommend|strong buy|what.*(should|to) buy)\b", msg_lower):
        return _top_picks_summary(df)

    if re.search(r"\b(top gainer|biggest gainer|most gained|up the most|rising|best performer)\b", msg_lower):
        return _gainers_losers(df, "gainers")

    if re.search(r"\b(top loser|biggest loser|most lost|down the most|falling|worst)\b", msg_lower):
        return _gainers_losers(df, "losers")

    if re.search(r"\b(most active|highest volume|most traded|most liquid)\b", msg_lower):
        return _most_active(df)

    for sec in [s for s in df["Sector"].unique() if s and s != "Unknown"]:
        if sec.lower() in msg_lower:
            return _sector_analysis(sec, df)

    if re.search(r"\bsector\b", msg_lower):
        perf = (df[df["Sector"] != "Unknown"].groupby("Sector")["% Change"]
                .mean().sort_values(ascending=False))
        lines = "\n".join(f"  {i+1}. **{s}**: {p:+.2f}%" for i, (s, p) in enumerate(perf.items()))
        return f"**All Sectors (Avg % Change)**\n\n{lines}"

    if re.search(r"\b(how many|total|dataset|universe|stocks in)\b", msg_lower):
        total = len(df)
        g = (df["% Change"] > 0).sum()
        l = (df["% Change"] < 0).sum()
        return (f"**Dataset Overview**\n\n"
                f"- Total stocks: **{total:,}**\n"
                f"- Sectors: **{df['Sector'].nunique()}**\n"
                f"- Countries: **{df['Country'].nunique()}**\n"
                f"- Gainers today: **{g:,}** ({g/total*100:.1f}%)\n"
                f"- Losers today: **{l:,}** ({l/total*100:.1f}%)\n"
                f"- Unchanged: **{total-g-l:,}**")

    for key, content in KNOWLEDGE_BASE.items():
        if key in msg_lower:
            return content

    # Any ticker mentioned
    ignore = {"I","A","AN","THE","IS","IT","IN","OF","TO","DO","BE","GO","MY","WE","US","ON",
              "AT","BY","AS","UP","OR","AND","FOR","ARE","WAS","RSI","PE","MA","EMA","SMA",
              "ETF","IPO","CEO","CFO","CTO","EPS","ROI","AI"}
    for t in re.findall(r"\b([A-Z]{1,5})\b", msg.upper()):
        if t not in ignore and not df[df["Symbol"].str.upper() == t].empty:
            return _stock_analysis_csv(t, df)

    return ("I can help with:\n\n"
            "- **Ticker**: type `AAPL` or `analyze MSFT`\n"
            "- **Top picks**: `top picks today`\n"
            "- **Gainers/Losers**: `top gainers` or `biggest losers`\n"
            "- **Most active**: `most active stocks`\n"
            "- **Sector**: `technology sector` or `all sectors`\n"
            "- **Compare**: `AAPL vs MSFT`\n"
            "- **Education**: `RSI`, `MACD`, `moving average`, `momentum`, `short squeeze`, "
            "`market cap`, `PE ratio`, `breakout`, `earnings`, `stop loss`, `diversification`, "
            "`sector rotation`, `yfinance`"
            )


def agent_respond(user_msg: str, df: pd.DataFrame) -> str:
    return _agent_respond_inner(user_msg, df) + DISCLAIMER


# ═══════════════════════════════════════════════════════════════════════════════
# PAGES
# ═══════════════════════════════════════════════════════════════════════════════

def page_market_overview(df: pd.DataFrame):
    st.title("🌐 Market Overview")
    st.caption("NASDAQ market breadth and sector performance from screener data")

    total = len(df)
    gainers = int((df["% Change"] > 0).sum())
    losers = int((df["% Change"] < 0).sum())

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Stocks", f"{total:,}")
    c2.metric("Gainers 🟢", f"{gainers:,}", f"{gainers/total*100:.1f}%")
    c3.metric("Losers 🔴", f"{losers:,}", f"-{losers/total*100:.1f}%")
    c4.metric("Unchanged ⚪", f"{total-gainers-losers:,}")

    st.markdown("---")
    sec_df = df[df["Sector"] != "Unknown"]
    sec_breadth = (
        sec_df.groupby("Sector")
        .apply(lambda g: pd.Series({
            "Gainers": int((g["% Change"] > 0).sum()),
            "Losers": int((g["% Change"] < 0).sum()),
            "Avg % Change": g["% Change"].mean(),
        }))
        .reset_index()
        .sort_values("Avg % Change", ascending=False)
    )

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Sector Breadth")
        fig = px.bar(sec_breadth.melt(id_vars="Sector", value_vars=["Gainers", "Losers"]),
                     x="Sector", y="value", color="variable", barmode="group",
                     color_discrete_map={"Gainers": "#22c55e", "Losers": "#ef4444"})
        fig.update_layout(height=360, xaxis_tickangle=-40, legend_title="")
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        st.subheader("Average % Change by Sector")
        fig2 = px.bar(sec_breadth.sort_values("Avg % Change"),
                      x="Avg % Change", y="Sector", orientation="h",
                      color="Avg % Change", color_continuous_scale="RdYlGn")
        fig2.update_layout(height=360, coloraxis_showscale=False)
        st.plotly_chart(fig2, use_container_width=True)

    st.subheader("% Change Distribution")
    fig3 = px.histogram(df[(df["% Change"] > -20) & (df["% Change"] < 20)],
                        x="% Change", nbins=80, color_discrete_sequence=["#6366f1"])
    fig3.add_vline(x=0, line_dash="dash", line_color="white")
    fig3.update_layout(height=260)
    st.plotly_chart(fig3, use_container_width=True)

    c1, c2 = st.columns(2)
    qualified = df[(df["Market Cap"] >= 50e6) & (df["Volume"] >= 50_000)]
    with c1:
        st.subheader("Top 10 Gainers")
        top_g = qualified.nlargest(10, "% Change")
        st.dataframe(top_g[["Symbol","Name","Last Sale","% Change","Volume"]].style.format(
            {"Last Sale":"${:.2f}","% Change":"{:+.2f}%","Volume":"{:,.0f}"}),
            use_container_width=True, height=320)
    with c2:
        st.subheader("Top 10 Losers")
        top_l = qualified.nsmallest(10, "% Change")
        st.dataframe(top_l[["Symbol","Name","Last Sale","% Change","Volume"]].style.format(
            {"Last Sale":"${:.2f}","% Change":"{:+.2f}%","Volume":"{:,.0f}"}),
            use_container_width=True, height=320)


def page_screener(df: pd.DataFrame):
    st.title("📊 NASDAQ Stock Screener")
    st.caption(f"Browse all {len(df):,} stocks from the NASDAQ screener snapshot")

    with st.expander("🔧 Filters", expanded=True):
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
    st.metric("Matching Stocks", f"{len(filtered):,}", f"of {len(df):,} total")

    st.dataframe(
        filtered[["Symbol","Name","Last Sale","Net Change","% Change","Volume","Market Cap","Sector","Industry","Country"]]
        .head(500).style.format({"Last Sale":"${:.2f}","Net Change":"{:+.2f}",
                                  "% Change":"{:+.2f}%","Volume":"{:,.0f}","Market Cap":"${:,.0f}"}),
        use_container_width=True, height=540)


def page_top_picks(df: pd.DataFrame):
    st.title("🚀 Top High-Gain Picks")
    st.caption("Stocks ranked by composite score. Enable live data for RSI, analyst targets, and MA signals.")

    with st.sidebar:
        st.subheader("🔧 Filters")
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
        fetch_live = st.checkbox("⚡ Fetch Live Data (yfinance)", value=False,
                                  help="Adds RSI, MA signals, and analyst targets. ~30s for top 30.")

    filtered = filter_df(df, min_price, 1e9, min_cap, min_vol, sector, direction)
    if filtered.empty:
        st.warning("No stocks match the current filters.")
        return

    scored = compute_scores(filtered).nlargest(top_n, "score").reset_index(drop=True)

    if fetch_live:
        symbols = tuple(scored["Symbol"].tolist()[:30])
        with st.spinner(f"Fetching live data for top {len(symbols)} candidates via yfinance…"):
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
                "Symbol": sym,
                "Name": row["Name"][:35],
                "Price": live["current_price"],
                "RSI": live["rsi"],
                "vs 52w Low": live["pct_from_52w_low"],
                "Analyst Upside %": live["analyst_upside"],
                "Beta": live["beta"],
                "Sector": row["Sector"],
                "Live Score": ls,
            })

        if not rows:
            st.error("Could not retrieve live data. yfinance may be rate-limited — try again in a moment.")
            return

        result_df = pd.DataFrame(rows).sort_values("Live Score", ascending=False).reset_index(drop=True)

        st.subheader("⭐ Top 3 Live-Scored Picks")
        for col, (_, r) in zip(st.columns(3), result_df.head(3).iterrows()):
            with col:
                st.metric(r["Symbol"], f"${r['Price']:.2f}", f"Score: {r['Live Score']:.0f}/100")
                st.caption(r["Name"])
                upside = f"+{r['Analyst Upside %']:.1f}%" if r["Analyst Upside %"] else "No target"
                st.write(f"RSI: {r['RSI']:.1f} | Analyst: {upside}")

        st.subheader("All Live-Scored Picks")
        st.dataframe(
            result_df.style.format({
                "Price": "${:.2f}", "RSI": "{:.1f}", "vs 52w Low": "{:+.1f}%",
                "Analyst Upside %": lambda x: f"+{x:.1f}%" if x else "N/A",
                "Beta": lambda x: f"{x:.2f}" if x else "N/A", "Live Score": "{:.1f}",
            }).background_gradient(subset=["Live Score"], cmap="RdYlGn"),
            use_container_width=True, height=500)
    else:
        st.info("💡 Enable **Fetch Live Data** in the sidebar to add RSI, analyst targets, and MA signals from yfinance.")

        for col, (_, r) in zip(st.columns(5), scored.head(5).iterrows()):
            with col:
                st.metric(r["Symbol"], f"${r['Last Sale']:.2f}", f"{r['% Change']:+.2f}%")
                st.caption(f"Score: **{r['score']:.0f}/100**")

        disp = scored[["Symbol","Name","Last Sale","% Change","Volume","Market Cap","Sector","score"]].rename(
            columns={"Last Sale":"Price","score":"Score"})
        st.dataframe(
            disp.style.format({"Price":"${:.2f}","% Change":"{:+.2f}%",
                               "Volume":"{:,.0f}","Market Cap":"${:,.0f}","Score":"{:.1f}"})
            .background_gradient(subset=["Score"], cmap="RdYlGn"),
            use_container_width=True, height=480)

        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Score Distribution")
            fig = px.histogram(scored, x="score", nbins=20, color_discrete_sequence=["#6366f1"])
            fig.update_layout(height=280)
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            st.subheader("Sector Breakdown")
            sec_c = scored.groupby("Sector").size().reset_index(name="Count")
            fig2 = px.bar(sec_c.sort_values("Count"), x="Count", y="Sector",
                          orientation="h", color="Count", color_continuous_scale="viridis")
            fig2.update_layout(height=280, coloraxis_showscale=False)
            st.plotly_chart(fig2, use_container_width=True)


def page_stock_detail(df: pd.DataFrame):
    st.title("🔍 Stock Detail")
    st.caption("CSV snapshot + live data from yfinance: RSI, moving averages, analyst targets, and 52-week range")

    symbols = sorted(df["Symbol"].dropna().unique().tolist())
    default_idx = symbols.index("AAPL") if "AAPL" in symbols else 0
    selected = st.selectbox("Choose a stock", symbols, index=default_idx)

    rows = df[df["Symbol"] == selected]
    if rows.empty:
        st.error("Stock not found.")
        return

    row = rows.iloc[0]
    scored = compute_scores(rows)
    csv_score = float(scored["score"].iloc[0])

    st.subheader(f"{selected} — {row['Name']}")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Last Price (CSV)", f"${row['Last Sale']:.2f}", f"{row['Net Change']:+.2f}")
    c2.metric("Daily % Change", f"{row['% Change']:+.2f}%")
    c3.metric("Volume", f"{row['Volume']:,.0f}")
    c4.metric("Market Cap", _fmt_mcap(row["Market Cap"]))

    st.markdown(f"**Sector:** {row['Sector']}  |  **Industry:** {row.get('Industry','N/A')}  |  **Country:** {row['Country']}")
    st.markdown("---")

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
            lc1.metric("Live Price", f"${price:.2f}")
            rsi = live["rsi"]
            lc2.metric("RSI (14)", f"{rsi:.1f}",
                        "Oversold 🟢" if rsi < 35 else ("Overbought 🔴" if rsi > 70 else "Neutral ⚪"))
            if live["analyst_upside"] is not None:
                lc3.metric("Analyst Upside", f"{live['analyst_upside']:+.1f}%",
                            f"Target: ${live['analyst_target']:.2f}")
            else:
                lc3.metric("Analyst Upside", "N/A")
            lc4.metric("Beta", f"{live['beta']:.2f}" if live["beta"] else "N/A")

            st.subheader("Price Chart — 3 Months")
            fig = go.Figure()
            fig.add_trace(go.Candlestick(x=hist.index, open=hist["Open"], high=hist["High"],
                                          low=hist["Low"], close=hist["Close"], name="Price"))
            if len(hist) >= 20:
                fig.add_trace(go.Scatter(x=hist.index, y=hist["Close"].rolling(20).mean(),
                                          name="MA20", line=dict(color="#f59e0b", width=1.5)))
            if len(hist) >= 50:
                fig.add_trace(go.Scatter(x=hist.index, y=hist["Close"].rolling(50).mean(),
                                          name="MA50", line=dict(color="#3b82f6", width=1.5)))
            fig.update_layout(xaxis_rangeslider_visible=False, height=420)
            st.plotly_chart(fig, use_container_width=True)

            cl, cr = st.columns(2)
            with cl:
                st.subheader("52-Week Range")
                lo, hi = live["week52_low"], live["week52_high"]
                pct = float(np.clip((price - lo) / (hi - lo + 1e-9), 0, 1))
                st.progress(pct)
                st.caption(f"Low: ${lo:.2f}  ·  Now: ${price:.2f}  ·  High: ${hi:.2f}")
            with cr:
                st.subheader("Volume (30d)")
                vfig = px.bar(hist.tail(30), x=hist.tail(30).index, y="Volume",
                              color_discrete_sequence=["#6366f1"])
                vfig.update_layout(height=220, showlegend=False)
                st.plotly_chart(vfig, use_container_width=True)

            live_score, breakdown = score_live(live)
            st.subheader(f"Live High-Gain Score: {live_score:.1f} / 100")
            sfig = px.bar(x=list(breakdown.keys()), y=list(breakdown.values()),
                          color=list(breakdown.values()), color_continuous_scale="RdYlGn",
                          range_color=[0, 100])
            sfig.update_layout(height=240, coloraxis_showscale=False, yaxis_range=[0, 100])
            st.plotly_chart(sfig, use_container_width=True)
    else:
        st.info("Click **Fetch Live Analysis** to load the yfinance chart, RSI, analyst targets, and live score.")
        st.subheader(f"CSV Score: {csv_score:.1f} / 100")
        csv_factors = {
            "Momentum": float(scored["momentum_score"].iloc[0]),
            "Volume": float(scored["volume_score"].iloc[0]),
            "Market Cap": float(scored["mcap_score"].iloc[0]),
            "Price Range": float(scored["price_score"].iloc[0]),
            "Direction": float(scored["direction_score"].iloc[0]),
        }
        sfig = px.bar(x=list(csv_factors.keys()), y=list(csv_factors.values()),
                      color=list(csv_factors.values()), color_continuous_scale="RdYlGn",
                      range_color=[0, 100])
        sfig.update_layout(height=240, coloraxis_showscale=False, yaxis_range=[0, 100])
        st.plotly_chart(sfig, use_container_width=True)

    st.markdown("---")
    st.subheader(f"Top Sector Peers — {row['Sector']}")
    peers = df[(df["Sector"] == row["Sector"]) & (df["Symbol"] != selected)]
    if not peers.empty:
        peer_sc = compute_scores(peers).nlargest(10, "score")
        st.dataframe(
            peer_sc[["Symbol","Name","Last Sale","% Change","Volume","score"]].rename(
                columns={"Last Sale":"Price","score":"Score"})
            .style.format({"Price":"${:.2f}","% Change":"{:+.2f}%","Volume":"{:,.0f}","Score":"{:.1f}"}),
            use_container_width=True, height=300)


def page_ai_agent(df: pd.DataFrame):
    st.title("💬 AI Stock Analysis Agent")
    st.caption("Local agent — no external AI needed. Analyzes stocks from NASDAQ data and explains market concepts.")

    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "pending_q" not in st.session_state:
        st.session_state.pending_q = None

    with st.sidebar:
        st.subheader("💡 Quick Questions")
        quick_qs = [
            "Show me the top picks today",
            "Top gainers today",
            "Biggest losers today",
            "Most active stocks",
            "Technology sector",
            "Healthcare sector",
            "Explain RSI",
            "What is a short squeeze?",
            "Explain momentum investing",
            "What is market cap?",
            "How do earnings affect stocks?",
            "Explain stop loss orders",
        ]
        for q in quick_qs:
            if st.button(q, use_container_width=True, key=f"q_{q[:18]}"):
                st.session_state.pending_q = q
                st.rerun()

        st.markdown("---")
        if st.button("🗑 Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

        st.markdown("---")
        st.caption("**Tips:**")
        st.caption("• Type a ticker: `AAPL`")
        st.caption("• Compare: `AAPL vs MSFT`")
        st.caption("• Concepts: `explain MACD`")
        st.caption("• For live charts, use Stock Detail")

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    prompt = st.session_state.pending_q
    if prompt:
        st.session_state.pending_q = None
    else:
        prompt = st.chat_input("Ask about any stock, strategy, or market concept…")

    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        response = agent_respond(prompt, df)
        with st.chat_message("assistant"):
            st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    df = load_data()

    st.sidebar.title("📈 Stock Analyzer")
    st.sidebar.markdown("---")

    page = st.sidebar.radio("Navigate", [
        "🌐 Market Overview",
        "📊 Screener",
        "🚀 Top Picks",
        "🔍 Stock Detail",
        "💬 AI Agent",
    ])

    st.sidebar.markdown("---")
    gainers = int((df["% Change"] > 0).sum())
    losers = int((df["% Change"] < 0).sum())
    st.sidebar.caption(f"**{len(df):,} stocks** loaded")
    st.sidebar.caption(f"🟢 {gainers:,} gainers  |  🔴 {losers:,} losers")
    st.sidebar.caption("CSV data · Live via yfinance")

    pages = {
        "🌐 Market Overview": page_market_overview,
        "📊 Screener": page_screener,
        "🚀 Top Picks": page_top_picks,
        "🔍 Stock Detail": page_stock_detail,
        "💬 AI Agent": page_ai_agent,
    }
    pages[page](df)


if __name__ == "__main__":
    main()
