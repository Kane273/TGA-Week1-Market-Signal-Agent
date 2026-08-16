# ⚡ NASDAQ Terminal

An AI-powered NASDAQ stock screener and analysis tool built with Streamlit. Combines live market data with a GPT agent that can answer questions, look up tickers, compare stocks, and analyze sectors — all in a dark finance terminal aesthetic.

---

## Features

### 📊 Market Overview
- Live market breadth: total stocks, gainers, losers, and unchanged counts
- Sentiment badge (Bullish / Neutral / Bearish) derived from gainer/loser ratio
- Sector performance charts — stacked breadth (% gainers vs losers) and average % change per sector, both horizontal for easy reading
- Return distribution histogram across all 7,000+ NASDAQ stocks
- Top 10 gainers and losers (filtered to $50M+ market cap and 50K+ daily volume)

### 🔍 Screener
- Filter the full NASDAQ universe by sector, market cap range, price range, and % change
- Sortable, styled dataframe with color-coded gain/loss column
- Summary metrics: stocks shown, average % change, total volume

### 🚀 Top Picks
- Algorithmic momentum score (0–100) combining % change, volume ratio, RSI proximity, and market cap
- Optional live data fetch via yfinance for real-time RSI, moving averages, 52-week range, and candlestick charts
- Medal cards (gold / silver / bronze) for the top 3 picks
- Full ranked table with color-coded score column

### 🔎 Stock Detail
- Search any NASDAQ symbol
- Live price card with day change, market cap, volume, and 52-week range
- 30-day candlestick chart (OHLCV) fetched from yfinance
- Visual 52-week range indicator showing current price position

### 🤖 AI Agent
- GPT-powered conversational agent (gpt-5.6-luna via Replit AI Integrations)
- Tool-calling architecture: the model selects the right data tool, the app executes it against live data, then GPT streams a synthesized answer
- 8 built-in tools: `lookup_stock`, `get_top_gainers`, `get_top_losers`, `get_most_active`, `get_top_picks`, `get_sector_analysis`, `get_market_overview`, `compare_stocks`
- Streaming responses via `st.write_stream`
- Conversation history (last 20 turns sent to the model)
- Quick-prompt buttons in sidebar for common questions
- Rule-based fallback if the API is unavailable
- Financial disclaimer appended to every response

---

## Tech Stack

| Layer | Library |
|---|---|
| App framework | [Streamlit](https://streamlit.io) ≥ 1.35 |
| Market data | [yfinance](https://github.com/ranaroussi/yfinance) ≥ 0.2.40 |
| Screener dataset | NASDAQ bulk screener CSV (bundled, refreshed on load) |
| Charts | [Plotly](https://plotly.com/python/) ≥ 5.20 |
| Data wrangling | [pandas](https://pandas.pydata.org) ≥ 2.0, [numpy](https://numpy.org) ≥ 1.26 |
| AI | OpenAI Python SDK via Replit AI Integrations proxy |
| Model | `gpt-5.6-luna` (tool-calling with `reasoning_effort="none"`) |

---

## Project Structure

```
streamlit-app/
├── app.py                  # Single-file Streamlit app (~1,900 lines)
├── nasdaq_screener.csv     # Bundled NASDAQ screener snapshot
├── requirements.txt        # Python dependencies
└── .streamlit/
    └── config.toml         # Server config (CORS off, headless, dynamic port)

artifacts/
└── stock-analyzer/         # Replit artifact config (workflow, routing)
```

### Key sections inside `app.py`

| Section | What it does |
|---|---|
| `_inject_css()` | Full dark-theme CSS injection — glassmorphism cards, gradient typography, chart containers, sidebar nav styling |
| `_stat_card()` | HTML metric card with radial glow and top-edge gradient accent |
| `_medal_card()` | Gold / silver / bronze HTML card for top picks |
| `_section_hdr()` | Section header with violet left-bar accent |
| `_badge()` | Pill badge for sentiment labels |
| `_chart_dark()` | Applies consistent dark Plotly theme to any figure |
| `load_data()` | Loads and cleans the NASDAQ CSV; derives Market Cap and normalizes columns |
| `fetch_live_data()` | Parallel yfinance fetch (ThreadPoolExecutor) for RSI, OHLCV, 52-week range |
| `get_openai_client()` | `@st.cache_resource` OpenAI client via `AI_INTEGRATIONS_OPENAI_BASE_URL` + `AI_INTEGRATIONS_OPENAI_API_KEY` |
| `_TOOL_SPECS` | OpenAI function-calling tool definitions (8 tools) |
| `_dispatch_tool()` | Routes tool calls to Python implementations against live `df` |
| `agent_respond_stream()` | Generator: tool-selection call → execute tools → stream synthesis. Falls back to rule-based on error |
| `page_market_overview()` | Market Overview page |
| `page_screener()` | Screener page |
| `page_top_picks()` | Top Picks page |
| `page_stock_detail()` | Stock Detail page |
| `page_ai_agent()` | AI Agent chat page |
| `main()` | Entry point — injects CSS, renders sidebar nav, dispatches to page functions |

---

## Environment Variables

Set automatically by Replit AI Integrations. Do not set these manually.

| Variable | Purpose |
|---|---|
| `AI_INTEGRATIONS_OPENAI_BASE_URL` | Replit proxy base URL for the OpenAI-compatible API |
| `AI_INTEGRATIONS_OPENAI_API_KEY` | Dummy key required by the OpenAI SDK (handled by the proxy) |

---

## Running Locally

```bash
pip install -r streamlit-app/requirements.txt
streamlit run streamlit-app/app.py
```

> **Note:** The AI Agent requires `AI_INTEGRATIONS_OPENAI_BASE_URL` and `AI_INTEGRATIONS_OPENAI_API_KEY` to be set. Without them, the agent falls back to a rule-based response system automatically.

---

## AI Agent — How It Works

```
User message
    │
    ▼
Step 1 — Tool selection (non-streamed)
  gpt-5.6-luna receives the user message + system prompt + 8 tool definitions
  reasoning_effort="none" required for tool calls with gpt-5.6 models
    │
    ├─ Tool call requested → Step 2
    └─ No tool needed → yield direct answer → done
    │
    ▼
Step 2 — Tool execution
  _dispatch_tool() runs the selected function(s) against the live DataFrame
  Results formatted as JSON strings and appended as tool messages
    │
    ▼
Step 3 — Streaming synthesis
  gpt-5.6-luna receives tool results and streams a natural-language answer
  st.write_stream() renders tokens as they arrive
    │
    ▼
  Financial disclaimer appended
```

**Fallback chain:**
1. API call fails → rule-based `_agent_respond_inner_fallback()`
2. Fallback fails → generic error message with the exception text

---

## Design Notes

- **Color palette:** `#07091a` background · `#6366f1` indigo accent · `#a78bfa` violet highlights · `#10b981` green · `#f43f5e` rose
- **Fonts:** Inter (UI) · JetBrains Mono (tickers, code)
- **Chart containers:** CSS targets `[data-testid="stPlotlyChart"]` to apply glass-card treatment automatically to every Plotly chart
- **Metric cards:** `::before` pseudo-element draws a gradient top-edge glow on each card
- **Sidebar nav:** Radio circles hidden; selected state uses `:has(input:checked)` for a custom pill highlight

---

## Known Limitations

- The NASDAQ CSV is a snapshot; screener data is only as fresh as the last app restart
- yfinance live-fetch is on-demand (Top Picks and Stock Detail pages only) and subject to Yahoo Finance rate limits
- The `use_container_width` Streamlit parameter is deprecated; future Streamlit versions will require `width='stretch'`
- AI Agent conversation history is session-only (not persisted across browser refreshes)
