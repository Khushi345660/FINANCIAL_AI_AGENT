"""
Enhanced Financial Data Insight Agent — Indian Stock Market (NSE/BSE)
=======================================================================
Features:
  ✅ Tool 1  — Fetch 30-day stock data (NSE/BSE via yfinance)
  ✅ Tool 2  — Trend Detection (up / down / volatile / stable)
  ✅ Tool 3  — RSI (Relative Strength Index) Indicator
  ✅ Tool 4  — Moving Averages (SMA 7, SMA 20)
  ✅ Tool 5  — Breakout & Breakdown Detection
  ✅ Tool 6  — Compare 2 Indian Stocks side-by-side
  ✅ Gradio UI with Tabs
  ✅ Reasoning log for every tool
  ✅ Handles failure cases gracefully
  ✅ Indian market focus (auto-appends .NS for NSE)

Indian Ticker Examples:
  RELIANCE.NS, TCS.NS, INFY.NS, HDFCBANK.NS, WIPRO.NS,
  SBIN.NS, ICICIBANK.NS, BAJFINANCE.NS, ADANIENT.NS, ITC.NS
"""

import gradio as gr
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta


# ══════════════════════════════════════════════════════════
# HELPER — Normalize Indian ticker
# ══════════════════════════════════════════════════════════
def normalize_ticker(ticker: str) -> str:
    ticker = ticker.strip().upper()
    # If user types RELIANCE → auto-add .NS (NSE)
    if ticker and "." not in ticker:
        ticker = ticker + ".NS"
    return ticker


# ══════════════════════════════════════════════════════════
# TOOL 1 — Fetch Stock Data
# ══════════════════════════════════════════════════════════
def fetch_stock_data(ticker: str, days: int = 60) -> tuple:
    """Fetches historical data. Uses 60 days so RSI/MA have enough data."""
    log = f"[Tool 1 - fetch_stock_data]\n"
    log += f"  → Ticker: {ticker}\n"

    try:
        end_date   = datetime.today()
        start_date = end_date - timedelta(days=days)
        log += f"  → Period: {start_date.date()} to {end_date.date()}\n"

        stock = yf.Ticker(ticker)
        df = stock.history(start=start_date, end=end_date)

        if df.empty:
            log += f"  ❌ FAILURE: No data for '{ticker}'. Check if ticker is valid (e.g. RELIANCE.NS)\n"
            return None, None, log

        info = {}
        try:
            info = stock.info
        except:
            pass

        company_name = info.get("longName", ticker)
        df.index = df.index.tz_localize(None)
        df = df[["Open", "High", "Low", "Close", "Volume"]].copy()

        log += f"  ✅ SUCCESS: {len(df)} trading days fetched for {company_name}\n"
        log += f"  → Close range: ₹{df['Close'].min():.2f} – ₹{df['Close'].max():.2f}\n"
        return df, company_name, log

    except Exception as e:
        log += f"  ❌ EXCEPTION: {e}\n"
        return None, None, log


# ══════════════════════════════════════════════════════════
# TOOL 2 — Trend Detection
# ══════════════════════════════════════════════════════════
def analyze_trend(df: pd.DataFrame, ticker: str) -> tuple:
    log = "[Tool 2 - analyze_trend]\n"
    closes = df["Close"].tail(30)

    first_price = closes.iloc[0]
    last_price  = closes.iloc[-1]
    pct_change  = ((last_price - first_price) / first_price) * 100
    volatility  = (closes.std() / closes.mean()) * 100

    log += f"  → 30-day price change: {pct_change:+.2f}%\n"
    log += f"  → Volatility (CV): {volatility:.2f}%\n"

    VOLATILITY_THRESHOLD = 5.0
    GROWTH_THRESHOLD     = 3.0

    if volatility > VOLATILITY_THRESHOLD:
        trend = "volatile"
    elif pct_change > GROWTH_THRESHOLD:
        trend = "up"
    elif pct_change < -GROWTH_THRESHOLD:
        trend = "down"
    else:
        trend = "stable"

    log += f"  → Trend: {trend.upper()}\n"

    direction_word = "gained" if pct_change > 0 else "lost"
    insight = f"**Price moved from ₹{first_price:.2f} → ₹{last_price:.2f}** ({direction_word} {abs(pct_change):.2f}%)\n\n"

    if trend == "volatile":
        insight += f"⚠️ **Risk Warning:** High volatility (CV={volatility:.2f}%). Large price swings — trade cautiously."
    elif trend == "up":
        insight += f"📈 **Bullish Trend:** Strong upward movement of {pct_change:.2f}%. Positive market sentiment."
    elif trend == "down":
        insight += f"📉 **Bearish Trend:** Downward movement of {abs(pct_change):.2f}%. Possible selling pressure."
    else:
        insight += f"➡️ **Sideways/Stable:** Low volatility, minimal movement. Market in consolidation phase."

    return trend, pct_change, volatility, insight, log


# ══════════════════════════════════════════════════════════
# TOOL 3 — RSI Indicator
# ══════════════════════════════════════════════════════════
def calculate_rsi(df: pd.DataFrame, period: int = 14) -> tuple:
    log = f"[Tool 3 - calculate_rsi] Period={period}\n"

    closes = df["Close"]
    delta  = closes.diff()
    gain   = delta.where(delta > 0, 0.0)
    loss   = -delta.where(delta < 0, 0.0)

    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()

    rs  = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))

    current_rsi = rsi.iloc[-1]
    log += f"  → Current RSI: {current_rsi:.2f}\n"

    if current_rsi >= 70:
        rsi_signal = "Overbought 🔴"
        rsi_advice = f"RSI = **{current_rsi:.2f}** → Stock is **overbought**. Possible price correction ahead. Consider booking profits."
        log += "  → Signal: OVERBOUGHT (>70)\n"
    elif current_rsi <= 30:
        rsi_signal = "Oversold 🟢"
        rsi_advice = f"RSI = **{current_rsi:.2f}** → Stock is **oversold**. Potential buying opportunity if fundamentals are strong."
        log += "  → Signal: OVERSOLD (<30)\n"
    else:
        rsi_signal = "Neutral ⚪"
        rsi_advice = f"RSI = **{current_rsi:.2f}** → Stock is in **neutral zone** (30–70). No extreme pressure."
        log += "  → Signal: NEUTRAL (30-70)\n"

    return current_rsi, rsi_signal, rsi_advice, rsi.dropna(), log


# ══════════════════════════════════════════════════════════
# TOOL 4 — Moving Averages
# ══════════════════════════════════════════════════════════
def calculate_moving_averages(df: pd.DataFrame) -> tuple:
    log = "[Tool 4 - calculate_moving_averages]\n"

    closes = df["Close"]
    sma7  = closes.rolling(window=7).mean()
    sma20 = closes.rolling(window=20).mean()

    current_price = closes.iloc[-1]
    current_sma7  = sma7.iloc[-1]
    current_sma20 = sma20.iloc[-1]

    log += f"  → Current Price: ₹{current_price:.2f}\n"
    log += f"  → SMA 7  : ₹{current_sma7:.2f}\n"
    log += f"  → SMA 20 : ₹{current_sma20:.2f}\n"

    # Golden Cross / Death Cross detection
    if current_sma7 > current_sma20:
        ma_signal = "Golden Cross 🌟"
        ma_advice = (
            f"**SMA7 (₹{current_sma7:.2f}) > SMA20 (₹{current_sma20:.2f})** → "
            "**Golden Cross** — Short-term momentum is bullish. Uptrend likely to continue."
        )
        log += "  → Golden Cross: SMA7 > SMA20 (Bullish)\n"
    else:
        ma_signal = "Death Cross 💀"
        ma_advice = (
            f"**SMA7 (₹{current_sma7:.2f}) < SMA20 (₹{current_sma20:.2f})** → "
            "**Death Cross** — Short-term momentum is bearish. Downtrend may persist."
        )
        log += "  → Death Cross: SMA7 < SMA20 (Bearish)\n"

    # Price vs MA position
    if current_price > current_sma20:
        ma_advice += f"\n\nPrice (₹{current_price:.2f}) is **above SMA20** → Bullish bias."
    else:
        ma_advice += f"\n\nPrice (₹{current_price:.2f}) is **below SMA20** → Bearish bias."

    return current_sma7, current_sma20, ma_signal, ma_advice, sma7, sma20, log


# ══════════════════════════════════════════════════════════
# TOOL 5 — Breakout & Breakdown Detection
# ══════════════════════════════════════════════════════════
def detect_breakout_breakdown(df: pd.DataFrame) -> tuple:
    log = "[Tool 5 - detect_breakout_breakdown]\n"

    recent    = df.tail(30)
    last_5    = df.tail(5)

    resistance = recent["High"].max()    # 30-day high = resistance
    support    = recent["Low"].min()     # 30-day low  = support
    current    = df["Close"].iloc[-1]
    prev_close = df["Close"].iloc[-2]

    log += f"  → 30-day Resistance (High): ₹{resistance:.2f}\n"
    log += f"  → 30-day Support (Low)    : ₹{support:.2f}\n"
    log += f"  → Current Price           : ₹{current:.2f}\n"
    log += f"  → Previous Close          : ₹{prev_close:.2f}\n"

    # Check breakout / breakdown with 1% buffer
    buffer = 0.01

    if current > resistance * (1 - buffer):
        signal = "🚀 BREAKOUT"
        detail = (
            f"**BREAKOUT DETECTED!**\n"
            f"Price (₹{current:.2f}) has broken above the 30-day resistance of ₹{resistance:.2f}.\n"
            f"This is a **strong bullish signal** — momentum traders may enter long positions.\n"
            f"Watch for volume confirmation!"
        )
        log += "  → BREAKOUT: Price near/above 30-day resistance\n"
    elif current < support * (1 + buffer):
        signal = "💥 BREAKDOWN"
        detail = (
            f"**BREAKDOWN DETECTED!**\n"
            f"Price (₹{current:.2f}) has broken below the 30-day support of ₹{support:.2f}.\n"
            f"This is a **strong bearish signal** — consider protective stops.\n"
            f"High risk of further downside!"
        )
        log += "  → BREAKDOWN: Price near/below 30-day support\n"
    else:
        pct_to_resistance = ((resistance - current) / current) * 100
        pct_to_support    = ((current - support) / current) * 100
        signal = "↔️ Consolidating"
        detail = (
            f"**No Breakout/Breakdown yet.**\n"
            f"Price (₹{current:.2f}) is trading within the range.\n"
            f"- **{pct_to_resistance:.1f}%** away from resistance (₹{resistance:.2f})\n"
            f"- **{pct_to_support:.1f}%** away from support (₹{support:.2f})\n"
            f"Watch for a breakout above ₹{resistance:.2f} or breakdown below ₹{support:.2f}."
        )
        log += f"  → CONSOLIDATING: {pct_to_resistance:.1f}% from resistance, {pct_to_support:.1f}% from support\n"

    return signal, detail, resistance, support, log


# ══════════════════════════════════════════════════════════
# MAIN AGENT — Single Stock
# ══════════════════════════════════════════════════════════
def run_single_agent(ticker_raw: str):
    ticker = normalize_ticker(ticker_raw)
    full_log = "=" * 55 + f"\n🤖 AGENT LOG — {ticker}\n" + "=" * 55 + "\n\n"

    # Tool 1 — Fetch
    df, company_name, log1 = fetch_stock_data(ticker, days=60)
    full_log += log1 + "\n"

    if df is None:
        err = (
            f"❌ **Could not fetch data for '{ticker}'**\n\n"
            "**Possible fixes:**\n"
            "- Use NSE format: `RELIANCE.NS`, `TCS.NS`, `INFY.NS`\n"
            "- Use BSE format: `500325.BO` (BSE code)\n"
            "- Check if the company is listed on NSE/BSE\n"
            "- Try popular tickers: HDFCBANK.NS, SBIN.NS, ITC.NS"
        )
        return err, "", "", "", "", "", full_log, None

    display_ticker = f"{company_name} ({ticker})"

    # Tool 2 — Trend
    trend, pct_change, volatility, trend_insight, log2 = analyze_trend(df, ticker)
    full_log += log2 + "\n"

    # Tool 3 — RSI
    rsi_val, rsi_signal, rsi_advice, rsi_series, log3 = calculate_rsi(df)
    full_log += log3 + "\n"

    # Tool 4 — Moving Averages
    sma7, sma20, ma_signal, ma_advice, sma7_series, sma20_series, log4 = calculate_moving_averages(df)
    full_log += log4 + "\n"

    # Tool 5 — Breakout/Breakdown
    bb_signal, bb_detail, resistance, support, log5 = detect_breakout_breakdown(df)
    full_log += log5 + "\n"

    full_log += "=" * 55 + "\n✅ ANALYSIS COMPLETE\n" + "=" * 55

    # Summary card
    current_price = df["Close"].iloc[-1]
    summary = f"""
## 📊 {display_ticker}

| Metric | Value |
|--------|-------|
| **Current Price** | ₹{current_price:.2f} |
| **30-Day Change** | {pct_change:+.2f}% |
| **Trend** | {trend.upper()} |
| **RSI (14)** | {rsi_val:.2f} — {rsi_signal} |
| **SMA 7** | ₹{sma7:.2f} |
| **SMA 20** | ₹{sma20:.2f} |
| **MA Signal** | {ma_signal} |
| **Breakout/Breakdown** | {bb_signal} |
| **30D Resistance** | ₹{resistance:.2f} |
| **30D Support** | ₹{support:.2f} |
"""

    # Price table (last 30 days)
    display_df = df.tail(30).copy()
    display_df = display_df[["Open", "High", "Low", "Close", "Volume"]].copy()
    for col in ["Open", "High", "Low", "Close"]:
        display_df[col] = display_df[col].map("₹{:.2f}".format)
    display_df["Volume"] = display_df["Volume"].map("{:,.0f}".format)
    display_df.index = display_df.index.strftime("%Y-%m-%d")
    display_df.index.name = "Date"

    return (
        summary,
        f"**Trend:** {trend_insight}",
        f"**RSI Signal:** {rsi_advice}",
        f"**Moving Averages:** {ma_advice}",
        f"**Breakout/Breakdown:** {bb_detail}",
        bb_signal,
        full_log,
        display_df.reset_index()
    )


# ══════════════════════════════════════════════════════════
# TOOL 6 — Compare 2 Stocks
# ══════════════════════════════════════════════════════════
def run_comparison(ticker1_raw: str, ticker2_raw: str):
    t1 = normalize_ticker(ticker1_raw)
    t2 = normalize_ticker(ticker2_raw)

    full_log = "=" * 55 + f"\n🤖 COMPARISON LOG — {t1} vs {t2}\n" + "=" * 55 + "\n\n"

    if t1 == t2:
        return "❌ Please enter two **different** stock tickers to compare.", full_log, None

    results = {}
    for ticker in [t1, t2]:
        df, company, log = fetch_stock_data(ticker, days=60)
        full_log += log + "\n"

        if df is None:
            return f"❌ Could not fetch data for **{ticker}**. Check the ticker symbol.", full_log, None

        trend, pct_change, volatility, _, log2 = analyze_trend(df, ticker)
        rsi_val, rsi_signal, _, _, log3       = calculate_rsi(df)
        sma7, sma20, ma_signal, _, _, _, log4 = calculate_moving_averages(df)
        bb_signal, _, resistance, support, log5 = detect_breakout_breakdown(df)

        full_log += log2 + log3 + log4 + log5 + "\n"

        results[ticker] = {
            "company"    : company,
            "price"      : df["Close"].iloc[-1],
            "change_30d" : pct_change,
            "volatility" : volatility,
            "trend"      : trend,
            "rsi"        : rsi_val,
            "rsi_signal" : rsi_signal,
            "sma7"       : sma7,
            "sma20"      : sma20,
            "ma_signal"  : ma_signal,
            "bb_signal"  : bb_signal,
            "resistance" : resistance,
            "support"    : support,
        }

    r1, r2 = results[t1], results[t2]

    # Decide winner per metric
    def winner(val1, val2, higher_is_better=True):
        if higher_is_better:
            return "🏆" if val1 > val2 else ("🏆" if val2 > val1 else "🤝")
        else:
            return "🏆" if val1 < val2 else ("🏆" if val2 < val1 else "🤝")

    w_change = (winner(r1["change_30d"], r2["change_30d"]), winner(r2["change_30d"], r1["change_30d"]))
    w_vol    = (winner(r1["volatility"], r2["volatility"], False), winner(r2["volatility"], r1["volatility"], False))
    w_rsi    = ("📈" if 40 < r1["rsi"] < 60 else "⚠️", "📈" if 40 < r2["rsi"] < 60 else "⚠️")

    comparison_md = f"""
## ⚔️ Stock Comparison: {r1['company']} vs {r2['company']}

| Metric | {t1} | {t2} |
|--------|------|------|
| **Company** | {r1['company']} | {r2['company']} |
| **Current Price** | ₹{r1['price']:.2f} | ₹{r2['price']:.2f} |
| **30-Day Change** | {r1['change_30d']:+.2f}% {w_change[0]} | {r2['change_30d']:+.2f}% {w_change[1]} |
| **Volatility (CV)** | {r1['volatility']:.2f}% {w_vol[0]} | {r2['volatility']:.2f}% {w_vol[1]} |
| **Trend** | {r1['trend'].upper()} | {r2['trend'].upper()} |
| **RSI (14)** | {r1['rsi']:.2f} — {r1['rsi_signal']} {w_rsi[0]} | {r2['rsi']:.2f} — {r2['rsi_signal']} {w_rsi[1]} |
| **SMA 7** | ₹{r1['sma7']:.2f} | ₹{r2['sma7']:.2f} |
| **SMA 20** | ₹{r1['sma20']:.2f} | ₹{r2['sma20']:.2f} |
| **MA Signal** | {r1['ma_signal']} | {r2['ma_signal']} |
| **Breakout/Breakdown** | {r1['bb_signal']} | {r2['bb_signal']} |
| **30D Resistance** | ₹{r1['resistance']:.2f} | ₹{r2['resistance']:.2f} |
| **30D Support** | ₹{r1['support']:.2f} | ₹{r2['support']:.2f} |

---

## 🧠 Comparison Verdict

"""

    # Generate verdict
    verdict_parts = []

    # Return comparison
    if r1["change_30d"] > r2["change_30d"]:
        verdict_parts.append(f"📈 **Better Returns:** {t1} outperformed {t2} with {r1['change_30d']:+.2f}% vs {r2['change_30d']:+.2f}% in 30 days.")
    elif r2["change_30d"] > r1["change_30d"]:
        verdict_parts.append(f"📈 **Better Returns:** {t2} outperformed {t1} with {r2['change_30d']:+.2f}% vs {r1['change_30d']:+.2f}% in 30 days.")
    else:
        verdict_parts.append("🤝 **Returns:** Both stocks performed similarly in the last 30 days.")

    # Risk comparison
    if r1["volatility"] < r2["volatility"]:
        verdict_parts.append(f"🛡️ **Lower Risk:** {t1} is less volatile (CV={r1['volatility']:.2f}%) compared to {t2} (CV={r2['volatility']:.2f}%).")
    else:
        verdict_parts.append(f"🛡️ **Lower Risk:** {t2} is less volatile (CV={r2['volatility']:.2f}%) compared to {t1} (CV={r1['volatility']:.2f}%).")

    # RSI health
    for t, r in [(t1, r1), (t2, r2)]:
        if r["rsi"] > 70:
            verdict_parts.append(f"⚠️ **{t} is overbought** (RSI={r['rsi']:.1f}) — possible correction risk.")
        elif r["rsi"] < 30:
            verdict_parts.append(f"🟢 **{t} is oversold** (RSI={r['rsi']:.1f}) — potential buying opportunity.")

    # Breakout signals
    for t, r in [(t1, r1), (t2, r2)]:
        if "BREAKOUT" in r["bb_signal"]:
            verdict_parts.append(f"🚀 **{t} shows a BREAKOUT signal** — bullish momentum building.")
        elif "BREAKDOWN" in r["bb_signal"]:
            verdict_parts.append(f"💥 **{t} shows a BREAKDOWN signal** — bearish pressure detected.")

    comparison_md += "\n\n".join(verdict_parts)
    comparison_md += "\n\n> ⚠️ *This is for educational purposes only. Not financial advice. Always do your own research before investing.*"

    # Comparison table for dataframe display
    comp_table = pd.DataFrame({
        "Metric"          : ["Current Price", "30-Day Change (%)", "Volatility (CV%)",
                             "Trend", "RSI (14)", "RSI Signal", "SMA 7", "SMA 20",
                             "MA Signal", "Breakout/Breakdown", "Resistance (30D)", "Support (30D)"],
        t1                : [f"₹{r1['price']:.2f}", f"{r1['change_30d']:+.2f}%",
                             f"{r1['volatility']:.2f}%", r1["trend"].upper(),
                             f"{r1['rsi']:.2f}", r1["rsi_signal"],
                             f"₹{r1['sma7']:.2f}", f"₹{r1['sma20']:.2f}",
                             r1["ma_signal"], r1["bb_signal"],
                             f"₹{r1['resistance']:.2f}", f"₹{r1['support']:.2f}"],
        t2                : [f"₹{r2['price']:.2f}", f"{r2['change_30d']:+.2f}%",
                             f"{r2['volatility']:.2f}%", r2["trend"].upper(),
                             f"{r2['rsi']:.2f}", r2["rsi_signal"],
                             f"₹{r2['sma7']:.2f}", f"₹{r2['sma20']:.2f}",
                             r2["ma_signal"], r2["bb_signal"],
                             f"₹{r2['resistance']:.2f}", f"₹{r2['support']:.2f}"],
    })

    full_log += "=" * 55 + "\n✅ COMPARISON COMPLETE\n" + "=" * 55
    return comparison_md, full_log, comp_table


# ══════════════════════════════════════════════════════════
# GRADIO UI
# ══════════════════════════════════════════════════════════
with gr.Blocks(title="Indian Stock Market Agent", theme=gr.themes.Soft()) as demo:

    gr.Markdown("""
    # 🇮🇳 Indian Stock Market — AI Insight Agent
    **Powered by NSE/BSE data via Yahoo Finance**
    
    > Enter stock tickers in NSE format (e.g., `RELIANCE.NS`, `TCS.NS`, `HDFCBANK.NS`)  
    > Or just type the name without `.NS` — it will be added automatically!
    
    ⚠️ *For educational purposes only. Not financial advice.*
    """)

    with gr.Tabs():

        # ── TAB 1: Single Stock Analysis ──────────────────
        with gr.TabItem("📈 Single Stock Analysis"):
            with gr.Row():
                single_ticker = gr.Textbox(
                    label="NSE/BSE Ticker",
                    placeholder="e.g. RELIANCE or RELIANCE.NS",
                    scale=3
                )
                single_btn = gr.Button("🔍 Analyze", variant="primary", scale=1)

            summary_out    = gr.Markdown(label="📊 Summary")

            with gr.Row():
                trend_out  = gr.Markdown(label="📉 Trend")
                rsi_out    = gr.Markdown(label="📊 RSI")

            with gr.Row():
                ma_out     = gr.Markdown(label="📐 Moving Averages")
                bb_out     = gr.Markdown(label="🚀 Breakout/Breakdown")

            bb_signal_out  = gr.Textbox(label="Signal", interactive=False)

            with gr.Accordion("📅 Price Table (Last 30 Days)", open=False):
                price_table = gr.Dataframe(interactive=False)

            with gr.Accordion("🧠 Agent Reasoning Log", open=False):
                single_log  = gr.Textbox(lines=25, interactive=False)

            gr.Examples(
                examples=[
                    ["RELIANCE"], ["TCS"], ["INFY"], ["HDFCBANK"],
                    ["WIPRO"], ["SBIN"], ["ITC"], ["BAJFINANCE"],
                    ["ADANIENT"], ["ICICIBANK"]
                ],
                inputs=single_ticker,
                label="🏦 Popular NSE Stocks"
            )

            single_btn.click(
                fn=run_single_agent,
                inputs=single_ticker,
                outputs=[summary_out, trend_out, rsi_out, ma_out, bb_out,
                         bb_signal_out, single_log, price_table]
            )
            single_ticker.submit(
                fn=run_single_agent,
                inputs=single_ticker,
                outputs=[summary_out, trend_out, rsi_out, ma_out, bb_out,
                         bb_signal_out, single_log, price_table]
            )

        # ── TAB 2: Stock Comparison ────────────────────────
        with gr.TabItem("⚔️ Compare 2 Stocks"):
            gr.Markdown("### Compare any two NSE/BSE listed stocks side-by-side")

            with gr.Row():
                comp_ticker1 = gr.Textbox(label="Stock 1", placeholder="e.g. RELIANCE or RELIANCE.NS", scale=2)
                comp_ticker2 = gr.Textbox(label="Stock 2", placeholder="e.g. TCS or TCS.NS", scale=2)
                comp_btn     = gr.Button("⚔️ Compare", variant="primary", scale=1)

            comp_result = gr.Markdown(label="Comparison Result")

            with gr.Accordion("📊 Comparison Table", open=True):
                comp_table_out = gr.Dataframe(interactive=False)

            with gr.Accordion("🧠 Agent Reasoning Log", open=False):
                comp_log = gr.Textbox(lines=30, interactive=False)

            gr.Examples(
                examples=[
                    ["RELIANCE", "TCS"],
                    ["HDFCBANK", "ICICIBANK"],
                    ["INFY", "WIPRO"],
                    ["SBIN", "BAJFINANCE"],
                    ["ADANIENT", "ITC"],
                ],
                inputs=[comp_ticker1, comp_ticker2],
                label="🔥 Popular Comparisons"
            )

            comp_btn.click(
                fn=run_comparison,
                inputs=[comp_ticker1, comp_ticker2],
                outputs=[comp_result, comp_log, comp_table_out]
            )

    gr.Markdown("""
    ---
    **Supported Exchanges:**  
    🟦 NSE → Add `.NS` suffix (e.g., `RELIANCE.NS`)  
    🟧 BSE → Add `.BO` suffix (e.g., `500325.BO`)  
    
    **Indicators Used:** RSI(14) • SMA7 • SMA20 • 30-Day Resistance/Support  
    **Data Source:** Yahoo Finance (real-time delayed)
    """)

if __name__ == "__main__":
   demo.launch()