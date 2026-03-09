import streamlit as st
import pandas as pd
import requests
from ta.momentum import RSIIndicator
from ta.trend import MACD, EMAIndicator
from concurrent.futures import ThreadPoolExecutor

st.set_page_config("🚀 AI Crypto Screener", layout="wide")
st.markdown("""
    <style>
        .main { background-color: #0e1117; color: white; }
        .stButton > button { background-color: #4CAF50; color: white; font-size: 18px; }
        .stDataFrame { font-size: 16px; }
    </style>
""", unsafe_allow_html=True)

st.title("🚀 AI Agent: Pump/Dump Predictor")
st.caption("💥 Shows coins with likely pump/dump potential using volume, price action, and momentum")

# === CONFIG ===
MAX_COINS = 500
RSI_BUY_THRESHOLD = 30
RSI_SELL_THRESHOLD = 70
VOLUME_SPIKE_RATIO = 1.5
PRICE_PUMP_THRESHOLD = 2.5
PRICE_DUMP_THRESHOLD = -2.5

# === UTILITIES ===
def get_all_coins():
    url = "https://api.coingecko.com/api/v3/coins/list"
    res = requests.get(url)
    if res.status_code == 200:
        return res.json()
    return []

def get_coin_market_data(coin_id):
    url = f"https://api.coingecko.com/api/v3/coins/{coin_id}"
    try:
        res = requests.get(url)
        if res.status_code == 200:
            return res.json()
    except:
        return None
    return None

def get_ohlcv(symbol):
    try:
        url = f"https://api.binance.com/api/v3/klines?symbol={symbol}USDT&interval=1h&limit=100"
        res = requests.get(url)
        if res.status_code != 200:
            return None
        raw = res.json()
        df = pd.DataFrame(raw, columns=["timestamp", "open", "high", "low", "close", "volume", "_", "quote_volume", "trades", "taker_base_vol", "taker_quote_vol", "ignore"])
        df["close"] = pd.to_numeric(df["close"])
        df["open"] = pd.to_numeric(df["open"])
        df["volume"] = pd.to_numeric(df["volume"])
        return df
    except:
        return None

def analyze_coin(coin):
    coin_id = coin['id']
    symbol = coin['symbol'].upper()

    market_data = get_coin_market_data(coin_id)
    if not market_data or 'market_data' not in market_data:
        return None

    market_cap = market_data['market_data'].get('market_cap', {}).get('usd', 0)
    circulating_supply = market_data['market_data'].get('circulating_supply', 0)
    if market_cap < 10_000_000 or circulating_supply == 0:
        return None

    df = get_ohlcv(symbol)
    if df is None or df.empty or len(df) < 10:
        return None

    try:
        rsi = RSIIndicator(df['close']).rsi().iloc[-1]
        macd = MACD(df['close']).macd_diff().iloc[-1]
        ema_fast = EMAIndicator(df['close'], window=9).ema_indicator().iloc[-1]
        ema_slow = EMAIndicator(df['close'], window=21).ema_indicator().iloc[-1]

        volume_now = df['volume'].iloc[-1]
        volume_avg = df['volume'].iloc[-10:-1].mean()
        volume_ratio = volume_now / volume_avg if volume_avg > 0 else 0

        price_now = df['close'].iloc[-1]
        price_past = df['close'].iloc[-4]  # 3 hours ago
        price_change_pct = ((price_now - price_past) / price_past) * 100
    except:
        return None

    reasons = []
    signal = None

    if volume_ratio >= VOLUME_SPIKE_RATIO:
        reasons.append(f"Volume spike x{volume_ratio:.1f}")
        if price_change_pct >= PRICE_PUMP_THRESHOLD:
            signal = "PUMP"
            reasons.append(f"Price +{price_change_pct:.2f}%")
        elif price_change_pct <= PRICE_DUMP_THRESHOLD:
            signal = "DUMP"
            reasons.append(f"Price {price_change_pct:.2f}%")

    # fallback logic if no clear pump/dump but technicals align
    if not signal:
        if rsi < RSI_BUY_THRESHOLD and macd > 0 and ema_fast > ema_slow:
            signal = "LIKELY PUMP"
            reasons.append("RSI < 30, MACD > 0, EMA Bullish")
        elif rsi > RSI_SELL_THRESHOLD and macd < 0 and ema_fast < ema_slow:
            signal = "LIKELY DUMP"
            reasons.append("RSI > 70, MACD < 0, EMA Bearish")

    if not signal:
        return None

    return {
        "Coin": coin['name'],
        "Symbol": symbol,
        "Signal": signal,
        "Price Change %": round(price_change_pct, 2),
        "Volume Spike x": round(volume_ratio, 2),
        "RSI": round(rsi, 2),
        "MACD": round(macd, 4),
        "EMA Fast": round(ema_fast, 4),
        "EMA Slow": round(ema_slow, 4),
        "Reason": ", ".join(reasons)
    }

# === UI BUTTON ===
start = st.button("🚨 Find Pump/Dump Coins")
if start:
    with st.spinner("Scanning for high-volume movers and momentum setups..."):
        coins = get_all_coins()[:MAX_COINS]
        results = []

        with ThreadPoolExecutor(max_workers=16) as executor:
            futures = [executor.submit(analyze_coin, coin) for coin in coins]
            for future in futures:
                result = future.result()
                if result:
                    results.append(result)

        if results:
            df = pd.DataFrame(results)
            st.success(f"🚀 Found {len(df)} coins with pump or dump potential!")
            st.dataframe(df.sort_values("Signal"), use_container_width=True)
        else:
            st.warning("😕 No coins found even with fallback logic. Market may be calm.")
