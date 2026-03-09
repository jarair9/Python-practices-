# import streamlit as st
# import pandas as pd
# import requests
# import ta
# from concurrent.futures import ThreadPoolExecutor


# st.set_page_config("📊 Crypto Screener", layout="wide")
# st.title("📊 Real-Time Crypto RSI Screener")
# st.markdown("""
#     <style>
#     .stDataFrame th, .stDataFrame td, .stTable td {
#         font-size: 16px !important;
#         padding: 4px 8px !important;
#         height: 30px !important;
#     }
#     .stDataFrame div[data-testid="stDataFrameContainer"] {
#         max-height: 500px !important;
#         overflow: auto;
#         padding: 0px !important;
#         margin: 0px !important;
#     }
#     .stDataFrame table {
#         border-collapse: collapse !important;
#         border: none !important;
#     }
#     .stDataFrame td, .stDataFrame th {
#         border: none !important;
#     }
#     </style>
# """, unsafe_allow_html=True)


# # === Sidebar Filters ===
# with st.sidebar:
#     st.header("⚙️ Filter Settings")
#     timeframe = st.selectbox("⏱️ Timeframe", ["1m", "5m", "15m", "1h", "4h", "1d"])
#     rsi_mode = st.radio("📈 RSI Condition", ["Below", "Above"])
    
#     rsi_threshold = st.slider("🎯 RSI Threshold", 10, 90, 30 if rsi_mode == "Below" else 70)
#     rsi_buffer = st.slider("🧭 RSI Buffer", 0, 10, 2, help="Allow some flexibility above/below RSI threshold")
#     rsi_length = st.slider("📐 RSI Period", 5, 50, 14)
#     smooth_rsi = st.checkbox("📉 Smooth RSI (EMA)", value=False)

#     st.markdown("---")
#     top100_volume = st.checkbox("🔥 Only Top 100 by Volume (Binance)", value=False)
#     new_listings = st.checkbox("🆕 Newly Listed (last 30 days)", value=False)
#     market_cap_filter = st.checkbox("💰 Filter by Market Cap", value=False)
    
#     if market_cap_filter:
#         min_cap, max_cap = st.slider("💵 Market Cap ($)", 0, 10_000_000_000, (50_000_000, 500_000_000), step=10_000_000)
    
#     symbol_search = st.text_input("🔍 Search for Symbol (e.g., BTCUSDT)")

# start_scan = st.button("🚀 Start Scan")

# # === Cached Data Loaders ===
# @st.cache_data(ttl=600)
# def get_coin_gecko_data():
#     all_data = []
#     for page in range(1, 5):
#         url = "https://api.coingecko.com/api/v3/coins/markets"
#         params = {
#             "vs_currency": "usd",
#             "order": "volume_desc",
#             "per_page": 250,
#             "page": page,
#             "sparkline": False
#         }
#         try:
#             response = requests.get(url, params=params, timeout=10)
#             response.raise_for_status()
#             data = response.json()
#             if not data:
#                 break
#             all_data.extend(data)
#         except:
#             break

#     df = pd.DataFrame(all_data)
#     df["symbol_uc"] = df["symbol"].str.upper() + "USDT"
#     df["last_updated"] = pd.to_datetime(df["last_updated"], errors="coerce", utc=True)
#     return df[["id", "symbol_uc", "market_cap", "total_volume", "name", "last_updated"]]

# @st.cache_data(ttl=600)
# def get_binance_symbols():
#     try:
#         res = requests.get("https://api.binance.com/api/v3/exchangeInfo", timeout=10).json()
#         return [
#             s["symbol"] for s in res["symbols"]
#             if s["quoteAsset"] == "USDT" and s["status"] == "TRADING" and not any(x in s["symbol"] for x in ["UP", "DOWN", "BULL", "BEAR"])
#         ]
#     except:
#         return []

# def fetch_ohlcv(symbol, interval="15m", limit=200):
#     try:
#         url = f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval={interval}&limit={limit}"
#         data = requests.get(url, timeout=5).json()
#         df = pd.DataFrame(data, columns=["time", "open", "high", "low", "close", "volume", *["_"]*6])
#         df["close"] = pd.to_numeric(df["close"])
#         return df
#     except:
#         return None

# def analyze_symbol(symbol):
#     df = fetch_ohlcv(symbol, interval=timeframe)
#     if df is None or df.empty:
#         return None

#     try:
#         rsi = ta.momentum.RSIIndicator(df["close"], window=rsi_length).rsi()
#         if smooth_rsi:
#             rsi = rsi.ewm(span=5).mean()
#         current_rsi = rsi.iloc[-1]
#         price = df["close"].iloc[-1]
        
#         if pd.isna(current_rsi):
#             return None

#                 # RSI filtering with buffer
#         if rsi_mode == "Below":
#             if current_rsi > rsi_threshold + rsi_buffer:
#                 return None
#         elif rsi_mode == "Above":
#             if current_rsi < rsi_threshold - rsi_buffer:
#                 return None


#         return {
#             "Symbol": symbol,
#             "Price": round(price, 4),
#             "RSI": round(current_rsi, 2)
#         }

#     except Exception as e:
#         return None

# # === Run Scan ===
# if start_scan:
#     st.info("📡 Fetching and scanning data... Please wait.")

#     binance_symbols = get_binance_symbols()
#     gecko_data = get_coin_gecko_data()

#     if top100_volume:
#         top100 = gecko_data.sort_values("total_volume", ascending=False).head(100)
#         binance_symbols = [s for s in binance_symbols if s in top100["symbol_uc"].values]

#     if new_listings:
#         fresh = gecko_data[
#             gecko_data["last_updated"] > pd.Timestamp.utcnow() - pd.Timedelta(days=30)
#         ]
#         binance_symbols = [s for s in binance_symbols if s in fresh["symbol_uc"].values]

#     if market_cap_filter:
#         cap_filtered = gecko_data[
#             (gecko_data["market_cap"] >= min_cap) & (gecko_data["market_cap"] <= max_cap)
#         ]
#         binance_symbols = [s for s in binance_symbols if s in cap_filtered["symbol_uc"].values]

#     if symbol_search:
#         binance_symbols = [s for s in binance_symbols if symbol_search.upper() in s]

#     st.write(f"🔎 Scanning {len(binance_symbols)} USDT pairs...")

#     results = []
#     with ThreadPoolExecutor(max_workers=30) as executor:
#         for result in executor.map(analyze_symbol, binance_symbols):
#             if result:
#                 results.append(result)

#     if results:
#         df = pd.DataFrame(results)
#         df = df.sort_values("RSI", ascending=(rsi_mode == "Below"))

#         st.success(f"✅ Found {len(df)} matching coins.")

#         # Dynamically calculate height
#         row_height = 38  # ~38px per row (safe average with padding)
#         header_height = 40
#         total_height = header_height + row_height * len(df)

#         st.dataframe(
#             df.style.background_gradient(cmap="RdYlGn_r", subset=["RSI"]),
#             use_container_width=True,
#             height=min(total_height, 800)  # cap max height to avoid scroll on too many rows
#         )

#     else:
#         st.warning("❌ No matching coins found. Try adjusting the filters.")



import streamlit as st
import pandas as pd
import requests
import ta
from concurrent.futures import ThreadPoolExecutor

# ==== Page Config ====
st.set_page_config("📊 Crypto Screener", layout="wide")
st.title("📊 Real-Time Crypto RSI Screener")

# ==== Custom Table CSS ====
st.markdown("""
    <style>
    .stDataFrame th, .stDataFrame td {
        font-size: 16px !important;
        padding: 4px 8px !important;
    }
    .stDataFrame div[data-testid="stDataFrameContainer"] {
        max-height: 500px !important;
        overflow: auto;
    }
    </style>
""", unsafe_allow_html=True)

# ==== Sidebar Filters ====
with st.sidebar:
    st.header("⚙️ Filter Settings")
    timeframe = st.selectbox("⏱️ Timeframe", ["1m", "5m", "15m", "1h", "4h", "1d"])
    rsi_mode = st.radio("📈 RSI Condition", ["Below", "Above"])
    rsi_threshold = st.slider("🎯 RSI Threshold", 10, 90, 30 if rsi_mode == "Below" else 70)
    rsi_buffer = st.slider("🧭 RSI Buffer", 0, 10, 2)
    rsi_length = st.slider("📐 RSI Period", 5, 50, 14)
    smooth_rsi = st.checkbox("📉 Smooth RSI (EMA)", value=False)

    st.markdown("---")
    top100_volume = st.checkbox("🔥 Only Top 100 by Volume (Binance)", value=False)
    new_listings = st.checkbox("🆕 Newly Listed (last 30 days)", value=False)
    market_cap_filter = st.checkbox("💰 Filter by Market Cap", value=False)
    
    if market_cap_filter:
        min_cap, max_cap = st.slider("💵 Market Cap ($)", 0, 10_000_000_000, (50_000_000, 500_000_000), step=10_000_000)
    
    symbol_search = st.text_input("🔍 Search for Symbol (e.g., BTCUSDT)")

start_scan = st.button("🚀 Start Scan")

# ==== Cached Data Loaders ====
@st.cache_data(ttl=86400)
def get_binance_symbols():
    try:
        res = requests.get("https://api.binance.com/api/v3/exchangeInfo", timeout=10).json()
        return [
            s["symbol"] for s in res["symbols"]
            if s["quoteAsset"] == "USDT" and s["status"] == "TRADING"
            and not any(x in s["symbol"] for x in ["UP", "DOWN", "BULL", "BEAR"])
        ]
    except:
        return []

@st.cache_data(ttl=600)
def get_coin_gecko_data():
    all_data = []
    for page in range(1, 5):
        url = "https://api.coingecko.com/api/v3/coins/markets"
        params = {"vs_currency": "usd", "order": "volume_desc", "per_page": 250, "page": page}
        try:
            data = requests.get(url, params=params, timeout=10).json()
            if not data:
                break
            all_data.extend(data)
        except:
            break
    df = pd.DataFrame(all_data)
    df["symbol_uc"] = df["symbol"].str.upper() + "USDT"
    df["last_updated"] = pd.to_datetime(df["last_updated"], errors="coerce", utc=True)
    return df[["id", "symbol_uc", "market_cap", "total_volume", "name", "last_updated"]]

# ==== Data Fetchers ====
def fetch_ohlcv(symbol, interval="15m", limit=200):
    try:
        url = f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval={interval}&limit={limit}"
        data = requests.get(url, timeout=5).json()
        df = pd.DataFrame(data, columns=["time", "open", "high", "low", "close", "volume", *["_"]*6])
        df["close"] = pd.to_numeric(df["close"])
        return df
    except:
        return None

def analyze_symbol(symbol):
    df = fetch_ohlcv(symbol, interval=timeframe)
    if df is None or df.empty:
        return None
    try:
        rsi = ta.momentum.RSIIndicator(df["close"], window=rsi_length).rsi()
        if smooth_rsi:
            rsi = rsi.ewm(span=5).mean()
        current_rsi = rsi.iloc[-1]
        if pd.isna(current_rsi):
            return None

        # RSI filter with buffer
        if rsi_mode == "Below" and current_rsi > rsi_threshold + rsi_buffer:
            return None
        if rsi_mode == "Above" and current_rsi < rsi_threshold - rsi_buffer:
            return None

        return {"Symbol": symbol, "Price": round(df["close"].iloc[-1], 4), "RSI": round(current_rsi, 2)}
    except:
        return None

# ==== Table Color Function ====
def color_rsi(val):
    if val < 30:
        color = "#2ecc71"  # green
    elif val > 70:
        color = "#e74c3c"  # red
    else:
        color = "#f1c40f"  # yellow
    return f"background-color: {color}; color: white;"

# ==== Run Scan ====
if start_scan:
    with st.spinner("📡 Fetching and scanning data... Please wait."):
        binance_symbols = set(get_binance_symbols())
        gecko_data = get_coin_gecko_data()

        # Apply all filters in one step
        if top100_volume:
            gecko_data = gecko_data.nlargest(100, "total_volume")
        if new_listings:
            gecko_data = gecko_data[gecko_data["last_updated"] > pd.Timestamp.utcnow() - pd.Timedelta(days=30)]
        if market_cap_filter:
            gecko_data = gecko_data[(gecko_data["market_cap"] >= min_cap) & (gecko_data["market_cap"] <= max_cap)]
        if symbol_search:
            gecko_data = gecko_data[gecko_data["symbol_uc"].str.contains(symbol_search.upper())]

        # Keep only symbols on Binance
        symbols_to_scan = list(binance_symbols.intersection(set(gecko_data["symbol_uc"])))

        st.write(f"🔎 Scanning {len(symbols_to_scan)} USDT pairs...")
        results = []
        progress = st.progress(0)

        with ThreadPoolExecutor(max_workers=30) as executor:
            for i, result in enumerate(executor.map(analyze_symbol, symbols_to_scan), 1):
                if result:
                    results.append(result)
                progress.progress(i / len(symbols_to_scan))

        if results:
            df = pd.DataFrame(results).sort_values("RSI", ascending=(rsi_mode == "Below"))
            st.success(f"✅ Found {len(df)} matching coins.")

            row_height = 38
            header_height = 40
            total_height = header_height + row_height * len(df)

            st.dataframe(
                df.style.applymap(color_rsi, subset=["RSI"]),
                use_container_width=True,
                height=min(total_height, 800)
            )
        else:
            st.warning("❌ No matching coins found. Try adjusting the filters.")
