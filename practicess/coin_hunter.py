# Filename: streamlit_crypto_gemini_agent.py
"""
Streamlit Crypto Fundamental Analysis Agent (Gemini Flash 2.0)
- CoinGecko for market & token info
- Scoring engine (market, tokenomics, social)
- Gemini for human-readable reasoning & final verdict
- Export CSV, local watchlist
Notes: Put GEMINI_API_KEY in .env. Gemini SDK calls will gracefully fallback to a template if the SDK call fails.
"""

import os
import time
import json
from typing import Dict, Any, List, Tuple

import streamlit as st
import requests
import pandas as pd
import numpy as np
from dotenv import load_dotenv

# Gemini SDK
try:
    import google.generativeai as genai
    GEMINI_SDK_AVAILABLE = True
except Exception:
    GEMINI_SDK_AVAILABLE = False

# Load env
load_dotenv()
COINGECKO_BASE = "https://api.coingecko.com/api/v3"
GEMINI_API_KEY = os.getenv("AIzaSyCphPLGWXvbNMQlPy5WXFrbF_6E_Em721U", "")

if GEMINI_API_KEY and GEMINI_SDK_AVAILABLE:
    genai.configure(api_key=GEMINI_API_KEY)
    # In some SDK versions it's used via generate_text / GenerativeModel — we'll try both patterns later.

# ------------------------
# CoinGecko helpers
# ------------------------
@st.cache_data(ttl=60)
def coingecko_get(path: str, params: dict = None) -> Dict[str, Any]:
    url = f"{COINGECKO_BASE}/{path}"
    try:
        r = requests.get(url, params=params, timeout=15)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        st.error(f"CoinGecko request failed: {e}")
        return {}

@st.cache_data(ttl=300)
def get_top_coins(vs_currency: str = "usd", per_page: int = 100) -> pd.DataFrame:
    res = coingecko_get("coins/markets", params={
        "vs_currency": vs_currency,
        "order": "market_cap_desc",
        "per_page": per_page,
        "page": 1,
        "price_change_percentage": "1h,24h,7d"
    })
    if not res:
        return pd.DataFrame()
    df = pd.DataFrame(res)
    return df

@st.cache_data(ttl=300)
def get_coin_details(id: str) -> Dict[str, Any]:
    return coingecko_get(f"coins/{id}", params={"localization": "false", "tickers": "false", "market_data": "true", "community_data": "true", "developer_data": "true"})

# ------------------------
# Simple scoring engine
# ------------------------
def score_market_features(coin: Dict[str, Any]) -> Tuple[float, Dict[str, float]]:
    weights = {"volume": 0.4, "market_cap": 0.3, "momentum": 0.3}
    v = coin.get("total_volume", 0) or 0
    mc = coin.get("market_cap", 0) or 0
    momentum = coin.get("price_change_percentage_24h", 0) or 0

    vol_score = min(np.log1p(v) / 20.0, 1.0)
    mc_score = min(np.log1p(mc) / 30.0, 1.0)
    mom_score = (momentum + 50) / 100.0
    mom_score = np.clip(mom_score, 0, 1)

    final = weights['volume'] * vol_score + weights['market_cap'] * mc_score + weights['momentum'] * mom_score
    breakdown = {"volume": vol_score, "market_cap": mc_score, "momentum": mom_score}
    return float(final * 100), {k: float(v * 100) for k, v in breakdown.items()}

def score_tokenomics(details: Dict[str, Any]) -> Tuple[float, Dict[str, float]]:
    circ = details.get("market_data", {}).get("circulating_supply") or 0
    total = details.get("market_data", {}).get("total_supply") or 0
    if total and circ:
        circ_ratio = circ / total
    else:
        circ_ratio = 1.0
    circ_score = np.clip(circ_ratio, 0, 1)
    vesting_penalty = 0.0
    final = 0.7 * circ_score + 0.3 * (1 - vesting_penalty)
    breakdown = {"circulating_ratio": circ_score, "vesting_penalty": 1 - vesting_penalty}
    return float(final * 100), {k: float(v * 100) for k, v in breakdown.items()}

def score_social(coin_id: str, details: Dict[str, Any]) -> Tuple[float, Dict[str, float]]:
    cm = details.get("community_data", {})
    tw = cm.get("twitter_followers") or 0
    reddit = cm.get("reddit_subscribers") or 0
    repos = details.get("developer_data", {}).get("forks_count", 0) or 0

    tw_score = min(np.log1p(tw) / 10.0, 1.0)
    reddit_score = min(np.log1p(reddit) / 8.0, 1.0)
    dev_score = min(np.log1p(repos) / 4.0, 1.0)

    final = 0.5 * tw_score + 0.3 * reddit_score + 0.2 * dev_score
    breakdown = {"twitter": tw_score, "reddit": reddit_score, "dev_activity": dev_score}
    return float(final * 100), {k: float(v * 100) for k, v in breakdown.items()}

def aggregate_scores(market_score: float, token_score: float, social_score: float) -> Tuple[str, float, Dict[str, float]]:
    w_market = 0.4
    w_token = 0.35
    w_social = 0.25
    overall = market_score * w_market + token_score * w_token + social_score * w_social
    label = "HOLD"
    if overall > 70:
        label = "BUY"
    elif overall < 40:
        label = "SELL"
    else:
        label = "HOLD"
    breakdown = {"market": market_score, "tokenomics": token_score, "social": social_score}
    return label, float(overall), breakdown

# ------------------------
# Gemini reasoning wrapper
# ------------------------
def _call_gemini_raw(prompt: str, max_tokens: int = 512) -> str:
    """
    Safe wrapper for various versions of google.generativeai SDK.
    - Tries genai.generate_text (common)
    - Then tries GenerativeModel(...).generate_content(...)
    - Returns text string on success, raises Exception on failure.
    """
    if not GEMINI_API_KEY or not GEMINI_SDK_AVAILABLE:
        raise RuntimeError("Gemini SDK not available or GEMINI_API_KEY missing")

    # Attempt 1: modern helper (may exist in some SDK versions)
    try:
        # some SDKs: genai.generate_text(model="gemini-1.5-flash", prompt=prompt, max_output_tokens=max_tokens)
        resp = genai.generate_text(model="gemini-1.5-flash", prompt=prompt, max_output_tokens=max_tokens)
        if isinstance(resp, dict) and resp.get("candidates"):
            text = resp["candidates"][0].get("content") or resp["candidates"][0].get("output")
            if text:
                return text
        # If object-like with .text attr
        if hasattr(resp, "text"):
            return resp.text
    except Exception:
        pass

    # Attempt 2: GenerativeModel interface
    try:
        model_obj = genai.GenerativeModel("gemini-1.5-flash")
        # Some SDKs use generate_content or generate; handle both
        if hasattr(model_obj, "generate_content"):
            out = model_obj.generate_content(prompt)
            if isinstance(out, str):
                return out
            # try attributes
            if hasattr(out, "text"):
                return out.text
            if isinstance(out, dict) and out.get("candidates"):
                return out["candidates"][0].get("content", "")
        elif hasattr(model_obj, "generate"):
            out = model_obj.generate(prompt=prompt, max_output_tokens=max_tokens)
            # adapt to expected structure
            if isinstance(out, dict) and out.get("candidates"):
                return out["candidates"][0].get("content", "")
            if hasattr(out, "candidates") and len(out.candidates) > 0:
                return getattr(out.candidates[0], "content", "")
    except Exception:
        pass

    # If all attempts fail, raise
    raise RuntimeError("Unable to call Gemini via installed SDK. Check gemini SDK version and key.")

def generate_reasoning_with_gemini(coin_id: str, evidence: List[str], overall_score: float) -> str:
    """
    Compose a clear prompt and call Gemini. If Gemini unavailable or call fails,
    fall back to a concise template-based explanation.
    """
    prompt = (
        f"You are a top-tier crypto research analyst. "
        f"Based on the facts below about '{coin_id}', produce:\n"
        "1) A one-line verdict: BUY / HOLD / SELL\n"
        "2) A confidence score (0-100)\n"
        "3) A 3-bullet concise justification (each bullet 1 short sentence)\n"
        "4) Top 3 risks (1 short sentence each)\n\n"
        f"Facts / evidence:\n" + "\n".join([f"- {e}" for e in evidence[:15]]) + f"\n\nOverallScoreEstimate (from model): {overall_score:.1f}%\n\n"
        "Be objective, concise, and cite the most important fact(s) in each bullet. "
        "Answer exactly in plain text."
    )

    # Try to call Gemini
    if GEMINI_API_KEY and GEMINI_SDK_AVAILABLE:
        try:
            raw = _call_gemini_raw(prompt, max_tokens=512)
            # Clean up and return
            return raw.strip()
        except Exception as e:
            # SDK failure - fallback to template
            fallback_reason = f"(Gemini call failed: {e})\n\n"
    else:
        fallback_reason = "(Gemini not configured or SDK missing)\n\n"

    # Fallback templated output
    reasons = "\n".join([f"- {e}" for e in evidence[:5]])
    return fallback_reason + f"Top evidence:\n{reasons}\n\nOverall confidence: {overall_score:.1f}%"

# ------------------------
# Streamlit UI
# ------------------------
def main():
    st.set_page_config(page_title="Crypto Gemini FA Agent", layout="wide")
    st.title("🚀 Crypto Fundamental Analysis Agent — Gemini Flash")

    st.sidebar.header("Settings & Keys")
    per_page = st.sidebar.slider("Top N coins (CoinGecko)", min_value=10, max_value=250, value=100, step=10)
    show_raw = st.sidebar.checkbox("Show raw JSON details", value=False)
    target_mode = st.sidebar.radio("Mode", ["Top Coins (by market cap)", "Trending (CoinGecko)"], index=0)

    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("Coin Universe")
        if target_mode == "Top Coins (by market cap)":
            df_coins = get_top_coins(per_page=per_page)
        else:
            # trending (lightweight)
            trending = coingecko_get("search/trending", params=None).get("coins", [])
            items = [c.get("item", {}) for c in trending][:per_page]
            df_coins = pd.DataFrame(items) if items else pd.DataFrame()

        if df_coins.empty:
            st.warning("CoinGecko fetch failed or returned no data.")
            return

        search = st.text_input("Search coin (name or symbol)")
        if search:
            filtered = df_coins[df_coins['id'].str.contains(search, case=False) | df_coins.get('symbol', '').str.contains(search, case=False) | df_coins.get('name', '').str.contains(search, case=False)]
        else:
            filtered = df_coins

        selected = st.selectbox("Select coin", options=filtered['id'].tolist())

        st.markdown("---")
        st.write("Showing coins from CoinGecko — data cached for a few minutes.")

    with col2:
        if not selected:
            st.info("Select a coin to analyze.")
            return

        st.subheader(f"Analysis: {selected}")
        with st.spinner("Fetching details..."):
            details = get_coin_details(selected)
            market_row = df_coins[df_coins['id'] == selected].iloc[0].to_dict()

        if show_raw:
            st.json(details)

        # Scoring
        m_score, m_break = score_market_features(market_row)
        t_score, t_break = score_tokenomics(details)
        s_score, s_break = score_social(selected, details)

        label, overall, breakdown = aggregate_scores(m_score, t_score, s_score)

        # Evidence collection
        evidence = []
        evidence.append(f"24h change: {market_row.get('price_change_percentage_24h')}%")
        evidence.append(f"7d change: {market_row.get('price_change_percentage_7d_in_currency') or market_row.get('price_change_percentage_7d')}%")
        evidence.append(f"Market cap: ${market_row.get('market_cap'):,.0f}")
        evidence.append(f"24h volume: ${market_row.get('total_volume'):,.0f}")
        evidence.append(f"Twitter followers: {details.get('community_data', {}).get('twitter_followers')}")
        evidence.append(f"Reddit subscribers: {details.get('community_data', {}).get('reddit_subscribers')}")
        evidence.append(f"GitHub forks: {details.get('developer_data', {}).get('forks_count')}")
        circ = details.get("market_data", {}).get("circulating_supply")
        total = details.get("market_data", {}).get("total_supply")
        if circ and total:
            evidence.append(f"Circulating/Total supply: {int(circ):,} / {int(total):,}")
        else:
            evidence.append(f"Circulating supply: {circ}")

        # Render score cards
        st.metric("Decision", label, delta=None)
        st.progress(int(min(overall, 100)))
        st.write(f"**Overall Confidence:** {overall:.1f}%")

        st.write("### Score Breakdown")
        col_a, col_b, col_c = st.columns(3)
        col_a.metric("Market Score", f"{m_score:.1f}%")
        col_b.metric("Tokenomics Score", f"{t_score:.1f}%")
        col_c.metric("Social Score", f"{s_score:.1f}%")

        st.write("#### Component Details")
        st.write(pd.DataFrame({
            'component': ['market_volume', 'market_cap', 'momentum', 'circulating_ratio', 'twitter', 'reddit', 'dev_forks'],
            'value': [m_break.get('volume', 0), m_break.get('market_cap', 0), m_break.get('momentum', 0), t_break.get('circulating_ratio', 0), s_break.get('twitter', 0), s_break.get('reddit', 0), s_break.get('dev_activity', 0)]
        }))

        # LLM reasoning via Gemini
        st.write("### Reasoning & Evidence")
        reasoning = generate_reasoning_with_gemini(selected, evidence, overall)
        # Gemini output may include newlines and bullets; render safely
        st.markdown("```\n" + reasoning + "\n```")

        # Export / Actions
        st.markdown("---")
        st.write("### Actions")
        col1a, col1b = st.columns(2)
        if col1a.button("Save analysis to CSV"):
            out = pd.DataFrame([{ 'id': selected, 'decision': label, 'confidence': overall, 'market_score': m_score, 'tokenomics_score': t_score, 'social_score': s_score }])
            fname = f"analysis_{selected}_{int(time.time())}.csv"
            out.to_csv(fname, index=False)
            st.success(f"Saved CSV to working directory as {fname}.")

        if col1b.button("Add to watchlist (local)"):
            wl = st.session_state.get('watchlist', set())
            wl.add(selected)
            st.session_state['watchlist'] = wl
            st.success(f"Added {selected} to watchlist")

        if st.session_state.get('watchlist'):
            st.write("Watchlist:", list(st.session_state.get('watchlist')))

        st.markdown("---")
        st.write("Prototype notes: This is a starting point. Next steps: add whale flows, vesting schedules, news streaming (LunarCrush/CryptoPanic), and vector DB RAG for whitepapers & tweets.")

if __name__ == '__main__':
    main()
