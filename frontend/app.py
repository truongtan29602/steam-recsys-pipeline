"""Steam RecSys Streamlit Frontend — 4 sections as required by the rubric."""

import os
import random

import requests
import streamlit as st

API_URL = os.environ.get("API_BASE_URL", "http://localhost:8000")

st.set_page_config(
    page_title="Steam RecSys",
    page_icon="🎮",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------
# Sidebar — User selection
# ---------------------------------------------------------------
st.sidebar.title("🎮 Steam RecSys")
st.sidebar.markdown("EPITA AIS3 — Recommendation System")

# Fetch users from API
with st.spinner("Loading users..."):
    try:
        resp = requests.get(f"{API_URL}/users", params={"limit": 200}, timeout=10)
        resp.raise_for_status()
        user_list = resp.json()
    except Exception:
        user_list = []

st.sidebar.markdown("---")
if user_list:
    user_options = [u["user_id"] for u in user_list]
    selected_user = st.sidebar.selectbox("👤 Select a user", user_options)
else:
    selected_user = st.sidebar.text_input("👤 Enter user ID", "anonymous")

st.sidebar.markdown(f"**Logged in as:** {selected_user[:20]}...")

# Health
try:
    health = requests.get(f"{API_URL}/health", timeout=5).json()
    st.sidebar.markdown(f"🟢 API healthy | {health.get('faiss_items', '?')} items")
except Exception:
    st.sidebar.markdown("🔴 API unreachable")

st.sidebar.markdown("---")
st.sidebar.caption("Two-Tower enriched (43 feats) + LightGBM LambdaRank")
st.sidebar.caption("NDCG@10: 1.9% → 30.1%")


# ---------------------------------------------------------------
# Helper: display game cards
# ---------------------------------------------------------------
def display_cards(items, model_label, key_prefix=""):
    """Display a row of game cards."""
    if not items:
        st.warning("No recommendations available.")
        return

    cols = st.columns(min(len(items), 5))
    for i, item in enumerate(items[:20]):
        with cols[i % 5]:
            title = item.get("title", f"Game {item.get('item_id', '?')}")
            category = item.get("category", "Unknown")
            with st.container(border=True):
                st.markdown(f"**{title[:40]}**")
                st.caption(f"📂 {category}")
                if i % 5 == 4 and i < 19:
                    pass  # force new row
        if (i + 1) % 5 == 0 and i + 1 < len(items):
            cols = st.columns(5)

    st.caption(f"ℹ️ Model: {model_label}")


# ---------------------------------------------------------------
# SECTION 1: Guest Homepage — Popularity
# ---------------------------------------------------------------
st.title("🏠 Steam RecSys")
st.markdown("#### Trending on Steam (Popularity)")

with st.spinner("Fetching trending..."):
    try:
        r = requests.get(f"{API_URL}/recommendations/popular", timeout=10)
        r.raise_for_status()
        data = r.json()
        display_cards(data["items"], data["model"], "guest")
        st.caption(f"⏱ {data['latency_ms']:.1f} ms")
    except Exception as e:
        st.warning(f"Trending unavailable: {e}")

# ---------------------------------------------------------------
# SECTION 2: Logged-in Homepage — Personalized
# ---------------------------------------------------------------
st.markdown("---")
st.markdown(f"#### 🎯 For You — Personalized Recommendations")

col1, col2 = st.columns([3, 1])
with col2:
    n_recs = st.slider("Recommendations", 5, 20, 10)

with st.spinner("Generating personalized recommendations..."):
    try:
        r = requests.get(
            f"{API_URL}/recommendations/personalized/{selected_user}",
            params={"k": n_recs}, timeout=15,
        )
        r.raise_for_status()
        data = r.json()
        display_cards(data["items"], data["model"], "personalized")
        st.caption(f"⏱ {data['latency_ms']:.1f} ms")
    except Exception as e:
        st.warning(f"Personalized recommendations unavailable: {e}")

# ---------------------------------------------------------------
# SECTION 3: User History + Because you liked X
# ---------------------------------------------------------------
st.markdown("---")
st.markdown("#### 🕹️ Your Play History & Because You Liked...")

with st.spinner("Fetching play history..."):
    try:
        r = requests.get(f"{API_URL}/users/{selected_user}/history", params={"k": 15}, timeout=10)
        r.raise_for_status()
        hist_data = r.json()
        if hist_data["items"]:
            st.markdown("**Games you played (test period):**")
            display_cards(hist_data["items"], hist_data["model"], "history")

            # Pick the most-played game for "Because you liked X"
            top_game = hist_data["items"][0]
            st.markdown(f"---")
            st.markdown(f"#### 🔗 Because you liked **{top_game['title'][:40]}**")
            with st.spinner("Finding similar games..."):
                r2 = requests.get(
                    f"{API_URL}/recommendations/because/{selected_user}/{top_game['item_id']}",
                    params={"k": 10}, timeout=10,
                )
                r2.raise_for_status()
                because_data = r2.json()
                display_cards(because_data["items"], because_data["model"], "because")
                st.caption(f"⏱ {because_data['latency_ms']:.1f} ms")
    except Exception as e:
        st.warning(f"History unavailable: {e}")

# ---------------------------------------------------------------
# SECTION 4: Item Similarity Search
# ---------------------------------------------------------------
st.markdown("---")
st.markdown("#### 🔍 Similar Items Search")

search_id = st.text_input("Enter a game ID to find similar games:", "10")
col_s1, col_s2 = st.columns([1, 3])
with col_s1:
    search_btn = st.button("🔎 Search", use_container_width=True)
with col_s2:
    n_similar = st.slider("Similar items", 5, 20, 10, key="similar_k")

if search_btn or "similar_results" in st.session_state:
    if search_btn:
        with st.spinner("Searching..."):
            try:
                r = requests.get(f"{API_URL}/items/{search_id}/similar",
                                 params={"k": n_similar}, timeout=10)
                r.raise_for_status()
                st.session_state["similar_results"] = r.json()
            except Exception as e:
                st.error(f"Search failed: {e}")
                st.session_state["similar_results"] = None

    if st.session_state.get("similar_results"):
        data = st.session_state["similar_results"]
        st.markdown(f"**Games similar to `{search_id}`:**")
        display_cards(data["items"], data["model"], "similar")
        st.caption(f"⏱ {data['latency_ms']:.1f} ms")

# ---------------------------------------------------------------
# Footer
# ---------------------------------------------------------------
st.markdown("---")
cols_f = st.columns(4)
cols_f[0].metric("NDCG@10", "30.1%", "+28.2 pts")
cols_f[1].metric("Recall@20", "8.03%", "two-tower")
cols_f[2].metric("Catalog", "11.8K", "items")
cols_f[3].metric("Features", "43+8", "item+user")
st.caption("Malo Fargeas — Task 4 Ranking | EPITA AIS3 2026")

# Force refresh on reload
if "seed" not in st.session_state:
    st.session_state["seed"] = random.randint(0, 10000)
