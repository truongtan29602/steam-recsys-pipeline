"""Portfolio-ready Steam RecSys Streamlit frontend.

The app mirrors the four product surfaces required in the rubric:
guest trending, personalized homepage, item-to-item similarity, and
"because you liked X" recommendations.
"""

import os
import random

import requests
import streamlit as st

API_URL = os.environ.get("API_BASE_URL", "http://localhost:8000")
DEFAULT_IMAGE = "https://cdn.cloudflare.steamstatic.com/steam/apps/10/header.jpg"

st.set_page_config(
    page_title="Steam RecSys",
    page_icon="🎮",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    :root {
        --card-bg: rgba(20, 31, 48, 0.74);
        --accent: #66d9ef;
        --accent-2: #a6e22e;
    }
    .main .block-container { padding-top: 1.4rem; max-width: 1400px; }
    div[data-testid="stMetric"] {
        background: linear-gradient(135deg, rgba(35,48,72,.92), rgba(19,28,44,.92));
        border: 1px solid rgba(102,217,239,.22);
        border-radius: 18px;
        padding: 1rem;
        box-shadow: 0 14px 30px rgba(0,0,0,.18);
    }
    .hero {
        background: radial-gradient(circle at 10% 20%, rgba(102,217,239,.28), transparent 26%),
                    linear-gradient(135deg, #101827 0%, #16223a 48%, #102617 100%);
        border: 1px solid rgba(255,255,255,.08);
        border-radius: 28px;
        padding: 2rem 2.2rem;
        margin-bottom: 1.2rem;
        box-shadow: 0 18px 50px rgba(0,0,0,.24);
    }
    .hero h1 { margin: 0 0 .4rem 0; font-size: 3.1rem; letter-spacing: -.04em; }
    .hero p { color: rgba(255,255,255,.78); font-size: 1.05rem; max-width: 980px; }
    .badge-row { display: flex; flex-wrap: wrap; gap: .5rem; margin-top: 1rem; }
    .badge {
        background: rgba(255,255,255,.09);
        border: 1px solid rgba(255,255,255,.12);
        border-radius: 999px;
        padding: .35rem .7rem;
        font-size: .82rem;
        color: rgba(255,255,255,.9);
    }
    .section-title { margin-top: 1.2rem; margin-bottom: .2rem; }
    .section-subtitle { color: rgba(250,250,250,.65); margin-bottom: .7rem; }
    .game-card {
        border-radius: 18px;
        overflow: hidden;
        border: 1px solid rgba(255,255,255,.10);
        background: var(--card-bg);
        min-height: 238px;
        box-shadow: 0 12px 30px rgba(0,0,0,.20);
        transition: transform .18s ease, border-color .18s ease;
    }
    .game-card:hover { transform: translateY(-3px); border-color: rgba(102,217,239,.55); }
    .game-img {
        height: 92px;
        width: 100%;
        object-fit: cover;
        background: linear-gradient(135deg, #263954, #14251f);
    }
    .game-body { padding: .82rem .9rem .95rem .9rem; }
    .game-title { font-weight: 750; line-height: 1.18; min-height: 2.35rem; }
    .game-meta { color: rgba(255,255,255,.64); font-size: .80rem; margin-top: .45rem; }
    .model-pill {
        display: inline-block;
        margin-top: .65rem;
        color: #041218;
        background: linear-gradient(135deg, var(--accent), var(--accent-2));
        border-radius: 999px;
        padding: .18rem .48rem;
        font-size: .70rem;
        font-weight: 700;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------
# Sidebar — User selection
# ---------------------------------------------------------------
st.sidebar.title("🎮 Steam RecSys")
st.sidebar.markdown("**EPITA AIS3** · real-time recommendations")

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

st.sidebar.markdown(f"**Logged in as:** `{selected_user[:20]}...`")

# Health
try:
    health = requests.get(f"{API_URL}/health", timeout=5).json()
    st.sidebar.markdown(f"🟢 API healthy | {health.get('faiss_items', '?')} items")
except Exception:
    st.sidebar.markdown("🔴 API unreachable")

st.sidebar.markdown("---")
st.sidebar.caption("Retrieval: enriched Two-Tower + FAISS")
st.sidebar.caption("Ranking: LightGBM LambdaRank")
st.sidebar.caption("Observed NDCG@10: 1.9% → 30.1%")


# ---------------------------------------------------------------
# Helper: display game cards
# ---------------------------------------------------------------
def _safe_image(url: str | None) -> str:
    """Return a usable image URL for cards."""
    if not url or url.lower() in {"nan", "none", ""}:
        return DEFAULT_IMAGE
    return url


def display_cards(items, model_label, key_prefix="", max_cards=10):
    """Display responsive, metadata-rich game cards."""
    if not items:
        st.info("No recommendations available for this section yet.")
        return

    for start in range(0, min(len(items), max_cards), 5):
        cols = st.columns(5)
        for offset, item in enumerate(items[start:start + 5]):
            i = start + offset
            with cols[offset]:
                title = item.get("title", f"Game {item.get('item_id', '?')}")
                category = item.get("category", "Unknown")
                item_id = item.get("item_id", "?")
                image_url = _safe_image(item.get("image_url"))
                st.markdown(
                    f"""
                    <div class="game-card">
                        <img class="game-img" src="{image_url}" alt="{title}">
                        <div class="game-body">
                            <div class="game-title">{i + 1}. {title[:54]}</div>
                            <div class="game-meta">🎮 Steam ID: <code>{item_id}</code></div>
                            <div class="game-meta">📂 {category[:42]}</div>
                            <span class="model-pill">{model_label}</span>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    st.caption(f"ℹ️ Served by: {model_label}")


def section_header(icon: str, title: str, subtitle: str):
    st.markdown(f"<h2 class='section-title'>{icon} {title}</h2>", unsafe_allow_html=True)
    st.markdown(f"<div class='section-subtitle'>{subtitle}</div>", unsafe_allow_html=True)


# ---------------------------------------------------------------
# SECTION 1: Guest Homepage — Popularity
# ---------------------------------------------------------------
st.markdown(
    """
    <div class="hero">
        <h1>Steam Recommendation System</h1>
        <p>
            A production-style demo with PostgreSQL-backed user history, FastAPI model serving,
            FAISS retrieval, LightGBM re-ranking, and a Streamlit product interface.
        </p>
        <div class="badge-row">
            <span class="badge">Popularity for guests</span>
            <span class="badge">Two-Tower personalization</span>
            <span class="badge">LightGBM LambdaRank</span>
            <span class="badge">Item embeddings for similarity</span>
            <span class="badge">Docker Compose ready</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

top_metrics = st.columns(4)
top_metrics[0].metric("Retrieval Recall@20", "8.03%", "Two-Tower test")
top_metrics[1].metric("Ranking NDCG@10", "30.11%", "+28.21 pts")
top_metrics[2].metric("Catalog", "11.8K", "Steam games")
top_metrics[3].metric("Serving", "FastAPI", "REST + Swagger")

section_header("🔥", "Trending on Steam", "Guest homepage powered by training-period popularity.")

with st.spinner("Fetching trending..."):
    try:
        r = requests.get(f"{API_URL}/recommendations/popular", timeout=10)
        r.raise_for_status()
        data = r.json()
        display_cards(data["items"], data["model"], "guest")
        st.caption(f"⏱ API latency: {data['latency_ms']:.1f} ms")
    except Exception as e:
        st.warning(f"Trending unavailable: {e}")

# ---------------------------------------------------------------
# SECTION 2: Logged-in Homepage — Personalized
# ---------------------------------------------------------------
st.markdown("---")
section_header("🎯", "For You", "Logged-in homepage using Two-Tower retrieval followed by LightGBM re-ranking.")

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
        st.caption(f"⏱ API latency: {data['latency_ms']:.1f} ms")
    except Exception as e:
        st.warning(f"Personalized recommendations unavailable: {e}")

# ---------------------------------------------------------------
# SECTION 3: User History + Because you liked X
# ---------------------------------------------------------------
st.markdown("---")
section_header("🕹️", "Your Play History", "Recent test-period interactions simulate live deployment context.")

with st.spinner("Fetching play history..."):
    try:
        r = requests.get(f"{API_URL}/users/{selected_user}/history", params={"k": 15}, timeout=10)
        r.raise_for_status()
        hist_data = r.json()
        if hist_data["items"]:
            st.markdown("**Games you played during the held-out period:**")
            display_cards(hist_data["items"], hist_data["model"], "history", max_cards=5)

            # Pick the most-played game for "Because you liked X"
            top_game = hist_data["items"][0]
            st.markdown(f"---")
            section_header("🔗", f"Because you liked {top_game['title'][:40]}", "Nearest neighbors from the same learned item embedding space.")
            with st.spinner("Finding similar games..."):
                r2 = requests.get(
                    f"{API_URL}/recommendations/because/{selected_user}/{top_game['item_id']}",
                    params={"k": 10}, timeout=10,
                )
                r2.raise_for_status()
                because_data = r2.json()
                display_cards(because_data["items"], because_data["model"], "because")
                st.caption(f"⏱ API latency: {because_data['latency_ms']:.1f} ms")
        else:
            st.info("This user has no held-out history. Personalized recommendations will fall back gracefully.")
    except Exception as e:
        st.warning(f"History unavailable: {e}")

# ---------------------------------------------------------------
# SECTION 4: Item Similarity Search
# ---------------------------------------------------------------
st.markdown("---")
section_header("🔍", "Similar Items Search", "Item page experience: query a Steam app ID and retrieve nearest games by cosine similarity.")

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
        st.caption(f"⏱ API latency: {data['latency_ms']:.1f} ms")

# ---------------------------------------------------------------
# Footer
# ---------------------------------------------------------------
st.markdown("---")
cols_f = st.columns(4)
cols_f[0].metric("NDCG@10", "30.1%", "+28.2 pts")
cols_f[1].metric("Recall@20", "8.03%", "Two-Tower")
cols_f[2].metric("Coverage", "3.3%", "ranked top-10")
cols_f[3].metric("Features", "43+", "ranking/retrieval")
st.caption("Steam RecSys Team — Tasks 1–6 sync branch | EPITA AIS3 2026")

# Force refresh on reload
if "seed" not in st.session_state:
    st.session_state["seed"] = random.randint(0, 10000)
