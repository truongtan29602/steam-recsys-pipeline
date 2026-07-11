"""Portfolio-ready Steam RecSys Streamlit frontend.

The app mirrors the four product surfaces required in the rubric:
guest trending, personalized homepage, item-to-item similarity, and
"because you liked X" recommendations.
"""

import os
import random
from html import escape

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
        --page-bg: #08111f;
        --panel-bg: rgba(15, 25, 42, 0.94);
        --card-bg: rgba(255, 255, 255, 0.96);
        --card-text: #102033;
        --muted-text: #53657c;
        --accent: #4cc9f0;
        --accent-2: #80ffdb;
        --warning: #ffd166;
    }
    .stApp {
        color: #f8fbff;
        background:
            radial-gradient(circle at 8% 5%, rgba(76, 201, 240, 0.24), transparent 28rem),
            radial-gradient(circle at 92% 12%, rgba(128, 255, 219, 0.18), transparent 26rem),
            linear-gradient(135deg, #07101f 0%, #0c1628 48%, #111827 100%);
    }
    .main .block-container {
        padding-top: 1.6rem;
        padding-bottom: 2.5rem;
        max-width: 1440px;
    }
    h1, h2, h3, h4, h5, h6, p, li, label, span, div {
        text-rendering: optimizeLegibility;
    }
    div[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #07111f 0%, #101a2c 100%);
        border-right: 1px solid rgba(255,255,255,.08);
    }
    div[data-testid="stSidebar"] * {
        color: #eef6ff !important;
    }
    div[data-testid="stSidebar"] .stCaption,
    div[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
        color: rgba(238,246,255,.78) !important;
    }
    .stMarkdown, .stCaption, .stSlider, .stTextInput, .stSelectbox {
        color: #f8fbff;
    }
    div[data-testid="stAlert"] {
        border-radius: 16px;
        border: 1px solid rgba(255,255,255,.12);
    }
    div[data-testid="stMetric"] {
        background: linear-gradient(145deg, rgba(255,255,255,.98), rgba(224,241,255,.94));
        border: 1px solid rgba(76,201,240,.34);
        border-radius: 22px;
        padding: 1.05rem;
        box-shadow: 0 18px 42px rgba(0,0,0,.26);
    }
    div[data-testid="stMetric"] * {
        color: #102033 !important;
    }
    .hero {
        position: relative;
        overflow: hidden;
        background:
            linear-gradient(120deg, rgba(9, 20, 38, .92), rgba(13, 36, 55, .90)),
            radial-gradient(circle at 18% 20%, rgba(76,201,240,.38), transparent 26%),
            radial-gradient(circle at 85% 35%, rgba(128,255,219,.32), transparent 32%);
        border: 1px solid rgba(255,255,255,.16);
        border-radius: 32px;
        padding: 2.35rem 2.45rem;
        margin-bottom: 1.35rem;
        box-shadow: 0 24px 70px rgba(0,0,0,.34);
    }
    .hero h1 {
        margin: 0 0 .55rem 0;
        font-size: clamp(2.4rem, 5vw, 4.2rem);
        line-height: .95;
        letter-spacing: -.055em;
        color: #ffffff;
        text-shadow: 0 5px 28px rgba(0,0,0,.45);
    }
    .hero p {
        color: rgba(255,255,255,.88);
        font-size: 1.12rem;
        max-width: 980px;
        line-height: 1.65;
    }
    .badge-row { display: flex; flex-wrap: wrap; gap: .5rem; margin-top: 1rem; }
    .badge {
        background: rgba(255,255,255,.14);
        border: 1px solid rgba(255,255,255,.24);
        border-radius: 999px;
        padding: .42rem .78rem;
        font-size: .84rem;
        font-weight: 700;
        color: #ffffff;
        backdrop-filter: blur(10px);
    }
    .section-title {
        margin-top: 1.35rem;
        margin-bottom: .25rem;
        color: #ffffff;
        letter-spacing: -.025em;
    }
    .section-subtitle {
        color: rgba(232,241,255,.82);
        margin-bottom: .95rem;
        font-size: 1.02rem;
    }
    .game-card {
        border-radius: 18px;
        overflow: hidden;
        border: 1px solid rgba(255,255,255,.70);
        background: var(--card-bg);
        min-height: 260px;
        box-shadow: 0 18px 38px rgba(0,0,0,.30);
        transition: transform .18s ease, border-color .18s ease, box-shadow .18s ease;
    }
    .game-card:hover {
        transform: translateY(-5px);
        border-color: rgba(76,201,240,.95);
        box-shadow: 0 24px 52px rgba(0,0,0,.38);
    }
    .game-img {
        height: 104px;
        width: 100%;
        object-fit: cover;
        background: linear-gradient(135deg, #1d3557, #0b132b);
    }
    .game-body { padding: .95rem 1rem 1.05rem 1rem; }
    .game-title {
        color: var(--card-text);
        font-weight: 850;
        line-height: 1.22;
        min-height: 3rem;
        font-size: .98rem;
    }
    .game-meta {
        color: var(--muted-text);
        font-size: .80rem;
        margin-top: .48rem;
        line-height: 1.35;
    }
    .game-meta code {
        color: #0b4f6c;
        background: rgba(76,201,240,.14);
        border-radius: 6px;
        padding: .05rem .24rem;
    }
    .model-pill {
        display: inline-block;
        margin-top: .65rem;
        color: #06111d;
        background: linear-gradient(135deg, var(--accent), var(--accent-2));
        border-radius: 999px;
        padding: .24rem .56rem;
        font-size: .70rem;
        font-weight: 850;
        box-shadow: 0 8px 18px rgba(76,201,240,.22);
    }
    div[data-testid="stTextInput"] input,
    div[data-testid="stSelectbox"] div[data-baseweb="select"] > div {
        color: #102033 !important;
        background: rgba(255,255,255,.98) !important;
        border-radius: 14px !important;
    }
    .stButton > button {
        border: 0;
        border-radius: 14px;
        background: linear-gradient(135deg, #4cc9f0, #80ffdb);
        color: #06111d;
        font-weight: 850;
        box-shadow: 0 14px 28px rgba(76,201,240,.22);
    }
    .stButton > button:hover {
        color: #06111d;
        transform: translateY(-1px);
        box-shadow: 0 18px 36px rgba(76,201,240,.30);
    }
    hr {
        border-color: rgba(255,255,255,.14) !important;
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
                title_text = escape(str(title)[:58])
                category_text = escape(str(category)[:46])
                item_id_text = escape(str(item_id))
                image_url_text = escape(str(image_url), quote=True)
                model_label_text = escape(str(model_label))
                st.markdown(
                    f"""
                    <div class="game-card">
                        <img class="game-img" src="{image_url_text}" alt="{title_text}">
                        <div class="game-body">
                            <div class="game-title">{i + 1}. {title_text}</div>
                            <div class="game-meta">🎮 Steam ID: <code>{item_id_text}</code></div>
                            <div class="game-meta">📂 {category_text}</div>
                            <span class="model-pill">{model_label_text}</span>
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
