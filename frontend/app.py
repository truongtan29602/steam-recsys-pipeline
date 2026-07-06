import os
import requests
import streamlit as st

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")

st.set_page_config(page_title="Steam Recommender", page_icon="🎮", layout="wide")

def api_get(path):
    try:
        r = requests.get(f"{API_BASE_URL}{path}", timeout=3)
        r.raise_for_status()
        return r.json()
    except Exception:
        return None

def hero():
    st.markdown("""
<div style="padding:2rem;border-radius:24px;background:linear-gradient(135deg,#1b2838,#2a475e,#66c0f4);">
<h1>🎮 Steam Recommender System</h1>
<p style="font-size:1.1rem;">Production-style demo shell for Steam recommendations.</p>
<p>Setup only: models, ranking, FAISS, and DB loading come later.</p>
</div>
""", unsafe_allow_html=True)

def card(title, category, model):
    st.markdown(f"""
<div style="border:1px solid rgba(102,192,244,.3);border-radius:18px;padding:1rem;min-height:165px;background:#1b2838;">
<div style="font-si
import>🎮</div>
<h3>{title}</h3>
<p style="color:#acdbf5;">{category}</p>
<small>Model: {model}</small>
</div>
""", unsafe_allow_html=True)

def render_cards(payload, fallback_model):
    items = (payload or {}).get("items") or [
        {"title":"Placeholder Game A", "category":"Action"},
        {"title":"Placeholder Game B", "category":"Adventure"},
        {"title":"Placeholder Game C", "category":"Strategy"},
    ]
    model = (payload or {}).get("model", fallback_model)
    cols = st.columns(3)
    for col, item in zip(cols, items):
        with col:
            card(item.get("title", "Unknown"), item.get("category", "TBD"), model)

hero()
with st.sidebar:
    st.title("Navigation")
    page = st.radio("Choose a section", ["Homepage: Guest", "Homepage: Logged in", "Item page: Similar items", "Because you liked X", "System status"])
    st.caption(f"API: {API_BASE_URL}")

if page == "Homepage: Guest":
    st.subheader("Popular right now")
    st.caption("Unknown users fall back to popularity recommendations.")
    render_cards(api_get("/recommendations/popular"), "Popularity placeholder")
elif page == "Homepage: Logged in":
    user_id = st.text_input("User ID", "demo_user")
    st.subheader("Personalized for you")
    st.caption("Two-tower retrieval + LightGBM ranking placeholder.")
    render_cards(api_get(f"/recommendations/personalized/{user_id}"), "Two-tower + LightGBM placeholder")
elif page == "Item page: Similar items":
    item_id = st.text_input("Item ID", "app_730")
    st.subheader("Similar games")
    st.caption("Cosine similarity over learned item embeddings placeholder.")
    render_cards(api_get(f"/items/{item_id}/similar"), "Embedding cosine placeholder")
elif page == "Because you liked X":
    liked = st.text_input("Liked game", "Portal 2")
    st.subheader(f"Because you liked {liked}")
    st.caption("This row will use the same item embeddings later.")
    render_cards(None, "Same embeddings placeholder")
else:
    st.subheader("System status")
    health = api_get("/health")
    if health:
        st.success("API reachable")
        st.json(health)
    else:
        st.warning("API is not reachable yet")
    st.info("PostgreSQL and pgAdmin are included in Docker Compose.")
