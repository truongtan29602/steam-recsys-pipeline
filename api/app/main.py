from fastapi import FastAPI
from pydantic import BaseModel

class RecommendationSection(BaseModel):
    section: str
    model: str
    status: str
    items: list[dict]

app = FastAPI(title="Steam Recommender API", description="Setup-only API skeleton.", version="0.1.0")

ITEMS = [
    {"id": "app_730", "title": "Counter-Strike 2", "category": "Action"},
    {"id": "app_570", "title": "Dota 2", "category": "MOBA"},
    {"id": "app_440", "title": "Team Fortress 2", "category": "Action"},
]

@app.get("/health")
def health():
    return {"status": "ok", "mode": "setup-only"}

@app.get("/recommendations/popular", response_model=RecommendationSection)
def popular():
    return RecommendationSection(section="Homepage guest", model="Popularity placeholder", status="Not implemented yet", items=ITEMS)

@app.get("/recommendations/personalized/{user_id}", response_model=RecommendationSection)
def personalized(user_id: str):
    return RecommendationSection(section="Homepage logged in", model="Two-tower + LightGBM placeholder", status="Not implemented yet", items=ITEMS)

@app.get("/items/{item_id}/similar", response_model=RecommendationSection)
def similar_items(item_id: str):
    return RecommendationSection(section="Similar items", model="Embedding cosine placeholder", status="Not implemented yet", items=ITEMS)
