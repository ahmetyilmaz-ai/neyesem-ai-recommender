from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from .recommender import recommend
from .homepage_recommender import generate_homepage_recommendations


app = FastAPI(
    title="NeYesem AI Recommender",
    description="Cold-start ve content-based yemek öneri API'si",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class RecommendRequest(BaseModel):
    query: str
    limit: int = 10


@app.get("/")
def root():
    return {
        "service": "NeYesem AI Recommender",
        "status": "ok",
        "docs": "/docs",
        "endpoints": {
            "health": "/health",
            "homepage": "/homepage?limit=8",
            "recommend": "/recommend",
        },
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": "neyesem-ai-recommender",
    }


@app.get("/homepage")
def homepage(limit: int = 8):
    return generate_homepage_recommendations(limit=limit)


@app.post("/recommend")
def recommend_items(request: RecommendRequest):
    return recommend(request.query, limit=request.limit)
