from typing import Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .homepage_recommender import generate_homepage_recommendations
from .semantic_recommender import semantic_recommend
from .compare_engine import compare_items
from .db_output_adapter import adapt_homepage_response, adapt_recommend_response


app = FastAPI(
    title="NeYesem AI Recommender",
    description="Semantic search, comparison, cold-start and content-based food recommendation API",
    version="0.3.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class RecommendationContext(BaseModel):
    hour: Optional[int] = None
    day_type: Optional[str] = None
    city: Optional[str] = None


class UserProfile(BaseModel):
    diet: Optional[str] = None
    allergies: list[str] = Field(default_factory=list)
    disliked_items: list[str] = Field(default_factory=list)
    preferred_categories: list[str] = Field(default_factory=list)
    max_budget: Optional[float] = None


class RecommendRequest(BaseModel):
    query: str
    limit: int = 10
    context: Optional[RecommendationContext] = None
    user_profile: Optional[UserProfile] = None


def model_to_dict(value):
    if value is None:
        return {}
    if hasattr(value, "model_dump"):
        return value.model_dump()
    if hasattr(value, "dict"):
        return value.dict()
    return dict(value)


@app.get("/")
def root():
    return {
        "service": "NeYesem AI Recommender",
        "status": "ok",
        "engine": "sentence-transformer-faiss",
        "docs": "/docs",
        "endpoints": {
            "health": "/health",
            "homepage": "/homepage?limit=8",
            "recommend": "/recommend",
            "compare": "/compare",
            "db_homepage": "/db/homepage?limit=8",
            "db_recommend": "/db/recommend",
            "db_compare": "/db/compare",
        },
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": "neyesem-ai-recommender",
        "engine": "sentence-transformer-faiss",
        "features": [
            "cold_start_homepage",
            "semantic_recommendation",
            "price_comparison",
            "db_compatible_output",
        ],
    }


@app.get("/homepage")
def homepage(limit: int = 8):
    return generate_homepage_recommendations(limit=limit)


@app.post("/recommend")
def recommend_items(request: RecommendRequest):
    return semantic_recommend(
        query=request.query,
        limit=request.limit,
        context=model_to_dict(request.context),
        user_profile=model_to_dict(request.user_profile),
    )


@app.post("/compare")
def compare_recommendations(request: RecommendRequest):
    return compare_items(
        query=request.query,
        limit=request.limit,
        context=model_to_dict(request.context),
        user_profile=model_to_dict(request.user_profile),
    )


@app.get("/db/homepage")
def db_homepage(limit: int = 8):
    response = generate_homepage_recommendations(limit=limit)
    return adapt_homepage_response(response)


@app.post("/db/recommend")
def db_recommend_items(request: RecommendRequest):
    response = semantic_recommend(
        query=request.query,
        limit=request.limit,
        context=model_to_dict(request.context),
        user_profile=model_to_dict(request.user_profile),
    )
    return adapt_recommend_response(response)


@app.post("/db/compare")
def db_compare_recommendations(request: RecommendRequest):
    return compare_items(
        query=request.query,
        limit=request.limit,
        context=model_to_dict(request.context),
        user_profile=model_to_dict(request.user_profile),
    )
