from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .homepage_recommender import generate_homepage_recommendations
from .recommender import RecommenderNotReadyError, load_recommender, recommend
from .compare_engine import compare_items
from .db_output_adapter import adapt_homepage_response


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        app.state.recommender_engine = load_recommender()
        app.state.recommender_ready = True
        app.state.recommender_error = None
    except Exception as exc:
        app.state.recommender_engine = None
        app.state.recommender_ready = False
        app.state.recommender_error = str(exc)
    yield
    app.state.recommender_engine = None


app = FastAPI(
    title="NeYesem AI Recommender",
    description="Semantic FAISS recommendation and comparison API",
    version="0.4.0",
    lifespan=lifespan,
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
    hour: Optional[int] = None
    allergies: list[str] = Field(default_factory=list)
    diet: Optional[str] = None


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


class CompareRequest(BaseModel):
    query: str
    limit: int = 10
    hour: Optional[int] = None
    allergies: list[str] = Field(default_factory=list)
    diet: Optional[str] = None
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


def get_loaded_engine(request: Request):
    engine = getattr(request.app.state, "recommender_engine", None)
    if engine is None:
        error = getattr(request.app.state, "recommender_error", None)
        raise HTTPException(
            status_code=503,
            detail={
                "message": "AI recommender is not ready. Run python .\\src\\build_index.py first.",
                "error": error,
            },
        )
    return engine


@app.get("/")
def root():
    return {
        "service": "NeYesem AI Recommender",
        "status": "ok",
        "engine": "sentence-transformer-faiss-lifespan",
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
def health(request: Request):
    return {
        "status": "ok" if getattr(request.app.state, "recommender_ready", False) else "degraded",
        "service": "neyesem-ai-recommender",
        "engine": "sentence-transformer-faiss-lifespan",
        "model_loaded": bool(getattr(request.app.state, "recommender_ready", False)),
        "model_error": getattr(request.app.state, "recommender_error", None),
        "features": [
            "cold_start_homepage",
            "semantic_recommendation",
            "lifespan_model_loading",
            "business_filtering",
            "grouped_platform_prices",
            "db_compatible_output",
        ],
    }


@app.get("/homepage")
def homepage(limit: int = 8):
    return generate_homepage_recommendations(limit=limit)


@app.post("/recommend")
def recommend_items(request: Request, payload: RecommendRequest):
    engine = get_loaded_engine(request)
    try:
        return recommend(
            engine=engine,
            query=payload.query,
            limit=payload.limit,
            hour=payload.hour,
            allergies=payload.allergies,
            diet=payload.diet,
        )
    except RecommenderNotReadyError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.post("/compare")
def compare_recommendations(payload: CompareRequest):
    context = model_to_dict(payload.context)
    user_profile = model_to_dict(payload.user_profile)
    if payload.hour is not None:
        context["hour"] = payload.hour
    if payload.diet is not None:
        user_profile["diet"] = payload.diet
    if payload.allergies:
        user_profile["allergies"] = payload.allergies
    return compare_items(
        query=payload.query,
        limit=payload.limit,
        context=context,
        user_profile=user_profile,
    )


@app.get("/db/homepage")
def db_homepage(limit: int = 8):
    response = generate_homepage_recommendations(limit=limit)
    return adapt_homepage_response(response)


@app.post("/db/recommend")
def db_recommend_items(request: Request, payload: RecommendRequest):
    engine = get_loaded_engine(request)
    try:
        return recommend(
            engine=engine,
            query=payload.query,
            limit=payload.limit,
            hour=payload.hour,
            allergies=payload.allergies,
            diet=payload.diet,
        )
    except RecommenderNotReadyError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.post("/db/compare")
def db_compare_recommendations(payload: CompareRequest):
    return compare_recommendations(payload)
