import json
import re
from pathlib import Path
from typing import Any

import numpy as np
from sentence_transformers import SentenceTransformer


ROOT_DIR = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = ROOT_DIR / "artifacts"

METADATA_PATH = ARTIFACTS_DIR / "item_metadata.json"
EMBEDDINGS_PATH = ARTIFACTS_DIR / "item_embeddings.npy"
FAISS_INDEX_PATH = ARTIFACTS_DIR / "faiss_index.bin"
CONFIG_PATH = ARTIFACTS_DIR / "semantic_config.json"

DEFAULT_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


CATEGORY_KEYWORDS = {
    "pizza": ["pizza"],
    "burger": ["burger", "hamburger"],
    "döner": ["döner", "doner"],
    "pide": ["pide", "lahmacun"],
    "tatlı": ["tatlı", "tatli", "waffle", "pasta", "baklava", "dondurma", "sütlaç", "sutlac", "kazandibi", "magnolia"],
    "tavuk": ["tavuk", "chicken", "kanat"],
    "içecek": ["su", "kola", "ayran", "ice tea", "fanta", "sprite", "limonata", "pepsi"],
    "sağlıklı": ["salata", "fit", "bowl", "ızgara", "izgara"],
    "kebap": ["kebap", "kebab", "adana", "urfa", "şiş", "sis"],
    "çorba": ["çorba", "corba", "soup"],
}

FILLING_WORDS = [
    "menü",
    "menu",
    "döner",
    "doner",
    "burger",
    "pizza",
    "pide",
    "lahmacun",
    "tavuk",
    "köfte",
    "kofte",
    "dürüm",
    "durum",
    "tantuni",
    "kebap",
    "kebab",
    "taco",
]

HEAVY_MORNING_WORDS = [
    "döner",
    "doner",
    "kebap",
    "kebab",
    "lahmacun",
    "burger",
    "pizza",
    "tantuni",
    "kanat",
]

MORNING_FRIENDLY_WORDS = [
    "kahvaltı",
    "kahvalti",
    "tost",
    "sandviç",
    "sandvic",
    "çorba",
    "corba",
    "salata",
    "bowl",
    "sütlaç",
    "sutlac",
]

ALLERGY_KEYWORDS = {
    "laktoz": ["süt", "sut", "peynir", "yoğurt", "yogurt", "ayran", "dondurma", "krema", "milk", "cheese"],
    "gluten": ["ekmek", "pide", "lahmacun", "pizza", "burger", "makarna", "waffle", "pasta", "börek", "borek"],
    "soya": ["soya", "soy"],
    "yer fıstığı": ["yer fıstığı", "yer fistigi", "fıstık", "fistik", "peanut"],
}

MEAT_WORDS = [
    "et",
    "dana",
    "tavuk",
    "chicken",
    "köfte",
    "kofte",
    "döner",
    "doner",
    "kebap",
    "kebab",
    "sucuk",
    "kanat",
    "tantuni",
]


_ENGINE = None


def normalize_text(value: Any) -> str:
    value = str(value or "").lower().strip()

    replacements = {
        "ı": "i",
        "ğ": "g",
        "ü": "u",
        "ş": "s",
        "ö": "o",
        "ç": "c",
    }

    for source, target in replacements.items():
        value = value.replace(source, target)

    value = re.sub(r"[^a-z0-9\s]", " ", value)
    value = re.sub(r"\s+", " ", value)

    return value.strip()


def has_any(text: str, words: list[str]) -> bool:
    normalized = normalize_text(text)
    tokens = set(normalized.split())

    for word in words:
        normalized_word = normalize_text(word)

        if not normalized_word:
            continue

        if len(normalized_word) <= 3:
            if normalized_word in tokens:
                return True
        else:
            if normalized_word in normalized:
                return True

    return False


def infer_intent(query: str) -> dict[str, Any]:
    normalized = normalize_text(query)

    categories = []

    for category, keywords in CATEGORY_KEYWORDS.items():
        if has_any(normalized, keywords):
            categories.append(category)

    budget_max = None

    numbers = re.findall(r"\d+", normalized)
    if numbers:
        possible_budget = max(int(number) for number in numbers)
        if 50 <= possible_budget <= 3000:
            budget_max = possible_budget

    if budget_max is None and has_any(normalized, ["ucuz", "uygun", "bütçe", "butce", "pahalı olmasın", "pahali olmasin"]):
        budget_max = 250

    if has_any(normalized, ["spor", "protein", "proteinli", "yüksek protein", "yuksek protein"]):
        preference = "protein"
    elif has_any(normalized, ["çok aç", "cok ac", "açım", "acim", "doyurucu", "büyük", "buyuk"]):
        preference = "filling"
    elif has_any(normalized, ["ucuz", "uygun", "pahalı olmasın", "pahali olmasin"]):
        preference = "cheap"
    elif has_any(normalized, ["sağlıklı", "saglikli", "hafif", "fit"]):
        preference = "healthy"
    elif has_any(normalized, ["tatlı", "tatli", "sweet"]):
        preference = "sweet"
        if "tatlı" not in categories:
            categories.append("tatlı")
    else:
        preference = "balanced"

    return {
        "raw_query": query,
        "normalized_query": normalized,
        "categories": categories,
        "budget_max": budget_max,
        "preference": preference,
    }


def safe_float(value: Any) -> float | None:
    if value is None:
        return None

    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def model_to_plain_dict(value: Any) -> dict[str, Any]:
    if value is None:
        return {}

    if isinstance(value, dict):
        return value

    if hasattr(value, "model_dump"):
        return value.model_dump()

    if hasattr(value, "dict"):
        return value.dict()

    return {}


class SemanticRecommendationEngine:
    def __init__(self):
        if not METADATA_PATH.exists() or not EMBEDDINGS_PATH.exists():
            raise FileNotFoundError(
                "Semantic artifacts bulunamadı. Önce şu komutu çalıştır: python .\\src\\build_index.py"
            )

        config = {}

        if CONFIG_PATH.exists():
            with CONFIG_PATH.open("r", encoding="utf-8") as file:
                config = json.load(file)

        self.model_name = config.get("model_name", DEFAULT_MODEL_NAME)

        print(f"Semantic model yükleniyor: {self.model_name}")
        self.model = SentenceTransformer(self.model_name)

        with METADATA_PATH.open("r", encoding="utf-8") as file:
            self.items = json.load(file)

        self.embeddings = np.load(EMBEDDINGS_PATH).astype("float32")

        self.faiss_index = None
        self.faiss_available = False

        if FAISS_INDEX_PATH.exists():
            try:
                import faiss

                with FAISS_INDEX_PATH.open("rb") as file:
                    index_bytes = np.frombuffer(file.read(), dtype="uint8")
                self.faiss_index = faiss.deserialize_index(index_bytes)
                self.faiss_available = True
                print("FAISS index RAM'e yüklendi.")
            except Exception as exc:
                print(f"FAISS yüklenemedi, NumPy fallback kullanılacak: {exc}")

        print(f"Semantic recommender hazır. Ürün sayısı: {len(self.items)}")

    def search(self, query: str, top_k: int) -> list[tuple[dict[str, Any], float]]:
        query_embedding = self.model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype("float32")

        top_k = min(top_k, len(self.items))

        if self.faiss_available and self.faiss_index is not None:
            scores, indices = self.faiss_index.search(query_embedding, top_k)

            results = []

            for index, score in zip(indices[0], scores[0]):
                if index < 0:
                    continue

                results.append((self.items[int(index)], float(score)))

            return results

        scores = self.embeddings @ query_embedding[0]
        best_indices = np.argsort(scores)[::-1][:top_k]

        return [
            (self.items[int(index)], float(scores[int(index)]))
            for index in best_indices
        ]


def get_engine() -> SemanticRecommendationEngine:
    global _ENGINE

    if _ENGINE is None:
        _ENGINE = SemanticRecommendationEngine()

    return _ENGINE


def should_filter_item(item: dict[str, Any], intent: dict[str, Any], context: dict[str, Any], user_profile: dict[str, Any]) -> bool:
    item_name = item.get("item_name") or ""
    item_text = item.get("item_text") or normalize_text(item_name)
    category = normalize_text(item.get("category"))
    price = safe_float(item.get("price"))

    disliked_items = user_profile.get("disliked_items") or []
    for disliked in disliked_items:
        if normalize_text(disliked) and normalize_text(disliked) in item_text:
            return True

    allergies = user_profile.get("allergies") or []
    for allergy in allergies:
        allergy_key = normalize_text(allergy)
        keywords = ALLERGY_KEYWORDS.get(allergy_key, [])

        if keywords and has_any(item_text, keywords):
            return True

    diet = normalize_text(user_profile.get("diet"))

    if diet in ["vegan", "vejetaryen", "vegetarian"]:
        if has_any(item_text, MEAT_WORDS):
            return True

    # Protein / spor odaklı sorgularda sos, tatlı, içecek ve yan ürünleri temizle.
    if intent.get("preference") == "protein" or has_any(intent.get("normalized_query", ""), ["protein", "spor", "proteinli"]):
        protein_exclude_words = [
            "sos",
            "ketçap",
            "ketcap",
            "mayonez",
            "brownie",
            "browni",
            "waffle",
            "dondurma",
            "baklava",
            "tatlı",
            "tatli",
            "kola",
            "ayran",
            "pepsi",
            "fanta",
            "sprite",
            "ice tea",
            "fuse tea",
        ]

        protein_positive_words = [
            "tavuk",
            "chicken",
            "et",
            "dana",
            "köfte",
            "kofte",
            "döner",
            "doner",
            "protein",
            "pilav",
            "ızgara",
            "izgara",
            "burger",
        ]

        if has_any(item_text, protein_exclude_words):
            return True

        # Yeterince protein çağrışımı yoksa protein sorgusunda ürünü ele.
        if not has_any(item_text, protein_positive_words):
            return True

    max_budget = user_profile.get("max_budget") or intent.get("budget_max")
    max_budget = safe_float(max_budget)

    if max_budget is not None and price is not None:
        if price > max_budget * 1.35:
            return True

    city = normalize_text(context.get("city"))

    if city:
        item_city = normalize_text(item.get("city"))
        if item_city and item_city != city:
            return True

    return False


def calculate_context_score(item: dict[str, Any], intent: dict[str, Any], context: dict[str, Any], user_profile: dict[str, Any]) -> float:
    score = 0.0

    item_text = item.get("item_text") or normalize_text(item.get("item_name"))
    category = normalize_text(item.get("category"))
    price = safe_float(item.get("price"))
    discount = safe_float(item.get("discount_rate"))

    preference = intent.get("preference")
    budget_max = safe_float(user_profile.get("max_budget") or intent.get("budget_max"))

    if budget_max is not None and price is not None:
        if price <= budget_max:
            score += 0.18

    if price is not None:
        if preference == "cheap":
            score += max(0.0, 1.0 - min(price, 600) / 600) * 0.20
        elif preference == "balanced":
            score += max(0.0, 1.0 - min(price, 900) / 900) * 0.10
        elif preference == "filling":
            if 80 <= price <= 450:
                score += 0.16
            elif price < 80:
                score -= 0.12

    if discount is not None and discount > 0:
        score += min(discount, 40) / 100 * 0.16

    if preference == "filling":
        if has_any(item_text, FILLING_WORDS):
            score += 0.18

    if preference == "healthy":
        if category in ["saglikli", "salata"] or has_any(item_text, CATEGORY_KEYWORDS["sağlıklı"]):
            score += 0.18

    if preference == "protein":
        if has_any(item_text, ["tavuk", "chicken", "et", "köfte", "kofte", "protein", "ızgara", "izgara"]):
            score += 0.20

    preferred_categories = user_profile.get("preferred_categories") or []
    for preferred in preferred_categories:
        if normalize_text(preferred) == category or has_any(item_text, [preferred]):
            score += 0.12

    hour = context.get("hour")

    try:
        hour = int(hour) if hour is not None else None
    except (TypeError, ValueError):
        hour = None

    if hour is not None:
        if 5 <= hour <= 10:
            if has_any(item_text, HEAVY_MORNING_WORDS):
                score -= 0.25
            if has_any(item_text, MORNING_FRIENDLY_WORDS):
                score += 0.16
        elif 11 <= hour <= 14:
            if has_any(item_text, FILLING_WORDS):
                score += 0.08
        elif 18 <= hour <= 23:
            if has_any(item_text, FILLING_WORDS):
                score += 0.10

    day_type = normalize_text(context.get("day_type"))

    if day_type == "weekend":
        if category in ["tatli", "pizza", "burger"]:
            score += 0.05

    return score


def build_reason(item: dict[str, Any], intent: dict[str, Any], semantic_score: float, context_score: float) -> str:
    parts = []

    categories = intent.get("categories") or []
    preference = intent.get("preference")

    if categories:
        parts.append(f"{', '.join(categories)} isteğine anlamsal olarak yakın")

    if preference == "filling":
        parts.append("doyurucu tercihine uygun")
    elif preference == "cheap":
        parts.append("fiyat odaklı seçildi")
    elif preference == "healthy":
        parts.append("hafif/sağlıklı tercihle uyumlu")
    elif preference == "protein":
        parts.append("protein odaklı tercihle uyumlu")

    price = safe_float(item.get("price"))

    if price is not None:
        parts.append(f"{price:.2f} TL")

    discount = safe_float(item.get("discount_rate"))

    if discount is not None and discount > 0:
        parts.append(f"%{discount:.1f} indirim")

    parts.append(f"semantik skor {semantic_score:.2f}")

    if context_score != 0:
        parts.append(f"bağlam skoru {context_score:.2f}")

    return ", ".join(parts) + "."


def semantic_recommend(
    query: str,
    limit: int = 10,
    context: dict[str, Any] | None = None,
    user_profile: dict[str, Any] | None = None,
) -> dict[str, Any]:
    context = context or {}
    user_profile = user_profile or {}

    try:
        engine = get_engine()
    except Exception as exc:
        try:
            from .recommender import recommend as legacy_recommend
        except Exception:
            from recommender import recommend as legacy_recommend

        fallback = legacy_recommend(query, limit=limit)
        fallback["engine"] = "tfidf_fallback"
        fallback["fallback_reason"] = str(exc)
        return fallback

    intent = infer_intent(query)

    search_text = query

    if intent.get("categories"):
        search_text += " " + " ".join(intent["categories"])

    if intent.get("preference") == "filling":
        search_text += " doyurucu büyük porsiyon menü"
    elif intent.get("preference") == "healthy":
        search_text += " sağlıklı hafif fit"
    elif intent.get("preference") == "protein":
        search_text += " protein tavuk ızgara et"

    top_k = max(limit * 8, 50)

    raw_results = engine.search(search_text, top_k=top_k)

    scored = []
    seen = set()

    for item, semantic_score in raw_results:
        if should_filter_item(item, intent, context, user_profile):
            continue

        dedupe_key = (
            normalize_text(item.get("platform")),
            normalize_text(item.get("restaurant_name")),
            normalize_text(item.get("item_name")),
        )

        if dedupe_key in seen:
            continue

        seen.add(dedupe_key)

        context_score = calculate_context_score(item, intent, context, user_profile)

        final_score = semantic_score * 0.70 + context_score * 0.30

        scored.append(
            {
                "platform": item.get("platform"),
                "restaurant_name": item.get("restaurant_name"),
                "restaurant_rating": item.get("restaurant_rating"),
                "item_name": item.get("item_name"),
                "category": item.get("category"),
                "price": item.get("price"),
                "original_price": item.get("original_price"),
                "discount_rate": item.get("discount_rate"),
                "product_url": item.get("product_url"),
                "image_url": item.get("image_url"),
                "city": item.get("city"),
                "score": round(float(final_score), 4),
                "ml_similarity": round(float(semantic_score), 4),
                "semantic_score": round(float(semantic_score), 4),
                "context_score": round(float(context_score), 4),
                "reason": build_reason(item, intent, semantic_score, context_score),
            }
        )

    scored = sorted(scored, key=lambda item: item["score"], reverse=True)

    return {
        "engine": "sentence_transformer_faiss",
        "intent": intent,
        "context": context,
        "user_profile": user_profile,
        "count": min(len(scored), limit),
        "recommendations": scored[:limit],
    }


if __name__ == "__main__":
    import sys

    query = " ".join(sys.argv[1:]).strip() or "ucuz doyurucu bir şey öner"

    output = semantic_recommend(
        query,
        limit=10,
        context={"hour": 19, "day_type": "weekday", "city": "bursa"},
        user_profile={"diet": "Standart", "allergies": [], "preferred_categories": []},
    )

    print(json.dumps(output, ensure_ascii=False, indent=2))

