import json
import re
from statistics import mean
from typing import Any

try:
    from .semantic_recommender import semantic_recommend, normalize_text, safe_float
except ImportError:
    from semantic_recommender import semantic_recommend, normalize_text, safe_float


GROUP_STOPWORDS = {
    "orta", "buyuk", "kucuk", "boy", "adet", "gr", "g", "kg", "ml", "lt", "cl",
    "menu", "menusu", "ekstra", "double", "tek", "yarim", "tam", "with", "and", "ve",
    "ozel", "super", "mega", "maxi", "large", "small", "medium", "cocuk", "kids",
}

PRODUCT_SYNONYMS = {
    "hamburger": "burger",
    "chicken": "tavuk",
    "grilled": "izgara",
    "fit": "fit",
    "doner": "doner",
    "döner": "doner",
    "kofte": "kofte",
    "köfte": "kofte",
}


def clean_number(value: Any) -> float | None:
    number = safe_float(value)
    if number is None:
        return None
    return round(float(number), 2)


def item_key(item: dict[str, Any]) -> tuple[str, str, str]:
    return (
        normalize_text(item.get("platform")),
        normalize_text(item.get("restaurant_name")),
        normalize_text(item.get("item_name")),
    )


def item_to_public_dict(item: dict[str, Any]) -> dict[str, Any]:
    return {
        "platform": item.get("platform"),
        "restaurant_name": item.get("restaurant_name"),
        "restaurant_rating": clean_number(item.get("restaurant_rating")),
        "item_name": item.get("item_name"),
        "category": item.get("category"),
        "price": clean_number(item.get("price")),
        "original_price": clean_number(item.get("original_price")),
        "discount_rate": clean_number(item.get("discount_rate")),
        "product_url": item.get("product_url"),
        "city": item.get("city"),
        "ai_score": clean_number(item.get("score")),
        "semantic_score": clean_number(item.get("semantic_score") or item.get("ml_similarity")),
        "context_score": clean_number(item.get("context_score")),
        "reason": item.get("reason"),
    }


def calculate_price_score(price: float, min_price: float, max_price: float) -> float:
    if max_price <= min_price:
        return 1.0
    return 1.0 - ((price - min_price) / (max_price - min_price))


def calculate_discount_score(discount: float | None) -> float:
    if discount is None or discount <= 0:
        return 0.0
    return min(discount, 50.0) / 50.0


def canonical_tokens(item_name: Any) -> set[str]:
    text = normalize_text(item_name)
    text = re.sub(r"\b\d+(?:[.,]\d+)?\b", " ", text)
    tokens = []
    for token in text.split():
        token = PRODUCT_SYNONYMS.get(token, token)
        if len(token) < 3:
            continue
        if token in GROUP_STOPWORDS:
            continue
        tokens.append(token)
    return set(tokens)


def token_similarity(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    overlap = len(left & right)
    union = len(left | right)
    return overlap / union if union else 0.0


def product_signature(item: dict[str, Any]) -> str:
    category = normalize_text(item.get("category")) or "genel"
    tokens = sorted(canonical_tokens(item.get("item_name")))
    if not tokens:
        tokens = [normalize_text(item.get("item_name")) or "urun"]
    return f"{category}:{'-'.join(tokens[:5])}"


def choose_group_name(items: list[dict[str, Any]]) -> str:
    if not items:
        return "Benzer Ürün Grubu"
    return max(items, key=lambda item: clean_number(item.get("semantic_score") or item.get("ml_similarity")) or 0).get("item_name") or "Benzer Ürün Grubu"


def build_platform_comparison(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for item in items:
        platform = item.get("platform") or "unknown"
        grouped.setdefault(platform, []).append(item)

    result = []
    for platform, platform_items in grouped.items():
        prices = [
            clean_number(item.get("price"))
            for item in platform_items
            if clean_number(item.get("price")) is not None
        ]
        if not prices:
            continue
        cheapest = min(platform_items, key=lambda item: clean_number(item.get("price")) or float("inf"))
        result.append(
            {
                "platform": platform,
                "item_count": len(platform_items),
                "min_price": round(min(prices), 2),
                "max_price": round(max(prices), 2),
                "avg_price": round(mean(prices), 2),
                "cheapest_item": item_to_public_dict(cheapest),
            }
        )
    return sorted(result, key=lambda row: row["min_price"])


def build_cross_platform_comparisons(items: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    clusters: list[dict[str, Any]] = []

    for item in items:
        item_category = normalize_text(item.get("category"))
        item_tokens = canonical_tokens(item.get("item_name"))
        placed = False

        for cluster in clusters:
            if cluster["category"] != item_category:
                continue
            similarity = token_similarity(item_tokens, cluster["tokens"])
            same_signature = product_signature(item) == cluster["signature"]
            if same_signature or similarity >= 0.50:
                cluster["items"].append(item)
                cluster["tokens"] |= item_tokens
                placed = True
                break

        if not placed:
            clusters.append(
                {
                    "category": item_category,
                    "tokens": set(item_tokens),
                    "signature": product_signature(item),
                    "items": [item],
                }
            )

    result = []
    for cluster in clusters:
        cluster_items = cluster["items"]
        by_platform: dict[str, list[dict[str, Any]]] = {}
        for item in cluster_items:
            by_platform.setdefault(item.get("platform") or "unknown", []).append(item)

        platform_prices = []
        for platform, platform_items in by_platform.items():
            cheapest = min(platform_items, key=lambda item: clean_number(item.get("price")) or float("inf"))
            platform_prices.append(
                {
                    "platform": platform,
                    "price": clean_number(cheapest.get("price")),
                    "original_price": clean_number(cheapest.get("original_price")),
                    "discount_rate": clean_number(cheapest.get("discount_rate")),
                    "restaurant_name": cheapest.get("restaurant_name"),
                    "item_name": cheapest.get("item_name"),
                    "product_url": cheapest.get("product_url"),
                    "item": item_to_public_dict(cheapest),
                }
            )

        platform_prices = [row for row in platform_prices if row["price"] is not None]
        if not platform_prices:
            continue

        prices = [row["price"] for row in platform_prices]
        cheapest_row = min(platform_prices, key=lambda row: row["price"])
        most_expensive_row = max(platform_prices, key=lambda row: row["price"])
        price_gap = most_expensive_row["price"] - cheapest_row["price"]
        saving_rate = round((price_gap / most_expensive_row["price"]) * 100, 2) if most_expensive_row["price"] else 0.0

        result.append(
            {
                "group_name": choose_group_name(cluster_items),
                "category": cluster_items[0].get("category"),
                "platform_count": len(platform_prices),
                "item_count": len(cluster_items),
                "min_price": round(min(prices), 2),
                "max_price": round(max(prices), 2),
                "avg_price": round(mean(prices), 2),
                "price_gap": round(price_gap, 2),
                "saving_rate_percent": saving_rate,
                "cheapest_platform": cheapest_row["platform"],
                "cheapest_item": cheapest_row,
                "platform_prices": sorted(platform_prices, key=lambda row: row["price"]),
            }
        )

    return sorted(
        result,
        key=lambda row: (row["platform_count"], row["saving_rate_percent"], -row["min_price"]),
        reverse=True,
    )[:limit]


def compare_items(
    query: str,
    limit: int = 10,
    context: dict[str, Any] | None = None,
    user_profile: dict[str, Any] | None = None,
) -> dict[str, Any]:
    context = context or {}
    user_profile = user_profile or {}
    candidate_limit = max(limit * 10, 80)

    recommendation_response = semantic_recommend(
        query=query,
        limit=candidate_limit,
        context=context,
        user_profile=user_profile,
    )
    raw_items = recommendation_response.get("recommendations", [])

    seen = set()
    candidates = []
    for item in raw_items:
        price = clean_number(item.get("price"))
        if price is None or price <= 0:
            continue
        key = item_key(item)
        if key in seen:
            continue
        seen.add(key)
        candidates.append(item)

    if not candidates:
        return {
            "engine": "semantic_compare",
            "query": query,
            "intent": recommendation_response.get("intent"),
            "context": context,
            "user_profile": user_profile,
            "candidate_count": 0,
            "price_analysis": None,
            "cheapest_items": [],
            "best_value_items": [],
            "platform_comparison": [],
            "cross_platform_comparisons": [],
        }

    prices = [clean_number(item.get("price")) for item in candidates]
    prices = [price for price in prices if price is not None]
    min_price = min(prices)
    max_price = max(prices)
    avg_price = mean(prices)

    cheapest_item = min(candidates, key=lambda item: clean_number(item.get("price")) or float("inf"))
    most_expensive_item = max(candidates, key=lambda item: clean_number(item.get("price")) or 0)

    scored_for_value = []
    for item in candidates:
        price = clean_number(item.get("price")) or 0.0
        semantic_score = clean_number(item.get("semantic_score") or item.get("ml_similarity")) or 0.0
        context_score = clean_number(item.get("context_score")) or 0.0
        discount_score = calculate_discount_score(clean_number(item.get("discount_rate")))
        price_score = calculate_price_score(price, min_price, max_price)
        compare_score = semantic_score * 0.45 + context_score * 0.15 + price_score * 0.30 + discount_score * 0.10
        enriched = dict(item)
        enriched["compare_score"] = round(compare_score, 4)
        enriched["price_score"] = round(price_score, 4)
        enriched["discount_score"] = round(discount_score, 4)
        scored_for_value.append(enriched)

    cheapest_items = sorted(candidates, key=lambda item: clean_number(item.get("price")) or float("inf"))[:limit]
    best_value_items = sorted(scored_for_value, key=lambda item: item.get("compare_score", 0), reverse=True)[:limit]

    price_gap = max_price - min_price
    saving_rate = round((price_gap / max_price) * 100, 2) if max_price > 0 else 0.0

    return {
        "engine": "semantic_compare",
        "query": query,
        "intent": recommendation_response.get("intent"),
        "context": context,
        "user_profile": user_profile,
        "candidate_count": len(candidates),
        "price_analysis": {
            "min_price": round(min_price, 2),
            "max_price": round(max_price, 2),
            "avg_price": round(avg_price, 2),
            "price_gap": round(price_gap, 2),
            "saving_rate_percent": saving_rate,
            "cheapest_item": item_to_public_dict(cheapest_item),
            "most_expensive_item": item_to_public_dict(most_expensive_item),
        },
        "cheapest_items": [item_to_public_dict(item) for item in cheapest_items],
        "best_value_items": [
            {
                **item_to_public_dict(item),
                "compare_score": clean_number(item.get("compare_score")),
                "price_score": clean_number(item.get("price_score")),
                "discount_score": clean_number(item.get("discount_score")),
            }
            for item in best_value_items
        ],
        "platform_comparison": build_platform_comparison(candidates),
        "cross_platform_comparisons": build_cross_platform_comparisons(candidates, limit),
    }


if __name__ == "__main__":
    import sys
    query = " ".join(sys.argv[1:]).strip() or "pizza karşılaştır"
    output = compare_items(
        query=query,
        limit=10,
        context={"hour": 19, "day_type": "weekday", "city": "bursa"},
        user_profile={"diet": "Standart", "allergies": [], "max_budget": None},
    )
    print(json.dumps(output, ensure_ascii=False, indent=2))
