import json
from statistics import mean
from typing import Any

try:
    from .semantic_recommender import semantic_recommend, normalize_text, safe_float
except ImportError:
    from semantic_recommender import semantic_recommend, normalize_text, safe_float


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

        cheapest = min(
            platform_items,
            key=lambda item: clean_number(item.get("price")) or float("inf"),
        )
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


def compare_items(
    query: str,
    limit: int = 10,
    context: dict[str, Any] | None = None,
    user_profile: dict[str, Any] | None = None,
) -> dict[str, Any]:
    context = context or {}
    user_profile = user_profile or {}
    candidate_limit = max(limit * 8, 60)

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
        compare_score = (
            semantic_score * 0.45
            + context_score * 0.15
            + price_score * 0.30
            + discount_score * 0.10
        )
        enriched = dict(item)
        enriched["compare_score"] = round(compare_score, 4)
        enriched["price_score"] = round(price_score, 4)
        enriched["discount_score"] = round(discount_score, 4)
        scored_for_value.append(enriched)

    cheapest_items = sorted(
        candidates,
        key=lambda item: clean_number(item.get("price")) or float("inf"),
    )[:limit]
    best_value_items = sorted(
        scored_for_value,
        key=lambda item: item.get("compare_score", 0),
        reverse=True,
    )[:limit]

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
