import json
import re
import sys
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


DATA_PATH = Path("data/all_items.json")

CATEGORY_KEYWORDS = {
    "pizza": ["pizza"],
    "burger": ["burger", "hamburger"],
    "döner": ["döner", "doner"],
    "pide": ["pide", "lahmacun"],
    "tatlı": ["tatlı", "tatli", "waffle", "pasta", "baklava", "dondurma", "sütlaç", "sutlac", "kazandibi", "magnolia"],
    "tavuk": ["tavuk", "chicken", "kanat", "şiş", "sis"],
    "içecek": ["su", "kola", "ayran", "ice tea", "fanta", "sprite", "içecek", "icecek"],
    "sağlıklı": ["salata", "fit", "bowl", "taco", "tacofit", "ızgara", "izgara"],
}

FILLING_WORDS = [
    "menü", "menu", "döner", "doner", "burger", "pizza", "pide",
    "lahmacun", "tavuk", "köfte", "kofte", "dürüm", "durum",
    "tantuni", "kebap", "kebab", "taco"
]

BAD_ITEM_WORDS = [
    "detaylar", "sepet", "minimum", "teslimat", "kampanya",
    "cookie", "çerez", "cerez", "şu anda", "su anda", "temsili",
    "anasayfa", "restoran", "filtre", "sırala", "sirala"
]

ADDON_WORDS = [
    "ketçap", "ketcap", "mayonez", "sos", "acı sos", "aci sos",
    "ranch", "barbekü", "barbeku", "cheddar sos", "kenar sos",
    "ekstra", "peçete", "pecete"
]


def normalize_text(value):
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


def is_bad_item_name(name):
    raw = str(name or "").strip()
    normalized = normalize_text(raw)

    if not raw:
        return True

    if len(raw) < 3:
        return True

    if normalized.isdigit():
        return True

    if any(normalize_text(word) in normalized for word in BAD_ITEM_WORDS):
        return True

    return False


def is_addon_item(name):
    normalized = normalize_text(name)
    return any(normalize_text(word) in normalized for word in ADDON_WORDS)


def load_items():
    with DATA_PATH.open("r", encoding="utf-8") as file:
        items = json.load(file)

    df = pd.DataFrame(items)

    for column in [
        "platform",
        "restaurant_name",
        "restaurant_rating",
        "item_name",
        "price",
        "original_price",
        "discount_rate",
        "product_url",
        "city",
    ]:
        if column not in df.columns:
            df[column] = None

    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df["original_price"] = pd.to_numeric(df["original_price"], errors="coerce")
    df["discount_rate"] = pd.to_numeric(df["discount_rate"], errors="coerce")
    df["restaurant_rating"] = pd.to_numeric(df["restaurant_rating"], errors="coerce")

    df = df.dropna(subset=["price"]).copy()
    df = df[~df["item_name"].apply(is_bad_item_name)].copy()
    df = df[(df["price"] > 0) & (df["price"] <= 2500)].copy()

    df["item_text"] = df["item_name"].fillna("").apply(normalize_text)
    df["restaurant_text"] = df["restaurant_name"].fillna("").apply(normalize_text)
    df["platform_text"] = df["platform"].fillna("").apply(normalize_text)

    df["model_text"] = (
        df["item_text"] + " " +
        df["item_text"] + " " +
        df["restaurant_text"] + " " +
        df["platform_text"]
    )

    return df.reset_index(drop=True)


def infer_intent(user_text):
    normalized = normalize_text(user_text)

    categories = []

    for category, keywords in CATEGORY_KEYWORDS.items():
        if any(normalize_text(keyword) in normalized for keyword in keywords):
            categories.append(category)

    budget_max = None

    numbers = re.findall(r"\d+", normalized)
    if numbers:
        possible_budget = max(int(number) for number in numbers)
        if 50 <= possible_budget <= 2000:
            budget_max = possible_budget

    if budget_max is None and any(word in normalized for word in ["ucuz", "uygun", "butce", "butçe", "bütce", "bütçe", "pahali olmasin"]):
        budget_max = 250

    if any(word in normalized for word in ["cok ac", "çok aç", "doyurucu", "acim", "açım"]):
        preference = "filling"
    elif any(word in normalized for word in ["ucuz", "uygun", "pahali olmasin", "pahalı olmasın"]):
        preference = "cheap"
    elif any(word in normalized for word in ["saglikli", "sağlıklı", "hafif"]):
        preference = "healthy"
    elif any(word in normalized for word in ["tatli", "tatlı", "sweet"]):
        preference = "sweet"
        if "tatlı" not in categories:
            categories.append("tatlı")
    else:
        preference = "balanced"

    return {
        "raw_query": user_text,
        "normalized_query": normalized,
        "categories": categories,
        "budget_max": budget_max,
        "preference": preference,
    }


def item_matches_category(row, category):
    item_text = row.get("item_text", "")
    restaurant_text = row.get("restaurant_text", "")

    keywords = CATEGORY_KEYWORDS.get(category, [category])
    keywords = [normalize_text(keyword) for keyword in keywords]

    if category in ["pizza", "burger", "döner", "pide", "tatlı", "tavuk", "içecek"]:
        return any(keyword in item_text for keyword in keywords)

    return any(keyword in item_text or keyword in restaurant_text for keyword in keywords)


def filter_candidates(df, intent):
    candidates = df.copy()

    categories = intent.get("categories") or []
    budget_max = intent.get("budget_max")
    preference = intent.get("preference")

    if categories:
        mask = pd.Series(False, index=candidates.index)

        for category in categories:
            mask = mask | candidates.apply(lambda row: item_matches_category(row, category), axis=1)

        candidates = candidates[mask].copy()

    if categories and "içecek" not in categories:
        candidates = candidates[~candidates["item_name"].apply(is_addon_item)].copy()

    if preference == "filling":
        candidates = candidates[~candidates["item_name"].apply(is_addon_item)].copy()

        filling_mask = candidates["item_text"].apply(
            lambda text: any(normalize_text(word) in text for word in FILLING_WORDS)
        )

        if filling_mask.sum() >= 5:
            candidates = candidates[filling_mask].copy()

    if budget_max:
        candidates = candidates[candidates["price"] <= float(budget_max) * 1.25].copy()

    return candidates.reset_index(drop=True)


def build_query_text(intent):
    parts = [intent.get("normalized_query", "")]

    for category in intent.get("categories") or []:
        parts.append(category)
        parts.extend(CATEGORY_KEYWORDS.get(category, []))

    preference = intent.get("preference")

    if preference == "filling":
        parts.extend(FILLING_WORDS)
    elif preference == "healthy":
        parts.extend(CATEGORY_KEYWORDS["sağlıklı"])
    elif preference == "sweet":
        parts.extend(CATEGORY_KEYWORDS["tatlı"])

    return " ".join(parts)


def calculate_business_score(row, intent):
    score = 0.0

    price = row.get("price")
    discount = row.get("discount_rate")
    rating = row.get("restaurant_rating")
    preference = intent.get("preference")
    budget_max = intent.get("budget_max")

    if pd.notna(price):
        if budget_max and price <= budget_max:
            score += 0.20

        if preference == "cheap":
            score += max(0, 1 - min(price, 500) / 500) * 0.25
        elif preference == "balanced":
            score += max(0, 1 - min(price, 700) / 700) * 0.15
        elif preference == "filling":
            if 80 <= price <= 450:
                score += 0.20
            elif price < 80:
                score -= 0.15

    if pd.notna(discount):
        capped_discount = min(float(discount), 40)
        score += (capped_discount / 100) * 0.20

    if pd.notna(rating):
        score += min(float(rating), 5) / 5 * 0.10

    if preference == "filling":
        item_text = row.get("item_text", "")
        if any(normalize_text(word) in item_text for word in FILLING_WORDS):
            score += 0.20

    return score


def build_reason(row, intent):
    parts = []

    categories = intent.get("categories") or []
    preference = intent.get("preference")

    if categories:
        parts.append(f"{', '.join(categories)} isteğine uygun")

    if preference == "filling":
        parts.append("doyurucu seçenek")
    elif preference == "cheap":
        parts.append("fiyat odaklı seçildi")
    elif preference == "healthy":
        parts.append("daha hafif/sağlıklı seçenek")
    elif preference == "sweet":
        parts.append("tatlı isteğine uygun")

    if pd.notna(row.get("price")):
        parts.append(f"{row.get('price'):.2f} TL")

    if pd.notna(row.get("discount_rate")) and row.get("discount_rate") > 0:
        parts.append(f"%{row.get('discount_rate'):.1f} indirim")

    if not parts:
        return "Metin benzerliği ve fiyat skoruna göre önerildi."

    return ", ".join(parts) + "."


def recommend(user_text, limit=10):
    df = load_items()
    intent = infer_intent(user_text)

    candidates = filter_candidates(df, intent)

    if candidates.empty:
        return {
            "intent": intent,
            "count": 0,
            "recommendations": [],
        }

    vectorizer = TfidfVectorizer(
        min_df=1,
        ngram_range=(1, 2),
        sublinear_tf=True,
    )

    item_vectors = vectorizer.fit_transform(candidates["model_text"])
    query_text = build_query_text(intent)
    query_vector = vectorizer.transform([query_text])

    similarity = cosine_similarity(query_vector, item_vectors).flatten()

    candidates = candidates.copy()
    candidates["ml_similarity"] = similarity
    candidates["business_score"] = candidates.apply(lambda row: calculate_business_score(row, intent), axis=1)

    candidates["score"] = (
        candidates["ml_similarity"] * 0.70 +
        candidates["business_score"] * 0.30
    )

    candidates["dedupe_key"] = (
        candidates["platform"].fillna("") + "|" +
        candidates["restaurant_name"].fillna("") + "|" +
        candidates["item_name"].fillna("").apply(normalize_text)
    )

    candidates = candidates.sort_values(["score", "ml_similarity", "price"], ascending=[False, False, True])
    candidates = candidates.drop_duplicates(subset=["dedupe_key"])

    results = []

    for _, row in candidates.head(limit).iterrows():
        original_price = row.get("original_price")
        discount_rate = row.get("discount_rate")

        results.append(
            {
                "platform": row.get("platform"),
                "restaurant_name": row.get("restaurant_name"),
                "item_name": row.get("item_name"),
                "price": round(float(row.get("price")), 2),
                "original_price": None if pd.isna(original_price) else round(float(original_price), 2),
                "discount_rate": None if pd.isna(discount_rate) else round(float(discount_rate), 2),
                "product_url": row.get("product_url"),
                "score": round(float(row.get("score")), 4),
                "ml_similarity": round(float(row.get("ml_similarity")), 4),
                "reason": build_reason(row, intent),
            }
        )

    return {
        "intent": intent,
        "count": len(results),
        "recommendations": results,
    }


if __name__ == "__main__":
    query = " ".join(sys.argv[1:]).strip()

    if not query:
        query = input("Ne yemek istersin? ")

    output = recommend(query, limit=10)

    print(json.dumps(output, ensure_ascii=False, indent=2))
