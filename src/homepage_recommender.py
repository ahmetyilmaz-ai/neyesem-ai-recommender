import argparse
import json
from pathlib import Path

import pandas as pd

from recommender import (
    load_items,
    normalize_text,
    CATEGORY_KEYWORDS,
    FILLING_WORDS,
    is_addon_item,
)


OUTPUT_PATH = Path("data/homepage_recommendations.json")


MAIN_FOOD_CATEGORIES = [
    "pizza",
    "burger",
    "döner",
    "pide",
    "tavuk",
]

HEALTHY_WORDS = [
    "salata",
    "fit",
    "bowl",
    "taco",
    "tacofit",
    "ızgara",
    "izgara",
    "vejetaryen",
    "veggie",
]

BAD_HOMEPAGE_WORDS = [
    "sos",
    "ketçap",
    "ketcap",
    "mayonez",
    "peçete",
    "pecete",
    "detaylar",
    "ekstra",
    "pepsi",
    "coca cola",
    "coca-cola",
    "sprite",
    "fanta",
    "fuse tea",
    "ice tea",
    "ayran",
    "limonata",
]


def tokenize(text):
    return normalize_text(text).split()


def has_any(text, words):
    """
    Kısa kelimelerde substring eşleşmesi yapma.
    Örn: "fit" kelimesi "profiterol" içinde geçtiği için sağlıklı sayılmasın.
    """
    normalized = normalize_text(text)
    tokens = set(tokenize(text))

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


def item_matches_category(row, category):
    """
    Ana sayfa önerilerinde kategori eşleşmesini ürün adına göre yap.
    Restoran adına göre yaparsak Pizza Bulls restoranındaki kola/sos da pizza sanılıyor.
    """
    item_text = row.get("item_text", "")

    keywords = CATEGORY_KEYWORDS.get(category, [category])
    return has_any(item_text, keywords)


def is_clean_homepage_item(row):
    item_name = row.get("item_name", "")
    price = row.get("price")

    if is_addon_item(item_name):
        return False

    if has_any(item_name, BAD_HOMEPAGE_WORDS):
        return False

    # Ürün adı sadece içecek / yan ürün ise ana sayfa karma önerilerine sokma.
    # İçecek istenirse arama önerisinde ayrıca bulunabilir.
    item_tokens = set(tokenize(item_name))
    drink_tokens = {"su", "kola", "ayran", "pepsi", "sprite", "fanta"}

    if item_tokens and item_tokens.issubset(drink_tokens):
        return False

    if pd.isna(price):
        return False

    if price <= 20 or price > 1500:
        return False

    return True


def base_score(row):
    score = 0.0

    price = row.get("price")
    discount = row.get("discount_rate")
    rating = row.get("restaurant_rating")
    original_price = row.get("original_price")

    if pd.notna(price):
        # Ana sayfada çok pahalı ürünleri biraz geri at
        if 80 <= price <= 350:
            score += 0.35
        elif 350 < price <= 600:
            score += 0.20
        elif price < 80:
            score += 0.10

    if pd.notna(discount) and discount > 0:
        # Scrape hatalarından etkilenmemek için indirimi 40 ile sınırla
        score += min(float(discount), 40) / 100 * 0.25

    if pd.notna(rating):
        score += min(float(rating), 5) / 5 * 0.15

    if pd.notna(price) and pd.notna(original_price):
        # Aşırı uçuk eski fiyatları biraz cezalandır
        if original_price > price * 5:
            score -= 0.10

    return score


def score_budget_filling(row):
    score = base_score(row)

    item_text = row.get("item_text", "")

    if has_any(item_text, FILLING_WORDS):
        score += 0.35

    price = row.get("price")
    if pd.notna(price):
        if price <= 250:
            score += 0.25
        elif price <= 350:
            score += 0.10

    return score


def score_discount(row):
    score = base_score(row)

    discount = row.get("discount_rate")
    price = row.get("price")

    if pd.notna(discount):
        if 10 <= discount <= 60:
            score += float(discount) / 100 * 0.60
        elif discount > 60:
            # Aşırı indirimler veri hatası olabilir, yine de tamamen çöpe atma
            score += 0.20

    if pd.notna(price) and price <= 500:
        score += 0.15

    return score


def score_category(row, category):
    score = base_score(row)

    if item_matches_category(row, category):
        score += 0.50

    return score


def score_healthy(row):
    score = base_score(row)

    text = row.get("item_text", "") + " " + row.get("restaurant_text", "")

    if has_any(text, HEALTHY_WORDS):
        score += 0.50

    price = row.get("price")
    if pd.notna(price) and price <= 500:
        score += 0.15

    return score


def select_diverse_items(df, score_column, limit=8, max_per_restaurant=2, max_per_platform=5):
    selected = []
    restaurant_counts = {}
    platform_counts = {}

    sorted_df = df.sort_values(score_column, ascending=False)

    for _, row in sorted_df.iterrows():
        restaurant = row.get("restaurant_name") or "unknown"
        platform = row.get("platform") or "unknown"

        if restaurant_counts.get(restaurant, 0) >= max_per_restaurant:
            continue

        if platform_counts.get(platform, 0) >= max_per_platform:
            continue

        selected.append(row)

        restaurant_counts[restaurant] = restaurant_counts.get(restaurant, 0) + 1
        platform_counts[platform] = platform_counts.get(platform, 0) + 1

        if len(selected) >= limit:
            break

    return selected


def item_to_dict(row):
    def clean_number(value):
        if pd.isna(value):
            return None
        return round(float(value), 2)

    return {
        "platform": row.get("platform"),
        "restaurant_name": row.get("restaurant_name"),
        "item_name": row.get("item_name"),
        "price": clean_number(row.get("price")),
        "original_price": clean_number(row.get("original_price")),
        "discount_rate": clean_number(row.get("discount_rate")),
        "product_url": row.get("product_url"),
        "city": row.get("city"),
        "score": clean_number(row.get("homepage_score")),
    }


def make_section(title, description, df, score_column="homepage_score", limit=8):
    rows = select_diverse_items(df, score_column=score_column, limit=limit)

    return {
        "title": title,
        "description": description,
        "items": [item_to_dict(row) for row in rows],
    }


def generate_homepage_recommendations(limit=8):
    df = load_items()

    df = df[df.apply(is_clean_homepage_item, axis=1)].copy()

    sections = []

    # 1. Uygun fiyatlı doyurucu seçenekler
    budget_df = df.copy()
    budget_df["homepage_score"] = budget_df.apply(score_budget_filling, axis=1)
    budget_df = budget_df[budget_df["price"] <= 350].copy()

    sections.append(
        make_section(
            title="Uygun Fiyatlı Doyurucular",
            description="Fiyatı görece uygun, ana yemek sayılabilecek doyurucu seçenekler.",
            df=budget_df,
            limit=limit,
        )
    )

    # 2. Pizza & Burger
    pizza_burger_df = df[
        df.apply(lambda row: item_matches_category(row, "pizza") or item_matches_category(row, "burger"), axis=1)
    ].copy()

    pizza_burger_df["homepage_score"] = pizza_burger_df.apply(
        lambda row: max(score_category(row, "pizza"), score_category(row, "burger")),
        axis=1,
    )

    sections.append(
        make_section(
            title="Pizza & Burger Önerileri",
            description="Popüler fast food kategorilerinden seçilmiş ürünler.",
            df=pizza_burger_df,
            limit=limit,
        )
    )

    # 3. Tatlı
    dessert_df = df[
        df.apply(lambda row: item_matches_category(row, "tatlı"), axis=1)
    ].copy()

    dessert_df["homepage_score"] = dessert_df.apply(lambda row: score_category(row, "tatlı"), axis=1)

    sections.append(
        make_section(
            title="Tatlı Kaçamağı",
            description="Tatlı, pasta, waffle ve benzeri seçenekler.",
            df=dessert_df,
            limit=limit,
        )
    )

    # 4. İndirimli ürünler
    discount_df = df[
        df["discount_rate"].notna() & (df["discount_rate"] > 0)
    ].copy()

    discount_df["homepage_score"] = discount_df.apply(score_discount, axis=1)

    sections.append(
        make_section(
            title="İndirimli Fırsatlar",
            description="İndirim oranı ve fiyat dengesi iyi olan seçenekler.",
            df=discount_df,
            limit=limit,
        )
    )

    # 5. Hafif / sağlıklı
    healthy_df = df[
        df.apply(lambda row: has_any(row.get("item_text", ""), HEALTHY_WORDS), axis=1)
    ].copy()

    healthy_df["homepage_score"] = healthy_df.apply(score_healthy, axis=1)

    sections.append(
        make_section(
            title="Hafif ve Sağlıklı Seçenekler",
            description="Daha hafif, fit veya sağlıklı sayılabilecek ürünler.",
            df=healthy_df,
            limit=limit,
        )
    )

    # 6. Karışık ana yemekler
    main_food_df = df[
        df.apply(
            lambda row: any(item_matches_category(row, category) for category in MAIN_FOOD_CATEGORIES),
            axis=1,
        )
    ].copy()

    main_food_df["homepage_score"] = main_food_df.apply(base_score, axis=1)

    sections.append(
        make_section(
            title="Bugünün Karışık Önerileri",
            description="Farklı platform ve restoranlardan karışık ana yemek önerileri.",
            df=main_food_df,
            limit=limit,
        )
    )

    return {
        "type": "cold_start_homepage_recommendations",
        "description": "Kullanıcı geçmişi olmadan fiyat, kategori, indirim ve ürün içeriğine göre oluşturulan ana sayfa önerileri.",
        "section_count": len(sections),
        "sections": sections,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=8)
    parser.add_argument("--save", action="store_true")
    args = parser.parse_args()

    output = generate_homepage_recommendations(limit=args.limit)

    if args.save:
        OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

        with OUTPUT_PATH.open("w", encoding="utf-8") as file:
            json.dump(output, file, ensure_ascii=False, indent=2)

        print(f"Kaydedildi: {OUTPUT_PATH}")

    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
