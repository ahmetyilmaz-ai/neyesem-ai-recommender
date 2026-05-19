from .recommender import normalize_text, CATEGORY_KEYWORDS


def infer_category_name(item_name):
    text = normalize_text(item_name)

    category_map = {
        "Pizza": ["pizza"],
        "Burger": ["burger", "hamburger"],
        "Döner": ["döner", "doner"],
        "Pide & Lahmacun": ["pide", "lahmacun"],
        "Tatlı": ["tatlı", "tatli", "waffle", "pasta", "baklava", "dondurma", "sütlaç", "sutlac", "kazandibi", "magnolia"],
        "Tavuk": ["tavuk", "chicken", "kanat"],
        "İçecek": ["su", "kola", "ayran", "ice tea", "fanta", "sprite", "limonata", "pepsi"],
        "Sağlıklı": ["salata", "fit", "bowl", "ızgara", "izgara"],
    }

    for category, keywords in category_map.items():
        for keyword in keywords:
            if normalize_text(keyword) in text:
                return category

    return "Genel"


def platform_display_name(platform):
    mapping = {
        "getir_yemek": "Getir Yemek",
        "trendyol": "Trendyol Go",
        "trendyol_go": "Trendyol Go",
        "yemeksepeti": "Yemeksepeti",
    }

    return mapping.get(str(platform or ""), str(platform or "Bilinmeyen Platform"))


def to_number(value):
    if value is None:
        return None

    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def recommendation_item_to_db_format(item, rank):
    item_name = item.get("item_name")
    restaurant_name = item.get("restaurant_name")
    platform = item.get("platform")

    price = to_number(item.get("price"))
    original_price = to_number(item.get("original_price"))
    discount_rate = to_number(item.get("discount_rate"))

    return {
        "sira": rank,
        "platform": {
            "ad": platform_display_name(platform),
            "kod": platform,
        },
        "sehir": {
            "ad": item.get("city") or "bursa",
        },
        "restoran": {
            "id": item.get("restaurant_id"),
            "ad": restaurant_name,
            "puan": to_number(item.get("restaurant_rating")),
        },
        "kategori": {
            "ad": infer_category_name(item_name),
        },
        "urun": {
            "id": item.get("product_id") or item.get("item_id"),
            "ad": item_name,
            "fiyat": price,
            "orijinal_fiyat": original_price,
            "indirim_yuzdesi": discount_rate,
            "gorsel_url": item.get("image_url") or "",
            "urun_url": item.get("product_url"),
            "musait_mi": True,
        },
        "ai": {
            "skor": to_number(item.get("score")),
            "benzerlik_skoru": to_number(item.get("ml_similarity")),
            "neden": item.get("reason"),
        },
    }


def adapt_recommend_response(response):
    recommendations = response.get("recommendations", [])

    return {
        "tip": "ai_recommendation_response",
        "intent": response.get("intent"),
        "toplam_oneri": len(recommendations),
        "oneriler": [
            recommendation_item_to_db_format(item, index + 1)
            for index, item in enumerate(recommendations)
        ],
    }


def adapt_homepage_response(response):
    sections = response.get("sections", [])

    adapted_sections = []

    for section in sections:
        items = section.get("items", [])

        adapted_sections.append(
            {
                "baslik": section.get("title"),
                "aciklama": section.get("description"),
                "toplam_urun": len(items),
                "urunler": [
                    recommendation_item_to_db_format(item, index + 1)
                    for index, item in enumerate(items)
                ],
            }
        )

    return {
        "tip": "cold_start_homepage_response",
        "aciklama": response.get("description"),
        "bolum_sayisi": len(adapted_sections),
        "bolumler": adapted_sections,
    }
