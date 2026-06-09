import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
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
    "doner": ["döner", "doner"],
    "pide": ["pide", "lahmacun"],
    "tatli": ["tatlı", "tatli", "waffle", "pasta", "baklava", "dondurma", "sütlaç", "sutlac", "kazandibi", "magnolia", "brownie", "browni"],
    "tavuk": ["tavuk", "chicken", "kanat", "şiş", "sis"],
    "icecek": ["su", "kola", "ayran", "ice tea", "fanta", "sprite", "içecek", "icecek", "pepsi", "limonata", "soda"],
    "saglikli": ["salata", "fit", "bowl", "ızgara", "izgara", "hafif"],
    "corba": ["çorba", "corba", "soup"],
}

HEAVY_MORNING_WORDS = [
    "burger", "hamburger", "kebap", "kebab", "adana", "urfa", "lahmacun",
    "tantuni", "iskender", "döner", "doner", "pizza", "kanat", "pirzola",
]

LACTOSE_WORDS = [
    "süt", "sut", "milk", "peynir", "cheese", "yoğurt", "yogurt", "ayran",
    "dondurma", "krema", "kremalı", "kremali", "magnolia", "sütlaç", "sutlac",
    "kazandibi", "profiterol", "cheesecake", "mozzarella", "parmesan",
    "pasta", "kek", "cake", "tiramisu", "muffin", "milkshake", "latte",
    "kaşar", "kasar", "labne", "kaymak", "muhallebi", "trileçe", "trilece",
    "supangle", "künefe", "kunefe", "waffle", "tatlı", "tatli",
]

MEAT_WORDS = [
    "et", "dana", "tavuk", "chicken", "köfte", "kofte", "döner", "doner",
    "kebap", "kebab", "sucuk", "kanat", "tantuni", "burger",
    "dürüm", "durum", "lahmacun", "iskender", "pastırma", "pastirma",
    "jambon", "salam", "sosis", "kavurma", "pirzola", "bonfile", "şiş", "sis",
    "piliç", "pilic", "kıyma", "kiyma", "nugget", "wrap",
]

SEAFOOD_WORDS = [
    "balık", "balik", "hamsi", "midye", "karides", "somon", "levrek",
    "çipura", "cipura", "kalamar", "ahtapot", "deniz",
]

# Vegan, et/deniz ürünlerine ek olarak süt ve yumurta ürünlerini de dışlar.
EGG_WORDS = ["yumurta", "omlet", "menemen", "mayonez"]

# Adında et/süt kelimesi geçmese bile bu kategoriler neredeyse her zaman risklidir.
# Bu sayede "Double Dürüm" (et) vegan filtresinden, "Doğum Günü Pasta" (süt)
# laktoz filtresinden kaçamaz.
MEAT_CATEGORIES = {"doner", "kebap", "tavuk"}
LACTOSE_CATEGORIES = {"tatli", "pizza"}

DRINK_WORDS = [
    "su", "ayran", "kola", "pepsi", "fanta", "sprite", "ice tea", "fuse tea",
    "limonata", "soda", "maden suyu", "şalgam", "salgam", "cola", "zero sugar",
]

ADDON_WORDS = [
    "sos", "ketçap", "ketcap", "mayonez", "ranch", "barbekü", "barbeku",
    "acı sos", "aci sos", "cheddar sos", "ekstra", "peçete", "pecete",
]

DESSERT_WORDS = [
    "tatlı", "tatli", "waffle", "brownie", "browni", "dondurma", "baklava",
    "magnolia", "sütlaç", "sutlac", "kazandibi", "profiterol", "kruvasan", "croissant",
    "cheesecake", "pasta", "cookie", "kurabiye",
]

PROTEIN_WORDS = [
    "protein", "tavuk", "chicken", "et", "dana", "köfte", "kofte", "döner", "doner",
    "kebap", "kebab", "pilav", "ızgara", "izgara", "şiş", "sis", "kanat", "burger",
]

FILLING_WORDS = [
    "menü", "menu", "döner", "doner", "burger", "pizza", "pide", "lahmacun",
    "tavuk", "köfte", "kofte", "dürüm", "durum", "tantuni", "kebap", "kebab",
    "pilav", "kumpir", "makarna",
]

HEALTHY_WORDS = [
    "salata", "fit", "bowl", "ızgara", "izgara", "hafif", "protein", "tavuk",
]

SIDE_DISH_WORDS = [
    "patates", "kızartma", "kizartma", "fries", "cips", "chips", "nugget sos",
]

PROMO_WORDS = [
    "ilkyemek", "kupon", "kampanya", "indirim kodu", "kod", "promo", "promosyon",
]


# Literal eşleşmede boost'u bozan dolgu kelimeleri (anlam taşımayanlar).
QUERY_STOPWORDS = {
    "bir", "sey", "seyler", "oner", "onerir", "onerisi", "istiyorum", "isterim",
    "lutfen", "biraz", "bana", "icin", "ile", "gibi", "olsun", "olan", "var",
    "yemek", "yiyecek", "canim", "istiyom", "ver", "bul",
    # Tercih/sıfat kelimeleri: bunlar preference kurallarıyla işlenir; ürün adında
    # geçmeleri (ör. "Proteini Yüksek ...") gerçek protein kaynağı anlamına gelmez,
    # bu yüzden literal isim eşleşmesini sürüklemesinler.
    "protein", "proteini", "proteinli", "yuksek", "ucuz", "uygun", "ekonomik",
    "doyurucu", "saglikli", "hafif", "fit", "lezzetli", "taze", "spor", "spordan",
    "ciktim", "acim", "aciktim",
}


class RecommenderNotReadyError(RuntimeError):
    pass


def normalize_text(value: Any) -> str:
    value = str(value or "").lower().strip()
    replacements = {"ı": "i", "ğ": "g", "ü": "u", "ş": "s", "ö": "o", "ç": "c"}
    for source, target in replacements.items():
        value = value.replace(source, target)
    value = re.sub(r"[^a-z0-9\s]", " ", value)
    value = re.sub(r"\s+", " ", value)
    return value.strip()


def safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except TypeError:
        pass
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def clean_number(value: Any) -> float | None:
    number = safe_float(value)
    if number is None:
        return None
    return round(float(number), 2)


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
        elif normalized_word in normalized:
            return True
    return False


def item_text(item: dict[str, Any]) -> str:
    return " ".join(
        [
            str(item.get("item_name") or ""),
            str(item.get("category") or ""),
            str(item.get("restaurant_name") or ""),
        ]
    )


def product_text(item: dict[str, Any]) -> str:
    # Ürün filtresinde restoran adını kullanmıyoruz. Yoksa "Manville Burger" içindeki
    # burger kelimesi, "Patates Kızartması" gibi yan ürünleri protein sanabiliyor.
    return " ".join([str(item.get("item_name") or ""), str(item.get("category") or "")])


def is_promo_or_code_product(item: dict[str, Any]) -> bool:
    raw_name = str(item.get("item_name") or "").strip()
    normalized = normalize_text(raw_name)
    if has_any(normalized, PROMO_WORDS):
        return True
    if re.fullmatch(r"[a-z]+\d+[a-z0-9]*", normalized):
        return True
    if re.fullmatch(r"[a-z0-9]{8,}", normalized) and any(char.isdigit() for char in normalized):
        return True
    return False


def infer_intent(query: str) -> dict[str, Any]:
    normalized = normalize_text(query)
    categories = []
    for category, keywords in CATEGORY_KEYWORDS.items():
        if has_any(normalized, keywords):
            categories.append(category)

    numbers = re.findall(r"\d+", normalized)
    budget_max = None
    if numbers:
        possible_budget = max(int(number) for number in numbers)
        if 50 <= possible_budget <= 3000:
            budget_max = possible_budget
    if budget_max is None and has_any(normalized, ["ucuz", "uygun", "butce", "bütçe", "pahali olmasin"]):
        budget_max = 250

    if has_any(normalized, ["spor", "protein", "proteinli", "yuksek protein", "yüksek protein"]):
        preference = "protein"
    elif has_any(normalized, ["cok ac", "çok aç", "acim", "açım", "doyurucu", "buyuk", "büyük"]):
        preference = "filling"
    elif has_any(normalized, ["ucuz", "uygun", "pahali olmasin", "pahalı olmasın"]):
        preference = "cheap"
    elif has_any(normalized, ["saglikli", "sağlıklı", "hafif", "fit"]):
        preference = "healthy"
    elif has_any(normalized, ["tatli", "tatlı", "sweet"]):
        preference = "sweet"
    else:
        preference = "balanced"

    return {
        "raw_query": query,
        "normalized_query": normalized,
        "categories": categories,
        "budget_max": budget_max,
        "preference": preference,
    }


class RecommenderEngine:
    def __init__(self, model: SentenceTransformer, items: list[dict[str, Any]], embeddings: np.ndarray, faiss_index: Any | None):
        self.model = model
        self.items = items
        self.embeddings = embeddings.astype("float32")
        self.faiss_index = faiss_index

    def search(self, query: str, top_k: int = 200) -> list[tuple[dict[str, Any], float]]:
        if not self.items:
            return []
        top_k = min(top_k, len(self.items))
        query_vector = self.model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype("float32")

        if self.faiss_index is not None:
            scores, indices = self.faiss_index.search(query_vector, top_k)
            results = []
            for index, score in zip(indices[0], scores[0]):
                if index < 0:
                    continue
                results.append((self.items[int(index)], float(score)))
            return results

        scores = self.embeddings @ query_vector[0]
        best_indices = np.argsort(scores)[::-1][:top_k]
        return [(self.items[int(index)], float(scores[int(index)])) for index in best_indices]


def load_recommender() -> RecommenderEngine:
    if not METADATA_PATH.exists() or not EMBEDDINGS_PATH.exists():
        raise RecommenderNotReadyError("Semantic artifacts bulunamadı. Önce python .\\src\\build_index.py çalıştırılmalı.")

    config = {}
    if CONFIG_PATH.exists():
        with CONFIG_PATH.open("r", encoding="utf-8") as file:
            config = json.load(file)

    model_name = config.get("model_name", DEFAULT_MODEL_NAME)
    model = SentenceTransformer(model_name)

    with METADATA_PATH.open("r", encoding="utf-8") as file:
        items = json.load(file)

    embeddings = np.load(EMBEDDINGS_PATH).astype("float32")
    faiss_index = None

    if FAISS_INDEX_PATH.exists():
        try:
            import faiss
            try:
                faiss_index = faiss.read_index(str(FAISS_INDEX_PATH))
            except Exception:
                with FAISS_INDEX_PATH.open("rb") as file:
                    index_bytes = np.frombuffer(file.read(), dtype="uint8")
                faiss_index = faiss.deserialize_index(index_bytes)
        except Exception:
            faiss_index = None

    return RecommenderEngine(model=model, items=items, embeddings=embeddings, faiss_index=faiss_index)


def should_drop_item(item: dict[str, Any], intent: dict[str, Any], hour: int | None, allergies: list[str], diet: str | None) -> bool:
    text = item_text(item)
    product = product_text(item)
    normalized_category = normalize_text(item.get("category"))
    normalized_diet = normalize_text(diet)
    normalized_allergies = [normalize_text(allergy) for allergy in allergies or []]
    preference = intent.get("preference")
    categories = intent.get("categories") or []

    if is_promo_or_code_product(item):
        return True

    if hour is not None and 6 <= int(hour) <= 11 and has_any(product, HEAVY_MORNING_WORDS):
        return True

    if "laktoz" in normalized_allergies or "lactose" in normalized_allergies:
        if normalized_category in LACTOSE_CATEGORIES or has_any(product, LACTOSE_WORDS):
            return True

    if normalized_diet in ["vegan", "vejetaryen", "vegetarian"]:
        if normalized_category in MEAT_CATEGORIES:
            return True
        if has_any(product, MEAT_WORDS) or has_any(product, SEAFOOD_WORDS):
            return True

    # Vegan ek olarak süt ve yumurta içeren ürünleri de dışlar (vejetaryen sütü kabul eder).
    if normalized_diet == "vegan":
        if normalized_category in LACTOSE_CATEGORIES:
            return True
        if has_any(product, LACTOSE_WORDS) or has_any(product, EGG_WORDS):
            return True

    if "icecek" not in categories and (normalized_category in ["icecek", "içecek"] or has_any(product, DRINK_WORDS)):
        return True

    if has_any(product, ADDON_WORDS):
        return True

    if preference == "protein":
        if has_any(product, DESSERT_WORDS) or has_any(product, DRINK_WORDS) or has_any(product, SIDE_DISH_WORDS):
            return True
        if not has_any(product, PROTEIN_WORDS):
            return True

    if preference == "filling":
        if has_any(product, DESSERT_WORDS) or has_any(product, DRINK_WORDS):
            return True
        if not has_any(product, FILLING_WORDS):
            return True

    if preference == "healthy":
        if has_any(product, DESSERT_WORDS) or has_any(product, DRINK_WORDS):
            return True
        if not has_any(product, HEALTHY_WORDS):
            return True

    return False


def calculate_business_score(item: dict[str, Any], semantic_score: float, intent: dict[str, Any], hour: int | None) -> float:
    text = product_text(item)
    price = clean_number(item.get("price")) or 0.0
    discount = clean_number(item.get("discount_rate")) or 0.0
    preference = intent.get("preference")
    budget_max = clean_number(intent.get("budget_max"))

    score = semantic_score * 0.70

    # Kategori eşleşmesi: sorgu "tavuk" ise tavuk ürünlerini güçlü şekilde öne çıkar.
    # Tek kelimelik kategori sorgularında semantik benzerlik zayıf kaldığı için kritik.
    categories = intent.get("categories") or []
    if categories:
        item_category = normalize_text(item.get("category"))
        matched = False
        for category in categories:
            keywords = CATEGORY_KEYWORDS.get(category, [category])
            if has_any(text, keywords) or normalize_text(category) == item_category:
                matched = True
                break
        if matched:
            score += 0.45

    # GENEL literal eşleşme: kullanıcının yazdığı kelime(ler) ürün adında geçiyorsa öne çıkar.
    # Kategori listesinde OLMAYAN her kelime için de çalışır (mantı, iskender, sushi, kumpir...).
    # Böylece "her yemeğe ayrı kural yazma" derdi ortadan kalkar.
    query_tokens = [
        token for token in normalize_text(intent.get("normalized_query")).split()
        if len(token) >= 3 and token not in QUERY_STOPWORDS
    ]
    if query_tokens:
        item_name_norm = normalize_text(item.get("item_name"))
        literal_hits = sum(1 for token in query_tokens if token in item_name_norm)
        if literal_hits:
            score += min(literal_hits, 2) * 0.35

    if budget_max is not None and price <= budget_max:
        score += 0.12

    if preference == "cheap":
        score += max(0.0, 1.0 - min(price, 600.0) / 600.0) * 0.16
    elif preference == "protein":
        if has_any(text, PROTEIN_WORDS):
            score += 0.22
        if 80 <= price <= 450:
            score += 0.08
    elif preference == "filling":
        if has_any(text, FILLING_WORDS):
            score += 0.18
        if 80 <= price <= 450:
            score += 0.10
    elif preference == "healthy":
        if has_any(text, HEALTHY_WORDS):
            score += 0.18

    if discount > 0:
        score += min(discount, 40.0) / 100.0 * 0.12

    if hour is not None:
        if 11 <= int(hour) <= 14 and has_any(text, FILLING_WORDS):
            score += 0.04
        if 18 <= int(hour) <= 23 and has_any(text, FILLING_WORDS + PROTEIN_WORDS):
            score += 0.05

    return round(float(score), 4)


def build_reason(item: dict[str, Any], semantic_score: float, final_score: float, hour: int | None, allergies: list[str], diet: str | None) -> str:
    parts = [f"semantik skor {semantic_score:.2f}", f"final skor {final_score:.2f}"]
    price = clean_number(item.get("price"))
    discount = clean_number(item.get("discount_rate"))
    if price is not None:
        parts.append(f"{price:.2f} TL")
    if discount is not None and discount > 0:
        parts.append(f"%{discount:.1f} indirim")
    if hour is not None:
        parts.append(f"saat {hour} bağlamı dikkate alındı")
    if allergies:
        parts.append("alerjen filtreleri uygulandı")
    if diet:
        parts.append(f"diyet tercihi: {diet}")
    return ", ".join(parts) + "."


def build_candidate_rows(raw_results: list[tuple[dict[str, Any], float]], intent: dict[str, Any], hour: int | None, allergies: list[str], diet: str | None) -> list[dict[str, Any]]:
    rows = []
    for item, semantic_score in raw_results:
        if should_drop_item(item, intent=intent, hour=hour, allergies=allergies, diet=diet):
            continue

        price = clean_number(item.get("price"))
        if price is None or price <= 0:
            continue

        final_score = calculate_business_score(item, semantic_score, intent, hour)
        row = {
            "platform": item.get("platform"),
            "city": item.get("city") or "bursa",
            "restaurant_name": item.get("restaurant_name"),
            "restaurant_rating": clean_number(item.get("restaurant_rating")),
            "category": item.get("category") or "Genel",
            "item_name": item.get("item_name"),
            "price": price,
            "original_price": clean_number(item.get("original_price")),
            "discount_rate": clean_number(item.get("discount_rate")),
            "product_url": item.get("product_url"),
            "image_url": item.get("image_url") or "",
            "semantic_score": round(float(semantic_score), 4),
            "score": final_score,
            "reason": build_reason(item, semantic_score, final_score, hour, allergies, diet),
        }
        row["restaurant_key"] = normalize_text(row["restaurant_name"])
        row["item_key"] = normalize_text(row["item_name"])
        rows.append(row)
    return rows


def group_platform_prices(rows: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    if not rows:
        return []

    df = pd.DataFrame(rows)
    df = df.sort_values(["restaurant_key", "item_key", "platform", "price", "score"], ascending=[True, True, True, True, False])
    df = df.drop_duplicates(subset=["restaurant_key", "item_key", "platform"], keep="first")

    grouped_items = []
    for _, group in df.groupby(["restaurant_key", "item_key"], sort=False):
        group_by_price = group.sort_values("price", ascending=True)
        best_price_row = group_by_price.iloc[0]
        best_score_row = group.sort_values("score", ascending=False).iloc[0]

        platforms = []
        for _, row in group_by_price.iterrows():
            platforms.append(
                {
                    "name": row.get("platform"),
                    "price": clean_number(row.get("price")),
                    "original_price": clean_number(row.get("original_price")),
                    "discount_rate": clean_number(row.get("discount_rate")),
                    "url": row.get("product_url"),
                }
            )

        prices = [platform["price"] for platform in platforms if platform["price"] is not None]
        min_price = min(prices) if prices else None
        max_price = max(prices) if prices else None
        price_gap = round(max_price - min_price, 2) if min_price is not None and max_price is not None else None
        max_score = round(float(group["score"].max()), 4)
        max_semantic_score = round(float(group["semantic_score"].max()), 4)

        grouped_items.append(
            {
                "sort_price": min_price if min_price is not None else float("inf"),
                "sort_score": max_score,
                "sehir": {"ad": best_price_row.get("city")},
                "restoran": {
                    "ad": best_price_row.get("restaurant_name"),
                    "puan": clean_number(best_price_row.get("restaurant_rating")),
                },
                "kategori": {"ad": best_price_row.get("category")},
                "urun": {
                    "ad": best_price_row.get("item_name"),
                    "fiyat": min_price,
                    "orijinal_fiyat": clean_number(best_price_row.get("original_price")),
                    "indirim_yuzdesi": clean_number(best_price_row.get("discount_rate")),
                    "gorsel_url": best_price_row.get("image_url") or "",
                    "urun_url": best_price_row.get("product_url"),
                    "musait_mi": True,
                    "platforms": platforms,
                    "platform_sayisi": len(platforms),
                    "en_ucuz_platform": platforms[0]["name"] if platforms else None,
                    "fiyat_farki": price_gap,
                },
                "ai": {
                    "skor": max_score,
                    "semantic_skoru": max_semantic_score,
                    "neden": best_score_row.get("reason"),
                },
            }
        )

    grouped_items = sorted(grouped_items, key=lambda item: (-item["sort_score"], item["sort_price"]))

    # Çeşitlilik: tek bir zincir (ör. Komagene'nin tüm şubeleri) ya da aynı isimli ürün
    # sonuçları domine etmesin. Restoran adındaki ", Şube/Mahalle" eki atılıp ZİNCİR
    # bazında sayılır; aynı isimli ürün bir kez gösterilir. Limit dolmazsa kalanlardan tamamlanır.
    max_per_chain = 2
    primary, overflow = [], []
    chain_counts: dict[str, int] = {}
    seen_names: set[str] = set()
    for item in grouped_items:
        name_key = normalize_text(item.get("urun", {}).get("ad"))
        chain_key = normalize_text((item.get("restoran", {}).get("ad") or "").split(",")[0])
        if (name_key and name_key in seen_names) or chain_counts.get(chain_key, 0) >= max_per_chain:
            overflow.append(item)
            continue
        seen_names.add(name_key)
        chain_counts[chain_key] = chain_counts.get(chain_key, 0) + 1
        primary.append(item)
    grouped_items = (primary + overflow)[:limit]

    for index, item in enumerate(grouped_items, start=1):
        item["sira"] = index
        item.pop("sort_price", None)
        item.pop("sort_score", None)

    return grouped_items


def recommend(
    engine: RecommenderEngine,
    query: str,
    limit: int = 10,
    hour: int | None = None,
    allergies: list[str] | None = None,
    diet: str | None = None,
) -> dict[str, Any]:
    allergies = allergies or []
    intent = infer_intent(query)
    search_query = query
    # Sorguda kategori tespit edildiyse (ör. "tavuk", "pizza") embedding'i o yöne
    # çekmek için kategori anahtar kelimelerini sorguya ekle.
    for category in intent.get("categories", []):
        keywords = CATEGORY_KEYWORDS.get(category, [])
        if keywords:
            search_query += " " + " ".join(keywords[:3])
    if intent.get("preference") == "protein":
        search_query += " tavuk protein pilav ızgara et"
    elif intent.get("preference") == "filling":
        search_query += " doyurucu menü ana yemek"
    elif intent.get("preference") == "healthy":
        search_query += " sağlıklı hafif fit ızgara salata"

    raw_results = engine.search(query=search_query, top_k=max(limit * 30, 200))

    # HİBRİT RETRIEVAL: ürün adında sorgu kelimesi geçen ürünleri de aday havuzuna ekle.
    # Zayıf embedding'li tek kelimeler (kumpir, mantı, iskender...) semantik top-K'ya
    # giremese bile, veride varsa bu yolla kesin yakalanır. (Sonsuz keyword yazma derdi yok.)
    query_tokens = [
        token for token in normalize_text(query).split()
        if len(token) >= 3 and token not in QUERY_STOPWORDS
    ]
    if query_tokens:
        seen_keys = {
            (normalize_text(it.get("platform")), normalize_text(it.get("restaurant_name")),
             normalize_text(it.get("item_name")))
            for it, _ in raw_results
        }
        for it in engine.items:
            name_norm = normalize_text(it.get("item_name"))
            if any(token in name_norm for token in query_tokens):
                key = (normalize_text(it.get("platform")),
                       normalize_text(it.get("restaurant_name")), name_norm)
                if key not in seen_keys:
                    seen_keys.add(key)
                    raw_results.append((it, 0.45))  # makul taban semantik skoru

    rows = build_candidate_rows(raw_results, intent=intent, hour=hour, allergies=allergies, diet=diet)

    # ALAKA FİLTRESİ: sorguda anlamlı kelime varsa ve adında/kategorisinde gerçekten
    # eşleşen ürünler bulunuyorsa, sadece onları göster (ör. "iskender" -> sadece iskender).
    # Hiç eşleşme yoksa (veride olmayan yemek) semantik sonuçlara düşülür; boş ekran olmaz.
    detected_categories = intent.get("categories") or []
    # Filtreyi YALNIZCA bilinen kategori sorgularında uygula (tavuk, pizza, tatlı...).
    # "acıktım" gibi belirsiz kelimelerde / serbest cümlelerde filtre uygulanmaz:
    # hibrit retrieval + boost zaten alakalı ürünü öne çıkarır, çeşitlilik korunur.
    apply_relevance = bool(detected_categories)
    if apply_relevance:
        def _row_relevant(row):
            name = normalize_text(row.get("item_name"))
            if any(token in name for token in query_tokens):
                return True
            row_category = normalize_text(row.get("category"))
            for category in detected_categories:
                keywords = CATEGORY_KEYWORDS.get(category, [category])
                if has_any(name, keywords) or normalize_text(category) == row_category:
                    return True
            return False

        relevant_rows = [row for row in rows if _row_relevant(row)]
        if relevant_rows:
            rows = relevant_rows

    grouped = group_platform_prices(rows, limit=limit)

    return {
        "tip": "ai_grouped_recommendation_response",
        "engine": "sentence_transformer_faiss_lifespan",
        "intent": intent,
        "filters": {
            "hour": hour,
            "allergies": allergies,
            "diet": diet,
        },
        "toplam_oneri": len(grouped),
        "oneriler": grouped,
        "recommendations": grouped,
    }


if __name__ == "__main__":
    import sys
    query = " ".join(sys.argv[1:]).strip() or "pizza öner"
    engine = load_recommender()
    output = recommend(engine=engine, query=query, limit=10, hour=19, allergies=[], diet="Standart")
    print(json.dumps(output, ensure_ascii=False, indent=2))
