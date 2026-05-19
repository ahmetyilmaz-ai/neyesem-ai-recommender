import json
import re
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer


ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT_DIR / "data" / "all_items.json"
ARTIFACTS_DIR = ROOT_DIR / "artifacts"

MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

METADATA_PATH = ARTIFACTS_DIR / "item_metadata.json"
EMBEDDINGS_PATH = ARTIFACTS_DIR / "item_embeddings.npy"
FAISS_INDEX_PATH = ARTIFACTS_DIR / "faiss_index.bin"
CONFIG_PATH = ARTIFACTS_DIR / "semantic_config.json"


CATEGORY_KEYWORDS = {
    "Pizza": ["pizza"],
    "Burger": ["burger", "hamburger"],
    "Döner": ["döner", "doner"],
    "Pide & Lahmacun": ["pide", "lahmacun"],
    "Tatlı": ["tatlı", "tatli", "waffle", "pasta", "baklava", "dondurma", "sütlaç", "sutlac", "kazandibi", "magnolia"],
    "Tavuk": ["tavuk", "chicken", "kanat"],
    "İçecek": ["su", "kola", "ayran", "ice tea", "fanta", "sprite", "limonata", "pepsi"],
    "Sağlıklı": ["salata", "fit", "bowl", "ızgara", "izgara"],
    "Kebap": ["kebap", "kebab", "şiş", "sis", "adana", "urfa"],
    "Çiğ Köfte": ["çiğ köfte", "cig kofte", "çiğköfte", "cigkofte"],
}

BAD_ITEM_WORDS = [
    "detaylar",
    "sepet",
    "minimum",
    "teslimat",
    "kampanya",
    "cookie",
    "çerez",
    "cerez",
    "şu anda",
    "su anda",
    "temsili",
    "anasayfa",
    "restoran",
    "filtre",
    "sırala",
    "sirala",
]

ADDON_WORDS = [
    "ketçap",
    "ketcap",
    "mayonez",
    "sos",
    "acı sos",
    "aci sos",
    "ranch",
    "barbekü",
    "barbeku",
    "cheddar sos",
    "kenar sos",
    "ekstra",
    "peçete",
    "pecete",
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


def to_float(value):
    if value is None:
        return None

    if isinstance(value, (int, float)):
        return float(value)

    try:
        return float(str(value).replace("TL", "").replace("₺", "").replace(",", ".").strip())
    except ValueError:
        return None


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


def infer_category(item_name):
    text = normalize_text(item_name)

    for category, keywords in CATEGORY_KEYWORDS.items():
        for keyword in keywords:
            if normalize_text(keyword) in text:
                return category

    return "Genel"


def build_semantic_text(item, category):
    item_name = item.get("item_name") or ""
    restaurant_name = item.get("restaurant_name") or ""
    platform = item.get("platform") or ""
    price = item.get("price")
    discount = item.get("discount_rate")

    parts = [
        f"Ürün: {item_name}",
        f"Restoran: {restaurant_name}",
        f"Platform: {platform}",
        f"Kategori: {category}",
    ]

    if price is not None:
        parts.append(f"Fiyat: {price} TL")

    if discount is not None:
        parts.append(f"İndirim: %{discount}")

    return ". ".join(parts)


def load_and_clean_items():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"{DATA_PATH} bulunamadı.")

    with DATA_PATH.open("r", encoding="utf-8") as file:
        raw_items = json.load(file)

    cleaned = []

    for index, item in enumerate(raw_items):
        item_name = item.get("item_name")

        if is_bad_item_name(item_name):
            continue

        price = to_float(item.get("price"))

        if price is None or price <= 0 or price > 2500:
            continue

        original_price = to_float(item.get("original_price"))
        discount_rate = to_float(item.get("discount_rate"))
        restaurant_rating = to_float(item.get("restaurant_rating"))

        category = infer_category(item_name)

        normalized_item = {
            "id": item.get("id") or item.get("item_id") or item.get("product_id") or f"item_{index}",
            "platform": item.get("platform"),
            "restaurant_name": item.get("restaurant_name"),
            "restaurant_rating": restaurant_rating,
            "item_name": item_name,
            "category": category,
            "price": price,
            "original_price": original_price,
            "discount_rate": discount_rate,
            "product_url": item.get("product_url"),
            "image_url": item.get("image_url") or "",
            "city": item.get("city") or "bursa",
            "item_text": normalize_text(item_name),
            "restaurant_text": normalize_text(item.get("restaurant_name")),
            "is_addon": is_addon_item(item_name),
        }

        normalized_item["semantic_text"] = build_semantic_text(normalized_item, category)

        cleaned.append(normalized_item)

    return cleaned


def main():
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    items = load_and_clean_items()

    if not items:
        raise RuntimeError("Indexlenecek ürün bulunamadı.")

    texts = [item["semantic_text"] for item in items]

    print(f"Model yükleniyor: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)

    print(f"{len(texts)} ürün embedding'e çevriliyor...")
    embeddings = model.encode(
        texts,
        batch_size=64,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype("float32")

    print(f"Embedding shape: {embeddings.shape}")

    np.save(EMBEDDINGS_PATH, embeddings)

    with METADATA_PATH.open("w", encoding="utf-8") as file:
        json.dump(items, file, ensure_ascii=False, indent=2)

    faiss_available = False

    try:
        import faiss

        index = faiss.IndexFlatIP(embeddings.shape[1])
        index.add(embeddings)
        faiss.write_index(index, str(FAISS_INDEX_PATH))
        faiss_available = True

        print(f"FAISS index kaydedildi: {FAISS_INDEX_PATH}")

    except Exception as exc:
        print(f"FAISS index oluşturulamadı, NumPy fallback kullanılacak: {exc}")

    config = {
        "model_name": MODEL_NAME,
        "item_count": len(items),
        "embedding_dim": int(embeddings.shape[1]),
        "faiss_available": faiss_available,
    }

    with CONFIG_PATH.open("w", encoding="utf-8") as file:
        json.dump(config, file, ensure_ascii=False, indent=2)

    print()
    print("Semantic index build tamamlandı.")
    print(f"Metadata: {METADATA_PATH}")
    print(f"Embeddings: {EMBEDDINGS_PATH}")
    print(f"Config: {CONFIG_PATH}")


if __name__ == "__main__":
    main()
