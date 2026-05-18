from pathlib import Path

path = Path("src/homepage_recommender.py")
text = path.read_text(encoding="utf-8")

old = '''def has_any(text, words):
    normalized = normalize_text(text)
    return any(normalize_text(word) in normalized for word in words)


def item_matches_category(row, category):
    item_text = row.get("item_text", "")
    restaurant_text = row.get("restaurant_text", "")

    keywords = CATEGORY_KEYWORDS.get(category, [category])
    keywords = [normalize_text(keyword) for keyword in keywords]

    return any(keyword in item_text or keyword in restaurant_text for keyword in keywords)
'''

new = '''def tokenize(text):
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
'''

if old not in text:
    raise RuntimeError("has_any / item_matches_category bloğu bulunamadı.")

text = text.replace(old, new)

old = '''BAD_HOMEPAGE_WORDS = [
    "sos",
    "ketçap",
    "ketcap",
    "mayonez",
    "peçete",
    "pecete",
    "detaylar",
    "ekstra",
]
'''

new = '''BAD_HOMEPAGE_WORDS = [
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
'''

if old not in text:
    raise RuntimeError("BAD_HOMEPAGE_WORDS bloğu bulunamadı.")

text = text.replace(old, new)

old = '''def is_clean_homepage_item(row):
    item_name = row.get("item_name", "")
    price = row.get("price")

    if is_addon_item(item_name):
        return False

    if has_any(item_name, BAD_HOMEPAGE_WORDS):
        return False

    if pd.isna(price):
        return False

    if price <= 20 or price > 1500:
        return False

    return True
'''

new = '''def is_clean_homepage_item(row):
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
'''

if old not in text:
    raise RuntimeError("is_clean_homepage_item bloğu bulunamadı.")

text = text.replace(old, new)

old = '''    healthy_df = df[
        df.apply(lambda row: has_any(row.get("item_text", "") + " " + row.get("restaurant_text", ""), HEALTHY_WORDS), axis=1)
    ].copy()
'''

new = '''    healthy_df = df[
        df.apply(lambda row: has_any(row.get("item_text", ""), HEALTHY_WORDS), axis=1)
    ].copy()
'''

if old not in text:
    raise RuntimeError("healthy_df bloğu bulunamadı.")

text = text.replace(old, new)

path.write_text(text, encoding="utf-8")

print("Homepage recommender kalite filtreleri güncellendi.")
