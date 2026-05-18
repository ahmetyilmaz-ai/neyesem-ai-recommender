from pathlib import Path

path = Path("src/homepage_recommender.py")
text = path.read_text(encoding="utf-8")

old = '''HEALTHY_WORDS = [
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
'''

new = '''HEALTHY_WORDS = [
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

HEALTHY_EXCLUDE_WORDS = [
    "doritos",
    "cips",
    "chips",
    "parti",
    "nachos",
    "patates kızartması",
    "patates kizartmasi",
    "kızartma",
    "kizartma",
    "çikolata",
    "cikolata",
    "dondurma",
    "baklava",
    "tatlı",
    "tatli",
    "profiterol",
    "mayonez",
    "sos",
    "amerikan salatası",
    "amerikan salatasi",
    "rus salatası",
    "rus salatasi",
]
'''

if old not in text:
    raise RuntimeError("HEALTHY_WORDS bloğu bulunamadı.")

text = text.replace(old, new)

old = '''    healthy_df = df[
        df.apply(lambda row: has_any(row.get("item_text", ""), HEALTHY_WORDS), axis=1)
    ].copy()

    healthy_df["homepage_score"] = healthy_df.apply(score_healthy, axis=1)
'''

new = '''    healthy_df = df[
        df.apply(lambda row: has_any(row.get("item_text", ""), HEALTHY_WORDS), axis=1)
    ].copy()

    healthy_df = healthy_df[
        ~healthy_df.apply(lambda row: has_any(row.get("item_text", ""), HEALTHY_EXCLUDE_WORDS), axis=1)
    ].copy()

    healthy_df["homepage_score"] = healthy_df.apply(score_healthy, axis=1)
'''

if old not in text:
    raise RuntimeError("healthy_df bloğu bulunamadı.")

text = text.replace(old, new)

path.write_text(text, encoding="utf-8")

print("Sağlıklı bölüm filtresi güncellendi.")
