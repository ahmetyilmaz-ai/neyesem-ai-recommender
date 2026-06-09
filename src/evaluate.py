"""
Öneri kalitesi değerlendirmesi — precision@k.

Etiketli bir sorgu seti üzerinde recommender çalıştırılır; her sorgu için dönen
ilk k sonucun ne kadarının "alakalı" olduğu ölçülür. Alaka, sonucun ürün adı veya
kategorisinin sorgunun beklenen anahtar kelimelerinden birini içermesiyle tanımlanır.

Çalıştırma:
    python -m src.evaluate
    python -m src.evaluate --k 5

Çıktı: sorgu bazında precision@k + genel ortalama (mean precision@k).
Bu, "öneri kalitesini nasıl ölçtük?" sorusunun nesnel cevabıdır.
"""

import argparse

try:
    from .recommender import load_recommender, recommend, normalize_text
except ImportError:
    from recommender import load_recommender, recommend, normalize_text


# Etiketli değerlendirme seti: her sorgu için "alakalı sayılacak" anahtar kelimeler.
EVAL_SET = [
    {"query": "tavuk",                 "relevant": ["tavuk", "chicken", "pilic", "kanat", "nugget"]},
    {"query": "pizza",                 "relevant": ["pizza"]},
    {"query": "burger",                "relevant": ["burger", "hamburger"]},
    {"query": "döner",                 "relevant": ["doner", "durum"]},
    {"query": "lahmacun",              "relevant": ["lahmacun"]},
    {"query": "mantı",                 "relevant": ["manti"]},
    {"query": "iskender",              "relevant": ["iskender", "doner", "kebap"]},
    {"query": "kebap",                 "relevant": ["kebap", "kebab", "sis", "adana", "urfa"]},
    {"query": "çorba",                 "relevant": ["corba", "soup"]},
    {"query": "salata",                "relevant": ["salata", "salad"]},
    {"query": "tatlı",                 "relevant": ["tatli", "waffle", "baklava", "dondurma", "pasta",
                                                     "sutlac", "kunefe", "tiramisu", "kurabiye", "magnolia", "kek"]},
    {"query": "kumpir",                "relevant": ["kumpir"]},
    {"query": "tantuni",               "relevant": ["tantuni"]},
    {"query": "kola",                  "relevant": ["kola", "cola"]},
    {"query": "ayran",                 "relevant": ["ayran"]},
    {"query": "yüksek proteinli",      "relevant": ["tavuk", "chicken", "et", "kofte", "doner", "kebap",
                                                     "protein", "pilav", "izgara", "burger", "balik", "somon"]},
    {"query": "ucuz doyurucu",         "relevant": ["durum", "doner", "kofte", "pide", "kumpir", "pilav",
                                                     "tavuk", "menu", "tantuni", "lahmacun", "kumru", "burger"]},
    {"query": "hafif sağlıklı",        "relevant": ["salata", "salad", "izgara", "bowl", "somon", "ton", "fit"]},
    {"query": "tatlı bir şey",         "relevant": ["tatli", "waffle", "baklava", "dondurma", "pasta",
                                                     "sutlac", "kunefe", "kurabiye", "kek", "magnolia"]},
    {"query": "çok acıktım doyurucu",  "relevant": ["durum", "doner", "kofte", "pide", "kumpir", "pilav",
                                                     "tavuk", "menu", "tantuni", "lahmacun", "burger", "et"]},
]


def is_relevant(item, relevant_keywords):
    text = normalize_text(item.get("urun", {}).get("ad")) + " " + \
           normalize_text(item.get("kategori", {}).get("ad"))
    return any(keyword in text for keyword in relevant_keywords)


def evaluate(engine, k=5):
    print(f"{'Sorgu':<26} {'Dönen':>5} {'Alakalı':>7} {'P@%d' % k:>7}")
    print("-" * 50)

    precisions = []
    for case in EVAL_SET:
        result = recommend(engine=engine, query=case["query"], limit=k)
        items = result.get("oneriler", [])
        returned = len(items)
        relevant = sum(1 for it in items if is_relevant(it, case["relevant"]))
        precision = relevant / returned if returned else 0.0
        precisions.append(precision)
        print(f"{case['query']:<26} {returned:>5} {relevant:>7} {precision:>7.2f}")

    mean_p = sum(precisions) / len(precisions) if precisions else 0.0
    print("-" * 50)
    print(f"{'ORTALAMA (mean precision@%d)' % k:<40} {mean_p:>7.2f}")
    print(f"\nDeğerlendirilen sorgu: {len(EVAL_SET)} | Ortalama precision@{k}: {mean_p:.1%}")
    return mean_p


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, default=5, help="precision@k için k")
    args = parser.parse_args()

    print("Recommender yükleniyor...\n")
    engine = load_recommender()
    evaluate(engine, k=args.k)


if __name__ == "__main__":
    main()
