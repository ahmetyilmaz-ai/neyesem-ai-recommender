"""
Sahte / şüpheli indirim tespiti — PİYASA REFERANSLI.

Kavram: Gerçek "sahte indirim", bir restoranın üstü-çizili ESKİ FİYATINI şişirip
göstermelik indirim sunmasıdır ("350'ydi 297 oldu!" ama ürün hiç 350 değildi).
Bunu dürüstçe yargılamanın yolu, o ürünün PİYASA FİYATINI bilmektir: aynı ürünün
(boyut-duyarlı) tüm ilanlarının MEDYANI = gerçekçi piyasa fiyatı.

Ana sinyaller:
  1) Şişirilmiş referans : üstü-çizili "eski fiyat" >> piyasa medyanı.
  2) İndirimli ama pahalı: "indirimli" fiyat bile piyasa medyanının üstünde.
Yan sinyaller:
  3) Absürt markup (eski/yeni >= 3x) ve  4) gerçekçi olmayan indirim oranı.

Not: "Başka platformda daha ucuz" tek başına SAHTE indirim DEĞİLDİR (o restoranın
indirimi kendi içinde gerçek olabilir); bu yüzden o bilgi yalnızca DESTEKLEYİCİ
kanıt olarak taşınır, ana karar piyasa medyanına dayanır.
"""

import re
from collections import defaultdict
from statistics import median
from typing import Any


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _normalize(value: Any) -> str:
    value = str(value or "").lower().strip()
    for src, dst in {"ı": "i", "ğ": "g", "ü": "u", "ş": "s", "ö": "o", "ç": "c"}.items():
        value = value.replace(src, dst)
    value = re.sub(r"[^a-z0-9\s]", " ", value)
    # ÖNEMLİ: boyut/birim (1 lt, 330 ml, 500 gr) ürün KİMLİĞİNİN parçası — silinirse
    # "Ayran 1lt" ile "Ayran 175ml" aynı sanılır ve sahte fiyat farkı uydurulur.
    # Birimi kanonik hale getir (boşluğu kaldır) ki "1 lt" == "1lt" eşleşsin ama 1lt != 175ml.
    value = re.sub(r"\b(\d+)\s*(gr|g|ml|cl|lt|l|adet|kg|cm)\b", r"\1\2", value)
    # Sadece pazarlama/porsiyon sıfatlarını sadeleştir (boyut sayısını DEĞİL).
    value = re.sub(r"\b(buyuk|kucuk|orta|aile|boyu|boy|mega|jumbo|tek|cift|porsiyon|menu|menusu)\b", " ", value)
    return re.sub(r"\s+", " ", value).strip()


# Piyasa medyanı güvenilir olsun diye o üründen en az bu kadar ilan gerek.
MIN_MARKET_SAMPLE = 4
# Üstü-çizili eski fiyat, piyasa medyanının bu katından fazlaysa "şişirilmiş referans".
INFLATED_REF_FACTOR = 1.6
# "İndirimli" fiyat bile piyasa medyanının bu katından fazlaysa indirim göstermelik.
STILL_ABOVE_FACTOR = 1.15
# Aynı ürün başka platformda en fazla bu kadar ucuz olabilir (destekleyici kanıt sınırı);
# ötesi farklı boyut/ürün eşleşmesidir.
MAX_CROSS_PLATFORM_RATIO = 2.5

ABSURD_MARKUP_FACTOR = 3.0    # eski / indirimli oranı
IMPLAUSIBLE_DISCOUNT = 70.0   # % üstü şüpheli (yemekte nadiren gerçek)

# Zayıf-tek-sinyalleri elemek için: en az bu skora ulaşan ilanlar raporlanır.
MIN_REPORT_SCORE = 1.5


def detect_suspicious_discounts(items: list[dict[str, Any]], limit: int = 20) -> dict[str, Any]:
    # Aynı ürünün (boyut-duyarlı normalize ad) tüm ilanlarını topla → piyasa referansı.
    product_listings: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in items:
        price = _safe_float(item.get("price"))
        if price and price > 0:
            product_listings[_normalize(item.get("item_name"))].append(
                {"price": price, "platform": item.get("platform"),
                 "restaurant": item.get("restaurant_name")}
            )

    # Her ürün için piyasa medyanı + örnek sayısı (gerçekçi "asıl fiyat").
    market: dict[str, dict[str, Any]] = {}
    for key, listings in product_listings.items():
        prices = [lst["price"] for lst in listings]
        market[key] = {"median": median(prices), "count": len(prices)}

    flagged: list[dict[str, Any]] = []

    for item in items:
        price = _safe_float(item.get("price"))
        original = _safe_float(item.get("original_price"))
        discount = _safe_float(item.get("discount_rate"))
        if price is None or price <= 0:
            continue

        claims_discount = (discount is not None and discount > 0) or (
            original is not None and original > price
        )
        if not claims_discount:
            continue

        name_key = _normalize(item.get("item_name"))
        ref = market.get(name_key, {})
        med = ref.get("median")
        sample = ref.get("count", 0)
        reliable_market = med is not None and med > 0 and sample >= MIN_MARKET_SAMPLE

        reasons: list[str] = []
        score = 0.0

        # 1) ŞİŞİRİLMİŞ REFERANS: üstü-çizili eski fiyat piyasanın çok üstünde.
        if reliable_market and original is not None and original >= med * INFLATED_REF_FACTOR:
            reasons.append(
                f"'Eski fiyat' {original:.0f} TL gösteriliyor ama bu ürünün piyasa "
                f"fiyatı ~{med:.0f} TL — referans fiyat şişirilmiş, indirim göstermelik."
            )
            score += 2.0 + min((original / med) - 1.0, 3.0)

        # 2) İNDİRİMLİ AMA HÂLÂ PAHALI: indirimli fiyat bile piyasanın üstünde.
        if reliable_market and price >= med * STILL_ABOVE_FACTOR:
            reasons.append(
                f"'İndirimli' fiyat {price:.0f} TL, piyasa medyanı ~{med:.0f} TL'nin "
                f"üstünde — indirim gerçek bir avantaj sağlamıyor."
            )
            score += 1.5 + min((price / med) - 1.0, 2.0)

        # 3) ABSÜRT MARKUP (piyasa verisi olmadan da çalışır)
        if original is not None and original > price * ABSURD_MARKUP_FACTOR:
            reasons.append(
                f"Eski fiyat ({original:.0f} TL) indirimli fiyatın {original / price:.1f} "
                f"katı — şişirilmiş referans fiyat işareti."
            )
            score += 1.0

        # 4) Gerçekçi olmayan indirim oranı (zayıf yan sinyal)
        if discount is not None and discount >= IMPLAUSIBLE_DISCOUNT:
            reasons.append(f"%{discount:.0f} indirim yemek için gerçekçi değil.")
            score += 0.5

        # DESTEKLEYİCİ KANIT (karar verici değil): aynı ürün başka platformda daha ucuz mu?
        cheaper_alt = None
        listings = product_listings.get(name_key, [])
        if len({lst["platform"] for lst in listings}) >= 2:
            elsewhere = [
                lst for lst in listings
                if lst["platform"] != item.get("platform")
                and lst["price"] < price
                and lst["price"] >= price / MAX_CROSS_PLATFORM_RATIO
            ]
            if elsewhere:
                cheaper_alt = min(elsewhere, key=lambda x: x["price"])

        if reasons and score >= MIN_REPORT_SCORE:
            flagged.append(
                {
                    "platform": item.get("platform"),
                    "restaurant_name": item.get("restaurant_name"),
                    "item_name": item.get("item_name"),
                    "category": item.get("category"),
                    "price": round(price, 2),
                    "original_price": round(original, 2) if original else None,
                    "discount_rate": round(discount, 2) if discount else None,
                    "product_url": item.get("product_url"),
                    # Piyasa referansı (mobil "Piyasa ~X TL" diye gösterebilir)
                    "market_median": round(med, 2) if reliable_market else None,
                    "market_sample": sample,
                    # Destekleyici cross-platform kanıt
                    "cheaper_platform": cheaper_alt["platform"] if cheaper_alt else None,
                    "cheaper_price": round(cheaper_alt["price"], 2) if cheaper_alt else None,
                    "suspicion_score": round(score, 2),
                    "reasons": reasons,
                }
            )

    flagged.sort(key=lambda row: row["suspicion_score"], reverse=True)

    return {
        "type": "suspicious_discount_report",
        "description": "Piyasa fiyatına göre yanıltıcı/şişirilmiş görünen indirimler.",
        "total_scanned": len(items),
        "suspicious_count": len(flagged),
        "items": flagged[:limit],
    }
