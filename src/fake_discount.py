"""
Sahte / şüpheli indirim tespiti.

Bir indirim "şüpheli" sayılır çünkü gösterilen tasarruf gerçek değildir. Üç sinyal:

1. Şişirilmiş orijinal fiyat: orijinal fiyat, aynı kategorideki ürünlerin medyan
   fiyatının çok üstündeyken indirimli fiyat hâlâ ortalama civarındaysa -> sahte
   referans fiyat.
2. Absürt markup: orijinal fiyat, indirimli fiyatın anlamsız bir katıysa
   (ör. 5×) -> büyük olasılıkla uydurma.
3. Gerçekçi olmayan indirim oranı: yemekte %70+ indirim genelde sahte/veri hatası.

Kategori medyanı, "platformlar/restoranlar arası" ortak fiyat referansı görevi görür:
tek bir restoranın şişirdiği fiyat, kategorinin geneline göre yakalanır.
"""

import re
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
    return re.sub(r"\s+", " ", value).strip()


# Eşikler (yemek alan bilgisi).
INFLATED_ORIGINAL_FACTOR = 2.0   # orijinal fiyat kategori medyanının kaç katı ise şişirilmiş
STILL_AVERAGE_FACTOR = 0.9       # indirimli fiyat hâlâ medyanın bu oranı üstünde mi
ABSURD_MARKUP_FACTOR = 4.0       # orijinal / indirimli oranı
IMPLAUSIBLE_DISCOUNT = 70.0      # % üstü şüpheli


def detect_suspicious_discounts(items: list[dict[str, Any]], limit: int = 20) -> dict[str, Any]:
    # 1) Kategori medyan fiyatları (gerçek/indirimli fiyat üzerinden).
    by_category: dict[str, list[float]] = {}
    for item in items:
        price = _safe_float(item.get("price"))
        if price and price > 0:
            by_category.setdefault(_normalize(item.get("category")), []).append(price)

    category_median = {
        category: median(prices)
        for category, prices in by_category.items() if prices
    }

    flagged: list[dict[str, Any]] = []

    for item in items:
        price = _safe_float(item.get("price"))
        original = _safe_float(item.get("original_price"))
        discount = _safe_float(item.get("discount_rate"))

        if price is None or price <= 0:
            continue

        # Yalnızca indirim iddiası olan ürünleri değerlendir.
        claims_discount = (discount is not None and discount > 0) or (
            original is not None and original > price
        )
        if not claims_discount:
            continue

        reasons: list[str] = []
        score = 0.0
        med = category_median.get(_normalize(item.get("category")))

        if (
            original is not None and med
            and original > med * INFLATED_ORIGINAL_FACTOR
            and price >= med * STILL_AVERAGE_FACTOR
        ):
            reasons.append(
                f"Orijinal fiyat ({original:.0f} TL) kategori ortalamasının "
                f"({med:.0f} TL) çok üstünde; indirimli fiyat hâlâ ortalama civarında."
            )
            score += 2.0

        if original is not None and original > price * ABSURD_MARKUP_FACTOR:
            reasons.append(
                f"Orijinal fiyat, indirimli fiyatın {original / price:.1f} katı — "
                f"şişirilmiş referans fiyat olabilir."
            )
            score += 1.5

        if discount is not None and discount >= IMPLAUSIBLE_DISCOUNT:
            reasons.append(f"%{discount:.0f} indirim yemek için gerçekçi değil.")
            score += 1.0

        if reasons:
            flagged.append(
                {
                    "platform": item.get("platform"),
                    "restaurant_name": item.get("restaurant_name"),
                    "item_name": item.get("item_name"),
                    "category": item.get("category"),
                    "price": round(price, 2),
                    "original_price": round(original, 2) if original else None,
                    "discount_rate": round(discount, 2) if discount else None,
                    "category_median_price": round(med, 2) if med else None,
                    "product_url": item.get("product_url"),
                    "suspicion_score": round(score, 2),
                    "reasons": reasons,
                }
            )

    flagged.sort(key=lambda row: row["suspicion_score"], reverse=True)

    return {
        "type": "suspicious_discount_report",
        "description": "Kategori medyan fiyatına göre şişirilmiş/sahte görünen indirimler.",
        "total_scanned": len(items),
        "suspicious_count": len(flagged),
        "items": flagged[:limit],
    }
