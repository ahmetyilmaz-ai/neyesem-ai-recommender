"""
Sahte / şüpheli indirim tespiti — PLATFORMLAR ARASI temelli.

Asıl sinyal: bir platform "%X indirim!" diyor ama AYNI ürün başka platformda,
indirimsiz, bu "indirimli" fiyattan daha ucuza satılıyorsa indirim yanıltıcıdır.
(Kategori ortalaması kıyası yanıltıcıydı: aile boyu/premium ürünler normal indirimle
bile "şüpheli" görünüyordu — o yaklaşım bırakıldı.)

Ek sinyaller: absürt markup (orijinal >> indirimli) ve gerçekçi olmayan indirim oranı.
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
    # boyut/birim ekleri (aile boyu, 500 gr, 1 lt...) ürün kimliğini bozmasın diye sadeleştir
    value = re.sub(r"\b\d+\s*(gr|g|ml|cl|lt|l|adet|kg|cm)\b", " ", value)
    value = re.sub(r"\b(buyuk|kucuk|orta|aile|boyu|boy|mega|jumbo|tek|cift|porsiyon|menu|menusu)\b", " ", value)
    return re.sub(r"\s+", " ", value).strip()


ABSURD_MARKUP_FACTOR = 3.0    # orijinal / indirimli oranı
IMPLAUSIBLE_DISCOUNT = 60.0   # % üstü şüpheli


def detect_suspicious_discounts(items: list[dict[str, Any]], limit: int = 20) -> dict[str, Any]:
    # Aynı ürünün (normalize ad) tüm platform/restoran fiyatlarını topla.
    product_listings: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in items:
        price = _safe_float(item.get("price"))
        if price and price > 0:
            product_listings[_normalize(item.get("item_name"))].append(
                {"price": price, "platform": item.get("platform"),
                 "restaurant": item.get("restaurant_name"), "item": item}
            )

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

        reasons: list[str] = []
        score = 0.0
        cheaper_alt = None

        # 1) PLATFORMLAR ARASI: aynı ürün başka platformda bu indirimli fiyattan ucuz mu?
        name_key = _normalize(item.get("item_name"))
        listings = product_listings.get(name_key, [])
        platforms = {lst["platform"] for lst in listings}
        if len(platforms) >= 2:
            elsewhere = [
                lst for lst in listings
                if lst["platform"] != item.get("platform") and lst["price"] < price
            ]
            if elsewhere:
                cheapest = min(elsewhere, key=lambda x: x["price"])
                cheaper_alt = cheapest
                reasons.append(
                    f"Aynı ürün {cheapest['platform']} platformunda indirimsiz "
                    f"{cheapest['price']:.0f} TL — bu '%{(discount or 0):.0f} indirim' "
                    f"aslında daha pahalı."
                )
                score += 3.0 + (price - cheapest["price"]) / max(price, 1)

        # 2) Absürt markup
        if original is not None and original > price * ABSURD_MARKUP_FACTOR:
            reasons.append(
                f"Orijinal fiyat ({original:.0f} TL) indirimli fiyatın "
                f"{original / price:.1f} katı — şişirilmiş referans fiyat olabilir."
            )
            score += 1.5

        # 3) Gerçekçi olmayan indirim oranı
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
                    "product_url": item.get("product_url"),
                    # cross-platform karşılaştırma bilgisi (mobil gösterebilir)
                    "cheaper_platform": cheaper_alt["platform"] if cheaper_alt else None,
                    "cheaper_price": round(cheaper_alt["price"], 2) if cheaper_alt else None,
                    "suspicion_score": round(score, 2),
                    "reasons": reasons,
                }
            )

    flagged.sort(key=lambda row: row["suspicion_score"], reverse=True)

    return {
        "type": "suspicious_discount_report",
        "description": "Platformlar arası kıyasla yanıltıcı/sahte görünen indirimler.",
        "total_scanned": len(items),
        "suspicious_count": len(flagged),
        "items": flagged[:limit],
    }
