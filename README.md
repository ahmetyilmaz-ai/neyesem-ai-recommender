# NeYesem AI Recommender

Bu repo, **NeYesem projesinin AI öneri ve karşılaştırma motorudur**.

Görev nettir: gerçek yemek verisini alır, semantic embedding ile anlamlandırır, kullanıcı isteğine göre öneri üretir ve fiyat karşılaştırması yapar.

Bu sistem ürün uydurmaz. Rastgele öneri basmaz. Elindeki `data/all_items.json` verisini kullanır. Kullanıcının yazdığı metni, fiyatı, indirimi, kategoriyi, saat bilgisini, şehir bilgisini, bütçeyi ve kullanıcı profilini birlikte değerlendirir.

---

## Sistem Ne Yapıyor?

### 1. Cold-start ana sayfa önerisi

Kullanıcı daha hiçbir şey yazmadan ana sayfa için öneri bölümleri üretir.

```txt
Uygun Fiyatlı Doyurucular
Pizza & Burger Önerileri
Tatlı Kaçamağı
İndirimli Fırsatlar
Hafif ve Sağlıklı Seçenekler
Bugünün Karışık Önerileri
```

Endpoint:

```http
GET /homepage?limit=8
```

### 2. Semantic AI önerisi

Kullanıcı serbest metin yazabilir.

```txt
pizza öner
çok açım ucuz doyurucu bir şey öner
spordan çıktım yüksek proteinli ucuz bir şey öner
hasta gibiyim sıcak çorba tarzı bir şey öner
hafif sağlıklı bir şey olsun
```

Sistem bunu kelime eşleşmesi gibi değil, anlam benzerliğiyle işler.

Endpoint:

```http
POST /recommend
```

### 3. Karşılaştırmalı öneri / fiyat analizi

Sistem sadece öneri vermez. Aynı zamanda aday ürünleri karşılaştırır.

Endpoint:

```http
POST /compare
```

Bu endpoint şunları döndürür:

```txt
price_analysis      -> min fiyat, max fiyat, ortalama fiyat, fiyat farkı
cheapest_items      -> en ucuz ürünler
best_value_items    -> fiyat + semantic skor + indirim dengesine göre en avantajlı ürünler
platform_comparison -> platform bazlı min/max/ortalama fiyat özeti
```

Bu yüzden sistem artık sadece "öneri motoru" değil, aynı zamanda **semantic price comparison engine** olarak çalışır.

---

## Kullanılan AI Mimarisi

Güncel mimari:

```txt
Hugging Face SentenceTransformer
+ Dense Embeddings
+ FAISS Vector Search
+ Business Rule Scoring
+ Context-Aware Ranking
+ User Profile Filtering
+ Price Comparison Layer
```

Kullanılan model:

```txt
sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
```

Ürünler offline olarak embedding'e çevrilir. Kullanıcı isteği geldiğinde sadece sorgu embedding'e çevrilir ve FAISS index üzerinden en yakın ürünler bulunur.

---

## Offline / Online Çalışma Mantığı

### Offline build

Veri güncellendiğinde çalışır.

```txt
data/all_items.json okunur
ürün metinleri temizlenir
semantic text oluşturulur
SentenceTransformer ile embedding çıkarılır
FAISS index oluşturulur
artifacts/ klasörüne kaydedilir
```

Komut:

```powershell
python .\src\build_index.py
```

Üretilen dosyalar:

```txt
artifacts/faiss_index.bin
artifacts/item_embeddings.npy
artifacts/item_metadata.json
artifacts/semantic_config.json
```

### Online inference

FastAPI açıldığında index RAM'e yüklenir.

Kullanıcı istek attığında:

```txt
query embedding'e çevrilir
FAISS en yakın ürünleri bulur
context ve user_profile kuralları uygulanır
öneri veya karşılaştırma sonucu döner
```

Ağır model işi her request'te yapılmaz. Ağır iş offline build aşamasındadır.

---

## Proje Yapısı

```txt
.
├── artifacts/
│   ├── faiss_index.bin
│   ├── item_embeddings.npy
│   ├── item_metadata.json
│   └── semantic_config.json
│
├── data/
│   ├── all_items.json
│   └── homepage_recommendations.json
│
├── src/
│   ├── __init__.py
│   ├── api.py
│   ├── build_index.py
│   ├── compare_engine.py
│   ├── db_output_adapter.py
│   ├── homepage_recommender.py
│   ├── recommender.py
│   └── semantic_recommender.py
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

## Dosyalar

### `src/build_index.py`

Offline index build scriptidir. `all_items.json` dosyasını okur, ürünleri temizler, embedding çıkarır ve FAISS index oluşturur.

```powershell
python .\src\build_index.py
```

### `src/semantic_recommender.py`

Ana semantic öneri motorudur. Kullanıcı sorgusunu embedding'e çevirir, FAISS ile aday ürünleri bulur, context ve profil kurallarıyla skorlar.

```powershell
python .\src\semantic_recommender.py "spordan çıktım yüksek proteinli ucuz bir şey öner"
```

### `src/compare_engine.py`

Karşılaştırma motorudur. Semantic recommender'dan geniş aday havuzu alır, fiyat analizi yapar, en ucuzları ve en avantajlı ürünleri hesaplar.

```powershell
python .\src\compare_engine.py "pizza karşılaştır"
```

### `src/homepage_recommender.py`

Ana sayfa cold-start önerileri üretir.

```powershell
python .\src\homepage_recommender.py --limit 8 --save
```

### `src/db_output_adapter.py`

AI çıktısını backend/DB tarafına daha uygun formata çevirir.

### `src/api.py`

FastAPI servisidir. Frontend ve backend buraya istek atar.

---

## Endpointler

```txt
GET  /
GET  /health
GET  /homepage
POST /recommend
POST /compare
GET  /db/homepage
POST /db/recommend
POST /db/compare
```

### Health check

```http
GET /health
```

### Ana sayfa

```http
GET /homepage?limit=8
```

### Semantic öneri

```http
POST /recommend
```

Request:

```json
{
  "query": "spordan çıktım yüksek proteinli ucuz bir şey öner",
  "limit": 5,
  "context": {
    "hour": 19,
    "day_type": "weekday",
    "city": "bursa"
  },
  "user_profile": {
    "diet": "Standart",
    "allergies": [],
    "disliked_items": [],
    "preferred_categories": ["tavuk"],
    "max_budget": 350
  }
}
```

### Karşılaştırma

```http
POST /compare
```

Request:

```json
{
  "query": "pizza karşılaştır",
  "limit": 5,
  "context": {
    "hour": 19,
    "day_type": "weekday",
    "city": "bursa"
  },
  "user_profile": {
    "diet": "Standart",
    "allergies": [],
    "disliked_items": [],
    "preferred_categories": [],
    "max_budget": 400
  }
}
```

Response ana alanları:

```txt
engine: semantic_compare
price_analysis
cheapest_items
best_value_items
platform_comparison
```

### DB/backend uyumlu endpointler

```http
GET  /db/homepage?limit=8
POST /db/recommend
POST /db/compare
```

DB tarafı bu endpointleri kullanabilir. Çıktılar platform, şehir, restoran, kategori, ürün ve AI skoru mantığına daha yakın döner.

---

## Kurulum

```powershell
python -m pip install -r .\requirements.txt
```

Ana paketler:

```txt
pandas
numpy
scikit-learn
joblib
fastapi
uvicorn
sentence-transformers
faiss-cpu
```

İlk kurulumdan sonra index build:

```powershell
python .\src\build_index.py
```

API çalıştırma:

```powershell
uvicorn src.api:app --reload
```

Swagger:

```txt
http://127.0.0.1:8000/docs
```

---

## Frontend Entegrasyonu

Chrome/Web için base URL:

```txt
http://127.0.0.1:8000
```

Android emulator için:

```txt
http://10.0.2.2:8000
```

Gerçek telefonda test için:

```powershell
uvicorn src.api:app --reload --host 0.0.0.0 --port 8000
```

Frontend ana sayfa için `/homepage`, arama/öneri kutusu için `/recommend`, karşılaştırma ekranı için `/compare` kullanmalıdır.

---

## Veri Temizleme ve Business Rules

Sistem çöp veriyi öneriye sokmamak için filtreler uygular.

Elenen veya cezalandırılan örnekler:

```txt
sos
ketçap
mayonez
peçete
yan ürünler
anlamsız scrape metinleri
protein sorgusunda tatlı/içecek/sos ürünleri
sağlıklı sorgusunda abur cubur ürünler
```

Context ve profil kuralları da skora yansır:

```txt
şehir
saat
hafta içi / hafta sonu
alerjenler
diyet tercihi
beğenilmeyen ürünler
maksimum bütçe
tercih edilen kategoriler
```

---

## Test Komutları

Semantic öneri:

```powershell
python .\src\semantic_recommender.py "spordan çıktım yüksek proteinli ucuz bir şey öner"
```

Karşılaştırma:

```powershell
python .\src\compare_engine.py "pizza karşılaştır"
```

API karşılaştırma testi:

```powershell
$body = @{
  query = "pizza karşılaştır"
  limit = 5
  context = @{
    hour = 19
    day_type = "weekday"
    city = "bursa"
  }
  user_profile = @{
    diet = "Standart"
    allergies = @()
    disliked_items = @()
    preferred_categories = @()
    max_budget = 400
  }
} | ConvertTo-Json -Depth 5

Invoke-RestMethod `
  -Uri "http://127.0.0.1:8000/compare" `
  -Method POST `
  -ContentType "application/json; charset=utf-8" `
  -Body ([System.Text.Encoding]::UTF8.GetBytes($body))
```

---

## Son Durum

Bu repo artık şunları yapar:

```txt
Cold-start ana sayfa önerisi
Semantic AI önerisi
Fiyat karşılaştırması
Platform bazlı fiyat özeti
En ucuz ürün listesi
En avantajlı ürün listesi
DB/backend uyumlu çıktı
FastAPI servis katmanı
```

Kısaca:

```txt
Veri gelir.
Embedding çıkarılır.
FAISS index kurulur.
Kullanıcı sorgusu anlamsal olarak aranır.
Ürünler skorlanır.
Gerekirse fiyat karşılaştırması yapılır.
API servis eder.
```
