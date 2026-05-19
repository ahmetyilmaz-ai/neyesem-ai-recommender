# NeYesem AI Recommender

Bu repo, **NeYesem projesinin AI öneri motorudur**.

Görevi nettir:

> Gerçek yemek verisini alır, anlamsal olarak işler, kullanıcı isteğine ve bağlamına göre mantıklı yemek önerileri üretir.

Bu sistem ürün uydurmaz. Rastgele öneri basmaz. Elindeki gerçek ürün verisini kullanır. Kullanıcının yazdığı metni, fiyatı, indirimi, kategoriyi, saat bilgisini, şehir bilgisini ve kullanıcı profilini birlikte değerlendirir.

---

## Projenin Rolü

NeYesem mimarisi üç ana parçadan oluşur:

```txt
Scraper / Data Pipeline
→ Yemek platformlarından ürün verilerini toplar
→ data/all_items.json üretir

AI Recommender
→ all_items.json dosyasını okur
→ ürünleri semantic embedding'e çevirir
→ FAISS index oluşturur
→ kullanıcıya öneri üretir
→ FastAPI üzerinden servis eder

Frontend / Backend
→ AI API'den gelen önerileri kullanıcıya gösterir
→ DB/backend uyumlu endpointleri kullanabilir
```

Bu repo ikinci parçadır:

```txt
NeYesem AI Recommender
```

Yani bu repo veri toplamaz. Toplanmış veriyi akıllıca işler.

---

## Sistem Şu An Ne Yapıyor?

Sistem iki ana öneri tipi üretir.

### 1. Ana Sayfa Önerileri

Kullanıcı uygulamayı açtığında daha hiçbir şey yazmamış olabilir. Sistem yine boş durmaz.

Ana sayfa için cold-start öneri bölümleri üretir:

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

Bu endpoint Flutter ana sayfasında kullanılabilir.

---

### 2. Kullanıcı İsteğine Göre Semantic AI Önerisi

Kullanıcı serbest metin yazabilir:

```txt
pizza öner
çok açım ucuz ve doyurucu bir şey öner
spordan çıktım yüksek proteinli ucuz bir şey öner
hasta gibiyim sıcak çorba tarzı bir şey öner
tatlı bir şey istiyorum ama pahalı olmasın
hafif sağlıklı bir şey olsun
```

Sistem bu metni kelime kelime basit arama gibi değil, **anlamsal olarak** işler.

Endpoint:

```http
POST /recommend
```

---

## Kullanılan AI Mimarisi

Bu repo artık basit TF-IDF arama motoru değildir.

Güncel mimari:

```txt
Hugging Face SentenceTransformer
+ Dense Embeddings
+ FAISS Vector Search
+ Business Rule Scoring
+ Context-Aware Ranking
+ User Profile Filtering
```

Kullanılan model:

```txt
sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
```

Bu model ürün metinlerini ve kullanıcı sorgusunu aynı anlamsal vektör uzayına taşır. Böylece sadece birebir kelime eşleşmesi değil, anlam yakınlığı üzerinden öneri yapılır.

Örnek:

```txt
Kullanıcı: spordan çıktım proteinli bir şey öner
Sistem: tavuk, proteinli pilav, grilled chicken, chicken burger gibi ürünleri öne çıkarır
```

---

## Offline / Online Mimari

Sistemin en önemli mimari ayrımı budur.

### Offline Build Aşaması

Bu aşama veri güncellendiğinde çalışır.

```txt
data/all_items.json okunur
ürün metinleri temizlenir
ürünlerden semantic text oluşturulur
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

Bu işlem her kullanıcı isteğinde yapılmaz. Veri güncellenince yapılır.

---

### Online Inference Aşaması

FastAPI çalışırken sistem hazır indexi RAM'e alır.

Kullanıcı istek attığında:

```txt
kullanıcı sorgusu embedding'e çevrilir
FAISS en yakın ürünleri bulur
context ve user_profile kuralları uygulanır
final score hesaplanır
en iyi öneriler JSON olarak döner
```

Yani API isteği sırasında tüm ürünler baştan vektörleştirilmez. Sistem hazır FAISS index üzerinden çalışır.

---

## Skorlama Mantığı

Sistem tek bir şeye bakmaz. Final skor birkaç sinyalden oluşur.

```txt
semantic_score
+ context_score
+ price_score
+ discount_score
+ category_score
+ user_profile_score
```

### Semantic Score

Kullanıcı sorgusu ile ürün arasındaki anlamsal yakınlıktır.

Örnek:

```txt
"yüksek proteinli ucuz bir şey"
→ "Proteini Yüksek Tavuklu Pilav"
→ yüksek semantic score
```

### Context Score

Kullanıcının bağlamı dikkate alınır:

```txt
saat
gün tipi
şehir
```

Örnek:

```txt
Sabah saatlerinde ağır kebap/burger ürünleri geriye düşebilir.
Akşam saatlerinde doyurucu ana yemekler öne çıkabilir.
```

### User Profile Score

Kullanıcı profili dikkate alınır:

```txt
diyet tercihi
alerjenler
beğenilmeyen ürünler
tercih edilen kategoriler
maksimum bütçe
```

Örnek:

```txt
Laktoz alerjisi varsa sütlü ürünler elenir.
Kullanıcı tavuk seviyorsa tavuk ürünleri skor kazanır.
Max bütçe 350 TL ise aşırı pahalı ürünler geriye düşer.
```

---

## Request Formatı

`POST /recommend` endpointi şu formatı destekler:

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

---

## Örnek Response

```json
{
  "engine": "sentence_transformer_faiss",
  "intent": {
    "raw_query": "spordan çıktım yüksek proteinli ucuz bir şey öner",
    "normalized_query": "spordan ciktim yuksek proteinli ucuz bir sey oner",
    "categories": [],
    "budget_max": 250,
    "preference": "protein"
  },
  "context": {
    "hour": 19,
    "day_type": "weekday",
    "city": "bursa"
  },
  "count": 5,
  "recommendations": [
    {
      "platform": "trendyol",
      "restaurant_name": "Etkolik",
      "item_name": "Fit Grilled Chicken",
      "category": "Tavuk",
      "price": 90.0,
      "original_price": 518.9,
      "discount_rate": 82.66,
      "score": 0.4953,
      "semantic_score": 0.4744,
      "context_score": 0.544,
      "reason": "protein odaklı tercihle uyumlu, 90.00 TL, %82.7 indirim, semantik skor 0.47, bağlam skoru 0.54."
    }
  ]
}
```

---

## DB / Backend Uyumlu Endpointler

Backend ve DB tarafı için ayrıca uyumlu çıktı formatı vardır.

```http
GET /db/homepage?limit=8
POST /db/recommend
```

Bu endpointlerde öneriler şu yapıya ayrılır:

```txt
platform
şehir
restoran
kategori
ürün
ai skoru
```

Yani PostgreSQL tarafındaki platform/restoran/kategori/ürün mantığına daha yakın JSON döner.

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

## Dosyalar Ne İşe Yarar?

### `data/all_items.json`

Scraper/data pipeline tarafından üretilen ana veri dosyasıdır.

AI sistemi ürünleri buradan okur.

---

### `src/build_index.py`

Offline index build scriptidir.

Şunları yapar:

```txt
all_items.json okur
ürünleri temizler
semantic text oluşturur
SentenceTransformer ile embedding çıkarır
FAISS index oluşturur
artifacts/ klasörüne kaydeder
```

Komut:

```powershell
python .\src\build_index.py
```

---

### `src/semantic_recommender.py`

Ana semantic öneri motorudur.

Şunları yapar:

```txt
FAISS index'i RAM'e yükler
kullanıcı sorgusunu embedding'e çevirir
FAISS ile en yakın ürünleri bulur
context ve user_profile filtreleri uygular
final score hesaplar
öneri JSON'u döndürür
```

Terminalden test:

```powershell
python .\src\semantic_recommender.py "spordan çıktım yüksek proteinli ucuz bir şey öner"
```

---

### `src/homepage_recommender.py`

Kullanıcı hiçbir şey yazmadan ana sayfa cold-start önerileri üretir.

Komut:

```powershell
python .\src\homepage_recommender.py --limit 8 --save
```

---

### `src/db_output_adapter.py`

AI çıktısını backend/DB tarafına daha uygun formata çevirir.

Kullanılan endpointler:

```txt
/db/homepage
/db/recommend
```

---

### `src/api.py`

FastAPI servisidir.

Frontend/backend buraya istek atar.

Endpointler:

```txt
GET  /
GET  /health
GET  /homepage
POST /recommend
GET  /db/homepage
POST /db/recommend
```

---

## Kurulum

Bağımlılıkları kur:

```powershell
python -m pip install -r .\requirements.txt
```

`requirements.txt` içinde ana paketler:

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

---

## İlk Kurulumdan Sonra Index Build

Model ve FAISS index için:

```powershell
python .\src\build_index.py
```

İlk çalıştırmada Hugging Face modeli indirilebilir. Sonraki çalıştırmalarda cache üzerinden kullanılır.

---

## API'yi Çalıştırma

```powershell
uvicorn src.api:app --reload
```

API adresi:

```txt
http://127.0.0.1:8000
```

Swagger dokümantasyonu:

```txt
http://127.0.0.1:8000/docs
```

---

## Health Check

```http
GET /health
```

Örnek cevap:

```json
{
  "status": "ok",
  "service": "neyesem-ai-recommender",
  "engine": "sentence-transformer-faiss"
}
```

---

## Frontend Entegrasyonu

Flutter veya başka bir frontend lokal geliştirmede şu base URL'i kullanır:

```txt
http://127.0.0.1:8000
```

Chrome üzerinde çalışırken bu adres doğrudur.

Android emulator için:

```txt
http://10.0.2.2:8000
```

Gerçek telefonda test için API şöyle açılır:

```powershell
uvicorn src.api:app --reload --host 0.0.0.0 --port 8000
```

---

## Frontend İçin Kullanım

Ana sayfa:

```http
GET http://127.0.0.1:8000/homepage?limit=8
```

Kullanıcı arama/istek kutusu:

```http
POST http://127.0.0.1:8000/recommend
```

Backend/DB uyumlu çıktı:

```http
POST http://127.0.0.1:8000/db/recommend
```

---

## Kullanıcı Geçmişi Oluşunca Ne Olur?

Kullanıcı sipariş verdikçe sistem daha kişisel hale gelir.

Toplanabilecek sinyaller:

```txt
en çok sipariş edilen kategoriler
tekrar tercih edilen restoranlar
ortalama harcama aralığı
beğenilen / beğenilmeyen ürünler
sipariş saatleri
diyet ve alerjen bilgileri
```

Bu bilgiler `user_profile` alanı üzerinden öneri skoruna eklenir.

Örnek:

```txt
Kullanıcı tavuk ürünlerini seviyorsa tavuk ürünleri skor kazanır.
Laktoz alerjisi varsa sütlü ürünler elenir.
Max bütçe varsa pahalı ürünler geriye düşer.
```

---

## Veri Temizleme

Sistem çöp veriyi öneriye sokmamak için filtreleme yapar.

Elenen örnekler:

```txt
sos
ketçap
mayonez
peçete
anlamsız scrape metinleri
yan ürünler
protein sorgusunda tatlı/içecek/sos ürünleri
sağlıklı sorgusunda abur cubur ürünler
```

Kurallar net. Ürün neyse o.

---

## Test Komutları

Semantic recommender test:

```powershell
python .\src\semantic_recommender.py "spordan çıktım yüksek proteinli ucuz bir şey öner"
```

Ana sayfa önerisi üret:

```powershell
python .\src\homepage_recommender.py --limit 8 --save
```

API test:

```powershell
$body = @{
  query = "spordan çıktım yüksek proteinli ucuz bir şey öner"
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
    preferred_categories = @("tavuk")
    max_budget = 350
  }
} | ConvertTo-Json -Depth 5

Invoke-RestMethod `
  -Uri "http://127.0.0.1:8000/recommend" `
  -Method POST `
  -ContentType "application/json; charset=utf-8" `
  -Body ([System.Text.Encoding]::UTF8.GetBytes($body))
```

---

## Geliştirme Akışı

Veri güncellendiğinde:

```powershell
python .\src\build_index.py
python .\src\homepage_recommender.py --limit 8 --save
```

Sonra API çalıştırılır:

```powershell
uvicorn src.api:app --reload
```

Frontend/backend aynı endpointleri çağırmaya devam eder.

---

## Son Durum

Bu repo artık AI tarafında görevini yapar.

```txt
Veri gelir.
Embedding çıkarılır.
FAISS index kurulur.
Kullanıcı sorgusu anlamsal olarak aranır.
Context ve profil kuralları uygulanır.
AI öneri üretir.
API servis eder.
```
