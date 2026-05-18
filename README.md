# NeYesem AI Recommender

Bu repo, **NeYesem projesinin AI öneri motorudur**.

Görevi nettir:

> Kullanıcının ne yemek istediğini anlamak, eldeki gerçek ürün verileriyle eşleştirmek ve mantıklı yemek önerileri üretmek.

Bu sistem ürün uydurmaz. Rastgele öneri basmaz. Kullanıcının isteğini, ürün adlarını, kategorileri, fiyatları ve indirimleri dikkate alır; elindeki gerçek veri üzerinden karar verir.

---

## Projenin Rolü

NeYesem projesi üç ana parçadan oluşur:

```txt
Scraper / Data Pipeline
→ Yemek platformlarından ürün verilerini toplar
→ all_items.json üretir

AI Recommender
→ all_items.json dosyasını okur
→ kullanıcıya öneri üretir
→ API olarak frontend'e servis eder

Frontend / Mobil Uygulama
→ AI API'den gelen önerileri kullanıcıya gösterir
```

Bu repo ikinci parçadır:

```txt
NeYesem AI Recommender
```

Yani bu repo yemek verisini toplamaz. Toplanmış veriyi akıllıca kullanır.

---

## Sistem Ne Yapar?

Bu sistem iki farklı öneri tipi üretir.

### 1. Ana Sayfa Önerileri

Kullanıcı uygulamayı açtığında daha hiçbir şey yazmamış olabilir. Bu durumda sistem boş durmaz.

Ana sayfa için hazır öneri bölümleri üretir:

```txt
Uygun Fiyatlı Doyurucular
Pizza & Burger Önerileri
Tatlı Kaçamağı
İndirimli Fırsatlar
Hafif ve Sağlıklı Seçenekler
Bugünün Karışık Önerileri
```

Bu öneriler şu endpoint ile gelir:

```http
GET /homepage?limit=8
```

### 2. Kullanıcı İsteğine Göre AI Önerisi

Kullanıcı şuna benzer bir şey yazabilir:

```txt
pizza öner
çok açım ucuz ve doyurucu bir şey öner
tatlı bir şey istiyorum ama pahalı olmasın
sağlıklı hafif bir şey öner
burger öner
```

Sistem bu metni analiz eder, ilgili ürünleri bulur ve gerçek veri üzerinden öneri listesi üretir.

Bu öneriler şu endpoint ile gelir:

```http
POST /recommend
```

---

## Kullanılan AI / ML Mantığı

Bu projede öneri sistemi **content-based recommendation** mantığıyla çalışır.

Temel akış:

```txt
1. all_items.json okunur
2. Ürün adları, restoran adları ve platform bilgileri temizlenir
3. Kullanıcının yazdığı metin normalize edilir
4. Ürünler TF-IDF ile vektörleştirilir
5. Kullanıcı isteği de aynı vektör uzayına taşınır
6. Cosine similarity ile benzer ürünler bulunur
7. Fiyat, indirim, kategori ve tercih skorları eklenir
8. En yüksek skorlu ürünler öneri olarak döndürülür
```

Kullanılan ana teknikler:

```txt
TF-IDF
Cosine Similarity
Content-Based Recommendation
Cold-Start Recommendation
Rule-Based Ranking
Data Cleaning
Category Filtering
```

Sistem özellikle başlangıç aşaması için uygundur. Çünkü kullanıcı geçmişi olmadan da öneri üretebilir.

---

## Cold-Start Recommendation Mantığı

Yeni kullanıcı geldiğinde sistemin elinde şu bilgiler yoktur:

```txt
Daha önce ne sipariş etti?
Neye tıkladı?
Hangi restoranları seviyor?
Hangi fiyat aralığını tercih ediyor?
```

Bu durumda sistem kullanıcı geçmişine güvenemez. O yüzden ana sayfa önerileri şu sinyallerle oluşturulur:

```txt
Ürün kategorisi
Fiyat
İndirim oranı
Restoran adı
Platform
Ürün adı
Ana yemek olup olmaması
Yan ürün / sos / içecek filtreleri
Sağlıklı ürün filtreleri
```

Böylece kullanıcı daha hiçbir şey yapmadan mantıklı bir ana sayfa görür.

---


## Kullanıcı Geçmişi Oluşunca Ne Olur?

Şu an sistem kullanıcı geçmişi olmadan çalışır. Buna **cold-start recommendation** diyoruz.

Kullanıcı sipariş vermeye başladığında sistem daha akıllı hale gelir. Çünkü artık sadece genel ürün verisine değil, kullanıcının davranışına da bakabilir.

Kullanıcı geçmişinden şu sinyaller çıkarılabilir:

```txt
En çok sipariş verdiği kategoriler
Tekrar tercih ettiği restoranlar
Sık kullandığı platformlar
Ortalama harcama aralığı
Beğendiği / beğenmediği ürünler
Sipariş verdiği saat aralıkları

---

## Proje Yapısı

```txt
.
├── data/
│   ├── all_items.json
│   └── homepage_recommendations.json
│
├── src/
│   ├── __init__.py
│   ├── api.py
│   ├── recommender.py
│   └── homepage_recommender.py
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

## Dosyalar Ne İşe Yarar?

### `data/all_items.json`

Scraper/data pipeline tarafından üretilen ana veri dosyasıdır.

Bu dosyada farklı platformlardan gelen ürünler ortak formatta bulunur.

Örnek ürün:

```json
{
  "platform": "getir_yemek",
  "restaurant_name": "Pizza Bulls",
  "item_name": "Orta Boy Sucuklu Pizza",
  "price": 339.9,
  "original_price": 339.9,
  "discount_rate": null,
  "product_url": "https://..."
}
```

AI sistemi bütün önerilerini bu dosya üzerinden üretir.

---

### `src/recommender.py`

Kullanıcının yazdığı metne göre öneri üretir.

Örnek:

```powershell
python .\src\recommender.py "pizza öner"
```

Bu dosya şunları yapar:

```txt
Kullanıcı isteğini analiz eder
Kategori çıkarır
TF-IDF modeli kurar
Cosine similarity hesaplar
Fiyat ve kategori skorları ekler
Önerileri JSON olarak döndürür
```

---

### `src/homepage_recommender.py`

Kullanıcı hiçbir şey yazmadan ana sayfa önerileri üretir.

Örnek:

```powershell
python .\src\homepage_recommender.py --limit 8
```

Dosyaya kaydetmek için:

```powershell
python .\src\homepage_recommender.py --limit 8 --save
```

Bu komut şunu üretir:

```txt
data/homepage_recommendations.json
```

---

### `src/api.py`

FastAPI servisidir.

Frontend buraya istek atar.

Ana endpointler:

```txt
GET  /
GET  /health
GET  /homepage
POST /recommend
```

---

## Kurulum

Bağımlılıkları kur:

```powershell
python -m pip install -r .\requirements.txt
```

`requirements.txt` içinde kullanılan ana paketler:

```txt
pandas
numpy
scikit-learn
joblib
fastapi
uvicorn
```

---

## API'yi Çalıştırma

AI API'yi başlatmak için:

```powershell
uvicorn src.api:app --reload
```

API çalışınca şu adres açılır:

```txt
http://127.0.0.1:8000
```

Swagger dokümantasyonu:

```txt
http://127.0.0.1:8000/docs
```

---

## API Endpointleri

### Health Check

```http
GET /health
```

Örnek cevap:

```json
{
  "status": "ok",
  "service": "neyesem-ai-recommender"
}
```

---

### Ana Sayfa Önerileri

```http
GET /homepage?limit=8
```

Bu endpoint kullanıcı hiçbir şey yazmadan gösterilecek ana sayfa önerilerini döndürür.

Örnek cevap yapısı:

```json
{
  "type": "cold_start_homepage_recommendations",
  "section_count": 6,
  "sections": [
    {
      "title": "Uygun Fiyatlı Doyurucular",
      "description": "Fiyatı görece uygun, ana yemek sayılabilecek doyurucu seçenekler.",
      "items": []
    }
  ]
}
```

Frontend ana sayfada bu section'ları kart kart gösterebilir.

---

### Kullanıcı İsteğine Göre Öneri

```http
POST /recommend
```

Request body:

```json
{
  "query": "pizza öner",
  "limit": 10
}
```

Örnek cevap:

```json
{
  "intent": {
    "raw_query": "pizza öner",
    "normalized_query": "pizza oner",
    "categories": ["pizza"],
    "budget_max": null,
    "preference": "balanced"
  },
  "count": 10,
  "recommendations": [
    {
      "platform": "getir_yemek",
      "restaurant_name": "Pizza Bulls",
      "item_name": "Orta Boy Sucuklu Pizza",
      "price": 339.9,
      "original_price": 339.9,
      "discount_rate": null,
      "product_url": "https://...",
      "score": 0.2496,
      "ml_similarity": 0.2841,
      "reason": "pizza isteğine uygun, 339.90 TL."
    }
  ]
}
```
---

## Flutter Entegrasyonu

Flutter tarafı lokal geliştirmede şu base URL'i kullanır:

```txt
http://127.0.0.1:8000
```

Chrome üzerinde çalışırken bu adres doğrudur.

Android emulator için base URL şu olur:

```txt
http://10.0.2.2:8000
```

Gerçek telefonda çalıştırılacaksa bilgisayarın local IP adresi kullanılır ve API şu şekilde başlatılır:

```powershell
uvicorn src.api:app --reload --host 0.0.0.0 --port 8000
```

---

## Frontend İçin Kullanım

Ana sayfa için:

```http
GET http://127.0.0.1:8000/homepage?limit=8
```

Kullanıcı arama/istek kutusu için:

```http
POST http://127.0.0.1:8000/recommend
```

Body:

```json
{
  "query": "ucuz doyurucu bir şey öner",
  "limit": 10
}
```

Frontend bu cevaptaki `recommendations` listesini kart olarak basar.

Olay bu kadar.

---

## Öneri Skoru Nasıl Hesaplanır?

Sistem tek bir şeye bakmaz. Birkaç sinyali beraber kullanır.

### Metin Benzerliği

Kullanıcının isteği ile ürün metni arasındaki benzerlik hesaplanır.

Örnek:

```txt
Kullanıcı: pizza öner
Ürün: Orta Boy Sucuklu Pizza
→ yüksek benzerlik
```

### Kategori Uyumu

Sistem bazı kategorileri tanır:

```txt
pizza
burger
döner
pide
tatlı
tavuk
içecek
sağlıklı
```

Kullanıcı isteğinde kategori varsa ürünler buna göre filtrelenir.

### Fiyat Skoru

Ucuz, uygun fiyatlı veya bütçe odaklı isteklerde fiyat etkisi artar.

Örnek:

```txt
çok açım ucuz bir şey öner
```

Bu durumda sistem düşük/orta fiyatlı ve doyurucu ürünleri öne çıkarır.

### Doyuruculuk Skoru

Doyurucu ürünlerde şu kelimeler dikkate alınır:

```txt
menü
döner
burger
pizza
pide
lahmacun
tavuk
köfte
dürüm
tantuni
kebap
```

### İndirim Skoru

İndirim oranı varsa skora eklenir. Çok uç değerler kontrollü değerlendirilir.

---

## Veri Temizleme

Öneri sistemi çöp veriyi ekrana basmamak için filtreleme yapar.

Temizlenen/elenen örnekler:

```txt
Detaylar
Sadece sayı olan ürün adları
Sos
Ketçap
Mayonez
Peçete
Yan ürünler
Anlamsız scrape metinleri
```

Ana sayfa önerilerinde ayrıca içecek ve yan ürünlerin yanlış kategoriye düşmesi engellenir.

Örnek:

```txt
Pizza Bulls restoranındaki Fuse Tea, pizza önerisi sayılmaz.
Profiterol, sağlıklı ürün sayılmaz.
Doritos Taco Parti, sağlıklı ürün sayılmaz.
```

Kurallar net. Ürün neyse o.

---

## Komutlar

### Kullanıcı isteğine göre öneri üret

```powershell
python .\src\recommender.py "pizza öner"
```

```powershell
python .\src\recommender.py "çok açım ucuz ve doyurucu bir şey öner"
```

```powershell
python .\src\recommender.py "tatlı bir şey istiyorum ama pahalı olmasın"
```

### Ana sayfa önerisi üret

```powershell
python .\src\homepage_recommender.py --limit 8
```

### Ana sayfa önerisini dosyaya kaydet

```powershell
python .\src\homepage_recommender.py --limit 8 --save
```

### API'yi başlat

```powershell
uvicorn src.api:app --reload
```

---

## Örnek Kullanıcı Senaryoları

### Senaryo 1

Kullanıcı:

```txt
pizza öner
```

Sistem:

```txt
Pizza kategorisini çıkarır
Pizza ürünlerini filtreler
TF-IDF benzerliği hesaplar
Gerçek pizza ürünlerini listeler
```

### Senaryo 2

Kullanıcı:

```txt
çok açım ucuz ve doyurucu bir şey öner
```

Sistem:

```txt
Preference = filling
Budget = düşük/orta
Doyurucu ana yemekleri öne çıkarır
Fiyatı yüksek olanları geriye atar
```

### Senaryo 3

Kullanıcı:

```txt
tatlı bir şey istiyorum ama pahalı olmasın
```

Sistem:

```txt
Tatlı kategorisini çıkarır
Ucuz tatlıları öne çıkarır
İndirim varsa skora ekler
```

## Geliştirme Akışı

Veri güncellendikten sonra `data/all_items.json` dosyası yenilenir.

Sonra ana sayfa önerileri tekrar üretilebilir:

```powershell
python .\src\homepage_recommender.py --limit 8 --save
```

API tekrar çalıştırılır:

```powershell
uvicorn src.api:app --reload
```

Frontend aynı endpointleri çağırmaya devam eder.

---

## Son Durum

Bu repo artık AI tarafında görevini yapar.

Elindeki gerçek yemek verisini okur, temizler, anlamlandırır, skorlar ve frontend'in kullanabileceği öneri çıktısına çevirir.

Kısaca:

```txt
Veri gelir.
AI karar verir.
API servis eder.
Frontend gösterir.
```
