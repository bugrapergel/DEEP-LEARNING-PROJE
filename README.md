# DEEP-LEARNING-PROJE
# KITTI Autonomous Perception Dashboard · v4.0

Cyberpunk temalı 4 panelli otonom araç algılama görselleştirme aracı.  


---

##  Proje Yapısı

```
DEEP-LEARNING-PROJE/
│
├── main.py          ← Arayüzü başlatan ana yürütülebilir dosya
├── finetune.py      ← PointPillars Fine-Tuning (Adam Opt, 160 Epoch) eğitim betiği
├── dashboard.py     ← Ana pencere (CustomTkinter 4-panel grid)
├── config.py        ← Renkler, sabitler, font tanımları
├── data.py          ← KITTICalib, KITTIObj, KITTIDataset sınıfları
├── lidar.py         ← LiDAR nokta bulutu ve Z-Coloring işlemleri
├── rendering.py     ← 3D Bounding Box projeksiyonu ve BEV/Camera Render motoru
├── ai_engine.py     ← Kural tabanlı otonom karar ve eylem metni üreticisi
└── requirements.txt ← Proje bağımlılıkları
```

---

##  Kurulum

### 1. Repoyu klonla
```bash
git clone [https://github.com/------/DEEP-LEARNING-PROJE.git](https://github.com/-------/DEEP-LEARNING-PROJE.git)
cd DEEP-LEARNING-PROJE
```

### 2. Sanal ortam oluştur (önerilir)
```bash
python -m venv .venv

# Windows:
.venv\Scripts\activate

# Linux / macOS:
source .venv/bin/activate
```

### 3. Bağımlılıkları yükle
```bash
pip install -r requirements.txt
```

---

##  Çalıştırma

```bash
python main.py
```

---

##  KITTI Veri Seti

[KITTI Vision Benchmark](http://www.cvlibs.net/datasets/kitti/eval_object.php) sayfasından  
**object detection** veri setini indirin. İhtiyacın olan 3 klasör:

```
KITTI/
├── image_2/     ← .png kamera görüntüleri
├── label_2/     ← .txt etiket dosyaları
└── calib/       ← .txt kalibrasyon dosyaları
└── velodyne/    ←  .bin LİDAR(nokta bulutu) dosyaları
```

Uygulamayı açtıktan sonra sol paneldeki **"…"** butonlarıyla bu 4 klasörü seç,  
ardından **"◈ VERİ SETİ YÜKLE"** butonuna tıkla.

---


##  Panel Açıklamaları

| Panel | İçerik |
|-------|--------|
| **Sol Üst** | Veri seti yolları, frame geçişleri, dinamik nesne listesi ve tespit edilen anlık metrik istatistikler. |
| **Sol Alt** | [SCAN/PLAN/ACT] formatında Yapay Zeka Karar Analizi, güvenli takip mesafesi uyarıları ve otonom müdahale logları. |
| **Sağ Üst** | Orijinal kamera görüntüsü üzerine kalibrasyon matrisleriyle (P2) izdüşümü alınmış tespit edilen nesneler ve 3D Bounding Box projeksiyonları. |
| **Sağ Alt** | Kuş bakışı (Bird's Eye View) gerçek LiDAR nokta bulutu görünümü, Z-Coloring (Yükseklik bazlı renklendirme) ve nesne vurgulama. |

---

## 🔧 Gereksinimler

- Python 3.9+
- Windows 10/11, Linux veya macOS
- DLL bağımlılığı yok
