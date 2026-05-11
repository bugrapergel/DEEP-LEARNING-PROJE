# KITTI Autonomous Perception Dashboard · v4.0

Cyberpunk temalı 4 panelli otonom araç algılama görselleştirme aracı.  
Gerçek Velodyne `.bin` dosyası gerekmez — pseudo-LiDAR motoru halkaları ve nokta bulutunu sentetik olarak üretir.

---

## 📁 Proje Yapısı

```
kitti_dashboard/
│
├── main.py          ← Buradan çalıştırırsın
├── dashboard.py     ← Ana pencere (CustomTkinter 4-panel grid)
├── config.py        ← Renkler, sabitler, font tanımları
├── data.py          ← KITTICalib, KITTIObj, KITTIDataset
├── lidar.py         ← Pseudo-LiDAR nokta bulutu + lazer halkaları
├── rendering.py     ← 3D BBox çizimi, BEVRenderer, CameraRenderer
├── ai_engine.py     ← Yapay zeka karar metni üreticisi
└── requirements.txt
```

---

## ⚙️ Kurulum

### 1. Repoyu klonla
```bash
git clone https://github.com/KULLANICI_ADIN/kitti-dashboard.git
cd kitti-dashboard
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

## ▶️ Çalıştırma

```bash
python main.py
```

---

## 📂 KITTI Veri Seti

[KITTI Vision Benchmark](http://www.cvlibs.net/datasets/kitti/eval_object.php) sayfasından  
**object detection** veri setini indirin. İhtiyacın olan 3 klasör:

```
KITTI/
├── image_2/     ← .png kamera görüntüleri
├── label_2/     ← .txt etiket dosyaları
└── calib/       ← .txt kalibrasyon dosyaları
```

Uygulamayı açtıktan sonra sol paneldeki **"…"** butonlarıyla bu 3 klasörü seç,  
ardından **"◈ VERİ SETİ YÜKLE"** butonuna tıkla.

---

## 🖥️ Panel Açıklamaları

| Panel | İçerik |
|-------|--------|
| **Sol Üst** | Dizin seçimi, frame navigasyonu, nesne listesi, istatistikler |
| **Sol Alt** | AI karar analizi, vision modu (ısı haritası / segmentasyon), LiDAR halka yoğunluğu |
| **Sağ Üst** | Kamera görüntüsü + 3D bounding box projeksiyonu (P2 matrisi) |
| **Sağ Alt** | Kuş bakışı LiDAR görünümü (pseudo nokta bulutu + lazer halkaları) |

---

## 🔧 Gereksinimler

- Python 3.9+
- Windows 10/11, Linux veya macOS
- DLL bağımlılığı yok