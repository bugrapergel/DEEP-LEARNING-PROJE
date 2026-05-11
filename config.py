# ══════════════════════════════════════════════════════════════════════════════
#  config.py  —  Global tema, sabitler, sınıf renkleri
# ══════════════════════════════════════════════════════════════════════════════

# ── Cyberpunk Dark-Navy Renk Paleti ───────────────────────────────────────────
C_BG0 = "#03070f"   # Siyaha yakın zemin
C_BG1 = "#060d1a"   # Panel arka planı
C_BG2 = "#0a1628"   # Kart / satır arka planı
C_BG3 = "#0e1e35"   # Hover / seçili satır
C_CYN = "#00f0ff"   # Neon Cyan  — birincil vurgu
C_GRN = "#00ff88"   # Neon Yeşil — onay / normal
C_RED = "#ff1a3d"   # Neon Kırmızı — KRİTİK
C_YLW = "#ffe500"   # Neon Sarı — uyarı / ok
C_ORG = "#ff6a00"   # Turuncu — cyclist
C_PRP = "#bf00ff"   # Mor — segmentasyon
C_WHT = "#e8f4ff"   # Beyaz-mavi metin
C_DIM = "#2a4a65"   # Soluk metin
C_BDR = "#0d2040"   # Kenar

# ── Algılama Sabitleri ────────────────────────────────────────────────────────
RISK_M     = 5.0    # Kritik mesafe eşiği (metre)
BEV_R      = 55.0   # BEV görüntü yarıçapı (metre)
SYNTH_DENS = 600    # Sentetik nokta yoğunluğu (nokta/m³)
RING_COUNT = 18     # LiDAR halka sayısı (varsayılan)
RING_MAX   = 50.0   # Maksimum halka mesafesi (metre)

# ── Font Tanımları ────────────────────────────────────────────────────────────
FNT_TITLE = ("Courier New", 16, "bold")
FNT_HDR   = ("Courier New", 12, "bold")
FNT_MD    = ("Courier New", 11)
FNT_SM    = ("Courier New", 10)
FNT_XS    = ("Courier New",  9)

# ── Sınıf → BGR Rengi (OpenCV) ────────────────────────────────────────────────
CLASS_BGR = {
    "Car":        ( 30, 220, 255),
    "Pedestrian": ( 50, 255, 120),
    "Cyclist":    (  0, 130, 255),
    "Truck":      (255,  60, 255),
    "Van":        (200,  80, 255),
    "Tram":       (  0, 210, 255),
    "Misc":       (160, 160, 160),
    "DontCare":   ( 50,  50,  50),
}

# ── Sınıf → Türkçe Görüntü Adı ───────────────────────────────────────────────
DISPLAY = {
    "Car":        "Binek Araç",
    "Pedestrian": "Yaya",
    "Cyclist":    "Bisiklet",
    "Truck":      "Kamyon",
    "Van":        "Minibüs",
    "Tram":       "Tramvay",
    "Misc":       "Diğer",
    "DontCare":   "Engel",
}