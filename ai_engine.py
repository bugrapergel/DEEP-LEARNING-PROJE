# ══════════════════════════════════════════════════════════════════════════════
#  ai_engine.py  —  Yapay Zeka Karar Motoru
# ══════════════════════════════════════════════════════════════════════════════

import math
import random
from typing import Tuple

from config import DISPLAY
from data import KITTIObj


class AIEngine:
    """
    Seçili nesne için metin tabanlı otonom sürüş kararı üretir.
    Gerçek bir model değil; demo amaçlı şablon tabanlı sistem.
    """

    # ── Normal Durum Şablonları ───────────────────────────────────────────────
    _log_templates = {
        "Car": [
            "[TRACK]  Binek araç algılandı — şerit içi.\n"
            "[PLAN]   Güvenli takip mesafesi {dist:.1f}m → {safe:.1f}m hedefleniyor.\n"
            "[ACT]    Hız kontrolörü aktif. ACC moduna geçiliyor...",

            "[PRED]   Araç #ID önde seyrediyor ({dist:.1f}m).\n"
            "[RISK]   TTC (çarpışma süresi) = {ttc:.1f}s\n"
            "[ACT]    Standby fren basıncı ±{brk:.0f}% hazır.",

            "[SCAN]   Şerit değiştirme analizi başlatıldı.\n"
            "[MAP]    Sol: Müsait  ·  Sağ: Müsait\n"
            "[ACT]    Araç takip modunda sabit seyir.",
        ],
        "Pedestrian": [
            "[ALERT]  Yaya tespit edildi — {dist:.1f}m.\n"
            "[RULE]   Yayaya öncelik protokolü devreye girdi.\n"
            "[ACT]    Araç hızı ≤10 km/h sınırına düşürülüyor...",

            "[WARN]   Şerit içinde yaya ({dist:.1f}m). Hareket vektörü hesaplanıyor.\n"
            "[PRED]   Çarpışma riski: %{risk:.0f}\n"
            "[ACT]    Acil duruş mesafesi = {dist:.1f}m  →  DURDUR.",
        ],
        "Cyclist": [
            "[TRACK]  Bisikletli algılandı ({dist:.1f}m).\n"
            "[PLAN]   Güvenli geçiş yolu hesaplanıyor...\n"
            "[ACT]    Sağa {offset:.1f}m ötelenme manevra planı üretildi.",

            "[SCAN]   Bisikletli hız tahmini: ~{spd:.1f} km/h\n"
            "[RISK]   Şerit çakışma olasılığı: %{risk:.0f}\n"
            "[ACT]    Mesafe korunuyor, geçiş fırsatı bekleniyor.",
        ],
        "Truck": [
            "[SCAN]   Ağır vasıta ({dist:.1f}m). Kör nokta analizi çalışıyor.\n"
            "[PLAN]   Soldan geçiş senaryosu hesaplandı.\n"
            "[ACT]    Yeterli mesafe sağlanana kadar hız kilidi aktif.",
        ],
        "default": [
            "[TRACK]  Nesne algılandı — Tip:{cls} · Mesafe:{dist:.1f}m\n"
            "[PLAN]   Sınıflandırma güveni düşük. Gözlem modu açıldı.\n"
            "[ACT]    Güvenli seyir modu sürdürülüyor.",
        ],
    }

    # ── Kritik Durum Şablonları ───────────────────────────────────────────────
    _crit_templates = [
        "[⚠ KRİTİK]  {dist:.1f}m — ACİL FREN PROTOKOLÜ BAŞLATILDI!\n"
        "[SENSOR]    Çarpışma öngörüsü POZİTİF. TTC ≤ 1.2s\n"
        "[ACT]       Fren basıncı maksimuma çıkarıldı. DURULUYOR.",

        "[⚠ KRİTİK]  {cls} {dist:.1f}m mesafede! YAVAŞLIYORUM.\n"
        "[RULE]      Güvenli dur mesafesi (2.4m) aşıldı.\n"
        "[ACT]       Hazırlık alarmı verildi — müdahale süresi <0.3s.",
    ]

    def analyze(self, obj: KITTIObj, frame_idx: int) -> Tuple[str, bool]:
        """
        Nesne için karar metni üretir.
        Döndürür: (mesaj_str, is_critical)
        """
        rnd    = random.Random(frame_idx * 31 + hash(obj.cls) % 97)
        params = dict(
            cls    = obj.label,
            dist   = obj.dist,
            safe   = obj.dist + 2.8,
            ttc    = max(0.5, obj.dist / max(0.1, 50 / 3.6)),
            brk    = rnd.uniform(15, 45),
            risk   = rnd.uniform(75, 99) if obj.critical else rnd.uniform(10, 70),
            offset = rnd.uniform(0.8, 1.6),
            spd    = rnd.uniform(8, 25),
        )
        if obj.critical:
            tmpl = rnd.choice(self._crit_templates)
            return tmpl.format(**params), True
        tmpl_list = self._log_templates.get(obj.cls, self._log_templates["default"])
        tmpl      = tmpl_list[frame_idx % len(tmpl_list)]
        return tmpl.format(**params), False

    def kinematic_line(self, obj: KITTIObj) -> str:
        """Nesnenin kinematik özetini tek satır olarak döndürür."""
        yaw_deg = math.degrees(obj.ry)
        status  = "KRİTİK" if obj.critical else "Normal"
        return (
            f"ID:{DISPLAY.get(obj.cls, obj.cls)}  ·  "
            f"Z={obj.z:.2f}m  ·  X={obj.x:.2f}m  ·  "
            f"Yaw={yaw_deg:.1f}°  ·  [{status}]"
        )