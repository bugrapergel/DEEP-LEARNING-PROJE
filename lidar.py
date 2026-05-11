# ══════════════════════════════════════════════════════════════════════════════
#  lidar.py  —  Pseudo-LiDAR Engine: nesne noktaları + zemin + lazer halkaları
# ══════════════════════════════════════════════════════════════════════════════

import math
import numpy as np
from typing import List

from config import BEV_R, SYNTH_DENS, RING_COUNT, RING_MAX
from data import KITTIObj


class PseudoLiDAR:
    """
    Gerçek Velodyne .bin olmadan tam LiDAR görünümü:
      1) Her nesne içine ~600 pt/m³ yoğun sentetik nokta bulutu
      2) Yol düzleminde dağınık zemin noktaları
      3) Ego-araç merkezli kavisli lazer halkaları (KITTI ring pattern)
    """

    RNG = np.random.default_rng(7)

    # ── Nesne + Zemin Noktaları ───────────────────────────────────────────────

    @classmethod
    def generate(cls, objects: List[KITTIObj]) -> np.ndarray:
        """
        Verilen nesneler için (N, 4) float32 nokta bulutu üretir.
        Sütunlar: [x_cam, y_cam, z_cam, intensity]
        """
        parts = []

        for obj in objects:
            if obj.cls == "DontCare":
                continue
            vol    = max(0.1, obj.h * obj.w * obj.l)
            n      = int(vol * SYNTH_DENS)
            lx     = cls.RNG.uniform(-obj.l / 2, obj.l / 2, n).astype(np.float32)
            ly     = cls.RNG.uniform(-obj.h,      0.0,      n).astype(np.float32)
            lz     = cls.RNG.uniform(-obj.w / 2, obj.w / 2, n).astype(np.float32)
            c, s   = math.cos(obj.ry), math.sin(obj.ry)
            wx     = (c * lx + s * lz + obj.x).astype(np.float32)
            wz     = (-s * lx + c * lz + obj.z).astype(np.float32)
            wy     = (ly + obj.y).astype(np.float32)
            intens = cls.RNG.uniform(0.4, 0.95, n).astype(np.float32)
            parts.append(np.stack([wx, wy, wz, intens], axis=1))

        # Zemin düzlemi (yol: y ≈ 1.65 m kamera altı)
        n_gnd = 6000
        gx = cls.RNG.uniform(-18,  18,   n_gnd).astype(np.float32)
        gz = cls.RNG.uniform(2,    BEV_R, n_gnd).astype(np.float32)
        gy = cls.RNG.uniform(1.55, 1.75,  n_gnd).astype(np.float32)
        gi = cls.RNG.uniform(0.05, 0.30,  n_gnd).astype(np.float32)
        parts.append(np.stack([gx, gy, gz, gi], axis=1))

        return np.vstack(parts) if parts else np.zeros((0, 4), np.float32)

    # ── LiDAR Halkaları ───────────────────────────────────────────────────────

    @classmethod
    def lidar_rings(cls,
                    n_rings: int = RING_COUNT,
                    max_r:   float = RING_MAX) -> np.ndarray:
        """
        Ego-araç merkezli kavisli lazer halkaları üretir.
        Gerçek LiDAR davranışını taklit eden logaritmik aralık + gürültü.
        (M, 4) float32 döndürür; y ≈ yol seviyesi.
        """
        all_rings = []
        for i in range(1, n_rings + 1):
            r      = max_r * (i / n_rings) ** 1.15   # logaritmik aralık
            angles = np.linspace(
                -math.pi * 0.68,
                 math.pi * 0.68,
                 int(360 * (i / n_rings) * 2.5 + 40),
            )
            noise  = cls.RNG.uniform(-0.015, 0.015, len(angles)).astype(np.float32)
            r_n    = (r + cls.RNG.uniform(-0.05, 0.05, len(angles))).astype(np.float32)
            x      = (r_n * np.sin(angles + noise)).astype(np.float32)
            z      = (r_n * np.cos(angles + noise)).astype(np.float32)
            y      = np.full(len(x), 1.65, dtype=np.float32)
            # Uzaktaki noktalar daha soluk (1/r² zayıflama)
            intensity = np.clip(
                1.0 / (1.0 + 0.008 * r_n ** 2), 0.04, 1.0
            ).astype(np.float32)
            mask = z > 0.5   # yalnızca ön yarım küre
            all_rings.append(
                np.stack([x[mask], y[mask], z[mask], intensity[mask]], axis=1)
            )
        return np.vstack(all_rings)