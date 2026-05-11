# ══════════════════════════════════════════════════════════════════════════════
#  rendering.py  —  3D Bounding Box + BEV + Kamera görüntü üreticileri
# ══════════════════════════════════════════════════════════════════════════════

import math
import numpy as np
import cv2
from typing import List

from config import CLASS_BGR, BEV_R
from data import KITTIObj, KITTICalib


# ══════════════════════════════════════════════════════════════════════════════
#  3D BOUNDING BOX  ──  Köşe hesabı + kenar çizimi
# ══════════════════════════════════════════════════════════════════════════════

# Kutu kenar indeksleri (alt taban → üst taban → dikeyler)
EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 0),   # alt taban
    (4, 5), (5, 6), (6, 7), (7, 4),   # üst taban
    (0, 4), (1, 5), (2, 6), (3, 7),   # dikeyler
]
# Ön yüz kenarları (beyaz çizilir)
FRONT = {(0, 1), (0, 4), (1, 5), (4, 5)}


def box_corners_cam(obj: KITTIObj) -> np.ndarray:
    """Nesnenin 8 köşesini kamera koordinatında döndürür (8, 3)."""
    hl, hw = obj.l / 2, obj.w / 2
    xs = np.array([ hl,  hl, -hl, -hl,  hl,  hl, -hl, -hl])
    ys = np.array([-obj.h] * 4 + [0] * 4)
    zs = np.array([ hw, -hw, -hw,  hw,  hw, -hw, -hw,  hw])
    c, s = math.cos(obj.ry), math.sin(obj.ry)
    rx   =  c * xs + s * zs + obj.x
    rz   = -s * xs + c * zs + obj.z
    return np.stack([rx, ys + obj.y, rz], axis=1)


def draw_box3d(img: np.ndarray, uv: np.ndarray,
               valid: np.ndarray, color: tuple,
               highlight: bool = False) -> None:
    """Projekte edilmiş köşeleri kullanarak 3D kutu çizer."""
    t_edge  = 4 if highlight else 3
    t_front = 5 if highlight else 4

    def pt(i):
        return (int(uv[i, 0]), int(uv[i, 1]))

    for i, j in EDGES:
        if not (valid[i] and valid[j]):
            continue
        pair     = (min(i, j), max(i, j))
        is_front = pair in {(min(a, b), max(a, b)) for a, b in FRONT}
        cv2.line(
            img, pt(i), pt(j),
            (255, 255, 255) if is_front else color,
            t_front if is_front else t_edge,
            cv2.LINE_AA,
        )

    if highlight:
        for i, j in EDGES:
            if not (valid[i] and valid[j]):
                continue
            overlay = img.copy()
            cv2.line(overlay, pt(i), pt(j), (0, 240, 255), 8, cv2.LINE_AA)
            cv2.addWeighted(overlay, 0.25, img, 0.75, 0, img)


def draw_label_3d(img: np.ndarray, obj: KITTIObj,
                  uv: np.ndarray, valid: np.ndarray,
                  highlight: bool = False) -> None:
    """3D kutunun üstüne sınıf + mesafe etiketi çizer."""
    vc = uv[valid]
    if len(vc) == 0:
        return
    mx   = int(vc[:, 0].mean())
    ty   = max(int(vc[:, 1].min()) - 10, 16)
    text = f"{obj.cls}  {obj.dist:.1f}m"
    font = cv2.FONT_HERSHEY_DUPLEX
    sc   = 0.62 if highlight else 0.55
    (tw, th), bl = cv2.getTextSize(text, font, sc, 1)
    x0, y0 = mx - tw // 2 - 5, ty - th - bl - 2
    x1, y1 = mx + tw // 2 + 5, ty + bl + 1
    ov = img.copy()
    cv2.rectangle(ov, (x0, y0), (x1, y1), (4, 8, 18), -1)
    cv2.addWeighted(ov, 0.65, img, 0.35, 0, img)
    color = CLASS_BGR.get(obj.cls, (200, 200, 200))
    if highlight:
        color = (0, 240, 255)
    cv2.putText(img, text, (mx - tw // 2, ty), font, sc, color, 1, cv2.LINE_AA)


# ══════════════════════════════════════════════════════════════════════════════
#  BEV RENDERER  ──  Kuş Bakışı Görünüm
# ══════════════════════════════════════════════════════════════════════════════

class BEVRenderer:
    def __init__(self, W: int = 560, H: int = 460, r: float = BEV_R):
        self.W, self.H, self.R = W, H, r
        self.sx = W / (2 * r)
        self.sz = H / r

    def _px(self, x, z):
        """Kamera koordinatını piksel konumuna çevirir."""
        return (
            (self.W // 2 + np.asarray(x) * self.sx).astype(np.int32),
            (self.H     - np.asarray(z) * self.sz).astype(np.int32),
        )

    def render(self, cloud: np.ndarray, rings: np.ndarray,
               objects: List[KITTIObj], sel: int = -1) -> np.ndarray:
        img = np.zeros((self.H, self.W, 3), np.uint8)

        # ── Izgara çizgileri ──────────────────────────────────────────────
        for d in range(10, int(self.R) + 1, 10):
            py = int(self.H - d * self.sz)
            cv2.line(img, (0, py), (self.W, py), (10, 20, 10), 1)
            cv2.putText(img, f"{d}m", (5, py - 3),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.36, (28, 55, 28), 1)
        cv2.line(img, (self.W // 2, 0), (self.W // 2, self.H), (10, 20, 10), 1)

        # ── LiDAR halkaları (beyaz tonları) ──────────────────────────────
        if rings is not None and len(rings):
            mask = (
                (rings[:, 2] > 0.3) &
                (rings[:, 2] < self.R) &
                (np.abs(rings[:, 0]) < self.R)
            )
            rx, rz = rings[mask, 0], rings[mask, 2]
            px, py = self._px(rx, rz)
            v      = (px >= 0) & (px < self.W) & (py >= 0) & (py < self.H)
            intens = np.clip((rings[mask][v, 3] * 200).astype(np.uint8), 25, 200)
            img[py[v], px[v]] = np.stack([intens, intens, intens], axis=1)

        # ── Yoğun nesne nokta bulutu (mavi tonları) ──────────────────────
        if cloud is not None and len(cloud):
            mask = (
                (cloud[:, 2] > 0.2) &
                (cloud[:, 2] < self.R) &
                (np.abs(cloud[:, 0]) < self.R)
            )
            cx, cz = cloud[mask, 0], cloud[mask, 2]
            px, py = self._px(cx, cz)
            v      = (px >= 0) & (px < self.W) & (py >= 0) & (py < self.H)
            intens = np.clip((cloud[mask][v, 3] * 255).astype(np.uint8), 60, 255)
            col    = np.stack([intens // 4, intens // 2, intens], axis=1)
            img[py[v], px[v]] = col

        # ── Nesne kutuları + yaw okları ───────────────────────────────────
        for idx, obj in enumerate(objects):
            if obj.cls == "DontCare":
                continue
            color   = CLASS_BGR.get(obj.cls, (180, 180, 180))
            cx_px   = int(self.W // 2 + obj.x * self.sx)
            cz_px   = int(self.H     - obj.z * self.sz)
            hl, hw  = obj.l * self.sx / 2, obj.w * self.sx / 2
            c, s    = math.cos(-obj.ry), math.sin(-obj.ry)
            corners = []
            for sx2, sz2 in [(-1, -1), (1, -1), (1, 1), (-1, 1)]:
                lx, lz = sx2 * hl, sz2 * hw
                corners.append((
                    int(cx_px + c * lx + s * lz),
                    int(cz_px - (-s * lx + c * lz)),
                ))
            pts     = np.array(corners, np.int32)
            thk     = 3 if idx == sel else 1
            box_col = (0, 40, 255) if obj.critical else color

            if idx == sel:
                ov = img.copy()
                cv2.fillPoly(ov, [pts], (0, 30, 60))
                cv2.addWeighted(ov, 0.35, img, 0.65, 0, img)
            cv2.polylines(img, [pts], True, box_col, thk, cv2.LINE_AA)

            # Yaw oku
            al  = max(int(hl * 1.8), 18)
            tip = (
                int(cx_px + math.sin(obj.ry) * al),
                int(cz_px - math.cos(obj.ry) * al),
            )
            oc = (0, 60, 255) if obj.critical else (0, 220, 255)
            cv2.arrowedLine(img, (cx_px, cz_px), tip, oc, 2,
                            cv2.LINE_AA, tipLength=0.42)
            if idx == sel:
                cv2.arrowedLine(img, (cx_px, cz_px), tip, (0, 255, 255),
                                3, cv2.LINE_AA, tipLength=0.45)

            cv2.putText(img, f"#{idx} {obj.dist:.0f}m",
                        (cx_px + 4, cz_px - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.32,
                        (180, 180, 180), 1, cv2.LINE_AA)

        # ── Ego araç simgesi ──────────────────────────────────────────────
        tri = np.array([
            [self.W // 2 - 7, self.H - 4],
            [self.W // 2 + 7, self.H - 4],
            [self.W // 2,     self.H - 22],
        ], np.int32)
        cv2.fillPoly(img, [tri], (0, 255, 180))
        cv2.putText(img, "EGO", (self.W // 2 - 10, self.H - 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 200, 140), 1)
        return img


# ══════════════════════════════════════════════════════════════════════════════
#  CAMERA RENDERER  ──  3D BBox Projeksiyonu + Vision Efektleri
# ══════════════════════════════════════════════════════════════════════════════

class CameraRenderer:
    def render(self, img_bgr: np.ndarray,
               objects: List[KITTIObj],
               calib: KITTICalib,
               sel: int = -1,
               mode: str = "normal") -> np.ndarray:
        """
        mode: 'normal' | 'heatmap' | 'segmentation'
        """
        out = img_bgr.copy()

        if mode == "heatmap":
            gray = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)
            heat = cv2.applyColorMap(gray, cv2.COLORMAP_JET)
            out  = cv2.addWeighted(heat, 0.75, out, 0.25, 0)
        elif mode == "segmentation":
            out  = self._segment_road(out)

        for idx, obj in enumerate(objects):
            if obj.cls == "DontCare":
                continue
            highlight = (idx == sel)
            color     = CLASS_BGR.get(obj.cls, (180, 180, 180))
            if obj.critical:
                color = (0, 40, 255)

            x1, y1, x2, y2 = [int(v) for v in obj.bbox2d]
            cv2.rectangle(out, (x1, y1), (x2, y2),
                          (0, 30, 200) if obj.critical else color,
                          2, cv2.LINE_AA)

            if calib:
                corners = box_corners_cam(obj)
                uv, valid = calib.project(corners)
                if valid.sum() >= 2:
                    draw_box3d(out, uv, valid, color, highlight)
                    draw_label_3d(out, obj, uv, valid, highlight)

        return out

    @staticmethod
    def _segment_road(img: np.ndarray) -> np.ndarray:
        """Gri yol piksellerini mor renk katmanıyla vurgular."""
        hsv  = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(
            hsv,
            np.array([0,   0,  60]),
            np.array([180, 50, 180]),
        )
        H, W      = img.shape[:2]
        road_mask = np.zeros((H, W), np.uint8)
        road_mask[H // 3:, :] = mask[H // 3:, :]
        road_mask = cv2.morphologyEx(
            road_mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8)
        )
        out          = img.copy()
        purple_layer = np.zeros_like(img)
        purple_layer[:] = (180, 0, 180)
        out[road_mask > 0] = cv2.addWeighted(
            img, 0.35, purple_layer, 0.65, 0
        )[road_mask > 0]
        return out