# ══════════════════════════════════════════════════════════════════════════════
#  data.py  —  KITTI veri yapıları: Kalibrasyon, Nesne, Dataset
# ══════════════════════════════════════════════════════════════════════════════

import math
import numpy as np
import cv2
from pathlib import Path
from typing import List

from config import RISK_M, DISPLAY


class KITTICalib:
    """P2 (3×4) projeksiyon, R0 düzeltme, Tr velodyne→cam."""

    def __init__(self, path: str):
        self.P2 = np.eye(3, 4, dtype=np.float64)
        self.R0 = np.eye(3,    dtype=np.float64)
        self.Tr = np.eye(3, 4, dtype=np.float64)
        self._load(path)

    def _load(self, path: str):
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                key, *vals = line.split()
                arr = np.array(vals, dtype=np.float64)
                if key.startswith("P2"):
                    self.P2 = arr.reshape(3, 4)
                elif key.startswith("R0_rect") or key.startswith("R_rect"):
                    self.R0 = arr.reshape(3, 3)
                elif key.startswith("Tr_velo_to_cam"):
                    self.Tr = arr.reshape(3, 4)

    def project(self, pts_cam: np.ndarray):
        """(N,3) kamera koordinatı → (N,2) piksel + geçerlilik maskesi."""
        N  = len(pts_cam)
        h  = np.hstack([pts_cam, np.ones((N, 1))])
        R4 = np.eye(4)
        R4[:3, :3] = self.R0
        pr    = (self.P2 @ R4 @ h.T).T
        valid = pr[:, 2] > 0.05
        uv    = np.full((N, 2), -1.0)
        uv[valid] = pr[valid, :2] / pr[valid, 2:3]
        return uv, valid


class KITTIObj:
    """Tek bir KITTI etiket satırını temsil eder."""

    __slots__ = [
        "cls", "h", "w", "l", "x", "y", "z", "ry",
        "bbox2d", "truncated", "occluded", "alpha", "dist",
    ]

    def __init__(self, line: str):
        p = line.strip().split()
        self.cls       = p[0]
        self.truncated = float(p[1])
        self.occluded  = int(p[2])
        self.alpha     = float(p[3])
        self.bbox2d    = list(map(float, p[4:8]))
        self.h, self.w, self.l = float(p[8]),  float(p[9]),  float(p[10])
        self.x, self.y, self.z = float(p[11]), float(p[12]), float(p[13])
        self.ry        = float(p[14])
        self.dist      = math.hypot(self.x, self.z)

    @property
    def critical(self) -> bool:
        return self.dist < RISK_M and self.cls != "DontCare"

    @property
    def label(self) -> str:
        return DISPLAY.get(self.cls, self.cls)


def load_labels(path) -> List[KITTIObj]:
    """Bir etiket dosyasından tüm nesneleri yükler."""
    with open(path) as f:
        return [KITTIObj(line) for line in f if line.strip()]


class KITTIDataset:
    """image_2 / label_2 / calib üçlüsünden oluşan KITTI veri seti."""

    def __init__(self, img_dir: str, lbl_dir: str, cal_dir: str):
        self.img_dir = Path(img_dir)
        self.lbl_dir = Path(lbl_dir)
        self.cal_dir = Path(cal_dir)
        imgs     = sorted(self.img_dir.glob("*.png")) or \
                   sorted(self.img_dir.glob("*.jpg"))
        self.ids = [f.stem for f in imgs]

    def __len__(self) -> int:
        return len(self.ids)

    def get(self, idx: int):
        """(img_bgr, objects, calib, frame_id) döndürür."""
        fid   = self.ids[idx]
        img   = cv2.imread(str(self.img_dir / f"{fid}.png"))
        if img is None:
            img = cv2.imread(str(self.img_dir / f"{fid}.jpg"))
        lp    = self.lbl_dir / f"{fid}.txt"
        cp    = self.cal_dir / f"{fid}.txt"
        objs  = load_labels(lp)         if lp.exists() else []
        calib = KITTICalib(str(cp))     if cp.exists() else None
        return img, objs, calib, fid