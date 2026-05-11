# ══════════════════════════════════════════════════════════════════════════════
#  dashboard.py  —  Ana Dashboard Penceresi (CustomTkinter 4-Panel Grid)
# ══════════════════════════════════════════════════════════════════════════════

import numpy as np
import cv2
from PIL import Image
from tkinter import filedialog
from typing import Optional, List

import customtkinter as ctk

from config import (
    C_BG0, C_BG1, C_BG2, C_BG3, C_CYN, C_GRN, C_RED,
    C_YLW, C_DIM, C_BDR, C_ORG, C_PRP, C_WHT,
    FNT_HDR, FNT_MD, FNT_SM, FNT_XS, RING_COUNT,
)
from data import KITTIDataset, KITTIObj
from lidar import PseudoLiDAR
from rendering import BEVRenderer, CameraRenderer
from ai_engine import AIEngine

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("dark-blue")


class Dashboard(ctk.CTk):

    def __init__(self):
        super().__init__()
        self.title("KITTI AUTONOMOUS PERCEPTION  ·  v4.0  NEURAL-VISION")
        self.geometry("1640x1000")
        self.minsize(1280, 800)
        self.configure(fg_color=C_BG0)

        # ── Uygulama Durumu ───────────────────────────────────────────────
        self.dataset   : Optional[KITTIDataset] = None
        self.cur_idx   : int                    = 0
        self.objects   : List[KITTIObj]         = []
        self.sel_id    : int                    = 0
        self.playing   : bool                   = False
        self._play_job                          = None
        self.cam_mode  : str                    = "normal"
        self._defer_job                         = None
        self._pulse_state                       = True

        # ── Render Motorları ──────────────────────────────────────────────
        self.cam_rndr = CameraRenderer()
        self.bev_rndr = BEVRenderer(W=560, H=440)
        self.ai_eng   = AIEngine()

        self._build_ui()
        self._pulse()

    # ══════════════════════════════════════════════════════════════════════
    #  UI İNŞASI
    # ══════════════════════════════════════════════════════════════════════

    def _build_ui(self):
        self._build_header()
        self._build_main_grid()
        self._build_footer()

    def _build_header(self):
        hdr = ctk.CTkFrame(self, fg_color=C_BG1, height=50, corner_radius=0)
        hdr.pack(fill="x")
        hdr.pack_propagate(False)

        self.pulse_dot = ctk.CTkLabel(
            hdr, text="◉", font=("Courier New", 16), text_color=C_GRN)
        self.pulse_dot.pack(side="left", padx=(14, 6), pady=10)

        ctk.CTkLabel(
            hdr, text="KITTI  AUTONOMOUS  PERCEPTION  DASHBOARD",
            font=("Courier New", 15, "bold"), text_color=C_CYN,
        ).pack(side="left", pady=10)

        self.frame_lbl = ctk.CTkLabel(
            hdr, text="FRAME —/—", font=FNT_MD, text_color=C_DIM)
        self.frame_lbl.pack(side="right", padx=20)

        self.status_lbl = ctk.CTkLabel(
            hdr, text="● HAZIR", font=FNT_MD, text_color=C_GRN)
        self.status_lbl.pack(side="right", padx=10)

    def _build_main_grid(self):
        main = ctk.CTkFrame(self, fg_color=C_BG0, corner_radius=0)
        main.pack(fill="both", expand=True, padx=5, pady=(4, 2))
        main.columnconfigure(0, weight=1, minsize=300)
        main.columnconfigure(1, weight=3)
        main.columnconfigure(2, weight=2)
        main.rowconfigure(0, weight=1)
        main.rowconfigure(1, weight=1)

        self._build_left_top(main)
        self._build_left_bottom(main)
        self._build_camera_panel(main)
        self._build_bev_panel(main)

    def _build_footer(self):
        ftr = ctk.CTkFrame(self, fg_color=C_BG1, height=28, corner_radius=0)
        ftr.pack(fill="x", side="bottom")
        ftr.pack_propagate(False)
        self.footer_lbl = ctk.CTkLabel(
            ftr, text="KITTI Perception Dashboard v4.0  ·  Hazır",
            font=("Courier New", 9), text_color=C_DIM)
        self.footer_lbl.pack(side="left", padx=14, pady=4)
        ctk.CTkLabel(
            ftr, text="numpy  ·  opencv  ·  Pillow  ·  customtkinter",
            font=("Courier New", 9), text_color=C_DIM,
        ).pack(side="right", padx=14, pady=4)

    # ── Sol Üst: Veri Seti + Nesne Listesi ───────────────────────────────

    def _build_left_top(self, parent):
        frame = self._card(parent, "◈  VERİ SETİ  &  NESNE LİSTESİ")
        frame.grid(row=0, column=0, sticky="nsew", padx=(0, 4), pady=(0, 4))

        inner = ctk.CTkScrollableFrame(
            frame, fg_color="transparent",
            scrollbar_fg_color=C_BG2,
            scrollbar_button_color=C_BDR)
        inner.pack(fill="both", expand=True, padx=6, pady=4)

        # Dizin girişleri
        self._lbl(inner, "VERİ YOLLARI", C_DIM, FNT_XS)
        self.img_ent = self._dir_row(inner, "image_2 dizini")
        self.lbl_ent = self._dir_row(inner, "label_2 dizini")
        self.cal_ent = self._dir_row(inner, "calib dizini")

        ctk.CTkButton(
            inner, text="◈  VERİ SETİ YÜKLE",
            fg_color=C_GRN, text_color="#000",
            hover_color="#00cc66", font=("Courier New", 11, "bold"),
            height=34, corner_radius=4, command=self._load,
        ).pack(fill="x", padx=2, pady=(6, 8))

        # Navigasyon
        self._lbl(inner, "NAVİGASYON", C_DIM, FNT_XS)
        self.frame_nav_lbl = self._lbl(inner, "Frame — / —", C_CYN, FNT_MD)
        nav = ctk.CTkFrame(inner, fg_color="transparent")
        nav.pack(fill="x", pady=2)
        for txt, fn, w in [
            ("◀◀", self._p10, 44), ("◀", self._p1, 36),
            ("▶",  self._n1,  36), ("▶▶", self._n10, 44),
        ]:
            self._nav_btn(nav, txt, fn, w)

        self.play_btn = ctk.CTkButton(
            inner, text="▶  OYNAT", height=30, corner_radius=3,
            fg_color=C_BG2, hover_color=C_BG3,
            text_color=C_CYN, font=FNT_MD, command=self._toggle_play)
        self.play_btn.pack(fill="x", padx=2, pady=2)

        # Nesne tablosu
        self._sep(inner)
        self._lbl(inner, "ID   TİP             MESAFE   DURUM", C_DIM, FNT_XS)
        self._sep(inner)
        self.obj_table = ctk.CTkFrame(inner, fg_color="transparent")
        self.obj_table.pack(fill="x")

        # İstatistik kartları
        self._sep(inner)
        stats = ctk.CTkFrame(inner, fg_color="transparent")
        stats.pack(fill="x", pady=4)
        self.stat_lbls = {}
        for key, title in [
            ("total", "TOPLAM"), ("crit", "KRİTİK"),
            ("near",  "EN YAKIN"), ("mode", "MOD"),
        ]:
            col = ctk.CTkFrame(stats, fg_color=C_BG2, corner_radius=6)
            col.pack(side="left", padx=2, fill="y", expand=True)
            self._lbl(col, title, C_DIM, FNT_XS).pack(pady=(5, 0))
            lbl = ctk.CTkLabel(
                col, text="—", font=("Courier New", 13, "bold"),
                text_color=C_CYN)
            lbl.pack(pady=(2, 5))
            self.stat_lbls[key] = lbl

    # ── Sol Alt: AI Karar Analizi ─────────────────────────────────────────

    def _build_left_bottom(self, parent):
        frame = self._card(parent, "◈  YAPAY ZEKA KARAR ANALİZİ")
        frame.grid(row=1, column=0, sticky="nsew", padx=(0, 4), pady=0)

        # kinematic_lbl önce oluşturulmalı (ai_box'tan önce referans alınıyor)
        self.kinematic_lbl = ctk.CTkLabel(
            frame, text="— nesne seçin —",
            font=FNT_XS, text_color=C_DIM,
            anchor="w", wraplength=270)
        self.kinematic_lbl.pack(fill="x", padx=10, pady=(4, 2))

        self.ai_box = ctk.CTkTextbox(
            frame, height=130, fg_color="#020810",
            text_color=C_GRN, font=("Courier New", 10),
            border_width=1, border_color=C_BDR,
            corner_radius=4, state="disabled", wrap="word")
        self.ai_box.pack(fill="x", padx=8, pady=(0, 4))

        self._sep(frame)

        # Vision modu butonları
        efx = ctk.CTkFrame(frame, fg_color="transparent")
        efx.pack(fill="x", padx=8, pady=4)
        ctk.CTkLabel(efx, text="VISION MODU",
                     font=FNT_XS, text_color=C_DIM).pack(anchor="w")
        btn_row = ctk.CTkFrame(efx, fg_color="transparent")
        btn_row.pack(fill="x", pady=2)

        self.heat_btn = ctk.CTkButton(
            btn_row, text="🌡  ISI HARİTASI", width=130, height=30,
            fg_color=C_BG2, hover_color=C_BG3,
            text_color="#ff8844", font=FNT_SM, corner_radius=3,
            command=lambda: self._set_mode("heatmap"))
        self.heat_btn.pack(side="left", padx=(0, 4))

        self.seg_btn = ctk.CTkButton(
            btn_row, text="🔮  SEGMENTASYON", width=140, height=30,
            fg_color=C_BG2, hover_color=C_BG3,
            text_color="#bf00ff", font=FNT_SM, corner_radius=3,
            command=lambda: self._set_mode("segmentation"))
        self.seg_btn.pack(side="left")

        ctk.CTkButton(
            btn_row, text="⬜  NORMAL", width=90, height=30,
            fg_color=C_BG2, hover_color=C_BG3,
            text_color=C_DIM, font=FNT_SM, corner_radius=3,
            command=lambda: self._set_mode("normal"),
        ).pack(side="left", padx=4)

        self._sep(frame)

        # LiDAR halka yoğunluğu kaydırıcısı
        ring_row = ctk.CTkFrame(frame, fg_color="transparent")
        ring_row.pack(fill="x", padx=8, pady=4)
        ctk.CTkLabel(ring_row, text="LiDAR HALKA YOĞUNLUĞU",
                     font=FNT_XS, text_color=C_DIM).pack(anchor="w")
        self.ring_slider = ctk.CTkSlider(
            ring_row, from_=4, to=30, number_of_steps=26,
            fg_color=C_BG2, progress_color=C_CYN,
            button_color=C_CYN, button_hover_color="#00c0dd",
            command=self._ring_changed)
        self.ring_slider.set(RING_COUNT)
        self.ring_slider.pack(fill="x", pady=2)

    # ── Sağ Üst: Kamera ──────────────────────────────────────────────────

    def _build_camera_panel(self, parent):
        frame = self._card(parent, "◈  KAMERA  ·  3D Projeksiyon  (P2)")
        frame.grid(row=0, column=1, columnspan=2, sticky="nsew",
                   padx=(0, 0), pady=(0, 4))
        self.cam_lbl = ctk.CTkLabel(frame, text="", fg_color="#000",
                                     corner_radius=3)
        self.cam_lbl.pack(fill="both", expand=True, padx=6, pady=(0, 6))
        frame.bind("<Configure>", lambda e: self._deferred_refresh())

    # ── Sağ Alt: LiDAR BEV ───────────────────────────────────────────────

    def _build_bev_panel(self, parent):
        frame = self._card(parent, "◈  LiDAR BEV  ·  Pseudo Point Cloud + Rings")
        frame.grid(row=1, column=1, columnspan=2, sticky="nsew",
                   padx=0, pady=0)
        self.bev_lbl = ctk.CTkLabel(frame, text="", fg_color="#000",
                                     corner_radius=3)
        self.bev_lbl.pack(fill="both", expand=True, padx=6, pady=(0, 6))

    # ══════════════════════════════════════════════════════════════════════
    #  YARDIMCI WIDGET ÜRETİCİLER
    # ══════════════════════════════════════════════════════════════════════

    def _card(self, parent, title: str) -> ctk.CTkFrame:
        """Başlıklı kart çerçevesi oluşturur. İçi pack kullanır."""
        f = ctk.CTkFrame(parent, fg_color=C_BG1,
                         corner_radius=8, border_width=1,
                         border_color=C_BDR)
        ctk.CTkLabel(f, text=title, font=FNT_HDR,
                     text_color=C_CYN, anchor="w",
        ).pack(side="top", fill="x", padx=12, pady=(8, 2))
        return f

    @staticmethod
    def _lbl(parent, text: str, color: str, font) -> ctk.CTkLabel:
        lbl = ctk.CTkLabel(parent, text=text, font=font,
                            text_color=color, anchor="w")
        lbl.pack(fill="x", padx=6, pady=(4, 0))
        return lbl

    @staticmethod
    def _sep(parent) -> None:
        ctk.CTkFrame(parent, fg_color=C_BDR, height=1
        ).pack(fill="x", padx=6, pady=3)

    def _dir_row(self, parent, placeholder: str) -> ctk.CTkEntry:
        row = ctk.CTkFrame(parent, fg_color="transparent")
        row.pack(fill="x", pady=1)
        e = ctk.CTkEntry(
            row, placeholder_text=placeholder,
            font=FNT_SM, height=28,
            fg_color=C_BG0, border_color=C_BDR,
            text_color=C_CYN, corner_radius=3)
        e.pack(side="left", fill="x", expand=True, padx=(0, 2))
        ctk.CTkButton(
            row, text="…", width=30, height=28,
            fg_color=C_BG2, hover_color=C_BDR,
            text_color=C_DIM, font=FNT_SM, corner_radius=3,
            command=lambda en=e: self._browse(en),
        ).pack(side="right")
        return e

    @staticmethod
    def _browse(entry: ctk.CTkEntry) -> None:
        d = filedialog.askdirectory(title="Dizin Seç")
        if d:
            entry.delete(0, "end")
            entry.insert(0, d)

    def _nav_btn(self, parent, txt: str, fn, w: int) -> None:
        ctk.CTkButton(
            parent, text=txt, width=w, height=30,
            fg_color=C_BG2, hover_color=C_BG3,
            text_color=C_CYN, font=FNT_MD,
            corner_radius=3, command=fn,
        ).pack(side="left", padx=1)

    # ══════════════════════════════════════════════════════════════════════
    #  VERİ SETİ YÜKLEME
    # ══════════════════════════════════════════════════════════════════════

    def _load(self) -> None:
        img = self.img_ent.get().strip()
        lbl = self.lbl_ent.get().strip()
        cal = self.cal_ent.get().strip()
        if not (img and lbl and cal):
            self._status("image_2, label_2 ve calib dizinlerini girin.", err=True)
            return
        try:
            self.dataset = KITTIDataset(img, lbl, cal)
            self.cur_idx = 0
            self._status(f"✓  {len(self.dataset)} frame yüklendi.")
            self._refresh()
        except Exception as exc:
            self._status(f"HATA: {exc}", err=True)

    def _status(self, msg: str, err: bool = False) -> None:
        color = C_RED if err else C_GRN
        self.status_lbl.configure(text=f"● {msg}", text_color=color)
        self.footer_lbl.configure(text=msg)

    # ══════════════════════════════════════════════════════════════════════
    #  ANA GÜNCELLEME (REFRESH)
    # ══════════════════════════════════════════════════════════════════════

    def _refresh(self) -> None:
        if not self.dataset or not len(self.dataset):
            return
        self.cur_idx = max(0, min(self.cur_idx, len(self.dataset) - 1))

        img_bgr, objects, calib, fid = self.dataset.get(self.cur_idx)
        self.objects = [o for o in objects if o.cls != "DontCare"]

        self.frame_lbl.configure(
            text=f"FRAME  {self.cur_idx+1:04d}  /  {len(self.dataset):04d}  ·  {fid}")
        self.frame_nav_lbl.configure(
            text=f"Frame: {self.cur_idx+1} / {len(self.dataset)}")

        if self.sel_id >= len(self.objects):
            self.sel_id = 0

        # İstatistikler
        crits = [o for o in self.objects if o.critical]
        near  = min(self.objects, key=lambda o: o.dist).dist \
                if self.objects else 0.0
        self.stat_lbls["total"].configure(text=str(len(self.objects)))
        self.stat_lbls["crit"].configure(
            text=str(len(crits)),
            text_color=C_RED if crits else C_GRN)
        self.stat_lbls["near"].configure(text=f"{near:.1f}m")
        self.stat_lbls["mode"].configure(
            text="ACİL" if crits else "OTOMATİK",
            text_color=C_RED if crits else C_GRN)

        self._rebuild_table()
        self._update_ai()

        # Nokta bulutu + halkalar
        cloud = PseudoLiDAR.generate(self.objects)
        rings = PseudoLiDAR.lidar_rings(n_rings=int(self.ring_slider.get()))

        # Kamera görüntüsü
        cam_out = self.cam_rndr.render(
            img_bgr if img_bgr is not None else np.zeros((375, 1242, 3), np.uint8),
            self.objects, calib, sel=self.sel_id, mode=self.cam_mode,
        )
        W_c = max(self.cam_lbl.winfo_width(),  100)
        H_c = max(self.cam_lbl.winfo_height(), 80)
        self._set_img(self.cam_lbl, cam_out, W_c, H_c)

        # BEV görüntüsü
        bev_out = self.bev_rndr.render(cloud, rings, self.objects, self.sel_id)
        W_b = max(self.bev_lbl.winfo_width(),  100)
        H_b = max(self.bev_lbl.winfo_height(), 80)
        self._set_img(self.bev_lbl, bev_out, W_b, H_b)

    def _rebuild_table(self) -> None:
        for w in self.obj_table.winfo_children():
            w.destroy()
        for idx, obj in enumerate(self.objects[:35]):
            bg      = C_BG3 if idx == self.sel_id else C_BG2
            is_crit = obj.critical
            row     = ctk.CTkFrame(self.obj_table, fg_color=bg,
                                    height=28, corner_radius=2)
            row.pack(fill="x", pady=1, padx=1)
            row.pack_propagate(False)
            row.bind("<Button-1>", lambda e, i=idx: self._select(i))
            for txt, col, wid in [
                (f"{idx:02d}",       C_CYN,                    30),
                (obj.label,          C_WHT,                    80),
                (f"{obj.dist:.1f}m", C_WHT,                    58),
                ("KRİTİK" if is_crit else "Normal",
                 C_RED    if is_crit else C_GRN,               64),
            ]:
                lbl = ctk.CTkLabel(row, text=txt,
                                   font=("Courier New", 11),
                                   text_color=col, width=wid, anchor="w")
                lbl.pack(side="left", padx=3)
                lbl.bind("<Button-1>", lambda e, i=idx: self._select(i))

    def _update_ai(self) -> None:
        if not self.objects:
            return
        obj      = self.objects[self.sel_id]
        msg, is_crit = self.ai_eng.analyze(obj, self.cur_idx)
        self.ai_box.configure(state="normal")
        self.ai_box.delete("1.0", "end")
        self.ai_box.insert("end", msg)
        self.ai_box.configure(
            state="disabled",
            text_color=C_RED if is_crit else C_GRN)
        self.kinematic_lbl.configure(
            text=self.ai_eng.kinematic_line(obj),
            text_color=C_RED if is_crit else C_DIM)

    def _select(self, idx: int) -> None:
        self.sel_id = idx
        self._refresh()

    # ══════════════════════════════════════════════════════════════════════
    #  GÖRÜNTÜ GÖSTERME
    # ══════════════════════════════════════════════════════════════════════

    @staticmethod
    def _set_img(label: ctk.CTkLabel, bgr: np.ndarray,
                 max_w: int, max_h: int) -> None:
        rgb    = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        h, w   = rgb.shape[:2]
        sc     = min(max_w / w, max_h / h, 1.0)
        nw, nh = max(1, int(w * sc)), max(1, int(h * sc))
        rgb    = cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_LINEAR)
        pil    = Image.fromarray(rgb)
        cim    = ctk.CTkImage(light_image=pil, dark_image=pil, size=(nw, nh))
        label.configure(image=cim, text="")
        label._cim = cim   # referans tutulmazsa garbage collected olur

    # ══════════════════════════════════════════════════════════════════════
    #  NAVİGASYON
    # ══════════════════════════════════════════════════════════════════════

    def _p1(self):
        if self.dataset:
            self.cur_idx = max(0, self.cur_idx - 1)
            self._refresh()

    def _n1(self):
        if self.dataset:
            self.cur_idx = min(self.cur_idx + 1, len(self.dataset) - 1)
            self._refresh()

    def _p10(self):
        if self.dataset:
            self.cur_idx = max(0, self.cur_idx - 10)
            self._refresh()

    def _n10(self):
        if self.dataset:
            self.cur_idx = min(self.cur_idx + 10, len(self.dataset) - 1)
            self._refresh()

    def _toggle_play(self) -> None:
        self.playing = not self.playing
        if self.playing:
            self.play_btn.configure(text="⏸  DURDUR", text_color=C_YLW)
            self._play_loop()
        else:
            self.play_btn.configure(text="▶  OYNAT", text_color=C_CYN)
            if self._play_job:
                self.after_cancel(self._play_job)

    def _play_loop(self) -> None:
        if not self.playing:
            return
        self._n1()
        if self.dataset and self.cur_idx < len(self.dataset) - 1:
            self._play_job = self.after(130, self._play_loop)
        else:
            self._toggle_play()

    # ══════════════════════════════════════════════════════════════════════
    #  VISION MODU + HALKA KAYDIRICI
    # ══════════════════════════════════════════════════════════════════════

    def _set_mode(self, mode: str) -> None:
        self.cam_mode = mode
        self.heat_btn.configure(fg_color=C_BG3 if mode == "heatmap"      else C_BG2)
        self.seg_btn.configure( fg_color=C_BG3 if mode == "segmentation" else C_BG2)
        self._refresh()

    def _ring_changed(self, _) -> None:
        self._deferred_refresh()

    # ══════════════════════════════════════════════════════════════════════
    #  ANİMASYON + RESIZE DEFERRED REFRESH
    # ══════════════════════════════════════════════════════════════════════

    def _pulse(self) -> None:
        self._pulse_state = not self._pulse_state
        self.pulse_dot.configure(
            text_color=C_GRN if self._pulse_state else "#006633")
        self.after(900, self._pulse)

    def _deferred_refresh(self) -> None:
        if self._defer_job:
            self.after_cancel(self._defer_job)
        self._defer_job = self.after(80, self._refresh)