"""Dark customtkinter UI for Dragonfly Lab."""
from __future__ import annotations

import math
import queue
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog

import customtkinter as ctk
import cv2
import numpy as np
from PIL import Image, ImageTk

from . import core

# ---- palette (dark) -------------------------------------------------------
BG, PANEL, CARD, BORDER = "#0b0d12", "#12151c", "#181c25", "#262b38"
TEXT, MUTED = "#d7dbe6", "#7d8499"
ACCENT, ACCENT_H = "#6d5efc", "#8478ff"
DANGER, DANGER_H = "#3a1f27", "#55283a"

# overlay colours (BGR for OpenCV)
C_BODY, C_WING = (191, 212, 45), (246, 92, 139)
C_DUST, C_DUST_DONE = (36, 191, 251), (128, 222, 74)
C_MASK = (94, 63, 244)


class App(ctk.CTk):
    def __init__(self) -> None:
        super().__init__(fg_color=BG)
        self.title("Dragonfly Lab")
        self.geometry("1440x900")
        self.minsize(1050, 650)

        # image / analysis state
        self.src: Path | None = None
        self.full: np.ndarray | None = None
        self.proc: np.ndarray | None = None
        self.scale = 1.0
        self.mask: np.ndarray | None = None
        self.det: core.BodyWings | None = None
        self.specks: list[core.Speck] = []
        self.sp_x = self.sp_y = self.sp_r = np.zeros(0)
        self.undo_stack: list[np.ndarray] = []

        # view state
        self.zoom, self.ox, self.oy = 1.0, 0.0, 0.0
        self._drag: str | None = None
        self._last = (0, 0)
        self._photo = None
        self._render_pending = False

        # background work
        self.q: queue.Queue = queue.Queue()
        self.busy = False

        self._build()
        self.bind_all("<Control-z>", lambda e: self.undo())
        self.bind_all("<Command-z>", lambda e: self.undo())
        self.after(60, self._poll)

    # ------------------------------------------------------------------ UI
    def _build(self) -> None:
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        side = ctk.CTkScrollableFrame(self, width=290, fg_color=PANEL, corner_radius=0,
                                      scrollbar_button_color=BORDER, scrollbar_button_hover_color=MUTED)
        side.grid(row=0, column=0, sticky="nsw")

        ctk.CTkLabel(side, text="🪰  Dragonfly Lab", font=ctk.CTkFont(size=20, weight="bold"),
                     text_color=TEXT, anchor="w").pack(fill="x", pady=(8, 2))
        ctk.CTkLabel(side, text="CPU-only wing · body · dust inspector", text_color=MUTED,
                     anchor="w").pack(fill="x", pady=(0, 10))

        self._btn(side, "Open image…", self.open_image, primary=True)

        self._section(side, "DETECT")
        self.sens_bw = self._slider(side, "Body / wing sensitivity", 0.5, 2.0, 1.0, 30)
        self._btn(side, "Detect body & wings", self.detect_body_wings)
        self.sens_dust = self._slider(side, "Dust sensitivity", 1, 10, 5, 9, "{:.0f}")
        self.max_speck = self._slider(side, "Max speck size (px)", 6, 60, 20, 54, "{:.0f}")
        self.ignore_body = ctk.BooleanVar(value=True)
        ctk.CTkSwitch(side, text="Ignore body while hunting dust", variable=self.ignore_body,
                      text_color=TEXT, progress_color=ACCENT, button_color=TEXT,
                      fg_color=BORDER).pack(anchor="w", pady=(4, 6))
        self._btn(side, "Detect dust specks", self.detect_dust)

        self._section(side, "MASK")
        self.tool = ctk.StringVar(value="Brush")
        ctk.CTkSegmentedButton(side, values=["Brush", "Pick"], variable=self.tool,
                               command=lambda v: self._cursor(None), selected_color=ACCENT,
                               selected_hover_color=ACCENT_H, unselected_color=CARD,
                               unselected_hover_color=BORDER, fg_color=CARD,
                               text_color=TEXT).pack(fill="x", pady=(2, 2))
        ctk.CTkLabel(side, text="Brush: L-drag paint · R-drag erase\nPick: click speck / wing / body",
                     text_color=MUTED, justify="left", anchor="w").pack(fill="x")
        self.brush = self._slider(side, "Brush size (px)", 2, 150, 14, 148, "{:.0f}")
        self._btn(side, "Mask all dust", self.mask_dust)
        self._btn(side, "Mask body", lambda: self._mask_region("body"))
        self._btn(side, "Mask wings", lambda: self._mask_region("wings"))
        row = ctk.CTkFrame(side, fg_color="transparent")
        row.pack(fill="x", pady=(4, 0))
        self._btn(row, "Undo", self.undo, side="left")
        self._btn(row, "Clear mask", self.clear_mask, side="left", danger=True)

        self._section(side, "VIEW")
        self.v_body, self.v_wings = ctk.BooleanVar(value=True), ctk.BooleanVar(value=True)
        self.v_dust, self.v_mask = ctk.BooleanVar(value=True), ctk.BooleanVar(value=True)
        for text, var in (("Body", self.v_body), ("Wings", self.v_wings),
                          ("Dust specks", self.v_dust), ("Mask", self.v_mask)):
            ctk.CTkSwitch(side, text=text, variable=var, command=self.request_render, text_color=TEXT,
                          progress_color=ACCENT, button_color=TEXT, fg_color=BORDER).pack(anchor="w", pady=2)
        self._btn(side, "Fit to window", self.fit_and_render)

        self._section(side, "EXPORT")
        self._btn(side, "Save mask + masked copy", self.save, primary=True)
        self.stats = ctk.CTkLabel(side, text="", text_color=MUTED, justify="left", anchor="w")
        self.stats.pack(fill="x", pady=(10, 0))
        ctk.CTkLabel(side, text="Wheel: zoom · Shift/middle-drag: pan", text_color=MUTED,
                     anchor="w").pack(fill="x", pady=(8, 0))

        # canvas
        self.canvas = tk.Canvas(self, bg=BG, highlightthickness=0, cursor="crosshair")
        self.canvas.grid(row=0, column=1, sticky="nsew")
        c = self.canvas
        c.bind("<Configure>", lambda e: self.request_render())
        c.bind("<ButtonPress-1>", lambda e: self._press(e, 1))
        c.bind("<ButtonPress-3>", lambda e: self._press(e, 3))
        c.bind("<ButtonPress-2>", lambda e: self._press(e, 2))
        for seq in ("<B1-Motion>", "<B2-Motion>", "<B3-Motion>"):
            c.bind(seq, self._motion)
        c.bind("<Motion>", self._cursor)
        c.bind("<Leave>", lambda e: c.delete("brush"))
        for seq in ("<ButtonRelease-1>", "<ButtonRelease-2>", "<ButtonRelease-3>"):
            c.bind(seq, self._release)
        c.bind("<MouseWheel>", self._wheel)
        c.bind("<Button-4>", lambda e: self._wheel(e, 1))
        c.bind("<Button-5>", lambda e: self._wheel(e, -1))

        # status bar
        bar = ctk.CTkFrame(self, fg_color=PANEL, corner_radius=0, height=32)
        bar.grid(row=1, column=0, columnspan=2, sticky="ew")
        self.status_lbl = ctk.CTkLabel(bar, text="Open an image to begin.", text_color=MUTED, anchor="w")
        self.status_lbl.pack(side="left", padx=12, pady=4)
        self.progress = ctk.CTkProgressBar(bar, mode="indeterminate", width=180, height=6,
                                           progress_color=ACCENT, fg_color=BORDER)
        self.progress.pack(side="right", padx=12)
        self.progress.set(0)

    def _section(self, parent, text: str) -> None:
        ctk.CTkLabel(parent, text=text, text_color=MUTED, anchor="w",
                     font=ctk.CTkFont(size=11, weight="bold")).pack(fill="x", pady=(18, 4))
        ctk.CTkFrame(parent, height=1, fg_color=BORDER).pack(fill="x", pady=(0, 6))

    def _btn(self, parent, text, cmd, primary=False, danger=False, side=None):
        fg, hov = (ACCENT, ACCENT_H) if primary else (DANGER, DANGER_H) if danger else (CARD, BORDER)
        b = ctk.CTkButton(parent, text=text, command=cmd, fg_color=fg, hover_color=hov, text_color=TEXT,
                          border_width=0 if primary else 1, border_color=BORDER, height=34)
        if side:
            b.pack(side=side, expand=True, fill="x", padx=2)
        else:
            b.pack(fill="x", pady=3)
        return b

    def _slider(self, parent, text, lo, hi, init, steps, fmt="{:.1f}") -> ctk.DoubleVar:
        var = ctk.DoubleVar(value=init)
        row = ctk.CTkFrame(parent, fg_color="transparent")
        row.pack(fill="x", pady=(6, 0))
        ctk.CTkLabel(row, text=text, text_color=TEXT, anchor="w").pack(side="left")
        val = ctk.CTkLabel(row, text=fmt.format(init), text_color=MUTED)
        val.pack(side="right")
        ctk.CTkSlider(parent, from_=lo, to=hi, number_of_steps=steps, variable=var, fg_color=BORDER,
                      progress_color=ACCENT, button_color=ACCENT, button_hover_color=ACCENT_H,
                      command=lambda v: (val.configure(text=fmt.format(v)), self._cursor(None))
                      ).pack(fill="x", pady=(2, 4))
        return var

    def status(self, text: str) -> None:
        self.status_lbl.configure(text=text)

    # ------------------------------------------------------- background tasks
    def _run(self, label: str, work, done) -> None:
        if self.busy:
            self.status("Busy – wait for the current task to finish.")
            return
        self.busy = True
        self.status(label)
        self.progress.start()

        def target():
            try:
                out = (work(), None)
            except Exception as ex:  # noqa: BLE001 - shown in the status bar
                out = (None, ex)
            self.q.put((done, *out))

        threading.Thread(target=target, daemon=True).start()

    def _poll(self) -> None:
        try:
            while True:
                done, res, err = self.q.get_nowait()
                self.busy = False
                self.progress.stop()
                self.progress.set(0)
                if err is not None:
                    self.status(f"Error: {err}")
                else:
                    done(res)
        except queue.Empty:
            pass
        self.after(60, self._poll)

    # ------------------------------------------------------------- actions
    def open_image(self) -> None:
        path = filedialog.askopenfilename(
            title="Choose a dragonfly photo",
            filetypes=[("Images", "*.jpg *.jpeg *.png *.tif *.tiff *.bmp *.webp"), ("All files", "*.*")])
        if not path:
            return
        p = Path(path)

        def work():
            full = core.imread(p)
            proc, scale = core.make_proxy(full)
            saved = core.load_mask(core.work_dir_for(p), proc.shape[:2])
            return full, proc, scale, saved

        def done(res):
            self.src, (self.full, self.proc, self.scale, saved) = p, res
            self.mask = saved if saved is not None else np.zeros(self.proc.shape[:2], np.uint8)
            self.det, self.specks, self.undo_stack = None, [], []
            self._set_specks([])
            self.fit_and_render()
            h, w = self.full.shape[:2]
            self.status(f"{p.name}  ·  {w}×{h}px" + ("  ·  restored saved mask" if saved is not None else ""))
            self._update_stats()

        self._run("Loading image…", work, done)

    def detect_body_wings(self) -> None:
        if not self._need_image():
            return
        proc, sens = self.proc, self.sens_bw.get()

        def done(det):
            self.det = det
            self.status(f"Body {'found' if det.body.any() else 'NOT found'} · {len(det.wing_contours)} wing region(s). "
                        "Adjust sensitivity and re-run if needed.")
            self._update_stats()
            self.request_render()

        self._run("Detecting body and wings…", lambda: core.detect_body_and_wings(proc, sens), done)

    def detect_dust(self) -> None:
        if not self._need_image():
            return
        proc, s, md = self.proc, self.sens_dust.get(), int(self.max_speck.get())
        body = self.det.body if (self.det and self.ignore_body.get()) else None

        def done(specks):
            self._set_specks(specks)
            self.status(f"{len(specks)} dust speck(s) found. Use Pick to toggle one, or Mask all dust.")
            self._update_stats()
            self.request_render()

        self._run("Hunting dust specks…", lambda: core.detect_dust(proc, body, s, md, body is not None), done)

    def _set_specks(self, specks) -> None:
        self.specks = specks
        self.sp_x = np.array([s.cx for s in specks])
        self.sp_y = np.array([s.cy for s in specks])
        self.sp_r = np.array([s.r for s in specks])

    # ------------------------------------------------------------- masking
    def _push_undo(self) -> None:
        self.undo_stack.append(self.mask.copy())
        del self.undo_stack[:-15]

    def undo(self) -> None:
        if self.undo_stack and self.mask is not None:
            self.mask = self.undo_stack.pop()
            self._update_stats()
            self.request_render()

    def clear_mask(self) -> None:
        if self._need_image():
            self._push_undo()
            self.mask[:] = 0
            self._update_stats()
            self.request_render()

    def _paint_speck(self, s: core.Speck, value: int) -> None:
        cnt = s.contour.astype(np.int32)
        cv2.drawContours(self.mask, [cnt], -1, value, cv2.FILLED)
        cv2.drawContours(self.mask, [cnt], -1, value, 5)    # a little margin around the speck

    def mask_dust(self) -> None:
        if not self._need_image():
            return
        if not self.specks:
            self.status("No dust detected yet – press “Detect dust specks” first.")
            return
        self._push_undo()
        for s in self.specks:
            self._paint_speck(s, 255)
        self._update_stats()
        self.request_render()

    def _mask_region(self, which: str) -> None:
        if not self._need_image():
            return
        if self.det is None:
            self.status("Run “Detect body & wings” first.")
            return
        self._push_undo()
        region = self.det.body > 0 if which == "body" else self.det.wings > 0
        self.mask[region] = 255
        self._update_stats()
        self.request_render()

    def _pick(self, e, force: bool | None) -> None:
        x, y = self._to_img(e.x, e.y)
        xi, yi = int(x), int(y)
        H, W = self.mask.shape
        if not (0 <= xi < W and 0 <= yi < H):
            return
        if len(self.specks):
            d = np.hypot(self.sp_x - x, self.sp_y - y)
            i = int(d.argmin())
            if d[i] <= self.sp_r[i] + 14 / self.zoom:
                s = self.specks[i]
                now = self.mask[min(int(s.cy), H - 1), min(int(s.cx), W - 1)] > 0
                want = (not now) if force is None else force
                if want != now:
                    self._push_undo()
                    self._paint_speck(s, 255 if want else 0)
                    self._update_stats()
                    self.request_render()
                return
        if self.det is None:
            return
        region = None
        if self.det.body[yi, xi]:
            region = self.det.body > 0
        elif self.det.wings[yi, xi]:
            region = self.det.wings == self.det.wings[yi, xi]
        if region is not None:
            now = self.mask[yi, xi] > 0
            want = (not now) if force is None else force
            self._push_undo()
            self.mask[region] = 255 if want else 0
            self._update_stats()
            self.request_render()

    def _stamp(self, p0, p1, value: int) -> None:
        r = max(1, int(round(self.brush.get() / 2)))
        a = (int(p0[0]), int(p0[1]))
        b = (int(p1[0]), int(p1[1]))
        cv2.line(self.mask, a, b, value, 2 * r)
        cv2.circle(self.mask, b, r, value, -1)

    # -------------------------------------------------------------- saving
    def save(self) -> None:
        if not self._need_image():
            return
        src, full, mask, det, specks, scale = self.src, self.full, self.mask.copy(), self.det, self.specks, self.scale

        def done(out):
            self.status(f"Saved → {out}")

        self._run("Writing full-resolution files…",
                  lambda: core.save_outputs(src, full, mask, det, specks, scale), done)

    def _update_stats(self) -> None:
        if self.mask is None:
            return
        wings = len(self.det.wing_contours) if self.det else 0
        pct = 100.0 * float((self.mask > 0).mean())
        self.stats.configure(text=f"Wings: {wings}   Dust: {len(self.specks)}\nMasked: {pct:.2f}% of image")

    def _need_image(self) -> bool:
        if self.proc is None:
            self.status("Open an image first.")
            return False
        return True

    # ---------------------------------------------------------------- view
    def _to_img(self, sx: float, sy: float) -> tuple[float, float]:
        return self.ox + sx / self.zoom, self.oy + sy / self.zoom

    def fit_and_render(self) -> None:
        if self.proc is None:
            return
        self.update_idletasks()
        cw, ch = max(self.canvas.winfo_width(), 200), max(self.canvas.winfo_height(), 200)
        H, W = self.proc.shape[:2]
        self.zoom = min(cw / W, ch / H) * 0.96
        self.ox = W / 2 - cw / (2 * self.zoom)
        self.oy = H / 2 - ch / (2 * self.zoom)
        self.request_render()

    def request_render(self) -> None:
        if not self._render_pending:
            self._render_pending = True
            self.after(8, self.render)

    def render(self) -> None:
        self._render_pending = False
        cv = self.canvas
        cv.delete("img", "hud", "hint")
        cw, ch = cv.winfo_width(), cv.winfo_height()
        if self.proc is None:
            cv.create_text(cw / 2, ch / 2, text="Open an image to begin", fill=MUTED,
                           font=("Helvetica", 18), tags="hint")
            return
        H, W = self.proc.shape[:2]
        z = self.zoom
        ix0, iy0 = max(0, int(math.floor(self.ox))), max(0, int(math.floor(self.oy)))
        ix1, iy1 = min(W, int(math.ceil(self.ox + cw / z))), min(H, int(math.ceil(self.oy + ch / z)))
        if ix1 - ix0 < 1 or iy1 - iy0 < 1:
            return
        dw, dh = max(1, round((ix1 - ix0) * z)), max(1, round((iy1 - iy0) * z))
        view = cv2.resize(self.proc[iy0:iy1, ix0:ix1], (dw, dh),
                          interpolation=cv2.INTER_AREA if z < 1 else cv2.INTER_LINEAR)

        def tint(m: np.ndarray, color, alpha: float) -> None:
            mm = cv2.resize(m[iy0:iy1, ix0:ix1], (dw, dh), interpolation=cv2.INTER_NEAREST) > 0
            if mm.any():
                view[mm] = (view[mm] * (1 - alpha) + np.array(color, np.float32) * alpha).astype(np.uint8)

        def tf(c: np.ndarray) -> np.ndarray:
            return ((c.reshape(-1, 2).astype(np.float32) - (ix0, iy0)) * z).astype(np.int32)

        if self.det is not None:
            if self.v_wings.get():
                tint(self.det.wings, C_WING, 0.30)
            if self.v_body.get():
                tint(self.det.body, C_BODY, 0.38)
        if self.v_mask.get():
            tint(self.mask, C_MASK, 0.55)
        if self.det is not None:
            if self.v_wings.get():
                for i, (c, (cx, cy)) in enumerate(zip(self.det.wing_contours, self.det.wing_centroids), 1):
                    cv2.polylines(view, [tf(c)], True, C_WING, 1, cv2.LINE_AA)
                    cv2.putText(view, f"W{i}", (int((cx - ix0) * z) - 10, int((cy - iy0) * z)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (240, 240, 240), 1, cv2.LINE_AA)
            if self.v_body.get():
                for c in self.det.body_contours:
                    cv2.polylines(view, [tf(c)], True, C_BODY, 1, cv2.LINE_AA)

        if self.v_dust.get() and len(self.specks):
            vis = np.flatnonzero((self.sp_x >= ix0 - 20) & (self.sp_x < ix1 + 20) &
                                 (self.sp_y >= iy0 - 20) & (self.sp_y < iy1 + 20))
            for i in vis[:4000]:
                s = self.specks[i]
                done = self.mask[min(int(s.cy), H - 1), min(int(s.cx), W - 1)] > 0
                cv2.circle(view, (int((s.cx - ix0) * z), int((s.cy - iy0) * z)),
                           int(max(7, s.r * z + 5)), C_DUST_DONE if done else C_DUST, 1, cv2.LINE_AA)

        self._photo = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(view, cv2.COLOR_BGR2RGB)))
        cv.create_image(round((ix0 - self.ox) * z), round((iy0 - self.oy) * z), anchor="nw",
                        image=self._photo, tags="img")
        cv.tag_lower("img")
        cv.create_text(12, ch - 10, anchor="sw", fill=MUTED, tags="hud",
                       text=f"{z * self.scale * 100:.0f}% of original pixels")

    def _cursor(self, e) -> None:
        c = self.canvas
        c.delete("brush")
        if e is None:
            return
        self._mouse = (e.x, e.y)
        if self.proc is not None and self.tool.get() == "Brush":
            r = self.brush.get() / 2 * self.zoom
            c.create_oval(e.x - r, e.y - r, e.x + r, e.y + r, outline="#e5e7eb", tags="brush")

    # --------------------------------------------------------------- mouse
    def _press(self, e, button: int) -> None:
        if self.proc is None:
            return
        self._last = (e.x, e.y)
        if button == 2 or (e.state & 0x1):
            self._drag = "pan"
            return
        if self.tool.get() == "Pick":
            self._pick(e, None if button == 1 else False)
            return
        self._push_undo()
        self._drag = "paint" if button == 1 else "erase"
        p = self._to_img(e.x, e.y)
        self._stamp(p, p, 255 if self._drag == "paint" else 0)
        self.request_render()

    def _motion(self, e) -> None:
        if self.proc is None:
            return
        self._cursor(e)
        if self._drag == "pan":
            self.ox -= (e.x - self._last[0]) / self.zoom
            self.oy -= (e.y - self._last[1]) / self.zoom
        elif self._drag in ("paint", "erase"):
            self._stamp(self._to_img(*self._last), self._to_img(e.x, e.y), 255 if self._drag == "paint" else 0)
        else:
            return
        self._last = (e.x, e.y)
        self.request_render()

    def _release(self, e) -> None:
        if self._drag in ("paint", "erase"):
            self._update_stats()
        self._drag = None

    def _wheel(self, e, direction: int | None = None) -> None:
        if self.proc is None:
            return
        d = direction if direction is not None else (1 if e.delta > 0 else -1)
        bx, by = self._to_img(e.x, e.y)
        self.zoom = min(48.0, max(0.05, self.zoom * (1.18 ** d)))
        self.ox, self.oy = bx - e.x / self.zoom, by - e.y / self.zoom
        self._cursor(e)
        self.request_render()


def main() -> None:
    ctk.set_appearance_mode("dark")
    App().mainloop()


if __name__ == "__main__":
    main()
