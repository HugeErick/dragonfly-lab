from __future__ import annotations

import math
import queue
import threading
import tkinter as tk
from dataclasses import replace
from pathlib import Path
from tkinter import colorchooser, filedialog, messagebox, simpledialog

import customtkinter as ctk
import cv2
import numpy as np
from PIL import Image, ImageTk

from . import benchmark, core

# ---- palette (dark) -------------------------------------------------------
BG, PANEL, CARD, BORDER = "#0b0d12", "#12151c", "#181c25", "#262b38"
TEXT, MUTED = "#d7dbe6", "#7d8499"
ACCENT, ACCENT_H = "#6d5efc", "#8478ff"
DANGER, DANGER_H = "#3a1f27", "#55283a"

# overlay colours (BGR for OpenCV). Region colours come from their class.
C_DUST, C_DUST_DONE = (36, 191, 251), (128, 222, 74)
C_MASK = (94, 63, 244)
C_ZONE = (120, 120, 120)
# dust-assist suggestions are drawn as SQUARES: candidate / needs review / also found by our own detector
C_SUG, C_SUG_REVIEW, C_SUG_MATCH = (255, 170, 60), (200, 110, 255), (60, 235, 255)

NODE_HIT, EDGE_HIT = 10, 8          # click tolerance in screen px

TOOLS = [("Brush", "Brush"), ("Pick", "Pick"), ("Nodes", "Nodes"), ("Draw", "Draw"),
         ("Dust", "Dust −"), ("AddDust", "Dust ＋"), ("TagDust", "Tag dust"), ("Assist", "Assist")]
TOOL_HELP = {
    "Brush": "L-drag paint mask · R-drag erase",
    "Pick": "Click a region / speck to toggle its mask (R-click = unmask)",
    "Dust": "Click a dust circle to delete it (false positive)",
    "AddDust": "Click a speck the detector missed to add it · it sits on whatever class is under it",
    "TagDust": "Click a dust circle to attach it to the ACTIVE class · R-click = back to automatic",
    "Nodes": "Click a region to select it · drag a node to reshape · click an edge to add a node · "
             "R-click a node to delete it · drag inside the selected region to move it · Del removes it",
    "Draw": "Click to place points · click the first point, double-click or Enter to close · "
            "R-click undoes a point · Esc cancels",
    "Assist": "Squares = helper suggestions (blue: candidate · pink: needs review · yellow: your detector "
              "found it too) · click = accept as dust · R-click = reject",
}


def _hex(rgb) -> str:
    return "#%02x%02x%02x" % tuple(int(v) for v in rgb)


def _bgr(rgb) -> tuple[int, int, int]:
    return int(rgb[2]), int(rgb[1]), int(rgb[0])


class _Slider(ctk.CTkSlider):
    """CTkSlider that ignores the mouse wheel, so scrolling the side panel never changes a value."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        for seq in ("<MouseWheel>", "<Button-4>", "<Button-5>"):
            try:
                tk.Canvas.unbind(self._canvas, seq)
            except Exception:  # noqa: BLE001 - cosmetic safeguard
                pass

    def _mouse_scroll_event(self, event=None):          # older/newer customtkinter call this by name
        return None


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
        self.specks: list[core.Speck] = []
        self.sp_x = self.sp_y = self.sp_r = np.zeros(0)
        self.out_dir: Path | None = None            # src/<image id>/ of the open image
        self._dust_rev = 0                          # bumps whenever the dust list changes
        self._zone_cls_cache: tuple | None = None   # (key, class id per speck)

        # optional dust-assist layer: suggestions only, never ground truth until a person accepts them
        self.suggestions: list[core.Suggestion] = []
        self.sg_x = self.sg_y = self.sg_r = np.zeros(0)
        self.sg_pending = np.zeros(0, bool)
        self.sg_match = np.zeros(0, int)            # index of the speck that already covers it (-1 = none)
        self._recurrent: tuple | None = None        # (proxy shape, (N, 2) sensor-dust positions, proxy px)

        # segmentation state: user-defined classes + editable polygons
        self.classes: list[core.ClassDef] = core.default_classes()
        self.regions: list[core.Region] = []
        self.active_cls = core.BODY_ID
        self.sel: core.Region | None = None
        self.draft: list[tuple[float, float]] = []      # polygon being drawn (image coords)
        self.undo_stack: list[tuple] = []
        self._rev = 0                                   # bumps whenever geometry changes
        self._zone_cache: tuple | None = None
        self._pre: tuple | None = None                  # undo snapshot waiting for the first drag motion
        self._node = 0
        self._cls_btns: dict[int, ctk.CTkButton] = {}
        self._cls_vars: list[ctk.BooleanVar] = []

        # view state
        self.zoom, self.ox, self.oy = 1.0, 0.0, 0.0
        self._drag: str | None = None
        self._last = (0, 0)
        self._mouse = (0, 0)
        self._photo = None
        self._render_pending = False

        # background work
        self.q: queue.Queue = queue.Queue()
        self.busy = False

        self._build()
        self.bind_all("<Control-z>", self._undo_key)
        self.bind_all("<Command-z>", self._undo_key)
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
        ctk.CTkLabel(side, text="Ground-truth annotation assistant", text_color=MUTED,
                     anchor="w").pack(fill="x", pady=(0, 10))

        self._btn(side, "Open image…", self.open_image, primary=True)

        # ---- detect
        self._section(side, "DETECT")
        self.sens_bw = self._slider(side, "Body / wing sensitivity", 0.5, 2.0, 1.0, 30)
        self.node_eps = self._slider(side, "Node spacing (simplify px)", 0.5, 15, 3.0, 29)
        self._btn(side, "Detect body & wings", self.detect_body_wings)
        self.sens_dust = self._slider(side, "Dust sensitivity", 1, 10, 5, 9, "{:.0f}")
        self.max_speck = self._slider(side, "Max speck size (px)", 6, 60, 20, 54, "{:.0f}")
        self.avoid_fly = ctk.BooleanVar(value=True)
        ctk.CTkSwitch(side, text="Keep dust away from guarded classes", variable=self.avoid_fly,
                      command=self.request_render, text_color=TEXT, progress_color=ACCENT,
                      button_color=TEXT, fg_color=BORDER).pack(anchor="w", pady=(4, 2))
        self.keep_away = self._slider(side, "Keep-away margin (px)", 0, 150, 40, 150, "{:.0f}")
        self._btn(side, "Detect dust specks", self.detect_dust)
        self._build_assist(side)

        # ---- classes
        self._section(side, "CLASSES")
        self.cls_frame = ctk.CTkFrame(side, fg_color=CARD, corner_radius=6)
        self.cls_frame.pack(fill="x")
        row = ctk.CTkFrame(side, fg_color="transparent")
        row.pack(fill="x", pady=(4, 0))
        self._btn(row, "＋ New", self.new_class, side="left")
        self._btn(row, "Rename", self.rename_class, side="left")
        self._btn(row, "Delete", self.delete_class, side="left", danger=True)
        ctk.CTkLabel(side, text="swatch: colour · name: make active (new regions use it)\n"
                                "show: visible · guard: dust detection keeps away",
                     text_color=MUTED, justify="left", anchor="w", wraplength=260).pack(fill="x", pady=(4, 0))

        # ---- tools
        self._section(side, "TOOL")
        self.tool = ctk.StringVar(value="Brush")
        self._tool_btns: dict[str, ctk.CTkButton] = {}
        for chunk in (TOOLS[:4], TOOLS[4:]):
            row = ctk.CTkFrame(side, fg_color="transparent")
            row.pack(fill="x", pady=1)
            for key, label in chunk:
                b = ctk.CTkButton(row, text=label, height=30, border_width=1, border_color=BORDER,
                                  fg_color=CARD, hover_color=BORDER, text_color=TEXT,
                                  font=ctk.CTkFont(size=12), command=lambda k=key: self._select_tool(k))
                b.pack(side="left", expand=True, fill="x", padx=2)
                self._tool_btns[key] = b
        self._style_tools()
        self.tool_help = ctk.CTkLabel(side, text=TOOL_HELP["Brush"], text_color=MUTED, justify="left",
                                      anchor="w", wraplength=260)
        self.tool_help.pack(fill="x")
        self.brush = self._slider(side, "Brush size (px)", 2, 150, 14, 148, "{:.0f}")
        self.add_takes_active = ctk.BooleanVar(value=False)
        ctk.CTkSwitch(side, text="New dust takes the active class", variable=self.add_takes_active,
                      text_color=TEXT, progress_color=ACCENT, button_color=TEXT,
                      fg_color=BORDER).pack(anchor="w", pady=(4, 2))

        row = ctk.CTkFrame(side, fg_color="transparent")
        row.pack(fill="x", pady=(6, 0))
        self._btn(row, "Subdivide", self.subdivide_selected, side="left")
        self._btn(row, "Simplify", self.simplify_selected, side="left")
        row = ctk.CTkFrame(side, fg_color="transparent")
        row.pack(fill="x", pady=(2, 0))
        self._btn(row, "→ Active class", self.assign_selected, side="left")
        self._btn(row, "Delete region", self.delete_selected, side="left", danger=True)

        # ---- mask
        self._section(side, "MASK")
        self._btn(side, "Mask all dust", self.mask_dust)
        self._btn(side, "Mask active class", self.mask_active_class)
        self._btn(side, "Mask selected region", self.mask_selected)
        row = ctk.CTkFrame(side, fg_color="transparent")
        row.pack(fill="x", pady=(4, 0))
        self._btn(row, "Undo", self.undo, side="left")
        self._btn(row, "Clear mask", self.clear_mask, side="left", danger=True)

        # ---- view
        self._section(side, "VIEW")
        self.v_dust, self.v_mask = ctk.BooleanVar(value=True), ctk.BooleanVar(value=True)
        self.v_zone, self.v_sug = ctk.BooleanVar(value=False), ctk.BooleanVar(value=True)
        views = [("Dust specks", self.v_dust), ("Mask", self.v_mask), ("Keep-away zone", self.v_zone)]
        if core.ASSIST_AVAILABLE:
            views.append(("Assist suggestions", self.v_sug))
        for text, var in views:
            ctk.CTkSwitch(side, text=text, variable=var, command=self.request_render, text_color=TEXT,
                          progress_color=ACCENT, button_color=TEXT, fg_color=BORDER).pack(anchor="w", pady=2)
        self._btn(side, "Fit to window", self.fit_and_render)

        # ---- export
        self._section(side, "EXPORT")
        self._btn(side, "Output annotation", self.export, primary=True)
        self._btn(side, "Build benchmark package", self.build_package)
        ctk.CTkLabel(side, text="Package = last exported annotation of every image → benchmark/ "
                                "(20 development / 10 test split, metadata, README).",
                     text_color=MUTED, justify="left", anchor="w", wraplength=260).pack(fill="x")
        self.stats = ctk.CTkLabel(side, text="", text_color=MUTED, justify="left", anchor="w", wraplength=260)
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
        c.bind("<Double-Button-1>", lambda e: self._double(e, 1))
        c.bind("<Double-Button-3>", lambda e: self._double(e, 3))
        c.bind("<Double-Button-2>", lambda e: self._double(e, 2))
        for seq in ("<B1-Motion>", "<B2-Motion>", "<B3-Motion>"):
            c.bind(seq, self._motion)
        c.bind("<Motion>", self._cursor)
        c.bind("<Leave>", lambda e: c.delete("brush"))
        c.bind("<Enter>", lambda e: c.focus_set())
        for seq in ("<ButtonRelease-1>", "<ButtonRelease-2>", "<ButtonRelease-3>"):
            c.bind(seq, self._release)
        c.bind("<MouseWheel>", self._wheel)
        c.bind("<Button-4>", lambda e: self._wheel(e, 1))
        c.bind("<Button-5>", lambda e: self._wheel(e, -1))
        c.bind("<Delete>", lambda e: self.delete_selected())
        c.bind("<Return>", lambda e: self._finish_draft())
        c.bind("<Escape>", self._escape)

        # status bar
        bar = ctk.CTkFrame(self, fg_color=PANEL, corner_radius=0, height=32)
        bar.grid(row=1, column=0, columnspan=2, sticky="ew")
        self.status_lbl = ctk.CTkLabel(bar, text="Open an image to begin.", text_color=MUTED, anchor="w")
        self.status_lbl.pack(side="left", padx=12, pady=4)
        self.progress = ctk.CTkProgressBar(bar, mode="indeterminate", width=180, height=6,
                                           progress_color=ACCENT, fg_color=BORDER)
        self.progress.pack(side="right", padx=12)
        self.progress.set(0)

        self._rebuild_classes()

    def _build_assist(self, side) -> None:
        """Optional second opinion from dust_assist.py (detection only, never ground truth)."""
        self.as_save = ctk.BooleanVar(value=False)
        self._section(side, "ASSIST · 2nd opinion (optional)")
        if not core.ASSIST_AVAILABLE:
            why = f" ({core.ASSIST_ERROR})" if core.ASSIST_ERROR else ""
            ctk.CTkLabel(side, text=f"dust_assist.py is not available{why} – helper disabled.",
                         text_color=MUTED, justify="left", anchor="w", wraplength=260).pack(fill="x")
            return
        ctk.CTkLabel(side, text="Proposals from a second detector. They are NOT ground truth: review each "
                                "one with the Assist tool (click = accept, R-click = reject).",
                     text_color=MUTED, justify="left", anchor="w", wraplength=260).pack(fill="x")
        self.as_sens = self._slider(side, "Assist sensitivity", 1, 10, 5, 9, "{:.0f}")
        self.as_paper = ctk.BooleanVar(value=True)
        self.as_rec = ctk.BooleanVar(value=True)
        self.as_full = ctk.BooleanVar(value=False)
        for text, var in (("Also propose dust on paper / label", self.as_paper),
                          ("Use recurrent-dust session (if loaded)", self.as_rec),
                          ("Full resolution (slow, lots of RAM)", self.as_full),
                          ("Save suggestions file on export (not GT)", self.as_save)):
            ctk.CTkSwitch(side, text=text, variable=var, text_color=TEXT, progress_color=ACCENT,
                          button_color=TEXT, fg_color=BORDER).pack(anchor="w", pady=2)
        self._btn(side, "Suggest dust (helper)", self.assist_detect)
        row = ctk.CTkFrame(side, fg_color="transparent")
        row.pack(fill="x", pady=(4, 0))
        self._btn(row, "Accept candidates", self.assist_accept_candidates, side="left")
        self._btn(row, "Clear", self.assist_clear, side="left", danger=True)
        row = ctk.CTkFrame(side, fg_color="transparent")
        row.pack(fill="x", pady=(2, 0))
        self._btn(row, "Propose paper/info", self.assist_paper_region, side="left")
        self._btn(row, "Session…", self.assist_load_session, side="left")
        ctk.CTkLabel(side, text="Accept candidates skips the ones marked “review”. Session… = pick ≥ 3 photos "
                                "of the same shoot (same size) to flag sensor dust that repeats.",
                     text_color=MUTED, justify="left", anchor="w", wraplength=260).pack(fill="x", pady=(2, 0))

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
        _Slider(parent, from_=lo, to=hi, number_of_steps=steps, variable=var, fg_color=BORDER,
                      progress_color=ACCENT, button_color=ACCENT, button_hover_color=ACCENT_H,
                      command=lambda v: (val.configure(text=fmt.format(v)), self._cursor(None),
                                         self.request_render())
                      ).pack(fill="x", pady=(2, 4))
        return var

    def status(self, text: str) -> None:
        self.status_lbl.configure(text=text)

    def _select_tool(self, key: str) -> None:
        self.tool.set(key)
        self._style_tools()
        self._on_tool(key)

    def _style_tools(self) -> None:
        for key, b in self._tool_btns.items():
            b.configure(fg_color=ACCENT if key == self.tool.get() else CARD)

    def _on_tool(self, v: str) -> None:
        self.tool_help.configure(text=TOOL_HELP.get(v, ""))
        if v != "Draw":
            self.draft = []
        self._cursor(None)
        self.request_render()

    # ------------------------------------------------------- class manager
    def _rebuild_classes(self) -> None:
        f = self.cls_frame
        for w in f.winfo_children():
            w.destroy()
        self._cls_btns, self._cls_vars = {}, []
        f.grid_columnconfigure(1, weight=1)
        small = ctk.CTkFont(size=10)
        ctk.CTkLabel(f, text="show", font=small, text_color=MUTED).grid(row=0, column=2, padx=2)
        ctk.CTkLabel(f, text="guard", font=small, text_color=MUTED).grid(row=0, column=3, padx=(2, 6))
        for r, c in enumerate(self.classes, 1):
            hx = _hex(c.color)
            ctk.CTkButton(f, text="", width=22, height=22, corner_radius=4, fg_color=hx, hover_color=hx,
                          command=lambda c=c: self.pick_color(c)).grid(row=r, column=0, padx=(6, 4), pady=2)
            b = ctk.CTkButton(f, text=c.name, anchor="w", height=26, fg_color="transparent",
                              hover_color=BORDER, text_color=TEXT,
                              command=lambda c=c: self.set_active(c.id))
            b.grid(row=r, column=1, sticky="ew", pady=2)
            self._cls_btns[c.id] = b
            for col, attr, pad in ((2, "visible", (2, 2)), (3, "guard", (2, 6))):
                var = ctk.BooleanVar(value=getattr(c, attr))
                self._cls_vars.append(var)
                ctk.CTkCheckBox(f, text="", variable=var, width=24, checkbox_width=18, checkbox_height=18,
                                fg_color=ACCENT, hover_color=ACCENT_H, border_color=MUTED,
                                command=lambda c=c, a=attr, v=var: self._set_flag(c, a, v.get())
                                ).grid(row=r, column=col, padx=pad)
        self._style_active()

    def _style_active(self) -> None:
        for cid, b in self._cls_btns.items():
            b.configure(fg_color=ACCENT if cid == self.active_cls else "transparent")

    def _cls(self, cid: int) -> core.ClassDef | None:
        return next((c for c in self.classes if c.id == cid), None)

    def set_active(self, cid: int) -> None:
        self.active_cls = cid
        self._style_active()
        c = self._cls(cid)
        if c:
            self.status(f"Active class: {c.name}")

    def _set_flag(self, c: core.ClassDef, attr: str, value: bool) -> None:
        setattr(c, attr, bool(value))
        if attr == "visible" and not value and self.sel is not None and self.sel.cls == c.id:
            self.sel = None
        self.request_render()

    def pick_color(self, c: core.ClassDef) -> None:
        rgb, _ = colorchooser.askcolor(color=_hex(c.color), parent=self, title=f"Colour for “{c.name}”")
        if rgb:
            c.color = tuple(int(v) for v in rgb)
            self._rebuild_classes()
            self.request_render()

    def _unique_name(self, name: str, ignore: int | None = None) -> str:
        used = {c.name.lower() for c in self.classes if c.id != ignore}
        out, k = name, 2
        while out.lower() in used:
            out, k = f"{name} {k}", k + 1
        return out

    def new_class(self) -> None:
        name = simpledialog.askstring("New class", "Class name:", parent=self)
        if not name or not name.strip():
            return
        cid = max((c.id for c in self.classes), default=0) + 1
        n_custom = sum(1 for k in self.classes if k.id not in core.BUILTIN_IDS)
        color = core.PALETTE[n_custom % len(core.PALETTE)]
        self.classes.append(core.ClassDef(cid, self._unique_name(name.strip()), color))
        self.active_cls = cid
        self._rebuild_classes()
        self.status(f"Class “{self.classes[-1].name}” created and active. Use Draw, or select a region "
                    "and press “→ Active class”.")

    def rename_class(self) -> None:
        c = self._cls(self.active_cls)
        if c is None:
            return
        name = simpledialog.askstring("Rename class", "New name:", initialvalue=c.name, parent=self)
        if name and name.strip():
            c.name = self._unique_name(name.strip(), ignore=c.id)
            self._rebuild_classes()
            self.request_render()

    def delete_class(self) -> None:
        c = self._cls(self.active_cls)
        if c is None:
            return
        if c.id in core.BUILTIN_IDS:
            self.status("The built-in specimen / wings / paper_info classes can't be deleted "
                        "(rename them instead).")
            return
        n = sum(r.cls == c.id for r in self.regions)
        self._push_undo()
        self.regions = [r for r in self.regions if r.cls != c.id]
        self.classes = [k for k in self.classes if k.id != c.id]
        self._set_specks([replace(sp, cls=None) if sp.cls == c.id else sp for sp in self.specks])
        self.active_cls = core.BODY_ID
        self.sel = None
        self._rebuild_classes()
        self._regions_changed()
        self.status(f"Class “{c.name}” deleted with {n} region(s) (Ctrl+Z to undo).")

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
            out = core.resolve_output_dir(p)
            saved = core.load_mask(out, proc.shape[:2], core.clean_stem(p))
            proj = core.load_project(out, proc.shape[:2])
            specks = core.load_dust(out, scale, {c.id for c in proj[0]} if proj else None)
            return full, proc, scale, saved, specks, proj, out

        def done(res):
            self.src = p
            self.full, self.proc, self.scale, saved, specks, proj, self.out_dir = res
            self.mask = saved if saved is not None else np.zeros(self.proc.shape[:2], np.uint8)
            self.undo_stack, self.sel, self.draft = [], None, []
            if proj is not None:                       # this image has a saved project -> use its classes
                self.classes, self.regions = proj
            else:                                      # keep the classes you already made, drop old polygons
                self.regions = []
            core.ensure_defaults(self.classes)
            if self._cls(self.active_cls) is None:
                self.active_cls = core.BODY_ID
            self._rebuild_classes()
            self._set_suggestions([])
            self._set_specks(specks)
            self._rev += 1
            self.fit_and_render()
            h, w = self.full.shape[:2]
            note = "  ·  restored previous annotation" if (saved is not None or specks or proj) else ""
            ident = "" if self.out_dir.name == core.clean_stem(p) else \
                f"  ·  ⚠ same file name as another photo → saved as ID “{self.out_dir.name}”"
            self.status(f"{p.name}  ·  {w}×{h}px{note}{ident}")
            self._update_stats()

        self._run("Loading image…", work, done)

    def detect_body_wings(self) -> None:
        if not self._need_image():
            return
        proc, sens, eps = self.proc, self.sens_bw.get(), float(self.node_eps.get())

        def done(det):
            self._push_undo()
            self._apply_detection(det, eps)
            body = sum(r.cls == core.BODY_ID for r in self.regions)
            wings = sum(r.cls == core.WING_ID for r in self.regions)
            self.status(f"Body {'found' if body else 'NOT found'} · {wings} wing region(s). "
                        "Refine with the Nodes tool, or adjust sensitivity and re-run "
                        "(re-detect only replaces regions you haven't edited).")

        self._run("Detecting body and wings…", lambda: core.detect_body_and_wings(proc, sens), done)

    def _apply_detection(self, det: core.BodyWings, eps: float) -> None:
        # only the auto body / wing regions are replaced (an auto paper/info proposal stays until re-proposed)
        self.regions = [r for r in self.regions if not (r.auto and r.cls in (core.BODY_ID, core.WING_ID))] \
            + core.regions_from_detection(det, eps)
        if core.ensure_defaults(self.classes):
            self._rebuild_classes()
        self.sel = None
        self._regions_changed()

    def _guard_ids(self) -> set[int]:
        return {c.id for c in self.classes if c.guard}

    def detect_dust(self) -> None:
        if not self._need_image():
            return
        proc = self.proc
        s, md, margin = self.sens_dust.get(), int(self.max_speck.get()), int(self.keep_away.get())
        avoid, sens_bw, eps = self.avoid_fly.get(), self.sens_bw.get(), float(self.node_eps.get())
        fly = core.rasterize(self.regions, proc.shape[:2], self._guard_ids()) if avoid else None
        need_det = avoid and not fly.any()             # nothing to keep away from yet -> auto-detect first

        def work():
            det, f = None, fly
            if need_det:
                det = core.detect_body_and_wings(proc, sens_bw)
                f = core.fly_from_detection(det)
            zone = core.exclusion_zone(f, margin, proc.shape[:2]) if avoid else None
            return det, core.detect_dust(proc, zone, s, md)

        def done(res):
            det, specks = res
            self._push_undo()
            if det is not None:
                self._apply_detection(det, eps)
            kept = [sp for sp in self.specks if sp.manual]
            self._set_specks(list(specks) + kept)
            warn = f"  ({len(kept)} hand-added kept)" if kept else ""
            if avoid and not self._guard_ids():
                warn += "  (no class has “guard” enabled – nothing excluded)"
            elif avoid and not any(r.cls in self._guard_ids() for r in self.regions):
                warn += "  (no guarded regions found – tune body/wing sensitivity or draw them)"
            self.status(f"{len(specks)} dust speck(s) found.{warn} Review them: Dust − removes false ones, "
                        "Dust ＋ adds missed ones.")
            self._update_stats()
            self.request_render()

        self._run("Hunting dust specks…", work, done)

    def _set_specks(self, specks) -> None:
        self.specks = core.ensure_uids(list(specks))
        self._dust_rev += 1
        self.sp_x = np.array([s.cx for s in self.specks])
        self.sp_y = np.array([s.cy for s in self.specks])
        self.sp_r = np.array([s.r for s in self.specks])
        self._refresh_matches()

    # ------------------------------------------------ assist (optional suggestion layer)
    def _assist_ok(self) -> bool:
        if not core.ASSIST_AVAILABLE:
            self.status("The dust-assist helper (dust_assist.py) is not available.")
            return False
        return True

    def _set_suggestions(self, sugs) -> None:
        self.suggestions = list(sugs)
        self.sg_x = np.array([s.cx for s in self.suggestions])
        self.sg_y = np.array([s.cy for s in self.suggestions])
        self.sg_r = np.array([s.r for s in self.suggestions])
        self.sg_pending = np.array([s.decision == "pending" for s in self.suggestions], bool)
        self._refresh_matches()

    def _refresh_matches(self) -> None:
        """Pending suggestions that our own detector already covers (both found the same speck)."""
        match = np.full(len(self.suggestions), -1, int)
        if len(self.specks) and self.sg_pending.any():
            for i in np.flatnonzero(self.sg_pending):
                d = np.hypot(self.sp_x - self.sg_x[i], self.sp_y - self.sg_y[i])
                j = int(d.argmin())
                if d[j] <= max(self.sp_r[j], self.sg_r[i]) + 3.0:
                    match[i] = j
        self.sg_match = match

    def _recurrent_for(self, shape, scale: float, hi_res: bool):
        if not self.as_rec.get() or self._recurrent is None:
            return None
        sh, pts = self._recurrent
        if sh != shape or not len(pts):
            return None
        return pts / scale if hi_res else pts

    def assist_detect(self) -> None:
        if not self._need_image() or not self._assist_ok():
            return
        proc, full, scale = self.proc, self.full, self.scale
        shape = proc.shape[:2]
        # the helper protects the HUMAN-corrected specimen (every guarded class except paper/info) and
        # takes the paper/info regions as its document mask; with none drawn yet it guesses on its own
        guarded = {c.id for c in self.classes if c.guard and c.id != core.INFO_ID}
        spec = core.rasterize(self.regions, shape, guarded)
        doc = core.rasterize(self.regions, shape, {core.INFO_ID})
        use_spec, use_doc = bool(spec.any()), bool(doc.any())
        sens, paper = float(self.as_sens.get()), bool(self.as_paper.get())
        hi = bool(self.as_full.get()) and scale < 1.0
        rec = self._recurrent_for(shape, scale, hi)
        old_rejected = [s for s in self.suggestions if s.decision == "rejected"]

        def work():
            if hi:
                H, W = full.shape[:2]
                img, to_proc = full, scale
                sp = cv2.resize(spec, (W, H), interpolation=cv2.INTER_NEAREST) if use_spec else None
                dc = cv2.resize(doc, (W, H), interpolation=cv2.INTER_NEAREST) if use_doc else None
            else:
                img, to_proc = proc, 1.0
                sp, dc = (spec if use_spec else None), (doc if use_doc else None)
            return core.assist_suggestions(img, sp, dc, rec, paper, sens, to_proc)

        def done(res):
            sugs, det = res
            if old_rejected:                    # what you already rejected stays rejected on a re-run
                ox = np.array([s.cx for s in old_rejected])
                oy = np.array([s.cy for s in old_rejected])
                orr = np.array([s.r for s in old_rejected])
                for n, s in enumerate(sugs):
                    d = np.hypot(ox - s.cx, oy - s.cy)
                    j = int(d.argmin())
                    if d[j] <= max(orr[j], s.r) + 3.0:
                        sugs[n] = replace(s, decision="rejected")
            self._push_undo()
            self._set_suggestions(sugs)
            pend = [s for s in sugs if s.decision == "pending"]
            n_rev = sum(s.status == "review" for s in pend)
            n_match = int((self.sg_match >= 0).sum())
            base = "your body / wing regions" if use_spec else "its own automatic protection (no body / wing regions yet)"
            extra = f" · {len(rec)} recurrent position(s) used" if rec is not None else ""
            self.status(f"{len(pend)} suggestion(s): {len(pend) - n_rev} candidate, {n_rev} to review, "
                        f"{n_match} also found by your detector · protected via {base}{extra}. "
                        "Assist tool: click accepts, R-click rejects.")
            self._update_stats()
            self.request_render()

        self._run("Asking the dust-assist helper…", work, done)

    def _nearest_suggestion(self, e) -> int | None:
        if not self.sg_pending.any():
            return None
        x, y = self._to_img(e.x, e.y)
        d = np.hypot(self.sg_x - x, self.sg_y - y) * self.zoom
        d[~self.sg_pending] = np.inf
        i = int(d.argmin())
        return i if d[i] <= max(10.0, self.sg_r[i] * self.zoom + 8) else None

    def _assist_press(self, e, button: int) -> None:
        if not self._assist_ok():
            return
        i = self._nearest_suggestion(e)
        if i is None:
            return
        if button == 1:
            self._accept_suggestions([i])
        else:
            self._reject_suggestion(i)

    def _accept_suggestions(self, idx: list[int]) -> None:
        """A person confirms the suggestion as dust: it becomes an ordinary hand-curated speck
        (origin "manual", suggested_by "dust_assist"). If our own detector already has that speck, no
        duplicate is added."""
        sugs, new, dup = list(self.suggestions), [], 0
        for i in idx:
            s = sugs[i]
            if self.sg_match[i] >= 0:
                dup += 1
            else:
                sp = core.speck_from_suggestion(s)
                if self.add_takes_active.get():
                    sp = replace(sp, cls=self.active_cls)
                new.append(sp)
            sugs[i] = replace(s, decision="accepted")
        self._push_undo()
        self._set_suggestions(sugs)
        if new:
            self._set_specks(self.specks + new)
        note = f" · {dup} already in your list" if dup else ""
        self.status(f"Accepted {len(new)} new dust speck(s){note} · {len(self.specks)} total (Ctrl+Z to undo).")
        self._update_stats()
        self.request_render()

    def _reject_suggestion(self, i: int) -> None:
        self._push_undo()
        sugs = list(self.suggestions)
        sugs[i] = replace(sugs[i], decision="rejected")
        self._set_suggestions(sugs)
        self.status(f"Suggestion rejected · {int(self.sg_pending.sum())} pending (Ctrl+Z to undo).")
        self._update_stats()
        self.request_render()

    def assist_accept_candidates(self) -> None:
        if not self._need_image() or not self._assist_ok():
            return
        idx = [i for i, s in enumerate(self.suggestions) if s.decision == "pending" and s.status == "candidate"]
        if not idx:
            self.status("No pending “candidate” suggestions (the ones marked “review” are never bulk-accepted).")
            return
        if not messagebox.askyesno(
                "Accept candidates",
                f"Accept {len(idx)} pending “candidate” suggestion(s) as dust?\n\nSuggestions marked “review” "
                "(near the specimen, near the border, or too large) stay for you to check one by one.\n"
                "You can undo with Ctrl+Z.", parent=self):
            return
        self._accept_suggestions(idx)

    def assist_clear(self) -> None:
        if not self.suggestions:
            return
        self._push_undo()
        self._set_suggestions([])
        self.status("Suggestions cleared (Ctrl+Z to undo). Accepted specks stay in the dust list.")
        self._update_stats()
        self.request_render()

    def assist_paper_region(self) -> None:
        if not self._need_image() or not self._assist_ok():
            return
        proc, eps = self.proc, float(self.node_eps.get())

        def done(regs):
            if not regs:
                self.status("The helper found no clear paper / label region – draw it with the Draw tool.")
                return
            self._push_undo()
            self.regions = [r for r in self.regions if not (r.auto and r.cls == core.INFO_ID)] + regs
            if core.ensure_defaults(self.classes):
                self._rebuild_classes()
            self.sel = None
            self._regions_changed()
            self.status(f"{len(regs)} paper / info region(s) proposed – a starting point only: reshape them "
                        "with Nodes or delete them.")

        self._run("Looking for paper / label…", lambda: core.assist_document_regions(proc, eps), done)

    def assist_load_session(self) -> None:
        if not self._assist_ok():
            return
        paths = filedialog.askopenfilenames(
            title="Photos of the same session (same size, at least 3)",
            filetypes=[("Images", "*.jpg *.jpeg *.png *.tif *.tiff *.bmp *.webp"), ("All files", "*.*")])
        if not paths:
            return
        if len(paths) < 3:
            self.status("Pick at least 3 photos of the same session.")
            return

        def work():
            proxies, sizes = [], set()
            for p in paths:
                img = core.imread(p)
                sizes.add(img.shape[:2])
                proxies.append(core.make_proxy(img)[0])
            if len(sizes) != 1:
                raise ValueError("the session photos must all have the same dimensions")
            return proxies[0].shape[:2], core.assist_recurrent_positions(proxies)

        def done(res):
            shape, pts = res
            self._recurrent = (shape, pts)
            self.status(f"Recurrent-dust session: {len(pts)} sensor position(s) from {len(paths)} photos. Used by "
                        "“Suggest dust” on photos of the same size (switch it off under Assist).")

        self._run("Looking for dust that repeats across the session…", work, done)

    # ------------------------------------------------------ undo / snapshots
    def _snapshot(self) -> tuple:
        return (self.mask.copy(), list(self.specks),
                [core.Region(r.cls, r.pts.copy(), r.auto) for r in self.regions],
                [replace(c) for c in self.classes], list(self.suggestions))

    def _push_undo(self, snap: tuple | None = None) -> None:
        if self.mask is None:
            return
        self.undo_stack.append(snap if snap is not None else self._snapshot())
        del self.undo_stack[:-15]

    def _commit_pre(self) -> None:
        if self._pre is not None:
            self._push_undo(self._pre)
            self._pre = None

    def _undo_key(self, e) -> None:
        if e.widget.winfo_class() in ("Entry", "Text"):    # let text fields keep their own undo
            return
        self.undo()

    def undo(self) -> None:
        if self.undo_stack and self.mask is not None:
            self.mask, specks, self.regions, self.classes, sugs = self.undo_stack.pop()
            self.sel, self.draft = None, []
            if self._cls(self.active_cls) is None:
                self.active_cls = core.BODY_ID
            self._rebuild_classes()
            self._set_suggestions(sugs)
            self._set_specks(specks)
            self._regions_changed()

    def _regions_changed(self) -> None:
        self._rev += 1
        self._update_stats()
        self.request_render()

    # ------------------------------------------------------------- masking
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

    def _paint_region(self, r: core.Region, value: int) -> None:
        if len(r.pts) >= 3:
            cv2.fillPoly(self.mask, [np.round(r.pts).astype(np.int32)], value)

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

    def mask_active_class(self) -> None:
        if not self._need_image():
            return
        todo = [r for r in self.regions if r.cls == self.active_cls]
        if not todo:
            self.status("The active class has no regions yet.")
            return
        self._push_undo()
        for r in todo:
            self._paint_region(r, 255)
        self._update_stats()
        self.request_render()

    def mask_selected(self) -> None:
        if not self._need_image():
            return
        if self.sel is None:
            self.status("Select a region first (Nodes tool).")
            return
        self._push_undo()
        self._paint_region(self.sel, 255)
        self._update_stats()
        self.request_render()

    def _speck_class_ids(self) -> list:
        """Class id each speck sits on (None = background). Cached; stale while a polygon is dragged."""
        if not self.specks or self.proc is None:
            return []
        key = (self._rev, self._dust_rev)
        c = self._zone_cls_cache
        if c is not None and (c[0] == key or (self._drag in ("node", "move") and len(c[1]) == len(self.specks))):
            return c[1]
        ids = core.assign_zones(self.specks, self.regions, self.classes, self.proc.shape[:2])
        self._zone_cls_cache = (key, ids)
        return ids

    def _nearest_speck(self, e) -> int | None:
        if not len(self.specks):
            return None
        x, y = self._to_img(e.x, e.y)
        d = np.hypot(self.sp_x - x, self.sp_y - y) * self.zoom
        i = int(d.argmin())
        return i if d[i] <= max(10.0, self.sp_r[i] * self.zoom + 8) else None

    def _add_speck(self, e) -> None:
        x, y = self._to_img(e.x, e.y)
        H, W = self.mask.shape
        if not (0 <= x < W and 0 <= y < H):
            return
        if self._nearest_speck(e) is not None:
            self.status("There is already a dust circle there (Dust − removes it, Tag dust changes its class).")
            return
        s = core.speck_at(self.proc, x, y, int(self.max_speck.get()), max(3.0, self.max_speck.get() / 4))
        if self.add_takes_active.get():
            s = replace(s, cls=self.active_cls)
        self._push_undo()
        self._set_specks(self.specks + [s])
        zid = self._speck_class_ids()[-1]
        c = self._cls(zid) if zid is not None else None
        self.status(f"Dust added as {core.dust_id(self.specks[-1].uid)} on "
                    f"{c.name if c else 'background'} · {len(self.specks)} total (Ctrl+Z to undo).")
        self._update_stats()
        self.request_render()

    def _tag_speck(self, e, button: int) -> None:
        i = self._nearest_speck(e)
        if i is None:
            return
        c = self._cls(self.active_cls)
        if button == 1 and c is None:
            return
        s = self.specks[i]
        new = c.id if button == 1 else None
        if s.cls == new:
            return
        self._push_undo()
        specks = list(self.specks)
        specks[i] = replace(s, cls=new)
        self._set_specks(specks)
        self.status(f"{core.dust_id(s.uid)} → {c.name if new is not None else 'automatic (by position)'}")
        self._update_stats()
        self.request_render()

    def _remove_speck(self, e) -> None:
        if not len(self.specks):
            self.status("No dust circles to remove.")
            return
        x, y = self._to_img(e.x, e.y)
        d = np.hypot(self.sp_x - x, self.sp_y - y) * self.zoom          # distance in screen px
        i = int(d.argmin())
        if d[i] > max(10.0, self.sp_r[i] * self.zoom + 8):
            return
        s = self.specks[i]
        H, W = self.mask.shape
        self._push_undo()
        if self.mask[min(int(s.cy), H - 1), min(int(s.cx), W - 1)] > 0:
            self._paint_speck(s, 0)                                       # it was never dust: unmask it too
        self._set_specks(self.specks[:i] + self.specks[i + 1:])
        self.status(f"Removed 1 dust circle · {len(self.specks)} left (Ctrl+Z to undo).")
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
        r = self._region_at(x, y, edges=False)
        if r is None:
            return
        now = self.mask[yi, xi] > 0
        want = (not now) if force is None else force
        self._push_undo()
        self._paint_region(r, 255 if want else 0)
        self._update_stats()
        self.request_render()

    def _stamp(self, p0, p1, value: int) -> None:
        r = max(1, int(round(self.brush.get() / 2)))
        a = (int(p0[0]), int(p0[1]))
        b = (int(p1[0]), int(p1[1]))
        cv2.line(self.mask, a, b, value, 2 * r)
        cv2.circle(self.mask, b, r, value, -1)

    # ----------------------------------------------------- region editing
    def _region_at(self, x: float, y: float, edges: bool = True) -> core.Region | None:
        """Smallest visible region under the point (or whose outline is within click tolerance)."""
        vis = {c.id for c in self.classes if c.visible}
        tol = EDGE_HIT / self.zoom
        best, best_area = None, 0.0
        for r in self.regions:
            if r.cls not in vis or len(r.pts) < 3:
                continue
            if not core.point_in_poly(r.pts, x, y):
                if not (edges and core.nearest_edge(r.pts, x, y)[0] <= tol):
                    continue
            a = abs(float(cv2.contourArea(r.pts.astype(np.float32))))
            if best is None or a < best_area:
                best, best_area = r, a
        return best

    def _need_sel(self) -> bool:
        if self.sel is None:
            self.status("Select a region first (Nodes tool, click it).")
            return False
        return True

    def delete_selected(self) -> None:
        if self.sel is None or self.proc is None:
            return
        self._push_undo()
        self.regions = [r for r in self.regions if r is not self.sel]
        self.sel = None
        self._regions_changed()
        self.status("Region deleted (Ctrl+Z to undo).")

    def assign_selected(self) -> None:
        if not self._need_sel():
            return
        c = self._cls(self.active_cls)
        if c is None or self.sel.cls == c.id:
            return
        self._push_undo()
        self.sel.cls, self.sel.auto = c.id, False
        self._regions_changed()
        self.status(f"Region moved to class “{c.name}”.")

    def subdivide_selected(self) -> None:
        if not self._need_sel():
            return
        self._push_undo()
        self.sel.pts, self.sel.auto = core.subdivide(self.sel.pts), False
        self._regions_changed()
        self.status(f"{len(self.sel.pts)} nodes.")

    def simplify_selected(self) -> None:
        if not self._need_sel():
            return
        self._push_undo()
        self.sel.pts = core.simplify_poly(self.sel.pts, max(1.0, float(self.node_eps.get())))
        self.sel.auto = False
        self._regions_changed()
        self.status(f"{len(self.sel.pts)} nodes.")

    def _finish_draft(self) -> None:
        if not self.draft or self.proc is None:
            return
        c = self._cls(self.active_cls)
        if len(self.draft) < 3 or c is None:
            self.status("Need at least 3 points.")
            return
        self._push_undo()
        H, W = self.proc.shape[:2]
        pts = np.clip(np.array(self.draft, np.float32), 0, (W - 1, H - 1)).astype(np.float32)
        self.regions.append(core.Region(c.id, pts, auto=False))
        self.sel, self.draft = self.regions[-1], []
        self._regions_changed()
        self.status(f"{c.name} region added ({len(pts)} nodes). Switch to Nodes to fine-tune it.")

    def _escape(self, e=None) -> None:
        self.draft, self.sel = [], None
        self.request_render()

    # -------------------------------------------------------------- saving
    def export(self) -> None:
        if not self._need_image():
            return
        src, full, mask = self.src, self.full, self.mask.copy()
        classes = [replace(c) for c in self.classes]
        regions = [core.Region(r.cls, r.pts.copy(), r.auto) for r in self.regions]
        specks, scale, shape = list(self.specks), self.scale, self.proc.shape[:2]
        out_dir = self.out_dir
        sugs = (list(self.suggestions) or None) if (core.ASSIST_AVAILABLE and self.as_save.get()) else None

        def work():
            out = core.save_annotation(src, out_dir, full, mask, classes, regions, specks, scale, shape)
            if sugs is not None:                    # provenance only - not part of the GT package
                core.write_suggestions(out, src, full.shape[:2], sugs, scale)
            return out

        def done(out):
            extra = f"  (+ {core.SUGGESTIONS_JSON}, not GT)" if sugs is not None else ""
            self.status(f"Annotation saved → {out}{extra}")

        self._run("Writing full-resolution annotation…", work, done)

    def build_package(self) -> None:
        def done(res):
            self.status(f"Benchmark: {res.n_dev} development / {res.n_test} test → {res.root}")
            messagebox.showinfo("Benchmark package", res.summary(), parent=self)

        self._run("Building benchmark package…", benchmark.build_package, done)

    def _update_stats(self) -> None:
        if self.mask is None:
            return
        counts: dict[int, int] = {}
        for r in self.regions:
            counts[r.cls] = counts.get(r.cls, 0) + 1
        parts = "   ".join(f"{c.name}: {counts.get(c.id, 0)}" for c in self.classes)
        pct = 100.0 * float((self.mask > 0).mean())
        by_id = {c.id: c for c in self.classes}
        zones = self._speck_class_ids()
        actionable = sum(1 for z in zones if z is None or not by_id[z].guard) if zones else 0
        pend = int(self.sg_pending.sum()) if len(self.sg_pending) else 0
        sug = f"\nAssist suggestions pending: {pend}" if pend else ""
        self.stats.configure(text=f"{parts}\nDust: {len(self.specks)} ({actionable} actionable)   "
                                  f"Masked: {pct:.2f}% of image{sug}")

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

    def _get_zone(self, H: int, W: int) -> np.ndarray:
        """Keep-away zone from the guarded classes (cached; stale while a node is being dragged)."""
        guard = self._guard_ids()
        key = (self._rev, int(self.keep_away.get()), tuple(sorted(guard)))
        if self._zone_cache is not None:
            if self._zone_cache[0] == key or self._drag in ("node", "move"):
                return self._zone_cache[1]
        fly = core.rasterize(self.regions, (H, W), guard)
        zone = core.exclusion_zone(fly, int(self.keep_away.get()), (H, W))
        self._zone_cache = (key, zone)
        return zone

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
        off_x, off_y = round((ix0 - self.ox) * z), round((iy0 - self.oy) * z)   # view origin on canvas

        def tint(m: np.ndarray, color, alpha: float) -> None:
            mm = cv2.resize(m[iy0:iy1, ix0:ix1], (dw, dh), interpolation=cv2.INTER_NEAREST) > 0
            if mm.any():
                view[mm] = (view[mm] * (1 - alpha) + np.array(color, np.float32) * alpha).astype(np.uint8)

        def tf(p: np.ndarray) -> np.ndarray:
            return ((p.reshape(-1, 2).astype(np.float32) - (ix0, iy0)) * z).astype(np.int32)

        cm = {c.id: c for c in self.classes}

        if self.v_zone.get():
            tint(self._get_zone(H, W), C_ZONE, 0.25)

        # class-coloured fills
        fills = [r for r in self.regions if r.cls in cm and cm[r.cls].visible and len(r.pts) >= 3]
        if fills:
            ov = view.copy()
            for r in fills:
                cv2.fillPoly(ov, [tf(r.pts)], _bgr(cm[r.cls].color))
            view[:] = cv2.addWeighted(ov, 0.35, view, 0.65, 0)

        if self.v_mask.get():
            tint(self.mask, C_MASK, 0.55)

        # outlines + labels
        seen: dict[int, int] = {}
        for r in self.regions:
            seen[r.cls] = seen.get(r.cls, 0) + 1
            c = cm.get(r.cls)
            if c is None or not c.visible or len(r.pts) < 3:
                continue
            col, pts = _bgr(c.color), tf(r.pts)
            cv2.polylines(view, [pts], True, col, 2 if r is self.sel else 1, cv2.LINE_AA)
            cx, cy = pts.mean(axis=0)
            cv2.putText(view, f"{c.name} {seen[r.cls]}", (int(cx) - 14, int(cy)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (240, 240, 240), 1, cv2.LINE_AA)

        # nodes of the selected region
        if self.sel is not None and len(self.sel.pts) >= 3:
            c = cm.get(self.sel.cls)
            col = _bgr(c.color) if c else (255, 255, 255)
            for nx, ny in tf(self.sel.pts):
                cv2.circle(view, (int(nx), int(ny)), 5, (255, 255, 255), -1, cv2.LINE_AA)
                cv2.circle(view, (int(nx), int(ny)), 5, col, 1, cv2.LINE_AA)

        if self.v_dust.get() and len(self.specks):
            vis = np.flatnonzero((self.sp_x >= ix0 - 20) & (self.sp_x < ix1 + 20) &
                                 (self.sp_y >= iy0 - 20) & (self.sp_y < iy1 + 20))
            zc = self._speck_class_ids()
            for i in vis[:4000]:
                s = self.specks[i]
                done = self.mask[min(int(s.cy), H - 1), min(int(s.cx), W - 1)] > 0
                zid = zc[i] if i < len(zc) else None
                col = C_DUST_DONE if done else (_bgr(cm[zid].color) if zid in cm else C_DUST)
                cv2.circle(view, (int((s.cx - ix0) * z), int((s.cy - iy0) * z)),
                           int(max(7, s.r * z + 5)), col, 1, cv2.LINE_AA)

        if self.v_sug.get() and len(self.suggestions):
            vis = np.flatnonzero(self.sg_pending & (self.sg_x >= ix0 - 20) & (self.sg_x < ix1 + 20) &
                                 (self.sg_y >= iy0 - 20) & (self.sg_y < iy1 + 20))
            for i in vis[:4000]:
                s = self.suggestions[i]
                matched = self.sg_match[i] >= 0
                col = C_SUG_MATCH if matched else (C_SUG_REVIEW if s.status == "review" else C_SUG)
                px, py = int((s.cx - ix0) * z), int((s.cy - iy0) * z)
                hs = int(max(8, s.r * z + 6))
                cv2.rectangle(view, (px - hs, py - hs), (px + hs, py + hs), col, 2 if matched else 1, cv2.LINE_AA)

        # polygon being drawn
        if self.draft:
            c = cm.get(self.active_cls)
            col = _bgr(c.color) if c else (255, 255, 255)
            pts = tf(np.array(self.draft, np.float32))
            cv2.polylines(view, [pts], False, col, 2, cv2.LINE_AA)
            last = (int(pts[-1][0]), int(pts[-1][1]))
            cv2.line(view, last, (self._mouse[0] - off_x, self._mouse[1] - off_y), col, 1, cv2.LINE_AA)
            for k, (px, py) in enumerate(pts):
                cv2.circle(view, (int(px), int(py)), 6 if k == 0 else 4,
                           (128, 222, 74) if k == 0 else (255, 255, 255), -1, cv2.LINE_AA)

        self._photo = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(view, cv2.COLOR_BGR2RGB)))
        cv.create_image(off_x, off_y, anchor="nw", image=self._photo, tags="img")
        cv.tag_lower("img")
        cv.create_text(12, ch - 10, anchor="sw", fill=MUTED, tags="hud",
                       text=f"{z * self.scale * 100:.0f}% of original pixels")

    def _cursor(self, e) -> None:
        c = self.canvas
        c.delete("brush")
        if e is None:
            return
        self._mouse = (e.x, e.y)
        if self.proc is None:
            return
        tool = self.tool.get()
        if tool == "Brush":
            r = self.brush.get() / 2 * self.zoom
            c.create_oval(e.x - r, e.y - r, e.x + r, e.y + r, outline="#e5e7eb", tags="brush")
        elif tool == "Draw" and self.draft:
            self.request_render()                       # rubber-band line to the cursor

    # --------------------------------------------------------------- mouse
    def _press(self, e, button: int) -> None:
        if self.proc is None:
            return
        self.canvas.focus_set()
        self._last = (e.x, e.y)
        if button == 2 or (e.state & 0x1):
            self._drag = "pan"
            return
        tool = self.tool.get()
        if tool == "Pick":
            self._pick(e, None if button == 1 else False)
            return
        if tool == "Dust":
            self._remove_speck(e)
            return
        if tool == "AddDust":
            if button == 1:
                self._add_speck(e)
            return
        if tool == "TagDust":
            self._tag_speck(e, button)
            return
        if tool == "Assist":
            self._assist_press(e, button)
            return
        if tool == "Nodes":
            self._nodes_press(e, button)
            return
        if tool == "Draw":
            self._draw_press(e, button)
            return
        self._push_undo()                               # brush
        self._drag = "paint" if button == 1 else "erase"
        p = self._to_img(e.x, e.y)
        self._stamp(p, p, 255 if self._drag == "paint" else 0)
        self.request_render()

    def _double(self, e, button: int) -> None:
        """A quick 2nd click arrives as <Double-…>: close the polygon in Draw, else treat it as a normal press."""
        if self.proc is None:
            return
        if button == 1 and self.tool.get() == "Draw" and not (e.state & 0x1):
            if len(self.draft) >= 3:
                self._finish_draft()
            return
        self._press(e, button)

    def _draw_press(self, e, button: int) -> None:
        if button == 3:
            if self.draft:
                self.draft.pop()
                self.request_render()
            return
        p = self._to_img(e.x, e.y)
        if len(self.draft) >= 3:
            fx, fy = self.draft[0]
            if math.hypot(fx - p[0], fy - p[1]) * self.zoom <= NODE_HIT:
                self._finish_draft()
                return
        self.draft.append(p)
        self.request_render()

    def _nodes_press(self, e, button: int) -> None:
        x, y = self._to_img(e.x, e.y)
        z, sel = self.zoom, self.sel
        if sel is not None:
            d = np.hypot(sel.pts[:, 0] - x, sel.pts[:, 1] - y) * z
            i = int(d.argmin())
            if d[i] <= NODE_HIT:                                    # on a node
                if button == 3:
                    if len(sel.pts) > 3:
                        self._push_undo()
                        sel.pts, sel.auto = np.delete(sel.pts, i, axis=0), False
                        self._regions_changed()
                    else:
                        self.status("A region needs at least 3 nodes.")
                    return
                self._pre, self._drag, self._node = self._snapshot(), "node", i
                return
            if button == 1:                                         # on an edge -> insert a node and drag it
                dist, k, proj = core.nearest_edge(sel.pts, x, y)
                if dist * z <= EDGE_HIT:
                    self._push_undo()
                    sel.pts = np.insert(sel.pts, k + 1, proj.reshape(1, 2), axis=0)
                    sel.auto = False
                    self._drag, self._node = "node", k + 1
                    self._regions_changed()
                    return
        if button != 1:
            return
        hit = self._region_at(x, y)
        if hit is None:
            self.sel = None
        elif hit is sel:
            self._pre, self._drag = self._snapshot(), "move"        # drag the whole region
        else:
            self.sel = hit
            c = self._cls(hit.cls)
            self.status(f"Selected {c.name if c else '?'} region · {len(hit.pts)} nodes")
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
        elif self._drag == "node" and self.sel is not None:
            self._commit_pre()
            H, W = self.proc.shape[:2]
            x, y = self._to_img(e.x, e.y)
            self.sel.pts[self._node] = (min(max(x, 0), W - 1), min(max(y, 0), H - 1))
            self.sel.auto = False
            self._rev += 1
        elif self._drag == "move" and self.sel is not None:
            self._commit_pre()
            d = np.array([(e.x - self._last[0]) / self.zoom, (e.y - self._last[1]) / self.zoom], np.float32)
            self.sel.pts = self.sel.pts + d
            self.sel.auto = False
            self._rev += 1
        else:
            return
        self._last = (e.x, e.y)
        self.request_render()

    def _release(self, e) -> None:
        if self._drag in ("paint", "erase"):
            self._update_stats()
        elif self._drag in ("node", "move"):
            self._update_stats()
            self.request_render()
        self._drag, self._pre = None, None

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
