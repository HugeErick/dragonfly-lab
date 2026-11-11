"""Builds the benchmark package (Sprint 1, deliverables 1 and 6) from the annotated images.

Nothing here is manual: it scans `src/<image id>/` for finished annotations, freezes a reproducible
development / test split, copies the originals + the GT files into the agreed layout and writes
metadata.csv, README.md and (once) annotation_rules.md.

    python -m dragonfly_lab.benchmark            # same as the "Build benchmark package" button
"""
from __future__ import annotations

import csv
import json
import random
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from . import core

BENCH_ROOT = core.PROJECT_ROOT / "benchmark"
SEED = 42                      # fixed -> the split is reproducible
TEST_FRACTION = 1 / 3          # 30 images -> 20 development / 10 test
LEVELS = ("light", "medium", "heavy")     # dust load, by tertile of the dust count inside the package
SPLIT_FILE = "split.json"
OVERRIDE_FILE = "split_override.csv"      # optional: "id,split" lines to force an assignment

META_COLUMNS = [
    "id", "split", "image_file", "format", "width", "height", "file_bytes", "sha256",
    "exif_orientation", "source_folder", "dust_level", "dust_total", "dust_actionable",
    "dust_non_actionable", "dust_background", "dust_specimen", "dust_wing", "dust_paper_info",
    "dust_other", "has_specimen", "has_wings", "has_info", "specimen_area_pct", "wings_area_pct",
    "info_area_pct", "dust_area_pct", "regions_hand_edited", "dust_from_assist",
]


@dataclass
class BuildResult:
    root: Path
    n_dev: int = 0
    n_test: int = 0
    skipped: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def summary(self) -> str:
        lines = [f"Benchmark written to {self.root}",
                 f"development: {self.n_dev} images   test: {self.n_test} images"]
        if self.skipped:
            lines.append(f"\nSkipped {len(self.skipped)} annotation folder(s):")
            lines += [f"  - {s}" for s in self.skipped]
        lines += [f"\n{n}" for n in self.notes]
        return "\n".join(lines)


# ------------------------------------------------------------------ collecting
def _mask(path: Path) -> np.ndarray:
    m = cv2.imdecode(np.fromfile(str(path), np.uint8), cv2.IMREAD_UNCHANGED)
    if m is None:
        raise ValueError(f"cannot decode {path.name}")
    if m.ndim != 2:
        raise ValueError(f"{path.name} is not single-channel")
    return m


def _collect(d: Path) -> dict:
    """Reads one annotation folder. Raises ValueError with a human-readable reason if it is not
    ready to be packaged (so it is skipped, never silently half-packaged)."""
    missing = [f for f in core.GT_FILES if not (d / f).exists()]
    if missing:
        raise ValueError(f"missing {', '.join(missing)} - open the image and press “Output annotation”")
    originals = sorted(d.glob("original.*"))
    if not originals:
        raise ValueError("missing original.<ext> - open the image and press “Output annotation” again")
    orig = originals[0]
    proj = json.loads((d / core.ANNOT_FILE).read_text(encoding="utf-8"))
    dust = json.loads((d / core.DUST_JSON).read_text(encoding="utf-8"))
    if any("zone" not in it for it in dust.get("instances", [])):
        raise ValueError("dust_instances.json is from an older version - re-export it")

    with Image.open(orig) as im:
        w, h = im.size                                  # stored pixel grid (no EXIF rotation)
    areas = {}
    for key, fname in (("specimen", core.SPECIMEN_MASK), ("wings", core.WINGS_MASK),
                       ("info", core.INFO_MASK), ("dust", core.DUST_MASK)):
        m = _mask(d / fname)
        if m.shape != (h, w):
            raise ValueError(f"{fname} is {m.shape[1]}x{m.shape[0]} but the original is {w}x{h}")
        areas[key] = 100.0 * float((m > 0).mean())
    if (dust.get("width"), dust.get("height")) != (w, h):
        raise ValueError("dust_instances.json width/height do not match the original")

    inst = dust.get("instances", [])
    zone = lambda z: sum(1 for it in inst if it["zone"] == z)       # noqa: E731
    known = {"background", "specimen", "wing", "paper_info"}
    src = proj.get("source", {})
    return {
        "dir": d, "orig": orig, "id": d.name, "w": w, "h": h,
        "sha": core.file_sha256(orig), "bytes": orig.stat().st_size,
        "exif": src.get("exif_orientation", core.exif_orientation(orig)),
        "folder": src.get("folder", ""),
        "dust_total": len(inst),
        "dust_actionable": sum(1 for it in inst if it["actionable"]),
        "dust_background": zone("background"), "dust_specimen": zone("specimen"),
        "dust_wing": zone("wing"), "dust_paper_info": zone("paper_info"),
        "dust_other": sum(1 for it in inst if it["zone"] not in known),
        "dust_from_assist": sum(1 for it in inst if it.get("suggested_by")),
        "areas": areas,
        "edited": sum(1 for r in proj.get("regions", []) if not r.get("auto", False)),
    }


def _assign_levels(items: list[dict]) -> None:
    order = sorted(items, key=lambda it: (it["dust_total"], it["id"]))
    n = len(order)
    for rank, it in enumerate(order):
        it["level"] = LEVELS[min(2, rank * 3 // max(n, 1))]


# ------------------------------------------------------------------ the split
def _read_override(root: Path) -> dict[str, str]:
    p = root / OVERRIDE_FILE
    out: dict[str, str] = {}
    if p.exists():
        for row in csv.reader(p.read_text(encoding="utf-8").splitlines()):
            if len(row) >= 2 and row[1].strip().lower() in ("development", "test"):
                out[row[0].strip()] = row[1].strip().lower()
    return out


def assign_split(items: list[dict], frozen: dict[str, str], override: dict[str, str]) -> dict[str, str]:
    """development / test assignment, stratified by dust level and deterministic (fixed seed).
    Images that already have an assignment keep it (the split is frozen once made)."""
    ids = [it["id"] for it in items]
    result = {i: frozen[i] for i in ids if i in frozen}
    result.update({i: v for i, v in override.items() if i in ids})
    new = [it for it in items if it["id"] not in result]
    if new:
        target = round(len(ids) * TEST_FRACTION)
        need = min(len(new), max(0, target - sum(v == "test" for v in result.values())))
        groups = {lv: sorted(it["id"] for it in new if it["level"] == lv) for lv in LEVELS}
        quota = {lv: need * len(g) / len(new) for lv, g in groups.items()}
        alloc = {lv: int(q) for lv, q in quota.items()}
        for lv in sorted(LEVELS, key=lambda k: -(quota[k] - alloc[k]))[: need - sum(alloc.values())]:
            alloc[lv] += 1
        rng = random.Random(SEED)
        for lv in LEVELS:
            g = groups[lv]
            rng.shuffle(g)
            for k, i in enumerate(g):
                result[i] = "test" if k < alloc[lv] else "development"
    return result


# ------------------------------------------------------------------- the build
def build_package(root: Path | None = None, annotations_root: Path | None = None) -> BuildResult:
    root = Path(root) if root else BENCH_ROOT
    ann_root = Path(annotations_root) if annotations_root else core.OUTPUT_ROOT
    res = BuildResult(root)

    items = []
    for d in sorted(p for p in ann_root.glob("*") if (p / core.ANNOT_FILE).exists()):
        try:
            items.append(_collect(d))
        except (ValueError, OSError, KeyError, json.JSONDecodeError) as ex:
            res.skipped.append(f"{d.name}: {ex}")
    if not items:
        res.notes.append("Nothing to package: no finished annotations were found.")
        return res
    dup = {it["sha"] for it in items if sum(o["sha"] == it["sha"] for o in items) > 1}
    if dup:
        res.notes.append("WARNING: the same photo appears under more than one id (identical file hash).")

    _assign_levels(items)
    root.mkdir(parents=True, exist_ok=True)
    sp = root / SPLIT_FILE
    frozen = {}
    if sp.exists():
        try:
            frozen = json.loads(sp.read_text(encoding="utf-8")).get("assignments", {})
        except (ValueError, OSError):
            frozen = {}
    split = assign_split(items, frozen, _read_override(root))
    sp.write_text(json.dumps({
        "seed": SEED, "test_fraction": TEST_FRACTION, "stratified_by": "dust_level",
        "note": "Frozen: once an image is assigned it keeps its split. Force one with split_override.csv.",
        "assignments": dict(sorted(split.items())),
    }, indent=2), encoding="utf-8")

    for part in ("development", "test"):                 # rebuilt from scratch every time
        if (root / part).exists():
            shutil.rmtree(root / part)
    rows = []
    for it in sorted(items, key=lambda x: x["id"]):
        part = split[it["id"]]
        img_dir, ann_dir = root / part / "images", root / part / "annotations" / it["id"]
        img_dir.mkdir(parents=True, exist_ok=True)
        ann_dir.mkdir(parents=True, exist_ok=True)
        image_file = f"{it['id']}{it['orig'].suffix.lower()}"
        shutil.copy2(it["orig"], img_dir / image_file)
        for f in core.GT_FILES:
            shutil.copy2(it["dir"] / f, ann_dir / f)
        # the json inside the package points at the packaged image name
        dj = json.loads((ann_dir / core.DUST_JSON).read_text(encoding="utf-8"))
        dj["image_id"], dj["image"] = it["id"], image_file
        (ann_dir / core.DUST_JSON).write_text(json.dumps(dj, indent=2), encoding="utf-8")
        rows.append({
            "id": it["id"], "split": part, "image_file": f"{part}/images/{image_file}",
            "format": {".jpg": "JPEG", ".jpeg": "JPEG", ".png": "PNG", ".tif": "TIFF", ".tiff": "TIFF",
                       ".bmp": "BMP", ".webp": "WEBP"}.get(it["orig"].suffix.lower(), it["orig"].suffix),
            "width": it["w"], "height": it["h"], "file_bytes": it["bytes"], "sha256": it["sha"],
            "exif_orientation": it["exif"], "source_folder": it["folder"], "dust_level": it["level"],
            "dust_total": it["dust_total"], "dust_actionable": it["dust_actionable"],
            "dust_non_actionable": it["dust_total"] - it["dust_actionable"],
            "dust_background": it["dust_background"], "dust_specimen": it["dust_specimen"],
            "dust_wing": it["dust_wing"], "dust_paper_info": it["dust_paper_info"],
            "dust_other": it["dust_other"],
            "has_specimen": int(it["areas"]["specimen"] > 0), "has_wings": int(it["areas"]["wings"] > 0),
            "has_info": int(it["areas"]["info"] > 0),
            "specimen_area_pct": round(it["areas"]["specimen"], 3),
            "wings_area_pct": round(it["areas"]["wings"], 3),
            "info_area_pct": round(it["areas"]["info"], 3),
            "dust_area_pct": round(it["areas"]["dust"], 4),
            "regions_hand_edited": it["edited"],
            "dust_from_assist": it["dust_from_assist"],
        })
    with open(root / "metadata.csv", "w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=META_COLUMNS)
        wr.writeheader()
        wr.writerows(rows)

    res.n_dev = sum(r["split"] == "development" for r in rows)
    res.n_test = len(rows) - res.n_dev
    (root / "README.md").write_text(_readme(rows), encoding="utf-8")
    if not (root / "annotation_rules.md").exists():       # human-editable: never overwritten
        (root / "annotation_rules.md").write_text(RULES, encoding="utf-8")
    if len(rows) != 30:
        res.notes.append(f"Note: the package has {len(rows)} images (the plan calls for 30: 20 development / "
                         "10 test). Annotate / export the missing ones and build again; existing "
                         "assignments stay frozen.")
    if any(r["has_wings"] == 0 or r["has_specimen"] == 0 for r in rows):
        res.notes.append("Note: some images have an empty specimen or wings mask "
                         "(see has_specimen / has_wings in metadata.csv).")
    return res


# --------------------------------------------------------------------- texts
def _readme(rows: list[dict]) -> str:
    dev = [r for r in rows if r["split"] == "development"]
    tst = [r for r in rows if r["split"] == "test"]
    lv = lambda rs, k: sum(r["dust_level"] == k for r in rs)          # noqa: E731
    dust = lambda rs: sum(r["dust_total"] for r in rs)                # noqa: E731
    act = lambda rs: sum(r["dust_actionable"] for r in rs)            # noqa: E731
    return f"""# Dragonfly cleaning benchmark - Ground Truth package

Generated by Dragonfly Lab (`python -m dragonfly_lab.benchmark`). Do not edit by hand: re-run the
build instead. Rules that are not obvious from the masks are in `annotation_rules.md`.

| split | images | dust load (light / medium / heavy) | dust specks | actionable |
|---|---|---|---|---|
| development | {len(dev)} | {lv(dev, 'light')} / {lv(dev, 'medium')} / {lv(dev, 'heavy')} | {dust(dev)} | {act(dev)} |
| test | {len(tst)} | {lv(tst, 'light')} / {lv(tst, 'medium')} / {lv(tst, 'heavy')} | {dust(tst)} | {act(tst)} |

**The test annotations must stay out of method tuning.** Use `development/` to build and tune; run the
final configuration on `test/` once and hand the outputs to the common evaluator.

## Layout

```
benchmark/
  README.md
  annotation_rules.md
  metadata.csv                 one row per image (see columns below)
  split.json                   frozen development/test assignment (+ seed)
  development/
    images/<ID>.<ext>          original photo, byte-exact
    annotations/<ID>/
      specimen_mask.png
      wings_mask.png
      dust_mask.png
      info_mask.png
      dust_instances.json
  test/                        same structure - keep out of method tuning
```

`<ID>` is the stable image id (the file name without extension). It is the same in the image, the
annotation folder and `metadata.csv`.

## Classes and formats

| class | file | meaning |
|---|---|---|
| specimen | `specimen_mask.png` | body of the dragonfly (head, thorax, abdomen, legs), wings excluded |
| wings | `wings_mask.png` | wings, as an independent region |
| dust | `dust_mask.png` + `dust_instances.json` | real dust specks: global mask and one instance per speck |
| paper / info | `info_mask.png` | millimetre paper, label or any informative region, separate from the specimen |

* Masks are single-channel PNG, **exactly the width x height of the original**, 255 = class, 0 = not.
  Empty (all 0) means the class is absent in that image.
* Read the original with its stored pixel grid: `cv2.imread(path, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)`
  (Pillow's `Image.open` already does this). Do not apply EXIF rotation, or the masks will not line up.
  `exif_orientation` in `metadata.csv` flags files that carry such a tag.
* `dust_instances.json` coordinates are original-image pixels, origin top-left:

```json
{{
  "image_id": "IMG_001", "image": "IMG_001.jpg", "width": 6000, "height": 4000,
  "count": 2, "count_actionable": 1,
  "instances": [
    {{"id": "MOTA_007", "class": "background", "class_id": null, "zone": "background",
     "origin": "auto", "class_source": "auto", "actionable": true,
     "center": [1234.5, 876.0], "radius": 9.5, "bbox": [1225, 867, 19, 18], "area": 241.5,
     "dark": true, "contrast": 18.4, "contour": [[1225, 870], [1230, 867]]}},
    {{"id": "MOTA_008", "class": "wings", "class_id": 2, "zone": "wing",
     "class_source": "auto", "actionable": false, "...": "..."}}
  ]
}}
```

* `id` - stable per image (`MOTA_<number>`); deleting a speck never renumbers the others.
* `origin` - `auto` (found by the detector) or `manual` (added by a person).
* `suggested_by` - only present when a person accepted a suggestion of the optional dust-assist helper
  (`"dust_assist"`); those specks have `origin` `manual`. Rejected or unreviewed suggestions are not part
  of the package (`dust_suggestions.json` is provenance, not ground truth, and is never copied here).
* `class` / `zone` - what the speck sits on: `specimen`, `wing`, `paper_info`, a custom class, or
  `background`. `class_source` says whether that came from geometry (`auto`) or was set by hand (`manual`).
* `actionable` - `true` only for dust on the background. Dust over the dragonfly or any protected
  class is annotated but **must not be cleaned automatically**; its existence does not authorise a
  method to modify the specimen.
* `dark` - darker than the surroundings; `contrast` - mean absolute difference from the local background.
* `dust_mask.png` contains **every** speck in the list (actionable or not), outline filled plus a thin ring.

## metadata.csv columns

`id, split, image_file, format, width, height, file_bytes, sha256, exif_orientation, source_folder,
dust_level` (light / medium / heavy = tertile of the dust count inside this package), `dust_total,
dust_actionable, dust_non_actionable, dust_background, dust_specimen, dust_wing, dust_paper_info,
dust_other, has_specimen, has_wings, has_info, *_area_pct` (percentage of the image covered by each
mask), `regions_hand_edited` (polygons touched by a person after auto-detection), `dust_from_assist` (dust
specks a person accepted from the optional dust-assist helper).

## Split

Stratified by `dust_level`, fixed seed ({SEED}), one third of the images to `test`. The assignment is
stored in `split.json` and frozen: it does not change when the package is rebuilt.
"""


RULES = """# Annotation rules

Short, common rules so two people read the regions the same way. Edit this file freely: the build
never overwrites it. Add a line here whenever a decision is not obvious from the masks.

## Classes

* **specimen** - the dragonfly body: head, thorax, abdomen, legs and other non-wing structures.
  Wings are *not* part of it. The whole animal is `specimen OR wings`.
* **wings** - every wing, as its own region, including the translucent parts and the veins.
  Where a wing meets the body the boundary follows the body outline (the auto-detector subtracts the
  body from the wings); keep that convention when correcting by hand.
* **dust** - real dust specks (small, compact, clearly not part of the animal, the paper or the
  background texture). Each speck is one instance with its own id. Veins, hairs, label letters and
  paper texture are not dust.
* **paper / info** - millimetre paper, label or any written / informative area. Kept apart from the
  specimen. It is a protected class.

## Dust: zone and actionable

* The zone of a speck is the class it sits on; outside every class it is `background`.
  A speck that only touches a region still belongs to that class.
* `actionable = true` only on the background. Dust over the specimen, wings, paper/info or any protected
  class is annotated but `actionable = false`: methods must not change the specimen to remove it.
* The class can be set by hand (Tag-dust tool); then `class_source = "manual"`.

## Conventions of the tool

* Masks are drawn as polygons on a working image of at most 2400 px on the long side and rasterized at the
  original resolution, so edges are accurate to about one working-image pixel.
* The brush mask of the editor (`edit_mask.png`) is **not** part of the ground truth.
* Suggestions of the optional dust-assist helper are proposals, never ground truth: a speck is dust only
  after a person accepted it (it then carries `suggested_by` in `dust_instances.json`).
* Annotation never resizes or rotates the photo. EXIF orientation is ignored on purpose.

## Decisions taken during annotation

(Add here, e.g. "tarsi shorter than 2 px are left out of the specimen", "label shadows are not dust".)
"""


def main() -> None:
    res = build_package()
    print(res.summary())
    sys.exit(0 if res.n_dev + res.n_test else 1)


if __name__ == "__main__":
    main()
