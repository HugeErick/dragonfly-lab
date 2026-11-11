# Dragonfly Lab

CPU-only **ground-truth annotation assistant** for dragonfly macro photos. It finds the specimen and wings, hunts for dust specks, lets a person correct everything by hand, exports full-resolution masks plus a JSON list of dust instances, and assembles the whole thing into the **benchmark package** used to compare dust-cleaning methods. No ML models: classical computer vision (OpenCV + NumPy) with a dark customtkinter UI.

> In this project the app is an *assisted annotation tool*, not an automatic source of GT. Automatic detection speeds up the work; every final mask and every reference speck is reviewed by a person.

## What "Ground Truth" and "benchmark" mean here

* **Ground Truth (GT)** - for each photo, the human-verified answer to "where is the specimen, where are the wings, where is the paper/label, and which specks are real dust". It is independent of any cleaning method.
* **Benchmark** - a fixed set of 30 photos (20 *development* for building and tuning methods, 10 *test* reserved for the final comparison) together with their GT. Every team runs its cleaning method on the same photos, and a **common evaluator** (Team 1) compares each method's output against the GT, e.g. "was the dust on the background removed?" and "was the specimen left untouched?".
* The GT stays frozen while methods are evaluated. If a method reveals a possible annotation error, it is logged and reviewed in a controlled way; the GT is not edited to improve a score.

## Requirements

* Python 3.10+
* [uv](https://docs.astral.sh/uv/) (recommended)
* Dependencies (installed automatically): `customtkinter`, `numpy`, `opencv-python-headless`, `pillow`

## Install and run

From the project root:

```powershell
uv sync
uv run dragonfly-lab
```

or `uv run python -m dragonfly_lab.app`. Do not run `app.py` directly (relative imports). Always launch from the project root: annotations go to `<project>/src/<image id>/` and the package to `<project>/benchmark/`; the root is found by looking for `pyproject.toml`, falling back to the current directory.

If you edit the code and `uv run` still shows the old behaviour: `uv sync --reinstall-package dragonfly-lab`.

## Workflow (per image)

1. **Open image…** and pick a photo (jpg, png, tif, bmp, webp). Re-opening restores the previous annotation.
2. **Detect body & wings**, adjust *Body / wing sensitivity* and re-run until close. Detection produces editable polygons.
3. **Fix the regions** (Nodes / Draw / Pick), and draw the **paper_info** region (millimetre paper, label) with *Draw* while that class is active.
4. **Detect dust specks**, then review them: **Dust −** deletes false positives, **Dust ＋** adds specks the detector missed (hand-added specks survive a re-detection), **Tag dust** attaches a speck to the active class by hand.
5. **Output annotation** writes the files of that image.

Once all images are annotated: **Build benchmark package** (or `uv run dragonfly-lab-benchmark`, or `python -m dragonfly_lab.benchmark`).

## Classes

| Class (built-in) | Meaning | `guard` default |
|---|---|---|
| `specimen` | dragonfly body (wings excluded) | on |
| `wings` | wings, as an independent region | on |
| `paper_info` | millimetre paper / label / written area | on |

Built-in classes can be renamed but not deleted; you can add your own. `guard` = dust detection keeps away from that class, and dust that sits on a guarded class is marked *not actionable*. Dust itself is not a class but a set of instances.

## Dust: id, class and actionable

Every speck has a stable id (`MOTA_007`; deleting a speck never renumbers the others) and the class it sits on:

* **By position (default):** the class whose regions overlap the speck most; outside every class it is `background`.
* **By hand:** with the *Tag dust* tool, click a speck to attach it to the **active class** (right-click = back to automatic). The switch *New dust takes the active class* makes *Dust ＋* do the same when adding.
* `actionable` is `true` only for dust on the background (or on a class whose `guard` is off). Dust over the dragonfly or any protected class is still annotated but a method may **not** modify the specimen to remove it.

On the canvas, circles are yellow on the background, the class colour on a class, and green once covered by the mask.

## Controls

| Action | Input |
|---|---|
| Zoom | Mouse wheel over the image |
| Pan | Shift + drag, or middle-drag |
| Scroll the side panel | Mouse wheel (sliders ignore the wheel on purpose) |
| Undo | Ctrl+Z (Cmd+Z on macOS) |

| Tool | Use |
|---|---|
| Brush | L-drag paint the edit mask, R-drag erase |
| Pick | click a region / speck to toggle its edit mask |
| Nodes | select a region, drag nodes, click an edge to add a node, R-click a node to delete, Del removes the region |
| Draw | click points, close on the first point / double-click / Enter, R-click undoes a point, Esc cancels |
| Dust − | click a circle to delete it |
| Dust ＋ | click a missed speck to add it |
| Tag dust | attach a speck to the active class |

## Settings

| Setting | Effect |
|---|---|
| Body / wing sensitivity | Higher finds more foreground; lower is stricter. |
| Node spacing | Polygon simplification of auto-detected regions. |
| Dust sensitivity (1-10) | Higher detects fainter specks (and more false positives). |
| Max speck size (px) | Larger blobs are ignored. |
| Keep dust away from guarded classes / margin | Discards specks touching the guarded zone (+ margin). |
| Brush size (px) | Brush / eraser size. |

Sizes are in the working image (a proxy of at most 2400 px on the long side). The HUD shows the zoom relative to the original pixels. Masks are rasterized at full resolution from the polygons, so they never depend on the proxy.

## Output: one annotation folder per image

*Output annotation* writes `src/<image id>/`. The **image id** is the file name without its image extension(s); if two *different* photos share a name (e.g. `light/IMG_001.jpg` and `heavy/IMG_001.jpg`) the second becomes `IMG_001_2` instead of overwriting the first.

```
src/IMG_001/
    original.jpg           byte-exact copy of the photo
    specimen_mask.png      full-resolution, single channel, same width x height as the original
    wings_mask.png         255 = class, 0 = not; all-zero = class absent
    dust_mask.png          every dust speck still in the list
    info_mask.png          paper / label
    dust_instances.json    one entry per dust speck
    extra_classes/         masks of user-defined classes (only if any)
    annotation.json        editable project (classes + polygons) - tool-internal, reloaded on open
    edit_mask.png          brush mask of the editor - tool-internal, not part of the GT
```

Pixel grid: images are read exactly as stored (EXIF rotation is **not** applied), so masks line up with the original file. The EXIF tag is recorded in `annotation.json` / `metadata.csv`.

### `dust_instances.json`

Coordinates are original-image pixels, origin top-left.

```json
{
  "image_id": "IMG_001", "image": "original.jpg", "width": 6000, "height": 4000,
  "count": 2, "count_actionable": 1,
  "instances": [
    {"id": "MOTA_007", "class": "background", "class_id": null, "zone": "background",
     "origin": "auto", "class_source": "auto", "actionable": true,
     "center": [1234.5, 876.0], "radius": 9.5, "bbox": [1225, 867, 19, 18], "area": 241.5,
     "dark": true, "contrast": 18.4, "contour": [[1225, 870], [1230, 867]]},
    {"id": "MOTA_008", "class": "wings", "class_id": 2, "zone": "wing",
     "origin": "manual", "class_source": "manual", "actionable": false, "...": "..."}
  ]
}
```

`zone` is the rename-proof label (`specimen`, `wing`, `paper_info`, custom class slug, `background`); `class` is the class name as displayed. `origin` says whether the speck was auto-detected or added by a person.

## Benchmark package

**Build benchmark package** reads the last exported annotation of every image and writes `benchmark/`:

```
benchmark/
  README.md  annotation_rules.md  metadata.csv  split.json
  development/images/<ID>.<ext>      development/annotations/<ID>/{specimen,wings,dust,info}_mask.png + dust_instances.json
  test/images/<ID>.<ext>             test/annotations/<ID>/...      (keep out of method tuning)
```

* **Split:** one third of the images go to `test` (30 images -> 20 development / 10 test), stratified by dust load (light / medium / heavy = tertile of the dust count), fixed seed. It is **frozen** in `split.json`: rebuilding never reshuffles an image that already has a split. To force an assignment, create `benchmark/split_override.csv` with lines `IMG_012,test`.
* `metadata.csv`: one row per image (format, size, sha256, EXIF flag, source folder, dust counts by zone, mask coverage, ...).
* `README.md` is regenerated on every build; `annotation_rules.md` is written once and then belongs to the team (add decisions as they are taken).
* The builder skips (and reports) any annotation folder that is incomplete, from an older tool version, or whose masks do not match the original's width x height.
* Only the last *exported* state is packaged: press **Output annotation** after the final edit of each image.

## How it works

**Body and wings** (`core.detect_body_and_wings`): model the backdrop from border colours (Lab distance), foreground = colour difference OR dense vein texture, body = the largest thick strongly-different structure, wings = foreground minus body split into pieces. Results become editable polygons (`regions_from_detection`).

**Dust** (`core.detect_dust`): subtract a large median blur, threshold with a robust noise estimate, keep compact components outside the guarded zone, ranked by contrast (up to 3000), numbered in reading order. `core.speck_at` outlines one speck at a click for manual additions; `core.assign_zones` decides which class each speck sits on.

## Sprint 1 status (Definition of Done)

| Item | Status |
|---|---|
| 30 images selected, 20 development / 10 test | split is automatic and frozen (`benchmark.py`); the choice of the 30 photos is made outside the tool (Drive bank) |
| Tool validated on 5 pilot images | the tool covers the full flow (load, segment, correct, save, reopen, export); the pilot itself, with timings, is a team activity |
| Annotation protocol | classes, zones and `actionable` rule are fixed by the tool and written to `annotation_rules.md`; the team completes its own decisions there |
| 30 images with masks and dust registered | produced with this tool; annotation and review of each image are human |
| Masks pass resolution / alignment / consistency checks | human QA. The tool only guarantees by construction: exact original width x height, no resizing or rotation, byte-exact original, and the builder refuses mismatching masks |
| GT package independent of the software | `benchmark/` = plain PNG / JSON / CSV, no dependency on the app |
| Team 1 can consume it without per-image manual changes | fixed names / layout / `metadata.csv`; to be confirmed against the common evaluator's input format |

So the **tooling for Sprint 1 is complete**; the sprint is done once the humans have annotated and QA'd the 30 images and `Build benchmark package` has produced the final package. Sprint 2 (applying extra cleaning methods and evaluating them) needs the common evaluator from Team 1, which is not part of this repository.

## Notes for existing projects

* Re-exporting an image removes files from older versions (`<name>.txt` YOLO labels, `classes.txt`, `class_masks/`, `<name>_mask.png`, `<name>.<ext>` copy). The brush mask is migrated to `edit_mask.png` on the next export.
* Old projects with `body` / `wing` class names keep them; the exporter maps by class id, not by name. The new `paper_info` class is added automatically.
* If an old image has an EXIF rotation, re-check it: it is now shown on its stored pixel grid.

## Project layout

```
dragonfly-lab/
├─ pyproject.toml
├─ README.md
├─ benchmark/                  generated package
└─ src/
   ├─ dragonfly_lab/
   │  ├─ app.py                UI (customtkinter + canvas rendering)
   │  ├─ core.py               detection, classes, dust, per-image export
   │  └─ benchmark.py          split, metadata and package builder
   └─ <image id>/              per-image annotations
```

## Tips

* If the body is not found, raise *Body / wing sensitivity*; if background shows up as wings, lower it.
* Check the **Keep-away zone** view before detecting dust. Dust on translucent wings is intentionally not auto-reported (the dragonfly is excluded); add it with *Dust ＋* if it is real - it will be marked not actionable.
* Too many false specks on veins or edges: lower *Dust sensitivity* or *Max speck size*, or delete them with *Dust −*.
