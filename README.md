# Dragonfly Lab

CPU-only inspector for dragonfly macro photos. It finds the body and wings, hunts for dust specks on the background, lets you clean everything up by hand, and exports masks plus a JSON list of dust instances. No ML models: it is classical computer vision (OpenCV + NumPy) with a dark customtkinter UI.

## Features

- **Body and wing detection** from colour distance to the backdrop plus vein texture, with an adjustable sensitivity.
- **Dust detection** for small, compact blobs that are darker or brighter than their local surroundings.
- **Keep-away zone**: dust is never reported on or near the dragonfly (body + wings + a configurable margin).
- **Manual editing**: brush/eraser, pick-to-toggle for specks, wings and body, and a *Remove dust* tool that deletes individual dust circles.
- **Undo** (Ctrl+Z / Cmd+Z) covers mask edits and removed dust circles.
- **Output annotation** exports full-resolution masks and `dust_instances.json`.
- Re-opening an image restores its previous mask and dust list.

## Requirements

- Python 3.10+
- [uv](https://docs.astral.sh/uv/) (recommended)
- Dependencies (installed automatically): `customtkinter`, `numpy`, `opencv-python-headless`, `pillow`

## Install and run

From the project root:

```powershell
uv sync
uv run dragonfly-lab
```

or

```powershell
uv run python -m dragonfly_lab.app
```

Do not run `app.py` directly (`uv run .\src\dragonfly_lab\app.py`). It uses relative imports and fails with *"attempted relative import with no known parent package"*.

Always launch from the project root. Exports go to `<project>/src/`, and the project root is found by looking for `pyproject.toml`, falling back to the current directory.

If you edit the code and `uv run` still shows the old behaviour, force a reinstall:

```powershell
uv sync --reinstall-package dragonfly-lab
```

## Workflow

1. **Open image…** and pick a photo (jpg, png, tif, bmp, webp).
2. **Detect body & wings.** Adjust *Body / wing sensitivity* and re-run until the overlay looks right.
3. **Detect dust specks.** Tune *Dust sensitivity*, *Max speck size* and *Keep-away margin*. If body/wings were not detected yet, they are detected first.
4. **Clean up:**
   - *Remove dust*: click a circle to delete it.
   - *Pick*: click a speck, wing or body to toggle it in the mask (right-click unmasks).
   - *Brush*: left-drag paints, right-drag erases.
5. Optionally use **Mask all dust**, **Mask body** or **Mask wings** to fill the mask in one go.
6. **Output annotation** to write the files.

## Controls

| Action | Input |
|---|---|
| Zoom | Mouse wheel |
| Pan | Shift + drag, or middle-drag |
| Paint / erase (Brush) | Left-drag / right-drag |
| Toggle / unmask (Pick) | Left-click / right-click |
| Delete a dust circle (Remove dust) | Click the circle |
| Undo | Ctrl+Z (Cmd+Z on macOS) |

### View toggles

Body, wings, dust specks, mask, and **Keep-away zone** (shows the excluded area in grey). Dust circles are yellow, and turn green once covered by the mask.

## Settings

| Setting | Effect |
|---|---|
| Body / wing sensitivity | Higher finds more foreground; lower is stricter. |
| Dust sensitivity (1–10) | Higher detects fainter specks (and more false positives). |
| Max speck size (px) | Larger blobs are ignored (rejects wing fragments and debris). |
| Keep dust away from dragonfly | Discards any speck that touches the dragonfly zone. |
| Keep-away margin (px) | Distance around body and wings that is off-limits to dust. |
| Brush size (px) | Size of the brush/eraser. |

All sizes are in the working image (a proxy of at most 2400 px on the long side). The HUD shows the zoom relative to the original pixels.

## Output

*Output annotation* writes to `src/<image name>/`. The name is the photo's file name with all trailing image extensions removed, so `IMG_001.jpg` and even `IMG_001.jpg.jpg` both give `IMG_001`:

```
src/IMG_001/
  IMG_001.jpg            copy of the original image
  IMG_001_mask.png       full mask (brush strokes + picked body/wings/dust)
  wings_mask.png         detected wings
  dust_mask.png          every dust circle still in the list
  dust_instances.json    one entry per dust circle
```

All masks are full-resolution, single-channel PNGs (255 = masked, 0 = not). `wings_mask.png` is empty if wings were never detected.

### `dust_instances.json`

Coordinates are in original-image pixels.

```json
{
  "image": "IMG_001.jpg",
  "width": 6000,
  "height": 4000,
  "count": 1,
  "instances": [
    {
      "id": 1,
      "center": [1234.5, 876.0],
      "radius": 9.5,
      "bbox": [1225, 867, 19, 18],
      "area": 241.5,
      "dark": true,
      "contrast": 18.4,
      "contour": [[1225, 870], [1230, 867]]
    }
  ]
}
```

`dark` is true for specks darker than their surroundings. `contrast` is the mean absolute difference from the local background.

Dust circles you removed are excluded from both `dust_mask.png` and the JSON. Re-opening the image loads the mask and dust list back from this folder.

## How it works

**Body and wings** (`core.detect_body_and_wings`)
1. Cluster the border colours to model the backdrop, then measure each pixel's Lab distance to it.
2. Foreground = colour difference OR dense vein texture (Canny edge density), cleaned up with morphology.
3. Body = the largest thick, strongly different structure that survives a large opening, grown back to recover thin parts.
4. Wings = foreground minus body, split into connected pieces (up to 8).

**Keep-away zone** (`core.exclusion_zone`)

Body and wings are merged, gaps are bridged and holes filled, then the shape is grown by the margin using a distance transform.

**Dust** (`core.detect_dust`)
1. Subtract a large median blur (the speck-free local background).
2. Threshold the difference using a robust noise estimate and the sensitivity.
3. Keep only compact components: small, roughly round, not touching the keep-away zone.
4. Rank by contrast and keep up to 3000.

## Project layout

```
dragonfly-lab/
├─ pyproject.toml
├─ README.md
└─ src/
   ├─ dragonfly_lab/
   │  ├─ app.py      UI (customtkinter + canvas rendering)
   │  └─ core.py     detection, masks, export
   └─ <image name>/  generated annotations
```

## Tips

- If the body is not found, raise *Body / wing sensitivity*. If background shows up as wings, lower it.
- Check the **Keep-away zone** view before detecting dust. If the dragonfly is only partly outlined, dust near the missed parts will still be reported.
- Too many false specks on wing veins or edges: lower *Dust sensitivity* or *Max speck size*.
- Dust on translucent wings is intentionally not reported; the whole dragonfly is excluded.
