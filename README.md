# Side-scan sonar preprocessing (`core/preprocess.py`)

Turns a raw XTF side-scan file into cleaned, tiled images ready for YOLO,
while keeping enough metadata to trace any detection back to a GPS position
and a size in metres.

```python
from core.preprocess import preprocess_xtf, locate_tile_pixel, bbox_to_metres

res = preprocess_xtf("data/NBP050504A.XTF", out_dir="out",
                     tile_size=640, overlap=0.2)

# later, once YOLO returns a detection in tile `tid` at pixel (x, y):
fix  = locate_tile_pixel(tid, x, y, tile_dir=res["tile_dir"])
dims = bbox_to_metres(fix, bbox_px=(x0, y0, w, h), shadow_len_px=35)
```

Every stage is a standalone function, so a FastAPI worker can call them
individually rather than running the whole pipeline.

## Stage 1 — read

| function | what it does |
| --- | --- |
| `read_xtf(path)` | pings, sorted chronologically |
| `diagnose_channels(fh, pings)` | per-channel storage direction + usability |
| `select_channel_pair(fh, pings, prefer="auto")` | picks the port/stbd pair |
| `extract_ping_metadata(pings, ch, name)` | **the ping table** everything keys off |
| `build_side_images(fh, pings, ch)` | stacks both channels via `pyxtf.concatenate_channel` |

The ping table has one row per ping — `ping_index`, `ping_number`,
`timestamp`, `lat`, `lon`, `heading`, `altitude`, `slant_range`, sample
counts — and **image row `i` is always table row `i`**. Stage 2 and 3 append
`nadir_sample`, `altitude_m`, `meters_per_pixel`,
`meters_per_pixel_alongtrack` and `bad_ping` to it.

## Stage 2 — clean

Applied in this order, each its own function, each saving a PNG:

1. `track_bottom` + `remove_nadir` — find the first seabed return per ping and
   crop the water-column blind spot.
2. `slant_to_ground_range` — `ground = sqrt(slant² - altitude²)`, resampled
   onto one shared grid so every column is a fixed number of metres wide.
   Sets `meters_per_pixel`.
3. `normalize_gain` — divide out the survey's average across-track falloff,
   and each ping's own level, so brightness is even near-to-far.
4. `despeckle_lee` — log transform (speckle is multiplicative), then a Lee
   filter, which collapses flat areas to their local mean while leaving
   high-variance edges and targets alone.

Optional: `resample_alongtrack` makes pixels square (see *Two scales* below).

## Which PNG goes to the model

**Whole-image detector →** `out_square/waterfall.png` (+ `waterfall.json`).
**Tile-based detector →** `out_square/tiles/*.png` (+ per-tile sidecars).

The `stage*.png` files are **diagnostics only**. They are stretched over the
whole array so each step is visible, and they still contain the junk pings a
line opens with. `waterfall.png` differs in two ways that matter for
inference: the junk pings are cropped out, and the contrast stretch is
measured over good rows only — the same limits the tiles use, so both render
identically (verified pixel-exact).

Because it is cropped, its row 0 is not ping 0. `waterfall.json` records
`row_offset` and `locate_waterfall_pixel` applies it:

```python
fix  = locate_waterfall_pixel(x, y, "out_square")     # same dict as tiles
dims = bbox_to_metres(fix, (x, y, w, h), shadow_len_px=...)
```

## Stage 3 — tile

`tile_waterfall` writes `<tile_id>.png` plus a `<tile_id>.json` sidecar
carrying `row_offset`, `col_offset`, `ping_start`, `ping_end`,
`meters_per_pixel` and `source_xtf_filename`, and a `tile_index.json` +
`ping_table.csv` for the whole survey.

`locate_tile_pixel(tile_id, x, y, tile_dir=...)` inverts all of it, returning
`full_row`, `full_col`, `ping_start`, `ping_end`, `lat`, `lon`, `heading`,
`altitude`, `slant_range` and `meters_per_pixel`.

**`lat`/`lon` are the seabed position of that pixel, not the towfish
position.** A pixel at the swath edge is up to a full swath width from the
track line, so the ping's own fix is offset by the pixel's ground range,
perpendicular to the heading, on the side the pixel falls. The towfish
position is returned separately as `ship_lat`/`ship_lon`.

## Two scales

Across track a pixel is a resampled range bin; along track it is one ping,
however far the fish moved. On the test file that is 0.286 m vs 0.756 m — a
2.6x difference — so `bbox_to_metres` uses the along-track scale for
`length_m` and the across-track scale for `width_m`. `height_m` comes from the
shadow: an object of height `h` at slant range `R` under a fish at altitude
`a` casts a shadow `L = h·R/a`.

Pass `square_pixels=True` to `preprocess_xtf` to resample rows so both scales
match; the ping table is resampled with the image, so `row i ↔ table row i`
still holds and `ping_start`/`ping_end` still name real pings.

## Verifying

```bash
python run_pipeline.py            # full run + summary
python verify_stage3.py           # tiling + locate_tile_pixel checks
```

`verify_stage3.py` asserts that every tile is pixel-identical to the full
image it was cut from, and hand-checks all 13 fields `locate_tile_pixel`
returns against values computed independently.
