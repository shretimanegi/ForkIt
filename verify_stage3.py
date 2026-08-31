"""Independent verification of the tiling and backtracking metadata."""
import json, os
import numpy as np, cv2
from core.preprocess import (preprocess_xtf, load_tile_index, locate_tile_pixel,
                             bbox_to_metres, display_limits, _apply_limits)

import sys
if len(sys.argv) < 2:
    sys.exit("Pass an XTF path: python3 verify_stage3.py /path/to/file.xtf")
XTF = sys.argv[1]
res = preprocess_xtf(XTF, out_dir="out", tile_size=640, overlap=0.2)
cleaned, tab, idx = res["cleaned"], res["ping_table"], res["tile_index"]
TD = res["tile_dir"]

print("=" * 72)
print("A. TILE GRID")
print("=" * 72)
print(f"image {cleaned.shape}, tile 640, overlap 0.2 -> stride 512")
print(f"{len(idx.tiles)} tiles kept")
rows = sorted({t['row_offset'] for t in idx.tiles}); cols = sorted({t['col_offset'] for t in idx.tiles})
print(f"row_offsets {rows}\ncol_offsets {cols}")
ov = [rows[i+1]-rows[i] for i in range(len(rows)-1)]
print(f"row strides {ov}  -> overlap {[640-o for o in ov]} px "
      f"({[round((640-o)/640*100) for o in ov]}%)")

print("\n" + "=" * 72)
print("B. SIDECAR JSON (one tile)")
print("=" * 72)
tid = idx.tiles[len(idx.tiles)//2]["tile_id"]
print(json.dumps(json.load(open(f"{TD}/{tid}.json")), indent=2))

print("\n" + "=" * 72)
print("C. ROUND-TRIP: does the tile PNG really hold the pixels it claims?")
print("=" * 72)
# The tiler stretches with limits measured over good pings only, so the
# reference rendering here has to use the same mask to compare like with like.
lo, hi = display_limits(cleaned, row_mask=~tab["bad_ping"].to_numpy(bool))
full_u8 = _apply_limits(cleaned, lo, hi)
worst = 0
for t in idx.tiles:
    png = cv2.imread(os.path.join(TD, t["png"]), cv2.IMREAD_GRAYSCALE)
    r, c, s = t["row_offset"], t["col_offset"], t["tile_size"]
    assert png.shape == (s, s), (t["tile_id"], png.shape)
    worst = max(worst, int(np.abs(png.astype(int) - full_u8[r:r+s, c:c+s].astype(int)).max()))
assert worst == 0, f"tile pixels disagree with the full image (max diff {worst})"
print(f"all {len(idx.tiles)} tiles match the full image exactly "
      f"(max abs pixel diff = {worst})")

print("\n" + "=" * 72)
print("D. locate_tile_pixel() vs hand-computed values")
print("=" * 72)
index, table = load_tile_index(TD)          # reload from disk, as FastAPI would
TILE, X, Y = tid, 400, 123
fix = locate_tile_pixel(TILE, X, Y, tile_dir=TD)
meta = index.get(TILE)

exp_row = meta["row_offset"] + Y
exp_col = meta["col_offset"] + X
ping = tab.iloc[exp_row]
half = cleaned.shape[1] / 2.0
exp_side = "port" if exp_col < half else "stbd"
exp_gr = abs(exp_col + 0.5 - half) * res["meters_per_pixel"]

checks = [
    ("full_row",        fix["full_row"],       exp_row),
    ("full_col",        fix["full_col"],       exp_col),
    ("ping_start",      fix["ping_start"],     int(ping["ping_index"])),
    ("ping_end",        fix["ping_end"],       int(ping["ping_index"])),
    ("ping_number",     fix["ping_number"],    int(ping["ping_number"])),
    ("ship_lat",        fix["ship_lat"],       float(ping["lat"])),
    ("ship_lon",        fix["ship_lon"],       float(ping["lon"])),
    ("heading",         fix["heading"],        float(ping["heading"])),
    ("altitude",        fix["altitude"],       float(ping["altitude_m"])),
    ("slant_range",     fix["slant_range"],    float(ping["slant_range"])),
    ("side",            fix["side"],           exp_side),
    ("ground_range_m",  fix["ground_range_m"], exp_gr),
    ("m/px across",     fix["meters_per_pixel"], res["meters_per_pixel"]),
]
print(f"tile={TILE}  x={X}  y={Y}\n")
print(f"{'field':22}{'returned':>26}{'expected':>26}  ok")
allok = True
for name, got, exp in checks:
    ok = (got == exp) if isinstance(exp, str) else abs(float(got)-float(exp)) < 1e-6
    allok &= ok
    print(f"{name:22}{str(got):>26}{str(exp):>26}  {'OK' if ok else 'MISMATCH'}")
print(f"\nall hand-checks pass: {allok}")

print("\n" + "=" * 72)
print("E. GEOREFERENCING SANITY")
print("=" * 72)
import math
dlat = fix["lat"]-fix["ship_lat"]; dlon = fix["lon"]-fix["ship_lon"]
d = math.hypot(dlat*110540.0, dlon*111320.0*math.cos(math.radians(fix["lat"])))
brg = (math.degrees(math.atan2(dlon*111320.0*math.cos(math.radians(fix["lat"])),
                               dlat*110540.0)) + 360) % 360
print(f"ship  lat/lon : {fix['ship_lat']:.6f}, {fix['ship_lon']:.6f}")
print(f"pixel lat/lon : {fix['lat']:.6f}, {fix['lon']:.6f}")
print(f"offset        : {d:.2f} m  (ground_range_m = {fix['ground_range_m']:.2f} m)")
print(f"bearing       : {brg:.1f} deg   heading = {fix['heading']:.1f} deg, "
      f"side = {fix['side']}  (expect heading{'+' if fix['side']=='stbd' else '-'}90 "
      f"= {(fix['heading']+(90 if fix['side']=='stbd' else -90))%360:.1f})")

print("\n" + "=" * 72)
print("F. bbox -> report schema fields")
print("=" * 72)
bbox = (X-20, Y-14, 40, 28)   # x, y, w, h in tile pixels
m = bbox_to_metres(fix, bbox, shadow_len_px=35)
print(f"bbox_px = {bbox}, shadow_len_px = 35")
for k, v in m.items():
    print(f"  {k:15} {v:.3f}")
print(f"\n  -> report row: lat={fix['lat']:.6f} lon={fix['lon']:.6f} "
      f"ping_start={fix['ping_start']} ping_end={fix['ping_end']}")
