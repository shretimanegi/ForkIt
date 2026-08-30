"""End-to-end run of the preprocessing pipeline on one XTF file."""
import sys, time
from core.preprocess import preprocess_xtf

XTF = sys.argv[1] if len(sys.argv) > 1 else "/Users/jaswant/sensor-debris/data/NBP050504A.XTF"

t = time.time()
res = preprocess_xtf(XTF, out_dir="out", tile_size=640, overlap=0.2)
print(f"pipeline finished in {time.time()-t:.1f}s\n")

tab, idx = res["ping_table"], res["tile_index"]
print(f"channels        : port={res['channels'].port_name} stbd={res['channels'].stbd_name} "
      f"@ {res['channels'].frequency_khz:.0f} kHz")
print(f"pings           : {len(tab)}   bad-flagged: {int(tab['bad_ping'].sum())}")
print(f"cleaned image   : {res['cleaned'].shape}")
print(f"m/px acrosstrack: {res['meters_per_pixel']:.4f}")
print(f"m/px alongtrack : {tab['meters_per_pixel_alongtrack'].median():.4f}")
print(f"altitude (m)    : {tab['altitude_m'].min():.1f} .. {tab['altitude_m'].max():.1f}")
print(f"tiles written   : {len(idx.tiles)} -> {res['tile_dir']}")
print("\nPNGs:")
for k, v in res["pngs"].items():
    print(f"  {k:28s} {v}")
