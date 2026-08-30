"""
Side-scan sonar (XTF) preprocessing pipeline for YOLO-based debris detection.

The module is organised in three stages, each built from small importable
functions so a FastAPI backend can call them individually:

  Stage 1  read_xtf / diagnose_channels / select_channel_pair
           extract_ping_metadata / build_side_images / build_raw_waterfall
  Stage 2  track_bottom -> remove_nadir -> slant_to_ground_range
           -> normalize_gain -> despeckle_lee
  Stage 3  tile_waterfall / locate_tile_pixel

Geometry conventions
--------------------
* A "waterfall" is a 2-D array, one row per ping, columns running across
  track:  ``[ port far ... port near | stbd near ... stbd far ]``.
  The nadir (directly beneath the fish) is the vertical centre line.
* Row ``i`` of every waterfall corresponds to row ``i`` of the ping metadata
  table, which is ordered chronologically (row 0 = earliest ping).
  ``pyxtf.concatenate_channel`` internally reverses the ping order, so
  :func:`build_side_images` flips the result back.
* Internally the two sides are carried in a :class:`SideImages` container with
  both sides oriented **nadir-first** (column 0 = zero slant range).  Only
  :meth:`SideImages.assemble` mirrors the port side for display, so the
  mirroring logic lives in exactly one place.
* ``meters_per_pixel`` is only meaningful after
  :func:`slant_to_ground_range`; before that a column is a constant *slant*
  range increment, not a constant ground distance.
"""

from __future__ import annotations

import json
import os
import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import pandas as pd
import pyxtf

__all__ = [
    # stage 1
    "PingChannels", "ChannelDiagnostic", "SideImages",
    "read_xtf", "diagnose_channels", "select_channel_pair",
    "extract_ping_metadata", "build_side_images", "build_raw_waterfall",
    # stage 2
    "track_bottom", "remove_nadir", "slant_to_ground_range",
    "normalize_gain", "despeckle_lee", "flag_bad_pings",
    "add_alongtrack_scale", "resample_alongtrack",
    # stage 3
    "TileIndex", "tile_waterfall", "load_tile_index", "locate_tile_pixel",
    "bbox_to_metres", "save_inference_waterfall", "locate_waterfall_pixel",
    # orchestration
    "preprocess_xtf",
    # helpers
    "to_display_uint8", "save_png", "display_limits",
]


# ==========================================================================
# Containers
# ==========================================================================

@dataclass
class PingChannels:
    """Which XTF channel index carries the port and starboard imagery."""

    port: int
    stbd: int
    port_name: str = ""
    stbd_name: str = ""
    frequency_khz: float = 0.0
    port_nadir_at: str = "start"   # "start" or "end" -- how samples are stored
    stbd_nadir_at: str = "start"


@dataclass
class ChannelDiagnostic:
    """Per-channel data-quality summary produced by :func:`diagnose_channels`."""

    index: int
    name: str
    kind: str                 # "port" / "stbd" / "other"
    frequency_khz: float
    nadir_at: str             # inferred sample storage direction
    bottom_sample: float      # median first-bottom-return sample (nadir-first)
    altitude_corr: float      # corr. of detected bottom vs header altitude
    saturation: float         # 0..1, how pinned the profile is at its hot end
    usable: bool
    score: float
    note: str = ""


@dataclass
class SideImages:
    """The port and starboard halves of a survey, both oriented nadir-first.

    ``port`` and ``stbd`` are ``(n_pings, n_cols)`` float32 arrays where column
    0 is the nadir.  ``meters_per_pixel`` is populated by
    :func:`slant_to_ground_range`; before that it is ``None`` and
    ``range_per_sample`` (metres of *slant* range per column) applies instead.
    """

    port: np.ndarray
    stbd: np.ndarray
    range_per_sample: float
    meters_per_pixel: Optional[float] = None
    stage: str = "raw"
    history: List[str] = field(default_factory=list)

    @property
    def n_pings(self) -> int:
        return int(self.port.shape[0])

    @property
    def half_width(self) -> int:
        return int(self.port.shape[1])

    def assemble(self) -> np.ndarray:
        """Join the two sides into a display waterfall (port mirrored)."""
        return np.concatenate([self.port[:, ::-1], self.stbd], axis=1)

    def replaced(self, port: np.ndarray, stbd: np.ndarray, stage: str,
                 **kw: Any) -> "SideImages":
        """Return a copy carrying new pixel data and an extended history."""
        return SideImages(
            port=port, stbd=stbd,
            range_per_sample=kw.pop("range_per_sample", self.range_per_sample),
            meters_per_pixel=kw.pop("meters_per_pixel", self.meters_per_pixel),
            stage=stage, history=self.history + [stage],
        )


# ==========================================================================
# Stage 1 -- read the XTF file
# ==========================================================================

def read_xtf(path: str) -> Tuple[Any, List[Any]]:
    """Read an XTF file and return ``(file_header, sonar_pings)``.

    Pings are returned sorted chronologically (index 0 = earliest ping); this
    is the canonical ordering for every array and table the pipeline produces.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(path)

    file_header, packets = pyxtf.xtf_read(path)
    pings = packets.get(pyxtf.XTFHeaderType.sonar, [])
    if not pings:
        raise ValueError(f"no sonar ping packets found in {path}")

    pings = sorted(pings, key=pyxtf.XTFPingHeader.get_time)
    return file_header, pings


def _channel_names(file_header: Any) -> List[Tuple[int, str, str, float]]:
    """``[(index, name, kind, frequency_khz), ...]`` for every sonar channel."""
    out = []
    for idx, info in enumerate(file_header.sonar_info):
        name = info.ChannelName.decode(errors="replace").strip("\x00 ").strip()
        if info.TypeOfChannel == pyxtf.XTFChannelType.port.value:
            kind = "port"
        elif info.TypeOfChannel == pyxtf.XTFChannelType.stbd.value:
            kind = "stbd"
        else:
            kind = "other"
        out.append((idx, name, kind, float(info.Frequency)))
    return out


def _first_bottom_return(profile: np.ndarray, min_sample: int = 4,
                         smooth: int = 9, frac: float = 0.18,
                         run: int = 4) -> int:
    """Index of the first sustained strong echo in a nadir-first ping profile.

    The water column is quiet; the seabed arrives as a step up.  We smooth,
    threshold at a fraction of the ping's robust maximum, and require the
    signal to *stay* above threshold for ``run`` samples so a single noise
    spike in the water column cannot trigger a false bottom.
    Returns ``-1`` when no bottom is found.
    """
    s = np.asarray(profile, dtype=np.float32)
    if s.size <= min_sample + run:
        return -1
    k = max(1, int(smooth))
    sm = np.convolve(s, np.ones(k, np.float32) / k, mode="same")

    hi = float(np.percentile(sm, 99))
    if hi <= 0:
        return -1
    thr = frac * hi

    above = sm > thr
    # require `run` consecutive samples above threshold
    kernel = np.ones(run, np.int32)
    sustained = np.convolve(above.astype(np.int32), kernel, mode="valid") == run
    idx = np.flatnonzero(sustained)
    idx = idx[idx >= min_sample]
    return int(idx[0]) if idx.size else -1


def diagnose_channels(file_header: Any, pings: Sequence[Any],
                      max_pings: int = 200) -> List[ChannelDiagnostic]:
    """Inspect every sonar channel and report orientation and usability.

    Two things vary between files and even between channels of the same file:

    * **Storage direction.**  Some channels store samples nadir-first, others
      far-range-first.  Inferred from the location of the strongest rising
      edge in the median ping profile, cross-checked (when the header carries
      a real altitude) against which direction correlates with altitude.
    * **Usability.**  A channel whose profile is pinned at one end carries no
      usable dynamic range (a dead or badly gain-staged transducer).

    Returns one :class:`ChannelDiagnostic` per channel, best first.
    """
    pings = list(pings)
    step = max(1, len(pings) // max_pings)
    sub = pings[::step]

    header_alt = np.array([float(p.SensorPrimaryAltitude) for p in sub])
    alt_ok = np.isfinite(header_alt) & (header_alt > 1.0)
    use_alt = alt_ok.sum() >= max(10, 0.5 * len(sub))

    diags: List[ChannelDiagnostic] = []
    for idx, name, kind, freq in _channel_names(file_header):
        try:
            stack = np.stack([p.data[idx].astype(np.float32) for p in sub])
        except (IndexError, ValueError):
            continue

        prof = np.median(stack, axis=0)
        n = prof.size
        k = max(1, n // 64)
        sm = np.convolve(prof, np.ones(k, np.float32) / k, mode="same")

        # --- storage direction -------------------------------------------
        grad_idx = int(np.argmax(np.gradient(sm)))
        nadir_at = "start" if grad_idx < n / 2 else "end"

        corr = float("nan")
        if use_alt:
            rps = _range_per_sample_for(sub[0], idx)
            fwd, rev = [], []
            for p in sub:
                d = p.data[idx].astype(np.float32)
                fwd.append(_first_bottom_return(d))
                rev.append(_first_bottom_return(d[::-1]))
            fwd = np.asarray(fwd, float)
            rev = np.asarray(rev, float)
            expect = header_alt / max(rps, 1e-9)
            m = alt_ok & (fwd >= 0) & (rev >= 0)
            if m.sum() >= 10:
                cf = _safe_corr(fwd[m], expect[m])
                cr = _safe_corr(rev[m], expect[m])
                if np.isfinite(cf) or np.isfinite(cr):
                    nadir_at = "start" if np.nan_to_num(cf, nan=-2) >= \
                        np.nan_to_num(cr, nan=-2) else "end"
                    corr = max(np.nan_to_num(cf, nan=-2),
                               np.nan_to_num(cr, nan=-2))

        oriented = prof if nadir_at == "start" else prof[::-1]
        bottom = _first_bottom_return(oriented)

        # --- usability ----------------------------------------------------
        p99 = max(float(np.percentile(prof, 99)), 1e-6)
        sat = max(float(np.median(prof[:3])), float(np.median(prof[-3:]))) / p99
        usable = True
        note = ""
        if sat > 0.5:
            usable, note = False, "profile saturated at one end (no usable falloff)"
        elif bottom < 0:
            usable, note = False, "no bottom return detected"
        elif bottom <= 2:
            usable, note = False, "bottom at sample 0 (near-range saturation)"

        score = (1.0 if usable else 0.0) + (0.0 if np.isnan(corr) else max(corr, 0.0)) \
            + (1.0 - min(sat, 1.0))
        diags.append(ChannelDiagnostic(
            index=idx, name=name, kind=kind, frequency_khz=freq,
            nadir_at=nadir_at, bottom_sample=float(bottom),
            altitude_corr=corr, saturation=float(sat), usable=usable,
            score=float(score), note=note,
        ))

    diags.sort(key=lambda d: -d.score)
    return diags


def _safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    if a.size < 3 or np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def _range_per_sample_for(ping: Any, channel: int) -> float:
    ch = ping.ping_chan_headers[channel]
    n = int(ch.NumSamples) or len(ping.data[channel])
    return float(ch.SlantRange) / max(n, 1)


def select_channel_pair(file_header: Any, pings: Optional[Sequence[Any]] = None,
                        prefer: str = "auto") -> PingChannels:
    """Choose the port/starboard channel pair to image.

    ``prefer``
        ``"auto"``  best-scoring usable pair (needs ``pings``; recommended --
        a nominally better high-frequency channel is worthless if its
        transducer is saturated, which :func:`diagnose_channels` detects)
        ``"high"`` / ``"low"``  highest / lowest frequency pair
        ``(port_idx, stbd_idx)``  explicit indices
    """
    chans = _channel_names(file_header)
    ports = [c for c in chans if c[2] == "port"]
    stbds = [c for c in chans if c[2] == "stbd"]
    if not ports or not stbds:
        raise ValueError(
            f"expected both a port and a starboard channel, found "
            f"port={ports} stbd={stbds}"
        )

    diags = {d.index: d for d in diagnose_channels(file_header, pings)} \
        if pings is not None else {}

    if isinstance(prefer, (tuple, list)):
        p_idx, s_idx = int(prefer[0]), int(prefer[1])
        p = next(c for c in chans if c[0] == p_idx)
        s = next(c for c in chans if c[0] == s_idx)
    elif prefer == "auto":
        if not diags:
            raise ValueError("prefer='auto' requires the pings argument")
        # Prefer a pair whose two channels are both usable and share a frequency.
        def pair_score(pc, sc):
            dp, ds = diags.get(pc[0]), diags.get(sc[0])
            if dp is None or ds is None:
                return -1e9
            same_freq = 1.0 if abs(pc[3] - sc[3]) < 1e-6 else 0.0
            return dp.score + ds.score + same_freq + 1e-3 * pc[3]
        p, s = max(((pc, sc) for pc in ports for sc in stbds),
                   key=lambda t: pair_score(*t))
        if pair_score(p, s) <= -1e8:
            raise ValueError("no usable port/starboard pair found")
    elif prefer in ("high", "low"):
        rev = prefer == "high"
        p = sorted(ports, key=lambda c: c[3], reverse=rev)[0]
        s = sorted(stbds, key=lambda c: c[3], reverse=rev)[0]
    else:
        raise ValueError(f"unknown prefer={prefer!r}")

    return PingChannels(
        port=p[0], stbd=s[0], port_name=p[1], stbd_name=s[1],
        frequency_khz=p[3],
        port_nadir_at=diags[p[0]].nadir_at if p[0] in diags else "start",
        stbd_nadir_at=diags[s[0]].nadir_at if s[0] in diags else "start",
    )


def extract_ping_metadata(pings: Sequence[Any], channels: PingChannels,
                          source_xtf_filename: str = "") -> pd.DataFrame:
    """Build the ping metadata table the rest of the pipeline depends on.

    One row per ping, in the same order as the waterfall rows.  Columns:

    ``ping_index``    position of the ping in the chronologically sorted file
    ``ping_number``   the sonar's own PingNumber field
    ``timestamp``     ping time
    ``lat`` / ``lon`` SensorYcoordinate / SensorXcoordinate
    ``heading``       SensorHeading, degrees
    ``altitude``      SensorPrimaryAltitude, metres, as recorded
    ``slant_range``   SlantRange from ping_chan_headers, metres
    ``n_samples_*``   samples per side for this ping

    Stage 2's :func:`track_bottom` appends an ``altitude_m`` column measured
    from the imagery itself; that is the one the geometry uses, because the
    recorded altitude is missing or wrong on plenty of surveys.
    """
    rows: List[Dict[str, Any]] = []
    for i, ping in enumerate(pings):
        hdrs = ping.ping_chan_headers
        ch_p = hdrs[channels.port]
        ch_s = hdrs[channels.stbd]
        rows.append({
            "ping_index": i,
            "ping_number": int(ping.PingNumber),
            "timestamp": pd.Timestamp(ping.get_time()),
            "lat": float(ping.SensorYcoordinate),
            "lon": float(ping.SensorXcoordinate),
            "heading": float(ping.SensorHeading),
            "altitude": float(ping.SensorPrimaryAltitude),
            "sensor_depth": float(ping.SensorDepth),
            "slant_range": float(ch_s.SlantRange or ch_p.SlantRange),
            "slant_range_port": float(ch_p.SlantRange),
            "slant_range_stbd": float(ch_s.SlantRange),
            "n_samples_port": int(ch_p.NumSamples),
            "n_samples_stbd": int(ch_s.NumSamples),
        })

    df = pd.DataFrame(rows)
    df.attrs["source_xtf_filename"] = source_xtf_filename
    df.attrs["port_channel"] = channels.port
    df.attrs["stbd_channel"] = channels.stbd
    df.attrs["frequency_khz"] = channels.frequency_khz
    return df


def build_side_images(file_header: Any, pings: Sequence[Any],
                      channels: PingChannels) -> SideImages:
    """Stack both channels into a :class:`SideImages`, both sides nadir-first.

    Uses :func:`pyxtf.concatenate_channel` per side.  That helper emits rows in
    reverse chronological order, so both sides are flipped back; the result is
    row-aligned with :func:`extract_ping_metadata`.  Channels stored
    far-range-first (see :func:`diagnose_channels`) are reversed so that
    column 0 is the nadir on both sides.
    """
    pings = list(pings)
    port = np.flipud(pyxtf.concatenate_channel(
        pings, file_header, channel=channels.port)).astype(np.float32)
    stbd = np.flipud(pyxtf.concatenate_channel(
        pings, file_header, channel=channels.stbd)).astype(np.float32)

    if channels.port_nadir_at == "end":
        port = port[:, ::-1]
    if channels.stbd_nadir_at == "end":
        stbd = stbd[:, ::-1]

    w = min(port.shape[1], stbd.shape[1])
    port, stbd = np.ascontiguousarray(port[:, :w]), np.ascontiguousarray(stbd[:, :w])

    rps = _range_per_sample_for(pings[0], channels.stbd)
    return SideImages(port=port, stbd=stbd, range_per_sample=rps,
                      stage="raw", history=["raw"])


def build_raw_waterfall(file_header: Any, pings: Sequence[Any],
                        channels: PingChannels) -> np.ndarray:
    """Convenience wrapper: the raw display waterfall as a single array."""
    return build_side_images(file_header, pings, channels).assemble()


# ==========================================================================
# Display helpers
# ==========================================================================

def to_display_uint8(img: np.ndarray, lo_pct: float = 2.0,
                     hi_pct: float = 98.0) -> np.ndarray:
    """Percentile-stretch a float image to uint8 for viewing/saving.

    Zero samples (channel padding and the blanked nadir band) are excluded
    from the percentile estimate so the stretch is driven by real returns.
    """
    a = np.asarray(img, dtype=np.float32)
    finite = a[np.isfinite(a)]
    valid = finite[finite > 0]
    if valid.size == 0:
        valid = finite
    if valid.size == 0:
        return np.zeros(a.shape, np.uint8)
    lo, hi = np.percentile(valid, [lo_pct, hi_pct])
    if hi <= lo:
        hi = lo + 1.0
    out = (np.nan_to_num(a, nan=lo) - lo) / (hi - lo)
    return (np.clip(out, 0.0, 1.0) * 255.0).astype(np.uint8)


def save_png(img: np.ndarray, path: str, stretch: bool = True) -> str:
    """Write an image to ``path`` as PNG, creating parent directories."""
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    out = to_display_uint8(img) if stretch else np.asarray(img, np.uint8)
    if not cv2.imwrite(path, out):
        raise IOError(f"failed to write {path}")
    return path


# ==========================================================================
# Stage 2 -- clean the image
# ==========================================================================

def _interp_nan(x: np.ndarray) -> np.ndarray:
    """Fill NaNs in a 1-D array by linear interpolation over their neighbours."""
    x = np.asarray(x, dtype=np.float64).copy()
    bad = ~np.isfinite(x)
    if bad.all():
        return np.zeros_like(x)
    if bad.any():
        good = ~bad
        x[bad] = np.interp(np.flatnonzero(bad), np.flatnonzero(good), x[good])
    return x


def _median_filter_1d(x: np.ndarray, size: int) -> np.ndarray:
    """Odd-window running median (edges handled by reflection)."""
    size = int(size) | 1
    if size <= 1:
        return x.astype(np.float64)
    pad = size // 2
    padded = np.pad(np.asarray(x, np.float64), pad, mode="reflect")
    windows = np.lib.stride_tricks.sliding_window_view(padded, size)
    return np.median(windows, axis=-1)


def track_bottom(sides: SideImages, ping_table: pd.DataFrame,
                 smooth_pings: int = 31, min_sample: int = 4,
                 frac: float = 0.18) -> pd.DataFrame:
    """Stage 2.1a -- find the first seabed return in every ping.

    For each ping and each side the first sustained strong echo after the
    direct water-column return is located, failures are interpolated over, and
    the track is median-filtered along track (the seabed depth beneath a towed
    fish changes slowly, so a per-ping jitter of tens of samples is detection
    noise, not bathymetry).

    The two sides are then combined with a **minimum**: the water column ends
    at the fish's altitude, which is one physical number shared by both sides,
    and the earliest bottom return is the one that measured it.  This also
    guarantees ``slant_range >= altitude`` for every retained sample, so the
    ground-range conversion never takes the root of a negative number.

    Adds these columns to ``ping_table`` (in place, and returns it):
    ``nadir_sample_port``, ``nadir_sample_stbd``, ``nadir_sample``,
    ``altitude_m``.
    """
    rps = sides.range_per_sample
    tracks = {}
    for name, arr in (("port", sides.port), ("stbd", sides.stbd)):
        raw = np.array([_first_bottom_return(row, min_sample=min_sample,
                                             frac=frac)
                        for row in arr], dtype=np.float64)
        raw[raw < 0] = np.nan
        tracks[name] = _median_filter_1d(_interp_nan(raw), smooth_pings)

    combined = np.minimum(tracks["port"], tracks["stbd"])
    combined = _median_filter_1d(combined, smooth_pings)
    combined = np.clip(np.round(combined), 0, sides.half_width - 2).astype(int)

    ping_table["nadir_sample_port"] = np.round(tracks["port"]).astype(int)
    ping_table["nadir_sample_stbd"] = np.round(tracks["stbd"]).astype(int)
    ping_table["nadir_sample"] = combined
    ping_table["altitude_m"] = combined * rps
    return ping_table


def remove_nadir(sides: SideImages, ping_table: pd.DataFrame) -> SideImages:
    """Stage 2.1b -- crop the water-column blind spot out of every ping.

    Each row is shifted left by its own ``nadir_sample`` so that column 0 of
    both sides is the first seabed return.  Rows are ragged (the altitude
    varies along track), so the tail of each row is filled with NaN and
    carried as "no data" through the rest of the pipeline.
    """
    nadir = ping_table["nadir_sample"].to_numpy(int)
    if nadir.size != sides.n_pings:
        raise ValueError("ping_table rows do not match the number of pings")

    n, w = sides.n_pings, sides.half_width
    out = {}
    for name, arr in (("port", sides.port), ("stbd", sides.stbd)):
        dst = np.full((n, w), np.nan, np.float32)
        for i in range(n):
            k = int(nadir[i])
            dst[i, : w - k] = arr[i, k:]
        out[name] = dst

    return sides.replaced(out["port"], out["stbd"], stage="nadir_removed")


def slant_to_ground_range(sides: SideImages, ping_table: pd.DataFrame,
                          half_width: Optional[int] = None,
                          max_ground_range_pct: float = 50.0,
                          max_stretch: float = 8.0,
                          blank_near_nadir_m: float = 0.0) -> SideImages:
    """Stage 2.2 -- project slant range onto the seabed at a constant scale.

    A side-scan sample is timed along the slanted path from fish to seabed.
    With the fish ``a`` metres above a flat bottom, a sample at slant range
    ``r`` actually lies ``sqrt(r^2 - a^2)`` metres horizontally from the
    nadir, so raw side-scan is compressed near the nadir and progressively
    less so further out.  Each ping is resampled onto one shared ground-range
    grid, which makes every column a fixed number of metres wide -- the
    ``meters_per_pixel`` the detection report needs to turn a bounding box
    into a length.

    ``max_ground_range_pct`` picks the grid's outer edge from the percentile
    of per-ping maximum ground range (the default, the median, keeps the grid
    filled for about half the pings and pads the rest rather than throwing
    away swath for every ping to satisfy the deepest one).

    Very close to the nadir the stretch factor ``dg/dr = sqrt(g^2+a^2)/g``
    diverges, so a handful of slant samples get smeared across many ground
    pixels -- the ragged high-contrast band down the middle of an
    uncorrected mosaic is interpolation, not measurement.  ``max_stretch``
    blanks that strip per ping, at the ground range where the stretch exceeds
    it (``g < a / sqrt(T^2 - 1)``); set it to ``0`` to keep everything.
    ``blank_near_nadir_m`` additionally blanks a fixed strip.
    """
    rps = sides.range_per_sample
    nadir = ping_table["nadir_sample"].to_numpy(int)
    alt = ping_table["altitude_m"].to_numpy(float)
    n, w = sides.n_pings, sides.half_width
    W = int(half_width or w)

    # Per-ping maximum ground range, given each ping's own altitude.
    # remove_nadir shifts each row left and pads the tail with NaN, so the
    # last *valid* sample is still the last sample the sonar recorded --
    # slant range (w - 1) * rps, independent of the nadir crop.
    max_slant = (w - 1) * rps
    per_ping_max = np.sqrt(np.maximum(max_slant ** 2 - alt ** 2, 0.0))
    g_max = float(np.percentile(per_ping_max, max_ground_range_pct))
    if not np.isfinite(g_max) or g_max <= 0:
        raise ValueError("could not determine a positive ground-range extent")

    mpp = g_max / W
    grid = np.arange(W, dtype=np.float64) * mpp

    out = {}
    for name, arr in (("port", sides.port), ("stbd", sides.stbd)):
        dst = np.full((n, W), np.nan, np.float32)
        for i in range(n):
            row = arr[i]
            valid = np.flatnonzero(np.isfinite(row))
            if valid.size < 2:
                continue
            j = np.arange(valid[0], valid[-1] + 1)
            slant = (nadir[i] + j) * rps
            ground = np.sqrt(np.maximum(slant ** 2 - alt[i] ** 2, 0.0))
            # `ground` is monotonically increasing, so np.interp applies.
            keep = grid <= ground[-1]
            dst[i, keep] = np.interp(grid[keep], ground,
                                     np.nan_to_num(row[j], nan=0.0))
        if max_stretch and max_stretch > 1.0:
            g_blank = alt / np.sqrt(max_stretch ** 2 - 1.0)
            for i in range(n):
                dst[i, : int(round(g_blank[i] / mpp))] = np.nan
        if blank_near_nadir_m > 0:
            dst[:, : int(round(blank_near_nadir_m / mpp))] = np.nan
        out[name] = dst

    new = sides.replaced(out["port"], out["stbd"], stage="ground_range",
                         meters_per_pixel=mpp)
    ping_table["meters_per_pixel"] = mpp
    ping_table["max_ground_range_m"] = g_max
    if max_stretch and max_stretch > 1.0:
        ping_table["nadir_blank_px"] = np.round(
            alt / np.sqrt(max_stretch ** 2 - 1.0) / mpp).astype(int)
    else:
        ping_table["nadir_blank_px"] = 0
    return new


def normalize_gain(sides: SideImages, smooth_cols: int = 51,
                   equalize_pings: bool = True,
                   ping_smooth: int = 801) -> SideImages:
    """Stage 2.3 -- flatten the across-track brightness falloff.

    Sonar returns fall off steeply with range, so a raw waterfall is a bright
    band near the nadir fading to black at the swath edge -- and a detector
    trained on evenly-lit tiles sees that gradient rather than the targets.
    The average intensity profile of the whole survey (a robust per-column
    median, so bright targets do not drag it) is the empirical falloff; each
    column is divided by it.

    ``equalize_pings`` additionally divides each ping by its own median
    relative to a long running median of its neighbours.  This removes two
    things: the individual hot or dropped pings that show up as horizontal
    streaks, and the multi-hundred-ping blocks of raised gain left behind when
    an operator changed a setting mid-line.  ``ping_smooth`` must be wider
    than about twice the longest such block to flatten it -- but the wider it
    is, the more genuine along-track brightness change (a change of sediment)
    it also removes, so it is the knob to back off if real contrast is lost.
    """
    out = {}
    for name, arr in (("port", sides.port), ("stbd", sides.stbd)):
        a = arr.astype(np.float32).copy()
        a[a <= 0] = np.nan

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            profile = np.nanmedian(a, axis=0)
        profile = _interp_nan(profile)
        profile = _median_filter_1d(profile, smooth_cols)
        profile[profile <= 0] = np.nan
        profile = _interp_nan(profile)
        profile = np.maximum(profile, 1e-6)

        a = a / profile[None, :]

        if equalize_pings:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                per_ping = np.nanmedian(a, axis=1)
            per_ping = _interp_nan(per_ping)
            # Only correct each ping's departure from its local neighbourhood.
            baseline = _median_filter_1d(per_ping, ping_smooth)
            ratio = np.where(baseline > 0, per_ping / np.maximum(baseline, 1e-6), 1.0)
            ratio = np.clip(_interp_nan(ratio), 0.2, 5.0)
            a = a / ratio[:, None]

        out[name] = a.astype(np.float32)

    # Put both sides back on a common scale so the seam is invisible.
    both = np.concatenate([out["port"].ravel(), out["stbd"].ravel()])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        ref = np.nanmedian(both)
    if np.isfinite(ref) and ref > 0:
        for name in out:
            out[name] = (out[name] / ref).astype(np.float32)

    return sides.replaced(out["port"], out["stbd"], stage="gain_normalized")


def _lee_filter(img: np.ndarray, size: int = 7,
                noise_pct: float = 25.0) -> np.ndarray:
    """Lee filter for additive noise, NaN-safe.

    ``out = mean + k * (img - mean)`` with ``k = max(var - nv, 0) / var``:
    flat areas (``var ~ nv``) collapse to the local mean, while edges and
    targets (``var >> nv``) are passed through almost untouched.  That is what
    separates it from a blur.
    """
    a = np.asarray(img, np.float32)
    mask = np.isfinite(a).astype(np.float32)
    filled = np.nan_to_num(a, nan=0.0).astype(np.float32)

    k = (int(size) | 1, int(size) | 1)
    wsum = cv2.boxFilter(mask, -1, k, normalize=False, borderType=cv2.BORDER_REFLECT)
    wsum = np.maximum(wsum, 1e-6)
    mean = cv2.boxFilter(filled, -1, k, normalize=False,
                         borderType=cv2.BORDER_REFLECT) / wsum
    sq = cv2.boxFilter(filled * filled, -1, k, normalize=False,
                       borderType=cv2.BORDER_REFLECT) / wsum
    var = np.maximum(sq - mean * mean, 0.0)

    v = var[np.isfinite(a) & (wsum > k[0])]
    noise_var = float(np.percentile(v, noise_pct)) if v.size else 0.0

    weight = np.where(var > 0, np.maximum(var - noise_var, 0.0) / np.maximum(var, 1e-12), 0.0)
    out = mean + weight * (filled - mean)
    return np.where(np.isfinite(a), out, np.nan).astype(np.float32)


def despeckle_lee(sides: SideImages, size: int = 7, noise_pct: float = 25.0,
                  return_linear: bool = False) -> SideImages:
    """Stage 2.4 -- suppress speckle without blurring real edges.

    Sonar speckle is multiplicative: the grain grows with the signal, so a
    filter tuned for the dark far range over-smooths the bright near range and
    vice versa.  A log transform turns that multiplicative noise into additive
    noise of roughly constant variance, which is the model the Lee filter
    assumes, so the filter is applied in the log domain.

    The result is returned in the log domain by default.  That is deliberate:
    log-compressed sonar is both the conventional display form and the better
    input for a detector, since it stops a single bright specular return from
    consuming the whole dynamic range.  Pass ``return_linear=True`` to undo it.
    """
    out = {}
    for name, arr in (("port", sides.port), ("stbd", sides.stbd)):
        a = np.asarray(arr, np.float32)
        logd = np.log1p(np.maximum(a, 0.0))
        filt = _lee_filter(logd, size=size, noise_pct=noise_pct)
        out[name] = (np.expm1(filt) if return_linear else filt).astype(np.float32)

    stage = "despeckled_linear" if return_linear else "despeckled_log"
    return sides.replaced(out["port"], out["stbd"], stage=stage)


def flag_bad_pings(sides: SideImages, ping_table: pd.DataFrame,
                   z: float = 4.0) -> pd.DataFrame:
    """Mark pings whose overall level is a wild outlier.

    Every survey line contains a few dropped or saturated pings -- the hard
    white and black streaks across a waterfall.  They survive cleaning as
    strong horizontal edges, which a detector will happily fire on, so they
    are flagged here and the tiler can refuse tiles built mostly from them.
    Outliers are measured against a median/MAD baseline, which the outliers
    themselves cannot drag around the way a mean and standard deviation could.

    Adds a boolean ``bad_ping`` column to ``ping_table``.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        level = np.nanmedian(np.concatenate([sides.port, sides.stbd], axis=1), axis=1)
    finite = np.isfinite(level)
    med = float(np.median(level[finite])) if finite.any() else 0.0
    mad = float(np.median(np.abs(level[finite] - med))) if finite.any() else 0.0
    scale = mad * 1.4826 if mad > 0 else (float(np.std(level[finite])) or 1.0)

    bad = ~finite | (np.abs(level - med) > z * scale)
    ping_table["ping_level"] = level
    ping_table["bad_ping"] = bad
    return ping_table


def add_alongtrack_scale(ping_table: pd.DataFrame,
                         window: int = 51) -> pd.DataFrame:
    """Add the along-track metres-per-pixel, from the platform's GPS track.

    Across-track and along-track scales are *different numbers*: across track a
    pixel is a resampled range bin, along track it is however far the fish
    moved between pings.  A detection report that needs both a length and a
    width has to use the right one for each axis, so both are carried.

    The advance is measured between pings ``window`` apart rather than between
    neighbours.  GPS fixes arrive more slowly than pings -- on this data more
    than half of all consecutive pairs share a fix and are exactly 0 m apart --
    so a per-ping difference is mostly zeros and its median is simply 0.
    Measuring across a window and dividing by the number of pings spanned
    recovers the real advance rate.

    Adds ``alongtrack_m`` (advance since the previous ping, smoothed) and
    ``meters_per_pixel_alongtrack`` (the same value: one ping is one row).
    """
    lat = ping_table["lat"].to_numpy(float)
    lon = ping_table["lon"].to_numpy(float)
    n = lat.size
    if n < 2:
        ping_table["alongtrack_m"] = 0.0
        ping_table["meters_per_pixel_alongtrack"] = 0.0
        return ping_table

    coslat = np.cos(np.radians(np.nanmedian(lat)))
    half = max(1, int(window) // 2)
    i = np.arange(n)
    lo = np.clip(i - half, 0, n - 1)
    hi = np.clip(i + half, 0, n - 1)
    span = np.maximum(hi - lo, 1)

    dx = (lon[hi] - lon[lo]) * 111320.0 * coslat
    dy = (lat[hi] - lat[lo]) * 110540.0
    rate = np.sqrt(dx ** 2 + dy ** 2) / span

    good = rate[np.isfinite(rate) & (rate > 0)]
    fallback = float(np.median(good)) if good.size else 0.0
    rate = np.where(np.isfinite(rate) & (rate > 0), rate, fallback)

    ping_table["alongtrack_m"] = rate
    ping_table["meters_per_pixel_alongtrack"] = rate
    return ping_table


def resample_alongtrack(sides: SideImages, ping_table: pd.DataFrame
                        ) -> Tuple[SideImages, pd.DataFrame]:
    """Rescale rows so a pixel is square -- the same metres across and along.

    Across track a pixel is a resampled range bin (here ~0.29 m); along track
    it is one ping, however far the fish moved (here ~0.76 m).  Left alone,
    everything in the image is squashed along track by that ratio, so a round
    object looks like an ellipse and a detector has to learn the distortion.
    Rows are resampled by the ratio of the two scales to remove it.

    ``ping_table`` is resampled with the image and returned alongside, so the
    invariant the rest of the pipeline relies on -- image row ``i`` is table
    row ``i`` -- still holds.  Each new row keeps the ``ping_index`` of the
    ping it came from, so a tile's ``ping_start``/``ping_end`` still name real
    pings; several rows simply share one ping now.

    Returns ``(sides, ping_table)``; both are new objects.
    """
    if sides.meters_per_pixel is None:
        raise ValueError("run slant_to_ground_range first: no meters_per_pixel")
    if "meters_per_pixel_alongtrack" not in ping_table:
        add_alongtrack_scale(ping_table)

    mpp_x = float(sides.meters_per_pixel)
    mpp_y = float(np.nanmedian(ping_table["meters_per_pixel_alongtrack"]))
    if not np.isfinite(mpp_y) or mpp_y <= 0:
        raise ValueError("no usable along-track scale; check the GPS track")

    factor = mpp_y / mpp_x
    n = sides.n_pings
    new_n = max(1, int(round(n * factor)))
    interp = cv2.INTER_AREA if factor < 1.0 else cv2.INTER_LINEAR

    out = {}
    for name, arr in (("port", sides.port), ("stbd", sides.stbd)):
        # NaN is no-data, not a value to interpolate; resample a mask with it
        # and re-punch the holes so the blanked nadir keeps a hard edge.
        mask = np.isfinite(arr).astype(np.float32)
        filled = np.nan_to_num(arr, nan=0.0).astype(np.float32)
        r_img = cv2.resize(filled, (arr.shape[1], new_n), interpolation=interp)
        r_msk = cv2.resize(mask, (arr.shape[1], new_n), interpolation=interp)
        out[name] = np.where(r_msk > 0.5, r_img, np.nan).astype(np.float32)

    src = np.clip(np.round(np.arange(new_n) / factor).astype(int), 0, n - 1)
    table = ping_table.iloc[src].reset_index(drop=True).copy()
    table["source_row"] = src
    table["meters_per_pixel_alongtrack"] = mpp_x

    new_sides = sides.replaced(out["port"], out["stbd"],
                               stage="alongtrack_resampled")
    return new_sides, table


def display_limits(img: np.ndarray, lo_pct: float = 2.0,
                   hi_pct: float = 98.0,
                   row_mask: Optional[np.ndarray] = None) -> Tuple[float, float]:
    """The intensity limits :func:`to_display_uint8` would pick for ``img``.

    Exported so every tile of a survey can be stretched with the *same*
    limits.  Per-tile stretching would make identical seabed look different
    from tile to tile, which is exactly the inconsistency a detector should
    not have to learn around.

    ``row_mask`` selects the rows the percentiles are measured over.  Pass the
    good pings: a block of saturated junk pings drags the upper percentile far
    above anything in the real imagery, and every tile in the survey then
    renders as flat mid-grey to leave headroom for rows no tile contains.
    """
    a = np.asarray(img, np.float32)
    if row_mask is not None:
        row_mask = np.asarray(row_mask, bool)
        if row_mask.shape[0] == a.shape[0] and row_mask.any():
            a = a[row_mask]
    v = a[np.isfinite(a)]
    v = v[v > 0] if (v > 0).any() else v
    if v.size == 0:
        return 0.0, 1.0
    lo, hi = np.percentile(v, [lo_pct, hi_pct])
    return float(lo), float(hi if hi > lo else lo + 1.0)


def _apply_limits(img: np.ndarray, lo: float, hi: float) -> np.ndarray:
    a = np.nan_to_num(np.asarray(img, np.float32), nan=lo, posinf=hi, neginf=lo)
    return (np.clip((a - lo) / (hi - lo), 0.0, 1.0) * 255.0).astype(np.uint8)


# ==========================================================================
# Stage 3 -- tiling with backtracking metadata
# ==========================================================================

@dataclass
class TileIndex:
    """Everything needed to map a tile pixel back to the world.

    Written to ``<tile_dir>/tile_index.json`` next to the tiles, so a
    detection served by a different process (a FastAPI worker, say) can
    resolve coordinates without re-reading the XTF.
    """

    tile_dir: str
    source_xtf_filename: str
    image_shape: Tuple[int, int]
    tile_size: int
    overlap: float
    meters_per_pixel: float
    tiles: List[Dict[str, Any]]
    ping_table_csv: str = "ping_table.csv"

    def to_json(self) -> Dict[str, Any]:
        d = {
            "tile_dir": self.tile_dir,
            "source_xtf_filename": self.source_xtf_filename,
            "image_shape": list(self.image_shape),
            "tile_size": self.tile_size,
            "overlap": self.overlap,
            "meters_per_pixel": self.meters_per_pixel,
            "ping_table_csv": self.ping_table_csv,
            "tiles": self.tiles,
        }
        return d

    def get(self, tile_id: str) -> Dict[str, Any]:
        for t in self.tiles:
            if t["tile_id"] == tile_id:
                return t
        raise KeyError(f"unknown tile_id {tile_id!r}")


def _offsets(total: int, size: int, stride: int, start: int = 0) -> List[int]:
    """Tile origins covering ``[start, total)``, last one clamped to the edge."""
    span = total - start
    if span <= size:
        return [max(0, min(start, total - size))]
    offs = list(range(start, total - size + 1, stride))
    if offs[-1] != total - size:
        offs.append(total - size)
    return offs


def tile_waterfall(image: np.ndarray, ping_table: pd.DataFrame, out_dir: str,
                   source_xtf_filename: str, meters_per_pixel: float,
                   tile_size: int = 640, overlap: float = 0.2,
                   min_valid_frac: float = 0.35,
                   max_bad_ping_frac: float = 0.25,
                   lo_hi: Optional[Tuple[float, float]] = None) -> TileIndex:
    """Slice the cleaned waterfall into overlapping tiles with sidecar JSON.

    ``overlap`` (a fraction of ``tile_size``) keeps a target that straddles a
    tile boundary whole in at least one tile.

    Every tile gets a ``<tile_id>.json`` sidecar carrying ``row_offset``,
    ``col_offset``, ``ping_start``, ``ping_end``, ``meters_per_pixel`` and
    ``source_xtf_filename`` -- enough to invert the tiling for any detection.

    Tiles that are largely no-data (the blanked nadir strip, the ragged far
    edge) or largely flagged pings are skipped: they cannot contain a real
    target, and feeding them to a detector only invites false positives.
    Every sidecar records its own ``valid_fraction`` and ``bad_ping_fraction``,
    so a caller that wants a different threshold can filter without retiling.

    The row grid starts at the first usable ping rather than at row 0.  Survey
    lines routinely open with a block of junk pings while the fish settles; a
    grid anchored at row 0 would put that block in the first tile, fail it on
    ``max_bad_ping_frac``, and take several hundred perfectly good rows down
    with it because no other tile origin reaches them.
    """
    os.makedirs(out_dir, exist_ok=True)
    img = np.asarray(image, np.float32)
    H, W = img.shape
    stride = max(1, int(round(tile_size * (1.0 - overlap))))

    bad = ping_table["bad_ping"].to_numpy(bool) if "bad_ping" in ping_table \
        else np.zeros(H, bool)
    lo, hi = lo_hi if lo_hi is not None else display_limits(img, row_mask=~bad)
    at_mpp = ping_table["meters_per_pixel_alongtrack"].to_numpy(float) \
        if "meters_per_pixel_alongtrack" in ping_table else np.full(H, np.nan)

    usable = np.flatnonzero(~bad)
    row_start = int(usable[0]) if usable.size else 0

    tiles: List[Dict[str, Any]] = []
    for r in _offsets(H, tile_size, stride, start=row_start):
        for c in _offsets(W, tile_size, stride):
            patch = img[r:r + tile_size, c:c + tile_size]
            valid_frac = float(np.isfinite(patch).mean())
            bad_frac = float(bad[r:r + tile_size].mean())
            if valid_frac < min_valid_frac or bad_frac > max_bad_ping_frac:
                continue

            tile_id = f"r{r:06d}_c{c:06d}"
            rows = ping_table.iloc[r:r + tile_size]
            mid = rows.iloc[len(rows) // 2]
            meta = {
                "tile_id": tile_id,
                "png": f"{tile_id}.png",
                "row_offset": int(r),
                "col_offset": int(c),
                "tile_size": int(tile_size),
                "ping_start": int(rows["ping_index"].iloc[0]),
                "ping_end": int(rows["ping_index"].iloc[-1]),
                "ping_number_start": int(rows["ping_number"].iloc[0]),
                "ping_number_end": int(rows["ping_number"].iloc[-1]),
                "meters_per_pixel": float(meters_per_pixel),
                "meters_per_pixel_alongtrack": float(
                    np.nanmedian(at_mpp[r:r + tile_size])),
                "source_xtf_filename": source_xtf_filename,
                "center_lat": float(mid["lat"]),
                "center_lon": float(mid["lon"]),
                "center_heading": float(mid["heading"]),
                "timestamp_start": str(rows["timestamp"].iloc[0]),
                "timestamp_end": str(rows["timestamp"].iloc[-1]),
                "valid_fraction": round(valid_frac, 4),
                "bad_ping_fraction": round(bad_frac, 4),
            }

            cv2.imwrite(os.path.join(out_dir, meta["png"]),
                        _apply_limits(patch, lo, hi))
            with open(os.path.join(out_dir, f"{tile_id}.json"), "w") as fh:
                json.dump(meta, fh, indent=2)
            tiles.append(meta)

    index = TileIndex(
        tile_dir=os.path.abspath(out_dir),
        source_xtf_filename=source_xtf_filename,
        image_shape=(int(H), int(W)), tile_size=int(tile_size),
        overlap=float(overlap), meters_per_pixel=float(meters_per_pixel),
        tiles=tiles,
    )
    ping_table.to_csv(os.path.join(out_dir, index.ping_table_csv), index=False)
    with open(os.path.join(out_dir, "tile_index.json"), "w") as fh:
        json.dump(index.to_json(), fh, indent=2)
    return index


def load_tile_index(tile_dir: str) -> Tuple[TileIndex, pd.DataFrame]:
    """Load ``tile_index.json`` and its ping table from a tile directory."""
    with open(os.path.join(tile_dir, "tile_index.json")) as fh:
        d = json.load(fh)
    index = TileIndex(
        tile_dir=tile_dir, source_xtf_filename=d["source_xtf_filename"],
        image_shape=tuple(d["image_shape"]), tile_size=d["tile_size"],
        overlap=d["overlap"], meters_per_pixel=d["meters_per_pixel"],
        tiles=d["tiles"], ping_table_csv=d.get("ping_table_csv", "ping_table.csv"),
    )
    table = pd.read_csv(os.path.join(tile_dir, index.ping_table_csv))
    return index, table


def _offset_latlon(lat: float, lon: float, bearing_deg: float,
                   distance_m: float) -> Tuple[float, float]:
    """Move ``distance_m`` from (lat, lon) along ``bearing_deg``."""
    b = np.radians(bearing_deg)
    dlat = distance_m * np.cos(b) / 110540.0
    dlon = distance_m * np.sin(b) / (111320.0 * max(np.cos(np.radians(lat)), 1e-6))
    return float(lat + dlat), float(lon + dlon)


def locate_tile_pixel(tile_id: str, x: int, y: int,
                      tile_dir: Optional[str] = None,
                      index: Optional[TileIndex] = None,
                      ping_table: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
    """Map a pixel inside a tile back to its ping, its metadata and its position.

    ``x`` is the column and ``y`` the row *within the tile*.  Pass either
    ``tile_dir`` (the index is loaded from disk) or an already-loaded
    ``index``/``ping_table`` pair.

    The returned ``lat``/``lon`` are the **seabed position of that pixel**, not
    the towfish position: a pixel at the swath edge is up to a full swath
    width away from the track line, so reporting the ping's own fix would put
    every detection on top of the survey line.  The fix is offset by the
    pixel's ground range along the beam, which is perpendicular to the
    heading, to the side the pixel falls on.  The towfish position is returned
    alongside as ``ship_lat``/``ship_lon``.

    Both scales are returned: ``meters_per_pixel`` across track (for the width
    of a box, and for a shadow length measured across track) and
    ``meters_per_pixel_alongtrack`` (for its length).
    """
    if index is None or ping_table is None:
        if tile_dir is None:
            raise ValueError("pass either tile_dir or (index and ping_table)")
        index, ping_table = load_tile_index(tile_dir)

    tile = index.get(tile_id)
    size = tile["tile_size"]
    if not (0 <= x < size and 0 <= y < size):
        raise ValueError(f"({x}, {y}) is outside a {size}x{size} tile")

    full_row = tile["row_offset"] + int(y)
    full_col = tile["col_offset"] + int(x)
    if not (0 <= full_row < index.image_shape[0]):
        raise ValueError(f"row {full_row} is outside the waterfall")

    fix = _locate_full_pixel(
        full_row, full_col, image_width=index.image_shape[1],
        ping_table=ping_table,
        meters_per_pixel=float(tile["meters_per_pixel"]),
        meters_per_pixel_alongtrack=float(tile["meters_per_pixel_alongtrack"]),
        source_xtf_filename=tile["source_xtf_filename"],
    )
    fix.update({
        "tile_id": tile_id, "x": int(x), "y": int(y),
        "tile_ping_start": int(tile["ping_start"]),
        "tile_ping_end": int(tile["ping_end"]),
    })
    return fix


def _locate_full_pixel(full_row: int, full_col: int, image_width: int,
                       ping_table: pd.DataFrame, meters_per_pixel: float,
                       meters_per_pixel_alongtrack: float,
                       source_xtf_filename: str) -> Dict[str, Any]:
    """Resolve a pixel of the full waterfall to its ping and world position."""
    ping = ping_table.iloc[int(full_row)]
    mpp = float(meters_per_pixel)

    # Column -> across-track geometry.  The waterfall is
    # [port far ... port near | stbd near ... stbd far], so the nadir is the
    # centre line and ground range grows outwards from it.
    half = image_width / 2.0
    side = "port" if full_col < half else "stbd"
    ground_range_m = abs(full_col + 0.5 - half) * mpp
    bearing = float(ping["heading"]) + (90.0 if side == "stbd" else -90.0)
    lat, lon = _offset_latlon(float(ping["lat"]), float(ping["lon"]),
                              bearing, ground_range_m)

    return {
        "full_row": int(full_row),
        "full_col": int(full_col),
        # one pixel comes from exactly one ping, so the schema's ping span is
        # degenerate here; a tile's or a box's own span is set by the caller.
        "ping_start": int(ping["ping_index"]),
        "ping_end": int(ping["ping_index"]),
        "ping_number": int(ping["ping_number"]),
        "timestamp": str(ping["timestamp"]),
        "lat": lat,
        "lon": lon,
        "ship_lat": float(ping["lat"]),
        "ship_lon": float(ping["lon"]),
        "heading": float(ping["heading"]),
        "altitude": float(ping.get("altitude_m", ping.get("altitude", np.nan))),
        "altitude_header": float(ping.get("altitude", np.nan)),
        "slant_range": float(ping["slant_range"]),
        "ground_range_m": float(ground_range_m),
        "side": side,
        "meters_per_pixel": mpp,
        "meters_per_pixel_alongtrack": float(meters_per_pixel_alongtrack),
        "source_xtf_filename": source_xtf_filename,
    }


def bbox_to_metres(fix: Dict[str, Any], bbox_px: Sequence[float],
                   shadow_len_px: Optional[float] = None) -> Dict[str, float]:
    """Turn a YOLO box (and optional shadow) into metres, given a pixel fix.

    ``bbox_px`` is ``(x, y, w, h)`` in tile pixels; ``fix`` is the result of
    :func:`locate_tile_pixel` for the box centre.

    ``length_m`` uses the along-track scale and ``width_m`` the across-track
    one, because the two axes of the image are not the same number of metres
    per pixel.

    ``height_m`` comes from the shadow, which is how object height is measured
    on side-scan: an object of height ``h`` standing ``R`` metres away from a
    fish flying at altitude ``a`` casts a shadow of length ``L = h * R / a``,
    so ``h = L * a / R``.  The estimate degrades near the nadir, where ``R``
    approaches ``a`` and shadows get very short.
    """
    _, _, w, h = bbox_px
    mpp_x = float(fix["meters_per_pixel"])
    mpp_y = float(fix["meters_per_pixel_alongtrack"])
    out = {
        "width_m": float(w) * mpp_x,
        "length_m": float(h) * mpp_y,
        "height_m": float("nan"),
        "shadow_len_px": float("nan") if shadow_len_px is None else float(shadow_len_px),
    }
    if shadow_len_px is not None:
        shadow_m = float(shadow_len_px) * mpp_x
        slant = float(np.hypot(fix["ground_range_m"], fix["altitude"]))
        if slant > 0:
            out["height_m"] = shadow_m * float(fix["altitude"]) / slant
    return out


# ==========================================================================
# Orchestration
# ==========================================================================

def preprocess_xtf(xtf_path: str, out_dir: str = "out",
                   prefer_channels: str = "auto",
                   tile_size: int = 640, overlap: float = 0.2,
                   square_pixels: bool = False,
                   save_intermediate: bool = True,
                   tile_subdir: str = "tiles") -> Dict[str, Any]:
    """Run the whole pipeline: XTF in, YOLO-ready tiles plus metadata out.

    Returns a dict with the ping table, the cleaned waterfall, the
    :class:`TileIndex`, and the paths of every PNG written.  Each stage is
    also available as a standalone function if a caller wants to interleave
    its own steps.
    """
    os.makedirs(out_dir, exist_ok=True)
    name = os.path.basename(xtf_path)
    pngs: Dict[str, str] = {}

    def dump(key: str, img: np.ndarray) -> None:
        if save_intermediate or key in ("stage1_raw", "stage2_4_despeckled"):
            pngs[key] = save_png(img, os.path.join(out_dir, f"{key}.png"))

    # --- stage 1 ---------------------------------------------------------
    file_header, pings = read_xtf(xtf_path)
    channels = select_channel_pair(file_header, pings, prefer=prefer_channels)
    ping_table = extract_ping_metadata(pings, channels, name)
    sides = build_side_images(file_header, pings, channels)
    dump("stage1_raw", sides.assemble())

    # --- stage 2 ---------------------------------------------------------
    track_bottom(sides, ping_table)
    sides = remove_nadir(sides, ping_table)
    dump("stage2_1_nadir_removed", sides.assemble())

    sides = slant_to_ground_range(sides, ping_table)
    dump("stage2_2_ground_range", sides.assemble())

    sides = normalize_gain(sides)
    dump("stage2_3_gain_normalized", sides.assemble())

    sides = despeckle_lee(sides)
    cleaned = sides.assemble()
    dump("stage2_4_despeckled", cleaned)

    flag_bad_pings(sides, ping_table)
    add_alongtrack_scale(ping_table)

    if square_pixels:
        sides, ping_table = resample_alongtrack(sides, ping_table)
        cleaned = sides.assemble()
        dump("stage2_5_square_pixels", cleaned)

    # --- stage 3 ---------------------------------------------------------
    waterfall_meta = save_inference_waterfall(
        cleaned, ping_table, out_dir, source_xtf_filename=name,
        meters_per_pixel=float(sides.meters_per_pixel))

    tile_dir = os.path.join(out_dir, tile_subdir)
    index = tile_waterfall(
        cleaned, ping_table, tile_dir, source_xtf_filename=name,
        meters_per_pixel=float(sides.meters_per_pixel),
        tile_size=tile_size, overlap=overlap,
        lo_hi=display_limits(cleaned,
                             row_mask=~ping_table["bad_ping"].to_numpy(bool)),
    )

    return {
        "xtf_path": xtf_path,
        "channels": channels,
        "ping_table": ping_table,
        "sides": sides,
        "cleaned": cleaned,
        "tile_index": index,
        "tile_dir": tile_dir,
        "waterfall": waterfall_meta,
        "waterfall_png": os.path.join(out_dir, waterfall_meta["png"]),
        "pngs": pngs,
        "meters_per_pixel": float(sides.meters_per_pixel),
    }


# ==========================================================================
# Whole-waterfall output, for detectors that take the full image
# ==========================================================================

def save_inference_waterfall(image: np.ndarray, ping_table: pd.DataFrame,
                             out_dir: str, source_xtf_filename: str,
                             meters_per_pixel: float,
                             name: str = "waterfall",
                             trim_bad_pings: bool = True) -> Dict[str, Any]:
    """Write the model-ready waterfall PNG plus the sidecar that inverts it.

    This is *not* the same picture as the ``stage*.png`` files.  Those are
    diagnostics, stretched over the whole array so you can see what each step
    did.  This one is prepared for a detector, which needs two more things:

    * **The junk pings are trimmed.**  A survey line opens with saturated
      pings while the fish settles; left in, they are a band of white a
      detector will happily fire on.
    * **The stretch is measured over the good rows only**, exactly as the
      tiles are, so the same seabed renders the same way in both.  Measured
      over the whole array the junk drags the upper percentile up and
      everything real comes out flat mid-grey.

    Because the image is trimmed, its row 0 is not ping 0.  The sidecar
    records ``row_offset``, and :func:`locate_waterfall_pixel` applies it, so
    a detection still resolves to the right ping.

    Returns the sidecar dict; writes ``<name>.png`` and ``<name>.json``.
    """
    os.makedirs(out_dir, exist_ok=True)
    img = np.asarray(image, np.float32)
    H, W = img.shape

    bad = ping_table["bad_ping"].to_numpy(bool) if "bad_ping" in ping_table \
        else np.zeros(H, bool)

    # Take the longest unbroken run of good pings.  Trimming only the leading
    # and trailing junk is not enough: a settling fish typically emits a few
    # separate bursts of bad pings, so a first-good-to-last-good crop starts
    # in a gap between them and still opens on a band of white.
    interior: List[List[int]] = []
    if trim_bad_pings and (~bad).any():
        edges = np.flatnonzero(np.diff(np.concatenate(([1], bad.astype(int), [1]))) != 0)
        runs = [(int(a), int(b)) for a, b in zip(edges[::2], edges[1::2]) if b > a]
        r0, r1 = max(runs, key=lambda ab: ab[1] - ab[0])
        # If bad pings are scattered rather than clustered at the start, the
        # longest clean run can be a small slice of the line.  Keeping the
        # whole span and reporting the gaps beats silently binning the survey.
        if (r1 - r0) < 0.5 * H:
            good = np.flatnonzero(~bad)
            r0, r1 = int(good[0]), int(good[-1]) + 1
            b = bad[r0:r1]
            e = np.flatnonzero(np.diff(np.concatenate(([0], b.astype(int), [0]))) != 0)
            interior = [[int(a), int(z)] for a, z in zip(e[::2], e[1::2])]
    else:
        r0, r1 = 0, H

    lo, hi = display_limits(img, row_mask=~bad)
    crop = img[r0:r1]
    png_path = os.path.join(out_dir, f"{name}.png")
    if not cv2.imwrite(png_path, _apply_limits(crop, lo, hi)):
        raise IOError(f"failed to write {png_path}")

    rows = ping_table.iloc[r0:r1]
    at = ping_table.get("meters_per_pixel_alongtrack")
    meta = {
        "png": f"{name}.png",
        "source_xtf_filename": source_xtf_filename,
        "row_offset": r0,
        "col_offset": 0,
        "height": int(r1 - r0),
        "width": int(W),
        "full_image_shape": [int(H), int(W)],
        "ping_start": int(rows["ping_index"].iloc[0]),
        "ping_end": int(rows["ping_index"].iloc[-1]),
        "meters_per_pixel": float(meters_per_pixel),
        "meters_per_pixel_alongtrack": float(
            np.nanmedian(at.to_numpy(float)[r0:r1])) if at is not None else float("nan"),
        "display_lo": float(lo), "display_hi": float(hi),
        "trimmed_bad_pings": bool(trim_bad_pings),
        "rows_dropped": int(H - (r1 - r0)),
        # rows of THIS png that are still flagged, if any survived the crop;
        # discard detections landing inside them
        "interior_bad_runs": interior,
        "ping_table_csv": "ping_table.csv",
    }
    ping_table.to_csv(os.path.join(out_dir, meta["ping_table_csv"]), index=False)
    with open(os.path.join(out_dir, f"{name}.json"), "w") as fh:
        json.dump(meta, fh, indent=2)
    return meta


def locate_waterfall_pixel(x: int, y: int, out_dir: str,
                           name: str = "waterfall") -> Dict[str, Any]:
    """:func:`locate_tile_pixel` for the whole-waterfall PNG.

    ``x``/``y`` are pixel coordinates *in the written PNG*.  The sidecar's
    ``row_offset`` is added back, so the trimming is invisible to the caller
    and the returned ``ping_start``/``ping_end`` name real pings.

    Returns the same dict as :func:`locate_tile_pixel`, minus the tile fields.
    """
    with open(os.path.join(out_dir, f"{name}.json")) as fh:
        meta = json.load(fh)
    table = pd.read_csv(os.path.join(out_dir, meta["ping_table_csv"]))

    if not (0 <= x < meta["width"] and 0 <= y < meta["height"]):
        raise ValueError(
            f"({x}, {y}) is outside the {meta['width']}x{meta['height']} waterfall")

    fix = _locate_full_pixel(
        meta["row_offset"] + int(y), meta["col_offset"] + int(x),
        image_width=meta["width"], ping_table=table,
        meters_per_pixel=meta["meters_per_pixel"],
        meters_per_pixel_alongtrack=meta["meters_per_pixel_alongtrack"],
        source_xtf_filename=meta["source_xtf_filename"],
    )
    fix.update({"x": int(x), "y": int(y)})
    return fix
