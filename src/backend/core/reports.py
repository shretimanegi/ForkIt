"""
Detection report generator for side-scan sonar debris detection.

Converts raw YOLO model outputs into the **frozen Day-1 JSON schema** agreed
by all three pairs, and writes structured reports in JSON, CSV, and GeoJSON
formats.

The schema
----------
Every detection record contains exactly these fields::

    {
        "id":             str,     # unique detection identifier
        "class":          str,     # object class (shipwreck, ghost_net, pipe, ...)
        "confidence":     float,   # calibrated confidence 0.0–1.0
        "bbox_px":        [x, y, w, h],  # bounding box in tile/waterfall pixels
        "ping_start":     int,     # first ping row containing the detection
        "ping_end":       int,     # last ping row
        "lat":            float,   # seabed latitude (WGS84)
        "lon":            float,   # seabed longitude (WGS84)
        "length_m":       float,   # along-track extent
        "width_m":        float,   # across-track extent
        "height_m":       float,   # estimated height from shadow geometry
        "shadow_len_px":  float    # measured shadow length in pixels
    }

Extra fields (``side``, ``ground_range_m``, ``timestamp``, ``ship_lat``,
``ship_lon``, ``heading``, ``altitude``, ``source_xtf_filename``,
``tile_id``) are carried alongside but are not part of the core schema.

Usage
-----
::

    from core.reports import (
        process_tile_detections, generate_report,
        merge_overlapping_detections,
    )

    # After running YOLO on each tile:
    all_dets = []
    for tile_id, yolo_results in per_tile_results.items():
        all_dets.extend(
            process_tile_detections(tile_id, yolo_results, tile_dir)
        )
    all_dets = merge_overlapping_detections(all_dets)
    generate_report(all_dets, "detections.json", fmt="json")
    generate_report(all_dets, "detections.csv",  fmt="csv")
    generate_report(all_dets, "detections.geojson", fmt="geojson")
"""

from __future__ import annotations

import csv
import json
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import geojson
import numpy as np
import pandas as pd

from core.geotag import (
    apply_layback,
    locate_pixel_geodesic,
)
from core.preprocess import (
    bbox_to_metres,
    load_tile_index,
)

__all__ = [
    "SCHEMA_FIELDS",
    "CLASS_NAMES",
    "build_detection_record",
    "process_tile_detections",
    "process_waterfall_detections",
    "merge_overlapping_detections",
    "generate_report",
    "to_json",
    "to_csv",
    "to_geojson",
]

# The frozen Day-1 schema — the contract all three pairs build against.
SCHEMA_FIELDS = [
    "id", "class", "confidence", "bbox_px",
    "ping_start", "ping_end",
    "lat", "lon",
    "length_m", "width_m", "height_m",
    "shadow_len_px",
]

# Class index → human-readable name.
# Must match Pair 1's YOLO training class order.
CLASS_NAMES: Dict[int, str] = {
    0: "shipwreck",
    1: "ghost_net",
    2: "pipe",
    3: "cylinder",
    4: "debris",
}


# ======================================================================
# Single detection record builder
# ======================================================================

def build_detection_record(
    det: Dict[str, Any],
    fix: Dict[str, Any],
    dims: Dict[str, float],
    det_id: str,
) -> Dict[str, Any]:
    """Build one detection record matching the frozen Day-1 schema.

    Parameters
    ----------
    det : dict
        Raw model output.  Expected keys:

        - ``class_id`` (int) or ``class`` (str): object class
        - ``confidence`` (float): detection confidence 0.0–1.0
        - ``bbox_px`` (list): ``[x, y, w, h]`` in tile/waterfall pixels
        - ``shadow_len_px`` (float, optional): shadow length in pixels

    fix : dict
        Pixel→world fix from :func:`locate_tile_pixel`,
        :func:`locate_waterfall_pixel`, or :func:`locate_pixel_geodesic`.
    dims : dict
        Physical dimensions from :func:`bbox_to_metres`.
    det_id : str
        Unique identifier for this detection (e.g. ``"det_001"``).

    Returns
    -------
    dict
        Schema-compliant detection record with all required fields plus
        extra metadata fields.
    """
    # Resolve class name.
    if "class" in det and isinstance(det["class"], str):
        class_name = det["class"]
    elif "class_id" in det:
        class_name = CLASS_NAMES.get(int(det["class_id"]), f"class_{det['class_id']}")
    else:
        class_name = "unknown"

    bbox = list(det.get("bbox_px", [0, 0, 0, 0]))
    shadow = det.get("shadow_len_px", None)

    ping_start = fix.get("ping_start", 0)
    ping_end = fix.get("ping_end", 0)

    # Core schema fields.
    record: Dict[str, Any] = {
        "id": det_id,
        "class": class_name,
        "confidence": round(float(det.get("confidence", 0.0)), 4),
        "bbox_px": bbox,
        "ping_start": int(ping_start),
        "ping_end": int(ping_end),
        "lat": round(float(fix.get("lat", 0.0)), 7),
        "lon": round(float(fix.get("lon", 0.0)), 7),
        "length_m": round(float(dims.get("length_m", 0.0)), 2),
        "width_m": round(float(dims.get("width_m", 0.0)), 2),
        "height_m": round(float(dims.get("height_m", float("nan"))), 2),
        "shadow_len_px": float(shadow) if shadow is not None else float("nan"),
    }

    # Extra metadata — not part of the core schema but useful for the
    # dashboard and debugging.  These don't break the contract.
    record["_extra"] = {
        "side": fix.get("side", ""),
        "ground_range_m": round(float(fix.get("ground_range_m", 0.0)), 2),
        "timestamp": fix.get("timestamp", ""),
        "ship_lat": fix.get("ship_lat", None),
        "ship_lon": fix.get("ship_lon", None),
        "heading": fix.get("heading", None),
        "altitude": fix.get("altitude", None),
        "source_xtf_filename": fix.get("source_xtf_filename", ""),
        "tile_id": fix.get("tile_id", ""),
        "meters_per_pixel": fix.get("meters_per_pixel", None),
        "meters_per_pixel_alongtrack": fix.get("meters_per_pixel_alongtrack", None),
    }

    return record


# ======================================================================
# Batch processing: tile-based and waterfall-based
# ======================================================================

def process_tile_detections(
    tile_id: str,
    yolo_results: List[Dict[str, Any]],
    tile_dir: str,
    cable_length_m: float = 0.0,
    index=None,
    ping_table: Optional[pd.DataFrame] = None,
    id_prefix: str = "det",
    id_offset: int = 0,
) -> List[Dict[str, Any]]:
    """Process all detections from one tile into schema-compliant records.

    Parameters
    ----------
    tile_id : str
        Tile identifier (e.g. ``"r000512_c000000"``).
    yolo_results : list of dict
        Each dict must contain at least ``class_id`` (int), ``confidence``
        (float), ``bbox_px`` ([x, y, w, h]).  Optional: ``shadow_len_px``.
    tile_dir : str
        Path to the tile directory containing ``tile_index.json``.
    cable_length_m : float
        Layback cable length.  0 = no correction.
    index : TileIndex, optional
        Pre-loaded tile index (avoids re-reading from disk).
    ping_table : pd.DataFrame, optional
        Pre-loaded ping table.
    id_prefix : str
        Prefix for detection IDs.
    id_offset : int
        Starting number for detection IDs.

    Returns
    -------
    list of dict
        Schema-compliant detection records.
    """
    if index is None or ping_table is None:
        index, ping_table = load_tile_index(tile_dir)

    # Apply layback if needed (will overwrite existing columns if present).
    if cable_length_m > 0:
        apply_layback(ping_table, cable_length_m)

    records = []
    for i, det in enumerate(yolo_results):
        det_id = f"{id_prefix}_{id_offset + i + 1:04d}"
        bbox = det.get("bbox_px", [0, 0, 0, 0])
        # Centre of the bounding box for the pixel fix.
        cx = int(bbox[0] + bbox[2] / 2)
        cy = int(bbox[1] + bbox[3] / 2)

        # Clamp to tile bounds.
        tile_meta = index.get(tile_id)
        ts = tile_meta["tile_size"]
        cx = max(0, min(cx, ts - 1))
        cy = max(0, min(cy, ts - 1))

        tile_meta = index.get(tile_id)
        
        full_row = tile_meta["row_offset"] + cy
        full_col = tile_meta["col_offset"] + cx
        
        fix = locate_pixel_geodesic(
            full_row=full_row,
            full_col=full_col,
            image_width=index.image_shape[1],
            ping_table=ping_table,
            meters_per_pixel=tile_meta["meters_per_pixel"],
            meters_per_pixel_alongtrack=tile_meta["meters_per_pixel_alongtrack"],
            source_xtf_filename=tile_meta["source_xtf_filename"],
            use_corrected=(cable_length_m > 0)
        )
        
        # Calculate ping span properly using the ping table
        row_start = tile_meta["row_offset"] + int(bbox[1])
        row_end = tile_meta["row_offset"] + int(bbox[1] + bbox[3]) - 1
        row_start = max(0, min(row_start, len(ping_table) - 1))
        row_end = max(0, min(row_end, len(ping_table) - 1))
        
        fix["ping_start"] = int(ping_table.iloc[row_start]["ping_index"])
        fix["ping_end"] = int(ping_table.iloc[row_end]["ping_index"])
        
        fix.update({
            "tile_id": tile_id, "x": int(cx), "y": int(cy),
        })
        
        shadow = det.get("shadow_len_px", None)
        dims = bbox_to_metres(fix, bbox, shadow_len_px=shadow)

        records.append(build_detection_record(det, fix, dims, det_id))

    return records


def process_waterfall_detections(
    yolo_results: List[Dict[str, Any]],
    out_dir: str,
    cable_length_m: float = 0.0,
    id_prefix: str = "det",
    id_offset: int = 0,
) -> List[Dict[str, Any]]:
    """Process all detections from a whole-waterfall image.

    Parameters
    ----------
    yolo_results : list of dict
        Each dict: ``class_id``, ``confidence``, ``bbox_px`` [x,y,w,h],
        optional ``shadow_len_px``.
    out_dir : str
        Directory containing ``waterfall.png`` and ``waterfall.json``.
    cable_length_m : float
        Layback cable length.  0 = no correction.
    id_prefix : str
        Prefix for detection IDs.
    id_offset : int
        Starting number for detection IDs.

    Returns
    -------
    list of dict
        Schema-compliant detection records.
    """
    with open(os.path.join(out_dir, "waterfall.json")) as fh:
        meta = json.load(fh)
    table = pd.read_csv(os.path.join(out_dir, meta["ping_table_csv"]))

    if cable_length_m > 0:
        apply_layback(table, cable_length_m)

    records = []
    for i, det in enumerate(yolo_results):
        det_id = f"{id_prefix}_{id_offset + i + 1:04d}"
        bbox = det.get("bbox_px", [0, 0, 0, 0])
        cx = int(bbox[0] + bbox[2] / 2)
        cy = int(bbox[1] + bbox[3] / 2)

        full_row = meta["row_offset"] + cy
        full_col = meta["col_offset"] + cx
        
        fix = locate_pixel_geodesic(
            full_row=full_row,
            full_col=full_col,
            image_width=meta["width"],
            ping_table=table,
            meters_per_pixel=meta["meters_per_pixel"],
            meters_per_pixel_alongtrack=meta["meters_per_pixel_alongtrack"],
            source_xtf_filename=meta["source_xtf_filename"],
            use_corrected=(cable_length_m > 0)
        )
        
        row_start = meta["row_offset"] + int(bbox[1])
        row_end = meta["row_offset"] + int(bbox[1] + bbox[3]) - 1
        row_start = max(0, min(row_start, len(table) - 1))
        row_end = max(0, min(row_end, len(table) - 1))
        
        fix["ping_start"] = int(table.iloc[row_start]["ping_index"])
        fix["ping_end"] = int(table.iloc[row_end]["ping_index"])
        fix.update({"x": int(cx), "y": int(cy)})

        shadow = det.get("shadow_len_px", None)
        dims = bbox_to_metres(fix, bbox, shadow_len_px=shadow)

        records.append(build_detection_record(det, fix, dims, det_id))

    return records


# ======================================================================
# Overlapping tile deduplication (NMS on world coordinates)
# ======================================================================

def _iou_latlon(a: Dict, b: Dict) -> float:
    """Approximate IoU between two detections using their metric bounding boxes.

    We project both boxes into a local metre grid centred on detection A
    and compute the axis-aligned intersection.  This is approximate (no
    rotation) but sufficient for NMS on overlapping tiles where heading
    changes slowly.
    """
    ax, ay, aw, ah = a["bbox_px"]
    bx, by, bw, bh = b["bbox_px"]

    mpp_x_a = a.get("_extra", {}).get("meters_per_pixel", 1.0) or 1.0
    mpp_y_a = a.get("_extra", {}).get("meters_per_pixel_alongtrack", 1.0) or 1.0
    mpp_x_b = b.get("_extra", {}).get("meters_per_pixel", 1.0) or 1.0
    mpp_y_b = b.get("_extra", {}).get("meters_per_pixel_alongtrack", 1.0) or 1.0

    # Convert to metre extents.
    a_w_m = aw * mpp_x_a
    a_h_m = ah * mpp_y_a
    b_w_m = bw * mpp_x_b
    b_h_m = bh * mpp_y_b

    # Centre distance in metres (flat-Earth at this scale is fine).
    coslat = np.cos(np.radians(a["lat"]))
    dx = (b["lon"] - a["lon"]) * 111320.0 * coslat
    dy = (b["lat"] - a["lat"]) * 110540.0

    # Axis-aligned boxes centred at (0,0) and (dx,dy).
    a_x0, a_x1 = -a_w_m / 2, a_w_m / 2
    a_y0, a_y1 = -a_h_m / 2, a_h_m / 2
    b_x0, b_x1 = dx - b_w_m / 2, dx + b_w_m / 2
    b_y0, b_y1 = dy - b_h_m / 2, dy + b_h_m / 2

    inter_w = max(0.0, min(a_x1, b_x1) - max(a_x0, b_x0))
    inter_h = max(0.0, min(a_y1, b_y1) - max(a_y0, b_y0))
    inter = inter_w * inter_h
    union = a_w_m * a_h_m + b_w_m * b_h_m - inter

    return inter / union if union > 0 else 0.0


def merge_overlapping_detections(
    detections: List[Dict[str, Any]],
    iou_threshold: float = 0.3,
) -> List[Dict[str, Any]]:
    """Deduplicate detections from overlapping tiles using NMS on world coords.

    Tiles overlap (typically 20%), so the same target can appear in two or
    more tiles.  This function performs class-aware non-maximum suppression
    in world-coordinate space: for each pair of same-class detections whose
    metric bounding boxes overlap by more than ``iou_threshold``, the one
    with lower confidence is suppressed.

    Parameters
    ----------
    detections : list of dict
        Schema-compliant detection records.
    iou_threshold : float
        Overlap threshold for suppression.  0.3 is conservative: real
        duplicates from a 20% tile overlap will have IoU ≈ 0.5–0.9.

    Returns
    -------
    list of dict
        Filtered detections with duplicates removed.
    """
    if len(detections) <= 1:
        return list(detections)

    # Sort by confidence descending.
    dets = sorted(detections, key=lambda d: -d["confidence"])
    keep = [True] * len(dets)

    for i in range(len(dets)):
        if not keep[i]:
            continue
        for j in range(i + 1, len(dets)):
            if not keep[j]:
                continue
            # Only suppress within the same class.
            if dets[i]["class"] != dets[j]["class"]:
                continue
            if _iou_latlon(dets[i], dets[j]) >= iou_threshold:
                keep[j] = False  # suppress the lower-confidence detection

    return [d for d, k in zip(dets, keep) if k]


# ======================================================================
# Report generation
# ======================================================================

def _report_header(detections: List[Dict], source: str = "") -> Dict[str, Any]:
    """Metadata header for the JSON report."""
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "generator": "SSS-pipeline / core.reports",
        "schema_version": "1.0",
        "source_xtf": source or (
            detections[0].get("_extra", {}).get("source_xtf_filename", "")
            if detections else ""
        ),
        "total_detections": len(detections),
        "classes": sorted(set(d["class"] for d in detections)) if detections else [],
    }


def _strip_extra(det: Dict[str, Any], include_extra: bool = False) -> Dict[str, Any]:
    """Return a copy with ``_extra`` inlined or removed."""
    out = {k: v for k, v in det.items() if k != "_extra"}
    if include_extra and "_extra" in det:
        out.update(det["_extra"])
    return out


def _sanitise_for_json(value: Any) -> Any:
    """Replace NaN/Inf with None for JSON serialisation."""
    if isinstance(value, float):
        if np.isnan(value) or np.isinf(value):
            return None
        return value
    if isinstance(value, dict):
        return {k: _sanitise_for_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_sanitise_for_json(v) for v in value]
    return value


def to_json(
    detections: List[Dict[str, Any]],
    path: str,
    include_extra: bool = False,
    indent: int = 2,
) -> str:
    """Write detections to a JSON file with a metadata header.

    Parameters
    ----------
    detections : list of dict
        Schema-compliant detection records.
    path : str
        Output file path.
    include_extra : bool
        If True, ``_extra`` metadata fields are inlined into each record.
    indent : int
        JSON indentation.

    Returns
    -------
    str
        Absolute path to the written file.
    """
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)

    payload = {
        "metadata": _report_header(detections),
        "detections": [
            _sanitise_for_json(_strip_extra(d, include_extra))
            for d in detections
        ],
    }

    with open(path, "w") as fh:
        json.dump(payload, fh, indent=indent, default=str)

    return os.path.abspath(path)


def to_csv(
    detections: List[Dict[str, Any]],
    path: str,
    include_extra: bool = False,
) -> str:
    """Write detections to a flat CSV file, one row per detection.

    ``bbox_px`` is expanded into four columns: ``bbox_x``, ``bbox_y``,
    ``bbox_w``, ``bbox_h``.

    Parameters
    ----------
    detections : list of dict
        Schema-compliant detection records.
    path : str
        Output file path.
    include_extra : bool
        If True, extra metadata columns are included.

    Returns
    -------
    str
        Absolute path to the written file.
    """
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)

    if not detections:
        with open(path, "w", newline="") as fh:
            fh.write(",".join(SCHEMA_FIELDS) + "\n")
        return os.path.abspath(path)

    # Flatten records.
    rows = []
    for det in detections:
        flat = _strip_extra(det, include_extra)
        # Expand bbox_px into individual columns.
        bbox = flat.pop("bbox_px", [0, 0, 0, 0])
        flat["bbox_x"] = bbox[0] if len(bbox) > 0 else 0
        flat["bbox_y"] = bbox[1] if len(bbox) > 1 else 0
        flat["bbox_w"] = bbox[2] if len(bbox) > 2 else 0
        flat["bbox_h"] = bbox[3] if len(bbox) > 3 else 0
        # Replace NaN with empty string for CSV.
        for k, v in flat.items():
            if isinstance(v, float) and (np.isnan(v) or np.isinf(v)):
                flat[k] = ""
        rows.append(flat)

    # Determine column order: core schema fields first, then extras.
    core_cols = [
        "id", "class", "confidence",
        "bbox_x", "bbox_y", "bbox_w", "bbox_h",
        "ping_start", "ping_end",
        "lat", "lon",
        "length_m", "width_m", "height_m",
        "shadow_len_px",
    ]
    extra_cols = sorted(set().union(*(r.keys() for r in rows)) - set(core_cols))
    if not include_extra:
        extra_cols = []
    columns = core_cols + extra_cols

    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    return os.path.abspath(path)


def to_geojson(
    detections: List[Dict[str, Any]],
    path: str,
    include_extra: bool = False,
) -> str:
    """Write detections to a GeoJSON FeatureCollection.

    Each detection becomes a Point feature at its (lon, lat) with all
    schema fields as properties.  The output can be loaded directly into
    QGIS, geojson.io, or any GIS tool.

    Parameters
    ----------
    detections : list of dict
        Schema-compliant detection records.
    path : str
        Output file path.
    include_extra : bool
        If True, extra metadata is included as properties.

    Returns
    -------
    str
        Absolute path to the written file.
    """
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)

    features = []
    for det in detections:
        lat = det.get("lat", 0.0)
        lon = det.get("lon", 0.0)

        # Build properties from all fields except lat/lon (which go into
        # the geometry).
        props = _sanitise_for_json(_strip_extra(det, include_extra))
        # bbox_px as a string so GIS tools display it cleanly.
        if "bbox_px" in props:
            props["bbox_px"] = str(props["bbox_px"])

        feature = geojson.Feature(
            geometry=geojson.Point((lon, lat)),
            properties=props,
            id=det.get("id", ""),
        )
        features.append(feature)

    collection = geojson.FeatureCollection(
        features,
        # Custom metadata in the collection root.
        metadata=_sanitise_for_json(_report_header(detections)),
    )

    with open(path, "w") as fh:
        geojson.dump(collection, fh, indent=2)

    return os.path.abspath(path)


def generate_report(
    detections: List[Dict[str, Any]],
    output_path: str,
    fmt: str = "json",
    include_extra: bool = False,
) -> str:
    """Write a detection report in the specified format.

    Parameters
    ----------
    detections : list of dict
        Schema-compliant detection records.
    output_path : str
        Output file path.
    fmt : str
        Output format: ``"json"``, ``"csv"``, or ``"geojson"``.
    include_extra : bool
        Whether to include extra metadata fields.

    Returns
    -------
    str
        Absolute path to the written file.

    Raises
    ------
    ValueError
        If ``fmt`` is not one of the supported formats.
    """
    fmt = fmt.lower().strip()
    writers = {
        "json": to_json,
        "csv": to_csv,
        "geojson": to_geojson,
    }
    if fmt not in writers:
        raise ValueError(
            f"unsupported format {fmt!r}; choose from {list(writers.keys())}"
        )
    return writers[fmt](detections, output_path, include_extra=include_extra)
