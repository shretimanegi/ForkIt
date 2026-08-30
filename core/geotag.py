"""
Geodesic coordinate math and layback correction for side-scan sonar.

Provides WGS84-accurate geodesic calculations using ``pyproj`` as a
production-grade replacement for the flat-Earth approximations in
:mod:`core.preprocess`, plus towfish layback correction for towed sonar
systems.

Flat-Earth formulas are fine for within-swath offsets (~100 m), but layback
distances (50–200 m) and aggregate survey-scale calculations benefit from
proper ellipsoidal math.  Both approaches agree to sub-metre precision at
swath scale, so :mod:`core.preprocess` keeps its fast flat-Earth path for
per-pixel operations during tiling; this module is used for final report
coordinates and layback.

Usage
-----
::

    from core.geotag import (
        offset_latlon_geodesic, inverse_geodesic,
        apply_layback, locate_pixel_geodesic,
    )

    # Move 100 m north from the equator on WGS84
    lat2, lon2 = offset_latlon_geodesic(0.0, 0.0, 0.0, 100.0)

    # Correct towfish positions for 80 m of cable behind the ship
    corrected_table = apply_layback(ping_table, cable_length_m=80.0)
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
from pyproj import Geod

__all__ = [
    "WGS84",
    "offset_latlon_geodesic",
    "inverse_geodesic",
    "apply_layback",
    "locate_pixel_geodesic",
]

# WGS84 ellipsoid — the geodetic standard for GPS coordinates.
WGS84 = Geod(ellps="WGS84")


# ======================================================================
# Core geodesic primitives
# ======================================================================

def offset_latlon_geodesic(
    lat: float,
    lon: float,
    bearing_deg: float,
    distance_m: float,
) -> Tuple[float, float]:
    """Move ``distance_m`` along ``bearing_deg`` from ``(lat, lon)`` on WGS84.

    This is the geodesic (Karney, 2013) equivalent of :func:`preprocess._offset_latlon`.
    The two agree to <1 m for distances under ~500 m at mid-latitudes.

    Parameters
    ----------
    lat, lon : float
        Starting position in decimal degrees (WGS84).
    bearing_deg : float
        Forward azimuth in degrees clockwise from north.
    distance_m : float
        Distance to travel in metres.  May be negative (travels in the
        reverse direction).

    Returns
    -------
    (lat2, lon2) : tuple of float
        Destination position in decimal degrees.
    """
    if distance_m == 0.0:
        return lat, lon
    # pyproj.Geod.fwd takes (lon, lat) — note the reversed order.
    lon2, lat2, _ = WGS84.fwd(lon, lat, bearing_deg, distance_m)
    return float(lat2), float(lon2)


def inverse_geodesic(
    lat1: float, lon1: float,
    lat2: float, lon2: float,
) -> Tuple[float, float, float]:
    """Geodesic inverse problem: distance and azimuths between two points.

    Parameters
    ----------
    lat1, lon1 : float
        First point (decimal degrees, WGS84).
    lat2, lon2 : float
        Second point (decimal degrees, WGS84).

    Returns
    -------
    (distance_m, fwd_azimuth_deg, back_azimuth_deg) : tuple of float
        ``fwd_azimuth_deg`` is the bearing from point 1 to point 2.
        ``back_azimuth_deg`` is the bearing from point 2 back to point 1.
    """
    fwd_az, back_az, dist = WGS84.inv(lon1, lat1, lon2, lat2)
    return float(dist), float(fwd_az), float(back_az)


# ======================================================================
# Layback correction
# ======================================================================

def apply_layback(
    ping_table: pd.DataFrame,
    cable_length_m: float,
    behind_ship: bool = True,
) -> pd.DataFrame:
    """Correct towfish GPS fixes for cable layback.

    When a side-scan sonar is towed behind a ship, the GPS antenna is on
    the ship, but the sonar is ``cable_length_m`` behind the ship along
    the survey track.  The towfish's true position is therefore offset
    backwards along the heading by the cable length.

    For AUV-mounted sonar (where the GPS is on the vehicle itself),
    ``cable_length_m=0`` means no correction is applied.

    Adds ``lat_corrected`` and ``lon_corrected`` columns to the table
    and returns it.  When ``cable_length_m == 0``, the corrected columns
    are simply copies of the original lat/lon.

    Parameters
    ----------
    ping_table : pd.DataFrame
        Must contain ``lat``, ``lon``, and ``heading`` columns.
    cable_length_m : float
        Cable/tow length in metres.  Zero means no correction.
    behind_ship : bool
        If True (default), the fish is behind the ship, so the fix is
        shifted opposite to the heading.  If False, the fix is shifted
        along the heading (unusual, but some setups mount sensors ahead).

    Returns
    -------
    pd.DataFrame
        The same DataFrame with ``lat_corrected`` and ``lon_corrected``
        added in place.
    """
    if cable_length_m == 0.0:
        ping_table["lat_corrected"] = ping_table["lat"].copy()
        ping_table["lon_corrected"] = ping_table["lon"].copy()
        return ping_table

    lats = ping_table["lat"].to_numpy(float)
    lons = ping_table["lon"].to_numpy(float)
    headings = ping_table["heading"].to_numpy(float)

    # Bearing for the offset: 180° from heading puts the fish behind the
    # ship; 0° from heading puts it ahead.
    bearing_offset = 180.0 if behind_ship else 0.0
    bearings = (headings + bearing_offset) % 360.0

    # Vectorised pyproj call — much faster than a Python loop.
    distances = np.full_like(lons, cable_length_m)
    lons_c, lats_c, _ = WGS84.fwd(lons, lats, bearings, distances)

    ping_table["lat_corrected"] = lats_c.astype(float)
    ping_table["lon_corrected"] = lons_c.astype(float)
    return ping_table


# ======================================================================
# Pixel → world coordinates (geodesic version)
# ======================================================================

def locate_pixel_geodesic(
    full_row: int,
    full_col: int,
    image_width: int,
    ping_table: pd.DataFrame,
    meters_per_pixel: float,
    meters_per_pixel_alongtrack: float,
    source_xtf_filename: str,
    use_corrected: bool = True,
) -> Dict[str, Any]:
    """Resolve a full-waterfall pixel to its seabed position using WGS84 geodesics.

    This is the geodesic equivalent of :func:`preprocess._locate_full_pixel`.
    It uses ``pyproj`` for the bearing offset from the survey track to the
    seabed point, and optionally uses layback-corrected coordinates.

    Parameters
    ----------
    full_row, full_col : int
        Row and column in the full (untiled) waterfall image.
    image_width : int
        Total width of the waterfall in pixels.
    ping_table : pd.DataFrame
        The ping metadata table from preprocessing.
    meters_per_pixel : float
        Across-track scale (from slant-to-ground range correction).
    meters_per_pixel_alongtrack : float
        Along-track scale (from GPS track).
    source_xtf_filename : str
        Name of the source XTF file for provenance.
    use_corrected : bool
        If True and ``lat_corrected``/``lon_corrected`` exist in the table,
        use the layback-corrected coordinates.  Otherwise uses raw lat/lon.

    Returns
    -------
    dict
        Same structure as ``preprocess._locate_full_pixel``, with the addition
        of ``lat_corrected``/``lon_corrected`` fields when available.
    """
    ping = ping_table.iloc[int(full_row)]
    mpp = float(meters_per_pixel)

    # Determine which lat/lon to use as the towfish position.
    if use_corrected and "lat_corrected" in ping_table.columns:
        fish_lat = float(ping["lat_corrected"])
        fish_lon = float(ping["lon_corrected"])
    else:
        fish_lat = float(ping["lat"])
        fish_lon = float(ping["lon"])

    # Column → across-track geometry.
    # Waterfall: [port far ... port near | stbd near ... stbd far]
    half = image_width / 2.0
    side = "port" if full_col < half else "stbd"
    ground_range_m = abs(full_col + 0.5 - half) * mpp

    # Bearing perpendicular to the heading, toward the correct side.
    heading = float(ping["heading"])
    bearing = heading + (90.0 if side == "stbd" else -90.0)

    # Geodesic offset from the (possibly layback-corrected) towfish position.
    lat, lon = offset_latlon_geodesic(fish_lat, fish_lon, bearing, ground_range_m)

    result = {
        "full_row": int(full_row),
        "full_col": int(full_col),
        "ping_start": int(ping["ping_index"]),
        "ping_end": int(ping["ping_index"]),
        "ping_number": int(ping["ping_number"]),
        "timestamp": str(ping["timestamp"]),
        "lat": lat,
        "lon": lon,
        "ship_lat": float(ping["lat"]),
        "ship_lon": float(ping["lon"]),
        "heading": heading,
        "altitude": float(ping.get("altitude_m", ping.get("altitude", np.nan))),
        "altitude_header": float(ping.get("altitude", np.nan)),
        "slant_range": float(ping["slant_range"]),
        "ground_range_m": float(ground_range_m),
        "side": side,
        "meters_per_pixel": mpp,
        "meters_per_pixel_alongtrack": float(meters_per_pixel_alongtrack),
        "source_xtf_filename": source_xtf_filename,
    }

    # Include layback-corrected fish position if available.
    if "lat_corrected" in ping_table.columns:
        result["fish_lat_corrected"] = fish_lat
        result["fish_lon_corrected"] = fish_lon

    return result
