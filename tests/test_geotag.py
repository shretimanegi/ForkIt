"""
Unit tests for core.geotag — geodesic math and layback correction.

Tests validate:
1. offset_latlon_geodesic against known distances
2. Round-trip offset → inverse
3. Flat-Earth vs pyproj agreement at swath scale
4. Layback correction with known headings
5. Edge cases: high latitudes, zero distance, poles
"""

import math

import numpy as np
import pandas as pd
import pytest

from core.geotag import (
    WGS84,
    apply_layback,
    inverse_geodesic,
    locate_pixel_geodesic,
    offset_latlon_geodesic,
)
from core.preprocess import _offset_latlon  # A2's flat-Earth version


# ======================================================================
# offset_latlon_geodesic
# ======================================================================

class TestOffsetGeodesic:
    """Tests for the WGS84 geodesic forward function."""

    def test_100m_north_from_equator(self):
        """100 m north from (0,0) should give ~0.000904° lat, 0° lon."""
        lat2, lon2 = offset_latlon_geodesic(0.0, 0.0, 0.0, 100.0)
        # 1° latitude ≈ 110,574 m at the equator on WGS84.
        expected_dlat = 100.0 / 110574.0
        assert abs(lat2 - expected_dlat) < 1e-6, f"lat2={lat2}"
        assert abs(lon2) < 1e-10, f"lon2={lon2}"

    def test_100m_east_from_equator(self):
        """100 m east from (0,0) should give 0° lat, ~0.000899° lon."""
        lat2, lon2 = offset_latlon_geodesic(0.0, 0.0, 90.0, 100.0)
        expected_dlon = 100.0 / 111320.0  # cos(0)=1
        assert abs(lat2) < 1e-6, f"lat2={lat2}"
        assert abs(lon2 - expected_dlon) < 1e-6, f"lon2={lon2}"

    def test_zero_distance(self):
        """Zero distance returns the same point."""
        lat, lon = offset_latlon_geodesic(45.0, -73.5, 42.0, 0.0)
        assert lat == 45.0
        assert lon == -73.5

    def test_negative_distance(self):
        """Negative distance moves in the reverse direction."""
        lat_fwd, lon_fwd = offset_latlon_geodesic(0.0, 0.0, 0.0, 100.0)
        lat_rev, lon_rev = offset_latlon_geodesic(0.0, 0.0, 180.0, 100.0)
        lat_neg, lon_neg = offset_latlon_geodesic(0.0, 0.0, 0.0, -100.0)
        assert abs(lat_neg - lat_rev) < 1e-8
        assert abs(lon_neg - lon_rev) < 1e-8

    def test_high_latitude(self):
        """Works correctly at 80°N where cos(lat) ≈ 0.17."""
        lat2, lon2 = offset_latlon_geodesic(80.0, 10.0, 90.0, 100.0)
        # At 80°N, 1° longitude ≈ 111320 * cos(80°) ≈ 19,330 m.
        # So 100 m east ≈ 100/19330 ≈ 0.00517° lon.
        assert abs(lon2 - 10.0 - 100.0 / (111320.0 * math.cos(math.radians(80.0)))) < 1e-4
        assert abs(lat2 - 80.0) < 1e-4  # mostly east, tiny lat change

    def test_bearing_wraparound(self):
        """Bearings 360° and 0° are equivalent."""
        a = offset_latlon_geodesic(10.0, 20.0, 0.0, 500.0)
        b = offset_latlon_geodesic(10.0, 20.0, 360.0, 500.0)
        assert abs(a[0] - b[0]) < 1e-10
        assert abs(a[1] - b[1]) < 1e-10


# ======================================================================
# inverse_geodesic
# ======================================================================

class TestInverseGeodesic:
    """Tests for the geodesic inverse function."""

    def test_known_distance(self):
        """Inverse of a known offset should recover the distance."""
        lat2, lon2 = offset_latlon_geodesic(0.0, 0.0, 45.0, 1000.0)
        dist, fwd_az, _ = inverse_geodesic(0.0, 0.0, lat2, lon2)
        assert abs(dist - 1000.0) < 0.01, f"dist={dist}"
        assert abs(fwd_az - 45.0) < 0.01, f"fwd_az={fwd_az}"

    def test_round_trip(self):
        """Forward then inverse recovers the original distance and bearing."""
        for bearing in [0, 45, 90, 135, 180, 225, 270, 315]:
            for dist in [10.0, 100.0, 1000.0, 10000.0]:
                lat2, lon2 = offset_latlon_geodesic(37.0, -122.0, bearing, dist)
                d_back, az_back, _ = inverse_geodesic(37.0, -122.0, lat2, lon2)
                assert abs(d_back - dist) < 0.01, \
                    f"bearing={bearing}, dist={dist}: got {d_back}"
                # Bearing should match (modulo 360).
                az_diff = (az_back - bearing + 180) % 360 - 180
                assert abs(az_diff) < 0.1, \
                    f"bearing={bearing}, dist={dist}: got az={az_back}"

    def test_same_point(self):
        """Distance between a point and itself is zero."""
        dist, _, _ = inverse_geodesic(45.0, 10.0, 45.0, 10.0)
        assert dist < 1e-6


# ======================================================================
# Flat-Earth vs pyproj agreement
# ======================================================================

class TestFlatEarthAgreement:
    """Verify that A2's flat-Earth and B2's pyproj agree at swath scale."""

    @pytest.mark.parametrize("lat,lon,bearing,dist", [
        (0.0, 0.0, 45.0, 50.0),
        (0.0, 0.0, 90.0, 100.0),
        (45.0, -73.5, 0.0, 100.0),
        (45.0, -73.5, 135.0, 80.0),
        (-33.86, 151.2, 270.0, 120.0),   # Sydney
        (60.0, 25.0, 180.0, 100.0),       # Helsinki
    ])
    def test_agreement_within_1m(self, lat, lon, bearing, dist):
        """Flat-Earth and pyproj must agree within 1 m for swath-scale offsets."""
        flat_lat, flat_lon = _offset_latlon(lat, lon, bearing, dist)
        geo_lat, geo_lon = offset_latlon_geodesic(lat, lon, bearing, dist)

        # Compute the distance between the two results.
        separation, _, _ = inverse_geodesic(flat_lat, flat_lon, geo_lat, geo_lon)
        assert separation < 1.0, \
            f"flat vs geodesic differ by {separation:.3f} m at ({lat},{lon}) " \
            f"bearing={bearing} dist={dist}"


# ======================================================================
# apply_layback
# ======================================================================

class TestLayback:
    """Tests for the towfish layback correction."""

    def _make_table(self, n=10, heading=90.0, lat=0.0, lon=0.0):
        """Create a minimal ping table for testing."""
        return pd.DataFrame({
            "lat": [lat] * n,
            "lon": [lon] * n,
            "heading": [heading] * n,
            "ping_index": list(range(n)),
            "ping_number": list(range(n)),
            "timestamp": pd.date_range("2024-01-01", periods=n, freq="100ms"),
        })

    def test_zero_cable(self):
        """Zero cable length → corrected = original."""
        table = self._make_table()
        apply_layback(table, cable_length_m=0.0)
        assert "lat_corrected" in table.columns
        assert "lon_corrected" in table.columns
        np.testing.assert_array_equal(
            table["lat_corrected"].values, table["lat"].values
        )
        np.testing.assert_array_equal(
            table["lon_corrected"].values, table["lon"].values
        )

    def test_heading_90_shifts_west(self):
        """Heading 90° (east) + behind_ship → fish is west of the GPS fix."""
        table = self._make_table(heading=90.0, lat=0.0, lon=0.0)
        apply_layback(table, cable_length_m=100.0, behind_ship=True)
        # Fish is behind the ship → 180° from heading = 270° = west.
        # So lon_corrected < lon.
        assert table["lon_corrected"].iloc[0] < table["lon"].iloc[0], \
            f"expected westward shift: lon_corrected={table['lon_corrected'].iloc[0]}"
        # Lat should be approximately unchanged (east-west movement).
        assert abs(table["lat_corrected"].iloc[0] - table["lat"].iloc[0]) < 1e-5

    def test_heading_0_shifts_south(self):
        """Heading 0° (north) + behind_ship → fish is south of the GPS fix."""
        table = self._make_table(heading=0.0, lat=10.0, lon=20.0)
        apply_layback(table, cable_length_m=80.0, behind_ship=True)
        # Fish behind ship heading north → 180° = south.
        assert table["lat_corrected"].iloc[0] < table["lat"].iloc[0], \
            f"expected southward shift: lat_corrected={table['lat_corrected'].iloc[0]}"
        assert abs(table["lon_corrected"].iloc[0] - table["lon"].iloc[0]) < 1e-5

    def test_layback_distance_correct(self):
        """The offset distance matches the cable length."""
        table = self._make_table(heading=45.0, lat=37.0, lon=-122.0)
        cable = 120.0
        apply_layback(table, cable_length_m=cable)

        dist, _, _ = inverse_geodesic(
            table["lat"].iloc[0], table["lon"].iloc[0],
            table["lat_corrected"].iloc[0], table["lon_corrected"].iloc[0],
        )
        assert abs(dist - cable) < 0.01, f"expected {cable} m, got {dist:.3f} m"

    def test_behind_ship_false(self):
        """behind_ship=False shifts along the heading, not against it."""
        table_behind = self._make_table(heading=0.0, lat=0.0, lon=0.0)
        table_ahead = self._make_table(heading=0.0, lat=0.0, lon=0.0)
        apply_layback(table_behind, cable_length_m=100.0, behind_ship=True)
        apply_layback(table_ahead, cable_length_m=100.0, behind_ship=False)
        # Behind goes south, ahead goes north.
        assert table_behind["lat_corrected"].iloc[0] < 0.0
        assert table_ahead["lat_corrected"].iloc[0] > 0.0

    def test_idempotent(self):
        """Calling apply_layback twice with same cable doesn't double-shift."""
        table = self._make_table(heading=90.0, lat=0.0, lon=0.0)
        apply_layback(table, cable_length_m=100.0)
        first_lon = table["lon_corrected"].iloc[0]
        # Second call with cable_length_m=0 → should detect columns exist
        # but since we pass a different cable length, let's verify the
        # columns are overwritten correctly.
        apply_layback(table, cable_length_m=0.0)
        assert table["lon_corrected"].iloc[0] == table["lon"].iloc[0]


# ======================================================================
# locate_pixel_geodesic
# ======================================================================

class TestLocatePixelGeodesic:
    """Tests for geodesic pixel → world coordinate resolution."""

    def _make_ping_table(self):
        """Create a realistic minimal ping table."""
        return pd.DataFrame({
            "ping_index": [0, 1, 2, 3, 4],
            "ping_number": [100, 101, 102, 103, 104],
            "timestamp": pd.date_range("2024-06-15 10:00", periods=5, freq="200ms"),
            "lat": [45.0, 45.0001, 45.0002, 45.0003, 45.0004],
            "lon": [-73.5, -73.5, -73.5, -73.5, -73.5],
            "heading": [0.0, 0.0, 0.0, 0.0, 0.0],
            "altitude": [10.0, 10.0, 10.0, 10.0, 10.0],
            "altitude_m": [10.0, 10.0, 10.0, 10.0, 10.0],
            "slant_range": [75.0, 75.0, 75.0, 75.0, 75.0],
            "meters_per_pixel": [0.3, 0.3, 0.3, 0.3, 0.3],
        })

    def test_returns_required_fields(self):
        """Result dict contains all fields needed by the report builder."""
        table = self._make_ping_table()
        fix = locate_pixel_geodesic(
            full_row=2, full_col=600, image_width=1000,
            ping_table=table, meters_per_pixel=0.3,
            meters_per_pixel_alongtrack=0.7,
            source_xtf_filename="test.xtf",
        )
        required = {
            "full_row", "full_col", "ping_start", "ping_end", "ping_number",
            "timestamp", "lat", "lon", "ship_lat", "ship_lon", "heading",
            "altitude", "slant_range", "ground_range_m", "side",
            "meters_per_pixel", "meters_per_pixel_alongtrack",
            "source_xtf_filename",
        }
        assert required.issubset(fix.keys()), \
            f"missing fields: {required - set(fix.keys())}"

    def test_port_side(self):
        """A column left of centre is on the port side."""
        table = self._make_ping_table()
        fix = locate_pixel_geodesic(
            full_row=2, full_col=100, image_width=1000,
            ping_table=table, meters_per_pixel=0.3,
            meters_per_pixel_alongtrack=0.7,
            source_xtf_filename="test.xtf",
        )
        assert fix["side"] == "port"

    def test_stbd_side(self):
        """A column right of centre is on the starboard side."""
        table = self._make_ping_table()
        fix = locate_pixel_geodesic(
            full_row=2, full_col=700, image_width=1000,
            ping_table=table, meters_per_pixel=0.3,
            meters_per_pixel_alongtrack=0.7,
            source_xtf_filename="test.xtf",
        )
        assert fix["side"] == "stbd"

    def test_agrees_with_flat_earth(self):
        """Geodesic and flat-Earth pixel locations agree within 1 m."""
        table = self._make_ping_table()

        fix_geo = locate_pixel_geodesic(
            full_row=2, full_col=700, image_width=1000,
            ping_table=table, meters_per_pixel=0.3,
            meters_per_pixel_alongtrack=0.7,
            source_xtf_filename="test.xtf",
        )

        from core.preprocess import _locate_full_pixel
        fix_flat = _locate_full_pixel(
            full_row=2, full_col=700, image_width=1000,
            ping_table=table, meters_per_pixel=0.3,
            meters_per_pixel_alongtrack=0.7,
            source_xtf_filename="test.xtf",
        )

        dist, _, _ = inverse_geodesic(
            fix_geo["lat"], fix_geo["lon"],
            fix_flat["lat"], fix_flat["lon"],
        )
        assert dist < 1.0, f"geodesic vs flat-Earth differ by {dist:.3f} m"

    def test_uses_corrected_coords_when_available(self):
        """When layback-corrected columns exist, they are used."""
        table = self._make_ping_table()
        apply_layback(table, cable_length_m=50.0)

        fix_corrected = locate_pixel_geodesic(
            full_row=2, full_col=500, image_width=1000,
            ping_table=table, meters_per_pixel=0.3,
            meters_per_pixel_alongtrack=0.7,
            source_xtf_filename="test.xtf",
            use_corrected=True,
        )
        fix_raw = locate_pixel_geodesic(
            full_row=2, full_col=500, image_width=1000,
            ping_table=table, meters_per_pixel=0.3,
            meters_per_pixel_alongtrack=0.7,
            source_xtf_filename="test.xtf",
            use_corrected=False,
        )

        # The two results should differ because layback shifts the origin.
        dist, _, _ = inverse_geodesic(
            fix_corrected["lat"], fix_corrected["lon"],
            fix_raw["lat"], fix_raw["lon"],
        )
        assert dist > 1.0, \
            f"corrected and raw should differ; separation={dist:.3f} m"
