"""
Unit tests for core.reports — Detection report generator.

Tests validate:
1. build_detection_record produces all schema fields
2. JSON schema compliance
3. CSV output format
4. GeoJSON output format
5. Overlapping tile deduplication (NMS)
6. Fake detection data handling
"""

import json
import os
import tempfile
from typing import Any, Dict

import geojson
import pandas as pd
import pytest

from core.reports import (
    CLASS_NAMES,
    SCHEMA_FIELDS,
    build_detection_record,
    generate_report,
    merge_overlapping_detections,
    process_tile_detections,
    process_waterfall_detections,
)


@pytest.fixture
def sample_detection() -> Dict[str, Any]:
    return {
        "class_id": 0,
        "confidence": 0.85,
        "bbox_px": [10, 20, 30, 40],
        "shadow_len_px": 15,
    }


@pytest.fixture
def sample_fix() -> Dict[str, Any]:
    return {
        "full_row": 100,
        "full_col": 500,
        "ping_start": 50,
        "ping_end": 50,
        "lat": 45.123,
        "lon": -73.456,
        "side": "port",
        "ground_range_m": 50.0,
    }


@pytest.fixture
def sample_dims() -> Dict[str, float]:
    return {
        "length_m": 5.0,
        "width_m": 2.5,
        "height_m": 1.5,
    }


class TestReportBuilder:
    def test_build_detection_record(self, sample_detection, sample_fix, sample_dims):
        det_id = "det_0001"
        record = build_detection_record(
            sample_detection, sample_fix, sample_dims, det_id
        )

        # Check core schema fields exist
        for field in SCHEMA_FIELDS:
            assert field in record, f"Missing required field: {field}"

        assert record["id"] == det_id
        assert record["class"] == CLASS_NAMES[0]
        assert record["confidence"] == 0.85
        assert record["bbox_px"] == [10, 20, 30, 40]
        assert record["lat"] == 45.123
        assert record["lon"] == -73.456
        assert record["length_m"] == 5.0
        assert record["width_m"] == 2.5
        assert record["height_m"] == 1.5
        assert record["shadow_len_px"] == 15.0
        
        # Check extras
        assert "_extra" in record
        assert record["_extra"]["side"] == "port"
        assert record["_extra"]["ground_range_m"] == 50.0

    def test_build_detection_record_unknown_class(self, sample_fix, sample_dims):
        det = {"class_id": 99, "confidence": 0.5, "bbox_px": [0,0,10,10]}
        record = build_detection_record(det, sample_fix, sample_dims, "det_99")
        assert record["class"] == "class_99"

    def test_build_detection_record_string_class(self, sample_fix, sample_dims):
        det = {"class": "custom_debris", "confidence": 0.5, "bbox_px": [0,0,10,10]}
        record = build_detection_record(det, sample_fix, sample_dims, "det_custom")
        assert record["class"] == "custom_debris"


class TestReportGeneration:
    @pytest.fixture
    def sample_records(self, sample_detection, sample_fix, sample_dims) -> list[Dict]:
        return [
            build_detection_record(sample_detection, sample_fix, sample_dims, "det_001"),
            build_detection_record(
                {"class_id": 1, "confidence": 0.9, "bbox_px": [100, 200, 50, 50]},
                {"lat": 46.0, "lon": -74.0, "ping_start": 200, "ping_end": 200},
                {"length_m": 10.0, "width_m": 5.0, "height_m": float('nan')},
                "det_002"
            )
        ]

    def test_generate_json(self, sample_records):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "report.json")
            generate_report(sample_records, path, fmt="json")
            
            assert os.path.exists(path)
            with open(path) as f:
                data = json.load(f)
                
            assert "metadata" in data
            assert data["metadata"]["total_detections"] == 2
            assert "detections" in data
            assert len(data["detections"]) == 2
            
            # Check sanitization (NaN -> null)
            assert data["detections"][1]["height_m"] is None

    def test_generate_csv(self, sample_records):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "report.csv")
            generate_report(sample_records, path, fmt="csv")
            
            assert os.path.exists(path)
            df = pd.read_csv(path)
            
            assert len(df) == 2
            assert list(df.columns[:3]) == ["id", "class", "confidence"]
            assert "bbox_x" in df.columns
            assert df.iloc[0]["bbox_x"] == 10
            # NaN is empty string in CSV, so pandas reads as NaN
            assert pd.isna(df.iloc[1]["height_m"])

    def test_generate_geojson(self, sample_records):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "report.geojson")
            generate_report(sample_records, path, fmt="geojson")
            
            assert os.path.exists(path)
            with open(path) as f:
                data = geojson.load(f)
                
            assert data.type == "FeatureCollection"
            assert len(data.features) == 2
            
            f1 = data.features[0]
            assert f1.type == "Feature"
            assert f1.geometry.type == "Point"
            assert f1.geometry.coordinates == [-73.456, 45.123] # [lon, lat]
            assert f1.properties["id"] == "det_001"
            assert f1.properties["class"] == "shipwreck"


class TestMergeDetections:
    def test_merge_overlapping(self):
        # Create two overlapping detections of the SAME class
        det1 = {
            "id": "det_1",
            "class": "shipwreck",
            "confidence": 0.9,
            "bbox_px": [0, 0, 100, 100], # w=100, h=100
            "lat": 0.0,
            "lon": 0.0,
            "_extra": {"meters_per_pixel": 1.0, "meters_per_pixel_alongtrack": 1.0}
        }
        
        # Second one is very close (10m away), same class, lower confidence
        # 10m is ~0.00009 deg lon
        det2 = {
            "id": "det_2",
            "class": "shipwreck",
            "confidence": 0.5,
            "bbox_px": [10, 10, 100, 100],
            "lat": 0.0,
            "lon": 0.00009,
            "_extra": {"meters_per_pixel": 1.0, "meters_per_pixel_alongtrack": 1.0}
        }
        
        # Third is same location but DIFFERENT class
        det3 = {
            "id": "det_3",
            "class": "ghost_net",
            "confidence": 0.4,
            "bbox_px": [0, 0, 100, 100],
            "lat": 0.0,
            "lon": 0.0,
            "_extra": {"meters_per_pixel": 1.0, "meters_per_pixel_alongtrack": 1.0}
        }
        
        # Fourth is same class, far away
        det4 = {
            "id": "det_4",
            "class": "shipwreck",
            "confidence": 0.8,
            "bbox_px": [0, 0, 100, 100],
            "lat": 1.0,
            "lon": 1.0,
            "_extra": {"meters_per_pixel": 1.0, "meters_per_pixel_alongtrack": 1.0}
        }

        merged = merge_overlapping_detections([det1, det2, det3, det4], iou_threshold=0.3)
        
        # Should keep det1 (highest conf), suppress det2 (same class, overlap), 
        # keep det3 (different class), keep det4 (no overlap)
        assert len(merged) == 3
        ids = [d["id"] for d in merged]
        assert "det_1" in ids
        assert "det_2" not in ids
        assert "det_3" in ids
        assert "det_4" in ids

class TestIntegrationBugFixes:
    def test_waterfall_layback_shifts_coordinates(self):
        with tempfile.TemporaryDirectory() as tmp:
            # Create a mock ping table where ship is at 0,0 moving North (0 deg)
            # If layback is applied, the corrected pos should be South of the ship (negative lat)
            df = pd.DataFrame({
                "ping_index": [0, 1, 2],
                "ping_number": [100, 101, 102],
                "lat": [0.0, 0.0, 0.0],
                "lon": [0.0, 0.0, 0.0],
                "heading": [0.0, 0.0, 0.0],
                "timestamp": ["12:00:00", "12:00:01", "12:00:02"],
                "altitude_m": [10.0, 10.0, 10.0],
                "slant_range": [50.0, 50.0, 50.0]
            })
            df.to_csv(os.path.join(tmp, "ping_table.csv"), index=False)
            
            meta = {
                "ping_table_csv": "ping_table.csv",
                "width": 1000,
                "height": 3,
                "row_offset": 0,
                "col_offset": 0,
                "meters_per_pixel": 0.1,
                "meters_per_pixel_alongtrack": 0.1,
                "source_xtf_filename": "test.xtf"
            }
            with open(os.path.join(tmp, "waterfall.json"), "w") as fh:
                json.dump(meta, fh)
                
            yolo = [{"bbox_px": [500, 1, 10, 1]}] # center is 505 (stbd near nadir), cy is 1
            
            # Run without layback
            recs_0 = process_waterfall_detections(yolo, tmp, cable_length_m=0.0)
            lat_0 = recs_0[0]["lat"]
            
            # Run with layback (100m)
            recs_100 = process_waterfall_detections(yolo, tmp, cable_length_m=100.0)
            lat_100 = recs_100[0]["lat"]
            
            assert lat_100 < lat_0, f"Layback 100m (lat {lat_100}) should be South of 0m (lat {lat_0}) when heading North"
            
    def test_square_pixel_ping_spans(self):
        # In square_pixels=True, multiple image rows might map to the same ping index
        # Let's say we resampled so every real ping is duplicated into 2 image rows
        with tempfile.TemporaryDirectory() as tmp:
            df = pd.DataFrame({
                "ping_index": [0, 0, 1, 1, 2, 2],
                "ping_number": [10, 10, 11, 11, 12, 12],
                "lat": [0.0]*6, "lon": [0.0]*6, "heading": [0.0]*6,
                "timestamp": [""]*6, "altitude_m": [10.0]*6, "slant_range": [50.0]*6
            })
            df.to_csv(os.path.join(tmp, "ping_table.csv"), index=False)
            
            meta = {
                "ping_table_csv": "ping_table.csv",
                "width": 100,
                "height": 6,
                "row_offset": 0,
                "col_offset": 0,
                "meters_per_pixel": 0.1,
                "meters_per_pixel_alongtrack": 0.1,
                "source_xtf_filename": "test.xtf"
            }
            with open(os.path.join(tmp, "waterfall.json"), "w") as fh:
                json.dump(meta, fh)
                
            # Detection spanning image rows 1 to 4
            # Row 1 -> ping 0
            # Row 4 -> ping 2
            yolo = [{"bbox_px": [50, 1, 10, 4]}] 
            
            recs = process_waterfall_detections(yolo, tmp, cable_length_m=0.0)
            assert recs[0]["ping_start"] == 0
            assert recs[0]["ping_end"] == 2

