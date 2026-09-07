import os
import math
import uuid
import json
import asyncio
import shutil
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from fastapi.responses import FileResponse
import pandas as pd
from ultralytics import YOLO

# Import Pair 2's preprocessing and reporting pipeline
from core.preprocess import preprocess_xtf
from core.reports import process_tile_detections, merge_overlapping_detections

app = FastAPI(title="Sonar Debris Detection API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "uploads"
OUTPUT_DIR = "outputs"
MODEL_PATH = "models/best.pt"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Pre-load YOLO model at startup
yolo_model = None
if os.path.exists(MODEL_PATH):
    yolo_model = YOLO(MODEL_PATH)

class DetectionItem(BaseModel):
    id: int
    class_name: str
    confidence: float
    bbox_px: List[float]
    ping_start: Optional[int] = None
    ping_end: Optional[int] = None
    lat: float
    lon: float
    length_m: Optional[float] = None
    width_m: Optional[float] = None
    height_m: Optional[float] = None
    shadow_len_px: Optional[int] = None

jobs_db: Dict[str, Dict[str, Any]] = {}

def safe_int(val: Any) -> Optional[int]:
    """Safely converts NaN, None, or float values to int or None."""
    if val is None:
        return None
    try:
        f = float(val)
        return None if math.isnan(f) else int(f)
    except (ValueError, TypeError):
        return None

def safe_float(val: Any, precision: int = 4) -> Optional[float]:
    """Safely converts NaN, None, or numerical values to rounded float or None."""
    if val is None:
        return None
    try:
        f = float(val)
        return None if math.isnan(f) else round(f, precision)
    except (ValueError, TypeError):
        return None

def execute_pipeline(file_path: str, job_out_dir: str) -> Dict[str, Any]:
    """Synchronous CPU worker running preprocessing, YOLO inference, and georeferencing."""
    # 1. Tile the raw XTF sonar imagery
    res = preprocess_xtf(file_path, out_dir=job_out_dir, tile_size=640, overlap=0.2)
    
    tile_dir = res["tile_dir"]
    tile_index = res["tile_index"]
    ping_table = res["ping_table"]
    
    # 2. Run YOLOv8 on all generated tiles
    raw_detections = []
    det_counter = 1
    
    global yolo_model
    if yolo_model is None and os.path.exists(MODEL_PATH):
        yolo_model = YOLO(MODEL_PATH)

    if yolo_model:
        for tile in tile_index.tiles:
            tile_id = str(tile.get("tile_id", ""))
            
            # Dynamically resolve filename / path
            candidate_name = (
                tile.get("filename")
                or tile.get("file_name")
                or tile.get("path")
                or tile.get("tile_path")
                or f"{tile_id}.png"
            )
            
            if os.path.isabs(str(candidate_name)) or os.path.exists(str(candidate_name)):
                tile_img_path = str(candidate_name)
            else:
                tile_img_path = os.path.join(tile_dir, os.path.basename(str(candidate_name)))

            if not os.path.exists(tile_img_path):
                alt_path = os.path.join(tile_dir, f"{tile_id}.png")
                if os.path.exists(alt_path):
                    tile_img_path = alt_path
                else:
                    continue

            results = yolo_model(tile_img_path, verbose=False)
            boxes = results[0].boxes

            yolo_results = []
            for box in boxes:
                cls_id = int(box.cls[0].item())
                conf = float(box.conf[0].item())
                xywh = [float(val) for val in box.xywh[0].tolist()]

                yolo_results.append({
                    "class_id": cls_id,
                    "confidence": conf,
                    "bbox_px": xywh
                })

            if not yolo_results:
                continue

            # 3. Resolve tile pixels to WGS84 geodesic coordinates
            recs = process_tile_detections(
                tile_id=tile_id,
                yolo_results=yolo_results,
                tile_dir=tile_dir,
                ping_table=ping_table,
                id_prefix="det",
                id_offset=det_counter
            )
            raw_detections.extend(recs)
            det_counter += len(recs)

    # 4. Suppress duplicate detections from overlapping tiles
    merged = merge_overlapping_detections(raw_detections, iou_threshold=0.3)
    
    # 5. Format to schema matching both DetectionItem and React frontend with NaN guards
    clean_detections = []
    for idx, d in enumerate(merged, start=1):
        clean_detections.append({
            "id": idx,
            "class": str(d.get("class", "shipwreck")),
            "class_name": str(d.get("class", "shipwreck")),
            "confidence": safe_float(d.get("confidence", 0.0), 3) or 0.0,
            "bbox_px": [safe_float(b, 2) or 0.0 for b in d.get("bbox_px", [0, 0, 0, 0])],
            "ping_start": safe_int(d.get("ping_start")),
            "ping_end": safe_int(d.get("ping_end")),
            "lat": safe_float(d.get("lat", 0.0), 6) or 0.0,
            "lon": safe_float(d.get("lon", 0.0), 6) or 0.0,
            "length_m": safe_float(d.get("length_m")),
            "width_m": safe_float(d.get("width_m")),
            "height_m": safe_float(d.get("height_m")),
            "shadow_len_px": safe_int(d.get("shadow_len_px")),
        })

    return {
        "metadata": {
            "pings": len(ping_table),
            "meters_per_pixel": float(res["meters_per_pixel"]),
            "tile_count": len(tile_index.tiles),
            "tile_dir": str(tile_dir),
            "pngs": res.get("pngs", {})
        },
        "results": clean_detections
    }

async def run_pipeline_task(job_id: str, file_path: str):
    try:
        jobs_db[job_id]["status"] = "processing"
        job_out_dir = os.path.join(OUTPUT_DIR, job_id)
        os.makedirs(job_out_dir, exist_ok=True)

        # Run pipeline in a background worker thread
        output = await asyncio.to_thread(execute_pipeline, file_path, job_out_dir)

        jobs_db[job_id]["metadata"] = output["metadata"]
        jobs_db[job_id]["results"] = output["results"]
        jobs_db[job_id]["status"] = "completed"

    except Exception as exc:
        jobs_db[job_id]["status"] = "failed"
        jobs_db[job_id]["error"] = str(exc)

@app.post("/upload", status_code=status.HTTP_202_ACCEPTED)
async def upload_sonar_file(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    job_id = str(uuid.uuid4())
    saved_file_path = os.path.join(UPLOAD_DIR, f"{job_id}_{file.filename}")

    with open(saved_file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    jobs_db[job_id] = {
        "status": "queued",
        "filename": file.filename,
        "file_path": saved_file_path,
        "results": [],
        "metadata": {},
        "error": None
    }

    background_tasks.add_task(run_pipeline_task, job_id, saved_file_path)

    return {
        "job_id": job_id,
        "filename": file.filename,
        "status": "queued",
        "message": "File accepted and processing queued"
    }

@app.get("/status/{job_id}")
async def get_job_status(job_id: str):
    if job_id not in jobs_db:
        raise HTTPException(status_code=404, detail="Job ID not found")
    return {
        "job_id": job_id,
        "status": jobs_db[job_id]["status"],
        "metadata": jobs_db[job_id].get("metadata"),
        "error": jobs_db[job_id].get("error")
    }

@app.get("/waterfall/{job_id}")
async def get_waterfall(job_id: str):
    if job_id not in jobs_db:
        raise HTTPException(status_code=404, detail="Job ID not found")

    waterfall_path = os.path.join(OUTPUT_DIR, job_id, "waterfall.png")
    if not os.path.isfile(waterfall_path):
        waterfall_path = os.path.join(OUTPUT_DIR, job_id, "waterfall_slant_corrected.png")
        if not os.path.isfile(waterfall_path):
            raise HTTPException(status_code=404, detail="Waterfall image not available")

    return FileResponse(waterfall_path, media_type="image/png", filename="waterfall.png")

@app.get("/results/{job_id}", response_model=List[DetectionItem])
async def get_results(job_id: str):
    if job_id not in jobs_db:
        raise HTTPException(status_code=404, detail="Job ID not found")

    current_status = jobs_db[job_id]["status"]
    if current_status in ["queued", "processing"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Results not ready. Current job status: {current_status}"
        )
    if current_status == "failed":
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Pipeline processing failed: {jobs_db[job_id].get('error')}"
        )

    return jobs_db[job_id]["results"]