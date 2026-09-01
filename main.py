import os
import uuid
import asyncio
import shutil
from typing import List, Dict, Any
from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Import Pair 2's preprocessing pipeline
from core.preprocess import preprocess_xtf

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
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

class DetectionItem(BaseModel):
    id: int
    class_name: str
    confidence: float
    bbox_px: List[int]
    ping_start: int
    ping_end: int
    lat: float
    lon: float
    length_m: float
    width_m: float
    height_m: float
    shadow_len_px: int

jobs_db: Dict[str, Dict[str, Any]] = {}

def execute_preprocessing(file_path: str, job_out_dir: str):
    """Synchronous CPU worker running Pair 2's preprocessing."""
    return preprocess_xtf(file_path, out_dir=job_out_dir, tile_size=640, overlap=0.2)

async def run_pipeline_task(job_id: str, file_path: str):
    try:
        jobs_db[job_id]["status"] = "processing"
        
        job_out_dir = os.path.join(OUTPUT_DIR, job_id)
        os.makedirs(job_out_dir, exist_ok=True)
        
        # Run Pair 2's preprocessing in a background thread
        res = await asyncio.to_thread(execute_preprocessing, file_path, job_out_dir)
        
        # Store metadata extracted by Pair 2
        jobs_db[job_id]["metadata"] = {
            "pings": len(res["ping_table"]),
            "meters_per_pixel": float(res["meters_per_pixel"]),
            "tile_count": len(res["tile_index"].tiles),
            "tile_dir": str(res["tile_dir"]),
            "pngs": res.get("pngs", {})
        }
        
        # Placeholder mock detections (Pair 1 YOLO will replace this on Day 4)
        jobs_db[job_id]["results"] = [
            {
                "id": 1,
                "class_name": "shipwreck",
                "confidence": 0.94,
                "bbox_px": [120, 340, 260, 510],
                "ping_start": 1020,
                "ping_end": 1085,
                "lat": 18.9220,
                "lon": 72.8347,
                "length_m": 24.5,
                "width_m": 6.2,
                "height_m": 3.1,
                "shadow_len_px": 85
            }
        ]
        
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