import os
import uuid
import asyncio
import shutil
from typing import List, Dict, Any
from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI(title="Sonar Debris Detection API")

# Enable CORS for B3's frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Step 1: Storage directories
UPLOAD_DIR = "uploads"
OUTPUT_DIR = "outputs"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Step 2: Frozen Schema Contract
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

# Step 3: In-Memory Job Registry
jobs_db: Dict[str, Dict[str, Any]] = {}

# Step 4: Background Worker
async def run_pipeline_task(job_id: str, file_path: str):
    try:
        jobs_db[job_id]["status"] = "processing"
        
        # Simulated pipeline delay (Days 3-4: replace with real pipeline calls)
        await asyncio.sleep(5)
        
        # Mock detection result payload matching the contract
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
            },
            {
                "id": 2,
                "class_name": "ghost_net",
                "confidence": 0.81,
                "bbox_px": [410, 600, 470, 680],
                "ping_start": 1450,
                "ping_end": 1490,
                "lat": 18.9255,
                "lon": 72.8390,
                "length_m": 8.0,
                "width_m": 4.5,
                "height_m": 1.2,
                "shadow_len_px": 30
            }
        ]
        jobs_db[job_id]["status"] = "completed"
        
    except Exception as exc:
        jobs_db[job_id]["status"] = "failed"
        jobs_db[job_id]["error"] = str(exc)

# Step 5: Routes
@app.post("/upload", status_code=status.HTTP_202_ACCEPTED)
async def upload_sonar_file(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    job_id = str(uuid.uuid4())
    saved_file_path = os.path.join(UPLOAD_DIR, f"{job_id}_{file.filename}")
    
    # Stream and save file to local disk
    with open(saved_file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    jobs_db[job_id] = {
        "status": "queued",
        "filename": file.filename,
        "file_path": saved_file_path,
        "results": [],
        "error": None
    }
    
    # Enqueue background execution
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
            detail=f"Job processing failed: {jobs_db[job_id].get('error')}"
        )
        
    return jobs_db[job_id]["results"]