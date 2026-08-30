import uuid
from typing import List
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI(title="Sonar Debris Detection API")

# Enable CORS for frontend (B3)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Day 1 Frozen Contract Schema
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

# In-memory job state tracking
jobs = {}

@app.post("/upload")
async def upload_sonar_file(file: UploadFile = File(...)):
    job_id = str(uuid.uuid4())
    jobs[job_id] = "completed"
    return {
        "job_id": job_id,
        "filename": file.filename,
        "status": "completed",
        "message": "File received successfully"
    }

@app.get("/status/{job_id}")
async def get_job_status(job_id: str):
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job ID not found")
    return {"job_id": job_id, "status": jobs[job_id]}

@app.get("/results/{job_id}", response_model=List[DetectionItem])
async def get_results(job_id: str):
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job ID not found")
    
    return [
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