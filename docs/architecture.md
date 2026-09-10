# System Architecture

## High-level flow

```text
User
  |
  v
React Dashboard (Vite, :5173)
  |  POST /upload            multipart XTF
  |  GET  /status/{job_id}   polled until completed
  |  GET  /waterfall/{job_id}
  |  GET  /results/{job_id}
  v
FastAPI Backend (uvicorn, :8000)
  |
  v
Async job runner
  |
  +--> Stage 1  Read XTF
  |      pyxtf parses ping packets
  |      -> ping metadata table (time, lat, lon, heading, altitude, slant range)
  |      -> raw waterfall array (port + starboard stacked, nadir centred)
  |
  +--> Stage 2  Clean
  |      bottom tracking / nadir removal
  |      slant range -> ground range projection
  |      gain normalisation
  |      log transform + Lee filter despeckling
  |
  +--> Stage 3  Tile
  |      640x640 tiles, 20% overlap
  |      -> one JSON sidecar per tile
  |
  +--> Stage 4  Detect
  |      YOLO11n-seg inference per tile
  |      cross-tile de-duplication by geographic IoU
  |
  +--> Stage 5  Georeference
         invert tiling -> originating ping
         offset sonar fix by ground range, perpendicular to heading
         convert pixels to metres per axis
         |
         v
       Detection records (JSON / CSV / GeoJSON)
         |
         v
       React Dashboard: table, map, waterfall viewer
```

## Components

### Frontend
React 18 served by Vite from `src/frontend/`. Handles file upload, polls job status, and renders
three views of a completed job: the processed waterfall image, a sortable
detection table, and a Leaflet map with a marker per detection. Report export
to JSON, CSV and GeoJSON happens client-side.

The frontend calls the backend over an absolute URL (`VITE_API_BASE_URL`,
default `http://localhost:8000`), so the two run as independent processes and
the backend enables CORS.

### Backend API
FastAPI application in `src/backend/main.py`, exposing four endpoints. It is started from inside `src/backend/`, so `models/`, `uploads/` and `outputs/` all resolve inside that folder and the service is self-contained. Upload returns `202 Accepted` with
a `job_id` immediately and runs the pipeline in the background, so a large
survey file does not block the request. Job state is tracked in memory.

### Preprocessing pipeline
Three stages, each a set of importable functions rather than a script, so the
API calls them directly.

**Stage 1 — Read.** Parses the XTF and builds the ping metadata table that
everything downstream keys off. Channel selection is automatic: each sonar
channel is scored for usability and for the direction its samples are stored in.
This matters in practice — on the validation file the 410 kHz channel pair is
saturated and carries no usable signal, and the port channel stores its samples
far-range-first. Choosing channels naively by frequency would image noise.

**Stage 2 — Clean.** Four corrections in order:

1. *Bottom tracking and nadir removal.* Locates the first seabed return in each
   ping and crops the water-column blind zone. Altitude is measured from the
   imagery rather than read from the file header, because the recorded altitude
   is zero or physically implausible on many pings.
2. *Slant range to ground range.* Each sample is projected onto the seabed with
   `ground = sqrt(slant² − altitude²)` and resampled onto a shared grid, giving
   a constant metres-per-pixel across track.
3. *Gain normalisation.* The survey's average across-track intensity profile is
   divided out so brightness is even from near to far range.
4. *Despeckling.* Sonar speckle is multiplicative, so a log transform converts
   it to additive noise before a Lee filter is applied — this suppresses grain
   in flat areas while leaving genuine edges and targets intact.

**Stage 3 — Tile.** Cuts 640×640 tiles with 20% overlap so an object straddling
a boundary survives whole in at least one tile, and writes a JSON sidecar per
tile recording its offsets, its ping span, its metres-per-pixel and its source
file. Tiles that are mostly no-data or built from faulty pings are skipped.

### Machine Learning Model
YOLO11n-seg, fine-tuned for 50 epochs at 640 px input. Inference runs per tile.
Because tiles overlap, objects near a boundary are detected more than once, so
detections are merged by geographic intersection-over-union rather than by
tile-local pixel overlap.

### Georeferencing and reporting
Converts model output into the shared detection schema. Two details carry most
of the accuracy:

- **Position.** A detection is offset from the sonar's own GPS fix by its ground
  range, perpendicular to the platform heading, on whichever side of the track
  it falls. Reporting the platform fix directly would stack every detection onto
  the survey line — an error of up to a full swath width.
- **Size.** Along-track and across-track scales are different numbers: across
  track a pixel is a resampled range bin, along track it is however far the
  platform moved between pings. Each axis is converted with its own scale.
  Object height is derived from acoustic shadow geometry.

`pyproj` provides WGS84 geodesic calculations and towfish layback correction for
final report coordinates.

## The traceability invariant

Every array in the pipeline maintains **image row `i` = ping table row `i`**.
Each transformation that changes the row count resamples the ping table
alongside the pixels. This is what allows any pixel in any tile to be resolved
back to the exact ping that produced it, and is the reason a detection can carry
a real coordinate and a real size rather than only a pixel box.

## Data format

Input is XTF (eXtended Triton Format), the standard container for side-scan
sonar. Each ping packet carries the acoustic return for every channel plus the
platform's navigation state at that instant, which is what makes georeferencing
possible without a separate navigation file.
