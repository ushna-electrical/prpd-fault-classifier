r"""FastAPI backend for the PRPD fault classifier web tool.

Local development:

    cd Code/webapp/backend
    ..\..\.venv\Scripts\uvicorn app.main:app --reload --port 8000

Every API route is under /api (so a reverse proxy can route "/api/*" to this
process and serve the frontend's static files for everything else from one
origin). Deployment-specific settings are read from environment variables,
all optional -- unset, they reproduce the original local-dev layout:

    PRPD_CORS_ORIGINS        comma-separated allowed origins (default "*")
    PRPD_LIBRARY_IMAGES_DIR  Fault Library example images (default: the bundled
                              copy at Code/webapp/library_images, falling back to
                              ../Table 5.5 next to Code/ if that's not present)
    PRPD_STAGE2_DIR          dashboard source files (default: Code/stage2_output)
    PRPD_SERVE_FRONTEND      "1" to also serve webapp/frontend/ as static files
                              at "/" from this same process (handy for a
                              single-process deployment with no separate static
                              host/Nginx needed); default off

See Code/webapp/README.md and Code/webapp/DEPLOYMENT.md for full instructions.
"""

from __future__ import annotations

import io
import os
import tempfile
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles
from PIL import Image, UnidentifiedImageError

from . import inference, pdf_report, results_data

SENSOR_NAMES = {
    "1": "UHF Antenna (Ch1)",
    "2": "Coupling Capacitor (IEC Ch2)",
    "3": "HFCT (IEC Ch3)",
}

app = FastAPI(title="PRPD Fault Classifier API")
api = APIRouter(prefix="/api")

# Same-origin deployments (one reverse proxy, or PRPD_SERVE_FRONTEND=1) don't
# need CORS at all; split deployments (e.g. frontend on Vercel, backend on a
# VPS) should set PRPD_CORS_ORIGINS to the frontend's real origin(s) instead
# of leaving this at "*". Comma-separated, e.g. "https://my-app.vercel.app".
_cors_origins = [o.strip() for o in os.environ.get("PRPD_CORS_ORIGINS", "*").split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Fault Library example images: the curated Table 5.5 set (one image per fault
# per sensor). Bundled into the repo at webapp/library_images/ (a copy, so
# deployments that only have Code/ -- e.g. Render -- still have it); falls
# back to the original external ../Table 5.5/ for local dev if the bundled
# copy isn't there, and PRPD_LIBRARY_IMAGES_DIR overrides both.
#
# Mounted on `app` directly (full "/api/..." path spelled out), not on the
# `api` router: APIRouter.include_router() does not apply its prefix to Mount
# sub-apps (a FastAPI/Starlette limitation, confirmed by testing -- a Mount
# added via api.mount("/x", ...) still lands at "/x", not "/api/x").
_library_images_env = os.environ.get("PRPD_LIBRARY_IMAGES_DIR")
_bundled_library_images = inference.CODE_ROOT / "webapp" / "library_images"
if _library_images_env:
    TABLE_5_5_DIR = Path(_library_images_env)
elif _bundled_library_images.is_dir():
    TABLE_5_5_DIR = _bundled_library_images
else:
    TABLE_5_5_DIR = inference.CODE_ROOT.parent / "Table 5.5"
if TABLE_5_5_DIR.is_dir():
    app.mount("/api/library-images", StaticFiles(directory=str(TABLE_5_5_DIR)), name="library-images")

# Thesis result files (CSV/JSON/PNG) backing the Results Dashboard, so every
# number the dashboard shows can be opened/downloaded from its real source file.
_stage2_env = os.environ.get("PRPD_STAGE2_DIR")
STAGE2_DIR = Path(_stage2_env) if _stage2_env else inference.CODE_ROOT / "stage2_output"
if STAGE2_DIR.is_dir():
    app.mount("/api/stage2-output", StaticFiles(directory=str(STAGE2_DIR)), name="stage2-output")

MAX_UPLOAD_BYTES = 15 * 1024 * 1024  # 15 MB
VALID_SENSORS = {"1", "2", "3"}


@api.get("/health")
def health():
    return {"status": "ok"}


@api.get("/classes")
def classes(sensor: str = "2"):
    if sensor not in VALID_SENSORS:
        raise HTTPException(400, "sensor must be 1, 2 or 3")
    model = inference.get_model(sensor)
    return {
        "sensor": sensor,
        "classes": [
            {"code": c, "display_name": inference.DISPLAY_NAMES.get(c, c)}
            for c in model.classes
        ],
        "expected_image_size": list(model.expected_size),
    }


@api.get("/dashboard/sensor-comparison")
def dashboard_sensor_comparison():
    return results_data.sensor_comparison()


@api.get("/dashboard/confusion-matrix")
def dashboard_confusion_matrix():
    return results_data.ensemble_confusion_matrix()


@api.get("/dashboard/calibration")
def dashboard_calibration():
    return results_data.calibration()


@api.post("/predict")
async def predict(
    file: UploadFile = File(...),
    sensor: str = Form("2"),
    voltage: Optional[float] = Form(None),
    tie_margin: float = Form(0.15),
    top_k: int = Form(3),
):
    raw, filename = await _read_upload(file)
    result, check = _run_prediction(raw, filename, sensor, voltage, tie_margin)
    return _format_response(result, check, sensor, top_k)


@api.post("/report")
async def report(
    file: UploadFile = File(...),
    sensor: str = Form("2"),
    voltage: Optional[float] = Form(None),
    tie_margin: float = Form(0.15),
):
    raw, filename = await _read_upload(file)
    result, check = _run_prediction(raw, filename, sensor, voltage, tie_margin)
    formatted = _format_response(result, check, sensor, top_k=3)
    pdf_bytes = pdf_report.build_report(raw, formatted, SENSOR_NAMES.get(sensor, f"Sensor {sensor}"))
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": 'attachment; filename="prpd_classification_report.pdf"'},
    )


async def _read_upload(file: UploadFile) -> tuple[bytes, str]:
    if file.content_type not in ("image/png", "image/jpeg"):
        raise HTTPException(400, f"unsupported content type: {file.content_type} (expected PNG or JPEG)")
    raw = await file.read()
    if not raw:
        raise HTTPException(400, "empty file")
    if len(raw) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, f"file exceeds the {MAX_UPLOAD_BYTES // (1024 * 1024)} MB limit")
    try:
        Image.open(io.BytesIO(raw)).verify()
    except UnidentifiedImageError:
        raise HTTPException(400, "file is not a readable image (expected a PNG/JPG PRPD plot)")
    return raw, file.filename or "upload.png"


def _run_prediction(raw: bytes, filename: str, sensor: str, voltage: Optional[float], tie_margin: float):
    if sensor not in VALID_SENSORS:
        raise HTTPException(400, "sensor must be 1, 2 or 3")
    model = inference.get_model(sensor)

    tmp_path = Path(tempfile.NamedTemporaryFile(delete=False, suffix=_suffix(filename)).name)
    tmp_path.write_bytes(raw)
    try:
        result = model.predict(tmp_path)
    finally:
        tmp_path.unlink(missing_ok=True)

    check = model.voltage_check(result, voltage, tie_margin) if voltage is not None else None
    return result, check


def _suffix(filename: Optional[str]) -> str:
    return Path(filename or "upload.png").suffix or ".png"


def _format_response(result: dict, check: Optional[dict], sensor: str, top_k: int) -> dict:
    fault = result["fault"]
    final_fault = check["final"] if check else fault
    return {
        "sensor": sensor,
        "fault": fault,
        "fault_display": inference.DISPLAY_NAMES.get(fault, fault),
        "confidence": result["confidence"],
        "confidence_tier": inference.confidence_tier(result["confidence"]),
        "top": [
            {"fault": c, "fault_display": inference.DISPLAY_NAMES.get(c, c), "probability": p}
            for c, p in result["top"][:top_k]
        ],
        "probs": {c: p for c, p in result["probs"].items()},
        "prototypical": {"fault": result["prototypical"][0], "probability": result["prototypical"][1]},
        "resnet18": {"fault": result["resnet18"][0], "probability": result["resnet18"][1]},
        "models_agree": result["prototypical"][0] == result["resnet18"][0],
        "image_size": list(result["size"]),
        "size_matches_training": result["size_ok"],
        "voltage_check": check,
        "final_fault": final_fault,
        "final_fault_display": inference.DISPLAY_NAMES.get(final_fault, final_fault),
    }


app.include_router(api)

# Optional: serve the static frontend from this same process (one process, one
# port, no separate static host needed). Mounted last and at "/" so it only
# catches requests the /api routes above didn't. Off by default so the
# existing "two separate servers" local-dev workflow is unaffected.
if os.environ.get("PRPD_SERVE_FRONTEND") == "1":
    _frontend_env = os.environ.get("PRPD_FRONTEND_DIR")
    FRONTEND_DIR = Path(_frontend_env) if _frontend_env else inference.CODE_ROOT / "webapp" / "frontend"
    if FRONTEND_DIR.is_dir():
        app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")
