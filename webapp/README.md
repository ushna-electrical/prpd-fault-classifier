# PRPD Fault Classifier &mdash; web tool

A field/presentation web tool built on the thesis **"AI-Based Evaluation of PRPD Patterns for Medium
Voltage Cables"**. It wraps the thesis's selected model (the Sensor-2 equal-weight ensemble of a
prototypical few-shot network and a ResNet-18, on the ten fault classes shared by all three sensors)
behind a classify page, a fault library, a read-only results dashboard, and a one-click PDF report.

Model inference lives entirely in the Python backend (`backend/`); the frontend (`frontend/`) is plain
HTML/CSS/JS with no build step and talks to the backend over HTTP.

For anything beyond local development (a VPS, a subdomain, Vercel for the frontend, etc.), see
[`DEPLOYMENT.md`](DEPLOYMENT.md).

## Prerequisites

- The parent project's Python virtual environment, already set up at `Code/.venv`, with `torch`,
  `torchvision`, `numpy`, `pillow` and `matplotlib` installed (see `Code/README.md`).
- The exported final models: `Code/final_model_sensor1_10class/`, `..._sensor2_10class/`,
  `..._sensor3_10class/`. If missing, generate them once from `Code/`:

  ```bash
  py export_final_model.py --sensor 1
  py export_final_model.py --sensor 2
  py export_final_model.py --sensor 3
  ```

- `Code/voltage_ranges.json`. If missing: `py build_voltage_ranges.py` (needs `metadata.xlsx` one level
  above `Code/`).
- The data this tool reads but never copies: `../Dataset/` (Fault Library deep-links use the curated
  `../Table 5.5/` set instead), and `Code/stage2_output/` (the dashboard's source files).

## 1. Install backend dependencies

```bash
cd Code/webapp/backend
..\..\.venv\Scripts\pip install -r requirements.txt     # Windows
# or: ../../.venv/bin/pip install -r requirements.txt   # macOS/Linux
```

## 2. (Once) generate the ensemble confusion matrix

The dashboard's confusion-matrix section reads a file that isn't produced anywhere else in the
project &mdash; only the two component models' own confusion matrices existed. Generate it once
(re-run any time the Sensor-2 model is re-exported):

```bash
cd Code
.venv\Scripts\python.exe webapp\backend\scripts\make_ensemble_confusion_matrix.py
```

This evaluates the deployed `final_model_sensor2_10class` ensemble on its held-out test split and
writes `stage2_output/ensemble_confusion_matrix_sensor2_merged23_10class_*` (counts CSV, row-percent
CSV, per-class precision/recall/F1 CSV, and a 300 DPI PNG). It prints the resulting accuracy/macro-F1
so you can confirm they still land at 0.863/0.870 (matching `stage2_output/results_sensor_2_merged23.txt`).

## 3. Run the backend

```bash
cd Code/webapp/backend
..\..\.venv\Scripts\python.exe -m uvicorn app.main:app --reload --port 8000
```

Check it's up: `http://127.0.0.1:8000/api/health` should return `{"status":"ok"}`.

Every route lives under `/api` (so a reverse proxy only needs one rule to route API traffic to this
process &mdash; see `DEPLOYMENT.md`):

| Method | Path | Purpose |
|---|---|---|
| GET | `/api/health` | liveness check |
| GET | `/api/classes?sensor=2` | class list + display names for a sensor's model |
| POST | `/api/predict` | classify an uploaded image (`file`, `sensor`, optional `voltage`, `tie_margin`, `top_k`) |
| POST | `/api/report` | same inputs as `/predict`, returns a PDF report |
| GET | `/api/dashboard/sensor-comparison` | S1/S2/S3 comparison data |
| GET | `/api/dashboard/confusion-matrix` | Sensor-2 ensemble confusion matrix + per-class metrics |
| GET | `/api/dashboard/calibration` | ECE/MCE/Brier/NLL, reliability bins, coverage-vs-accuracy |
| GET | `/api/library-images/<file>` | static Fault Library example images (from `../Table 5.5/`) |
| GET | `/api/stage2-output/<file>` | static source files backing the dashboard (CSV/JSON/PNG) |

A model only works on its own sensor's images; `sensor` must be `"1"`, `"2"` or `"3"`.

## 4. Serve the frontend

Any static file server pointed at `frontend/` works. For local use, from the `Ushna/` directory (one
level above `Code/`, so the Fault Library's drag-and-drop test images under `Dataset/` are also
reachable if you use them for testing):

```bash
cd Ushna
python -m http.server 5500
```

Then open `http://127.0.0.1:5500/Code/webapp/frontend/index.html`.

The frontend expects the backend at `http://127.0.0.1:8000` by default &mdash; that's set in
`frontend/js/config.js` (`window.PRPD_CONFIG.API_BASE_URL`), the one file to edit if you run the
backend elsewhere (see `DEPLOYMENT.md` for the same-origin and split-deployment cases).

## Running the tests

```bash
cd Code/webapp/backend
..\..\.venv\Scripts\python.exe -m pytest tests/ -v
```

Covers: image preprocessing (crop boxes, output shapes, no augmentation at inference time, training
image size), class-label mapping (the ten shared classes, the fault_3 -> F2 merge, display names,
`resnet_keep_idx` consistency), and the threshold logic (the three confidence tiers, and the real
`voltage_check` tie-break across several tie/no-tie/out-of-range scenarios).

## Project structure

```
webapp/
  DEPLOYMENT.md          VPS / split-with-Vercel / env var reference
  deploy/                Nginx + systemd templates for a VPS deployment
  backend/
    app/
      main.py          FastAPI app and routes, all under /api; reads PRPD_* env vars for deployment
      inference.py      wraps test_model.FinalModel (Code/test_model.py), adds display names + confidence tiers
      results_data.py   reads stage2_output/ files for the dashboard, with source paths attached
      pdf_report.py     one-click classification PDF
    scripts/
      make_ensemble_confusion_matrix.py   generates the dashboard's confusion-matrix source files
    tests/               pytest suite (see above)
    requirements.txt
  frontend/
    vercel.json           makes this folder deployable to Vercel as-is
    index.html           Classify (two-pane: upload left, results right)
    library.html          Fault Library (10 model classes, hash-routed, e.g. #fault_7)
    dashboard.html         Results Dashboard (read-only, every number links to its source file)
    about.html              Project summary, objectives, method, limitations (thesis wording)
    css/
    js/
      config.js            the one file to edit per deployment (backend URL)
```

## Notes

- Inference reuses `Code/test_model.py`'s `FinalModel` class as-is; the backend does not
  reimplement preprocessing or the ensemble/voltage logic.
- The dashboard never hardcodes a result number in JavaScript &mdash; every value is fetched from the
  backend, which reads it from a real file under `stage2_output/` (or recomputes it live from saved
  raw probabilities, for the reliability diagram) and returns the source path alongside it.
- No dates appear anywhere in the UI or the PDF report, per the project's own requirement.
- Open-set detection (flagging images of conditions outside the ten modelled classes) is out of scope,
  matching the thesis.
