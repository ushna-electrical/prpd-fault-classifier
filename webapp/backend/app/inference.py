"""Backend wrapper around the sensor-1/2/3 final models in Code/final_model_sensor<N>_10class/.

Reuses test_model.py's FinalModel and voltage_check logic unchanged (that module
already implements the full ensemble + voltage tie-break pipeline verified in
Stage 0/1); this file only adds what the CLI didn't need: display names and
the 3-tier confidence label for the field tool.
"""

from __future__ import annotations

import sys
from pathlib import Path

CODE_ROOT = Path(__file__).resolve().parents[3]  # app/ -> backend/ -> webapp/ -> Code/
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

import torch  # noqa: E402

# This service only ever runs CPU inference at batch size 1 -- extra intra-op
# threads buy nothing here but each carries its own memory arenas, which
# matters on a RAM-capped host (e.g. Render's free 512MB tier). Set before any
# model loads. OMP_NUM_THREADS/MKL_NUM_THREADS (set in render.yaml) cover the
# BLAS libraries underneath; this covers torch's own thread pool.
torch.set_num_threads(1)

from test_model import FinalModel  # noqa: E402

# Display label per class code. fault_2 absorbs the merged fault_3 label: the
# 10-class models never output "fault_3" (it was folded into fault_2 at data-prep
# time, confirmed against manifests_sensor_2_merged23 and thesis ch. 8), so this
# is purely a label -- there is no runtime remapping to do.
DISPLAY_NAMES: dict[str, str] = {
    "fault_2": "F2 (includes F3)",
    "fault_4": "F4",
    "fault_5": "F5",
    "fault_6": "F6",
    "fault_7": "F7",
    "fault_8": "F8",
    "fault_9": "F9",
    "fault_10": "F10",
    "fault_12": "F12",
    "fault_13": "F13",
}

# Confidence tiers from the thesis's field-tool recommendation
# (chapters/09_calibration.tex, "Confidence Threshold for the Field Tool"):
# accept >= 0.9, review between 0.7 and 0.9, manual inspection below 0.7.
ACCEPT_THRESHOLD = 0.9
REVIEW_THRESHOLD = 0.7


def confidence_tier(confidence: float) -> dict:
    if confidence >= ACCEPT_THRESHOLD:
        return {"tier": "accept", "label": "High confidence"}
    if confidence >= REVIEW_THRESHOLD:
        return {"tier": "review", "label": "Review recommended"}
    return {"tier": "manual", "label": "Low confidence — verify manually"}


_models: dict[str, FinalModel] = {}


def get_model(sensor: str) -> FinalModel:
    """Lazily load and cache each sensor's model (loading is not free)."""
    if sensor not in _models:
        _models[sensor] = FinalModel(sensor)
    return _models[sensor]
