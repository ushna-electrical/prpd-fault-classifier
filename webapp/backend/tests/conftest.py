"""Makes `app` (and the Code/ packages it imports) importable regardless of the
directory pytest is invoked from."""

import sys
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[1]  # webapp/backend/
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

# Importing app.inference also pushes Code/ onto sys.path (needed for
# prpd_fewshot/prpd_resnet/prpd_classifier imports used by the tests below).
import app.inference  # noqa: E402,F401
