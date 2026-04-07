from pathlib import Path
import os

# In the Docker container, DT_PROJECT_PATH points to the project root
# Fall back to computing from file location if env var is not set
_project_path = os.environ.get("DT_PROJECT_PATH", None)
if _project_path:
    PROJECT_ROOT = Path(_project_path)
else:
    PROJECT_ROOT = Path(__file__).resolve().parents[4]

MODEL_PATH = PROJECT_ROOT / "assets" / "best.onnx"

CONF_THRESHOLD = 0.2
STOP_DISTANCE = 0.5
FORWARD_PWM = 0.5
