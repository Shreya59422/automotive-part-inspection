
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Set PATCHCORE_CHECKPOINT to your model's actual location.
checkpoint_env = os.getenv("PATCHCORE_CHECKPOINT")

PATCHCORE_CHECKPOINT = (
    Path(checkpoint_env).expanduser()
    if checkpoint_env
    else PROJECT_ROOT / "models" / "patchcore" / "model.ckpt"
)
