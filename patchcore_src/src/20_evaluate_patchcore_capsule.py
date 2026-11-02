
from pathlib import Path

from anomalib.data import MVTecAD
from anomalib.models import Patchcore
from anomalib.engine import Engine


# Dataset
datamodule = MVTecAD(
    root=Path("data/mvtec"),
    category="capsule",
    train_batch_size=8,
    eval_batch_size=8,
    num_workers=0,
)

# Model
model = Patchcore()

# Load trained checkpoint
checkpoint = Path(
    "results/anomaly_detection/Patchcore/MVTecAD/"
    "capsule/v0/weights/lightning/model.ckpt"
)

# Engine
engine = Engine(
    default_root_dir="results/anomaly_detection"
)

print("Evaluating Capsule PatchCore checkpoint...")

engine.test(
    model=model,
    datamodule=datamodule,
    ckpt_path=str(checkpoint),
)

print("Capsule evaluation completed.")
