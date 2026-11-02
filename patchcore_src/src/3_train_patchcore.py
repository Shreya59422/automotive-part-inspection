from pathlib import Path

from anomalib.data import MVTecAD
from anomalib.models import Patchcore
from anomalib.engine import Engine

# Dataset
datamodule = MVTecAD(
    root=Path("data/mvtec"),
    category="bottle",
    train_batch_size=8,
    eval_batch_size=8,
    num_workers=0,
)

# Model
model = Patchcore()

# Training and testing
engine = Engine(default_root_dir="results/anomaly_detection")

engine.fit(
    model=model,
    datamodule=datamodule,
)

engine.test(
    model=model,
    datamodule=datamodule,
)

print("PatchCore training and testing completed.")