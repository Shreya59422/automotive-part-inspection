
from pathlib import Path

from anomalib.data import MVTecAD
from anomalib.models import Patchcore
from anomalib.engine import Engine


# Dataset configuration
datamodule = MVTecAD(
    root=Path("data/mvtec"),
    category="carpet",
    train_batch_size=8,
    eval_batch_size=8,
    num_workers=0,
)


# PatchCore model
model = Patchcore()


# Training and testing engine
engine = Engine(
    default_root_dir="results/anomaly_detection"
)


# Train
print("Starting Carpet PatchCore training...")

engine.fit(
    model=model,
    datamodule=datamodule,
)


# Test
print("Starting Carpet PatchCore testing...")

engine.test(
    model=model,
    datamodule=datamodule,
)


print("Carpet PatchCore training and testing completed.")
