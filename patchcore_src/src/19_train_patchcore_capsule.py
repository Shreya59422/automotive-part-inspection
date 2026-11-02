
from pathlib import Path

from anomalib.data import MVTecAD
from anomalib.models import Patchcore
from anomalib.engine import Engine


# -----------------------------
# 1. Dataset configuration
# -----------------------------

datamodule = MVTecAD(
    root=Path("data/mvtec"),
    category="capsule",
    train_batch_size=8,
    eval_batch_size=8,
    num_workers=0,
)


# -----------------------------
# 2. PatchCore model
# -----------------------------

model = Patchcore()


# -----------------------------
# 3. Training and testing
# -----------------------------

engine = Engine(
    default_root_dir="results/anomaly_detection"
)

print("Starting Capsule PatchCore training...")

engine.fit(
    model=model,
    datamodule=datamodule,
)

print("Capsule training completed.")

print("Starting Capsule PatchCore testing...")

engine.test(
    model=model,
    datamodule=datamodule,
)

print("Capsule PatchCore training and testing completed.")
