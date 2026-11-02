
from pathlib import Path

from anomalib.data import MVTecAD
from anomalib.models import Patchcore
from anomalib.engine import Engine


# Hazelnut dataset
datamodule = MVTecAD(
    root=Path("data/mvtec"),
    category="hazelnut",
    train_batch_size=8,
    eval_batch_size=8,
    num_workers=0,
)


# PatchCore model
model = Patchcore()


# Separate results for Hazelnut
engine = Engine(
    default_root_dir="results/anomaly_detection"
)


# Train Hazelnut model
engine.fit(
    model=model,
    datamodule=datamodule,
)


# Test Hazelnut model
engine.test(
    model=model,
    datamodule=datamodule,
)


print("Hazelnut PatchCore training and testing completed.")
