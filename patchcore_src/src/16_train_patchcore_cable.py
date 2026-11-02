
from pathlib import Path

from anomalib.data import MVTecAD
from anomalib.models import Patchcore
from anomalib.engine import Engine


# Cable dataset
datamodule = MVTecAD(
    root=Path("data/mvtec"),
    category="cable",
    train_batch_size=8,
    eval_batch_size=8,
    num_workers=0,
)

# PatchCore model
model = Patchcore()

# Separate results for Cable
engine = Engine(
    default_root_dir="results/anomaly_detection"
)

# Train Cable model
engine.fit(
    model=model,
    datamodule=datamodule,
)

# Test Cable model
engine.test(
    model=model,
    datamodule=datamodule,
)

print("Cable PatchCore training and testing completed.")
