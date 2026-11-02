
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

# Checkpoint
checkpoint_path = (
    "results/anomaly_detection/Patchcore/MVTecAD/"
    "bottle/v0/weights/lightning/model.ckpt"
)

# Engine
engine = Engine(default_root_dir="results/anomaly_detection")

# Predict
predictions = engine.predict(
    model=model,
    datamodule=datamodule,
    ckpt_path=checkpoint_path,
)

print("\nPrediction output type:", type(predictions))

if predictions:
    first_batch = predictions[0]

    print("\nFirst prediction type:", type(first_batch))

    if isinstance(first_batch, list):
        first_prediction = first_batch[0]
    else:
        first_prediction = first_batch

    print("\nPrediction object:")
    print(first_prediction)

    print("\nAvailable fields:")
    print(
        [
            name
            for name in dir(first_prediction)
            if not name.startswith("_")
        ]
    )

print("\nInspection completed.")
