
from pathlib import Path

from anomalib.data import MVTecAD
from anomalib.models import Patchcore
from anomalib.engine import Engine

datamodule = MVTecAD(
    root=Path("data/mvtec"),
    category="bottle",
    train_batch_size=8,
    eval_batch_size=8,
    num_workers=0,
)

model = Patchcore()
engine = Engine(default_root_dir="results/anomaly_detection")

checkpoint_path = (
    "results/anomaly_detection/Patchcore/MVTecAD/"
    "bottle/v0/weights/lightning/model.ckpt"
)

predictions = engine.predict(
    model=model,
    datamodule=datamodule,
    ckpt_path=checkpoint_path,
)

print("\nPrediction type:", type(predictions))

if predictions:
    first_batch = predictions[0]

    print("\nFirst batch type:", type(first_batch))
    print("\nAvailable fields:")

    for field in [
        "image_path",
        "pred_score",
        "pred_label",
        "gt_label",
    ]:
        value = getattr(first_batch, field, None)

        if value is not None:
            print(f"\n{field}:")
            print(value)
