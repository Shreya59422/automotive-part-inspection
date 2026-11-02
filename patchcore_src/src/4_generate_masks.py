
from pathlib import Path

import numpy as np
from PIL import Image

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

# Output folders
mask_dir = Path("results/prediction_masks")
map_dir = Path("results/anomaly_maps")

mask_dir.mkdir(parents=True, exist_ok=True)
map_dir.mkdir(parents=True, exist_ok=True)

# Save predictions
saved_masks = 0
saved_maps = 0

if predictions:
    for batch in predictions:
        for index in range(batch.batch_size):

            image_path = Path(batch.image_path[index])
            category = image_path.parent.name
            filename = image_path.stem

            # Save binary prediction mask
            predicted_mask = batch.pred_mask[index].cpu().numpy()

            predicted_mask = np.squeeze(predicted_mask)
            predicted_mask = predicted_mask.astype(np.uint8) * 255

            output_category = mask_dir / category
            output_category.mkdir(parents=True, exist_ok=True)

            Image.fromarray(predicted_mask).save(
                output_category / f"{filename}_pred.png"
            )

            saved_masks += 1

            # Save anomaly map
            anomaly_map = batch.anomaly_map[index].cpu().numpy()

            anomaly_map = np.squeeze(anomaly_map)

            # Normalize anomaly map for visualization
            anomaly_min = anomaly_map.min()
            anomaly_max = anomaly_map.max()

            if anomaly_max > anomaly_min:
                anomaly_map = (
                    (anomaly_map - anomaly_min)
                    / (anomaly_max - anomaly_min)
                ) * 255
            else:
                anomaly_map = np.zeros_like(anomaly_map)

            anomaly_map = anomaly_map.astype(np.uint8)

            output_map_category = map_dir / category
            output_map_category.mkdir(
                parents=True,
                exist_ok=True,
            )

            Image.fromarray(anomaly_map).save(
                output_map_category / f"{filename}_anomaly.png"
            )

            saved_maps += 1

print("\nPrediction completed.")
print("Binary masks saved:", saved_masks)
print("Anomaly maps saved:", saved_maps)
print("Mask folder:", mask_dir)
print("Anomaly map folder:", map_dir)
