
from pathlib import Path
import numpy as np
from sklearn.metrics import precision_recall_curve

from anomalib.data import MVTecAD
from anomalib.engine import Engine
from anomalib.models import Patchcore

CHECKPOINT = Path(
    "results/screw_synthetic_experiment/Patchcore/"
    "MVTecAD/screw/v0/weights/lightning/model.ckpt"
)

dm = MVTecAD(
    root=Path("data/mvtec"),
    category="screw",
    train_batch_size=8,
    eval_batch_size=1,
    num_workers=0,
    val_split_mode="synthetic",
    val_split_ratio=0.2,
    seed=42,
)

dm.setup()

model = Patchcore.load_from_checkpoint(
    CHECKPOINT, map_location="cpu"
)
model.eval()

engine = Engine(
    default_root_dir="results/validation_threshold_check"
)

predictions = engine.predict(
    model=model,
    dataloaders=dm.val_dataloader(),
)

labels = []
scores = []

for batch in predictions:
    mask = batch.gt_mask.detach().cpu().numpy()
    anomaly = batch.anomaly_map.detach().cpu().numpy()

    for i in range(len(mask)):
        labels.extend((np.squeeze(mask[i]) > 0.5).ravel())
        scores.extend(np.squeeze(anomaly[i]).ravel())

labels = np.asarray(labels, dtype=np.uint8)
scores = np.asarray(scores, dtype=np.float32)

if len(np.unique(labels)) < 2:
    raise RuntimeError("Validation data does not contain both classes.")

precision, recall, thresholds = precision_recall_curve(
    labels, scores
)

f1 = (
    2 * precision[:-1] * recall[:-1]
    / (precision[:-1] + recall[:-1] + 1e-10)
)

best = int(np.argmax(f1))

print("\n========== VALIDATION THRESHOLD DIAGNOSIS ==========")
print("Validation pixels:", len(labels))
print("Defect pixels:", int(labels.sum()))
print("Current saved pixel threshold:", float(
    model.post_processor.pixel_threshold.item()
))
print("Best validation raw-score threshold:", float(thresholds[best]))
print("Validation precision:", round(float(precision[best]), 4))
print("Validation recall:", round(float(recall[best]), 4))
print("Validation F1:", round(float(f1[best]), 4))
print("\nThis is a diagnostic threshold, not a test-set result.")
