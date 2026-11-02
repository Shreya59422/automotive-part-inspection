
from pathlib import Path
import torch
from anomalib.data import MVTecAD
from anomalib.models import Patchcore
from anomalib.engine import Engine
from sklearn.metrics import confusion_matrix, classification_report

# Load test dataset
datamodule = MVTecAD(
    root=Path("data/mvtec"),
    category="bottle",
    train_batch_size=8,
    eval_batch_size=8,
    num_workers=0,
)

# Load trained PatchCore
model = Patchcore()
engine = Engine(default_root_dir="results/anomaly_detection")

checkpoint_path = (
    "results/anomaly_detection/Patchcore/MVTecAD/"
    "bottle/v0/weights/lightning/model.ckpt"
)

# Generate predictions for all test images
predictions = engine.predict(
    model=model,
    datamodule=datamodule,
    ckpt_path=checkpoint_path,
)

# Collect actual and predicted labels
actual_labels = []
predicted_labels = []

for batch in predictions:
    actual_labels.extend(batch.gt_label.cpu().numpy().astype(int).tolist())
    predicted_labels.extend(batch.pred_label.cpu().numpy().astype(int).tolist())

# Calculate confusion matrix
tn, fp, fn, tp = confusion_matrix(
    actual_labels,
    predicted_labels,
    labels=[0, 1],
).ravel()

# Calculate metrics
precision = tp / (tp + fp) if (tp + fp) else 0
recall = tp / (tp + fn) if (tp + fn) else 0
fnr = fn / (fn + tp) if (fn + tp) else 0
f1 = (
    2 * precision * recall / (precision + recall)
    if (precision + recall)
    else 0
)

# Display results
print("\nIMAGE-LEVEL EVALUATION")
print("---------------------")
print("Total images:", len(actual_labels))
print("True Positives (TP):", tp)
print("True Negatives (TN):", tn)
print("False Positives (FP):", fp)
print("False Negatives (FN):", fn)
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1 Score:  {f1:.4f}")
print(f"FNR:       {fnr:.4f}")

print("\nClassification Report:")
print(
    classification_report(
        actual_labels,
        predicted_labels,
        labels=[0, 1],
        target_names=["Good", "Defective"],
        zero_division=0,
    )
)
