
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
)

from anomalib.data import MVTecAD
from anomalib.engine import Engine
from anomalib.models import Patchcore


# ------------------------------------------
# 1. Check the trained checkpoint
# ------------------------------------------

CHECKPOINT_PATH = Path(
    "results/screw_synthetic_experiment/Patchcore/"
    "MVTecAD/screw/v0/weights/lightning/model.ckpt"
)

if not CHECKPOINT_PATH.exists():
    raise FileNotFoundError(
        f"Checkpoint not found: {CHECKPOINT_PATH.resolve()}"
    )


# ------------------------------------------
# 2. Load the test dataset
# ------------------------------------------

datamodule = MVTecAD(
    root=Path("data/mvtec"),
    category="screw",
    train_batch_size=8,
    eval_batch_size=1,
    num_workers=0,
    val_split_mode="synthetic",
    val_split_ratio=0.2,
    seed=42,
)

datamodule.setup()


# ------------------------------------------
# 3. Load the trained PatchCore model
# ------------------------------------------

model = Patchcore.load_from_checkpoint(
    CHECKPOINT_PATH,
    map_location="cpu",
)

model.eval()

engine = Engine(
    default_root_dir="results/patchcore_final_evaluation"
)


# ------------------------------------------
# 4. Predict all test images
# ------------------------------------------

predictions = engine.predict(
    model=model,
    dataloaders=datamodule.test_dataloader(),
)

if predictions is None:
    raise RuntimeError("No predictions were returned.")


# ------------------------------------------
# 5. Collect image and pixel predictions
# ------------------------------------------

image_labels = []
image_scores = []
image_predictions = []

pixel_labels = []
pixel_scores = []
pixel_predictions = []

image_count = 0

for batch in predictions:

    gt_label = batch.gt_label.detach().cpu().numpy().reshape(-1)
    pred_score = batch.pred_score.detach().cpu().numpy().reshape(-1)
    pred_label = batch.pred_label.detach().cpu().numpy().reshape(-1)

    gt_mask = batch.gt_mask.detach().cpu().numpy()
    anomaly_map = batch.anomaly_map.detach().cpu().numpy()

    pred_mask = batch.pred_mask.detach().cpu().numpy()

    for i in range(len(gt_label)):

        image_count += 1

        image_labels.append(int(gt_label[i]))
        image_scores.append(float(pred_score[i]))
        image_predictions.append(int(pred_label[i]))

        mask = np.squeeze(gt_mask[i]) > 0.5
        anomaly = np.squeeze(anomaly_map[i])
        predicted_mask = np.squeeze(pred_mask[i]) > 0.5

        if mask.shape != anomaly.shape:
            raise ValueError(
                f"Mask and anomaly map shapes differ: "
                f"{mask.shape} vs {anomaly.shape}"
            )

        if mask.shape != predicted_mask.shape:
            raise ValueError(
                f"Ground truth and predicted mask shapes differ: "
                f"{mask.shape} vs {predicted_mask.shape}"
            )

        pixel_labels.extend(mask.astype(np.uint8).ravel())
        pixel_scores.extend(anomaly.astype(np.float32).ravel())
        pixel_predictions.extend(
            predicted_mask.astype(np.uint8).ravel()
        )


# ------------------------------------------
# 6. Calculate evaluation metrics
# ------------------------------------------

image_labels = np.asarray(image_labels)
image_scores = np.asarray(image_scores)
image_predictions = np.asarray(image_predictions)

pixel_labels = np.asarray(pixel_labels)
pixel_scores = np.asarray(pixel_scores)
pixel_predictions = np.asarray(pixel_predictions)


def calculate_auc(labels, scores, name):

    if len(np.unique(labels)) < 2:
        raise ValueError(
            f"{name} requires both normal and defective labels."
        )

    return roc_auc_score(labels, scores)


image_auc = calculate_auc(
    image_labels, image_scores, "Image AUROC"
)

image_ap = average_precision_score(
    image_labels, image_scores
)

pixel_auc = calculate_auc(
    pixel_labels, pixel_scores, "Pixel AUROC"
)

pixel_ap = average_precision_score(
    pixel_labels, pixel_scores
)

image_f1 = f1_score(
    image_labels, image_predictions, zero_division=0
)

image_precision = precision_score(
    image_labels, image_predictions, zero_division=0
)

image_recall = recall_score(
    image_labels, image_predictions, zero_division=0
)

pixel_f1 = f1_score(
    pixel_labels, pixel_predictions, zero_division=0
)

pixel_precision = precision_score(
    pixel_labels, pixel_predictions, zero_division=0
)

pixel_recall = recall_score(
    pixel_labels, pixel_predictions, zero_division=0
)


# ------------------------------------------
# 7. Print the results
# ------------------------------------------

print("\n========== PATCHCORE FINAL EVALUATION ==========")

print("Total test images:", image_count)

print("\nIMAGE-LEVEL METRICS")
print("Image AUROC:", round(image_auc, 6))
print("Image Average Precision:", round(image_ap, 6))
print("Image F1-score:", round(image_f1, 6))
print("Image Precision:", round(image_precision, 6))
print("Image Recall:", round(image_recall, 6))

print("\nPIXEL-LEVEL METRICS")
print("Pixel AUROC:", round(pixel_auc, 6))
print("Pixel Average Precision:", round(pixel_ap, 6))
print("Pixel F1-score:", round(pixel_f1, 6))
print("Pixel Precision:", round(pixel_precision, 6))
print("Pixel Recall:", round(pixel_recall, 6))

print("\nIMAGE CONFUSION MATRIX")
print(confusion_matrix(image_labels, image_predictions))

print("\nPIXEL CONFUSION MATRIX")
print(confusion_matrix(pixel_labels, pixel_predictions))


# ------------------------------------------
# 8. Save the metrics
# ------------------------------------------

results = pd.DataFrame([
    {"Metric": "Image AUROC", "Score": image_auc},
    {"Metric": "Image Average Precision", "Score": image_ap},
    {"Metric": "Image F1-score", "Score": image_f1},
    {"Metric": "Image Precision", "Score": image_precision},
    {"Metric": "Image Recall", "Score": image_recall},
    {"Metric": "Pixel AUROC", "Score": pixel_auc},
    {"Metric": "Pixel Average Precision", "Score": pixel_ap},
    {"Metric": "Pixel F1-score", "Score": pixel_f1},
    {"Metric": "Pixel Precision", "Score": pixel_precision},
    {"Metric": "Pixel Recall", "Score": pixel_recall},
])

output_path = (
    Path.cwd() / "results" / "patchcore_final_metrics.csv"
)

output_path.parent.mkdir(parents=True, exist_ok=True)
results.to_csv(output_path, index=False)

print("\nSaved metrics to:", output_path)
