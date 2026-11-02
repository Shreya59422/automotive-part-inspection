
from pathlib import Path
import numpy as np
from PIL import Image
from sklearn.metrics import precision_score, recall_score, f1_score
pred_dir = Path("results/prediction_masks/contamination")
gt_dir = Path("data/mvtec/bottle/ground_truth/contamination")

precisions = []
recalls = []
f1_scores = []
ious = []

for i in range(21):
    name = f"{i:03d}"

    pred_path = pred_dir / f"{name}_pred.png"
    gt_path = gt_dir / f"{name}_mask.png"

    if not pred_path.exists() or not gt_path.exists():
        print(f"Missing mask: {name}")
        continue

    pred = np.array(Image.open(pred_path).convert("L")) > 0
    gt = np.array(Image.open(gt_path).convert("L")) > 0

    if pred.shape != gt.shape:
        pred = np.array(
            Image.fromarray(pred.astype(np.uint8)).resize(
                (gt.shape[1], gt.shape[0]),
                Image.Resampling.NEAREST
            )
        ) > 0

    pred_flat = pred.ravel()
    gt_flat = gt.ravel()

    precision = precision_score(
        gt_flat, pred_flat, zero_division=0
    )
    recall = recall_score(
        gt_flat, pred_flat, zero_division=0
    )
    f1 = f1_score(
        gt_flat, pred_flat, zero_division=0
    )

    intersection = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    iou = intersection / union if union else 1.0

    precisions.append(precision)
    recalls.append(recall)
    f1_scores.append(f1)
    ious.append(iou)

print("\nCONTAMINATION EVALUATION")
print("----------------------")
print("Images evaluated:", len(f1_scores))

if f1_scores:
    print(f"Mean Precision: {np.mean(precisions):.4f}")
    print(f"Mean Recall:    {np.mean(recalls):.4f}")
    print(f"Mean F1 Score:  {np.mean(f1_scores):.4f}")
    print(f"Mean IoU:       {np.mean(ious):.4f}")
