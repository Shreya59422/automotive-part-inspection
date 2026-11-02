
from pathlib import Path
import pandas as pd
import torch
from anomalib.data import MVTecAD
from anomalib.models import Patchcore
from anomalib.engine import Engine
from sklearn.metrics import confusion_matrix

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

# Collect image-level scores and actual labels
scores = []
actual_labels = []

for batch in predictions:
    scores.extend(batch.pred_score.cpu().numpy().tolist())
    actual_labels.extend(batch.gt_label.cpu().numpy().astype(int).tolist())

# Test different anomaly thresholds
thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

results = []

for threshold in thresholds:
    predicted_labels = [
        1 if score >= threshold else 0
        for score in scores
    ]

    tn, fp, fn, tp = confusion_matrix(
        actual_labels,
        predicted_labels,
        labels=[0, 1],
    ).ravel()

    precision = tp / (tp + fp) if tp + fp else 0
    recall = tp / (tp + fn) if tp + fn else 0
    fnr = fn / (fn + tp) if fn + tp else 0
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall
        else 0
    )

    results.append({
        "Threshold": threshold,
        "TP": tp,
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "Precision": round(precision, 4),
        "Recall": round(recall, 4),
        "F1 Score": round(f1, 4),
        "FNR": round(fnr, 4),
    })

# Save results
df = pd.DataFrame(results)

output_path = Path("results/threshold_analysis.csv")
output_path.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(output_path, index=False)

print("\nTHRESHOLD ANALYSIS")
print("-----------------")
print(df.to_string(index=False))
print(f"\nResults saved to: {output_path}")
