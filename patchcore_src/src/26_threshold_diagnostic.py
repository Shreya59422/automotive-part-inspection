from pathlib import Path
import numpy as np
import torch
from anomalib.engine import Engine
from anomalib.data import MVTecAD
from anomalib.models import Patchcore

checkpoint = Path("results/patchcore_screw_backup.ckpt")

datamodule = MVTecAD(
    root=Path("data/mvtec"),
    category="screw",
    train_batch_size=1,
    eval_batch_size=1,
    num_workers=0,
)

model = Patchcore()
engine = Engine(default_root_dir="results/anomaly_detection")

predictions = engine.predict(
    model=model,
    datamodule=datamodule,
    ckpt_path=str(checkpoint),
    return_predictions=True,
)

for batch in predictions:
    paths = batch.image_path
    if isinstance(paths, (str, Path)):
        paths = [str(paths)]
    else:
        paths = [str(p) for p in paths]

    for i, path in enumerate(paths):
        normalized_path = path.replace("\\", "/")
        if "thread_side/000.png" not in normalized_path:
            continue

        anomaly_map = batch.anomaly_map[i].detach().cpu().numpy()
        gt_mask = batch.gt_mask[i].detach().cpu().numpy() > 0

        print("\nThreshold comparison for:", path)
        print(f"{'Threshold':>10} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Pred pixels':>12}")

        for threshold in [0.10, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50]:
            pred = anomaly_map >= threshold
            tp = np.logical_and(pred, gt_mask).sum()
            fp = np.logical_and(pred, ~gt_mask).sum()
            fn = np.logical_and(~pred, gt_mask).sum()

            precision = tp / (tp + fp) if tp + fp else 0.0
            recall = tp / (tp + fn) if tp + fn else 0.0
            f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

            print(f"{threshold:10.2f} {precision:10.4f} {recall:10.4f} {f1:10.4f} {int(pred.sum()):12d}")

        break
    else:
        continue
    break
else:
    print("Target image was not found.")
