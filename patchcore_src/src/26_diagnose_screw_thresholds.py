from pathlib import Path
import numpy as np
import torch
from anomalib.engine import Engine
from anomalib.data import MVTecAD
from anomalib.models import Patchcore

checkpoint = Path(
    "results/anomaly_detection/Patchcore/MVTecAD/"
    "screw/v0/weights/lightning/model.ckpt"
)

datamodule = MVTecAD(
    root=Path("data/mvtec"),
    category="screw",
    train_batch_size=1,
    eval_batch_size=1,
    num_workers=0,
)

engine = Engine(default_root_dir="results/anomaly_detection")
predictions = engine.predict(
    model=Patchcore(),
    datamodule=datamodule,
    ckpt_path=str(checkpoint),
    return_predictions=True,
)

thresholds = [0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50]
totals = {t: [0, 0, 0] for t in thresholds}

def to_numpy(value, i):
    if isinstance(value, torch.Tensor):
        return value[i].detach().cpu().numpy()
    if isinstance(value, (list, tuple)):
        return np.asarray(value[i])
    return np.asarray(value)

for batch in predictions:
    paths = batch.image_path
    if isinstance(paths, (str, Path)):
        paths = [str(paths)]
    else:
        paths = [str(p) for p in paths]

    for i, path in enumerate(paths):
        amap = np.squeeze(to_numpy(batch.anomaly_map, i)).astype(np.float32)
        mask = np.squeeze(to_numpy(batch.gt_mask, i)) > 0

        if amap.shape != mask.shape:
            print("Shape mismatch:", path, amap.shape, mask.shape)
            continue

        if "thread_side/000.png" in path.replace("\\", "/"):
            print("\nTHREAD_SIDE/000")
            print("Anomaly map min/max:", float(amap.min()), float(amap.max()))
            print("Ground-truth pixels:", int(mask.sum()))
            for t in thresholds:
                print(f"Threshold {t:.2f}:",
                      int((amap >= t).sum()), "predicted pixels")

        for t in thresholds:
            pred = amap >= t
            tp = int((pred & mask).sum())
            fp = int((pred & ~mask).sum())
            fn = int((~pred & mask).sum())
            totals[t][0] += tp
            totals[t][1] += fp
            totals[t][2] += fn

print("\nTHRESHOLD | PRECISION | RECALL | F1")
for t, (tp, fp, fn) in totals.items():
    precision = tp / (tp + fp) if tp + fp else 0
    recall = tp / (tp + fn) if tp + fn else 0
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall else 0
    )
    print(f"{t:.2f}      | {precision:.4f}    | {recall:.4f} | {f1:.4f}")
