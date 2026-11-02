from pathlib import Path
import numpy as np
import torch
from anomalib.engine import Engine
from anomalib.data import MVTecAD
from anomalib.models import Patchcore
checkpoint = Path(
    "results/screw_synthetic_experiment/Patchcore/MVTecAD/screw/v0/weights/lightning/model.ckpt"
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

for batch in predictions:
    paths = batch.image_path
    if isinstance(paths, (str, Path)):
        paths = [str(paths)]
    else:
        paths = [str(p) for p in paths]

    for i, path in enumerate(paths):
        if "thread_side/000.png" not in path.replace("\\", "/"):
            continue

        anomaly = batch.anomaly_map[i].detach().cpu().numpy().squeeze()
        gt = batch.gt_mask[i].detach().cpu().numpy().squeeze() > 0

        print("\nImage:", path)
        print("Anomaly-map shape:", anomaly.shape)
        print("Ground-truth shape:", gt.shape)
        print("Anomaly min/max:", float(anomaly.min()), float(anomaly.max()))
        print("Ground-truth defect pixels:", int(gt.sum()))

        if gt.shape != anomaly.shape:
            from PIL import Image
            gt = np.asarray(
                Image.fromarray(gt.astype(np.uint8) * 255).resize(
                    (anomaly.shape[1], anomaly.shape[0]),
                    Image.Resampling.NEAREST,
                )
            ) > 0

        print("Resized defect pixels:", int(gt.sum()))

        for percentile in [99, 99.5, 99.9]:
            cutoff = np.percentile(anomaly, percentile)
            top = anomaly >= cutoff
            overlap = int(np.logical_and(top, gt).sum())
            precision = overlap / int(top.sum()) if top.sum() else 0
            recall = overlap / int(gt.sum()) if gt.sum() else 0

            print(
                f"Top {100-percentile:.1f}% scores: "
                f"cutoff={cutoff:.5f}, "
                f"overlap={overlap}, "
                f"precision={precision:.4f}, "
                f"recall={recall:.4f}"
            )

        break
    else:
        continue
    break
