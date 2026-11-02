
from pathlib import Path

import numpy as np
from anomalib.engine import Engine
from anomalib.data import MVTecAD
from anomalib.models import Patchcore

checkpoint = Path(
    "results/screw_synthetic_experiment/Patchcore/"
    "MVTecAD/screw/v0/weights/lightning/model.ckpt"
)

datamodule = MVTecAD(
    root=Path("data/mvtec"),
    category="screw",
    train_batch_size=1,
    eval_batch_size=1,
    num_workers=0,
)

engine = Engine(
    default_root_dir="results/localization_score_check"
)

predictions = engine.predict(
    model=Patchcore(),
    datamodule=datamodule,
    ckpt_path=str(checkpoint),
    return_predictions=True,
)

target_found = False

for batch in predictions:
    paths = batch.image_path

    if isinstance(paths, (str, Path)):
        paths = [str(paths)]
    else:
        paths = [str(p) for p in paths]

    for i, path in enumerate(paths):
        normalized = str(path).replace("\\", "/")

        if "thread_side/000.png" not in normalized:
            continue

        anomaly = (
            batch.anomaly_map[i]
            .detach()
            .cpu()
            .numpy()
            .squeeze()
        )

        mask = (
            batch.gt_mask[i]
            .detach()
            .cpu()
            .numpy()
            .squeeze()
        ) > 0

        defect_scores = anomaly[mask]
        normal_scores = anomaly[~mask]

        print("\n========== DEFECT SCORE COMPARISON ==========")
        print("Image:", path)
        print("Defect pixels:", defect_scores.size)
        print("Normal pixels:", normal_scores.size)

        print("\nDefect-region scores:")
        print("Mean:", defect_scores.mean())
        print("Median:", np.median(defect_scores))
        print("Maximum:", defect_scores.max())

        print("\nNormal-region scores:")
        print("Mean:", normal_scores.mean())
        print("Median:", np.median(normal_scores))
        print("Maximum:", normal_scores.max())

        print("\nDefect score percentile:")
        print(
            "Defect mean percentile among all pixels:",
            100 * np.mean(anomaly <= defect_scores.mean())
        )

        target_found = True
        break

    if target_found:
        break

if not target_found:
    print("Target image not found.")
