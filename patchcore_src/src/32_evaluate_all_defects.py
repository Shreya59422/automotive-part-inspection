from pathlib import Path

import numpy as np
import pandas as pd

from anomalib.data import MVTecAD
from anomalib.engine import Engine


# --------------------------------------------------
# 1. Load the existing trained PatchCore checkpoint
# --------------------------------------------------

CHECKPOINT_PATH = Path(
    "results/screw_synthetic_experiment/Patchcore/"
    "MVTecAD/screw/v0/weights/lightning/model.ckpt"
)

if not CHECKPOINT_PATH.exists():
    raise FileNotFoundError(
        f"Checkpoint not found: {CHECKPOINT_PATH.resolve()}"
    )


# --------------------------------------------------
# 2. Load the screw test dataset
# --------------------------------------------------

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


# --------------------------------------------------
# 3. Load PatchCore
# --------------------------------------------------

from anomalib.models import Patchcore

engine = Engine(
    default_root_dir="results/localization_all_defects"
)

model = Patchcore.load_from_checkpoint(
    CHECKPOINT_PATH,
    map_location="cpu"
)

model.eval()


# --------------------------------------------------
# 4. Predict all screw test images
# --------------------------------------------------

predictions = engine.predict(
    model=model,
    dataloaders=datamodule.test_dataloader(),
)

if predictions is None:
    raise RuntimeError("Prediction returned no results.")


# --------------------------------------------------
# 5. Compare defect-region and normal-region scores
# --------------------------------------------------

records = []

for batch in predictions:

    anomaly_map = (
        batch.anomaly_map.detach().cpu().numpy()
    )

    ground_truth = (
        batch.gt_mask.detach().cpu().numpy()
    )

    image_paths = batch.image_path

    if isinstance(image_paths, (str, Path)):
        image_paths = [image_paths]

    for i in range(len(anomaly_map)):

        image_path = str(image_paths[i])

        # Consider only test images
        if "/test/" not in image_path.replace("\\", "/"):
            continue

        anomaly = np.squeeze(anomaly_map[i])
        mask = np.squeeze(ground_truth[i]) > 0.5

        # Skip normal images without defect pixels
        if not np.any(mask):
            continue

        defect_scores = anomaly[mask]
        normal_scores = anomaly[~mask]

        if len(normal_scores) == 0:
            continue

        defect_mean = float(np.mean(defect_scores))
        normal_mean = float(np.mean(normal_scores))

        # Percentile rank of the mean defect score
        percentile = float(
            100 * np.mean(anomaly <= defect_mean)
        )

        records.append({
            "image": Path(image_path).name,
            "defect_type": Path(image_path).parent.name,
            "defect_mean": defect_mean,
            "normal_mean": normal_mean,
            "mean_score_gap": defect_mean - normal_mean,
            "defect_median": float(np.median(defect_scores)),
            "normal_median": float(np.median(normal_scores)),
            "defect_mean_percentile": percentile,
        })


# --------------------------------------------------
# 6. Summarize results
# --------------------------------------------------

if not records:
    raise RuntimeError(
        "No defective test images were found. "
        "Check the test dataloader and mask values."
    )

df = pd.DataFrame(records)

output_path = (
    Path.cwd() / "results" / "localization_all_defects_summary.csv"
)

output_path.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(output_path, index=False)

print("\n========== ALL DEFECTIVE IMAGES ==========")

print("Number of defective images:", len(df))

print(
    "Images where defect mean score > normal mean score:",
    int((df["mean_score_gap"] > 0).sum()),
    "out of",
    len(df),
)

print(
    "Average defect-region score:",
    round(df["defect_mean"].mean(), 6),
)

print(
    "Average normal-region score:",
    round(df["normal_mean"].mean(), 6),
)

print(
    "Average score gap:",
    round(df["mean_score_gap"].mean(), 6),
)

print(
    "Average defect mean percentile:",
    round(df["defect_mean_percentile"].mean(), 2),
)

print("\n========== RESULTS BY DEFECT TYPE ==========")

print(
    df.groupby("defect_type")[
        ["defect_mean", "normal_mean", "mean_score_gap",
         "defect_mean_percentile"]
    ].mean().round(4).to_string()
)

print("\nSaved results to:", output_path)