
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from anomalib.engine import Engine
from anomalib.data import MVTecAD
from anomalib.models import Patchcore


# --------------------------------------------------
# 1. PATHS AND CONFIGURATION
# --------------------------------------------------

PROJECT_ROOT = Path.cwd()

CHECKPOINT = (
    PROJECT_ROOT
    / "results"
    / "screw_synthetic_experiment"
    / "Patchcore"
    / "MVTecAD"
    / "screw"
    / "v0"
    / "weights"
    / "lightning"
    / "model.ckpt"
)

IMAGE_NAME = "000.png"

IMAGE_PATH_PART = "thread_side/000.png"

RAW_MASK_PATH = (
    PROJECT_ROOT
    / "data"
    / "mvtec"
    / "screw"
    / "ground_truth"
    / "thread_side"
    / "000_mask.png"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "results"
    / "localization_diagnostic_resized.png"
)


# --------------------------------------------------
# 2. CHECK REQUIRED FILES
# --------------------------------------------------

if not CHECKPOINT.exists():
    raise FileNotFoundError(
        f"Checkpoint not found: {CHECKPOINT}"
    )

if not RAW_MASK_PATH.exists():
    raise FileNotFoundError(
        f"Ground-truth mask not found: {RAW_MASK_PATH}"
    )


# --------------------------------------------------
# 3. LOAD DATASET
# --------------------------------------------------

datamodule = MVTecAD(
    root=PROJECT_ROOT / "data" / "mvtec",
    category="screw",
    train_batch_size=1,
    eval_batch_size=1,
    num_workers=0,
)


# --------------------------------------------------
# 4. LOAD CHECKPOINT AND PREDICT
# --------------------------------------------------

engine = Engine(
    default_root_dir=PROJECT_ROOT / "results" / "localization_diagnostic"
)

predictions = engine.predict(
    model=Patchcore(),
    datamodule=datamodule,
    ckpt_path=str(CHECKPOINT),
    return_predictions=True,
)


# --------------------------------------------------
# 5. FIND TARGET IMAGE
# --------------------------------------------------

target_found = False

for batch in predictions:

    paths = batch.image_path

    if isinstance(paths, (str, Path)):
        paths = [str(paths)]
    else:
        paths = [str(p) for p in paths]

    for i, path in enumerate(paths):

        normalized_path = str(path).replace("\\", "/")

        if IMAGE_PATH_PART not in normalized_path:
            continue

        target_found = True

        # ------------------------------------------
        # 6. LOAD ORIGINAL IMAGE
        # ------------------------------------------

        image = np.asarray(
            Image.open(path).convert("RGB")
        )

        # ------------------------------------------
        # 7. GET ANOMALY MAP
        # ------------------------------------------

        anomaly = (
            batch.anomaly_map[i]
            .detach()
            .cpu()
            .numpy()
            .squeeze()
            .astype(np.float32)
        )

        # ------------------------------------------
        # 8. GET ANOMALIB GROUND-TRUTH MASK
        # ------------------------------------------

        batch_mask = (
            batch.gt_mask[i]
            .detach()
            .cpu()
            .numpy()
            .squeeze()
        )

        batch_mask_binary = batch_mask > 0

        # ------------------------------------------
        # 9. LOAD ORIGINAL DATASET MASK
        # ------------------------------------------

        raw_mask = np.asarray(
            Image.open(RAW_MASK_PATH).convert("L")
        )

        raw_mask_binary = raw_mask > 0

        # Resize Anomalib mask to original mask size
        batch_mask_resized = np.asarray(
            Image.fromarray(
                batch_mask_binary.astype(np.uint8) * 255
            ).resize(
                (raw_mask.shape[1], raw_mask.shape[0]),
                Image.Resampling.NEAREST,
            )
        ) > 0

        # ------------------------------------------
        # 10. CHECK MASK ALIGNMENT
        # ------------------------------------------

        intersection = np.logical_and(
            batch_mask_resized,
            raw_mask_binary,
        ).sum()

        union = np.logical_or(
            batch_mask_resized,
            raw_mask_binary,
        ).sum()

        mask_iou = intersection / (union + 1e-8)

        print("\n========== MASK ALIGNMENT CHECK ==========")
        print("Original mask shape:", raw_mask.shape)
        print("Batch mask shape:", batch_mask.shape)
        print("Original defect pixels:", raw_mask_binary.sum())
        print("Batch defect pixels:", batch_mask_binary.sum())
        print("Mask IoU:", round(float(mask_iou), 4))

        # Use the original dataset mask for visualization
        mask = raw_mask_binary

        # ------------------------------------------
        # 11. RESIZE ANOMALY MAP
        # ------------------------------------------

        anomaly_resized = np.asarray(
            Image.fromarray(anomaly).resize(
                (image.shape[1], image.shape[0]),
                Image.Resampling.BILINEAR,
            )
        )

        # Normalize only for visualization
        anomaly_min = float(anomaly_resized.min())
        anomaly_max = float(anomaly_resized.max())

        anomaly_display = (
            (anomaly_resized - anomaly_min)
            / (anomaly_max - anomaly_min + 1e-8)
        )

        # ------------------------------------------
        # 12. PRINT DIAGNOSTIC INFORMATION
        # ------------------------------------------

        print("\n========== LOCALIZATION DIAGNOSTIC ==========")
        print("Image:", path)
        print("Image shape:", image.shape)
        print("Anomaly map shape:", anomaly.shape)
        print("Resized anomaly shape:", anomaly_resized.shape)
        print("Ground-truth mask shape:", mask.shape)
        print("Anomaly minimum:", anomaly.min())
        print("Anomaly maximum:", anomaly.max())
        print("Ground-truth defect pixels:", np.count_nonzero(mask))

        # ------------------------------------------
        # 13. CREATE OUTPUT DIRECTORY
        # ------------------------------------------

        OUTPUT_PATH.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        # ------------------------------------------
        # 14. CREATE FOUR-PANEL VISUALIZATION
        # ------------------------------------------

        fig, axes = plt.subplots(
            1,
            4,
            figsize=(20, 6),
        )

        # Panel 1: Original image
        axes[0].imshow(image)
        axes[0].set_title("Original image")

        # Panel 2: Ground-truth mask
        axes[1].imshow(
            mask,
            cmap="gray",
            vmin=0,
            vmax=1,
        )
        axes[1].set_title("Ground-truth mask")

        # Panel 3: Anomaly map
        axes[2].imshow(
            anomaly_display,
            cmap="jet",
            vmin=0,
            vmax=1,
        )
        axes[2].set_title("Resized anomaly map")

        # Panel 4: Overlay and ground-truth boundary
        axes[3].imshow(image)

        axes[3].imshow(
            anomaly_display,
            cmap="jet",
            alpha=0.45,
            vmin=0,
            vmax=1,
        )

        axes[3].contour(
            mask.astype(float),
            levels=[0.5],
            colors="white",
            linewidths=0.8,
        )

        axes[3].set_title(
            "Anomaly overlay + GT boundary"
        )

        for ax in axes:
            ax.axis("off")

        plt.tight_layout()

        # Save using an absolute Windows-compatible path
        fig.savefig(
            str(OUTPUT_PATH.resolve()),
            dpi=150,
            bbox_inches="tight",
        )

        plt.close(fig)

        print("\nSaved diagnostic image:")
        print(OUTPUT_PATH.resolve())

        break

    if target_found:
        break


# --------------------------------------------------
# 15. HANDLE TARGET NOT FOUND
# --------------------------------------------------

if not target_found:
    print(
        "\nTarget image was not found in predictions."
    )
