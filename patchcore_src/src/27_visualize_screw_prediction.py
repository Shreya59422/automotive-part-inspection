from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
from anomalib.engine import Engine
from anomalib.data import MVTecAD
from anomalib.models import Patchcore

checkpoint = Path("results/patchcore_screw_backup.ckpt")
target = "thread_side/000.png"

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
        if target not in path.replace("\\", "/"):
            continue

        image = np.asarray(Image.open(path).convert("RGB"))
        anomaly = batch.anomaly_map[i].detach().cpu().numpy().squeeze()
        gt = batch.gt_mask[i].detach().cpu().numpy().squeeze() > 0
        pred = anomaly >= 0.25

        fig, axes = plt.subplots(1, 4, figsize=(16, 4))
        axes[0].imshow(image)
        axes[0].set_title("Original image")
        axes[1].imshow(gt, cmap="gray")
        axes[1].set_title("Ground-truth mask")
        axes[2].imshow(anomaly, cmap="inferno")
        axes[2].set_title("Anomaly map")
        axes[3].imshow(pred, cmap="gray")
        axes[3].set_title("Prediction at 0.25")

        for ax in axes:
            ax.axis("off")

        plt.tight_layout()
        output = Path("results/thread_side_000_diagnostic.png")
        output.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output, dpi=150, bbox_inches="tight")
        plt.close()

        print("Saved visualization:", output.resolve())
        break
    else:
        continue
    break
else:
    print("Target image not found.")
