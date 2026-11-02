from pathlib import Path
import numpy as np
from PIL import Image

root = Path("data/mvtec/screw")
mask_path = root / "ground_truth/thread_side/000_mask.png"
image_path = root / "test/thread_side/000.png"
viz_path = Path(
    "results/anomaly_detection/Patchcore/MVTecAD/"
    "screw/v0/images/thread_side/000.png"
)

mask = np.array(Image.open(mask_path).convert("L"))
image = np.array(Image.open(image_path).convert("L"))
viz = Image.open(viz_path)

ground_truth = mask > 0

print("IMAGE:", image.shape)
print("GROUND-TRUTH MASK:", mask.shape)
print("VISUALIZATION:", viz.size)
print("Ground-truth defect pixels:", int(ground_truth.sum()))
print("Ground-truth defect percentage:",
      round(100 * ground_truth.mean(), 4), "%")
print("Ground-truth unique values:", np.unique(mask))
print("Visualization panel dimensions:",
      viz.width // 4, "x", viz.height)
