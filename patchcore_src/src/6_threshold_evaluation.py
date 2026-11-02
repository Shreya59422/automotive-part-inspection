
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image

from sklearn.metrics import f1_score


# Ground-truth mask folder
mask_dir = Path("data/mvtec/bottle/ground_truth")

# Prediction visualization folder
prediction_dir = Path(
    "results/anomaly_detection/Patchcore/MVTecAD/"
    "bottle/latest/images"
)

print("Threshold evaluation setup")
print("Ground truth:", mask_dir)
print("Predictions:", prediction_dir)

# Check folders
if mask_dir.exists():
    print("Ground-truth folder exists")
else:
    print("Ground-truth folder not found")

if prediction_dir.exists():
    print("Prediction folder exists")
else:
    print("Prediction folder not found")

print("Next: extract anomaly maps and compare with ground truth.")
