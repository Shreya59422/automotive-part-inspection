
from pathlib import Path

import numpy as np
import pandas as pd

from anomalib.data import MVTecAD
from anomalib.models import Patchcore
from anomalib.engine import Engine

from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)


# Dataset
datamodule = MVTecAD(
    root=Path("data/mvtec"),
    category="bottle",
    train_batch_size=8,
    eval_batch_size=8,
    num_workers=0,
)


# Model
model = Patchcore()


# Engine
engine = Engine(default_root_dir="results/anomaly_detection")


# Checkpoint
checkpoint_path = (
    "results/anomaly_detection/Patchcore/MVTecAD/"
    "bottle/v0/weights/lightning/model.ckpt"
)


# Evaluate
results = engine.test(
    model=model,
    datamodule=datamodule,
    ckpt_path=checkpoint_path,
)

print("\nPatchCore evaluation completed.")
print("Evaluation results:")
print(results)
