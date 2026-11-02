
from pathlib import Path
import torch

from anomalib.models import Patchcore

checkpoint = Path(
    "results/screw_synthetic_experiment/Patchcore/"
    "MVTecAD/screw/v0/weights/lightning/model.ckpt"
)

print("\n========== LOADING PATCHCORE ==========")

model = Patchcore.load_from_checkpoint(
    checkpoint,
    map_location="cpu",
)

model.eval()

print("\n========== MODEL THRESHOLD INSPECTION ==========")

print("\nModel type:", type(model).__name__)

# Inspect threshold-related model attributes
for name in dir(model):
    if "threshold" in name.lower():
        try:
            value = getattr(model, name)
            if not callable(value):
                print(f"{name}: {value}")
        except Exception:
            pass

# Inspect threshold-related buffers
print("\n========== REGISTERED BUFFERS ==========")

for name, value in model.named_buffers():
    if "threshold" in name.lower():
        print(name, "=", value.detach().cpu().numpy())

# Inspect post-processing configuration
print("\n========== POST-PROCESSOR ==========")

if hasattr(model, "post_processor"):
    print(model.post_processor)
else:
    print("No post_processor attribute found.")

print("\n========== INSPECTION COMPLETE ==========")
