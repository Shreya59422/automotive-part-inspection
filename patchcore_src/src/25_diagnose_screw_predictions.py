from pathlib import Path
import numpy as np
import torch
from anomalib.engine import Engine
from anomalib.data import MVTecAD
from anomalib.models import Patchcore

checkpoint = Path("results/patchcore_screw_backup.ckpt")

datamodule = MVTecAD(
    root=Path("data/mvtec"),
    category="screw",
    train_batch_size=1,
    eval_batch_size=1,
    num_workers=0,
)

model = Patchcore()
engine = Engine(default_root_dir="results/anomaly_detection")

predictions = engine.predict(
    model=model,
    datamodule=datamodule,
    ckpt_path=str(checkpoint),
    return_predictions=True,
)

print("Total prediction batches:", len(predictions))

for batch in predictions:
    paths = batch.image_path
    if isinstance(paths, (str, Path)):
        paths = [str(paths)]
    else:
        paths = [str(p) for p in paths]

    for i, path in enumerate(paths):
        if Path(path).stem != "000":
            continue
        if "thread_side" not in path.replace("\\", "/"):
            continue

        def get_item(value, index):
            if value is None:
                return None
            if isinstance(value, torch.Tensor):
                return value[index].detach().cpu().numpy()
            if isinstance(value, (list, tuple)):
                return value[index]
            return value

        anomaly_map = get_item(batch.anomaly_map, i)
        pred_mask = get_item(batch.pred_mask, i)
        gt_mask = get_item(batch.gt_mask, i)

        print("\nIMAGE:", path)

        for name, value in [
            ("Anomaly map", anomaly_map),
            ("Predicted mask", pred_mask),
            ("Ground-truth mask", gt_mask),
        ]:
            if value is None:
                print(name, "is None")
                continue
            array = np.asarray(value)
            print(f"\n{name}:")
            print("  Shape:", array.shape)
            print("  Min:", float(array.min()))
            print("  Max:", float(array.max()))
            print("  Unique values (up to 10):",
                  np.unique(array)[:10])

            if name == "Predicted mask":
                binary = array > 0
                print("  Predicted defect pixels:",
                      int(binary.sum()))
            elif name == "Ground-truth mask":
                binary = array > 0
                print("  Actual defect pixels:",
                      int(binary.sum()))

        break
    else:
        continue
    break
else:
    print("Could not find thread_side/000.png in predictions.")

