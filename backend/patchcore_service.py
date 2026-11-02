
from pathlib import Path

from anomalib.engine import Engine
from anomalib.models import Patchcore

from backend.model_config import PATCHCORE_CHECKPOINT


def predict_anomaly(image_path):

    # ------------------------------------------
    # 1. Validate input paths
    # ------------------------------------------

    image_path = Path(image_path)
    checkpoint = Path(PATCHCORE_CHECKPOINT)

    if not image_path.exists():
        raise FileNotFoundError(
            f"Image not found: {image_path}"
        )

    if not checkpoint.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint}"
        )

    # ------------------------------------------
    # 2. Initialize PatchCore
    # ------------------------------------------

    model = Patchcore()
    engine = Engine()

    # ------------------------------------------
    # 3. Run inference
    # ------------------------------------------

    predictions = engine.predict(
        model=model,
        data_path=str(image_path),
        ckpt_path=str(checkpoint),
        return_predictions=True,
    )

    if not predictions:
        raise RuntimeError(
            "PatchCore returned no predictions."
        )

    # ------------------------------------------
    # 4. Extract prediction
    # ------------------------------------------

    prediction = predictions[0]

    anomaly_map = (
        prediction.anomaly_map
        .detach()
        .cpu()
        .numpy()
    )

    anomaly_score = float(
        prediction.pred_score.flatten()[0].item()
    )

    predicted_anomaly = bool(
        prediction.pred_label.flatten()[0].item()
    )

    prediction_mask = (
        prediction.pred_mask
        .detach()
        .cpu()
        .numpy()
    )

    # ------------------------------------------
    # 5. Diagnostic information
    # ------------------------------------------

    print("\n========== PATCHCORE DIAGNOSTICS ==========")

    print("Image:", image_path.name)
    print("Checkpoint:", checkpoint.name)

    print("Anomaly Score:", anomaly_score)

    print("Anomaly Map Shape:", anomaly_map.shape)

    print("Minimum:", float(anomaly_map.min()))

    print("Maximum:", float(anomaly_map.max()))

    print("Mean:", float(anomaly_map.mean()))

    print("Standard Deviation:", float(anomaly_map.std()))

    print("Predicted Anomaly:", predicted_anomaly)

    print("===========================================\n")

    # ------------------------------------------
    # 6. Return results
    # ------------------------------------------

    return {
        "anomaly_score": anomaly_score,
        "predicted_anomaly": predicted_anomaly,
        "anomaly_map": anomaly_map,
        "prediction_mask": prediction_mask,
    }
