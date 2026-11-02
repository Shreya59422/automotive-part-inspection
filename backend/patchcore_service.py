from pathlib import Path

from anomalib.engine import Engine
from anomalib.models import Patchcore

from backend.model_config import get_patchcore_checkpoint


def predict_anomaly(image_path):

    # 1. Validate image
    image_path = Path(image_path)

    if not image_path.is_file():
        raise FileNotFoundError(
            f"Image not found: {image_path}"
        )

    # 2. Resolve checkpoint from local storage or Hugging Face
    checkpoint = get_patchcore_checkpoint()

    if not checkpoint.is_file():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint}"
        )

    print("Loading PatchCore model...")
    print("Checkpoint:", checkpoint)

    # 3. Initialize PatchCore and Engine
    model = Patchcore()
    engine = Engine()

    # 4. Run inference
    predictions = engine.predict(
        model=model,
        data_path=str(image_path),
        ckpt_path=str(checkpoint),
        return_predictions=True,
    )

    if predictions is None or len(predictions) == 0:
        raise RuntimeError(
            "PatchCore returned no predictions."
        )

    # 5. Extract prediction
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

    # 6. Diagnostic information
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

    # 7. Return results
    return {
        "anomaly_score": anomaly_score,
        "predicted_anomaly": predicted_anomaly,
        "anomaly_map": anomaly_map,
        "prediction_mask": prediction_mask,
    }