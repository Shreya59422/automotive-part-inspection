import cv2
import numpy as np
from PIL import Image


def _to_numpy(anomaly_map):
    """Convert tensor/list/array anomaly map to a 2D float32 array."""
    if hasattr(anomaly_map, "detach"):
        anomaly_map = anomaly_map.detach().cpu().numpy()

    array = np.asarray(anomaly_map, dtype=np.float32)
    array = np.nan_to_num(array, nan=0.0, posinf=0.0, neginf=0.0)

    # Remove leading batch/channel dimensions until a 2D map remains.
    while array.ndim > 2:
        array = array[0]

    if array.ndim == 1:
        array = array.reshape(1, -1)

    if array.ndim != 2:
        raise ValueError(
            f"Expected a 2D anomaly map after squeezing dimensions; got {array.shape}"
        )

    return array


def _normalize_map(anomaly_map):
    """Normalize anomaly map to uint8 range 0-255 without losing float data."""
    array = _to_numpy(anomaly_map)

    minimum = float(array.min())
    maximum = float(array.max())

    if maximum - minimum < 1e-8:
        normalized = np.zeros(array.shape, dtype=np.uint8)
    else:
        normalized = (
            (array - minimum) / (maximum - minimum) * 255.0
        ).astype(np.uint8)

    return normalized


def create_heatmap(anomaly_map, output_size=None):
    """
    Create a colored anomaly heatmap.

    Returns an RGB uint8 numpy array suitable for PIL and Streamlit.
    """
    normalized = _normalize_map(anomaly_map)

    if output_size is not None:
        width, height = output_size
        normalized = cv2.resize(
            normalized,
            (int(width), int(height)),
            interpolation=cv2.INTER_CUBIC,
        )

    colored = cv2.applyColorMap(normalized, cv2.COLORMAP_JET)
    colored = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)

    return colored.astype(np.uint8)


def create_overlay(original_image, heatmap, alpha=0.45):
    """
    Blend the original RGB image with a colored anomaly heatmap.

    Returns an RGB uint8 numpy array.
    """
    if isinstance(original_image, Image.Image):
        original = np.asarray(original_image.convert("RGB"))
    else:
        original = np.asarray(original_image)

    original = np.nan_to_num(original)

    if np.issubdtype(original.dtype, np.floating):
        if original.size and original.min() >= 0 and original.max() <= 1:
            original = original * 255.0

    original = np.clip(original, 0, 255).astype(np.uint8)

    if original.ndim == 2:
        original = cv2.cvtColor(original, cv2.COLOR_GRAY2RGB)
    elif original.ndim == 3 and original.shape[-1] == 4:
        original = cv2.cvtColor(original, cv2.COLOR_RGBA2RGB)
    elif original.ndim == 3 and original.shape[-1] == 1:
        original = cv2.cvtColor(original, cv2.COLOR_GRAY2RGB)

    heat = np.asarray(heatmap)

    if heat.ndim == 2:
        heat = cv2.applyColorMap(
            np.clip(heat, 0, 255).astype(np.uint8),
            cv2.COLORMAP_JET,
        )
        heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)

    if heat.ndim == 3 and heat.shape[-1] == 4:
        heat = heat[..., :3]

    if np.issubdtype(heat.dtype, np.floating):
        if heat.size and heat.min() >= 0 and heat.max() <= 1:
            heat = heat * 255.0

    heat = np.clip(heat, 0, 255).astype(np.uint8)

    height, width = original.shape[:2]
    if heat.shape[:2] != (height, width):
        heat = cv2.resize(
            heat,
            (width, height),
            interpolation=cv2.INTER_LINEAR,
        )

    alpha = float(np.clip(alpha, 0.0, 1.0))
    blended = cv2.addWeighted(original, 1.0 - alpha, heat, alpha, 0)

    return blended.astype(np.uint8)
