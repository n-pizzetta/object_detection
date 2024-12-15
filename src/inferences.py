"""
Inference Module.

Provides functions to run inference using the YOLO model on images.
"""

from typing import Optional

import cv2

from src.model_loader import load_model


def run_inference(
    image_path: str,
    model_path: str = "models/yolov8n.pt",
    remote_url: Optional[str] = None,
):
    """
    Run inference using the YOLO model on a given image.

    Args:
        image_path (str): Path to the input image.
        model_path (str, optional): Local path to the model weights.
        Defaults to "models/yolov8n.pt".
        remote_url (str, optional): URL to fetch model if not found
        locally. Defaults to None.

    Returns:
        InferenceResults: Results from the YOLO model prediction.
    """
    model = load_model(
        local_path=model_path,
        remote_url=remote_url,
        auto_download=True,
    )
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Image not found at {image_path}")
    results = model.predict(img)
    return results
