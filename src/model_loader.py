"""Load the Yjson file and download it if it is not found locally."""

import os

import requests
import torch
from ultralytics import YOLO


def load_model(
    local_path: str = "models/yolov8n.pt",
    remote_url: str = None,
    auto_download: bool = False,
) -> YOLO:
    """
    Load the YOLO model.

    If the model is not found locally and a remote URL is provided
    (with auto_download=True), it will download the model before loading.

    Args:
        local_path (str): Path to local model weights file.
        remote_url (str): URL to download the model if not found locally.
        auto_download (bool): If True, download the model if not
        available locally.

    Returns:
        Model: Loaded YOLO model.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if not os.path.exists(local_path) and remote_url and auto_download:
        print(f"Local model not found. Downloading from {remote_url}...")
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        response = requests.get(remote_url, stream=True)
        response.raise_for_status()
        with open(local_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

    if not os.path.exists(local_path):
        raise FileNotFoundError(
            f"Model not found at {local_path}. "
            "Provide a remote_url and set auto_download=True to download."
        )

    model = YOLO(local_path)
    model.to(device)
    return model
