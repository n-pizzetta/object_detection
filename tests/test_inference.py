"""Tests for the inference module."""

import os

import cv2
import numpy as np
import pytest

from src.inferences import run_inference


def test_run_inference_missing_image():
    """
    Test running inference on a non-existent image.

    Should raise FileNotFoundError.
    """
    with pytest.raises(FileNotFoundError):
        run_inference(image_path="tests/data/nonexistent_image.jpg")


def test_run_inference_without_model():
    """
    Test running inference.

    It does it without a local model and without providing a remote URL.

    Should raise FileNotFoundError.
    """
    # Create a dummy image for testing
    test_image = "tests/data/test_image.jpg"
    os.makedirs(os.path.dirname(test_image), exist_ok=True)
    dummy_image = np.ones((100, 100, 3), dtype=np.uint8) * 255
    cv2.imwrite(test_image, dummy_image)

    with pytest.raises(FileNotFoundError):
        run_inference(
            image_path=test_image,
            model_path="models/nonexistent_model.pt",
        )
