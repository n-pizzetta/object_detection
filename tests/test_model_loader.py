"""Test the model_loader module."""

import pytest

from src.model_loader import load_model


def test_load_model_local_only():
    """
    Test loading a model that does not exist locally.

    It does it without providing a remote URL.

    Should raise FileNotFoundError.
    """
    with pytest.raises(FileNotFoundError):
        load_model(local_path="models/nonexistent_model.pt")
