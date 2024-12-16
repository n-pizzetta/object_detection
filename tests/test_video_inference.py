"""Tests for the video inference module."""

from unittest.mock import MagicMock, patch

from video_inference import main


@patch("os.path.exists", return_value=True)  # Mock file existence check
@patch("video_inference.load_model")  # Mock model loading
@patch("cv2.VideoCapture")  # Mock video capture
def test_main(mock_video_capture, mock_load_model, mock_path_exists):
    """Test the main function."""
    mock_capture = MagicMock()
    mock_capture.isOpened.return_value = True
    mock_capture.read.return_value = (True, MagicMock())
    mock_video_capture.return_value = mock_capture

    mock_model = MagicMock()
    mock_load_model.return_value = mock_model

    # Mock arguments
    test_args = [
        "--input",
        "test_video.mp4",
        "--output",
        "test_output",
        "--model_path",
        "models/yolov8n.pt",
        "--confidence",
        "0.5",
        "--iou",
        "0.5",
        "--device",
        "cpu",
    ]
    with patch("sys.argv", ["video_inference.py"] + test_args):
        main()
        mock_load_model.assert_called_once_with(
            local_path="models/yolov8n.pt",
            remote_url="https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt",
            auto_download=True,
        )
        mock_model.predict.assert_called_once_with(
            source="test_video.mp4",
            conf=0.5,
            iou=0.5,
            verbose=False,
            save=True,
            project="test_output",
            name="results",
        )
