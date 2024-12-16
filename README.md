# Object Detection
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)

A YOLO-based object detection project with a clean setup using `pip`, pre-commit hooks, linting, and testing.

<p align="center">
  <img src="./results/result_example.gif" width=100%>
</p>

## 📋 Table of Contents

- [Features](#features)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
- [Usage](#usage)
- [Testing](#testing)
- [CI/CD](#cicd)
- [License](#license)

## 🚀 Features

- **YOLO Model Loading and Inference:** Efficient object detection using YOLOv8.
- **Local or Remote Model Management:** Automatically load models from local storage or download them if absent.
- **Code Quality Tools:** Automated formatting and linting with pre-commit hooks (Black, isort, Flake8).
- **Testing:** Robust testing framework using `pytest`.
- **Continuous Integration:** Automated testing and linting with GitHub Actions.

## 🛠 Getting Started

### Prerequisites

- Python 3.7 or higher
- Git

### Installation

1. **Clone the Repository:**

```bash
git clone https://github.com/n-pizzetta/object_detection.git
cd object_detection
```

2. **Create a Virtual Environment:**

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install Dependencies:**

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

4. **Install Pre-commit Hooks:**

```bash
pre-commit install
```

## 🎬 Usage

### Running Video Inference

Run the script to perform inference on a video file or webcam:

```bash
python src/video_inference.py --input <path_to_video> --output <output_directory>
```

For example:
```bash
python src/video_inference.py --input data/test_video.mp4 --output .
```

**Optional Arguments:**
- `--config <path>`: Path to a custom configuration file (default: `config.yaml`).
- `--model_path <path>`: Path to the YOLO model file (default: `models/yolov8n.pt`).
- `--remote_url <url>`: Remote URL to download the model if not found locally.
- `--confidence <float>`: Confidence threshold for detections (default: `0.25`).
- `--iou <float>`: IoU threshold for Non-Max Suppression (default: `0.45`).
- `--device <str>`: Device to run the model (`cpu` or `cuda`, default: auto-detect).

**Example:**
```bash
python src/video_inference.py \
  --input data/test_video.mp4 \
  --output results \
  --model_path models/yolov8n.pt \
  --confidence 0.3 \
  --iou 0.5 \
  --device cuda
```

**Notes:**<br>
If the model isn't found locally at `--model_path`, it will attempt to download from `--remote_url` if provided.

--- 

Let me know if you need additional details for this section! 🚀

## 🧪 Testing

Run tests using `pytest`:

```bash
pytest
```

## 🛠 CI/CD

The project uses GitHub Actions for Continuous Integration. On each push or pull request, the CI pipeline will:
- Checkout the code.
- Set up Python.
- Install dependencies.
- Run linting checks.
- Execute tests.


## 📄 License

This project is licensed under the MIT License. See the LICENSE file for details.
