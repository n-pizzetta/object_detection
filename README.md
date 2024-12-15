# Object Detection

A YOLO-based object detection project with a clean setup using `pip`, pre-commit hooks, linting, and testing.

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

### Running Inference

```bash
from src.inference import run_inference

image_path = "path/to/your/image.jpg"
model_path = "models/yolov8n.pt"  # Local path
remote_url = "https://example.com/models/yolov8n.pt"  # Remote URL

results = run_inference(image_path, model_path=model_path, remote_url=remote_url)

# Process results as needed
print(results)
```

**Notes:**<br>
If the model isn't found locally at model_path, it will attempt to download from remote_url if provided.


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
