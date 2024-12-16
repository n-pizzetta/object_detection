#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Video inference using YOLO model."""

import argparse
import os

import cv2
import yaml

from model_loader import load_model


def load_config(config_path="config.yaml"):
    """Load configuration from a YAML file."""
    if os.path.exists(config_path):
        with open(config_path, "r") as file:
            return yaml.safe_load(file)
    print(f"Config file {config_path} not found.")
    return {}


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="YOLO Video Inference")
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to configuration file.",
    )
    parser.add_argument(
        "--input",
        type=str,
        help="Path to input video file.\
            If not provided, webcam will be used.",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Path to save the output video.\
        If not provided, output will not be saved.",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        help="Path to the YOLO model file.",
    )
    parser.add_argument(
        "--remote_url",
        type=str,
        help="Remote URL to download the YOLO model if not found locally.",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        help="Confidence threshold for detections.",
    )
    parser.add_argument(
        "--iou",
        type=float,
        help="IoU threshold for Non-Max Suppression.",
    )
    parser.add_argument(
        "--device",
        type=str,
        help=(
            "Device to run the model on (e.g., 'cpu', 'cuda')."
            "If not specified, auto-detect."
        ),
    )
    args = parser.parse_args()
    return args


def main():
    """Execute main function."""
    args = parse_args()
    config = load_config(args.config)

    # Use CLI arguments to override the configuration
    input_video = args.input or config.get("paths", {}).get(
        "input_video", "data/test_video.mp4"
    )
    output_dir = args.output or config.get("paths", {}).get("output_directory", ".")
    model_path = args.model_path or config.get("paths", {}).get(
        "model_path", "models/yolov8n.pt"
    )
    remote_url = args.remote_url or config.get("paths", {}).get("remote_url")
    confidence = args.confidence or config.get("model", {}).get(
        "confidence_threshold", 0.25
    )
    iou = args.iou or config.get("model", {}).get("iou_threshold", 0.45)
    device = args.device or config.get("device", {}).get("default", "")

    # Load YOLO model
    try:
        model = load_model(
            local_path=model_path,
            remote_url=remote_url,
            auto_download=True,
        )
    except FileNotFoundError as e:
        print(e)
        return

    # Set device
    if args.device:
        model.to(device)
    else:
        model.to("cuda" if cv2.cuda.getCudaEnabledDeviceCount() > 0 else "cpu")

    # Initialize video capture
    if args.input:
        if not os.path.exists(args.input):
            print(f"Input video file {args.input} does not exist.")
            return

        # Run inference
        model.predict(
            source=input_video,
            conf=confidence,
            iou=iou,
            verbose=False,
            save=True,
            project=output_dir,
            name="results",
        )


if __name__ == "__main__":
    main()
