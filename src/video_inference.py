#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Video inference using YOLO model."""

import argparse
import os

import cv2

from src.model_loader import load_model


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="YOLO Video Inference")
    parser.add_argument(
        "--input",
        type=str,
        default="",
        help="Path to input video file.\
            If not provided, webcam will be used.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="",
        help="Path to save the output video.\
        If not provided, output will not be saved.",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="models/yolov8n.pt",
        help="Path to the YOLO model file.",
    )
    parser.add_argument(
        "--remote_url",
        type=str,
        default=None,
        help="Remote URL to download the YOLO model if not found locally.",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.25,
        help="Confidence threshold for detections.",
    )
    parser.add_argument(
        "--iou",
        type=float,
        default=0.45,
        help="IoU threshold for Non-Max Suppression.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="",
        help="Device to run the model on (e.g., 'cpu', 'cuda').\
            If not specified, auto-detect.",
    )
    args = parser.parse_args()
    return args


def main():
    """Execute main function."""
    args = parse_args()

    # Load YOLO model
    try:
        model = load_model(
            local_path=args.model_path,
            remote_url=args.remote_url,
            auto_download=True,
        )
    except FileNotFoundError as e:
        print(e)
        return

    # Set device
    if args.device:
        model.to(args.device)
    else:
        model.to("cuda" if cv2.cuda.getCudaEnabledDeviceCount() > 0 else "cpu")

    # Initialize video capture
    if args.input:
        if not os.path.exists(args.input):
            print(f"Input video file {args.input} does not exist.")
            return
        cap = cv2.VideoCapture(args.input)
    else:
        cap = cv2.VideoCapture(0)  # Use webcam

    if not cap.isOpened():
        print("Error: Could not open video source.")
        return

    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = (
        cap.get(cv2.CAP_PROP_FPS) if args.input else 30.0
    )  # Default to 30 FPS for webcam

    # Initialize video writer if output path is provided
    if args.output:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(args.output, fourcc, fps, (width, height))
        if not out.isOpened():
            print(
                f"Error: Could not open video writer with path {args.output}.",
            )
            return
    else:
        out = None

    print("Starting video inference. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("End of video stream.")
            break

        # Run inference
        results = model.predict(
            frame, conf=args.confidence, iou=args.iou, verbose=False
        )

        # Render results on the frame
        annotated_frame = results[0].plot()

        # Display the frame
        cv2.imshow("YOLO Video Inference", annotated_frame)

        # Write the frame to the output video if writer is initialized
        if out:
            out.write(annotated_frame)

        # Exit on 'q' key
        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("Quitting video inference.")
            break

    # Release resources
    cap.release()
    if out:
        out.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
