"""Validation entry point for GLS-YOLO."""

from ultralytics import YOLO


def main():
    # Replace this path with the trained checkpoint if it is stored elsewhere.
    model = YOLO("runs/detect/gls-yolo/weights/best.pt")
    model.val(data="data/poppy.yaml", imgsz=640, batch=16, split="test")


if __name__ == "__main__":
    main()
