"""Training entry point for GLS-YOLO on the UAV poppy dataset."""

from ultralytics import YOLO


def main():
    model = YOLO("ultralytics/cfg/models/11/gls-yolo.yaml")
    model.train(
        data="data/poppy.yaml",
        epochs=100,
        imgsz=640,
        batch=16,
        workers=16,
        optimizer="auto",
        lr0=0.01,
        lrf=0.01,
        momentum=0.937,
        project="runs/detect",
        name="gls-yolo",
    )


if __name__ == "__main__":
    main()
