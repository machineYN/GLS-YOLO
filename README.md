# GLS-YOLO: UAV Poppy Detection

## Description

This repository contains the code for **GLS-YOLO: A Gradient-Aware Local-Global Modulated Framework for UAV Poppy Detection**.

The project is based on Ultralytics YOLO11s and implements the final GLS-YOLO model described in the manuscript. The model adds three components to improve UAV-based poppy plant detection in complex farmland scenes:

- **GRAConv**: Gradient-Region Aggregation Convolution for preserving weak edge and shape cues during downsampling.
- **LGCM**: Lightweight Global-Local Context Modulator for shallow feature refinement.
- **SEAM**: Separated and Enhancement Attention Module for recalibrating fused neck features.

## Dataset Information

The UAV-based poppy plant detection dataset used in the manuscript is provided separately for PeerJ review and publication as **raw data**.

Dataset DOI:

```text
https://doi.org/10.6084/m9.figshare.32676051
```

Dataset summary:

```text
Original UAV images: 470
Augmented annotated images: 4,794
Training images: 3,835
Validation images: 479
Test images: 480
Object class: poppy
Annotation format: YOLO TXT bounding-box labels
Image format: JPG
```

Expected dataset directory format:

```text
VOC2007/
|-- images/
|   |-- train/
|   |-- val/
|   `-- test/
|-- labels/
|   |-- train/
|   |-- val/
|   `-- test/
|-- train.txt
|-- val.txt
`-- test.txt
```

The dataset path is configured in:

```text
data/poppy.yaml
```

Before running training or validation, update the `path` field in `data/poppy.yaml` if the dataset is stored in a different location.

## Code Information

Important files:

```text
GLS-Yolo/
|-- README.md
|-- pyproject.toml
|-- train_gls_yolo.py
|-- val_gls_yolo.py
|-- gls-yolo.yaml
|-- data/
|   `-- poppy.yaml
`-- ultralytics/
    |-- cfg/models/11/gls-yolo.yaml
    |-- nn/modules/gls.py
    |-- nn/modules/__init__.py
    `-- nn/tasks.py
```

The final GLS-YOLO model definition is:

```text
ultralytics/cfg/models/11/gls-yolo.yaml
```

The custom modules are implemented in:

```text
ultralytics/nn/modules/gls.py
```

## Requirements

The experiments reported in the manuscript used:

```text
Operating system: Ubuntu 20.04.5 LTS
Python: 3.9.0
PyTorch: 2.0.0
CUDA: cu117
Ultralytics: 8.3.169
GPU: NVIDIA GeForce RTX 3060
CPU: Intel Xeon E5-2683 v4
```

Install the repository in editable mode:

```bash
pip install -e .
```

If PyTorch is not already installed, install a CUDA-compatible PyTorch version first. For the environment used in the manuscript:

```bash
pip install torch==2.0.0 torchvision --index-url https://download.pytorch.org/whl/cu117
```

Core Python dependencies are declared in `pyproject.toml`.

## Usage Instructions

### 1. Prepare the dataset

Place the dataset in the expected YOLO format, then edit `data/poppy.yaml` if needed:

```yaml
path: D:/data/VOC2007
train: images/train
val: images/val
test: images/test

nc: 1
names:
  0: poppy
```

### 2. Train GLS-YOLO

Run:

```bash
python train_gls_yolo.py
```

Equivalent Ultralytics command:

```bash
yolo detect train model=ultralytics/cfg/models/11/gls-yolo.yaml data=data/poppy.yaml epochs=100 imgsz=640 batch=16 workers=16 optimizer=auto lr0=0.01 lrf=0.01 momentum=0.937
```

### 3. Validate GLS-YOLO

After training, validate the trained model:

```bash
python val_gls_yolo.py
```

Equivalent Ultralytics command:

```bash
yolo detect val model=runs/detect/gls-yolo/weights/best.pt data=data/poppy.yaml imgsz=640 batch=16 split=test
```

## Methodology

The workflow used in the manuscript is:

1. Collect UAV images of poppy plants under different flight heights, viewing angles, illumination conditions, and vegetation densities.
2. Annotate poppy plants with bounding boxes in YOLO TXT format.
3. Normalize images to a unified spatial resolution.
4. Apply offline augmentation, including horizontal flipping, vertical flipping, random region masking, and multi-scale random cropping.
5. Split the augmented dataset into training, validation, and test sets with an approximate 8:1:1 ratio.
6. Train GLS-YOLO for 100 epochs with input size 640 x 640 and batch size 16.
7. Evaluate using Precision, Recall, mAP@0.5, mAP@0.5:0.95, parameters, and GFLOPs.

## Results Reported in the Manuscript

```text
mAP@0.5:      0.958
mAP@0.5:0.95: 0.774
GFLOPs:       20.9
Parameters:   9.53M
```

## Citation

If you use this code or dataset, please cite the associated manuscript:

```text
Zhang H, Wang C, Feng J, Feng X. GLS-YOLO: A Gradient-Aware Local-Global Modulated Framework for UAV Poppy Detection.
```

## License

This project is based on Ultralytics YOLO and follows the AGPL-3.0 license. See `LICENSE` for details.

## Contribution Guidelines

This repository is provided to support review and reproduction of the manuscript. Issues or suggestions can be reported through the repository issue tracker if the code is hosted on a public repository.
