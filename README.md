# YOLOv1 TensorFlow — Arthropod Object Detection

This project trains and evaluates a YOLOv1-style object detector in TensorFlow/Keras on the Kaggle dataset “[Arthropod Taxonomy Orders Object Detection (v6).](https://www.kaggle.com/datasets/mistag/arthropod-taxonomy-orders-object-detection-dataset)” It includes two approaches:

- A model trained from scratch
- A transfer-learning model that uses MobileNetV2 (pretrained on ImageNet) as a feature extractor

Both training and inference workflows are implemented in the notebook `YOLO_V1.ipynb`, with utilities for data parsing, visualization, training curves, and prediction display.

## Folder structure

```
.
├─ LICENSE                 # Project license
├─ README.md               # This document
├─ YOLO_V1.ipynb           # Main notebook (data, training, inference)
└─ (no large model artifacts are tracked in this repo)
```

## Dataset

- Source: Kaggle — Arthropod Taxonomy Orders Object Detection Dataset (Version 6)
  - https://www.kaggle.com/datasets/mistag/arthropod-taxonomy-orders-object-detection-dataset/versions/6
- The dataset provides images and JSON annotations (bounding boxes and class tags) for multiple arthropod orders.
- The notebook expects the base path to be set via `ROOT_DATA`, for example:
  - On Kaggle: `"/kaggle/input/arthropod-taxonomy-orders-object-detection-dataset/ArTaxOr"`
  - Locally: update `ROOT_DATA` to your dataset location.

## What’s in the notebook

`YOLO_V1.ipynb` contains:

- Data utilities: recursive JSON discovery, annotation parsing to YOLOv1 grid format, dataset building (`tf.data`)
- Visualization: class distribution plots, image + bounding boxes render, training curves
- Models:
  - YOLOv1 from scratch (Conv/Pool blocks + FC head; output reshaped to S×S×(5B+C))
  - YOLOv1 with transfer learning (MobileNetV2 backbone + custom head)
- Custom loss: YOLOv1 loss (localization, confidence with no‑obj term, and classification)
- Training: callbacks (checkpointing, early stopping, LR scheduler), split into train/val/test with stratification
- Inference: batch sampling and side‑by‑side comparison of predictions vs ground truth

Key configuration:

- Grid size `S = 7`, boxes per cell `B = 2`, number of classes `C = 7`
- Image size `448×448`

## How to run

### Option A: Kaggle (recommended)

1. Open the notebook in a Kaggle Notebook environment.
2. Add the dataset as an input to the notebook.
3. Run all cells. Models and history files are saved under `/kaggle/working/` during training.

### Option B: Local execution

1. Create and activate a Python environment (Python 3.9–3.11 recommended).
2. Install dependencies (see below).
3. Download and extract the Kaggle dataset locally; update `ROOT_DATA` in the notebook to point to your dataset path.
4. Run the notebook with Jupyter or VS Code.

## Dependencies

The notebook installs some packages in-place and uses the following stack:

- TensorFlow 2.x and Keras
- NumPy, Pandas, Matplotlib, Seaborn
- scikit-learn
- tqdm

## Models and outputs

Trained model artifacts are not stored in this repository due to size constraints. You can download them from the Kaggle Notebook's Output tab:

- Kaggle notebook (Outputs): https://www.kaggle.com/code/jlgonrod/c04-01-yolov1

During Kaggle runs, files are written under `/kaggle/working/`, for example:

- `/kaggle/working/yolov1_model.h5` (YOLOv1 trained from scratch)
- `/kaggle/working/yolov1_model_tl.h5` (YOLOv1 transfer learning with MobileNetV2)
- `/kaggle/working/history_*.pkl` (training histories)

## License

See `LICENSE` for details.

## Acknowledgments

- Dataset by the Kaggle community: Arthropod Taxonomy Orders Object Detection
- MobileNetV2 weights from TensorFlow/Keras
- Kaggle GPU resources