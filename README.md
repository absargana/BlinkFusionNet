# BlinkFusionNet

Reference implementation for **BlinkFusionNet: Gated multimodal fusion for robust blink event detection in sign language corpora**.

The repository contains the final code used for preprocessing, training, and evaluation. Dataset files, experimental outputs, and trained model checkpoints are not included.

## Files

- `preprocess.py` filters and crops detected faces, computes dense optical flow, extracts MediaPipe eye landmarks, and verifies alignment between the generated streams.
- `train.py` trains the RGB, optical-flow, and geometric-stream model with feature-wise gated fusion and a bidirectional LSTM.
- `evaluate.py` performs non-overlapping window evaluation and stride-1 frame evaluation with mean probability aggregation.

## Installation

Python 3.10 or later is recommended.

```bash
python -m venv .venv
# Linux/macOS
source .venv/bin/activate
# Windows PowerShell
# .venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Install a PyTorch build compatible with the CUDA version on your system when GPU acceleration is required. InsightFace may additionally require an ONNX Runtime build suitable for the target machine.

## Expected data layout

Each processed sequence directory is expected to contain:

```text
sequence_group_10/
├── frame_000001.jpg
├── frame_000002.jpg
├── ...
├── ground_truth.csv
├── mediapipe_landmarks.txt
└── optical_flow/
    ├── flow_000001.jpg
    └── ...
```

`ground_truth.csv` must contain a `frame_index` column and a binary or text blink-label column. Sequence directory names should end in `_group_6` or `_group_10`.

## Preprocessing

Set the input and output directories with environment variables. Defaults are `data/stage0` and `data/stage1`.

```bash
# Linux/macOS
export BLINK_INPUT_ROOT=/path/to/stage0
export BLINK_OUTPUT_ROOT=/path/to/stage1
python preprocess.py

# Windows PowerShell
$env:BLINK_INPUT_ROOT = "C:\path\to\stage0"
$env:BLINK_OUTPUT_ROOT = "C:\path\to\stage1"
python preprocess.py
```

Preprocessing modifies the copied Stage 1 data by removing frames without valid face detections and updating the corresponding ground-truth rows. Keep an unchanged copy of the source data.

## Training

Set `BLINK_TRAIN_DATA` to the processed training-data root:

```bash
export BLINK_TRAIN_DATA=/path/to/processed/train
python train.py
```

On Windows PowerShell:

```powershell
$env:BLINK_TRAIN_DATA = "C:\path\to\processed\train"
python train.py
```

The checked-in configuration trains the `T=10` model. To train `T=6`, set `ACCEPT_GROUPS = [6]` and `TIME_SEQUENCE = 6` in the `Config` class. Training writes the best checkpoint and its CSV history to the working directory.

## Evaluation

Set the test-data, checkpoint, and output paths:

```bash
export BLINK_TEST_DATA=/path/to/processed/test
export BLINK_MODEL_PATH=/path/to/checkpoint.pth
export BLINK_OUTPUT_DIR=/path/to/evaluation/output
python evaluate.py
```

On Windows PowerShell:

```powershell
$env:BLINK_TEST_DATA = "C:\path\to\processed\test"
$env:BLINK_MODEL_PATH = "C:\path\to\checkpoint.pth"
$env:BLINK_OUTPUT_DIR = "C:\path\to\evaluation\output"
python evaluate.py
```

The checked-in evaluation configuration is for `T=10`. For a `T=6` checkpoint, set `ACCEPT_GROUPS = [6]` and `TIME_SEQUENCE = 6` in the `Config` class.

## Data availability

The source sign-language corpora and annotations are not redistributed in this repository. Obtain access from the respective corpus providers and follow their licence and consent conditions.

## Citation

If you use this code, please cite the associated article:

> Sargano, A. B., Miquel Blasco, C., Susman, M., Börstell, C., & Kimmelman, V. (2026). BlinkFusionNet: Gated multimodal fusion for robust blink event detection in sign language corpora. *Computer Vision and Image Understanding*. In press. https://www.sciencedirect.com/science/article/pii/S1077314226003450

```bibtex
@article{sargano2026blinkfusionnet,
  title   = {BlinkFusionNet: Gated Multimodal Fusion for Robust Blink Event Detection in Sign Language Corpora},
  author  = {Sargano, Allah Bux and Miquel Blasco, Carla and Susman, Margaux and B{\"o}rstell, Carl and Kimmelman, Vadim},
  journal = {Computer Vision and Image Understanding},
  year    = {2026},
  note    = {In press},
  url     = {https://www.sciencedirect.com/science/article/pii/S1077314226003450}
}
```

## Licence

This code is released under the MIT License. See `LICENSE`.
