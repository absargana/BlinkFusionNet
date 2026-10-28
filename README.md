# BlinkFusionNet

Reference implementation for **BlinkFusionNet: Gated multimodal fusion for robust blink event detection in sign language corpora**.

The repository contains the final code used for preprocessing, training, and evaluation, together with the released semi-automatic ELAN blink-annotation tool. Dataset files and experimental outputs are not included. The annotation tool's verified inference checkpoint is distributed through Git LFS.

## Files

- `preprocess.py` filters and crops detected faces, computes dense optical flow, extracts MediaPipe eye landmarks, and verifies alignment between the generated streams.
- `train.py` trains the RGB, optical-flow, and geometric-stream model with feature-wise gated fusion and a bidirectional LSTM.
- `evaluate.py` performs non-overlapping window evaluation and stride-1 frame evaluation with mean probability aggregation.
- `annotation_tool/` contains the semi-automatic blink pre-annotation tool, its verified checkpoint, installation requirements, and detailed ELAN verification instructions.

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

## Blink annotation tool

The released annotation tool proposes blink intervals for sign-language videos and writes them to an ELAN `.eaf` file plus blink-only CSV/XLSX tables. It uses the published six-frame checkpoint and preserves the original preprocessing, model, probability aggregation, hysteresis thresholds, and temporal cleanup logic.

Input videos may use different constant frame rates. The tool reads each video's native FPS and samples it by timestamp onto the checkpoint's original 25-FPS temporal timeline; generated ELAN intervals remain aligned to the original video time. Built-in mapping checks cover 24, 25, 29.97, 30, 50, and 60 FPS.

This is a **semi-automatic pre-annotation tool**. Every proposed blink must be opened with the source video in ELAN and verified by a human annotator. Accuracy can vary with lighting, motion blur, face scale, head rotation, glasses, and hand–face occlusion. The unreviewed machine output should not be treated as final linguistic annotation.

The annotation tool has its own pinned dependency file. Install it from the tool directory rather than using the repository-level requirements:

```bash
git lfs install
git clone https://github.com/absargana/BlinkFusionNet.git
cd BlinkFusionNet/annotation_tool
git lfs pull
python -m venv .venv
# Activate .venv for your operating system, then run:
python -m pip install -r requirements.txt
python blink_annotator.py --self-test
```

See [`annotation_tool/README.md`](annotation_tool/README.md) for platform-specific installation commands, checkpoint verification, video-processing examples, outputs, troubleshooting, limitations, privacy guidance, and the required manual-verification workflow.

## Data availability

The source sign-language corpora and annotations are not redistributed in this repository. Obtain access from the respective corpus providers and follow their licence and consent conditions. Here is the link for French sign language videos: https://www.ortolang.fr/market/corpora/dicta-sign-lsf-v2/v1 

## Citation

If you use this code, please cite the associated article:

> Sargano, A. B., Miquel Blasco, C., Susman, M., Börstell, C., & Kimmelman, V. (2026). BlinkFusionNet: Gated multimodal fusion for robust blink event detection in sign language corpora. *Computer Vision and Image Understanding*, 104978. https://doi.org/10.1016/j.cviu.2026.104978

```bibtex
@article{sargano2026blinkfusionnet,
  title   = {BlinkFusionNet: Gated Multimodal Fusion for Robust Blink Event Detection in Sign Language Corpora},
  author  = {Sargano, Allah Bux and Miquel Blasco, Carla and Susman, Margaux and B{\"o}rstell, Carl and Kimmelman, Vadim},
  journal = {Computer Vision and Image Understanding},
  year    = {2026},
  pages   = {104978},
  doi     = {10.1016/j.cviu.2026.104978},
  url     = {https://doi.org/10.1016/j.cviu.2026.104978}
}
```

## Licence

This code is released under the MIT License. See `LICENSE`.
