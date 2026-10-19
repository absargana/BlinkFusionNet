# BlinkFusionNet Semi-Automatic Blink Annotation Tool

This tool generates **candidate eye-blink annotations** from sign-language videos. It writes the detected blink intervals to an ELAN `.eaf` file and to blink-only CSV/XLSX tables. The ELAN output is intended to be opened with the original video so that an annotator can verify, correct, add or remove blink events.

The tool is therefore **semi-automatic**, not a replacement for human annotation. Its reliability depends on the video, including face scale, lighting, motion blur, head rotation, glasses and hand–face occlusion. 

## What has—and has not—changed from the original version

The released inference logic is preserved from the original 25-FPS annotation tool:

- the same `model_gated_fusion_seed_123.pth` checkpoint;
- six-frame temporal windows;
- InsightFace largest-face cropping, resized to 256 × 192 pixels;
- Farneback optical flow with the original parameters;
- MediaPipe eye landmarks and the original eye indices;
- the same RGB, optical-flow and eye-geometry model inputs;
- maximum aggregation of overlapping-window probabilities;
- hysteresis thresholds `0.30/0.18`;
- minimum blink length of three model frames;
- merge gap of three model frames; and
- the original merge/prune ordering.

The only scientific extension is **automatic input-frame-rate handling**. The checkpoint was developed on a 25-FPS temporal timeline, so changing the model to run natively at 30 or 60 FPS would change the duration represented by each six-frame sequence. This release instead keeps the model at 25 FPS.

## Frame-rate handling

For every video, the tool reads the native FPS and constructs a uniform 25-FPS model timeline:

```text
source video at native FPS
        ↓
nearest source frame at each 25-FPS model timestamp
        ↓
unchanged BlinkFusionNet inference
        ↓
blink intervals written at original-video times
```

Examples:

| Source video | Processing |
|---|---|
| 25 FPS | Every source frame is used directly. |
| 29.97 or 30 FPS | Source frames nearest the 25-FPS timestamps are selected. |
| 50 FPS | Approximately every second source frame is selected. |
| 60 FPS | Source frames nearest each 40-ms model timestamp are selected. |
| Below 25 FPS | Nearest source frames must sometimes be repeated. The tool warns that extra manual review is required. |

The mapping uses timestamps rather than a rounded integer frame interval. A `frame_mapping.csv` file records the processed frame, model-timeline frame, original source frame and timestamp.

### Constant versus variable frame rate

This release supports ordinary **constant-frame-rate** videos, including fractional rates such as 23.976 and 29.97 FPS. OpenCV supplies the source FPS and frame count.

For a genuinely variable-frame-rate video, first convert it to a constant-frame-rate video while preserving duration and audio/video synchronization. The tool stops when it cannot obtain a valid FPS rather than silently creating inaccurate ELAN times.

## Requirements

- Python 3.9 or a compatible Python version
- Windows, Linux or macOS
- CPU processing is supported; a compatible NVIDIA GPU is faster
- Sufficient disk space for temporary face crops and optical-flow images
- ELAN for manual verification of the generated `.eaf` files

Install the Python dependencies in a dedicated environment:

```bash
python -m venv blinkfusionnet_env
```

Activate it, then install:

```bash
pip install -r requirements.txt
```

The provided requirements use CPU ONNX Runtime for portability. Users with a compatible NVIDIA/CUDA installation may replace `onnxruntime` with the matching `onnxruntime-gpu` package. Install PyTorch using the command recommended for your operating system and CUDA version if the standard installation does not match your hardware.

InsightFace may download its `buffalo_l` model files on the first run. Internet access may therefore be required once. Users are responsible for reviewing the licences and usage conditions of third-party models and dependencies.

## Checkpoint

Place the released checkpoint beside the script:

```text
annotation_tool/
├── blink_annotator.py
├── model_gated_fusion_seed_123.pth
├── requirements.txt
└── README.md
```

The tool verifies the exact released checkpoint before processing:

```text
SHA-256: CFEB5281BB2CE56A08D8234BA4F8D3C15E0CC48F2055575F647B5B1CBECB8D65
```

If the hash differs, processing stops. This prevents accidental use of an incompatible model while appearing to reproduce the published tool.

## Quick start

Process one video:

```bash
python blink_annotator.py --input path/to/video.mp4 --output path/to/results
```

Process every supported video directly inside a folder:

```bash
python blink_annotator.py --input path/to/videos --output path/to/results
```

Specify the checkpoint explicitly when it is not beside the script:

```bash
python blink_annotator.py --input path/to/videos --output path/to/results --model path/to/model_gated_fusion_seed_123.pth
```

Existing result folders are protected. To intentionally replace the result for a video:

```bash
python blink_annotator.py --input path/to/video.mp4 --output path/to/results --overwrite
```

Check frame-rate mapping without loading the model or processing a video:

```bash
python blink_annotator.py --self-test
```

## Outputs

For `example.mp4`, the tool creates:

```text
results/
└── example/
    ├── frames/
    │   ├── 000001.bmp
    │   ├── ...
    │   ├── frame_mapping.csv
    │   ├── mediapipe_landmarks.txt
    │   └── optical_flow/
    └── annotations/
        ├── blinks_example.eaf
        ├── predictions_example.csv
        ├── predictions_example.xlsx
        └── processing_summary.json
```

The CSV and XLSX prediction tables intentionally contain only frames classified as blink frames. The remaining processed frames are non-blink predictions. The ELAN file contains only blink intervals on the `Blinks` tier.

`frame_mapping.csv` and `processing_summary.json` are traceability files, not additional linguistic annotation tiers.

## Required manual verification in ELAN

1. Open `blinks_<video-name>.eaf` in ELAN.
2. If ELAN does not locate the media automatically, link the original source video through **Edit → Linked Files**.
3. Inspect every generated interval on the `Blinks` tier.
4. Correct interval boundaries when necessary.
5. Delete false-positive blink intervals.
6. Add missed blink intervals.
7. Save the verified EAF under a new name so the machine-generated pre-annotation remains available for audit.

Recommended naming:

```text
blinks_example_machine.eaf
blinks_example_human_verified.eaf
```

The human-verified file—not the unreviewed machine output—should be treated as the final linguistic annotation.

## Interpretation and limitations

The tool detects eye blinks, not every possible eye-related non-manual marker. Performance can be affected by:

- severe hand–face or hand–eye occlusion;
- strong out-of-plane head rotation;
- small or low-resolution faces;
- motion blur or dropped video frames;
- glare, dark glasses or reflections;
- unusual lighting;
- failure of face or eye-landmark detection; and
- videos whose temporal characteristics differ substantially from the development data.

Because sign-language recordings often contain rapid manual activity near the face, human verification is a required part of the intended workflow. The tool reduces the amount of video that must be searched manually by proposing candidate blink intervals; it does not guarantee that every blink is found or that every proposal is correct.

## Reproducibility note

The model FPS, architecture and operating point are intentionally not exposed as command-line tuning parameters. Changing the six-frame sequence length, 25-FPS model timeline, hysteresis thresholds or temporal cleanup rules would create a different method and should be separately validated.

## Citation

If you use this annotation tool or the BlinkFusionNet model, please cite:

> *BlinkFusionNet: Gated Multimodal Fusion for Robust Blink Event Detection in Sign Language Corpora*. Computer Vision and Image Understanding, 2026. <https://doi.org/10.1016/j.cviu.2026.104978>

Source repository: <https://github.com/absargana/BlinkFusionNet>

## Privacy

Face crops and optical-flow images are written to the result directory. These files may constitute sensitive biometric data. Store, share and delete them according to the consent, ethics approval and data-management requirements governing the source videos.
