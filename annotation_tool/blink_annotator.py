"""Semi-automatic BlinkFusionNet annotation tool.

The scientific inference path is intentionally identical to the original
25-FPS annotation script: InsightFace largest-face crops, Farneback optical
flow, MediaPipe eye landmarks, the six-frame BlinkFusionNet checkpoint,
max probability aggregation, 0.30/0.18 hysteresis, and 3-frame cleanup.

The only scientific extension is frame-rate handling.  Every input video is
sampled by timestamp onto the checkpoint's original 25-FPS model timeline.
Predicted blink intervals are then written at the corresponding times in the
original video so that the generated ELAN file stays synchronized.

This is a semi-automatic pre-annotation tool.  Its blink annotations must be
opened with the source video in ELAN and verified/corrected by a human.
"""

import os
import re
import sys
import math
import glob
import time
import argparse
import datetime
import hashlib
import json
import mimetypes
import shutil
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

import cv2

import torch
import torch.nn as nn
from PIL import Image
# ========= Third-party models / transforms =========
# Avoid a network version check during an otherwise local annotation run.
os.environ.setdefault('NO_ALBUMENTATIONS_UPDATE', '1')
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torchvision.models import efficientnet_b0

# ----------------------------
# Configuration
# ----------------------------
class Cfg:
    IMAGE_SIZE = 128
    TIME_SEQUENCE = 6
    # Temporal sampling used to train/evaluate the released checkpoint.
    # This is deliberately fixed; native input FPS is detected automatically.
    FRAME_RATE = 25.0
    GEOM_FEAT_DIM = 4  # (EAR, dEAR, 2 dummies) to match trained checkpoint
    NUM_CLASSES = 2
    DROPOUT = 0.5

    # Aggregation for smearing window prob across T frames
    PROB_AGG = 'mean'  # will be set to 'max' in recall-friendly mode

    # Hysteresis and cleanup (may be overridden)
    HYST_HIGH = 0.35
    HYST_LOW = 0.25
    MIN_BLINK_LEN = 4
    MERGE_GAP = 2

    CHECKPOINT_FILENAME = "model_gated_fusion_seed_123.pth"
    CHECKPOINT_SHA256 = "CFEB5281BB2CE56A08D8234BA4F8D3C15E0CC48F2055575F647B5B1CBECB8D65"
    VIDEO_EXTENSIONS = ('.mp4', '.avi', '.mov', '.mkv', '.mpeg', '.mpg')


# ----------------------------
# Utilities
# ----------------------------
def tryint(s):
    try:
        return int(s)
    except:
        return s

def alphanum_key(s):
    return [tryint(c) for c in re.split('([0-9]+)', s)]

def natural_sort(l):
    return sorted(l, key=alphanum_key)


def ensure_dir(p):
    os.makedirs(p, exist_ok=True)
    return p


# ----------------------------
# Video → Cropped-face frames
# ----------------------------
def build_model_timeline(total_frames: int, source_fps: float, model_fps: float):
    """Map a constant-rate source video onto the checkpoint's FPS timeline.

    The mapping is timestamp based rather than an integer frame step.  It is
    therefore valid for rates such as 23.976, 25, 29.97, 30, 50 and 60 FPS.
    Higher-rate videos are downsampled; lower-rate videos necessarily repeat
    nearest source frames and are reported with a quality warning.
    """
    if total_frames <= 0:
        raise RuntimeError("The video reports no readable frames.")
    if not math.isfinite(source_fps) or source_fps <= 0:
        raise RuntimeError("The video frame rate could not be determined.")
    if not math.isfinite(model_fps) or model_fps <= 0:
        raise RuntimeError("The model reference frame rate must be positive.")

    duration_s = (total_frames - 1) / source_fps
    model_count = int(math.floor(duration_s * model_fps + 1e-9)) + 1
    model_indices = np.arange(model_count, dtype=np.int64)
    model_times_ms = model_indices.astype(np.float64) * (1000.0 / model_fps)
    source_indices = np.rint((model_times_ms / 1000.0) * source_fps).astype(np.int64)
    source_indices = np.clip(source_indices, 0, total_frames - 1)
    source_times_ms = source_indices.astype(np.float64) * (1000.0 / source_fps)
    return source_indices, model_times_ms, source_times_ms


def crop_faces_from_video(video_path: str, frames_out: str, fps: float):
    """Apply the original largest-face crop on the 25-FPS model timeline.

    As in the original annotation tool, a model-time sample is omitted when no
    face is detected or the crop is empty.  A mapping table is retained so that
    this historical behaviour cannot compress or shift exported ELAN times.
    """
    ensure_dir(frames_out)
    try:
        from insightface.app import FaceAnalysis
    except Exception as e:
        raise RuntimeError("insightface is required for face cropping. Install it first.") from e

    app = FaceAnalysis()
    app.prepare(ctx_id=0 if torch.cuda.is_available() else -1)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    vid_fps = float(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if not math.isfinite(vid_fps) or vid_fps <= 0:
        cap.release()
        raise RuntimeError(
            f"Unable to read a valid FPS from {video_path}. Convert the video "
            "to a constant-frame-rate file before annotation."
        )
    if total_frames <= 0:
        cap.release()
        raise RuntimeError(f"Unable to read the frame count from {video_path}.")

    source_indices, model_times_ms, source_times_ms = build_model_timeline(
        total_frames, vid_fps, fps
    )
    schedule = {}
    for model_idx, source_idx in enumerate(source_indices.tolist()):
        schedule.setdefault(int(source_idx), []).append(model_idx)

    print(
        f"Source FPS={vid_fps:.6g}; model FPS={fps:.6g}; "
        f"source frames={total_frames}; model-time samples={len(source_indices)}"
    )
    if vid_fps < fps - 0.05:
        print(
            "WARNING: source FPS is below the 25-FPS model reference. Nearest "
            "frames will be repeated; all blink annotations require careful review."
        )

    mapping_rows = []
    saved_idx = 1
    frame_idx = 0

    pbar = tqdm(desc=f"Cropping faces: {os.path.basename(video_path)}", unit='frame')
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx in schedule:
            faces = app.get(frame)
            if len(faces) > 0:
                bboxes = [f['bbox'] for f in faces]
                areas = [(b[2]-b[0])*(b[3]-b[1]) for b in bboxes]
                i = int(np.argmax(areas))
                x1, y1, x2, y2 = map(int, bboxes[i])
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
                crop = frame[y1:y2, x1:x2]
                if crop.size == 0:
                    frame_idx += 1
                    pbar.update(1)
                    continue
                resized = cv2.resize(crop, (256, 192))
                for model_idx in schedule[frame_idx]:
                    out_name = f"{saved_idx:06d}.bmp"
                    out_path = os.path.join(frames_out, out_name)
                    if not cv2.imwrite(out_path, resized):
                        cap.release()
                        pbar.close()
                        raise RuntimeError(f"Could not write cropped frame: {out_path}")
                    mapping_rows.append({
                        'processed_frame': saved_idx - 1,
                        'model_frame': int(model_idx),
                        'source_frame': int(frame_idx),
                        'time_ms': float(model_times_ms[model_idx]),
                        'nearest_source_time_ms': float(source_times_ms[model_idx]),
                        'source_fps': float(vid_fps),
                        'model_fps': float(fps),
                        'frame_file': out_name,
                    })
                    saved_idx += 1
        frame_idx += 1
        pbar.update(1)

    pbar.close()
    cap.release()
    mapping = pd.DataFrame(mapping_rows)
    if len(mapping) < Cfg.TIME_SEQUENCE:
        raise RuntimeError(
            f"Only {len(mapping)} usable face crops were created; at least "
            f"{Cfg.TIME_SEQUENCE} are required."
        )
    mapping.to_csv(os.path.join(frames_out, 'frame_mapping.csv'), index=False)
    dropped = len(source_indices) - len(mapping)
    if dropped:
        print(
            f"Face detection omitted {dropped}/{len(source_indices)} model-time "
            "samples, matching the original tool's behaviour."
        )
    return mapping


# ----------------------------
# Optical flow per frames folder
# ----------------------------
def compute_optical_flow_for_folder(frames_dir: str, flow_subfolder: str = "optical_flow"):
    image_files = natural_sort([
        f for f in os.listdir(frames_dir)
        if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))
    ])
    if not image_files:
        return

    flow_out = ensure_dir(os.path.join(frames_dir, flow_subfolder))

    prev = cv2.imread(os.path.join(frames_dir, image_files[0]))
    if prev is None:
        raise RuntimeError(f"Could not read first frame in {frames_dir}")
    prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
    cv2.imwrite(os.path.join(flow_out, image_files[0]), np.zeros_like(prev))

    for i in tqdm(range(1, len(image_files)), desc=f"Optical flow: {os.path.basename(frames_dir)}"):
        cur = cv2.imread(os.path.join(frames_dir, image_files[i]))
        if cur is None:
            cv2.imwrite(os.path.join(flow_out, image_files[i]), np.zeros_like(prev))
            continue
        cur_gray = cv2.cvtColor(cur, cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(prev_gray, cur_gray, None,
                                            pyr_scale=0.5, levels=3, winsize=15,
                                            iterations=3, poly_n=5, poly_sigma=1.2, flags=0)
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        hsv = np.zeros_like(cur)
        hsv[..., 0] = ang * 180 / np.pi / 2
        hsv[..., 1] = 255
        hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        cv2.imwrite(os.path.join(flow_out, image_files[i]), bgr)
        prev_gray = cur_gray


# ----------------------------
# Eye landmarks per frames folder
# ----------------------------
def extract_eye_landmarks_for_folder(frames_dir: str, out_name: str = "mediapipe_landmarks.txt"):
    try:
        import mediapipe as mp
    except Exception as e:
        raise RuntimeError("mediapipe is required for eye landmarks. Install it first.") from e

    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.2,
    )

    LEFT = [362, 385, 387, 263, 373, 380]
    RIGHT = [33, 160, 158, 133, 153, 144]

    image_files = natural_sort([
        f for f in os.listdir(frames_dir)
        if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))
    ])

    last_good = None
    all_lm = []

    for f in tqdm(image_files, desc=f"Landmarks: {os.path.basename(frames_dir)}"):
        p = os.path.join(frames_dir, f)
        img = cv2.imread(p)
        if img is None:
            all_lm.append(last_good if last_good is not None else np.zeros(24))
            continue
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        res = face_mesh.process(img_rgb)
        if res.multi_face_landmarks:
            fl = res.multi_face_landmarks[0]
            h, w, _ = img.shape
            L = np.array([[fl.landmark[i].x * w, fl.landmark[i].y * h] for i in LEFT])
            R = np.array([[fl.landmark[i].x * w, fl.landmark[i].y * h] for i in RIGHT])
            lm = np.concatenate([L, R]).flatten()  # 12*2 = 24
            last_good = lm
            all_lm.append(lm)
        else:
            all_lm.append(last_good if last_good is not None else np.zeros(24))

    np.savetxt(os.path.join(frames_dir, out_name), np.array(all_lm), fmt='%.4f')
    face_mesh.close()


# ----------------------------
# Model definition (matches checkpoint naming)
# ----------------------------
class Augmentation:
    def __init__(self, size: int):
        self.tf = A.Compose([
            A.Resize(size, size),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])
    def __call__(self, img_np: np.ndarray):
        return self.tf(image=img_np)['image']

class EfficientNetStream(nn.Module):
    def __init__(self, out_dim: int):
        super().__init__()
        base = efficientnet_b0(weights=None)
        self.features = nn.Sequential(*list(base.children())[:-1])  # [features, avgpool]
        in_feat = base.classifier[1].in_features
        self.fc = nn.Linear(in_feat, out_dim)
    def forward(self, x):  # x: [B, T, C, H, W]
        B, T, C, H, W = x.size()
        x = x.view(B*T, C, H, W)
        feat = self.features(x)              # [B*T, 1280, 1, 1]
        feat = feat.squeeze(-1).squeeze(-1)  # [B*T, 1280]
        feat = self.fc(feat)                 # [B*T, out_dim]
        return feat.view(B, T, -1)           # [B, T, out_dim]

class TemporalBiLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes, dropout):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)
    def forward(self, x):  # x: [B, T, D]
        o, _ = self.lstm(x)                 # [B, T, 2H]
        x = self.drop(o[:, -1, :])          # last timestep
        return self.fc(x)                   # [B, num_classes]

class GatedFusion(nn.Module):
    def __init__(self, input_dim, hidden_dim=32):
        super().__init__()
        self.gating_network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid(),
        )
    def forward(self, x):
        return x * self.gating_network(x)

class DualStreamBlinkModel(nn.Module):
    def __init__(self, cfg: Cfg):
        super().__init__()
        self.spatial_stream = EfficientNetStream(256)
        self.temporal_stream = EfficientNetStream(256)
        fusion_in = 256*2 + cfg.GEOM_FEAT_DIM
        self.gated_fusion = GatedFusion(fusion_in)
        self.temporal = TemporalBiLSTM(fusion_in, 128, cfg.NUM_CLASSES, cfg.DROPOUT)
    def forward(self, rgb, flow, geom):
        rgb_f  = self.spatial_stream(rgb)   # [B, T, 256]
        flow_f = self.temporal_stream(flow) # [B, T, 256]
        fused = torch.cat([rgb_f, flow_f, geom], dim=-1)
        fused = self.gated_fusion(fused)
        return self.temporal(fused)


# ----------------------------
# Inference helpers
# ----------------------------

def calculate_ear(lm):
    A = np.linalg.norm(lm[1] - lm[5])
    B = np.linalg.norm(lm[2] - lm[4])
    C = np.linalg.norm(lm[0] - lm[3])
    return (A + B) / (2.0 * C) if C > 1e-6 else 3.0


def aggregate_probs(probs_per_frame: List[List[float]], method: str):
    if method == 'max':
        return np.array([max(p) if p else 0.0 for p in probs_per_frame], dtype=np.float32)
    return np.array([float(np.mean(p)) if p else 0.0 for p in probs_per_frame], dtype=np.float32)


def apply_hysteresis(probs: np.ndarray, high: float, low: float):
    out = np.zeros_like(probs, dtype=np.int32)
    on = False
    for i, p in enumerate(probs):
        if not on and p >= high:
            on = True
        elif on and p < low:
            on = False
        out[i] = 1 if on else 0
    return out


def merge_and_prune(preds: np.ndarray, min_len: int, merge_gap: int):
    n = len(preds)
    segs, i = [], 0
    while i < n:
        if preds[i] == 1:
            j = i
            while j+1 < n and preds[j+1] == 1:
                j += 1
            segs.append([i, j])
            i = j + 1
        else:
            i += 1
    segs = [s for s in segs if (s[1]-s[0]+1) >= min_len]
    merged = []
    for s in segs:
        if not merged:
            merged.append(s)
        else:
            if s[0] - merged[-1][1] - 1 <= merge_gap:
                merged[-1][1] = s[1]
            else:
                merged.append(s)
    out = np.zeros(n, dtype=np.int32)
    for s in merged:
        out[s[0]:s[1]+1] = 1
    return out


# ----------------------------
# Dataset wrapper for a frames folder
# ----------------------------
class FramesFolder:
    def __init__(self, frames_dir: str, cfg: Cfg):
        self.cfg = cfg
        self.tf = Augmentation(cfg.IMAGE_SIZE)
        self.frames_dir = frames_dir
        self.image_paths = [os.path.join(frames_dir, f) for f in natural_sort([
            f for f in os.listdir(frames_dir)
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))
        ])]
        flow_dir = os.path.join(frames_dir, "optical_flow")
        self.flow_paths = [os.path.join(flow_dir, f) for f in natural_sort([
            f for f in os.listdir(flow_dir)
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))
        ])] if os.path.isdir(flow_dir) else []
        lm_path = os.path.join(frames_dir, "mediapipe_landmarks.txt")
        self.landmarks = np.loadtxt(lm_path).reshape(-1, 12, 2) if os.path.isfile(lm_path) else None

    def __len__(self):
        return len(self.image_paths)


# ----------------------------
# ELAN + CSV exporters
# ----------------------------
import xml.etree.ElementTree as ET
from xml.dom import minidom

def blink_segments(predictions: np.ndarray, min_duration_frames: int):
    blinks = []
    is_on = False
    start = 0
    for i, p in enumerate(predictions):
        if p == 1 and not is_on:
            is_on = True
            start = i
        elif p == 0 and is_on:
            is_on = False
            if (i - start) >= min_duration_frames:
                blinks.append((start, i-1))
    if is_on and (len(predictions) - start) >= min_duration_frames:
        blinks.append((start, len(predictions)-1))
    return blinks


def event_boundary_ms(frame_index: int, mapping: pd.DataFrame, model_fps: float):
    if frame_index < len(mapping):
        return int(round(float(mapping.iloc[frame_index]['time_ms'])))
    return int(round(float(mapping.iloc[-1]['time_ms']) + 1000.0 / model_fps))


def save_blinks_to_elan(video_path: str, predictions: np.ndarray, mapping: pd.DataFrame,
                        model_fps: float, out_dir: str, min_duration_frames: int):
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    blinks = blink_segments(predictions, min_duration_frames)

    def frame_to_ms(f):
        return str(event_boundary_ms(f, mapping, model_fps))

    root = ET.Element('ANNOTATION_DOCUMENT', {
        'AUTHOR': 'Blink-Detector', 'DATE': datetime.date.today().isoformat(),
        'FORMAT': '3.0', 'VERSION': '3.0'
    })
    header = ET.SubElement(root, 'HEADER', {'MEDIA_FILE': '', 'TIME_UNITS': 'milliseconds'})
    resolved_video = Path(video_path).resolve()
    mime_type = mimetypes.guess_type(str(resolved_video))[0] or 'video/mp4'
    ET.SubElement(header, 'MEDIA_DESCRIPTOR', {
        'MEDIA_URL': resolved_video.as_uri(),
        'MIME_TYPE': mime_type,
        'RELATIVE_MEDIA_URL': os.path.basename(video_path),
    })
    ET.SubElement(header, 'PROPERTY', {'NAME': 'URN', 'VALUE': 'urn:nl-mpi-tools-elan-eaf:generated'})
    time_order = ET.SubElement(root, 'TIME_ORDER')

    ts_map, ts_id = {}, 1
    for s, e in blinks:
        for f in [s, e+1]:
            if f not in ts_map:
                tid = f"ts{ts_id}"; ts_map[f] = tid; ts_id += 1
                ET.SubElement(time_order, 'TIME_SLOT', {'TIME_SLOT_ID': tid, 'TIME_VALUE': frame_to_ms(f)})

    tier = ET.SubElement(root, 'TIER', {'LINGUISTIC_TYPE_REF': 'default-lt', 'TIER_ID': 'Blinks'})
    a_id = 1
    for s, e in blinks:
        ann = ET.SubElement(tier, 'ANNOTATION')
        align = ET.SubElement(ann, 'ALIGNABLE_ANNOTATION', {
            'ANNOTATION_ID': f"a{a_id}",
            'TIME_SLOT_REF1': ts_map[s],
            'TIME_SLOT_REF2': ts_map[e+1],
        })
        ET.SubElement(align, 'ANNOTATION_VALUE').text = "blink"
        a_id += 1

    ET.SubElement(root, 'LINGUISTIC_TYPE', {'GRAPHIC_REFERENCES': 'false', 'LINGUISTIC_TYPE_ID': 'default-lt'})
    pretty = minidom.parseString(ET.tostring(root, encoding='utf-8')).toprettyxml(indent="  ")
    ensure_dir(out_dir)
    fp = os.path.join(out_dir, f"blinks_{video_name}.eaf")
    with open(fp, 'w', encoding='utf-8') as f:
        f.write(pretty)
    return fp


def save_predictions_csv(video_name: str, probs: np.ndarray, preds_bin: np.ndarray,
                         mapping: pd.DataFrame, out_dir: str):
    if not (len(probs) == len(preds_bin) == len(mapping)):
        raise RuntimeError("Prediction and frame-mapping lengths do not match.")
    df = pd.DataFrame({
        'frame': np.arange(len(probs)),
        'model_frame': mapping['model_frame'].to_numpy(dtype=int),
        'source_frame': mapping['source_frame'].to_numpy(dtype=int),
        'time_ms': np.rint(mapping['time_ms'].to_numpy(dtype=float)).astype(int),
        'prob_blink': probs,
        'pred_binary': preds_bin,
    })

    # ✅ Keep only blink frames
    df = df[df['pred_binary'] == 1].reset_index(drop=True)

    ensure_dir(out_dir)
    fp_csv = os.path.join(out_dir, f"predictions_{video_name}.csv")
    df.to_csv(fp_csv, index=False)
    fp_xlsx = os.path.join(out_dir, f"predictions_{video_name}.xlsx")
    df.to_excel(fp_xlsx, index=False)
    return fp_csv, fp_xlsx



# ----------------------------
# Core inference on one frames folder
# ----------------------------
@torch.no_grad()
def infer_on_frames_folder(frames_dir: str, model: nn.Module, device: torch.device, cfg: Cfg):
    ds = FramesFolder(frames_dir, cfg)
    if len(ds.image_paths) == 0:
        raise RuntimeError(f"No frames found in {frames_dir}")
    if ds.flow_paths and len(ds.flow_paths) != len(ds.image_paths):
        raise RuntimeError(f"Mismatch: {len(ds.image_paths)} frames vs {len(ds.flow_paths)} flow images in {frames_dir}")
    if ds.landmarks is None:
        raise RuntimeError(f"Missing mediapipe_landmarks.txt in {frames_dir}")

    # Preload tensors once
    rgb_t, flow_t = [], []
    for i in range(len(ds)):
        rgb_np  = np.array(Image.open(ds.image_paths[i]).convert('RGB'))
        rgb_t.append(ds.tf(rgb_np))
        flow_np = np.array(Image.open(ds.flow_paths[i]).convert('RGB')) if ds.flow_paths else rgb_np
        flow_t.append(ds.tf(flow_np))
    rgb_t  = torch.stack(rgb_t)
    flow_t = torch.stack(flow_t)

    n = min(len(ds), len(rgb_t), len(flow_t), len(ds.landmarks))
    probs_per_frame: List[List[float]] = [[] for _ in range(n)]
    model.eval()

    for start in tqdm(range(n - cfg.TIME_SEQUENCE + 1), desc=f"Sliding window: {os.path.basename(frames_dir)}"):
        idxs = slice(start, start + cfg.TIME_SEQUENCE)
        rgb_tensor  = rgb_t[idxs].unsqueeze(0).to(device)
        flow_tensor = flow_t[idxs].unsqueeze(0).to(device)

        # Geometric features (EAR + delta + 2 dummies to keep GEOM=4)
        geom_seq = []
        prev_ear = -1.0
        for fidx in range(start, start + cfg.TIME_SEQUENCE):
            lm = ds.landmarks[fidx]
            l_ear = calculate_ear(lm[:6])
            r_ear = calculate_ear(lm[6:12])
            ear = (l_ear + r_ear) / 2.0
            d_ear = (ear - prev_ear) if prev_ear >= 0 else 0.0
            geom_seq.append([ear, d_ear, 0.0, 0.0])
            prev_ear = ear
        geom_tensor = torch.tensor(np.array(geom_seq, dtype=np.float32)).unsqueeze(0).to(device)

        logits = model(rgb_tensor, flow_tensor, geom_tensor)
        prob_blink = torch.softmax(logits, dim=-1)[0, 1].item()
        for i in range(cfg.TIME_SEQUENCE):
            probs_per_frame[start + i].append(prob_blink)

    probs = aggregate_probs(probs_per_frame, cfg.PROB_AGG)
    return probs


# ----------------------------
# Pipeline for one video
# ----------------------------

def process_video(video_path: str, out_root: str, model: nn.Module, device: torch.device, cfg: Cfg, fps: int):
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    work_dir = ensure_dir(os.path.join(out_root, video_name))
    frames_dir = ensure_dir(os.path.join(work_dir, 'frames'))

    # 1) Frames (cropped faces)
    mapping = crop_faces_from_video(video_path, frames_dir, fps=fps)

    # 2) Optical flow + landmarks
    compute_optical_flow_for_folder(frames_dir)
    extract_eye_landmarks_for_folder(frames_dir)

    # 3) Inference → probs per frame
    probs = infer_on_frames_folder(frames_dir, model, device, cfg)

    # 4) Convert probs → binary with recall-friendly hysteresis + cleanup
    preds_h = apply_hysteresis(probs, cfg.HYST_HIGH, cfg.HYST_LOW)
    preds_h = merge_and_prune(preds_h, cfg.MIN_BLINK_LEN, cfg.MERGE_GAP)

    # 5) Export ELAN + CSV/XLSX
    ann_dir = ensure_dir(os.path.join(work_dir, 'annotations'))
    fp_eaf = save_blinks_to_elan(
        video_path, preds_h, mapping, cfg.FRAME_RATE, ann_dir, cfg.MIN_BLINK_LEN
    )
    fp_csv, fp_xlsx = save_predictions_csv(video_name, probs, preds_h, mapping, ann_dir)

    segments = blink_segments(preds_h, cfg.MIN_BLINK_LEN)
    summary = {
        'video': os.path.basename(video_path),
        'source_fps': float(mapping.iloc[0]['source_fps']),
        'model_reference_fps': float(cfg.FRAME_RATE),
        'usable_model_samples': int(len(mapping)),
        'blink_frames': int(preds_h.sum()),
        'blink_events': int(len(segments)),
        'checkpoint_sha256': cfg.CHECKPOINT_SHA256,
        'verification_required': True,
        'tool_scope': 'semi-automatic blink pre-annotation',
    }
    with open(os.path.join(ann_dir, 'processing_summary.json'), 'w', encoding='utf-8') as handle:
        json.dump(summary, handle, indent=2)

    return work_dir, fp_eaf, fp_csv, fp_xlsx, probs, preds_h


# ----------------------------
# Loading model
# ----------------------------

def file_sha256(path: str):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest().upper()


def load_model(model_path: str, device: torch.device, cfg: Cfg):
    observed_hash = file_sha256(model_path)
    if observed_hash != cfg.CHECKPOINT_SHA256:
        raise RuntimeError(
            "The supplied checkpoint is not the released BlinkFusionNet model. "
            f"Expected SHA-256 {cfg.CHECKPOINT_SHA256}, observed {observed_hash}."
        )
    model = DualStreamBlinkModel(cfg).to(device)
    state = torch.load(model_path, map_location=device)
    # sanity check LSTM input size
    expected_in = 256*2 + cfg.GEOM_FEAT_DIM
    # try to find a known key to infer ckpt input size
    some_key = None
    for k in state.keys():
        if 'temporal.lstm.weight_ih_l0' in k:
            some_key = k
            break
    if some_key is not None:
        ckpt_in = state[some_key].shape[1]
        if expected_in != ckpt_in:
            raise ValueError(f"GEOM_FEAT_DIM mismatch. Set to 4 so that 256+256+GEOM={ckpt_in}.")
    model.load_state_dict(state, strict=True)
    return model


# ----------------------------
# Recall-friendly operating point
# ----------------------------

def enable_recall_friendly():
    # 1) Aggregate windowed probabilities with MAX to pick strongest evidence
    Cfg.PROB_AGG = 'max'
    # 2) Lower hysteresis thresholds (wider band toward positives)
    Cfg.HYST_HIGH = 0.30
    Cfg.HYST_LOW  = 0.18
    # 3) Allow slightly shorter detections and more generous merging
    Cfg.MIN_BLINK_LEN = 3
    Cfg.MERGE_GAP     = 3


# ----------------------------
# Public command-line entry point
# ----------------------------

def run_fps_self_checks():
    identity, times_25, _ = build_model_timeline(101, 25.0, Cfg.FRAME_RATE)
    if not np.array_equal(identity, np.arange(101)):
        raise AssertionError("25-FPS identity mapping failed.")
    if not np.allclose(times_25[:4], [0.0, 40.0, 80.0, 120.0]):
        raise AssertionError("25-FPS model timeline failed.")

    for source_fps in (29.97, 30.0, 50.0, 60.0):
        selected, model_times, source_times = build_model_timeline(601, source_fps, Cfg.FRAME_RATE)
        maximum_error = float(np.max(np.abs(model_times - source_times)))
        if maximum_error > (500.0 / source_fps + 1e-6):
            raise AssertionError(f"Nearest-timestamp mapping failed at {source_fps} FPS.")
        if np.any(np.diff(selected) < 0):
            raise AssertionError(f"Non-monotonic mapping at {source_fps} FPS.")

    selected_24, _, _ = build_model_timeline(241, 24.0, Cfg.FRAME_RATE)
    if not np.any(np.diff(selected_24) == 0):
        raise AssertionError("Low-FPS nearest-frame repetition check failed.")
    print("FPS self-checks PASS: 24, 25, 29.97, 30, 50 and 60 FPS.")


def parse_args():
    default_model = Path(__file__).resolve().with_name(Cfg.CHECKPOINT_FILENAME)
    parser = argparse.ArgumentParser(
        description=(
            "Generate semi-automatic blink pre-annotations for ELAN while "
            "normalizing source videos to the model's fixed 25-FPS timeline."
        )
    )
    parser.add_argument('--input', help='Input video file or directory of videos.')
    parser.add_argument('--output', help='Directory in which result folders are created.')
    parser.add_argument('--model', default=str(default_model),
                        help=f'Released checkpoint (default: {default_model}).')
    parser.add_argument('--overwrite', action='store_true',
                        help='Replace an existing result folder for the same video.')
    parser.add_argument('--self-test', action='store_true',
                        help='Run frame-rate mapping checks and exit.')
    return parser.parse_args()


def collect_videos(input_path: Path):
    if input_path.is_file():
        if input_path.suffix.lower() not in Cfg.VIDEO_EXTENSIONS:
            raise RuntimeError(f"Unsupported video extension: {input_path.suffix}")
        return [input_path]
    if not input_path.is_dir():
        raise FileNotFoundError(f"Input path does not exist: {input_path}")
    return sorted(
        [p for p in input_path.iterdir()
         if p.is_file() and p.suffix.lower() in Cfg.VIDEO_EXTENSIONS],
        key=lambda p: p.name.lower()
    )


def prepare_video_output(output_root: Path, video_path: Path, overwrite: bool):
    output_root.mkdir(parents=True, exist_ok=True)
    video_output = output_root / video_path.stem
    if video_output.exists():
        if not overwrite:
            raise FileExistsError(
                f"Output already exists: {video_output}. Use --overwrite to replace it."
            )
        resolved_root = output_root.resolve()
        resolved_video_output = video_output.resolve()
        if resolved_video_output.parent != resolved_root:
            raise RuntimeError(f"Refusing to remove unexpected output path: {resolved_video_output}")
        shutil.rmtree(resolved_video_output)


def main():
    args = parse_args()
    run_fps_self_checks()
    if args.self_test:
        return
    if not args.input or not args.output:
        raise SystemExit("--input and --output are required unless --self-test is used.")

    videos_path = Path(args.input).expanduser().resolve()
    out_path = Path(args.output).expanduser().resolve()
    model_path = Path(args.model).expanduser().resolve()
    if not model_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {model_path}")

    # Preserve the released operating point and checkpoint temporal contract.
    Cfg.FRAME_RATE = 25.0
    enable_recall_friendly()

    videos = collect_videos(videos_path)
    if not videos:
        raise RuntimeError(f"No supported videos found in: {videos_path}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.backends.cudnn.benchmark = True
    print(f"Using device: {device}")
    print(f"Model reference FPS: {Cfg.FRAME_RATE:g}")
    model = load_model(str(model_path), device, Cfg)

    for video in videos:
        prepare_video_output(out_path, video, args.overwrite)
        print("\n" + "-"*70)
        print(f"Processing: {video}")
        work_dir, fp_eaf, fp_csv, fp_xlsx, probs, preds_bin = process_video(
            str(video), str(out_path), model, device, Cfg, fps=Cfg.FRAME_RATE
        )
        print(f"\nDone. Semi-automatic outputs for {video.name}:")
        print(f"   Work dir:  {work_dir}")
        print(f"   ELAN:      {fp_eaf}")
        print(f"   CSV:       {fp_csv}")
        print(f"   Excel:     {fp_xlsx}")
        print("   REQUIRED: open the ELAN file with the source video and verify every blink.")


if __name__ == '__main__':
    main()
