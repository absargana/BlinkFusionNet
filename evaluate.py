import os
import re
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image

import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from sklearn.metrics import precision_recall_fscore_support

import albumentations as A
from albumentations.pytorch import ToTensorV2

# =========================
# CONFIGURATION
# =========================
class Config:
    # Root of Stage1-style test data
    DATA_ROOT = os.environ.get("BLINK_TEST_DATA", os.path.join("data", "test"))

    # Use any of these groups: [6], [10], or [6, 10]
    ACCEPT_GROUPS = [10]

    # Model checkpoint / meta
    MODEL_PATH = os.environ.get(
        "BLINK_MODEL_PATH", os.path.join("checkpoints", "blinkfusionnet_t10_seed123.pth")
    )
    META_PATH = MODEL_PATH.replace(".pth", ".meta.json")

    # Output dir for CSV
    OUTPUT_DIR = os.environ.get("BLINK_OUTPUT_DIR", os.path.join("outputs", "evaluation"))

    # Window settings (must match training)
    TIME_SEQUENCE = 10
    STRIDE = 1            # not used directly now for metrics
    WINDOW_STRIDE = None  # if None, we use TIME_SEQUENCE for window metrics

    IMAGE_SIZE = 128
    GEOM_FEAT_DIM = 4
    NUM_CLASSES = 2
    DROPOUT = 0.5

    # frame-wise aggregation over windows: "mean" or "max"
   # FRAME_AGGREGATION_METHOD = "mean"
    PROB_AGG = "mean"

    LABEL_COLUMN = "label"         # used by load_ground_truth_csv
    FRAME_COLUMN = "frame_index"   # optional

    # Test threshold:
    # - Set to a recall-friendly value (e.g. 0.80 based on sweep)
    # - If USE_META_THRESHOLD=True, this may be overridden by best_thr from META_PATH
    THRESHOLD = 0.5
    USE_META_THRESHOLD = False

    GROUP_LABEL_MODE = "majority"  # "majority" or "any"


cfg = Config()

# =========================
# SMALL HELPERS
# =========================

def tryint(s):
    try:
        return int(s)
    except Exception:
        return s


def alphanum_key(s):
    return [tryint(c) for c in re.split(r"([0-9]+)", s)]


def natural_sort(l):
    return sorted(l, key=alphanum_key)


def extract_group_id(name: str):
    m = re.search(r"group_(\d+)", name)
    return int(m.group(1)) if m else None


# ---- label helpers (compatible with train_v2) ----
_BIN_MAP = {
    "1": 1, "0": 0,
    "blink": 1, "no blink": 0,
    "blinks": 1, "no-blink": 0, "no_blink": 0,
    "positive": 1, "negative": 0,
}


def coerce_series_to_binary(series: pd.Series):
    if pd.api.types.is_numeric_dtype(series):
        vals = series.fillna(0).round().astype(int)
        if set(vals.unique().tolist()).issubset({0, 1}):
            return vals.astype(int)

    mapped = []
    unknown = 0
    for v in series.astype(str).fillna("").tolist():
        s = v.strip().lower()
        if s in _BIN_MAP:
            mapped.append(_BIN_MAP[s])
        else:
            try:
                fv = float(s)
                mapped.append(int(round(fv)))
            except Exception:
                mapped.append(None)
                unknown += 1

    mapped = pd.Series(mapped, index=series.index, dtype="float").ffill().fillna(0).astype(int)
    if unknown > 0 and unknown / max(len(series), 1) > 0.5:
        raise ValueError("More than half labels were non-mappable strings.")
    return mapped


def detect_label_column(df: pd.DataFrame, preferred: str):
    if preferred in df.columns:
        return preferred
    for cand in df.columns:
        cname = str(cand).lower()
        if "blink" in cname or cname in ("label", "labels", "y", "target"):
            return cand
    for cand in df.columns:
        col = df[cand]
        if pd.api.types.is_numeric_dtype(col):
            uniq = set(pd.Series(col).dropna().astype(int).unique().tolist())
            if uniq.issubset({0, 1}):
                return cand
    return None


def load_ground_truth_csv(csv_path, label_col, frame_col=None):
    df = pd.read_csv(csv_path)
    use_col = detect_label_column(df, label_col)
    if use_col is None:
        raise ValueError(f"No suitable label column found in {csv_path}. Columns: {list(df.columns)}")
    labels_raw = df[use_col]
    labels = coerce_series_to_binary(labels_raw)
    frame_idx = None
    if frame_col and frame_col in df.columns:
        frame_idx = df[frame_col].to_numpy().astype(int)
    return labels.to_numpy(), frame_idx


def make_group_label(window_labels, mode: str = "any"):
    if mode == "any":
        return int(np.any(window_labels == 1))
    elif mode == "majority":
        ones = int(np.sum(window_labels == 1))
        zeros = len(window_labels) - ones
        return 1 if ones >= zeros else 0
    else:
        raise ValueError(f"Unknown GROUP_LABEL_MODE: {mode}")


def aggregate_probs(probs_per_frame, method: str):
    if method == "max":
        return np.array([np.max(p) if p else 0.0 for p in probs_per_frame], dtype=np.float32)
    # default: "mean"
    return np.array([np.mean(p) if p else 0.0 for p in probs_per_frame], dtype=np.float32)


def compute_classwise_metrics(y_true, y_pred):
    """
    Returns:
      (p0, r0, f10), (p1, r1, f11), macro_f1
      where class 0 = non-blink, class 1 = blink
    """
    p, r, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=[0, 1], zero_division=0
    )
    macro_f1 = float(f1.mean())
    return (float(p[0]), float(r[0]), float(f1[0])), (float(p[1]), float(r[1]), float(f1[1])), macro_f1


def calculate_ear(lm: np.ndarray) -> float:
    """
    EAR for one eye (6 points: [p0..p5]).
    lm: [6, 2]
    """
    A = np.linalg.norm(lm[1] - lm[5])
    B = np.linalg.norm(lm[2] - lm[4])
    C = np.linalg.norm(lm[0] - lm[3])
    return (A + B) / (2.0 * C) if C > 1e-6 else 3.0


# =========================
# AUGMENTATION FOR INFERENCE
# =========================
class Augmentation:
    def __init__(self, size: int):
        self.tf = A.Compose([
            A.Resize(size, size),
            A.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])

    def __call__(self, img_np: np.ndarray):
        return self.tf(image=img_np)["image"]


# =========================
# MODEL (matches training)
# =========================
class EfficientNetStream(nn.Module):
    def __init__(self, out_dim: int):
        super().__init__()
        base = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
        self.features = nn.Sequential(*list(base.children())[:-1])  # all but classifier
        in_feat = base.classifier[1].in_features
        self.fc = nn.Linear(in_feat, out_dim)

    def forward(self, x):  # x: [B, T, C, H, W]
        B, T, C, H, W = x.size()
        x = x.view(B * T, C, H, W)
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

    def forward(self, x):  # x: [B, T, D]
        gate = self.gating_network(x)
        return x * gate


class DualStreamBlinkModel(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.spatial_stream = EfficientNetStream(256)
        self.temporal_stream = EfficientNetStream(256)
        fusion_in = 256 * 2 + cfg.GEOM_FEAT_DIM
        self.gated_fusion = GatedFusion(fusion_in)
        self.temporal = TemporalBiLSTM(fusion_in, 128, cfg.NUM_CLASSES, cfg.DROPOUT)

    def forward(self, rgb, flow, geom):
        rgb_f = self.spatial_stream(rgb)
        flow_f = self.temporal_stream(flow)
        fused = torch.cat([rgb_f, flow_f, geom], dim=-1)
        fused = self.gated_fusion(fused)
        return self.temporal(fused)


# =========================
# SEQUENCE INDEXING (Stage1_new_Data)
# =========================
def build_sequence_metadata_stage1(seq_dir: str, cfg: Config):
    """
    Stage1_new_Data sequence structure:
      - frame_*.jpg
      - optical_flow/frame_*.jpg
      - mediapipe_landmarks.txt (N x 24 -> N x 12 x 2)
      - ground_truth.csv
    """
    frames_dir = seq_dir
    flow_dir = os.path.join(seq_dir, "optical_flow")

    if not os.path.isdir(frames_dir):
        raise FileNotFoundError(f"Missing frames in {seq_dir}")
    if not os.path.isdir(flow_dir):
        raise FileNotFoundError(f"Missing optical_flow in {seq_dir}")

    img_files = [f for f in os.listdir(frames_dir)
                 if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))
                 and f.lower() != "optical_flow"]
    image_paths = [os.path.join(frames_dir, f) for f in natural_sort(img_files)]

    flow_files = [f for f in os.listdir(flow_dir)
                  if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))]
    flow_paths = [os.path.join(flow_dir, f) for f in natural_sort(flow_files)]

    lm_path = os.path.join(seq_dir, "mediapipe_landmarks.txt")
    gt_csv = os.path.join(seq_dir, "ground_truth.csv")

    if not os.path.isfile(lm_path):
        raise FileNotFoundError(f"Missing mediapipe_landmarks.txt in {seq_dir}")
    if not os.path.isfile(gt_csv):
        raise FileNotFoundError(f"Missing ground_truth.csv in {seq_dir}")

    lm_flat = np.loadtxt(lm_path, dtype=np.float32)
    if lm_flat.ndim == 1:
        lm_flat = lm_flat[None, :]
    landmarks = lm_flat.reshape(-1, 12, 2)

    labels, frame_idx = load_ground_truth_csv(gt_csv, cfg.LABEL_COLUMN, frame_col=cfg.FRAME_COLUMN)

    n = min(len(image_paths), len(flow_paths), landmarks.shape[0], len(labels))
    if n == 0:
        raise RuntimeError(f"No aligned samples in {seq_dir}")

    return {
        "image_paths": image_paths[:n],
        "flow_paths": flow_paths[:n],
        "landmarks": landmarks[:n],
        "labels": labels[:n],
        "n_frames": n,
    }


def discover_sequences(cfg: Config):
    seq_dirs = []
    for name in os.listdir(cfg.DATA_ROOT):
        seq_dir = os.path.join(cfg.DATA_ROOT, name)
        if not os.path.isdir(seq_dir):
            continue
        gid = extract_group_id(name)
        if gid is None:
            continue
        if cfg.ACCEPT_GROUPS and gid not in cfg.ACCEPT_GROUPS:
            continue
        seq_dirs.append(seq_dir)
    return natural_sort(seq_dirs)


def index_sequences(cfg: Config):
    seq_dirs = discover_sequences(cfg)
    seq_infos = []
    metadata_cache = {}
    for sd in seq_dirs:
        cache = build_sequence_metadata_stage1(sd, cfg)
        name = os.path.basename(sd)
        gid = extract_group_id(name)
        cache["group_id"] = gid
        metadata_cache[sd] = cache
        seq_infos.append({
            "path": sd,
            "name": name,
            "n_frames": cache["n_frames"],
            "labels": cache["labels"],
            "group_id": gid,
        })
    return seq_infos, metadata_cache


# =========================
# PRELOAD SEQUENCE TENSORS
# =========================
def preload_sequence_tensors(seq_name, cache, cfg: Config):
    n_frames = cache["n_frames"]
    landmarks = cache["landmarks"]
    labels = cache["labels"]

    tf = Augmentation(cfg.IMAGE_SIZE)
    rgb_tensors, flow_tensors = [], []

    print(f"[SEQ] Preloading tensors for {seq_name} (n_frames={n_frames})")
    for idx in tqdm(range(n_frames), desc=f"Preload {seq_name}", leave=False):
        rgb_np = np.array(Image.open(cache["image_paths"][idx]).convert("RGB"))
        flow_np = np.array(Image.open(cache["flow_paths"][idx]).convert("RGB"))
        rgb_tensors.append(tf(rgb_np))
        flow_tensors.append(tf(flow_np))

    rgb_tensors = torch.stack(rgb_tensors)   # [N, C, H, W] on CPU
    flow_tensors = torch.stack(flow_tensors) # [N, C, H, W] on CPU

    return rgb_tensors, flow_tensors, landmarks, labels


# =========================
# RUN WINDOWS FOR GIVEN STRIDE
# =========================
def eval_sequence_with_stride(seq_info, rgb_tensors, flow_tensors, landmarks, labels,
                              model, cfg, device, stride: int):
    """
    Runs sliding windows with a given stride over a preloaded sequence.
    Returns:
      window_records: list of dicts (sequence, group_id, window_idx, start_frame, end_frame, prob_blink)
      frame_probs:    np.array [N_frames] aggregated per-frame probs (mean / max)
      frame_labels:   np.array [N_frames] ground-truth labels
    """
    seq_name = seq_info["name"]
    n_frames = seq_info["n_frames"]

    window_records = []
    probs_per_frame = [[] for _ in range(n_frames)]

    win_len = cfg.TIME_SEQUENCE
    stride = max(int(stride), 1)

    model.eval()
    with torch.no_grad():
        win_idx = 0
        for start in tqdm(range(0, n_frames - win_len + 1, stride),
                          desc=f"Windows {seq_name} (stride={stride})", leave=False):
            end = start + win_len - 1
            idxs = slice(start, start + win_len)

            rgb_window = rgb_tensors[idxs].unsqueeze(0).to(device)   # [1, T, C, H, W] after view in model
            flow_window = flow_tensors[idxs].unsqueeze(0).to(device)

            # ----- geom sequence (EAR etc.) -----
            geom_seq = []
            prev_ear = -1.0
            for fidx in range(start, start + win_len):
                lm = landmarks[fidx]
                l_ear = calculate_ear(lm[:6])
                r_ear = calculate_ear(lm[6:12])
                avg_ear = (l_ear + r_ear) / 2.0
                delta_ear = (avg_ear - prev_ear) if prev_ear >= 0 else 0.0
                geom = np.array([avg_ear, delta_ear, 0.0, 0.0], dtype=np.float32)[:cfg.GEOM_FEAT_DIM]
                geom_seq.append(geom)
                prev_ear = avg_ear
            geom_tensor = torch.tensor(np.array(geom_seq, dtype=np.float32)).unsqueeze(0).to(device)
            # ------------------------------------

            logits = model(rgb_window, flow_window, geom_tensor)
            prob_blink = torch.softmax(logits, dim=-1)[0, 1].item()

            # smear prob to all frames in window
            for i in range(win_len):
                probs_per_frame[start + i].append(prob_blink)

            window_records.append({
                "sequence": seq_name,
                "group_id": seq_info["group_id"],
                "window_idx": win_idx,
                "start_frame": start + 1,
                "end_frame": end + 1,
                "prob_blink": float(prob_blink),
            })
            win_idx += 1

    frame_probs = aggregate_probs(probs_per_frame, cfg.PROB_AGG)
    frame_labels = labels[:len(frame_probs)].astype(int)
    return window_records, frame_probs, frame_labels


# =========================
# MAIN
# =========================
def main():
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    # threshold from meta (optional)
    if cfg.USE_META_THRESHOLD and os.path.isfile(cfg.META_PATH):
        try:
            with open(cfg.META_PATH, "r") as f:
                meta = json.load(f)
            if "best_thr" in meta:
                cfg.THRESHOLD = float(meta["best_thr"])
                print(f"[META] Loaded best_thr={cfg.THRESHOLD:.3f} from {cfg.META_PATH}")
        except Exception as e:
            print(f"[META] Could not load meta file: {e}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    # load model
    model = DualStreamBlinkModel(cfg).to(device)
    state = torch.load(cfg.MODEL_PATH, map_location=device)
    if isinstance(state, dict) and "state_dict" in state:
        model.load_state_dict(state["state_dict"])
    else:
        model.load_state_dict(state)
    print(f"[INFO] Loaded model from {cfg.MODEL_PATH}")

    # index sequences
    seq_infos, metadata_cache = index_sequences(cfg)
    print(f"[INFO] Found {len(seq_infos)} sequences in DATA_ROOT filtered by groups {cfg.ACCEPT_GROUPS}")

    thr = cfg.THRESHOLD
    print(f"[INFO] Using threshold={thr:.3f} for metrics")

    # Global accumulators
    all_window_records = []
    all_frame_records = []
    summary_window_rows = []
    summary_frame_rows = []

    all_window_gt = []
    all_window_pred = []
    all_frame_gt = []
    all_frame_pred = []

    # per-sequence evaluation
    for seq_info in seq_infos:
        seq_name = seq_info["name"]
        print(f"\n========== Evaluating sequence: {seq_name} ==========")
        cache = metadata_cache[seq_info["path"]]

        # Preload tensors once
        rgb_tensors, flow_tensors, landmarks, labels = preload_sequence_tensors(
            seq_name, cache, cfg
        )

        # --------------------------------------------------
        # PASS 1: WINDOW/GROUP METRICS (NON-OVERLAP)
        # --------------------------------------------------
        window_stride = cfg.WINDOW_STRIDE or cfg.TIME_SEQUENCE

        window_records_nonoverlap, frame_probs_nonoverlap, frame_labels_nonoverlap = \
            eval_sequence_with_stride(seq_info, rgb_tensors, flow_tensors, landmarks, labels,
                                      model, cfg, device, stride=window_stride)

        # compute window-level ground-truth labels (grouped by majority / any)
        prob_w = np.array([r["prob_blink"] for r in window_records_nonoverlap], dtype=np.float32)
        y_true_w = []
        for r in window_records_nonoverlap:
            s = r["start_frame"] - 1
            e = r["end_frame"] - 1
            win_labels = frame_labels_nonoverlap[s:e + 1]
            y_true_w.append(make_group_label(win_labels, cfg.GROUP_LABEL_MODE))
        y_true_w = np.array(y_true_w, dtype=int)
        y_pred_w = (prob_w >= thr).astype(int)

        (p0_w, r0_w, f0_w), (p1_w, r1_w, f1_w), macro_f1_w = compute_classwise_metrics(y_true_w, y_pred_w)
        print(f"[WINDOW] stride={window_stride}, #windows={len(window_records_nonoverlap)}")
        print(f"  NonBlink P/R/F1 = ({p0_w:.3f}/{r0_w:.3f}/{f0_w:.3f})")
        print(f"  Blink    P/R/F1 = ({p1_w:.3f}/{r1_w:.3f}/{f1_w:.3f}) | MacroF1={macro_f1_w:.4f}")

        # accumulate overall window metrics
        all_window_gt.extend(y_true_w.tolist())
        all_window_pred.extend(y_pred_w.tolist())

        # per-sequence window summary row
        summary_window_rows.append({
            "sequence": seq_name,
            "group_id": seq_info["group_id"],
            "num_windows": len(window_records_nonoverlap),
            "stride": window_stride,
            "threshold": thr,
            "nonblink_precision": p0_w,
            "nonblink_recall": r0_w,
            "nonblink_f1": f0_w,
            "blink_precision": p1_w,
            "blink_recall": r1_w,
            "blink_f1": f1_w,
            "macro_f1": macro_f1_w,
        })

        # --------------------------------------------------
        # PASS 2: FRAME METRICS (DENSE, STRIDE = 1)
        # --------------------------------------------------
        frame_stride = 1
        window_records_dense, frame_probs_dense, frame_labels_dense = \
            eval_sequence_with_stride(seq_info, rgb_tensors, flow_tensors, landmarks, labels,
                                      model, cfg, device, stride=frame_stride)

        # collect window predictions for CSV (dense windows)
        all_window_records.extend(window_records_dense)

        # frame-level metrics
        y_true_f = frame_labels_dense.astype(int)
        prob_f = frame_probs_dense.astype(np.float32)
        y_pred_f = (prob_f >= thr).astype(int)

        (p0_f, r0_f, f0_f), (p1_f, r1_f, f1_f), macro_f1_f = compute_classwise_metrics(y_true_f, y_pred_f)
        print(f"[FRAME ] stride={frame_stride}, #frames={len(frame_labels_dense)} (agg={cfg.PROB_AGG})")
        print(f"  NonBlink P/R/F1 = ({p0_f:.3f}/{r0_f:.3f}/{f0_f:.3f})")
        print(f"  Blink    P/R/F1 = ({p1_f:.3f}/{r1_f:.3f}/{f1_f:.3f}) | MacroF1={macro_f1_f:.4f}")

        # accumulate overall frame metrics
        all_frame_gt.extend(y_true_f.tolist())
        all_frame_pred.extend(y_pred_f.tolist())

        # per-sequence frame summary
        summary_frame_rows.append({
            "sequence": seq_name,
            "group_id": seq_info["group_id"],
            "num_frames": len(frame_labels_dense),
            "threshold": thr,
            "nonblink_precision": p0_f,
            "nonblink_recall": r0_f,
            "nonblink_f1": f0_f,
            "blink_precision": p1_f,
            "blink_recall": r1_f,
            "blink_f1": f1_f,
            "macro_f1": macro_f1_f,
        })

        # collect frame predictions for CSV
        for idx, (p, y) in enumerate(zip(prob_f, y_true_f)):
            all_frame_records.append({
                "sequence": seq_name,
                "group_id": seq_info["group_id"],
                "frame_index": idx + 1,
                "prob_blink": float(p),
                "label": int(y),
            })

    # ================= OVERALL METRICS =================
    print("\n================ OVERALL METRICS (all sequences) ================")

    if all_window_gt:
        y_true_w_all = np.array(all_window_gt, dtype=int)
        y_pred_w_all = np.array(all_window_pred, dtype=int)
        (p0_w, r0_w, f0_w), (p1_w, r1_w, f1_w), macro_f1_w = compute_classwise_metrics(
            y_true_w_all, y_pred_w_all
        )
        print(f"[WINDOW] ALL: #windows={len(y_true_w_all)}")
        print(f"  NonBlink P/R/F1 = ({p0_w:.3f}/{r0_w:.3f}/{f0_w:.3f})")
        print(f"  Blink    P/R/F1 = ({p1_w:.3f}/{r1_w:.3f}/{f1_w:.3f}) | MacroF1={macro_f1_w:.4f}")

        # overall row for window summary CSV
        summary_window_rows.append({
            "sequence": "ALL_VIDEOS",
            "group_id": -1,
            "num_windows": len(y_true_w_all),
            "stride": cfg.WINDOW_STRIDE or cfg.TIME_SEQUENCE,
            "threshold": thr,
            "nonblink_precision": p0_w,
            "nonblink_recall": r0_w,
            "nonblink_f1": f0_w,
            "blink_precision": p1_w,
            "blink_recall": r1_w,
            "blink_f1": f1_w,
            "macro_f1": macro_f1_w,
        })

    if all_frame_gt:
        y_true_f_all = np.array(all_frame_gt, dtype=int)
        y_pred_f_all = np.array(all_frame_pred, dtype=int)
        (p0_f, r0_f, f0_f), (p1_f, r1_f, f1_f), macro_f1_f = compute_classwise_metrics(
            y_true_f_all, y_pred_f_all
        )
        print(f"[FRAME ] ALL: #frames={len(y_true_f_all)}")
        print(f"  NonBlink P/R/F1 = ({p0_f:.3f}/{r0_f:.3f}/{f0_f:.3f})")
        print(f"  Blink    P/R/F1 = ({p1_f:.3f}/{r1_f:.3f}/{f1_f:.3f}) | MacroF1={macro_f1_f:.4f}")

        # overall row for frame summary CSV
        summary_frame_rows.append({
            "sequence": "ALL_VIDEOS",
            "group_id": -1,
            "num_frames": len(y_true_f_all),
            "threshold": thr,
            "nonblink_precision": p0_f,
            "nonblink_recall": r0_f,
            "nonblink_f1": f0_f,
            "blink_precision": p1_f,
            "blink_recall": r1_f,
            "blink_f1": f1_f,
            "macro_f1": macro_f1_f,
        })

    # save CSVs
    print(f"\n[INFO] Saving CSVs under: {cfg.OUTPUT_DIR}")
    if all_window_records:
        pd.DataFrame(all_window_records).to_csv(
            os.path.join(cfg.OUTPUT_DIR, "window_predictions_all.csv"), index=False
        )
    if all_frame_records:
        pd.DataFrame(all_frame_records).to_csv(
            os.path.join(cfg.OUTPUT_DIR, "frame_predictions_all.csv"), index=False
        )
    if summary_window_rows:
        pd.DataFrame(summary_window_rows).to_csv(
            os.path.join(cfg.OUTPUT_DIR, "metrics_window_summary.csv"), index=False
        )
    if summary_frame_rows:
        pd.DataFrame(summary_frame_rows).to_csv(
            os.path.join(cfg.OUTPUT_DIR, "metrics_frame_summary.csv"), index=False
        )
    print("[DONE] Testing complete.")


if __name__ == "__main__":
    main()
