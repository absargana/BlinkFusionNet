import os
import re
import random
from collections import Counter
from typing import List, Dict, Any

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler

from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

import albumentations as A
from albumentations.pytorch import ToTensorV2

from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from sklearn.metrics import f1_score, precision_recall_fscore_support


# ============================================================
# CONFIG
# ============================================================
class Config:
    # Root where Stage1_new_Data sequences live
    DATA_ROOT = os.environ.get("BLINK_TRAIN_DATA", os.path.join("data", "train"))

    # NEW: which groups to use (folder names end with _group_6, _group_10, etc.)
    # Examples:
    #   [6]      → only use ..._group_6 sequences
    #   [10]     → only use ..._group_10 sequences
    #   [6, 10]  → use both types
    ACCEPT_GROUPS = [10]          # <--- set this as you like

    TIME_SEQUENCE = 10            # must be consistent with your chosen group (6 or 10)
    IMAGE_SIZE = 128
    VAL_SPLIT = 0.15
    EPOCHS = 60
    PATIENCE = 10
    BATCH_SIZE = 16
    LR = 1e-4
    WEIGHT_DECAY = 1e-4
    DROPOUT = 0.5
    NUM_WORKERS = 8
    GEOM_FEAT_DIM = 4
    NUM_CLASSES = 2
    SEED = 123



cfg = Config()


# ============================================================
# SEEDING
# ============================================================
def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


seed_everything(cfg.SEED)
torch.backends.cudnn.benchmark = True  # speed for fixed-size inputs

def normalize_blink_labels(series: pd.Series) -> pd.Series:
    """
    Convert various blink label formats to integer {0,1}.

    Accepts e.g.:
      - numeric: 0, 1, 0.0, 1.0
      - strings: 'blink', 'no blink', 'Blink', 'NonBlink', etc.

    Returns a Series[int] with the same index.
    """
    # First try the simple case: already numeric 0/1
    try:
        s = series.astype(float)
        # If all values are 0/1, this is fine
        if set(np.unique(s)) <= {0.0, 1.0}:
            return s.astype(int)
    except Exception:
        pass

    mapped = []
    for v in series:
        # Handle strings
        if isinstance(v, str):
            s = v.strip().lower()
            if s in {"blink", "b", "1", "yes", "y", "true"}:
                mapped.append(1)
            elif s in {"no blink", "nonblink", "non blink", "0", "nb", "none", "false", "no", "n"}:
                mapped.append(0)
            else:
                raise ValueError(
                    f"normalize_blink_labels: unknown string label value '{v}'. "
                    f"Please extend the mapping."
                )
        # Handle numbers
        else:
            try:
                fv = float(v)
            except Exception:
                raise ValueError(
                    f"normalize_blink_labels: unsupported label type/value {repr(v)}"
                )
            if fv == 1.0:
                mapped.append(1)
            elif fv == 0.0:
                mapped.append(0)
            else:
                raise ValueError(
                    f"normalize_blink_labels: numeric label must be 0/1, got {fv}"
                )

    return pd.Series(mapped, index=series.index, dtype=int)

# ============================================================
# UTILITIES
# ============================================================
def tryint(s):
    try:
        return int(s)
    except Exception:
        return s


def alphanum_key(s):
    return [tryint(c) for c in re.split(r"([0-9]+)", s)]


def natural_sort(l):
    return sorted(l, key=alphanum_key)


def parse_frame_index_from_name(filename: str) -> int:
    """
    Extract numeric index from a filename, e.g.:
        frame_2.jpg   -> 2
        frame_0023.png -> 23
    """
    m = re.search(r"(\d+)", filename)
    if not m:
        raise ValueError(f"Could not parse frame index from {filename}")
    return int(m.group(1))


def calculate_ear(lm: np.ndarray) -> float:
    """
    EAR for one eye (6 points: [p0..p5]).
    lm: [6, 2]
    """
    A = np.linalg.norm(lm[1] - lm[5])
    B = np.linalg.norm(lm[2] - lm[4])
    C = np.linalg.norm(lm[0] - lm[3])
    return (A + B) / (2.0 * C) if C > 1e-6 else 3.0


# ============================================================
# AUGMENTATION
# ============================================================
class Augmentation:
    """
    Albumentations transforms (train / val).
    """

    def __init__(self, size: int, is_train: bool = True):
        if is_train:
            self.transform = A.Compose([
                A.HorizontalFlip(p=0.5),
                A.ShiftScaleRotate(shift_limit=0.05,
                                   scale_limit=0.05,
                                   rotate_limit=15,
                                   p=0.5),
                A.RandomBrightnessContrast(p=0.4),
                A.MotionBlur(p=0.3),
                A.ISONoise(p=0.3),
                A.CoarseDropout(max_holes=8,
                                max_height=16,
                                max_width=16,
                                p=0.5),
                A.Resize(size, size),
                A.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ])
        else:
            self.transform = A.Compose([
                A.Resize(size, size),
                A.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ])

    def __call__(self, image: np.ndarray) -> torch.Tensor:
        return self.transform(image=image)["image"]


# ============================================================
# DATA PREPARATION FROM Stage1_new_Data
# ============================================================
def detect_label_column(gt_df: pd.DataFrame) -> str:
    """
    Try to find the blink label column in ground_truth.csv.
    Priority for exact names, then any column containing 'label'/'blink'.
    """
    cols_lower = {c.lower(): c for c in gt_df.columns}

    for k in ["blink_label", "blink", "label", "class"]:
        if k in cols_lower:
            return cols_lower[k]

    candidates = [
        c for c in gt_df.columns
        if ("label" in c.lower()) or ("blink" in c.lower())
    ]
    if len(candidates) == 0:
        raise ValueError(f"Could not find a blink/label column in ground_truth columns: {list(gt_df.columns)}")
    return candidates[0]


def build_sequence_metadata(cfg: Config):
    """
    Build:
      - valid_seq_infos: list of dicts with { 'path', 'group_labels' }
      - metadata_cache:  seq_path -> { 'image_paths', 'flow_paths', 'landmarks' }

    Uses:
      - Stage1_new_Data/<seq_name>/frame_*.jpg
      - Stage1_new_Data/<seq_name>/optical_flow/frame_*.jpg
      - Stage1_new_Data/<seq_name>/mediapipe_landmarks.txt
      - Stage1_new_Data/<seq_name>/ground_truth.csv (with 'frame_index' + label col)
    """
    valid_seq_infos: List[Dict[str, Any]] = []
    metadata_cache: Dict[str, Dict[str, Any]] = {}

    all_seq_names = [
        d for d in os.listdir(cfg.DATA_ROOT)
        if os.path.isdir(os.path.join(cfg.DATA_ROOT, d))
    ]
    all_seq_names = natural_sort(all_seq_names)

    allowed_groups = set(cfg.ACCEPT_GROUPS) if getattr(cfg, "ACCEPT_GROUPS", None) else None

    seq_names = []
    for d in all_seq_names:
        m = re.search(r"_group_(\d+)$", d)
        if m:
            g = int(m.group(1))
            if (allowed_groups is None) or (g in allowed_groups):
                seq_names.append(d)
        else:
            # If folder has no _group_X suffix:
            #   - include it only when no filtering is configured
            if allowed_groups is None:
                seq_names.append(d)

    print(f"[INFO] Found {len(seq_names)} candidate sequence folders under {cfg.DATA_ROOT} "
          f"after filtering by ACCEPT_GROUPS={cfg.ACCEPT_GROUPS}")

    for seq_name in tqdm(seq_names, desc="Scanning sequences"):
        seq_path = os.path.join(cfg.DATA_ROOT, seq_name)

        # 1) RGB frames
        all_files = os.listdir(seq_path)
        img_files = [
            f for f in all_files
            if f.lower().startswith("frame_")
            and f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))
        ]
        if not img_files:
            print(f"[WARN] No frame_*.jpg images in {seq_path}, skipping.")
            continue
        img_files = natural_sort(img_files)
        img_paths = [os.path.join(seq_path, f) for f in img_files]

        # 2) Optical flow
        flow_dir = os.path.join(seq_path, "optical_flow")
        if not os.path.isdir(flow_dir):
            print(f"[WARN] No optical_flow folder in {seq_path}, skipping.")
            continue
        flow_files = [
            f for f in os.listdir(flow_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))
        ]
        flow_files = natural_sort(flow_files)
        flow_paths = [os.path.join(flow_dir, f) for f in flow_files]

        if len(img_paths) != len(flow_paths):
            print(f"[WARN] RGB/flow length mismatch in {seq_path} "
                  f"({len(img_paths)} vs {len(flow_paths)}), skipping.")
            continue

        # 3) Landmarks
        lm_path = os.path.join(seq_path, "mediapipe_landmarks.txt")
        if not os.path.isfile(lm_path):
            print(f"[WARN] No mediapipe_landmarks.txt in {seq_path}, skipping.")
            continue
        try:
            lm_arr = np.loadtxt(lm_path, dtype=np.float32)
        except Exception as e:
            print(f"[WARN] Could not load landmarks for {seq_path}: {e}, skipping.")
            continue

        lm_arr = lm_arr.reshape(-1, 12, 2)
        if lm_arr.shape[0] != len(img_paths):
            print(f"[WARN] Landmarks length mismatch in {seq_path} "
                  f"({lm_arr.shape[0]} vs {len(img_paths)}), skipping.")
            continue

        # 4) ground_truth.csv
        gt_path = os.path.join(seq_path, "ground_truth.csv")
        if not os.path.isfile(gt_path):
            print(f"[WARN] No ground_truth.csv in {seq_path}, skipping.")
            continue
        try:
            gt_df = pd.read_csv(gt_path)
        except Exception as e:
            print(f"[WARN] Could not read {gt_path}: {e}, skipping.")
            continue

        if "frame_index" not in gt_df.columns:
            print(f"[WARN] ground_truth.csv in {seq_path} has no 'frame_index' column, skipping.")
            continue

        label_col = detect_label_column(gt_df)
        labels_int = normalize_blink_labels(gt_df[label_col])
        frame_to_label = dict(zip(gt_df["frame_index"].astype(int), labels_int))

        # Build per-frame label list aligned with img_files via frame_index in file name
        frame_labels = []
        missing = False
        for f in img_files:
            try:
                fid = parse_frame_index_from_name(f)
            except ValueError:
                print(f"[WARN] Could not parse frame index from {f} in {seq_path}, skipping sequence.")
                missing = True
                break
            if fid not in frame_to_label:
                print(f"[WARN] frame_index={fid} not found in GT for {seq_path}, skipping sequence.")
                missing = True
                break
            frame_labels.append(int(frame_to_label[fid]))

        if missing:
            continue

        frame_labels = np.array(frame_labels, dtype=np.int64)
        n_frames = len(frame_labels)
        T = cfg.TIME_SEQUENCE

        if n_frames < T:
            print(f"[WARN] Not enough frames for one window in {seq_path} "
                  f"(n_frames={n_frames}, T={T}), skipping.")
            continue

        # Only use full non-overlapping windows
        n_groups = n_frames // T
        effective_len = n_groups * T

        frame_labels = frame_labels[:effective_len]
        img_paths = img_paths[:effective_len]
        flow_paths = flow_paths[:effective_len]
        lm_arr = lm_arr[:effective_len]

        # Group labels: majority vote (aligned with your original group-label logic)
        group_labels = []
        for g in range(n_groups):
            chunk = frame_labels[g * T:(g + 1) * T]
            mean_val = chunk.mean()
            # majority: >= 0.5 → blink
            group_label = 1 if mean_val >= 0.5 else 0
            group_labels.append(group_label)
        group_labels = np.array(group_labels, dtype=np.int64)

        # Cache metadata
        metadata_cache[seq_path] = {
            "image_paths": img_paths,
            "flow_paths": flow_paths,
            "landmarks": lm_arr,
        }
        valid_seq_infos.append({
            "path": seq_path,
            "group_labels": group_labels,
        })

    print(f"[INFO] Valid sequences for training/validation: {len(valid_seq_infos)}")
    if not valid_seq_infos:
        raise RuntimeError("No valid sequences found in DATA_ROOT; cannot train.")

    return valid_seq_infos, metadata_cache


# ============================================================
# GROUP SAMPLES (WINDOWS)
# ============================================================
def create_group_samples(cfg: Config, all_seq_infos: List[Dict[str, Any]]):
    """
    Convert sequence-level group_labels into per-window samples:
      {
        'seq_path': seq_path,
        'start_frame': start_frame_index,
        'label': group_label (0/1)
      }
    """
    group_samples = []
    for seq_info in all_seq_infos:
        seq_path = seq_info["path"]
        group_labels = seq_info["group_labels"]
        for group_idx, group_label in enumerate(group_labels):
            start_frame = group_idx * cfg.TIME_SEQUENCE
            group_samples.append({
                "seq_path": seq_path,
                "start_frame": start_frame,
                "label": int(group_label),
            })
    return group_samples


# ============================================================
# DATASET
# ============================================================
class GroupBlinkSequenceDataset(torch.utils.data.Dataset):
    def __init__(self, group_samples, metadata_cache, cfg: Config, is_train: bool = True):
        self.group_samples = group_samples
        self.metadata_cache = metadata_cache
        self.cfg = cfg
        self.is_train = is_train
        self.transform = Augmentation(cfg.IMAGE_SIZE, is_train)

    def __len__(self):
        return len(self.group_samples)

    def __getitem__(self, idx):
        info = self.group_samples[idx]
        seq_path = info["seq_path"]
        group_start = info["start_frame"]
        label = info["label"]

        cache = self.metadata_cache[seq_path]
        img_paths = cache["image_paths"]
        flow_paths = cache["flow_paths"]
        landmarks = cache["landmarks"]  # [N,12,2]

        rgb_seq, flow_seq, geom_seq = [], [], []
        prev_ear = -1.0

        for i in range(self.cfg.TIME_SEQUENCE):
            frame_idx = group_start + i

            rgb_np = np.array(Image.open(img_paths[frame_idx]).convert("RGB"))
            flow_np = np.array(Image.open(flow_paths[frame_idx]).convert("RGB"))

            rgb_seq.append(self.transform(rgb_np))
            flow_seq.append(self.transform(flow_np))

            lm = landmarks[frame_idx]  # [12,2]
            l_ear = calculate_ear(lm[:6])
            r_ear = calculate_ear(lm[6:12])
            avg_ear = (l_ear + r_ear) / 2.0
            delta_ear = (avg_ear - prev_ear) if prev_ear >= 0 else 0.0

            geom_feat = np.array(
                [avg_ear, delta_ear, 0.0, 0.0],
                dtype=np.float32
            )[: self.cfg.GEOM_FEAT_DIM]

            geom_seq.append(geom_feat)
            prev_ear = avg_ear

        rgb_seq = torch.stack(rgb_seq)  # [T,3,H,W]
        flow_seq = torch.stack(flow_seq)
        geom_seq = torch.tensor(np.stack(geom_seq, axis=0), dtype=torch.float32)  # [T,G]

        return rgb_seq, flow_seq, geom_seq, torch.tensor(label, dtype=torch.long)


# ============================================================
# MODEL (EfficientNet B0 dual-stream + Gated Fusion + BiLSTM)
# ============================================================
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
    def __init__(self, input_dim: int, hidden_dim: int, num_classes: int, dropout: float):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x):  # x: [B, T, D]
        o, _ = self.lstm(x)           # [B, T, 2H]
        x = self.drop(o[:, -1, :])    # last timestep
        return self.fc(x)             # [B, num_classes]


class GatedFusion(nn.Module):
    """
    Gated fusion module to weigh modalities (RGB, flow, geom).
    """

    def __init__(self, input_dim: int, hidden_dim: int = 32):
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

        fusion_input_dim = 256 * 2 + cfg.GEOM_FEAT_DIM
        self.gated_fusion = GatedFusion(fusion_input_dim)
        self.temporal = TemporalBiLSTM(fusion_input_dim, 128, cfg.NUM_CLASSES, cfg.DROPOUT)

    def forward(self, rgb, flow, geom):
        # rgb, flow: [B,T,3,H,W]; geom: [B,T,G]
        rgb_feat = self.spatial_stream(rgb)      # [B,T,256]
        flow_feat = self.temporal_stream(flow)   # [B,T,256]
        fused = torch.cat([rgb_feat, flow_feat, geom], dim=-1)  # [B,T,512+G]
        gated = self.gated_fusion(fused)
        return self.temporal(gated)              # [B,2]


# ============================================================
# FOCAL LOSS
# ============================================================
class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, alpha: float = 0.25, reduction: str = "mean"):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, logits, targets):
        ce_loss = nn.functional.cross_entropy(logits, targets, reduction="none")
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        else:
            return focal_loss


# ============================================================
# TRAINING LOOP
# ============================================================
def train_and_eval(cfg: Config, model_save_name: str):
    # ---- Build metadata from Stage1_new_Data ----
    valid_seq_infos, metadata_cache = build_sequence_metadata(cfg)

    # ---- Train/val split by sequence ----
    indices = np.arange(len(valid_seq_infos))
    train_idx, val_idx = train_test_split(indices,
                                          test_size=cfg.VAL_SPLIT,
                                          random_state=cfg.SEED,
                                          shuffle=True)

    train_seq_infos = [valid_seq_infos[i] for i in train_idx]
    val_seq_infos = [valid_seq_infos[i] for i in val_idx]

    train_samples = create_group_samples(cfg, train_seq_infos)
    val_samples = create_group_samples(cfg, val_seq_infos)

    print(f"[INFO] Train windows: {len(train_samples)} | Val windows: {len(val_samples)}")

    # ---- WeightedRandomSampler for class imbalance ----
    train_labels = [s["label"] for s in train_samples]
    class_counts = Counter(train_labels)
    print(f"[INFO] Class counts in train windows: {class_counts}")

    class_weight_per_label = {c: 1.0 / max(cnt, 1) for c, cnt in class_counts.items()}
    sample_weights = [class_weight_per_label[lbl] for lbl in train_labels]

    sampler = WeightedRandomSampler(weights=sample_weights,
                                    num_samples=len(sample_weights),
                                    replacement=True)

    train_loader = DataLoader(
        GroupBlinkSequenceDataset(train_samples, metadata_cache, cfg, is_train=True),
        batch_size=cfg.BATCH_SIZE,
        sampler=sampler,
        num_workers=cfg.NUM_WORKERS,
        pin_memory=True,
    )

    val_loader = DataLoader(
        GroupBlinkSequenceDataset(val_samples, metadata_cache, cfg, is_train=False),
        batch_size=cfg.BATCH_SIZE,
        shuffle=False,
        num_workers=cfg.NUM_WORKERS,
        pin_memory=True,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    model = DualStreamBlinkModel(cfg).to(device)
    criterion = FocalLoss(gamma=2.0, alpha=0.25)
    optimizer = optim.AdamW(model.parameters(),
                            lr=cfg.LR,
                            weight_decay=cfg.WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                     mode="max",
                                                     factor=0.3,
                                                     patience=3)
    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())

    best_f1 = 0.0
    patience = 0

    # history for CSV
    history = []

    for epoch in range(cfg.EPOCHS):
        # --------------- TRAIN ---------------
        model.train()
        total_loss = 0.0
        train_preds, train_labels_collect = [], []

        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{cfg.EPOCHS} [Train]"):
            rgb, flow, geom, labels = batch
            rgb = rgb.to(device)
            flow = flow.to(device)
            geom = geom.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                logits = model(rgb, flow, geom)
                loss = criterion(logits, labels)

            # collect predictions for metrics
            preds = torch.argmax(logits, dim=-1)
            train_preds.extend(preds.detach().cpu().numpy())
            train_labels_collect.extend(labels.detach().cpu().numpy())

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()

        avg_train_loss = total_loss / max(len(train_loader), 1)

        # ---- train metrics ----
        if len(train_preds) > 0:
            train_p, train_r, train_f1, _ = precision_recall_fscore_support(
                train_labels_collect,
                train_preds,
                labels=[0, 1],
                zero_division=0,
            )
            train_macro_f1 = train_f1.mean()
        else:
            train_p = train_r = train_f1 = np.array([0.0, 0.0])
            train_macro_f1 = 0.0

        # --------------- VALIDATION ---------------
        model.eval()
        val_preds, val_labels_collect = [], []
        val_total_loss = 0.0

        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch + 1}/{cfg.EPOCHS} [Val]"):
                rgb, flow, geom, labels = batch
                rgb = rgb.to(device)
                flow = flow.to(device)
                geom = geom.to(device)
                labels = labels.to(device)

                with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                    logits = model(rgb, flow, geom)
                    loss = criterion(logits, labels)

                val_total_loss += loss.item()

                preds = torch.argmax(logits, dim=-1)
                val_preds.extend(preds.cpu().numpy())
                val_labels_collect.extend(labels.cpu().numpy())

        avg_val_loss = val_total_loss / max(len(val_loader), 1)

        if len(val_preds) > 0:
            val_p, val_r, val_f1, _ = precision_recall_fscore_support(
                val_labels_collect,
                val_preds,
                labels=[0, 1],
                zero_division=0,
            )
            val_macro_f1 = val_f1.mean()
        else:
            val_p = val_r = val_f1 = np.array([0.0, 0.0])
            val_macro_f1 = 0.0

        current_lr = optimizer.param_groups[0]["lr"]

        # ---------- logging ----------
        print(
            f"[Epoch {epoch + 1}] "
            f"Train: NonBlink P/R/F1=({train_p[0]:.3f}/{train_r[0]:.3f}/{train_f1[0]:.3f}), "
            f"Blink P/R/F1=({train_p[1]:.3f}/{train_r[1]:.3f}/{train_f1[1]:.3f}), "
            f"MacroF1={train_macro_f1:.3f} | "
            f"Loss={avg_train_loss:.4f}"
        )
        print(
            f"[Epoch {epoch + 1}] "
            f" Val : NonBlink P/R/F1=({val_p[0]:.3f}/{val_r[0]:.3f}/{val_f1[0]:.3f}), "
            f"Blink P/R/F1=({val_p[1]:.3f}/{val_r[1]:.3f}/{val_f1[1]:.3f}), "
            f"MacroF1={val_macro_f1:.3f} | "
            f"ValLoss={avg_val_loss:.4f} | "
            f"LR={current_lr:.1e}"
        )

        # ---------- save to history ----------
        history.append({
            "epoch": epoch + 1,
            "train_loss": float(avg_train_loss),
            "val_loss": float(avg_val_loss),
            "train_macro_f1": float(train_macro_f1),
            "val_macro_f1": float(val_macro_f1),
            "train_prec_nonblink": float(train_p[0]),
            "train_rec_nonblink": float(train_r[0]),
            "train_f1_nonblink": float(train_f1[0]),
            "train_prec_blink": float(train_p[1]),
            "train_rec_blink": float(train_r[1]),
            "train_f1_blink": float(train_f1[1]),
            "val_prec_nonblink": float(val_p[0]),
            "val_rec_nonblink": float(val_r[0]),
            "val_f1_nonblink": float(val_f1[0]),
            "val_prec_blink": float(val_p[1]),
            "val_rec_blink": float(val_r[1]),
            "val_f1_blink": float(val_f1[1]),
            "lr": float(current_lr),
        })

        # ---------- scheduler & early stopping on val macro F1 ----------
        scheduler.step(val_macro_f1)

        if val_macro_f1 > best_f1:
            best_f1 = val_macro_f1
            patience = 0
            torch.save(model.state_dict(), model_save_name)
            print(f"✅ New best model saved: {model_save_name} "
                  f"(Val MacroF1: {best_f1:.4f})")
        else:
            patience += 1
            if patience >= cfg.PATIENCE:
                print(f"[INFO] Early stopping triggered (no improvement for {patience} epochs).")
                break

    # ---------- write CSV log ----------
    log_path = os.path.splitext(model_save_name)[0] + "_log.csv"
    df_hist = pd.DataFrame(history)
    df_hist.to_csv(log_path, index=False)
    print(f"[INFO] Training history saved to: {log_path}")

# ============================================================
# ENTRY POINT
# ============================================================
if __name__ == "__main__":
    model_filename = f"g10model_gated_fusion_stage1_seed_{cfg.SEED}.pth"
    print(f"[INFO] Starting training with Stage1_new_Data. "
          f"Model will be saved as: {model_filename}")
    train_and_eval(cfg, model_filename)
