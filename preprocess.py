import os
import re
import shutil
from typing import List, Tuple

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch


# =============================================================================
# FIXED PATHS (EDIT HERE IF NEEDED)
# =============================================================================
INPUT_ROOT = os.environ.get("BLINK_INPUT_ROOT", os.path.join("data", "stage0"))
OUTPUT_ROOT = os.environ.get("BLINK_OUTPUT_ROOT", os.path.join("data", "stage1"))
GT_FILENAME = "ground_truth.csv"
TARGET_SIZE = (256, 192)  # (width, height) for final training images
# =============================================================================


def ensure_dir(p: str) -> str:
    os.makedirs(p, exist_ok=True)
    return p


def tryint(s):
    try:
        return int(s)
    except Exception:
        return s


def alphanum_key(s: str):
    return [tryint(c) for c in re.split(r"([0-9]+)", s)]


def natural_sort(l: List[str]) -> List[str]:
    return sorted(l, key=alphanum_key)


def parse_frame_index_from_name(filename: str) -> int:
    """
    Extract numeric index from a filename, e.g.:
        frame_2.jpg   -> 2
        frame_0023.png -> 23
    """
    m = re.search(r'(\d+)', filename)
    if not m:
        raise ValueError(f"Could not parse frame index from {filename}")
    return int(m.group(1))


# ----------------------------
# 1) Filter & crop frames + update GT (InsightFace)
# ----------------------------
def filter_frames_and_update_gt(
    seq_dir: str,
    gt_filename: str,
    target_size: Tuple[int, int],
    crop_and_resize: bool = True,
) -> List[str]:
    """
    Step 1:
      - Run InsightFace on each frame_*.jpg
      - If NO face:
          * print message
          * delete the frame file
          * drop corresponding GT row (by frame_index)
      - If face exists:
          * optionally crop to largest face and resize to target_size
            (OVERWRITES the original image)

    Returns:
      A natural-sorted list of remaining frame filenames.
    """
    seq_dir = os.path.abspath(seq_dir)
    gt_path = os.path.join(seq_dir, gt_filename)

    if not os.path.isfile(gt_path):
        raise FileNotFoundError(f"ground truth file not found: {gt_path}")

    gt_df = pd.read_csv(gt_path)

    # Collect frame files
    image_files = [
        f for f in os.listdir(seq_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))
        and f.lower().startswith("frame_")
    ]
    if not image_files:
        print(f"[WARN] No frame_*.jpg images found in {seq_dir}")
        return []

    image_files = natural_sort(image_files)

    # Setup InsightFace detector (same style as original script)
    try:
        from insightface.app import FaceAnalysis
    except Exception as e:
        raise RuntimeError("insightface is required for face filtering. Install it first.") from e

    app = FaceAnalysis()
    app.prepare(ctx_id=0 if torch.cuda.is_available() else -1)

    removed_indices = []

    pbar = tqdm(image_files, desc=f"Face filter: {os.path.basename(seq_dir)}", unit="frame")
    for fname in pbar:
        img_path = os.path.join(seq_dir, fname)
        try:
            frame_idx = parse_frame_index_from_name(fname)
        except ValueError:
            print(f"[WARN] Skipping file (no numeric index found): {fname}")
            continue

        img = cv2.imread(img_path)
        if img is None:
            print(f"[WARN] Could not read image: {img_path} → removing image and GT row if present.")
            removed_indices.append(frame_idx)
            try:
                os.remove(img_path)
            except OSError:
                pass
            continue

        # Use InsightFace with default threshold (as in the original cropping script)
        faces = app.get(img)
        if len(faces) == 0:
            print(f"[INFO] NO FACE detected in {fname} (frame_index={frame_idx}) → removing this frame and its GT.")
            removed_indices.append(frame_idx)
            try:
                os.remove(img_path)
            except OSError:
                pass
            continue

        if crop_and_resize:
            # Choose largest face, same logic as original
            bboxes = [f["bbox"] for f in faces]
            areas = [(b[2] - b[0]) * (b[3] - b[1]) for b in bboxes]
            best_i = int(np.argmax(areas))
            x1, y1, x2, y2 = map(int, bboxes[best_i])

            h, w, _ = img.shape
            x1 = max(0, min(x1, w - 1))
            x2 = max(0, min(x2, w))
            y1 = max(0, min(y1, h - 1))
            y2 = max(0, min(y2, h))

            if x2 <= x1 or y2 <= y1:
                print(
                    f"[WARN] Degenerate bbox for {fname} (frame_index={frame_idx}) "
                    f"→ treating as NO FACE and removing."
                )
                removed_indices.append(frame_idx)
                try:
                    os.remove(img_path)
                except OSError:
                    pass
                continue

            crop = img[y1:y2, x1:x2]
            resized = cv2.resize(crop, target_size)
            cv2.imwrite(img_path, resized)

    pbar.close()

    # Update GT: remove rows whose frame_index was removed
    if removed_indices:
        removed_set = set(removed_indices)
        before = len(gt_df)
        gt_df = gt_df[~gt_df["frame_index"].isin(removed_set)]
        after = len(gt_df)
        print(f"[INFO] Removed {before - after} GT rows because frames had no face.")

    # Also ensure GT only contains entries for frames that actually exist on disk
    remaining_files = [
        f for f in os.listdir(seq_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))
        and f.lower().startswith("frame_")
    ]
    remaining_files = natural_sort(remaining_files)
    remaining_indices = set()
    for f in remaining_files:
        try:
            idx = parse_frame_index_from_name(f)
            remaining_indices.add(idx)
        except ValueError:
            continue

    before2 = len(gt_df)
    gt_df = gt_df[gt_df["frame_index"].isin(remaining_indices)]
    after2 = len(gt_df)
    if after2 != before2:
        print(f"[INFO] Removed {before2 - after2} additional GT rows with no corresponding frame file on disk.")

    # Save GT back
    gt_df.to_csv(gt_path, index=False)
    print(
        f"[DONE] Face filtering in {os.path.basename(seq_dir)}: "
        f"{len(remaining_files)} frames kept after removal."
    )

    return remaining_files


# ----------------------------
# 2) Optical flow on remaining frames
# ----------------------------
def compute_optical_flow_for_folder(
    seq_dir: str,
    frame_files: List[str],
    flow_subfolder: str = "optical_flow",
) -> None:
    """
    Compute dense Farneback optical flow between consecutive frames
    (using the updated/filtered frame_files list) and save HSV→BGR
    visualizations aligned 1:1 with RGB frames.
    """
    if not frame_files:
        print(f"[WARN] No frames to compute optical flow in {seq_dir}")
        return

    flow_dir = ensure_dir(os.path.join(seq_dir, flow_subfolder))

    first_path = os.path.join(seq_dir, frame_files[0])
    prev = cv2.imread(first_path)
    if prev is None:
        raise RuntimeError(f"Could not read first frame for flow: {first_path}")
    prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)

    # First frame: zero-flow image
    zero_flow = np.zeros_like(prev)
    cv2.imwrite(os.path.join(flow_dir, frame_files[0]), zero_flow)

    pbar = tqdm(range(1, len(frame_files)), desc=f"Optical flow: {os.path.basename(seq_dir)}", unit="frame")
    for i in pbar:
        fname = frame_files[i]
        cur_path = os.path.join(seq_dir, fname)
        cur = cv2.imread(cur_path)
        if cur is None:
            print(f"[WARN] Could not read frame for flow: {cur_path} → writing zero-flow image.")
            cv2.imwrite(os.path.join(flow_dir, fname), zero_flow)
            continue

        cur_gray = cv2.cvtColor(cur, cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(
            prev_gray,
            cur_gray,
            None,
            pyr_scale=0.5,
            levels=3,
            winsize=15,
            iterations=3,
            poly_n=5,
            poly_sigma=1.2,
            flags=0,
        )
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        hsv = np.zeros_like(cur)
        hsv[..., 0] = ang * 180 / np.pi / 2
        hsv[..., 1] = 255
        hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        cv2.imwrite(os.path.join(flow_dir, fname), bgr)
        prev_gray = cur_gray

    pbar.close()


# ----------------------------
# 3) Eye landmarks on FINAL images (MediaPipe, low conf)
# ----------------------------
def compute_eye_landmarks_for_folder(
    seq_dir: str,
    frame_files: List[str],
    out_name: str = "mediapipe_landmarks.txt",
) -> None:
    """
    Run MediaPipe FaceMesh on each FINAL (cropped/resized) frame and save
    12 (6+6) eye landmarks as 24 floats (x,y) per frame to out_name.

    Landmarks are computed directly in pixel coordinates of the image used
    for training, so they are guaranteed to match EXACTLY the training RGB
    stream.

    If FaceMesh fails on a frame:
       - Print a warning
       - Reuse last valid landmarks (or zeros for very first frames)
       - This keeps the landmarks length identical to the number of frames.
    """
    try:
        import mediapipe as mp
    except Exception as e:
        raise RuntimeError("mediapipe is required for eye landmarks. Install it first.") from e

    if not frame_files:
        print(f"[WARN] No frames found for landmarks in {seq_dir}")
        return

    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        # More permissive than default
        min_detection_confidence=0.15,
    )

    LEFT = [362, 385, 387, 263, 373, 380]
    RIGHT = [33, 160, 158, 133, 153, 144]

    all_lm: List[np.ndarray] = []
    last_good: np.ndarray = None

    pbar = tqdm(frame_files, desc=f"Landmarks (MediaPipe): {os.path.basename(seq_dir)}", unit="frame")
    for fname in pbar:
        img_path = os.path.join(seq_dir, fname)
        img = cv2.imread(img_path)
        if img is None:
            print(f"[WARN] Could not read image for landmarks: {img_path}")
            lm = last_good if last_good is not None else np.zeros(24, dtype=np.float32)
            all_lm.append(lm)
            continue

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        res = face_mesh.process(img_rgb)

        if res.multi_face_landmarks:
            fl = res.multi_face_landmarks[0]
            h, w, _ = img.shape
            L = np.array([[fl.landmark[i].x * w, fl.landmark[i].y * h] for i in LEFT], dtype=np.float32)
            R = np.array([[fl.landmark[i].x * w, fl.landmark[i].y * h] for i in RIGHT], dtype=np.float32)
            lm_eye = np.concatenate([L, R], axis=0).reshape(-1)  # 12*2=24
            last_good = lm_eye
            all_lm.append(lm_eye)
        else:
            print(f"[WARN] MediaPipe could not find landmarks in {fname} → reusing last valid landmarks.")
            lm = last_good if last_good is not None else np.zeros(24, dtype=np.float32)
            all_lm.append(lm)

    pbar.close()
    face_mesh.close()

    lm_arr = np.vstack(all_lm)
    out_path = os.path.join(seq_dir, out_name)
    np.savetxt(out_path, lm_arr, fmt="%.4f")

    print(
        f"[DONE] MediaPipe eye landmarks for {os.path.basename(seq_dir)}: "
        f"{lm_arr.shape[0]} frames × {lm_arr.shape[1]} values saved to {out_path}"
    )


# ----------------------------
# 4) Full pipeline for ONE Stage1 sequence
# ----------------------------
def preprocess_stage1_sequence(
    seq_dir: str,
    gt_filename: str,
    target_size: Tuple[int, int],
) -> None:
    """
    Run all preprocessing steps on a single Stage1 sequence folder:
      1) Filter frames by face + crop/resize + update GT (InsightFace)
      2) Optical flow on remaining frames
      3) Eye landmarks on FINAL images (MediaPipe)
      4) Consistency checks (RGB, flow, landmarks, GT)
    """
    seq_dir = os.path.abspath(seq_dir)
    print(f"\n[SEQ] Preprocessing Stage1 sequence folder: {seq_dir}")

    # Step 1: filter & crop + update GT
    frame_files = filter_frames_and_update_gt(
        seq_dir=seq_dir,
        gt_filename=gt_filename,
        target_size=target_size,
        crop_and_resize=True,
    )

    if not frame_files:
        print(f"[WARN] No frames remain after face filtering in {seq_dir}. Skipping further steps.")
        return

    # Step 2: optical flow (on final images)
    compute_optical_flow_for_folder(seq_dir, frame_files)

    # Step 3: landmarks on FINAL images (cropped/resized)
    compute_eye_landmarks_for_folder(seq_dir, frame_files)

    # Step 4: consistency checks
    rgb_files = [
        f for f in os.listdir(seq_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))
        and f.lower().startswith("frame_")
    ]
    rgb_files = natural_sort(rgb_files)

    flow_dir = os.path.join(seq_dir, "optical_flow")
    flow_files: List[str] = []
    if os.path.isdir(flow_dir):
        flow_files = [
            f for f in os.listdir(flow_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))
        ]
        flow_files = natural_sort(flow_files)

    gt_path = os.path.join(seq_dir, gt_filename)
    gt_rows = 0
    if os.path.isfile(gt_path):
        gt_rows = len(pd.read_csv(gt_path))

    lm_path = os.path.join(seq_dir, "mediapipe_landmarks.txt")
    lm_rows = 0
    if os.path.isfile(lm_path):
        lm_rows = sum(1 for _ in open(lm_path, "r", encoding="utf-8"))

    print(
        f"[CHECK] {os.path.basename(seq_dir)}: "
        f"RGB={len(rgb_files)}, FLOW={len(flow_files)}, GT_rows={gt_rows}, LM_rows={lm_rows}"
    )

    if not (len(rgb_files) == len(flow_files) == gt_rows == lm_rows):
        print(
            "[WARN] STREAM MISMATCH detected in "
            f"{os.path.basename(seq_dir)} — please inspect manually."
        )
    else:
        print(f"[OK] All streams aligned for {os.path.basename(seq_dir)}.")


# ----------------------------
# 5) Copy Stage0 → Stage1 and run pipeline
# ----------------------------
def copy_stage0_to_stage1_if_needed(seq_name: str) -> str:
    """
    Copy INPUT_ROOT/seq_name to OUTPUT_ROOT/seq_name if not already present.
    Returns the Stage1 path.
    """
    src = os.path.join(INPUT_ROOT, seq_name)
    dst = os.path.join(OUTPUT_ROOT, seq_name)

    if not os.path.isdir(src):
        raise FileNotFoundError(f"Source sequence folder not found: {src}")

    if os.path.isdir(dst):
        print(f"[INFO] Stage1 folder already exists, will reuse: {dst}")
        return dst

    print(f"[COPY] {src}  →  {dst}")
    shutil.copytree(src, dst)
    return dst


def preprocess_all_sequences():
    """
    Main entry: iterate over all sequence folders in INPUT_ROOT and
    preprocess them into OUTPUT_ROOT.
    """
    ensure_dir(OUTPUT_ROOT)

    seq_names = [
        d for d in os.listdir(INPUT_ROOT)
        if os.path.isdir(os.path.join(INPUT_ROOT, d))
    ]
    seq_names = natural_sort(seq_names)

    print(f"[INFO] Found {len(seq_names)} sequence folders in Stage0 (INPUT_ROOT).")

    for seq_name in seq_names:
        stage1_dir = copy_stage0_to_stage1_if_needed(seq_name)
        preprocess_stage1_sequence(stage1_dir, gt_filename=GT_FILENAME, target_size=TARGET_SIZE)


if __name__ == "__main__":
    print(f"[CONFIG] INPUT_ROOT  = {INPUT_ROOT}")
    print(f"[CONFIG] OUTPUT_ROOT = {OUTPUT_ROOT}")
    preprocess_all_sequences()
