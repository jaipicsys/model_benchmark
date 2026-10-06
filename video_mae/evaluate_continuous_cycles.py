#!/usr/bin/env python3
"""
Continuous-cycle VideoMAE evaluation using cycle20GT.ods

Purpose
-------
Evaluate a fine-tuned VideoMAE model on continuous cycle videos using
the same sliding-window approach as the jumbled-video evaluator.

Default test setup:
    window_len = 6.5 seconds
    stride     = 1.0 second
    fps        = 15
    num_frames = 16
    ROI        = 280,0,440,690

Ground truth
------------
cycle20GT.ods contains 20 rows (cycles 1..20). Each step column contains
the START time of that step. The interval for a step is therefore:

    step_start <= t < next_step_start

For a VideoMAE window, the ground-truth class is assigned from the CENTER
of the window. The script also marks windows whose full 6.5-second interval
crosses a ground-truth step boundary, so you can separately inspect:

    1. Accuracy on ALL windows
    2. Accuracy on NON-BOUNDARY windows

This is important for continuous videos because a 6.5-second window can
naturally contain two different production steps.

Expected video names
--------------------
The script searches the video directory for:
    cycle1.mkv, cycle2.mkv, ...
    cycle01.mkv, cycle02.mkv, ...
and also .mp4/.avi/.mov.

If your names are different, use --video_pattern.

Dependencies
------------
This script expects the same VideoMAE inference helper used by your
existing pipeline, specifically:

    extract_session_embeddings(...)

It tries to import it from extract_embeddings.py.

Run
---
python3 evaluate_continuous_cycles.py \
    --videomae_dir runs/roi_v4/videomae_finetuned/best_model \
    --video_dir . \
    --gt cycle20GT.ods \
    --window_len 6.5 \
    --stride 1.0 \
    --fps 15 \
    --num_frames 16 \
    --roi 280,0,440,690

Output:
    continuous_cycle_report_6.5s.xlsx
"""

import argparse
import re
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from transformers import VideoMAEImageProcessor, VideoMAEForVideoClassification


# ---------------------------------------------------------------------
# Class-name normalization
# ---------------------------------------------------------------------

def canonical_class_name(name):
    """Normalize class names so GT/model naming differences don't count wrong."""
    if name is None:
        return ""

    s = str(name).strip().lower()
    s = s.replace("–", "-").replace("—", "-")
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"[_-]+", "_", s)

    aliases = {
        "step1": "start_step1",
        "start_step1": "start_step1",

        "step2": "start_step2",
        "start_step2": "start_step2",

        "step4": "step4",
        "start_step4": "step4",

        "step5": "step5",
        "start_step5": "step5",

        "step6": "step6",
        "start_step6": "step6",

        "step6.5": "step6.5",
        "start_step6.5": "step6.5",

        "step7": "step7",
        "start_step7": "step7",

        "step8": "start_step8",
        "start_step8": "start_step8",

        "step9": "step9",
        "start_step9": "step9",

        "step10": "step10",
        "start_step10": "step10",

        "step11": "start_step11",
        "start_step11": "start_step11",

        "stop_step1": "stop_step1",
        "stop_step2": "stop_step2",
    }

    return aliases.get(s, s)


def display_class_name(name):
    pretty = {
        "start_step1": "Step 1",
        "start_step2": "Step 2",
        "step4": "Step 4",
        "step5": "Step 5",
        "step6": "Step 6",
        "step6.5": "Step 6.5",
        "step7": "Step 7",
        "start_step8": "Step 8",
        "step9": "Step 9",
        "step10": "Step 10",
        "start_step11": "Step 11",
        "stop_step1": "Stop Step 1",
        "stop_step2": "Stop Step 2",
    }
    return pretty.get(name, name)


def classes_match(a, b):
    return canonical_class_name(a) == canonical_class_name(b)


# ---------------------------------------------------------------------
# GT parsing
# ---------------------------------------------------------------------

GT_COLUMN_MAP = {
    "Start_step1 – Visual check": "start_step1",
    "start_step2 – Heatshield placement": "start_step2",
    "STEP4 – Place bracket": "step4",
    "STEP5 – Place drilled lever assembly": "step5",
    "Step6 – Fit spindle": "step6",
    "Step 6.5 - Rivit fitting": "step6.5",
    "Step7 – Fit Actuator": "step7",
    "Step8- Fit bracket nuts": "start_step8",
    "Step9- Fasten bracket nuts using torque gun": "step9",
    "Step10- Fasten actuator fitting": "step10",
    "Step11 – Place pipe bracket": "start_step11",
    "Stop_step1 - Fit pipe bracket": "stop_step1",
    "Stop_step2 - Final tightenning": "stop_step2",
}


def time_to_seconds(value):
    if pd.isna(value):
        return None

    if hasattr(value, "hour") and hasattr(value, "minute"):
        return (
            value.hour * 3600
            + value.minute * 60
            + value.second
            + getattr(value, "microsecond", 0) / 1e6
        )

    s = str(value).strip()
    if not s or s.lower() in {"nan", "none", "from", "to"}:
        return None

    parts = s.split(":")
    try:
        if len(parts) == 3:
            return (
                float(parts[0]) * 3600
                + float(parts[1]) * 60
                + float(parts[2])
            )
        if len(parts) == 2:
            return float(parts[0]) * 60 + float(parts[1])
        return float(s)
    except ValueError:
        return None


def load_ground_truth(gt_path):
    df = pd.read_excel(gt_path, engine="odf")

    # The uploaded GT has the cycle number in the "cycle" column.
    if "cycle" not in df.columns:
        raise ValueError(
            f'Could not find "cycle" column in {gt_path}. '
            f"Columns found: {list(df.columns)}"
        )

    cycles = {}

    for _, row in df.iterrows():
        cycle_value = row["cycle"]
        if pd.isna(cycle_value):
            continue

        cycle_num = int(float(cycle_value))
        starts = []

        for source_col, label in GT_COLUMN_MAP.items():
            if source_col not in df.columns:
                continue

            t = time_to_seconds(row[source_col])
            if t is not None:
                starts.append((t, label))

        starts.sort(key=lambda x: x[0])

        # Remove duplicate timestamps while preserving first occurrence.
        cleaned = []
        seen = set()
        for t, label in starts:
            key = (round(t, 6), label)
            if key not in seen:
                cleaned.append((t, label))
                seen.add(key)

        cycles[cycle_num] = cleaned

    return cycles


def ground_truth_at(cycle_starts, t):
    """
    Return (label, start, end).

    The GT labels represent step START times, so each step owns the interval
    from its start until the next step starts.
    """
    if not cycle_starts:
        return None, None, None

    for i, (start, label) in enumerate(cycle_starts):
        end = cycle_starts[i + 1][0] if i + 1 < len(cycle_starts) else None

        if t >= start and (end is None or t < end):
            return label, start, end

    return None, None, None


def boundary_info(cycle_starts, window_start, window_end):
    """
    A window is marked boundary=True if any GT step transition occurs inside
    the window. This does NOT remove the window; it only labels it.
    """
    boundaries = []
    for t, _ in cycle_starts[1:]:
        if window_start < t < window_end:
            boundaries.append(t)

    return bool(boundaries), boundaries


# ---------------------------------------------------------------------
# VideoMAE loading
# ---------------------------------------------------------------------

def parse_roi(s):
    parts = [int(x.strip()) for x in s.split(",")]
    if len(parts) != 4:
        raise ValueError("--roi must be x,y,w,h")
    return tuple(parts)


def load_model(model_dir):
    processor = VideoMAEImageProcessor.from_pretrained(model_dir)
    model = VideoMAEForVideoClassification.from_pretrained(model_dir)
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    return model, processor, device


def get_id2label(model):
    id2label = model.config.id2label
    result = {}
    for k, v in id2label.items():
        result[int(k)] = v
    return result


# ---------------------------------------------------------------------
# Existing inference helper
# ---------------------------------------------------------------------

def import_inference_helper():
    """
    Import the user's existing helper rather than duplicating its sampling
    implementation. This keeps the continuous test consistent with the
    existing jumbled-video evaluator.

    Expected function:
        extract_session_embeddings(
            video_path,
            model,
            processor,
            window_len,
            stride,
            fps,
            num_frames,
            roi
        )
    """
    try:
        from extract_embeddings import extract_session_embeddings
        return extract_session_embeddings
    except ImportError as e:
        raise ImportError(
            "\nCould not import extract_session_embeddings from "
            "extract_embeddings.py.\n\n"
            "Keep this evaluator in the same directory as your existing "
            "extract_embeddings.py, or change import_inference_helper() "
            "to match your current helper module.\n"
        ) from e


# ---------------------------------------------------------------------
# Video duration / window timestamps
# ---------------------------------------------------------------------

def video_duration(video_path):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    cap.release()

    if fps and fps > 0 and frames and frames > 0:
        return frames / fps

    return None


def build_window_times(duration, window_len, stride):
    if duration is None or duration <= 0:
        return []

    if duration < window_len:
        return []

    starts = []
    t = 0.0

    # Match the usual sliding-window convention.
    while t + window_len <= duration + 1e-6:
        starts.append((t, t + window_len))
        t += stride

    return starts


# ---------------------------------------------------------------------
# Video discovery
# ---------------------------------------------------------------------

def cycle_number_from_filename(path):
    m = re.search(r"cycle[_-]?(\d+)", path.stem.lower())
    return int(m.group(1)) if m else None


def find_cycle_videos(video_dir, pattern=None):
    video_dir = Path(video_dir)

    if pattern:
        paths = sorted(video_dir.glob(pattern))
    else:
        paths = []
        for ext in ("*.mkv", "*.mp4", "*.avi", "*.mov"):
            paths.extend(video_dir.glob(ext))

    items = []
    for p in paths:
        n = cycle_number_from_filename(p)
        if n is not None:
            items.append((n, p))

    items.sort(key=lambda x: x[0])
    return items


# ---------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate VideoMAE on continuous production cycles."
    )

    parser.add_argument("--videomae_dir", required=True)
    parser.add_argument("--video_dir", default=".")
    parser.add_argument("--gt", default="cycle20GT.ods")

    parser.add_argument("--window_len", type=float, default=6.5)
    parser.add_argument("--stride", type=float, default=1.0)
    parser.add_argument("--fps", type=float, default=15)
    parser.add_argument("--num_frames", type=int, default=16)
    parser.add_argument("--roi", default="280,0,440,690")

    parser.add_argument(
        "--video_pattern",
        default=None,
        help='Optional glob such as "cycle*.mkv".'
    )

    parser.add_argument(
        "--output",
        default=None,
        help="Output XLSX filename. Defaults to continuous_cycle_report_<window>s.xlsx"
    )

    args = parser.parse_args()

    roi = parse_roi(args.roi)
    gt_path = Path(args.gt)

    if not gt_path.exists():
        raise FileNotFoundError(f"Ground-truth file not found: {gt_path}")

    if args.output:
        output_path = Path(args.output)
    else:
        output_path = Path(
            f"continuous_cycle_report_{args.window_len:g}s.xlsx"
        )

    print("=" * 70)
    print("Continuous Cycle VideoMAE Evaluation")
    print("=" * 70)
    print(f"Model       : {args.videomae_dir}")
    print(f"Video dir   : {args.video_dir}")
    print(f"GT          : {args.gt}")
    print(f"Window      : {args.window_len:.2f} s")
    print(f"Stride      : {args.stride:.2f} s")
    print(f"Sampling FPS: {args.fps:g}")
    print(f"Frames      : {args.num_frames}")
    print(f"ROI         : {roi}")
    print()

    gt_cycles = load_ground_truth(gt_path)
    print(f"Loaded GT for {len(gt_cycles)} cycles.")

    videos = find_cycle_videos(args.video_dir, args.video_pattern)

    if not videos:
        raise RuntimeError(
            f"No cycle videos found in {args.video_dir}. "
            f"Expected names such as cycle1.mkv, cycle2.mkv, ..."
        )

    print(f"Found {len(videos)} cycle videos.")

    model, processor, device = load_model(args.videomae_dir)
    id2label = get_id2label(model)

    print(f"Device      : {device}")
    print(f"Model labels: {id2label}")
    print()

    extract_session_embeddings = import_inference_helper()

    all_rows = []
    cycle_rows = []

    for cycle_num, video_path in videos:
        print(f"[Cycle {cycle_num:02d}] {video_path.name}")

        if cycle_num not in gt_cycles:
            print("  WARNING: no GT row for this cycle; skipping.")
            continue

        duration = video_duration(video_path)
        if duration is None:
            print("  WARNING: could not determine duration.")
            continue

        window_times = build_window_times(
            duration, args.window_len, args.stride
        )

        if not window_times:
            print("  WARNING: video shorter than window length.")
            continue

        # The existing helper is expected to return:
        #   probs, times
        #
        # times may be window-start timestamps or equivalent timestamps
        # from the existing jumbled evaluator.
        result = extract_session_embeddings(
            str(video_path),
            model,
            processor,
            args.window_len,
            args.stride,
            args.fps,
            args.num_frames,
            roi,
        )

        if isinstance(result, tuple) and len(result) >= 2:
            probs = result[0]
            inference_times = result[1]
        else:
            raise RuntimeError(
                "extract_session_embeddings() must return at least "
                "(probs, times)."
            )

        probs = np.asarray(probs)
        if probs.ndim == 1:
            probs = probs.reshape(1, -1)

        pred_ids = probs.argmax(axis=1)
        pred_conf = probs.max(axis=1)

        # If helper provides one timestamp per prediction, use it.
        # Otherwise fall back to our own sliding-window starts.
        if inference_times is None or len(inference_times) != len(pred_ids):
            inference_times = [x[0] for x in window_times]

        correct = 0
        nonboundary_correct = 0
        nonboundary_total = 0
        boundary_total = 0

        for i, (pred_id, conf, t) in enumerate(
            zip(pred_ids, pred_conf, inference_times)
        ):
            try:
                start = float(t)
            except Exception:
                start = float(i * args.stride)

            end = start + args.window_len
            center = start + args.window_len / 2.0

            gt_label, gt_start, gt_end = ground_truth_at(
                gt_cycles[cycle_num], center
            )

            pred_raw = id2label.get(int(pred_id), str(int(pred_id)))
            pred_label = canonical_class_name(pred_raw)

            is_correct = classes_match(pred_label, gt_label)

            is_boundary, boundary_times = boundary_info(
                gt_cycles[cycle_num], start, end
            )

            if is_correct:
                correct += 1

            if is_boundary:
                boundary_total += 1
            else:
                nonboundary_total += 1
                if is_correct:
                    nonboundary_correct += 1

            all_rows.append({
                "cycle": cycle_num,
                "video": video_path.name,
                "window_index": i,
                "window_start_sec": round(start, 3),
                "window_end_sec": round(end, 3),
                "window_center_sec": round(center, 3),

                "ground_truth": display_class_name(gt_label),
                "ground_truth_canonical": gt_label,

                "prediction": display_class_name(pred_label),
                "prediction_canonical": pred_label,

                "confidence": round(float(conf), 6),
                "correct": bool(is_correct),

                "boundary_window": bool(is_boundary),
                "boundary_times_sec": ", ".join(
                    f"{x:.2f}" for x in boundary_times
                ),

                "gt_step_start_sec": (
                    round(gt_start, 3) if gt_start is not None else None
                ),
                "gt_step_end_sec": (
                    round(gt_end, 3) if gt_end is not None else None
                ),
            })

        total = len(pred_ids)
        acc = 100.0 * correct / total if total else 0.0
        nb_acc = (
            100.0 * nonboundary_correct / nonboundary_total
            if nonboundary_total else 0.0
        )

        cycle_rows.append({
            "cycle": cycle_num,
            "video": video_path.name,
            "duration_sec": round(duration, 3),
            "windows": total,
            "correct": correct,
            "wrong": total - correct,
            "accuracy_percent": round(acc, 2),
            "boundary_windows": boundary_total,
            "non_boundary_windows": nonboundary_total,
            "non_boundary_correct": nonboundary_correct,
            "non_boundary_accuracy_percent": round(nb_acc, 2),
        })

        print(
            f"  duration={duration:.1f}s | windows={total} | "
            f"accuracy={acc:.2f}% | non-boundary={nb_acc:.2f}%"
        )

    if not all_rows:
        raise RuntimeError("No inference results were produced.")

    results_df = pd.DataFrame(all_rows)
    cycles_df = pd.DataFrame(cycle_rows)

    total = len(results_df)
    correct = int(results_df["correct"].sum())
    overall_acc = 100.0 * correct / total if total else 0.0

    nb = results_df[~results_df["boundary_window"]]
    nb_correct = int(nb["correct"].sum())
    nb_acc = 100.0 * nb_correct / len(nb) if len(nb) else 0.0

    avg_conf = float(results_df["confidence"].mean())

    summary_df = pd.DataFrame([
        ["Window length (sec)", args.window_len],
        ["Stride (sec)", args.stride],
        ["Sampling FPS", args.fps],
        ["Frames per window", args.num_frames],
        ["ROI", ",".join(map(str, roi))],
        ["Cycles evaluated", len(cycles_df)],
        ["Total windows", total],
        ["Correct windows", correct],
        ["Wrong windows", total - correct],
        ["Overall accuracy (%)", round(overall_acc, 2)],
        ["Boundary windows", int(results_df["boundary_window"].sum())],
        ["Non-boundary windows", len(nb)],
        ["Non-boundary correct", nb_correct],
        ["Non-boundary accuracy (%)", round(nb_acc, 2)],
        ["Average confidence (%)", round(avg_conf * 100.0, 2)],
    ], columns=["Metric", "Value"])

    # Per-class statistics based on GT class.
    per_class = []
    for gt_label, g in results_df.groupby("ground_truth_canonical", dropna=False):
        n = len(g)
        c = int(g["correct"].sum())
        per_class.append({
            "ground_truth": display_class_name(gt_label),
            "windows": n,
            "correct": c,
            "wrong": n - c,
            "accuracy_percent": round(100.0 * c / n, 2) if n else 0.0,
            "avg_confidence_percent": round(
                100.0 * g["confidence"].mean(), 2
            ),
        })

    per_class_df = pd.DataFrame(per_class).sort_values(
        "ground_truth"
    )

    # Confusion matrix.
    confusion = pd.crosstab(
        results_df["ground_truth"],
        results_df["prediction"],
        margins=True,
        margins_name="TOTAL",
    )

    # Boundary-only report.
    boundary_df = results_df[results_df["boundary_window"]].copy()

    # Save everything into one Excel workbook.
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        results_df.to_excel(writer, sheet_name="Window Results", index=False)
        cycles_df.to_excel(writer, sheet_name="Cycle Summary", index=False)
        summary_df.to_excel(writer, sheet_name="Summary", index=False)
        per_class_df.to_excel(writer, sheet_name="Per Class Summary", index=False)
        confusion.to_excel(writer, sheet_name="Confusion Matrix")
        boundary_df.to_excel(
            writer, sheet_name="Boundary Windows", index=False
        )

    print()
    print("=" * 70)
    print("FINAL RESULT")
    print("=" * 70)
    print(f"Overall accuracy       : {overall_acc:.2f}%")
    print(f"Non-boundary accuracy  : {nb_acc:.2f}%")
    print(f"Total windows          : {total}")
    print(f"Correct                : {correct}")
    print(f"Wrong                  : {total - correct}")
    print(f"Boundary windows       : {int(results_df['boundary_window'].sum())}")
    print(f"Average confidence     : {avg_conf * 100.0:.2f}%")
    print(f"Report saved            : {output_path}")


if __name__ == "__main__":
    main()

