"""
End-to-end pipeline for a NEW continuous operator video: runs Phase 2
(sliding-window embedding extraction) -> Phase 4 (segmentation head
inference) -> smoothing -> Phase 5 (sequence verification) -> report.

This is what you'd run in "production": given one operator's session
video, get back a structured pass/fail report on step order and
idle/fumble anomalies.

Usage:
    python run_pipeline.py \
        --video_path new_sessions/operator_07_session.mp4 \
        --videomae_dir ./videomae_finetuned/best_model \
        --segmentation_head segmentation_head.pt \
        --procedure_json example_canonical_procedure.json \
        --window_len 2.0 --stride 1.0 --fps 15 \
        --min_segment_duration 1.0 --long_idle_threshold_sec 15.0 \
        --report_out report_operator_07.json
"""

import argparse
import json

import torch
from transformers import VideoMAEForVideoClassification, VideoMAEImageProcessor

from extract_embeddings import extract_session_embeddings
from segment_utils import apply_min_duration_filter, labels_to_segments
from segmentation_head import TCNSegmentationHead, build_features, predict_sequence
from sequence_verification import load_canonical_procedure, verify_sequence


def load_segmentation_head(checkpoint_path):
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    model = TCNSegmentationHead(
        input_dim=ckpt["input_dim"],
        num_classes=len(ckpt["class_names"]),
        hidden_dim=ckpt["hidden_dim"],
        num_levels=ckpt["num_levels"],
        kernel_size=ckpt["kernel_size"],
    )
    model.load_state_dict(ckpt["state_dict"])
    return model, ckpt["class_names"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_path", type=str, required=True)
    parser.add_argument("--videomae_dir", type=str, required=True,
                         help="Phase 1 fine-tuned model dir")
    parser.add_argument("--segmentation_head", type=str, required=True,
                         help="Phase 4 checkpoint (.pt) from train_segmentation.py")
    parser.add_argument("--procedure_json", type=str, required=True)
    parser.add_argument("--window_len", type=float, default=2.0)
    parser.add_argument("--stride", type=float, default=1.0)
    parser.add_argument("--fps", type=float, default=15.0)
    parser.add_argument("--num_frames", type=int, default=16)
    parser.add_argument("--min_segment_duration", type=float, default=1.0)
    parser.add_argument("--long_idle_threshold_sec", type=float, default=15.0)
    parser.add_argument("--idle_labels", type=str, nargs="+", default=["idle_fumble"])
    parser.add_argument("--report_out", type=str, default=None)
    args = parser.parse_args()

    print("Loading models...")
    processor = VideoMAEImageProcessor.from_pretrained(args.videomae_dir)
    videomae_model = VideoMAEForVideoClassification.from_pretrained(args.videomae_dir)
    seg_head, class_names = load_segmentation_head(args.segmentation_head)
    idx_to_label = {i: name for i, name in enumerate(class_names)}

    print("Phase 2: extracting sliding-window embeddings...")
    window_times, embeddings, probs, timing = extract_session_embeddings(
        args.video_path, videomae_model, processor,
        window_len=args.window_len, stride=args.stride,
        fps=args.fps, num_frames=args.num_frames,
    )
    print(f"  {timing['device']} | model-only: {timing['per_window_sec_model_only']*1000:.1f} ms/window "
          f"| incl. decode: {timing['per_window_sec_incl_decode']*1000:.1f} ms/window")
    window_starts = [w[0] for w in window_times]
    window_ends = [w[1] for w in window_times]

    print("Phase 4: running segmentation head...")
    features = build_features(embeddings, probs)
    pred_idx, _ = predict_sequence(seg_head, features)
    pred_labels = [idx_to_label[i] for i in pred_idx]

    print("Smoothing predicted timeline...")
    raw_segments = labels_to_segments(window_starts, window_ends, pred_labels)
    segments = apply_min_duration_filter(raw_segments, args.min_segment_duration)

    print("Phase 5: verifying against canonical procedure...")
    canonical_steps = load_canonical_procedure(args.procedure_json)
    report = verify_sequence(
        segments, canonical_steps,
        idle_labels=tuple(args.idle_labels),
        long_idle_threshold_sec=args.long_idle_threshold_sec,
    )
    report["predicted_segments"] = segments  # include the raw timeline for review

    print(json.dumps(report, indent=2))
    if args.report_out:
        with open(args.report_out, "w") as f:
            json.dump(report, f, indent=2)
        print(f"\nReport saved to {args.report_out}")


if __name__ == "__main__":
    main()