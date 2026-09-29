"""
Given a continuous video, reports which time range matched each class.

This runs your fine-tuned VideoMAE classifier (Phase 1) over sliding
windows (Phase 2), then smooths the raw per-window predictions directly
into clean segments -- WITHOUT needing a trained TCN segmentation head
(segmentation_head.py / train_segmentation.py), since that requires
boundary-annotated continuous sessions you don't have yet.

This is a reasonable, useful stand-in until you do have enough annotated
sessions to train the TCN: it uses the same smoothing logic
(segment_utils.labels_to_segments + apply_min_duration_filter) that Phase
4 output goes through, just applied directly to the raw classifier
predictions instead of to a TCN's refined predictions. Expect it to be a
bit noisier at action boundaries than the full Phase 4 pipeline would be,
since there's no learned temporal smoothing -- only the rule-based
min-duration filter.

USAGE

Single video:
    python detect_segments.py \
        --videomae_dir runs/20260925_130237/videomae_finetuned/best_model \
        --video caa_start_2steps_1.mkv \
        --window_len 2.0 --stride 1.0 --fps 15 --num_frames 16 \
        --min_segment_duration 1.0

Every video in a folder:
    python detect_segments.py \
        --videomae_dir runs/20260925_130237/videomae_finetuned/best_model \
        --video_dir videos/ \
        --window_len 2.0 --stride 1.0 --fps 15 --num_frames 16
"""

import argparse
import json
import os
from datetime import datetime

import numpy as np
import torch
from transformers import VideoMAEForVideoClassification, VideoMAEImageProcessor

from extract_embeddings import extract_session_embeddings, resolve_roi
from segment_utils import apply_min_duration_filter, labels_to_segments
from pipeline_runner import compute_model_fingerprint, VIDEO_EXTENSIONS


def detect_segments_for_video(video_path, model, processor, class_names,
                               window_len, stride, fps, num_frames,
                               min_segment_duration, confidence_threshold=None, roi=None):
    window_times, embeddings, probs, timing = extract_session_embeddings(
        video_path, model, processor,
        window_len=window_len, stride=stride, fps=fps, num_frames=num_frames, roi=roi,
    )
    window_starts = [w[0] for w in window_times]
    window_ends = [w[1] for w in window_times]

    pred_idx = probs.argmax(axis=1)
    max_conf = probs.max(axis=1)
    pred_labels = [class_names[i] for i in pred_idx]

    if confidence_threshold is not None:
        # A closed-set classifier always picks its most-plausible known label,
        # even for content that matches none of them (e.g. a dummy/unrelated
        # video) -- there is no built-in "none of the above". This is a
        # stopgap for that: low-confidence windows get relabeled "uncertain"
        # rather than trusting a forced top-1 guess. The durable fix is
        # training an explicit background/no_action class on exactly this
        # kind of negative footage -- this flag only reduces the symptom.
        pred_labels = [
            lbl if conf >= confidence_threshold else "uncertain"
            for lbl, conf in zip(pred_labels, max_conf)
        ]

    raw_segments = labels_to_segments(window_starts, window_ends, pred_labels)
    segments = apply_min_duration_filter(raw_segments, min_segment_duration)

    return segments, raw_segments, timing


def format_segments(segments):
    lines = []
    for seg in segments:
        duration = seg["end_sec"] - seg["start_sec"]
        lines.append(f"  {seg['start_sec']:7.2f}s -> {seg['end_sec']:7.2f}s "
                      f"({duration:5.2f}s)  {seg['label']}")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--videomae_dir", type=str, required=True)
    parser.add_argument("--video", action="append", default=None,
                         help="Repeatable. A single video path.")
    parser.add_argument("--video_dir", type=str, default=None,
                         help="Process every video file in this folder.")
    parser.add_argument("--window_len", type=float, default=2.0)
    parser.add_argument("--stride", type=float, default=1.0)
    parser.add_argument("--fps", type=float, default=15.0)
    parser.add_argument("--num_frames", type=int, default=16)
    parser.add_argument("--min_segment_duration", type=float, default=2.5,
                         help="Segments shorter than this (seconds) get merged into "
                              "whichever neighboring segment is longer. Must be set "
                              "GREATER than --window_len to have any effect: even a "
                              "single flickered window produces a segment spanning the "
                              "full window_len, so a threshold <= window_len will never "
                              "filter anything.")
    parser.add_argument("--confidence_threshold", type=float, default=None,
                         help="Stopgap for a closed-set classifier's lack of a 'none of "
                              "the above' option: windows whose top prediction is below "
                              "this probability (0-1) are labeled 'uncertain' instead of "
                              "a forced guess. Useful against false positives on footage "
                              "that matches none of your trained classes, but the durable "
                              "fix is training an explicit background/no_action class on "
                              "that kind of footage. Off by default.")
    parser.add_argument("--roi", type=str, default=None,
                         help="Override the ROI, as 'x,y,w,h'. If omitted, auto-loads "
                              "roi.json from --videomae_dir if the model was trained "
                              "with one.")
    parser.add_argument("--runs_dir", type=str, default="runs")
    args = parser.parse_args()

    if not args.video and not args.video_dir:
        parser.error("Provide --video and/or --video_dir")

    video_paths = list(args.video) if args.video else []
    if args.video_dir:
        for fname in sorted(os.listdir(args.video_dir)):
            if fname.lower().endswith(VIDEO_EXTENSIONS):
                video_paths.append(os.path.join(args.video_dir, fname))
    video_paths = sorted(set(os.path.abspath(v) for v in video_paths))

    if not video_paths:
        raise RuntimeError("No videos found -- check --video / --video_dir.")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(args.runs_dir, timestamp + "_segments")
    os.makedirs(run_dir, exist_ok=True)

    fingerprint = compute_model_fingerprint(args.videomae_dir)
    print(f"Model dir: {args.videomae_dir}")
    print(f"Model fingerprint: {fingerprint}\n")

    processor = VideoMAEImageProcessor.from_pretrained(args.videomae_dir)
    model = VideoMAEForVideoClassification.from_pretrained(args.videomae_dir)
    class_names = [model.config.id2label[i] for i in range(len(model.config.id2label))]
    print(f"Classes: {class_names}\n")

    roi = resolve_roi(args.videomae_dir, args.roi)

    manifest = {
        "run_dir": run_dir,
        "videomae_dir": os.path.abspath(args.videomae_dir),
        "model_fingerprint": fingerprint,
        "class_names": class_names,
        "window_len": args.window_len,
        "stride": args.stride,
        "fps": args.fps,
        "num_frames": args.num_frames,
        "min_segment_duration": args.min_segment_duration,
        "confidence_threshold": args.confidence_threshold,
        "roi": list(roi) if roi else None,
        "videos": [],
    }

    for video_path in video_paths:
        stem = os.path.splitext(os.path.basename(video_path))[0]
        print(f"[{video_path}]")

        segments, raw_segments, timing = detect_segments_for_video(
            video_path, model, processor, class_names,
            args.window_len, args.stride, args.fps, args.num_frames,
            args.min_segment_duration, args.confidence_threshold, roi,
        )

        print(format_segments(segments))
        print(f"  ({timing['device']} | {timing['per_window_sec_incl_decode']*1000:.1f} ms/window)\n")

        out_path = os.path.join(run_dir, stem + "_segments.json")
        result = {
            "video_path": video_path,
            "model_fingerprint": fingerprint,
            "segments": segments,
            "raw_segments_before_smoothing": raw_segments,
        }
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2)

        manifest["videos"].append({
            "video_path": video_path,
            "segments_json": os.path.abspath(out_path),
            "num_segments": len(segments),
        })

    manifest_path = os.path.join(run_dir, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Full results: {run_dir}")


if __name__ == "__main__":
    main()
