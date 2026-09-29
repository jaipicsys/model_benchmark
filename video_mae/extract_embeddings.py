"""
Phase 2 -- Sliding-window inference over continuous (untrimmed) video.

Runs the Phase 1 fine-tuned VideoMAE model over overlapping windows of a
continuous session and saves, per window:
    - (start_sec, end_sec)
    - a pooled embedding vector (mean-pooled last_hidden_state)
    - the classifier's softmax probabilities over your action classes

These per-window embeddings+probs are the INPUT to the Phase 4 temporal
segmentation head (segmentation_head.py) -- both at training time (aligned
against your Phase 3 boundary annotations) and at inference time.

Also reports inference timing (total + per-window average), printed to
stdout and included in the saved .npz, so you have real measured numbers
for your deployment hardware rather than estimates.

Usage (single video):
    python extract_embeddings.py \
        --model_dir ./videomae_finetuned/best_model \
        --video_path sessions/session_001.mp4 \
        --out_path embeddings/session_001.npz \
        --window_len 2.0 --stride 1.0 --fps 15 --num_frames 16

Usage (batch over a folder of videos):
    python extract_embeddings.py \
        --model_dir ./videomae_finetuned/best_model \
        --video_dir sessions/ \
        --out_dir embeddings/ \
        --window_len 2.0 --stride 1.0 --fps 15 --num_frames 16

ROI: if the model was trained with --roi (see finetune_videomae.py), this
script auto-loads the matching roi.json from --model_dir and applies the
same crop -- no need to pass --roi manually unless overriding it.
"""

import argparse
import json
import os
import time

import numpy as np
import torch
from decord import VideoReader, cpu
from transformers import VideoMAEForVideoClassification, VideoMAEImageProcessor


def resolve_roi(model_dir, cli_roi_str=None):
    """
    Determines which ROI to use: an explicit --roi on the command line always
    wins; otherwise, auto-load roi.json saved alongside the model by
    finetune_videomae.py (so inference automatically matches whatever crop,
    if any, the model was actually trained with -- no need to retype
    coordinates and risk a train/inference mismatch). Returns None if
    neither is set (use the full frame).
    """
    if cli_roi_str:
        try:
            roi = tuple(int(v) for v in cli_roi_str.split(","))
            assert len(roi) == 4
            return roi
        except (ValueError, AssertionError):
            raise ValueError(f"--roi must be 'x,y,w,h' (four integers), got: {cli_roi_str}")

    roi_path = os.path.join(model_dir, "roi.json")
    if os.path.exists(roi_path):
        with open(roi_path) as f:
            data = json.load(f)
        if data.get("roi"):
            roi = tuple(data["roi"])
            print(f"Auto-loaded ROI from {roi_path}: {roi}")
            return roi
    return None


def _sample_frames(vr, start_f, end_f, num_frames, roi=None):
    end_f = max(end_f, start_f + 1)
    indices = np.linspace(start_f, end_f - 1, num=num_frames)
    indices = np.clip(np.round(indices).astype(int), 0, len(vr) - 1)
    frames = vr.get_batch(indices).asnumpy()  # (T, H, W, C)
    if roi is not None:
        x, y, w, h = roi
        frames = frames[:, y:y + h, x:x + w, :]
    return frames


@torch.no_grad()
def extract_session_embeddings(
    video_path,
    model,
    processor,
    window_len=2.0,
    stride=1.0,
    fps=15,
    num_frames=16,
    device="cuda" if torch.cuda.is_available() else "cpu",
    roi=None,
):
    """
    Returns:
        window_times: list of (start_sec, end_sec)
        embeddings:   np.ndarray (N, hidden_size)
        probs:        np.ndarray (N, num_classes)
        timing:       dict with total_sec, per_window_sec (model-only time,
                       excludes video decode/frame sampling), and
                       per_window_sec_incl_decode (wall time per window
                       including decode -- closer to real deployment
                       throughput if you decode frames fresh each window)

    roi: optional (x, y, w, h) fixed crop in native pixel coordinates,
         applied to every sampled frame before the processor resizes it.
         MUST match whatever ROI (if any) the model was trained with --
         extract_embeddings.py's CLI auto-loads this from the model dir's
         roi.json when present, so you normally don't need to pass it
         manually here unless calling this function directly.
    """
    vr = VideoReader(video_path, ctx=cpu(0))
    total_frames = len(vr)
    native_fps = vr.get_avg_fps() or fps
    duration_sec = total_frames / native_fps

    window_times = []
    t = 0.0
    while t + window_len <= duration_sec + 1e-6:
        window_times.append((t, t + window_len))
        t += stride
    if not window_times:
        # video shorter than one window: use the whole thing as a single window
        window_times = [(0.0, duration_sec)]

    embeddings, probs_all = [], []
    model.eval().to(device)

    if device.startswith("cuda"):
        torch.cuda.synchronize()
    wall_start = time.perf_counter()
    model_only_time = 0.0

    for start_sec, end_sec in window_times:
        start_f = int(start_sec * native_fps)
        end_f = int(end_sec * native_fps)
        frames = _sample_frames(vr, start_f, end_f, num_frames, roi=roi)

        inputs = processor(list(frames), return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(device)

        if device.startswith("cuda"):
            torch.cuda.synchronize()
        model_start = time.perf_counter()

        # pooled embedding from the backbone (mean over patch tokens)
        backbone_out = model.videomae(pixel_values=pixel_values)
        pooled = backbone_out.last_hidden_state.mean(dim=1)  # (1, hidden)

        # classification probs from the full model (shares the same forward pass
        # conceptually; cheap to also call the classifier head directly)
        logits = model.classifier(model.fc_norm(pooled)) if hasattr(model, "fc_norm") \
            else model(pixel_values=pixel_values).logits
        p = torch.softmax(logits, dim=-1)

        if device.startswith("cuda"):
            torch.cuda.synchronize()
        model_only_time += time.perf_counter() - model_start

        embeddings.append(pooled.squeeze(0).cpu().numpy())
        probs_all.append(p.squeeze(0).cpu().numpy())

    if device.startswith("cuda"):
        torch.cuda.synchronize()
    wall_total = time.perf_counter() - wall_start

    n = len(window_times)
    timing = {
        "device": device,
        "num_windows": n,
        "total_sec_incl_decode": wall_total,
        "total_sec_model_only": model_only_time,
        "per_window_sec_incl_decode": wall_total / n,
        "per_window_sec_model_only": model_only_time / n,
    }

    return window_times, np.stack(embeddings), np.stack(probs_all), timing


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, required=True,
                         help="Phase 1 fine-tuned model dir (contains config.json etc.)")
    parser.add_argument("--video_path", type=str, default=None)
    parser.add_argument("--video_dir", type=str, default=None)
    parser.add_argument("--out_path", type=str, default=None)
    parser.add_argument("--out_dir", type=str, default=None)
    parser.add_argument("--window_len", type=float, default=2.0, help="seconds")
    parser.add_argument("--stride", type=float, default=1.0, help="seconds")
    parser.add_argument("--fps", type=float, default=15.0,
                         help="fallback fps if video metadata is unreliable")
    parser.add_argument("--num_frames", type=int, default=16)
    parser.add_argument("--roi", type=str, default=None,
                         help="Override the ROI, as 'x,y,w,h'. If omitted, auto-loads "
                              "roi.json from --model_dir if the model was trained with "
                              "one (recommended -- keeps train/inference consistent).")
    args = parser.parse_args()

    processor = VideoMAEImageProcessor.from_pretrained(args.model_dir)
    model = VideoMAEForVideoClassification.from_pretrained(args.model_dir)
    class_names = [model.config.id2label[i] for i in range(len(model.config.id2label))]
    roi = resolve_roi(args.model_dir, args.roi)

    def process_one(video_path, out_path):
        window_times, embeddings, probs, timing = extract_session_embeddings(
            video_path, model, processor,
            window_len=args.window_len, stride=args.stride,
            fps=args.fps, num_frames=args.num_frames, roi=roi,
        )
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        np.savez(
            out_path,
            window_starts=np.array([w[0] for w in window_times]),
            window_ends=np.array([w[1] for w in window_times]),
            embeddings=embeddings,
            probs=probs,
            class_names=np.array(class_names),
            timing=np.array([str(timing)]),  # stored as string for portability
            roi=np.array([str(roi)]),
        )
        print(f"[{video_path}] -> {out_path}  ({len(window_times)} windows)")
        print(f"  device: {timing['device']}")
        print(f"  model-only:      {timing['total_sec_model_only']:.3f}s total, "
              f"{timing['per_window_sec_model_only']*1000:.1f} ms/window")
        print(f"  incl. decode:    {timing['total_sec_incl_decode']:.3f}s total, "
              f"{timing['per_window_sec_incl_decode']*1000:.1f} ms/window")

    if args.video_path:
        assert args.out_path, "--out_path required with --video_path"
        process_one(args.video_path, args.out_path)
    elif args.video_dir:
        assert args.out_dir, "--out_dir required with --video_dir"
        for fname in sorted(os.listdir(args.video_dir)):
            if fname.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
                session_id = os.path.splitext(fname)[0]
                process_one(
                    os.path.join(args.video_dir, fname),
                    os.path.join(args.out_dir, session_id + ".npz"),
                )
    else:
        raise ValueError("Provide either --video_path or --video_dir")


if __name__ == "__main__":
    main()
