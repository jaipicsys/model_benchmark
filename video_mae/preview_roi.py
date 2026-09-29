"""
Preview a region of interest (ROI) on an actual video frame, before
committing to it for training. Saves two images:
    1. the full frame with the ROI box drawn on it
    2. just the cropped region -- exactly what the model will actually see

USAGE

Check a specific box against a frame:
    python preview_roi.py \
        --video videos/caa_start_2steps_1.mkv \
        --roi 280,0,440,690

Check whatever ROI a trained model was saved with, against a new video
(useful to confirm the same box still makes sense for different footage,
e.g. a different camera or slightly different framing than what it was
trained on):
    python preview_roi.py \
        --video videos/op1_startonly_1.mp4 \
        --videomae_dir runs/20260925_160109/videomae_finetuned/best_model

Check a different point in time in the video (default is the first frame):
    python preview_roi.py --video videos/x.mkv --roi 280,0,440,690 --frame_time 5.0
"""

import argparse
import os

import numpy as np
from decord import VideoReader, cpu
from PIL import Image, ImageDraw

from extract_embeddings import resolve_roi


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--video", type=str, required=True)
    parser.add_argument("--roi", type=str, default=None,
                         help="'x,y,w,h' in native pixel coordinates. Either this or "
                              "--videomae_dir (to auto-load a trained model's roi.json) "
                              "is required.")
    parser.add_argument("--videomae_dir", type=str, default=None,
                         help="Auto-load the ROI saved alongside this trained model, "
                              "instead of specifying --roi directly.")
    parser.add_argument("--frame_time", type=float, default=0.0,
                         help="Timestamp (seconds) of the frame to preview. Default: "
                              "the very first frame.")
    parser.add_argument("--out_prefix", type=str, default="roi_preview",
                         help="Output files are '<out_prefix>_box.jpg' (full frame with "
                              "the ROI box drawn on it) and '<out_prefix>_cropped.jpg' "
                              "(just the cropped region).")
    args = parser.parse_args()

    if not args.roi and not args.videomae_dir:
        parser.error("Provide --roi directly, or --videomae_dir to auto-load one")

    roi = resolve_roi(args.videomae_dir or "", args.roi)
    if roi is None:
        parser.error("No ROI resolved -- check --roi format or that --videomae_dir "
                      "actually has a roi.json with a non-null roi")

    vr = VideoReader(args.video, ctx=cpu(0))
    native_fps = vr.get_avg_fps() or 15
    frame_idx = int(args.frame_time * native_fps)
    frame_idx = max(0, min(frame_idx, len(vr) - 1))
    frame = vr[frame_idx].asnumpy()  # (H, W, C)

    frame_h, frame_w = frame.shape[:2]
    x, y, w, h = roi
    if x < 0 or y < 0 or x + w > frame_w or y + h > frame_h:
        print(f"WARNING: ROI {roi} extends outside this frame's bounds "
              f"({frame_w}x{frame_h}) -- it will be clipped. Double-check this ROI "
              f"actually matches this video's framing/resolution.")

    img = Image.fromarray(frame).convert("RGB")
    draw = ImageDraw.Draw(img)
    draw.rectangle([x, y, x + w, y + h], outline=(255, 0, 0), width=6)
    draw.text((x + 10, max(y + 10, 10)), f"ROI: x={x} y={y} w={w} h={h}", fill=(255, 0, 0))

    box_path = f"{args.out_prefix}_box.jpg"
    img.save(box_path)

    cropped = frame[max(y, 0):y + h, max(x, 0):x + w, :]
    cropped_path = f"{args.out_prefix}_cropped.jpg"
    Image.fromarray(cropped).convert("RGB").save(cropped_path)

    print(f"Video native resolution: {frame_w}x{frame_h}")
    print(f"ROI: {roi}")
    print(f"Frame previewed: t={args.frame_time}s (frame index {frame_idx})")
    print(f"Saved: {box_path}  (full frame with ROI box)")
    print(f"Saved: {cropped_path}  (cropped region only -- what the model actually sees)")


if __name__ == "__main__":
    main()
