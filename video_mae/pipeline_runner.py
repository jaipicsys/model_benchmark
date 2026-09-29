"""
End-to-end orchestrator: fine-tune (optional) -> extract embeddings -> eval,
run as one traceable unit so a stale .npz can never silently get compared
against the wrong model checkpoint.

WHAT THIS FIXES, specifically:
    1. finetune_videomae.py always writes to the same --output_dir, so a
       retrain silently overwrites the previous checkpoint. This script
       gives every fine-tuning run its own timestamped output directory,
       so old checkpoints are never clobbered.
    2. extract_embeddings.py's .npz files carry no record of which model
       checkpoint produced them, so a stale .npz can get evaluated against
       a *different*, newer model without any warning (this is exactly
       what happened with new_video.npz vs op2.npz earlier). This script
       computes a fingerprint of the model checkpoint (based on the
       weights file's size + modification time) and stamps it into every
       .npz it writes -- eval_on_video.py's evaluate_embeddings() already
       reads and reports this fingerprint if present.
    3. Every run's embeddings + eval reports land in their own timestamped
       folder under --runs_dir, alongside a manifest.json summarizing
       exactly what model and what videos were used. Nothing from a
       previous run is ever overwritten or reused by accident.

USAGE

Case A -- process every video in a folder (annotations auto-matched by
filename stem, e.g. op2.mkv -> op2_ground_truth.json or op2.json in the
same folder, or in --annotations_dir if given separately). Videos with no
matching annotation file just get predictions saved, no accuracy report:

    python pipeline_runner.py \
        --videomae_dir ./videomae_finetuned/best_model \
        --video_dir videos/ \
        --window_len 2.0 --stride 1.0 --fps 15 --num_frames 16

Case B -- explicit list of videos (still supported, and combinable with
--video_dir -- both lists are merged):

    python pipeline_runner.py \
        --videomae_dir ./videomae_finetuned/best_model \
        --video op2.mkv op2_ground_truth.json \
        --video pr.mkv pr_ground_truth.json \
        --video rg.mkv rg_ground_truth.json \
        --video si.mkv si_ground_truth.json \
        --window_len 2.0 --stride 1.0 --fps 15 --num_frames 16

Case C -- fine-tune first, then automatically extract+eval every video in
a folder with the freshly trained model (guarantees you can never
accidentally evaluate against an old checkpoint):

    python pipeline_runner.py \
        --do_finetune \
        --data_dir data_dir \
        --model_ckpt MCG-NJU/videomae-base-finetuned-kinetics \
        --num_frames 16 --epochs 30 --batch_size 4 --lr 5e-5 \
        --freeze_backbone_layers 8 \
        --video_dir videos/ \
        --window_len 2.0 --stride 1.0 --fps 15
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime

import numpy as np
import torch
from transformers import VideoMAEForVideoClassification, VideoMAEImageProcessor

from extract_embeddings import extract_session_embeddings, resolve_roi
from eval_on_video import evaluate_embeddings, format_report

VIDEO_EXTENSIONS = (".mp4", ".avi", ".mov", ".mkv")


def find_annotation_for_video(video_path, annotations_dir):
    """
    Looks for a ground-truth annotation matching a video's filename stem,
    trying (in order): "<stem>_ground_truth.json", then "<stem>.json",
    inside annotations_dir. Returns the path if found, else None (the
    video is then processed as predict-only, no accuracy report).
    """
    stem = os.path.splitext(os.path.basename(video_path))[0]
    for candidate_name in (stem + "_ground_truth.json", stem + ".json"):
        candidate_path = os.path.join(annotations_dir, candidate_name)
        if os.path.exists(candidate_path):
            return candidate_path
    return None


def build_video_list(args):
    """
    Merges --video_dir (every video file in the folder, annotation
    auto-matched by filename stem) with any explicit --video entries.
    Returns a list of (video_path, annotation_path_or_None), de-duplicated
    by absolute video path (explicit --video entries take precedence over
    an auto-match from --video_dir for the same file).
    """
    videos = {}  # abs video_path -> annotation_path_or_None

    if args.video_dir:
        annotations_dir = args.annotations_dir or args.video_dir
        for fname in sorted(os.listdir(args.video_dir)):
            if fname.lower().endswith(VIDEO_EXTENSIONS):
                video_path = os.path.join(args.video_dir, fname)
                annotation_path = find_annotation_for_video(video_path, annotations_dir)
                videos[os.path.abspath(video_path)] = annotation_path

    if args.video:
        for v in args.video:
            video_path = v[0]
            annotation_path = v[1] if len(v) == 2 else None
            videos[os.path.abspath(video_path)] = annotation_path  # explicit entry wins

    return list(videos.items())


def compute_model_fingerprint(model_dir):
    """
    Cheap fingerprint of a model checkpoint: hashes the weights file's size
    + modification time (not its full contents -- hashing a 300MB+ file on
    every run would be slow, and size+mtime already changes on every save
    Trainer/save_model performs, which is all we need to detect "this is a
    different checkpoint than before").
    """
    weights_path = None
    for candidate in ("model.safetensors", "pytorch_model.bin"):
        p = os.path.join(model_dir, candidate)
        if os.path.exists(p):
            weights_path = p
            break
    if weights_path is None:
        raise FileNotFoundError(f"No model weights file found under {model_dir}")

    stat = os.stat(weights_path)
    signature = f"{os.path.abspath(model_dir)}|{stat.st_size}|{stat.st_mtime}"
    return hashlib.sha1(signature.encode()).hexdigest()[:12]


def run_finetune(args, run_dir):
    """Shells out to finetune_videomae.py with a fresh, run-specific output_dir
    so it can never overwrite a previous run's checkpoint."""
    finetune_output_dir = os.path.join(run_dir, "videomae_finetuned")
    cmd = [
        sys.executable, "finetune_videomae.py",
        "--data_dir", args.data_dir,
        "--output_dir", finetune_output_dir,
        "--model_ckpt", args.model_ckpt,
        "--num_frames", str(args.num_frames),
        "--epochs", str(args.epochs),
        "--batch_size", str(args.batch_size),
        "--lr", str(args.lr),
        "--freeze_backbone_layers", str(args.freeze_backbone_layers),
        "--early_stopping_patience", str(args.early_stopping_patience),
        "--early_stopping_threshold", str(args.early_stopping_threshold),
        "--color_jitter_strength", str(args.color_jitter_strength),
    ]
    if args.roi:
        cmd += ["--roi", args.roi]
    print(f"\n=== Phase 1: fine-tuning ===\n$ {' '.join(cmd)}\n")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        raise RuntimeError("finetune_videomae.py failed -- see output above. "
                            "Aborting before extract/eval to avoid using a bad checkpoint.")

    best_model_dir = os.path.join(finetune_output_dir, "best_model")
    if not os.path.isdir(best_model_dir):
        raise RuntimeError(f"Expected {best_model_dir} to exist after training but it doesn't.")

    if args.cleanup_checkpoints:
        # Intermediate checkpoint-N folders (optimizer/scheduler state, periodic
        # weight snapshots) are only useful for resuming an interrupted run --
        # once training has finished successfully and best_model/ is saved,
        # they're just disk bloat. Each one can be 300-600MB+, and left
        # unchecked across many runs this is exactly what fills the disk and
        # crashes the NEXT training run's checkpoint write (as happened here
        # twice already) -- not a code bug, just accumulated leftover files.
        import glob
        import shutil
        removed_bytes = 0
        for ckpt_dir in glob.glob(os.path.join(finetune_output_dir, "checkpoint-*")):
            removed_bytes += sum(
                os.path.getsize(os.path.join(dp, f))
                for dp, _, files in os.walk(ckpt_dir) for f in files
            )
            shutil.rmtree(ckpt_dir)
        if removed_bytes:
            print(f"Cleaned up intermediate checkpoints: freed {removed_bytes / 1e9:.2f} GB")

    return best_model_dir


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument("--do_finetune", action="store_true",
                         help="Run Phase 1 fine-tuning first, using a fresh output dir. "
                              "If not set, --videomae_dir must point to an existing model.")
    parser.add_argument("--videomae_dir", type=str, default=None,
                         help="Existing fine-tuned model dir. Ignored if --do_finetune is set "
                              "(the freshly trained model is used instead).")

    # finetune passthrough args (only used with --do_finetune)
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--model_ckpt", type=str,
                         default="MCG-NJU/videomae-base-finetuned-kinetics")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--freeze_backbone_layers", type=int, default=8)
    parser.add_argument("--early_stopping_patience", type=int, default=5,
                         help="Passthrough to finetune_videomae.py. 0 disables early "
                              "stopping and always runs the full --epochs.")
    parser.add_argument("--early_stopping_threshold", type=float, default=0.0,
                         help="Passthrough to finetune_videomae.py. Minimum eval_loss "
                              "decrease to count as improvement -- 0.0 means ANY "
                              "decrease resets patience, which in practice can prevent "
                              "early stopping from ever triggering on a slowly-improving "
                              "loss curve. Consider e.g. 0.001.")
    parser.add_argument("--color_jitter_strength", type=float, default=0.2,
                         help="Passthrough to finetune_videomae.py.")

    # shared extract_embeddings args
    parser.add_argument("--num_frames", type=int, default=16)
    parser.add_argument("--window_len", type=float, default=2.0)
    parser.add_argument("--stride", type=float, default=1.0)
    parser.add_argument("--fps", type=float, default=15.0)
    parser.add_argument("--roi", type=str, default=None,
                         help="'x,y,w,h' fixed crop. With --do_finetune, passed through "
                              "to training. Either way, also used (or auto-loaded from "
                              "the model's roi.json if omitted) for extraction/eval, so "
                              "train and inference always use the same crop.")

    # videos to run through extract+eval. Two ways to specify, combinable:
    #   --video_dir folder/               all videos in folder, annotation auto-matched
    #   --video path.mp4                  (predictions only, no accuracy report)
    #   --video path.mp4 ground_truth.json  (predictions + accuracy report)
    parser.add_argument("--video_dir", type=str, default=None,
                         help="Folder of videos to process. Ground-truth annotations are "
                              "auto-matched by filename stem: '<stem>_ground_truth.json' or "
                              "'<stem>.json', looked up in --annotations_dir (or --video_dir "
                              "itself if --annotations_dir is not given). Videos with no "
                              "matching annotation are processed predict-only.")
    parser.add_argument("--annotations_dir", type=str, default=None,
                         help="Folder to look for ground-truth JSONs matched to --video_dir "
                              "videos. Defaults to --video_dir itself.")
    parser.add_argument("--video", action="append", nargs="+",
                         metavar=("VIDEO_PATH", "ANNOTATION_JSON"),
                         help="Repeatable. 1 value = predict only; 2 values = predict + eval. "
                              "Can be combined with --video_dir; explicit --video entries take "
                              "precedence over an auto-matched annotation for the same file.")

    parser.add_argument("--runs_dir", type=str, default="runs")
    parser.add_argument("--run_name", type=str, default=None,
                         help="Name for this run's folder under --runs_dir, instead of "
                              "the default timestamp (e.g. 'roi_v2_30clips'). Makes runs "
                              "easier to find/remember than a bare timestamp. Fails if a "
                              "folder with this name already exists, rather than silently "
                              "overwriting a previous run's results -- pick a different "
                              "name, or delete the old folder first if you really mean to "
                              "replace it.")
    parser.add_argument("--cleanup_checkpoints", action="store_true", default=True,
                         help="After a successful --do_finetune run, delete the "
                              "intermediate checkpoint-N folders (optimizer/scheduler "
                              "state, periodic snapshots) and keep only best_model/. "
                              "These are only needed to resume an interrupted run, and "
                              "left unchecked they're what fills the disk over repeated "
                              "runs. On by default; pass --no-cleanup_checkpoints to "
                              "keep them (e.g. if you want to resume training later).")
    parser.add_argument("--no-cleanup_checkpoints", dest="cleanup_checkpoints",
                         action="store_false")
    args = parser.parse_args()

    if args.do_finetune and not args.data_dir:
        parser.error("--do_finetune requires --data_dir")
    if not args.do_finetune and not args.videomae_dir:
        parser.error("Provide --videomae_dir, or use --do_finetune to train a new model")
    if not args.video_dir and not args.video:
        parser.error("Provide --video_dir and/or one or more --video entries")

    if args.video:
        for v in args.video:
            if len(v) not in (1, 2):
                parser.error(f"--video takes 1 or 2 values, got {len(v)}: {v}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    folder_name = args.run_name if args.run_name else timestamp
    run_dir = os.path.join(args.runs_dir, folder_name)

    if args.run_name and os.path.exists(run_dir):
        raise FileExistsError(
            f"runs/{folder_name} already exists -- refusing to overwrite it. "
            f"Pick a different --run_name, or delete/rename the existing folder "
            f"first if you really mean to replace it."
        )

    os.makedirs(run_dir, exist_ok=True)
    embeddings_dir = os.path.join(run_dir, "embeddings")
    reports_dir = os.path.join(run_dir, "reports")
    os.makedirs(embeddings_dir, exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)
    print(f"Run directory: {run_dir}")

    # --- Phase 1 (optional) ---
    if args.do_finetune:
        videomae_dir = run_finetune(args, run_dir)
    else:
        videomae_dir = args.videomae_dir
        if not os.path.isdir(videomae_dir):
            raise FileNotFoundError(f"--videomae_dir does not exist: {videomae_dir}")

    fingerprint = compute_model_fingerprint(videomae_dir)
    print(f"\nModel dir: {videomae_dir}")
    print(f"Model fingerprint: {fingerprint}  "
          f"(changes any time the checkpoint is retrained/resaved)")

    roi = resolve_roi(videomae_dir, args.roi)

    # --- Phase 2: extract embeddings for every video, tagged with the fingerprint ---
    video_list = build_video_list(args)
    if not video_list:
        raise RuntimeError("No videos found -- check --video_dir contents or --video paths.")

    print(f"\n=== Phase 2: extracting embeddings ({len(video_list)} video(s)) ===")
    n_with_annotation = sum(1 for _, a in video_list if a)
    print(f"({n_with_annotation}/{len(video_list)} have a matched ground-truth annotation)")

    processor = VideoMAEImageProcessor.from_pretrained(videomae_dir)
    model = VideoMAEForVideoClassification.from_pretrained(videomae_dir)
    class_names = [model.config.id2label[i] for i in range(len(model.config.id2label))]

    manifest = {
        "run_dir": run_dir,
        "run_name": folder_name,
        "created_at": timestamp,
        "videomae_dir": os.path.abspath(videomae_dir),
        "model_fingerprint": fingerprint,
        "window_len": args.window_len,
        "stride": args.stride,
        "fps": args.fps,
        "num_frames": args.num_frames,
        "roi": list(roi) if roi else None,
        "class_names": class_names,
        "videos": [],
    }

    for video_path, annotation_path in video_list:
        stem = os.path.splitext(os.path.basename(video_path))[0]
        npz_path = os.path.join(embeddings_dir, stem + ".npz")

        print(f"\n[{video_path}]")
        window_times, embeddings, probs, timing = extract_session_embeddings(
            video_path, model, processor,
            window_len=args.window_len, stride=args.stride,
            fps=args.fps, num_frames=args.num_frames, roi=roi,
        )
        np.savez(
            npz_path,
            window_starts=np.array([w[0] for w in window_times]),
            window_ends=np.array([w[1] for w in window_times]),
            embeddings=embeddings,
            probs=probs,
            roi=np.array([str(roi)]),
            class_names=np.array(class_names),
            model_fingerprint=np.array([fingerprint]),
            videomae_dir=np.array([os.path.abspath(videomae_dir)]),
        )
        print(f"  -> {npz_path}  ({len(window_times)} windows)")
        print(f"  {timing['device']} | model-only: {timing['per_window_sec_model_only']*1000:.1f} ms/window "
              f"| incl. decode: {timing['per_window_sec_incl_decode']*1000:.1f} ms/window")

        video_entry = {
            "video_path": os.path.abspath(video_path),
            "embeddings_npz": os.path.abspath(npz_path),
            "annotation_json": os.path.abspath(annotation_path) if annotation_path else None,
            "num_windows": len(window_times),
            "timing": timing,
        }

        # --- Phase 2 eval (only if a ground-truth annotation was given) ---
        if annotation_path:
            print(f"  Evaluating against {annotation_path} ...")
            try:
                report = evaluate_embeddings(npz_path, annotation_path)
                print("  " + format_report(report).replace("\n", "\n  "))
                report_path = os.path.join(reports_dir, stem + "_eval.json")
                with open(report_path, "w") as f:
                    json.dump(report, f, indent=2)
                video_entry["eval_report"] = report
                video_entry["eval_report_path"] = os.path.abspath(report_path)
            except ValueError as e:
                print(f"  Skipped eval: {e}")
                video_entry["eval_error"] = str(e)

        manifest["videos"].append(video_entry)

    manifest_path = os.path.join(run_dir, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    # --- summary ---
    print(f"\n=== Summary ===")
    print(f"Model fingerprint: {fingerprint}")
    for v in manifest["videos"]:
        name = os.path.basename(v["video_path"])
        if "eval_report" in v:
            acc = v["eval_report"]["accuracy"]
            print(f"  {name}: accuracy {acc:.3f}")
        elif "eval_error" in v:
            print(f"  {name}: no eval ({v['eval_error']})")
        else:
            print(f"  {name}: predictions only, no ground truth provided")
    print(f"\nFull manifest: {manifest_path}")


if __name__ == "__main__":
    main()