"""
Fine-tune VideoMAE (or VideoMAE V2) for operator action classification.

Expected data layout (standard "ImageFolder-style" for video):

    data_dir/
        train/
            pick_up_part/
                clip001.mp4
                clip002.mp4
                ...
            assemble/
                clip001.mp4
                ...
            idle_fumble/
                clip001.mp4
                ...
        val/
            pick_up_part/
                clip010.mp4
            assemble/
                clip010.mp4
            idle_fumble/
                clip010.mp4

Each subfolder name under train/ and val/ is treated as a class label.
Clip length can vary (your 3-10s range is fine) -- frames are uniformly
sampled to a fixed count regardless of raw duration.

Install (once):
    pip install --break-system-packages transformers accelerate decord \
        torch torchvision scikit-learn

Run:
    python finetune_videomae.py \
        --data_dir /path/to/data_dir \
        --output_dir ./videomae_finetuned \
        --model_ckpt MCG-NJU/videomae-base-finetuned-kinetics \
        --num_frames 16 \
        --epochs 30 \
        --batch_size 4 \
        --lr 5e-5 \
        --freeze_backbone_layers 8 \
        --roi 280,0,440,690

Notes:
    - model_ckpt options (HuggingFace hub):
        "MCG-NJU/videomae-base-finetuned-kinetics"   (VideoMAE v1, base, K400 fine-tuned)
        "MCG-NJU/videomae-base"                      (VideoMAE v1, base, MAE-pretrained only)
        "OpenGVLab/VideoMAEv2-Base"                   (VideoMAE V2, if available on hub in
                                                        your environment -- check availability;
                                                        VideoMAE v1 checkpoints are the most
                                                        reliably available via `transformers`)
      Start from a Kinetics-fine-tuned checkpoint if available -- it's already adapted
      from "reconstruct masked patches" toward "classify actions", which is closer to
      your end task and tends to fine-tune faster/better with small data.

    - freeze_backbone_layers: number of the FIRST N encoder blocks to freeze.
      With ~50 clips total, freezing most of the backbone and only training the
      last few blocks + classification head is strongly recommended to avoid
      overfitting. Set to 0 to fine-tune the whole network end-to-end.

    - Overfitting safeguards (on by default):
        * Early stopping on eval_loss (not eval_accuracy -- accuracy saturates
          at 1.0 early on small datasets and stops being informative, while
          eval_loss keeps revealing whether the model is still generalizing
          or just sharpening confidence on memorized training patterns).
          Training stops once eval_loss hasn't improved for
          --early_stopping_patience evaluations, well before --epochs is
          reached if the data is small. Tune with --early_stopping_patience
          and --early_stopping_threshold, or disable with
          --early_stopping_patience 0.
        * Color jitter augmentation (brightness/contrast/saturation), applied
          consistently across all frames within a clip (so motion stays
          coherent) alongside the existing temporal jitter + horizontal flip.
          Tune strength with --color_jitter_strength, or disable with
          --color_jitter_strength 0.
      Neither of these replaces having enough data and enough operator/session
      diversity -- they reduce overfitting for a given dataset, they don't
      substitute for a dataset that's still too small or too narrow.

    - --roi x,y,w,h: fixed crop applied to every frame before resize/augmentation,
      in the video's native pixel coordinates. Only makes sense for a fixed
      camera position (the same box is used for every clip). Cropping out
      irrelevant background (floor, shelving, other equipment) reduces
      visual noise the model has to learn to ignore, and effectively gives
      more resolution to the region that actually matters, since the model's
      input size (224x224) is fixed regardless of how much of the frame you
      crop away first. Saved alongside the trained model as roi.json so
      extract_embeddings.py / detect_segments.py can pick it up automatically
      at inference time -- train and inference must use the same crop.
"""

import argparse
import os
import random

import numpy as np
import torch
from decord import VideoReader, cpu
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import Dataset
from transformers import (
    EarlyStoppingCallback,
    TrainingArguments,
    Trainer,
    VideoMAEForVideoClassification,
    VideoMAEImageProcessor,
)


# --------------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------------- #
class VideoClipDataset(Dataset):
    """
    Reads clips from a class-labeled folder structure and uniformly samples
    a fixed number of frames per clip, regardless of the clip's raw duration.
    """

    def __init__(self, root_dir, class_to_idx, processor, num_frames=16, train=True,
                 color_jitter_strength=0.2, roi=None):
        self.samples = []  # list of (filepath, label_idx)
        self.class_to_idx = class_to_idx
        self.processor = processor
        self.num_frames = num_frames
        self.train = train
        self.color_jitter_strength = color_jitter_strength
        self.roi = roi  # (x, y, w, h) in the video's native pixel coordinates, or None

        for class_name, idx in class_to_idx.items():
            class_dir = os.path.join(root_dir, class_name)
            if not os.path.isdir(class_dir):
                continue
            for fname in os.listdir(class_dir):
                # skip macOS AppleDouble metadata sidecar files (e.g. "._clip.mp4"),
                # which get created when copying from a Mac and are not real videos
                if fname.startswith("._"):
                    continue
                if fname.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
                    self.samples.append((os.path.join(class_dir, fname), idx))

        if len(self.samples) == 0:
            raise RuntimeError(f"No video files found under {root_dir}")

    def __len__(self):
        return len(self.samples)

    def _sample_frame_indices(self, total_frames):
        """Uniform sampling across the full clip, regardless of duration."""
        if total_frames <= self.num_frames:
            # short clip: repeat frames to pad up to num_frames
            indices = np.linspace(0, total_frames - 1, num=self.num_frames)
            indices = np.round(indices).astype(int)
        else:
            if self.train:
                # small random jitter within each segment (temporal augmentation)
                segment_len = total_frames / self.num_frames
                indices = []
                for i in range(self.num_frames):
                    start = int(i * segment_len)
                    end = int((i + 1) * segment_len)
                    end = max(end, start + 1)
                    end = min(end, total_frames)
                    indices.append(random.randint(start, end - 1))
                indices = np.array(indices)
            else:
                indices = np.linspace(0, total_frames - 1, num=self.num_frames)
                indices = np.round(indices).astype(int)
        return np.clip(indices, 0, total_frames - 1)

    def _crop_roi(self, frames):
        """Crops every frame to the fixed (x, y, w, h) region of interest, in
        the video's native pixel coordinates. No-op if self.roi is None."""
        if self.roi is None:
            return frames
        x, y, w, h = self.roi
        return frames[:, y:y + h, x:x + w, :]

    def _resize_frames(self, frames, size=256):
        """
        Cheap nearest-neighbor downsize, applied right after decoding and
        before flip/jitter. The model's actual input is 224x224 (via the
        processor's own resize/crop), so frames were always going to be
        shrunk from their native decoded resolution (e.g. 1280x720) down to
        roughly this size -- doing it here first, before the augmentation
        steps, means flip and color jitter operate on ~256x256 instead of
        full native resolution, which is where most of their CPU cost was
        coming from. Quality impact is negligible since 256 is still above
        the 224 the model ultimately sees.
        """
        T, H, W, C = frames.shape
        if H <= size and W <= size:
            return frames
        row_idx = np.linspace(0, H - 1, size).astype(int)
        col_idx = np.linspace(0, W - 1, size).astype(int)
        return frames[:, row_idx][:, :, col_idx]

    def _apply_color_jitter(self, frames):
        """
        Random brightness/contrast/saturation jitter, applied with the SAME
        random factors across every frame in the clip -- jittering each frame
        independently would flicker brightness frame-to-frame and corrupt the
        motion signal the model is supposed to learn from. Strength 0 disables.
        """
        if self.color_jitter_strength <= 0:
            return frames

        s = self.color_jitter_strength
        frames = frames.astype(np.float32)

        # brightness: additive shift
        brightness = 1.0 + random.uniform(-s, s)
        frames = frames * brightness

        # contrast: scale around the clip's own mean
        contrast = 1.0 + random.uniform(-s, s)
        mean = frames.mean(axis=(0, 1, 2), keepdims=True)
        frames = (frames - mean) * contrast + mean

        # saturation: blend toward grayscale
        saturation = 1.0 + random.uniform(-s, s)
        gray = frames.mean(axis=-1, keepdims=True)
        frames = gray + (frames - gray) * saturation

        return np.clip(frames, 0, 255).astype(np.uint8)

    def __getitem__(self, idx):
        filepath, label = self.samples[idx]
        vr = VideoReader(filepath, ctx=cpu(0))
        total_frames = len(vr)
        frame_indices = self._sample_frame_indices(total_frames)
        frames = vr.get_batch(frame_indices).asnumpy()  # (T, H, W, C), uint8
        frames = self._crop_roi(frames)  # crop to the workstation region first, if set
        frames = self._resize_frames(frames)  # shrink before flip/jitter -- see docstring

        # simple spatial augmentation for training: random horizontal flip
        if self.train and random.random() < 0.5:
            frames = frames[:, :, ::-1, :].copy()

        if self.train:
            frames = self._apply_color_jitter(frames)

        # VideoMAEImageProcessor expects a list of frames (T, H, W, C)
        inputs = self.processor(list(frames), return_tensors="pt")
        pixel_values = inputs["pixel_values"][0]  # (T, C, H, W)

        return {"pixel_values": pixel_values, "labels": label}


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1_macro": f1_score(labels, preds, average="macro"),
    }


# --------------------------------------------------------------------------- #
# Layer freezing
# --------------------------------------------------------------------------- #
def freeze_backbone_layers(model, num_layers_to_freeze):
    """
    Freezes the embeddings and the first `num_layers_to_freeze` encoder
    blocks of the VideoMAE backbone. The classification head (and any
    unfrozen later blocks) remain trainable.
    """
    if num_layers_to_freeze <= 0:
        return

    # freeze patch/positional embeddings
    for param in model.videomae.embeddings.parameters():
        param.requires_grad = False

    encoder_layers = model.videomae.encoder.layer
    n = min(num_layers_to_freeze, len(encoder_layers))
    for i in range(n):
        for param in encoder_layers[i].parameters():
            param.requires_grad = False

    print(f"Froze embeddings + first {n}/{len(encoder_layers)} encoder blocks.")
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable params: {trainable:,} / {total:,} ({100*trainable/total:.1f}%)")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True,
                         help="Root dir containing train/ and val/ subfolders")
    parser.add_argument("--output_dir", type=str, default="./videomae_finetuned")
    parser.add_argument("--model_ckpt", type=str,
                         default="MCG-NJU/videomae-base-finetuned-kinetics")
    parser.add_argument("--num_frames", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--freeze_backbone_layers", type=int, default=8,
                         help="Freeze first N encoder blocks. 0 = fine-tune all.")
    parser.add_argument("--fp16", action="store_true", default=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--early_stopping_patience", type=int, default=5,
                         help="Stop training if eval_loss hasn't improved for this many "
                              "evaluations (epochs). Set to 0 to disable and always run "
                              "the full --epochs.")
    parser.add_argument("--early_stopping_threshold", type=float, default=0.0,
                         help="Minimum eval_loss decrease to count as an improvement, "
                              "for early stopping purposes.")
    parser.add_argument("--color_jitter_strength", type=float, default=0.2,
                         help="Brightness/contrast/saturation jitter strength for "
                              "training augmentation (0 disables). Applied consistently "
                              "across all frames in a clip.")
    parser.add_argument("--roi", type=str, default=None,
                         help="Fixed region of interest to crop from every frame before "
                              "anything else, as 'x,y,w,h' in the video's native pixel "
                              "coordinates (e.g. '280,0,440,690'). Assumes a fixed camera "
                              "position -- the same box is used for every clip. Omit to "
                              "use the full frame.")
    args = parser.parse_args()

    roi = None
    if args.roi:
        try:
            roi = tuple(int(v) for v in args.roi.split(","))
            assert len(roi) == 4
        except (ValueError, AssertionError):
            parser.error(f"--roi must be 'x,y,w,h' (four integers), got: {args.roi}")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    train_dir = os.path.join(args.data_dir, "train")
    val_dir = os.path.join(args.data_dir, "val")

    class_names = sorted(
        d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))
    )
    class_to_idx = {name: i for i, name in enumerate(class_names)}
    idx_to_class = {i: name for name, i in class_to_idx.items()}
    print(f"Found {len(class_names)} classes: {class_names}")

    processor = VideoMAEImageProcessor.from_pretrained(args.model_ckpt)

    train_dataset = VideoClipDataset(
        train_dir, class_to_idx, processor, num_frames=args.num_frames, train=True,
        color_jitter_strength=args.color_jitter_strength, roi=roi,
    )
    val_dataset = VideoClipDataset(
        val_dir, class_to_idx, processor, num_frames=args.num_frames, train=False, roi=roi
    )
    print(f"Train clips: {len(train_dataset)} | Val clips: {len(val_dataset)}")

    model = VideoMAEForVideoClassification.from_pretrained(
        args.model_ckpt,
        num_labels=len(class_names),
        id2label=idx_to_class,
        label2id=class_to_idx,
        ignore_mismatched_sizes=True,  # replaces the pretrained classification head
    )

    freeze_backbone_layers(model, args.freeze_backbone_layers)

    steps_per_epoch = max(1, (len(train_dataset) + args.batch_size - 1) // args.batch_size)
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = max(1, int(total_steps * args.warmup_ratio))

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        warmup_steps=warmup_steps,  # computed from warmup_ratio for compatibility across
                                     # transformers versions (warmup_ratio itself was removed
                                     # in some newer releases -- see finetune README note)
        weight_decay=args.weight_decay,
        fp16=args.fp16 and torch.cuda.is_available(),
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",  # more sensitive to overfitting than accuracy,
                                             # which saturates at 1.0 early on small datasets
        greater_is_better=False,
        logging_steps=5,
        report_to=[],  # disable wandb/etc unless you want it
        dataloader_num_workers=2,
        remove_unused_columns=False,
        seed=args.seed,
    )

    callbacks = []
    if args.early_stopping_patience > 0:
        callbacks.append(EarlyStoppingCallback(
            early_stopping_patience=args.early_stopping_patience,
            early_stopping_threshold=args.early_stopping_threshold,
        ))
        print(f"Early stopping enabled: patience={args.early_stopping_patience} "
              f"evaluations, threshold={args.early_stopping_threshold} (on eval_loss)")
    else:
        print("Early stopping disabled -- will always run the full --epochs.")

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=callbacks,
    )

    trainer.train()

    final_metrics = trainer.evaluate()
    print("Final validation metrics:", final_metrics)

    trainer.save_model(os.path.join(args.output_dir, "best_model"))
    processor.save_pretrained(os.path.join(args.output_dir, "best_model"))

    # Save the ROI alongside the model so inference scripts can pick it up
    # automatically -- train and inference must use the SAME crop, and this
    # avoids relying on retyping the same coordinates correctly every time.
    import json
    with open(os.path.join(args.output_dir, "best_model", "roi.json"), "w") as f:
        json.dump({"roi": list(roi) if roi else None}, f)

    print(f"Saved best model to {os.path.join(args.output_dir, 'best_model')}")


if __name__ == "__main__":
    main()
