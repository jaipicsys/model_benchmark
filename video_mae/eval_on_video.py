"""
Quick evaluation of Phase 1's classifier on a continuous video, using only:
    - the .npz output of extract_embeddings.py (predicted probs per window)
    - a lightweight ground-truth annotation of the same video (same JSON
      format as example_annotation.json -- just start/end/label per action,
      written by watching the video once)

This does NOT train anything. It's purely a scoring step: for each sliding
window, it looks up what the model predicted and what actually happened
(based on the window's center timestamp), and reports window-level accuracy
plus a confusion matrix.

The scoring logic lives in evaluate_embeddings() so it can be reused by
pipeline_runner.py without duplicating it -- this file's main() is a thin
CLI wrapper around that function.

Usage:
    python eval_on_video.py \
        --embeddings_npz embeddings/new_video.npz \
        --annotation_json new_video_ground_truth.json
"""

import argparse

import numpy as np

from segment_utils import align_annotations_to_windows, load_annotation


def evaluate_embeddings(embeddings_npz_path, annotation_json_path):
    """
    Returns a report dict:
        {
          "n_correct", "n_total", "accuracy",
          "n_skipped_no_ground_truth",
          "labels_present": [...],
          "confusion_matrix": [[...], ...],   # rows=actual, cols=predicted, same order as labels_present
          "per_class_recall": {label: recall_or_None},
          "model_fingerprint": <str or None>,  # present if pipeline_runner.py tagged this npz
        }
    Raises ValueError if no windows overlap the annotation at all.
    """
    data = np.load(embeddings_npz_path, allow_pickle=True)
    window_starts, window_ends = data["window_starts"], data["window_ends"]
    probs = data["probs"]
    class_names = list(data["class_names"])
    model_fingerprint = str(data["model_fingerprint"][0]) if "model_fingerprint" in data else None

    pred_idx = probs.argmax(axis=1)
    pred_labels = [class_names[i] for i in pred_idx]

    annotation_segments = load_annotation(annotation_json_path)
    true_labels = align_annotations_to_windows(window_starts, window_ends, annotation_segments)

    scored = [(t, p) for t, p in zip(true_labels, pred_labels) if t != "unknown"]
    n_skipped = len(true_labels) - len(scored)
    if not scored:
        raise ValueError("No windows overlapped with the annotation -- "
                          "check timestamps/labels match.")

    n_correct = sum(1 for t, p in scored if t == p)
    n_total = len(scored)

    labels_present = sorted(set(t for t, _ in scored) | set(p for _, p in scored))
    idx = {l: i for i, l in enumerate(labels_present)}
    cm = np.zeros((len(labels_present), len(labels_present)), dtype=int)
    for t, p in scored:
        cm[idx[t], idx[p]] += 1

    per_class_recall = {}
    for i, l in enumerate(labels_present):
        total_true = cm[i].sum()
        per_class_recall[l] = (cm[i, i] / total_true) if total_true > 0 else None

    return {
        "n_correct": n_correct,
        "n_total": n_total,
        "accuracy": n_correct / n_total,
        "n_skipped_no_ground_truth": n_skipped,
        "labels_present": labels_present,
        "confusion_matrix": cm.tolist(),
        "per_class_recall": per_class_recall,
        "model_fingerprint": model_fingerprint,
    }


def format_report(report):
    """Human-readable rendering of a report dict from evaluate_embeddings()."""
    lines = []
    lines.append(f"Window-level accuracy: {report['n_correct']}/{report['n_total']} "
                  f"= {report['accuracy']:.3f}")
    lines.append(f"({report['n_skipped_no_ground_truth']} windows had no ground-truth "
                  f"coverage and were skipped)")
    if report.get("model_fingerprint"):
        lines.append(f"Model fingerprint: {report['model_fingerprint']}")
    lines.append("")

    labels_present = report["labels_present"]
    cm = np.array(report["confusion_matrix"])
    col_width = max(len(l) for l in labels_present) + 2
    header = " " * col_width + "".join(f"{l[:10]:>12}" for l in labels_present)
    lines.append("Confusion matrix (rows = actual, cols = predicted):")
    lines.append(header)
    for i, l in enumerate(labels_present):
        row = f"{l:<{col_width}}" + "".join(f"{cm[i, j]:>12}" for j in range(len(labels_present)))
        lines.append(row)

    lines.append("\nPer-class recall (of the windows truly labeled X, how many predicted correctly):")
    for l, recall in report["per_class_recall"].items():
        if recall is not None:
            lines.append(f"  {l}: {recall:.3f}")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--embeddings_npz", type=str, required=True,
                         help="Output of extract_embeddings.py for this video")
    parser.add_argument("--annotation_json", type=str, required=True,
                         help="Ground-truth annotation (same format as example_annotation.json)")
    args = parser.parse_args()

    try:
        report = evaluate_embeddings(args.embeddings_npz, args.annotation_json)
    except ValueError as e:
        print(str(e))
        return

    print(format_report(report))


if __name__ == "__main__":
    main()