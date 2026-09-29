"""
Shared utilities bridging Phase 3 (boundary annotations) and Phase 4
(temporal segmentation head), plus post-processing of predicted
per-window labels into a clean segment timeline.

Phase 3 annotation format (one JSON file per continuously-recorded session):

    {
      "session_id": "session_001",
      "segments": [
        {"label": "pick_up_part", "start_sec": 2.1,  "end_sec": 5.4},
        {"label": "idle_fumble",  "start_sec": 5.4,  "end_sec": 19.0},
        {"label": "assemble",     "start_sec": 19.0, "end_sec": 24.3}
      ]
    }

You do NOT need to annotate every single session -- just enough
(Phase 3 in our discussion suggested ~10-20 sessions covering your action
variety) to train/validate the Phase 4 segmentation head.
"""

import json

import numpy as np


def load_annotation(json_path):
    with open(json_path, "r") as f:
        data = json.load(f)
    return data["segments"]


def align_annotations_to_windows(window_starts, window_ends, annotation_segments,
                                  unknown_label="unknown"):
    """
    Assigns a ground-truth label to each sliding window based on which
    annotated segment covers the window's CENTER timestamp.

    This is what turns your Phase 3 boundary annotations (start/end times)
    into per-window training labels for the Phase 4 sequence model.
    """
    labels = []
    for s, e in zip(window_starts, window_ends):
        center = (s + e) / 2.0
        assigned = unknown_label
        for seg in annotation_segments:
            if seg["start_sec"] <= center < seg["end_sec"]:
                assigned = seg["label"]
                break
        labels.append(assigned)
    return labels


def labels_to_segments(window_starts, window_ends, labels):
    """
    Collapses a per-window label sequence into a list of contiguous segments:
        [{"label": ..., "start_sec": ..., "end_sec": ...}, ...]

    Consecutive windows with the same predicted label are merged into one
    segment spanning from the first window's start to the last window's end.
    """
    if len(labels) == 0:
        return []

    segments = []
    cur_label = labels[0]
    cur_start = window_starts[0]
    cur_end = window_ends[0]

    for i in range(1, len(labels)):
        if labels[i] == cur_label:
            cur_end = window_ends[i]
        else:
            segments.append({"label": cur_label, "start_sec": float(cur_start),
                              "end_sec": float(cur_end)})
            cur_label = labels[i]
            cur_start = window_starts[i]
            cur_end = window_ends[i]

    segments.append({"label": cur_label, "start_sec": float(cur_start),
                      "end_sec": float(cur_end)})
    return segments


def apply_min_duration_filter(segments, min_duration_sec=1.0):
    """
    Removes spurious very-short segments (label flicker) by merging them
    into whichever neighbor is longer. Repeats until stable or no more
    short segments remain.

    This is a simple, cheap smoothing step to run on top of the Phase 4
    model's raw per-window predictions before Phase 5 sequence verification.
    """
    segments = [dict(s) for s in segments]  # copy

    changed = True
    while changed and len(segments) > 1:
        changed = False
        for i, seg in enumerate(segments):
            duration = seg["end_sec"] - seg["start_sec"]
            if duration < min_duration_sec:
                # decide which neighbor to merge into (prefer the longer one)
                left_len = (segments[i - 1]["end_sec"] - segments[i - 1]["start_sec"]
                            if i > 0 else -1)
                right_len = (segments[i + 1]["end_sec"] - segments[i + 1]["start_sec"]
                             if i < len(segments) - 1 else -1)

                if left_len == -1 and right_len == -1:
                    continue  # only segment, nothing to merge into

                if left_len >= right_len:
                    segments[i - 1]["end_sec"] = seg["end_sec"]
                    del segments[i]
                else:
                    segments[i + 1]["start_sec"] = seg["start_sec"]
                    del segments[i]
                changed = True
                break  # restart scan after a structural change

    # re-merge any now-adjacent same-label segments produced by the merges above
    merged = []
    for seg in segments:
        if merged and merged[-1]["label"] == seg["label"]:
            merged[-1]["end_sec"] = seg["end_sec"]
        else:
            merged.append(seg)
    return merged


def majority_vote_smoothing(labels, window_radius=1):
    """
    Simple baseline smoothing: replaces each window's label with the
    majority label within a +/- window_radius neighborhood. Useful as a
    quick sanity-check baseline to compare against the Phase 4 model's
    learned smoothing.
    """
    smoothed = []
    n = len(labels)
    for i in range(n):
        lo, hi = max(0, i - window_radius), min(n, i + window_radius + 1)
        neighborhood = labels[lo:hi]
        vals, counts = np.unique(neighborhood, return_counts=True)
        smoothed.append(vals[np.argmax(counts)])
    return smoothed
