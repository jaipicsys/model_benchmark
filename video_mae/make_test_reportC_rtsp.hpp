// make_test_report_live.hpp
//
// Live RTSP step monitor: VideoMAE (ONNX Runtime) window classification + the offline post-processing,
// applied to frames coming from an RTSP stream (single header; implementation in make_test_report_live.cpp).
//
// DEPENDENCIES
//   OpenCV (videoio with FFmpeg, imgproc)   RTSP decoding + resize
//   ONNX Runtime (C++ API)                  VideoMAE model
//   nlohmann/json                           config.json / roi.json / optional events log
//
// MODEL EXPORT (one-time, in Python). The C++ side loads an ONNX file, not a HF folder:
//
//   import torch
//   from transformers import VideoMAEForVideoClassification
//   m = VideoMAEForVideoClassification.from_pretrained("runs/roi_v4/videomae_finetuned/best_model").eval()
//   class W(torch.nn.Module):
//       def __init__(s, m): super().__init__(); s.m = m
//       def forward(s, x):
//           pooled = s.m.videomae(pixel_values=x).last_hidden_state.mean(1)
//           logits = s.m.classifier(s.m.fc_norm(pooled)) if hasattr(s.m, "fc_norm") else s.m(pixel_values=x).logits
//           return logits, pooled
//   x = torch.zeros(1, 16, 3, 224, 224)
//   torch.onnx.export(W(m), x, "best_model/model.onnx", input_names=["pixel_values"],
//                     output_names=["logits", "pooled"], opset_version=17)
//
// Keep config.json, preprocessor_config.json and roi.json next to model.onnx.
#pragma once

#include <cstdint>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace vmae {

inline const std::string UNCERTAIN = "uncertain";
inline const std::string DEFAULT_STEP6_RENAME_FROM = "step6";
inline const std::string DEFAULT_STEP6_RENAME_TO = "step6.5";

using RenameMap = std::map<std::string, std::string>;

// ----------------------------------------------------------------------------- basic types
struct WindowTime {
    double start;
    double end;
};

struct Segment {  // predicted segment
    std::string label;
    double start_sec;
    double end_sec;
};

struct Detected {
    std::string label;
    double start;
    double end;
    std::optional<double> conf;
};

struct Roi {
    int x, y, w, h;
};

// ----------------------------------------------------------------------------- post-processing config
// Every threshold / switch of the post-processing lives here (nothing is hard-coded), so it can be
// retuned after the model is retrained.  Pipeline (see postprocess_segments):
//   1. per-window argmax, conf = max prob, conf < window_conf_thr  -> "uncertain"      (windows_to_segments)
//   2. labels_to_segments + apply_min_duration_filter(min_segment_duration); drop "uncertain" segments
//   3. segment confidence = mean conf of ALL windows whose midpoint lies in [start, end]
//   4. merge overlapping same-label segments (conf = max of the two)
//   5. drop segments with conf < seg_conf_thr
//   6. step-order filter: DP for the best non-decreasing-order subsequence (max sum conf*duration)
struct PostCfg {
    bool enable = true;                 // --no_postproc turns the whole thing off (raw detections)
    double window_conf_thr = 0.7;       // step 1 (overridden by --confidence_threshold if given)
    double seg_conf_thr = 0.7;          // step 5
    bool do_merge = true;               // step 4
    bool do_conf_filter = true;         // step 5
    bool do_order_filter = true;        // step 6
    bool debug = false;                 // print every window + all stages (--post_debug)
    std::vector<std::string> order = {"start_step1", "start_step2", "step4",  "step5",     "step6",
                                      "step6.5",     "step7",       "step8",  "step9",     "step10",
                                      "stop_step1",  "stop_step2"};
};

// ----------------------------------------------------------------------------- live config
struct LiveCfg {
    std::string rtsp_url = "rtsp://192.168.2.170:8554/cam1";
    bool rtsp_tcp = true;          // RTSP over TCP (--udp to switch)
    double buffer_sec = 12.0;      // how much ROI-cropped video is kept in RAM (~0.9 MB per frame for a 440x690 ROI)
    double status_interval = 5.0;  // seconds between "current step" status lines (0 = off)
    double complete_gap = 2.0;     // a step counts as finished when the latest window end is >= this many seconds past it
    double cycle_max_sec = 600.0;  // force-close a cycle that runs longer than this
    double cycle_idle_sec = 60.0;  // abandon a cycle if no new step appears for this long
    int confirm_updates = 2;       // a step is announced once it was present in this many consecutive window updates (flicker guard)
    double max_frame_gap = 1.0;    // skip a window if no buffered frame lies within this many seconds of a sample time
    double reconnect_sec = 2.0;    // wait before reconnecting a dropped stream
    double max_runtime_sec = 0.0;  // 0 = run until Ctrl-C
    std::string events_log;        // optional JSON-lines file with step/cycle events (empty = off)
};

struct Args {
    std::string videomae_dir;  // folder with config.json / preprocessor_config.json / roi.json
    std::string onnx_path;     // default: <videomae_dir>/model.onnx
    double window_len = 2.0, stride = 1.0, fps = 15.0;
    int num_frames = 16;
    double min_segment_duration = 2.5;
    std::optional<double> confidence_threshold;
    std::string roi;  // "x,y,w,h" override
    bool merge_step6 = false, use_cuda = true;  // CUDA on by default; pass --cpu to disable
    PostCfg post;
    LiveCfg live;
};

// ----------------------------------------------------------------------------- labels
std::string canon(const std::string& name, const RenameMap& rename);

// ----------------------------------------------------------------------------- segment_utils.py
std::vector<Segment> labels_to_segments(const std::vector<double>& window_starts,
                                        const std::vector<double>& window_ends,
                                        const std::vector<std::string>& labels);
std::vector<Segment> apply_min_duration_filter(std::vector<Segment> segments, double min_duration_sec = 1.0);

// ----------------------------------------------------------------------------- extract_embeddings.py
std::optional<Roi> resolve_roi(const std::string& model_dir, const std::string& cli_roi_str = "");

// Raw RGB frame (HWC, uint8), already ROI-cropped.
struct RawFrame {
    int h = 0, w = 0;
    std::vector<uint8_t> rgb;
};

// ONNX Runtime wrapper + VideoMAEImageProcessor-equivalent preprocessing.
class VideoMAEModel {
public:
    VideoMAEModel(const std::string& model_dir, const std::string& onnx_path, bool use_cuda);
    ~VideoMAEModel();
    VideoMAEModel(const VideoMAEModel&) = delete;
    VideoMAEModel& operator=(const VideoMAEModel&) = delete;

    struct Output {
        std::vector<double> probs;   // softmax
        std::vector<float> pooled;   // may be empty
    };
    Output run(const std::vector<RawFrame>& frames);  // frames.size() == num_frames
    const std::string& device() const;

private:
    struct Impl;
    std::unique_ptr<Impl> p_;
};

// ----------------------------------------------------------------------------- post-processing
struct Collapse {
    std::vector<std::string> uniq;
    std::vector<int> map;  // raw class index -> index in uniq
};
Collapse collapse_matrix(const std::vector<std::string>& class_names_raw, const RenameMap& rename);
std::vector<std::vector<double>> collapse_probs(const std::vector<std::vector<double>>& probs, const Collapse& c);

struct SegResult {
    std::vector<Segment> segs;
    std::vector<std::string> labels;
    std::vector<double> conf;
};
SegResult windows_to_segments(const std::vector<WindowTime>& wt, const std::vector<std::vector<double>>& probs_c,
                              const std::vector<std::string>& uniq, const Args& a);

std::optional<double> segment_confidence(const std::vector<WindowTime>& wt, const std::vector<double>& conf,
                                         const Segment& seg);

// Per-stage snapshots of postprocess_segments (for debug logging).
struct PostDebug {
    std::vector<Detected> unfiltered;      // everything windows_to_segments produced (incl. "uncertain")
    std::vector<Detected> after_drop_unc;  // step 2
    std::vector<Detected> after_merge;     // step 4
    std::vector<Detected> after_conf;      // step 5
    std::vector<Detected> final_;          // step 6 (= returned value)
    std::vector<Detected> dropped_unknown; // labels not in PostCfg::order (removed by the order filter)
};

// Steps 2-6 of the README. `segs` = output of windows_to_segments (min-duration filter already applied,
// "uncertain" segments still present).  Returns the final detections sorted by start time.
//
// Order-filter tie-breaking (explicit):
//   * segments are sorted by (start, end, original index);
//   * predecessor j for i: the j<i with idx(label j) <= idx(label i) and the highest best[j];
//     equal best[j] (|diff| < 1e-12) -> the LATER j (closest in time) wins;
//   * the chain end: the highest best[i]; equal -> the LATER i wins;
//   * labels that are not in PostCfg::order cannot be placed in the order and are dropped
//     (they are listed in PostDebug::dropped_unknown).
std::vector<Detected> postprocess_segments(const std::vector<WindowTime>& wt, const std::vector<double>& conf,
                                           const std::vector<Segment>& segs, const PostCfg& cfg,
                                           PostDebug* dbg = nullptr);

// ----------------------------------------------------------------------------- entry points
Args parse_args(int argc, char** argv);
int run_live(const Args& args);

}  // namespace vmae
