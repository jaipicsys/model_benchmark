// make_test_report.hpp
//
// C++ port of make_test_report.py + extract_embeddings.py + segment_utils.py
// (single header; implementation in make_test_report.cpp).
//
// DEPENDENCIES
//   OpenCV (videoio, imgproc)   replaces decord
//   ONNX Runtime (C++ API)      replaces transformers VideoMAE
//   xlnt                        replaces openpyxl
//   libzip + pugixml            read the .ods ground-truth file (replaces pandas+odf)
//   nlohmann/json               config.json / roi.json / results.json
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
constexpr double GT_MERGE_GAP = 3.0;  // GT ranges with same label closer than this are merged
constexpr int MAX_ROWS = 40;          // template capacity for GT rows / detected rows per sheet

using RenameMap = std::map<std::string, std::string>;

// ----------------------------------------------------------------------------- basic types
struct WindowTime {
    double start;
    double end;
};

struct Segment {  // used for predicted segments and Phase-3 annotations
    std::string label;
    double start_sec;
    double end_sec;
};

struct GtSegment {
    std::string label;
    double start;
    double end;
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

struct Timing {
    std::string device;
    int num_windows = 0;
    double total_sec_incl_decode = 0, total_sec_model_only = 0;
    double per_window_sec_incl_decode = 0, per_window_sec_model_only = 0;
};

struct SessionResult {
    std::vector<WindowTime> windows;
    std::vector<std::vector<float>> embeddings;  // empty if the ONNX model has no 2nd output
    std::vector<std::vector<double>> probs;      // N x num_classes
    Timing timing;
};

struct WindowsProbs {
    std::vector<WindowTime> windows;
    std::vector<std::vector<double>> probs;
};

struct ClipResult {
    std::string file, true_label, pred, note;
    double conf = 0;
};

struct Args {
    std::string videomae_dir;  // folder with config.json / preprocessor_config.json / roi.json
    std::string onnx_path;     // default: <videomae_dir>/model.onnx
    std::string individual_dir, continuous_dir;
    std::string gt = "cycle20GT.ods";
    std::string tmpl = "temporal_test_template.xlsx";
    double window_len = 2.0, stride = 1.0, fps = 15.0;
    int num_frames = 16;
    double min_segment_duration = 2.5;
    std::optional<double> confidence_threshold;
    std::string roi;  // "x,y,w,h" override
    std::string runs_dir = "runs";
    std::string out_dir, out_name;
    std::string clip_method = "any_detected";  // any_detected | dominant_segment | mean_prob
    double min_overlap = 1.0;
    bool keep_uncertain = false, merge_step6 = false, no_cache = false, use_cuda = true;  // CUDA on by default; pass --cpu to disable
    std::string tester, test_date;
};

// ----------------------------------------------------------------------------- labels
std::string canon(const std::string& name, const RenameMap& rename);
std::optional<std::string> label_from_filename(const std::string& path, const RenameMap& rename);

// ----------------------------------------------------------------------------- ground truth (.ods / .xlsx)
struct GtResult {
    std::map<int, std::vector<GtSegment>> gt;  // cycle -> segments sorted by start
    std::vector<std::string> labels;
};
GtResult load_gt(const std::string& path, const RenameMap& rename);

// ----------------------------------------------------------------------------- segment_utils.py
std::vector<Segment> load_annotation(const std::string& json_path);
std::vector<std::string> align_annotations_to_windows(const std::vector<double>& window_starts,
                                                      const std::vector<double>& window_ends,
                                                      const std::vector<Segment>& annotation_segments,
                                                      const std::string& unknown_label = "unknown");
std::vector<Segment> labels_to_segments(const std::vector<double>& window_starts,
                                        const std::vector<double>& window_ends,
                                        const std::vector<std::string>& labels);
std::vector<Segment> apply_min_duration_filter(std::vector<Segment> segments, double min_duration_sec = 1.0);
std::vector<std::string> majority_vote_smoothing(const std::vector<std::string>& labels, int window_radius = 1);

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

SessionResult extract_session_embeddings(const std::string& video_path, VideoMAEModel& model,
                                         double window_len = 2.0, double stride = 1.0, double fps = 15.0,
                                         int num_frames = 16, const std::optional<Roi>& roi = std::nullopt);

// ----------------------------------------------------------------------------- inference cache
class Engine {
public:
    Engine(const Args& args, const std::optional<Roi>& roi, const std::string& fingerprint);
    WindowsProbs windows_and_probs(const std::string& video_path);

private:
    const Args& a_;
    std::optional<Roi> roi_;
    std::unique_ptr<VideoMAEModel> model_;
    std::string cache_dir_;
};

// ----------------------------------------------------------------------------- post-processing / matching
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

struct ClipPred {
    std::string label;
    double conf;
    std::string note;
};
ClipPred clip_prediction(const std::vector<WindowTime>& wt, const std::vector<std::vector<double>>& probs_c,
                         const std::vector<std::string>& uniq, const std::vector<Segment>& segs,
                         const std::vector<double>& conf, const Args& a, const std::string* true_label = nullptr);

std::optional<double> segment_confidence(const std::vector<WindowTime>& wt, const std::vector<double>& conf,
                                         const Segment& seg);

// 1-based index of the matched detection per GT segment (nullopt = missed)
std::vector<std::optional<int>> match_gt_to_det(const std::vector<GtSegment>& gt, const std::vector<Detected>& det,
                                                double min_overlap);

// ----------------------------------------------------------------------------- misc / entry points
std::string compute_model_fingerprint(const std::string& model_dir, const std::string& onnx_path);
const std::vector<std::string>& video_extensions();
Args parse_args(int argc, char** argv);
int run_report(const Args& args);

}  // namespace vmae
