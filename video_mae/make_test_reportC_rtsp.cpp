// make_test_report_live.cpp
// Live RTSP version of the VideoMAE step detector.
// Same model preprocessing, window classification and post-processing as the offline report tool;
// the video-file testing and the Excel report generation are removed.  Frames are read from an RTSP stream,
// windows are classified as they complete, and every step (start time, finish time, duration) is logged.
//
// Build example:
//   g++ -std=c++17 -O2 make_test_report_live.cpp -o step_monitor $(pkg-config --cflags --libs opencv4) -lonnxruntime -pthread
//
// Run:
//   ./step_monitor --videomae_dir runs/roi_v4/videomae_finetuned/best_model --rtsp rtsp://192.168.2.170:8554/cam1
//       --window_len 2.0 --stride 1.0 --fps 15 --num_frames 16 --min_segment_duration 2.5 --events_log events.jsonl

#include "make_test_reportC_rtsp.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <cctype>
#include <chrono>
#include <cmath>
#include <csignal>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <deque>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <iterator>
#include <memory>
#include <mutex>
#include <numeric>
#include <regex>
#include <set>
#include <sstream>
#include <stdexcept>
#include <thread>

#include <nlohmann/json.hpp>
#include <onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>

namespace fs = std::filesystem;
using json = nlohmann::json;

namespace vmae {

// ============================================================================= small helpers
static std::string trim(const std::string& s) {
    size_t b = 0, e = s.size();
    while (b < e && std::isspace((unsigned char)s[b])) ++b;
    while (e > b && std::isspace((unsigned char)s[e - 1])) --e;
    return s.substr(b, e - b);
}
static std::string lower(std::string s) {
    for (auto& c : s) c = (char)std::tolower((unsigned char)c);
    return s;
}
static bool starts_with(const std::string& s, const std::string& p) { return s.compare(0, p.size(), p) == 0; }
static bool ends_with(const std::string& s, const std::string& p) {
    return s.size() >= p.size() && s.compare(s.size() - p.size(), p.size(), p) == 0;
}
static std::string pyfloat(double v) {  // Python-like str(float)
    if (std::isfinite(v) && v == std::floor(v) && std::fabs(v) < 1e15) {
        char b[64];
        std::snprintf(b, sizeof b, "%.1f", v);
        return b;
    }
    std::ostringstream o;
    o.precision(12);
    o << v;
    return o.str();
}
static std::string fmt(const char* f, ...) __attribute__((format(printf, 1, 2)));
static std::string fmt(const char* f, ...) {
    char buf[1024];
    va_list ap;
    va_start(ap, f);
    std::vsnprintf(buf, sizeof buf, f, ap);
    va_end(ap);
    return buf;
}
static double round_to(double v, int nd) {
    double m = std::pow(10.0, nd);
    return std::nearbyint(v * m) / m;
}
static std::string rename_repr(const RenameMap& r) {
    if (r.empty()) return "none";
    std::string s = "{";
    bool first = true;
    for (auto& kv : r) {
        if (!first) s += ", ";
        s += "'" + kv.first + "': '" + kv.second + "'";
        first = false;
    }
    return s + "}";
}

// ============================================================================= labels
static const std::regex STEP_RE(R"(^(?:(start|stop)_)?step_?(\d+(?:\.\d+)?)$)", std::regex::icase);

std::string canon(const std::string& name, const RenameMap& rename) {
    std::string s = lower(trim(name));
    s.erase(std::remove(s.begin(), s.end(), ' '), s.end());
    std::smatch m;
    if (std::regex_match(s, m, STEP_RE)) {
        std::string pre = lower(m[1].str()), n = m[2].str();
        if (pre == "stop") s = "stop_step" + n;
        else if (n == "1" || n == "2") s = "start_step" + n;
        else s = "step" + n;
    }
    auto it = rename.find(s);
    return it == rename.end() ? s : it->second;
}

// ============================================================================= segment_utils.py
std::vector<Segment> labels_to_segments(const std::vector<double>& ws, const std::vector<double>& we,
                                        const std::vector<std::string>& labels) {
    std::vector<Segment> segments;
    if (labels.empty()) return segments;
    std::string cur = labels[0];
    double cs = ws[0], ce = we[0];
    for (size_t i = 1; i < labels.size(); ++i) {
        if (labels[i] == cur) ce = we[i];
        else {
            segments.push_back({cur, cs, ce});
            cur = labels[i];
            cs = ws[i];
            ce = we[i];
        }
    }
    segments.push_back({cur, cs, ce});
    return segments;
}

std::vector<Segment> apply_min_duration_filter(std::vector<Segment> segments, double min_duration_sec) {
    bool changed = true;
    while (changed && segments.size() > 1) {
        changed = false;
        for (size_t i = 0; i < segments.size(); ++i) {
            double duration = segments[i].end_sec - segments[i].start_sec;
            if (duration < min_duration_sec) {
                double left_len = i > 0 ? segments[i - 1].end_sec - segments[i - 1].start_sec : -1;
                double right_len = i + 1 < segments.size() ? segments[i + 1].end_sec - segments[i + 1].start_sec : -1;
                if (left_len == -1 && right_len == -1) continue;
                if (left_len >= right_len) segments[i - 1].end_sec = segments[i].end_sec;
                else segments[i + 1].start_sec = segments[i].start_sec;
                segments.erase(segments.begin() + i);
                changed = true;
                break;
            }
        }
    }
    std::vector<Segment> merged;
    for (auto& s : segments) {
        if (!merged.empty() && merged.back().label == s.label) merged.back().end_sec = s.end_sec;
        else merged.push_back(s);
    }
    return merged;
}

// ============================================================================= extract_embeddings.py
std::optional<Roi> resolve_roi(const std::string& model_dir, const std::string& cli_roi_str) {
    if (!cli_roi_str.empty()) {
        std::vector<int> v;
        try {
            std::stringstream ss(cli_roi_str);
            std::string p;
            while (std::getline(ss, p, ',')) v.push_back(std::stoi(p));
        } catch (...) { v.clear(); }
        if (v.size() != 4) throw std::runtime_error("--roi must be 'x,y,w,h' (four integers), got: " + cli_roi_str);
        return Roi{v[0], v[1], v[2], v[3]};
    }
    fs::path rp = fs::path(model_dir) / "roi.json";
    if (fs::exists(rp)) {
        std::ifstream f(rp);
        json j = json::parse(f);
        if (j.contains("roi") && j["roi"].is_array() && j["roi"].size() == 4) {
            Roi r{j["roi"][0].get<int>(), j["roi"][1].get<int>(), j["roi"][2].get<int>(), j["roi"][3].get<int>()};
            std::printf("Auto-loaded ROI from %s: (%d, %d, %d, %d)\n", rp.string().c_str(), r.x, r.y, r.w, r.h);
            return r;
        }
    }
    return std::nullopt;
}

// ----- ONNX model + VideoMAEImageProcessor equivalent
struct VideoMAEModel::Impl {
    Ort::Env env{ORT_LOGGING_LEVEL_WARNING, "videomae"};
    Ort::SessionOptions opts;
    std::unique_ptr<Ort::Session> session;
    std::string in_name;
    std::vector<std::string> out_names;
    std::string device = "cpu";
    int resize_short = 224, crop_h = 224, crop_w = 224;
    float mean[3] = {0.485f, 0.456f, 0.406f};
    float stdv[3] = {0.229f, 0.224f, 0.225f};
};

VideoMAEModel::VideoMAEModel(const std::string& model_dir, const std::string& onnx_path, bool use_cuda)
    : p_(std::make_unique<Impl>()) {
    // preprocessor_config.json (defaults match VideoMAEImageProcessor)
    fs::path pc = fs::path(model_dir) / "preprocessor_config.json";
    if (fs::exists(pc)) {
        std::ifstream f(pc);
        json j = json::parse(f);
        auto edge = [](const json& v, int def, const char* k1, const char* k2) {
            if (v.is_number()) return v.get<int>();
            if (v.is_object()) {
                if (v.contains(k1)) return v[k1].get<int>();
                if (v.contains(k2)) return v[k2].get<int>();
            }
            return def;
        };
        if (j.contains("size")) p_->resize_short = edge(j["size"], 224, "shortest_edge", "height");
        if (j.contains("crop_size")) {
            p_->crop_h = edge(j["crop_size"], 224, "height", "shortest_edge");
            p_->crop_w = edge(j["crop_size"], 224, "width", "shortest_edge");
        }
        if (j.contains("image_mean") && j["image_mean"].size() == 3)
            for (int i = 0; i < 3; ++i) p_->mean[i] = j["image_mean"][i].get<float>();
        if (j.contains("image_std") && j["image_std"].size() == 3)
            for (int i = 0; i < 3; ++i) p_->stdv[i] = j["image_std"][i].get<float>();
    }
    p_->opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    if (use_cuda) {
        try {
            OrtCUDAProviderOptions co{};
            co.device_id = 0;
            p_->opts.AppendExecutionProvider_CUDA(co);
            p_->device = "cuda";
        } catch (const Ort::Exception& e) {
            std::fprintf(stderr, "[warn] CUDA provider unavailable (%s); falling back to CPU\n", e.what());
        }
    }
#ifdef _WIN32
    std::wstring wp(onnx_path.begin(), onnx_path.end());
    p_->session = std::make_unique<Ort::Session>(p_->env, wp.c_str(), p_->opts);
#else
    p_->session = std::make_unique<Ort::Session>(p_->env, onnx_path.c_str(), p_->opts);
#endif
    Ort::AllocatorWithDefaultOptions alloc;
    p_->in_name = p_->session->GetInputNameAllocated(0, alloc).get();
    for (size_t i = 0; i < p_->session->GetOutputCount(); ++i)
        p_->out_names.push_back(p_->session->GetOutputNameAllocated(i, alloc).get());
}
VideoMAEModel::~VideoMAEModel() = default;
const std::string& VideoMAEModel::device() const { return p_->device; }

VideoMAEModel::Output VideoMAEModel::run(const std::vector<RawFrame>& frames) {
    const int T = (int)frames.size(), H = p_->crop_h, W = p_->crop_w;
    std::vector<float> buf((size_t)T * 3 * H * W);
    for (int t = 0; t < T; ++t) {
        const RawFrame& rf = frames[t];
        cv::Mat src(rf.h, rf.w, CV_8UC3, const_cast<uint8_t*>(rf.rgb.data()));
        // resize shortest edge (HF: new_long = int(S * long / short))
        int sh = std::min(rf.h, rf.w), lg = std::max(rf.h, rf.w);
        int new_short = p_->resize_short, new_long = (int)((double)new_short * lg / sh);
        int nw = rf.w <= rf.h ? new_short : new_long, nh = rf.w <= rf.h ? new_long : new_short;
        cv::Mat rs;
        cv::resize(src, rs, cv::Size(nw, nh), 0, 0, (nw < rf.w || nh < rf.h) ? cv::INTER_AREA : cv::INTER_LINEAR);
        // center crop (zero-pad if image is smaller than crop)
        if (rs.rows < H || rs.cols < W) {
            cv::Mat padded(std::max(H, rs.rows), std::max(W, rs.cols), CV_8UC3, cv::Scalar(0, 0, 0));
            rs.copyTo(padded(cv::Rect(0, 0, rs.cols, rs.rows)));
            rs = padded;
        }
        int top = (rs.rows - H) / 2, left = (rs.cols - W) / 2;
        cv::Mat crop = rs(cv::Rect(left, top, W, H));
        for (int y = 0; y < H; ++y) {
            const uint8_t* row = crop.ptr<uint8_t>(y);
            for (int x = 0; x < W; ++x)
                for (int c = 0; c < 3; ++c)
                    buf[(((size_t)t * 3 + c) * H + y) * W + x] = (row[x * 3 + c] / 255.0f - p_->mean[c]) / p_->stdv[c];
        }
    }
    std::array<int64_t, 5> shape{1, T, 3, H, W};
    auto mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value in = Ort::Value::CreateTensor<float>(mem, buf.data(), buf.size(), shape.data(), shape.size());
    const char* in_names[] = {p_->in_name.c_str()};
    std::vector<const char*> on;
    for (auto& s : p_->out_names) on.push_back(s.c_str());
    auto outs = p_->session->Run(Ort::RunOptions{nullptr}, in_names, &in, 1, on.data(), on.size());

    Output o;
    size_t nl = outs[0].GetTensorTypeAndShapeInfo().GetElementCount();
    const float* lg = outs[0].GetTensorData<float>();
    double mx = *std::max_element(lg, lg + nl), sum = 0;
    o.probs.resize(nl);
    for (size_t i = 0; i < nl; ++i) { o.probs[i] = std::exp((double)lg[i] - mx); sum += o.probs[i]; }
    for (auto& v : o.probs) v /= sum;
    if (outs.size() > 1) {
        size_t np = outs[1].GetTensorTypeAndShapeInfo().GetElementCount();
        const float* pp = outs[1].GetTensorData<float>();
        o.pooled.assign(pp, pp + np);
    }
    return o;
}

static RawFrame make_raw(const cv::Mat& bgr, const std::optional<Roi>& roi) {
    cv::Mat rgb;
    cv::cvtColor(bgr, rgb, cv::COLOR_BGR2RGB);
    if (roi) {  // numpy-style slicing: clamp to frame
        int x0 = std::clamp(roi->x, 0, rgb.cols), y0 = std::clamp(roi->y, 0, rgb.rows);
        int x1 = std::clamp(roi->x + roi->w, 0, rgb.cols), y1 = std::clamp(roi->y + roi->h, 0, rgb.rows);
        if (x1 > x0 && y1 > y0) rgb = rgb(cv::Rect(x0, y0, x1 - x0, y1 - y0));
    }
    cv::Mat cont = rgb.isContinuous() ? rgb : rgb.clone();
    RawFrame rf;
    rf.h = cont.rows;
    rf.w = cont.cols;
    rf.rgb.assign(cont.data, cont.data + (size_t)cont.total() * 3);
    return rf;
}

// ============================================================================= post-processing / matching
Collapse collapse_matrix(const std::vector<std::string>& raw, const RenameMap& rename) {
    Collapse c;
    for (auto& n : raw) {
        std::string cn = canon(n, rename);
        auto it = std::find(c.uniq.begin(), c.uniq.end(), cn);
        if (it == c.uniq.end()) { c.uniq.push_back(cn); c.map.push_back((int)c.uniq.size() - 1); }
        else c.map.push_back((int)(it - c.uniq.begin()));
    }
    return c;
}

std::vector<std::vector<double>> collapse_probs(const std::vector<std::vector<double>>& probs, const Collapse& c) {
    std::vector<std::vector<double>> out(probs.size(), std::vector<double>(c.uniq.size(), 0.0));
    for (size_t i = 0; i < probs.size(); ++i)
        for (size_t j = 0; j < probs[i].size(); ++j) out[i][c.map[j]] += probs[i][j];
    return out;
}

SegResult windows_to_segments(const std::vector<WindowTime>& wt, const std::vector<std::vector<double>>& pc,
                              const std::vector<std::string>& uniq, const Args& a) {
    SegResult r;
    if (wt.empty()) return r;
    std::vector<double> starts, ends;
    for (size_t i = 0; i < wt.size(); ++i) {
        starts.push_back(wt[i].start);
        ends.push_back(wt[i].end);
        size_t best = std::max_element(pc[i].begin(), pc[i].end()) - pc[i].begin();
        double c = pc[i][best];
        std::string lab = uniq[best];
        if (a.confidence_threshold && c < *a.confidence_threshold) lab = UNCERTAIN;
        r.labels.push_back(lab);
        r.conf.push_back(c);
    }
    r.segs = apply_min_duration_filter(labels_to_segments(starts, ends, r.labels), a.min_segment_duration);
    return r;
}

std::optional<double> segment_confidence(const std::vector<WindowTime>& wt, const std::vector<double>& conf, const Segment& seg) {
    if (wt.empty() || conf.empty()) return std::nullopt;
    double sum = 0, all = 0;
    int n = 0;
    for (size_t i = 0; i < wt.size(); ++i) {
        double mid = (wt[i].start + wt[i].end) / 2;
        all += conf[i];
        if (mid >= seg.start_sec && mid <= seg.end_sec) { sum += conf[i]; ++n; }
    }
    return n ? sum / n : all / conf.size();
}

// ----------------------------------------------------------------------------- Step-2 post-processing (README section 6)
static std::string det_dump(const std::vector<Detected>& v) {
    std::string o;
    for (auto& d : v)
        o += fmt("      %-12s %7.2f - %7.2f  (%.2fs)  conf=%.3f\n", d.label.c_str(), d.start, d.end, d.end - d.start,
                 d.conf ? *d.conf : -1.0);
    return o.empty() ? "      (none)\n" : o;
}

std::vector<Detected> postprocess_segments(const std::vector<WindowTime>& wt, const std::vector<double>& conf,
                                           const std::vector<Segment>& segs, const PostCfg& cfg, PostDebug* dbg) {
    PostDebug local;
    PostDebug& D = dbg ? *dbg : local;
    D = PostDebug{};

    // everything before filtering (kept for debug / results.json)
    for (auto& s : segs) D.unfiltered.push_back({s.label, s.start_sec, s.end_sec, segment_confidence(wt, conf, s)});

    // step 2 (tail): drop "uncertain" segments; step 3: segment confidence over ALL windows
    std::vector<Detected> cur;
    for (auto& d : D.unfiltered) if (d.label != UNCERTAIN) cur.push_back(d);
    D.after_drop_unc = cur;

    // step 4: merge overlapping same-label segments (end = max, conf = MAX)
    if (cfg.do_merge && !cur.empty()) {
        std::stable_sort(cur.begin(), cur.end(), [](const Detected& a, const Detected& b) { return a.start < b.start; });
        std::vector<Detected> merged;
        for (auto& d : cur) {
            if (!merged.empty() && merged.back().label == d.label && d.start <= merged.back().end) {
                Detected& m = merged.back();
                m.end = std::max(m.end, d.end);
                m.conf = std::max(m.conf.value_or(0.0), d.conf.value_or(0.0));
            } else merged.push_back(d);
        }
        cur = merged;
    }
    D.after_merge = cur;

    // step 5: drop low-confidence segments
    if (cfg.do_conf_filter) {
        std::vector<Detected> keep;
        for (auto& d : cur) if (d.conf.value_or(0.0) >= cfg.seg_conf_thr) keep.push_back(d);
        cur = keep;
    }
    D.after_conf = cur;

    // step 6: step-order filter (DP, maximise sum of conf*duration over order-non-decreasing chains)
    if (cfg.do_order_filter && !cur.empty()) {
        auto order_idx = [&](const std::string& l) {
            auto it = std::find(cfg.order.begin(), cfg.order.end(), l);
            return it == cfg.order.end() ? -1 : (int)(it - cfg.order.begin());
        };
        std::vector<Detected> cand;
        for (auto& d : cur) (order_idx(d.label) >= 0 ? cand : D.dropped_unknown).push_back(d);
        std::stable_sort(cand.begin(), cand.end(), [](const Detected& a, const Detected& b) {
            return a.start != b.start ? a.start < b.start : a.end < b.end;
        });
        const int n = (int)cand.size();
        std::vector<double> w(n), best(n);
        std::vector<int> prev(n, -1), oi(n);
        for (int i = 0; i < n; ++i) {
            w[i] = cand[i].conf.value_or(0.0) * (cand[i].end - cand[i].start);
            oi[i] = order_idx(cand[i].label);
        }
        const double EPS = 1e-12;
        for (int i = 0; i < n; ++i) {
            int bj = -1;
            double bv = 0.0;  // max(0, best[j])
            for (int j = 0; j < i; ++j) {
                if (oi[j] > oi[i]) continue;
                if (best[j] > bv + EPS || (bj >= 0 && std::fabs(best[j] - bv) <= EPS && best[j] > 0)) { bv = best[j]; bj = j; }  // tie -> later j
            }
            best[i] = w[i] + bv;
            prev[i] = bj;
        }
        std::vector<Detected> chain;
        if (n > 0) {
            int e = 0;
            for (int i = 1; i < n; ++i) if (best[i] >= best[e] - EPS) e = i;  // tie -> later i
            for (int i = e; i >= 0; i = prev[i]) chain.push_back(cand[i]);
            std::reverse(chain.begin(), chain.end());
        }
        cur = chain;
    }
    D.final_ = cur;
    return cur;
}


// ============================================================================= live RTSP monitor
using Clock = std::chrono::steady_clock;

static std::atomic<bool> g_stop{false};
static void on_signal(int) { g_stop = true; }

static std::mutex g_log_m;

// Session clock: t = 0 at the arrival of the first frame; all window times are on this timeline.
struct SessionClock {
    std::atomic<bool> started{false};
    Clock::time_point t0;
    std::chrono::system_clock::time_point wall0;
    double now_t() const { return std::chrono::duration<double>(Clock::now() - t0).count(); }
};
static SessionClock g_clk;

static std::string hms(double t) {
    if (t < 0) t = 0;
    int h = (int)(t / 3600), m = (int)((t - h * 3600) / 60);
    double s = t - h * 3600 - m * 60;
    return fmt("%02d:%02d:%04.1f", h, m, s);
}
static std::string wall_of(std::chrono::system_clock::time_point tp) {
    std::time_t tt = std::chrono::system_clock::to_time_t(tp);
    int ms = (int)(std::chrono::duration_cast<std::chrono::milliseconds>(tp.time_since_epoch()).count() % 1000);
    std::tm tmv;
    localtime_r(&tt, &tmv);
    char b[32];
    std::strftime(b, sizeof b, "%H:%M:%S", &tmv);
    return fmt("%s.%01d", b, ms / 100);
}
// wall-clock time of session time t
static std::string wall_str(double t) {
    auto tp = g_clk.wall0 + std::chrono::duration_cast<std::chrono::system_clock::duration>(std::chrono::duration<double>(t));
    return wall_of(tp);
}
// "13:45:02.3 (t+00:01:23.0)"
static std::string stamp(double t) { return fmt("%s (t+%s)", wall_str(t).c_str(), hms(t).c_str()); }

static void say(const std::string& msg) {
    std::lock_guard<std::mutex> lk(g_log_m);
    std::printf("[%s] %s\n", wall_of(std::chrono::system_clock::now()).c_str(), msg.c_str());
    std::fflush(stdout);
}

// ----- frame buffer (ROI-cropped RGB frames stamped with session time)
struct StampedFrame {
    double t;
    std::shared_ptr<const RawFrame> f;
};

class FrameBuffer {
public:
    explicit FrameBuffer(double keep_sec) : keep_(keep_sec) {}
    void push(double t, RawFrame&& f) {
        auto sp = std::make_shared<const RawFrame>(std::move(f));
        std::lock_guard<std::mutex> lk(m_);
        q_.push_back({t, std::move(sp)});
        while (!q_.empty() && q_.front().t < t - keep_) q_.pop_front();
        last_t_ = t;
    }
    double last_t() const {
        std::lock_guard<std::mutex> lk(m_);
        return last_t_;
    }
    // nearest buffered frame to time t; false if empty
    bool nearest(double t, StampedFrame& out) const {
        std::lock_guard<std::mutex> lk(m_);
        if (q_.empty()) return false;
        auto it = std::lower_bound(q_.begin(), q_.end(), t, [](const StampedFrame& a, double v) { return a.t < v; });
        if (it == q_.end()) out = q_.back();
        else if (it == q_.begin()) out = *it;
        else {
            auto pv = std::prev(it);
            out = (t - pv->t <= it->t - t) ? *pv : *it;
        }
        return true;
    }

private:
    double keep_;
    mutable std::mutex m_;
    std::deque<StampedFrame> q_;
    double last_t_ = -1;
};

static void grabber_loop(const Args& a, const std::optional<Roi>& roi, FrameBuffer& buf) {
    bool first_ever = true;
    while (!g_stop) {
        say("connecting to " + a.live.rtsp_url + " ...");
        cv::VideoCapture cap;
        if (!cap.open(a.live.rtsp_url, cv::CAP_FFMPEG)) {
            say(fmt("[warn] cannot open stream, retrying in %.1f s", a.live.reconnect_sec));
            for (double w = 0; w < a.live.reconnect_sec && !g_stop; w += 0.1) std::this_thread::sleep_for(std::chrono::milliseconds(100));
            continue;
        }
        cap.set(cv::CAP_PROP_BUFFERSIZE, 1);
        say(fmt("stream connected (reported fps=%.1f)", cap.get(cv::CAP_PROP_FPS)));
        cv::Mat bgr;
        bool size_checked = false;
        while (!g_stop) {
            if (!cap.read(bgr) || bgr.empty()) {
                say("[warn] stream read failed -> reconnecting");
                break;
            }
            auto now = Clock::now();
            if (first_ever) {
                g_clk.t0 = now;
                g_clk.wall0 = std::chrono::system_clock::now();
                g_clk.started.store(true, std::memory_order_release);
                first_ever = false;
                say("first frame received: session clock t=0 starts now");
            }
            if (!size_checked) {
                size_checked = true;
                say(fmt("frame size %dx%d", bgr.cols, bgr.rows));
                if (roi && (roi->x + roi->w > bgr.cols || roi->y + roi->h > bgr.rows))
                    say(fmt("[warn] ROI (%d,%d,%d,%d) exceeds the frame; it will be clamped", roi->x, roi->y, roi->w, roi->h));
            }
            buf.push(std::chrono::duration<double>(now - g_clk.t0).count(), make_raw(bgr, roi));
        }
        cap.release();
        if (!g_stop) std::this_thread::sleep_for(std::chrono::milliseconds((int)(a.live.reconnect_sec * 1000)));
    }
}

// ----- live step tracking
struct StepRec {
    std::string label;
    double start = 0, end = 0, conf = 0;
    bool completed = false;
};

class LiveMonitor {
public:
    LiveMonitor(const Args& a, const std::vector<std::string>& uniq, const Collapse& col)
        : a_(a), a2_(a), uniq_(uniq), col_(col) {
        if (a_.post.enable && !a_.confidence_threshold) a2_.confidence_threshold = a_.post.window_conf_thr;
        if (!a_.live.events_log.empty()) ev_.open(a_.live.events_log, std::ios::app);
    }

    void on_window(const WindowTime& w, const std::vector<double>& probs, double infer_sec) {
        wt_.push_back(w);
        pc_.push_back(collapse_probs(std::vector<std::vector<double>>{probs}, col_)[0]);
        if (first_win_t_ < 0) first_win_t_ = w.start;

        SegResult sr = windows_to_segments(wt_, pc_, uniq_, a2_);
        PostDebug dbg;
        std::vector<Detected> fin = detect(sr, dbg);
        const double last_end = w.end;
        const double gap = a_.live.complete_gap;

        if (a_.post.debug) {
            std::string s = fmt("win %s..%s  label=%s conf=%.3f  (infer %.0f ms)", hms(w.start).c_str(), hms(w.end).c_str(),
                                sr.labels.back().c_str(), sr.conf.back(), infer_sec * 1000);
            say(s);
            if (a_.post.enable) {
                std::lock_guard<std::mutex> lk(g_log_m);
                std::printf("    unfiltered (%zu):\n%s    after merge (%zu) | after conf (%zu) | final (%zu):\n%s", dbg.unfiltered.size(),
                            det_dump(dbg.unfiltered).c_str(), dbg.after_merge.size(), dbg.after_conf.size(), dbg.final_.size(),
                            det_dump(dbg.final_).c_str());
                std::fflush(stdout);
            }
        }

        update_steps(fin, last_end);

        bool reset = false;
        if (!fin.empty() && !a_.post.order.empty() && fin.back().label == a_.post.order.back() &&
            last_end - fin.back().end >= gap - 1e-6) {
            finish_cycle("COMPLETE", fin, last_end, fin.back().end);
            reset = true;
        } else if (!wt_.empty() && last_end - first_win_t_ > a_.live.cycle_max_sec) {
            finish_cycle("TIMEOUT (cycle_max_sec)", fin, last_end, last_end + 1e9);
            reset = true;
        } else if (!fin.empty() && last_end - fin.back().end > a_.live.cycle_idle_sec) {
            finish_cycle("ABANDONED (no new step)", fin, last_end, last_end + 1e9);
            reset = true;
        } else if (fin.empty()) {
            // nothing confirmed: keep only a sliding horizon so stale windows cannot pollute the next cycle
            double keep_from = last_end - a_.live.cycle_idle_sec - a_.window_len;
            size_t cut = 0;
            while (cut < wt_.size() && wt_[cut].start < keep_from) ++cut;
            if (cut) erase_prefix(cut);
            steps_.clear();
            pending_.clear();
        }

        double nt = g_clk.now_t();
        if (a_.live.status_interval > 0 && (last_status_ < 0 || nt - last_status_ >= a_.live.status_interval)) {
            last_status_ = nt;
            if (!reset) status(sr, fin, last_end, infer_sec);
            else say(fmt("CYCLE %d | waiting for the next cycle to start", cycle_id_));
        }
    }

    void on_shutdown() {
        if (wt_.empty()) return;
        SegResult sr;
        PostDebug dbg;
        auto fin = detect(sr, dbg);
        if (fin.empty()) {
            say(fmt("stopped. CYCLE %d: no confirmed step.", cycle_id_));
            return;
        }
        double last_end = wt_.back().end;
        const Detected& L = fin.back();
        if (last_end - L.end < a_.live.complete_gap - 1e-6)
            say(fmt("stopped while IN STEP %s (started %s, %.1f s so far)", L.label.c_str(), stamp(L.start).c_str(), L.end - L.start));
        print_summary("INTERRUPTED", fin);
    }

private:
    const Args& a_;
    Args a2_;
    const std::vector<std::string>& uniq_;
    const Collapse& col_;
    std::vector<WindowTime> wt_;
    std::vector<std::vector<double>> pc_;
    std::vector<StepRec> steps_;  // announced steps of the current cycle
    std::map<std::string, int> pending_;  // label -> consecutive updates seen but not yet announced
    double first_win_t_ = -1, last_status_ = -1;
    int cycle_id_ = 1;
    std::ofstream ev_;

    std::vector<Detected> detect(SegResult& sr, PostDebug& dbg) {
        sr = windows_to_segments(wt_, pc_, uniq_, a2_);
        if (a_.post.enable) return postprocess_segments(wt_, sr.conf, sr.segs, a_.post, &dbg);
        std::vector<Detected> det;  // --no_postproc: raw segments minus "uncertain"
        for (auto& s : sr.segs)
            if (s.label != UNCERTAIN) det.push_back({s.label, s.start_sec, s.end_sec, segment_confidence(wt_, sr.conf, s)});
        return det;
    }

    void erase_prefix(size_t n) {
        wt_.erase(wt_.begin(), wt_.begin() + n);
        pc_.erase(pc_.begin(), pc_.begin() + n);
        first_win_t_ = wt_.empty() ? -1 : wt_.front().start;
    }

    void emit(json j) {
        if (!ev_.is_open()) return;
        j["wall"] = wall_of(std::chrono::system_clock::now());
        j["cycle"] = cycle_id_;
        ev_ << j.dump() << "\n";
        ev_.flush();
    }

    void update_steps(const std::vector<Detected>& fin, double last_end) {
        std::vector<bool> matched(steps_.size(), false);
        std::set<std::string> present;
        for (size_t k = 0; k < fin.size(); ++k) {
            const Detected& f = fin[k];
            const bool is_last = k + 1 == fin.size();
            const bool done = !is_last || (last_end - f.end >= a_.live.complete_gap - 1e-6);
            int idx = -1;
            for (size_t i = 0; i < steps_.size(); ++i)
                if (!matched[i] && steps_[i].label == f.label && f.start <= steps_[i].end + 0.5 && f.end >= steps_[i].start - 0.5) {
                    idx = (int)i;
                    break;
                }
            if (idx < 0) {
                // debounce: announce only after the step was present in `confirm_updates` consecutive window updates
                present.insert(f.label);
                if (++pending_[f.label] < a_.live.confirm_updates) continue;
                pending_.erase(f.label);
                steps_.push_back({f.label, f.start, f.end, f.conf.value_or(0.0), false});
                matched.push_back(true);
                idx = (int)steps_.size() - 1;
                say(fmt(">> STEP %-12s STARTED   at %s  (conf %.2f)", f.label.c_str(), stamp(f.start).c_str(), f.conf.value_or(0.0)));
                emit({{"event", "step_start"}, {"label", f.label}, {"start", f.start}, {"conf", f.conf.value_or(0.0)}});
            } else matched[idx] = true;
            StepRec& r = steps_[idx];
            if (r.completed) continue;  // finished steps are not re-opened
            r.start = f.start;
            r.end = f.end;
            r.conf = f.conf.value_or(0.0);
            if (done) {
                r.completed = true;
                say(fmt("<< STEP %-12s FINISHED  %s -> %s  duration %.1f s  (conf %.2f)", r.label.c_str(), stamp(r.start).c_str(),
                        wall_str(r.end).c_str(), r.end - r.start, r.conf));
                emit({{"event", "step_end"}, {"label", r.label}, {"start", r.start}, {"end", r.end}, {"duration", r.end - r.start},
                      {"conf", r.conf}});
            }
        }
        for (auto it = pending_.begin(); it != pending_.end();)
            it = present.count(it->first) ? std::next(it) : pending_.erase(it);
        // announced but no longer present and not finished -> the filters removed it again
        for (int i = (int)steps_.size() - 1; i >= 0; --i)
            if (i < (int)matched.size() && !matched[i] && !steps_[i].completed) {
                say(fmt("xx STEP %-12s RETRACTED (removed by post-processing)", steps_[i].label.c_str()));
                emit({{"event", "step_retracted"}, {"label", steps_[i].label}});
                steps_.erase(steps_.begin() + i);
            }
    }

    void print_summary(const std::string& kind, const std::vector<Detected>& fin) {
        std::string s = fmt("=== CYCLE %d %s ===  %zu step(s) confirmed", cycle_id_, kind.c_str(), fin.size());
        if (!fin.empty()) s += fmt(", span %s -> %s, %.1f s total", stamp(fin.front().start).c_str(), wall_str(fin.back().end).c_str(),
                                   fin.back().end - fin.front().start);
        say(s);
        {
            std::lock_guard<std::mutex> lk(g_log_m);
            int n = 0;
            for (auto& d : fin)
                std::printf("      %2d. %-12s %s -> %s  %6.1f s  conf %.2f\n", ++n, d.label.c_str(), wall_str(d.start).c_str(),
                            wall_str(d.end).c_str(), d.end - d.start, d.conf.value_or(0.0));
            std::string missing;
            for (auto& o : a_.post.order)
                if (std::none_of(fin.begin(), fin.end(), [&](const Detected& d) { return d.label == o; })) missing += (missing.empty() ? "" : ", ") + o;
            if (!missing.empty()) std::printf("      not detected in this cycle: %s\n", missing.c_str());
            std::fflush(stdout);
        }
    }

    void finish_cycle(const std::string& kind, const std::vector<Detected>& fin, double last_end, double cut_t) {
        (void)last_end;
        print_summary(kind, fin);
        json steps = json::array();
        for (auto& d : fin) steps.push_back({{"label", d.label}, {"start", d.start}, {"end", d.end}, {"duration", d.end - d.start}, {"conf", d.conf.value_or(0.0)}});
        emit({{"event", "cycle_end"}, {"kind", kind}, {"steps", steps}});
        size_t cut = 0;
        while (cut < wt_.size() && wt_[cut].start < cut_t) ++cut;
        erase_prefix(cut);
        steps_.clear();
        pending_.clear();
        ++cycle_id_;
    }

    void status(const SegResult& sr, const std::vector<Detected>& fin, double last_end, double infer_sec) {
        std::string s = fmt("CYCLE %d | ", cycle_id_);
        if (!fin.empty()) {
            const Detected& L = fin.back();
            if (last_end - L.end < a_.live.complete_gap - 1e-6)
                s += fmt("IN STEP %s since %s (%.1f s so far, conf %.2f)", L.label.c_str(), wall_str(L.start).c_str(), L.end - L.start,
                         L.conf.value_or(0.0));
            else s += fmt("last step %s finished at %s; between steps", L.label.c_str(), wall_str(L.end).c_str());
            s += fmt(" | %zu step(s) confirmed", fin.size());
        } else {
            s += fmt("no confirmed step yet | latest window: %s (%.2f)", sr.labels.back().c_str(), sr.conf.back());
        }
        s += fmt(" | infer %.0f ms/win", infer_sec * 1000);
        say(s);
    }
};

// ----- main loop
int run_live(const Args& a) {
    RenameMap rename;
    if (a.merge_step6) rename[DEFAULT_STEP6_RENAME_FROM] = DEFAULT_STEP6_RENAME_TO;

    std::optional<Roi> roi = resolve_roi(a.videomae_dir, a.roi);

    std::vector<std::string> raw_names;
    {
        std::ifstream f(fs::path(a.videomae_dir) / "config.json");
        if (!f) throw std::runtime_error("cannot open config.json in " + a.videomae_dir);
        json cfg = json::parse(f);
        auto& id2 = cfg.at("id2label");
        for (size_t i = 0; i < id2.size(); ++i) raw_names.push_back(id2.at(std::to_string(i)).get<std::string>());
    }
    Collapse col = collapse_matrix(raw_names, rename);
    const auto& uniq = col.uniq;
    auto join = [](const std::vector<std::string>& v) {
        std::string s = "[";
        for (size_t i = 0; i < v.size(); ++i) s += (i ? ", '" : "'") + v[i] + "'";
        return s + "]";
    };
    std::printf("Model classes (%zu): %s\n", raw_names.size(), join(raw_names).c_str());
    std::printf("Step labels    (%zu): %s   (rename map: %s)\n", uniq.size(), join(uniq).c_str(), rename_repr(rename).c_str());
    std::printf("Step order: %s\n", join(a.post.order).c_str());
    for (auto& o : a.post.order)
        if (std::find(uniq.begin(), uniq.end(), o) == uniq.end())
            std::printf("  [WARNING] order label '%s' is not one of the model's labels\n", o.c_str());
    std::printf("Window %.2fs / stride %.2fs / %d frames, min segment %.2fs, post-processing %s\n", a.window_len, a.stride, a.num_frames,
                a.min_segment_duration, a.post.enable ? "ON" : "OFF");

    VideoMAEModel model(a.videomae_dir, a.onnx_path, a.use_cuda);
    std::printf("Inference device: %s\n", model.device().c_str());
    if (a.use_cuda && model.device() != "cuda") std::printf("  [WARNING] CUDA requested but running on %s\n", model.device().c_str());

    setenv("OPENCV_FFMPEG_CAPTURE_OPTIONS", a.live.rtsp_tcp ? "rtsp_transport;tcp" : "rtsp_transport;udp", 1);
    std::signal(SIGINT, on_signal);
    std::signal(SIGTERM, on_signal);

    FrameBuffer buf(a.live.buffer_sec);
    std::thread grabber(grabber_loop, std::cref(a), std::cref(roi), std::ref(buf));
    LiveMonitor mon(a, uniq, col);

    long k = 0;
    long skipped = 0;
    double skip_from = 0, skip_to = 0;
    bool announced_wait = false;
    while (!g_stop) {
        if (!g_clk.started.load(std::memory_order_acquire)) {
            if (!announced_wait) { say("waiting for the first frame ..."); announced_wait = true; }
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            continue;
        }
        if (a.live.max_runtime_sec > 0 && g_clk.now_t() > a.live.max_runtime_sec) { say("max_runtime_sec reached, stopping"); break; }
        const double ws = k * a.stride, we = ws + a.window_len;
        if (buf.last_t() < we) {
            std::this_thread::sleep_for(std::chrono::milliseconds(20));
            continue;
        }
        // frames: same sampling as offline (linspace(start, end - 1/fps, num_frames)), nearest buffered frame by timestamp
        std::vector<RawFrame> frames;
        frames.reserve(a.num_frames);
        bool ok = true;
        for (int i = 0; i < a.num_frames && ok; ++i) {
            double tt = a.num_frames == 1 ? ws : ws + (a.window_len - 1.0 / a.fps) * i / (a.num_frames - 1);
            StampedFrame sf;
            if (!buf.nearest(tt, sf) || std::fabs(sf.t - tt) > a.live.max_frame_gap) ok = false;
            else frames.push_back(*sf.f);
        }
        if (!ok) {
            if (skipped == 0) skip_from = ws;
            skip_to = we;
            ++skipped;
            ++k;
            continue;
        }
        if (skipped) {
            say(fmt("[warn] %ld window(s) skipped (no frames: stream gap or processing lag) between %s and %s", skipped,
                    hms(skip_from).c_str(), hms(skip_to).c_str()));
            skipped = 0;
        }
        auto m0 = Clock::now();
        auto out = model.run(frames);
        double infer = std::chrono::duration<double>(Clock::now() - m0).count();
        double lag = g_clk.now_t() - we;
        if (lag > 3 * a.stride) say(fmt("[warn] processing lag %.1f s behind real time (infer %.0f ms)", lag, infer * 1000));
        mon.on_window({ws, we}, out.probs, infer);
        ++k;
    }
    g_stop = true;
    grabber.join();
    say("shutting down");
    mon.on_shutdown();
    return 0;
}

// ============================================================================= CLI
Args parse_args(int argc, char** argv) {
    Args a;
    auto usage = [&] {
        std::printf(
            "Usage: %s --videomae_dir DIR [--rtsp URL] [--onnx_path FILE] [--udp] [--cpu]\n"
            "  [--window_len S] [--stride S] [--fps F] [--num_frames N] [--min_segment_duration S]\n"
            "  [--confidence_threshold P] [--roi x,y,w,h] [--merge_step6]\n"
            "  post-processing: [--no_postproc] [--post_seg_conf P] [--post_no_merge] [--post_no_conf]\n"
            "                   [--post_no_order] [--post_order a,b,c,...] [--post_debug]\n"
            "  live: [--buffer_sec S] [--status_interval S] [--complete_gap S] [--cycle_max_sec S] [--cycle_idle_sec S]\n"
            "        [--confirm_updates N] [--max_frame_gap S] [--reconnect_sec S] [--max_runtime_sec S] [--events_log FILE.jsonl]\n"
            "  (default stream: %s, CUDA on by default)\n",
            argv[0], a.live.rtsp_url.c_str());
    };
    for (int i = 1; i < argc; ++i) {
        std::string k = argv[i], v;
        bool inline_v = false;
        if (starts_with(k, "--") && k.find('=') != std::string::npos) {
            v = k.substr(k.find('=') + 1);
            k = k.substr(0, k.find('='));
            inline_v = true;
        }
        auto val = [&]() -> std::string {
            if (inline_v) return v;
            if (i + 1 >= argc) throw std::runtime_error("missing value for " + k);
            return argv[++i];
        };
        if (k == "-h" || k == "--help") { usage(); std::exit(0); }
        else if (k == "--videomae_dir") a.videomae_dir = val();
        else if (k == "--onnx_path") a.onnx_path = val();
        else if (k == "--rtsp") a.live.rtsp_url = val();
        else if (k == "--udp") a.live.rtsp_tcp = false;
        else if (k == "--window_len") a.window_len = std::stod(val());
        else if (k == "--stride") a.stride = std::stod(val());
        else if (k == "--fps") a.fps = std::stod(val());
        else if (k == "--num_frames") a.num_frames = std::stoi(val());
        else if (k == "--min_segment_duration") a.min_segment_duration = std::stod(val());
        else if (k == "--confidence_threshold") a.confidence_threshold = std::stod(val());
        else if (k == "--roi") a.roi = val();
        else if (k == "--merge_step6") a.merge_step6 = true;
        else if (k == "--cuda") a.use_cuda = true;
        else if (k == "--cpu") a.use_cuda = false;
        else if (k == "--no_postproc") a.post.enable = false;
        else if (k == "--post_seg_conf") a.post.seg_conf_thr = std::stod(val());
        else if (k == "--post_no_merge") a.post.do_merge = false;
        else if (k == "--post_no_conf") a.post.do_conf_filter = false;
        else if (k == "--post_no_order") a.post.do_order_filter = false;
        else if (k == "--post_debug") a.post.debug = true;
        else if (k == "--post_order") {
            a.post.order.clear();
            std::stringstream ss(val());
            std::string p;
            while (std::getline(ss, p, ',')) if (!trim(p).empty()) a.post.order.push_back(trim(p));
        }
        else if (k == "--buffer_sec") a.live.buffer_sec = std::stod(val());
        else if (k == "--status_interval") a.live.status_interval = std::stod(val());
        else if (k == "--complete_gap") a.live.complete_gap = std::stod(val());
        else if (k == "--cycle_max_sec") a.live.cycle_max_sec = std::stod(val());
        else if (k == "--cycle_idle_sec") a.live.cycle_idle_sec = std::stod(val());
        else if (k == "--max_frame_gap") a.live.max_frame_gap = std::stod(val());
        else if (k == "--reconnect_sec") a.live.reconnect_sec = std::stod(val());
        else if (k == "--max_runtime_sec") a.live.max_runtime_sec = std::stod(val());
        else if (k == "--events_log") a.live.events_log = val();
        else if (k == "--confirm_updates") a.live.confirm_updates = std::stoi(val());
        else { usage(); throw std::runtime_error("unknown argument: " + k); }
    }
    if (a.videomae_dir.empty()) { usage(); throw std::runtime_error("--videomae_dir is required"); }
    if (a.onnx_path.empty()) a.onnx_path = (fs::path(a.videomae_dir) / "model.onnx").string();
    if (a.num_frames < 1 || a.window_len <= 0 || a.stride <= 0 || a.fps <= 0) throw std::runtime_error("invalid window/stride/fps/num_frames");
    return a;
}

}  // namespace vmae

int main(int argc, char** argv) {
    try {
        return vmae::run_live(vmae::parse_args(argc, argv));
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
}
