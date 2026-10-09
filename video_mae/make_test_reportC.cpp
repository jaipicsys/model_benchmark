// make_test_report.cpp
// C++ port of make_test_report.py + extract_embeddings.py + segment_utils.py.
// See make_test_report.hpp for dependencies and the ONNX export snippet.
//
// Build example:
//   g++ -std=c++17 -O2 make_test_report.cpp -o make_test_report \
//       $(pkg-config --cflags --libs opencv4) -lonnxruntime -lxlnt -lzip -lpugixml
//
// USAGE (same flags as the Python script, plus --onnx_path / --cuda):
//   ./make_test_report --videomae_dir runs/roi_v4/videomae_finetuned/best_model \
//       --individual_dir videos/ --continuous_dir caa_20cycles/ --gt cycle20GT.ods \
//       --template temporal_test_template.xlsx --window_len 2.0 --stride 1.0 --fps 15 \
//       --num_frames 16 --min_segment_duration 2.5 --confidence_threshold 0.7

#include "make_test_reportC.hpp"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <numeric>
#include <regex>
#include <set>
#include <sstream>
#include <stdexcept>

#include <nlohmann/json.hpp>
#include <onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>
#include <pugixml.hpp>
#include <xlnt/xlnt.hpp>
#include <zip.h>

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
static const std::regex FILE_RE(R"(^((?:(?:start|stop)_)?step_?\d+(?:\.\d+)?)_(?:op\d+_)?\d+$)", std::regex::icase);

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

std::optional<std::string> label_from_filename(const std::string& path, const RenameMap& rename) {
    std::string stem = fs::path(path).stem().string();
    std::smatch m;
    if (std::regex_match(stem, m, FILE_RE)) return canon(m[1].str(), rename);
    return std::nullopt;
}

// ============================================================================= ground truth
struct GCell {
    enum Kind { Empty, Num, Str } kind = Empty;
    double num = 0;
    std::string str;
};
using Grid = std::vector<std::vector<GCell>>;

static std::string cell_str(const GCell& c) {
    if (c.kind == GCell::Str) return c.str;
    if (c.kind == GCell::Num) return pyfloat(c.num);
    return "nan";
}

static std::optional<double> str_to_sec(const std::string& s) {
    try {
        std::stringstream ss(trim(s));
        std::string part;
        double sec = 0;
        while (std::getline(ss, part, ':')) sec = sec * 60 + std::stod(part);
        return sec;
    } catch (...) {
        return std::nullopt;
    }
}
static std::optional<double> to_sec(const GCell& c) {
    if (c.kind == GCell::Empty) return std::nullopt;
    if (c.kind == GCell::Num) return c.num;
    return str_to_sec(c.str);
}

static double parse_iso_duration(const std::string& s) {  // PT01H02M03.5S
    double total = 0;
    std::string num;
    for (char ch : s) {
        if (std::isdigit((unsigned char)ch) || ch == '.') num += ch;
        else if (ch == 'D' && !num.empty()) { total += std::stod(num) * 86400; num.clear(); }
        else if (ch == 'H' && !num.empty()) { total += std::stod(num) * 3600; num.clear(); }
        else if (ch == 'M' && !num.empty()) { total += std::stod(num) * 60; num.clear(); }
        else if (ch == 'S' && !num.empty()) { total += std::stod(num); num.clear(); }
        else num.clear();
    }
    return total;
}

static std::string node_text(const pugi::xml_node& n) {
    std::string out;
    for (auto c : n.children()) {
        if (c.type() == pugi::node_pcdata || c.type() == pugi::node_cdata) out += c.value();
        else out += node_text(c);
    }
    return out;
}

static Grid read_ods(const std::string& path) {
    int err = 0;
    zip_t* z = zip_open(path.c_str(), ZIP_RDONLY, &err);
    if (!z) throw std::runtime_error("cannot open ODS (zip): " + path);
    zip_stat_t st;
    if (zip_stat(z, "content.xml", 0, &st) != 0) { zip_close(z); throw std::runtime_error("no content.xml in " + path); }
    std::string buf(st.size, '\0');
    zip_file_t* f = zip_fopen(z, "content.xml", 0);
    zip_fread(f, buf.data(), st.size);
    zip_fclose(f);
    zip_close(z);

    pugi::xml_document doc;
    if (!doc.load_buffer(buf.data(), buf.size())) throw std::runtime_error("cannot parse ODS content.xml");
    auto table = doc.child("office:document-content").child("office:body").child("office:spreadsheet").child("table:table");
    if (!table) throw std::runtime_error("ODS has no sheet");

    Grid g;
    size_t ncols = 0;
    std::function<void(pugi::xml_node)> walk = [&](pugi::xml_node parent) {
        for (auto row : parent.children()) {
            std::string nm = row.name();
            if (nm == "table:table-header-rows" || nm == "table:table-row-group" || nm == "table:table-rows") {
                walk(row);
                continue;
            }
            if (nm != "table:table-row") continue;
            int rrep = row.attribute("table:number-rows-repeated").as_int(1);
            std::vector<GCell> cells;
            for (auto c : row.children()) {
                std::string cn = c.name();
                if (cn != "table:table-cell" && cn != "table:covered-table-cell") continue;
                int crep = c.attribute("table:number-columns-repeated").as_int(1);
                GCell cell;
                if (cn == "table:table-cell") {
                    std::string vt = c.attribute("office:value-type").as_string();
                    if (vt == "float" || vt == "percentage" || vt == "currency") {
                        cell.kind = GCell::Num;
                        cell.num = c.attribute("office:value").as_double();
                    } else if (vt == "time") {
                        cell.kind = GCell::Num;
                        cell.num = parse_iso_duration(c.attribute("office:time-value").as_string());
                    } else if (!vt.empty()) {
                        cell.kind = GCell::Str;
                        for (auto p : c.children("text:p")) {
                            if (!cell.str.empty()) cell.str += "\n";
                            cell.str += node_text(p);
                        }
                    }
                }
                crep = std::min(crep, cell.kind == GCell::Empty ? 256 : 1024);
                for (int k = 0; k < crep; ++k) cells.push_back(cell);
            }
            while (!cells.empty() && cells.back().kind == GCell::Empty) cells.pop_back();
            int reps = cells.empty() ? 1 : std::min(rrep, 1000);
            for (int k = 0; k < reps; ++k) {
                g.push_back(cells);
                ncols = std::max(ncols, cells.size());
            }
        }
    };
    walk(table);
    for (auto& r : g) r.resize(ncols);
    return g;
}

static Grid read_xlsx_grid(const std::string& path) {
    xlnt::workbook wb;
    wb.load(path);
    auto ws = wb.active_sheet();
    size_t nr = ws.highest_row(), nc = ws.highest_column().index;
    Grid g(nr, std::vector<GCell>(nc));
    for (size_t r = 0; r < nr; ++r)
        for (size_t c = 0; c < nc; ++c) {
            xlnt::cell_reference ref((xlnt::column_t::index_t)(c + 1), (xlnt::row_t)(r + 1));
            if (!ws.has_cell(ref)) continue;
            auto cell = ws.cell(ref);
            if (!cell.has_value()) continue;
            GCell& out = g[r][c];
            if (cell.is_date()) {  // time-of-day cells -> seconds
                double d = cell.value<double>();
                out.kind = GCell::Num;
                out.num = round_to((d - std::floor(d)) * 86400.0, 3);
            } else if (cell.data_type() == xlnt::cell::type::number) {
                out.kind = GCell::Num;
                out.num = cell.value<double>();
            } else {
                out.kind = GCell::Str;
                out.str = cell.to_string();
            }
        }
    return g;
}

GtResult load_gt(const std::string& path, const RenameMap& rename) {
    Grid df = ends_with(lower(path), ".ods") ? read_ods(path) : read_xlsx_grid(path);
    if (df.size() < 3) throw std::runtime_error("GT sheet too small");
    const size_t ncols = df[0].size();
    const auto& hdr = df[0];
    const auto& sub = df[1];

    int cyc_col = -1;
    for (size_t c = 0; c < ncols; ++c)
        if (lower(trim(cell_str(hdr[c]))) == "cycle") { cyc_col = (int)c; break; }
    if (cyc_col < 0) throw std::runtime_error("GT: no 'cycle' column in header row");

    std::vector<std::pair<std::string, int>> step_cols;
    for (size_t c = cyc_col + 1; c + 1 < ncols; ++c) {
        if (hdr[c].kind != GCell::Str || trim(hdr[c].str).empty()) continue;
        if (!starts_with(lower(trim(cell_str(sub[c]))), "from")) continue;
        if (lower(trim(cell_str(sub[c + 1]))) != "to") continue;
        std::string h = trim(hdr[c].str);
        // split at first '-', en dash or em dash
        size_t cut = std::string::npos;
        for (const char* d : {"-", "\xE2\x80\x93", "\xE2\x80\x94"}) {
            size_t p = h.find(d);
            if (p != std::string::npos) cut = std::min(cut, p);
        }
        std::string name = trim(cut == std::string::npos ? h : h.substr(0, cut));
        step_cols.emplace_back(canon(name, rename), (int)c);
    }

    GtResult res;
    for (size_t r = 2; r < df.size(); ++r) {
        const GCell& cc = df[r][cyc_col];
        int cyc;
        if (cc.kind == GCell::Num) cyc = (int)cc.num;
        else if (cc.kind == GCell::Str) {
            try { cyc = std::stoi(cc.str); } catch (...) { continue; }
        } else continue;

        std::vector<GtSegment> segs;
        for (auto& [lab, c] : step_cols) {
            auto s = to_sec(df[r][c]);
            auto e = to_sec(df[r][c + 1]);
            if (!s || !e || *e <= *s) {
                std::printf("  [GT warning] cycle %d %s: bad/missing times (%s -> %s), skipped\n", cyc, lab.c_str(),
                            cell_str(df[r][c]).c_str(), cell_str(df[r][c + 1]).c_str());
                continue;
            }
            segs.push_back({lab, *s, *e});
        }
        std::stable_sort(segs.begin(), segs.end(), [](const GtSegment& a, const GtSegment& b) {
            return a.label != b.label ? a.label < b.label : a.start < b.start;
        });
        std::vector<GtSegment> merged;
        for (auto& sg : segs) {
            if (!merged.empty() && merged.back().label == sg.label && sg.start <= merged.back().end + GT_MERGE_GAP)
                merged.back().end = std::max(merged.back().end, sg.end);
            else merged.push_back(sg);
        }
        std::stable_sort(merged.begin(), merged.end(), [](const GtSegment& a, const GtSegment& b) { return a.start < b.start; });
        res.gt[cyc] = merged;
    }
    for (auto& sc : step_cols) res.labels.push_back(sc.first);
    return res;
}

// ============================================================================= segment_utils.py
std::vector<Segment> load_annotation(const std::string& json_path) {
    std::ifstream f(json_path);
    if (!f) throw std::runtime_error("cannot open " + json_path);
    json j = json::parse(f);
    std::vector<Segment> out;
    for (auto& s : j.at("segments"))
        out.push_back({s.at("label").get<std::string>(), s.at("start_sec").get<double>(), s.at("end_sec").get<double>()});
    return out;
}

std::vector<std::string> align_annotations_to_windows(const std::vector<double>& ws, const std::vector<double>& we,
                                                      const std::vector<Segment>& ann, const std::string& unknown) {
    std::vector<std::string> labels;
    for (size_t i = 0; i < ws.size(); ++i) {
        double center = (ws[i] + we[i]) / 2.0;
        std::string assigned = unknown;
        for (auto& seg : ann)
            if (seg.start_sec <= center && center < seg.end_sec) { assigned = seg.label; break; }
        labels.push_back(assigned);
    }
    return labels;
}

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

std::vector<std::string> majority_vote_smoothing(const std::vector<std::string>& labels, int radius) {
    std::vector<std::string> out;
    int n = (int)labels.size();
    for (int i = 0; i < n; ++i) {
        int lo = std::max(0, i - radius), hi = std::min(n, i + radius + 1);
        std::map<std::string, int> cnt;  // sorted like np.unique
        for (int k = lo; k < hi; ++k) cnt[labels[k]]++;
        std::string best;
        int bc = -1;
        for (auto& kv : cnt)
            if (kv.second > bc) { bc = kv.second; best = kv.first; }
        out.push_back(best);
    }
    return out;
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

SessionResult extract_session_embeddings(const std::string& video_path, VideoMAEModel& model, double window_len,
                                         double stride, double fps, int num_frames, const std::optional<Roi>& roi) {
    cv::VideoCapture cap(video_path);
    if (!cap.isOpened()) throw std::runtime_error("cannot open video: " + video_path);
    const long total_frames = (long)cap.get(cv::CAP_PROP_FRAME_COUNT);
    double native_fps = cap.get(cv::CAP_PROP_FPS);
    if (native_fps <= 0) native_fps = fps;
    const double duration = total_frames / native_fps;

    SessionResult res;
    double t = 0.0;
    while (t + window_len <= duration + 1e-6) {
        res.windows.push_back({t, t + window_len});
        t += stride;
    }
    if (res.windows.empty()) res.windows.push_back({0.0, duration});

    using clk = std::chrono::steady_clock;
    auto wall0 = clk::now();
    double model_only = 0;

    for (auto& w : res.windows) {
        long start_f = (long)(w.start * native_fps), end_f = (long)(w.end * native_fps);
        end_f = std::max(end_f, start_f + 1);
        std::vector<long> idx(num_frames);
        for (int i = 0; i < num_frames; ++i) {
            double v = num_frames == 1 ? (double)start_f
                                       : start_f + (double)(end_f - 1 - start_f) * i / (num_frames - 1);
            idx[i] = std::clamp((long)std::nearbyint(v), 0L, std::max(total_frames - 1, 0L));
        }
        // sequential read from idx.front() .. idx.back()
        std::vector<RawFrame> frames(num_frames);
        cap.set(cv::CAP_PROP_POS_FRAMES, (double)idx.front());
        cv::Mat cur;
        size_t k = 0;
        RawFrame last_ok;
        bool have = false;
        for (long f = idx.front(); f <= idx.back() && k < idx.size(); ++f) {
            if (!cap.grab()) break;
            if (idx[k] != f) continue;
            if (!cap.retrieve(cur) || cur.empty()) break;
            last_ok = make_raw(cur, roi);
            have = true;
            while (k < idx.size() && idx[k] == f) frames[k++] = last_ok;
        }
        if (!have) throw std::runtime_error("could not decode frames from " + video_path);
        for (; k < idx.size(); ++k) frames[k] = last_ok;  // decoder ended early: repeat last frame

        auto m0 = clk::now();
        auto out = model.run(frames);
        model_only += std::chrono::duration<double>(clk::now() - m0).count();
        res.probs.push_back(std::move(out.probs));
        if (!out.pooled.empty()) res.embeddings.push_back(std::move(out.pooled));
    }
    double wall = std::chrono::duration<double>(clk::now() - wall0).count();
    int n = (int)res.windows.size();
    res.timing = {model.device(), n, wall, model_only, wall / n, model_only / n};
    return res;
}

// ============================================================================= model fingerprint / extensions
const std::vector<std::string>& video_extensions() {
    static const std::vector<std::string> e{".mp4", ".avi", ".mov", ".mkv"};
    return e;
}

// NOTE: pipeline_runner.compute_model_fingerprint() was not part of the supplied files, so this is a
// substitute: FNV-1a over config.json + the ONNX file size. Values will differ from the Python ones.
std::string compute_model_fingerprint(const std::string& model_dir, const std::string& onnx_path) {
    uint64_t h = 1469598103934665603ULL;
    auto mix = [&](const unsigned char* d, size_t n) {
        for (size_t i = 0; i < n; ++i) { h ^= d[i]; h *= 1099511628211ULL; }
    };
    std::ifstream cf(fs::path(model_dir) / "config.json", std::ios::binary);
    std::string cfg((std::istreambuf_iterator<char>(cf)), {});
    mix((const unsigned char*)cfg.data(), cfg.size());
    std::error_code ec;
    uint64_t sz = fs::exists(onnx_path) ? (uint64_t)fs::file_size(onnx_path, ec) : 0;
    mix((const unsigned char*)&sz, sizeof sz);
    char b[32];
    std::snprintf(b, sizeof b, "%016llx", (unsigned long long)h);
    return b;
}

// ============================================================================= Engine (cache)
Engine::Engine(const Args& args, const std::optional<Roi>& roi, const std::string& fingerprint) : a_(args), roi_(roi) {
    if (!a_.no_cache) {
        std::string rs = roi ? fmt("%d-%d-%d-%d", roi->x, roi->y, roi->w, roi->h) : "none";
        std::string key = fingerprint + "_w" + pyfloat(a_.window_len) + "_s" + pyfloat(a_.stride) + "_f" + pyfloat(a_.fps) +
                          "_n" + std::to_string(a_.num_frames) + "_roi" + rs;
        cache_dir_ = (fs::path(a_.runs_dir) / "report_cache" / key).string();
        fs::create_directories(cache_dir_);
    }
}

WindowsProbs Engine::windows_and_probs(const std::string& video_path) {
    std::string cpath;
    if (!cache_dir_.empty()) {
        auto sz = fs::file_size(video_path);
        cpath = (fs::path(cache_dir_) / (fs::path(video_path).stem().string() + "_" + std::to_string(sz) + ".bin")).string();
        std::ifstream f(cpath, std::ios::binary);
        if (f) {
            int64_t n = 0, k = 0;
            f.read((char*)&n, 8);
            f.read((char*)&k, 8);
            if (f && n > 0 && k > 0) {
                WindowsProbs wp;
                wp.windows.resize(n);
                wp.probs.assign(n, std::vector<double>(k));
                for (auto& w : wp.windows) { f.read((char*)&w.start, 8); f.read((char*)&w.end, 8); }
                for (auto& p : wp.probs) f.read((char*)p.data(), 8 * k);
                if (f) return wp;
            }
        }
    }
    if (!model_) {
        std::string onnx = a_.onnx_path.empty() ? (fs::path(a_.videomae_dir) / "model.onnx").string() : a_.onnx_path;
        model_ = std::make_unique<VideoMAEModel>(a_.videomae_dir, onnx, a_.use_cuda);
    }
    auto sr = extract_session_embeddings(video_path, *model_, a_.window_len, a_.stride, a_.fps, a_.num_frames, roi_);
    WindowsProbs wp{sr.windows, sr.probs};
    if (!cpath.empty() && !wp.probs.empty()) {
        std::ofstream f(cpath, std::ios::binary);
        int64_t n = (int64_t)wp.windows.size(), k = (int64_t)wp.probs[0].size();
        f.write((const char*)&n, 8);
        f.write((const char*)&k, 8);
        for (auto& w : wp.windows) { f.write((const char*)&w.start, 8); f.write((const char*)&w.end, 8); }
        for (auto& p : wp.probs) f.write((const char*)p.data(), 8 * k);
    }
    return wp;
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

ClipPred clip_prediction(const std::vector<WindowTime>& wt, const std::vector<std::vector<double>>& pc,
                         const std::vector<std::string>& uniq, const std::vector<Segment>& segs,
                         const std::vector<double>& conf, const Args& a, const std::string* true_label) {
    if (wt.empty()) return {UNCERTAIN, 0.0, "no windows (clip shorter than window_len?)"};
    std::vector<double> mids;
    for (auto& w : wt) mids.push_back((w.start + w.end) / 2);
    auto mean_all = [&] { return std::accumulate(conf.begin(), conf.end(), 0.0) / conf.size(); };
    auto seg_conf = [&](const Segment& s) {
        double sum = 0; int n = 0;
        for (size_t i = 0; i < mids.size(); ++i)
            if (mids[i] >= s.start_sec && mids[i] <= s.end_sec) { sum += conf[i]; ++n; }
        return n ? sum / n : mean_all();
    };

    std::string lab;
    double c;
    if (a.clip_method == "mean_prob") {
        std::vector<double> mp(uniq.size(), 0.0);
        for (auto& p : pc) for (size_t j = 0; j < mp.size(); ++j) mp[j] += p[j] / pc.size();
        size_t i = std::max_element(mp.begin(), mp.end()) - mp.begin();
        lab = uniq[i];
        c = mp[i];
        if (a.confidence_threshold && c < *a.confidence_threshold) lab = UNCERTAIN;
    } else {  // any_detected / dominant_segment
        std::vector<std::pair<std::string, double>> dur;  // insertion-ordered
        for (auto& s : segs) {
            auto it = std::find_if(dur.begin(), dur.end(), [&](auto& p) { return p.first == s.label; });
            if (it == dur.end()) dur.emplace_back(s.label, s.end_sec - s.start_sec);
            else it->second += s.end_sec - s.start_sec;
        }
        std::vector<std::pair<std::string, double>> real;
        for (auto& d : dur) if (d.first != UNCERTAIN) real.push_back(d);
        bool true_in_real = true_label && std::any_of(real.begin(), real.end(), [&](auto& p) { return p.first == *true_label; });
        if (a.clip_method == "any_detected" && true_in_real) lab = *true_label;
        else if (!real.empty()) {
            auto best = real.begin();
            for (auto it = real.begin(); it != real.end(); ++it) if (it->second > best->second) best = it;
            lab = best->first;
        } else lab = UNCERTAIN;

        std::vector<bool> mask(mids.size(), false);
        bool any_seg = false;
        for (auto& s : segs) {
            if (s.label != lab) continue;
            any_seg = true;
            for (size_t i = 0; i < mids.size(); ++i) if (mids[i] >= s.start_sec && mids[i] <= s.end_sec) mask[i] = true;
        }
        if (any_seg) {
            double sum = 0; int n = 0;
            for (size_t i = 0; i < mask.size(); ++i) if (mask[i]) { sum += conf[i]; ++n; }
            c = n ? sum / n : mean_all();
        } else c = mean_all();
    }

    std::string note;
    if (segs.size() > 1 || (!segs.empty() && segs[0].label != lab)) {
        note = "segments: ";
        for (size_t i = 0; i < segs.size(); ++i) {
            if (i) note += "; ";
            note += fmt("%s %.0f-%.0fs (%.2f)", segs[i].label.c_str(), segs[i].start_sec, segs[i].end_sec, seg_conf(segs[i]));
        }
    }
    return {lab, c, note};
}

std::vector<std::optional<int>> match_gt_to_det(const std::vector<GtSegment>& gt, const std::vector<Detected>& det, double min_overlap) {
    auto ov = [](const GtSegment& g, const Detected& d) { return std::max(0.0, std::min(g.end, d.end) - std::max(g.start, d.start)); };
    std::vector<std::optional<int>> out;
    for (auto& g : gt) {
        double bs = -1, bo = -1;
        int is = -1, io = -1;  // ties -> larger index (matches Python's max over (overlap, idx) tuples)
        for (size_t i = 0; i < det.size(); ++i) {
            double o = ov(g, det[i]);
            if (o < min_overlap) continue;
            if (det[i].label == g.label) { if (o >= bs) { bs = o; is = (int)i; } }
            else { if (o >= bo) { bo = o; io = (int)i; } }
        }
        int pick = is >= 0 ? is : io;
        out.push_back(pick < 0 ? std::nullopt : std::optional<int>(pick + 1));
    }
    return out;
}

// ============================================================================= Excel helpers (xlnt)
using OptFmt = std::optional<xlnt::format>;

static OptFmt cap_fmt(xlnt::worksheet& ws, const std::string& ref) {
    auto c = ws.cell(ref);
    return c.has_format() ? OptFmt(c.format()) : std::nullopt;
}
static void apply_fmt(xlnt::cell c, const OptFmt& f) {
    if (f) c.format(*f);
    else c.clear_format();
}
static void reset_cell(xlnt::cell c) {
    c.clear_value();
    c.clear_format();
}
static void set_s(xlnt::worksheet& ws, const std::string& ref, const std::string& v) { ws.cell(ref).value(v); }
static void set_n(xlnt::worksheet& ws, const std::string& ref, double v) { ws.cell(ref).value(v); }
static void set_f(xlnt::worksheet& ws, const std::string& ref, std::string f) {
    if (!f.empty() && f[0] == '=') f.erase(0, 1);
    ws.cell(ref).formula(f);
}
static std::string R(const char* col, int row) { return std::string(col) + std::to_string(row); }
static std::string R(char col, int row) { return std::string(1, col) + std::to_string(row); }

static std::pair<std::string, std::string> fill_step1(xlnt::workbook& wb, const std::vector<ClipResult>& clips,
                                                      const std::vector<std::string>& class_order) {
    auto ws = wb.sheet_by_title("Step1_Individual_Clips");
    std::map<char, OptFmt> st_in, st_th, st_cls;
    for (char c : std::string("ABCDF")) st_in[c] = cap_fmt(ws, R(c, 6));
    OptFmt st_e = cap_fmt(ws, "E6"), st_sec = cap_fmt(ws, "A57"), st_lbl = cap_fmt(ws, "A58"), st_note = cap_fmt(ws, "A66");
    std::map<int, OptFmt> st_val;
    for (int r = 58; r <= 62; ++r) st_val[r] = cap_fmt(ws, R('B', r));
    for (char c : std::string("ABCD")) { st_th[c] = cap_fmt(ws, R(c, 67)); st_cls[c] = cap_fmt(ws, R(c, 68)); }

    for (size_t row = 5; row <= std::max<size_t>(ws.highest_row(), 100); ++row)
        for (int col = 1; col <= 7; ++col)
            reset_cell(ws.cell(xlnt::cell_reference((xlnt::column_t::index_t)col, (xlnt::row_t)row)));

    set_s(ws, "A2",
          "Auto-filled by make_test_report: Clip Filename, True Label, Predicted Label, Confidence and Notes. "
          "'Correct?' and all summary stats compute in Excel. Predicted 'uncertain' = no class reached the confidence threshold.");

    int n = std::max<int>((int)clips.size(), 1), last = 4 + n;
    for (size_t i = 0; i < clips.size(); ++i) {
        int r = 5 + (int)i;
        const auto& c = clips[i];
        apply_fmt(ws.cell(R('A', r)), st_in['A']); set_s(ws, R('A', r), c.file);
        apply_fmt(ws.cell(R('B', r)), st_in['B']); set_s(ws, R('B', r), c.true_label);
        apply_fmt(ws.cell(R('C', r)), st_in['C']); set_s(ws, R('C', r), c.pred);
        apply_fmt(ws.cell(R('D', r)), st_in['D']); set_n(ws, R('D', r), round_to(c.conf, 4));
        apply_fmt(ws.cell(R('F', r)), st_in['F']); set_s(ws, R('F', r), c.note);
        apply_fmt(ws.cell(R('E', r)), st_e);
        set_f(ws, R('E', r), fmt("=IF(B%d=\"\",\"\",IF(B%d=C%d,\"Yes\",\"No\"))", r, r, r));
    }
    if (clips.empty()) {
        for (char c : std::string("ABCDF")) apply_fmt(ws.cell(R(c, 5)), st_in[c]);
        apply_fmt(ws.cell("E5"), st_e);
    }

    int sr = last + 3;
    auto rng = [&](char col) { return fmt("%c5:%c%d", col, col, last); };
    struct Row { const char* lab; std::string f; int tr; };
    std::vector<Row> rows = {
        {"Total Clips Tested", "=COUNTA(" + rng('B') + ")", 58},
        {"Correct", "=COUNTIF(" + rng('E') + ",\"Yes\")", 59},
        {"Overall Accuracy", fmt("=IFERROR(B%d/B%d,\"\")", sr + 2, sr + 1), 60},
        {"Avg. Confidence (correct predictions)", "=IFERROR(AVERAGEIF(" + rng('E') + ",\"Yes\"," + rng('D') + "),\"\")", 61},
        {"Avg. Confidence (wrong predictions)", "=IFERROR(AVERAGEIF(" + rng('E') + ",\"No\"," + rng('D') + "),\"\")", 62},
    };
    apply_fmt(ws.cell(R('A', sr)), st_sec);
    set_s(ws, R('A', sr), "Overall Summary");
    for (size_t k = 0; k < rows.size(); ++k) {
        int r = sr + 1 + (int)k;
        apply_fmt(ws.cell(R('A', r)), st_lbl);
        set_s(ws, R('A', r), rows[k].lab);
        apply_fmt(ws.cell(R('B', r)), st_val[rows[k].tr]);
        set_f(ws, R('B', r), rows[k].f);
    }

    int pr = sr + 8;
    apply_fmt(ws.cell(R('A', pr)), st_sec);
    set_s(ws, R('A', pr), "Per-Class Breakdown");
    apply_fmt(ws.cell(R('A', pr + 1)), st_note);
    set_s(ws, R('A', pr + 1), "One row per class (True Label).");
    const char* heads[] = {"Class Name", "Tested", "Correct", "Accuracy"};
    for (int i = 0; i < 4; ++i) {
        char col = 'A' + i;
        apply_fmt(ws.cell(R(col, pr + 2)), st_th[col]);
        set_s(ws, R(col, pr + 2), heads[i]);
    }
    int nrows = std::max<int>(15, (int)class_order.size());
    for (int k = 0; k < nrows; ++k) {
        int r = pr + 3 + k;
        for (char c : std::string("ABCD")) apply_fmt(ws.cell(R(c, r)), st_cls[c]);
        if (k < (int)class_order.size()) set_s(ws, R('A', r), class_order[k]);
        set_f(ws, R('B', r), fmt("=IF(A%d=\"\",\"\",COUNTIF(B$5:B$%d,A%d))", r, last, r));
        set_f(ws, R('C', r), fmt("=IF(A%d=\"\",\"\",COUNTIFS(B$5:B$%d,A%d,E$5:E$%d,\"Yes\"))", r, last, r, last));
        set_f(ws, R('D', r), fmt("=IF(A%d=\"\",\"\",IFERROR(C%d/B%d,\"\"))", r, r, r));
    }
    return {R('B', sr + 1), R('B', sr + 3)};
}

// One-time preparation of the Step-2 template sheet. Everything style-related (including the only
// number_format() calls) happens here, BEFORE any copy_sheet(), so every cycle sheet simply inherits it.
static void prepare_step2_proto(xlnt::worksheet proto) {
    auto& ws = proto;
    // example rows -> normal input style, example values cleared
    for (char c : std::string("ABCDE")) apply_fmt(ws.cell(R(c, 8)), cap_fmt(ws, R(c, 9)));
    for (char c : std::string("ABCDEF")) apply_fmt(ws.cell(R(c, 52)), cap_fmt(ws, R(c, 53)));
    for (const char* ref : {"B8", "C8", "D8", "E8", "B52", "C52", "D52", "F52"}) ws.cell(ref).clear_value();

    // Section B: confidence column G
    apply_fmt(ws.cell("G51"), cap_fmt(ws, "F51"));
    set_s(ws, "G51", "Confidence");
    apply_fmt(ws.cell("G52"), cap_fmt(ws, "F53"));
    ws.cell("G52").number_format(xlnt::number_format("0.000"));  // the only call for column G
    for (int r = 53; r < 52 + MAX_ROWS; ++r) apply_fmt(ws.cell(R('G', r)), cap_fmt(ws, "G52"));

    // Section C: matched confidence column K
    apply_fmt(ws.cell("K95"), cap_fmt(ws, "J95"));
    set_s(ws, "K95", "Matched Confidence");
    apply_fmt(ws.cell("K96"), cap_fmt(ws, "J97"));
    ws.cell("K96").number_format(xlnt::number_format("0.000"));  // the only call for column K
    for (int r = 97; r < 96 + MAX_ROWS; ++r) apply_fmt(ws.cell(R('K', r)), cap_fmt(ws, "K96"));
    for (int i = 0; i < MAX_ROWS; ++i) {
        int r = 96 + i;
        set_f(ws, R('K', r), fmt("=IF(E%d=\"\",\"\",IFERROR(INDEX($G$52:$G$91,E%d),\"\"))", r, r));
    }
    ws.column_properties(xlnt::column_t("G")).width = 14;
    ws.column_properties(xlnt::column_t("K")).width = 20;
}

static void fill_cycle_sheet(xlnt::workbook& wb, xlnt::worksheet proto, const std::string& sheet_name, const std::string& title,
                             const std::vector<GtSegment>& gt, const std::vector<Detected>& det,
                             const std::vector<std::optional<int>>& matches, const std::vector<std::optional<double>>& det_confs) {
    wb.copy_sheet(proto);
    auto ws = wb.sheet_by_index(wb.sheet_count() - 1);
    ws.title(sheet_name);
    set_s(ws, "A1", title);

    for (size_t i = 0; i < gt.size() && (int)i < MAX_ROWS; ++i) {
        int r = 8 + (int)i;
        set_s(ws, R('B', r), gt[i].label);
        set_n(ws, R('C', r), gt[i].start);
        set_n(ws, R('D', r), gt[i].end);
    }
    for (size_t i = 0; i < det.size() && (int)i < MAX_ROWS; ++i) {
        int r = 52 + (int)i;
        set_s(ws, R('B', r), det[i].label);
        set_n(ws, R('C', r), round_to(det[i].start, 2));
        set_n(ws, R('D', r), round_to(det[i].end, 2));
    }
    for (size_t i = 0; i < matches.size() && (int)i < MAX_ROWS; ++i)
        if (matches[i]) set_n(ws, R('E', 96 + (int)i), *matches[i]);

    // confidence values (formats, header, K formulas and widths were prepared on the template sheet)
    for (int i = 0; i < MAX_ROWS && i < (int)det_confs.size(); ++i)
        if (det_confs[i]) set_n(ws, R('G', 52 + i), round_to(*det_confs[i], 4));
    ws.column_properties(xlnt::column_t("G")).width = 14;
    ws.column_properties(xlnt::column_t("K")).width = 20;
}

static void build_dashboard_step2(xlnt::workbook& wb, const std::vector<std::pair<int, std::string>>& cyc_sheets) {
    auto ws = wb.sheet_by_title("Summary_Dashboard");
    auto s1 = wb.sheet_by_title("Step1_Individual_Clips");
    OptFmt st_l = cap_fmt(ws, "A14"), st_v = cap_fmt(ws, "B14");
    auto hdr = [&] { return cap_fmt(s1, "A4"); };    // re-read on every use (never keep style handles across edits)
    auto body = [&] { return cap_fmt(s1, "A58"); };

    auto plus = [&](const std::string& cell) {
        std::string o;
        for (size_t i = 0; i < cyc_sheets.size(); ++i) o += (i ? "+" : "") + cyc_sheets[i].second + "!" + cell;
        return o;
    };
    set_s(ws, "A11", fmt("STEP 2 -- Continuous Video (%zu cycles combined)", cyc_sheets.size()));
    set_f(ws, "B12", "=" + plus("B139"));
    set_f(ws, "B13", "=IFERROR((" + plus("B140") + ")/B12,\"\")");
    set_f(ws, "B14", "=" + plus("B144"));

    std::string sumif;
    for (size_t i = 0; i < cyc_sheets.size(); ++i) {
        const auto& s = cyc_sheets[i].second;
        sumif += (i ? "+" : "") + std::string("SUMIF(") + s + "!J96:J135,\"OK\"," + s + "!I96:I135)";
    }
    struct Ex { int r; const char* lab; std::string f; };
    std::vector<Ex> extra = {
        {15, "Missed Steps", "=" + plus("B141")},
        {16, "Wrong Label Steps", "=" + plus("B142")},
        {17, "Avg. |Start Offset| for OK matches (s)", "=IFERROR((" + sumif + ")/(" + plus("B140") + "),\"\")"},
    };
    for (auto& e : extra) {
        apply_fmt(ws.cell(R('A', e.r)), st_l);
        set_s(ws, R('A', e.r), e.lab);
        apply_fmt(ws.cell(R('B', e.r)), st_v);
        set_f(ws, R('B', e.r), e.f);
    }
    ws.cell("B17").number_format(xlnt::number_format("0.00"));

    int r0 = 20;
    set_s(ws, R('A', r0 - 1), "Per-cycle results");
    ws.cell(R('A', r0 - 1)).font(s1.cell("A57").font());
    const char* heads[] = {"Cycle", "GT Steps", "OK", "Missed", "Wrong Label", "False Positives", "Accuracy"};
    for (int i = 0; i < 7; ++i) {
        char col = 'A' + i;
        apply_fmt(ws.cell(R(col, r0)), hdr());
        set_s(ws, R(col, r0), heads[i]);
    }
    for (size_t i = 0; i < cyc_sheets.size(); ++i) {
        int r = r0 + 1 + (int)i;
        const auto& s = cyc_sheets[i].second;
        std::vector<std::string> vals = {"", "=" + s + "!B139", "=" + s + "!B140", "=" + s + "!B141", "=" + s + "!B142",
                                         "=" + s + "!B144", "=IFERROR(" + s + "!B140/" + s + "!B139,\"\")"};
        for (int k = 0; k < 7; ++k) {
            char col = 'A' + k;
            apply_fmt(ws.cell(R(col, r)), body());
            if (k == 0) set_s(ws, R(col, r), "cycle " + std::to_string(cyc_sheets[i].first));
            else set_f(ws, R(col, r), vals[k]);
        }
        if (i == 0) ws.cell(R('G', r)).number_format(xlnt::number_format("0.0%"));  // only call; others copy its style
        else apply_fmt(ws.cell(R('G', r)), cap_fmt(ws, R('G', r0 + 1)));
    }
    for (char col : std::string("DEFG")) ws.column_properties(xlnt::column_t(std::string(1, col))).width = 16;
}

// ============================================================================= CLI
Args parse_args(int argc, char** argv) {
    Args a;
    auto usage = [&] {
        std::printf(
            "Usage: %s --videomae_dir DIR [--onnx_path FILE] [--individual_dir DIR] [--continuous_dir DIR]\n"
            "  [--gt FILE] [--template FILE] [--window_len S] [--stride S] [--fps F] [--num_frames N]\n"
            "  [--min_segment_duration S] [--confidence_threshold P] [--roi x,y,w,h] [--runs_dir DIR]\n"
            "  [--out_dir DIR] [--out_name NAME] [--clip_method any_detected|dominant_segment|mean_prob]\n"
            "  [--min_overlap S] [--keep_uncertain] [--merge_step6] [--tester NAME] [--test_date D]\n"
            "  [--no_cache] [--cpu]   (CUDA is on by default)\n",
            argv[0]);
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
        else if (k == "--individual_dir") a.individual_dir = val();
        else if (k == "--continuous_dir") a.continuous_dir = val();
        else if (k == "--gt") a.gt = val();
        else if (k == "--template") a.tmpl = val();
        else if (k == "--window_len") a.window_len = std::stod(val());
        else if (k == "--stride") a.stride = std::stod(val());
        else if (k == "--fps") a.fps = std::stod(val());
        else if (k == "--num_frames") a.num_frames = std::stoi(val());
        else if (k == "--min_segment_duration") a.min_segment_duration = std::stod(val());
        else if (k == "--confidence_threshold") a.confidence_threshold = std::stod(val());
        else if (k == "--roi") a.roi = val();
        else if (k == "--runs_dir") a.runs_dir = val();
        else if (k == "--out_dir") a.out_dir = val();
        else if (k == "--out_name") a.out_name = val();
        else if (k == "--clip_method") {
            a.clip_method = val();
            if (a.clip_method != "any_detected" && a.clip_method != "dominant_segment" && a.clip_method != "mean_prob")
                throw std::runtime_error("bad --clip_method: " + a.clip_method);
        }
        else if (k == "--min_overlap") a.min_overlap = std::stod(val());
        else if (k == "--keep_uncertain") a.keep_uncertain = true;
        else if (k == "--merge_step6") a.merge_step6 = true;
        else if (k == "--tester") a.tester = val();
        else if (k == "--test_date") a.test_date = val();
        else if (k == "--no_cache") a.no_cache = true;
        else if (k == "--cuda") a.use_cuda = true;
        else if (k == "--cpu") a.use_cuda = false;
        else { usage(); throw std::runtime_error("unknown argument: " + k); }
    }
    if (a.videomae_dir.empty()) { usage(); throw std::runtime_error("--videomae_dir is required"); }
    if (a.onnx_path.empty()) a.onnx_path = (fs::path(a.videomae_dir) / "model.onnx").string();
    return a;
}

static std::vector<std::string> list_videos(const std::string& folder, const std::vector<std::string>& exts) {
    std::vector<std::string> names;
    for (auto& e : fs::directory_iterator(folder)) {
        if (!e.is_regular_file()) continue;
        std::string n = e.path().filename().string(), ln = lower(n);
        for (auto& x : exts) if (ends_with(ln, x)) { names.push_back(n); break; }
    }
    std::sort(names.begin(), names.end());
    std::vector<std::string> out;
    for (auto& n : names) out.push_back((fs::path(folder) / n).string());
    return out;
}

static int first_number(const std::string& s) {
    std::smatch m;
    static const std::regex re(R"((\d+))");
    return std::regex_search(s, m, re) ? std::stoi(m[1].str()) : 0;
}

// ============================================================================= main flow
int run_report(const Args& a) {
    if (a.individual_dir.empty() && a.continuous_dir.empty()) {
        std::fprintf(stderr, "Give --individual_dir and/or --continuous_dir\n");
        return 1;
    }
    RenameMap rename;
    if (a.merge_step6) rename[DEFAULT_STEP6_RENAME_FROM] = DEFAULT_STEP6_RENAME_TO;

    std::string fingerprint = compute_model_fingerprint(a.videomae_dir, a.onnx_path);
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
    std::printf("Model fingerprint: %s\n", fingerprint.c_str());
    std::printf("Model classes (%zu): %s\n", raw_names.size(), join(raw_names).c_str());
    std::printf("Excel labels   (%zu): %s   (rename map: %s)\n\n", uniq.size(), join(uniq).c_str(), rename_repr(rename).c_str());

    Engine engine(a, roi, fingerprint);

    char tsb[32];
    std::time_t now = std::time(nullptr);
    std::strftime(tsb, sizeof tsb, "%Y%m%d_%H%M%S", std::localtime(&now));
    std::string out_dir = a.out_dir.empty() ? (fs::path(a.runs_dir) / (std::string(tsb) + "_report")).string() : a.out_dir;
    fs::create_directories(out_dir);

    xlnt::workbook wb;
    wb.load(a.tmpl);
    json results = {{"model_fingerprint", fingerprint}, {"individual", json::array()}, {"continuous", json::array()}};
    std::vector<std::string> notes;

    // ---------------- Step 1
    if (!a.individual_dir.empty()) {
        std::printf("=== STEP 1: individual clips ===\n");
        std::vector<ClipResult> clips;
        for (auto& vp : list_videos(a.individual_dir, video_extensions())) {
            std::string base = fs::path(vp).filename().string();
            auto tl = label_from_filename(vp, rename);
            if (!tl) { std::printf("  [skip] cannot parse label from filename: %s\n", base.c_str()); continue; }
            if (std::find(uniq.begin(), uniq.end(), *tl) == uniq.end())
                std::printf("  [WARNING] true label '%s' (%s) is not one of the model's labels %s\n", tl->c_str(), base.c_str(), join(uniq).c_str());
            auto wp = engine.windows_and_probs(vp);
            auto pc = wp.windows.empty() ? std::vector<std::vector<double>>{} : collapse_probs(wp.probs, col);
            auto sr = windows_to_segments(wp.windows, pc, uniq, a);
            auto cp = clip_prediction(wp.windows, pc, uniq, sr.segs, sr.conf, a, &*tl);
            clips.push_back({base, *tl, cp.label, cp.note, cp.conf});
            std::printf("  %-34s true=%-11s pred=%-11s conf=%.3f %s\n", base.c_str(), tl->c_str(), cp.label.c_str(), cp.conf,
                        cp.label == *tl ? "" : "  <-- WRONG");
        }
        std::vector<std::string> seen;
        for (auto& c : clips) if (std::find(seen.begin(), seen.end(), c.true_label) == seen.end()) seen.push_back(c.true_label);
        std::vector<std::string> class_order;
        for (auto& u : uniq) if (std::find(seen.begin(), seen.end(), u) != seen.end()) class_order.push_back(u);
        for (auto& s : seen) if (std::find(uniq.begin(), uniq.end(), s) == uniq.end()) class_order.push_back(s);

        auto [tot_cell, acc_cell] = fill_step1(wb, clips, class_order);
        json ji = json::array();
        for (auto& c : clips)
            ji.push_back({{"file", c.file}, {"true", c.true_label}, {"pred", c.pred}, {"conf", c.conf}, {"note", c.note}});
        results["individual"] = ji;

        int ncor = 0, n_unc = 0;
        for (auto& c : clips) { ncor += c.pred == c.true_label; n_unc += c.pred == UNCERTAIN; }
        std::printf("\n  Step 1 accuracy: %d/%zu = %.1f%%\n", ncor, clips.size(), 100.0 * ncor / std::max<size_t>(clips.size(), 1));
        for (auto& cl : class_order) {
            int n = 0, k = 0;
            for (auto& c : clips) if (c.true_label == cl) { ++n; k += c.pred == cl; }
            std::printf("    %-12s %d/%d\n", cl.c_str(), k, n);
        }
        std::printf("\n");
        auto dash = wb.sheet_by_title("Summary_Dashboard");
        set_f(dash, "B8", "=Step1_Individual_Clips!" + tot_cell);
        set_f(dash, "B9", "=Step1_Individual_Clips!" + acc_cell);
        notes.push_back(fmt("Step1: %zu clips, method=%s, %d predicted 'uncertain'.", clips.size(), a.clip_method.c_str(), n_unc));
    }

    // ---------------- Step 2
    if (!a.continuous_dir.empty()) {
        std::printf("=== STEP 2: continuous videos ===\n");
        auto gtr = load_gt(a.gt, rename);
        std::set<std::string> bad;
        for (auto& [c, segs] : gtr.gt) for (auto& s : segs) if (std::find(uniq.begin(), uniq.end(), s.label) == uniq.end()) bad.insert(s.label);
        if (!bad.empty()) {
            std::printf("  [WARNING] GT labels not among model labels (will show as MISSED / WRONG): %s\n",
                        join(std::vector<std::string>(bad.begin(), bad.end())).c_str());
        }
        prepare_step2_proto(wb.sheet_by_title("Step2_Continuous_Video"));
        auto vids = list_videos(a.continuous_dir, video_extensions());
        std::stable_sort(vids.begin(), vids.end(), [](const std::string& x, const std::string& y) {
            return first_number(fs::path(x).filename().string()) < first_number(fs::path(y).filename().string());
        });
        std::vector<std::pair<int, std::string>> cyc_sheets;
        std::vector<std::pair<int, int>> trunc;
        static const std::regex cyc_re(R"(cycle[_\-]?(\d+))", std::regex::icase);
        long t_ok = 0, t_miss = 0, t_wrong = 0, t_fp = 0, t_gt = 0;

        for (auto& vp : vids) {
            std::string base = fs::path(vp).filename().string();
            std::smatch m;
            if (!std::regex_search(base, m, cyc_re) || !gtr.gt.count(std::stoi(m[1].str()))) {
                std::printf("  [skip] no GT row for %s\n", base.c_str());
                continue;
            }
            int cyc = std::stoi(m[1].str());
            auto wp = engine.windows_and_probs(vp);
            auto pc = collapse_probs(wp.probs, col);
            auto sr = windows_to_segments(wp.windows, pc, uniq, a);
            std::vector<Detected> det_all;
            for (auto& s : sr.segs) det_all.push_back({s.label, s.start_sec, s.end_sec, segment_confidence(wp.windows, sr.conf, s)});
            std::vector<Detected> det;
            for (auto& d : det_all) if (a.keep_uncertain || d.label != UNCERTAIN) det.push_back(d);
            int n_unc = (int)(det_all.size() - det.size());
            const auto& g = gtr.gt[cyc];
            if ((int)det.size() > MAX_ROWS) trunc.emplace_back(cyc, (int)det.size());
            std::vector<Detected> det_cap(det.begin(), det.begin() + std::min<size_t>(det.size(), MAX_ROWS));
            auto matches = match_gt_to_det(g, det_cap, a.min_overlap);

            char sn[32];
            std::snprintf(sn, sizeof sn, "Step2_cycle%02d", cyc);
            std::vector<std::optional<double>> dc;
            for (auto& d : det_cap) dc.push_back(d.conf);
            fill_cycle_sheet(wb, wb.sheet_by_title("Step2_Continuous_Video"), sn, fmt("Step 2 -- Continuous Video: cycle %d (%s)", cyc, base.c_str()), g, det_cap, matches, dc);
            cyc_sheets.emplace_back(cyc, sn);

            int ok = 0, miss = 0;
            for (size_t i = 0; i < g.size(); ++i) {
                if (!matches[i]) ++miss;
                else if (det_cap[*matches[i] - 1].label == g[i].label) ++ok;
            }
            int wrong = (int)g.size() - ok - miss;
            std::set<int> used;
            for (auto& mm : matches) if (mm) used.insert(*mm);
            int fp = 0;
            for (int i = 0; i < (int)det_cap.size(); ++i) if (!used.count(i + 1)) ++fp;

            json jg = json::array(), jd = json::array(), jm = json::array();
            for (auto& s : g) jg.push_back({{"label", s.label}, {"start", s.start}, {"end", s.end}});
            for (auto& d : det) {
                json o = {{"label", d.label}, {"start", d.start}, {"end", d.end}};
                o["conf"] = d.conf ? json(*d.conf) : json(nullptr);
                jd.push_back(o);
            }
            for (auto& mm : matches) jm.push_back(mm ? json(*mm) : json(nullptr));
            results["continuous"].push_back({{"cycle", cyc}, {"video", vp}, {"gt", jg}, {"detected", jd}, {"matches", jm},
                                             {"ok", ok}, {"missed", miss}, {"wrong_label", wrong}, {"false_positives", fp},
                                             {"uncertain_segments_dropped", n_unc}});
            t_ok += ok; t_miss += miss; t_wrong += wrong; t_fp += fp; t_gt += (long)g.size();
            std::printf("  cycle %2d: GT=%zu OK=%d missed=%d wrong=%d FP=%d (detected %zu, dropped %d uncertain)\n", cyc, g.size(),
                        ok, miss, wrong, fp, det.size(), n_unc);
        }
        wb.remove_sheet(wb.sheet_by_title("Step2_Continuous_Video"));
        if (!trunc.empty()) {
            std::string s;
            for (auto& t : trunc) s += fmt("(%d, %d) ", t.first, t.second);
            std::printf("  [WARNING] more than %d detected segments (template capacity), extras NOT in Excel: %s\n", MAX_ROWS, s.c_str());
        }
        std::printf("\n  Step 2 overall: %ld/%ld = %.1f%% OK, missed=%ld, wrong label=%ld, false positives=%ld\n\n", t_ok, t_gt,
                    100.0 * t_ok / std::max(t_gt, 1L), t_miss, t_wrong, t_fp);
        build_dashboard_step2(wb, cyc_sheets);
        notes.push_back(fmt("Step2: %zu cycles, one sheet each; 'uncertain' segments %s from Section B; match needs >= %s s overlap.",
                            cyc_sheets.size(), a.keep_uncertain ? "kept" : "excluded", pyfloat(a.min_overlap).c_str()));
        if (!trunc.empty()) notes.push_back(fmt("TRUNCATED detections (>%d) in %zu cycle(s)", MAX_ROWS, trunc.size()));
    }

    // ---------------- Run Info
    auto ri = wb.sheet_by_title("Run Info");
    fs::path mp = fs::absolute(fs::path(a.videomae_dir)).lexically_normal();
    if (mp.filename().empty()) mp = mp.parent_path();
    std::string run_name = mp.parent_path().parent_path().filename().string();  // <run>/videomae_finetuned/best_model
    char db[16];
    std::strftime(db, sizeof db, "%Y-%m-%d", std::localtime(&now));
    set_s(ri, "B6", a.test_date.empty() ? std::string(db) : a.test_date);
    if (!a.tester.empty()) set_s(ri, "B7", a.tester);
    set_s(ri, "B8", run_name);
    set_s(ri, "B9", fingerprint);
    set_s(ri, "B10", roi ? fmt("%d,%d,%d,%d", roi->x, roi->y, roi->w, roi->h) : "none");
    set_n(ri, "B11", a.window_len);
    set_n(ri, "B12", a.stride);
    set_n(ri, "B13", a.num_frames);
    if (a.confidence_threshold) set_n(ri, "B14", *a.confidence_threshold);
    else ri.cell("B14").clear_value();
    set_n(ri, "B15", a.min_segment_duration);
    std::string info = "fps=" + pyfloat(a.fps) + ". Label rename: " + rename_repr(rename) + ".";
    for (auto& n : notes) info += " " + n;
    set_s(ri, "B16", info);

    std::string out_name = a.out_name.empty() ? "temporal_test_report_" + run_name + ".xlsx" : a.out_name;
    std::string out_path = (fs::path(out_dir) / out_name).string();
    wb.save(out_path);  // formulas are recalculated by Excel/LibreOffice when the file is opened
    std::string rj = (fs::path(out_dir) / "results.json").string();
    std::ofstream(rj) << results.dump(2);
    std::printf("Saved: %s\nRaw results: %s\n", out_path.c_str(), rj.c_str());
    return 0;
}

}  // namespace vmae

int main(int argc, char** argv) {
    try {
        return vmae::run_report(vmae::parse_args(argc, argv));
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
}
