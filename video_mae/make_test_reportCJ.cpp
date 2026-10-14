// make_steps_report.cpp
//
// Runs the SAME detection pipeline as make_test_report.cpp (ONNX VideoMAE, sliding windows, min-duration
// filter, segment merge, confidence filter, step-order filter) on two folders:
//     continuous_tests/   and   Jumpled_tests/
// but logs to Excel in a simple format:
//     Sl No | Video Path | Video Name | detected steps with start / end times
//
// This file REUSES make_test_report.cpp / make_test_reportC.hpp (it #includes the .cpp directly, with its
// own main() renamed), so keep all three files in the same folder and compile ONLY this file:
//
//   g++ -std=c++17 -O2 make_steps_report.cpp -o make_steps_report \
//       $(pkg-config --cflags --libs opencv4) -lonnxruntime -lxlnt -lzip -lpugixml
//
// USAGE (all flags of make_test_report work; extra flags below):
//   ./make_steps_report --videomae_dir runs/roi_v4/videomae_finetuned/best_model \
//       --continuous_dir continuous_tests --jumbled_dir Jumpled_tests \
//       --window_len 2.0 --stride 1.0 --fps 15 --num_frames 16 \
//       --min_segment_duration 2.5 --confidence_threshold 0.7
//
// EXTRA FLAGS
//   --jumbled_dir DIR        folder of jumbled videos (default: Jumpled_tests; --jumpled_dir also accepted)
//   --jumbled_order_filter   also apply the step-order filter to the jumbled videos (OFF by default, because
//                            jumbled videos are out of order by definition and the filter would delete steps)
//   --post_no_order          (existing flag) turns the order filter off for continuous_tests
//   --continuous_dir DIR     default: continuous_tests

// Heavy third-party headers are included BEFORE the "main" rename so the rename can't touch them.
#include <cstdarg>
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

#define main vmae_original_main
#include "make_test_reportC.cpp"
#undef main

namespace vmae {

// ============================================================================= result rows
struct VideoRow {
    std::string path, name, error;
    std::vector<Detected> det;
};
struct Section {
    std::string title, folder;
    bool order_filter = true;
    std::vector<VideoRow> rows;
};

static std::string mmss(double sec) {
    int m = (int)(sec / 60);
    return fmt("%02d:%05.2f", m, sec - 60.0 * m);
}

// ============================================================================= Excel styling helpers
static void style_header(xlnt::worksheet& ws, const std::string& ref) {
    auto c = ws.cell(ref);
    c.fill(xlnt::fill::solid(xlnt::rgb_color(0x1F, 0x4E, 0x78)));
    c.font(xlnt::font().bold(true).color(xlnt::color::white()));
    c.alignment(xlnt::alignment().horizontal(xlnt::horizontal_alignment::center).wrap(true));
}
static void style_body(xlnt::worksheet& ws, const std::string& ref, bool wrap) {
    ws.cell(ref).alignment(xlnt::alignment().vertical(xlnt::vertical_alignment::top).wrap(wrap));
}
static void set_width(xlnt::worksheet& ws, const std::string& col, double w) {
    ws.column_properties(xlnt::column_t(col)).width = w;
}

// One row per video, detected steps listed one per line inside a single cell.
static void write_summary_sheet(xlnt::workbook& wb, const Section& sec) {
    auto ws = wb.create_sheet();
    ws.title(sec.title);
    const char* heads[] = {"Sl No", "Video Path", "Video Name", "No. of Steps Detected", "Detected Steps (label: start - end)"};
    for (int i = 0; i < 5; ++i) {
        set_s(ws, R((char)('A' + i), 1), heads[i]);
        style_header(ws, R((char)('A' + i), 1));
    }
    for (size_t i = 0; i < sec.rows.size(); ++i) {
        const auto& v = sec.rows[i];
        int r = 2 + (int)i;
        std::string text;
        for (auto& d : v.det) {
            if (!text.empty()) text += "\n";
            text += fmt("%s: %.2fs - %.2fs  [%s - %s]", d.label.c_str(), d.start, d.end, mmss(d.start).c_str(), mmss(d.end).c_str());
        }
        if (text.empty()) text = v.error.empty() ? "(none detected)" : "(error: " + v.error + ")";
        set_n(ws, R('A', r), (double)(i + 1));
        set_s(ws, R('B', r), v.path);
        set_s(ws, R('C', r), v.name);
        set_n(ws, R('D', r), (double)v.det.size());
        set_s(ws, R('E', r), text);
        for (char c : std::string("ABCDE")) style_body(ws, R(c, r), c == 'B' || c == 'E');
        ws.row_properties(r).height = 15.0 * std::max<size_t>(1, v.det.size());
    }
    set_width(ws, "A", 7);
    set_width(ws, "B", 50);
    set_width(ws, "C", 28);
    set_width(ws, "D", 12);
    set_width(ws, "E", 52);
    ws.freeze_panes("A2");
}

// One row per detected step (Sl No repeats) -> easy to filter / pivot.
static void write_detail_sheet(xlnt::workbook& wb, const std::vector<Section>& secs) {
    auto ws = wb.create_sheet();
    ws.title("Detail");
    const char* heads[] = {"Folder", "Sl No", "Video Path", "Video Name", "Step #", "Detected Step",
                           "Start (s)", "End (s)", "Start (mm:ss)", "End (mm:ss)", "Duration (s)", "Confidence"};
    for (int i = 0; i < 12; ++i) {
        set_s(ws, R((char)('A' + i), 1), heads[i]);
        style_header(ws, R((char)('A' + i), 1));
    }
    int r = 2;
    for (auto& sec : secs)
        for (size_t i = 0; i < sec.rows.size(); ++i) {
            const auto& v = sec.rows[i];
            auto common = [&](int row) {
                set_s(ws, R('A', row), sec.title);
                set_n(ws, R('B', row), (double)(i + 1));
                set_s(ws, R('C', row), v.path);
                set_s(ws, R('D', row), v.name);
            };
            if (v.det.empty()) {
                common(r);
                set_s(ws, R('F', r), v.error.empty() ? "(none detected)" : "(error)");
                ++r;
                continue;
            }
            for (size_t k = 0; k < v.det.size(); ++k, ++r) {
                const auto& d = v.det[k];
                common(r);
                set_n(ws, R('E', r), (double)(k + 1));
                set_s(ws, R('F', r), d.label);
                set_n(ws, R('G', r), round_to(d.start, 2));
                set_n(ws, R('H', r), round_to(d.end, 2));
                set_s(ws, R('I', r), mmss(d.start));
                set_s(ws, R('J', r), mmss(d.end));
                set_n(ws, R('K', r), round_to(d.end - d.start, 2));
                if (d.conf) set_n(ws, R('L', r), round_to(*d.conf, 3));
            }
        }
    const double widths[] = {18, 7, 50, 28, 8, 16, 10, 10, 13, 13, 12, 11};
    for (int i = 0; i < 12; ++i) set_width(ws, std::string(1, (char)('A' + i)), widths[i]);
    ws.freeze_panes("A2");
}

// ============================================================================= main flow
int run_steps_report(const Args& a, const std::string& jumbled_dir, bool jumbled_order_filter) {
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
    std::printf("Labels        (%zu): %s\n\n", uniq.size(), join(uniq).c_str());

    Engine engine(a, roi, fingerprint);

    // same rule as Step 2 of make_test_report: window threshold = post-processing threshold unless given
    Args a2 = a;
    if (a.post.enable && !a.confidence_threshold) a2.confidence_threshold = a.post.window_conf_thr;

    std::vector<Section> sections;
    if (!a.continuous_dir.empty()) sections.push_back({"Continuous_Tests", a.continuous_dir, a.post.do_order_filter, {}});
    if (!jumbled_dir.empty()) sections.push_back({"Jumpled_Tests", jumbled_dir, jumbled_order_filter, {}});

    json results = json::object();
    for (auto& sec : sections) {
        if (!fs::is_directory(sec.folder)) {
            std::printf("[skip] folder not found: %s\n", sec.folder.c_str());
            continue;
        }
        std::printf("=== %s (%s) | order filter: %s ===\n", sec.title.c_str(), sec.folder.c_str(), sec.order_filter ? "on" : "off");
        PostCfg cfg = a.post;
        cfg.do_order_filter = sec.order_filter;

        std::vector<std::string> vids = list_videos(sec.folder, video_extensions());
        std::stable_sort(vids.begin(), vids.end(), [](const std::string& x, const std::string& y) {
            return first_number(fs::path(x).filename().string()) < first_number(fs::path(y).filename().string());
        });

        json jrows = json::array();
        for (auto& vp : vids) {
            VideoRow row;
            row.name = fs::path(vp).filename().string();
            row.path = fs::absolute(fs::path(vp)).lexically_normal().string();
            auto t0 = std::chrono::steady_clock::now();
            try {
                auto wp = engine.windows_and_probs(vp);
                if (!wp.windows.empty()) {
                    auto pc = collapse_probs(wp.probs, col);
                    auto sr = windows_to_segments(wp.windows, pc, uniq, a2);
                    if (a.post.enable) {
                        row.det = postprocess_segments(wp.windows, sr.conf, sr.segs, cfg);
                    } else {  // raw detections (no post-processing): just drop "uncertain"
                        for (auto& s : sr.segs)
                            if (s.label != UNCERTAIN)
                                row.det.push_back({s.label, s.start_sec, s.end_sec, segment_confidence(wp.windows, sr.conf, s)});
                    }
                }
            } catch (const std::exception& e) {
                row.error = e.what();
                std::fprintf(stderr, "  [error] %s: %s\n", row.name.c_str(), e.what());
            }
            double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
            std::printf("  %s  (%.1fs)\n", row.name.c_str(), secs);
            for (auto& d : row.det)
                std::printf("      %-12s %7.2f - %7.2f  conf=%.3f\n", d.label.c_str(), d.start, d.end, d.conf.value_or(-1.0));
            if (row.det.empty()) std::printf("      (no steps detected)\n");

            json jd = json::array();
            for (auto& d : row.det)
                jd.push_back({{"label", d.label}, {"start", d.start}, {"end", d.end}, {"conf", d.conf ? json(*d.conf) : json(nullptr)}});
            jrows.push_back({{"path", row.path}, {"name", row.name}, {"detected", jd}});
            sec.rows.push_back(std::move(row));
        }
        results[sec.title] = jrows;
        std::printf("\n");
    }

    // ---------------- Excel
    xlnt::workbook wb;
    auto info = wb.active_sheet();
    info.title("Run Info");
    fs::path mp = fs::absolute(fs::path(a.videomae_dir)).lexically_normal();
    if (mp.filename().empty()) mp = mp.parent_path();
    std::string run_name = mp.parent_path().parent_path().filename().string();  // <run>/videomae_finetuned/best_model
    std::time_t now = std::time(nullptr);
    char tsb[32], db[32];
    std::strftime(tsb, sizeof tsb, "%Y%m%d_%H%M%S", std::localtime(&now));
    std::strftime(db, sizeof db, "%Y-%m-%d %H:%M", std::localtime(&now));
    struct KV { std::string k, v; };
    std::vector<KV> kv = {
        {"Date", db},
        {"Tester", a.tester},
        {"Model dir", fs::absolute(fs::path(a.videomae_dir)).string()},
        {"Model fingerprint", fingerprint},
        {"ROI", roi ? fmt("%d,%d,%d,%d", roi->x, roi->y, roi->w, roi->h) : "none"},
        {"Window length (s)", pyfloat(a.window_len)},
        {"Stride (s)", pyfloat(a.stride)},
        {"FPS", pyfloat(a.fps)},
        {"Num frames", std::to_string(a.num_frames)},
        {"Window conf threshold", pyfloat(a2.confidence_threshold.value_or(0.0))},
        {"Min segment duration (s)", pyfloat(a.min_segment_duration)},
        {"Post-processing", a.post.enable ? "on" : "off"},
        {"Segment conf threshold", a.post.do_conf_filter ? pyfloat(a.post.seg_conf_thr) : "off"},
        {"Merge overlapping", a.post.do_merge ? "yes" : "no"},
        {"Order filter (continuous)", a.post.do_order_filter ? "on" : "off"},
        {"Order filter (jumbled)", jumbled_order_filter ? "on" : "off"},
    };
    for (size_t i = 0; i < kv.size(); ++i) {
        int r = 1 + (int)i;
        set_s(info, R('A', r), kv[i].k);
        set_s(info, R('B', r), kv[i].v);
        info.cell(R('A', r)).font(xlnt::font().bold(true));
    }
    set_width(info, "A", 28);
    set_width(info, "B", 70);

    for (auto& sec : sections)
        if (fs::is_directory(sec.folder)) write_summary_sheet(wb, sec);
    write_detail_sheet(wb, sections);

    std::string out_dir = a.out_dir.empty() ? (fs::path(a.runs_dir) / (std::string(tsb) + "_steps_report")).string() : a.out_dir;
    fs::create_directories(out_dir);
    std::string out_name = a.out_name.empty() ? "steps_report_" + run_name + ".xlsx" : a.out_name;
    std::string out_path = (fs::path(out_dir) / out_name).string();
    wb.save(out_path);
    std::string rj = (fs::path(out_dir) / "results.json").string();
    std::ofstream(rj) << results.dump(2);
    std::printf("Saved: %s\nRaw results: %s\n", out_path.c_str(), rj.c_str());
    return 0;
}

}  // namespace vmae

int main(int argc, char** argv) {
    try {
        // pull out the flags that only this tool knows, pass everything else to the original parser
        std::string jumbled_dir = "Jumpled_tests";
        bool jumbled_order = false;
        std::vector<std::string> keep{argv[0]};
        for (int i = 1; i < argc; ++i) {
            std::string k = argv[i];
            std::string v;
            bool inline_v = false;
            if (k.rfind("--", 0) == 0 && k.find('=') != std::string::npos) {
                v = k.substr(k.find('=') + 1);
                k = k.substr(0, k.find('='));
                inline_v = true;
            }
            if (k == "--jumbled_dir" || k == "--jumpled_dir") {
                if (inline_v) jumbled_dir = v;
                else if (i + 1 < argc) jumbled_dir = argv[++i];
                else throw std::runtime_error("missing value for " + k);
            } else if (k == "--jumbled_order_filter") {
                jumbled_order = true;
            } else {
                keep.push_back(argv[i]);
            }
        }
        std::vector<char*> av;
        for (auto& s : keep) av.push_back(s.data());
        vmae::Args a = vmae::parse_args((int)av.size(), av.data());
        if (a.continuous_dir.empty()) a.continuous_dir = "continuous_tests";
        return vmae::run_steps_report(a, jumbled_dir, jumbled_order);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
}