// A second start in one process: its bar is its own.  The counters are per announcement, so a load that
// follows another begins at 0 and climbs to its own total -- it does not inherit the first start's placed
// bytes and trip the "placed more than the announced total" warning on its first byte.  The lines go to
// stderr, read back from a file here.
#include "basalt/core/load_progress.hpp"

#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>
#include <system_error>
#include <vector>

using basalt::core::LoadProgress;

int main() {
    const std::string path = (std::filesystem::current_path() / "load_progress_two_starts_test.txt").string();
    if (std::freopen(path.c_str(), "w", stderr) == nullptr) return 1;
    LoadProgress::placed(7);                     // outside a start: not counted
    for (const unsigned long long total : {100ull, 40ull}) {
        LoadProgress::announce_total(total);
        LoadProgress::shares({{"cuda0", total}});
        LoadProgress::into("cuda0");
        LoadProgress::placed(total / 2, 0);
        LoadProgress::placed(total - total / 2, 0);
        LoadProgress::complete();
    }
    std::fflush(stderr);
    const std::string after = path + ".after";
    if (std::freopen(after.c_str(), "w", stderr) == nullptr) return 1;
    std::ifstream in(path);
    std::vector<std::string> lines;
    for (std::string l; std::getline(in, l);) lines.push_back(l);
    in.close();
    std::error_code ec;
    std::filesystem::remove(path, ec);
    std::filesystem::remove(after, ec);

    int totals = 0, warnings = 0, fails = 0;
    unsigned long long cur_total = 0, prev_last_done = 0;
    long long prev_done = -1;
    for (const std::string& l : lines) {
        unsigned long long a = 0, b = 0;
        if (std::sscanf(l.c_str(), "basalt load: total %llu", &a) == 1) {
            ++totals;
            if (totals > 1 && prev_last_done != cur_total) {   // the start before this one ended at its total
                std::fprintf(stdout, "FAIL a start ended at %llu, its total was %llu\n", prev_last_done, cur_total);
                ++fails;
            }
            cur_total = a;
            prev_last_done = 0;
            prev_done = -1;
        } else if (std::sscanf(l.c_str(), "basalt load: weights %llu %llu", &a, &b) == 2) {
            if (b != cur_total || a > b) {
                std::fprintf(stdout, "FAIL weights line %llu of %llu, the start announced %llu\n", a, b, cur_total);
                ++fails;
            }
            if (prev_done >= 0 && (long long) a < prev_done) {
                std::fprintf(stdout, "FAIL done went backwards: %lld then %llu\n", prev_done, a);
                ++fails;
            }
            prev_done = (long long) a;
            prev_last_done = a;
        } else if (l.find("basalt load: warning: placed more than the announced total") == 0) {
            ++warnings;
        }
    }
    if (prev_last_done != cur_total) {
        std::fprintf(stdout, "FAIL the last start ended at %llu, its total was %llu\n", prev_last_done, cur_total);
        ++fails;
    }
    if (totals != 2) { std::fprintf(stdout, "FAIL two totals announced (%d)\n", totals); ++fails; }
    if (warnings != 0) { std::fprintf(stdout, "FAIL a start warned about the other's bytes (%d)\n", warnings); ++fails; }
    std::fprintf(stdout, "load_progress_two_starts_test: %s\n", fails ? "FAIL" : "ok");
    return fails ? 1 : 0;
}
