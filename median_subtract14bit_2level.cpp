// 2-level 7/7 pyramid median filter. Compact, no verification overhead.
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstring>

#if defined(_MSC_VER)
  #include <intrin.h>
  #define PREFETCH(p) _mm_prefetch(reinterpret_cast<const char*>(p), _MM_HINT_T0)
#else
  #define PREFETCH(p) __builtin_prefetch((p), 0, 3)
#endif

namespace {

constexpr int SHIFT = 7;
constexpr int TOP = 128, BOT = 128;
constexpr int MAX_COLS = 1024;

struct Pyramid {
    alignas(64) uint16_t top[TOP]{};
    alignas(64) uint16_t bot[TOP * BOT]{};

    void inc(uint16_t v) noexcept {
        ++top[v >> SHIFT];
        ++bot[v];
    }
    void dec(uint16_t v) noexcept {
        --top[v >> SHIFT];
        --bot[v];
    }
    uint16_t median(uint32_t target) const noexcept {
        auto acc = 0, t = 0;
        do { acc += top[t++]; } while (acc <= target);
        acc -= top[--t];
        auto v = t << SHIFT;
        do { acc += bot[v++]; } while (acc <= target);
        return static_cast<uint16_t>(--v);
    }
};

void median_subtract14bit(const Image<uint16_t>& src, Image<uint16_t>& dst,
                           int radius, int row_begin, int row_end)
{
    assert(radius > 4 && radius <= 64);
    assert(src.cols() % 32 == 0);
    assert(row_begin >= 0 && row_end <= src.rows() && row_begin < row_end);

    const auto rows = src.rows(), cols = src.cols();
    const auto rows_1 = rows - 1, cols_1 = cols - 1;
    const auto diam = 2 * radius + 1;
    const auto target = static_cast<uint32_t>(diam) * diam / 2;
    const auto edge = (radius + 1 + 7) & ~7;

    assert(edge >= radius && edge % 8 == 0);
    assert((cols - 2 * edge) % 8 == 0);
    assert(cols <= MAX_COLS);

    Pyramid pyr;
    alignas(64) uint16_t tmp_row[MAX_COLS];

    auto pix = [&](int r, int c) -> uint16_t {
        return src[std::clamp(r, 0, rows_1)][std::clamp(c, 0, cols_1)];
    };

    for (auto dr = -radius; dr <= radius; ++dr)
        for (auto dc = -radius; dc <= radius; ++dc)
            pyr.inc(pix(row_begin + dr, dc));

    auto slide_interior = [&](int row, int oc, int nc) {
        const auto r0 = row - radius;
        const auto r1 = r0 + diam;
        for (auto r = r0; r < r1; ++r) {
            pyr.dec(src[r][oc]);
            pyr.inc(src[r][nc]);
        }
    };
    auto slide_edge = [&](int row, int oc, int nc) {
        for (auto dr = -radius; dr <= radius; ++dr) {
            pyr.dec(pix(row + dr, oc));
            pyr.inc(pix(row + dr, nc));
        }
    };
    auto vert = [&](int row, int c0, int c1) {
        const auto r_old = std::clamp(row - radius, 0, rows_1);
        const auto r_new = std::clamp(row + radius + 1, 0, rows_1);
        for (auto c = c0; c <= c1; ++c) {
            const auto cc = std::clamp(c, 0, cols_1);
            pyr.dec(src[r_old][cc]);
            pyr.inc(src[r_new][cc]);
        }
    };

    auto dir = +1;
    for (auto row = row_begin; row < row_end; ++row) {
        const auto* __restrict srow = src[row];
        PREFETCH(src[std::min(row + 1, rows_1)]);

        const auto row_interior = (row >= radius && row + radius < rows);
        const auto use_interior = row_interior && (2 * edge < cols);

        auto emit = [&](int col) {
            tmp_row[col] = static_cast<uint16_t>(0x2000u + srow[col] - pyr.median(target));
        };

        if (dir > 0) {
            auto oc = -(radius + 1), nc = radius;
            if (use_interior) {
                emit(0); ++oc; ++nc;
                for (auto col = 1; col < edge; ++col, ++oc, ++nc) {
                    slide_edge(row, oc, nc);
                    emit(col);
                }
                for (auto col = edge; col < cols - edge; ++col, ++oc, ++nc) {
                    slide_interior(row, oc, nc);
                    emit(col);
                }
                for (auto col = cols - edge; col < cols; ++col, ++oc, ++nc) {
                    slide_edge(row, oc, nc);
                    emit(col);
                }
            } else {
                emit(0); ++oc; ++nc;
                for (auto col = 1; col < cols; ++col, ++oc, ++nc) {
                    slide_edge(row, oc, nc);
                    emit(col);
                }
            }
        } else {
            auto oc = (cols - 1) + (radius + 1), nc = (cols - 1) - radius;
            if (use_interior) {
                emit(cols - 1); --oc; --nc;
                for (auto col = cols - 2; col >= cols - edge; --col, --oc, --nc) {
                    slide_edge(row, oc, nc);
                    emit(col);
                }
                for (auto col = cols - edge - 1; col >= edge; --col, --oc, --nc) {
                    slide_interior(row, oc, nc);
                    emit(col);
                }
                for (auto col = edge - 1; col >= 0; --col, --oc, --nc) {
                    slide_edge(row, oc, nc);
                    emit(col);
                }
            } else {
                emit(cols - 1); --oc; --nc;
                for (auto col = cols - 2; col >= 0; --col, --oc, --nc) {
                    slide_edge(row, oc, nc);
                    emit(col);
                }
            }
        }

        std::memcpy(dst[row], tmp_row, cols * sizeof(uint16_t));

        if (row + 1 < row_end) {
            const auto c0 = (dir > 0) ? (cols - 1 - radius) : (-radius);
            const auto c1 = (dir > 0) ? (cols - 1 + radius) : radius;
            vert(row, c0, c1);
        }
        dir = -dir;
    }
}

} // namespace
