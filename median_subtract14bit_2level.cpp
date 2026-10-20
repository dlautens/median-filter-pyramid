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

struct Pyramid {
    alignas(64) uint16_t top[TOP]{};
    alignas(64) uint16_t bot[TOP * BOT]{};
};

inline void inc(Pyramid& p, uint16_t v) noexcept {
    ++p.top[v >> SHIFT];
    ++p.bot[(v >> SHIFT) * BOT + (v & (BOT - 1))];
}
inline void dec(Pyramid& p, uint16_t v) noexcept {
    --p.top[v >> SHIFT];
    --p.bot[(v >> SHIFT) * BOT + (v & (BOT - 1))];
}
inline uint16_t median(const Pyramid& p, uint32_t target) noexcept {
    uint32_t acc = 0;
    int t = 0;
    while (acc + p.top[t] <= target) acc += p.top[t++];
    int b = 0;
    const auto* bp = p.bot + t * BOT;
    while (acc + bp[b] <= target) acc += bp[b++];
    return static_cast<uint16_t>((t << SHIFT) | b);
}

void median_subtract14bit_pyramid(const Image<uint16_t>& src, Image<uint16_t>& dst,
                                  int radius, int row_begin, int row_end)
{
    const auto rows = src.rows(), cols = src.cols();
    const auto rows_1 = rows - 1, cols_1 = cols - 1;
    const auto diam = 2 * radius + 1;
    const auto target = static_cast<uint32_t>(diam) * diam / 2;
    const auto edge = (radius + 1 + 7) & ~7;

    assert(edge >= radius && edge % 8 == 0);
    assert((cols - 2 * edge) % 8 == 0);
    assert(cols <= 1024);

    Pyramid pyr;
    alignas(64) uint16_t tmp_row[1024];

    auto pix = [&](int r, int c) -> uint16_t {
        return src[std::clamp(r, 0, rows_1)][std::clamp(c, 0, cols_1)];
    };

    for (int dr = -radius; dr <= radius; ++dr)
        for (int dc = -radius; dc <= radius; ++dc)
            inc(pyr, pix(row_begin + dr, dc));

    auto slide_interior = [&](int row, int oc, int nc) {
        const int r0 = row - radius;
        for (int dr = 0; dr < diam; ++dr) {
            dec(pyr, src[r0 + dr][oc]);
            inc(pyr, src[r0 + dr][nc]);
        }
    };
    auto slide_edge = [&](int row, int oc, int nc) {
        for (int dr = -radius; dr <= radius; ++dr) {
            dec(pyr, pix(row + dr, oc));
            inc(pyr, pix(row + dr, nc));
        }
    };
    auto vert = [&](int row, int c0, int c1) {
        if (row >= radius && row + radius + 1 < rows) {
            const int r_old = row - radius, r_new = row + radius + 1;
            for (int c = c0; c <= c1; ++c) {
                const int cc = std::clamp(c, 0, cols_1);
                dec(pyr, src[r_old][cc]);
                inc(pyr, src[r_new][cc]);
            }
        } else {
            for (int c = c0; c <= c1; ++c) {
                dec(pyr, pix(row - radius, c));
                inc(pyr, pix(row + radius + 1, c));
            }
        }
    };

    int dir = 1;
    for (int row = row_begin; row < row_end; ++row) {
        const auto* __restrict srow = src[row];
        PREFETCH(src[std::min(row + 1, rows_1)]);

        const bool row_interior = (row >= radius && row + radius < rows);
        const bool use_interior = row_interior && (2 * edge < cols);

        auto emit = [&](int col) {
            tmp_row[col] = static_cast<uint16_t>(0x2000u + srow[col] - median(pyr, target));
        };

        if (dir > 0) {
            int oc = -(radius + 1), nc = radius;
            if (use_interior) {
                for (int col = 0; col < edge; ++col, ++oc, ++nc) {
                    if (col > 0) slide_edge(row, oc, nc);
                    emit(col);
                }
                for (int col = edge; col < cols - edge; ++col, ++oc, ++nc) {
                    slide_interior(row, oc, nc);
                    emit(col);
                }
                for (int col = cols - edge; col < cols; ++col, ++oc, ++nc) {
                    slide_edge(row, oc, nc);
                    emit(col);
                }
            } else {
                for (int col = 0; col < cols; ++col, ++oc, ++nc) {
                    if (col > 0) slide_edge(row, oc, nc);
                    emit(col);
                }
            }
        } else {
            int oc = (cols - 1) + (radius + 1), nc = (cols - 1) - radius;
            if (use_interior) {
                for (int col = cols - 1; col >= cols - edge; --col, --oc, --nc) {
                    if (col < cols - 1) slide_edge(row, oc, nc);
                    emit(col);
                }
                for (int col = cols - edge - 1; col >= edge; --col, --oc, --nc) {
                    slide_interior(row, oc, nc);
                    emit(col);
                }
                for (int col = edge - 1; col >= 0; --col, --oc, --nc) {
                    slide_edge(row, oc, nc);
                    emit(col);
                }
            } else {
                for (int col = cols - 1; col >= 0; --col, --oc, --nc) {
                    if (col < cols - 1) slide_edge(row, oc, nc);
                    emit(col);
                }
            }
        }

        std::memcpy(dst[row], tmp_row, cols * sizeof(uint16_t));

        if (row + 1 < row_end) {
            const int c0 = (dir > 0) ? (cols - 1 - radius) : (-radius);
            const int c1 = (dir > 0) ? (cols - 1 + radius) : radius;
            vert(row, c0, c1);
        }
        dir = -dir;
    }
}

} // namespace

void median_subtract14bit(const Image<uint16_t>& src, Image<uint16_t>& dst,
                           int radius, int row_begin, int row_end)
{
    assert(radius > 4 && radius <= 64);
    assert(src.cols() % 32 == 0);
    assert(row_begin >= 0 && row_end <= src.rows() && row_begin < row_end);
    median_subtract14bit_pyramid(src, dst, radius, row_begin, row_end);
}
