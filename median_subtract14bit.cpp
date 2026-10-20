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

constexpr int RADIUS_THRESHOLD = 32;

namespace {

constexpr int B1 = 32, B2 = 16, B3 = 32;
constexpr int S2 = 4, S3 = 5;

struct Pyramid {
    alignas(64) uint16_t top[B1]{};
    alignas(64) uint16_t mid[B1 * B2]{};
    alignas(64) uint16_t bot[B1 * B2 * B3]{};

    auto mid_ptr(int i) noexcept { return mid + i * B2; }
    auto bot_ptr(int i, int j) noexcept { return bot + (i * B2 + j) * B3; }
    auto mid_ptr(int i) const noexcept { return mid + i * B2; }
    auto bot_ptr(int i, int j) const noexcept { return bot + (i * B2 + j) * B3; }
};

inline void inc_pyramid(Pyramid& p, uint16_t v) noexcept {
    const auto b1 = v >> (S2 + S3);
    const auto b2 = (v >> S3) & (B2 - 1);
    const auto b3 = v & (B3 - 1);
    ++p.top[b1];
    ++p.mid_ptr(b1)[b2];
    ++p.bot_ptr(b1, b2)[b3];
}
inline void dec_pyramid(Pyramid& p, uint16_t v) noexcept {
    const auto b1 = v >> (S2 + S3);
    const auto b2 = (v >> S3) & (B2 - 1);
    const auto b3 = v & (B3 - 1);
    --p.top[b1];
    --p.mid_ptr(b1)[b2];
    --p.bot_ptr(b1, b2)[b3];
}

inline uint16_t median_from_pyramid(const Pyramid& p, uint32_t target) noexcept {
    uint32_t acc = 0;
    const auto* mbase = p.mid;
    const auto* bbase = p.bot;

    auto walk = [&](const uint16_t* b, int n) noexcept {
        auto i = 0;
        for (; i < n && (acc + b[i]) <= target; ++i) acc += b[i];
        return i;
    };

    const auto b1 = walk(p.top, B1);
    const auto b2 = walk(mbase + b1 * B2, B2);
    const auto b3 = walk(bbase + (b1 * B2 + b2) * B3, B3);
    return static_cast<uint16_t>((b1 << (S2 + S3)) | (b2 << S3) | b3);
}

// Forward declarations for the other paths (implementations as previously defined)
void median_subtract14bit_huang(const Image<uint16_t>& src, Image<uint16_t>& dst, int radius, int row_begin, int row_end);
void median_subtract14bit_multilevel(const Image<uint16_t>& src, Image<uint16_t>& dst, int radius, int row_begin, int row_end);

void median_subtract14bit_pyramid(const Image<uint16_t>& src, Image<uint16_t>& dst,
                                  int radius, int row_begin, int row_end)
{
    const auto rows = src.rows(), cols = src.cols();
    const auto rows_1 = rows - 1, cols_1 = cols - 1;
    const auto diam = 2 * radius + 1;
    const auto target = static_cast<uint32_t>(diam) * diam / 2;
    const auto edge = (radius + 7) & ~7;

    assert(edge >= radius && edge % 8 == 0);
    assert((cols - 2 * edge) % 8 == 0);
    assert(cols <= 1024);

    // large object first so smaller hot locals tend to sit higher on the stack
    Pyramid pyr;
    alignas(64) uint16_t tmp_row[1024];

    auto pix = [&](int r, int c) -> uint16_t {
        return src[std::clamp(r, 0, rows_1)][std::clamp(c, 0, cols_1)];
    };

    // initial window
    for (auto dr = -radius; dr <= radius; ++dr)
        for (auto dc = -radius; dc <= radius; ++dc)
            inc_pyramid(pyr, pix(row_begin + dr, dc));

    const auto vert_edge = (row_begin < radius) || (row_end > rows - radius);

    auto slide_interior = [&](int row, int oc, int nc) {
#if defined(__GNUC__)
#pragma GCC ivdep
#elif defined(_MSC_VER)
#pragma loop(ivdep)
#endif
        const auto r0 = row - radius;
        for (auto dr = 0; dr <= 2 * radius; ++dr) {
            dec_pyramid(pyr, src[r0 + dr][oc]);
            inc_pyramid(pyr, src[r0 + dr][nc]);
        }
    };
    auto slide_edge = [&](int row, int oc, int nc) {
        for (auto dr = -radius; dr <= radius; ++dr) {
            dec_pyramid(pyr, pix(row + dr, oc));
            inc_pyramid(pyr, pix(row + dr, nc));
        }
    };

    auto vert_interior = [&](int row) {
#if defined(__GNUC__)
#pragma GCC ivdep
#elif defined(_MSC_VER)
#pragma loop(ivdep)
#endif
        const auto r_old = row - radius;
        const auto r_new = row + radius + 1;
        for (auto c = 0; c < cols; ++c) {
            dec_pyramid(pyr, src[r_old][c]);
            inc_pyramid(pyr, src[r_new][c]);
        }
    };
    auto vert_edge = [&](int row) {
        for (auto c = 0; c < cols; ++c) {
            dec_pyramid(pyr, pix(row - radius, c));
            inc_pyramid(pyr, pix(row + radius + 1, c));
        }
    };

    auto dir = +1;
    for (auto row = row_begin; row < row_end; ++row) {
        const auto* __restrict srow = src[row];
        auto* __restrict drow = dst[row];
        auto* __restrict out = tmp_row;
        PREFETCH(src[std::min(row + 1, rows_1)]);

        const auto row_interior = (row >= radius && row + radius < rows);

        auto emit = [&](int col) {
            out[col] = static_cast<uint16_t>(0x2000u + srow[col] - median_from_pyramid(pyr, target));
        };

        if (dir > 0) {
            auto oc = -(radius + 1);
            auto nc = radius;
            for (auto col = 0; col < edge; ++col, ++oc, ++nc) {
                if (col > 0) slide_edge(row, oc, nc);
                emit(col);
            }
            for (auto col = edge; col < cols - edge; ++col, ++oc, ++nc) {
                if (row_interior) slide_interior(row, oc, nc);
                else              slide_edge(row, oc, nc);
                emit(col);
            }
            for (auto col = cols - edge; col < cols; ++col, ++oc, ++nc) {
                slide_edge(row, oc, nc);
                emit(col);
            }
        } else {
            auto oc = (cols - 1) + (radius + 1);
            auto nc = (cols - 1) - radius;
            for (auto col = cols - 1; col >= cols - edge; --col, --oc, --nc) {
                if (col < cols - 1) slide_edge(row, oc, nc);
                emit(col);
            }
            for (auto col = cols - edge - 1; col >= edge; --col, --oc, --nc) {
                if (row_interior) slide_interior(row, oc, nc);
                else              slide_edge(row, oc, nc);
                emit(col);
            }
            for (auto col = edge - 1; col >= 0; --col, --oc, --nc) {
                slide_edge(row, oc, nc);
                emit(col);
            }
        }

        std::memcpy(drow, tmp_row, cols * sizeof(uint16_t));

        if (row + 1 < row_end) {
            if (!vert_edge && row >= radius && row + radius + 1 < rows)
                vert_interior(row);
            else
                vert_edge(row);
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

    if (radius <= 32)
        median_subtract14bit_pyramid(src, dst, radius, row_begin, row_end);
    else if (radius <= RADIUS_THRESHOLD)
        median_subtract14bit_huang(src, dst, radius, row_begin, row_end);
    else
        median_subtract14bit_multilevel(src, dst, radius, row_begin, row_end);
}
