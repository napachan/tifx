#include "tifffile/selection.hpp"

#include <algorithm>
#include <cstdint>
#include <stdexcept>

namespace tifffile {

// ---- Target 1: Selection index computation ----

SelectionResult compute_selection_indices(
    const std::vector<int64_t>& frame_shape,
    const std::vector<AxisSelection>& selections
) {
    const size_t ndim = frame_shape.size();

    if (ndim == 0) {
        // Empty frame_axes → single page at index 0
        return {{0}, {}};
    }

    if (selections.size() != ndim) {
        throw std::invalid_argument(
            "selections length must match frame_shape length");
    }

    // Resolve each axis to concrete values and compute output shape
    std::vector<std::vector<int64_t>> axis_values(ndim);
    std::vector<int64_t> output_shape;

    for (size_t i = 0; i < ndim; ++i) {
        const auto& sel = selections[i];
        int64_t size = frame_shape[i];

        if (sel.is_integer) {
            int64_t idx = sel.start;
            if (idx < 0) idx += size;
            if (idx < 0 || idx >= size) {
                throw std::out_of_range(
                    "index out of range for axis");
            }
            axis_values[i].push_back(idx);
            // Integer selection: squeezed out of output shape
        } else {
            // Slice: start, stop, step already resolved by Python
            int64_t start = sel.start;
            int64_t stop = sel.stop;
            int64_t step = sel.step;

            if (step == 0) {
                throw std::invalid_argument("slice step cannot be zero");
            }

            // Count and collect values
            if (step > 0) {
                for (int64_t v = start; v < stop; v += step) {
                    axis_values[i].push_back(v);
                }
            } else {
                for (int64_t v = start; v > stop; v += step) {
                    axis_values[i].push_back(v);
                }
            }

            if (axis_values[i].empty()) {
                throw std::invalid_argument("empty selection for axis");
            }
            output_shape.push_back(
                static_cast<int64_t>(axis_values[i].size()));
        }
    }

    // Compute C-order strides
    std::vector<uint64_t> strides(ndim);
    strides[ndim - 1] = 1;
    for (size_t i = ndim - 1; i > 0; --i) {
        strides[i - 1] = strides[i] * static_cast<uint64_t>(frame_shape[i]);
    }

    // Compute total number of output indices
    uint64_t total = 1;
    for (size_t i = 0; i < ndim; ++i) {
        total *= axis_values[i].size();
    }

    // Iterative nested loop: enumerate all combinations
    std::vector<uint64_t> flat_indices(total);

    // Use mixed-radix counter approach
    std::vector<size_t> counters(ndim, 0);
    const auto& sizes = axis_values;  // alias for clarity

    for (uint64_t out_idx = 0; out_idx < total; ++out_idx) {
        uint64_t flat = 0;
        for (size_t d = 0; d < ndim; ++d) {
            flat += static_cast<uint64_t>(
                        axis_values[d][counters[d]]) *
                    strides[d];
        }
        flat_indices[out_idx] = flat;

        // Increment mixed-radix counter (rightmost digit first)
        for (size_t d = ndim; d > 0; --d) {
            size_t dim = d - 1;
            ++counters[dim];
            if (counters[dim] < sizes[dim].size()) {
                break;
            }
            counters[dim] = 0;
        }
    }

    return {std::move(flat_indices), std::move(output_shape)};
}

// ---- Target 2: Tile/strip position computation ----

std::pair<std::vector<int32_t>, std::vector<int32_t>>
compute_segment_positions(const TileLayout& layout) {
    const int64_t n = layout.n_segments;
    std::vector<int32_t> positions(n * 5);
    std::vector<int32_t> shapes(n * 4);

    if (layout.is_tiled) {
        const int64_t stwidth = layout.stwidth;
        const int64_t stlength = layout.stlength;
        const int64_t stdepth = layout.stdepth;
        const int64_t samples = layout.samples;
        const int64_t imwidth = layout.imwidth;
        const int64_t imlength = layout.imlength;
        const int64_t imdepth = layout.imdepth;

        const int64_t W = (imwidth + stwidth - 1) / stwidth;
        const int64_t L = (imlength + stlength - 1) / stlength;
        const int64_t D = (imdepth + stdepth - 1) / stdepth;

        for (int64_t i = 0; i < n; ++i) {
            int32_t s = static_cast<int32_t>(i / (W * L * D));
            int32_t d = static_cast<int32_t>(((i / (W * L)) % D) * stdepth);
            int32_t h = static_cast<int32_t>(((i / W) % L) * stlength);
            int32_t w = static_cast<int32_t>((i % W) * stwidth);

            int64_t p = i * 5;
            positions[p]     = s;
            positions[p + 1] = d;
            positions[p + 2] = h;
            positions[p + 3] = w;
            positions[p + 4] = 0;

            int64_t q = i * 4;
            shapes[q]     = static_cast<int32_t>(stdepth);
            shapes[q + 1] = static_cast<int32_t>(stlength);
            shapes[q + 2] = static_cast<int32_t>(stwidth);
            shapes[q + 3] = static_cast<int32_t>(samples);
        }
    } else {
        // Strips: shapes vary per segment
        const int64_t stlength = layout.stlength;
        const int64_t stwidth = layout.stwidth;
        const int64_t stdepth = layout.stdepth;
        const int64_t samples = layout.samples;
        const int64_t imlength = layout.imlength;
        const int64_t imdepth = layout.imdepth;

        const int64_t L = (imlength + stlength - 1) / stlength;

        for (int64_t i = 0; i < n; ++i) {
            int32_t s = static_cast<int32_t>(i / (L * imdepth));
            int32_t d = static_cast<int32_t>(((i / L) % imdepth) * stdepth);
            int32_t h = static_cast<int32_t>((i % L) * stlength);

            int64_t p = i * 5;
            positions[p]     = s;
            positions[p + 1] = d;
            positions[p + 2] = h;
            positions[p + 3] = 0;
            positions[p + 4] = 0;

            int64_t q = i * 4;
            shapes[q]     = static_cast<int32_t>(stdepth);
            shapes[q + 1] = static_cast<int32_t>(
                std::min(stlength, imlength - h));
            shapes[q + 2] = static_cast<int32_t>(stwidth);
            shapes[q + 3] = static_cast<int32_t>(samples);
        }
    }

    return {std::move(positions), std::move(shapes)};
}

}  // namespace tifffile
