#pragma once

#include <cstdint>
#include <utility>
#include <vector>

namespace tifffile {

// --- Target 1: Selection index computation ---

struct AxisSelection {
    bool is_integer;            // true = single index (collapsed dim)
    int64_t start, stop, step;  // range params (for is_integer: start=index)
};

struct SelectionResult {
    std::vector<uint64_t> flat_indices;
    std::vector<int64_t> output_shape;
};

// Compute flat page indices from N-D selection.
// frame_shape: shape of each frame axis
// selections: one per axis, resolved from Python int/slice
SelectionResult compute_selection_indices(
    const std::vector<int64_t>& frame_shape,
    const std::vector<AxisSelection>& selections
);

// --- Target 2: Tile/strip position computation ---

struct TileLayout {
    int64_t n_segments;
    int64_t stdepth, stlength, stwidth, samples;
    int64_t imdepth, imlength, imwidth;
    bool is_tiled;
};

// Pre-compute all segment positions and shapes.
// Returns (positions, shapes) as flat int32 vectors.
// positions: n_segments * 5 elements, each row [s, d, h, w, 0]
// shapes:    n_segments * 4 elements, each row [depth, length, width, samples]
std::pair<std::vector<int32_t>, std::vector<int32_t>>
compute_segment_positions(const TileLayout& layout);

}  // namespace tifffile
