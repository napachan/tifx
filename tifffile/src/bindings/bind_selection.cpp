#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include "tifffile/selection.hpp"

namespace nb = nanobind;
using namespace tifffile;

void bind_selection(nb::module_& m) {

    // Target 1: selection_to_page_indices
    // Receives (frame_shape: tuple, frame_sel: dict, frame_axes: str)
    // Returns (flat_indices: ndarray[uint64], output_shape: tuple)
    m.def("selection_to_page_indices",
        [](nb::tuple frame_shape_py, nb::dict frame_sel_py,
           nb::str frame_axes_py) -> nb::tuple {

            std::string axes = nb::cast<std::string>(frame_axes_py);
            size_t ndim = axes.size();

            // Parse frame_shape
            std::vector<int64_t> frame_shape;
            frame_shape.reserve(ndim);
            for (size_t i = 0; i < nb::len(frame_shape_py); ++i) {
                frame_shape.push_back(
                    nb::cast<int64_t>(frame_shape_py[i]));
            }

            if (frame_shape.size() != ndim) {
                throw nb::value_error(
                    "frame_shape length must match frame_axes length");
            }

            // Parse selections from dict
            // For each axis char, look up in dict. If missing, use slice(None).
            std::vector<AxisSelection> selections(ndim);

            for (size_t i = 0; i < ndim; ++i) {
                char ax_char = axes[i];
                nb::str ax_key(&ax_char, 1);

                if (!frame_sel_py.contains(ax_key)) {
                    // No selection for this axis → full range
                    selections[i].is_integer = false;
                    selections[i].start = 0;
                    selections[i].stop = frame_shape[i];
                    selections[i].step = 1;
                    continue;
                }

                nb::object sel = frame_sel_py[ax_key];

                if (nb::isinstance<nb::int_>(sel) ||
                    nb::hasattr(sel, "__index__")) {
                    // Integer selection
                    int64_t idx = nb::cast<int64_t>(sel);
                    if (idx < 0) idx += frame_shape[i];
                    if (idx < 0 || idx >= frame_shape[i]) {
                        throw nb::index_error(
                            ("index out of range for axis '" +
                             std::string(1, ax_char) + "'").c_str());
                    }
                    selections[i].is_integer = true;
                    selections[i].start = idx;
                    selections[i].stop = idx + 1;
                    selections[i].step = 1;
                } else {
                    // Slice: use PySlice_GetIndicesEx to resolve
                    Py_ssize_t start, stop, step, slicelength;
                    int rc = PySlice_GetIndicesEx(
                        sel.ptr(),
                        static_cast<Py_ssize_t>(frame_shape[i]),
                        &start, &stop, &step, &slicelength);
                    if (rc != 0) {
                        throw nb::python_error();
                    }
                    if (slicelength == 0) {
                        throw nb::value_error(
                            ("empty selection for axis '" +
                             std::string(1, ax_char) + "'").c_str());
                    }
                    selections[i].is_integer = false;
                    selections[i].start = static_cast<int64_t>(start);
                    selections[i].stop = static_cast<int64_t>(stop);
                    selections[i].step = static_cast<int64_t>(step);
                }
            }

            // Compute — estimate output size, only release GIL if large
            uint64_t est_size = 1;
            for (const auto& s : selections) {
                if (s.is_integer) continue;
                int64_t len = 0;
                if (s.step > 0)
                    len = (s.stop - s.start + s.step - 1) / s.step;
                else
                    len = (s.start - s.stop - s.step - 1) / (-s.step);
                est_size *= static_cast<uint64_t>(len);
            }

            SelectionResult result;
            if (est_size > 1000) {
                nb::gil_scoped_release release;
                result = compute_selection_indices(frame_shape, selections);
            } else {
                result = compute_selection_indices(frame_shape, selections);
            }

            // Build numpy array from flat_indices via capsule ownership
            auto* vec = new std::vector<uint64_t>(
                std::move(result.flat_indices));
            size_t n = vec->size();
            nb::capsule owner(vec, [](void* p) noexcept {
                delete static_cast<std::vector<uint64_t>*>(p);
            });
            nb::ndarray<nb::numpy, uint64_t, nb::ndim<1>> arr(
                vec->data(), {n}, owner);

            // Build output shape tuple
            nb::tuple out_shape = nb::steal<nb::tuple>(
                PyTuple_New(
                    static_cast<Py_ssize_t>(result.output_shape.size())));
            for (size_t i = 0; i < result.output_shape.size(); ++i) {
                PyTuple_SET_ITEM(
                    out_shape.ptr(),
                    static_cast<Py_ssize_t>(i),
                    PyLong_FromLongLong(result.output_shape[i]));
            }

            return nb::make_tuple(arr, out_shape);
        },
        nb::arg("frame_shape"), nb::arg("frame_sel"),
        nb::arg("frame_axes"),
        "Compute flat page indices from N-D frame selection."
    );

    // Target 2: compute_segment_positions
    // Receives (n_segments, stshape, imshape, is_tiled)
    // Returns (positions: ndarray[int32, n*5], shapes: ndarray[int32, n*4])
    m.def("compute_segment_positions",
        [](int64_t n_segments, nb::tuple stshape_py,
           nb::tuple imshape_py, bool is_tiled) -> nb::tuple {

            TileLayout layout;
            layout.n_segments = n_segments;
            layout.stdepth  = nb::cast<int64_t>(stshape_py[0]);
            layout.stlength = nb::cast<int64_t>(stshape_py[1]);
            layout.stwidth  = nb::cast<int64_t>(stshape_py[2]);
            layout.samples  = nb::cast<int64_t>(stshape_py[3]);
            layout.imdepth  = nb::cast<int64_t>(imshape_py[0]);
            layout.imlength = nb::cast<int64_t>(imshape_py[1]);
            layout.imwidth  = nb::cast<int64_t>(imshape_py[2]);
            layout.is_tiled = is_tiled;

            std::pair<std::vector<int32_t>, std::vector<int32_t>> result;
            {
                nb::gil_scoped_release release;
                result = compute_segment_positions(layout);
            }

            // Positions: capsule-owned numpy array
            auto* pos_vec = new std::vector<int32_t>(
                std::move(result.first));
            size_t pos_n = pos_vec->size();
            nb::capsule pos_owner(pos_vec, [](void* p) noexcept {
                delete static_cast<std::vector<int32_t>*>(p);
            });
            nb::ndarray<nb::numpy, int32_t, nb::ndim<1>> pos_arr(
                pos_vec->data(), {pos_n}, pos_owner);

            // Shapes: capsule-owned numpy array
            auto* shp_vec = new std::vector<int32_t>(
                std::move(result.second));
            size_t shp_n = shp_vec->size();
            nb::capsule shp_owner(shp_vec, [](void* p) noexcept {
                delete static_cast<std::vector<int32_t>*>(p);
            });
            nb::ndarray<nb::numpy, int32_t, nb::ndim<1>> shp_arr(
                shp_vec->data(), {shp_n}, shp_owner);

            return nb::make_tuple(pos_arr, shp_arr);
        },
        nb::arg("n_segments"), nb::arg("stshape"),
        nb::arg("imshape"), nb::arg("is_tiled"),
        "Pre-compute all segment positions and shapes."
    );
}
