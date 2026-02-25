#include <nanobind/nanobind.h>
#include "tifffile/common.hpp"

namespace nb = nanobind;

void bind_enums(nb::module_& m);
void bind_format(nb::module_& m);
void bind_ifd(nb::module_& m);
void bind_file_reader(nb::module_& m);
void bind_selection(nb::module_& m);

NB_MODULE(_tifffile_ext, m) {
    m.doc() = "C++ accelerated TIFF file operations";
    m.attr("__version__") = "2026.1.28";

    // Suppress leak warnings for module-level cached objects (e.g. CppTiffFormat)
    nb::set_leak_warnings(false);

    nb::exception<tifffile::TiffFileError>(m, "TiffFileError");

    bind_enums(m);
    bind_format(m);
    bind_ifd(m);
    bind_file_reader(m);
    bind_selection(m);
}
