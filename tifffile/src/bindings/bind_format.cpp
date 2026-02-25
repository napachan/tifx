#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include "tifffile/tiff_format.hpp"

namespace nb = nanobind;
using namespace tifffile;

void bind_format(nb::module_& m) {
    nb::class_<TiffFormat>(m, "CppTiffFormat")
        .def(nb::init<int, char, int, int, int, int>(),
             nb::arg("version"), nb::arg("byteorder"),
             nb::arg("offsetsize"), nb::arg("tagnosize"),
             nb::arg("tagsize"), nb::arg("tagoffsetthreshold"))
        .def_ro("version", &TiffFormat::version)
        .def_ro("byteorder", &TiffFormat::byteorder)
        .def_ro("offsetsize", &TiffFormat::offsetsize)
        .def_ro("tagnosize", &TiffFormat::tagnosize)
        .def_ro("tagsize", &TiffFormat::tagsize)
        .def_ro("tagoffsetthreshold", &TiffFormat::tagoffsetthreshold)
        .def_prop_ro("is_bigtiff", &TiffFormat::is_bigtiff)
        .def_prop_ro("is_ndpi", &TiffFormat::is_ndpi)
        .def_prop_ro("offsetformat", &TiffFormat::offsetformat)
        .def_prop_ro("tagnoformat", &TiffFormat::tagnoformat)
        .def_prop_ro("tagformat1", &TiffFormat::tagformat1)
        .def_prop_ro("tagformat2", &TiffFormat::tagformat2)
        .def_static("classic_le", &TiffFormat::classic_le)
        .def_static("classic_be", &TiffFormat::classic_be)
        .def_static("big_le", &TiffFormat::big_le)
        .def_static("big_be", &TiffFormat::big_be)
        .def_static("ndpi_le", &TiffFormat::ndpi_le)
        .def("__eq__", &TiffFormat::operator==)
        .def("__hash__", &TiffFormat::hash)
        .def("__repr__", [](const TiffFormat& f) {
            std::string bits = f.version == 42 ? "32" : "64";
            std::string endian = f.byteorder == '<' ? "little" : "big";
            std::string ndpi = f.is_ndpi() ? " with 64-bit offsets" : "";
            return "<tifffile.CppTiffFormat " + bits + "-bit " +
                   endian + "-endian" + ndpi + ">";
        });
}
