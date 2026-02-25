#include <nanobind/nanobind.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include "tifffile/ifd_parser.hpp"
#include "tifffile/tiff_format.hpp"

namespace nb = nanobind;
using namespace tifffile;

void bind_ifd(nb::module_& m) {
    nb::class_<IfdEntry>(m, "IfdEntry")
        .def_ro("code", &IfdEntry::code)
        .def_ro("dtype", &IfdEntry::dtype)
        .def_ro("count", &IfdEntry::count)
        .def_ro("valueoffset", &IfdEntry::valueoffset)
        .def_ro("tag_file_offset", &IfdEntry::tag_file_offset)
        .def_ro("is_inline", &IfdEntry::is_inline)
        .def_ro("inline_size", &IfdEntry::inline_size)
        .def("get_inline_bytes", [](const IfdEntry& e) {
            return nb::bytes(
                reinterpret_cast<const char*>(e.inline_bytes.data()),
                e.inline_size
            );
        })
        .def("__repr__", [](const IfdEntry& e) {
            return "<IfdEntry code=" + std::to_string(e.code) +
                   " dtype=" + std::to_string(e.dtype) +
                   " count=" + std::to_string(e.count) +
                   " offset=" + std::to_string(e.valueoffset) +
                   (e.is_inline ? " inline" : " outofline") + ">";
        });

    // parse_ifd: simple version without default arguments
    m.def("parse_ifd",
        [](nb::bytes data, uint32_t tagno, uint64_t file_offset,
           const TiffFormat& fmt
        ) -> nb::list {
            const auto* data_ptr = reinterpret_cast<const uint8_t*>(data.c_str());
            size_t data_size = data.size();

            auto result = IfdParser::parse(
                data_ptr, data_size,
                tagno, file_offset, fmt
            );

            nb::list entries;
            for (auto& e : result.entries) {
                entries.append(nb::cast(std::move(e)));
            }
            return entries;
        },
        nb::arg("data"), nb::arg("tagno"), nb::arg("file_offset"),
        nb::arg("fmt"),
        "Parse one IFD's tag entries from raw bytes."
    );

    // parse_ifd_filtered: only return entries matching specified tag codes
    m.def("parse_ifd_filtered",
        [](nb::bytes data, uint32_t tagno, uint64_t file_offset,
           const TiffFormat& fmt, nb::set codes
        ) -> nb::list {
            const auto* data_ptr = reinterpret_cast<const uint8_t*>(data.c_str());
            size_t data_size = data.size();

            auto result = IfdParser::parse(
                data_ptr, data_size,
                tagno, file_offset, fmt
            );

            nb::list entries;
            for (auto& e : result.entries) {
                if (codes.contains(nb::int_(e.code))) {
                    entries.append(nb::cast(std::move(e)));
                }
            }
            return entries;
        },
        nb::arg("data"), nb::arg("tagno"), nb::arg("file_offset"),
        nb::arg("fmt"), nb::arg("codes"),
        "Parse IFD entries filtered by tag codes."
    );

    // scan_ifd_chain - returns packed bytes of uint64_t offsets
    m.def("scan_ifd_chain",
        [](nb::bytes file_data, uint64_t first_offset,
           const TiffFormat& fmt, size_t max_pages
        ) -> nb::bytes {
            const auto* ptr = reinterpret_cast<const uint8_t*>(file_data.c_str());
            auto result = IfdParser::scan_ifd_chain(
                ptr, file_data.size(), first_offset, fmt, max_pages
            );
            return nb::bytes(
                reinterpret_cast<const char*>(result.offsets.data()),
                result.offsets.size() * sizeof(uint64_t)
            );
        },
        nb::arg("file_data"), nb::arg("first_offset"),
        nb::arg("fmt"), nb::arg("max_pages") = size_t(0),
        "Scan IFD chain from memory buffer. Returns packed uint64_t offsets."
    );
}
