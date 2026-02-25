#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <cstring>

#include "tifffile/enums.hpp"
#include "tifffile/file_reader.hpp"
#include "tifffile/ifd_parser.hpp"
#include "tifffile/tiff_format.hpp"

namespace nb = nanobind;
using namespace tifffile;

// Extract tag values for specific codes from a single IFD at the given offset.
// For LONG/SHORT/LONG8 type tags, returns the values.
// target_codes: codes to extract. Returns values as uint64_t vectors.
static void extract_tag_values(
    const uint8_t* file_data, size_t file_size,
    uint64_t ifd_offset, const TiffFormat& fmt,
    const std::vector<uint16_t>& target_codes,
    std::vector<std::vector<uint64_t>>& out_values
) {
    const ByteOrder bo = fmt.byte_order();
    const uint8_t* p = file_data + ifd_offset;
    size_t remaining = file_size - static_cast<size_t>(ifd_offset);

    // Read tagno
    uint64_t tagno;
    if (fmt.tagnosize == 2) {
        if (remaining < 2) return;
        tagno = read_u16(p, bo);
    } else {
        if (remaining < 8) return;
        tagno = read_u64(p, bo);
    }
    if (tagno > 4096) return;

    const bool is_bigtiff = fmt.is_bigtiff();
    const int tag_entry_size = is_bigtiff ? 20 : 12;
    const int inline_capacity = is_bigtiff ? 8 : 4;
    const uint8_t* tags_start = p + fmt.tagnosize;

    size_t tags_bytes = static_cast<size_t>(tagno) * tag_entry_size;
    if (remaining < fmt.tagnosize + tags_bytes) return;

    // Initialize output
    out_values.resize(target_codes.size());

    for (uint64_t i = 0; i < tagno; i++) {
        const uint8_t* entry = tags_start + i * tag_entry_size;
        uint16_t code = read_u16(entry, bo);

        // Check if this code is one we want
        size_t target_idx = SIZE_MAX;
        for (size_t j = 0; j < target_codes.size(); j++) {
            if (target_codes[j] == code) {
                target_idx = j;
                break;
            }
        }
        if (target_idx == SIZE_MAX) continue;

        uint16_t dtype = read_u16(entry + 2, bo);
        uint64_t count;
        if (is_bigtiff) {
            count = read_u64(entry + 4, bo);
        } else {
            count = read_u32(entry + 4, bo);
        }

        // Get element size
        size_t elem_size = datatype_size(static_cast<CppDatatype>(dtype));
        uint64_t value_size = count * elem_size;

        // Read values
        const uint8_t* val_ptr;
        if (value_size <= static_cast<uint64_t>(inline_capacity)) {
            // Inline
            val_ptr = entry + (is_bigtiff ? 12 : 8);
        } else {
            // Out-of-line: read offset
            uint64_t val_offset;
            if (is_bigtiff) {
                val_offset = read_u64(entry + 12, bo);
            } else {
                val_offset = read_u32(entry + 8, bo);
            }
            if (val_offset + value_size > file_size) continue;
            val_ptr = file_data + val_offset;
        }

        // Extract values based on dtype
        auto& values = out_values[target_idx];
        values.resize(count);
        for (uint64_t k = 0; k < count; k++) {
            switch (dtype) {
                case 3:  // SHORT
                    values[k] = read_u16(val_ptr + k * 2, bo);
                    break;
                case 4:  // LONG
                    values[k] = read_u32(val_ptr + k * 4, bo);
                    break;
                case 16: // LONG8
                    values[k] = read_u64(val_ptr + k * 8, bo);
                    break;
                case 1:  // BYTE
                    values[k] = val_ptr[k];
                    break;
                default:
                    // For other types, store raw first 4/8 bytes as offset
                    if (elem_size <= 4) {
                        values[k] = read_u32(val_ptr + k * elem_size, bo);
                    } else {
                        values[k] = read_u64(val_ptr + k * elem_size, bo);
                    }
                    break;
            }
        }
    }
}


void bind_file_reader(nb::module_& m) {
    nb::class_<FileReader>(m, "CppFileReader")
        .def(nb::init<const std::string&>(),
             nb::arg("path"),
             "Open and memory-map a file by path.")
        .def_prop_ro("size", &FileReader::size,
             "Size of file in bytes.")
        .def_prop_ro("is_mmap", &FileReader::is_mmap,
             "Whether file is memory-mapped.")
        .def("read", [](const FileReader& self, uint64_t offset, size_t count) {
                auto data = self.read(offset, count);
                return nb::bytes(
                    reinterpret_cast<const char*>(data.data()),
                    data.size()
                );
             },
             nb::arg("offset"), nb::arg("count"),
             "Read bytes from file (returns a copy).")
        .def("__repr__", [](const FileReader& self) {
                return "<CppFileReader size=" + std::to_string(self.size()) +
                       (self.is_mmap() ? " mmap" : " buffer") + ">";
             });

    // scan_ifd_chain using FileReader (mmap-based, GIL-free)
    m.def("scan_ifd_chain_file",
        [](FileReader& reader, uint64_t first_offset,
           const TiffFormat& fmt, size_t max_pages) -> nb::tuple {
            ScanChainResult scan_result;
            {
                nb::gil_scoped_release release;
                scan_result = IfdParser::scan_ifd_chain(
                    reader.data(), reader.size(),
                    first_offset, fmt, max_pages
                );
            }
            nb::bytes offsets_bytes(
                reinterpret_cast<const char*>(scan_result.offsets.data()),
                scan_result.offsets.size() * sizeof(uint64_t)
            );
            return nb::make_tuple(
                offsets_bytes,
                scan_result.next_page_offset,
                scan_result.circular_ifd_index,
                scan_result.circular_target_offset
            );
        },
        nb::arg("reader"), nb::arg("first_offset"),
        nb::arg("fmt"), nb::arg("max_pages") = size_t(0),
        "Scan IFD chain using memory-mapped file (GIL-free)."
    );

    // Bulk extract tag values for multiple IFDs at once
    // Returns dict mapping tag_code -> list of (tuple of values per page)
    // For pages where a tag is missing, returns empty tuple.
    m.def("bulk_extract_tag_values",
        [](FileReader& reader, nb::bytes ifd_offsets_packed,
           const TiffFormat& fmt, nb::list code_list) -> nb::dict {

            size_t n_pages = ifd_offsets_packed.size() / sizeof(uint64_t);
            const auto* offsets = reinterpret_cast<const uint64_t*>(
                ifd_offsets_packed.c_str());

            // Convert codes
            std::vector<uint16_t> codes;
            codes.reserve(nb::len(code_list));
            for (size_t i = 0; i < nb::len(code_list); i++) {
                codes.push_back(nb::cast<uint16_t>(code_list[i]));
            }

            // Preallocate results: codes × pages
            // result[code_idx][page_idx] = vector<uint64_t>
            std::vector<std::vector<std::vector<uint64_t>>> all_results(
                codes.size(),
                std::vector<std::vector<uint64_t>>(n_pages)
            );

            {
                nb::gil_scoped_release release;

                for (size_t p = 0; p < n_pages; p++) {
                    std::vector<std::vector<uint64_t>> page_values;
                    extract_tag_values(
                        reader.data(), reader.size(),
                        offsets[p], fmt, codes, page_values
                    );
                    for (size_t c = 0; c < codes.size(); c++) {
                        if (c < page_values.size()) {
                            all_results[c][p] = std::move(page_values[c]);
                        }
                    }
                }
            }

            // Build Python dict: {code: [tuple_per_page, ...]}
            nb::dict result;
            for (size_t c = 0; c < codes.size(); c++) {
                nb::list pages_list;
                for (size_t p = 0; p < n_pages; p++) {
                    auto& vals = all_results[c][p];
                    nb::tuple t = nb::steal<nb::tuple>(
                        PyTuple_New(static_cast<Py_ssize_t>(vals.size()))
                    );
                    for (size_t v = 0; v < vals.size(); v++) {
                        PyTuple_SET_ITEM(
                            t.ptr(),
                            static_cast<Py_ssize_t>(v),
                            PyLong_FromUnsignedLongLong(vals[v])
                        );
                    }
                    pages_list.append(t);
                }
                result[nb::int_(codes[c])] = pages_list;
            }
            return result;
        },
        nb::arg("reader"), nb::arg("ifd_offsets_packed"),
        nb::arg("fmt"), nb::arg("codes"),
        "Bulk extract tag values for multiple IFDs. Returns {code: [tuple_per_page]}."
    );
}
