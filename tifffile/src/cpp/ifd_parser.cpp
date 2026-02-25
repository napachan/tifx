#include "tifffile/ifd_parser.hpp"

#include <algorithm>
#include <cstring>

namespace tifffile {

// Size in bytes for each TIFF data type
static size_t dtype_element_size(uint16_t dtype) {
    return datatype_size(static_cast<CppDatatype>(dtype));
}

IfdParseResult IfdParser::parse(
    const uint8_t* data, size_t data_size,
    uint32_t tagno,
    uint64_t file_offset,
    const TiffFormat& fmt,
    const uint8_t* ndpi_ext, size_t ndpi_ext_size,
    const uint8_t* next_ifd_bytes, size_t next_ifd_size
) {
    IfdParseResult result;
    result.entries.reserve(tagno);
    result.next_ifd_offset = 0;

    const bool is_ndpi = fmt.is_ndpi();
    const bool is_bigtiff = fmt.is_bigtiff();
    const ByteOrder bo = fmt.byte_order();
    const int tag_entry_size = is_bigtiff ? 20 : 12;
    const int value_field_offset = 4;  // code(2) + dtype(2)
    const int value_data_offset = is_bigtiff ? 12 : 8;  // after code+dtype+count
    const int inline_capacity = is_bigtiff ? 8 : 4;

    // For NDPI files, we have extended data to patch in
    // After the standard 12-byte entries, there are 4 extra bytes per tag
    // containing high bytes for offsets/values
    // The caller has already stitched them: each entry is 16 bytes

    for (uint32_t i = 0; i < tagno; i++) {
        size_t entry_offset;
        const uint8_t* entry;

        if (is_ndpi && ndpi_ext != nullptr) {
            // NDPI: entries are 12 bytes + 4 bytes high bits stitched together
            entry_offset = static_cast<size_t>(i) * 16;
            if (entry_offset + 16 > data_size) break;
            entry = data + entry_offset;
        } else {
            entry_offset = static_cast<size_t>(i) * tag_entry_size;
            if (entry_offset + tag_entry_size > data_size) break;
            entry = data + entry_offset;
        }

        IfdEntry e;
        e.code = read_u16(entry, bo);
        e.dtype = read_u16(entry + 2, bo);
        e.tag_file_offset = file_offset + static_cast<uint64_t>(i) * fmt.tagsize;

        // Count field
        if (is_bigtiff) {
            e.count = read_u64(entry + 4, bo);
        } else {
            e.count = read_u32(entry + 4, bo);
        }

        // Value / offset field
        size_t elem_size = dtype_element_size(e.dtype);
        uint64_t value_size = e.count * elem_size;

        e.inline_bytes.fill(0);

        if (is_ndpi && ndpi_ext != nullptr) {
            // NDPI: 12-byte entry + 4-byte high bits
            // value/offset is at entry[8..12] + entry[12..16]
            // making an 8-byte value
            std::memcpy(e.inline_bytes.data(), entry + 8, 4);
            std::memcpy(e.inline_bytes.data() + 4, entry + 12, 4);
            e.inline_size = 8;

            if (value_size > 4) {
                // Out-of-line: the 8 bytes are an offset
                e.valueoffset = read_u64(e.inline_bytes.data(), bo);
                e.is_inline = false;
            } else {
                e.valueoffset = e.tag_file_offset + 8;
                e.is_inline = true;
            }
        } else if (is_bigtiff) {
            // BigTIFF: value/offset field is 8 bytes at entry[12..20]
            std::memcpy(e.inline_bytes.data(), entry + 12, 8);
            e.inline_size = 8;

            if (value_size > 8) {
                e.valueoffset = read_u64(entry + 12, bo);
                e.is_inline = false;
            } else {
                e.valueoffset = e.tag_file_offset + 12;
                e.is_inline = true;
            }
        } else {
            // Classic TIFF: value/offset field is 4 bytes at entry[8..12]
            std::memcpy(e.inline_bytes.data(), entry + 8, 4);
            e.inline_size = 4;

            if (value_size > 4) {
                e.valueoffset = read_u32(entry + 8, bo);
                e.is_inline = false;
            } else {
                e.valueoffset = e.tag_file_offset + 8;
                e.is_inline = true;
            }
        }

        result.entries.push_back(std::move(e));
    }

    // Parse next IFD offset
    if (next_ifd_bytes != nullptr && next_ifd_size >= static_cast<size_t>(fmt.offsetsize)) {
        result.next_ifd_offset = fmt.read_offset(next_ifd_bytes);
    }

    return result;
}

ScanChainResult IfdParser::scan_ifd_chain(
    const uint8_t* file_data, size_t file_size,
    uint64_t first_offset,
    const TiffFormat& fmt,
    size_t max_pages
) {
    ScanChainResult result;
    result.next_page_offset = 0;
    result.circular_ifd_index = -1;
    result.circular_target_offset = 0;

    auto& offsets = result.offsets;
    offsets.reserve(std::min(max_pages > 0 ? max_pages : size_t(4096), size_t(4096)));

    uint64_t offset = first_offset;
    const ByteOrder bo = fmt.byte_order();

    while (offset > 0 && offset < file_size) {
        offsets.push_back(offset);

        if (max_pages > 0 && offsets.size() >= max_pages) break;

        // Read tagno at offset
        const uint8_t* p = file_data + offset;
        size_t remaining = file_size - static_cast<size_t>(offset);

        uint64_t tagno;
        if (fmt.tagnosize == 2) {
            if (remaining < 2) break;
            tagno = read_u16(p, bo);
        } else {
            if (remaining < 8) break;
            tagno = read_u64(p, bo);
        }

        if (tagno > 4096) break;  // suspicious

        // Skip past tagno + all tags to reach next IFD offset
        uint64_t next_offset_pos = offset + fmt.tagnosize +
            static_cast<uint64_t>(tagno) * fmt.tagsize;

        if (next_offset_pos + fmt.offsetsize > file_size) break;

        // Track the position of the next-IFD pointer
        result.next_page_offset = next_offset_pos;

        // Read next IFD offset
        offset = fmt.read_offset(file_data + next_offset_pos);

        // Quick circular check: self-reference or loop to first IFD
        if (offset == offsets.back() || offset == offsets.front()) {
            for (size_t i = 0; i < offsets.size(); i++) {
                if (offset == offsets[i]) {
                    result.circular_ifd_index = static_cast<int64_t>(i);
                    result.circular_target_offset = offset;
                    return result;
                }
            }
        }

        // Full circular check at intervals for rare patterns
        size_t n = offsets.size();
        if (n == 100 || (n >= 1000 && (n & 0xFFF) == 0)) {
            for (size_t i = 0; i < n; i++) {
                if (offset == offsets[i]) {
                    result.circular_ifd_index = static_cast<int64_t>(i);
                    result.circular_target_offset = offset;
                    return result;
                }
            }
        }
    }

    // Final circular check: does the next offset loop back?
    if (offset > 0 && offset < file_size && !offsets.empty()) {
        for (size_t i = 0; i < offsets.size(); i++) {
            if (offset == offsets[i]) {
                result.circular_ifd_index = static_cast<int64_t>(i);
                result.circular_target_offset = offset;
                break;
            }
        }
    }

    return result;
}

}  // namespace tifffile
