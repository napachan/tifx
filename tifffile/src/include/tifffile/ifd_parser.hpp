#pragma once

#include <cstdint>
#include <vector>

#include "tifffile/common.hpp"
#include "tifffile/enums.hpp"
#include "tifffile/ifd_entry.hpp"
#include "tifffile/tiff_format.hpp"

namespace tifffile {

class IfdParser {
public:
    // Parse one IFD's tag entries from raw bytes.
    // data: tag data bytes (tagsize * tagno bytes, already read from file)
    // tagno: number of tags
    // file_offset: offset of first tag in file
    // fmt: TIFF format descriptor
    // ndpi_ext: optional NDPI high-byte extension data (4 * tagno bytes)
    // next_ifd_bytes: bytes after tags containing next IFD offset
    static IfdParseResult parse(
        const uint8_t* data, size_t data_size,
        uint32_t tagno,
        uint64_t file_offset,
        const TiffFormat& fmt,
        const uint8_t* ndpi_ext = nullptr, size_t ndpi_ext_size = 0,
        const uint8_t* next_ifd_bytes = nullptr, size_t next_ifd_size = 0
    );

    // Scan an IFD chain from a memory-mapped file buffer.
    // Returns offsets + circular IFD info + next_page_offset.
    static ScanChainResult scan_ifd_chain(
        const uint8_t* file_data, size_t file_size,
        uint64_t first_offset,
        const TiffFormat& fmt,
        size_t max_pages = 0
    );
};

}  // namespace tifffile
