#pragma once

#include <array>
#include <cstdint>
#include <vector>

namespace tifffile {

struct IfdEntry {
    uint16_t code;        // tag code
    uint16_t dtype;       // TIFF data type (1-18)
    uint64_t count;       // number of values
    std::array<uint8_t, 8> inline_bytes;  // inline value bytes (padded to 8)
    uint64_t valueoffset;     // offset in file to value data
    uint64_t tag_file_offset; // position of this tag entry in the file
    bool is_inline;           // whether value fits in inline_bytes
    uint32_t inline_size;     // size of inline data actually used
};

struct IfdParseResult {
    std::vector<IfdEntry> entries;
    uint64_t next_ifd_offset;
};

struct ScanChainResult {
    std::vector<uint64_t> offsets;
    uint64_t next_page_offset;      // file offset of next-IFD pointer after last page
    int64_t circular_ifd_index;     // index of circularly-referenced IFD, or -1
    uint64_t circular_target_offset; // the offset value that causes the loop
};

}  // namespace tifffile
