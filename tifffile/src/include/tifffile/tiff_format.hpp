#pragma once

#include <cstdint>
#include <functional>
#include <string>

#include "tifffile/common.hpp"

namespace tifffile {

struct TiffFormat {
    int version;           // 42 = classic TIFF, 43 = BigTIFF
    char byteorder;        // '<' or '>'
    int offsetsize;        // 4 or 8
    int tagnosize;         // 2 (classic) or 8 (BigTIFF)
    int tagsize;           // 12 (classic) or 20 (BigTIFF)
    int tagoffsetthreshold; // 4 (classic) or 8 (BigTIFF)

    // Derived properties
    bool is_bigtiff() const { return version == 43; }
    bool is_ndpi() const { return version == 42 && offsetsize == 8; }

    ByteOrder byte_order() const {
        return byteorder == '<' ? ByteOrder::LittleEndian : ByteOrder::BigEndian;
    }

    // Read tag count from buffer
    uint64_t read_tagno(const uint8_t* p) const {
        auto bo = byte_order();
        if (tagnosize == 2) {
            return read_u16(p, bo);
        }
        return read_u64(p, bo);
    }

    // Read offset from buffer
    uint64_t read_offset(const uint8_t* p) const {
        auto bo = byte_order();
        if (offsetsize == 4) {
            return read_u32(p, bo);
        }
        return read_u64(p, bo);
    }

    // Struct format strings for Python interop
    std::string offsetformat() const {
        if (offsetsize == 8) {
            return std::string(1, byteorder) + "Q";
        }
        return std::string(1, byteorder) + "I";
    }

    std::string tagnoformat() const {
        if (tagnosize == 8) {
            return std::string(1, byteorder) + "Q";
        }
        return std::string(1, byteorder) + "H";
    }

    std::string tagformat1() const {
        return std::string(1, byteorder) + "HH";
    }

    std::string tagformat2() const {
        if (is_bigtiff()) {
            return std::string(1, byteorder) + "Q8s";
        }
        if (is_ndpi()) {
            return std::string(1, byteorder) + "I8s";
        }
        return std::string(1, byteorder) + "I4s";
    }

    // Static factories
    static TiffFormat classic_le() {
        return TiffFormat{42, '<', 4, 2, 12, 4};
    }

    static TiffFormat classic_be() {
        return TiffFormat{42, '>', 4, 2, 12, 4};
    }

    static TiffFormat big_le() {
        return TiffFormat{43, '<', 8, 8, 20, 8};
    }

    static TiffFormat big_be() {
        return TiffFormat{43, '>', 8, 8, 20, 8};
    }

    static TiffFormat ndpi_le() {
        return TiffFormat{42, '<', 8, 2, 12, 4};
    }

    bool operator==(const TiffFormat& other) const {
        return version == other.version &&
               byteorder == other.byteorder &&
               offsetsize == other.offsetsize;
    }

    size_t hash() const {
        size_t h = std::hash<int>{}(version);
        h ^= std::hash<char>{}(byteorder) + 0x9e3779b9 + (h << 6) + (h >> 2);
        h ^= std::hash<int>{}(offsetsize) + 0x9e3779b9 + (h << 6) + (h >> 2);
        return h;
    }
};

}  // namespace tifffile
