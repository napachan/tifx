#pragma once

#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <variant>
#include <vector>

namespace tifffile {

enum class ByteOrder : uint8_t {
    LittleEndian = 0,
    BigEndian = 1
};

class TiffFileError : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

// Endian read helpers

inline uint16_t read_u16_le(const uint8_t* p) {
    uint16_t v;
    std::memcpy(&v, p, sizeof(v));
#if defined(__BYTE_ORDER__) && __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
    v = __builtin_bswap16(v);
#elif defined(_MSC_VER)
    // x86/x64 is always little-endian, no swap needed
#endif
    return v;
}

inline uint16_t read_u16_be(const uint8_t* p) {
    return static_cast<uint16_t>(
        (static_cast<uint16_t>(p[0]) << 8) | static_cast<uint16_t>(p[1]));
}

inline uint32_t read_u32_le(const uint8_t* p) {
    uint32_t v;
    std::memcpy(&v, p, sizeof(v));
#if defined(__BYTE_ORDER__) && __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
    v = __builtin_bswap32(v);
#endif
    return v;
}

inline uint32_t read_u32_be(const uint8_t* p) {
    return (static_cast<uint32_t>(p[0]) << 24) |
           (static_cast<uint32_t>(p[1]) << 16) |
           (static_cast<uint32_t>(p[2]) << 8) |
           static_cast<uint32_t>(p[3]);
}

inline uint64_t read_u64_le(const uint8_t* p) {
    uint64_t v;
    std::memcpy(&v, p, sizeof(v));
#if defined(__BYTE_ORDER__) && __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
    v = __builtin_bswap64(v);
#endif
    return v;
}

inline uint64_t read_u64_be(const uint8_t* p) {
    return (static_cast<uint64_t>(p[0]) << 56) |
           (static_cast<uint64_t>(p[1]) << 48) |
           (static_cast<uint64_t>(p[2]) << 40) |
           (static_cast<uint64_t>(p[3]) << 32) |
           (static_cast<uint64_t>(p[4]) << 24) |
           (static_cast<uint64_t>(p[5]) << 16) |
           (static_cast<uint64_t>(p[6]) << 8) |
           static_cast<uint64_t>(p[7]);
}

inline uint16_t read_u16(const uint8_t* p, ByteOrder bo) {
    return bo == ByteOrder::LittleEndian ? read_u16_le(p) : read_u16_be(p);
}

inline uint32_t read_u32(const uint8_t* p, ByteOrder bo) {
    return bo == ByteOrder::LittleEndian ? read_u32_le(p) : read_u32_be(p);
}

inline uint64_t read_u64(const uint8_t* p, ByteOrder bo) {
    return bo == ByteOrder::LittleEndian ? read_u64_le(p) : read_u64_be(p);
}

inline float read_f32(const uint8_t* p, ByteOrder bo) {
    uint32_t u = read_u32(p, bo);
    float f;
    std::memcpy(&f, &u, sizeof(f));
    return f;
}

inline double read_f64(const uint8_t* p, ByteOrder bo) {
    uint64_t u = read_u64(p, bo);
    double d;
    std::memcpy(&d, &u, sizeof(d));
    return d;
}

// TagValue variant type for storing tag values
using TagValue = std::variant<
    std::monostate,                            // undefined/empty
    int64_t,                                   // single integer
    double,                                    // single float
    std::string,                               // ASCII string
    std::vector<uint8_t>,                      // UNDEFINED byte array
    std::vector<int64_t>,                      // integer array
    std::vector<double>,                       // float array
    std::vector<std::pair<int64_t, int64_t>>   // RATIONAL pairs
>;

}  // namespace tifffile
