#pragma once

#include <cstdint>

namespace tifffile {

// C++ enum classes matching Python IntEnums.
// These are used internally by C++ code; Python enums.py stays canonical.

enum class CppDatatype : uint16_t {
    BYTE = 1,
    ASCII = 2,
    SHORT = 3,
    LONG = 4,
    RATIONAL = 5,
    SBYTE = 6,
    UNDEFINED = 7,
    SSHORT = 8,
    SLONG = 9,
    SRATIONAL = 10,
    FLOAT = 11,
    DOUBLE = 12,
    IFD = 13,
    UNICODE = 14,
    COMPLEX = 15,
    LONG8 = 16,
    SLONG8 = 17,
    IFD8 = 18
};

// Size in bytes of each TIFF datatype element
inline size_t datatype_size(CppDatatype dt) {
    switch (dt) {
        case CppDatatype::BYTE:
        case CppDatatype::ASCII:
        case CppDatatype::SBYTE:
        case CppDatatype::UNDEFINED:
            return 1;
        case CppDatatype::SHORT:
        case CppDatatype::SSHORT:
        case CppDatatype::UNICODE:
            return 2;
        case CppDatatype::LONG:
        case CppDatatype::SLONG:
        case CppDatatype::FLOAT:
        case CppDatatype::IFD:
            return 4;
        case CppDatatype::RATIONAL:
        case CppDatatype::SRATIONAL:
        case CppDatatype::DOUBLE:
        case CppDatatype::LONG8:
        case CppDatatype::SLONG8:
        case CppDatatype::IFD8:
        case CppDatatype::COMPLEX:
            return 8;
        default:
            return 0;
    }
}

enum class CppCompression : uint32_t {
    NONE = 1,
    CCITTRLE = 2,
    CCITT_T4 = 3,
    CCITT_T6 = 4,
    LZW = 5,
    OJPEG = 6,
    JPEG = 7,
    ADOBE_DEFLATE = 8,
    JBIG_BW = 9,
    JBIG_COLOR = 10,
    JPEG_99 = 99,
    IMPACJ = 103,
    KODAK_262 = 262,
    JPEGXR_NDPI = 22610,
    NEXT = 32766,
    SONY_ARW = 32767,
    PACKED_RAW = 32769,
    SAMSUNG_SRW = 32770,
    CCIRLEW = 32771,
    SAMSUNG_SRW2 = 32772,
    PACKBITS = 32773,
    THUNDERSCAN = 32809,
    IT8CTPAD = 32895,
    IT8LW = 32896,
    IT8MP = 32897,
    IT8BL = 32898,
    PIXARFILM = 32908,
    PIXARLOG = 32909,
    DEFLATE = 32946,
    DCS = 32947,
    APERIO_JP2000_YCBC = 33003,
    JPEG_2000_LOSSY = 33004,
    APERIO_JP2000_RGB = 33005,
    ALT_JPEG = 33007,
    JBIG = 34661,
    SGILOG = 34676,
    SGILOG24 = 34677,
    LURADOC = 34692,
    JPEG2000 = 34712,
    NIKON_NEF = 34713,
    JBIG2 = 34715,
    MDI_BINARY = 34718,
    MDI_PROGRESSIVE = 34719,
    MDI_VECTOR = 34720,
    LERC = 34887,
    JPEG_LOSSY = 34892,
    LZMA = 34925,
    ZSTD_DEPRECATED = 34926,
    WEBP_DEPRECATED = 34927,
    PNG = 34933,
    JPEGXR = 34934,
    JETRAW = 48124,
    ZSTD = 50000,
    WEBP = 50001,
    JPEGXL = 50002,
    PIXTIFF = 50013,
    JPEGXL_DNG = 52546,
    EER_V0 = 65000,
    EER_V1 = 65001,
    EER_V2 = 65002
};

enum class CppPredictor : uint32_t {
    NONE = 1,
    HORIZONTAL = 2,
    FLOATINGPOINT = 3,
    HORIZONTALX2 = 34892,
    HORIZONTALX4 = 34893,
    FLOATINGPOINTX2 = 34894,
    FLOATINGPOINTX4 = 34895
};

enum class CppPhotometric : uint32_t {
    MINISWHITE = 0,
    MINISBLACK = 1,
    RGB = 2,
    PALETTE = 3,
    MASK = 4,
    SEPARATED = 5,
    YCBCR = 6,
    CIELAB = 8,
    ICCLAB = 9,
    ITULAB = 10,
    CFA = 32803,
    LOGL = 32844,
    LOGLUV = 32845,
    LINEAR_RAW = 34892,
    DEPTH_MAP = 51177,
    SEMANTIC_MASK = 52527
};

enum class CppFiletype : uint32_t {
    UNDEFINED = 0,
    REDUCEDIMAGE = 1,
    PAGE = 2,
    MASK = 4,
    MACRO = 8,
    ENHANCED = 16,
    DNG = 65536
};

enum class CppOfiletype : uint32_t {
    UNDEFINED = 0,
    IMAGE = 1,
    REDUCEDIMAGE = 2,
    PAGE = 3
};

enum class CppFillorder : uint32_t {
    MSB2LSB = 1,
    LSB2MSB = 2
};

enum class CppOrientation : uint32_t {
    TOPLEFT = 1,
    TOPRIGHT = 2,
    BOTRIGHT = 3,
    BOTLEFT = 4,
    LEFTTOP = 5,
    RIGHTTOP = 6,
    RIGHTBOT = 7,
    LEFTBOT = 8
};

enum class CppPlanarconfig : uint32_t {
    CONTIG = 1,
    SEPARATE = 2
};

enum class CppResunit : uint32_t {
    NONE = 1,
    INCH = 2,
    CENTIMETER = 3,
    MILLIMETER = 4,
    MICROMETER = 5
};

enum class CppExtrasample : uint32_t {
    UNSPECIFIED = 0,
    ASSOCALPHA = 1,
    UNASSALPHA = 2
};

enum class CppSampleformat : uint32_t {
    UINT = 1,
    INT = 2,
    IEEEFP = 3,
    VOID = 4,
    COMPLEXINT = 5,
    COMPLEXIEEEFP = 6
};

enum class CppChunkmode : uint32_t {
    STRILE = 0,
    PLANE = 1,
    PAGE = 2,
    FILE = 3
};

}  // namespace tifffile
