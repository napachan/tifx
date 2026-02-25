#include <nanobind/nanobind.h>
#include "tifffile/enums.hpp"

namespace nb = nanobind;
using namespace tifffile;

void bind_enums(nb::module_& m) {
    nb::enum_<CppDatatype>(m, "CppDatatype", nb::is_arithmetic())
        .value("BYTE", CppDatatype::BYTE)
        .value("ASCII", CppDatatype::ASCII)
        .value("SHORT", CppDatatype::SHORT)
        .value("LONG", CppDatatype::LONG)
        .value("RATIONAL", CppDatatype::RATIONAL)
        .value("SBYTE", CppDatatype::SBYTE)
        .value("UNDEFINED", CppDatatype::UNDEFINED)
        .value("SSHORT", CppDatatype::SSHORT)
        .value("SLONG", CppDatatype::SLONG)
        .value("SRATIONAL", CppDatatype::SRATIONAL)
        .value("FLOAT", CppDatatype::FLOAT)
        .value("DOUBLE", CppDatatype::DOUBLE)
        .value("IFD", CppDatatype::IFD)
        .value("UNICODE", CppDatatype::UNICODE)
        .value("COMPLEX", CppDatatype::COMPLEX)
        .value("LONG8", CppDatatype::LONG8)
        .value("SLONG8", CppDatatype::SLONG8)
        .value("IFD8", CppDatatype::IFD8)
        .export_values();

    nb::enum_<CppCompression>(m, "CppCompression", nb::is_arithmetic())
        .value("NONE", CppCompression::NONE)
        .value("CCITTRLE", CppCompression::CCITTRLE)
        .value("CCITT_T4", CppCompression::CCITT_T4)
        .value("CCITT_T6", CppCompression::CCITT_T6)
        .value("LZW", CppCompression::LZW)
        .value("OJPEG", CppCompression::OJPEG)
        .value("JPEG", CppCompression::JPEG)
        .value("ADOBE_DEFLATE", CppCompression::ADOBE_DEFLATE)
        .value("JBIG_BW", CppCompression::JBIG_BW)
        .value("JBIG_COLOR", CppCompression::JBIG_COLOR)
        .value("JPEG_99", CppCompression::JPEG_99)
        .value("PACKBITS", CppCompression::PACKBITS)
        .value("DEFLATE", CppCompression::DEFLATE)
        .value("APERIO_JP2000_YCBC", CppCompression::APERIO_JP2000_YCBC)
        .value("JPEG_2000_LOSSY", CppCompression::JPEG_2000_LOSSY)
        .value("APERIO_JP2000_RGB", CppCompression::APERIO_JP2000_RGB)
        .value("ALT_JPEG", CppCompression::ALT_JPEG)
        .value("JPEG2000", CppCompression::JPEG2000)
        .value("LERC", CppCompression::LERC)
        .value("JPEG_LOSSY", CppCompression::JPEG_LOSSY)
        .value("LZMA", CppCompression::LZMA)
        .value("ZSTD", CppCompression::ZSTD)
        .value("WEBP", CppCompression::WEBP)
        .value("JPEGXL", CppCompression::JPEGXL)
        .value("PNG", CppCompression::PNG)
        .value("JPEGXR", CppCompression::JPEGXR)
        .value("JPEGXR_NDPI", CppCompression::JPEGXR_NDPI)
        .value("PIXTIFF", CppCompression::PIXTIFF)
        .value("JPEGXL_DNG", CppCompression::JPEGXL_DNG)
        .value("EER_V0", CppCompression::EER_V0)
        .value("EER_V1", CppCompression::EER_V1)
        .value("EER_V2", CppCompression::EER_V2)
        .export_values();

    nb::enum_<CppPredictor>(m, "CppPredictor", nb::is_arithmetic())
        .value("NONE", CppPredictor::NONE)
        .value("HORIZONTAL", CppPredictor::HORIZONTAL)
        .value("FLOATINGPOINT", CppPredictor::FLOATINGPOINT)
        .value("HORIZONTALX2", CppPredictor::HORIZONTALX2)
        .value("HORIZONTALX4", CppPredictor::HORIZONTALX4)
        .value("FLOATINGPOINTX2", CppPredictor::FLOATINGPOINTX2)
        .value("FLOATINGPOINTX4", CppPredictor::FLOATINGPOINTX4)
        .export_values();

    nb::enum_<CppPhotometric>(m, "CppPhotometric", nb::is_arithmetic())
        .value("MINISWHITE", CppPhotometric::MINISWHITE)
        .value("MINISBLACK", CppPhotometric::MINISBLACK)
        .value("RGB", CppPhotometric::RGB)
        .value("PALETTE", CppPhotometric::PALETTE)
        .value("MASK", CppPhotometric::MASK)
        .value("SEPARATED", CppPhotometric::SEPARATED)
        .value("YCBCR", CppPhotometric::YCBCR)
        .value("CIELAB", CppPhotometric::CIELAB)
        .value("ICCLAB", CppPhotometric::ICCLAB)
        .value("ITULAB", CppPhotometric::ITULAB)
        .value("CFA", CppPhotometric::CFA)
        .value("LOGL", CppPhotometric::LOGL)
        .value("LOGLUV", CppPhotometric::LOGLUV)
        .value("LINEAR_RAW", CppPhotometric::LINEAR_RAW)
        .value("DEPTH_MAP", CppPhotometric::DEPTH_MAP)
        .value("SEMANTIC_MASK", CppPhotometric::SEMANTIC_MASK)
        .export_values();

    nb::enum_<CppPlanarconfig>(m, "CppPlanarconfig", nb::is_arithmetic())
        .value("CONTIG", CppPlanarconfig::CONTIG)
        .value("SEPARATE", CppPlanarconfig::SEPARATE)
        .export_values();

    nb::enum_<CppSampleformat>(m, "CppSampleformat", nb::is_arithmetic())
        .value("UINT", CppSampleformat::UINT)
        .value("INT", CppSampleformat::INT)
        .value("IEEEFP", CppSampleformat::IEEEFP)
        .value("VOID", CppSampleformat::VOID)
        .value("COMPLEXINT", CppSampleformat::COMPLEXINT)
        .value("COMPLEXIEEEFP", CppSampleformat::COMPLEXIEEEFP)
        .export_values();

    nb::enum_<CppFillorder>(m, "CppFillorder", nb::is_arithmetic())
        .value("MSB2LSB", CppFillorder::MSB2LSB)
        .value("LSB2MSB", CppFillorder::LSB2MSB)
        .export_values();

    nb::enum_<CppOrientation>(m, "CppOrientation", nb::is_arithmetic())
        .value("TOPLEFT", CppOrientation::TOPLEFT)
        .value("TOPRIGHT", CppOrientation::TOPRIGHT)
        .value("BOTRIGHT", CppOrientation::BOTRIGHT)
        .value("BOTLEFT", CppOrientation::BOTLEFT)
        .value("LEFTTOP", CppOrientation::LEFTTOP)
        .value("RIGHTTOP", CppOrientation::RIGHTTOP)
        .value("RIGHTBOT", CppOrientation::RIGHTBOT)
        .value("LEFTBOT", CppOrientation::LEFTBOT)
        .export_values();

    nb::enum_<CppFiletype>(m, "CppFiletype", nb::is_arithmetic())
        .value("UNDEFINED", CppFiletype::UNDEFINED)
        .value("REDUCEDIMAGE", CppFiletype::REDUCEDIMAGE)
        .value("PAGE", CppFiletype::PAGE)
        .value("MASK", CppFiletype::MASK)
        .value("MACRO", CppFiletype::MACRO)
        .value("ENHANCED", CppFiletype::ENHANCED)
        .value("DNG", CppFiletype::DNG)
        .export_values();

    nb::enum_<CppExtrasample>(m, "CppExtrasample", nb::is_arithmetic())
        .value("UNSPECIFIED", CppExtrasample::UNSPECIFIED)
        .value("ASSOCALPHA", CppExtrasample::ASSOCALPHA)
        .value("UNASSALPHA", CppExtrasample::UNASSALPHA)
        .export_values();

    nb::enum_<CppResunit>(m, "CppResunit", nb::is_arithmetic())
        .value("NONE", CppResunit::NONE)
        .value("INCH", CppResunit::INCH)
        .value("CENTIMETER", CppResunit::CENTIMETER)
        .value("MILLIMETER", CppResunit::MILLIMETER)
        .value("MICROMETER", CppResunit::MICROMETER)
        .export_values();
}
