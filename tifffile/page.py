# page.py

"""TIFF page classes: TiffPage, TiffFrame, TiffPages."""

from __future__ import annotations

import contextlib
import math
import os
import struct
import threading
import warnings
from collections.abc import Iterable, Sequence
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime as DateTime  # noqa: N812
from functools import cached_property
from typing import TYPE_CHECKING, cast, final, overload

import numpy

from .enums import (
    COMPRESSION,
    EXTRASAMPLE,
    PHOTOMETRIC,
    PLANARCONFIG,
    PREDICTOR,
    RESUNIT,
    SAMPLEFORMAT,
)
from .fileio import NullContext
from .tags import TiffTag, TiffTags
from .utils import (
    TiffFileError,
    astype,
    enumarg,
    logger,
    pformat,
    product,
    strptime,
)

if os.environ.get('TIFFFILE_NO_CPP'):
    _HAS_CPP = False
    _cpp_parse_ifd = None
    _cpp_parse_ifd_filtered = None
    _CppTiffFormat = None
    _CppFileReader = None
    _cpp_scan_ifd_chain_file = None
    _CPP_FORMATS: dict[tuple[int, str], object] = {}
else:
    try:
        from ._tifffile_ext import (
            CppFileReader as _CppFileReader,
            CppTiffFormat as _CppTiffFormat,
            bulk_extract_tag_values as _cpp_bulk_extract,
            parse_ifd as _cpp_parse_ifd,
            parse_ifd_filtered as _cpp_parse_ifd_filtered,
            scan_ifd_chain_file as _cpp_scan_ifd_chain_file,
        )

        _HAS_CPP = True
        # Pre-create the 4 format variants to avoid per-call allocation
        _CPP_FORMATS = {
            (42, '<'): _CppTiffFormat.classic_le(),
            (42, '>'): _CppTiffFormat.classic_be(),
            (43, '<'): _CppTiffFormat.big_le(),
            (43, '>'): _CppTiffFormat.big_be(),
        }
    except ImportError:
        _HAS_CPP = False
        _cpp_parse_ifd = None
        _cpp_parse_ifd_filtered = None
        _CppTiffFormat = None
        _CppFileReader = None
        _cpp_scan_ifd_chain_file = None
        _CPP_FORMATS = {}

if TYPE_CHECKING:
    from collections.abc import Callable, Container, Iterator
    from typing import IO, Any, TypeAlias

    from numpy.typing import NDArray

    from .codecs import TiffFormat
    from .fileio import FileHandle, StoredShape
    from .sequences import TiffPageSeries
    from .tifffile import TiffFile
    from .zarr import ZarrTiffStore

    OutputType: TypeAlias = str | IO[bytes] | NDArray[Any] | None


@final
class TiffPage:
    """TIFF image file directory (IFD).

    TiffPage instances are not thread-safe. All attributes are read-only.

    Parameters:
        parent:
            TiffFile instance to read page from.
            The file handle position must be at an offset to an IFD structure.
        index:
            Index of page in IFD tree.
        keyframe:
            Not used.

    Raises:
        TiffFileError: Invalid TIFF structure.

    """

    # instance attributes

    tags: TiffTags
    """Tags belonging to page."""

    parent: TiffFile
    """TiffFile instance page belongs to."""

    offset: int
    """Position of page in file."""

    shape: tuple[int, ...]
    """Shape of image array in page."""

    dtype: numpy.dtype[Any] | None
    """Data type of image array in page."""

    shaped: tuple[int, int, int, int, int]
    """Normalized 5-dimensional shape of image array in page:

        0. separate samplesperpixel or 1.
        1. imagedepth or 1.
        2. imagelength.
        3. imagewidth.
        4. contig samplesperpixel or 1.

    """

    axes: str
    """Character codes for dimensions in image array:
    'S' sample, 'X' width, 'Y' length, 'Z' depth.
    """

    dataoffsets: tuple[int, ...]
    """Positions of strips or tiles in file."""

    databytecounts: tuple[int, ...]
    """Size of strips or tiles in file."""

    _dtype: numpy.dtype[Any] | None
    _index: tuple[int, ...]  # index of page in IFD tree

    # default properties; might be updated from tags

    subfiletype: int = 0
    """:py:class:`FILETYPE` kind of image."""

    imagewidth: int = 0
    """Number of columns (pixels per row) in image."""

    imagelength: int = 0
    """Number of rows in image."""

    imagedepth: int = 1
    """Number of Z slices in image."""

    tilewidth: int = 0
    """Number of columns in each tile."""

    tilelength: int = 0
    """Number of rows in each tile."""

    tiledepth: int = 1
    """Number of Z slices in each tile."""

    samplesperpixel: int = 1
    """Number of components per pixel."""

    bitspersample: int = 1
    """Number of bits per pixel component."""

    sampleformat: int = 1
    """:py:class:`SAMPLEFORMAT` type of pixel components."""

    rowsperstrip: int = 2**32 - 1
    """Number of rows per strip."""

    compression: int = 1
    """:py:class:`COMPRESSION` scheme used on image data."""

    planarconfig: int = 1
    """:py:class:`PLANARCONFIG` type of storage of components in pixel."""

    fillorder: int = 1
    """Logical order of bits within byte of image data."""

    photometric: int = 0
    """:py:class:`PHOTOMETRIC` color space of image."""

    predictor: int = 1
    """:py:class:`PREDICTOR` applied to image data before compression."""

    extrasamples: tuple[int, ...] = ()
    """:py:class:`EXTRASAMPLE` interpretation of extra components in pixel."""

    subsampling: tuple[int, int] | None = None
    """Subsampling factors used for chrominance components."""

    subifds: tuple[int, ...] | None = None
    """Positions of SubIFDs in file."""

    jpegtables: bytes | None = None
    """JPEG quantization and Huffman tables."""

    jpegheader: bytes | None = None
    """JPEG header for NDPI."""

    software: str = ''
    """Software used to create image."""

    description: str = ''
    """Subject of image."""

    description1: str = ''
    """Value of second ImageDescription tag."""

    nodata: float = 0
    """Value used for missing data. The value of the GDAL_NODATA tag or 0."""

    def __init__(
        self,
        parent: TiffFile,
        /,
        index: int | Sequence[int],
        *,
        keyframe: TiffPage | None = None,
    ) -> None:
        from .tifffile import TIFF

        tag: TiffTag | None
        tiff = parent.tiff

        self.parent = parent
        self.shape = ()
        self.shaped = (0, 0, 0, 0, 0)
        self.dtype = self._dtype = None
        self.axes = ''
        self.tags = tags = TiffTags()
        self.dataoffsets = ()
        self.databytecounts = ()
        if isinstance(index, int):
            self._index = (index,)
        else:
            self._index = tuple(index)

        # read IFD structure and its tags from file
        fh = parent.filehandle
        self.offset = fh.tell()  # offset to this IFD
        try:
            tagno: int = struct.unpack(
                tiff.tagnoformat, fh.read(tiff.tagnosize)
            )[0]
            if tagno > 4096:
                msg = f'suspicious number of tags {tagno}'
                raise ValueError(msg)
        except Exception as exc:
            msg = f'corrupted tag list @{self.offset}'
            raise TiffFileError(msg) from exc

        tagoffset = self.offset + tiff.tagnosize  # fh.tell()
        tagsize = tagsize_ = tiff.tagsize

        data = fh.read(tagsize * tagno)
        if len(data) != tagsize * tagno:
            msg = 'corrupted IFD structure'
            raise TiffFileError(msg)
        if tiff.is_ndpi:
            # patch offsets/values for 64-bit NDPI file
            tagsize = 16
            fh.seek(8, os.SEEK_CUR)
            ext = fh.read(4 * tagno)  # high bits
            data = b''.join(
                data[i * 12 : i * 12 + 12] + ext[i * 4 : i * 4 + 4]
                for i in range(tagno)
            )

        if _HAS_CPP and not tiff.is_ndpi:
            cpp_fmt = _CPP_FORMATS.get((tiff.version, tiff.byteorder))
            entries = _cpp_parse_ifd(data, tagno, tagoffset, cpp_fmt)
            for entry in entries:
                try:
                    tag = TiffTag.from_ifd_entry(parent, entry)
                except TiffFileError as exc:
                    logger().error(
                        f'<TiffTag.from_ifd_entry> raised {exc!r:.128}'
                    )
                    continue
                tags.add(tag)
        else:
            tagindex = -tagsize
            for i in range(tagno):
                tagindex += tagsize
                tagdata = data[tagindex : tagindex + tagsize]
                try:
                    tag = TiffTag.fromfile(
                        parent, offset=tagoffset + i * tagsize_, header=tagdata
                    )
                except TiffFileError as exc:
                    logger().error(
                        f'<TiffTag.fromfile> raised {exc!r:.128}'
                    )
                    continue
                tags.add(tag)

        if not tags:
            return  # found in FIBICS

        for code, name in TIFF.TAG_ATTRIBUTES.items():
            value = tags.valueof(code)
            if value is None:
                continue
            if code in {270, 305} and not isinstance(value, str):
                # wrong string type for software or description
                continue
            setattr(self, name, value)

        value = tags.valueof(270, index=1)
        if isinstance(value, str):
            self.description1 = value

        if self.subfiletype == 0:
            value = tags.valueof(255)  # SubfileType
            if value == 2:
                self.subfiletype = 0b1  # reduced image
            elif value == 3:
                self.subfiletype = 0b10  # multi-page
        elif not isinstance(self.subfiletype, int):
            # files created by IDEAS
            logger().warning(f'{self!r} invalid {self.subfiletype=}')
            self.subfiletype = 0

        # consolidate private tags; remove them from self.tags
        # if self.is_andor:
        #     self.andor_tags
        # elif self.is_epics:
        #     self.epics_tags
        # elif self.is_ndpi:
        #     self.ndpi_tags
        # if self.is_sis and 34853 in tags:
        #     # TODO: cannot change tag.name
        #     tags[34853].name = 'OlympusSIS2'

        # dataoffsets and databytecounts
        # TileOffsets
        dataoffsets = tags.valueof(324)
        if dataoffsets is None:
            # StripOffsets
            dataoffsets = tags.valueof(273)
            if dataoffsets is None:
                # JPEGInterchangeFormat et al.
                dataoffsets = tags.valueof(513)
                if dataoffsets is None:
                    dataoffsets = ()
                    logger().error(f'{self!r} missing data offset tag')
        self.dataoffsets = dataoffsets
        del dataoffsets

        # TileByteCounts
        databytecounts = tags.valueof(325)
        if databytecounts is None:
            # StripByteCounts
            databytecounts = tags.valueof(279)
            if databytecounts is None:
                # JPEGInterchangeFormatLength et al.
                databytecounts = tags.valueof(514)
        self.databytecounts = databytecounts
        del databytecounts

        if (
            self.imagewidth == 0
            and self.imagelength == 0
            and self.dataoffsets
            and self.databytecounts
        ):
            # dimensions may be missing in some RAW formats
            # read dimensions from assumed JPEG encoded segment
            try:
                from .metadata import jpeg_shape

                fh.seek(self.dataoffsets[0])
                (
                    precision,
                    imagelength,
                    imagewidth,
                    samplesperpixel,
                ) = jpeg_shape(fh.read(min(self.databytecounts[0], 4096)))
            except Exception:  # noqa: S110
                pass
            else:
                self.imagelength = imagelength
                self.imagewidth = imagewidth
                self.samplesperpixel = samplesperpixel
                if 258 not in tags:
                    self.bitspersample = 8 if precision <= 8 else 16
                if 262 not in tags and samplesperpixel == 3:
                    self.photometric = PHOTOMETRIC.YCBCR
                if 259 not in tags:
                    self.compression = COMPRESSION.OJPEG
                if 278 not in tags:
                    self.rowsperstrip = imagelength

        elif self.compression == 6:
            # OJPEG hack. See libtiff v4.2.0 tif_dirread.c#L4082
            if 262 not in tags:
                # PhotometricInterpretation missing
                self.photometric = PHOTOMETRIC.YCBCR
            elif self.photometric == 2:
                # RGB -> YCbCr
                self.photometric = PHOTOMETRIC.YCBCR
            if 258 not in tags:
                # BitsPerSample missing
                self.bitspersample = 8
            if 277 not in tags and self.photometric in {0, 1, 2, 6}:
                # SamplesPerPixel missing
                self.samplesperpixel = 3

        elif self.is_lsm or (self.index != 0 and self.parent.is_lsm):
            # correct non standard LSM bitspersample tags
            tags[258]._fix_lsm_bitspersample()
            if self.compression == 1 and self.predictor != 1:
                # work around bug in LSM510 software
                self.predictor = PREDICTOR.NONE

        elif self.is_vista or (self.index != 0 and self.parent.is_vista):
            # ISS Vista writes wrong ImageDepth tag
            self.imagedepth = 1

        elif self.is_philips or (self.index != 0 and self.parent.is_philips):
            # Philips (DP v1.1) writes wrong ImageDepth and TileDepth tags
            self.imagedepth = 1
            self.tiledepth = 1

        elif self.is_stk:
            # read UIC1tag again now that plane count is known
            from .metadata import read_uic1tag

            tag = tags.get(33628)  # UIC1tag
            assert tag is not None
            fh.seek(tag.valueoffset)
            uic2tag = tags.get(33629)  # UIC2tag
            try:
                tag.value = read_uic1tag(
                    fh,
                    tiff.byteorder,
                    tag.dtype,
                    tag.count,
                    0,
                    planecount=uic2tag.count if uic2tag is not None else 1,
                )
            except Exception as exc:
                logger().warning(
                    f'{self!r} <tifffile.read_uic1tag> raised {exc!r:.128}'
                )

        elif parent._superres and self.compression in {65000, 65001, 65002}:
            horzbits = vertbits = 2
            if self.compression == 65002:
                horzbits = int(self.tags.valueof(65008, 2))
                vertbits = int(self.tags.valueof(65009, 2))
            self.imagewidth *= 2 ** (min(horzbits, parent._superres))
            self.imagelength *= 2 ** (min(vertbits, parent._superres))
            self.rowsperstrip *= 2 ** (min(vertbits, parent._superres))

        tag = tags.get(50839)
        if tag is not None:
            # decode IJMetadata tag
            from .metadata import imagej_metadata

            try:
                tag.value = imagej_metadata(
                    tag.value,
                    tags[50838].value,  # IJMetadataByteCounts
                    tiff.byteorder,
                )
            except Exception as exc:
                logger().warning(
                    f'{self!r} <tifffile.imagej_metadata> raised {exc!r:.128}'
                )

        # BitsPerSample
        value = tags.valueof(258)
        if value is not None:
            if self.bitspersample != 1:
                pass  # bitspersample was set by ojpeg hack
            elif tags[258].count == 1:
                self.bitspersample = int(value)
            else:
                # LSM might list more items than samplesperpixel
                value = value[: self.samplesperpixel]
                if any(v - value[0] for v in value):
                    self.bitspersample = value
                else:
                    self.bitspersample = int(value[0])

        # SampleFormat
        value = tags.valueof(339)
        if value is not None:
            if tags[339].count == 1:
                try:
                    self.sampleformat = SAMPLEFORMAT(value)
                except ValueError:
                    self.sampleformat = int(value)
            else:
                value = value[: self.samplesperpixel]
                if any(v - value[0] for v in value):
                    try:
                        self.sampleformat = SAMPLEFORMAT(value)
                    except ValueError:
                        self.sampleformat = int(value)
                else:
                    try:
                        self.sampleformat = SAMPLEFORMAT(value[0])
                    except ValueError:
                        self.sampleformat = int(value[0])
        elif self.bitspersample == 32 and (
            self.is_indica or (self.index != 0 and self.parent.is_indica)
        ):
            # IndicaLabsImageWriter does not write SampleFormat tag
            self.sampleformat = SAMPLEFORMAT.IEEEFP

        if 322 in tags:  # TileWidth
            self.rowsperstrip = 0
        elif 257 in tags:  # ImageLength
            if 278 not in tags or tags[278].count > 1:  # RowsPerStrip
                self.rowsperstrip = self.imagelength
            self.rowsperstrip = min(self.rowsperstrip, self.imagelength)
            # self.stripsperimage = math.floor(
            #    float(self.imagelength + self.rowsperstrip - 1) /
            #    self.rowsperstrip)

        # determine dtype
        dtypestr = TIFF.SAMPLE_DTYPES.get(
            (self.sampleformat, self.bitspersample), None
        )
        dtype = numpy.dtype(dtypestr) if dtypestr is not None else None
        self.dtype = self._dtype = dtype

        # determine shape of data
        imagelength = self.imagelength
        imagewidth = self.imagewidth
        imagedepth = self.imagedepth
        samplesperpixel = self.samplesperpixel

        if self.photometric == 2 or samplesperpixel > 1:  # PHOTOMETRIC.RGB
            if self.planarconfig == 1:
                self.shaped = (
                    1,
                    imagedepth,
                    imagelength,
                    imagewidth,
                    samplesperpixel,
                )
                if imagedepth == 1:
                    self.shape = (imagelength, imagewidth, samplesperpixel)
                    self.axes = 'YXS'
                else:
                    self.shape = (
                        imagedepth,
                        imagelength,
                        imagewidth,
                        samplesperpixel,
                    )
                    self.axes = 'ZYXS'
            else:
                self.shaped = (
                    samplesperpixel,
                    imagedepth,
                    imagelength,
                    imagewidth,
                    1,
                )
                if imagedepth == 1:
                    self.shape = (samplesperpixel, imagelength, imagewidth)
                    self.axes = 'SYX'
                else:
                    self.shape = (
                        samplesperpixel,
                        imagedepth,
                        imagelength,
                        imagewidth,
                    )
                    self.axes = 'SZYX'
        else:
            self.shaped = (1, imagedepth, imagelength, imagewidth, 1)
            if imagedepth == 1:
                self.shape = (imagelength, imagewidth)
                self.axes = 'YX'
            else:
                self.shape = (imagedepth, imagelength, imagewidth)
                self.axes = 'ZYX'

        if not self.databytecounts:
            self.databytecounts = (
                product(self.shape) * (self.bitspersample // 8),
            )
            if self.compression != 1:
                logger().error(f'{self!r} missing ByteCounts tag')

        if imagelength and self.rowsperstrip and not self.is_lsm:
            # fix incorrect number of strip bytecounts and offsets
            maxstrips = (
                int(
                    math.floor(imagelength + self.rowsperstrip - 1)
                    / self.rowsperstrip
                )
                * self.imagedepth
            )
            if self.planarconfig == 2:
                maxstrips *= self.samplesperpixel
            if maxstrips != len(self.databytecounts):
                logger().error(
                    f'{self!r} incorrect StripByteCounts count '
                    f'({len(self.databytecounts)} != {maxstrips})'
                )
                self.databytecounts = self.databytecounts[:maxstrips]
            if maxstrips != len(self.dataoffsets):
                logger().error(
                    f'{self!r} incorrect StripOffsets count '
                    f'({len(self.dataoffsets)} != {maxstrips})'
                )
                self.dataoffsets = self.dataoffsets[:maxstrips]

        value = tags.valueof(42113)  # GDAL_NODATA
        if value is not None and dtype is not None:
            try:
                pytype = type(dtype.type(0).item())
                value = value.replace(',', '.')  # comma decimal separator
                self.nodata = pytype(value)
                if not numpy.can_cast(
                    numpy.min_scalar_type(self.nodata), dtype
                ):
                    msg = f'{self.nodata} is not castable to {dtype}'
                    raise ValueError(msg)
            except Exception as exc:
                logger().warning(
                    f'{self!r} parsing GDAL_NODATA tag raised {exc!r:.128}'
                )
                self.nodata = 0

        mcustarts = tags.valueof(65426)
        if mcustarts is not None and self.is_ndpi:
            # use NDPI JPEG McuStarts as tile offsets
            from .metadata import ndpi_jpeg_tile

            mcustarts = mcustarts.astype(numpy.int64)
            high = tags.valueof(65432)
            if high is not None:
                # McuStartsHighBytes
                high = high.astype(numpy.uint64)
                high <<= 32
                mcustarts += high.astype(numpy.int64)
            fh.seek(self.dataoffsets[0])
            jpegheader = fh.read(mcustarts[0])
            try:
                (
                    self.tilelength,
                    self.tilewidth,
                    self.jpegheader,
                ) = ndpi_jpeg_tile(jpegheader)
            except ValueError as exc:
                logger().warning(
                    f'{self!r} <tifffile.ndpi_jpeg_tile> raised {exc!r:.128}'
                )
            else:
                # TODO: optimize tuple(ndarray.tolist())
                databytecounts = numpy.diff(
                    mcustarts, append=self.databytecounts[0]
                )
                self.databytecounts = tuple(databytecounts.tolist())
                mcustarts += self.dataoffsets[0]
                self.dataoffsets = tuple(mcustarts.tolist())

    @cached_property
    def decode(
        self,
    ) -> Callable[
        ...,
        tuple[
            NDArray[Any] | None,
            tuple[int, int, int, int, int],
            tuple[int, int, int, int],
        ],
    ]:
        """Return decoded segment, its shape, and indices in image.

        The decode function is implemented as a closure and has the following
        signature:

        Parameters:
            data (Union[bytes, None]):
                Encoded bytes of segment (strip or tile) or None for empty
                segments.
            index (int):
                Index of segment in Offsets and Bytecount tag values.
            jpegtables (Optional[bytes]):
                For JPEG compressed segments only, value of JPEGTables tag
                if any.

        Returns:
            - Decoded segment or None for empty segments.
            - Position of segment in image array of normalized shape
              (separate sample, depth, length, width, contig sample).
            - Shape of segment (depth, length, width, contig samples).
              The shape of strips depends on their linear index.

        Raises:
            ValueError or NotImplementedError:
                Decoding is not supported.
            TiffFileError:
                Invalid TIFF structure.

        """
        if self.hash in self.parent._parent._decoders:
            return self.parent._parent._decoders[self.hash]

        from .decoders import build_decode_closure

        decode = build_decode_closure(self)
        self.parent._parent._decoders[self.hash] = decode
        return decode

    def segments(
        self,
        *,
        lock: threading.RLock | NullContext | None = None,
        maxworkers: int | None = None,
        func: Callable[..., Any] | None = None,  # TODO: type this
        sort: bool = False,
        buffersize: int | None = None,
        _fullsize: bool | None = None,
        segment_filter: Sequence[int] | None = None,
    ) -> Iterator[
        tuple[
            NDArray[Any] | None,
            tuple[int, int, int, int, int],
            tuple[int, int, int, int],
        ]
    ]:
        """Return iterator over decoded tiles or strips.

        Parameters:
            lock:
                Reentrant lock to synchronize file seeks and reads.
            maxworkers:
                Maximum number of threads to concurrently decode segments.
            func:
                Function to process decoded segment.
            sort:
                Read segments from file in order of their offsets.
            buffersize:
                Approximate number of bytes to read from file in one pass.
                The default is :py:attr:`_TIFF.BUFFERSIZE`.
            _fullsize:
                Internal use.
            segment_filter:
                Indices of segments (strips or tiles) to decode.
                If provided, only the specified segments are read and
                decoded.  Other segments are skipped entirely.
                The indices refer to the linear segment order (matching
                dataoffsets/databytecounts).

        Yields:
            - Decoded segment or None for empty segments.
            - Position of segment in image array of normalized shape
              (separate sample, depth, length, width, contig sample).
            - Shape of segment (depth, length, width, contig samples).
              The shape of strips depends on their linear index.

        """
        keyframe = self.keyframe  # self or keyframe
        fh = self.parent.filehandle
        if lock is None:
            lock = fh.lock
        if _fullsize is None:
            _fullsize = keyframe.is_tiled

        decodeargs: dict[str, Any] = {'_fullsize': bool(_fullsize)}
        if keyframe.compression in {6, 7, 34892, 33007}:  # JPEG
            decodeargs['jpegtables'] = self.jpegtables
            decodeargs['jpegheader'] = keyframe.jpegheader

        if func is None:

            def decode(args, decodeargs=decodeargs, decode=keyframe.decode):
                return decode(*args, **decodeargs)

        else:

            def decode(args, decodeargs=decodeargs, decode=keyframe.decode):
                return func(decode(*args, **decodeargs))

        number_segments = product(self.chunked)

        if segment_filter is not None:
            # Read only specified segments
            offsets = self.dataoffsets
            bytecounts = self.databytecounts
            filtered_offsets = [offsets[i] for i in segment_filter]
            filtered_bytecounts = [bytecounts[i] for i in segment_filter]
            seg_indices = list(segment_filter)
            seg_length = len(segment_filter)
        else:
            filtered_offsets = self.dataoffsets
            filtered_bytecounts = self.databytecounts
            seg_indices = None  # read_segments defaults to range(length)
            seg_length = number_segments

        if maxworkers is None or maxworkers < 1:
            maxworkers = keyframe.maxworkers
        if maxworkers < 2:
            for segment in fh.read_segments(
                filtered_offsets,
                filtered_bytecounts,
                indices=seg_indices,
                length=seg_length,
                lock=lock,
                sort=sort,
                buffersize=buffersize,
                flat=True,
            ):
                yield decode(segment)
        else:
            # reduce memory overhead by processing chunks of up to
            # buffersize of segments because ThreadPoolExecutor.map is not
            # collecting iterables lazily
            with ThreadPoolExecutor(maxworkers) as executor:
                for segments in fh.read_segments(
                    filtered_offsets,
                    filtered_bytecounts,
                    indices=seg_indices,
                    length=seg_length,
                    lock=lock,
                    sort=sort,
                    buffersize=buffersize,
                    flat=False,
                ):
                    yield from executor.map(decode, segments)

    def asarray(
        self,
        *,
        out: OutputType = None,
        squeeze: bool = True,
        lock: threading.RLock | NullContext | None = None,
        device: str | None = None,
        maxworkers: int | None = None,
        buffersize: int | None = None,
    ) -> NDArray[Any]:
        """Return image from page as NumPy array.

        Parameters:
            out:
                Specifies how image array is returned.
                By default, a new NumPy array is created.
                If a *numpy.ndarray*, a writable array to which the image
                is copied.
                If *'memmap'*, directly memory-map the image data in the
                file if possible; else create a memory-mapped array in a
                temporary file.
                If a *string* or *open file*, the file used to create a
                memory-mapped array.
            squeeze:
                Remove all length-1 dimensions (except X and Y) from
                image array.
                If *False*, return the image array with normalized
                5-dimensional shape :py:attr:`TiffPage.shaped`.
            lock:
                Reentrant lock to synchronize seeks and reads from file.
                The default is the lock of the parent's file handle.
            device:
                If not *None*, return a ``torch.Tensor`` on the specified
                device instead of a NumPy array.
                For example, ``'cuda'`` or ``'cuda:0'``.
                Requires PyTorch.
            maxworkers:
                Maximum number of threads to concurrently decode segments.
                If *None* or *0*, use up to :py:attr:`_TIFF.MAXWORKERS`
                threads. See remarks in :py:meth:`TiffFile.asarray`.
            buffersize:
                Approximate number of bytes to read from file in one pass.
                The default is :py:attr:`_TIFF.BUFFERSIZE`.

        Returns:
            NumPy array of decompressed, unpredicted, and unpacked image data
            read from Strip/Tile Offsets/ByteCounts, formatted according to
            shape and dtype metadata found in tags and arguments.
            Photometric conversion, premultiplied alpha, orientation, and
            colorimetry corrections are not applied.
            Specifically, CMYK images are not converted to RGB, MinIsWhite
            images are not inverted, color palettes are not applied,
            gamma is not corrected, and CFA images are not demosaciced.
            Exception are YCbCr JPEG compressed images, which are converted to
            RGB.

        Raises:
            ValueError:
                Format of image in file is not supported and cannot be decoded.

        """
        from .tifffile import TIFF, create_output

        keyframe = self.keyframe  # self or keyframe

        if 0 in keyframe.shaped or keyframe._dtype is None:
            return numpy.empty((0,), keyframe.dtype)

        if len(self.dataoffsets) == 0:
            msg = 'missing data offset'
            raise TiffFileError(msg)

        fh = self.parent.filehandle
        if lock is None:
            lock = fh.lock

        if (
            isinstance(out, str)
            and out == 'memmap'
            and keyframe.is_memmappable
        ):
            # direct memory map array in file
            with lock:
                closed = fh.closed
                if closed:
                    warnings.warn(
                        f'{self!r} reading array from closed file',
                        UserWarning,
                        stacklevel=2,
                    )
                    fh.open()
                result = fh.memmap_array(
                    keyframe.parent.byteorder + keyframe._dtype.char,
                    keyframe.shaped,
                    offset=self.dataoffsets[0],
                )

        elif keyframe.is_contiguous:
            # read contiguous bytes to array
            if keyframe.is_subsampled:
                msg = 'chroma subsampling not supported'
                raise NotImplementedError(msg)
            if out is not None:
                out = create_output(out, keyframe.shaped, keyframe._dtype)
            with lock:
                closed = fh.closed
                if closed:
                    warnings.warn(
                        f'{self!r} reading array from closed file',
                        UserWarning,
                        stacklevel=2,
                    )
                    fh.open()
                fh.seek(self.dataoffsets[0])
                result = fh.read_array(
                    keyframe.parent.byteorder + keyframe._dtype.char,
                    product(keyframe.shaped),
                    out=out,
                )
            if keyframe.fillorder == 2:
                import imagecodecs

                result = imagecodecs.bitorder_decode(result, out=result)
            if keyframe.predictor != 1:
                # predictors without compression
                unpredict = TIFF.UNPREDICTORS[keyframe.predictor]
                if keyframe.predictor == 2:
                    result = unpredict(result, axis=-2, out=result)
                else:
                    # floatpred cannot decode in-place
                    out = unpredict(result, axis=-2, out=result)
                    result[:] = out

        elif (
            keyframe.jpegheader is not None
            and keyframe is self
            and 273 in self.tags  # striped ...
            and self.is_tiled  # but reported as tiled
            # TODO: imagecodecs can decode larger JPEG
            and self.imagewidth <= 65500
            and self.imagelength <= 65500
        ):
            # decode the whole NDPI JPEG strip
            with lock:
                closed = fh.closed
                if closed:
                    warnings.warn(
                        f'{self!r} reading array from closed file',
                        UserWarning,
                        stacklevel=2,
                    )
                    fh.open()
                fh.seek(self.tags[273].value[0])  # StripOffsets
                data = fh.read(self.tags[279].value[0])  # StripByteCounts
            decompress = TIFF.DECOMPRESSORS[self.compression]
            result = decompress(
                data,
                bitspersample=self.bitspersample,
                out=out,
                # shape=(self.imagelength, self.imagewidth)
            )
            del data

        else:
            # decode individual strips or tiles
            with lock:
                closed = fh.closed
                if closed:
                    warnings.warn(
                        f'{self!r} reading array from closed file',
                        UserWarning,
                        stacklevel=2,
                    )
                    fh.open()
                # init TiffPage.decode function under lock
                keyframe.decode  # noqa: B018

            result = create_output(out, keyframe.shaped, keyframe._dtype)

            def func(
                decoderesult: tuple[
                    NDArray[Any] | None,
                    tuple[int, int, int, int, int],
                    tuple[int, int, int, int],
                ],
                keyframe: TiffPage = keyframe,
                out: NDArray[Any] = result,
            ) -> None:
                # copy decoded segments to output array
                segment, (s, d, h, w, _), shape = decoderesult
                if segment is None:
                    out[
                        s, d : d + shape[0], h : h + shape[1], w : w + shape[2]
                    ] = keyframe.nodata
                else:
                    out[
                        s, d : d + shape[0], h : h + shape[1], w : w + shape[2]
                    ] = segment[
                        : keyframe.imagedepth - d,
                        : keyframe.imagelength - h,
                        : keyframe.imagewidth - w,
                    ]
                # except IndexError:
                #     pass  # corrupted file, for example, with too many strips

            for _ in self.segments(
                func=func,
                lock=lock,
                maxworkers=maxworkers,
                buffersize=buffersize,
                sort=True,
                _fullsize=False,
            ):
                pass

        result.shape = keyframe.shaped
        if squeeze:
            try:
                result.shape = keyframe.shape
            except ValueError as exc:
                logger().warning(
                    f'{self!r} <asarray> failed to reshape '
                    f'{result.shape} to {keyframe.shape}, raised {exc!r:.128}'
                )

        if closed:
            # TODO: close file if an exception occurred above
            fh.close()

        if device is not None:
            from .gpu import numpy_to_tensor, parse_device

            dev = parse_device(device)
            if dev is not None:
                return numpy_to_tensor(result, dev)
        return result

    def aszarr(self, **kwargs: Any) -> ZarrTiffStore:
        """Return image from page as Zarr store.

        Parameters:
            **kwargs: Passed to :py:class:`ZarrTiffStore`.

        """
        from .zarr import ZarrTiffStore

        return ZarrTiffStore(self, **kwargs)

    def asrgb(
        self,
        *,
        uint8: bool = False,
        alpha: Container[int] | None = None,
        **kwargs: Any,
    ) -> NDArray[Any]:
        """Return image as RGB(A). Work in progress. Do not use.

        :meta private:

        """
        from .utils import apply_colormap

        data = self.asarray(**kwargs)
        keyframe = self.keyframe  # self or keyframe

        if keyframe.photometric == PHOTOMETRIC.PALETTE:
            colormap = keyframe.colormap
            if colormap is None:
                msg = 'no colormap'
                raise ValueError(msg)
            if (
                colormap.shape[1] < 2**keyframe.bitspersample
                or keyframe.dtype is None
                or keyframe.dtype.char not in 'BH'
            ):
                msg = 'cannot apply colormap'
                raise ValueError(msg)
            if uint8:
                if colormap.max() > 255:
                    colormap >>= 8
                colormap = colormap.astype(numpy.uint8)
            if 'S' in keyframe.axes:
                data = data[..., 0] if keyframe.planarconfig == 1 else data[0]
            data = apply_colormap(data, colormap)

        elif keyframe.photometric == PHOTOMETRIC.RGB:
            if keyframe.extrasamples:
                if alpha is None:
                    alpha = EXTRASAMPLE
                for i, exs in enumerate(keyframe.extrasamples):
                    if exs in EXTRASAMPLE:
                        if keyframe.planarconfig == 1:
                            data = data[..., [0, 1, 2, 3 + i]]
                        else:
                            data = data[:, [0, 1, 2, 3 + i]]
                        break
            elif keyframe.planarconfig == 1:
                data = data[..., :3]
            else:
                data = data[:, :3]
            # TODO: convert to uint8?

        # elif keyframe.photometric == PHOTOMETRIC.MINISBLACK:
        #     raise NotImplementedError
        # elif keyframe.photometric == PHOTOMETRIC.MINISWHITE:
        #     raise NotImplementedError
        # elif keyframe.photometric == PHOTOMETRIC.SEPARATED:
        #     raise NotImplementedError
        else:
            raise NotImplementedError
        return data

    def _gettags(
        self,
        codes: Container[int] | None = None,
        /,
        lock: threading.RLock | None = None,
    ) -> list[tuple[int, TiffTag]]:
        """Return list of (code, TiffTag)."""
        return [
            (tag.code, tag)
            for tag in self.tags
            if codes is None or tag.code in codes
        ]

    def _nextifd(self) -> int:
        """Return offset to next IFD from file."""
        fh = self.parent.filehandle
        tiff = self.parent.tiff
        fh.seek(self.offset)
        tagno = struct.unpack(tiff.tagnoformat, fh.read(tiff.tagnosize))[0]
        fh.seek(self.offset + tiff.tagnosize + tagno * tiff.tagsize)
        return int(
            struct.unpack(tiff.offsetformat, fh.read(tiff.offsetsize))[0]
        )

    def aspage(self) -> TiffPage:
        """Return TiffPage instance."""
        return self

    @property
    def index(self) -> int:
        """Index of page in IFD chain."""
        return self._index[-1]

    @property
    def treeindex(self) -> tuple[int, ...]:
        """Index of page in IFD tree."""
        return self._index

    @property
    def keyframe(self) -> TiffPage:
        """Self."""
        return self

    @keyframe.setter
    def keyframe(self, index: TiffPage) -> None:
        return

    @property
    def name(self) -> str:
        """Name of image array."""
        index = self._index if len(self._index) > 1 else self._index[0]
        return f'TiffPage {index}'

    @property
    def ndim(self) -> int:
        """Number of dimensions in image array."""
        return len(self.shape)

    @cached_property
    def dims(self) -> tuple[str, ...]:
        """Names of dimensions in image array."""
        from .tifffile import TIFF

        names = TIFF.AXES_NAMES
        return tuple(names[ax] for ax in self.axes)

    @cached_property
    def sizes(self) -> dict[str, int]:
        """Ordered map of dimension names to lengths."""
        from .tifffile import TIFF

        shape = self.shape
        names = TIFF.AXES_NAMES
        return {names[ax]: shape[i] for i, ax in enumerate(self.axes)}

    @cached_property
    def coords(self) -> dict[str, NDArray[Any]]:
        """Ordered map of dimension names to coordinate arrays."""
        from .tifffile import TIFF

        resolution = self.get_resolution()
        coords: dict[str, NDArray[Any]] = {}

        for ax, size in zip(self.axes, self.shape, strict=True):
            name = TIFF.AXES_NAMES[ax]
            value = None
            step: float = 1

            if ax == 'X':
                step = resolution[0]
            elif ax == 'Y':
                step = resolution[1]
            elif ax == 'S':
                value = self._sample_names()
            elif ax == 'Z' and resolution[0] == resolution[1]:
                # a ZResolution tag doesn't exist
                # use XResolution if it agrees with YResolution
                step = resolution[0]

            if value is not None:
                coords[name] = numpy.asarray(value)
            elif step == 0 or step == 1 or size == 0:  # noqa: PLR1714
                coords[name] = numpy.arange(size)
            else:
                coords[name] = numpy.linspace(
                    0, size / step, size, endpoint=False, dtype=numpy.float32
                )
            assert len(coords[name]) == size
        return coords

    @cached_property
    def attr(self) -> dict[str, Any]:
        """Arbitrary metadata associated with image array."""
        # TODO: what to return?
        return {}

    @cached_property
    def size(self) -> int:
        """Number of elements in image array."""
        return product(self.shape)

    @cached_property
    def nbytes(self) -> int:
        """Number of bytes in image array."""
        if self.dtype is None:
            return 0
        return self.size * self.dtype.itemsize

    @property
    def colormap(self) -> NDArray[numpy.uint16] | None:
        """Value of Colormap tag."""
        return self.tags.valueof(320)

    @property
    def iccprofile(self) -> bytes | None:
        """Value of InterColorProfile tag."""
        return self.tags.valueof(34675)

    @property
    def transferfunction(self) -> NDArray[numpy.uint16] | None:
        """Value of TransferFunction tag."""
        return self.tags.valueof(301)

    def get_resolution(
        self,
        unit: RESUNIT | int | str | None = None,
        scale: float | None = None,
    ) -> tuple[float, float]:
        """Return number of pixels per unit in X and Y dimensions.

        By default, the XResolution and YResolution tag values are returned.
        Missing tag values are set to 1.

        Parameters:
            unit:
                Unit of measurement of returned values.
                The default is the value of the ResolutionUnit tag.
            scale:
                Factor to convert resolution values to meter unit.
                The default is determined from the ResolutionUnit tag.

        """
        scales = {
            1: 1,  # meter, no unit
            2: 100 / 2.54,  # INCH
            3: 100,  # CENTIMETER
            4: 1000,  # MILLIMETER
            5: 1000000,  # MICROMETER
        }
        if unit is not None:
            unit = enumarg(RESUNIT, unit)
            try:
                if scale is None:
                    resolutionunit = self.tags.valueof(296, default=2)
                    scale = scales[resolutionunit]
            except Exception as exc:
                logger().warning(
                    f'{self!r} <get_resolution> raised {exc!r:.128}'
                )
                scale = 1
            else:
                scale2 = scales[unit]
                if scale % scale2 == 0:
                    scale //= scale2
                else:
                    scale /= scale2
        elif scale is None:
            scale = 1

        resolution: list[float] = []
        n: int
        d: int
        for code in 282, 283:
            try:
                n, d = self.tags.valueof(code, default=(1, 1))
                if d == 0:
                    value = n * scale
                elif n % d == 0:
                    value = n // d * scale
                else:
                    value = n / d * scale
            except Exception:
                value = 1
            resolution.append(value)
        return resolution[0], resolution[1]

    @cached_property
    def resolution(self) -> tuple[float, float]:
        """Number of pixels per resolutionunit in X and Y directions."""
        # values are returned in (somewhat unexpected) XY order to
        # keep symmetry with the TiffWriter.write resolution argument
        resolution = self.get_resolution()
        return float(resolution[0]), float(resolution[1])

    @property
    def resolutionunit(self) -> int:
        """Unit of measurement for X and Y resolutions."""
        return self.tags.valueof(296, default=2)

    @property
    def datetime(self) -> DateTime | None:
        """Date and time of image creation."""
        value = self.tags.valueof(306)
        if value is None:
            return None
        try:
            return strptime(value)
        except (TypeError, ValueError):
            pass
        return None

    @property
    def tile(self) -> tuple[int, ...] | None:
        """Tile depth, length, and width."""
        if not self.is_tiled:
            return None
        if self.tiledepth > 1:
            return (self.tiledepth, self.tilelength, self.tilewidth)
        return (self.tilelength, self.tilewidth)

    @cached_property
    def chunks(self) -> tuple[int, ...]:
        """Shape of images in tiles or strips."""
        shape: list[int] = []
        if self.tiledepth > 1:
            shape.append(self.tiledepth)
        if self.is_tiled:
            shape.extend((self.tilelength, self.tilewidth))
        else:
            shape.extend((self.rowsperstrip, self.imagewidth))
        if self.planarconfig == 1 and self.samplesperpixel > 1:
            shape.append(self.samplesperpixel)
        return tuple(shape)

    @cached_property
    def chunked(self) -> tuple[int, ...]:
        """Shape of chunked image."""
        shape: list[int] = []
        if self.planarconfig == 2 and self.samplesperpixel > 1:
            shape.append(self.samplesperpixel)
        if self.is_tiled:
            if self.imagedepth > 1:
                shape.append(
                    (self.imagedepth + self.tiledepth - 1) // self.tiledepth
                )
            shape.append(
                (self.imagelength + self.tilelength - 1) // self.tilelength
            )
            shape.append(
                (self.imagewidth + self.tilewidth - 1) // self.tilewidth
            )
        else:
            if self.imagedepth > 1:
                shape.append(self.imagedepth)
            shape.append(
                (self.imagelength + self.rowsperstrip - 1) // self.rowsperstrip
            )
            shape.append(1)
        if self.planarconfig == 1 and self.samplesperpixel > 1:
            shape.append(1)
        return tuple(shape)

    @cached_property
    def hash(self) -> int:
        """Checksum to identify pages in same series.

        Pages with the same hash can use the same decode function.
        The hash is calculated from the following properties:
        :py:attr:`TiffFile.tiff`,
        :py:attr:`TiffPage.shaped`,
        :py:attr:`TiffPage.rowsperstrip`,
        :py:attr:`TiffPage.tilewidth`,
        :py:attr:`TiffPage.tilelength`,
        :py:attr:`TiffPage.tiledepth`,
        :py:attr:`TiffPage.sampleformat`,
        :py:attr:`TiffPage.bitspersample`,
        :py:attr:`TiffPage.fillorder`,
        :py:attr:`TiffPage.predictor`,
        :py:attr:`TiffPage.compression`,
        :py:attr:`TiffPage.extrasamples`, and
        :py:attr:`TiffPage.photometric`.

        """
        return hash(
            (
                *self.shaped,
                self.parent.tiff,
                self.rowsperstrip,
                self.tilewidth,
                self.tilelength,
                self.tiledepth,
                self.sampleformat,
                self.bitspersample,
                self.fillorder,
                self.predictor,
                self.compression,
                self.extrasamples,
                self.photometric,
            )
        )

    @cached_property
    def pages(self) -> TiffPages | None:
        """Sequence of sub-pages, SubIFDs."""
        if 330 not in self.tags:
            return None
        return TiffPages(self, index=self.index)

    @cached_property
    def maxworkers(self) -> int:
        """Maximum number of threads for decoding segments.

        A value of 0 disables multi-threading also when stacking pages.

        """
        from .tifffile import TIFF

        try:
            import imagecodecs
        except ImportError:
            imagecodecs = None  # type: ignore[assignment]

        if self.is_contiguous or self.dtype is None:
            return 0
        if self.compression in TIFF.IMAGE_COMPRESSIONS:
            return min(TIFF.MAXWORKERS, len(self.dataoffsets))
        bytecount = product(self.chunks) * self.dtype.itemsize
        if bytecount < 2048:
            # disable multi-threading for small segments
            return 0
        if self.compression == 5 and bytecount < 14336:
            # disable multi-threading for small LZW compressed segments
            return 0
        if len(self.dataoffsets) < 4:
            return 1
        if imagecodecs is not None and (
            self.compression != 1 or self.fillorder != 1 or self.predictor != 1
        ):
            return min(TIFF.MAXWORKERS, len(self.dataoffsets))
        return 2  # optimum for large number of uncompressed tiles

    @cached_property
    def is_contiguous(self) -> bool:
        """Image data is stored contiguously.

        Contiguous image data can be read from
        ``offset=TiffPage.dataoffsets[0]`` with ``size=TiffPage.nbytes``.
        Excludes prediction and fillorder.

        """
        if (
            self.sampleformat == 5
            or self.compression != 1
            or self.bitspersample not in {8, 16, 32, 64}
        ):
            return False
        if 322 in self.tags:  # TileWidth
            if (
                self.imagewidth != self.tilewidth
                or self.imagelength % self.tilelength
                or self.tilewidth % 16
                or self.tilelength % 16
            ):
                return False
            if (
                32997 in self.tags  # ImageDepth
                and 32998 in self.tags  # TileDepth
                and (
                    self.imagelength != self.tilelength
                    or self.imagedepth % self.tiledepth
                )
            ):
                return False
        offsets = self.dataoffsets
        bytecounts = self.databytecounts
        if len(offsets) == 0:
            return False
        if len(offsets) == 1:
            return True
        if self.is_stk or self.is_lsm:
            return True
        if sum(bytecounts) != self.nbytes:
            return False
        return all(
            bytecounts[i] != 0 and offsets[i] + bytecounts[i] == offsets[i + 1]
            for i in range(len(offsets) - 1)
        )

    @cached_property
    def is_final(self) -> bool:
        """Image data are stored in final form. Excludes byte-swapping."""
        return (
            self.is_contiguous
            and self.fillorder == 1
            and self.predictor == 1
            and not self.is_subsampled
        )

    @cached_property
    def is_memmappable(self) -> bool:
        """Image data in file can be memory-mapped to NumPy array."""
        return (
            self.parent.filehandle.is_file
            and self.is_final
            # and (self.bitspersample == 8 or self.parent.isnative)
            # aligned?
            and self.dtype is not None
            and self.dataoffsets[0] % self.dtype.itemsize == 0
        )

    def __repr__(self) -> str:
        index = self._index if len(self._index) > 1 else self._index[0]
        return f'<tifffile.TiffPage {index} @{self.offset}>'

    def __str__(self) -> str:
        return self._str()

    def _str(self, detail: int = 0, width: int = 79) -> str:
        """Return string containing information about TiffPage."""
        if self.keyframe != self:
            return TiffFrame._str(
                self, detail, width  # type: ignore[arg-type]
            )
        attr = ''
        for name in ('memmappable', 'final', 'contiguous'):
            attr = getattr(self, 'is_' + name)
            if attr:
                attr = name.upper()
                break

        def tostr(name: str, /, skip: int = 1) -> str:
            obj = getattr(self, name)
            if obj == skip:
                return ''
            try:
                value = obj.name
            except AttributeError:
                return ''
            return str(value)

        info = '  '.join(
            s.lower()
            for s in (
                'x'.join(str(i) for i in self.shape),
                f'{SAMPLEFORMAT(self.sampleformat).name}{self.bitspersample}',
                ' '.join(
                    i
                    for i in (
                        PHOTOMETRIC(self.photometric).name,
                        'REDUCED' if self.is_reduced else '',
                        'MASK' if self.is_mask else '',
                        'TILED' if self.is_tiled else '',
                        tostr('compression'),
                        tostr('planarconfig'),
                        tostr('predictor'),
                        tostr('fillorder'),
                        attr,
                    )
                    if i
                ),
                '|'.join(f.upper() for f in sorted(self.flags)),
            )
            if s
        )
        index = self._index if len(self._index) > 1 else self._index[0]
        info = f'TiffPage {index} @{self.offset}  {info}'
        if detail <= 0:
            return info
        info_list = [info, self.tags._str(detail + 1, width=width)]
        if detail > 1:
            for name in ('ndpi_tags',):
                attr = getattr(self, name, '')
                if attr:
                    info_list.append(
                        f'{name.upper()}\n'
                        f'{pformat(attr, width=width, height=detail * 8)}'
                    )
        if detail > 3:
            try:
                data = self.asarray()
                info_list.append(
                    f'DATA\n{pformat(data, width=width, height=detail * 8)}'
                )
            except Exception:  # noqa: S110
                pass
        return '\n\n'.join(info_list)

    def _sample_names(self) -> list[str] | None:
        """Return names of samples."""
        if 'S' not in self.axes:
            return None
        samples = self.shape[self.axes.find('S')]
        extrasamples = len(self.extrasamples)
        if samples < 1 or extrasamples > 2:
            return None
        match self.photometric:
            case 0:
                names = ['WhiteIsZero']
            case 1:
                names = ['BlackIsZero']
            case 2:
                names = ['Red', 'Green', 'Blue']
            case 5:
                names = ['Cyan', 'Magenta', 'Yellow', 'Black']
            case 6:
                if self.compression in {6, 7, 34892, 33007}:
                    # YCBCR -> RGB for JPEG
                    names = ['Red', 'Green', 'Blue']
                else:
                    names = ['Luma', 'Cb', 'Cr']
            case _:
                return None
        if extrasamples > 0:
            names += [enumarg(EXTRASAMPLE, self.extrasamples[0]).name.title()]
        if extrasamples > 1:
            names += [enumarg(EXTRASAMPLE, self.extrasamples[1]).name.title()]
        if len(names) != samples:
            return None
        return names

    @cached_property
    def flags(self) -> set[str]:
        r"""Set of ``is\_\*`` properties that are True."""
        from .tifffile import TIFF

        return {
            name.lower()
            for name in TIFF.PAGE_FLAGS
            if getattr(self, 'is_' + name)
        }

    @cached_property
    def eer_tags(self) -> dict[str, Any] | None:
        """Consolidated metadata from EER tags 65001-65009."""
        if not self.is_eer:
            return None
        from .metadata import eer_xml_metadata

        result = {}
        for code in range(65001, 65007):
            value = self.tags.valueof(code)
            if (
                value is None
                or not isinstance(value, bytes)
                or not value.startswith(b'<metadata>')
            ):
                continue
            try:
                result.update(eer_xml_metadata(value.decode()))
            except Exception as exc:
                logger().warning(
                    f'{self!r} eer_xml_metadata failed for tag {code}'
                    f'{exc!r:.128}'
                )
        return result

    @cached_property
    def nuvu_tags(self) -> dict[str, Any] | None:
        """Consolidated metadata from Nuvu tags."""
        if not self.is_nuvu:
            return None
        result: dict[str, Any] = {}
        used: set[int] = set()
        for tag in self.tags:
            if (
                tag.code < 65000
                or tag.code in used
                or tag.dtype != 2
                or tag.value[:7] != "Field '"
            ):
                continue
            try:
                value = tag.value.split("'")
                name = value[3]
                code = int(value[1])
            except Exception as exc:
                logger().warning(
                    f'{self!r} corrupted Nuvu tag {tag.code} ({exc})'
                )
                continue
            result[name] = self.tags.valueof(code)
            used.add(code)
        return result

    @cached_property
    def andor_tags(self) -> dict[str, Any] | None:
        """Consolidated metadata from Andor tags."""
        if not self.is_andor:
            return None
        result = {'Id': self.tags[4864].value}  # AndorId
        for tag in self.tags:  # list(self.tags.values()):
            code = tag.code
            if not 4864 < code < 5031:
                continue
            name = tag.name
            name = name[5:] if len(name) > 5 else name
            result[name] = tag.value
            # del self.tags[code]
        return result

    @cached_property
    def epics_tags(self) -> dict[str, Any] | None:
        """Consolidated metadata from EPICS areaDetector tags.

        Use the :py:func:`epics_datetime` function to get a datetime object
        from the epicsTSSec and epicsTSNsec tags.

        """
        if not self.is_epics:
            return None
        result = {}
        for tag in self.tags:  # list(self.tags.values()):
            code = tag.code
            if not 65000 <= code < 65500:
                continue
            value = tag.value
            match code:
                case 65000:
                    # not a POSIX timestamp
                    # https://github.com/bluesky/area-detector-handlers/issues/20
                    result['timeStamp'] = float(value)
                case 65001:
                    result['uniqueID'] = int(value)
                case 65002:
                    result['epicsTSSec'] = int(value)
                case 65003:
                    result['epicsTSNsec'] = int(value)
                case _:
                    key, value = value.split(':', 1)
                    result[key] = astype(value)
            # del self.tags[code]
        return result

    @cached_property
    def ndpi_tags(self) -> dict[str, Any] | None:
        """Consolidated metadata from Hamamatsu NDPI tags."""
        # TODO: parse 65449 ini style comments
        from .tifffile import TIFF

        if not self.is_ndpi:
            return None
        tags = self.tags
        result = {}
        for name in ('Make', 'Model', 'Software'):
            result[name] = tags[name].value
        for code, name in TIFF.NDPI_TAGS.items():
            if code in tags:
                result[name] = tags[code].value
                # del tags[code]
        if 'McuStarts' in result:
            mcustarts = result['McuStarts']
            if 'McuStartsHighBytes' in result:
                high = result['McuStartsHighBytes'].astype(numpy.uint64)
                high <<= 32
                mcustarts = mcustarts.astype(numpy.uint64)
                mcustarts += high
                del result['McuStartsHighBytes']
            result['McuStarts'] = mcustarts
        return result

    @cached_property
    def geotiff_tags(self) -> dict[str, Any] | None:
        """Consolidated metadata from GeoTIFF tags."""
        from .tifffile import TIFF

        if not self.is_geotiff:
            return None
        tags = self.tags

        gkd = tags.valueof(34735)  # GeoKeyDirectoryTag
        if gkd is None or len(gkd) < 2 or gkd[0] != 1:
            logger().warning(f'{self!r} invalid GeoKeyDirectoryTag')
            return {}

        result = {
            'KeyDirectoryVersion': gkd[0],
            'KeyRevision': gkd[1],
            'KeyRevisionMinor': gkd[2],
            # 'NumberOfKeys': gkd[3],
        }
        # deltags = ['GeoKeyDirectoryTag']
        geokeys = TIFF.GEO_KEYS
        geocodes = TIFF.GEO_CODES
        for index in range(gkd[3]):
            try:
                keyid, tagid, count, offset = gkd[
                    4 + index * 4 : index * 4 + 8
                ]
            except Exception as exc:
                logger().warning(
                    f'{self!r} corrupted GeoKeyDirectoryTag '
                    f'raised {exc!r:.128}'
                )
                continue
            if tagid == 0:
                value = offset
            else:
                try:
                    value = tags[tagid].value[offset : offset + count]
                except TiffFileError as exc:
                    logger().warning(
                        f'{self!r} corrupted GeoKeyDirectoryTag {tagid} '
                        f'raised {exc!r:.128}'
                    )
                    continue
                except KeyError as exc:
                    logger().warning(
                        f'{self!r} GeoKeyDirectoryTag {tagid} not found, '
                        f'raised {exc!r:.128}'
                    )
                    continue
                if tagid == 34737 and count > 1 and value[-1] == '|':
                    value = value[:-1]
                value = value if count > 1 else value[0]
            if keyid in geocodes:
                with contextlib.suppress(ValueError):
                    value = geocodes[keyid](value)
            try:
                key = geokeys(keyid).name
            except ValueError:
                key = keyid
            result[key] = value

        value = tags.valueof(33920)  # IntergraphMatrixTag
        if value is not None:
            value = numpy.array(value)
            if value.size == 16:
                value = value.reshape((4, 4)).tolist()
            result['IntergraphMatrix'] = value

        value = tags.valueof(33550)  # ModelPixelScaleTag
        if value is not None:
            result['ModelPixelScale'] = numpy.array(value).tolist()

        value = tags.valueof(33922)  # ModelTiepointTag
        if value is not None:
            value = numpy.array(value).reshape((-1, 6)).squeeze().tolist()
            result['ModelTiepoint'] = value

        value = tags.valueof(34264)  # ModelTransformationTag
        if value is not None:
            value = numpy.array(value).reshape((4, 4)).tolist()
            result['ModelTransformation'] = value

        # if 33550 in tags and 33922 in tags:
        #     sx, sy, sz = tags[33550].value  # ModelPixelScaleTag
        #     tiepoints = tags[33922].value  # ModelTiepointTag
        #     transforms = []
        #     for tp in range(0, len(tiepoints), 6):
        #         i, j, k, x, y, z = tiepoints[tp : tp + 6]
        #         transforms.append(
        #             [
        #                 [sx, 0.0, 0.0, x - i * sx],
        #                 [0.0, -sy, 0.0, y + j * sy],
        #                 [0.0, 0.0, sz, z - k * sz],
        #                 [0.0, 0.0, 0.0, 1.0],
        #             ]
        #         )
        #     if len(tiepoints) == 6:
        #         transforms = transforms[0]
        #     result['ModelTransformation'] = transforms

        rpcc = tags.valueof(50844)  # RPCCoefficientTag
        if rpcc is not None:
            result['RPCCoefficient'] = {
                'ERR_BIAS': rpcc[0],
                'ERR_RAND': rpcc[1],
                'LINE_OFF': rpcc[2],
                'SAMP_OFF': rpcc[3],
                'LAT_OFF': rpcc[4],
                'LONG_OFF': rpcc[5],
                'HEIGHT_OFF': rpcc[6],
                'LINE_SCALE': rpcc[7],
                'SAMP_SCALE': rpcc[8],
                'LAT_SCALE': rpcc[9],
                'LONG_SCALE': rpcc[10],
                'HEIGHT_SCALE': rpcc[11],
                'LINE_NUM_COEFF': rpcc[12:33],
                'LINE_DEN_COEFF ': rpcc[33:53],
                'SAMP_NUM_COEFF': rpcc[53:73],
                'SAMP_DEN_COEFF': rpcc[73:],
            }
        return result

    @cached_property
    def shaped_description(self) -> str | None:
        """Description containing array shape if exists, else None."""
        for description in (self.description, self.description1):
            if not description or '"mibi.' in description:
                return None
            if description[:1] == '{' and '"shape":' in description:
                return description
            if description[:6] == 'shape=':
                return description
        return None

    @cached_property
    def imagej_description(self) -> str | None:
        """ImageJ description if exists, else None."""
        for description in (self.description, self.description1):
            if not description:
                return None
            if description[:7] == 'ImageJ=' or description[:7] == 'SCIFIO=':
                return description
        return None

    @cached_property
    def is_jfif(self) -> bool:
        """JPEG compressed segments contain JFIF metadata."""
        if (
            self.compression not in {6, 7, 34892, 33007}
            or len(self.dataoffsets) < 1
            or self.dataoffsets[0] == 0
            or len(self.databytecounts) < 1
            or self.databytecounts[0] < 11
        ):
            return False
        fh = self.parent.filehandle
        fh.seek(self.dataoffsets[0] + 6)
        data = fh.read(4)
        return data == b'JFIF'  # or data == b'Exif'

    @property
    def is_frame(self) -> bool:
        """Object is :py:class:`TiffFrame` instance."""
        return False

    @property
    def is_virtual(self) -> bool:
        """Page does not have IFD structure in file."""
        return False

    @property
    def is_subifd(self) -> bool:
        """Page is SubIFD of another page."""
        return len(self._index) > 1

    @property
    def is_reduced(self) -> bool:
        """Page is reduced image of another image."""
        return bool(self.subfiletype & 0b1)

    @property
    def is_multipage(self) -> bool:
        """Page is part of multi-page image."""
        return bool(self.subfiletype & 0b10)

    @property
    def is_mask(self) -> bool:
        """Page is transparency mask for another image."""
        return bool(self.subfiletype & 0b100)

    @property
    def is_mrc(self) -> bool:
        """Page is part of Mixed Raster Content."""
        return bool(self.subfiletype & 0b1000)

    @property
    def is_tiled(self) -> bool:
        """Page contains tiled image."""
        return self.tilewidth > 0  # return 322 in self.tags  # TileWidth

    @property
    def is_subsampled(self) -> bool:
        """Page contains chroma subsampled image."""
        if self.subsampling is not None:
            return self.subsampling != (1, 1)
        return self.photometric == 6  # YCbCr
        # RGB JPEG usually stored as subsampled YCbCr
        # self.compression == 7
        # and self.photometric == 2
        # and self.planarconfig == 1

    @property
    def is_imagej(self) -> bool:
        """Page contains ImageJ description metadata."""
        return self.imagej_description is not None

    @property
    def is_shaped(self) -> bool:
        """Page contains Tifffile JSON metadata."""
        return self.shaped_description is not None

    @property
    def is_mdgel(self) -> bool:
        """Page contains MDFileTag tag."""
        return (
            37701 not in self.tags  # AgilentBinary
            and 33445 in self.tags  # MDFileTag
        )

    @property
    def is_agilent(self) -> bool:
        """Page contains Agilent Technologies tags."""
        # tag 270 and 285 contain color names
        return 285 in self.tags and 37701 in self.tags  # AgilentBinary

    @property
    def is_mediacy(self) -> bool:
        """Page contains Media Cybernetics Id tag."""
        tag = self.tags.get(50288)  # MC_Id
        try:
            return tag is not None and tag.value[:7] == b'MC TIFF'
        except Exception:
            return False

    @property
    def is_stk(self) -> bool:
        """Page contains UIC1Tag tag."""
        return 33628 in self.tags

    @property
    def is_lsm(self) -> bool:
        """Page contains CZ_LSMINFO tag."""
        return 34412 in self.tags

    @property
    def is_fluoview(self) -> bool:
        """Page contains FluoView MM_STAMP tag."""
        return 34362 in self.tags

    @property
    def is_nih(self) -> bool:
        """Page contains NIHImageHeader tag."""
        return 43314 in self.tags

    @property
    def is_volumetric(self) -> bool:
        """Page contains SGI ImageDepth tag with value > 1."""
        return self.imagedepth > 1

    @property
    def is_vista(self) -> bool:
        """Software tag is 'ISS Vista'."""
        return self.software == 'ISS Vista'

    @property
    def is_metaseries(self) -> bool:
        """Page contains MDS MetaSeries metadata in ImageDescription tag."""
        if self.index != 0 or self.software != 'MetaSeries':
            return False
        d = self.description
        return d.startswith('<MetaData>') and d.endswith('</MetaData>')

    @property
    def is_ome(self) -> bool:
        """Page contains OME-XML in ImageDescription tag."""
        if self.index != 0 or not self.description:
            return False
        return self.description[-10:].strip().endswith('OME>')

    @property
    def is_scn(self) -> bool:
        """Page contains Leica SCN XML in ImageDescription tag."""
        if self.index != 0 or not self.description:
            return False
        return self.description[-10:].strip().endswith('</scn>')

    @property
    def is_micromanager(self) -> bool:
        """Page contains MicroManagerMetadata tag."""
        return 51123 in self.tags

    @property
    def is_andor(self) -> bool:
        """Page contains Andor Technology tags 4864-5030."""
        return 4864 in self.tags

    @property
    def is_nuvu(self) -> bool:
        """Page contains Nuvu cameras tags >= 65000."""
        return (
            65000 in self.tags
            and 65001 in self.tags
            and self.tags[65000].dtype == 2
            and self.tags[65000].value.startswith("Field '65001' is ")
        )

    @property
    def is_pilatus(self) -> bool:
        """Page contains Pilatus tags."""
        return self.software[:8] == 'TVX TIFF' and self.description[:2] == '# '

    @property
    def is_epics(self) -> bool:
        """Page contains EPICS areaDetector tags."""
        return (
            self.description == 'EPICS areaDetector'
            or self.software == 'EPICS areaDetector'
        )

    @property
    def is_tvips(self) -> bool:
        """Page contains TVIPS metadata."""
        return 37706 in self.tags

    @property
    def is_fei(self) -> bool:
        """Page contains FEI_SFEG or FEI_HELIOS tags."""
        return 34680 in self.tags or 34682 in self.tags

    @property
    def is_sem(self) -> bool:
        """Page contains CZ_SEM tag."""
        return 34118 in self.tags

    @property
    def is_svs(self) -> bool:
        """Page contains Aperio metadata."""
        return self.description[:7] == 'Aperio '

    @property
    def is_bif(self) -> bool:
        """Page contains Ventana metadata."""
        try:
            return 700 in self.tags and (
                # avoid reading XMP tag from file at this point
                # b'<iScan' in self.tags[700].value[:4096]
                'Ventana' in self.software
                or self.software[:17] == 'ScanOutputManager'
                or self.description
                in {'Label Image', 'Label_Image', 'Probability_Image'}
            )
        except Exception:
            return False

    @property
    def is_scanimage(self) -> bool:
        """Page contains ScanImage metadata."""
        return (
            self.software[:3] == 'SI.'
            or self.description[:6] == 'state.'
            or 'scanimage.SI' in self.description[-256:]
        )

    @property
    def is_indica(self) -> bool:
        """Page contains IndicaLabs metadata."""
        return self.software[:21] == 'IndicaLabsImageWriter'

    @property
    def is_avs(self) -> bool:
        """Page contains Argos AVS XML metadata."""
        try:
            return (
                65000 in self.tags and self.tags.valueof(65000)[:6] == '<Argos'
            )
        except Exception:
            return False

    @property
    def is_qpi(self) -> bool:
        """Page contains PerkinElmer tissue images metadata."""
        # The ImageDescription tag contains XML with a top-level
        # <PerkinElmer-QPI-ImageDescription> element
        return self.software[:15] == 'PerkinElmer-QPI'

    @property
    def is_geotiff(self) -> bool:
        """Page contains GeoTIFF metadata."""
        return 34735 in self.tags  # GeoKeyDirectoryTag

    @property
    def is_gdal(self) -> bool:
        """Page contains GDAL metadata."""
        # startswith '<GDALMetadata>'
        return 42112 in self.tags  # GDAL_METADATA

    @property
    def is_astrotiff(self) -> bool:
        """Page contains AstroTIFF FITS metadata."""
        return (
            self.description[:7] == 'SIMPLE '
            and self.description[-3:] == 'END'
        )

    @property
    def is_streak(self) -> bool:
        """Page contains Hamamatsu streak metadata."""
        return (
            self.description[:1] == '['
            and '],' in self.description[1:32]
            # and self.tags.get(315, '').value[:19] == 'Copyright Hamamatsu'
        )

    @property
    def is_dng(self) -> bool:
        """Page contains DNG metadata."""
        return 50706 in self.tags  # DNGVersion

    @property
    def is_tiffep(self) -> bool:
        """Page contains TIFF/EP metadata."""
        return 37398 in self.tags  # TIFF/EPStandardID

    @property
    def is_sis(self) -> bool:
        """Page contains Olympus SIS metadata."""
        return 33560 in self.tags or 33471 in self.tags

    @property
    def is_ndpi(self) -> bool:
        """Page contains NDPI metadata."""
        return 65420 in self.tags and 271 in self.tags

    @property
    def is_philips(self) -> bool:
        """Page contains Philips DP metadata."""
        return self.software[:10] == 'Philips DP' and self.description[
            -16:
        ].strip().endswith('</DataObject>')

    @property
    def is_eer(self) -> bool:
        """Page contains EER acquisition metadata."""
        return (
            self.parent.is_bigtiff
            # and self.compression in {1, 65000, 65001, 65002}
            and 65001 in self.tags
            and self.tags[65001].dtype == 7
            and self.tags[65001].value[:10] == b'<metadata>'
        )


@final
class TiffFrame:
    """Lightweight TIFF image file directory (IFD).

    The purpose of TiffFrame is to reduce resource usage and speed up reading
    image data from file compared to TiffPage.
    Properties other than `offset`, `index`, `dataoffsets`, `databytecounts`,
    `subifds`, and `jpegtables` are assumed to be identical with a specified
    TiffPage instance, the keyframe.
    TiffFrame instances have no `tags` property.
    Virtual frames just reference the image data in the file. They may not
    have an IFD structure in the file.

    TiffFrame instances are not thread-safe. All attributes are read-only.

    Parameters:
        parent:
            TiffFile instance to read frame from.
            The file handle position must be at an offset to an IFD structure.
            Only a limited number of tag values are read from file.
        index:
            Index of frame in IFD tree.
        offset:
            Position of frame in file.
        keyframe:
            TiffPage instance with same hash as frame.
        dataoffsets:
            Data offsets of "virtual frame".
        databytecounts:
            Data bytecounts of "virtual frame".

    """

    __slots__ = (
        '_index',
        '_keyframe',
        'databytecounts',
        'dataoffsets',
        'jpegtables',
        'offset',
        'parent',
        'subifds',
    )

    is_mdgel: bool = False
    pages: TiffPages | None = None
    # tags = {}

    parent: TiffFile
    """TiffFile instance frame belongs to."""

    offset: int
    """Position of frame in file."""

    dataoffsets: tuple[int, ...]
    """Positions of strips or tiles in file."""

    databytecounts: tuple[int, ...]
    """Size of strips or tiles in file."""

    subifds: tuple[int, ...] | None
    """Positions of SubIFDs in file."""

    jpegtables: bytes | None
    """JPEG quantization and/or Huffman tables."""

    _keyframe: TiffPage | None
    _index: tuple[int, ...]  # index of frame in IFD tree.

    def __init__(
        self,
        parent: TiffFile,
        /,
        index: int | Sequence[int],
        *,
        offset: int | None = None,
        keyframe: TiffPage | None = None,
        dataoffsets: tuple[int, ...] | None = None,
        databytecounts: tuple[int, ...] | None = None,
    ):
        self._keyframe = None
        self.parent = parent

        self.offset = int(offset) if offset else 0
        self.subifds = None
        self.jpegtables = None
        self.dataoffsets = ()
        self.databytecounts = ()
        if isinstance(index, int):
            self._index = (index,)
        else:
            self._index = tuple(index)

        if dataoffsets is not None and databytecounts is not None:
            # initialize "virtual frame" from offsets and bytecounts
            self.offset = 0 if offset is None else offset
            self.dataoffsets = dataoffsets
            self.databytecounts = databytecounts
            self._keyframe = keyframe
            return

        if offset is None:
            self.offset = parent.filehandle.tell()
        else:
            parent.filehandle.seek(offset)

        if keyframe is None:
            tags = {273, 279, 324, 325, 330, 347}
        elif keyframe.is_contiguous:
            # use databytecounts from keyframe
            tags = {256, 273, 324, 330}
            self.databytecounts = keyframe.databytecounts
        else:
            tags = {256, 273, 279, 324, 325, 330, 347}

        for code, tag in self._gettags(tags):
            if code in {273, 324}:
                self.dataoffsets = tag.value
            elif code in {279, 325}:
                self.databytecounts = tag.value
            elif code == 330:
                self.subifds = tag.value
            elif code == 347:
                self.jpegtables = tag.value
            elif keyframe is None or (
                code == 256 and keyframe.tags[256].value != tag.value
            ):
                msg = 'incompatible keyframe'
                raise RuntimeError(msg)

        if not self.dataoffsets:
            logger().warning(f'{self!r} is missing required tags')
        elif keyframe is not None and len(self.dataoffsets) != len(
            keyframe.dataoffsets
        ):
            msg = 'incompatible keyframe'
            raise RuntimeError(msg)

        if keyframe is not None:
            self.keyframe = keyframe

    def _gettags(
        self,
        codes: Container[int] | None = None,
        /,
        lock: threading.RLock | None = None,
    ) -> list[tuple[int, TiffTag]]:
        """Return list of (code, TiffTag) from file."""
        fh = self.parent.filehandle
        tiff = self.parent.tiff
        unpack = struct.unpack
        rlock: Any = NullContext() if lock is None else lock
        tags = []

        with rlock:
            fh.seek(self.offset)
            try:
                tagno = unpack(tiff.tagnoformat, fh.read(tiff.tagnosize))[0]
                if tagno > 4096:
                    msg = f'suspicious number of tags {tagno}'
                    raise ValueError(msg)
            except Exception as exc:
                msg = f'corrupted tag list @{self.offset}'
                raise TiffFileError(msg) from exc

            tagoffset = self.offset + tiff.tagnosize  # fh.tell()
            tagsize = tiff.tagsize
            tagbytes = fh.read(tagsize * tagno)

            if _HAS_CPP and not tiff.is_ndpi:
                cpp_fmt = _CPP_FORMATS.get(
                    (tiff.version, tiff.byteorder)
                )
                if codes:
                    entries = _cpp_parse_ifd_filtered(
                        tagbytes, tagno, tagoffset, cpp_fmt,
                        set(codes),
                    )
                else:
                    entries = _cpp_parse_ifd(
                        tagbytes, tagno, tagoffset, cpp_fmt
                    )
                for entry in entries:
                    try:
                        tag = TiffTag.from_ifd_entry(
                            self.parent, entry
                        )
                    except TiffFileError as exc:
                        logger().error(
                            f'{self!r} <TiffTag.from_ifd_entry>'
                            f' raised {exc!r:.128}'
                        )
                        continue
                    tags.append((entry.code, tag))
            else:
                tagindex = -tagsize
                codeformat = tiff.tagformat1[:2]
                for _ in range(tagno):
                    tagindex += tagsize
                    code = unpack(
                        codeformat,
                        tagbytes[tagindex : tagindex + 2],
                    )[0]
                    if codes and code not in codes:
                        continue
                    try:
                        tag = TiffTag.fromfile(
                            self.parent,
                            offset=tagoffset + tagindex,
                            header=tagbytes[
                                tagindex : tagindex + tagsize
                            ],
                        )
                    except TiffFileError as exc:
                        logger().error(
                            f'{self!r} <TiffTag.fromfile>'
                            f' raised {exc!r:.128}'
                        )
                        continue
                    tags.append((code, tag))

        return tags

    def _nextifd(self) -> int:
        """Return offset to next IFD from file."""
        return TiffPage._nextifd(self)  # type: ignore[arg-type]

    def aspage(self) -> TiffPage:
        """Return TiffPage from file.

        Raise ValueError if frame is virtual.

        """
        if self.is_virtual:
            msg = 'cannot return virtual frame as page'
            raise ValueError(msg)
        fh = self.parent.filehandle
        closed = fh.closed
        if closed:
            # this is an inefficient resort in case a user calls aspage
            # of a TiffFrame with a closed FileHandle.
            warnings.warn(
                f'{self!r} reading TiffPage from closed file',
                UserWarning,
                stacklevel=2,
            )
            fh.open()
        try:
            fh.seek(self.offset)
            page = TiffPage(self.parent, index=self.index)
        finally:
            if closed:
                fh.close()
        return page

    def asarray(self, **kwargs: Any) -> NDArray[Any]:
        """Return image from frame as NumPy array.

        Parameters:
            **kwargs: Arguments passed to :py:meth:`TiffPage.asarray`.

        """
        return TiffPage.asarray(self, **kwargs)  # type: ignore[arg-type]

    def aszarr(self, **kwargs: Any) -> ZarrTiffStore:
        """Return image from frame as Zarr store.

        Parameters:
            **kwargs: Arguments passed to :py:class:`ZarrTiffStore`.

        """
        from .zarr import ZarrTiffStore

        return ZarrTiffStore(self, **kwargs)

    def asrgb(self, *args: Any, **kwargs: Any) -> NDArray[Any]:
        """Return image from frame as RGB(A). Work in progress. Do not use.

        :meta private:

        """
        return TiffPage.asrgb(self, *args, **kwargs)  # type: ignore[arg-type]

    def segments(self, *args: Any, **kwargs: Any) -> Iterator[
        tuple[
            NDArray[Any] | None,
            tuple[int, int, int, int, int],
            tuple[int, int, int, int],
        ]
    ]:
        """Return iterator over decoded tiles or strips.

        Parameters:
            **kwargs: Arguments passed to :py:meth:`TiffPage.segments`.

        :meta private:

        """
        return TiffPage.segments(
            self, *args, **kwargs  # type: ignore[arg-type]
        )

    @property
    def index(self) -> int:
        """Index of frame in IFD chain."""
        return self._index[-1]

    @property
    def treeindex(self) -> tuple[int, ...]:
        """Index of frame in IFD tree."""
        return self._index

    @property
    def keyframe(self) -> TiffPage | None:
        """TiffPage with same properties as this frame."""
        return self._keyframe

    @keyframe.setter
    def keyframe(self, keyframe: TiffPage, /) -> None:
        if self._keyframe == keyframe:
            return
        if self._keyframe is not None:
            msg = 'cannot reset keyframe'
            raise RuntimeError(msg)
        if len(self.dataoffsets) != len(keyframe.dataoffsets):
            msg = 'incompatible keyframe'
            raise RuntimeError(msg)
        if keyframe.is_contiguous:
            self.databytecounts = keyframe.databytecounts
        self._keyframe = keyframe

    @property
    def is_frame(self) -> bool:
        """Object is :py:class:`TiffFrame` instance."""
        return True

    @property
    def is_virtual(self) -> bool:
        """Frame does not have IFD structure in file."""
        return self.offset <= 0

    @property
    def is_subifd(self) -> bool:
        """Frame is SubIFD of another page."""
        return len(self._index) > 1

    @property
    def is_final(self) -> bool:
        assert self._keyframe is not None
        return self._keyframe.is_final

    @property
    def is_contiguous(self) -> bool:
        assert self._keyframe is not None
        return self._keyframe.is_contiguous

    @property
    def is_memmappable(self) -> bool:
        assert self._keyframe is not None
        return self._keyframe.is_memmappable

    @property
    def hash(self) -> int:
        assert self._keyframe is not None
        return self._keyframe.hash

    @property
    def shape(self) -> tuple[int, ...]:
        assert self._keyframe is not None
        return self._keyframe.shape

    @property
    def shaped(self) -> tuple[int, int, int, int, int]:
        assert self._keyframe is not None
        return self._keyframe.shaped

    @property
    def chunks(self) -> tuple[int, ...]:
        assert self._keyframe is not None
        return self._keyframe.chunks

    @property
    def chunked(self) -> tuple[int, ...]:
        assert self._keyframe is not None
        return self._keyframe.chunked

    @property
    def tile(self) -> tuple[int, ...] | None:
        assert self._keyframe is not None
        return self._keyframe.tile

    @property
    def name(self) -> str:
        index = self._index if len(self._index) > 1 else self._index[0]
        return f'TiffFrame {index}'

    @property
    def ndim(self) -> int:
        assert self._keyframe is not None
        return self._keyframe.ndim

    @property
    def dims(self) -> tuple[str, ...]:
        assert self._keyframe is not None
        return self._keyframe.dims

    @property
    def sizes(self) -> dict[str, int]:
        assert self._keyframe is not None
        return self._keyframe.sizes

    @property
    def coords(self) -> dict[str, NDArray[Any]]:
        assert self._keyframe is not None
        return self._keyframe.coords

    @property
    def size(self) -> int:
        assert self._keyframe is not None
        return self._keyframe.size

    @property
    def nbytes(self) -> int:
        assert self._keyframe is not None
        return self._keyframe.nbytes

    @property
    def dtype(self) -> numpy.dtype[Any] | None:
        assert self._keyframe is not None
        return self._keyframe.dtype

    @property
    def axes(self) -> str:
        assert self._keyframe is not None
        return self._keyframe.axes

    def get_resolution(
        self,
        unit: RESUNIT | int | None = None,
        scale: float | None = None,
    ) -> tuple[float, float]:
        assert self._keyframe is not None
        return self._keyframe.get_resolution(unit, scale)

    @property
    def resolution(self) -> tuple[float, float]:
        assert self._keyframe is not None
        return self._keyframe.resolution

    @property
    def resolutionunit(self) -> int:
        assert self._keyframe is not None
        return self._keyframe.resolutionunit

    @property
    def datetime(self) -> DateTime | None:
        # TODO: TiffFrame.datetime can differ from TiffPage.datetime?
        assert self._keyframe is not None
        return self._keyframe.datetime

    @property
    def compression(self) -> int:
        assert self._keyframe is not None
        return self._keyframe.compression

    @property
    def decode(
        self,
    ) -> Callable[
        ...,
        tuple[
            NDArray[Any] | None,
            tuple[int, int, int, int, int],
            tuple[int, int, int, int],
        ],
    ]:
        assert self._keyframe is not None
        return self._keyframe.decode

    def __repr__(self) -> str:
        index = self._index if len(self._index) > 1 else self._index[0]
        return f'<tifffile.TiffFrame {index} @{self.offset}>'

    def __str__(self) -> str:
        return self._str()

    def _str(self, detail: int = 0, width: int = 79) -> str:
        """Return string containing information about TiffFrame."""
        if self._keyframe is None:
            info = ''
            kf = None
        else:
            info = '  '.join(
                s
                for s in (
                    'x'.join(str(i) for i in self.shape),
                    str(self.dtype),
                )
            )
            kf = self._keyframe._str(width=width - 11)
        if detail > 3:
            of = pformat(self.dataoffsets, width=width - 9, height=detail - 3)
            bc = pformat(
                self.databytecounts, width=width - 13, height=detail - 3
            )
            info = f'\n Keyframe {kf}\n Offsets {of}\n Bytecounts {bc}'
        index = self._index if len(self._index) > 1 else self._index[0]
        return f'TiffFrame {index} @{self.offset}  {info}'


@final
class TiffPages(Sequence[TiffPage | TiffFrame]):
    """Sequence of TIFF image file directories (IFD chain).

    TiffPages instances have a state, such as a cache and keyframe, and are not
    thread-safe. All attributes are read-only.

    Parameters:
        arg:
            If a *TiffFile*, the file position must be at offset to offset to
            TiffPage.
            If a *TiffPage* or *TiffFrame*, page offsets are read from the
            SubIFDs tag.
            Only the first page is initially read from the file.
        index:
            Position of IFD chain in IFD tree.

    """

    parent: TiffFile | None = None
    """TiffFile instance pages belongs to."""

    _pages: list[TiffPage | TiffFrame | int]  # list of pages
    _keyframe: TiffPage | None
    _tiffpage: type[TiffPage | TiffFrame]  # class used for reading pages
    _indexed: bool
    _cached: bool
    _cache: bool
    _offset: int
    _nextpageoffset: int | None
    _index: tuple[int, ...] | None

    def __init__(
        self,
        arg: TiffFile | TiffPage | TiffFrame,
        /,
        *,
        index: Sequence[int] | int | None = None,
    ) -> None:
        from .tifffile import TiffFile

        offset: int
        self.parent = None
        self._pages = []  # cache of TiffPages, TiffFrames, or their offsets
        self._indexed = False  # True if offsets to all pages were read
        self._cached = False  # True if all pages were read into cache
        self._tiffpage = TiffPage  # class used for reading pages
        self._keyframe = None  # page that is currently used as keyframe
        self._cache = False  # do not cache frames or pages (if not keyframe)
        self._offset = 0
        self._nextpageoffset = None
        self._ifd_offsets_packed: bytes | None = None  # packed uint64 offsets

        if index is None:
            self._index = None
        elif isinstance(index, (int, numpy.integer)):
            self._index = (int(index),)
        else:
            self._index = tuple(index)

        if isinstance(arg, TiffFile):
            # read offset to first page from current file position
            self.parent = arg
            fh = self.parent.filehandle
            self._nextpageoffset = fh.tell()
            offset = struct.unpack(
                self.parent.tiff.offsetformat,
                fh.read(self.parent.tiff.offsetsize),
            )[0]
            if offset == 0:
                logger().warning(f'{arg!r} contains no pages')
                self._indexed = True
                return
        elif arg.subifds is not None:
            # use offsets from SubIFDs tag
            offsets = arg.subifds
            self.parent = arg.parent
            fh = self.parent.filehandle
            if len(offsets) == 0 or offsets[0] == 0:
                logger().warning(f'{arg!r} contains invalid SubIFDs')
                self._indexed = True
                return
            offset = offsets[0]
        else:
            self._indexed = True
            return

        self._offset = offset
        if offset >= fh.size:
            logger().warning(
                f'{self!r} invalid offset to first page {offset!r}'
            )
            self._indexed = True
            return

        pageindex: int | tuple[int, ...] = (
            0 if self._index is None else (*self._index, 0)
        )

        # read and cache first page
        fh.seek(offset)
        page = TiffPage(self.parent, index=pageindex)
        self._pages.append(page)
        self._keyframe = page
        if self._nextpageoffset is None:
            # offsets from SubIFDs tag
            self._pages.extend(offsets[1:])
            self._indexed = True
            self._cached = True

    @property
    def pages(self) -> list[TiffPage | TiffFrame | int]:
        """Deprecated. Use the TiffPages sequence interface.

        :meta private:

        """
        warnings.warn(
            '<tifffile.TiffPages.pages> is deprecated since 2024.5.22. '
            'Use the TiffPages sequence interface.',
            DeprecationWarning,
            stacklevel=2,
        )
        return self._pages

    @property
    def first(self) -> TiffPage:
        """First page as TiffPage if exists, else raise IndexError."""
        return cast(TiffPage, self._pages[0])

    @property
    def is_multipage(self) -> bool:
        """IFD chain contains more than one page."""
        try:
            self._seek(1)
        except IndexError:
            return False
        return True

    @property
    def cache(self) -> bool:
        """Pages and frames are being cached.

        When set to *False*, the cache is cleared.

        """
        return self._cache

    @cache.setter
    def cache(self, value: bool, /) -> None:
        value = bool(value)
        if self._cache and not value:
            self._clear()
        self._cache = value

    @property
    def useframes(self) -> bool:
        """Use TiffFrame (True) or TiffPage (False)."""
        return self._tiffpage == TiffFrame

    @useframes.setter
    def useframes(self, value: bool, /) -> None:
        self._tiffpage = TiffFrame if value else TiffPage

    @property
    def keyframe(self) -> TiffPage | None:
        """TiffPage used as keyframe for new TiffFrames."""
        return self._keyframe

    def set_keyframe(self, index: int, /) -> None:
        """Set keyframe to TiffPage specified by `index`.

        If not found in the cache, the TiffPage at `index` is loaded from file
        and added to the cache.

        """
        if not isinstance(index, (int, numpy.integer)):
            msg = f'indices must be integers, not {type(index)}'
            raise TypeError(msg)
        index = int(index)
        if index < 0:
            index %= len(self)
        if self._keyframe is not None and self._keyframe.index == index:
            return
        if index == 0:
            self._keyframe = cast(TiffPage, self._pages[0])
            return
        if self._indexed or index < len(self._pages):
            page = self._pages[index]
            if isinstance(page, TiffPage):
                self._keyframe = page
                return
            if isinstance(page, TiffFrame):
                # remove existing TiffFrame
                self._pages[index] = page.offset
        # load TiffPage from file
        tiffpage = self._tiffpage
        self._tiffpage = TiffPage
        try:
            self._keyframe = cast(TiffPage, self._getitem(index))
        finally:
            self._tiffpage = tiffpage
        # always cache keyframes
        self._pages[index] = self._keyframe

    @property
    def next_page_offset(self) -> int | None:
        """Offset where offset to new page can be stored."""
        if not self._indexed:
            self._seek(-1)
        return self._nextpageoffset

    def get(
        self,
        key: int,
        /,
        default: TiffPage | TiffFrame | None = None,
        *,
        validate: int = 0,
        cache: bool = False,
        aspage: bool = True,
    ) -> TiffPage | TiffFrame:
        """Return specified page from cache or file.

        The specified TiffPage or TiffFrame is read from file if it is not
        found in the cache.

        Parameters:
            key:
                Index of requested page in IFD chain.
            default:
                Page or frame to return if key is out of bounds.
                By default, an IndexError is raised if key is out of bounds.
            validate:
                If non-zero, raise RuntimeError if value does not match hash
                of TiffPage or TiffFrame.
            cache:
                Store returned page in cache for future use.
            aspage:
                Return TiffPage instance.

        """
        try:
            return self._getitem(
                key, validate=validate, cache=cache, aspage=aspage
            )
        except IndexError:
            if default is None:
                raise
        return default

    def _load(
        self,
        keyframe: TiffPage | bool | None = True,  # noqa: FBT001, FBT002
        /,
    ) -> None:
        """Read all remaining pages from file."""
        assert self.parent is not None
        if self._cached:
            return
        pages = self._pages
        if not pages:
            return
        if not self._indexed:
            self._seek(-1)
        if not self._cache:
            return
        # Try C++ bulk loading for uniform files
        self._try_bulk_load()
        if self._cached:
            return
        fh = self.parent.filehandle
        if keyframe is not None:
            keyframe = self._keyframe
        for i, page in enumerate(pages):
            if isinstance(page, (int, numpy.integer)):
                pageindex: int | tuple[int, ...] = (
                    i if self._index is None else (*self._index, i)
                )
                fh.seek(page)
                pages[i] = self._tiffpage(
                    self.parent, index=pageindex, keyframe=keyframe
                )
        self._cached = True

    def _load_frames_cpp(self) -> None:
        """Bulk load all TiffFrames using C++ mmap-based extraction."""
        assert self.parent is not None
        pages = self._pages
        keyframe = self._keyframe
        fh = self.parent.filehandle
        tiff = self.parent.tiff

        cpp_fmt = _CPP_FORMATS.get((tiff.version, tiff.byteorder))
        if cpp_fmt is None:
            msg = 'unsupported TIFF format for C++ bulk load'
            raise TiffFileError(msg)

        # Use stored packed offsets from _seek_cpp if available
        n_pages = len(pages)
        if self._ifd_offsets_packed is not None:
            all_packed = self._ifd_offsets_packed
            n_total = len(all_packed) // 8
            # Fast check: common case is page[0] loaded, rest are ints
            if (
                n_pages == n_total
                and n_pages > 1
                and not isinstance(pages[0], (int, numpy.integer))
                and isinstance(pages[1], (int, numpy.integer))
            ):
                # Common case: skip page[0], use rest directly
                int_indices = range(1, n_pages)
                offsets_packed = all_packed[8:]
                int_offsets = struct.unpack_from(
                    f'={n_pages - 1}Q', offsets_packed
                )
            else:
                # Mixed: collect which entries are ints
                int_indices = [
                    i for i in range(n_pages)
                    if isinstance(pages[i], (int, numpy.integer))
                ]
                if not int_indices:
                    self._cached = True
                    return
                all_offsets = struct.unpack_from(
                    f'={n_total}Q', all_packed
                )
                int_offsets = [all_offsets[i] for i in int_indices]
                offsets_packed = struct.pack(
                    f'={len(int_offsets)}Q', *int_offsets
                )
        else:
            # Fallback: collect int offsets from _pages
            int_indices = []
            int_offsets_list: list[int] = []
            for i, page in enumerate(pages):
                if isinstance(page, (int, numpy.integer)):
                    int_indices.append(i)
                    int_offsets_list.append(int(page))
            if not int_offsets_list:
                self._cached = True
                return
            int_offsets = int_offsets_list
            offsets_packed = struct.pack(
                f'={len(int_offsets)}Q', *int_offsets
            )

        reader = _CppFileReader(fh.path)

        # Determine which tags to extract based on keyframe
        is_contiguous = keyframe is not None and keyframe.is_contiguous
        if is_contiguous:
            codes = [273, 324, 330]
        else:
            codes = [273, 279, 324, 325, 330]

        tag_data = _cpp_bulk_extract(
            reader, offsets_packed, cpp_fmt, codes
        )

        # Build lookup
        strip_offsets = tag_data.get(273, [])
        strip_counts = tag_data.get(279, [])
        tile_offsets = tag_data.get(324, [])
        tile_counts = tag_data.get(325, [])
        subifds_vals = tag_data.get(330, [])

        parent = self.parent
        kf_bytecounts = keyframe.databytecounts if is_contiguous else None
        has_index = self._index is not None
        index_prefix = self._index

        for idx_pos, page_idx in enumerate(int_indices):
            offset = int_offsets[idx_pos]

            # Determine dataoffsets
            if idx_pos < len(tile_offsets) and tile_offsets[idx_pos]:
                dataoffsets = tile_offsets[idx_pos]
            elif idx_pos < len(strip_offsets) and strip_offsets[idx_pos]:
                dataoffsets = strip_offsets[idx_pos]
            else:
                dataoffsets = ()

            # Determine databytecounts
            if kf_bytecounts is not None:
                databytecounts = kf_bytecounts
            elif idx_pos < len(tile_counts) and tile_counts[idx_pos]:
                databytecounts = tile_counts[idx_pos]
            elif idx_pos < len(strip_counts) and strip_counts[idx_pos]:
                databytecounts = strip_counts[idx_pos]
            else:
                databytecounts = ()

            pageindex = (
                (*index_prefix, page_idx) if has_index else page_idx
            )

            # Inline TiffFrame construction (avoid function call overhead).
            # CAUTION: Must mirror TiffFrame.__init__ attribute initialization.
            # Update this block if TiffFrame.__init__ changes.
            frame = TiffFrame.__new__(TiffFrame)
            frame._keyframe = keyframe
            frame.parent = parent
            frame.offset = offset
            frame.subifds = None
            frame.jpegtables = None
            frame.dataoffsets = dataoffsets
            frame.databytecounts = databytecounts
            frame._index = (
                (pageindex,) if isinstance(pageindex, int) else
                tuple(pageindex)
            )

            # Handle SubIFDs
            if idx_pos < len(subifds_vals) and subifds_vals[idx_pos]:
                frame.subifds = subifds_vals[idx_pos]

            pages[page_idx] = frame

        self._cached = True

    def _load_virtual_frames(self) -> None:
        """Calculate virtual TiffFrames."""
        assert self.parent is not None
        pages = self._pages
        try:
            if len(pages) > 1:
                msg = 'pages already loaded'
                raise ValueError(msg)
            page = cast(TiffPage, pages[0])
            if not page.is_contiguous:
                msg = 'data not contiguous'
                raise ValueError(msg)
            self._seek(4)
            # following pages are int
            delta = cast(int, pages[2]) - cast(int, pages[1])
            if (
                cast(int, pages[3]) - cast(int, pages[2]) != delta
                or cast(int, pages[4]) - cast(int, pages[3]) != delta
            ):
                msg = 'page offsets not equidistant'
                raise ValueError(msg)
            page1 = self._getitem(1, validate=page.hash)
            offsetoffset = page1.dataoffsets[0] - page1.offset
            if offsetoffset < 0 or offsetoffset > delta:
                msg = 'page offsets not equidistant'
                raise ValueError(msg)
            pages = [page, page1]
            filesize = self.parent.filehandle.size - delta

            for index, offset in enumerate(
                range(page1.offset + delta, filesize, delta)
            ):
                index += 2  # noqa: PLW2901
                d = index * delta
                dataoffsets = tuple(i + d for i in page.dataoffsets)
                offset_or_none = offset if offset < 2**31 - 1 else None
                pages.append(
                    TiffFrame(
                        page.parent,
                        index=(
                            index
                            if self._index is None
                            else (*self._index, index)
                        ),
                        offset=offset_or_none,
                        dataoffsets=dataoffsets,
                        databytecounts=page.databytecounts,
                        keyframe=page,
                    )
                )
            self._pages = pages
            self._cache = True
            self._cached = True
            self._indexed = True
        except Exception as exc:
            if self.parent.filehandle.size >= 2147483648:
                logger().warning(
                    f'{self!r} <_load_virtual_frames> raised {exc!r:.128}'
                )

    def _clear(self, /, *, fully: bool = True) -> None:
        """Delete all but first page from cache. Set keyframe to first page."""
        pages = self._pages
        if not pages:
            return
        self._keyframe = cast(TiffPage, pages[0])
        if fully:
            # delete all but first TiffPage/TiffFrame
            for i, page in enumerate(pages[1:]):
                if not isinstance(page, int) and page.offset is not None:
                    pages[i + 1] = page.offset
        else:
            # delete only TiffFrames
            for i, page in enumerate(pages):
                if isinstance(page, TiffFrame) and page.offset is not None:
                    pages[i] = page.offset
        self._cached = False
        self._ifd_offsets_packed = None

    def _seek(self, index: int, /) -> int:
        """Seek file to offset of page specified by index and return offset."""
        assert self.parent is not None

        pages = self._pages
        lenpages = len(pages)
        if lenpages == 0:
            msg = 'index out of range'
            raise IndexError(msg)

        fh = self.parent.filehandle
        if fh.closed:
            msg = 'seek of closed file'
            raise ValueError(msg)

        if self._indexed or 0 <= index < lenpages:
            page = pages[index]
            offset = page if isinstance(page, int) else page.offset
            return fh.seek(offset)

        # C++ fast path: scan entire IFD chain at once using mmap
        if (
            _HAS_CPP
            and index < 0
            and not self.parent.tiff.is_ndpi
            and fh.is_file
        ):
            try:
                return self._seek_cpp(index)
            except Exception as exc:
                logger().debug(f'C++ scan failed: {exc!r:.128}')
                # fall through to Python path

        tiff = self.parent.tiff
        offsetformat = tiff.offsetformat
        offsetsize = tiff.offsetsize
        tagnoformat = tiff.tagnoformat
        tagnosize = tiff.tagnosize
        tagsize = tiff.tagsize
        unpack = struct.unpack

        page = pages[-1]
        offset = page if isinstance(page, int) else page.offset

        while lenpages < 2**32:
            # read offsets to pages from file until index is reached
            fh.seek(offset)
            # skip tags
            try:
                tagno = int(unpack(tagnoformat, fh.read(tagnosize))[0])
                if tagno > 4096:
                    msg = f'suspicious number of tags {tagno}'
                    raise TiffFileError(msg)
            except Exception as exc:
                logger().error(
                    f'{self!r} corrupted tag list of page '
                    f'{lenpages} @{offset} raised {exc!r:.128}',
                )
                del pages[-1]
                lenpages -= 1
                self._indexed = True
                break
            self._nextpageoffset = offset + tagnosize + tagno * tagsize
            fh.seek(self._nextpageoffset)

            # read offset to next page
            try:
                offset = int(unpack(offsetformat, fh.read(offsetsize))[0])
            except Exception as exc:
                logger().error(
                    f'{self!r} invalid offset to page '
                    f'{lenpages + 1} @{self._nextpageoffset} '
                    f'raised {exc!r:.128}'
                )
                self._indexed = True
                break
            if offset == 0:
                self._indexed = True
                break
            if offset >= fh.size:
                logger().error(f'{self!r} invalid page offset {offset!r}')
                self._indexed = True
                break

            pages.append(offset)
            lenpages += 1
            if 0 <= index < lenpages:
                break

            # detect some circular references
            if lenpages == 100:
                for i, p in enumerate(pages[:-1]):
                    if offset == (p if isinstance(p, int) else p.offset):
                        index = i
                        self._pages = pages[: i + 1]
                        self._indexed = True
                        logger().error(
                            f'{self!r} invalid circular reference to IFD '
                            f'{i} at {offset=}'
                        )
                        break

        if index >= lenpages:
            msg = 'index out of range'
            raise IndexError(msg)

        page = pages[index]
        return fh.seek(page if isinstance(page, int) else page.offset)

    def _seek_cpp(self, index: int, /) -> int:
        """Use C++ mmap-based IFD chain scanner for fast full scan."""
        assert self.parent is not None
        fh = self.parent.filehandle
        tiff = self.parent.tiff

        cpp_fmt = _CPP_FORMATS.get((tiff.version, tiff.byteorder))
        if cpp_fmt is None:
            msg = 'unsupported TIFF format for C++ scanner'
            raise TiffFileError(msg)

        # Memory-map the file and scan all IFD offsets
        reader = _CppFileReader(fh.path)
        offsets_bytes, next_page_offset, circ_idx, circ_off = (
            _cpp_scan_ifd_chain_file(reader, self._offset, cpp_fmt)
        )

        n = len(offsets_bytes) // 8
        if n == 0:
            self._indexed = True
            msg = 'index out of range'
            raise IndexError(msg)

        # Unpack packed uint64_t offsets to list of ints
        all_offsets = list(struct.unpack_from(f'={n}Q', offsets_bytes))

        # Keep pages[0] (the already-loaded TiffPage), replace the rest
        pages = self._pages
        if n > len(pages):
            pages.extend(all_offsets[len(pages):])
        self._indexed = True
        # Store packed offsets for reuse in _load_frames_cpp
        self._ifd_offsets_packed = bytes(offsets_bytes)

        # Update _nextpageoffset from C++ result
        if next_page_offset > 0:
            self._nextpageoffset = next_page_offset

        # Log circular IFD if detected by C++
        if circ_idx >= 0:
            logger().error(
                f'{self!r} invalid circular reference to IFD '
                f'{circ_idx} at offset={circ_off}'
            )

        lenpages = len(pages)
        if index < 0:
            index = lenpages + index
        if index < 0 or index >= lenpages:
            msg = 'index out of range'
            raise IndexError(msg)

        page = pages[index]
        return fh.seek(page if isinstance(page, int) else page.offset)

    def _getlist(
        self,
        key: int | slice | Iterable[int] | None = None,
        /,
        *,
        useframes: bool = True,
        validate: bool = True,
    ) -> list[TiffPage | TiffFrame]:
        """Return specified pages as list of TiffPages or TiffFrames.

        The first item is a TiffPage, and is used as a keyframe for
        following TiffFrames.

        """
        getitem = self._getitem
        _useframes = self.useframes

        match key:
            case None:
                key = iter(range(len(self)))
            case int() | numpy.integer():
                # return single TiffPage
                key = int(key)
                self.useframes = False
                if key == 0:
                    return [self.first]
                try:
                    return [getitem(key)]
                finally:
                    self.useframes = _useframes
            case slice():
                start, stop, _ = key.indices(2**31 - 1)
                if not self._indexed and max(stop, start) > len(self._pages):
                    self._seek(-1)
                key = iter(range(*key.indices(len(self._pages))))
            case Iterable():
                key = iter(key)
            case _:  # type: ignore[unreachable]
                msg = (
                    f'key must be integer, slice, or iterable, not {type(key)}'
                )
                raise TypeError(msg)

        # use first page as keyframe
        assert self._keyframe is not None
        keyframe = self._keyframe
        self.set_keyframe(next(key))
        validhash = self._keyframe.hash if validate else 0
        if useframes:
            self.useframes = True

        # Bulk-load TiffFrames from C++ if possible (avoids per-page I/O).
        if useframes:
            self._try_bulk_load()

        try:
            pages = [getitem(i, validate=validhash) for i in key]
            pages.insert(0, self._keyframe)
        finally:
            # restore state
            self._keyframe = keyframe
            if useframes:
                self.useframes = _useframes
        return pages

    def _getitem(
        self,
        key: int,
        /,
        *,
        validate: int = 0,  # hash
        cache: bool = False,
        aspage: bool = False,
    ) -> TiffPage | TiffFrame:
        """Return specified page from cache or file."""
        assert self.parent is not None
        key = int(key)
        pages = self._pages

        if key < 0:
            key %= len(self)
        elif self._indexed and key >= len(pages):
            msg = f'index {key} out of range({len(pages)})'
            raise IndexError(msg)

        tiffpage = TiffPage if aspage else self._tiffpage

        if key < len(pages):
            page = pages[key]
            if self._cache and not aspage:
                if not isinstance(page, (int, numpy.integer)):
                    if validate and validate != page.hash:
                        msg = 'page hash mismatch'
                        raise RuntimeError(msg)
                    return page
            elif isinstance(page, (TiffPage, tiffpage)):
                # page is not an int
                if (
                    validate
                    and validate != page.hash  # type: ignore[union-attr]
                ):
                    msg = 'page hash mismatch'
                    raise RuntimeError(msg)
                return page  # type: ignore[return-value]

        pageindex: int | tuple[int, ...] = (
            key if self._index is None else (*self._index, key)
        )
        self._seek(key)
        page = tiffpage(self.parent, index=pageindex, keyframe=self._keyframe)
        assert isinstance(page, (TiffPage, TiffFrame))
        if validate and validate != page.hash:
            msg = 'page hash mismatch'
            raise RuntimeError(msg)
        if self._cache or cache:
            pages[key] = page
        return page

    @overload
    def __getitem__(self, key: int, /) -> TiffPage | TiffFrame: ...

    @overload
    def __getitem__(
        self, key: slice | Iterable[int], /
    ) -> list[TiffPage | TiffFrame]: ...

    def __getitem__(
        self, key: int | slice | Iterable[int], /
    ) -> TiffPage | TiffFrame | list[TiffPage | TiffFrame]:
        pages = self._pages
        getitem = self._getitem

        if isinstance(key, (int, numpy.integer)):
            key = int(key)
            if key == 0:
                return cast(TiffPage, pages[key])
            return getitem(key)

        if isinstance(key, slice):
            start, stop, _ = key.indices(2**31 - 1)
            if not self._indexed and max(stop, start) > len(pages):
                self._seek(-1)
            if not self._cached:
                self._try_bulk_load()
            if self._cached and pages and not isinstance(
                pages[-1], (int, numpy.integer)
            ):
                # See __iter__ for why last-element check is sufficient
                return list(pages[key])
            return [getitem(i) for i in range(*key.indices(len(pages)))]

        if isinstance(key, Iterable):
            if not self._cached:
                self._try_bulk_load()
            if self._cached and pages and not isinstance(
                pages[-1], (int, numpy.integer)
            ):
                # See __iter__ for why last-element check is sufficient
                return [pages[k] for k in key]
            return [getitem(k) for k in key]

        msg = 'key must be an integer, slice, or iterable'
        raise TypeError(msg)

    def _try_bulk_load(self) -> None:
        """Try to bulk load all pages as TiffFrames via C++."""
        if (
            _HAS_CPP
            and not self._cached
            and self._keyframe is not None
            and self.parent is not None
            and not self.parent.tiff.is_ndpi
            and self.parent.filehandle.is_file
            and getattr(self.parent, 'is_uniform', False)
        ):
            try:
                if not self._indexed:
                    self._seek(-1)
                if len(self._pages) > 100:
                    self._load_frames_cpp()
            except Exception as exc:
                logger().debug(f'C++ bulk load failed: {exc!r:.128}')

    def __iter__(self) -> Iterator[TiffPage | TiffFrame]:
        # Bulk load frames via C++ for uniform files with many unloaded pages.
        self._try_bulk_load()
        if self._cached and self._pages and not isinstance(
            self._pages[-1], (int, numpy.integer)
        ):
            # All pages are loaded as objects; yield without _getitem overhead.
            # Last-element check is sufficient: _try_bulk_load only sets
            # _cached=True when ALL entries are TiffPage/TiffFrame objects.
            # SubIFD pages (which store trailing ints) never reach this path
            # because is_uniform is False for SubIFD files.
            yield from self._pages  # type: ignore[misc]
            return
        i = 0
        while True:
            try:
                yield self._getitem(i)
                i += 1
            except IndexError:
                break
        if self._cache:
            self._cached = True

    def __bool__(self) -> bool:
        """Return True if file contains any pages."""
        return len(self._pages) > 0

    def __len__(self) -> int:
        """Return number of pages in file."""
        if not self._indexed:
            self._seek(-1)
        return len(self._pages)

    def __repr__(self) -> str:
        return f'<tifffile.TiffPages @{self._offset}>'
