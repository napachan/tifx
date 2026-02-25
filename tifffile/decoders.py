# decoders.py

"""TIFF segment decode closure factory."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import numpy

from .utils import logger, unpack_rgb

if TYPE_CHECKING:
    from collections.abc import Callable
    from typing import Any

    from .tifffile import TiffPage

    NDArray = numpy.ndarray[Any, Any]

try:
    import imagecodecs
except ImportError:
    try:
        from . import _imagecodecs as imagecodecs  # type: ignore[no-redef]
    except ImportError:
        import _imagecodecs as imagecodecs  # type: ignore[no-redef]

if os.environ.get('TIFFFILE_NO_CPP'):
    _cpp_segment_positions = None
else:
    try:
        from ._tifffile_ext import (
            compute_segment_positions as _cpp_segment_positions,
        )
    except ImportError:
        _cpp_segment_positions = None

DecodeResult = tuple[
    'NDArray[Any] | None',
    tuple[int, int, int, int, int],
    tuple[int, int, int, int],
]


def build_decode_closure(
    page: TiffPage,
) -> Callable[..., DecodeResult]:
    """Return decode closure for the given TiffPage.

    The returned function decodes a single segment (strip or tile) and
    returns the decoded data, its position in the image array, and its shape.

    Parameters:
        page: TiffPage instance to build the decode closure for.

    Returns:
        A decode function with signature:
        ``(data, index, *, jpegtables=None, jpegheader=None, _fullsize=False)``

    """
    from .metadata import jpeg_decode_colorspace
    from .tifffile import TIFF
    from .utils import TiffFileError

    if page.dtype is None or page._dtype is None:

        def decode_raise_dtype(*args: Any, **kwargs: Any) -> DecodeResult:
            msg = (
                'data type not supported '
                f'(SampleFormat {page.sampleformat}, '
                f'{page.bitspersample}-bit)'
            )
            raise ValueError(msg)

        return decode_raise_dtype

    if 0 in page.shaped:

        def decode_raise_empty(*args: Any, **kwargs: Any) -> DecodeResult:
            msg = 'empty image'
            raise ValueError(msg)

        return decode_raise_empty

    decompress: Callable[..., Any] | None
    try:
        if page.compression == 1:
            decompress = None
        else:
            decompress = TIFF.DECOMPRESSORS[page.compression]
        if (
            page.compression in {65000, 65001, 65002}
            and not page.parent.is_eer
        ):
            raise KeyError(page.compression)
    except KeyError as exc:
        msg = str(exc)[1:-1]

        def decode_raise_compression(
            *args: Any, msg: str = msg, **kwargs: Any
        ) -> DecodeResult:
            raise ValueError(msg)

        return decode_raise_compression

    unpredict: Callable[..., Any] | None
    try:
        if page.predictor == 1:
            unpredict = None
        else:
            unpredict = TIFF.UNPREDICTORS[page.predictor]
    except KeyError as exc:
        if page.compression in TIFF.IMAGE_COMPRESSIONS:
            logger().warning(
                f'{page!r} ignoring predictor {page.predictor}'
            )
            unpredict = None

        else:
            msg = str(exc)[1:-1]

            def decode_raise_predictor(
                *args: Any, msg: str = msg, **kwargs: Any
            ) -> DecodeResult:
                raise ValueError(msg)

            return decode_raise_predictor

    if page.tags.get(339) is not None:
        tag = page.tags[339]  # SampleFormat
        if tag.count != 1 and any(i - tag.value[0] for i in tag.value):

            def decode_raise_sampleformat(
                *args: Any, **kwargs: Any
            ) -> DecodeResult:
                msg = f'sample formats do not match {tag.value}'
                raise ValueError(msg)

            return decode_raise_sampleformat

    if page.is_subsampled and (
        page.compression not in {6, 7, 34892, 33007}
        or page.planarconfig == 2
    ):

        def decode_raise_subsampling(
            *args: Any, **kwargs: Any
        ) -> DecodeResult:
            msg = (
                'chroma subsampling not supported without JPEG compression'
            )
            raise NotImplementedError(msg)

        return decode_raise_subsampling

    if page.compression == 50001 and page.samplesperpixel == 4:
        # WebP segments may be missing all-opaque alpha channel
        def decompress_webp_rgba(
            data: Any, out: Any = None, **kwargs: Any
        ) -> Any:
            return imagecodecs.webp_decode(data, hasalpha=True, out=out)

        decompress = decompress_webp_rgba

    # normalize segments shape to [depth, length, width, contig]
    if page.is_tiled:
        stshape = (
            page.tiledepth,
            page.tilelength,
            page.tilewidth,
            page.samplesperpixel if page.planarconfig == 1 else 1,
        )
    else:
        stshape = (
            1,
            page.rowsperstrip,
            page.imagewidth,
            page.samplesperpixel if page.planarconfig == 1 else 1,
        )

    stdepth, stlength, stwidth, samples = stshape
    _, imdepth, imlength, imwidth, samples = page.shaped

    if page.is_tiled:
        width = (imwidth + stwidth - 1) // stwidth
        length = (imlength + stlength - 1) // stlength
        depth = (imdepth + stdepth - 1) // stdepth

        def indices(
            segmentindex: int, /
        ) -> tuple[
            tuple[int, int, int, int, int], tuple[int, int, int, int]
        ]:
            return (
                (
                    segmentindex // (width * length * depth),
                    (segmentindex // (width * length)) % depth * stdepth,
                    (segmentindex // width) % length * stlength,
                    segmentindex % width * stwidth,
                    0,
                ),
                stshape,
            )

        def reshape(
            data: NDArray[Any],
            indices: tuple[int, int, int, int, int],
            shape: tuple[int, int, int, int],
            /,
        ) -> NDArray[Any]:
            size = shape[0] * shape[1] * shape[2] * shape[3]
            if data.ndim == 1 and data.size > size:
                data = data[:size]
            if data.size == size:
                return data.reshape(shape)
            try:
                return data.reshape(
                    (
                        min(imdepth - indices[1], shape[0]),
                        min(imlength - indices[2], shape[1]),
                        min(imwidth - indices[3], shape[2]),
                        samples,
                    )
                )
            except ValueError:
                pass
            try:
                return data.reshape(
                    (
                        min(imdepth - indices[1], shape[0]),
                        min(imlength - indices[2], shape[1]),
                        shape[2],
                        samples,
                    )
                )
            except ValueError:
                pass
            msg = (
                f'corrupted tile @ {indices} cannot be reshaped from '
                f'{data.shape} to {shape}'
            )
            raise TiffFileError(msg)

        def pad(
            data: NDArray[Any], shape: tuple[int, int, int, int], /
        ) -> tuple[NDArray[Any], tuple[int, int, int, int]]:
            if data.shape == shape:
                return data, shape
            padwidth = [
                (0, i - j)
                for i, j in zip(shape, data.shape, strict=True)
            ]
            data = numpy.pad(data, padwidth, constant_values=page.nodata)
            return data, shape

        def pad_none(
            shape: tuple[int, int, int, int], /
        ) -> tuple[int, int, int, int]:
            return shape

    else:
        # strips
        length = (imlength + stlength - 1) // stlength

        def indices(
            segmentindex: int, /
        ) -> tuple[
            tuple[int, int, int, int, int], tuple[int, int, int, int]
        ]:
            indices = (
                segmentindex // (length * imdepth),
                (segmentindex // length) % imdepth * stdepth,
                segmentindex % length * stlength,
                0,
                0,
            )
            shape = (
                stdepth,
                min(stlength, imlength - indices[2]),
                stwidth,
                samples,
            )
            return indices, shape

        def reshape(
            data: NDArray[Any],
            indices: tuple[int, int, int, int, int],
            shape: tuple[int, int, int, int],
            /,
        ) -> NDArray[Any]:
            size = shape[0] * shape[1] * shape[2] * shape[3]
            if data.ndim == 1 and data.size > size:
                data = data[:size]
            if data.size == size:
                try:
                    data.shape = shape
                except AttributeError:
                    data = data.reshape(shape)
                return data
            datashape = data.shape
            try:
                data.shape = shape[0], -1, shape[2], shape[3]
                data = data[:, : shape[1]]
                data.shape = shape
            except ValueError:
                pass
            else:
                return data
            msg = (
                'corrupted strip cannot be reshaped from '
                f'{datashape} to {shape}'
            )
            raise TiffFileError(msg)

        def pad(
            data: NDArray[Any], shape: tuple[int, int, int, int], /
        ) -> tuple[NDArray[Any], tuple[int, int, int, int]]:
            shape = (shape[0], stlength, shape[2], shape[3])
            if data.shape == shape:
                return data, shape
            padwidth = [
                (0, 0),
                (0, stlength - data.shape[1]),
                (0, 0),
                (0, 0),
            ]
            data = numpy.pad(data, padwidth, constant_values=page.nodata)
            return data, shape

        def pad_none(
            shape: tuple[int, int, int, int], /
        ) -> tuple[int, int, int, int]:
            return (shape[0], stlength, shape[2], shape[3])

    # Override indices() with pre-computed C++ lookup for many segments
    if _cpp_segment_positions is not None:
        _nseg = 1
        for _d in page.chunked:
            _nseg *= _d
        if _nseg > 100:
            try:
                _pos_arr, _shp_arr = _cpp_segment_positions(
                    _nseg,
                    stshape,
                    (imdepth, imlength, imwidth),
                    page.is_tiled,
                )
                # Convert to Python lists for fast O(1) element access
                _p = _pos_arr.tolist()
                _s = _shp_arr.tolist()

                def indices(
                    segmentindex: int,
                    /,
                    _p: list[int] = _p,
                    _s: list[int] = _s,
                ) -> tuple[
                    tuple[int, int, int, int, int],
                    tuple[int, int, int, int],
                ]:
                    i5 = segmentindex * 5
                    i4 = segmentindex * 4
                    return (
                        (
                            _p[i5],
                            _p[i5 + 1],
                            _p[i5 + 2],
                            _p[i5 + 3],
                            _p[i5 + 4],
                        ),
                        (
                            _s[i4],
                            _s[i4 + 1],
                            _s[i4 + 2],
                            _s[i4 + 3],
                        ),
                    )

            except Exception:
                pass

    if page.compression in {6, 7, 34892, 33007}:
        # JPEG needs special handling
        if page.fillorder == 2:
            logger().debug(f'{page!r} disabling LSB2MSB for JPEG')
        if unpredict:
            logger().debug(f'{page!r} disabling predictor for JPEG')
        if 28672 in page.tags:  # SonyRawFileType
            logger().warning(
                f'{page!r} SonyRawFileType might need additional '
                'unpacking (see issue #95)'
            )

        colorspace, outcolorspace = jpeg_decode_colorspace(
            page.photometric,
            page.planarconfig,
            page.extrasamples,
            page.is_jfif,
        )

        def decode_jpeg(
            data: bytes | None,
            index: int,
            /,
            *,
            jpegtables: bytes | None = None,
            jpegheader: bytes | None = None,
            _fullsize: bool = False,
        ) -> DecodeResult:
            segmentindex, shape = indices(index)
            if data is None:
                if _fullsize:
                    shape = pad_none(shape)
                return data, segmentindex, shape
            data_array: NDArray[Any] = imagecodecs.jpeg_decode(
                data,
                bitspersample=page.bitspersample,
                tables=jpegtables,
                header=jpegheader,
                colorspace=colorspace,
                outcolorspace=outcolorspace,
                shape=shape[1:3],
            )
            data_array = reshape(data_array, segmentindex, shape)
            if _fullsize:
                data_array, shape = pad(data_array, shape)
            return data_array, segmentindex, shape

        return decode_jpeg

    if page.compression in {65000, 65001, 65002}:
        # EER decoder requires shape and extra args
        horzbits = vertbits = 2
        if page.compression == 65002:
            skipbits = int(page.tags.valueof(65007, 7))
            horzbits = int(page.tags.valueof(65008, 2))
            vertbits = int(page.tags.valueof(65009, 2))
        elif page.compression == 65001:
            skipbits = 7
        else:
            skipbits = 8
        superres = page.parent._superres

        def decode_eer(
            data: bytes | None,
            index: int,
            /,
            *,
            jpegtables: bytes | None = None,
            jpegheader: bytes | None = None,
            _fullsize: bool = False,
        ) -> DecodeResult:
            segmentindex, shape = indices(index)
            if data is None:
                if _fullsize:
                    shape = pad_none(shape)
                return data, segmentindex, shape
            data_array = imagecodecs.eer_decode(
                data,
                shape[1:3],
                skipbits,
                horzbits,
                vertbits,
                superres=superres,
            )
            return data_array.reshape(shape), segmentindex, shape

        return decode_eer

    if page.compression == 48124:
        # Jetraw requires pre-allocated output buffer
        assert decompress is not None

        def decode_jetraw(
            data: bytes | None,
            index: int,
            /,
            *,
            jpegtables: bytes | None = None,
            jpegheader: bytes | None = None,
            _fullsize: bool = False,
        ) -> DecodeResult:
            segmentindex, shape = indices(index)
            if data is None:
                if _fullsize:
                    shape = pad_none(shape)
                return data, segmentindex, shape
            data_array = numpy.zeros(shape, numpy.uint16)
            decompress(data, out=data_array)
            return data_array.reshape(shape), segmentindex, shape

        return decode_jetraw

    if page.compression in TIFF.IMAGE_COMPRESSIONS:
        # presume codecs always return correct dtype, native byte order...
        if page.fillorder == 2:
            logger().debug(
                f'{page!r} '
                f'disabling LSB2MSB for compression {page.compression}'
            )
        if unpredict:
            logger().debug(
                f'{page!r} '
                f'disabling predictor for compression {page.compression}'
            )
        assert decompress is not None

        def decode_image(
            data: bytes | None,
            index: int,
            /,
            *,
            jpegtables: bytes | None = None,
            jpegheader: bytes | None = None,
            _fullsize: bool = False,
        ) -> DecodeResult:
            segmentindex, shape = indices(index)
            if data is None:
                if _fullsize:
                    shape = pad_none(shape)
                return data, segmentindex, shape
            data_array: NDArray[Any]
            data_array = decompress(data)  # type: ignore[misc]
            data_array = reshape(data_array, segmentindex, shape)
            if _fullsize:
                data_array, shape = pad(data_array, shape)
            return data_array, segmentindex, shape

        return decode_image

    dtype = numpy.dtype(page.parent.byteorder + page._dtype.char)

    unpack: Callable[[bytes], NDArray[Any]]

    if page.sampleformat == 5:
        # complex integer
        if unpredict is not None:
            msg = 'unpredicting complex integers not supported'
            raise NotImplementedError(msg)

        itype = numpy.dtype(
            f'{page.parent.byteorder}i{page.bitspersample // 16}'
        )
        ftype = numpy.dtype(
            f'{page.parent.byteorder}f{dtype.itemsize // 2}'
        )

        def unpack(data: bytes, /) -> NDArray[Any]:
            return numpy.frombuffer(data, itype).astype(ftype).view(dtype)

    elif page.bitspersample in {8, 16, 32, 64, 128}:
        # regular data types

        if (page.bitspersample * stwidth * samples) % 8:
            msg = 'data and sample size mismatch'
            raise ValueError(msg)
        if page.predictor in {3, 34894, 34895}:  # PREDICTOR.FLOATINGPOINT
            dtype = numpy.dtype(page._dtype.char)

        def unpack(data: bytes, /) -> NDArray[Any]:
            try:
                return numpy.frombuffer(data, dtype)
            except ValueError:
                bps = page.bitspersample // 8
                size = (len(data) // bps) * bps
                return numpy.frombuffer(data[:size], dtype)

    elif isinstance(page.bitspersample, tuple):
        # for example, RGB 565
        def unpack(data: bytes, /) -> NDArray[Any]:
            return unpack_rgb(data, dtype, page.bitspersample)

    elif page.bitspersample == 24 and dtype.char == 'f':
        # float24
        if unpredict is not None:
            msg = 'unpredicting float24 not supported'
            raise NotImplementedError(msg)

        def unpack(data: bytes, /) -> NDArray[Any]:
            return imagecodecs.float24_decode(
                data, byteorder=page.parent.byteorder
            )

    else:
        # bilevel and packed integers
        def unpack(data: bytes, /) -> NDArray[Any]:
            return imagecodecs.packints_decode(
                data, dtype, page.bitspersample, runlen=stwidth * samples
            )

    def decode_other(
        data: bytes | None,
        index: int,
        /,
        *,
        jpegtables: bytes | None = None,
        jpegheader: bytes | None = None,
        _fullsize: bool = False,
    ) -> DecodeResult:
        segmentindex, shape = indices(index)
        if data is None:
            if _fullsize:
                shape = pad_none(shape)
            return data, segmentindex, shape
        if page.fillorder == 2:
            data = imagecodecs.bitorder_decode(data)
        if decompress is not None:
            size = shape[0] * shape[1] * shape[2] * shape[3]
            data = decompress(data, out=size * dtype.itemsize)
        data_array = unpack(data)  # type: ignore[arg-type]
        data_array = reshape(data_array, segmentindex, shape)
        data_array = data_array.astype('=' + dtype.char, copy=False)
        if unpredict is not None:
            data_array = unpredict(data_array, axis=-2, out=data_array)
        if _fullsize:
            data_array, shape = pad(data_array, shape)
        return data_array, segmentindex, shape

    return decode_other
