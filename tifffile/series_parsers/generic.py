# series_parsers/generic.py

"""Generic, uniform, and shaped series parsers."""

from __future__ import annotations

import os
import struct
from typing import TYPE_CHECKING, cast

import numpy

from ..series import SeriesParser
from ..utils import check_shape, logger, product, reshape_axes

if TYPE_CHECKING:
    from typing import Any

    from ..tifffile import (
        TiffFile,
        TiffPage,
        TiffPageSeries,
        TiffPages,
    )

    NDArray = numpy.ndarray[Any, Any]

# C++ acceleration for bulk tag extraction
if os.environ.get('TIFFFILE_NO_CPP'):
    _HAS_CPP = False
else:
    try:
        from .._tifffile_ext import (
            CppFileReader as _CppFileReader,
            CppTiffFormat as _CppTiffFormat,
            bulk_extract_tag_values as _cpp_bulk_extract,
        )

        _HAS_CPP = True
        _CPP_FORMATS = {
            (42, '<'): _CppTiffFormat.classic_le(),
            (42, '>'): _CppTiffFormat.classic_be(),
            (43, '<'): _CppTiffFormat.big_le(),
            (43, '>'): _CppTiffFormat.big_be(),
        }
    except ImportError:
        _HAS_CPP = False

# Tag codes needed for page hash computation and frame creation
_HASH_TAG_CODES = [
    256,    # ImageWidth
    257,    # ImageLength
    258,    # BitsPerSample
    259,    # Compression
    262,    # Photometric
    266,    # FillOrder
    273,    # StripOffsets
    277,    # SamplesPerPixel
    278,    # RowsPerStrip
    279,    # StripByteCounts
    284,    # PlanarConfiguration
    317,    # Predictor
    322,    # TileWidth
    323,    # TileLength
    324,    # TileOffsets
    325,    # TileByteCounts
    330,    # SubIFDs
    338,    # ExtraSamples
    339,    # SampleFormat
    32997,  # ImageDepth
    32998,  # TileDepth
]


class UniformSeriesParser(SeriesParser):
    """Return all images in file as single series."""

    kind = 'uniform'

    def parse(self, tiff: TiffFile) -> list[TiffPageSeries] | None:
        from ..tifffile import TiffPageSeries

        tiff.pages.useframes = True
        tiff.pages.set_keyframe(0)
        page = tiff.pages.first
        validate = not (page.is_scanimage or page.is_nih)
        pages = tiff.pages._getlist(validate=validate)
        if len(pages) == 1:
            shape = page.shape
            axes = page.axes
        else:
            shape = (len(pages), *page.shape)
            axes = 'I' + page.axes
        dtype = page.dtype
        return [TiffPageSeries(pages, shape, dtype, axes, kind='uniform')]


class GenericSeriesParser(SeriesParser):
    """Return image series in file grouped by page hash."""

    kind = 'generic'

    def can_parse(self, tiff: TiffFile) -> bool:
        return True  # generic is always applicable

    def parse(self, tiff: TiffFile) -> list[TiffPageSeries] | None:
        from ..tifffile import (
            TiffPage,
            TiffPageSeries,
            pyramidize_series,
        )

        pages = tiff.pages

        # Try C++ fast path for large files
        if (
            _HAS_CPP
            and len(pages) > 100
            and tiff.filehandle.is_file
            and not tiff.tiff.is_ndpi
        ):
            try:
                result = self._parse_fast(tiff)
                if result is not None:
                    return result
            except Exception as exc:
                logger().debug(f'fast path failed: {exc!r:.128}')

        # Standard path
        pages._clear(fully=False)
        pages.useframes = False
        if pages.cache:
            pages._load()

        series: list[TiffPageSeries] = []
        keys: list[int] = []
        seriesdict: dict[int, list[TiffPage | Any]] = {}
        seen_offsets: dict[int, set[int]] = {}

        def addpage(page: TiffPage | Any, /) -> None:
            if not page.shape:
                return
            key = page.hash
            if key in seriesdict:
                if page.offset not in seen_offsets[key]:
                    seriesdict[key].append(page)
                    seen_offsets[key].add(page.offset)
            else:
                keys.append(key)
                seriesdict[key] = [page]
                seen_offsets[key] = {page.offset}

        for page in pages:
            addpage(page)
            if page.subifds is not None:
                for i, offset in enumerate(page.subifds):
                    if offset < 8:
                        continue
                    try:
                        tiff._fh.seek(offset)
                        subifd = TiffPage(tiff, (page.index, i))
                    except Exception as exc:
                        logger().warning(
                            f'{tiff!r} generic series raised {exc!r:.128}'
                        )
                    else:
                        addpage(subifd)

        for key in keys:
            pagelist = seriesdict[key]
            page = pagelist[0]
            shape = (len(pagelist), *page.shape)
            axes = 'I' + page.axes
            if 'S' not in axes:
                shape += (1,)
                axes += 'S'
            series.append(
                TiffPageSeries(
                    pagelist, shape, page.dtype, axes, kind='generic'
                )
            )

        tiff.is_uniform = len(series) == 1
        if not tiff.is_agilent:
            pyramidize_series(series)
        return series

    def _parse_fast(
        self, tiff: TiffFile
    ) -> list[TiffPageSeries] | None:
        """Fast series grouping using C++ bulk tag extraction.

        Avoids creating TiffPage objects for every page. Instead, extracts
        hash-relevant tags in bulk via mmap, computes hashes from raw values,
        and only creates TiffPage for keyframes (first page of each group).
        Remaining pages are created as lightweight TiffFrame objects.

        Returns None to fall back to the standard path.

        """
        from ..page import TiffFrame
        from ..tifffile import (
            TiffPage,
            TiffPageSeries,
            pyramidize_series,
        )

        pages = tiff.pages
        fh = tiff.filehandle

        # Ensure all IFD offsets are known
        if not pages._indexed:
            pages._seek(-1)

        raw_pages = pages._pages
        n_pages = len(raw_pages)
        if n_pages < 2:
            return None

        # Collect int offsets (pages not yet loaded as objects)
        all_offsets = []
        for p in raw_pages:
            if isinstance(p, (int, numpy.integer)):
                all_offsets.append(int(p))
            else:
                all_offsets.append(p.offset)

        # Get C++ format
        tiff_fmt = tiff.tiff
        cpp_fmt = _CPP_FORMATS.get((tiff_fmt.version, tiff_fmt.byteorder))
        if cpp_fmt is None:
            return None

        # Bulk extract all hash-relevant tags via C++ mmap
        offsets_packed = struct.pack(f'={n_pages}Q', *all_offsets)
        reader = _CppFileReader(fh.path)
        tag_data = _cpp_bulk_extract(
            reader, offsets_packed, cpp_fmt, _HASH_TAG_CODES
        )

        # Helper to get single tag value with default
        def tag1(code: int, idx: int, default: int = 0) -> int:
            vals = tag_data[code][idx]
            return vals[0] if vals else default

        # Compute hash for each page and group
        tiff_hash = hash(tiff_fmt)
        keys: list[int] = []
        seriesdict: dict[int, list[int]] = {}
        seen_offsets: dict[int, set[int]] = {}
        has_subifds = False

        for i in range(n_pages):
            imagewidth = tag1(256, i)
            imagelength = tag1(257, i)
            imagedepth = tag1(32997, i, 1)
            spp = tag1(277, i, 1)
            photometric = tag1(262, i)
            planarconfig = tag1(284, i, 1)

            # Compute shaped tuple (matches TiffPage logic)
            if photometric == 2 or spp > 1:
                if planarconfig == 1:
                    shaped = (1, imagedepth, imagelength, imagewidth, spp)
                else:
                    shaped = (spp, imagedepth, imagelength, imagewidth, 1)
            else:
                shaped = (1, imagedepth, imagelength, imagewidth, 1)

            if not imagewidth or not imagelength:
                # Skip empty pages
                continue

            # BitsPerSample: single value → int, multi-value → tuple
            bps_vals = tag_data[258][i]
            if len(bps_vals) == 1:
                bps = bps_vals[0]
            elif bps_vals:
                bps = tuple(bps_vals)
            else:
                bps = 1

            # ExtraSamples: always tuple
            es_vals = tag_data[338][i]
            extrasamples = tuple(es_vals) if es_vals else ()

            # RowsPerStrip defaults to imagelength
            rowsperstrip = tag1(278, i, imagelength)

            h = hash((
                *shaped,
                tiff_hash,
                rowsperstrip,
                tag1(322, i),        # tilewidth
                tag1(323, i),        # tilelength
                tag1(32998, i, 1),   # tiledepth
                tag1(339, i, 1),     # sampleformat
                bps,
                tag1(266, i, 1),     # fillorder
                tag1(317, i, 1),     # predictor
                tag1(259, i, 1),     # compression
                extrasamples,
                photometric,
            ))

            if h in seriesdict:
                # Dedup by offset (O(1) set lookup)
                page_offset = all_offsets[i]
                if page_offset not in seen_offsets[h]:
                    seriesdict[h].append(i)
                    seen_offsets[h].add(page_offset)
            else:
                keys.append(h)
                seriesdict[h] = [i]
                seen_offsets[h] = {all_offsets[i]}

            # Check SubIFDs
            if tag_data[330][i]:
                has_subifds = True

        if not keys:
            return None

        # If any page has SubIFDs, fall back to standard path
        # (SubIFDs need TiffPage creation with file seeks)
        if has_subifds:
            return None

        # Build series from groups
        series: list[TiffPageSeries] = []
        pages._clear(fully=False)
        pages.useframes = True

        for key in keys:
            page_indices = seriesdict[key]
            keyframe_idx = page_indices[0]

            # Create the keyframe as a full TiffPage
            pages.set_keyframe(keyframe_idx)
            keyframe = cast(TiffPage, pages.keyframe)
            if not keyframe.shape:
                continue

            pagelist: list[TiffPage | Any] = [keyframe]

            # Create TiffFrame for remaining pages in the group
            for idx in page_indices[1:]:
                offset = all_offsets[idx]

                # Get dataoffsets and databytecounts from bulk extraction
                tile_off = tag_data[324][idx]
                strip_off = tag_data[273][idx]
                dataoffsets = tile_off if tile_off else strip_off

                if keyframe.is_contiguous:
                    databytecounts = keyframe.databytecounts
                else:
                    tile_bc = tag_data[325][idx]
                    strip_bc = tag_data[279][idx]
                    databytecounts = tile_bc if tile_bc else strip_bc

                pageindex: int | tuple[int, ...] = (
                    idx
                    if pages._index is None
                    else (*pages._index, idx)
                )

                frame = TiffFrame(
                    tiff,
                    index=pageindex,
                    offset=offset if offset < 2**31 - 1 else None,
                    dataoffsets=dataoffsets,
                    databytecounts=databytecounts,
                    keyframe=keyframe,
                )
                pagelist.append(frame)

                # Cache the frame in pages list
                if pages._cache:
                    raw_pages[idx] = frame

            shape = (len(pagelist), *keyframe.shape)
            axes = 'I' + keyframe.axes
            if 'S' not in axes:
                shape += (1,)
                axes += 'S'

            series.append(
                TiffPageSeries(
                    pagelist, shape, keyframe.dtype, axes, kind='generic'
                )
            )

        tiff.is_uniform = len(series) == 1
        if not tiff.is_agilent:
            pyramidize_series(series)
        return series


class ShapedSeriesParser(SeriesParser):
    """Return image series in tifffile 'shaped' formatted file."""

    kind = 'shaped'

    def parse(self, tiff: TiffFile) -> list[TiffPageSeries] | None:
        from ..metadata import shaped_description_metadata
        from ..tifffile import (
            TiffFrame,
            TiffPage,
            TiffPageSeries,
            TiffPages,
            pyramidize_series,
        )

        def append(
            series: list[TiffPageSeries],
            pages: list[TiffPage | Any | None],
            axes: str | None,
            shape: tuple[int, ...] | None,
            reshape: tuple[int, ...],
            name: str,
            truncated: bool | None,
            /,
        ) -> None:
            assert isinstance(pages[0], TiffPage)
            page = pages[0]
            if not check_shape(page.shape, reshape):
                logger().warning(
                    f'{tiff!r} shaped series metadata does not match '
                    f'page shape {page.shape} != {tuple(reshape)}'
                )
                failed = True
            else:
                failed = False
            if failed or axes is None or shape is None:
                shape = page.shape
                axes = page.axes
                if len(pages) > 1:
                    shape = (len(pages), *shape)
                    axes = 'Q' + axes
                if failed:
                    reshape = shape
            size = product(shape)
            resize = product(reshape)
            if page.is_contiguous and resize > size and resize % size == 0:
                if truncated is None:
                    truncated = True
                axes = 'Q' + axes
                shape = (resize // size, *shape)
            try:
                axes = reshape_axes(axes, shape, reshape)
                shape = reshape
            except ValueError as exc:
                logger().error(
                    f'{tiff!r} shaped series failed to reshape, '
                    f'raised {exc!r:.128}'
                )
            series.append(
                TiffPageSeries(
                    pages,
                    shape,
                    page.dtype,
                    axes,
                    name=name,
                    kind='shaped',
                    truncated=bool(truncated),
                    squeeze=False,
                )
            )

        def detect_series(
            pages: TiffPages | list[TiffPage | Any | None],
            series: list[TiffPageSeries],
            /,
        ) -> list[TiffPageSeries] | None:
            shape: tuple[int, ...] | None
            reshape: tuple[int, ...]
            page: TiffPage | Any | None
            keyframe: TiffPage
            subifds: list[TiffPage | Any | None] = []
            subifd: TiffPage | Any
            keysubifd: TiffPage
            axes: str | None
            name: str

            lenpages = len(pages)
            index = 0
            while True:
                if index >= lenpages:
                    break

                if isinstance(pages, TiffPages):
                    pages.set_keyframe(index)
                    keyframe = cast(TiffPage, pages.keyframe)
                else:
                    keyframe = cast(TiffPage, pages[0])

                if keyframe.shaped_description is None:
                    logger().error(
                        f'{tiff!r} '
                        'invalid shaped series metadata or corrupted file'
                    )
                    return None
                axes = None
                shape = None
                metadata = shaped_description_metadata(
                    keyframe.shaped_description
                )
                name = metadata.get('name', '')
                reshape = metadata['shape']
                truncated = None if keyframe.subifds is None else False
                truncated = metadata.get('truncated', truncated)
                if 'axes' in metadata:
                    axes = cast(str, metadata['axes'])
                    if len(axes) == len(reshape):
                        shape = reshape
                    else:
                        axes = ''
                        logger().error(
                            f'{tiff!r} shaped series axes do not match shape'
                        )
                spages: list[TiffPage | Any | None] = [keyframe]
                size = product(reshape)
                if size > 0:
                    npages, mod = divmod(size, product(keyframe.shape))
                else:
                    npages = 1
                    mod = 0
                if mod:
                    logger().error(
                        f'{tiff!r} '
                        'shaped series shape does not match page shape'
                    )
                    return None

                if (
                    npages <= 1
                    and isinstance(pages, TiffPages)
                    and lenpages - index > 100
                ):
                    # Many single-page shaped series — fall back to generic
                    # parser which can use C++ bulk extraction
                    return None

                if 1 < npages <= lenpages - index:
                    assert keyframe._dtype is not None
                    size *= keyframe._dtype.itemsize
                    if truncated:
                        npages = 1
                    else:
                        page = pages[index + 1]
                        if (
                            keyframe.is_final
                            and page is not None
                            and keyframe.offset + size < page.offset
                            and keyframe.subifds is None
                        ):
                            truncated = False
                        else:
                            truncated = False
                            for j in range(index + 1, index + npages):
                                page = pages[j]
                                assert page is not None
                                page.keyframe = keyframe
                                spages.append(page)
                append(series, spages, axes, shape, reshape, name, truncated)
                index += npages

                if keyframe.subifds:
                    subifds_size = len(keyframe.subifds)
                    for i, offset in enumerate(keyframe.subifds):
                        if offset < 8:
                            continue
                        subifds = []
                        for j, page in enumerate(spages):
                            try:
                                if (
                                    page is None
                                    or page.subifds is None
                                    or len(page.subifds) < subifds_size
                                ):
                                    msg = (
                                        f'{page!r} contains invalid subifds'
                                    )
                                    raise ValueError(msg)
                                tiff._fh.seek(page.subifds[i])
                                if j == 0:
                                    subifd = TiffPage(
                                        tiff, (page.index, i)
                                    )
                                    keysubifd = subifd
                                else:
                                    subifd = TiffFrame(
                                        tiff,
                                        (page.index, i),
                                        keyframe=keysubifd,
                                    )
                            except Exception as exc:
                                logger().error(
                                    f'{tiff!r} shaped series '
                                    f'raised {exc!r:.128}'
                                )
                                return None
                            subifds.append(subifd)
                        if subifds:
                            series_or_none = detect_series(subifds, series)
                            if series_or_none is None:
                                return None
                            series = series_or_none
            return series

        tiff.pages.useframes = True
        series = detect_series(tiff.pages, [])
        if series is None:
            return None
        tiff.is_uniform = len(series) == 1
        pyramidize_series(series, reduced=True)
        return series
