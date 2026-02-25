# series_parsers/microscopy.py

"""Microscopy format series parsers: STK, FluoView, ScanImage, NIH,
MicroManager, NDTiff."""

from __future__ import annotations

import glob
import os
from typing import TYPE_CHECKING

import numpy

from ..series import SeriesParser
from ..utils import logger, order_axes, product

if TYPE_CHECKING:
    from typing import Any

    from ..tifffile import TiffFile, TiffPage, TiffPageSeries

    NDArray = numpy.ndarray[Any, Any]


class StkSeriesParser(SeriesParser):
    """Return series in STK file."""

    kind = 'stk'

    def parse(self, tiff: TiffFile) -> list[TiffPageSeries] | None:
        from ..tifffile import TiffPageSeries

        meta = tiff.stk_metadata
        if meta is None:
            return None
        page = tiff.pages.first
        planes = meta['NumberPlanes']
        name = meta.get('Name', '')
        if planes == 1:
            shape = (1, *page.shape)
            axes = 'I' + page.axes
        elif numpy.all(meta['ZDistance'] != 0):
            shape = (planes, *page.shape)
            axes = 'Z' + page.axes
        elif numpy.all(numpy.diff(meta['TimeCreated']) != 0):
            shape = (planes, *page.shape)
            axes = 'T' + page.axes
        else:
            shape = (planes, *page.shape)
            axes = 'I' + page.axes
        tiff.is_uniform = True
        series = TiffPageSeries(
            [page],
            shape,
            page.dtype,
            axes,
            name=name,
            truncated=planes > 1,
            kind='stk',
        )
        return [series]


class FluoViewSeriesParser(SeriesParser):
    """Return image series in FluoView file."""

    kind = 'fluoview'

    def parse(self, tiff: TiffFile) -> list[TiffPageSeries] | None:
        from ..tifffile import TIFF, TiffPageSeries

        meta = tiff.fluoview_metadata
        if meta is None:
            return None
        pages = tiff.pages._getlist(validate=False)
        mmhd = list(reversed(meta['Dimensions']))
        axes = ''.join(
            TIFF.MM_DIMENSIONS.get(i[0].upper(), 'Q') for i in mmhd
        )
        shape = tuple(int(i[1]) for i in mmhd)
        tiff.is_uniform = True
        return [
            TiffPageSeries(
                pages,
                shape,
                pages[0].dtype,
                axes,
                name=meta['ImageName'],
                kind='fluoview',
            )
        ]


class ScanImageSeriesParser(SeriesParser):
    """Return image series in ScanImage file."""

    kind = 'scanimage'

    def parse(self, tiff: TiffFile) -> list[TiffPageSeries] | None:
        from ..tifffile import TiffPageSeries

        pages = tiff.pages._getlist(validate=False)
        page = tiff.pages.first
        dtype = page.dtype
        shape = None

        meta = tiff.scanimage_metadata
        framedata = {} if meta is None else meta.get('FrameData', {})
        if 'SI.hChannels.channelSave' in framedata:
            try:
                channels = framedata['SI.hChannels.channelSave']
                try:
                    channels = len(channels)
                except TypeError:
                    channels = 1
                slices = None
                try:
                    frames = int(
                        framedata['SI.hStackManager.framesPerSlice']
                    )
                except Exception as exc:
                    slices = 1
                    if len(pages) % channels:
                        msg = 'unable to determine framesPerSlice'
                        raise ValueError(msg) from exc
                    frames = len(pages) // channels
                if slices is None:
                    slices = max(len(pages) // (frames * channels), 1)
                shape = (slices, frames, channels, *page.shape)
                axes = 'ZTC' + page.axes
            except Exception as exc:
                logger().warning(
                    f'{tiff!r} ScanImage series raised {exc!r:.128}'
                )

        if shape is None:
            shape = (len(pages), *page.shape)
            axes = 'I' + page.axes

        return [
            TiffPageSeries(pages, shape, dtype, axes, kind='scanimage')
        ]


class NihSeriesParser(SeriesParser):
    """Return all images in NIH Image file as single series."""

    kind = 'nih'

    def parse(self, tiff: TiffFile) -> list[TiffPageSeries] | None:
        from .generic import UniformSeriesParser

        series = UniformSeriesParser().parse(tiff)
        if series is not None:
            for s in series:
                s.kind = 'nih'
        return series


class MmstackSeriesParser(SeriesParser):
    """Return series in Micro-Manager stack file(s)."""

    kind = 'mmstack'

    def on_failure(self, tiff: TiffFile) -> bool:
        return True  # continue to try other parsers

    def parse(self, tiff: TiffFile) -> list[TiffPageSeries] | None:
        from ..metadata import read_micromanager_metadata
        from ..tifffile import TiffFile as TiffFileClass
        from ..tifffile import TiffFrame, TiffPageSeries

        settings = tiff.micromanager_metadata
        if (
            settings is None
            or 'Summary' not in settings
            or 'IndexMap' not in settings
        ):
            return None

        pages_list: list[TiffPage | Any | None]
        page_count: int

        summary = settings['Summary']
        indexmap = settings['IndexMap']
        indexmap = indexmap[indexmap[:, 4].argsort()]

        if (
            'MicroManagerVersion' not in summary
            or 'Frames' not in summary
        ):
            return None

        indexmap_shape = (
            numpy.max(indexmap[:, :4], axis=0) + 1
        ).tolist()
        indexmap_index = {'C': 0, 'Z': 1, 'T': 2, 'R': 3}

        axes = 'TR' if summary.get('TimeFirst', True) else 'RT'
        axes += 'ZC' if summary.get('SlicesFirst', True) else 'CZ'

        keys = {
            'C': 'Channels',
            'Z': 'Slices',
            'R': 'Positions',
            'T': 'Frames',
        }
        shape = tuple(
            max(
                indexmap_shape[indexmap_index[ax]],
                int(summary.get(keys[ax], 1)),
            )
            for ax in axes
        )
        size = product(shape)

        indexmap_order = tuple(indexmap_index[ax] for ax in axes)

        def add_file(t: TiffFile, indexmap: NDArray) -> int:
            page_count = 0
            offsets: list[int]
            offsets = indexmap[:, 4].tolist()
            indices = numpy.ravel_multi_index(
                indexmap[:, indexmap_order].T, shape
            ).tolist()
            keyframe = t.pages.first
            filesize = (
                t.filehandle.size - keyframe.databytecounts[0] - 162
            )
            index: int
            offset: int
            for item in zip(indices, offsets, strict=True):
                index, offset = item
                if offset == keyframe.offset:
                    pages_list[index] = keyframe
                    page_count += 1
                    continue
                if 0 < offset <= filesize:
                    dataoffsets = (offset + 162,)
                    databytecounts = keyframe.databytecounts
                    page_count += 1
                else:
                    dataoffsets = databytecounts = (0,)
                    offset = 0
                pages_list[index] = TiffFrame(
                    t,
                    index=index,
                    offset=offset,
                    dataoffsets=dataoffsets,
                    databytecounts=databytecounts,
                    keyframe=keyframe,
                )
            return page_count

        multifile = size > indexmap.shape[0]
        if multifile:
            if not tiff.filehandle.is_file:
                logger().warning(
                    f'{tiff!r} MMStack multi-file series cannot be read '
                    f'from {tiff.filehandle._fh!r}'
                )
                multifile = False
            elif '_MMStack' not in tiff.filename:
                logger().warning(
                    f'{tiff!r} MMStack file name is invalid'
                )
                multifile = False
            elif 'Prefix' in summary:
                prefix = summary['Prefix']
                if not tiff.filename.startswith(prefix):
                    logger().warning(
                        f'{tiff!r} MMStack file name is invalid'
                    )
                    multifile = False
            else:
                prefix = tiff.filename.split('_MMStack')[0]

        if multifile:
            pattern = os.path.join(
                tiff.filehandle.dirname, prefix + '_MMStack*.tif'
            )
            filenames = glob.glob(pattern)
            if len(filenames) == 1:
                multifile = False
            else:
                pages_list = [None] * size
                page_count = add_file(tiff, indexmap)
                for filename in filenames:
                    if tiff.filename == os.path.split(filename)[-1]:
                        continue
                    with TiffFileClass(filename) as t:
                        indexmap = read_micromanager_metadata(
                            t.filehandle, {'IndexMap'}
                        )['IndexMap']
                        indexmap = indexmap[indexmap[:, 4].argsort()]
                        page_count += add_file(t, indexmap)

        if multifile:
            pass
        elif size > indexmap.shape[0]:
            old_shape = shape
            min_index = numpy.min(indexmap[:, :4], axis=0)
            max_index = numpy.max(indexmap[:, :4], axis=0)
            indexmap = indexmap.copy()
            indexmap[:, :4] -= min_index
            shape = tuple(
                j - i + 1
                for i, j in zip(
                    min_index.tolist(),
                    max_index.tolist(),
                    strict=True,
                )
            )
            shape = tuple(shape[i] for i in indexmap_order)
            size = product(shape)
            pages_list = [None] * size
            page_count = add_file(tiff, indexmap)
            logger().warning(
                f'{tiff!r} MMStack series is missing files. '
                f'Returning subset {shape!r} of {old_shape!r}'
            )
        else:
            pages_list = [None] * size
            page_count = add_file(tiff, indexmap)

        if page_count != size:
            logger().warning(
                f'{tiff!r} MMStack is missing {size - page_count} pages.'
                ' Missing data are zeroed'
            )

        keyframe = tiff.pages.first
        return [
            TiffPageSeries(
                pages_list,
                shape=shape + keyframe.shape,
                dtype=keyframe.dtype,
                axes=axes + keyframe.axes,
                parent=tiff,
                kind='mmstack',
                multifile=multifile,
                squeeze=True,
            )
        ]


class NdtiffSeriesParser(SeriesParser):
    """Return series in NDTiff v2 and v3 files."""

    kind = 'ndtiff'

    def parse(self, tiff: TiffFile) -> list[TiffPageSeries] | None:
        from ..metadata import read_ndtiff_index
        from ..tifffile import TIFF
        from ..tifffile import TiffFile as TiffFileClass
        from ..tifffile import TiffFrame, TiffPageSeries

        if not tiff.filehandle.is_file:
            logger().warning(
                f'{tiff!r} NDTiff.index not found for '
                f'{tiff.filehandle._fh!r}'
            )
            return None

        indexfile = os.path.join(tiff.filehandle.dirname, 'NDTiff.index')
        if not os.path.exists(indexfile):
            logger().warning(f'{tiff!r} NDTiff.index not found')
            return None

        keyframes: dict[str, TiffPage] = {}
        shape: tuple[int, ...]
        dims: tuple[str, ...]
        page: TiffPage | Any
        pageindex = 0
        pixel_types = {
            0: ('uint8', 8),
            1: ('uint16', 16),
            2: ('uint8', 8),
            3: ('uint16', 10),
            4: ('uint16', 12),
            5: ('uint16', 14),
            6: ('uint16', 11),
        }

        indices: dict[tuple[int, ...], TiffPage | Any] = {}
        categories: dict[str, dict[str, int]] = {}
        first = True

        for (
            axes_dict,
            filename,
            dataoffset,
            width,
            height,
            pixeltype,
            compression,
            _metaoffset,
            _metabytecount,
            _metacompression,
        ) in read_ndtiff_index(indexfile):
            if filename in keyframes:
                pageindex += 1
                keyframe = keyframes[filename]
                page = TiffFrame(
                    keyframe.parent,
                    pageindex,
                    offset=None,
                    keyframe=keyframe,
                    dataoffsets=(dataoffset,),
                    databytecounts=keyframe.databytecounts,
                )
                if page.shape[:2] != (height, width):
                    msg = (
                        'NDTiff.index does not match TIFF shape '
                        f'{page.shape[:2]} != {(height, width)}'
                    )
                    raise ValueError(msg)
                if compression != 0:
                    msg = (
                        'NDTiff.index compression '
                        f'{compression} not supported'
                    )
                    raise ValueError(msg)
                if page.compression != 1:
                    msg = (
                        'NDTiff.index does not match TIFF compression '
                        f'{page.compression!r}'
                    )
                    raise ValueError(msg)
                if pixeltype not in pixel_types:
                    msg = (
                        f'NDTiff.index unknown pixel type {pixeltype}'
                    )
                    raise ValueError(msg)
                dtype, _ = pixel_types[pixeltype]
                if page.dtype != dtype:
                    msg = (
                        'NDTiff.index pixeltype does not match TIFF '
                        f'dtype {page.dtype} != {dtype}'
                    )
                    raise ValueError(msg)
            elif filename == tiff.filename:
                pageindex = 0
                page = tiff.pages.first
                keyframes[filename] = page
            else:
                pageindex = 0
                with TiffFileClass(
                    os.path.join(tiff.filehandle.dirname, filename)
                ) as t:
                    page = t.pages.first
                keyframes[filename] = page

            index: int | str
            if first:
                for axis, index in axes_dict.items():
                    if isinstance(index, str):
                        categories[axis] = {index: 0}
                        axes_dict[axis] = 0
                first = False
                dims = tuple(axes_dict.keys())
            elif categories:
                for axis, values in categories.items():
                    index = axes_dict[axis]
                    assert isinstance(index, str)
                    if index not in values:
                        values[index] = max(values.values()) + 1
                    axes_dict[axis] = values[index]

            if tuple(axes_dict.keys()) != dims:
                dims_ = tuple(axes_dict.keys())
                logger().warning(
                    f'{tiff!r} NDTiff.index '
                    f'axes_dict.keys={dims_} != {dims}'
                )
            indices[tuple(int(axes_dict[dim]) for dim in dims)] = page

        indices_array = numpy.array(
            list(indices.keys()), dtype=numpy.int32
        )
        min_index = numpy.min(indices_array, axis=0).tolist()
        max_index = numpy.max(indices_array, axis=0).tolist()
        shape = tuple(
            j - i + 1
            for i, j in zip(min_index, max_index, strict=True)
        )

        order = order_axes(indices_array, squeeze=False)
        shape = tuple(shape[i] for i in order)
        dims = tuple(dims[i] for i in order)
        indices = {
            tuple(idx[i] - min_index[i] for i in order): value
            for idx, value in indices.items()
        }

        pages_list: list[TiffPage | Any | None] = [
            indices.get(idx) for idx in numpy.ndindex(shape)
        ]

        if not keyframes:
            logger().error(f'{tiff!r} NDTiff.index has no keyframes')
            return None
        keyframe = next(iter(keyframes.values()))
        shape += keyframe.shape
        dims += keyframe.dims
        axes = ''.join(
            TIFF.AXES_CODES.get(i.lower(), 'Q') for i in dims
        )

        tiff.is_uniform = True
        return [
            TiffPageSeries(
                pages_list,
                shape=shape,
                dtype=keyframe.dtype,
                axes=axes,
                parent=tiff,
                kind='ndtiff',
                multifile=len(keyframes) > 1,
                squeeze=True,
            )
        ]
