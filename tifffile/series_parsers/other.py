# series_parsers/other.py

"""Other format series parsers: MDGel, EER, AVS, SIS."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy

from ..series import SeriesParser
from ..utils import logger

if TYPE_CHECKING:
    from typing import Any

    from ..tifffile import TiffFile, TiffPageSeries

    NDArray = numpy.ndarray[Any, Any]


class MdgelSeriesParser(SeriesParser):
    """Return image series in MD Gel file."""

    kind = 'mdgel'

    def parse(self, tiff: TiffFile) -> list[TiffPageSeries] | None:
        from collections.abc import Callable

        from ..tifffile import TiffPageSeries

        meta = tiff.mdgel_metadata
        if meta is None:
            return None
        transform: Callable[[NDArray], NDArray] | None
        tiff.pages.useframes = False
        tiff.pages.set_keyframe(0)

        if meta['FileTag'] in {2, 128}:
            dtype = numpy.dtype(numpy.float32)
            scale = meta['ScalePixel']
            scale = scale[0] / scale[1]
            if meta['FileTag'] == 2:

                def transform(a: NDArray, /) -> NDArray:
                    return a.astype(numpy.float32) ** 2 * scale

            else:

                def transform(a: NDArray, /) -> NDArray:
                    return a.astype(numpy.float32) * scale

        else:
            transform = None
        page = tiff.pages.first
        tiff.is_uniform = False
        return [
            TiffPageSeries(
                [page],
                page.shape,
                dtype,
                page.axes,
                transform=transform,
                kind='mdgel',
            )
        ]


class EerSeriesParser(SeriesParser):
    """Return image series in EER file."""

    kind = 'eer'

    def parse(self, tiff: TiffFile) -> list[TiffPageSeries] | None:
        from ..tifffile import TiffPageSeries

        series: list[TiffPageSeries] = []
        page = tiff.pages.first
        if page.compression == 1:
            if len(tiff.pages) < 2:
                return None
            series.append(
                TiffPageSeries(
                    [page],
                    page.shape,
                    page.dtype,
                    page.axes,
                    name='integrated',
                    kind='eer',
                )
            )
            tiff.is_uniform = False
            page = tiff.pages[1].aspage()

        assert page.compression in {65000, 65001, 65002}
        tiff.pages.useframes = True
        tiff.pages.set_keyframe(page.index)
        pages = tiff.pages._getlist(
            slice(page.index, None), validate=False
        )
        if len(pages) == 1:
            shape = page.shape
            axes = page.axes
        else:
            shape = (len(pages), *page.shape)
            axes = 'I' + page.axes
        series.insert(
            0,
            TiffPageSeries(
                pages, shape, page.dtype, axes, name='frames', kind='eer'
            ),
        )
        return series


class AvsSeriesParser(SeriesParser):
    """Return pyramidal image series in AVS file."""

    kind = 'avs'

    def parse(self, tiff: TiffFile) -> list[TiffPageSeries] | None:
        from .generic import GenericSeriesParser

        series = GenericSeriesParser().parse(tiff)
        if series is None:
            return None
        if len(series) != 3:
            logger().warning(
                f'{tiff!r} AVS series expected 3 series, '
                f'got {len(series)}'
            )
        s = series[0]
        s.kind = 'avs'
        if s.axes[0] == 'I':
            s._set_dimensions(s.shape, 'Z' + s.axes[1:], None, True)
        if s.is_pyramidal:
            s.name = 'Baseline'
        if len(series) == 3:
            series[1].name = 'Map'
            series[1].kind = 'avs'
            series[2].name = 'Macro'
            series[2].kind = 'avs'
        tiff.is_uniform = False
        return series


class SisSeriesParser(SeriesParser):
    """Return image series in Olympus SIS file."""

    kind = 'sis'

    def parse(self, tiff: TiffFile) -> list[TiffPageSeries] | None:
        from ..tifffile import TiffPageSeries

        meta = tiff.sis_metadata
        if meta is None:
            return None
        pages = tiff.pages._getlist(validate=False)
        page = pages[0]
        lenpages = len(pages)

        if 'shape' in meta and 'axes' in meta:
            shape = meta['shape'] + page.shape
            axes = meta['axes'] + page.axes
        else:
            shape = (lenpages, *page.shape)
            axes = 'I' + page.axes
        tiff.is_uniform = True
        return [
            TiffPageSeries(pages, shape, page.dtype, axes, kind='sis')
        ]
