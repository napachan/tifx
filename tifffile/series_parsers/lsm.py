# series_parsers/lsm.py

"""Zeiss LSM series parser."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from ..series import SeriesParser
from ..utils import product

if TYPE_CHECKING:
    from ..tifffile import TiffFile, TiffPageSeries


class LsmSeriesParser(SeriesParser):
    """Return main and thumbnail series in LSM file."""

    kind = 'lsm'

    def parse(self, tiff: TiffFile) -> list[TiffPageSeries] | None:
        from ..tifffile import TIFF, TiffPage, TiffPageSeries

        lsmi = tiff.lsm_metadata
        if lsmi is None:
            return None
        axes = TIFF.CZ_LSMINFO_SCANTYPE[lsmi['ScanType']]
        if tiff.pages.first.planarconfig == 1:
            axes = axes.replace('C', '').replace('X', 'XC')
        elif tiff.pages.first.planarconfig == 2:
            # keep axis for `get_shape(False)`
            pass
        elif tiff.pages.first.samplesperpixel == 1:
            axes = axes.replace('C', '')
        if lsmi.get('DimensionP', 0) > 0:
            axes = 'P' + axes
        if lsmi.get('DimensionM', 0) > 0:
            axes = 'M' + axes
        shape = tuple(
            int(lsmi[TIFF.CZ_LSMINFO_DIMENSIONS[i]]) for i in axes
        )

        name = lsmi.get('Name', '')
        pages = tiff.pages._getlist(slice(0, None, 2), validate=False)
        dtype = pages[0].dtype
        series: list[TiffPageSeries] = [
            TiffPageSeries(pages, shape, dtype, axes, name=name, kind='lsm')
        ]

        page = cast(TiffPage, tiff.pages[1])
        if page.is_reduced:
            pages = tiff.pages._getlist(slice(1, None, 2), validate=False)
            dtype = page.dtype
            cp = 1
            i = 0
            while cp < len(pages) and i < len(shape) - 2:
                cp *= shape[i]
                i += 1
            shape = shape[:i] + page.shape
            axes = axes[:i] + page.axes
            series.append(
                TiffPageSeries(
                    pages, shape, dtype, axes, name=name, kind='lsm'
                )
            )

        tiff.is_uniform = False
        return series
