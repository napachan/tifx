# series_parsers/imagej.py

"""ImageJ series parser."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..series import SeriesParser
from ..utils import logger, product

if TYPE_CHECKING:
    from ..tifffile import TiffFile, TiffPageSeries


class ImageJSeriesParser(SeriesParser):
    """Return image series in ImageJ file."""

    kind = 'imagej'

    def parse(self, tiff: TiffFile) -> list[TiffPageSeries] | None:
        from ..tifffile import TiffPageSeries

        # ImageJ's dimension order is TZCYXS
        meta = tiff.imagej_metadata
        if meta is None:
            return None

        pages = tiff.pages
        pages.useframes = True
        pages.set_keyframe(0)
        page = tiff.pages.first

        order = meta.get('order', 'czt').lower()
        frames = meta.get('frames', 1)
        slices = meta.get('slices', 1)
        channels = meta.get('channels', 1)
        images = meta.get('images', 1)  # not reliable

        if images < 1 or frames < 1 or slices < 1 or channels < 1:
            logger().warning(
                f'{tiff!r} ImageJ series metadata invalid or corrupted file'
            )
            return None

        if channels == 1:
            images = frames * slices
        elif page.shaped[0] > 1 and page.shaped[0] == channels:
            # Bio-Formats declares separate samples as channels
            images = frames * slices
        elif images == frames * slices and page.shaped[4] == channels:
            # RGB contig samples declared as channel
            channels = 1
        else:
            images = frames * slices * channels

        if images == 1 and pages.is_multipage:
            images = len(pages)

        nbytes = images * page.nbytes

        # ImageJ virtual hyperstacks store all image metadata in the first
        # page and image data are stored contiguously before the second
        # page, if any
        if not page.is_final:
            isvirtual = False
        elif page.dataoffsets[0] + nbytes > tiff.filehandle.size:
            logger().error(
                f'{tiff!r} ImageJ series metadata invalid or corrupted file'
            )
            return None
        elif images <= 1:
            isvirtual = True
        elif (
            pages.is_multipage
            and page.dataoffsets[0] + nbytes > pages[1].offset
        ):
            isvirtual = False
        else:
            isvirtual = True

        page_list = [page] if isvirtual else pages[:]

        shape: tuple[int, ...]
        axes: str

        match order:
            case 'czt' | 'default':
                axes = 'TZC'
                shape = (frames, slices, channels)
            case 'ctz':
                axes = 'ZTC'
                shape = (slices, frames, channels)
            case 'zct':
                axes = 'TCZ'
                shape = (frames, channels, slices)
            case 'ztc':
                axes = 'CTZ'
                shape = (channels, frames, slices)
            case 'tcz':
                axes = 'ZCT'
                shape = (slices, channels, frames)
            case 'tzc':
                axes = 'CZT'
                shape = (channels, slices, frames)
            case _:
                axes = 'TZC'
                shape = (frames, slices, channels)
                logger().warning(
                    f'{tiff!r} ImageJ series of unknown order {order!r}'
                )

        remain = images // product(shape)
        if remain > 1:
            logger().debug(
                f'{tiff!r} ImageJ series contains unidentified dimension'
            )
            shape = (remain, *shape)
            axes = 'I' + axes

        if page.shaped[0] > 1:
            # Bio-Formats declares separate samples as channels
            assert axes[-1] == 'C'
            shape = shape[:-1] + page.shape
            axes += page.axes[1:]
        else:
            shape += page.shape
            axes += page.axes

        if 'S' not in axes:
            shape += (1,)
            axes += 'S'

        truncated = (
            isvirtual and not pages.is_multipage and page.nbytes != nbytes
        )

        tiff.is_uniform = True
        return [
            TiffPageSeries(
                page_list,
                shape,
                page.dtype,
                axes,
                kind='imagej',
                truncated=truncated,
            )
        ]
