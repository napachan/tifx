# series_parsers/pathology.py

"""Pathology format series parsers: SVS, SCN, BIF, NDPI, Philips, QPI."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from ..series import SeriesParser
from ..utils import logger, product

if TYPE_CHECKING:
    from typing import Any

    from ..tifffile import TiffFile, TiffPage, TiffPageSeries


class SvsSeriesParser(SeriesParser):
    """Return image series in Aperio SVS file."""

    kind = 'svs'

    def parse(self, tiff: TiffFile) -> list[TiffPageSeries] | None:
        from ..tifffile import TiffPage, TiffPageSeries

        if not tiff.pages.first.is_tiled:
            return None

        series: list[TiffPageSeries] = []
        tiff.pages.cache = True
        tiff.pages.useframes = False
        tiff.pages.set_keyframe(0)
        tiff.pages._load()

        firstpage = tiff.pages.first
        if len(tiff.pages) == 1:
            tiff.is_uniform = False
            return [
                TiffPageSeries(
                    [firstpage],
                    firstpage.shape,
                    firstpage.dtype,
                    firstpage.axes,
                    name='Baseline',
                    kind='svs',
                )
            ]

        page = tiff.pages[1]
        thumbnail = TiffPageSeries(
            [page],
            page.shape,
            page.dtype,
            page.axes,
            name='Thumbnail',
            kind='svs',
        )

        levels: dict[tuple[int, ...], list[Any]] = {
            firstpage.shape: [firstpage]
        }
        index = 2
        while index < len(tiff.pages):
            page = cast(TiffPage, tiff.pages[index])
            if not page.is_tiled or page.is_reduced:
                break
            if page.shape in levels:
                levels[page.shape].append(page)
            else:
                levels[page.shape] = [page]
            index += 1

        zsize = len(levels[firstpage.shape])
        if not all(len(level) == zsize for level in levels.values()):
            logger().warning(
                f'{tiff!r} SVS series focal planes do not match'
            )
            zsize = 1
        baseline = TiffPageSeries(
            levels[firstpage.shape],
            (zsize, *firstpage.shape),
            firstpage.dtype,
            'Z' + firstpage.axes,
            name='Baseline',
            kind='svs',
        )
        for shape, level in levels.items():
            if shape == firstpage.shape:
                continue
            page = level[0]
            baseline.levels.append(
                TiffPageSeries(
                    level,
                    (zsize, *page.shape),
                    page.dtype,
                    'Z' + page.axes,
                    name='Resolution',
                    kind='svs',
                )
            )
        series.append(baseline)
        series.append(thumbnail)

        for _ in range(2):
            if index == len(tiff.pages):
                break
            page = tiff.pages[index]
            assert isinstance(page, TiffPage)
            name = 'Macro' if page.subfiletype == 9 else 'Label'
            series.append(
                TiffPageSeries(
                    [page],
                    page.shape,
                    page.dtype,
                    page.axes,
                    name=name,
                    kind='svs',
                )
            )
            index += 1
        tiff.is_uniform = False
        return series


class ScnSeriesParser(SeriesParser):
    """Return pyramidal image series in Leica SCN file."""

    kind = 'scn'

    def parse(self, tiff: TiffFile) -> list[TiffPageSeries] | None:
        from xml.etree import ElementTree

        from ..tifffile import TiffPageSeries

        scnxml = tiff.pages.first.description
        root = ElementTree.fromstring(scnxml)

        series: list[TiffPageSeries] = []
        tiff.pages.cache = True
        tiff.pages.useframes = False
        tiff.pages.set_keyframe(0)
        tiff.pages._load()

        for collection in root:
            if not collection.tag.endswith('collection'):
                continue
            for image in collection:
                if not image.tag.endswith('image'):
                    continue
                name = image.attrib.get('name', 'Unknown')
                for pixels in image:
                    if not pixels.tag.endswith('pixels'):
                        continue
                    resolutions: dict[int, dict[str, Any]] = {}
                    for dimension in pixels:
                        if not dimension.tag.endswith('dimension'):
                            continue
                        if int(image.attrib.get('sizeZ', 1)) > 1:
                            msg = (
                                'SCN series: Z-Stacks not supported. '
                                'Please submit a sample file.'
                            )
                            raise NotImplementedError(msg)
                        sizex = int(dimension.attrib['sizeX'])
                        sizey = int(dimension.attrib['sizeY'])
                        c = int(dimension.attrib.get('c', 0))
                        z = int(dimension.attrib.get('z', 0))
                        r = int(dimension.attrib.get('r', 0))
                        ifd = int(dimension.attrib['ifd'])
                        if r in resolutions:
                            level = resolutions[r]
                            level['channels'] = max(
                                level['channels'], c
                            )
                            level['sizez'] = max(level['sizez'], z)
                            level['ifds'][(c, z)] = ifd
                        else:
                            resolutions[r] = {
                                'size': [sizey, sizex],
                                'channels': c,
                                'sizez': z,
                                'ifds': {(c, z): ifd},
                            }
                    if not resolutions:
                        continue
                    levels = []
                    for _r, level in sorted(resolutions.items()):
                        shape: tuple[int, ...] = (
                            level['channels'] + 1,
                            level['sizez'] + 1,
                        )
                        axes = 'CZ'

                        ifds: list[Any | None] = [None] * product(shape)
                        for (c, z), ifd in sorted(
                            level['ifds'].items()
                        ):
                            ifds[c * shape[1] + z] = tiff.pages[ifd]

                        assert ifds[0] is not None
                        axes += ifds[0].axes
                        shape += ifds[0].shape
                        dtype = ifds[0].dtype

                        levels.append(
                            TiffPageSeries(
                                ifds,
                                shape,
                                dtype,
                                axes,
                                parent=tiff,
                                name=name,
                                kind='scn',
                            )
                        )
                    levels[0].levels.extend(levels[1:])
                    series.append(levels[0])

        tiff.is_uniform = False
        return series


class BifSeriesParser(SeriesParser):
    """Return image series in Ventana/Roche BIF file."""

    kind = 'bif'

    def parse(self, tiff: TiffFile) -> list[TiffPageSeries] | None:
        from ..tifffile import TiffPage, TiffPageSeries

        series: list[TiffPageSeries] = []
        baseline: TiffPageSeries | None = None
        tiff.pages.cache = True
        tiff.pages.useframes = False
        tiff.pages.set_keyframe(0)
        tiff.pages._load()

        for page in tiff.pages:
            page = cast(TiffPage, page)
            if page.description[:5] == 'Label':
                series.append(
                    TiffPageSeries(
                        [page],
                        page.shape,
                        page.dtype,
                        page.axes,
                        name='Label',
                        kind='bif',
                    )
                )
            elif (
                page.description == 'Thumbnail'
                or page.description[:11] == 'Probability'
            ):
                series.append(
                    TiffPageSeries(
                        [page],
                        page.shape,
                        page.dtype,
                        page.axes,
                        name='Thumbnail',
                        kind='bif',
                    )
                )
            elif 'level' not in page.description:
                series.append(
                    TiffPageSeries(
                        [page],
                        page.shape,
                        page.dtype,
                        page.axes,
                        name='Unknown',
                        kind='bif',
                    )
                )
            elif baseline is None:
                baseline = TiffPageSeries(
                    [page],
                    page.shape,
                    page.dtype,
                    page.axes,
                    name='Baseline',
                    kind='bif',
                )
                series.insert(0, baseline)
            else:
                baseline.levels.append(
                    TiffPageSeries(
                        [page],
                        page.shape,
                        page.dtype,
                        page.axes,
                        name='Resolution',
                        kind='bif',
                    )
                )

        logger().warning(f'{tiff!r} BIF series tiles are not stitched')
        tiff.is_uniform = False
        return series


class NdpiSeriesParser(SeriesParser):
    """Return pyramidal image series in NDPI file."""

    kind = 'ndpi'

    def parse(self, tiff: TiffFile) -> list[TiffPageSeries] | None:
        from .generic import GenericSeriesParser

        series = GenericSeriesParser().parse(tiff)
        if series is None:
            return None
        for s in series:
            s.kind = 'ndpi'
            if s.axes[0] == 'I':
                s._set_dimensions(s.shape, 'Z' + s.axes[1:], None, True)
            if s.is_pyramidal:
                name = s.keyframe.tags.valueof(65427)
                s.name = 'Baseline' if name is None else name
                continue
            mag = s.keyframe.tags.valueof(65421)
            if mag is not None:
                if mag == -1.0:
                    s.name = 'Macro'
                elif mag == -2.0:
                    s.name = 'Map'
        tiff.is_uniform = False
        return series


class PhilipsSeriesParser(SeriesParser):
    """Return pyramidal image series in Philips DP file."""

    kind = 'philips'

    def parse(self, tiff: TiffFile) -> list[TiffPageSeries] | None:
        from xml.etree import ElementTree

        from ..tifffile import TiffPage, TiffPageSeries

        series: list[TiffPageSeries] = []
        pages = tiff.pages
        pages.cache = False
        pages.useframes = False
        pages.set_keyframe(0)
        pages._load()

        meta = tiff.philips_metadata
        assert meta is not None

        try:
            tree = ElementTree.fromstring(meta)
        except ElementTree.ParseError as exc:
            logger().error(
                f'{tiff!r} Philips series raised {exc!r:.128}'
            )
            return None

        pixel_spacing = [
            tuple(
                float(v) for v in elem.text.replace('"', '').split()
            )
            for elem in tree.findall(
                './/*'
                '/DataObject[@ObjectType="PixelDataRepresentation"]'
                '/Attribute[@Name="DICOM_PIXEL_SPACING"]'
            )
            if elem.text is not None
        ]
        if len(pixel_spacing) < 2:
            logger().error(
                f'{tiff!r} Philips series {len(pixel_spacing)=} < 2'
            )
            return None

        series_dict: dict[str, list[TiffPage]] = {}
        series_dict['Level'] = []
        series_dict['Other'] = []
        for page in pages:
            assert isinstance(page, TiffPage)
            if page.description.startswith('Macro'):
                series_dict['Macro'] = [page]
            elif page.description.startswith('Label'):
                series_dict['Label'] = [page]
            elif not page.is_tiled:
                series_dict['Other'].append(page)
            else:
                series_dict['Level'].append(page)

        levels = series_dict.pop('Level')
        if len(levels) != len(pixel_spacing):
            logger().error(
                f'{tiff!r} Philips series '
                f'{len(levels)=} != {len(pixel_spacing)=}'
            )
            return None

        imagewidth0 = levels[0].imagewidth
        imagelength0 = levels[0].imagelength
        h0, w0 = pixel_spacing[0]
        for serie, (h, w) in zip(
            levels[1:], pixel_spacing[1:], strict=True
        ):
            page = serie.keyframe
            imagewidth = imagewidth0 // round(w / w0)
            imagelength = imagelength0 // round(h / h0)

            if page.imagewidth - page.tilewidth >= imagewidth:
                logger().warning(
                    f'{tiff!r} Philips series {page.index=} '
                    f'{page.imagewidth=}-{page.tilewidth=} '
                    f'>= {imagewidth=}'
                )
                page.imagewidth -= page.tilewidth - 1
            elif page.imagewidth < imagewidth:
                logger().warning(
                    f'{tiff!r} Philips series {page.index=} '
                    f'{page.imagewidth=} < {imagewidth=}'
                )
            else:
                page.imagewidth = imagewidth
            imagewidth = page.imagewidth

            if page.imagelength - page.tilelength >= imagelength:
                logger().warning(
                    f'{tiff!r} Philips series {page.index=} '
                    f'{page.imagelength=}-{page.tilelength=} '
                    f'>= {imagelength=}'
                )
                page.imagelength -= page.tilelength - 1
            else:
                page.imagelength = imagelength
            imagelength = page.imagelength

            if page.shaped[-1] > 1:
                page.shape = (imagelength, imagewidth, page.shape[-1])
            elif page.shaped[0] > 1:
                page.shape = (page.shape[0], imagelength, imagewidth)
            else:
                page.shape = (imagelength, imagewidth)
            page.shaped = (
                *page.shaped[:2],
                imagelength,
                imagewidth,
                *page.shaped[-1:],
            )

        series = [
            TiffPageSeries(
                [levels[0]], name='Baseline', kind='philips'
            )
        ]
        for i, page in enumerate(levels[1:]):
            series[0].levels.append(
                TiffPageSeries(
                    [page], name=f'Level{i + 1}', kind='philips'
                )
            )
        for key, value in series_dict.items():
            for page in value:
                series.append(
                    TiffPageSeries([page], name=key, kind='philips')
                )

        tiff.is_uniform = False
        return series


class QpiSeriesParser(SeriesParser):
    """Return image series in PerkinElmer QPI file."""

    kind = 'qpi'

    def parse(self, tiff: TiffFile) -> list[TiffPageSeries] | None:
        from ..tifffile import TiffPageSeries

        series: list[TiffPageSeries] = []
        pages = tiff.pages
        pages.cache = True
        pages.useframes = False
        pages.set_keyframe(0)
        pages._load()
        page0 = tiff.pages.first

        ifds: list[Any] = []
        index = 0
        axes = 'C' + page0.axes
        dtype = page0.dtype
        pshape = page0.shape
        while index < len(pages):
            page = pages[index]
            if page.shape != pshape:
                break
            ifds.append(page)
            index += 1
        shape: tuple[int, ...] = (len(ifds), *pshape)
        series.append(
            TiffPageSeries(
                ifds, shape, dtype, axes, name='Baseline', kind='qpi'
            )
        )

        if index < len(pages):
            page = pages[index]
            series.append(
                TiffPageSeries(
                    [page],
                    page.shape,
                    page.dtype,
                    page.axes,
                    name='Thumbnail',
                    kind='qpi',
                )
            )
            index += 1

        if page0.is_tiled:
            while index < len(pages):
                pshape = (pshape[0] // 2, pshape[1] // 2, *pshape[2:])
                ifds = []
                while index < len(pages):
                    page = pages[index]
                    if page.shape != pshape:
                        break
                    ifds.append(page)
                    index += 1
                if len(ifds) != len(series[0].pages):
                    break
                shape = (len(ifds), *pshape)
                series[0].levels.append(
                    TiffPageSeries(
                        ifds,
                        shape,
                        dtype,
                        axes,
                        name='Resolution',
                        kind='qpi',
                    )
                )

        if series[0].is_pyramidal and index < len(pages):
            page = pages[index]
            series.append(
                TiffPageSeries(
                    [page],
                    page.shape,
                    page.dtype,
                    page.axes,
                    name='Macro',
                    kind='qpi',
                )
            )
            index += 1
            if index < len(pages):
                page = pages[index]
                series.append(
                    TiffPageSeries(
                        [page],
                        page.shape,
                        page.dtype,
                        page.axes,
                        name='Label',
                        kind='qpi',
                    )
                )

        tiff.is_uniform = False
        return series
