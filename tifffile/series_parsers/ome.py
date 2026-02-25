# series_parsers/ome.py

"""OME-TIFF series parser."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, cast

import numpy

from ..series import SeriesParser
from ..utils import logger, squeeze_axes

if TYPE_CHECKING:
    from typing import Any

    from ..tifffile import TiffFile, TiffPage, TiffPageSeries

    NDArray = numpy.ndarray[Any, Any]


class OmeSeriesParser(SeriesParser):
    """Return image series in OME-TIFF file(s)."""

    kind = 'ome'

    def on_failure(self, tiff: TiffFile) -> bool:
        if tiff.is_imagej:
            tiff.pages._clear()
            return True
        return False

    def parse(self, tiff: TiffFile) -> list[TiffPageSeries] | None:
        # xml.etree found to be faster than lxml
        from xml.etree import ElementTree

        from ..tifffile import (
            TIFF,
            TiffFile as TiffFileClass,
            TiffFrame,
            TiffPage,
            TiffPageSeries,
        )

        omexml = tiff.ome_metadata
        if omexml is None:
            return None
        try:
            root = ElementTree.fromstring(omexml)
        except ElementTree.ParseError as exc:
            logger().error(f'{tiff!r} OME series raised {exc!r:.128}')
            return None

        keyframe: TiffPage
        ifds: list[TiffPage | Any | None]
        size: int = -1

        def load_pages(t: TiffFile, /) -> None:
            t.pages.cache = True
            t.pages.useframes = True
            t.pages.set_keyframe(0)
            t.pages._load(None)

        load_pages(tiff)

        root_uuid = root.attrib.get('UUID', None)
        tiff._files = {root_uuid: tiff}
        dirname = tiff._fh.dirname
        files_missing = 0
        moduloref: list[str | None] = []
        modulo: dict[str, dict[str, tuple[str, int]]] = {}
        series: list[TiffPageSeries] = []
        for element in root:
            if element.tag.endswith('BinaryOnly'):
                logger().debug(
                    f'{tiff!r} OME series is BinaryOnly, '
                    'not an OME-TIFF master file'
                )
                break
            if element.tag.endswith('StructuredAnnotations'):
                for annot in element:
                    if not annot.attrib.get('Namespace', '').endswith(
                        'modulo'
                    ):
                        continue
                    modulo[annot.attrib['ID']] = mod = {}
                    for value in annot:
                        for modulo_ns in value:
                            for along in modulo_ns:
                                if not along.tag[:-1].endswith('Along'):
                                    continue
                                axis = along.tag[-1]
                                newaxis = along.attrib.get('Type', 'other')
                                newaxis = TIFF.AXES_CODES[newaxis]
                                if 'Start' in along.attrib:
                                    step = float(
                                        along.attrib.get('Step', 1)
                                    )
                                    start = float(along.attrib['Start'])
                                    stop = float(
                                        along.attrib['End']
                                    ) + step
                                    labels = len(
                                        numpy.arange(start, stop, step)
                                    )
                                else:
                                    labels = len(
                                        [
                                            label
                                            for label in along
                                            if label.tag.endswith('Label')
                                        ]
                                    )
                                mod[axis] = (newaxis, labels)

            if not element.tag.endswith('Image'):
                continue

            for annot in element:
                if annot.tag.endswith('AnnotationRef'):
                    annotationref = annot.attrib['ID']
                    break
            else:
                annotationref = None

            attr = element.attrib
            name = attr.get('Name', None)

            for pixels in element:
                if not pixels.tag.endswith('Pixels'):
                    continue
                attr = pixels.attrib
                axes = ''.join(reversed(attr['DimensionOrder']))
                shape = [int(attr['Size' + ax]) for ax in axes]
                ifds = []
                spp = 1  # samples per pixel
                first = True

                for data in pixels:
                    if data.tag.endswith('Channel'):
                        attr = data.attrib
                        if first:
                            first = False
                            spp = int(attr.get('SamplesPerPixel', spp))
                            if spp > 1:
                                shape = [
                                    shape[i] // spp
                                    if ax == 'C'
                                    else shape[i]
                                    for i, ax in enumerate(axes)
                                ]
                        elif int(attr.get('SamplesPerPixel', 1)) != spp:
                            msg = (
                                'OME series cannot handle differing '
                                'SamplesPerPixel'
                            )
                            raise ValueError(msg)
                        continue

                    if not data.tag.endswith('TiffData'):
                        continue

                    attr = data.attrib
                    ifd_index = int(attr.get('IFD', 0))
                    num = int(
                        attr.get('NumPlanes', 1 if 'IFD' in attr else 0)
                    )
                    num = int(attr.get('PlaneCount', num))
                    idxs = [
                        int(attr.get('First' + ax, 0)) for ax in axes[:-2]
                    ]
                    try:
                        idx = int(
                            numpy.ravel_multi_index(idxs, shape[:-2])
                        )
                    except ValueError as exc:
                        logger().warning(
                            f'{tiff!r} '
                            'OME series contains invalid TiffData index, '
                            f'raised {exc!r:.128}',
                        )
                        continue
                    for uuid in data:
                        if not uuid.tag.endswith('UUID'):
                            continue
                        if (
                            root_uuid is None
                            and uuid.text is not None
                            and (
                                uuid.attrib.get('FileName', '').lower()
                                == tiff.filename.lower()
                            )
                        ):
                            root_uuid = uuid.text
                            tiff._files[root_uuid] = tiff._files[None]
                            del tiff._files[None]
                        elif uuid.text not in tiff._files:
                            if not tiff._multifile:
                                return []
                            filename = uuid.attrib['FileName']
                            try:
                                if not tiff.filehandle.is_file:
                                    raise ValueError
                                t = TiffFileClass(
                                    os.path.join(dirname, filename),
                                    _parent=tiff,
                                )
                                load_pages(t)
                            except (
                                OSError,
                                FileNotFoundError,
                                ValueError,
                            ) as exc:
                                if files_missing == 0:
                                    logger().warning(
                                        f'{tiff!r} OME series failed to '
                                        f'read {filename!r}, raised '
                                        f'{exc!r:.128}. '
                                        'Missing data are zeroed'
                                    )
                                files_missing += 1
                                if num:
                                    size = num
                                elif size == -1:
                                    msg = (
                                        'OME series missing '
                                        'NumPlanes or PlaneCount'
                                    )
                                    raise ValueError(msg) from exc
                                ifds.extend(
                                    [None] * (size + idx - len(ifds))
                                )
                                break
                            tiff._files[uuid.text] = t
                            t.close()
                        pages = tiff._files[uuid.text].pages
                        try:
                            size = num if num else len(pages)
                            ifds.extend(
                                [None] * (size + idx - len(ifds))
                            )
                            for i in range(size):
                                ifds[idx + i] = pages[ifd_index + i]
                        except IndexError as exc:
                            logger().warning(
                                f'{tiff!r} '
                                'OME series contains index out of range, '
                                f'raised {exc!r:.128}'
                            )
                        break
                    else:
                        pages = tiff.pages
                        try:
                            size = num if num else len(pages)
                            ifds.extend(
                                [None] * (size + idx - len(ifds))
                            )
                            for i in range(size):
                                ifds[idx + i] = pages[ifd_index + i]
                        except IndexError as exc:
                            logger().warning(
                                f'{tiff!r} '
                                'OME series contains index out of range, '
                                f'raised {exc!r:.128}'
                            )

                if not ifds or all(i is None for i in ifds):
                    continue

                # find a keyframe
                for ifd in ifds:
                    if ifd is not None and ifd == ifd.keyframe:
                        keyframe = cast(TiffPage, ifd)
                        break
                else:
                    for i, ifd in enumerate(ifds):
                        if ifd is not None:
                            isclosed = ifd.parent.filehandle.closed
                            if isclosed:
                                ifd.parent.filehandle.open()
                            ifd.parent.pages.set_keyframe(ifd.index)
                            keyframe = cast(
                                TiffPage, ifd.parent.pages[ifd.index]
                            )
                            ifds[i] = keyframe
                            if isclosed:
                                keyframe.parent.filehandle.close()
                            break

                # does the series spawn multiple files
                multifile = False
                for ifd in ifds:
                    if ifd and ifd.parent != keyframe.parent:
                        multifile = True
                        break

                if spp > 1:
                    if keyframe.planarconfig == 1:
                        shape += [spp]
                        axes += 'S'
                    else:
                        shape = [*shape[:-2], spp, *shape[-2:]]
                        axes = axes[:-2] + 'S' + axes[-2:]
                if 'S' not in axes:
                    shape += [1]
                    axes += 'S'

                size = max(product(shape) // keyframe.size, 1)
                if size < len(ifds):
                    logger().warning(
                        f'{tiff!r} '
                        f'OME series expected {size} frames, '
                        f'got {len(ifds)}'
                    )
                    ifds = ifds[:size]
                elif size > len(ifds):
                    logger().warning(
                        f'{tiff!r} '
                        f'OME series is missing {size - len(ifds)} frames.'
                        ' Missing data are zeroed'
                    )
                    ifds.extend([None] * (size - len(ifds)))

                squeezed = squeeze_axes(shape, axes)[0]
                if keyframe.shape != tuple(
                    squeezed[-len(keyframe.shape) :]
                ):
                    logger().warning(
                        f'{tiff!r} OME series cannot handle discontiguous '
                        f'storage ({keyframe.shape} != '
                        f'{tuple(squeezed[-len(keyframe.shape) :])})',
                    )
                    del ifds
                    continue

                keyframes: dict[str, TiffPage] = {
                    keyframe.parent.filehandle.name: keyframe
                }
                for i, page in enumerate(ifds):
                    if page is None:
                        continue
                    fh = page.parent.filehandle
                    if fh.name not in keyframes:
                        if page.keyframe != page:
                            isclosed = fh.closed
                            if isclosed:
                                fh.open()
                            page.parent.pages.set_keyframe(page.index)
                            page = page.parent.pages[  # noqa: PLW2901
                                page.index
                            ]
                            ifds[i] = page
                            if isclosed:
                                fh.close()
                        keyframes[fh.name] = cast(TiffPage, page)
                    if page.keyframe != page:
                        page.keyframe = keyframes[fh.name]

                moduloref.append(annotationref)
                series.append(
                    TiffPageSeries(
                        ifds,
                        shape,
                        keyframe.dtype,
                        axes,
                        parent=tiff,
                        name=name,
                        multifile=multifile,
                        kind='ome',
                    )
                )
                del ifds

        if files_missing > 1:
            logger().warning(
                f'{tiff!r} OME series failed to read {files_missing} files'
            )

        # apply modulo according to AnnotationRef
        for aseries, annotationref in zip(
            series, moduloref, strict=True
        ):
            if annotationref not in modulo:
                continue
            shape = list(aseries.get_shape(squeeze=False))
            axes = aseries.get_axes(squeeze=False)
            for axis, (newaxis, size) in modulo[annotationref].items():
                i = axes.index(axis)
                if shape[i] == size:
                    axes = axes.replace(axis, newaxis, 1)
                else:
                    shape[i] //= size
                    shape.insert(i + 1, size)
                    axes = axes.replace(axis, axis + newaxis, 1)
            aseries._set_dimensions(shape, axes, None)

        # pyramids
        for aseries in series:
            keyframe = aseries.keyframe
            if keyframe.subifds is None:
                continue
            if len(tiff._files) > 1:
                logger().warning(
                    f'{tiff!r} OME series cannot read multi-file pyramids'
                )
                break
            for level in range(len(keyframe.subifds)):
                found_keyframe = False
                ifds = []
                for page in aseries.pages:
                    if (
                        page is None
                        or page.subifds is None
                        or page.subifds[level] < 8
                    ):
                        ifds.append(None)
                        continue
                    page.parent.filehandle.seek(page.subifds[level])
                    if page.keyframe == page:
                        ifd = keyframe = TiffPage(
                            tiff, (page.index, level + 1)
                        )
                        found_keyframe = True
                    elif not found_keyframe:
                        msg = 'no keyframe found'
                        raise RuntimeError(msg)
                    else:
                        ifd = TiffFrame(
                            tiff,
                            (page.index, level + 1),
                            keyframe=keyframe,
                        )
                    ifds.append(ifd)
                if all(ifd_or_none is None for ifd_or_none in ifds):
                    logger().warning(
                        f'{tiff!r} OME series level {level + 1} is empty'
                    )
                    break
                shape = list(aseries.get_shape(squeeze=False))
                axes = aseries.get_axes(squeeze=False)
                for i, ax in enumerate(axes):
                    if ax == 'X':
                        shape[i] = keyframe.imagewidth
                    elif ax == 'Y':
                        shape[i] = keyframe.imagelength
                aseries.levels.append(
                    TiffPageSeries(
                        ifds,
                        tuple(shape),
                        keyframe.dtype,
                        axes,
                        parent=tiff,
                        name=f'level {level + 1}',
                        kind='ome',
                    )
                )

        tiff.is_uniform = (
            len(series) == 1 and len(series[0].levels) == 1
        )

        return series


def product(seq: Any) -> int:
    """Product of sequence of numbers."""
    from ..utils import product as _product

    return _product(seq)
