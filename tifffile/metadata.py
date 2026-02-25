# metadata.py

"""Metadata parsers for tifffile."""

from __future__ import annotations

import contextlib
import json
import logging
import math
import os
import re
import struct
import warnings
from collections.abc import Sequence
from datetime import datetime as DateTime
from functools import cached_property
from typing import IO, TYPE_CHECKING, final

import numpy

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator
    from types import TracebackType
    from typing import Any, Literal, Self, TypeAlias

    from numpy.typing import DTypeLike, NDArray

    from .fileio import FileHandle
    from .tifffile import TiffTagRegistry

    ByteOrder: TypeAlias = Literal['>', '<']

try:
    import imagecodecs
except ImportError:
    try:
        from . import _imagecodecs as imagecodecs  # type: ignore[no-redef]
    except ImportError:
        import _imagecodecs as imagecodecs  # type: ignore[no-redef]

from .enums import COMPRESSION
from .utils import (
    TiffFileError,
    asbool,
    astype,
    bytes2str,
    bytestr,
    indent,
    logger,
    matlabstr2py,
    pformat,
    product,
    recarray2dict,
    sequence,
    stripnull,
    strptime,
    xml2dict,
)

# TIFF singleton is lazily imported to avoid circular imports
# (tifffile.py imports from metadata.py, metadata.py needs TIFF from tifffile.py)
_TIFF_SINGLETON = None


def _get_tiff():
    global _TIFF_SINGLETON
    if _TIFF_SINGLETON is None:
        from .tifffile import TIFF
        _TIFF_SINGLETON = TIFF
    return _TIFF_SINGLETON


class _TIFFProxy:
    """Proxy that lazily resolves to the TIFF singleton."""
    def __getattr__(self, name):
        return getattr(_get_tiff(), name)


TIFF = _TIFFProxy()


def _get_version():
    """Lazy import of __version__ to avoid circular imports."""
    from .tifffile import __version__
    return __version__


class OmeXmlError(Exception):
    """Exception to indicate invalid OME-XML or unsupported cases."""


@final
class OmeXml:
    """Create OME-TIFF XML metadata.

    Parameters:
        **metadata:
            Additional OME-XML attributes or elements to be stored.

            Creator:
                Name of creating application. The default is 'tifffile'.
            UUID:
                Unique identifier.

    Examples:
        >>> omexml = OmeXml()
        >>> omexml.addimage(
        ...     dtype='uint16',
        ...     shape=(32, 256, 256),
        ...     storedshape=(32, 1, 1, 256, 256, 1),
        ...     axes='CYX',
        ...     Name='First Image',
        ...     PhysicalSizeX=2.0,
        ...     MapAnnotation={'key': 'value'},
        ...     Dataset={'Name': 'FirstDataset'},
        ... )
        >>> xml = omexml.tostring()
        >>> xml
        '<OME ...<Image ID="Image:0" Name="First Image">...</Image>...</OME>'
        >>> OmeXml.validate(xml)
        True

    """

    images: list[str]
    """OME-XML Image elements."""

    annotations: list[str]
    """OME-XML Annotation elements."""

    datasets: list[str]
    """OME-XML Dataset elements."""

    _xml: str
    _ifd: int

    def __init__(self, **metadata: Any) -> None:
        metadata = metadata.get('OME', metadata)

        self._ifd = 0
        self.images = []
        self.annotations = []
        self.datasets = []
        # TODO: parse other OME elements from metadata
        #   Project
        #   Folder
        #   Experiment
        #   Plate
        #   Screen
        #   Experimenter
        #   ExperimenterGroup
        #   Instrument
        #   ROI
        if 'UUID' in metadata:
            uuid = metadata['UUID'].split(':')[-1]
        else:
            from uuid import uuid1

            uuid = str(uuid1())
        creator = OmeXml._attribute(
            metadata,
            'Creator',
            default=f'tifffile.py {_get_version()}'
        )
        schema = 'http://www.openmicroscopy.org/Schemas/OME/2016-06'
        self._xml = (
            '{declaration}'
            f'<OME xmlns="{schema}" '
            'xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" '
            f'xsi:schemaLocation="{schema} {schema}/ome.xsd" '
            f'UUID="urn:uuid:{uuid}"{creator}>'
            '{datasets}'
            '{images}'
            '{annotations}'
            '</OME>'
        )

    def addimage(
        self,
        dtype: DTypeLike | None,
        shape: Sequence[int],
        storedshape: tuple[int, int, int, int, int, int],
        *,
        axes: str | None = None,
        **metadata: Any,
    ) -> None:
        """Add image to OME-XML.

        The OME model can handle up to 9 dimensional images for selected
        axes orders. Refer to the OME-XML specification for details.
        Non-TZCYXS (modulo) dimensions must be after a TZC dimension or
        require an unused TZC dimension.

        Parameters:
            dtype:
                Data type of image array.
            shape:
                Shape of image array.
            storedshape:
                Normalized shape describing how image array is stored in
                TIFF file as (pages, separate_samples, depth, length, width,
                contig_samples).
            axes:
                Character codes for dimensions in `shape`.
                By default, `axes` is determined from the DimensionOrder
                metadata attribute or matched to the `shape` in reverse order
                of TZC(S)YX(S) based on `storedshape`.
                The following codes are supported: 'S' sample, 'X' width,
                'Y' length, 'Z' depth, 'C' channel, 'T' time, 'A' angle,
                'P' phase, 'R' tile, 'H' lifetime, 'E' lambda, 'Q' other.
            **metadata:
                Additional OME-XML attributes or elements to be stored.

                Image/Pixels:
                    Name, Description,
                    DimensionOrder, TypeDescription,
                    PhysicalSizeX, PhysicalSizeXUnit,
                    PhysicalSizeY, PhysicalSizeYUnit,
                    PhysicalSizeZ, PhysicalSizeZUnit,
                    TimeIncrement, TimeIncrementUnit,
                    StructuredAnnotations, BooleanAnnotation, DoubleAnnotation,
                    LongAnnotation, CommentAnnotation, MapAnnotation,
                    Dataset
                Per Plane:
                    DeltaT, DeltaTUnit,
                    ExposureTime, ExposureTimeUnit,
                    PositionX, PositionXUnit,
                    PositionY, PositionYUnit,
                    PositionZ, PositionZUnit.
                Per Channel:
                    Name, AcquisitionMode, Color, ContrastMethod,
                    EmissionWavelength, EmissionWavelengthUnit,
                    ExcitationWavelength, ExcitationWavelengthUnit,
                    Fluor, IlluminationType, NDFilter,
                    PinholeSize, PinholeSizeUnit, PockelCellSetting.

        Raises:
            OmeXmlError: Image format not supported.

        """
        index = len(self.images)
        annotation_refs = []

        # get Image and Pixels metadata
        metadata = metadata.get('OME', metadata)
        metadata = metadata.get('Image', metadata)
        if isinstance(metadata, (list, tuple)):
            # multiple images
            metadata = metadata[index]
        if 'Pixels' in metadata:
            # merge with Image
            import copy

            metadata = copy.deepcopy(metadata)
            if 'ID' in metadata['Pixels']:
                del metadata['Pixels']['ID']
            metadata.update(metadata['Pixels'])
            del metadata['Pixels']

        try:
            dtype = numpy.dtype(dtype).name
            dtype = {
                'int8': 'int8',
                'int16': 'int16',
                'int32': 'int32',
                'uint8': 'uint8',
                'uint16': 'uint16',
                'uint32': 'uint32',
                'float32': 'float',
                'float64': 'double',
                'complex64': 'complex',
                'complex128': 'double-complex',
                'bool': 'bit',
            }[dtype]
        except KeyError:
            msg = f'data type {dtype!r} not supported'
            raise OmeXmlError(msg) from None

        if metadata.get('Type', dtype) != dtype:
            msg = (
                f'metadata Pixels Type {metadata["Type"]!r} '
                f'does not match array dtype {dtype!r}'
            )
            raise OmeXmlError(msg)

        samples = 1
        planecount, separate, depth, length, width, contig = storedshape
        if depth != 1:
            msg = 'ImageDepth not supported'
            raise OmeXmlError(msg)
        if not (separate == 1 or contig == 1):
            msg = 'invalid stored shape'
            raise ValueError(msg)

        shape = tuple(int(i) for i in shape)
        ndim = len(shape)
        if ndim < 1 or product(shape) <= 0:
            msg = 'empty arrays not supported'
            raise OmeXmlError(msg)

        if axes is None:
            # get axes from shape, stored shape, and DimensionOrder
            if contig != 1 or shape[-3:] == (length, width, 1):
                axes = 'YXS'
                samples = contig
            elif separate != 1 or (
                ndim == 6 and shape[-3:] == (1, length, width)
            ):
                axes = 'SYX'
                samples = separate
            else:
                axes = 'YX'
            if not len(axes) <= ndim <= (6 if 'S' in axes else 5):
                msg = f'{ndim} dimensions not supported'
                raise OmeXmlError(msg)
            hiaxes: str = metadata.get('DimensionOrder', 'XYCZT')[:1:-1]
            axes = hiaxes[(6 if 'S' in axes else 5) - ndim :] + axes
            assert len(axes) == len(shape)

        else:
            # validate axes against shape and stored shape
            axes = axes.upper()
            if len(axes) != len(shape):
                msg = 'axes do not match shape'
                raise ValueError(msg)
            if not (
                axes.endswith(('YX', 'YXS'))
                or (axes.endswith('YXC') and 'S' not in axes)
            ):
                msg = 'dimensions must end with YX or YXS'
                raise OmeXmlError(msg)
            unique = []
            for ax in axes:
                if ax not in 'TZCYXSAPRHEQ':
                    msg = f'dimension {ax!r} not supported'
                    raise OmeXmlError(msg)
                if ax in unique:
                    msg = f'multiple {ax!r} dimensions'
                    raise OmeXmlError(msg)
                unique.append(ax)
            if ndim > (9 if 'S' in axes else 8):
                msg = 'more than 8 dimensions not supported'
                raise OmeXmlError(msg)
            if contig != 1:
                samples = contig
                if ndim < 3:
                    msg = 'dimensions do not match stored shape'
                    raise ValueError(msg)
                if axes[-1] == 'C':
                    # allow C axis instead of S
                    if 'S' in axes:
                        msg = 'invalid axes'
                        raise ValueError(msg)
                    axes = axes.replace('C', 'S')
                elif axes[-1] != 'S':
                    msg = 'axes do not match stored shape'
                    raise ValueError(msg)
                if shape[-1] != contig or shape[-2] != width:
                    msg = 'shape does not match stored shape'
                    raise ValueError(msg)
            elif separate != 1:
                samples = separate
                if ndim < 3:
                    msg = 'dimensions do not match stored shape'
                    raise ValueError(msg)
                if axes[-3] == 'C':
                    # allow C axis instead of S
                    if 'S' in axes:
                        msg = 'invalid axes'
                        raise ValueError(msg)
                    axes = axes.replace('C', 'S')
                elif axes[-3] != 'S':
                    msg = 'axes do not match stored shape'
                    raise ValueError(msg)
                if shape[-3] != separate or shape[-1] != width:
                    msg = 'shape does not match stored shape'
                    raise ValueError(msg)

        if shape[axes.index('X')] != width or shape[axes.index('Y')] != length:
            msg = 'shape does not match stored shape'
            raise ValueError(msg)

        if 'S' in axes:
            hiaxes = axes[: min(axes.index('S'), axes.index('Y'))]
        else:
            hiaxes = axes[: axes.index('Y')]

        if any(ax in 'APRHEQ' for ax in hiaxes):
            # modulo axes
            modulo = {}
            dimorder = ''
            axestype = {
                'A': 'angle',
                'P': 'phase',
                'R': 'tile',
                'H': 'lifetime',
                'E': 'lambda',
                'Q': 'other',
            }
            axestypedescr = metadata.get('TypeDescription', {})
            for i, ax in enumerate(hiaxes):
                if ax in 'APRHEQ':
                    if ax in axestypedescr:
                        typedescr = f'TypeDescription="{axestypedescr[ax]}" '
                    else:
                        typedescr = ''
                    x = hiaxes[i - 1 : i]
                    if x and x in 'TZC':
                        # use previous axis
                        modulo[x] = axestype[ax], shape[i], typedescr
                    else:
                        # use next unused axis
                        for x in 'TZC':
                            if (
                                x not in dimorder
                                and x not in hiaxes
                                and x not in modulo
                            ):
                                modulo[x] = axestype[ax], shape[i], typedescr
                                dimorder += x
                                break
                        else:
                            # TODO: support any order of axes, such as, APRTZC
                            msg = 'more than 3 modulo dimensions'
                            raise OmeXmlError(msg)
                else:
                    dimorder += ax
            hiaxes = dimorder

            # TODO: use user-specified start, stop, step, or labels
            moduloalong = ''.join(
                f'<ModuloAlong{ax} Type="{axtype}" {typedescr}'
                f'Start="0" End="{size - 1}"/>'
                for ax, (axtype, size, typedescr) in modulo.items()
            )
            annotation_refs.append(
                f'<AnnotationRef ID="Annotation:{len(self.annotations)}"/>'
            )
            self.annotations.append(
                f'<XMLAnnotation ID="Annotation:{len(self.annotations)}" '
                'Namespace="openmicroscopy.org/omero/dimension/modulo">'
                '<Value>'
                '<Modulo namespace='
                '"http://www.openmicroscopy.org/Schemas/Additions/2011-09">'
                f'{moduloalong}'
                '</Modulo>'
                '</Value>'
                '</XMLAnnotation>'
            )
        else:
            modulo = {}

        hiaxes = hiaxes[::-1]
        for dimorder in (
            metadata.get('DimensionOrder', 'XYCZT'),
            'XYCZT',
            'XYZCT',
            'XYZTC',
            'XYCTZ',
            'XYTCZ',
            'XYTZC',
        ):
            if hiaxes in dimorder:
                break
        else:
            msg = f'dimension order {axes!r} not supported ({hiaxes=})'
            raise OmeXmlError(msg)

        dimsizes = []
        for ax in dimorder:
            if ax == 'S':
                continue
            size = shape[axes.index(ax)] if ax in axes else 1
            if ax == 'C':
                sizec = size
                size *= samples
            if ax in modulo:
                size *= modulo[ax][1]
            dimsizes.append(size)
        sizes = ''.join(
            f' Size{ax}="{size}"'
            for ax, size in zip(dimorder, dimsizes, strict=True)
        )

        # verify DimensionOrder in metadata is compatible
        if 'DimensionOrder' in metadata:
            omedimorder = metadata['DimensionOrder']
            omedimorder = ''.join(
                ax for ax in omedimorder if dimsizes[dimorder.index(ax)] > 1
            )
            if hiaxes not in omedimorder:
                msg = f'metadata DimensionOrder does not match {axes!r}'
                raise OmeXmlError(msg)

        # verify metadata Size values match shape
        for ax, size in zip(dimorder, dimsizes, strict=True):
            if metadata.get(f'Size{ax}', size) != size:
                msg = f'metadata Size{ax} does not match {shape!r}'
                raise OmeXmlError(msg)

        dimsizes[dimorder.index('C')] //= samples
        if planecount != product(dimsizes[2:]):
            msg = 'shape does not match stored shape'
            raise ValueError(msg)

        plane_list = []
        planeattributes = metadata.get('Plane', '')
        if planeattributes:
            cztorder = tuple(dimorder[2:].index(ax) for ax in 'CZT')
            for p in range(planecount):
                attributes = OmeXml._attributes(
                    planeattributes,
                    p,
                    'DeltaT',
                    'DeltaTUnit',
                    'ExposureTime',
                    'ExposureTimeUnit',
                    'PositionX',
                    'PositionXUnit',
                    'PositionY',
                    'PositionYUnit',
                    'PositionZ',
                    'PositionZUnit',
                )
                unraveled = numpy.unravel_index(p, dimsizes[2:], order='F')
                c, z, t = (int(unraveled[i]) for i in cztorder)
                plane_list.append(
                    f'<Plane TheC="{c}" TheZ="{z}" TheT="{t}"{attributes}/>'
                )
                # TODO: if possible, verify c, z, t match planeattributes
        planes = ''.join(plane_list)

        channel_list = []
        for c in range(sizec):
            lightpath = '<LightPath/>'
            # TODO: use LightPath elements from metadata
            #    'AnnotationRef',
            #    'DichroicRef',
            #    'EmissionFilterRef',
            #    'ExcitationFilterRef'
            attributes = OmeXml._attributes(
                metadata.get('Channel', ''),
                c,
                'Name',
                'AcquisitionMode',
                'Color',
                'ContrastMethod',
                'EmissionWavelength',
                'EmissionWavelengthUnit',
                'ExcitationWavelength',
                'ExcitationWavelengthUnit',
                'Fluor',
                'IlluminationType',
                'NDFilter',
                'PinholeSize',
                'PinholeSizeUnit',
                'PockelCellSetting',
            )
            channel_list.append(
                f'<Channel ID="Channel:{index}:{c}" '
                f'SamplesPerPixel="{samples}"'
                f'{attributes}>'
                f'{lightpath}'
                '</Channel>'
            )
        channels = ''.join(channel_list)

        # TODO: support more Image elements
        elements = OmeXml._elements(metadata, 'AcquisitionDate', 'Description')

        name = OmeXml._attribute(metadata, 'Name', default=f'Image{index}')
        attributes = OmeXml._attributes(
            metadata,
            None,
            'SignificantBits',
            'PhysicalSizeX',
            'PhysicalSizeXUnit',
            'PhysicalSizeY',
            'PhysicalSizeYUnit',
            'PhysicalSizeZ',
            'PhysicalSizeZUnit',
            'TimeIncrement',
            'TimeIncrementUnit',
        )
        if separate > 1 or contig > 1:
            interleaved = 'false' if separate > 1 else 'true'
            interleaved = f' Interleaved="{interleaved}"'
        else:
            interleaved = ''

        self._dataset(
            metadata.get('Dataset', {}), f'<ImageRef ID="Image:{index}"/>'
        )

        self._annotations(
            metadata.get('StructuredAnnotations', metadata), annotation_refs
        )
        annotationref = ''.join(annotation_refs)

        self.images.append(
            f'<Image ID="Image:{index}"{name}>'
            f'{elements}'
            f'<Pixels ID="Pixels:{index}" '
            f'DimensionOrder="{dimorder}" '
            f'Type="{dtype}"'
            f'{sizes}'
            f'{interleaved}'
            f'{attributes}>'
            f'{channels}'
            f'<TiffData IFD="{self._ifd}" PlaneCount="{planecount}"/>'
            f'{planes}'
            '</Pixels>'
            f'{annotationref}'
            '</Image>'
        )
        self._ifd += planecount

    def tostring(self, *, declaration: bool = False) -> str:
        """Return OME-XML string.

        Parameters:
            declaration: Include XML declaration.

        """
        # TODO: support other top-level elements
        datasets = ''.join(self.datasets)
        images = ''.join(self.images)
        annotations = ''.join(self.annotations)
        if annotations:
            annotations = (
                f'<StructuredAnnotations>{annotations}</StructuredAnnotations>'
            )
        if declaration:
            declaration_str = '<?xml version="1.0" encoding="UTF-8"?>'
        else:
            declaration_str = ''
        return self._xml.format(
            declaration=declaration_str,
            images=images,
            annotations=annotations,
            datasets=datasets,
        )

    def __repr__(self) -> str:
        return f'<tifffile.OmeXml @0x{id(self):016X}>'

    def __str__(self) -> str:
        """Return OME-XML string."""
        xml = self.tostring()
        try:
            from lxml import etree

            parser = etree.XMLParser(remove_blank_text=True)
            tree = etree.fromstring(xml, parser)
            xml = etree.tostring(
                tree, encoding='utf-8', pretty_print=True, xml_declaration=True
            ).decode()
        except ImportError:
            pass
        except Exception as exc:
            warnings.warn(
                f'<tifffile.OmeXml.__str__> {exc.__class__.__name__}: {exc}',
                UserWarning,
                stacklevel=2,
            )
        return xml

    @staticmethod
    def _escape(value: object, /) -> str:
        """Return escaped string of value."""
        if not isinstance(value, str):
            value = str(value)
        elif '&amp;' in value or '&gt;' in value or '&lt;' in value:
            return value
        value = value.replace('&', '&amp;')
        value = value.replace('>', '&gt;')
        return value.replace('<', '&lt;')

    @staticmethod
    def _element(
        metadata: dict[str, Any], name: str, default: str | None = None
    ) -> str:
        """Return XML formatted element if name in metadata."""
        value = metadata.get(name, default)
        if value is None:
            return ''
        return f'<{name}>{OmeXml._escape(value)}</{name}>'

    @staticmethod
    def _elements(metadata: dict[str, Any], /, *names: str) -> str:
        """Return XML formatted elements."""
        if not metadata:
            return ''
        elements = (OmeXml._element(metadata, name) for name in names)
        return ''.join(e for e in elements if e)

    @staticmethod
    def _attribute(
        metadata: dict[str, Any],
        name: str,
        /,
        index: int | None = None,
        default: Any = None,
    ) -> str:
        """Return XML formatted attribute if name in metadata."""
        value = metadata.get(name, default)
        if value is None:
            return ''
        if index is not None:
            if isinstance(value, (list, tuple)):
                try:
                    value = value[index]
                except IndexError as exc:
                    msg = f'list index out of range for attribute {name!r}'
                    raise IndexError(msg) from exc
            elif index > 0:
                msg = f'{type(value).__name__!r} is not a list or tuple'
                raise TypeError(msg)
        return f' {name}="{OmeXml._escape(value)}"'

    @staticmethod
    def _attributes(
        metadata: dict[str, Any],
        index_: int | None,
        /,
        *names: str,
    ) -> str:
        """Return XML formatted attributes."""
        if not metadata:
            return ''
        if index_ is None:
            attributes = (OmeXml._attribute(metadata, name) for name in names)
        elif isinstance(metadata, (list, tuple)):
            metadata = metadata[index_]
            attributes = (OmeXml._attribute(metadata, name) for name in names)
        elif isinstance(metadata, dict):
            attributes = (
                OmeXml._attribute(metadata, name, index_) for name in names
            )
        return ''.join(a for a in attributes if a)

    def _dataset(self, metadata: dict[str, Any] | None, imageref: str) -> None:
        """Add Dataset element to self.datasets."""
        index = len(self.datasets)
        if metadata is None:
            # dataset explicitly disabled
            return
        if not metadata and index == 0:
            # no dataset provided yet
            return
        if not metadata:
            # use previous dataset
            index -= 1
            if '<AnnotationRef' in self.datasets[index]:
                self.datasets[index] = self.datasets[index].replace(
                    '<AnnotationRef', f'{imageref}<AnnotationRef'
                )
            else:
                self.datasets[index] = self.datasets[index].replace(
                    '</Dataset>', f'{imageref}</Dataset>'
                )
            return

        # new dataset
        name = metadata.get('Name', '')
        if name:
            name = f' Name="{OmeXml._escape(name)}"'

        description = metadata.get('Description', '')
        if description:
            description = (
                f'<Description>{OmeXml._escape(description)}</Description>'
            )

        annotation_refs: list[str] = []
        self._annotations(metadata, annotation_refs)
        annotationref = ''.join(annotation_refs)

        self.datasets.append(
            f'<Dataset ID="Dataset:{index}"{name}>'
            f'{description}'
            f'{imageref}'
            f'{annotationref}'
            '</Dataset>'
        )

    def _annotations(
        self, metadata: dict[str, Any], annotation_refs: list[str]
    ) -> None:
        """Add annotations to self.annotations and annotation_refs."""
        for item in metadata.items():
            name, annotation_values = item
            if not annotation_values:
                continue
            if name not in {
                'BooleanAnnotation',
                'DoubleAnnotation',
                'LongAnnotation',
                'CommentAnnotation',
                'MapAnnotation',
                # 'FileAnnotation',
                # 'ListAnnotation',
                # 'TimestampAnnotation,
                # 'XmlAnnotation',
            }:
                continue

            if not isinstance(annotation_values, (list, tuple)):
                annotation_values = [annotation_values]

            for value in annotation_values:
                namespace = ''
                description = ''
                if isinstance(value, dict):
                    value = value.copy()  # noqa: PLW2901
                    description = value.pop('Description', '')
                    if description:
                        description = (
                            '<Description>'
                            f'{OmeXml._escape(description)}'
                            '</Description>'
                        )
                    namespace = value.pop('Namespace', '')
                    if namespace:
                        namespace = f' Namespace="{OmeXml._escape(namespace)}"'
                    value = value.pop('Value', value)  # noqa: PLW2901
                if name == 'MapAnnotation':
                    if not isinstance(value, dict):
                        msg = 'MapAnnotation is not a dict'
                        raise ValueError(msg)
                    values = [
                        f'<M K="{OmeXml._escape(k)}">{OmeXml._escape(v)}</M>'
                        for k, v in value.items()
                    ]
                elif name == 'BooleanAnnotation':
                    values = [f'{bool(value)}'.lower()]
                else:
                    values = [OmeXml._escape(str(value))]
                annotation_refs.append(
                    f'<AnnotationRef ID="Annotation:{len(self.annotations)}"/>'
                )
                self.annotations.append(
                    ''.join(
                        (
                            f'<{name} '
                            f'ID="Annotation:{len(self.annotations)}"'
                            f'{namespace}>',
                            description,
                            '<Value>',
                            ''.join(values),
                            '</Value>',
                            f'</{name}>',
                        )
                    )
                )

    @staticmethod
    def validate(
        omexml: str,
        /,
        omexsd: bytes | None = None,
        *,
        assert_: bool = True,
        _schema: list[Any] = [],  # noqa: B006  (etree.XMLSchema)
    ) -> bool | None:
        r"""Return if OME-XML is valid according to XMLSchema.

        Parameters:
            omexml:
                OME-XML string to validate.
            omexsd:
                Content of OME-XSD schema to validate against.
                By default, the 2016-06 OME XMLSchema is downloaded on first
                run.
            assert\_:
                Raise AssertionError if validation fails.
            _schema:
                Internal use.

        Raises:
            AssertionError:
                Validation failed and `assert\_` is *True*.

        """
        from lxml import etree

        if not _schema:
            if omexsd is None:
                omexsd_path = os.path.join(
                    os.path.dirname(__file__), 'ome.xsd'
                )
                if os.path.exists(omexsd_path):
                    with open(omexsd_path, 'rb') as fh:
                        omexsd = fh.read()
                else:
                    import urllib.request

                    with urllib.request.urlopen(
                        'https://www.openmicroscopy.org/'
                        'Schemas/OME/2016-06/ome.xsd'
                    ) as fh:
                        omexsd = fh.read()
            if omexsd.startswith(b'<?xml'):
                omexsd = omexsd.split(b'>', 1)[-1]
            try:
                _schema.append(
                    etree.XMLSchema(etree.fromstring(omexsd.decode()))
                )
            except Exception:
                # raise
                _schema.append(None)
        if _schema and _schema[0] is not None:
            if omexml.startswith('<?xml'):
                omexml = omexml.split('>', 1)[-1]
            tree = etree.fromstring(omexml)
            if assert_:
                _schema[0].assert_(tree)
                return True
            return bool(_schema[0].validate(tree))
        return None


def read_tags(
    fh: FileHandle,
    /,
    byteorder: ByteOrder,
    offsetsize: int,
    tagnames: TiffTagRegistry,
    *,
    maxifds: int | None = None,
    customtags: (
        dict[int, Callable[[FileHandle, ByteOrder, int, int, int], Any]] | None
    ) = None,
) -> list[dict[str, Any]]:
    """Read tag values from chain of IFDs.

    Parameters:
        fh:
            Binary file handle to read from.
            The file handle position must be at a valid IFD header.
        byteorder:
            Byte order of TIFF file.
        offsetsize:
            Size of offsets in TIFF file (8 for BigTIFF, else 4).
        tagnames:
            Map of tag codes to names.
            For example, :py:class:`_TIFF.GPS_TAGS` or
            :py:class:`_TIFF.IOP_TAGS`.
        maxifds:
            Maximum number of IFDs to read.
            By default, read the whole IFD chain.
        customtags:
            Mapping of tag codes to functions reading special tag value from
            file.

    Raises:
        TiffFileError: Invalid TIFF structure.

    Notes:
        This implementation does not support 64-bit NDPI files.

    """
    code: int
    dtype: int
    count: int
    valuebytes: bytes
    valueoffset: int

    if offsetsize == 4:
        offsetformat = byteorder + 'I'
        tagnosize = 2
        tagnoformat = byteorder + 'H'
        tagsize = 12
        tagformat1 = byteorder + 'HH'
        tagformat2 = byteorder + 'I4s'
    elif offsetsize == 8:
        offsetformat = byteorder + 'Q'
        tagnosize = 8
        tagnoformat = byteorder + 'Q'
        tagsize = 20
        tagformat1 = byteorder + 'HH'
        tagformat2 = byteorder + 'Q8s'
    else:
        msg = 'invalid offset size'
        raise ValueError(msg)

    if customtags is None:
        customtags = {}
    if maxifds is None:
        maxifds = 2**32

    result: list[dict[str, Any]] = []
    unpack = struct.unpack
    offset = fh.tell()
    while len(result) < maxifds:
        # loop over IFDs
        try:
            tagno = unpack(tagnoformat, fh.read(tagnosize))[0]
            if tagno > 4096:
                msg = f'suspicious number of tags {tagno}'
                raise TiffFileError(msg)
        except Exception as exc:
            logger().error(
                f'<tifffile.read_tags> corrupted tag list @{offset} '
                f'raised {exc!r:.128}'
            )
            break

        tags = {}
        data = fh.read(tagsize * tagno)
        pos = fh.tell()
        index = 0

        for _ in range(tagno):
            code, dtype = unpack(tagformat1, data[index : index + 4])
            count, valuebytes = unpack(
                tagformat2, data[index + 4 : index + tagsize]
            )
            index += tagsize
            name = tagnames.get(code, str(code))
            try:
                valueformat = TIFF.DATA_FORMATS[dtype]
            except KeyError:
                logger().error(f'invalid data type {dtype!r} for tag #{code}')
                continue

            valuesize = count * struct.calcsize(valueformat)
            if valuesize > offsetsize or code in customtags:
                valueoffset = unpack(offsetformat, valuebytes)[0]
                if valueoffset < 8 or valueoffset + valuesize > fh.size:
                    logger().error(
                        f'invalid value offset {valueoffset} for tag #{code}'
                    )
                    continue
                fh.seek(valueoffset)
                if code in customtags:
                    readfunc = customtags[code]
                    value = readfunc(fh, byteorder, dtype, count, offsetsize)
                elif dtype in {1, 2, 7}:
                    # BYTES, ASCII, UNDEFINED
                    value = fh.read(valuesize)
                    if len(value) != valuesize:
                        logger().warning(
                            '<tifffile.read_tags> '
                            f'could not read all values for tag #{code}'
                        )
                elif code in tagnames:
                    fmt = (
                        f'{byteorder}'
                        f'{count * int(valueformat[0])}'
                        f'{valueformat[1]}'
                    )
                    value = unpack(fmt, fh.read(valuesize))
                else:
                    value = read_numpy(fh, byteorder, dtype, count, offsetsize)
            elif dtype in {1, 2, 7}:
                # BYTES, ASCII, UNDEFINED
                value = valuebytes[:valuesize]
            else:
                fmt = (
                    f'{byteorder}'
                    f'{count * int(valueformat[0])}'
                    f'{valueformat[1]}'
                )
                value = unpack(fmt, valuebytes[:valuesize])

            process = (
                code not in customtags
                and code not in TIFF.TAG_TUPLE
                and dtype != 7  # UNDEFINED
            )
            if process and dtype == 2:
                # TIFF ASCII fields can contain multiple strings,
                #   each terminated with a NUL
                value = value.rstrip(b'\x00')
                try:
                    value = value.decode('utf-8').strip()
                except UnicodeDecodeError:
                    try:
                        value = value.decode('cp1252').strip()
                    except UnicodeDecodeError as exc:
                        logger().warning(
                            '<tifffile.read_tags> coercing invalid ASCII to '
                            f'bytes for tag #{code}, due to {exc!r:.128}'
                        )
            else:
                if code in TIFF.TAG_ENUM:
                    t = TIFF.TAG_ENUM[code]
                    try:
                        value = tuple(t(v) for v in value)
                    except ValueError as exc:
                        if code not in {259, 317}:
                            # ignore compression/predictor
                            logger().warning(
                                f'<tifffile.read_tags> tag #{code} '
                                f'raised {exc!r:.128}'
                            )
                if process and len(value) == 1:
                    value = value[0]
            tags[name] = value

        result.append(tags)

        # read offset to next page
        fh.seek(pos)
        offset = unpack(offsetformat, fh.read(offsetsize))[0]
        if offset == 0:
            break
        if offset >= fh.size:
            logger().error(f'<tifffile.read_tags> invalid next page {offset=}')
            break
        fh.seek(offset)

    return result


def read_exif_ifd(
    fh: FileHandle,
    byteorder: ByteOrder,
    dtype: int,
    count: int,
    offsetsize: int,
    /,
) -> dict[str, Any]:
    """Read EXIF tags from file."""
    exif = read_tags(fh, byteorder, offsetsize, TIFF.EXIF_TAGS, maxifds=1)[0]
    for name in ('ExifVersion', 'FlashpixVersion'):
        with contextlib.suppress(Exception):
            exif[name] = bytes2str(exif[name])
    if 'UserComment' in exif:
        idcode = exif['UserComment'][:8]
        try:
            if idcode == b'ASCII\x00\x00\x00':
                exif['UserComment'] = bytes2str(exif['UserComment'][8:])
            elif idcode == b'UNICODE\x00':
                exif['UserComment'] = exif['UserComment'][8:].decode('utf-16')
        except Exception:  # noqa: S110
            pass
    return exif


def read_gps_ifd(
    fh: FileHandle,
    byteorder: ByteOrder,
    dtype: int,
    count: int,
    offsetsize: int,
    /,
) -> dict[str, Any]:
    """Read GPS tags from file."""
    return read_tags(fh, byteorder, offsetsize, TIFF.GPS_TAGS, maxifds=1)[0]


def read_interoperability_ifd(
    fh: FileHandle,
    byteorder: ByteOrder,
    dtype: int,
    count: int,
    offsetsize: int,
    /,
) -> dict[str, Any]:
    """Read Interoperability tags from file."""
    return read_tags(fh, byteorder, offsetsize, TIFF.IOP_TAGS, maxifds=1)[0]


def read_bytes(
    fh: FileHandle,
    byteorder: ByteOrder,
    dtype: int,
    count: int,
    offsetsize: int,
    /,
) -> bytes:
    """Read tag data from file."""
    count *= numpy.dtype(
        'B' if dtype == 2 else byteorder + TIFF.DATA_FORMATS[dtype][-1]
    ).itemsize
    data = fh.read(count)
    if len(data) != count:
        logger().warning(
            '<tifffile.read_bytes> '
            f'failed to read {count} bytes, got {len(data)})'
        )
    return data


def read_utf8(
    fh: FileHandle,
    byteorder: ByteOrder,
    dtype: int,
    count: int,
    offsetsize: int,
    /,
) -> str:
    """Read unicode tag value from file."""
    return fh.read(count).decode()


def read_numpy(
    fh: FileHandle,
    byteorder: ByteOrder,
    dtype: int,
    count: int,
    offsetsize: int,
    /,
) -> NDArray[Any]:
    """Read NumPy array tag value from file."""
    return fh.read_array(
        'b' if dtype == 2 else byteorder + TIFF.DATA_FORMATS[dtype][-1], count
    )


def read_colormap(
    fh: FileHandle,
    byteorder: ByteOrder,
    dtype: int,
    count: int,
    offsetsize: int,
    /,
) -> NDArray[Any]:
    """Read ColorMap or TransferFunction tag value from file."""
    cmap = fh.read_array(byteorder + TIFF.DATA_FORMATS[dtype][-1], count)
    if count % 3 == 0:
        cmap = cmap.reshape((3, -1))
    return cmap


def read_json(
    fh: FileHandle,
    byteorder: ByteOrder,
    dtype: int,
    count: int,
    offsetsize: int,
    /,
) -> Any:
    """Read JSON tag value from file."""
    data = fh.read(count)
    try:
        return json.loads(bytes2str(data, 'utf-8'))
    except ValueError as exc:
        logger().warning(f'<tifffile.read_json> raised {exc!r:.128}')
    return None


def read_mm_header(
    fh: FileHandle,
    byteorder: ByteOrder,
    dtype: int,
    count: int,
    offsetsize: int,
    /,
) -> dict[str, Any]:
    """Read FluoView mm_header tag value from file."""
    meta = recarray2dict(
        fh.read_record(numpy.dtype(TIFF.MM_HEADER), byteorder=byteorder)
    )
    meta['Dimensions'] = [
        (bytes2str(d[0]).strip(), d[1], d[2], d[3], bytes2str(d[4]).strip())
        for d in meta['Dimensions']
    ]
    d = meta['GrayChannel']
    meta['GrayChannel'] = (
        bytes2str(d[0]).strip(),
        d[1],
        d[2],
        d[3],
        bytes2str(d[4]).strip(),
    )
    return meta


def read_mm_stamp(
    fh: FileHandle,
    byteorder: ByteOrder,
    dtype: int,
    count: int,
    offsetsize: int,
    /,
) -> NDArray[Any]:
    """Read FluoView mm_stamp tag value from file."""
    return fh.read_array(byteorder + 'f8', 8)


def read_uic1tag(
    fh: FileHandle,
    byteorder: ByteOrder,
    dtype: int,
    count: int,
    offsetsize: int,
    /,
    planecount: int = 0,
) -> dict[str, Any]:
    """Read MetaMorph STK UIC1Tag value from file.

    Return empty dictionary if planecount is unknown.

    """
    if dtype not in {4, 5} or byteorder != '<':
        msg = f'invalid UIC1Tag {byteorder}{dtype}'
        raise ValueError(msg)
    result = {}
    if dtype == 5:
        # pre MetaMorph 2.5 (not tested)
        values = fh.read_array('<u4', 2 * count).reshape((count, 2))
        result = {'ZDistance': values[:, 0] / values[:, 1]}
    else:
        for _ in range(count):
            tagid = struct.unpack('<I', fh.read(4))[0]
            if planecount > 1 and tagid in {28, 29, 37, 40, 41}:
                # silently skip unexpected tags
                fh.read(4)
                continue
            name, value = read_uic_tag(fh, tagid, planecount, True)
            if name == 'PlaneProperty':
                pos = fh.tell()
                fh.seek(value + 4)
                result.setdefault(name, []).append(read_uic_property(fh))
                fh.seek(pos)
            else:
                result[name] = value
    return result


def read_uic2tag(
    fh: FileHandle,
    byteorder: ByteOrder,
    dtype: int,
    count: int,
    offsetsize: int,
    /,
) -> dict[str, NDArray[Any]]:
    """Read MetaMorph STK UIC2Tag value from file."""
    if dtype != 5 or byteorder != '<':
        msg = 'invalid UIC2Tag'
        raise ValueError(msg)
    values = fh.read_array('<u4', 6 * count).reshape((count, 6))
    return {
        'ZDistance': values[:, 0] / values[:, 1],
        'DateCreated': values[:, 2],  # julian days
        'TimeCreated': values[:, 3],  # milliseconds
        'DateModified': values[:, 4],  # julian days
        'TimeModified': values[:, 5],  # milliseconds
    }


def read_uic3tag(
    fh: FileHandle,
    byteorder: ByteOrder,
    dtype: int,
    count: int,
    offsetsize: int,
    /,
) -> dict[str, NDArray[Any]]:
    """Read MetaMorph STK UIC3Tag value from file."""
    if dtype != 5 or byteorder != '<':
        msg = 'invalid UIC3Tag'
        raise ValueError(msg)
    values = fh.read_array('<u4', 2 * count).reshape((count, 2))
    return {'Wavelengths': values[:, 0] / values[:, 1]}


def read_uic4tag(
    fh: FileHandle,
    byteorder: ByteOrder,
    dtype: int,
    count: int,
    offsetsize: int,
    /,
) -> dict[str, NDArray[Any]]:
    """Read MetaMorph STK UIC4Tag value from file."""
    if dtype != 4 or byteorder != '<':
        msg = 'invalid UIC4Tag'
        raise ValueError(msg)
    result = {}
    while True:
        tagid: int = struct.unpack('<H', fh.read(2))[0]
        if tagid == 0:
            break
        name, value = read_uic_tag(fh, tagid, count, False)
        result[name] = value
    return result


def read_uic_tag(
    fh: FileHandle,
    tagid: int,
    planecount: int,
    offset: bool,  # noqa: FBT001
) -> tuple[str, Any]:
    """Read single UIC tag value from file and return tag name and value.

    UIC1Tags use an offset.

    """

    def read_int() -> int:
        return int(struct.unpack('<I', fh.read(4))[0])

    def read_int2() -> tuple[int, int]:
        value = struct.unpack('<2I', fh.read(8))
        return int(value[0]), (value[1])

    try:
        name, dtype = TIFF.UIC_TAGS[tagid]
    except IndexError:
        # unknown tag
        return f'_TagId{tagid}', read_int()

    Fraction = TIFF.UIC_TAGS[4][1]

    if offset:
        pos = fh.tell()
        if dtype not in {int, None}:
            off = read_int()
            if off < 8:
                # undocumented cases, or invalid offset
                if dtype is str:
                    return name, ''
                if tagid == 41:  # AbsoluteZValid
                    return name, off
                logger().warning(
                    '<tifffile.read_uic_tag> '
                    f'invalid offset for tag {name!r} @{off}'
                )
                return name, off
            fh.seek(off)

    value: Any

    if dtype is None:
        # skip
        name = '_' + name
        value = read_int()
    elif dtype is int:
        # int
        value = read_int()
    elif dtype is Fraction:
        # fraction
        value = read_int2()
        value = value[0] / value[1]
    elif dtype is julian_datetime:
        # datetime
        value = read_int2()
        try:
            value = julian_datetime(*value)
        except Exception as exc:
            value = None
            logger().warning(
                f'<tifffile.read_uic_tag> reading {name} raised {exc!r:.128}'
            )
    elif dtype is read_uic_property:
        # ImagePropertyEx
        value = read_uic_property(fh)
    elif dtype is str:
        # pascal string
        size = read_int()
        if 0 <= size < 2**10:
            value = struct.unpack(f'{size}s', fh.read(size))[0][:-1]
            value = bytes2str(value)
        elif offset:
            value = ''
            logger().warning(
                f'<tifffile.read_uic_tag> invalid string in tag {name!r}'
            )
        else:
            msg = f'invalid string size {size}'
            raise ValueError(msg)
    elif planecount == 0:
        value = None
    elif dtype == '%ip':
        # sequence of pascal strings
        value = []
        for _ in range(planecount):
            size = read_int()
            if 0 <= size < 2**10:
                string = struct.unpack(f'{size}s', fh.read(size))[0][:-1]
                value.append(bytes2str(string))
            elif offset:
                logger().warning(
                    f'<tifffile.read_uic_tag> invalid string in tag {name!r}'
                )
            else:
                msg = f'invalid string size: {size}'
                raise ValueError(msg)
    else:
        # struct or numpy type
        dtype = '<' + dtype
        if '%i' in dtype:
            dtype = dtype % planecount
        if '(' in dtype:
            # numpy type
            value = fh.read_array(dtype, 1)[0]
            if value.shape[-1] == 2:
                # assume fractions
                value = value[..., 0] / value[..., 1]
        else:
            # struct format
            value = struct.unpack(dtype, fh.read(struct.calcsize(dtype)))
            if len(value) == 1:
                value = value[0]

    if offset:
        fh.seek(pos + 4)

    return name, value


def read_uic_property(fh: FileHandle, /) -> dict[str, Any]:
    """Read UIC ImagePropertyEx or PlaneProperty tag from file."""
    size = struct.unpack('B', fh.read(1))[0]
    name = bytes2str(struct.unpack(f'{size}s', fh.read(size))[0])
    flags, prop = struct.unpack('<IB', fh.read(5))
    if prop == 1:
        value = struct.unpack('II', fh.read(8))
        value = value[0] / value[1]
    else:
        size = struct.unpack('B', fh.read(1))[0]
        value = bytes2str(
            struct.unpack(f'{size}s', fh.read(size))[0]
        )  # type: ignore[assignment]
    return {'name': name, 'flags': flags, 'value': value}


def read_cz_lsminfo(
    fh: FileHandle,
    byteorder: ByteOrder,
    dtype: int,
    count: int,
    offsetsize: int,
    /,
) -> dict[str, Any]:
    """Read CZ_LSMINFO tag value from file."""
    if byteorder != '<':
        msg = 'invalid CZ_LSMINFO structure'
        raise ValueError(msg)
    magic_number, structure_size = struct.unpack('<II', fh.read(8))
    if magic_number not in {50350412, 67127628}:
        msg = 'invalid CZ_LSMINFO structure'
        raise ValueError(msg)
    fh.seek(-8, os.SEEK_CUR)
    CZ_LSMINFO = TIFF.CZ_LSMINFO

    if structure_size < numpy.dtype(CZ_LSMINFO).itemsize:
        # adjust structure according to structure_size
        lsminfo: list[tuple[str, str]] = []
        size = 0
        for name, typestr in CZ_LSMINFO:
            size += numpy.dtype(typestr).itemsize
            if size > structure_size:
                break
            lsminfo.append((name, typestr))
    else:
        lsminfo = CZ_LSMINFO

    result = recarray2dict(
        fh.read_record(numpy.dtype(lsminfo), byteorder=byteorder)
    )

    # read LSM info subrecords at offsets
    for name, reader in TIFF.CZ_LSMINFO_READERS.items():
        if reader is None:
            continue
        offset = result.get('Offset' + name, 0)
        if offset < 8:
            continue
        fh.seek(offset)
        with contextlib.suppress(ValueError):
            result[name] = reader(fh)
    return result


def read_lsm_channeldatatypes(fh: FileHandle, /) -> NDArray[Any]:
    """Read LSM channel data type from file."""
    size = struct.unpack('<I', fh.read(4))[0]
    return fh.read_array('<u4', count=size)


def read_lsm_channelwavelength(fh: FileHandle, /) -> NDArray[Any]:
    """Read LSM channel wavelength ranges from file."""
    size = struct.unpack('<i', fh.read(4))[0]
    return fh.read_array('<2f8', count=size)


def read_lsm_positions(fh: FileHandle, /) -> NDArray[Any]:
    """Read LSM positions from file."""
    size = struct.unpack('<I', fh.read(4))[0]
    return fh.read_array('<3f8', count=size)


def read_lsm_timestamps(fh: FileHandle, /) -> NDArray[Any]:
    """Read LSM time stamps from file."""
    size, count = struct.unpack('<ii', fh.read(8))
    if size != (8 + 8 * count):
        logger().warning(
            '<tifffile.read_lsm_timestamps> invalid LSM TimeStamps block'
        )
        return numpy.empty((0,), '<f8')
    # return struct.unpack(f'<{count}d', fh.read(8 * count))
    return fh.read_array('<f8', count=count)


def read_lsm_eventlist(fh: FileHandle, /) -> list[tuple[float, int, str]]:
    """Read LSM events from file and return as list of (time, type, text)."""
    count = struct.unpack('<II', fh.read(8))[1]
    events = []
    while count > 0:
        esize, etime, etype = struct.unpack('<IdI', fh.read(16))
        etext = bytes2str(fh.read(esize - 16))
        events.append((etime, etype, etext))
        count -= 1
    return events


def read_lsm_channelcolors(fh: FileHandle, /) -> dict[str, Any]:
    """Read LSM ChannelColors structure from file."""
    result = {'Mono': False, 'Colors': [], 'ColorNames': []}
    pos = fh.tell()
    size, ncolors, nnames, coffset, noffset, mono = struct.unpack(
        '<IIIIII', fh.read(24)
    )
    if ncolors != nnames:
        logger().warning(
            '<tifffile.read_lsm_channelcolors> '
            'invalid LSM ChannelColors structure'
        )
        return result
    result['Mono'] = bool(mono)
    # Colors
    fh.seek(pos + coffset)
    colors = fh.read_array(numpy.uint8, count=ncolors * 4)
    colors = colors.reshape((ncolors, 4))
    result['Colors'] = colors.tolist()
    # ColorNames
    fh.seek(pos + noffset)
    buffer = fh.read(size - noffset)
    names = []
    while len(buffer) > 4:
        size = struct.unpack('<I', buffer[:4])[0]
        names.append(bytes2str(buffer[4 : 3 + size]))
        buffer = buffer[4 + size :]
    result['ColorNames'] = names
    return result


def read_lsm_lookuptable(fh: FileHandle, /) -> dict[str, Any]:
    """Read LSM lookup tables from file."""
    result: dict[str, Any] = {}
    (
        size,
        nsubblocks,
        nchannels,
        luttype,
        advanced,
        currentchannel,
    ) = struct.unpack('<iiiiii', fh.read(24))
    if size < 60:
        logger().warning(
            '<tifffile.read_lsm_lookuptable> '
            'invalid LSM LookupTables structure'
        )
        return result
    fh.read(9 * 4)  # reserved
    result['LutType'] = TIFF.CZ_LSM_LUTTYPE(luttype)
    result['Advanced'] = advanced
    result['NumberChannels'] = nchannels
    result['CurrentChannel'] = currentchannel
    result['SubBlocks'] = subblocks = []
    for _ in range(nsubblocks):
        sbtype = struct.unpack('<i', fh.read(4))[0]
        if sbtype <= 0:
            break
        size = struct.unpack('<i', fh.read(4))[0] - 8
        if 0 < sbtype < 4:
            data = fh.read_array('<f8', count=nchannels)
        elif sbtype == 4:
            # the data type is wrongly documented as f8
            data = fh.read_array('<i4', count=nchannels * 4)
            data = data.reshape((-1, 2, 2))
        elif sbtype == 5:
            # the data type is wrongly documented as f8
            nknots = struct.unpack('<i', fh.read(4))[0]  # undocumented
            data = fh.read_array('<i4', count=nchannels * nknots * 2)
            data = data.reshape((nchannels, nknots, 2))
        elif sbtype == 6:
            data = fh.read_array('<i2', count=nchannels * 4096)
            data = data.reshape((-1, 4096))
        else:
            logger().warning(
                '<tifffile.read_lsm_lookuptable> '
                f'invalid LSM SubBlock type {sbtype}'
            )
            break
        subblocks.append(
            {'Type': TIFF.CZ_LSM_SUBBLOCK_TYPE(sbtype), 'Data': data}
        )
    return result


def read_lsm_scaninfo(fh: FileHandle, /) -> dict[str, Any]:
    """Read LSM ScanInfo structure from file."""
    value: Any
    block: dict[str, Any] = {}
    blocks = [block]
    unpack = struct.unpack
    if struct.unpack('<I', fh.read(4))[0] != 0x10000000:
        # not a Recording sub block
        logger().warning(
            '<tifffile.read_lsm_scaninfo> invalid LSM ScanInfo structure'
        )
        return block
    fh.read(8)
    while True:
        entry, dtype, size = unpack('<III', fh.read(12))
        if dtype == 2:
            # ascii
            value = bytes2str(fh.read(size))
        elif dtype == 4:
            # long
            value = unpack('<i', fh.read(4))[0]
        elif dtype == 5:
            # rational
            value = unpack('<d', fh.read(8))[0]
        else:
            value = 0
        if entry in TIFF.CZ_LSMINFO_SCANINFO_ARRAYS:
            blocks.append(block)
            name = TIFF.CZ_LSMINFO_SCANINFO_ARRAYS[entry]
            newlist: list[dict[str, Any]] = []
            block[name] = newlist
            # TODO: fix types
            block = newlist  # type: ignore[assignment]
        elif entry in TIFF.CZ_LSMINFO_SCANINFO_STRUCTS:
            blocks.append(block)
            newdict: dict[str, Any] = {}
            # TODO: fix types
            block.append(newdict)  # type: ignore[attr-defined]
            block = newdict
        elif entry in TIFF.CZ_LSMINFO_SCANINFO_ATTRIBUTES:
            block[TIFF.CZ_LSMINFO_SCANINFO_ATTRIBUTES[entry]] = value
        elif entry == 0xFFFFFFFF:
            # end sub block
            block = blocks.pop()
        else:
            # unknown entry
            block[f'Entry0x{entry:x}'] = value
        if not blocks:
            break
    return block


def read_sis(
    fh: FileHandle,
    byteorder: ByteOrder,
    dtype: int,
    count: int,
    offsetsize: int,
    /,
) -> dict[str, Any]:
    """Read OlympusSIS structure from file.

    No specification is available. Only few fields are known.

    """
    result: dict[str, Any] = {}

    magic, minute, hour, day, month, year, name, tagcount = struct.unpack(
        '<4s6xhhhhh6x32sh', fh.read(60)
    )

    if magic != b'SIS0':
        msg = 'invalid OlympusSIS structure'
        raise ValueError(msg)

    result['name'] = bytes2str(name)
    with contextlib.suppress(ValueError):
        result['datetime'] = DateTime(
            1900 + year, month + 1, day, hour, minute
        )

    data = fh.read(8 * tagcount)
    for i in range(0, tagcount * 8, 8):
        tagtype, _count, offset = struct.unpack('<hhI', data[i : i + 8])
        fh.seek(offset)
        if tagtype == 1:
            # general data
            lenexp, xcal, ycal, mag, camname, pictype = struct.unpack(
                '<10xhdd8xd2x34s32s', fh.read(112)  # 220
            )
            m = math.pow(10, lenexp)
            result['pixelsizex'] = xcal * m
            result['pixelsizey'] = ycal * m
            result['magnification'] = mag
            result['cameraname'] = bytes2str(camname)
            result['picturetype'] = bytes2str(pictype)
        elif tagtype == 10:
            # channel data
            continue
            # TODO: does not seem to work?
            # (length, _, exptime, emv, _, camname, _, mictype,
            #  ) = struct.unpack('<h22sId4s32s48s32s', fh.read(152))  # 720
            # result['exposuretime'] = exptime
            # result['emvoltage'] = emv
            # result['cameraname2'] = bytes2str(camname)
            # result['microscopename'] = bytes2str(mictype)

    return result


def read_sis_ini(
    fh: FileHandle,
    byteorder: ByteOrder,
    dtype: int,
    count: int,
    offsetsize: int,
    /,
) -> dict[str, Any]:
    """Read OlympusSIS INI string from file."""
    try:
        return olympus_ini_metadata(bytes2str(fh.read(count)))
    except Exception as exc:
        logger().warning(
            f'<tifffile.olympus_ini_metadata> raised {exc!r:.128}'
        )
        return {}


def read_tvips_header(
    fh: FileHandle,
    byteorder: ByteOrder,
    dtype: int,
    count: int,
    offsetsize: int,
    /,
) -> dict[str, Any]:
    """Read TVIPS EM-MENU headers from file."""
    result: dict[str, Any] = {}
    header_v1 = TIFF.TVIPS_HEADER_V1
    header = fh.read_record(numpy.dtype(header_v1), byteorder=byteorder)
    for name, _typestr in header_v1:
        result[name] = header[name].tolist()
    if header['Version'] == 2:
        header_v2 = TIFF.TVIPS_HEADER_V2
        header = fh.read_record(numpy.dtype(header_v2), byteorder=byteorder)
        if header['Magic'] != 0xAAAAAAAA:
            logger().warning(
                '<tifffile.read_tvips_header> invalid TVIPS v2 magic number'
            )
            return {}
        # decode utf16 strings
        for name, typestr in header_v2:
            if typestr.startswith('V'):
                result[name] = bytes2str(
                    header[name].tobytes(), 'utf-16', 'ignore'
                )
            else:
                result[name] = header[name].tolist()
        # convert nm to m
        for axis in 'XY':
            header['PhysicalPixelSize' + axis] /= 1e9
            header['PixelSize' + axis] /= 1e9
    elif header.version != 1:
        logger().warning(
            '<tifffile.read_tvips_header> unknown TVIPS header version'
        )
        return {}
    return result


def read_fei_metadata(
    fh: FileHandle,
    byteorder: ByteOrder,
    dtype: int,
    count: int,
    offsetsize: int,
    /,
) -> dict[str, Any]:
    """Read FEI SFEG/HELIOS headers from file."""
    result: dict[str, Any] = {}
    section: dict[str, Any] = {}
    data = bytes2str(fh.read(count))
    for line in data.splitlines():
        line = line.strip()  # noqa: PLW2901
        if line.startswith('['):
            section = {}
            result[line[1:-1]] = section
            continue
        try:
            key, value = line.split('=')
        except ValueError:
            continue
        section[key] = astype(value)
    return result


def read_cz_sem(
    fh: FileHandle,
    byteorder: ByteOrder,
    dtype: int,
    count: int,
    offsetsize: int,
    /,
) -> dict[str, Any]:
    """Read Zeiss SEM tag from file.

    See https://sourceforge.net/p/gwyddion/mailman/message/29275000/ for
    unnamed values.

    """
    result: dict[str, Any] = {'': ()}
    value: Any
    key = None
    data = bytes2str(fh.read(count))
    for line in data.splitlines():
        if line.isupper():
            key = line.lower()
        elif key:
            try:
                name, value = line.split('=')
            except ValueError:
                try:
                    name, value = line.split(':', 1)
                except ValueError:
                    continue
            value = value.strip()
            unit = ''
            try:
                v, u = value.split()
                number = astype(v, (int, float))
                if number != v:
                    value = number
                    unit = u
            except Exception:
                number = astype(value, (int, float))
                if number != value:
                    value = number
                if value in {'No', 'Off'}:
                    value = False
                elif value in {'Yes', 'On'}:
                    value = True
            result[key] = (name.strip(), value)
            if unit:
                result[key] += (unit,)
            key = None
        else:
            result[''] += (astype(line, (int, float)),)
    return result


def read_nih_image_header(
    fh: FileHandle,
    byteorder: ByteOrder,
    dtype: int,
    count: int,
    offsetsize: int,
    /,
) -> dict[str, Any]:
    """Read NIH_IMAGE_HEADER tag value from file."""
    arr = fh.read_record(TIFF.NIH_IMAGE_HEADER, byteorder=byteorder)
    arr = arr.view(arr.dtype.newbyteorder(byteorder))
    result = recarray2dict(arr)
    result['XUnit'] = result['XUnit'][: result['XUnitSize']]
    result['UM'] = result['UM'][: result['UMsize']]
    return result


def read_scanimage_metadata(
    fh: FileHandle, /
) -> tuple[dict[str, Any], dict[str, Any], int]:
    """Read ScanImage BigTIFF v3 or v4 static and ROI metadata from file.

    The settings can be used to read image and metadata without parsing
    the TIFF file.

    Frame data and ROI groups can alternatively be obtained from the Software
    and Artist tags of any TIFF page.

    Parameters:
        fh: Binary file handle to read from.

    Returns:
        - Non-varying frame data, parsed with :py:func:`matlabstr2py`.
        - ROI group data, parsed from JSON.
        - Version of metadata (3 or 4).

    Raises:
        ValueError: File does not contain valid ScanImage metadata.

    """
    fh.seek(0)
    try:
        byteorder, version = struct.unpack('<2sH', fh.read(4))
        if byteorder != b'II' or version != 43:
            msg = 'not a BigTIFF file'
            raise ValueError(msg)
        fh.seek(16)
        magic, version, size0, size1 = struct.unpack('<IIII', fh.read(16))
        if magic != 117637889 or version not in {3, 4}:
            msg = f'invalid {magic=} or {version=}'
            raise ValueError(msg)
    except UnicodeDecodeError as exc:
        msg = 'file must be opened in binary mode'
        raise ValueError(msg) from exc
    except Exception as exc:
        msg = 'not a ScanImage BigTIFF v3 or v4 file'
        raise ValueError(msg) from exc

    frame_data = matlabstr2py(bytes2str(fh.read(size0)[:-1]))
    roi_data = read_json(fh, '<', 0, size1, 0) if size1 > 1 else {}
    return frame_data, roi_data, version


def read_micromanager_metadata(
    fh: FileHandle | IO[bytes], /, keys: Container[str] | None = None
) -> dict[str, Any]:
    """Return Micro-Manager non-TIFF settings from file.

    The settings can be used to read image data without parsing any TIFF
    structures.

    Parameters:
        fh: Open file handle to Micro-Manager TIFF file.
        keys: Name of keys to return in result.

    Returns:
        Micro-Manager non-TIFF settings, which may contain the following keys:

        - 'MajorVersion' (str)
        - 'MinorVersion' (str)
        - 'Summary' (dict):
          Specifies the dataset, such as shape, dimensions, and coordinates.
        - 'IndexMap' (numpy.ndarray):
          (channel, slice, frame, position, ifd_offset) indices of all frames.
        - 'DisplaySettings' (list[dict]):
          Image display settings such as channel contrast and colors.
        - 'Comments' (dict):
          User comments.

    Notes:
        Summary metadata are the same for all files in a dataset.
        DisplaySettings metadata are frequently corrupted, and Comments are
        often empty.
        The Summary and IndexMap metadata are stored at the beginning of
        the file, while DisplaySettings and Comments are towards the end.
        Excluding DisplaySettings and Comments from the results may
        significantly speed up reading metadata of interest.

    References:
        - https://micro-manager.org/Micro-Manager_File_Formats
        - https://github.com/micro-manager/NDTiffStorage

    """
    if keys is None:
        keys = {'Summary', 'IndexMap', 'DisplaySettings', 'Comments'}
    fh.seek(0)
    try:
        byteorder = {b'II': '<', b'MM': '>'}[fh.read(2)]
        fh.seek(8)
        (
            index_header,
            index_offset,
        ) = struct.unpack(byteorder + 'II', fh.read(8))
    except Exception as exc:
        msg = 'not a Micro-Manager TIFF file'
        raise ValueError(msg) from exc

    result = {}
    if index_header == 483729:
        # NDTiff >= v2
        result['MajorVersion'] = index_offset
        try:
            (
                summary_header,
                summary_length,
            ) = struct.unpack(byteorder + 'II', fh.read(8))
            if summary_header != 2355492:
                # NDTiff v3
                result['MinorVersion'] = summary_header
                if summary_length != 2355492:
                    msg = f'invalid {summary_length=}'
                    raise ValueError(msg)
                summary_length = struct.unpack(byteorder + 'I', fh.read(4))[0]
            if 'Summary' in keys:
                data = fh.read(summary_length)
                if len(data) != summary_length:
                    msg = 'not enough data'
                    raise ValueError(msg)
                result['Summary'] = json.loads(bytes2str(data, 'utf-8'))
        except Exception as exc:
            logger().warning(
                '<tifffile.read_micromanager_metadata> '
                f'failed to read NDTiffv{index_offset} summary settings, '
                f'raised {exc!r:.128}'
            )
        return result

    # Micro-Manager multipage TIFF or NDTiff v1
    try:
        (
            display_header,
            display_offset,
            comments_header,
            comments_offset,
            summary_header,
            summary_length,
        ) = struct.unpack(byteorder + 'IIIIII', fh.read(24))
    except Exception as exc:
        logger().warning(
            '<tifffile.read_micromanager_metadata> failed to read header, '
            f'raised {exc!r:.128}'
        )

    if 'Summary' in keys:
        try:
            if summary_header != 2355492:
                msg = f'invalid {summary_header=}'
                raise ValueError(msg)
            data = fh.read(summary_length)
            if len(data) != summary_length:
                msg = 'not enough data'
                raise ValueError(msg)
            result['Summary'] = json.loads(bytes2str(data, 'utf-8'))
        except Exception as exc:
            logger().warning(
                '<tifffile.read_micromanager_metadata> '
                f'failed to read summary settings, raised {exc!r:.128}'
            )

    if 'IndexMap' in keys:
        try:
            if index_header != 54773648:
                msg = f'invalid {index_header=}'
                raise ValueError(msg)
            fh.seek(index_offset)
            header, count = struct.unpack(byteorder + 'II', fh.read(8))
            if header != 3453623:
                msg = 'invalid header'
                raise ValueError(msg)
            data = fh.read(count * 20)
            result['IndexMap'] = numpy.frombuffer(
                data, byteorder + 'u4', count * 5
            ).reshape((-1, 5))
        except Exception as exc:
            logger().warning(
                '<tifffile.read_micromanager_metadata> '
                f'failed to read index map, raised {exc!r:.128}'
            )

    if 'DisplaySettings' in keys:
        try:
            if display_header != 483765892:
                msg = f'invalid {display_header=}'
                raise ValueError(msg)
            fh.seek(display_offset)
            header, count = struct.unpack(byteorder + 'II', fh.read(8))
            if header != 347834724:
                # display_offset might be wrapped at 4 GB
                fh.seek(display_offset + 2**32)
                header, count = struct.unpack(byteorder + 'II', fh.read(8))
                if header != 347834724:
                    msg = f'invalid display {header=}'
                    raise ValueError(msg)
            data = fh.read(count)
            if len(data) != count:
                msg = 'not enough data'
                raise ValueError(msg)
            result['DisplaySettings'] = json.loads(bytes2str(data, 'utf-8'))
        except json.decoder.JSONDecodeError:
            pass  # ignore frequent truncated JSON data
        except Exception as exc:
            logger().warning(
                '<tifffile.read_micromanager_metadata> '
                f'failed to read display settings, raised {exc!r:.128}'
            )

    result['MajorVersion'] = 0
    try:
        if comments_header == 99384722:
            # Micro-Manager multipage TIFF
            if 'Comments' in keys:
                fh.seek(comments_offset)
                header, count = struct.unpack(byteorder + 'II', fh.read(8))
                if header != 84720485:
                    # comments_offset might be wrapped at 4 GB
                    fh.seek(comments_offset + 2**32)
                    header, count = struct.unpack(byteorder + 'II', fh.read(8))
                    if header != 84720485:
                        msg = 'invalid comments header'
                        raise ValueError(msg)
                data = fh.read(count)
                if len(data) != count:
                    msg = 'not enough data'
                    raise ValueError(msg)
                result['Comments'] = json.loads(bytes2str(data, 'utf-8'))
        elif comments_header == 483729:
            # NDTiff v1
            result['MajorVersion'] = comments_offset
        elif comments_header == 0 and comments_offset == 0:
            pass
        elif 'Comments' in keys:
            msg = f'invalid {comments_header=}'
            raise ValueError(msg)
    except Exception as exc:
        logger().warning(
            '<tifffile.read_micromanager_metadata> failed to read comments, '
            f'raised {exc!r:.128}'
        )

    return result


def read_ndtiff_index(
    file: str | os.PathLike[Any], /
) -> Iterator[
    tuple[dict[str, int | str], str, int, int, int, int, int, int, int, int]
]:
    """Return iterator over fields in Micro-Manager NDTiff.index file.

    Parameters:
        file: Path of NDTiff.index file.

    Yields:
        Fields in NDTiff.index file:

        - axes_dict: Axes indices of frame in image.
        - filename: Name of file containing frame and metadata.
        - dataoffset: Offset of frame data in file.
        - width: Width of frame.
        - height: Height of frame.
        - pixeltype: Pixel type.
          0: 8-bit monochrome;
          1: 16-bit monochrome;
          2: 8-bit RGB;
          3: 10-bit monochrome;
          4: 12-bit monochrome;
          5: 14-bit monochrome;
          6: 11-bit monochrome.
        - compression: Pixel compression. 0: Uncompressed.
        - metaoffset: Offset of JSON metadata in file.
        - metabytecount: Length of metadata.
        - metacompression: Metadata compression. 0: Uncompressed.

    """
    with open(file, 'rb') as fh:
        while True:
            b = fh.read(4)
            if len(b) != 4:
                break
            k = struct.unpack('<i', b)[0]
            axes_dict = json.loads(fh.read(k))
            n = struct.unpack('<i', fh.read(4))[0]
            filename = fh.read(n).decode()
            (
                dataoffset,
                width,
                height,
                pixeltype,
                compression,
                metaoffset,
                metabytecount,
                metacompression,
            ) = struct.unpack('<IiiiiIii', fh.read(32))
            yield (
                axes_dict,
                filename,
                dataoffset,
                width,
                height,
                pixeltype,
                compression,
                metaoffset,
                metabytecount,
                metacompression,
            )


def read_gdal_structural_metadata(
    fh: FileHandle | IO[bytes], /
) -> dict[str, str] | None:
    """Read non-TIFF GDAL structural metadata from file.

    Return None if the file does not contain valid GDAL structural metadata.
    The metadata can be used to optimize reading image data from a COG file.

    """
    fh.seek(0)
    try:
        if fh.read(2) not in {b'II', b'MM'}:
            msg = 'not a TIFF file'
            raise ValueError(msg)
        fh.seek({b'*': 8, b'+': 16}[fh.read(1)])
        header = fh.read(43).decode()
        if header[:30] != 'GDAL_STRUCTURAL_METADATA_SIZE=':
            return None
        size = int(header[30:36])
        lines = fh.read(size).decode()
    except Exception:
        return None

    result: dict[str, Any] = {}
    try:
        for line in lines.splitlines():
            if '=' in line:
                key, value = line.split('=', 1)
                result[key.strip()] = value.strip()
    except Exception as exc:
        logger().warning(
            f'<tifffile.read_gdal_structural_metadata> raised {exc!r:.128}'
        )
        return None
    return result


def read_metaseries_catalog(fh: FileHandle | IO[bytes], /) -> None:
    """Read MetaSeries non-TIFF hint catalog from file.

    Raise ValueError if the file does not contain a valid hint catalog.

    """
    # TODO: implement read_metaseries_catalog
    raise NotImplementedError


def imagej_metadata_tag(
    metadata: dict[str, Any], byteorder: ByteOrder, /
) -> tuple[
    tuple[int, int, int, bytes, bool], tuple[int, int, int, bytes, bool]
]:
    """Return IJMetadata and IJMetadataByteCounts tags from metadata dict.

    Parameters:
        metadata:
            May contain the following keys and values:

            'Info' (str):
                Human-readable information as string.
            'Labels' (Sequence[str]):
                Human-readable label for each image.
            'Ranges' (Sequence[float]):
                Lower and upper values for each channel.
            'LUTs' (list[numpy.ndarray[(3, 256), 'uint8']]):
                Color palettes for each channel.
            'Plot' (bytes):
                Undocumented ImageJ internal format.
            'ROI', 'Overlays' (bytes):
                Undocumented ImageJ internal region of interest and overlay
                format. Can be created with the
                `roifile <https://pypi.org/project/roifile/>`_ package.
            'Properties' (dict[str, str]):
                Map of key, value items.

        byteorder:
            Byte order of TIFF file.

    Returns:
        IJMetadata and IJMetadataByteCounts tags in :py:meth:`TiffWriter.write`
        `extratags` format.

    """
    if not metadata:
        return ()  # type: ignore[return-value]
    header_list = [{'>': b'IJIJ', '<': b'JIJI'}[byteorder]]
    bytecount_list = [0]
    body_list = []

    def _string(data: str, byteorder: ByteOrder, /) -> bytes:
        return data.encode('utf-16' + {'>': 'be', '<': 'le'}[byteorder])

    def _doubles(data: Sequence[float], byteorder: ByteOrder, /) -> bytes:
        return struct.pack(f'{byteorder}{len(data)}d', *data)

    def _ndarray(data: NDArray[Any], byteorder: ByteOrder, /) -> bytes:
        return data.tobytes()

    def _bytes(data: bytes, byteorder: ByteOrder, /) -> bytes:
        return data

    metadata_types: tuple[
        tuple[str, bytes, Callable[[Any, ByteOrder], bytes]], ...
    ] = (
        ('Info', b'info', _string),
        ('Labels', b'labl', _string),
        ('Ranges', b'rang', _doubles),
        ('LUTs', b'luts', _ndarray),
        ('Plot', b'plot', _bytes),
        ('ROI', b'roi ', _bytes),
        ('Overlays', b'over', _bytes),
        ('Properties', b'prop', _string),
    )

    for item in metadata_types:
        key, mtype, func = item
        if key.lower() in metadata:
            key = key.lower()
        elif key not in metadata:
            continue
        if byteorder == '<':
            mtype = mtype[::-1]
        values = metadata[key]
        if isinstance(values, dict):
            values = [str(i) for item in values.items() for i in item]
            count = len(values)
        elif func is _doubles:
            values = [values]
            count = 1
        elif isinstance(values, (list, tuple)):
            count = len(values)
        else:
            values = [values]
            count = 1
        header_list.append(mtype + struct.pack(byteorder + 'I', count))
        for value in values:
            data = func(value, byteorder)
            body_list.append(data)
            bytecount_list.append(len(data))

    if not body_list:
        return ()  # type: ignore[return-value]
    body = b''.join(body_list)
    header = b''.join(header_list)
    data = header + body
    bytecount_list[0] = len(header)
    bytecounts = struct.pack(
        byteorder + ('I' * len(bytecount_list)), *bytecount_list
    )
    return (
        (50839, 1, len(data), data, True),
        (50838, 4, len(bytecounts) // 4, bytecounts, True),
    )


def imagej_metadata(
    data: bytes, bytecounts: Sequence[int], byteorder: ByteOrder, /
) -> dict[str, Any]:
    """Return IJMetadata tag value.

    Parameters:
        bytes:
            Encoded value of IJMetadata tag.
        bytecounts:
            Value of IJMetadataByteCounts tag.
        byteorder:
            Byte order of TIFF file.

    Returns:
        Metadata dict with optional items:

            'Info' (str):
                Human-readable information as string.
                Some formats, such as OIF or ScanImage, can be parsed into
                dicts with :py:func:`matlabstr2py` or the
                `oiffile.SettingsFile()` function of the
                `oiffile <https://pypi.org/project/oiffile/>`_  package.
            'Labels' (Sequence[str]):
                Human-readable labels for each channel.
            'Ranges' (Sequence[float]):
                Lower and upper values for each channel.
            'LUTs' (list[numpy.ndarray[(3, 256), 'uint8']]):
                Color palettes for each channel.
            'Plot' (bytes):
                Undocumented ImageJ internal format.
            'ROI', 'Overlays' (bytes):
                Undocumented ImageJ internal region of interest and overlay
                format. Can be parsed with the
                `roifile <https://pypi.org/project/roifile/>`_  package.
            'Properties' (dict[str, str]):
                Map of key, value items.

    """

    def _string(data: bytes, byteorder: ByteOrder, /) -> str:
        return data.decode('utf-16' + {'>': 'be', '<': 'le'}[byteorder])

    def _doubles(data: bytes, byteorder: ByteOrder, /) -> tuple[float, ...]:
        return struct.unpack(byteorder + ('d' * (len(data) // 8)), data)

    def _lut(data: bytes, byteorder: ByteOrder, /) -> NDArray[numpy.uint8]:
        return numpy.frombuffer(data, numpy.uint8).reshape((-1, 256))

    def _bytes(data: bytes, byteorder: ByteOrder, /) -> bytes:
        return data

    # big-endian
    metadata_types: dict[
        bytes, tuple[str, Callable[[bytes, ByteOrder], Any]]
    ] = {
        b'info': ('Info', _string),
        b'labl': ('Labels', _string),
        b'rang': ('Ranges', _doubles),
        b'luts': ('LUTs', _lut),
        b'plot': ('Plot', _bytes),
        b'roi ': ('ROI', _bytes),
        b'over': ('Overlays', _bytes),
        b'prop': ('Properties', _string),
    }
    # little-endian
    metadata_types.update({k[::-1]: v for k, v in metadata_types.items()})

    if len(bytecounts) == 0:
        msg = 'no ImageJ metadata'
        raise ValueError(msg)

    if data[:4] not in {b'IJIJ', b'JIJI'}:
        msg = 'invalid ImageJ metadata'
        raise ValueError(msg)

    header_size = bytecounts[0]
    if header_size < 12 or header_size > 804:
        msg = 'invalid ImageJ metadata header size'
        raise ValueError(msg)

    ntypes = (header_size - 4) // 8
    header = struct.unpack(
        byteorder + '4sI' * ntypes, data[4 : 4 + ntypes * 8]
    )
    pos = 4 + ntypes * 8
    counter = 0
    result = {}
    for mtype, count in zip(header[::2], header[1::2], strict=True):
        values = []
        name, func = metadata_types.get(mtype, (bytes2str(mtype), _bytes))
        for _ in range(count):
            counter += 1
            pos1 = pos + bytecounts[counter]
            values.append(func(data[pos:pos1], byteorder))
            pos = pos1
        result[name.strip()] = values[0] if count == 1 else values
    prop = result.get('Properties')
    if prop and len(prop) % 2 == 0:
        result['Properties'] = dict(
            prop[i : i + 2] for i in range(0, len(prop), 2)
        )
    return result


def imagej_description_metadata(description: str, /) -> dict[str, Any]:
    r"""Return metadata from ImageJ image description.

    Raise ValueError if not a valid ImageJ description.

    >>> description = 'ImageJ=1.11a\nimages=510\nhyperstack=true\n'
    >>> imagej_description_metadata(description)  # doctest: +SKIP
    {'ImageJ': '1.11a', 'images': 510, 'hyperstack': True}

    """

    def _bool(val: str, /) -> bool:
        return {'true': True, 'false': False}[val.lower()]

    result: dict[str, Any] = {}
    for line in description.splitlines():
        try:
            key, val = line.split('=')
        except Exception:  # noqa: S112
            continue
        key = key.strip()
        val = val.strip()
        for dtype in (int, float, _bool):
            try:
                val = dtype(val)  # type: ignore[assignment]
                break
            except Exception:  # noqa: S110
                pass
        result[key] = val

    if 'ImageJ' not in result and 'SCIFIO' not in result:
        msg = f'not an ImageJ image description: {result!r}'
        raise ValueError(msg)
    return result


def imagej_description(
    shape: Sequence[int],
    /,
    axes: str | None = None,
    *,
    rgb: bool | None = None,
    colormapped: bool = False,
    **metadata: Any,  # TODO: use TypedDict
) -> str:
    """Return ImageJ image description from data shape and metadata.

    Parameters:
        shape:
            Shape of image array.
        axes:
            Character codes for dimensions in `shape`.
            ImageJ can handle up to 6 dimensions in order TZCYXS.
            `Axes` and `shape` are used to determine the images, channels,
            slices, and frames entries of the image description.
        rgb:
            Image is RGB type.
        colormapped:
            Image is indexed color.
        **metadata:
            Additional items to be included in image description:

            hyperstack (bool):
                Image is a hyperstack.
                The default is True unless `colormapped` is true.
            mode (str):
                Display mode: 'grayscale', 'composite', or 'color'.
                The default is 'grayscale' unless `rgb` or `colormapped` are
                true. Ignored if `hyperstack` is false.
            loop (bool):
                Loop frames back and forth. The default is False.
            finterval (float):
                Frame interval in seconds.
            fps (float):
                Frames per seconds. The inverse of `finterval`.
            spacing (float):
                Voxel spacing in `unit` units.
            unit (str):
                Unit for `spacing` and X/YResolution tags.
                Usually 'um' (micrometer) or 'pixel'.
            xorigin, yorigin, zorigin (float):
                X, Y, and Z origins in pixel units.
            version (str):
                ImageJ version string. The default is '1.11a'.
            images, channels, slices, frames (int):
                Ignored.

    Examples:
        >>> imagej_description((51, 5, 2, 196, 171))  # doctest: +SKIP
        ImageJ=1.11a
        images=510
        channels=2
        slices=5
        frames=51
        hyperstack=true
        mode=grayscale
        loop=false

    """
    if 'colormaped' in metadata:
        warnings.warn(
            '<tifffile.imagej_description colormaped parameter '
            'is deprecated since 2026.2.28. Use colormapped instead.',
            DeprecationWarning,
            stacklevel=2,
        )
        colormapped = bool(metadata.pop('colormaped'))
    mode = metadata.pop('mode', None)
    hyperstack = metadata.pop('hyperstack', None)
    loop = metadata.pop('loop', None)
    version = metadata.pop('ImageJ', '1.11a')

    if colormapped:
        hyperstack = False
        rgb = False

    shape = imagej_shape(shape, rgb=rgb, axes=axes)
    rgb = shape[-1] in {3, 4}

    append = []
    result = [f'ImageJ={version}']
    result.append(f'images={product(shape[:-3])}')
    if hyperstack is None:
        hyperstack = True
        append.append('hyperstack=true')
    else:
        append.append(f'hyperstack={bool(hyperstack)}'.lower())
    if shape[2] > 1:
        result.append(f'channels={shape[2]}')
    if mode is None and not rgb and not colormapped:
        mode = 'grayscale'
    if hyperstack and mode:
        append.append(f'mode={mode}')
    if shape[1] > 1:
        result.append(f'slices={shape[1]}')
    if shape[0] > 1:
        result.append(f'frames={shape[0]}')
        if loop is None:
            append.append('loop=false')
    if loop is not None:
        append.append(f'loop={bool(loop)}'.lower())

    for key, value in metadata.items():
        if key not in {'images', 'channels', 'slices', 'frames', 'SCIFIO'}:
            val = str(value).lower() if isinstance(value, bool) else value
            append.append(f'{key.lower()}={val}')

    return '\n'.join(result + append + [''])


def imagej_shape(
    shape: Sequence[int],
    /,
    *,
    rgb: bool | None = None,
    axes: str | None = None,
) -> tuple[int, ...]:
    """Return shape normalized to 6D ImageJ hyperstack TZCYXS.

    Raise ValueError if not a valid ImageJ hyperstack shape or axes order.

    >>> imagej_shape((2, 3, 4, 5, 3), rgb=False)
    (2, 3, 4, 5, 3, 1)

    """
    shape = tuple(int(i) for i in shape)
    ndim = len(shape)
    if 1 > ndim > 6:
        msg = 'ImageJ hyperstack must be 2-6 dimensional'
        raise ValueError(msg)

    if axes:
        if len(axes) != ndim:
            msg = 'ImageJ hyperstack shape and axes do not match'
            raise ValueError(msg)
        i = 0
        axes = axes.upper()
        for ax in axes:
            j = 'TZCYXS'.find(ax)
            if j < i:
                msg = 'ImageJ hyperstack axes must be in TZCYXS order'
                raise ValueError(msg)
            i = j
        ndims = len(axes)
        newshape = []
        i = 0
        for ax in 'TZCYXS':
            if i < ndims and ax == axes[i]:
                newshape.append(shape[i])
                i += 1
            else:
                newshape.append(1)
        if newshape[-1] not in {1, 3, 4}:
            msg = 'ImageJ hyperstack must contain 1, 3, or 4 samples'
            raise ValueError(msg)
        return tuple(newshape)

    if rgb is None:
        rgb = shape[-1] in {3, 4} and ndim > 2
    if rgb and shape[-1] not in {3, 4}:
        msg = 'ImageJ hyperstack is not a RGB image'
        raise ValueError(msg)
    if not rgb and ndim == 6 and shape[-1] != 1:
        msg = 'ImageJ hyperstack is not a grayscale image'
        raise ValueError(msg)
    if rgb or shape[-1] == 1:
        return (1,) * (6 - ndim) + shape
    return (1,) * (5 - ndim) + shape + (1,)


def jpeg_decode_colorspace(
    photometric: int,
    planarconfig: int,
    extrasamples: tuple[int, ...],
    jfif: bool,  # noqa: FBT001
    /,
) -> tuple[int | None, int | str | None]:
    """Return JPEG and output color space for `jpeg_decode` function."""
    colorspace: int | None = None
    outcolorspace: int | str | None = None
    if extrasamples:
        pass
    elif photometric == 6:
        # YCBCR -> RGB
        outcolorspace = 2  # RGB
    elif photometric == 2:
        # RGB -> RGB
        if not jfif:
            # found in Aperio SVS
            colorspace = 2
        outcolorspace = 2
    elif photometric == 5:
        # CMYK
        outcolorspace = 4
    elif photometric > 3:
        outcolorspace = PHOTOMETRIC(photometric).name
    if planarconfig != 1:
        outcolorspace = 1  # decode separate planes to grayscale
    return colorspace, outcolorspace


def jpeg_shape(jpeg: bytes, /) -> tuple[int, int, int, int]:
    """Return bitdepth and shape of JPEG image."""
    i = 0
    while i < len(jpeg):
        marker = struct.unpack('>H', jpeg[i : i + 2])[0]
        i += 2

        if marker == 0xFFD8:
            # start of image
            continue
        if marker == 0xFFD9:
            # end of image
            break
        if 0xFFD0 <= marker <= 0xFFD7:
            # restart marker
            continue
        if marker == 0xFF01:
            # private marker
            continue

        length = struct.unpack('>H', jpeg[i : i + 2])[0]
        i += 2

        if 0xFFC0 <= marker <= 0xFFC3:
            # start of frame
            return struct.unpack('>BHHB', jpeg[i : i + 6])
        if marker == 0xFFDA:
            # start of scan
            break

        # skip to next marker
        i += length - 2

    msg = 'no SOF marker found'
    raise ValueError(msg)


def ndpi_jpeg_tile(jpeg: bytes, /) -> tuple[int, int, bytes]:
    """Return tile shape and JPEG header from JPEG with restart markers."""
    marker: int
    length: int
    factor: int
    ncomponents: int
    restartinterval: int = 0
    sofoffset: int = 0
    sosoffset: int = 0
    i: int = 0
    while i < len(jpeg):
        marker = struct.unpack('>H', jpeg[i : i + 2])[0]
        i += 2

        if marker == 0xFFD8:
            # start of image
            continue
        if marker == 0xFFD9:
            # end of image
            break
        if 0xFFD0 <= marker <= 0xFFD7:
            # restart marker
            continue
        if marker == 0xFF01:
            # private marker
            continue

        length = struct.unpack('>H', jpeg[i : i + 2])[0]
        i += 2

        if marker == 0xFFDD:
            # define restart interval
            restartinterval = struct.unpack('>H', jpeg[i : i + 2])[0]

        elif marker == 0xFFC0:
            # start of frame
            sofoffset = i + 1
            _precision, _imlength, _imwidth, ncomponents = struct.unpack(
                '>BHHB', jpeg[i : i + 6]
            )
            i += 6
            mcuwidth = 1
            mcuheight = 1
            for _ in range(ncomponents):
                _cid, factor, _table = struct.unpack('>BBB', jpeg[i : i + 3])
                i += 3
                mcuwidth = max(mcuwidth, factor >> 4)
                mcuheight = max(mcuheight, factor & 0b00001111)
            mcuwidth *= 8
            mcuheight *= 8
            i = sofoffset - 1

        elif marker == 0xFFDA:
            # start of scan
            sosoffset = i + length - 2
            break

        # skip to next marker
        i += length - 2

    if restartinterval == 0 or sofoffset == 0 or sosoffset == 0:
        msg = 'missing required JPEG markers'
        raise ValueError(msg)

    # patch jpeg header for tile size
    tilelength = mcuheight
    tilewidth = restartinterval * mcuwidth
    jpegheader = (
        jpeg[:sofoffset]
        + struct.pack('>HH', tilelength, tilewidth)
        + jpeg[sofoffset + 4 : sosoffset]
    )
    return tilelength, tilewidth, jpegheader


def shaped_description(shape: Sequence[int], /, **metadata: Any) -> str:
    """Return JSON image description from data shape and other metadata.

    Return UTF-8 encoded JSON.

    >>> shaped_description((256, 256, 3), axes='YXS')  # doctest: +SKIP
    '{"shape": [256, 256, 3], "axes": "YXS"}'

    """
    metadata.update(shape=shape)
    return json.dumps(metadata)  # .encode()


def shaped_description_metadata(description: str, /) -> dict[str, Any]:
    """Return metadata from JSON formatted image description.

    Raise ValueError if `description` is of unknown format.

    >>> description = '{"shape": [256, 256, 3], "axes": "YXS"}'
    >>> shaped_description_metadata(description)  # doctest: +SKIP
    {'shape': [256, 256, 3], 'axes': 'YXS'}
    >>> shaped_description_metadata('shape=(256, 256, 3)')
    {'shape': (256, 256, 3)}

    """
    if description[:6] == 'shape=':
        # old-style 'shaped' description; not JSON
        shape = tuple(int(i) for i in description[7:-1].split(','))
        return {'shape': shape}
    if description[:1] == '{' and description[-1:] == '}':
        # JSON description
        return json.loads(description)
    msg = 'invalid JSON image description'
    raise ValueError(msg, description)


def fluoview_description_metadata(
    description: str,
    /,
    ignoresections: Container[str] | None = None,
) -> dict[str, Any]:
    r"""Return metadata from FluoView image description.

    The FluoView image description format is unspecified. Expect failures.

    >>> descr = (
    ...     '[Intensity Mapping]\nMap Ch0: Range=00000 to 02047\n'
    ...     '[Intensity Mapping End]'
    ... )
    >>> fluoview_description_metadata(descr)
    {'Intensity Mapping': {'Map Ch0: Range': '00000 to 02047'}}

    """
    if not description.startswith('['):
        msg = 'invalid FluoView image description'
        raise ValueError(msg)
    if ignoresections is None:
        ignoresections = {'Region Info (Fields)', 'Protocol Description'}

    section: Any
    result: dict[str, Any] = {}
    sections = [result]
    comment = False
    for line in description.splitlines():
        if not comment:
            line = line.strip()  # noqa: PLW2901
        if not line:
            continue
        if line[0] == '[':
            if line[-5:] == ' End]':
                # close section
                del sections[-1]
                section = sections[-1]
                name = line[1:-5]
                if comment:
                    section[name] = '\n'.join(section[name])
                if name[:4] == 'LUT ':
                    a = numpy.array(section[name], dtype=numpy.uint8)
                    section[name] = a.reshape((-1, 3))
                continue
            # new section
            comment = False
            name = line[1:-1]
            if name[:4] == 'LUT ':
                section = []
            elif name in ignoresections:
                section = []
                comment = True
            else:
                section = {}
            sections.append(section)
            result[name] = section
            continue
        # add entry
        if comment:
            section.append(line)
            continue
        lines = line.split('=', 1)
        if len(line) == 1:
            section[lines[0].strip()] = None
            continue
        key, value = lines
        if key[:4] == 'RGB ':
            section.extend(int(rgb) for rgb in value.split())
        else:
            section[key.strip()] = astype(value.strip())
    return result


def pilatus_description_metadata(description: str, /) -> dict[str, Any]:
    """Return metadata from Pilatus image description.

    Return metadata from Pilatus pixel array detectors by Dectris, created
    by camserver or TVX software.

    >>> pilatus_description_metadata('# Pixel_size 172e-6 m x 172e-6 m')
    {'Pixel_size': (0.000172, 0.000172)}

    """
    result: dict[str, Any] = {}
    values: Any
    if not description.startswith('# '):
        return result
    for c in '#:=,()':
        description = description.replace(c, ' ')
    for lines in description.split('\n'):
        if lines[:2] != '  ':
            continue
        line = lines.split()
        name = line[0]
        if line[0] not in TIFF.PILATUS_HEADER:
            try:
                result['DateTime'] = strptime(
                    ' '.join(line), '%Y-%m-%dT%H %M %S.%f'
                )
            except Exception:
                result[name] = ' '.join(line[1:])
            continue
        indices, dtype = TIFF.PILATUS_HEADER[line[0]]
        if isinstance(indices[0], slice):
            # assumes one slice
            values = line[indices[0]]
        else:
            values = [line[i] for i in indices]
        if dtype is float and values[0] == 'not':
            values = ['NaN']
        values = tuple(dtype(v) for v in values)
        if dtype is str:
            values = ' '.join(values)
        elif len(values) == 1:
            values = values[0]
        result[name] = values
    return result


def svs_description_metadata(description: str, /) -> dict[str, Any]:
    """Return metadata from Aperio image description.

    The Aperio image description format is unspecified. Expect failures.

    >>> svs_description_metadata('Aperio Image Library v1.0')
    {'Header': 'Aperio Image Library v1.0'}

    """
    if not description.startswith('Aperio '):
        msg = 'invalid Aperio image description'
        raise ValueError(msg)
    result = {}
    items = description.split('|')
    result['Header'] = items[0]
    for item in items[1:]:
        try:
            key, value = item.split('=', maxsplit=1)
        except ValueError:
            # empty item or missing '='
            continue
        result[key.strip()] = astype(value.strip())
    return result


def stk_description_metadata(description: str, /) -> list[dict[str, Any]]:
    """Return metadata from MetaMorph image description.

    The MetaMorph image description format is unspecified. Expect failures.

    """
    description = description.strip()
    if not description:
        return []
    # try:
    #     description = bytes2str(description)
    # except UnicodeDecodeError as exc:
    #     logger().warning(
    #         '<tifffile.stk_description_metadata> raised {exc!r:.128}'
    #     )
    #     return []
    result = []
    for plane in description.split('\x00'):
        d = {}
        for line in plane.split('\r\n'):
            lines = line.split(':', 1)
            if len(lines) > 1:
                name, value = lines
                d[name.strip()] = astype(value.strip())
            else:
                value = lines[0].strip()
                if value:
                    if '' in d:
                        d[''].append(value)
                    else:
                        d[''] = [value]
        result.append(d)
    return result


def metaseries_description_metadata(description: str, /) -> dict[str, Any]:
    """Return metadata from MetaSeries image description."""
    if not description.startswith('<MetaData>'):
        msg = 'invalid MetaSeries image description'
        raise ValueError(msg)

    import uuid
    from xml.etree import ElementTree

    root = ElementTree.fromstring(description)
    types: dict[str, Callable[..., Any]] = {
        'float': float,
        'int': int,
        'bool': lambda x: asbool(x, 'on', 'off'),
        'time': lambda x: strptime(x, '%Y%m%d %H:%M:%S.%f'),
        'guid': uuid.UUID,
        # 'float-array':
        # 'colorref':
    }

    def parse(
        root: ElementTree.Element, result: dict[str, Any], /
    ) -> dict[str, Any]:
        # recursive
        for child in root:
            attrib = child.attrib
            if not attrib:
                result[child.tag] = parse(child, {})
                continue
            if 'id' in attrib:
                i = attrib['id']
                t = attrib['type']
                v = attrib['value']
                if t in types:
                    try:
                        result[i] = types[t](v)
                    except Exception:
                        result[i] = v
                else:
                    result[i] = v
        return result

    adict = parse(root, {})
    if 'Description' in adict:
        adict['Description'] = adict['Description'].replace('&#13;&#10;', '\n')
    return adict


def scanimage_description_metadata(description: str, /) -> Any:
    """Return metadata from ScanImage image description."""
    return matlabstr2py(description)


def scanimage_artist_metadata(artist: str, /) -> dict[str, Any] | None:
    """Return metadata from ScanImage artist tag."""
    try:
        return json.loads(artist)
    except ValueError as exc:
        logger().warning(
            f'<tifffile.scanimage_artist_metadata> raised {exc!r:.128}'
        )
    return None


def olympus_ini_metadata(inistr: str, /) -> dict[str, Any]:
    """Return OlympusSIS metadata from INI string.

    No specification is available.

    """

    def keyindex(key: str, /) -> tuple[str, int]:
        # split key into name and index
        index = 0
        i = len(key.rstrip('0123456789'))
        if i < len(key):
            index = int(key[i:]) - 1
            key = key[:i]
        return key, index

    value: Any
    result: dict[str, Any] = {}
    bands: list[dict[str, Any]] = []
    zpos: list[Any] | None = None
    tpos: list[Any] | None = None
    for line in inistr.splitlines():
        line = line.strip()  # noqa: PLW2901
        if line == '' or line[0] == ';':
            continue
        if line[0] == '[' and line[-1] == ']':
            section_name = line[1:-1]
            result[section_name] = section = {}
            if section_name == 'Dimension':
                result['axes'] = axes = []
                result['shape'] = shape = []
            elif section_name == 'ASD':
                result[section_name] = []
            elif section_name == 'Z':
                if 'Dimension' in result:
                    result[section_name]['ZPos'] = zpos = []
            elif section_name == 'Time':
                if 'Dimension' in result:
                    result[section_name]['TimePos'] = tpos = []
            elif section_name == 'Band':
                nbands = result['Dimension']['Band']
                bands = [{'LUT': []} for _ in range(nbands)]
                result[section_name] = bands
                iband = 0
        else:
            key, value = line.split('=')
            if value.strip() == '':
                value = None
            elif ',' in value:
                value = tuple(astype(v) for v in value.split(','))
            else:
                value = astype(value)

            if section_name == 'Dimension':
                section[key] = value
                axes.append(key)
                shape.append(value)
            elif section_name == 'ASD':
                if key == 'Count':
                    result['ASD'] = [{}] * value
                else:
                    key, index = keyindex(key)
                    result['ASD'][index][key] = value
            elif section_name == 'Band':
                if key[:3] == 'LUT':
                    lut = bands[iband]['LUT']
                    value = struct.pack('<I', value)
                    lut.append(
                        [ord(value[0:1]), ord(value[1:2]), ord(value[2:3])]
                    )
                else:
                    key, iband = keyindex(key)
                    bands[iband][key] = value
            elif key[:4] == 'ZPos' and zpos is not None:
                zpos.append(value)
            elif key[:7] == 'TimePos' and tpos is not None:
                tpos.append(value)
            else:
                section[key] = value

    if 'axes' in result:
        sisaxes = {'Band': 'C'}
        axes = []
        shape = []
        for i, x in zip(result['shape'], result['axes'], strict=True):
            if i > 1:
                axes.append(sisaxes.get(x, x[0].upper()))
                shape.append(i)
        result['axes'] = ''.join(axes)
        result['shape'] = tuple(shape)
    with contextlib.suppress(TypeError, ValueError):
        result['Z']['ZPos'] = numpy.array(
            result['Z']['ZPos'][: result['Dimension']['Z']], numpy.float64
        )
    with contextlib.suppress(TypeError, ValueError):
        result['Time']['TimePos'] = numpy.array(
            result['Time']['TimePos'][: result['Dimension']['Time']],
            numpy.int32,
        )
    for band in bands:
        band['LUT'] = numpy.array(band['LUT'], numpy.uint8)
    return result


def astrotiff_description_metadata(
    description: str, /, sep: str = ':'
) -> dict[str, Any]:
    """Return metadata from AstroTIFF image description."""
    logmsg = '<tifffile.astrotiff_description_metadata> '
    counts: dict[str, int] = {}
    result: dict[str, Any] = {}
    value: Any
    for line in description.splitlines():
        line = line.strip()  # noqa: PLW2901
        if not line:
            continue

        key = line[:8].strip()
        value = line[8:]

        if not value.startswith('='):
            # for example, COMMENT or HISTORY
            if key + f'{sep}0' not in result:
                result[key + f'{sep}0'] = value
                counts[key] = 1
            else:
                result[key + f'{sep}{counts[key]}'] = value
                counts[key] += 1
            continue

        value = value[1:]
        if '/' in value:
            value, comment = value.split('/', 1)
            comment = comment.strip()
        else:
            comment = ''
        value = value.strip()

        if not value:
            # undefined
            value = None
        elif value[0] == "'":
            # string
            if len(value) < 2:
                logger().warning(logmsg + f'{key}: invalid string {value!r}')
                continue
            if value[-1] == "'":
                value = value[1:-1]
            else:
                # string containing '/'
                if not ("'" in comment and '/' in comment):
                    logger().warning(
                        logmsg + f'{key}: invalid string {value!r}'
                    )
                    continue
                value, comment = line[9:].strip()[1:].split("'", 1)
                comment = comment.split('/', 1)[-1].strip()
            # TODO: string containing single quote '
        elif value[0] == '(' and value[-1] == ')':
            # complex number
            value = value[1:-1]
            dtype = float if '.' in value else int
            value = tuple(dtype(v.strip()) for v in value.split(','))
        elif value == 'T':
            value = True
        elif value == 'F':
            value = False
        elif '.' in value:
            value = float(value)
        else:
            try:
                value = int(value)
            except Exception:
                logger().warning(logmsg + f'{key}: invalid value {value!r}')
                continue

        if key in result:
            logger().warning(logmsg + f'{key}: duplicate key')

        result[key] = value
        if comment:
            result[key + f'{sep}COMMENT'] = comment
            if comment[0] == '[' and ']' in comment:
                result[key + f'{sep}UNIT'] = comment[1:].split(']', 1)[0]

    return result


def streak_description_metadata(
    description: str, fh: FileHandle, /
) -> dict[str, Any]:
    """Return metadata from Hamamatsu streak image description."""
    section_pattern = re.compile(
        r'\[([a-zA-Z0-9 _\-\.]+)\],([^\[]*)', re.DOTALL
    )
    properties_pattern = re.compile(
        r'([a-zA-Z0-9 _\-\.]+)=(\"[^\"]*\"|[\+\-0-9\.]+|[^,]*)'
    )
    result: dict[str, Any] = {}
    for section, values in section_pattern.findall(description.strip()):
        properties = {}
        for item in properties_pattern.findall(values):
            key, value = item
            value = value.strip()
            if not value or value == '"':
                value = None
            elif value[0] == '"' and value[-1] == '"':
                value = value[1:-1]
            if ',' in value:
                with contextlib.suppress(ValueError):
                    value = tuple(
                        (
                            float(v)
                            if '.' in value
                            else int(v[1:] if v[0] == '#' else v)
                        )
                        for v in value.split(',')
                    )
            elif '.' in value:
                with contextlib.suppress(ValueError):
                    value = float(value)
            else:
                with contextlib.suppress(ValueError):
                    value = int(value)
            properties[key] = value
        result[section] = properties

    if not fh.closed:
        pos = fh.tell()
        for scaling in ('ScalingXScaling', 'ScalingYScaling'):
            try:
                offset, count = result['Scaling'][scaling + 'File']
                fh.seek(offset)
                result['Scaling'][scaling] = fh.read_array(
                    dtype='<f4', count=count
                )
            except Exception:  # noqa: S110
                pass
        fh.seek(pos)

    return result


def eer_xml_metadata(xmlstr: str, /) -> dict[str, Any]:
    """Return metadata from EER XML tag values."""
    from xml.etree import ElementTree

    value: Any
    root = ElementTree.fromstring(xmlstr)
    result = {}
    for item in root.findall('./item'):
        key = item.attrib['name']
        value = item.text
        if value is None:
            continue
        if value == 'Yes':
            value = True
        elif value == 'No':
            value = False
        elif key == 'timestamp':
            # ISO 8601
            with contextlib.suppress(TypeError, ValueError):
                value = DateTime.fromisoformat(value)
        else:
            value = astype(value)
        result[key] = value
        if 'unit' in item.attrib:
            result[key + '.unit'] = item.attrib['unit']
    return result

