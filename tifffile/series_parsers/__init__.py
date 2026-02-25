# series_parsers/__init__.py

"""Format-specific series parsers."""

from .generic import GenericSeriesParser, ShapedSeriesParser, UniformSeriesParser
from .imagej import ImageJSeriesParser
from .lsm import LsmSeriesParser
from .microscopy import (
    FluoViewSeriesParser,
    MmstackSeriesParser,
    NdtiffSeriesParser,
    NihSeriesParser,
    ScanImageSeriesParser,
    StkSeriesParser,
)
from .ome import OmeSeriesParser
from .other import AvsSeriesParser, EerSeriesParser, MdgelSeriesParser, SisSeriesParser
from .pathology import (
    BifSeriesParser,
    NdpiSeriesParser,
    PhilipsSeriesParser,
    QpiSeriesParser,
    ScnSeriesParser,
    SvsSeriesParser,
)

__all__ = [
    'AvsSeriesParser',
    'BifSeriesParser',
    'EerSeriesParser',
    'FluoViewSeriesParser',
    'GenericSeriesParser',
    'ImageJSeriesParser',
    'LsmSeriesParser',
    'MdgelSeriesParser',
    'MmstackSeriesParser',
    'NdpiSeriesParser',
    'NdtiffSeriesParser',
    'NihSeriesParser',
    'OmeSeriesParser',
    'PhilipsSeriesParser',
    'QpiSeriesParser',
    'ScanImageSeriesParser',
    'ScnSeriesParser',
    'ShapedSeriesParser',
    'SisSeriesParser',
    'StkSeriesParser',
    'SvsSeriesParser',
    'UniformSeriesParser',
]
