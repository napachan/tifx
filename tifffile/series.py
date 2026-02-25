# series.py

"""Series parser infrastructure and registry."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .tifffile import TiffFile, TiffPageSeries

__all__ = [
    'SeriesParser',
    'SeriesParserRegistry',
    'get_default_registry',
]


class SeriesParser:
    """Base class for format-specific series parsers.

    Subclasses implement :py:meth:`parse` to detect and construct
    :py:class:`TiffPageSeries` from a :py:class:`TiffFile`.

    """

    kind: str = ''
    """Format identifier matching the ``is_<kind>`` flag on TiffFile."""

    def can_parse(self, tiff: TiffFile) -> bool:
        """Return whether this parser can handle the given file."""
        return bool(getattr(tiff, 'is_' + self.kind, False))

    def on_failure(self, tiff: TiffFile) -> bool:
        """Handle parse failure.

        Called when :py:meth:`parse` returns None or empty list.
        Return True to continue trying other parsers, False to stop.

        """
        return False

    def parse(self, tiff: TiffFile) -> list[TiffPageSeries] | None:
        """Parse series from the given TiffFile.

        Returns list of TiffPageSeries or None if parsing fails.

        """
        raise NotImplementedError


class SeriesParserRegistry:
    """Priority-ordered registry for series parsers."""

    def __init__(self) -> None:
        self._parsers: list[SeriesParser] = []
        self._generic: SeriesParser | None = None

    def register(self, parser: SeriesParser) -> None:
        """Register a format-specific parser."""
        self._parsers.append(parser)

    def register_generic(self, parser: SeriesParser) -> None:
        """Register the generic fallback parser."""
        self._generic = parser

    def get_parser(self, kind: str) -> SeriesParser | None:
        """Return parser by kind name."""
        for parser in self._parsers:
            if parser.kind == kind:
                return parser
        if self._generic is not None and self._generic.kind == kind:
            return self._generic
        return None

    def parse(self, tiff: TiffFile) -> list[TiffPageSeries]:
        """Parse series using registered parsers in priority order."""
        series: list[TiffPageSeries] | None = None

        for parser in self._parsers:
            if not parser.can_parse(tiff):
                continue
            series = parser.parse(tiff)
            if not series:
                if parser.on_failure(tiff):
                    continue
            break

        if not series and self._generic is not None:
            series = self._generic.parse(tiff)

        assert series is not None
        return series


_default_registry: SeriesParserRegistry | None = None


def get_default_registry() -> SeriesParserRegistry:
    """Return the default series parser registry, creating it if needed."""
    global _default_registry
    if _default_registry is None:
        _default_registry = _create_default_registry()
    return _default_registry


def _create_default_registry() -> SeriesParserRegistry:
    """Create and populate the default series parser registry."""
    from .series_parsers import (
        AvsSeriesParser,
        BifSeriesParser,
        EerSeriesParser,
        FluoViewSeriesParser,
        GenericSeriesParser,
        ImageJSeriesParser,
        LsmSeriesParser,
        MdgelSeriesParser,
        MmstackSeriesParser,
        NdpiSeriesParser,
        NdtiffSeriesParser,
        NihSeriesParser,
        OmeSeriesParser,
        PhilipsSeriesParser,
        QpiSeriesParser,
        ScanImageSeriesParser,
        ScnSeriesParser,
        ShapedSeriesParser,
        SisSeriesParser,
        StkSeriesParser,
        SvsSeriesParser,
        UniformSeriesParser,
    )

    registry = SeriesParserRegistry()
    # Order must match original dispatch order in TiffFile.series
    for parser_class in (
        ShapedSeriesParser,
        LsmSeriesParser,
        MmstackSeriesParser,
        OmeSeriesParser,
        ImageJSeriesParser,
        NdtiffSeriesParser,
        FluoViewSeriesParser,
        StkSeriesParser,
        SisSeriesParser,
        SvsSeriesParser,
        ScnSeriesParser,
        QpiSeriesParser,
        NdpiSeriesParser,
        BifSeriesParser,
        AvsSeriesParser,
        EerSeriesParser,
        PhilipsSeriesParser,
        ScanImageSeriesParser,
        # IndicaSeriesParser,  # TODO: rewrite _series_indica()
        NihSeriesParser,
        MdgelSeriesParser,
        UniformSeriesParser,
    ):
        registry.register(parser_class())
    registry.register_generic(GenericSeriesParser())
    return registry
