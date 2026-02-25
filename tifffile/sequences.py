# tifffile/sequences.py

"""Sequence classes for tifffile.

Classes: TiffPageSeries, FileSequence, TiffSequence, TiledSequence.

"""

from __future__ import annotations

import io
import os
import warnings
from collections.abc import Callable, Iterable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from functools import cached_property
from typing import TYPE_CHECKING, cast, final, overload

import numpy

from .utils import (
    natural_sorted,
    product,
    snipstr,
    squeeze_axes,
    unique_strings,
)

if TYPE_CHECKING:
    from collections.abc import Iterator
    from types import TracebackType
    from typing import IO, Any, Self

    from numpy.typing import DTypeLike, NDArray

    from .tifffile import TiffFile, TiffFrame, TiffPage
    from .zarr import ZarrFileSequenceStore, ZarrTiffStore

    OutputType = str | IO[bytes] | NDArray[Any] | None

if os.environ.get('TIFFFILE_NO_CPP'):
    _cpp_sel_to_indices = None
else:
    try:
        from ._tifffile_ext import (
            selection_to_page_indices as _cpp_sel_to_indices,
        )
    except ImportError:
        _cpp_sel_to_indices = None


def _mmap_read_contiguous(
    mv: memoryview,
    dataoffset: int,
    dtype_char: str,
    byteorder: str,
    total_pages: int,
    page_elems: int,
    page_selection: Any,
    spatial_shape: tuple[int, ...] | None = None,
    y_slice: slice | None = None,
    spatial_sel: tuple[int | slice, ...] | None = None,
) -> NDArray[Any] | None:
    """Read selected pages from contiguous mmap'd series, with byteswap.

    Parameters:
        mv: Memory-mapped view of the file.
        dataoffset: Byte offset to start of contiguous series data.
        dtype_char: Single-char dtype code (e.g. 'f', 'u1').
        byteorder: File byte order ('>' or '<').
        total_pages: Total number of pages in the series.
        page_elems: Number of elements per page.
        page_selection: Page selector — slice, range, or integer array
            for fancy indexing.
        spatial_shape: If provided, reshape each page to this shape before
            applying spatial selections.
        y_slice: Optional slice along the first spatial axis (rows).
            Applied together with page selection for efficient combined
            indexing.  Requires spatial_shape.
        spatial_sel: Optional full spatial selection tuple.  When given
            with spatial_shape, creates a 3D+ mmap view and applies the
            full selection (page + spatial) in one indexing operation.
            This is used for non-Y-only selections (e.g. fixed X column).
            Mutually exclusive with y_slice.

    Returns:
        NumPy array with native byte order, or None on failure.

    """
    try:
        dtype = numpy.dtype(byteorder + dtype_char)
        nbytes = total_pages * page_elems * dtype.itemsize
        if dataoffset + nbytes > len(mv):
            return None
        buf = mv[dataoffset : dataoffset + nbytes]

        if spatial_shape is not None and spatial_sel is not None:
            # Full spatial selection: reshape to (pages, *spatial) and
            # apply page + spatial indexing.
            all_data = numpy.frombuffer(buf, dtype).reshape(
                total_pages, *spatial_shape
            )

            # For large strided access (e.g. column extraction),
            # multithreaded copy overlaps memory latency across threads.
            # Convert page_selection to a range if possible.
            if isinstance(page_selection, slice):
                rng: range | None = range(
                    *page_selection.indices(total_pages)
                )
            elif isinstance(page_selection, range):
                rng = page_selection
            elif (
                isinstance(page_selection, numpy.ndarray)
                and page_selection.ndim == 1
                and len(page_selection) > 1
            ):
                d = int(page_selection[1] - page_selection[0])
                if d > 0 and numpy.all(
                    numpy.diff(page_selection) == d
                ):
                    rng = range(
                        int(page_selection[0]),
                        int(page_selection[-1]) + d,
                        d,
                    )
                else:
                    rng = None
            else:
                rng = None

            n_selected = len(rng) if rng is not None else 0

            if rng is not None and n_selected >= 500:
                import concurrent.futures
                import os as _os

                test = all_data[rng[0]][spatial_sel]
                result = numpy.empty(
                    (n_selected, *test.shape), dtype=dtype
                )
                nw = min(8, _os.cpu_count() or 4)
                chunk = max(100, n_selected // (nw * 4))

                def _copy(se: tuple[int, int]) -> None:
                    s, e = se
                    src = slice(rng[s], rng[e - 1] + 1, rng.step)
                    result[s:e] = all_data[src][
                        (slice(None),) + spatial_sel
                    ]

                tasks = [
                    (s, min(s + chunk, n_selected))
                    for s in range(0, n_selected, chunk)
                ]
                with concurrent.futures.ThreadPoolExecutor(
                    max_workers=nw
                ) as ex:
                    list(ex.map(_copy, tasks))
            else:
                idx = (page_selection,) + spatial_sel
                result = numpy.ascontiguousarray(all_data[idx])
        elif spatial_shape is not None and y_slice is not None:
            # Reshape to (total_pages, *spatial_shape) for sub-page access
            all_data = numpy.frombuffer(buf, dtype).reshape(
                total_pages, *spatial_shape
            )
            # Combined fancy index (pages) + slice (rows) in one pass
            result = numpy.array(all_data[page_selection, y_slice])
        else:
            all_data = numpy.frombuffer(buf, dtype).reshape(
                total_pages, page_elems
            )
            sliced = all_data[page_selection]
            result = numpy.ascontiguousarray(sliced)

        if not dtype.isnative:
            result = result.byteswap(True)
            result = result.view(result.dtype.newbyteorder('='))
        return result
    except Exception:
        return None


def _compute_strip_overlap(
    y_start: int,
    y_stop: int,
    rowsperstrip: int,
    imagelength: int,
    samplesperpixel: int = 1,
    planarconfig: int = 1,
    imagedepth: int = 1,
) -> tuple[list[int], int, int]:
    """Compute which strip indices overlap a Y range.

    Parameters:
        y_start: Start row of selection.
        y_stop: Stop row of selection (exclusive).
        rowsperstrip: Rows per strip.
        imagelength: Total image height.
        samplesperpixel: Samples per pixel.
        planarconfig: 1=contig, 2=separate.
        imagedepth: Image depth (for volumetric images).

    Returns:
        Tuple of (strip_indices, strip_y_start, first_strip_row):
        - strip_indices: List of segment indices to decode.
        - strip_y_start: Row offset of first strip in the output crop.
        - first_strip_row: Row of first strip in the image.

    """
    nstrips_y = (imagelength + rowsperstrip - 1) // rowsperstrip
    first_strip = y_start // rowsperstrip
    last_strip = (y_stop - 1) // rowsperstrip  # inclusive

    strip_indices: list[int] = []
    nsamples = samplesperpixel if planarconfig == 2 else 1

    for s in range(nsamples):
        for d in range(imagedepth):
            for strip_y in range(first_strip, last_strip + 1):
                idx = s * (imagedepth * nstrips_y) + d * nstrips_y + strip_y
                strip_indices.append(idx)

    first_strip_row = first_strip * rowsperstrip
    strip_y_start = y_start - first_strip_row

    return strip_indices, strip_y_start, first_strip_row


def _compute_tile_overlap(
    y_start: int,
    y_stop: int,
    x_start: int,
    x_stop: int,
    tilelength: int,
    tilewidth: int,
    imagelength: int,
    imagewidth: int,
    samplesperpixel: int = 1,
    planarconfig: int = 1,
    imagedepth: int = 1,
    tiledepth: int = 1,
) -> tuple[list[int], int, int, int, int]:
    """Compute which tile indices overlap a Y,X region.

    Parameters:
        y_start, y_stop: Row range of selection.
        x_start, x_stop: Column range of selection.
        tilelength: Tile height.
        tilewidth: Tile width.
        imagelength: Total image height.
        imagewidth: Total image width.
        samplesperpixel: Samples per pixel.
        planarconfig: 1=contig, 2=separate.
        imagedepth: Image depth.
        tiledepth: Tile depth.

    Returns:
        Tuple of (tile_indices, tile_y_start, tile_x_start,
                  first_tile_row, first_tile_col):
        - tile_indices: List of segment indices to decode.
        - tile_y_start: Row offset within the first tile row.
        - tile_x_start: Column offset within the first tile column.
        - first_tile_row: Row of first tile in the image.
        - first_tile_col: Column of first tile in the image.

    """
    ntiles_y = (imagelength + tilelength - 1) // tilelength
    ntiles_x = (imagewidth + tilewidth - 1) // tilewidth
    ndepth = (imagedepth + tiledepth - 1) // tiledepth

    first_tile_y = y_start // tilelength
    last_tile_y = (y_stop - 1) // tilelength
    first_tile_x = x_start // tilewidth
    last_tile_x = (x_stop - 1) // tilewidth

    tile_indices: list[int] = []
    nsamples = samplesperpixel if planarconfig == 2 else 1

    for s in range(nsamples):
        for d in range(ndepth):
            for ty in range(first_tile_y, last_tile_y + 1):
                for tx in range(first_tile_x, last_tile_x + 1):
                    idx = (
                        s * (ndepth * ntiles_y * ntiles_x)
                        + d * (ntiles_y * ntiles_x)
                        + ty * ntiles_x
                        + tx
                    )
                    tile_indices.append(idx)

    first_tile_row = first_tile_y * tilelength
    first_tile_col = first_tile_x * tilewidth
    tile_y_start = y_start - first_tile_row
    tile_x_start = x_start - first_tile_col

    return (
        tile_indices, tile_y_start, tile_x_start,
        first_tile_row, first_tile_col,
    )


def _split_axes(
    shape: tuple[int, ...], axes: str, page_size: int
) -> tuple[tuple[int, ...], str, tuple[int, ...], str, int]:
    """Split series dimensions into frame (across pages) and spatial (in page).

    Parameters:
        shape: Full series shape (unsqueezed).
        axes: Full series axes string (unsqueezed).
        page_size: Number of elements per page (keyframe.size).

    Returns:
        Tuple of (frame_shape, frame_axes, spatial_shape, spatial_axes,
        split_index).

    """
    p = 1
    for i in range(len(shape) - 1, -1, -1):
        p *= shape[i]
        if p == page_size:
            return shape[:i], axes[:i], shape[i:], axes[i:], i
    # Fallback: all dimensions are spatial (single-page series)
    return (), '', shape, axes, 0


def _normalize_selection(
    selection: dict[str, int | slice] | tuple[int | slice, ...],
    shape: tuple[int, ...],
    axes: str,
    split_index: int,
) -> tuple[dict[str, int | slice], tuple[int | slice, ...] | None]:
    """Normalize user selection into frame and spatial parts.

    Parameters:
        selection: Dict mapping axis codes/names to int or slice, or
            positional tuple matching all axes.
        shape: Full series shape (unsqueezed).
        axes: Full series axes string (unsqueezed).
        split_index: Index separating frame and spatial dimensions.

    Returns:
        Tuple of (frame_selection_dict, spatial_selection_tuple_or_None).

    """
    from .tifffile import TIFF

    if isinstance(selection, dict):
        codes = TIFF.AXES_CODES
        frame_axes = axes[:split_index]
        spatial_axes = axes[split_index:]

        frame_sel: dict[str, int | slice] = {}
        spatial_sel_dict: dict[str, int | slice] = {}

        for key, val in selection.items():
            # Resolve key to axis code
            if len(key) == 1 and key.upper() in axes:
                code = key.upper()
            elif key.lower() in codes:
                code = codes[key.lower()]
            else:
                msg = f'unknown axis {key!r} for axes {axes!r}'
                raise ValueError(msg)

            if code in frame_axes:
                frame_sel[code] = val
            elif code in spatial_axes:
                spatial_sel_dict[code] = val
            else:
                msg = f'axis {key!r} ({code}) not in series axes {axes!r}'
                raise ValueError(msg)

        spatial_sel: tuple[int | slice, ...] | None = None
        if spatial_sel_dict:
            spatial_sel = tuple(
                spatial_sel_dict.get(ax, slice(None))
                for ax in spatial_axes
            )

        return frame_sel, spatial_sel

    elif isinstance(selection, tuple):
        if len(selection) > len(shape):
            msg = (
                f'selection has {len(selection)} elements but series '
                f'has {len(shape)} dimensions'
            )
            raise ValueError(msg)

        frame_sel_dict: dict[str, int | slice] = {}
        for i, val in enumerate(selection[:split_index]):
            frame_sel_dict[axes[i]] = val

        spatial_sel_tuple: tuple[int | slice, ...] | None = None
        if len(selection) > split_index:
            spatial_sel_tuple = tuple(selection[split_index:])

        return frame_sel_dict, spatial_sel_tuple

    else:
        msg = f'selection must be a dict or tuple, not {type(selection)}'
        raise TypeError(msg)


def _selection_to_page_indices(
    frame_sel: dict[str, int | slice],
    frame_shape: tuple[int, ...],
    frame_axes: str,
) -> tuple[NDArray[Any], tuple[int, ...]]:
    """Compute flat page indices and output frame shape from frame selection.

    Parameters:
        frame_sel: Dict mapping axis codes to int or slice.
        frame_shape: Shape of frame dimensions.
        frame_axes: Axes string for frame dimensions.

    Returns:
        Tuple of (flat_page_indices, output_frame_shape).
        output_frame_shape excludes axes selected by integer (squeezed out).

    """
    if _cpp_sel_to_indices is not None:
        try:
            return _cpp_sel_to_indices(frame_shape, frame_sel, frame_axes)
        except Exception:
            pass

    ranges: list[NDArray[Any]] = []
    out_shape: list[int] = []

    for ax, size in zip(frame_axes, frame_shape):
        sel = frame_sel.get(ax, slice(None))
        if isinstance(sel, (int, numpy.integer)):
            idx = int(sel)
            if idx < 0:
                idx %= size
            if idx < 0 or idx >= size:
                msg = f'index {sel} out of range for axis {ax!r} with size {size}'
                raise IndexError(msg)
            ranges.append(numpy.array([idx]))
            # Integer selection: squeezed out of output shape
        elif isinstance(sel, slice):
            indices = range(*sel.indices(size))
            if not indices:
                msg = f'empty selection for axis {ax!r}'
                raise ValueError(msg)
            ranges.append(numpy.array(indices))
            out_shape.append(len(indices))
        else:
            msg = f'selection value must be int or slice, not {type(sel)}'
            raise TypeError(msg)

    if not ranges:
        return numpy.array([0]), tuple(out_shape)

    grids = numpy.meshgrid(*ranges, indexing='ij')
    coords = tuple(g.ravel() for g in grids)
    flat_indices = numpy.ravel_multi_index(coords, frame_shape)

    return flat_indices, tuple(out_shape)


@final
class TiffPageSeries(Sequence['TiffPage | TiffFrame | None']):
    """Sequence of TIFF pages making up multi-dimensional image.

    Many TIFF based formats, such as OME-TIFF, use series of TIFF pages to
    store chunks of larger, multi-dimensional images.
    The image shape and position of chunks in the multi-dimensional image is
    defined in format-specific metadata.
    All pages in a series must have the same :py:meth:`TiffPage.hash`,
    that is, the same shape, data type, and storage properties.
    Items of a series may be None (missing) or instances of
    :py:class:`TiffPage` or :py:class:`TiffFrame`, possibly belonging to
    different files.

    Parameters:
        pages:
            List of TiffPage, TiffFrame, or None.
            The file handles of TiffPages or TiffFrames may not be open.
        shape:
            Shape of image array in series.
        dtype:
            Data type of image array in series.
        axes:
            Character codes for dimensions in shape.
            Length must match shape.
        attr:
            Arbitrary metadata associated with series.
        index:
            Index of series in multi-series files.
        parent:
            TiffFile instance series belongs to.
        name:
            Name of series.
        kind:
            Nature of series, such as, 'ome' or 'imagej'.
        truncated:
            Series is truncated, for example, ImageJ hyperstack > 4 GB.
        multifile:
            Series contains pages from multiple files.
        squeeze:
            Remove length-1 dimensions (except X and Y) from shape and axes
            by default.
        transform:
            Function to transform image data after decoding.

    """

    levels: list[TiffPageSeries]
    """Multi-resolution, pyramidal levels. ``levels[0] is self``."""

    parent: TiffFile | None
    """TiffFile instance series belongs to."""

    keyframe: TiffPage
    """TiffPage of series."""

    dtype: numpy.dtype[Any]
    """Data type (native byte order) of image array in series."""

    kind: str
    """Nature of series."""

    name: str
    """Name of image series from metadata."""

    transform: Callable[[NDArray[Any]], NDArray[Any]] | None
    """Function to transform image data after decoding."""

    is_multifile: bool
    """Series contains pages from multiple files."""

    is_truncated: bool
    """Series contains single page describing multi-dimensional image."""

    _pages: list[TiffPage | TiffFrame | None]
    # List of pages in series.
    # Might contain only first page of contiguous series

    _index: int  # index of series in multi-series files
    _squeeze: bool
    _axes: str
    _axes_squeezed: str
    _shape: tuple[int, ...]
    _shape_squeezed: tuple[int, ...]
    _len: int
    _attr: dict[str, Any]

    def __init__(
        self,
        pages: Sequence[TiffPage | TiffFrame | None],
        /,
        shape: Sequence[int] | None = None,
        dtype: DTypeLike | None = None,
        axes: str | None = None,
        *,
        attr: dict[str, Any] | None = None,
        coords: Mapping[str, NDArray[Any] | None] | None = None,
        index: int | None = None,
        parent: TiffFile | None = None,
        name: str | None = None,
        kind: str | None = None,
        truncated: bool = False,
        multifile: bool = False,
        squeeze: bool = True,
        transform: Callable[[NDArray[Any]], NDArray[Any]] | None = None,
    ) -> None:
        self._shape = ()
        self._shape_squeezed = ()
        self._axes = ''
        self._axes_squeezed = ''
        self._attr = {} if attr is None else dict(attr)

        self._index = int(index) if index else 0
        self._pages = list(pages)
        self.levels = [self]
        npages = len(self._pages)
        try:
            # find open TiffPage
            keyframe = next(
                p.keyframe
                for p in self._pages
                if p is not None
                and p.keyframe is not None
                and not p.keyframe.parent.filehandle.closed
            )
        except StopIteration:
            keyframe = next(
                p.keyframe
                for p in self._pages
                if p is not None and p.keyframe is not None
            )

        if shape is None:
            shape = keyframe.shape
        if axes is None:
            axes = keyframe.axes
        if dtype is None:
            dtype = keyframe.dtype

        self.dtype = numpy.dtype(dtype)
        self.kind = kind if kind else ''
        self.name = name if name else ''
        self.transform = transform
        self.keyframe = keyframe
        self.is_multifile = bool(multifile)
        self.is_truncated = bool(truncated)

        if parent is not None:
            self.parent = parent
        elif self._pages:
            self.parent = self.keyframe.parent
        else:
            self.parent = None

        self._set_dimensions(shape, axes, coords, squeeze)

        if not truncated and npages == 1:
            s = product(keyframe.shape)
            if s > 0:
                self._len = int(product(self.shape) // s)
            else:
                self._len = npages
        else:
            self._len = npages

    def _set_dimensions(
        self,
        shape: Sequence[int],
        axes: str,
        coords: Mapping[str, NDArray[Any] | None] | None = None,
        squeeze: bool = True,  # noqa: FBT001, FBT002
        /,
    ) -> None:
        """Set shape, axes, and coords."""
        self._squeeze = bool(squeeze)
        self._shape = tuple(shape)
        self._axes = axes
        self._shape_squeezed, self._axes_squeezed, _ = squeeze_axes(
            shape, axes
        )

    @property
    def shape(self) -> tuple[int, ...]:
        """Shape of image array in series."""
        return self._shape_squeezed if self._squeeze else self._shape

    @property
    def axes(self) -> str:
        """Character codes for dimensions in image array."""
        return self._axes_squeezed if self._squeeze else self._axes

    @property
    def coords(self) -> dict[str, NDArray[Any]]:
        """Ordered map of dimension names to coordinate arrays."""
        raise NotImplementedError
        # return {
        #     name: numpy.arange(size)
        #     for name, size in zip(self.dims, self.shape)
        # }

    def get_shape(
        self, squeeze: bool | None = None  # noqa: FBT001
    ) -> tuple[int, ...]:
        """Return default, squeezed, or expanded shape of series.

        Parameters:
            squeeze: Remove length-1 dimensions from shape.

        """
        if squeeze is None:
            squeeze = self._squeeze
        return self._shape_squeezed if squeeze else self._shape

    def get_axes(self, squeeze: bool | None = None) -> str:  # noqa: FBT001
        """Return default, squeezed, or expanded axes of series.

        Parameters:
            squeeze: Remove length-1 dimensions from axes.

        """
        if squeeze is None:
            squeeze = self._squeeze
        return self._axes_squeezed if squeeze else self._axes

    def get_coords(
        self, squeeze: bool | None = None  # noqa: FBT001
    ) -> dict[str, NDArray[Any]]:
        """Return default, squeezed, or expanded coords of series.

        Parameters:
            squeeze: Remove length-1 dimensions from coords.

        """
        raise NotImplementedError

    def asarray(
        self,
        *,
        level: int | None = None,
        selection: (
            dict[str, int | slice] | tuple[int | slice, ...] | None
        ) = None,
        **kwargs: Any,
    ) -> NDArray[Any]:
        """Return images from series of pages as NumPy array.

        Parameters:
            level:
                Pyramid level to return.
                By default, the base layer is returned.
            selection:
                Subset of multi-dimensional image to return.
                If a dict, maps axis codes (e.g. ``'T'``, ``'Z'``) or
                dimension names (e.g. ``'time'``, ``'depth'``) to integer
                indices or slices.
                If a tuple, positional selections matching the series axes
                order.
                Only pages matching the selection are read from disk.
                Spatial dimensions (within a page) are sliced in memory.
            **kwargs:
                Additional arguments passed to :py:meth:`TiffFile.asarray`.

        """
        if self.parent is None:
            msg = 'no parent'
            raise ValueError(msg)
        if level is not None:
            return self.levels[level].asarray(
                selection=selection, **kwargs
            )
        if selection is not None:
            return self._asarray_selection(selection, **kwargs)
        result = self.parent.asarray(series=self, **kwargs)
        if self.transform is not None:
            result = self.transform(result)
        return result

    def _asarray_selection(
        self,
        selection: dict[str, int | slice] | tuple[int | slice, ...],
        *,
        squeeze: bool | None = None,
        out: OutputType = None,
        device: str | None = None,
        maxworkers: int | None = None,
        buffersize: int | None = None,
    ) -> NDArray[Any]:
        """Return selected subset of multi-dimensional image as NumPy array.

        Only pages matching the selection are read from disk.

        """
        from .tifffile import stack_pages

        assert self.parent is not None

        # 1. Expand tuple selection to match unsqueezed shape.
        #    Users provide tuples positional to the squeezed output shape,
        #    but _normalize_selection maps positions to unsqueezed axes.
        #    Track original length for squeeze decisions later.
        original_tuple_len = len(selection) if isinstance(selection, tuple) else 0
        if isinstance(selection, tuple):
            do_squeeze = self._squeeze if squeeze is None else squeeze
            if do_squeeze:
                from .utils import squeeze_axes

                _, _, squeezed_mask = squeeze_axes(
                    self._shape, self._axes
                )
                if not all(squeezed_mask):
                    expanded: list[int | slice] = []
                    sel_idx = 0
                    for kept in squeezed_mask:
                        if kept:
                            if sel_idx < len(selection):
                                expanded.append(selection[sel_idx])
                            else:
                                expanded.append(slice(None))
                            sel_idx += 1
                        else:
                            expanded.append(slice(None))
                    selection = tuple(expanded)

        # 2. Split axes using unsqueezed shape/axes
        frame_shape, frame_axes, spatial_shape, spatial_axes, split_idx = (
            _split_axes(self._shape, self._axes, self.keyframe.size)
        )

        # 3. Normalize selection
        frame_sel, spatial_sel = _normalize_selection(
            selection, self._shape, self._axes, split_idx
        )

        # 3. Single-page series: read full page, apply selection
        if not frame_axes:
            result = self.parent.asarray(
                series=self,
                squeeze=False,
                out=out,
                maxworkers=maxworkers,
                buffersize=buffersize,
            )
            if spatial_sel is not None:
                result = numpy.ascontiguousarray(result[spatial_sel])
            if squeeze is None:
                squeeze = self._squeeze
            if squeeze:
                result = result.squeeze()
            if self.transform is not None:
                result = self.transform(result)
            return result

        # 4. Compute page indices and output frame shape
        page_indices, out_frame_shape = _selection_to_page_indices(
            frame_sel, frame_shape, frame_axes
        )

        # 5. Try sub-page direct read for uncompressed memmappable pages
        #    with Y-axis (first spatial dim) selection
        result = self._try_subpage_read(
            page_indices,
            out_frame_shape,
            spatial_shape,
            spatial_axes,
            spatial_sel,
        )
        if result is not None:
            if squeeze is None:
                squeeze = self._squeeze
            if squeeze:
                result = result.squeeze()
            if self.transform is not None:
                result = self.transform(result)
            return result

        # 6. Compute segment filter for strip/tile skip optimization
        seg_filter = self._compute_segment_filter(
            spatial_shape, spatial_axes, spatial_sel
        )

        # 7. Compute crop slices for spatial selection
        #    This allows stack_pages to allocate a smaller output buffer.
        #    Crop only covers page-level dims (keyframe.shape), not extra
        #    series dims like trailing S=1.
        crop_slices: tuple[slice, ...] | None = None
        n_page_dims = len(self.keyframe.shape)
        if spatial_sel is not None and seg_filter is not None:
            crop_slices = tuple(
                s if isinstance(s, slice) else slice(s, s + 1)
                for s in spatial_sel[:n_page_dims]
            )

        # 8. Read selected pages
        pages = [self._getitem(int(idx)) for idx in page_indices]
        if seg_filter is not None:
            stacked = stack_pages(
                pages,
                maxworkers=maxworkers, buffersize=buffersize,
                segment_filter=seg_filter,
                crop=crop_slices,
            )
            # stacked shape: (npages, *cropped_spatial)
            # Build output shape, accounting for integer selections in
            # spatial_sel that produce length-1 dims
            cropped_spatial: list[int] = []
            for s, dim in zip(spatial_sel, spatial_shape):
                if isinstance(s, (int, numpy.integer)):
                    cropped_spatial.append(1)
                elif isinstance(s, slice):
                    cropped_spatial.append(
                        len(range(*s.indices(dim)))
                    )
                else:
                    cropped_spatial.append(dim)
            # Pad remaining spatial dims
            for dim in spatial_shape[len(spatial_sel):]:
                cropped_spatial.append(dim)

            target_shape = (*out_frame_shape, *cropped_spatial)
            result = stacked.reshape(target_shape)

            # Squeeze out integer-selected spatial dimensions
            squeeze_dims = []
            for i, s in enumerate(spatial_sel):
                if isinstance(s, (int, numpy.integer)):
                    squeeze_dims.append(len(out_frame_shape) + i)
            if squeeze_dims:
                result = numpy.squeeze(result, axis=tuple(squeeze_dims))

        else:
            if len(pages) == 1:
                page = pages[0]
                if page is None:
                    msg = 'selected page is None'
                    raise ValueError(msg)
                stacked = page.asarray(
                    maxworkers=maxworkers, buffersize=buffersize
                )
            else:
                stacked = stack_pages(
                    pages, maxworkers=maxworkers, buffersize=buffersize
                )

            # Reshape: (n_pages, *spatial) -> (*out_frame_shape, *spatial)
            target_shape = (*out_frame_shape, *spatial_shape)
            result = stacked.reshape(target_shape)

            # Apply spatial slicing
            if spatial_sel is not None:
                full_sel = (
                    tuple(slice(None) for _ in out_frame_shape) + spatial_sel
                )
                result = numpy.ascontiguousarray(result[full_sel])

        # 10. Squeeze only unselected length-1 dimensions
        #     Dimensions explicitly in the selection (even if length 1
        #     via a slice like 16:17) are kept.  Only "extra" unsqueezed
        #     dimensions not covered by the selection are candidates.
        if squeeze is None:
            squeeze = self._squeeze
        if squeeze:
            n_sel = original_tuple_len
            n_axes = len(self._axes)
            # Dimensions beyond what the selection covers are candidates
            # for squeezing (e.g., trailing S=1 in TZCYXS when only 5
            # elements in selection).
            if n_sel > 0 and n_sel < n_axes:
                squeeze_axes_list = []
                for i in range(result.ndim):
                    # Only squeeze trailing dims not in the selection
                    orig_idx = n_sel + (i - (result.ndim - (n_axes - n_sel)))
                    if orig_idx >= n_sel and result.shape[i] == 1:
                        squeeze_axes_list.append(i)
                if squeeze_axes_list:
                    result = numpy.squeeze(
                        result, axis=tuple(squeeze_axes_list)
                    )
            elif n_sel == 0:
                # Dict selection or no selection — squeeze normally
                result = result.squeeze()

        if self.transform is not None:
            result = self.transform(result)

        # Copy result into output array/memmap if requested
        if out is not None:
            from .tifffile import create_output

            out_array = create_output(out, result.shape, result.dtype)
            if out_array is not result:
                out_array[:] = result
            result = out_array

        if device is not None:
            from .gpu import numpy_to_tensor, parse_device

            dev = parse_device(device)
            if dev is not None:
                return numpy_to_tensor(result, dev)
        return result

    def _try_subpage_read(
        self,
        page_indices: NDArray[Any],
        out_frame_shape: tuple[int, ...],
        spatial_shape: tuple[int, ...],
        spatial_axes: str,
        spatial_sel: tuple[int | slice, ...] | None,
    ) -> NDArray[Any] | None:
        """Try fast mmap-based spatial selection for uncompressed pages.

        For uncompressed, memmappable, single-segment pages, uses mmap
        to avoid reading unneeded data:

        - Y-axis sub-range: reads only the needed rows (contiguous in
          memory), then applies remaining X/S selection in memory.
        - Non-Y selection on contiguous series: creates a full 3D+ mmap
          view and applies the spatial selection via numpy indexing,
          avoiding a full-file memory allocation.

        Returns None if the fast path is not applicable.

        """
        if spatial_sel is None:
            return None

        kf = self.keyframe

        # Must be uncompressed, single-segment, memmappable
        if not kf.is_memmappable or kf.compression != 1:
            return None
        if kf.predictor != 1 or kf.fillorder != 1:
            return None
        if len(kf.dataoffsets) != 1:
            return None

        assert self.parent is not None
        fh = self.parent.filehandle

        # Check if Y axis can be optimized (contiguous row sub-range)
        has_y_opt = False
        y_start = y_stop = n_rows = 0
        y_squeeze_dim = False

        if spatial_axes and spatial_axes[0] in ('Y', 'I'):
            y_sel = spatial_sel[0]
            if isinstance(y_sel, (int, numpy.integer)):
                y_start = int(y_sel)
                if y_start < 0:
                    y_start %= spatial_shape[0]
                y_stop = y_start + 1
                n_rows = 1
                y_squeeze_dim = True
                has_y_opt = True
            elif isinstance(y_sel, slice):
                y_start, y_stop, y_step = y_sel.indices(spatial_shape[0])
                if y_step == 1:
                    n_rows = y_stop - y_start
                    if 0 < n_rows < spatial_shape[0]:
                        has_y_opt = True

        if has_y_opt:
            # Y-optimized path: read only needed rows
            row_elements = 1
            for i, s in enumerate(spatial_shape):
                if i != 0:
                    row_elements *= s
            row_bytes = row_elements * self.dtype.itemsize

            remaining_spatial_sel: tuple[int | slice, ...] | None = None
            if len(spatial_sel) > 1:
                rest = spatial_sel[1:]
                if any(s != slice(None) for s in rest):
                    remaining_spatial_sel = rest

            npages = len(page_indices)
            result = self._try_contiguous_mmap_read(
                page_indices, npages, n_rows, y_start, y_stop,
                spatial_shape, fh,
            )
            if result is None:
                result = self._subpage_read_loop(
                    page_indices, npages, n_rows, y_start,
                    spatial_shape, row_bytes, fh,
                )

            cropped_spatial = (n_rows, *spatial_shape[1:])
            target_shape = (*out_frame_shape, *cropped_spatial)
            result = result.reshape(target_shape)

            if y_squeeze_dim:
                result = numpy.squeeze(
                    result, axis=len(out_frame_shape)
                )

            if remaining_spatial_sel is not None:
                n_frame = len(out_frame_shape)
                n_y = 0 if y_squeeze_dim else 1
                full_sel = (
                    tuple(slice(None) for _ in range(n_frame + n_y))
                    + remaining_spatial_sel
                )
                result = numpy.ascontiguousarray(result[full_sel])

            return result

        # Non-Y spatial selection on contiguous series: use mmap view
        # to apply full spatial indexing without allocating full data.
        if self.dataoffset is None:
            return None

        mv = fh._get_mmap_view()
        if mv is None:
            return None

        npages = self._len
        page_elems = kf.size
        if npages * page_elems != self.size:
            return None

        result = _mmap_read_contiguous(
            mv,
            self.dataoffset,
            self.dtype.char,
            self.parent.byteorder,
            npages,
            page_elems,
            page_selection=page_indices,
            spatial_shape=spatial_shape,
            spatial_sel=spatial_sel,
        )
        if result is None:
            return None

        # Reshape frame dims
        result_shape = result.shape
        # result is (n_selected_pages, *cropped_spatial)
        # Reshape to (*out_frame_shape, *cropped_spatial)
        target_shape = (*out_frame_shape, *result_shape[1:])
        result = result.reshape(target_shape)

        return result

    def _compute_segment_filter(
        self,
        spatial_shape: tuple[int, ...],
        spatial_axes: str,
        spatial_sel: tuple[int | slice, ...] | None,
    ) -> list[int] | None:
        """Compute which segment indices overlap the spatial selection.

        For compressed multi-strip or tiled pages, determines which
        segments need to be decoded.

        Returns segment indices list, or None if filtering is not applicable.

        """
        if spatial_sel is None:
            return None

        kf = self.keyframe
        # Must have multiple segments per page to benefit from filtering
        if len(kf.dataoffsets) < 2:
            return None
        # Uncompressed pages are handled by _try_subpage_read
        if kf.compression == 1:
            return None
        # Need Y axis as first spatial axis
        if not spatial_axes or spatial_axes[0] not in ('Y', 'I'):
            return None

        y_sel = spatial_sel[0]
        if isinstance(y_sel, (int, numpy.integer)):
            y_val = int(y_sel)
            if y_val < 0:
                y_val %= spatial_shape[0]
            y_start, y_stop = y_val, y_val + 1
        elif isinstance(y_sel, slice):
            y_start, y_stop, y_step = y_sel.indices(spatial_shape[0])
            if y_step != 1:
                return None
        else:
            return None

        n_rows = y_stop - y_start
        if n_rows <= 0 or n_rows == spatial_shape[0]:
            return None  # empty or full range, no benefit

        if kf.is_tiled:
            # Find X selection
            x_start, x_stop = 0, kf.imagewidth
            if len(spatial_sel) > 1 and len(spatial_axes) > 1:
                x_idx = next(
                    (i for i, ax in enumerate(spatial_axes) if ax == 'X'),
                    None,
                )
                if x_idx is not None and x_idx < len(spatial_sel):
                    x_sel = spatial_sel[x_idx]
                    if isinstance(x_sel, (int, numpy.integer)):
                        x_val = int(x_sel)
                        if x_val < 0:
                            x_val %= spatial_shape[x_idx]
                        x_start, x_stop = x_val, x_val + 1
                    elif isinstance(x_sel, slice):
                        xs, xe, xstep = x_sel.indices(spatial_shape[x_idx])
                        if xstep == 1 and 0 < xe - xs:
                            x_start, x_stop = xs, xe

            tile_indices, *_ = _compute_tile_overlap(
                y_start, y_stop, x_start, x_stop,
                kf.tilelength, kf.tilewidth,
                kf.imagelength, kf.imagewidth,
                kf.samplesperpixel, kf.planarconfig,
                kf.imagedepth, kf.tiledepth,
            )
            # Only filter if we skip at least some tiles
            total = product(kf.chunked)
            if len(tile_indices) < total:
                return tile_indices
            return None

        else:
            # Strips
            nstrips = (
                (kf.imagelength + kf.rowsperstrip - 1) // kf.rowsperstrip
            )
            if nstrips < 2:
                return None

            strip_indices, *_ = _compute_strip_overlap(
                y_start, y_stop,
                kf.rowsperstrip, kf.imagelength,
                kf.samplesperpixel, kf.planarconfig,
                kf.imagedepth,
            )
            total = product(kf.chunked)
            if len(strip_indices) < total:
                return strip_indices
            return None

    def _try_contiguous_mmap_read(
        self,
        page_indices: NDArray[Any],
        npages: int,
        n_rows: int,
        y_start: int,
        y_stop: int,
        spatial_shape: tuple[int, ...],
        fh: Any,
    ) -> NDArray[Any] | None:
        """Read sub-page data from contiguous series via mmap + numpy.

        Maps the entire series data as a numpy array, then uses combined
        fancy + slice indexing to select pages and rows in a single pass.
        Handles byte order normalization for non-native files.

        """
        dataoffset = self.dataoffset
        if dataoffset is None:
            return None

        mv = fh._get_mmap_view()
        if mv is None:
            return None

        assert self.parent is not None
        page_elems = product(spatial_shape)

        return _mmap_read_contiguous(
            mv,
            dataoffset,
            self.dtype.char,
            self.parent.byteorder,
            self._len,
            page_elems,
            page_selection=page_indices,
            spatial_shape=spatial_shape,
            y_slice=slice(y_start, y_stop),
        )
        # result shape: (npages, n_rows, *spatial_rest)

    def _subpage_read_loop(
        self,
        page_indices: NDArray[Any],
        npages: int,
        n_rows: int,
        y_start: int,
        spatial_shape: tuple[int, ...],
        row_bytes: int,
        fh: Any,
    ) -> NDArray[Any]:
        """Read sub-page data via mmap slicing or seek+read loop.

        Handles byte order normalization for non-native files.

        """
        assert self.parent is not None
        byteorder = self.parent.byteorder
        file_dtype = numpy.dtype(byteorder + self.dtype.char)
        native_dtype = self.dtype
        needs_swap = not file_dtype.isnative

        cropped_spatial = (n_rows, *spatial_shape[1:])
        result = numpy.empty((npages, *cropped_spatial), dtype=native_dtype)

        read_size = n_rows * row_bytes
        byte_offset_within_page = y_start * row_bytes

        # Try mmap path (avoids seek+read syscalls)
        mv = fh._get_mmap_view()
        if mv is not None:
            for i, pidx in enumerate(page_indices):
                page = self._getitem(int(pidx))
                if page is None:
                    result[i] = 0
                    continue
                offset = page.dataoffsets[0] + byte_offset_within_page
                buf = mv[offset : offset + read_size]
                chunk = numpy.ndarray(
                    shape=cropped_spatial, dtype=file_dtype, buffer=buf,
                )
                if needs_swap:
                    result[i] = chunk.byteswap()
                else:
                    result[i] = chunk
        else:
            # Fallback: seek+read
            for i, pidx in enumerate(page_indices):
                page = self._getitem(int(pidx))
                if page is None:
                    result[i] = 0
                    continue
                page_offset = page.dataoffsets[0] + byte_offset_within_page
                fh.seek(page_offset)
                data = fh.read(read_size)
                chunk = numpy.frombuffer(
                    data, dtype=file_dtype
                ).reshape(cropped_spatial)
                if needs_swap:
                    result[i] = chunk.byteswap()
                else:
                    result[i] = chunk

        return result

    def aszarr(
        self, *, level: int | None = None, **kwargs: Any
    ) -> ZarrTiffStore:
        """Return image array from series of pages as Zarr store.

        Parameters:
            level:
                Pyramid level to return.
                By default, a multi-resolution store is returned.
            **kwargs:
                Additional arguments passed to :py:class:`ZarrTiffStore`.

        """
        if self.parent is None:
            msg = 'no parent'
            raise ValueError(msg)

        from .zarr import ZarrTiffStore

        return ZarrTiffStore(self, level=level, **kwargs)

    @cached_property
    def dataoffset(self) -> int | None:
        """Offset to contiguous image data in file."""
        from .tifffile import TiffPage

        if not self._pages:
            return None
        pos = 0
        for page in self._pages:
            if page is None or len(page.dataoffsets) == 0:
                return None
            if not page.is_final:
                return None
            if not pos:
                pos = page.dataoffsets[0] + page.nbytes
                continue
            if pos != page.dataoffsets[0]:
                return None
            pos += page.nbytes

        page = self._pages[0]
        if page is None or len(page.dataoffsets) == 0:
            return None
        offset = page.dataoffsets[0]
        if (
            len(self._pages) == 1
            and isinstance(page, TiffPage)
            and (page.is_imagej or page.is_shaped or page.is_stk)
        ):
            # truncated files
            return offset
        if pos == offset + product(self.shape) * self.dtype.itemsize:
            return offset
        return None

    @property
    def is_pyramidal(self) -> bool:
        """Series contains multiple resolutions."""
        return len(self.levels) > 1

    @cached_property
    def attr(self) -> dict[str, Any]:
        """Arbitrary metadata associated with series."""
        return self._attr

    @property
    def ndim(self) -> int:
        """Number of array dimensions."""
        return len(self.shape)

    @property
    def dims(self) -> tuple[str, ...]:
        """Names of dimensions in image array."""
        from .tifffile import TIFF

        # return tuple(self.coords.keys())
        return tuple(
            unique_strings(TIFF.AXES_NAMES.get(ax, ax) for ax in self.axes)
        )

    @property
    def sizes(self) -> dict[str, int]:
        """Ordered map of dimension names to lengths."""
        # return dict(zip(self.coords.keys(), self.shape))
        return dict(zip(self.dims, self.shape, strict=True))

    @cached_property
    def size(self) -> int:
        """Number of elements in array."""
        return product(self.shape)

    @cached_property
    def nbytes(self) -> int:
        """Number of bytes in array."""
        return self.size * self.dtype.itemsize

    @property
    def pages(self) -> TiffPageSeries:
        # sequence of TiffPages or TiffFrame in series
        # a workaround to keep the old interface working
        return self

    def _getitem(self, key: int, /) -> TiffPage | TiffFrame | None:
        """Return specified page of series from cache or file."""
        key = int(key)
        if key < 0:
            key %= self._len
        if len(self._pages) == 1 and 0 < key < self._len:
            page = self._pages[0]
            assert page is not None
            assert self.parent is not None
            return self.parent.pages._getitem(page.index + key)
        return self._pages[key]

    @overload
    def __getitem__(
        self, key: int | numpy.integer[Any], /
    ) -> TiffPage | TiffFrame | None: ...

    @overload
    def __getitem__(
        self, key: slice | Iterable[int], /
    ) -> list[TiffPage | TiffFrame | None]: ...

    def __getitem__(
        self, key: int | numpy.integer[Any] | slice | Iterable[int], /
    ) -> TiffPage | TiffFrame | list[TiffPage | TiffFrame | None] | None:
        """Return specified page(s)."""
        if isinstance(key, (int, numpy.integer)):
            return self._getitem(int(key))
        if isinstance(key, slice):
            return [self._getitem(i) for i in range(*key.indices(self._len))]
        if isinstance(key, Iterable) and not isinstance(key, str):
            return [self._getitem(k) for k in key]
        msg = 'key must be an integer, slice, or iterable'
        raise TypeError(msg)

    def __iter__(self) -> Iterator[TiffPage | TiffFrame | None]:
        """Return iterator over pages in series."""
        if len(self._pages) == self._len:
            yield from self._pages
        else:
            assert self.parent is not None
            assert self._pages[0] is not None
            pages = self.parent.pages
            index = self._pages[0].index
            for i in range(self._len):
                yield pages[index + i]

    def __len__(self) -> int:
        """Return number of pages in series."""
        return self._len

    def __repr__(self) -> str:
        return f'<tifffile.TiffPageSeries {self._index} {self.kind}>'

    def __str__(self) -> str:
        s = '  '.join(
            s
            for s in (
                snipstr(f'{self.name!r}', 20) if self.name else '',
                'x'.join(str(i) for i in self.shape),
                str(self.dtype),
                self.axes,
                self.kind,
                (f'{len(self.levels)} Levels') if self.is_pyramidal else '',
                f'{len(self)} Pages',
                (f'@{self.dataoffset}') if self.dataoffset else '',
            )
            if s
        )
        return f'TiffPageSeries {self._index}  {s}'


class FileSequence(Sequence[str]):
    r"""Sequence of files containing compatible array data.

    Parameters:
        imread:
            Function to read image array from single file.
        files:
            Glob file name pattern or sequence of file names.
            If *None*, use '\*'.
            All files must contain array data of same shape and dtype.
            Binary streams are not supported.
        container:
            Name or open instance of ZIP file in which files are stored.
        sort:
            Function to sort file names if `files` is a pattern.
            The default is :py:func:`natural_sorted`.
            If *False*, disable sorting.
        parse:
            Function to parse sequence of sorted file names to dims, shape,
            chunk indices, and filtered file names.
            The default is :py:func:`parse_filenames` if `kwargs`
            contains `'pattern'`.
        **kwargs:
            Additional arguments passed to `parse` function.

    Examples:
        >>> filenames = ['temp_C001T002.tif', 'temp_C001T001.tif']
        >>> ims = TiffSequence(filenames, pattern=r'_(C)(\d+)(T)(\d+)')
        >>> ims[0]
        'temp_C001T002.tif'
        >>> ims.shape
        (1, 2)
        >>> ims.axes
        'CT'

    """

    imread: Callable[..., NDArray[Any]]
    """Function to read image array from single file."""

    shape: tuple[int, ...]
    """Shape of file series. Excludes shape of chunks in files."""

    axes: str
    """Character codes for dimensions in shape."""

    dims: tuple[str, ...]
    """Names of dimensions in shape."""

    indices: tuple[tuple[int, ...]]
    """Indices of files in shape."""

    _files: list[str]  # list of file names
    _container: Any  # TODO: container type?

    def __init__(
        self,
        imread: Callable[..., NDArray[Any]],
        files: (
            str | os.PathLike[Any] | Sequence[str | os.PathLike[Any]] | None
        ),
        *,
        container: str | os.PathLike[Any] | None = None,
        sort: Callable[..., Any] | bool | None = None,
        parse: Callable[..., Any] | None = None,
        **kwargs: Any,
    ) -> None:
        import fnmatch
        import glob
        import zipfile

        from .tifffile import TIFF, parse_filenames

        sort_func: Callable[..., list[str]] | None = None

        if files is None:
            files = '*'
        if sort is None:
            sort_func = natural_sorted
        elif callable(sort):
            sort_func = sort
        elif sort:
            sort_func = natural_sorted
        # elif not sort:
        #     sort_func = None

        self._container = container
        if container is not None:
            if isinstance(container, (str, os.PathLike)):
                self._container = zipfile.ZipFile(container)
            elif not hasattr(self._container, 'open'):
                msg = 'invalid container'
                raise ValueError(msg)
            if isinstance(files, str):
                files = fnmatch.filter(self._container.namelist(), files)
                if sort_func is not None:
                    files = sort_func(files)
        elif isinstance(files, os.PathLike):
            files = [os.fspath(files)]
        elif isinstance(files, str):
            files = glob.glob(files)
            if sort_func is not None:
                files = sort_func(files)
        elif sort is not None and sort_func is not None:
            # sort sequence if explicitly requested
            files = sort_func(f for f in files)

        files = [os.fspath(f) for f in files]  # type: ignore[union-attr]
        if not files:
            msg = 'no files found'
            raise ValueError(msg)

        if not callable(imread):
            msg = 'invalid imread function'
            raise TypeError(msg)

        if container:
            # redefine imread to read from container
            def imread_(
                filename: str, _imread: Any = imread, **kwargs: Any
            ) -> NDArray[Any]:
                with (
                    self._container.open(filename) as handle1,
                    io.BytesIO(handle1.read()) as handle2,
                ):
                    return _imread(handle2, **kwargs)

            imread = imread_

        if parse is None and kwargs.get('pattern'):
            parse = parse_filenames

        if parse:
            try:
                dims, shape, indices, files = parse(files, **kwargs)
            except ValueError as exc:
                msg = 'failed to parse file names'
                raise ValueError(msg) from exc
        else:
            dims = ('sequence',)
            shape = (len(files),)
            indices = tuple((i,) for i in range(len(files)))

        assert isinstance(files, list)
        assert isinstance(files[0], str)
        codes = TIFF.AXES_CODES
        axes = ''.join(codes.get(dim.lower(), dim[0].upper()) for dim in dims)

        self._files = files
        self.imread = imread
        self.axes = axes
        self.dims = tuple(dims)
        self.shape = tuple(shape)
        self.indices = indices

    def asarray(
        self,
        *,
        imreadargs: dict[str, Any] | None = None,
        chunkshape: tuple[int, ...] | None = None,
        chunkdtype: DTypeLike | None = None,
        axestiled: dict[int, int] | Sequence[tuple[int, int]] | None = None,
        ioworkers: int | None = 1,
        out_inplace: bool | None = None,
        out: OutputType = None,
        **kwargs: Any,
    ) -> NDArray[Any]:
        """Return images from files as NumPy array.

        Parameters:
            imreadargs:
                Arguments passed to :py:attr:`FileSequence.imread`.
            chunkshape:
                Shape of chunk in each file.
                Must match ``FileSequence.imread(file, **imreadargs).shape``.
                By default, this is determined by reading the first file.
            chunkdtype:
                Data type of chunk in each file.
                Must match ``FileSequence.imread(file, **imreadargs).dtype``.
                By default, this is determined by reading the first file.
            axestiled:
                Axes to be tiled.
                Map stacked sequence axis to chunk axis.
            ioworkers:
                Maximum number of threads to execute
                :py:attr:`FileSequence.imread` asynchronously.
                If *0*, use up to :py:attr:`_TIFF.MAXIOWORKERS` threads.
                Using threads can significantly improve runtime when reading
                many small files from a network share.
                If enabled, internal threading for the `imread` function
                should be disabled.
            out_inplace:
                :py:attr:`FileSequence.imread` decodes directly to the output
                instead of returning an array, which is copied to the output.
                Not all imread functions support this, especially in
                non-contiguous cases.
            out:
                Specifies how image array is returned.
                By default, create a new array.
                If a *numpy.ndarray*, a writable array to which the images
                are copied.
                If *'memmap'*, create a memory-mapped array in a temporary
                file.
                If a *string* or *open file*, the file used to create a
                memory-mapped array.
            **kwargs:
                Arguments passed to :py:attr:`FileSequence.imread` in
                addition to `imreadargs`.

        Raises:
            IndexError, ValueError: Array shapes do not match.

        """
        from .tifffile import TIFF, create_output, imread as _imread

        # TODO: deprecate kwargs?
        files = self._files
        if imreadargs is not None:
            kwargs |= imreadargs

        if ioworkers is None or ioworkers < 1:
            ioworkers = TIFF.MAXIOWORKERS
        ioworkers = min(len(files), ioworkers)
        assert isinstance(ioworkers, int)  # mypy bug?

        if out_inplace is None and self.imread == _imread:
            out_inplace = True
        else:
            out_inplace = bool(out_inplace)

        if chunkshape is None or chunkdtype is None:
            im = self.imread(files[0], **kwargs)
            chunkshape = im.shape
            chunkdtype = im.dtype
            del im
        chunkdtype = numpy.dtype(chunkdtype)
        assert chunkshape is not None

        if axestiled:
            tiled = TiledSequence(self.shape, chunkshape, axestiled=axestiled)
            result = create_output(out, tiled.shape, chunkdtype)

            def func(index: tuple[int | slice, ...], filename: str, /) -> None:
                # read single image from file into result
                # if index is None:
                #     return
                if out_inplace:
                    self.imread(filename, out=result[index], **kwargs)
                else:
                    im = self.imread(filename, **kwargs)
                    result[index] = im
                    del im  # delete memory-mapped file

            if ioworkers < 2:
                for index, filename in zip(
                    tiled.slices(self.indices), files, strict=True
                ):
                    func(index, filename)
            else:
                with ThreadPoolExecutor(ioworkers) as executor:
                    for _ in executor.map(
                        func, tiled.slices(self.indices), files
                    ):
                        pass
        else:
            shape = self.shape + chunkshape
            result = create_output(out, shape, chunkdtype)
            result = result.reshape((-1, *chunkshape))

            def func(index: tuple[int | slice, ...], filename: str, /) -> None:
                # read single image from file into result
                # if index is None:
                #     return
                index_ = int(
                    numpy.ravel_multi_index(
                        index,  # type: ignore[arg-type]
                        self.shape,
                    )
                )
                if out_inplace:
                    self.imread(filename, out=result[index_], **kwargs)
                else:
                    im = self.imread(filename, **kwargs)
                    result[index_] = im
                    del im  # delete memory-mapped file

            if ioworkers < 2:
                for index, filename in zip(self.indices, files, strict=True):
                    func(index, filename)
            else:
                with ThreadPoolExecutor(ioworkers) as executor:
                    for _ in executor.map(func, self.indices, files):
                        pass

            result.shape = shape

        return result

    def aszarr(self, **kwargs: Any) -> ZarrFileSequenceStore:
        """Return images from files as Zarr store.

        Parameters:
            **kwargs: Arguments passed to :py:class:`ZarrFileSequenceStore`.

        """
        from .zarr import ZarrFileSequenceStore

        return ZarrFileSequenceStore(self, **kwargs)

    def close(self) -> None:
        """Close open files."""
        if self._container is not None:
            self._container.close()
        self._container = None

    def commonpath(self) -> str:
        """Return longest common sub-path of each file in sequence."""
        if len(self._files) == 1:
            commonpath = os.path.dirname(self._files[0])
        else:
            commonpath = os.path.commonpath(self._files)
        return commonpath

    @property
    def files(self) -> list[str]:
        """Deprecated. Use the FileSequence sequence interface.

        :meta private:

        """
        warnings.warn(
            '<tifffile.FileSequence.files> is deprecated since 2024.5.22. '
            'Use the FileSequence sequence interface.',
            DeprecationWarning,
            stacklevel=2,
        )
        return self._files

    @property
    def files_missing(self) -> int:
        """Number of empty chunks."""
        return product(self.shape) - len(self._files)

    def __iter__(self) -> Iterator[str]:
        """Return iterator over all file names."""
        return iter(self._files)

    def __len__(self) -> int:
        return len(self._files)

    @overload
    def __getitem__(self, key: int, /) -> str: ...

    @overload
    def __getitem__(self, key: slice, /) -> list[str]: ...

    def __getitem__(self, key: int | slice, /) -> str | list[str]:
        return self._files[key]

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()

    def __repr__(self) -> str:
        return f'<tifffile.FileSequence @0x{id(self):016X}>'

    def __str__(self) -> str:
        file = str(self._container) if self._container else self._files[0]
        file = os.path.split(file)[-1]
        return '\n '.join(
            (
                self.__class__.__name__,
                file,
                f'files: {len(self._files)} ({self.files_missing} missing)',
                'shape: {}'.format(', '.join(str(i) for i in self.shape)),
                'dims: {}'.format(', '.join(s for s in self.dims)),
                # f'axes: {self.axes}',
            )
        )


@final
class TiffSequence(FileSequence):
    r"""Sequence of TIFF files containing compatible array data.

    Same as :py:class:`FileSequence` with the :py:func:`imread` function,
    `'\*.tif'` glob pattern, and `out_inplace` enabled by default.

    """

    def __init__(
        self,
        files: (
            str | os.PathLike[Any] | Sequence[str | os.PathLike[Any]] | None
        ) = None,
        *,
        imread: Callable[..., NDArray[Any]] | None = None,
        **kwargs: Any,
    ) -> None:
        if imread is None:
            from .tifffile import imread as _imread

            imread = _imread
        super().__init__(imread, '*.tif' if files is None else files, **kwargs)

    def __repr__(self) -> str:
        return f'<tifffile.TiffSequence @0x{id(self):016X}>'


@final
class TiledSequence:
    """Tiled sequence of chunks.

    Transform a sequence of stacked chunks to tiled chunks.

    Parameters:
        stackshape:
            Shape of stacked sequence excluding chunks.
        chunkshape:
            Shape of chunks.
        axestiled:
            Axes to be tiled. Map stacked sequence axis
            to chunk axis. By default, the sequence is not tiled.
        axes:
            Character codes for dimensions in stackshape and chunkshape.

    Examples:
        >>> ts = TiledSequence((1, 2), (3, 4), axestiled={1: 0}, axes='ABYX')
        >>> ts.shape
        (1, 6, 4)
        >>> ts.chunks
        (1, 3, 4)
        >>> ts.axes
        'AYX'

    """

    chunks: tuple[int, ...]
    """Shape of chunks in tiled sequence."""
    # with same number of dimensions as shape

    shape: tuple[int, ...]
    """Shape of tiled sequence including chunks."""

    axes: str | tuple[str, ...] | None
    """Dimensions codes of tiled sequence."""

    shape_squeezed: tuple[int, ...]
    """Shape of tiled sequence with length-1 dimensions removed."""

    axes_squeezed: str | tuple[str, ...] | None
    """Dimensions codes of tiled sequence with length-1 dimensions removed."""

    _stackdims: int
    """Number of dimensions in stack excluding chunks."""

    _chunkdims: int
    """Number of dimensions in chunks."""

    _shape_untiled: tuple[int, ...]
    """Shape of untiled sequence (stackshape + chunkshape)."""

    _axestiled: tuple[tuple[int, int], ...]
    """Map axes to tile from stack to chunks."""

    def __init__(
        self,
        stackshape: Sequence[int],
        chunkshape: Sequence[int],
        /,
        *,
        axestiled: dict[int, int] | Sequence[tuple[int, int]] | None = None,
        axes: str | Sequence[str] | None = None,
    ) -> None:
        self._stackdims = len(stackshape)
        self._chunkdims = len(chunkshape)
        self._shape_untiled = tuple(stackshape) + tuple(chunkshape)
        if axes is not None and len(axes) != len(self._shape_untiled):
            msg = 'axes length does not match stackshape + chunkshape'
            raise ValueError(msg)

        if axestiled:
            axestiled = dict(axestiled)
            for ax0, ax1 in axestiled.items():
                axestiled[ax0] = ax1 + self._stackdims
            self._axestiled = tuple(sorted(axestiled.items(), reverse=True))

            axes_list = [] if axes is None else list(axes)
            shape = list(self._shape_untiled)
            chunks = [1] * self._stackdims + list(chunkshape)
            used = set()
            for ax0, ax1 in self._axestiled:
                if ax0 in used or ax1 in used:
                    msg = 'duplicate axis'
                    raise ValueError(msg)
                used.add(ax0)
                used.add(ax1)
                shape[ax1] *= stackshape[ax0]
            for ax0, _ax1 in self._axestiled:
                del shape[ax0]
                del chunks[ax0]
                if axes_list:
                    del axes_list[ax0]
            self.shape = tuple(shape)
            self.chunks = tuple(chunks)
            if axes is None:
                self.axes = None
            elif isinstance(axes, str):
                self.axes = ''.join(axes_list)
            else:
                self.axes = tuple(axes_list)
        else:
            self._axestiled = ()
            self.shape = self._shape_untiled
            self.chunks = (1,) * self._stackdims + tuple(chunkshape)
            if axes is None:
                self.axes = None
            elif isinstance(axes, str):
                self.axes = axes
            else:
                self.axes = tuple(axes)

        assert len(self.shape) == len(self.chunks)
        if self.axes is not None:
            assert len(self.shape) == len(self.axes)

        if self.axes is None:
            self.shape_squeezed = tuple(i for i in self.shape if i > 1)
            self.axes_squeezed = None
        else:
            keep = ('X', 'Y', 'width', 'length', 'height')
            self.shape_squeezed = tuple(
                i
                for i, ax in zip(self.shape, self.axes, strict=True)
                if i > 1 or ax in keep
            )
            squeezed = tuple(
                ax
                for i, ax in zip(self.shape, self.axes, strict=True)
                if i > 1 or ax in keep
            )
            self.axes_squeezed = (
                ''.join(squeezed) if isinstance(self.axes, str) else squeezed
            )

    def indices(
        self, indices: Iterable[Sequence[int]], /
    ) -> Iterator[tuple[int, ...]]:
        """Return iterator over chunk indices of tiled sequence.

        Parameters:
            indices: Indices of chunks in stacked sequence.

        """
        chunkindex = [0] * self._chunkdims
        for index in indices:
            # if index is None:
            #     yield None
            # else:
            if len(index) != self._stackdims:
                msg = f'{len(index)} != {self._stackdims}'
                raise ValueError(msg)
            index_list = [*index, *chunkindex]
            for ax0, ax1 in self._axestiled:
                index_list[ax1] = index_list[ax0]
            for ax0, _ax1 in self._axestiled:
                del index_list[ax0]
            yield tuple(index_list)

    def slices(
        self, indices: Iterable[Sequence[int]] | None = None, /
    ) -> Iterator[tuple[int | slice, ...]]:
        """Return iterator over slices of chunks in tiled sequence.

        Parameters:
            indices: Indices of chunks in stacked sequence.

        """
        wholeslice: list[int | slice]
        chunkslice: list[int | slice] = [slice(None)] * self._chunkdims

        if indices is None:
            indices = numpy.ndindex(self._shape_untiled[: self._stackdims])

        for index in indices:
            # if index is None:
            #     yield None
            # else:
            assert len(index) == self._stackdims
            wholeslice = [*index, *chunkslice]
            for ax0, ax1 in self._axestiled:
                j = self._shape_untiled[ax1]
                i = cast(int, wholeslice[ax0]) * j
                wholeslice[ax1] = slice(i, i + j)
            for ax0, _ax1 in self._axestiled:
                del wholeslice[ax0]
            yield tuple(wholeslice)

    @property
    def ndim(self) -> int:
        """Number of dimensions of tiled sequence excluding chunks."""
        return len(self.shape)

    @property
    def is_tiled(self) -> bool:
        """Sequence is tiled."""
        return bool(self._axestiled)
