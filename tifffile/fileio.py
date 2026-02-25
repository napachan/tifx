# fileio.py

"""File I/O helpers for tifffile."""

from __future__ import annotations

import contextlib
import io
import logging
import mmap
import os
import struct
import threading
import time
from datetime import timedelta as TimeDelta
from collections.abc import Sequence
from functools import cached_property
from typing import IO, TYPE_CHECKING, cast, final, overload

import numpy

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator
    from types import TracebackType
    from typing import Any, Self

    from numpy.typing import DTypeLike, NDArray

from .utils import (
    format_size,
    indent,
    logger,
    product,
    sequence,
    snipstr,
    stripnull,
)


@final
class NullContext:
    """Null context manager. Can be used as a dummy reentrant lock.

    >>> with NullContext():
    ...     pass
    ...

    """

    __slots__ = ()

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        pass

    def __repr__(self) -> str:
        return 'NullContext()'


@final
class Timer:
    """Stopwatch for timing execution speed.

    Parameters:
        message:
            Message to print.
        end:
            End of print statement.
        started:
            Value of performance counter when started.
            The default is the current performance counter.

    Examples:
        >>> import time
        >>> with Timer('sleep:'):
        ...     time.sleep(1.05)
        sleep: 1.0... s

    """

    __slots__ = ('duration', 'started', 'stopped')

    started: float
    """Value of performance counter when started."""

    stopped: float
    """Value of performance counter when stopped."""

    duration: float
    """Duration between `started` and `stopped` in seconds."""

    def __init__(
        self,
        message: str | None = None,
        *,
        end: str = ' ',
        started: float | None = None,
    ) -> None:
        if message is not None:
            print(message, end=end, flush=True)
        self.duration = 0.0
        if started is None:
            started = time.perf_counter()
        self.started = self.stopped = started

    def start(self, message: str | None = None, *, end: str = ' ') -> float:
        """Start timer and return current time."""
        if message is not None:
            print(message, end=end, flush=True)
        self.duration = 0.0
        self.started = self.stopped = time.perf_counter()
        return self.started

    def stop(self, message: str | None = None, *, end: str = ' ') -> float:
        """Return duration of timer till start.

        Parameters:
            message: Message to print.
            end: End of print statement.

        """
        self.stopped = time.perf_counter()
        if message is not None:
            print(message, end=end, flush=True)
        self.duration = self.stopped - self.started
        return self.duration

    def print(
        self, message: str | None = None, *, end: str | None = None
    ) -> None:
        """Print duration from timer start till last stop or now.

        Parameters:
            message: Message to print.
            end: End of print statement.

        """
        msg = str(self)
        if message is not None:
            print(message, end=' ')
        print(msg, end=end, flush=True)

    @staticmethod
    def clock() -> float:
        """Return value of performance counter."""
        return time.perf_counter()

    def __str__(self) -> str:
        """Return duration from timer start till last stop or now."""
        if self.duration <= 0.0:
            # not stopped
            duration = time.perf_counter() - self.started
        else:
            duration = self.duration
        s = str(TimeDelta(seconds=duration))
        i = 0
        while i < len(s) and s[i : i + 2] in '0:0010203040506070809':
            i += 1
        if s[i : i + 1] == ':':
            i += 1
        return f'{s[i:]} s'

    def __repr__(self) -> str:
        return f'Timer(started={self.started})'

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.print()


@final
class StoredShape(Sequence[int]):
    """Normalized shape of image array in TIFF pages.

    Parameters:
        frames:
            Number of TIFF pages.
        separate_samples:
            Number of separate samples.
        depth:
            Image depth.
        length:
            Image length (height).
        width:
            Image width.
        contig_samples:
            Number of contiguous samples.
        extrasamples:
            Number of extra samples.

    """

    __slots__ = (
        'contig_samples',
        'depth',
        'extrasamples',
        'frames',
        'length',
        'separate_samples',
        'width',
    )

    frames: int
    """Number of TIFF pages."""

    separate_samples: int
    """Number of separate samples."""

    depth: int
    """Image depth. Value of ImageDepth tag or 1."""

    length: int
    """Image length (height). Value of ImageLength tag."""

    width: int
    """Image width. Value of ImageWidth tag."""

    contig_samples: int
    """Number of contiguous samples."""

    extrasamples: int
    """Number of extra samples. Count of ExtraSamples tag or 0."""

    def __init__(
        self,
        frames: int = 1,
        separate_samples: int = 1,
        depth: int = 1,
        length: int = 1,
        width: int = 1,
        contig_samples: int = 1,
        extrasamples: int = 0,
    ) -> None:
        if separate_samples != 1 and contig_samples != 1:
            msg = 'invalid samples'
            raise ValueError(msg)

        self.frames = int(frames)
        self.separate_samples = int(separate_samples)
        self.depth = int(depth)
        self.length = int(length)
        self.width = int(width)
        self.contig_samples = int(contig_samples)
        self.extrasamples = int(extrasamples)

    @property
    def size(self) -> int:
        """Product of all dimensions."""
        return (
            abs(self.frames)
            * self.separate_samples
            * self.depth
            * self.length
            * self.width
            * self.contig_samples
        )

    @property
    def samples(self) -> int:
        """Number of samples. Count of SamplesPerPixel tag."""
        assert self.separate_samples == 1 or self.contig_samples == 1
        samples = (
            self.separate_samples
            if self.separate_samples > 1
            else self.contig_samples
        )
        assert self.extrasamples < samples
        return samples

    @property
    def photometric_samples(self) -> int:
        """Number of photometric samples."""
        return self.samples - self.extrasamples

    @property
    def shape(self) -> tuple[int, int, int, int, int, int]:
        """Normalized 6D shape of image array in all pages."""
        return (
            self.frames,
            self.separate_samples,
            self.depth,
            self.length,
            self.width,
            self.contig_samples,
        )

    @property
    def page_shape(self) -> tuple[int, int, int, int, int]:
        """Normalized 5D shape of image array in single page."""
        return (
            self.separate_samples,
            self.depth,
            self.length,
            self.width,
            self.contig_samples,
        )

    @property
    def page_size(self) -> int:
        """Product of dimensions in single page."""
        return (
            self.separate_samples
            * self.depth
            * self.length
            * self.width
            * self.contig_samples
        )

    @property
    def squeezed(self) -> tuple[int, ...]:
        """Shape with length-1 removed, except for length and width."""
        shape = [self.length, self.width]
        if self.separate_samples > 1:
            shape.insert(0, self.separate_samples)
        elif self.contig_samples > 1:
            shape.append(self.contig_samples)
        if self.frames > 1:
            shape.insert(0, self.frames)
        return tuple(shape)

    @property
    def is_valid(self) -> bool:
        """Shape is valid."""
        return (
            self.frames >= 1
            and self.depth >= 1
            and self.length >= 1
            and self.width >= 1
            and (self.separate_samples == 1 or self.contig_samples == 1)
            and (
                self.contig_samples
                if self.contig_samples > 1
                else self.separate_samples
            )
            > self.extrasamples
        )

    @property
    def is_planar(self) -> bool:
        """Shape contains planar samples."""
        return self.separate_samples > 1

    @property
    def planarconfig(self) -> int | None:
        """Value of PlanarConfiguration tag."""
        if self.separate_samples > 1:
            return 2  # PLANARCONFIG.SEPARATE
        if self.contig_samples > 1:
            return 1  # PLANARCONFIG.CONTIG
        return None

    def __len__(self) -> int:
        return 6

    @overload
    def __getitem__(self, key: int, /) -> int: ...

    @overload
    def __getitem__(self, key: slice, /) -> tuple[int, ...]: ...

    def __getitem__(self, key: int | slice, /) -> int | tuple[int, ...]:
        return (
            self.frames,
            self.separate_samples,
            self.depth,
            self.length,
            self.width,
            self.contig_samples,
        )[key]

    def __hash__(self) -> int:
        return hash(
            (
                self.frames,
                self.separate_samples,
                self.depth,
                self.length,
                self.width,
                self.contig_samples,
            )
        )

    def __eq__(self, other: object, /) -> bool:
        return (
            isinstance(other, StoredShape)
            and self.frames == other.frames
            and self.separate_samples == other.separate_samples
            and self.depth == other.depth
            and self.length == other.length
            and self.width == other.width
            and self.contig_samples == other.contig_samples
        )

    def __repr__(self) -> str:
        return (
            '<StoredShape('
            f'frames={self.frames}, '
            f'separate_samples={self.separate_samples}, '
            f'depth={self.depth}, '
            f'length={self.length}, '
            f'width={self.width}, '
            f'contig_samples={self.contig_samples}, '
            f'extrasamples={self.extrasamples}'
            ')>'
        )


@final
class FileHandle:
    """Binary file handle.

    A limited, special purpose binary file handle that can:

    - handle embedded files (for example, LSM within LSM files).
    - re-open closed files (for multi-file formats, such as OME-TIFF).
    - read and write NumPy arrays and records from file-like objects.

    When initialized from another file handle, do not use the other handle
    unless this FileHandle is closed.

    FileHandle instances are not thread-safe.

    Parameters:
        file:
            File name or seekable binary stream, such as open file,
            BytesIO, or fsspec OpenFile.
        mode:
            File open mode if `file` is file name.
            The default is 'rb'. Files are always opened in binary mode.
        name:
            Name of file if `file` is binary stream.
        offset:
            Start position of embedded file.
            The default is the current file position.
        size:
            Size of embedded file.
            The default is the number of bytes from `offset` to
            the end of the file.

    """

    # TODO: make FileHandle a subclass of IO[bytes]

    __slots__ = (
        '_close',
        '_dir',
        '_fh',
        '_file',
        '_lock',
        '_mmap',
        '_mmap_fd',
        '_mmap_view',
        '_mode',
        '_name',
        '_offset',
        '_size',
    )

    _file: str | os.PathLike[Any] | FileHandle | IO[bytes] | None
    _fh: IO[bytes] | None
    _mode: str
    _name: str
    _dir: str
    _offset: int
    _size: int
    _close: bool
    _lock: threading.RLock | NullContext

    def __init__(
        self,
        file: str | os.PathLike[Any] | FileHandle | IO[bytes],
        /,
        mode: (
            Literal['r', 'r+', 'w', 'x', 'rb', 'r+b', 'wb', 'xb'] | None
        ) = None,
        *,
        name: str | None = None,
        offset: int | None = None,
        size: int | None = None,
    ) -> None:
        self._mode = 'rb' if mode is None else mode
        self._fh = None
        self._mmap: 'mmap.mmap | None' = None
        self._mmap_fd: int = -1
        self._mmap_view: memoryview | None = None
        self._file = file  # reference to original argument for re-opening
        self._name = name if name else ''
        self._dir = ''
        self._offset = -1 if offset is None else offset
        self._size = -1 if size is None else size
        self._close = True
        self._lock = NullContext()
        self.open()
        assert self._fh is not None

    def open(self) -> None:
        """Open or re-open file."""
        if self._fh is not None:
            return  # file is open

        if isinstance(self._file, os.PathLike):
            self._file = os.fspath(self._file)

        if isinstance(self._file, str):
            # file name
            if self._mode[-1:] != 'b':
                self._mode += 'b'
            if self._mode not in {'rb', 'r+b', 'wb', 'xb'}:
                msg = f'invalid mode {self._mode}'
                raise ValueError(msg)
            self._file = os.path.realpath(self._file)
            self._dir, self._name = os.path.split(self._file)
            self._fh = open(  # noqa: SIM115
                self._file, self._mode, encoding=None
            )
            self._close = True
            self._offset = max(0, self._offset)
        elif isinstance(self._file, FileHandle):
            # FileHandle
            self._fh = self._file._fh
            self._offset = max(0, self._offset)
            self._offset += self._file._offset
            self._close = False
            if not self._name:
                if self._offset:
                    name, ext = os.path.splitext(self._file._name)
                    self._name = f'{name}@{self._offset}{ext}'
                else:
                    self._name = self._file._name
            self._mode = self._file._mode
            self._dir = self._file._dir
        elif hasattr(self._file, 'seek'):
            # binary stream: open file, BytesIO, fsspec LocalFileOpener
            # cast to IO[bytes] even it might not be
            if isinstance(self._file, io.TextIOBase):
                msg = f'{self._file!r} is not open in binary mode'
                raise TypeError(msg)
            self._fh = cast(IO[bytes], self._file)
            try:
                self._fh.tell()
            except Exception:
                msg = 'binary stream is not seekable'
                raise ValueError(msg) from None

            if self._offset < 0:
                self._offset = self._fh.tell()
            self._close = False
            if not self._name:
                try:
                    self._dir, self._name = os.path.split(self._fh.name)
                except AttributeError:
                    try:
                        self._dir, self._name = os.path.split(
                            self._fh.path  # type: ignore[attr-defined]
                        )
                    except AttributeError:
                        self._name = 'Unnamed binary stream'
            with contextlib.suppress(AttributeError):
                self._mode = self._fh.mode
        elif hasattr(self._file, 'open'):
            # fsspec OpenFile
            _file: Any = self._file
            self._fh = cast(IO[bytes], _file.open())
            try:
                self._fh.tell()
            except Exception:
                with contextlib.suppress(Exception):
                    self._fh.close()
                msg = 'OpenFile is not seekable'
                raise ValueError(msg) from None

            if self._offset < 0:
                self._offset = self._fh.tell()
            self._close = True
            if not self._name:
                try:
                    self._dir, self._name = os.path.split(_file.path)
                except AttributeError:
                    self._name = 'Unnamed binary stream'
            with contextlib.suppress(AttributeError):
                self._mode = _file.mode

        else:
            msg = (
                'the first parameter must be a file name '
                'or seekable binary file object, '
                f'not {type(self._file)!r}'
            )
            raise ValueError(msg)

        assert self._fh is not None

        if self._offset:
            self._fh.seek(self._offset)

        if self._size < 0:
            pos = self._fh.tell()
            self._fh.seek(self._offset, os.SEEK_END)
            self._size = self._fh.tell()
            self._fh.seek(pos)

    def close(self) -> None:
        """Close file handle."""
        if self._mmap_view is not None:
            self._mmap_view.release()
            self._mmap_view = None
        if self._mmap is not None:
            with contextlib.suppress(Exception):
                self._mmap.close()
            self._mmap = None
        if self._mmap_fd >= 0:
            with contextlib.suppress(Exception):
                os.close(self._mmap_fd)
            self._mmap_fd = -1
        if self._close and self._fh is not None:
            with contextlib.suppress(Exception):
                self._fh.close()
            self._fh = None

    def fileno(self) -> int:
        """Return underlying file descriptor if exists, else raise OSError."""
        assert self._fh is not None
        try:
            return self._fh.fileno()
        except (OSError, AttributeError) as exc:
            msg = f'{type(self._fh)} does not have a file descriptor'
            raise OSError(msg) from exc

    def writable(self) -> bool:
        """Return True if stream supports writing."""
        assert self._fh is not None
        if hasattr(self._fh, 'writable'):
            return self._fh.writable()
        return False

    def seekable(self) -> bool:
        """Return True if stream supports random access."""
        return True

    def tell(self) -> int:
        """Return file's current position."""
        assert self._fh is not None
        return self._fh.tell() - self._offset

    def seek(self, offset: int, /, whence: int = 0) -> int:
        """Set file's current position.

        Parameters:
            offset:
                Position of file handle relative to position indicated
                by `whence`.
            whence:
                Relative position of `offset`.
                0 (`os.SEEK_SET`) beginning of file (default).
                1 (`os.SEEK_CUR`) current position.
                2 (`os.SEEK_END`) end of file.

        """
        assert self._fh is not None
        if self._offset:
            if whence == 0:
                return (
                    self._fh.seek(self._offset + offset, whence) - self._offset
                )
            if whence == 2 and self._size > 0:
                return (
                    self._fh.seek(self._offset + self._size + offset, 0)
                    - self._offset
                )
        return self._fh.seek(offset, whence)

    def read(self, size: int = -1, /) -> bytes:
        """Return bytes read from file.

        Parameters:
            size:
                Number of bytes to read from file.
                By default, read until the end of the file.

        """
        if size < 0 and self._offset:
            size = self._size
        assert self._fh is not None
        return self._fh.read(size)

    def readinto(self, buffer: bytes, /) -> int:
        """Read bytes from file into buffer.

        Parameters:
            buffer: Buffer to read into.

        Returns:
            Number of bytes read from file.

        """
        assert self._fh is not None
        return self._fh.readinto(buffer)  # type: ignore[attr-defined]

    def write(self, buffer: bytes | memoryview[Any], /) -> int:
        """Write bytes to file and return number of bytes written.

        Parameters:
            buffer: Bytes to write to file.

        Returns:
            Number of bytes written.

        """
        assert self._fh is not None
        return self._fh.write(buffer)

    def flush(self) -> None:
        """Flush write buffers of stream if applicable."""
        assert self._fh is not None
        if hasattr(self._fh, 'flush'):
            self._fh.flush()

    def memmap_array(
        self,
        dtype: DTypeLike | None,
        shape: tuple[int, ...],
        offset: int = 0,
        *,
        mode: str = 'r',
        order: str = 'C',
    ) -> NDArray[Any]:
        """Return `numpy.memmap` of array data stored in file.

        Parameters:
            dtype:
                Data type of array in file.
            shape:
                Shape of array in file.
            offset:
                Start position of array-data in file.
            mode:
                File is opened in this mode. The default is read-only.
            order:
                Order of ndarray memory layout. The default is 'C'.

        """
        if not self.is_file:
            msg = 'cannot memory-map file without fileno'
            raise ValueError(msg)
        assert self._fh is not None
        return numpy.memmap(
            self._fh,  # type: ignore[call-overload]
            dtype=dtype,
            mode=mode,
            offset=self._offset + offset,
            shape=shape,
            order=order,
        )

    def read_array(
        self,
        dtype: DTypeLike | None,
        count: int = -1,
        offset: int = 0,
        *,
        out: NDArray[Any] | None = None,
    ) -> NDArray[Any]:
        """Return NumPy array from file in native byte order.

        Parameters:
            dtype:
                Data type of array to read.
            count:
                Number of items to read. By default, all items are read.
            offset:
                Start position of array-data in file.
            out:
                NumPy array to read into. By default, a new array is created.

        """
        dtype = numpy.dtype(dtype)

        if count < 0:
            nbytes = self._size if out is None else out.nbytes
            count = nbytes // dtype.itemsize
        else:
            nbytes = count * dtype.itemsize

        result = numpy.empty(count, dtype) if out is None else out

        if result.nbytes != nbytes:
            msg = 'size mismatch'
            raise ValueError(msg)

        assert self._fh is not None

        # mmap fast path: reuse existing mmap (avoids seek+readinto)
        mv = self._mmap_view
        if mv is not None and out is None:
            start = offset if offset else self._fh.tell()
            end = start + nbytes
            if end <= len(mv):
                src = numpy.frombuffer(mv[start:end], dtype)
                if not dtype.isnative:
                    # byteswap to native byte order (one copy)
                    return src.byteswap().view(dtype.newbyteorder())
                return src.copy()

        if offset:
            self._fh.seek(self._offset + offset)

        try:
            n = self._fh.readinto(result)  # type: ignore[attr-defined]
        except AttributeError:
            result[:] = numpy.frombuffer(self._fh.read(nbytes), dtype).reshape(
                result.shape
            )
            n = nbytes

        if n != nbytes:
            msg = f'failed to read {nbytes} bytes, got {n}'
            raise ValueError(msg)

        if not result.dtype.isnative:
            if not dtype.isnative:
                result.byteswap(True)
            result = result.view(result.dtype.newbyteorder())
        elif result.dtype.isnative != dtype.isnative:
            result.byteswap(True)

        if out is not None and hasattr(out, 'flush'):
            out.flush()

        return result

    def read_record(
        self,
        dtype: DTypeLike | None,
        shape: tuple[int, ...] | int | None = 1,
        *,
        byteorder: Literal['S', '<', '>', '=', '|'] | None = None,
    ) -> numpy.recarray[Any, Any]:
        """Return NumPy record from file.

        Parameters:
            dtype:
                Data type of record array to read.
            shape:
                Shape of record array to read.
            byteorder:
                Byte order of record array to read.

        """
        assert self._fh is not None

        dtype = numpy.dtype(dtype)
        if byteorder is not None:
            dtype = dtype.newbyteorder(byteorder)

        try:
            record = numpy.rec.fromfile(  # type: ignore[call-overload]
                self._fh, dtype, shape
            )
        except Exception:
            if shape is None:
                shape = self._size // dtype.itemsize
            size = product(sequence(shape)) * dtype.itemsize
            # data = bytearray(size)
            # n = self._fh.readinto(data)
            # data = data[:n]
            # TODO: record is not writable
            data = self._fh.read(size)
            record = numpy.rec.fromstring(
                data,
                dtype,
                shape,
            )
        return record[0] if shape == 1 else record

    def write_empty(self, size: int, /) -> int:
        """Append null-bytes to file.

        The file position must be at the end of the file.

        Parameters:
            size: Number of null-bytes to write to file.

        """
        if size < 1:
            return 0
        assert self._fh is not None
        self._fh.seek(size - 1, os.SEEK_CUR)
        self._fh.write(b'\x00')
        return size

    def write_array(
        self,
        data: NDArray[Any],
        dtype: DTypeLike | None = None,
        /,
    ) -> int:
        """Write NumPy array to file in C contiguous order.

        Parameters:
            data: Array to write to file.

        """
        assert self._fh is not None
        pos = self._fh.tell()
        # writing non-contiguous arrays is very slow
        data = numpy.ascontiguousarray(data, dtype)
        try:
            data.tofile(self._fh)
        except io.UnsupportedOperation:
            # numpy cannot write to BytesIO
            self._fh.write(data.tobytes())
        return self._fh.tell() - pos

    def read_segments(
        self,
        offsets: Sequence[int],
        bytecounts: Sequence[int],
        /,
        indices: Sequence[int] | None = None,
        length: int | None = None,
        *,
        sort: bool = True,
        lock: threading.RLock | NullContext | None = None,
        buffersize: int | None = None,
        flat: bool = True,
    ) -> (
        Iterator[tuple[bytes | None, int]]
        | Iterator[list[tuple[bytes | None, int]]]
    ):
        """Return iterator over segments read from file and their indices.

        The purpose of this function is to

        - reduce small or random reads.
        - reduce acquiring reentrant locks.
        - synchronize seeks and reads.
        - limit size of segments read into memory at once.
          (ThreadPoolExecutor.map is not collecting iterables lazily).

        Parameters:
            offsets:
                Offsets of segments to read from file.
            bytecounts:
                Byte counts of segments to read from file.
            indices:
                Indices of segments in image.
                The default is `range(length)`.
            length:
                Number of segments to read from file.
                By default, `len(offsets)`.
            sort:
                Read segments from file in order of their offsets.
            lock:
                Reentrant lock to synchronize seeks and reads.
            buffersize:
                Approximate number of bytes to read from file in one pass.
                The default is :py:attr:`_TIFF.BUFFERSIZE`.
            flat:
                If *True*, return iterator over individual (segment, index)
                tuples.
                Else, return an iterator over a list of (segment, index)
                tuples that are acquired in one pass.

        Yields:
            Individual or lists of `(segment, index)` tuples.

        """
        # TODO: Cythonize this?
        assert self._fh is not None
        if length is None:
            length = len(offsets)
        if length < 1:
            return

        if indices is None:
            indices = tuple(range(length))

        # mmap fast path: zero-copy memoryview slicing, no locks needed
        mv = self._get_mmap_view()
        if mv is not None:
            file_size = len(mv)

            # build and sort segment list
            n = min(length, len(offsets), len(bytecounts))
            segments_list = [
                (indices[i], offsets[i], bytecounts[i]) for i in range(n)
            ]
            if n < length:
                logger().warning(
                    'tifffile.read_segments: '
                    f'expected {length} segments, got {n}'
                )
                segments_list.extend(
                    (indices[i], 0, 0) for i in range(n, length)
                )
            if sort:
                segments_list.sort(key=lambda x: x[1])

            if flat:
                for idx, off, bc in segments_list:
                    if off > 0 and bc > 0 and off + bc <= file_size:
                        yield (mv[off : off + bc], idx)
                    else:
                        yield (None, idx)
            else:
                if buffersize is None:
                    from .tifffile import TIFF

                    buffersize = TIFF.BUFFERSIZE

                # yield in batches sized by buffersize
                result: list[tuple[bytes | memoryview | None, int]] = []
                batch_size = 0
                for idx, off, bc in segments_list:
                    if off > 0 and bc > 0 and off + bc <= file_size:
                        result.append((mv[off : off + bc], idx))
                        batch_size += bc
                    else:
                        result.append((None, idx))
                    if batch_size >= buffersize:
                        yield result
                        result = []
                        batch_size = 0
                if result:
                    yield result
            return

        if length == 1:
            if bytecounts[0] > 0 and offsets[0] > 0:
                if lock is None:
                    lock = self._lock
                with lock:
                    self.seek(offsets[0])
                    data = self._fh.read(bytecounts[0])
            else:
                data = None
            index = 0 if indices is None else indices[0]
            yield (data, index) if flat else [(data, index)]
            return

        if lock is None:
            lock = self._lock
        if buffersize is None:
            from .tifffile import TIFF

            buffersize = TIFF.BUFFERSIZE

        # corrupted files may be missing some offsets or bytecounts
        segments = [
            (indices[i], offsets[i], bytecounts[i])
            for i in range(min(length, len(offsets), len(bytecounts)))
        ]
        if len(segments) < length:
            logger().warning(
                'tifffile.read_segments: '
                f'expected {length} segments, got {len(segments)}'
            )
            segments.extend(
                (indices[i], 0, 0) for i in range(len(segments), length)
            )

        if sort:
            segments = sorted(segments, key=lambda x: x[1])

        iscontig = True
        for i in range(length - 1):
            _, offset, bytecount = segments[i]
            nextoffset = segments[i + 1][1]
            if offset == 0 or bytecount == 0 or nextoffset == 0:
                continue
            if offset + bytecount != nextoffset:
                iscontig = False
                break

        seek = self.seek
        read = self._fh.read
        result_list: list[tuple[bytes | None, int]]

        if iscontig:
            # consolidate reads
            i = 0
            while i < length:
                j = i
                offset = -1
                bytecount = 0
                while bytecount <= buffersize and i < length:
                    _, o, b = segments[i]
                    if o > 0 and b > 0:
                        if offset < 0:
                            offset = o
                        bytecount += b
                    i += 1

                if offset < 0:
                    data = None
                else:
                    with lock:
                        seek(offset)
                        data = read(bytecount)
                start = 0
                stop = 0
                result_list = []
                while j < i:
                    index, offset, bytecount = segments[j]
                    if offset > 0 and bytecount > 0:
                        stop += bytecount
                        result_list.append(
                            (data[start:stop], index)  # type: ignore[index]
                        )
                        start = stop
                    else:
                        result_list.append((None, index))
                    j += 1
                if flat:
                    yield from result_list
                else:
                    yield result_list
            return

        i = 0
        while i < length:
            result_list = []
            size = 0
            with lock:
                while size <= buffersize and i < length:
                    index, offset, bytecount = segments[i]
                    if offset > 0 and bytecount > 0:
                        seek(offset)
                        result_list.append((read(bytecount), index))
                        size += bytecount
                    else:
                        result_list.append((None, index))
                    i += 1
            if flat:
                yield from result_list
            else:
                yield result_list

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()
        self._file = None

    # TODO: this may crash the Python interpreter under certain conditions
    # def __getattr__(self, name: str, /) -> Any:
    #     """Return attribute from underlying file object."""
    #     if self._offset:
    #         warnings.warn(
    #             '<tifffile.FileHandle> '
    #             f'{name} not implemented for embedded files',
    #             UserWarning,
    #         )
    #     return getattr(self._fh, name)

    def __repr__(self) -> str:
        return f'<tifffile.FileHandle {snipstr(self._name, 32)!r}>'

    def __str__(self) -> str:
        return '\n '.join(
            (
                'FileHandle',
                self._name,
                self._dir,
                f'{self._size} bytes',
                'closed' if self._fh is None else 'open',
            )
        )

    @property
    def name(self) -> str:
        """Name of file or stream."""
        return self._name

    @property
    def dirname(self) -> str:
        """Directory in which file is stored."""
        return self._dir

    @property
    def path(self) -> str:
        """Absolute path of file."""
        return os.path.join(self._dir, self._name)

    @property
    def extension(self) -> str:
        """File name extension of file or stream."""
        name, ext = os.path.splitext(self._name.lower())
        if ext and name.endswith('.ome'):
            ext = '.ome' + ext
        return ext

    @property
    def size(self) -> int:
        """Size of file in bytes."""
        return self._size

    @property
    def closed(self) -> bool:
        """File is closed."""
        return self._fh is None

    @property
    def lock(self) -> threading.RLock | NullContext:
        """Reentrant lock to synchronize reads and writes."""
        return self._lock

    @lock.setter
    def lock(self, value: bool, /) -> None:
        self.set_lock(value)

    def set_lock(self, lock: bool) -> None:  # noqa: FBT001
        """Set reentrant lock to synchronize reads and writes."""
        if bool(lock) == isinstance(self._lock, NullContext):
            self._lock = threading.RLock() if lock else NullContext()

    @property
    def has_lock(self) -> bool:
        """A reentrant lock is currently used to sync reads and writes."""
        return not isinstance(self._lock, NullContext)

    @property
    def is_file(self) -> bool:
        """File has fileno and can be memory-mapped."""
        try:
            self._fh.fileno()  # type: ignore[union-attr]
        except Exception:
            return False
        return True

    def _get_mmap_view(self) -> memoryview | None:
        """Return memoryview of memory-mapped file, or None."""
        if self._mmap_view is not None:
            return self._mmap_view
        if self._mmap is not None:
            return None  # mmap exists but view failed previously
        if not self.is_file or self._offset != 0 or self._mode != 'rb':
            return None
        try:
            # Open a separate file descriptor for mmap to avoid
            # interfering with the main file handle's seek/read on Windows
            fd = os.open(
                self._file,  # type: ignore[arg-type]
                os.O_RDONLY | getattr(os, 'O_BINARY', 0),
            )
            self._mmap_fd = fd
            self._mmap = mmap.mmap(fd, 0, access=mmap.ACCESS_READ)
            self._mmap_view = memoryview(self._mmap)
        except Exception:
            if self._mmap_fd >= 0:
                with contextlib.suppress(Exception):
                    os.close(self._mmap_fd)
                self._mmap_fd = -1
            return None
        return self._mmap_view


@final
class FileCache:
    """Keep FileHandles open.

    Parameters:
        size: Maximum number of files to keep open. The default is 8.
        lock: Reentrant lock to synchronize reads and writes.

    """

    __slots__ = ('files', 'keep', 'lock', 'past', 'size')

    size: int
    """Maximum number of files to keep open."""

    files: dict[FileHandle, int]
    """Reference counts of opened files."""

    keep: set[FileHandle]
    """Set of files to keep open."""

    past: list[FileHandle]
    """FIFO list of opened files."""

    lock: threading.RLock | NullContext
    """Reentrant lock to synchronize reads and writes."""

    def __init__(
        self,
        size: int | None = None,
        *,
        lock: threading.RLock | NullContext | None = None,
    ) -> None:
        self.past = []
        self.files = {}
        self.keep = set()
        self.size = 8 if size is None else int(size)
        self.lock = NullContext() if lock is None else lock

    def open(self, fh: FileHandle, /) -> None:
        """Open file, re-open if necessary."""
        with self.lock:
            if fh in self.files:
                self.files[fh] += 1
            elif fh.closed:
                fh.open()
                self.files[fh] = 1
                self.past.append(fh)
            else:
                self.files[fh] = 2
                self.keep.add(fh)
                self.past.append(fh)

    def close(self, fh: FileHandle, /) -> None:
        """Close least recently used open files."""
        with self.lock:
            if fh in self.files:
                self.files[fh] -= 1
            self._trim()

    def clear(self) -> None:
        """Close all opened files if not in use when opened first."""
        with self.lock:
            for fh, _refcount in list(self.files.items()):
                if fh not in self.keep:
                    fh.close()
                    del self.files[fh]
                    del self.past[self.past.index(fh)]

    def read(
        self,
        fh: FileHandle,
        /,
        offset: int,
        bytecount: int,
        whence: int = 0,
    ) -> bytes:
        """Return bytes read from binary file.

        Parameters:
            fh:
                File handle to read from.
            offset:
                Position in file to start reading from relative to the
                position indicated by `whence`.
            bytecount:
                Number of bytes to read.
            whence:
                Relative position of offset.
                0 (`os.SEEK_SET`) beginning of file (default).
                1 (`os.SEEK_CUR`) current position.
                2 (`os.SEEK_END`) end of file.

        """
        # this function is more efficient than
        # filecache.open(fh)
        # with lock:
        #     fh.seek()
        #     data = fh.read()
        # filecache.close(fh)
        with self.lock:
            b = fh not in self.files
            if b:
                if fh.closed:
                    fh.open()
                    self.files[fh] = 0
                else:
                    self.files[fh] = 1
                    self.keep.add(fh)
                self.past.append(fh)
            fh.seek(offset, whence)
            data = fh.read(bytecount)
            if b:
                self._trim()
        return data

    def write(
        self,
        fh: FileHandle,
        /,
        offset: int,
        data: bytes,
        whence: int = 0,
    ) -> int:
        """Write bytes to binary file.

        Parameters:
            fh:
                File handle to write to.
            offset:
                Position in file to start writing from relative to the
                position indicated by `whence`.
            value:
                Bytes to write.
            whence:
                Relative position of offset.
                0 (`os.SEEK_SET`) beginning of file (default).
                1 (`os.SEEK_CUR`) current position.
                2 (`os.SEEK_END`) end of file.

        """
        with self.lock:
            b = fh not in self.files
            if b:
                if fh.closed:
                    fh.open()
                    self.files[fh] = 0
                else:
                    self.files[fh] = 1
                    self.keep.add(fh)
                self.past.append(fh)
            fh.seek(offset, whence)
            written = fh.write(data)
            if b:
                self._trim()
        return written

    def _trim(self) -> None:
        """Trim file cache."""
        index = 0
        size = len(self.past)
        while index < size > self.size:
            fh = self.past[index]
            if fh not in self.keep and self.files[fh] <= 0:
                fh.close()
                del self.files[fh]
                del self.past[index]
                size -= 1
            else:
                index += 1

    def __len__(self) -> int:
        """Return number of open files."""
        return len(self.files)

    def __repr__(self) -> str:
        return f'<tifffile.FileCache @0x{id(self):016X}>'

