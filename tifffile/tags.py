"""TIFF tag classes."""

from __future__ import annotations

import os
import struct
import warnings
from typing import TYPE_CHECKING, cast, final, overload

from .enums import DATATYPE
from .utils import TiffFileError, enumarg, enumstr, logger, pformat

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence
    from typing import Any, TypeAlias

    from .codecs import TiffFormat
    from .fileio import FileHandle
    from .tifffile import TiffFile, TiffWriter

    TagTuple: TypeAlias = tuple[int | str, int | str, int | None, Any, bool]


@final
class TiffTagRegistry:
    """Registry of TIFF tag codes and names.

    Map tag codes and names to names and codes respectively.
    One tag code may be registered with several names, for example, 34853 is
    used for GPSTag or OlympusSIS2.
    Different tag codes may be registered with the same name, for example,
    37387 and 41483 are both named FlashEnergy.

    Parameters:
        arg: Mapping of codes to names.

    Examples:
        >>> tags = TiffTagRegistry([(34853, 'GPSTag'), (34853, 'OlympusSIS2')])
        >>> tags.add(37387, 'FlashEnergy')
        >>> tags.add(41483, 'FlashEnergy')
        >>> tags['GPSTag']
        34853
        >>> tags[34853]
        'GPSTag'
        >>> tags.getall(34853)
        ['GPSTag', 'OlympusSIS2']
        >>> tags.getall('FlashEnergy')
        [37387, 41483]
        >>> len(tags)
        4

    """

    __slots__ = ('_dict', '_list')

    _dict: dict[int | str, str | int]
    _list: list[dict[int | str, str | int]]

    def __init__(
        self,
        arg: TiffTagRegistry | dict[int, str] | Sequence[tuple[int, str]],
        /,
    ) -> None:
        self._dict = {}
        self._list = [self._dict]
        self.update(arg)

    def update(
        self,
        arg: TiffTagRegistry | dict[int, str] | Sequence[tuple[int, str]],
        /,
    ) -> None:
        """Add mapping of codes to names to registry.

        Parameters:
            arg: Mapping of codes to names.

        """
        if isinstance(arg, TiffTagRegistry):
            self._list.extend(arg._list)
            return
        if isinstance(arg, dict):
            arg = list(arg.items())
        for code, name in arg:
            self.add(code, name)

    def add(self, code: int, name: str, /) -> None:
        """Add code and name to registry."""
        for d in self._list:
            if code in d and d[code] == name:
                break
            if code not in d and name not in d:
                d[code] = name
                d[name] = code
                break
        else:
            self._list.append({code: name, name: code})

    def items(self) -> list[tuple[int, str]]:
        """Return all registry items as (code, name)."""
        items = (
            i for d in self._list for i in d.items() if isinstance(i[0], int)
        )
        return sorted(items, key=lambda i: i[0])  # type: ignore[arg-type]

    @overload
    def get(self, key: int, /, default: None) -> str | None: ...

    @overload
    def get(self, key: str, /, default: None) -> int | None: ...

    @overload
    def get(self, key: int, /, default: str) -> str: ...

    def get(
        self, key: int | str, /, default: str | None = None
    ) -> str | int | None:
        """Return first code or name if exists, else default.

        Parameters:
            key: tag code or name to lookup.
            default: value to return if key is not found.

        """
        for d in self._list:
            if key in d:
                return d[key]
        return default

    @overload
    def getall(self, key: int, /, default: None) -> list[str] | None: ...

    @overload
    def getall(self, key: str, /, default: None) -> list[int] | None: ...

    @overload
    def getall(self, key: int, /, default: list[str]) -> list[str]: ...

    def getall(
        self, key: int | str, /, default: list[str] | None = None
    ) -> list[str] | list[int] | None:
        """Return list of all codes or names if exists, else default.

        Parameters:
            key: tag code or name to lookup.
            default: value to return if key is not found.

        """
        result = [d[key] for d in self._list if key in d]
        return result if result else default  # type: ignore[return-value]

    @overload
    def __getitem__(self, key: int, /) -> str: ...

    @overload
    def __getitem__(self, key: str, /) -> int: ...

    def __getitem__(self, key: int | str, /) -> int | str:
        """Return first code or name. Raise KeyError if not found."""
        for d in self._list:
            if key in d:
                return d[key]
        raise KeyError(key)

    def __delitem__(self, key: int | str, /) -> None:
        """Delete all tags of code or name."""
        found = False
        for d in self._list:
            if key in d:
                found = True
                value = d[key]
                del d[key]
                del d[value]
        if not found:
            raise KeyError(key)

    def __contains__(self, item: int | str, /) -> bool:
        """Return if code or name is in registry."""
        return any(item in d for d in self._list)

    def __iter__(self) -> Iterator[tuple[int, str]]:
        """Return iterator over all items in registry."""
        return iter(self.items())

    def __len__(self) -> int:
        """Return number of registered tags."""
        size = 0
        for d in self._list:
            size += len(d)
        return size // 2

    def __repr__(self) -> str:
        return f'<tifffile.TiffTagRegistry @0x{id(self):016X}>'

    def __str__(self) -> str:
        return 'TiffTagRegistry(((\n  {}\n))'.format(
            ',\n  '.join(f'({code}, {name!r})' for code, name in self.items())
        )


@final
class TiffTag:
    """TIFF tag structure.

    TiffTag instances are not thread-safe. All attributes are read-only.

    Parameters:
        parent:
            TIFF file tag belongs to.
        offset:
            Position of tag structure in file.
        code:
            Decimal code of tag.
        dtype:
            Data type of tag value item.
        count:
            Number of items in tag value.
        valueoffset:
            Position of tag value in file.

    """

    __slots__ = (
        '_value',
        'code',
        'count',
        'dtype',
        'offset',
        'parent',
        'valueoffset',
    )

    parent: TiffFile | TiffWriter
    """TIFF file tag belongs to."""

    offset: int
    """Position of tag structure in file."""

    code: int
    """Decimal code of tag."""

    dtype: int
    """:py:class:`DATATYPE` of tag value item."""

    count: int
    """Number of items in tag value."""

    valueoffset: int
    """Position of tag value in file."""

    _value: Any

    def __init__(
        self,
        parent: TiffFile | TiffWriter,
        offset: int,
        code: int,
        dtype: DATATYPE | int,
        count: int,
        value: Any,
        valueoffset: int,
        /,
    ) -> None:
        self.parent = parent
        self.offset = int(offset)
        self.code = int(code)
        self.count = int(count)
        self._value = value
        self.valueoffset = valueoffset
        try:
            self.dtype = DATATYPE(dtype)
        except ValueError:
            self.dtype = int(dtype)

    @classmethod
    def fromfile(
        cls,
        parent: TiffFile,
        /,
        *,
        offset: int | None = None,
        header: bytes | None = None,
        validate: bool = True,
    ) -> TiffTag:
        """Return TiffTag instance from file.

        Parameters:
            parent:
                TiffFile instance tag is read from.
            offset:
                Position of tag structure in file.
                The default is the position of the file handle.
            header:
                Tag structure as bytes.
                The default is read from the file.
            validate:
                Raise TiffFileError if data type or value offset are invalid.

        Raises:
            TiffFileError:
                Data type or value offset are invalid and `validate` is *True*.

        """
        from .tifffile import TIFF

        tiff = parent.tiff
        fh = parent.filehandle

        if header is None:
            if offset is None:
                offset = fh.tell()
            else:
                fh.seek(offset)
            header = fh.read(tiff.tagsize)
        elif offset is None:
            offset = fh.tell()

        valueoffset = offset + tiff.tagsize - tiff.tagoffsetthreshold
        code, dtype, count, value = struct.unpack(
            tiff.tagformat1 + tiff.tagformat2[1:], header
        )

        try:
            valueformat = TIFF.DATA_FORMATS[dtype]
        except KeyError as exc:
            msg = (
                f'<tifffile.TiffTag {code} @{offset}> '
                f'invalid data type {dtype!r}'
            )
            if validate:
                raise TiffFileError(msg) from exc
            logger().error(msg)
            return cls(parent, offset, code, dtype, count, None, 0)

        valuesize = count * struct.calcsize(valueformat)
        if (
            valuesize > tiff.tagoffsetthreshold
            or code in TIFF.TAG_READERS  # TODO: only works with offsets?
        ):
            valueoffset = struct.unpack(tiff.offsetformat, value)[0]
            if validate and code in TIFF.TAG_LOAD:
                value = TiffTag._read_value(
                    parent, offset, code, dtype, count, valueoffset
                )
            elif valueoffset < 8 or valueoffset + valuesize > fh.size:
                msg = (
                    f'<tifffile.TiffTag {code} @{offset}> '
                    f'invalid value offset {valueoffset}'
                )
                if validate:
                    raise TiffFileError(msg)
                logger().warning(msg)
                value = None
            elif code in TIFF.TAG_LOAD:
                value = TiffTag._read_value(
                    parent, offset, code, dtype, count, valueoffset
                )
            else:
                value = None
        elif dtype in {1, 2, 7}:
            # BYTES, ASCII, UNDEFINED
            value = value[:valuesize]
        elif (
            tiff.is_ndpi
            and count == 1
            and dtype in {4, 9, 13}
            and value[4:] != b'\x00\x00\x00\x00'
        ):
            # NDPI IFD or LONG, for example, in StripOffsets or StripByteCounts
            value = struct.unpack('<Q', value)
        else:
            fmt = (
                f'{tiff.byteorder}'
                f'{count * int(valueformat[0])}'
                f'{valueformat[1]}'
            )
            value = struct.unpack(fmt, value[:valuesize])

        value = TiffTag._process_value(value, code, dtype, offset)

        return cls(parent, offset, code, dtype, count, value, valueoffset)

    @classmethod
    def from_ifd_entry(
        cls,
        parent: TiffFile,
        entry: Any,
        /,
        *,
        validate: bool = True,
    ) -> TiffTag:
        """Return TiffTag from C++ IfdEntry.

        Parameters:
            parent: TiffFile instance tag belongs to.
            entry: IfdEntry from C++ _tifffile_ext module.
            validate: Raise TiffFileError if data type or offset are invalid.

        """
        from .tifffile import TIFF

        tiff = parent.tiff
        fh = parent.filehandle

        code = entry.code
        dtype = entry.dtype
        count = entry.count
        offset = entry.tag_file_offset
        valueoffset = entry.valueoffset

        try:
            valueformat = TIFF.DATA_FORMATS[dtype]
        except KeyError as exc:
            msg = (
                f'<tifffile.TiffTag {code} @{offset}> '
                f'invalid data type {dtype!r}'
            )
            if validate:
                raise TiffFileError(msg) from exc
            logger().error(msg)
            return cls(parent, offset, code, dtype, count, None, 0)

        valuesize = count * struct.calcsize(valueformat)

        if (
            not entry.is_inline
            or code in TIFF.TAG_READERS
        ):
            # out-of-line value or special reader
            if not entry.is_inline:
                # valueoffset already decoded by C++
                pass
            else:
                # inline but needs TAG_READER: valueoffset points to
                # value field in tag entry
                pass

            if validate and code in TIFF.TAG_LOAD:
                value = TiffTag._read_value(
                    parent, offset, code, dtype, count, valueoffset
                )
            elif valueoffset < 8 or valueoffset + valuesize > fh.size:
                msg = (
                    f'<tifffile.TiffTag {code} @{offset}> '
                    f'invalid value offset {valueoffset}'
                )
                if validate:
                    raise TiffFileError(msg)
                logger().warning(msg)
                value = None
            elif code in TIFF.TAG_LOAD:
                value = TiffTag._read_value(
                    parent, offset, code, dtype, count, valueoffset
                )
            else:
                value = None
        else:
            # inline value
            value_bytes = entry.get_inline_bytes()
            if dtype in {1, 2, 7}:
                # BYTES, ASCII, UNDEFINED
                value = bytes(value_bytes[:valuesize])
            elif (
                tiff.is_ndpi
                and count == 1
                and dtype in {4, 9, 13}
                and len(value_bytes) >= 8
                and value_bytes[4:8] != b'\x00\x00\x00\x00'
            ):
                # NDPI 64-bit value in 32-bit tag
                value = struct.unpack('<Q', value_bytes[:8])
            else:
                fmt = (
                    f'{tiff.byteorder}'
                    f'{count * int(valueformat[0])}'
                    f'{valueformat[1]}'
                )
                value = struct.unpack(fmt, bytes(value_bytes[:valuesize]))

        value = TiffTag._process_value(value, code, dtype, offset)

        return cls(parent, offset, code, dtype, count, value, valueoffset)

    @staticmethod
    def _read_value(
        parent: TiffFile | TiffWriter,
        offset: int,
        code: int,
        dtype: int,
        count: int,
        valueoffset: int,
        /,
    ) -> Any:
        """Read tag value from file."""
        from .tifffile import TIFF

        try:
            valueformat = TIFF.DATA_FORMATS[dtype]
        except KeyError as exc:
            msg = f'<TiffTag {code} @{offset}> invalid {dtype=!r}'
            raise TiffFileError(msg) from exc

        fh = parent.filehandle
        byteorder = parent.tiff.byteorder
        offsetsize = parent.tiff.offsetsize

        valuesize = count * struct.calcsize(valueformat)
        if valueoffset < 8 or valueoffset + valuesize > fh.size:
            msg = f'<TiffTag {code} @{offset}> invalid {valueoffset=}'
            raise TiffFileError(msg)
        # if valueoffset % 2:
        #     logger().warning(
        #         f'<tifffile.TiffTag {code} @{offset}> '
        #         'value does not begin on word boundary'
        #     )

        fh.seek(valueoffset)
        if code in TIFF.TAG_READERS:
            readfunc = TIFF.TAG_READERS[code]
            try:
                value = readfunc(fh, byteorder, dtype, count, offsetsize)
            except Exception as exc:
                logger().warning(
                    f'<tifffile.TiffTag {code} @{offset}> raised {exc!r:.128}'
                )
            else:
                return value

        if dtype in {1, 2, 7}:
            # BYTES, ASCII, UNDEFINED
            value = fh.read(valuesize)
            if len(value) != valuesize:
                logger().warning(
                    f'<tifffile.TiffTag {code} @{offset}> '
                    'could not read all values'
                )
        elif code not in TIFF.TAG_TUPLE and count > 1024:
            from .metadata import read_numpy

            value = read_numpy(fh, byteorder, dtype, count, offsetsize)
        else:
            value = struct.unpack(
                f'{byteorder}{count * int(valueformat[0])}{valueformat[1]}',
                fh.read(valuesize),
            )
        return value

    @staticmethod
    def _process_value(
        value: Any, code: int, dtype: int, offset: int, /
    ) -> Any:
        """Process tag value."""
        from .tifffile import TIFF

        if (
            value is None
            or dtype in {1, 7}  # BYTE, UNDEFINED
            or code in TIFF.TAG_READERS
            or not isinstance(value, (bytes, str, tuple))
        ):
            return value

        if dtype == 2:
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
                        f'<tifffile.TiffTag {code} @{offset}> '
                        f'coercing invalid ASCII to bytes, due to {exc!r:.128}'
                    )
            return value

        if code in TIFF.TAG_ENUM:
            t = TIFF.TAG_ENUM[code]
            try:
                value = tuple(t(v) for v in value)
            except ValueError as exc:
                if code not in {259, 317}:  # ignore compression/predictor
                    logger().warning(
                        f'<tifffile.TiffTag {code} @{offset}> '
                        f'raised {exc!r:.128}'
                    )

        if len(value) == 1 and code not in TIFF.TAG_TUPLE:
            value = value[0]

        return value

    @property
    def value(self) -> Any:
        """Value of tag, delay-loaded from file if necessary."""
        if self._value is None:
            # print(
            #     f'_read_value {self.code} {TIFF.TAGS.get(self.code)} '
            #     f'{self.dtype}[{self.count}] @{self.valueoffset} '
            # )
            fh = self.parent.filehandle
            with fh.lock:
                closed = fh.closed
                if closed:
                    # this is an inefficient resort in case a user delay loads
                    # tag values from a TiffPage with a closed FileHandle.
                    warnings.warn(
                        f'{self!r} reading value from closed file',
                        UserWarning,
                        stacklevel=2,
                    )
                    fh.open()
                try:
                    value = TiffTag._read_value(
                        self.parent,
                        self.offset,
                        self.code,
                        self.dtype,
                        self.count,
                        self.valueoffset,
                    )
                finally:
                    if closed:
                        fh.close()
            self._value = TiffTag._process_value(
                value,
                self.code,
                self.dtype,
                self.offset,
            )
        return self._value

    @value.setter
    def value(self, value: Any, /) -> None:
        self._value = value

    @property
    def dtype_name(self) -> str:
        """Name of data type of tag value."""
        try:
            return self.dtype.name  # type: ignore[attr-defined]
        except AttributeError:
            return f'TYPE{self.dtype}'

    @property
    def name(self) -> str:
        """Name of tag from :py:attr:`_TIFF.TAGS` registry."""
        from .tifffile import TIFF

        return TIFF.TAGS.get(self.code, str(self.code))

    @property
    def dataformat(self) -> str:
        """Data type as `struct.pack` format."""
        from .tifffile import TIFF

        return TIFF.DATA_FORMATS[self.dtype]

    @property
    def valuebytecount(self) -> int:
        """Number of bytes of tag value in file."""
        from .tifffile import TIFF

        return self.count * struct.calcsize(TIFF.DATA_FORMATS[self.dtype])

    def astuple(self) -> TagTuple:
        """Return tag code, dtype, count, and encoded value.

        The encoded value is read from file if necessary.

        """
        from .tifffile import TIFF

        if isinstance(self.value, bytes):
            value = self.value
        else:
            tiff = self.parent.tiff
            dataformat = TIFF.DATA_FORMATS[self.dtype]
            count = self.count * int(dataformat[0])
            fmt = f'{tiff.byteorder}{count}{dataformat[1]}'
            try:
                if self.dtype == 2:
                    # ASCII
                    value = struct.pack(fmt, self.value.encode('ascii'))
                    if len(value) != count:
                        raise ValueError
                elif count == 1 and not isinstance(self.value, tuple):
                    value = struct.pack(fmt, self.value)
                else:
                    value = struct.pack(fmt, *self.value)
            except Exception as exc:
                if tiff.is_ndpi and count == 1:
                    msg = 'cannot pack 64-bit NDPI value to 32-bit dtype'
                    raise ValueError(msg) from exc
                fh = self.parent.filehandle
                pos = fh.tell()
                fh.seek(self.valueoffset)
                value = fh.read(struct.calcsize(fmt))
                fh.seek(pos)
        return self.code, int(self.dtype), self.count, value, True

    def overwrite(
        self,
        value: Any,
        /,
        *,
        dtype: DATATYPE | int | str | None = None,
        erase: bool = True,
    ) -> TiffTag:
        """Write new tag value to file and return new TiffTag instance.

        Warning: changing tag values in TIFF files might result in corrupted
        files or have unexpected side effects.

        The packed value is appended to the file if it is longer than the
        old value. The file position is left where it was.

        Overwriting tag values in NDPI files > 4 GB is only supported if
        single integer values and new offsets do not exceed the 32-bit range.

        Parameters:
            value:
                New tag value to write.
                Must be compatible with the `struct.pack` formats corresponding
                to the tag's data type.
            dtype:
                New tag data type. By default, the data type is not changed.
            erase:
                Overwrite previous tag values in file with zeros.

        Raises:
            struct.error:
                Value is not compatible with dtype or new offset exceeds
                TIFF size limit.
            ValueError:
                Invalid value or dtype, or old integer value in NDPI files
                exceeds 32-bit range.

        """
        from .tifffile import TIFF

        if self.offset < 8 or self.valueoffset < 8:
            msg = f'cannot rewrite tag at offset {self.offset} < 8'
            raise ValueError(msg)

        if hasattr(value, 'filehandle'):
            # passing a TiffFile instance is deprecated and no longer required
            # since 2021.7.30
            msg = (
                'TiffTag.overwrite got an unexpected TiffFile instance '
                'as first argument'
            )
            raise TypeError(msg)

        fh = self.parent.filehandle
        tiff = self.parent.tiff
        if tiff.is_ndpi:
            # only support files < 4GB
            if self.count == 1 and self.dtype in {4, 13}:
                if isinstance(self.value, tuple):
                    v = self.value[0]
                else:
                    v = self.value
                if v > 4294967295:
                    msg = 'cannot patch NDPI > 4 GB files'
                    raise ValueError(msg)
            tiff = TIFF.CLASSIC_LE

        if value is None:
            value = b''
        if dtype is None:
            dtype = self.dtype
        elif isinstance(dtype, str):
            if len(dtype) > 1 and dtype[0] in '<>|=':
                dtype = dtype[1:]
            try:
                dtype = TIFF.DATA_DTYPES[dtype]
            except KeyError:
                msg = f'unknown data type {dtype!r}'
                raise ValueError(msg) from None
        else:
            dtype = enumarg(DATATYPE, dtype)

        packedvalue: bytes | None = None
        dataformat: str
        try:
            dataformat = TIFF.DATA_FORMATS[dtype]
        except KeyError:
            msg = f'unknown data type {dtype!r}'
            raise ValueError(msg) from None

        if dtype == 2:
            # strings
            if isinstance(value, str):
                # enforce 7-bit ASCII on Unicode strings
                try:
                    value = value.encode('ascii')
                except UnicodeEncodeError as exc:
                    msg = 'TIFF strings must be 7-bit ASCII'
                    raise ValueError(msg) from exc
            elif not isinstance(value, bytes):
                msg = 'TIFF strings must be 7-bit ASCII'
                raise ValueError(msg)
            if len(value) == 0 or value[-1:] != b'\x00':
                value += b'\x00'
            count = len(value)
            value = (value,)

        elif isinstance(value, bytes):
            # pre-packed binary data
            dtsize = struct.calcsize(dataformat)
            if len(value) % dtsize:
                msg = 'invalid packed binary data'
                raise ValueError(msg)
            count = len(value) // dtsize
            packedvalue = value
            value = (value,)

        else:
            try:
                count = len(value)
            except TypeError:
                value = (value,)
                count = 1
            if dtype in {5, 10}:
                if count < 2 or count % 2:
                    msg = 'invalid RATIONAL value'
                    raise ValueError(msg)
                count //= 2  # rational

        if packedvalue is None:
            packedvalue = struct.pack(
                f'{tiff.byteorder}{count * int(dataformat[0])}{dataformat[1]}',
                *value,
            )
        newsize = len(packedvalue)
        oldsize = self.count * struct.calcsize(TIFF.DATA_FORMATS[self.dtype])
        valueoffset = self.valueoffset

        pos = fh.tell()
        try:
            if dtype != self.dtype:
                # rewrite data type
                fh.seek(self.offset + 2)
                fh.write(struct.pack(tiff.byteorder + 'H', dtype))

            if oldsize <= tiff.tagoffsetthreshold:
                if newsize <= tiff.tagoffsetthreshold:
                    # inline -> inline: overwrite
                    fh.seek(self.offset + 4)
                    fh.write(struct.pack(tiff.tagformat2, count, packedvalue))
                else:
                    # inline -> separate: append to file
                    fh.seek(0, os.SEEK_END)
                    valueoffset = fh.tell()
                    if valueoffset % 2:
                        # value offset must begin on a word boundary
                        fh.write(b'\x00')
                        valueoffset += 1
                    # write new offset
                    fh.seek(self.offset + 4)
                    fh.write(
                        struct.pack(
                            tiff.tagformat2,
                            count,
                            struct.pack(tiff.offsetformat, valueoffset),
                        )
                    )
                    # write new value
                    fh.seek(valueoffset)
                    fh.write(packedvalue)

            elif newsize <= tiff.tagoffsetthreshold:
                # separate -> inline: erase old value
                valueoffset = (
                    self.offset + 4 + struct.calcsize(tiff.tagformat2[:2])
                )
                fh.seek(self.offset + 4)
                fh.write(struct.pack(tiff.tagformat2, count, packedvalue))
                if erase:
                    fh.seek(self.valueoffset)
                    fh.write(b'\x00' * oldsize)
            elif newsize <= oldsize or self.valueoffset + oldsize == fh.size:
                # separate -> separate smaller: overwrite, erase remaining
                fh.seek(self.offset + 4)
                fh.write(struct.pack(tiff.tagformat2[:2], count))
                fh.seek(self.valueoffset)
                fh.write(packedvalue)
                if erase and oldsize - newsize > 0:
                    fh.write(b'\x00' * (oldsize - newsize))
            else:
                # separate -> separate larger: erase old value, append to file
                fh.seek(0, os.SEEK_END)
                valueoffset = fh.tell()
                if valueoffset % 2:
                    # value offset must begin on a word boundary
                    fh.write(b'\x00')
                    valueoffset += 1
                # write offset
                fh.seek(self.offset + 4)
                fh.write(
                    struct.pack(
                        tiff.tagformat2,
                        count,
                        struct.pack(tiff.offsetformat, valueoffset),
                    )
                )
                # write value
                fh.seek(valueoffset)
                fh.write(packedvalue)
                if erase:
                    fh.seek(self.valueoffset)
                    fh.write(b'\x00' * oldsize)

        finally:
            fh.seek(pos)  # must restore file position

        return TiffTag(
            self.parent,
            self.offset,
            self.code,
            dtype,
            count,
            value,
            valueoffset,
        )

    def _fix_lsm_bitspersample(self) -> None:
        """Correct LSM bitspersample tag.

        Old LSM writers may use a separate region for two 16-bit values,
        although they fit into the tag value element of the tag.

        """
        if self.code != 258 or self.count != 2:
            return
        # TODO: test this case; need example file
        logger().warning(f'{self!r} correcting LSM bitspersample tag')
        value = struct.pack('<HH', *self.value)
        self.valueoffset = struct.unpack('<I', value)[0]
        self.parent.filehandle.seek(self.valueoffset)
        self.value = struct.unpack('<HH', self.parent.filehandle.read(4))

    def __repr__(self) -> str:
        from .tifffile import TIFF

        name = '|'.join(TIFF.TAGS.getall(self.code, []))
        if name:
            name = ' ' + name
        return f'<tifffile.TiffTag {self.code}{name} @{self.offset}>'

    def __str__(self) -> str:
        return self._str()

    def _str(self, detail: int = 0, width: int = 79) -> str:
        """Return string containing information about TiffTag."""
        from .tifffile import TIFF

        height = 1 if detail <= 0 else 8 * detail
        dtype = self.dtype_name
        if self.count > 1:
            dtype += f'[{self.count}]'
        name = '|'.join(TIFF.TAGS.getall(self.code, []))
        if name:
            name = f'{self.code} {name} @{self.offset}'
        else:
            name = f'{self.code} @{self.offset}'
        line = f'TiffTag {name} {dtype} @{self.valueoffset} '
        line = line[:width]
        try:
            value = self.value
        except TiffFileError:
            value = 'CORRUPTED'
        else:
            try:
                if self.count == 1:
                    value = enumstr(value)
                else:
                    value = pformat(tuple(enumstr(v) for v in value))
            except Exception:
                if not isinstance(value, (tuple, list)):
                    pass
                elif height == 1:
                    value = value[:256]
                elif len(value) > 2048:
                    value = (
                        value[:1024] + value[-1024:]  # type: ignore[operator]
                    )
                value = pformat(value, width=width, height=height)
        if detail <= 0:
            line += '= '
            line += value[:width]
            line = line[:width]
        else:
            line += '\n' + value
        return line


@final
class TiffTags:
    """Multidict-like interface to TiffTag instances in TiffPage.

    Differences to a regular dict:

    - values are instances of :py:class:`TiffTag`.
    - keys are :py:attr:`TiffTag.code` (int).
    - multiple values can be stored per key.
    - can be indexed by :py:attr:`TiffTag.name` (`str`), slower than by key.
    - `iter()` returns values instead of keys.
    - `values()` and `items()` contain all values sorted by offset.
    - `len()` returns number of all values.
    - `get()` takes optional index argument.
    - some functions are not implemented, such as, `update` and `pop`.

    """

    __slots__ = ('_dict', '_list')

    _dict: dict[int, TiffTag]
    _list: list[dict[int, TiffTag]]

    def __init__(self) -> None:
        self._dict = {}
        self._list = [self._dict]

    def add(self, tag: TiffTag, /) -> None:
        """Add tag."""
        code = tag.code
        for d in self._list:
            if code not in d:
                d[code] = tag
                break
        else:
            self._list.append({code: tag})

    def keys(self) -> list[int]:
        """Return codes of all tags."""
        return list(self._dict.keys())

    def values(self) -> list[TiffTag]:
        """Return all tags in order they are stored in file."""
        tags = (t for d in self._list for t in d.values())
        return sorted(tags, key=lambda t: t.offset)

    def items(self) -> list[tuple[int, TiffTag]]:
        """Return all (code, tag) pairs in order tags are stored in file."""
        items = (i for d in self._list for i in d.items())
        return sorted(items, key=lambda i: i[1].offset)

    def valueof(
        self,
        key: int | str,
        /,
        default: Any = None,
        index: int | None = None,
    ) -> Any:
        """Return value of tag by code or name if exists, else default.

        Parameters:
            key:
                Code or name of tag to return.
            default:
                Another value to return if specified tag is corrupted or
                not found.
            index:
                Specifies tag in case of multiple tags with identical code.
                The default is the first tag.

        """
        tag = self.get(key, default=None, index=index)
        if tag is None:
            return default
        try:
            return tag.value
        except TiffFileError:
            return default  # corrupted tag

    def get(
        self,
        key: int | str,
        /,
        default: TiffTag | None = None,
        index: int | None = None,
    ) -> TiffTag | None:
        """Return tag by code or name if exists, else default.

        Parameters:
            key:
                Code or name of tag to return.
            default:
                Another tag to return if specified tag is corrupted or
                not found.
            index:
                Specifies tag in case of multiple tags with identical code.
                The default is the first tag.

        """
        if index is None:
            if key in self._dict:
                return self._dict[cast(int, key)]
            if not isinstance(key, str):
                return default
            index = 0
        try:
            tags = self._list[index]
        except IndexError:
            return default
        if key in tags:
            return tags[cast(int, key)]
        if not isinstance(key, str):
            return default
        for tag in tags.values():
            if tag.name == key:
                return tag
        return default

    def getall(
        self,
        key: int | str,
        /,
        default: Any = None,
    ) -> list[TiffTag] | None:
        """Return list of all tags by code or name if exists, else default.

        Parameters:
            key:
                Code or name of tags to return.
            default:
                Value to return if no tags are found.

        """
        result: list[TiffTag] = []
        for tags in self._list:
            if key in tags:
                result.append(tags[cast(int, key)])
            else:
                break
        if result:
            return result
        if not isinstance(key, str):
            return default
        for tags in self._list:
            for tag in tags.values():
                if tag.name == key:
                    result.append(tag)
                    break
            if not result:
                break
        return result if result else default

    def __getitem__(self, key: int | str, /) -> TiffTag:
        """Return first tag by code or name. Raise KeyError if not found."""
        if key in self._dict:
            return self._dict[cast(int, key)]
        if not isinstance(key, str):
            raise KeyError(key)
        for tag in self._dict.values():
            if tag.name == key:
                return tag
        raise KeyError(key)

    def __setitem__(self, code: int, tag: TiffTag, /) -> None:
        """Add tag."""
        assert tag.code == code
        self.add(tag)

    def __delitem__(self, key: int | str, /) -> None:
        """Delete all tags by code or name."""
        found = False
        for tags in self._list:
            if key in tags:
                found = True
                del tags[cast(int, key)]
            else:
                break
        if found:
            return
        if not isinstance(key, str):
            raise KeyError(key)
        for tags in self._list:
            for tag in tags.values():
                if tag.name == key:
                    del tags[tag.code]
                    found = True
                    break
            else:
                break
        if not found:
            raise KeyError(key)
        return

    def __contains__(self, item: object, /) -> bool:
        """Return if tag is in map."""
        if item in self._dict:
            return True
        if not isinstance(item, str):
            return False
        return any(tag.name == item for tag in self._dict.values())

    def __iter__(self) -> Iterator[TiffTag]:
        """Return iterator over all tags."""
        return iter(self.values())

    def __len__(self) -> int:
        """Return number of tags."""
        size = 0
        for d in self._list:
            size += len(d)
        return size

    def __repr__(self) -> str:
        return f'<tifffile.TiffTags @0x{id(self):016X}>'

    def __str__(self) -> str:
        return self._str()

    def _str(self, detail: int = 0, width: int = 79) -> str:
        """Return string with information about TiffTags."""
        info = []
        tlines = []
        vlines = []
        for tag in self:
            value = tag._str(width=width + 1)
            tlines.append(value[:width].strip())
            if detail > 0 and len(value) > width:
                try:
                    value = tag.value
                except Exception:  # noqa: S112
                    # delay load failed or closed file
                    continue
                if tag.code in {273, 279, 324, 325}:
                    if detail < 1:
                        value = value[:256]
                    elif len(value) > 1024:
                        value = value[:512] + value[-512:]
                    value = pformat(value, width=width, height=detail * 3)
                else:
                    value = pformat(value, width=width, height=detail * 8)
                if tag.count > 1:
                    vlines.append(
                        f'{tag.name} {tag.dtype_name}[{tag.count}]\n{value}'
                    )
                else:
                    vlines.append(f'{tag.name}\n{value}')
        info.append('\n'.join(tlines))
        if detail > 0 and vlines:
            info.append('\n')
            info.append('\n\n'.join(vlines))
        return '\n'.join(info)
