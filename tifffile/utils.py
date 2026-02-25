# utils.py

"""Utility functions for tifffile."""

from __future__ import annotations

import binascii
import collections
import enum
import io
import logging
import math
import os
import re
import sys
import warnings
from collections.abc import Callable, Iterable, Sequence
from datetime import datetime as DateTime
from datetime import timedelta as TimeDelta
from typing import TYPE_CHECKING, cast, overload

import numpy

from .enums import ORIENTATION

if TYPE_CHECKING:
    from collections.abc import Iterator
    from typing import Any

    from numpy.typing import ArrayLike, DTypeLike, NDArray


class TiffFileError(ValueError):
    """Exception to indicate invalid TIFF structure."""


def identityfunc(arg: Any, /, *args: Any, **kwargs: Any) -> Any:
    """Single argument identity function.

    >>> identityfunc('arg')
    'arg'

    """
    return arg


def nullfunc(*args: Any, **kwargs: Any) -> None:
    """Null function.

    >>> nullfunc('arg', kwarg='kwarg')

    """
    return


def sequence(value: Any, /) -> Sequence[Any]:
    """Return tuple containing value if value is not tuple or list.

    >>> sequence(1)
    (1,)
    >>> sequence([1])
    [1]
    >>> sequence('ab')
    ('ab',)

    """
    return value if isinstance(value, (tuple, list)) else (value,)


def product(iterable: Iterable[int], /) -> int:
    """Return product of integers.

    Equivalent of ``math.prod(iterable)``, but multiplying NumPy integers
    does not overflow.

    >>> product([2**8, 2**30])
    274877906944
    >>> product([])
    1

    """
    prod = 1
    for i in iterable:
        prod *= int(i)
    return prod


def peek_iterator(iterator: Iterator[Any], /) -> tuple[Any, Iterator[Any]]:
    """Return first item of iterator and iterator.

    >>> first, it = peek_iterator(iter((0, 1, 2)))
    >>> first
    0
    >>> list(it)
    [0, 1, 2]

    """
    first = next(iterator)

    def newiter(
        first: Any = first, iterator: Iterator[Any] = iterator
    ) -> Iterator[Any]:
        yield first
        yield from iterator

    return first, newiter()


def natural_sorted(iterable: Iterable[str], /) -> list[str]:
    """Return human-sorted list of strings.

    Use to sort file names.

    >>> natural_sorted(['f1', 'f2', 'f10'])
    ['f1', 'f2', 'f10']

    """

    def sortkey(x: str, /) -> list[int | str]:
        return [(int(c) if c.isdigit() else c) for c in re.split(numbers, x)]

    numbers = re.compile(r'(\d+)')
    return sorted(iterable, key=sortkey)


def format_size(size: float, /, threshold: float = 1536) -> str:
    """Return file size as string from byte size.

    >>> format_size(1234)
    '1234 B'
    >>> format_size(12345678901)
    '11.50 GiB'

    """
    if size < threshold:
        return f'{size} B'
    for unit in ('KiB', 'MiB', 'GiB', 'TiB', 'PiB'):
        size /= 1024.0
        if size < threshold:
            return f'{size:.2f} {unit}'
    return 'ginormous'


def epics_datetime(sec: int, nsec: int, /) -> DateTime:
    """Return datetime object from epicsTSSec and epicsTSNsec tag values.

    >>> epics_datetime(802117916, 103746502)
    datetime.datetime(2015, 6, 2, 11, 31, 56, 103746)

    """
    return DateTime.fromtimestamp(sec + 631152000 + nsec / 1e9)


def excel_datetime(timestamp: float, epoch: int | None = None, /) -> DateTime:
    """Return datetime object from timestamp in Excel serial format.

    Use to convert LSM time stamps.

    >>> excel_datetime(40237.029999999795)
    datetime.datetime(2010, 2, 28, 0, 43, 11, 999982)

    """
    if epoch is None:
        epoch = 693594
    return DateTime.fromordinal(epoch) + TimeDelta(timestamp)


def julian_datetime(julianday: int, millisecond: int = 0, /) -> DateTime:
    """Return datetime from days since 1/1/4713 BC and ms since midnight.

    Convert Julian dates according to MetaMorph.

    >>> julian_datetime(2451576, 54362783)
    datetime.datetime(2000, 2, 2, 15, 6, 2, 783000)

    """
    if julianday <= 1721423:
        # return DateTime.min  # ?
        msg = f'no datetime before year 1 ({julianday=})'
        raise ValueError(msg)

    a = julianday + 1
    if a > 2299160:
        alpha = math.trunc((a - 1867216.25) / 36524.25)
        a += 1 + alpha - alpha // 4
    b = a + (1524 if a > 1721423 else 1158)
    c = math.trunc((b - 122.1) / 365.25)
    d = math.trunc(365.25 * c)
    e = math.trunc((b - d) / 30.6001)

    day = b - d - math.trunc(30.6001 * e)
    month = e - (1 if e < 13.5 else 13)
    year = c - (4716 if month > 2.5 else 4715)

    hour, millisecond = divmod(millisecond, 1000 * 60 * 60)
    minute, millisecond = divmod(millisecond, 1000 * 60)
    second, millisecond = divmod(millisecond, 1000)

    return DateTime(year, month, day, hour, minute, second, millisecond * 1000)


def byteorder_isnative(byteorder: str, /) -> bool:
    """Return if byteorder matches system's byteorder.

    >>> byteorder_isnative('=')
    True

    """
    if byteorder in {'=', sys.byteorder}:
        return True
    keys = {'big': '>', 'little': '<'}
    return keys.get(byteorder, byteorder) == keys[sys.byteorder]


def byteorder_compare(byteorder: str, other: str, /) -> bool:
    """Return if byteorders match.

    >>> byteorder_compare('<', '<')
    True
    >>> byteorder_compare('>', '<')
    False

    """
    if byteorder in {other, '|'} or other == '|':
        return True
    if byteorder == '=':
        byteorder = {'big': '>', 'little': '<'}[sys.byteorder]
    elif other == '=':
        other = {'big': '>', 'little': '<'}[sys.byteorder]
    return byteorder == other


def recarray2dict(recarray: numpy.recarray[Any, Any], /) -> dict[str, Any]:
    """Return numpy.recarray as dictionary.

    >>> r = numpy.array(
    ...     [(1.0, 2, 'a'), (3.0, 4, 'bc')],
    ...     dtype=[('x', '<f4'), ('y', '<i4'), ('s', 'S2')],
    ... )
    >>> recarray2dict(r)
    {'x': [1.0, 3.0], 'y': [2, 4], 's': ['a', 'bc']}
    >>> recarray2dict(r[1])
    {'x': 3.0, 'y': 4, 's': 'bc'}

    """
    # TODO: subarrays
    value: Any
    result = {}
    for descr in recarray.dtype.descr:
        name, dtype = descr[:2]
        value = recarray[name]
        if value.ndim == 0:
            value = value.tolist()
            if dtype[1] == 'S':
                value = bytes2str(value)
        elif value.ndim == 1:
            value = value.tolist()
            if dtype[1] == 'S':
                value = [bytes2str(v) for v in value]
        result[name] = value
    return result


def xml2dict(
    xml: str,
    /,
    *,
    sanitize: bool = True,
    prefix: tuple[str, str] | None = None,
    sep: str = ',',
) -> dict[str, Any]:
    """Return XML as dictionary.

    Parameters:
        xml: XML data to convert.
        sanitize: Remove prefix from from etree Element.
        prefix: Prefixes for dictionary keys.
        sep: Sequence separator.

    Examples:
        >>> xml2dict(
        ...     '<?xml version="1.0" ?><root attr="name"><key>1</key></root>'
        ... )
        {'root': {'key': 1, 'attr': 'name'}}
        >>> xml2dict('<level1><level2>3.5322,-3.14</level2></level1>')
        {'level1': {'level2': (3.5322, -3.14)}}

    """
    try:
        from defusedxml import ElementTree
    except ImportError:
        from xml.etree import ElementTree

    at, tx = prefix if prefix else ('', '')

    def astype(value: Any, /) -> Any:
        # return string value as int, float, bool, tuple, or unchanged
        if not isinstance(value, str):
            return value
        if sep and sep in value:
            # sequence of numbers?
            values = []
            for val in value.split(sep):
                v = astype(val)
                if isinstance(v, str):
                    return value
                values.append(v)
            return tuple(values)
        for t in (int, float, asbool):
            try:
                return t(value)
            except (TypeError, ValueError):
                pass
        return value

    def etree2dict(t: Any, /) -> dict[str, Any]:
        # adapted from https://stackoverflow.com/a/10077069/453463
        key = t.tag
        if sanitize:
            key = key.rsplit('}', 1)[-1]
        d: dict[str, Any] = {key: {} if t.attrib else None}
        children = list(t)
        if children:
            dd = collections.defaultdict(list)
            for dc in map(etree2dict, children):
                for k, v in dc.items():
                    dd[k].append(astype(v))
            d = {
                key: {
                    k: astype(v[0]) if len(v) == 1 else astype(v)
                    for k, v in dd.items()
                }
            }
        if t.attrib:
            d[key].update((at + k, astype(v)) for k, v in t.attrib.items())
        if t.text:
            text = t.text.strip()
            if children or t.attrib:
                if text:
                    d[key][tx + 'value'] = astype(text)
            else:
                d[key] = astype(text)
        return d

    return etree2dict(ElementTree.fromstring(xml))


def hexdump(
    data: bytes,
    /,
    *,
    width: int = 75,
    height: int = 24,
    snipat: float | None = 0.75,
    modulo: int = 2,
    ellipsis: str | None = None,
) -> str:
    """Return hexdump representation of bytes.

    Parameters:
        data:
            Bytes to represent as hexdump.
        width:
            Maximum width of hexdump.
        height:
            Maximum number of lines of hexdump.
        snipat:
            Approximate position at which to split long hexdump.
        modulo:
            Number of bytes represented in line of hexdump are modulus
            of this value.
        ellipsis:
            Characters to insert for snipped content of long hexdump.
            The default is '...'.

    Examples:
        >>> import binascii
        >>> hexdump(binascii.unhexlify('49492a00080000000e00fe0004000100'))
        '49 49 2a 00 08 00 00 00 0e 00 fe 00 04 00 01 00 II*.............'

    """
    size = len(data)
    if size < 1 or width < 2 or height < 1:
        return ''
    if height == 1:
        addr = b''
        bytesperline = min(
            modulo * (((width - len(addr)) // 4) // modulo), size
        )
        if bytesperline < 1:
            return ''
        nlines = 1
    else:
        addr = b'%%0%ix: ' % len(b'%x' % size)
        bytesperline = min(
            modulo * (((width - len(addr % 1)) // 4) // modulo), size
        )
        if bytesperline < 1:
            return ''
        width = 3 * bytesperline + len(addr % 1)
        nlines = (size - 1) // bytesperline + 1

    if snipat is None or snipat == 1:
        snipat = height
    elif 0 < abs(snipat) < 1:
        snipat = math.floor(height * snipat)
    if snipat < 0:
        snipat += height
    assert isinstance(snipat, int)

    blocks: list[tuple[int, bytes | None]]

    if height == 1 or nlines == 1:
        blocks = [(0, data[:bytesperline])]
        addr = b''
        height = 1
        width = 3 * bytesperline
    elif not height or nlines <= height:
        blocks = [(0, data)]
    elif snipat <= 0:
        start = bytesperline * (nlines - height)
        blocks = [(start, data[start:])]  # (start, None)
    elif snipat >= height or height < 3:
        end = bytesperline * height
        blocks = [(0, data[:end])]  # (end, None)
    else:
        end1 = bytesperline * snipat
        end2 = bytesperline * (height - snipat - 2)
        if size % bytesperline:
            end2 += size % bytesperline
        else:
            end2 += bytesperline
        blocks = [
            (0, data[:end1]),
            (size - end1 - end2, None),
            (size - end2, data[size - end2 :]),
        ]

    if ellipsis is None:
        if addr and bytesperline > 3:
            elps = b' ' * (len(addr % 1) + bytesperline // 2 * 3 - 2)
            elps += b'...'
        else:
            elps = b'...'
    else:
        elps = ellipsis.encode('cp1252')

    result = []
    for start, bstr in blocks:
        if bstr is None:
            result.append(elps)  # 'skip %i bytes' % start)
            continue
        hexstr = binascii.hexlify(bstr)
        strstr = re.sub(br'[^\x20-\x7f]', b'.', bstr)
        for i in range(0, len(bstr), bytesperline):
            h = hexstr[2 * i : 2 * i + bytesperline * 2]
            r = (addr % (i + start)) if height > 1 else addr
            r += b' '.join(h[i : i + 2] for i in range(0, 2 * bytesperline, 2))
            r += b' ' * (width - len(r))
            r += strstr[i : i + bytesperline]
            result.append(r)
    return b'\n'.join(result).decode('ascii')


def isprintable(string: str | bytes, /) -> bool:
    r"""Return if all characters in string are printable.

    >>> isprintable('abc')
    True
    >>> isprintable(b'\01')
    False

    """
    string = string.strip()
    if not string:
        return True
    if isinstance(string, str):
        return string.isprintable()
    try:
        return string.decode().isprintable()
    except UnicodeDecodeError:
        pass
    return False


def clean_whitespace(string: str, /, *, compact: bool = False) -> str:
    r"""Return string with compressed whitespace.

    >>> clean_whitespace('  a  \n\n  b ')
    'a\n b'

    """
    string = (
        string.replace('\r\n', '\n')
        .replace('\r', '\n')
        .replace('\n\n', '\n')
        .replace('\t', ' ')
        .replace('  ', ' ')
        .replace('  ', ' ')
        .replace(' \n', '\n')
    )
    if compact:
        string = (
            string.replace('\n', ' ')
            .replace('[ ', '[')
            .replace('  ', ' ')
            .replace('  ', ' ')
            .replace('  ', ' ')
        )
    return string.strip()


def indent(*args: Any) -> str:
    """Return joined string representations of objects with indented lines.

    >>> print(indent('Title:', 'Text'))
    Title:
      Text

    """
    text = '\n'.join(str(arg) for arg in args)
    return '\n'.join(
        ('  ' + line if line else line) for line in text.splitlines() if line
    )[2:]


def pformat_xml(xml: str | bytes, /) -> str:
    """Return pretty formatted XML."""
    try:
        from lxml import etree

        if not isinstance(xml, bytes):
            xml = xml.encode()
        tree = etree.parse(io.BytesIO(xml))
        xml = etree.tostring(
            tree,
            pretty_print=True,
            xml_declaration=True,
            encoding=tree.docinfo.encoding,
        )
        assert isinstance(xml, bytes)
        xml = bytes2str(xml)
    except Exception:
        if isinstance(xml, bytes):
            xml = bytes2str(xml)
        xml = xml.replace('><', '>\n<')
    return xml.replace('  ', ' ').replace('\t', ' ')


def pformat(
    arg: Any,
    /,
    *,
    height: int | None = 24,
    width: int | None = 79,
    linewidth: int | None = 288,
    compact: bool = True,
) -> str:
    """Return pretty formatted representation of object as string.

    Whitespace might be altered. Long lines are cut off.

    """
    if height is None or height < 1:
        height = 1024
    if width is None or width < 1:
        width = 256
    if linewidth is None or linewidth < 1:
        linewidth = width

    npopt = numpy.get_printoptions()
    numpy.set_printoptions(threshold=100, linewidth=width)

    if isinstance(arg, bytes) and (
        arg[:5].lower() == b'<?xml' or arg[-4:] == b'OME>'
    ):
        arg = bytes2str(arg)

    if isinstance(arg, bytes):
        if isprintable(arg):
            arg = bytes2str(arg)
            arg = clean_whitespace(arg)
        else:
            numpy.set_printoptions(**npopt)
            return hexdump(arg, width=width, height=height, modulo=1)
        arg = arg.rstrip()
    elif isinstance(arg, str):
        if arg[:5].lower() == '<?xml' or arg[-4:] == 'OME>':
            arg = arg[: 4 * width] if height == 1 else pformat_xml(arg)
        # too slow
        # else:
        #    import textwrap
        #    return '\n'.join(
        #        textwrap.wrap(arg, width=width, max_lines=height, tabsize=2)
        #    )
        arg = arg.rstrip()
    elif isinstance(arg, numpy.record):
        arg = arg.pprint()
    # elif isinstance(arg, dict):
    #     from reprlib import Repr
    #
    #     arg = Repr(
    #         maxlevel=6,
    #         maxtuple=height,
    #         maxlist=height,
    #         maxarray=height,
    #         maxdict=height,
    #         maxset=height,
    #         maxfrozenset=6,
    #         maxdeque=6,
    #         maxstring=width,
    #         maxlong=40,
    #         maxother=height,
    #         indent='  ',
    #     ).repr(arg)
    else:
        import pprint

        arg = pprint.pformat(arg, width=width, compact=compact)

    numpy.set_printoptions(**npopt)

    if height == 1:
        arg = arg[: width * width]
        arg = clean_whitespace(arg, compact=True)
        return arg[:linewidth]

    argl = list(arg.splitlines())
    if len(argl) > height:
        arg = '\n'.join(
            line[:linewidth]
            for line in (*argl[: height // 2], '...', *argl[-height // 2 :])
        )
    else:
        arg = '\n'.join(line[:linewidth] for line in argl[:height])
    return arg


def snipstr(
    string: str,
    /,
    width: int = 79,
    *,
    snipat: float | None = None,
    ellipsis: str | None = None,
) -> str:
    """Return string cut to specified length.

    Parameters:
        string:
            String to snip.
        width:
            Maximum length of returned string.
        snipat:
            Approximate position at which to split long strings.
            The default is 0.5.
        ellipsis:
            Characters to insert between splits of long strings.
            The default is '\u2026'.

    Examples:
        >>> snipstr('abcdefghijklmnop', 8, ellipsis='...')
        'abc...op'

    """
    if snipat is None:
        snipat = 0.5
    if ellipsis is None:
        ellipsis = '\u2026'
    esize = len(ellipsis)

    splitlines = string.splitlines()
    # TODO: finish and test multiline snip

    result = []
    for line in splitlines:
        linelen = len(line)
        if linelen <= width:
            result.append(string)
            continue

        if snipat is None or snipat == 1:
            split = linelen
        elif 0 < abs(snipat) < 1:
            split = math.floor(linelen * snipat)
        else:
            split = int(snipat)

        if split < 0:
            split += linelen
            split = max(split, 0)

        if esize == 0 or width < esize + 1:
            if split <= 0:
                result.append(string[-width:])
            else:
                result.append(string[:width])
        elif split <= 0:
            result.append(ellipsis + string[esize - width :])
        elif split >= linelen or width < esize + 4:
            result.append(string[: width - esize] + ellipsis)
        else:
            splitlen = linelen - width + esize
            end1 = split - splitlen // 2
            end2 = end1 + splitlen
            result.append(string[:end1] + ellipsis + string[end2:])

    return '\n'.join(result)


def enumstr(enum: Any, /) -> str:
    """Return short string representation of Enum member.

    >>> enumstr(PHOTOMETRIC.RGB)
    'RGB'

    """
    name = enum.name
    if name is None:
        name = str(enum)
    return name


def enumarg(enum: type[enum.IntEnum], arg: Any, /) -> enum.IntEnum:
    """Return enum member from its name or value.

    Parameters:
        enum: Type of IntEnum.
        arg: Name or value of enum member.

    Returns:
        Enum member matching name or value.

    Raises:
        ValueError: No enum member matches name or value.

    Examples:
        >>> enumarg(PHOTOMETRIC, 2)
        <PHOTOMETRIC.RGB: 2>
        >>> enumarg(PHOTOMETRIC, 'RGB')
        <PHOTOMETRIC.RGB: 2>

    """
    try:
        return enum(arg)
    except Exception:
        try:
            return enum[arg.upper()]
        except Exception as exc:
            msg = f'invalid argument {arg!r}'
            raise ValueError(msg) from exc


def parse_kwargs(
    kwargs: dict[str, Any], /, *keys: str, **keyvalues: Any
) -> dict[str, Any]:
    """Return dict with keys from keys|keyvals and values from kwargs|keyvals.

    Existing keys are deleted from `kwargs`.

    >>> kwargs = {'one': 1, 'two': 2, 'four': 4}
    >>> kwargs2 = parse_kwargs(kwargs, 'two', 'three', four=None, five=5)
    >>> kwargs == {'one': 1}
    True
    >>> kwargs2 == {'two': 2, 'four': 4, 'five': 5}
    True

    """
    result = {}
    for key in keys:
        if key in kwargs:
            result[key] = kwargs[key]
            del kwargs[key]
    for key, value in keyvalues.items():
        if key in kwargs:
            result[key] = kwargs[key]
            del kwargs[key]
        else:
            result[key] = value
    return result


def update_kwargs(kwargs: dict[str, Any], /, **keyvalues: Any) -> None:
    """Update dict with keys and values if keys do not already exist.

    >>> kwargs = {'one': 1}
    >>> update_kwargs(kwargs, one=None, two=2)
    >>> kwargs == {'one': 1, 'two': 2}
    True

    """
    for key, value in keyvalues.items():
        if key not in kwargs:
            kwargs[key] = value


def kwargs_notnone(**kwargs: Any) -> dict[str, Any]:
    """Return dict of kwargs which values are not None.

    >>> kwargs_notnone(one=1, none=None)
    {'one': 1}

    """
    return dict(item for item in kwargs.items() if item[1] is not None)


def logger() -> logging.Logger:
    """Return logger for tifffile module."""
    return logging.getLogger('tifffile')


def matlabstr2py(matlabstr: str, /) -> Any:
    r"""Return Python object from Matlab string representation.

    Use to access ScanImage metadata.

    Parameters:
        matlabstr: String representation of Matlab objects.

    Returns:
        Matlab structures are returned as `dict`.
        Matlab arrays or cells are returned as `lists`.
        Other Matlab objects are returned as `str`, `bool`, `int`, or `float`.

    Examples:
        >>> matlabstr2py('1')
        1
        >>> matlabstr2py("['x y z' true false; 1 2.0 -3e4; NaN Inf @class]")
        [['x y z', True, False], [1, 2.0, -30000.0], [nan, inf, '@class']]
        >>> d = matlabstr2py(
        ...     "SI.hChannels.channelType = {'stripe' 'stripe'}\n"
        ...     "SI.hChannels.channelsActive = 2"
        ... )
        >>> d['SI.hChannels.channelType']
        ['stripe', 'stripe']

    """
    # TODO: handle invalid input
    # TODO: review unboxing of multidimensional arrays

    def lex(s: str, /) -> list[str]:
        # return sequence of tokens from Matlab string representation
        tokens = ['[']
        while True:
            t, i = next_token(s)
            if t is None:
                break
            if t == ';':
                tokens.extend((']', '['))
            elif t == '[':
                tokens.extend(('[', '['))
            elif t == ']':
                tokens.extend((']', ']'))
            else:
                tokens.append(t)
            s = s[i:]
        tokens.append(']')
        return tokens

    def next_token(s: str, /) -> tuple[str | None, int]:
        # return next token in Matlab string
        length = len(s)
        if length == 0:
            return None, 0
        i = 0
        while i < length and s[i] == ' ':
            i += 1
        if i == length:
            return None, i
        if s[i] in '{[;]}':
            return s[i], i + 1
        if s[i] == "'":
            j = i + 1
            while j < length and s[j] != "'":
                j += 1
            return s[i : j + 1], j + 1
        if s[i] == '<':
            j = i + 1
            while j < length and s[j] != '>':
                j += 1
            return s[i : j + 1], j + 1
        j = i
        while j < length and s[j] not in ' {[;]}':
            j += 1
        return s[i:j], j

    def value(s: str, *, fail: bool = False) -> Any:
        # return Python value of token
        s = s.strip()
        if not s:
            return s
        if len(s) == 1:
            try:
                return int(s)
            except Exception as exc:
                if fail:
                    raise ValueError from exc
                return s
        if s[0] == "'":
            if (fail and s[-1] != "'") or "'" in s[1:-1]:
                raise ValueError
            return s[1:-1]
        if s[0] == '<':
            if (fail and s[-1] != '>') or '<' in s[1:-1]:
                raise ValueError
            return s
        if fail and any(i in s for i in " ';[]{}"):
            raise ValueError
        if s[0] == '@':
            return s
        if s in {'true', 'True'}:
            return True
        if s in {'false', 'False'}:
            return False
        if s[:6] == 'zeros(':
            return numpy.zeros([int(i) for i in s[6:-1].split(',')]).tolist()
        if s[:5] == 'ones(':
            return numpy.ones([int(i) for i in s[5:-1].split(',')]).tolist()
        if '.' in s or 'e' in s:
            try:
                return float(s)
            except (TypeError, ValueError):
                pass
        try:
            return int(s)
        except (TypeError, ValueError):
            pass
        try:
            return float(s)  # nan, inf
        except (TypeError, ValueError) as exc:
            if fail:
                raise ValueError from exc
        return s

    def parse(s: str, /) -> Any:
        # return Python value from string representation of Matlab value
        s = s.strip()
        try:
            return value(s, fail=True)
        except ValueError:
            pass
        result: list[Any]
        addto: list[Any]
        result = addto = []
        levels = [addto]
        for t in lex(s):
            if t in '[{':
                addto = []
                levels.append(addto)
            elif t in ']}':
                if len(levels) < 2:
                    # unbalanced brackets
                    break
                x = levels.pop()
                addto = levels[-1]
                if len(x) == 1 and isinstance(x[0], (list, str)):
                    addto.append(x[0])
                else:
                    addto.append(x)
            else:
                addto.append(value(t))
        if len(result) == 1 and isinstance(result[0], (list, str)):
            return result[0]
        return result

    if '\r' in matlabstr or '\n' in matlabstr:
        # structure
        d = {}
        for line in matlabstr.splitlines():
            line = line.strip()  # noqa: PLW2901
            if not line or line[0] == '%' or '=' not in line:
                continue
            k, v = line.split('=', 1)
            k = k.strip()
            if any(c in k for c in " ';[]{}<>"):
                continue
            d[k] = parse(v)
        return d
    return parse(matlabstr)


def strptime(datetime_string: str, fmt: str | None = None, /) -> DateTime:
    """Return datetime corresponding to date string using common formats.

    Parameters:
        datetime_string:
            String representation of date and time.
        fmt:
            Format of `datetime_string`.
            By default, several datetime formats commonly found in TIFF files
            are parsed.

    Raises:
        ValueError: `datetime_string` does not match any known format.

    Examples:
        >>> strptime('2022:08:01 22:23:24')
        datetime.datetime(2022, 8, 1, 22, 23, 24)

    """
    formats = {
        '%Y:%m:%d %H:%M:%S': 1,  # TIFF6 specification
        '%Y%m%d %H:%M:%S.%f': 2,  # MetaSeries
        '%Y-%m-%dT%H %M %S.%f': 3,  # Pilatus
        '%Y-%m-%dT%H:%M:%S.%f': 4,  # ISO
        '%Y-%m-%dT%H:%M:%S': 5,  # ISO, microsecond is 0
        '%Y:%m:%d %H:%M:%S.%f': 6,
        '%d/%m/%Y %H:%M:%S': 7,
        '%d/%m/%Y %H:%M:%S.%f': 8,
        '%m/%d/%Y %I:%M:%S %p': 9,
        '%m/%d/%Y %I:%M:%S.%f %p': 10,
        '%Y%m%d %H:%M:%S': 11,
        '%Y/%m/%d %H:%M:%S': 12,
        '%Y/%m/%d %H:%M:%S.%f': 13,
        '%Y-%m-%dT%H:%M:%S%z': 14,
        '%Y-%m-%dT%H:%M:%S.%f%z': 15,
    }
    if fmt is not None:
        formats[fmt] = 0  # highest priority; replaces existing key if any
    for fmt_, _ in sorted(formats.items(), key=lambda item: item[1]):
        try:
            return DateTime.strptime(datetime_string, fmt_)
        except ValueError:
            pass
    msg = f'time data {datetime_string!r} does not match any format'
    raise ValueError(msg)


def stripnull(
    string: bytes, /, null: bytes | None = None, *, first: bool = True
) -> bytes: ...


def stripnull(
    string: str, /, null: str | None = None, *, first: bool = True
) -> str: ...


def stripnull(
    string: str | bytes,
    /,
    null: str | bytes | None = None,
    *,
    first: bool = True,
) -> str | bytes:
    r"""Return string truncated at first null character.

    Use to clean NULL terminated C strings.

    >>> stripnull(b'bytes\x00\x00')
    b'bytes'
    >>> stripnull(b'bytes\x00bytes\x00\x00', first=False)
    b'bytes\x00bytes'
    >>> stripnull('string\x00')
    'string'

    """
    # TODO: enable deprecation warning
    # warnings.warn(
    #     '<tifffile.stripnull is deprecated since 2025.3.18',
    #     DeprecationWarning,
    #     stacklevel=2,
    # )
    if null is None:
        null = b'\x00' if isinstance(string, bytes) else '\0'
    if first:
        i = string.find(null)  # type: ignore[arg-type]
        return string if i < 0 else string[:i]
    return string.rstrip(null)  # type: ignore[arg-type]


def stripascii(string: bytes, /) -> bytes:
    r"""Return string truncated at last byte that is 7-bit ASCII.

    Use to clean NULL separated and terminated TIFF strings.

    >>> stripascii(b'string\x00string\n\x01\x00')
    b'string\x00string\n'
    >>> stripascii(b'\x00')
    b''

    """
    # TODO: pythonize this
    i = len(string)
    while i:
        i -= 1
        if 8 < string[i] < 127:
            break
    else:
        i = -1
    return string[: i + 1]


def asbool(
    value: str,
    /,
    true: Sequence[str] | None = None,
    false: Sequence[str] | None = None,
) -> bool: ...


def asbool(
    value: bytes,
    /,
    true: Sequence[bytes] | None = None,
    false: Sequence[bytes] | None = None,
) -> bool: ...


def asbool(
    value: str | bytes,
    /,
    true: Sequence[str | bytes] | None = None,
    false: Sequence[str | bytes] | None = None,
) -> bool | bytes:
    """Return string as bool if possible, else raise TypeError.

    >>> asbool(b' False ')
    False
    >>> asbool('ON', ['on'], ['off'])
    True

    """
    value = value.strip().lower()
    isbytes = False
    if true is None:
        if isinstance(value, bytes):
            if value == b'true':
                return True
            isbytes = True
        elif value == 'true':
            return True
    elif value in true:
        return True
    if false is None:
        if isbytes or isinstance(value, bytes):
            if value == b'false':
                return False
        elif value == 'false':
            return False
    elif value in false:
        return False
    raise TypeError


def astype(value: Any, /, types: Sequence[Any] | None = None) -> Any:
    """Return argument as one of types if possible.

    >>> astype('42')
    42
    >>> astype('3.14')
    3.14
    >>> astype('True')
    True
    >>> astype(b'Neee-Wom')
    'Neee-Wom'

    """
    if types is None:
        types = int, float, asbool, bytes2str
    for typ in types:
        try:
            return typ(value)
        except (ValueError, AttributeError, TypeError, UnicodeEncodeError):
            pass
    return value


def rational(arg: float | tuple[int, int], /) -> tuple[int, int]:
    """Return rational numerator and denominator from float or two integers."""
    from fractions import Fraction

    if isinstance(arg, Sequence):
        f = Fraction(arg[0], arg[1])
    else:
        f = Fraction.from_float(arg)

    numerator, denominator = f.as_integer_ratio()
    if numerator > 4294967295 or denominator > 4294967295:
        s = 4294967295 / max(numerator, denominator)
        numerator = round(numerator * s)
        denominator = round(denominator * s)
    return numerator, denominator


def unique_strings(strings: Iterator[str], /) -> Iterator[str]:
    """Return iterator over unique strings.

    >>> list(unique_strings(iter(('a', 'b', 'a'))))
    ['a', 'b', 'a2']

    """
    known = set()
    for i, s in enumerate(strings):
        string = s
        if string in known:
            string += str(i)
        known.add(string)
        yield string


def bytes2str(
    b: bytes, /, encoding: str | None = None, errors: str = 'strict'
) -> str:
    """Return Unicode string from encoded bytes up to first NULL character."""
    if encoding is None or '16' not in encoding:
        i = b.find(b'\x00')
        if i >= 0:
            b = b[:i]
    else:
        # utf-16
        i = b.find(b'\x00\x00')
        if i >= 0:
            b = b[: i + i % 2]

    try:
        return b.decode('utf-8' if encoding is None else encoding, errors)
    except UnicodeDecodeError:
        if encoding is not None:
            raise
        return b.decode('cp1252', errors)


def bytestr(s: str | bytes, /, encoding: str = 'cp1252') -> bytes:
    """Return bytes from Unicode string, else pass through."""
    return s.encode(encoding) if isinstance(s, str) else s


def repeat_nd(a: ArrayLike, repeats: Sequence[int], /) -> NDArray[Any]:
    """Return read-only view into input array with elements repeated.

    Zoom image array by integer factors using nearest neighbor interpolation
    (box filter).

    Parameters:
        a: Input array.
        repeats: Number of repetitions to apply along each dimension of input.

    Examples:
        >>> repeat_nd([[1, 2], [3, 4]], (2, 2))
        array([[1, 1, 2, 2],
               [1, 1, 2, 2],
               [3, 3, 4, 4],
               [3, 3, 4, 4]])

    """
    reshape: list[int] = []
    shape: list[int] = []
    strides: list[int] = []
    a = numpy.asarray(a)
    for i, j, k in zip(a.strides, a.shape, repeats, strict=True):
        shape.extend((j, k))
        strides.extend((i, 0))
        reshape.append(j * k)
    return numpy.lib.stride_tricks.as_strided(
        a, shape, strides, writeable=False
    ).reshape(reshape)


def reshape_nd(
    data_or_shape: tuple[int, ...], ndim: int, /
) -> tuple[int, ...]: ...


def reshape_nd(data_or_shape: NDArray[Any], ndim: int, /) -> NDArray[Any]: ...


def reshape_nd(
    data_or_shape: tuple[int, ...] | NDArray[Any], ndim: int, /
) -> tuple[int, ...] | NDArray[Any]:
    """Return image array or shape with at least `ndim` dimensions.

    Prepend 1s to image shape as necessary.

    >>> import numpy
    >>> reshape_nd(numpy.empty(0), 1).shape
    (0,)
    >>> reshape_nd(numpy.empty(1), 2).shape
    (1, 1)
    >>> reshape_nd(numpy.empty((2, 3)), 3).shape
    (1, 2, 3)
    >>> reshape_nd(numpy.empty((3, 4, 5)), 3).shape
    (3, 4, 5)
    >>> reshape_nd((2, 3), 3)
    (1, 2, 3)

    """
    if isinstance(data_or_shape, tuple):
        shape = data_or_shape
    else:
        shape = data_or_shape.shape
    if len(shape) >= ndim:
        return data_or_shape
    shape = (1,) * (ndim - len(shape)) + shape
    if isinstance(data_or_shape, tuple):
        return shape
    return data_or_shape.reshape(shape)


def squeeze_axes(
    shape: Sequence[int],
    axes: str,
    /,
    skip: str | None = None,
) -> tuple[tuple[int, ...], str, tuple[bool, ...]]: ...


def squeeze_axes(
    shape: Sequence[int],
    axes: Sequence[str],
    /,
    skip: Sequence[str] | None = None,
) -> tuple[tuple[int, ...], Sequence[str], tuple[bool, ...]]: ...


def squeeze_axes(
    shape: Sequence[int],
    axes: str | Sequence[str],
    /,
    skip: str | Sequence[str] | None = None,
) -> tuple[tuple[int, ...], str | Sequence[str], tuple[bool, ...]]:
    """Return shape and axes with length-1 dimensions removed.

    Remove unused dimensions unless their axes are listed in `skip`.

    Parameters:
        shape:
            Sequence of dimension sizes.
        axes:
            Character codes for dimensions in `shape`.
        skip:
            Character codes for dimensions whose length-1 dimensions are
            not removed. The default is 'XY'.

    Returns:
        shape:
            Sequence of dimension sizes with length-1 dimensions removed.
        axes:
            Character codes for dimensions in output `shape`.
        squeezed:
            Dimensions were kept (True) or removed (False).

    Examples:
        >>> squeeze_axes((5, 1, 2, 1, 1), 'TZYXC')
        ((5, 2, 1), 'TYX', (True, False, True, True, False))
        >>> squeeze_axes((1,), 'Q')
        ((1,), 'Q', (True,))

    """
    if len(shape) != len(axes):
        msg = 'dimensions of axes and shape do not match'
        raise ValueError(msg)
    if not axes:
        return tuple(shape), axes, ()
    if skip is None:
        skip = 'X', 'Y', 'width', 'height', 'length'
    squeezed: list[bool] = []
    shape_squeezed: list[int] = []
    axes_squeezed: list[str] = []
    for size, ax in zip(shape, axes, strict=True):
        if size > 1 or ax in skip:
            squeezed.append(True)
            shape_squeezed.append(size)
            axes_squeezed.append(ax)
        else:
            squeezed.append(False)
    if len(shape_squeezed) == 0:
        squeezed[-1] = True
        shape_squeezed.append(shape[-1])
        axes_squeezed.append(axes[-1])
    if isinstance(axes, str):
        axes = ''.join(axes_squeezed)
    else:
        axes = tuple(axes_squeezed)
    return (tuple(shape_squeezed), axes, tuple(squeezed))


def transpose_axes(
    image: NDArray[Any],
    axes: str,
    /,
    asaxes: Sequence[str] | None = None,
) -> NDArray[Any]:
    """Return image array with its axes permuted to match specified axes.

    Parameters:
        image:
            Image array to permute.
        axes:
            Character codes for dimensions in image array.
        asaxes:
            Character codes for dimensions in output image array.
            The default is 'CTZYX'.

    Returns:
        Transposed image array.
        A length-1 dimension is added for added dimensions.
        A view of the input array is returned if possible.

    Examples:
        >>> import numpy
        >>> transpose_axes(
        ...     numpy.zeros((2, 3, 4, 5)), 'TYXC', asaxes='CTZYX'
        ... ).shape
        (5, 2, 1, 3, 4)

    """
    if asaxes is None:
        asaxes = 'CTZYX'
    for ax in axes:
        if ax not in asaxes:
            msg = f'unknown axis {ax}'
            raise ValueError(msg)
    # add missing axes to image
    shape = image.shape
    for ax in reversed(asaxes):
        if ax not in axes:
            axes = ax + axes
            shape = (1, *shape)
    image = image.reshape(shape)
    # transpose axes
    return image.transpose([axes.index(ax) for ax in asaxes])


def reshape_axes(
    axes: str,
    shape: Sequence[int],
    newshape: Sequence[int],
    /,
    unknown: str | None = None,
) -> str: ...


def reshape_axes(
    axes: Sequence[str],
    shape: Sequence[int],
    newshape: Sequence[int],
    /,
    unknown: str | None = None,
) -> Sequence[str]: ...


def reshape_axes(
    axes: str | Sequence[str],
    shape: Sequence[int],
    newshape: Sequence[int],
    /,
    unknown: str | None = None,
) -> str | Sequence[str]:
    """Return axes matching new shape.

    Parameters:
        axes:
            Character codes for dimensions in `shape`.
        shape:
            Input shape matching `axes`.
        newshape:
            Output shape matching output axes.
            Size must match size of `shape`.
        unknown:
            Character used for new axes in output. The default is 'Q'.

    Returns:
        Character codes for dimensions in `newshape`.

    Examples:
        >>> reshape_axes('YXS', (219, 301, 1), (219, 301))
        'YX'
        >>> reshape_axes('IYX', (12, 219, 301), (3, 4, 219, 1, 301, 1))
        'QQYQXQ'

    """
    shape = tuple(shape)
    newshape = tuple(newshape)
    if len(axes) != len(shape):
        msg = 'axes do not match shape'
        raise ValueError(msg)

    size = product(shape)
    newsize = product(newshape)
    if size != newsize:
        msg = f'cannot reshape {shape} to {newshape}'
        raise ValueError(msg)
    if not axes or not newshape:
        return '' if isinstance(axes, str) else ()

    lendiff = max(0, len(shape) - len(newshape))
    if lendiff:
        newshape = newshape + (1,) * lendiff

    i = len(shape) - 1
    prodns = 1
    prods = 1
    result = []
    for ns in newshape[::-1]:
        prodns *= ns
        while i > 0 and shape[i] == 1 and ns != 1:
            i -= 1
        if ns == shape[i] and prodns == prods * shape[i]:
            prods *= shape[i]
            result.append(axes[i])
            i -= 1
        elif unknown:
            result.append(unknown)
        else:
            unknown = 'Q'
            result.append(unknown)

    if isinstance(axes, str):
        axes = ''.join(reversed(result[lendiff:]))
    else:
        axes = tuple(reversed(result[lendiff:]))
    return axes


def order_axes(
    indices: ArrayLike,
    /,
    *,
    squeeze: bool = False,
) -> tuple[int, ...]:
    """Return order of axes sorted by variations in indices.

    Parameters:
        indices:
            Multi-dimensional indices of chunks in array.
        squeeze:
            Remove length-1 dimensions of nonvarying axes.

    Returns:
        Order of axes sorted by variations in indices.
        The axis with the least variations in indices is returned first,
        the axis varying fastest is last.

    Examples:
        First axis varies fastest, second axis is squeezed:
        >>> order_axes(
        ...     [(0, 2, 0), (1, 2, 0), (0, 2, 1), (1, 2, 1)], squeeze=True
        ... )
        (2, 0)

    """
    diff = numpy.sum(numpy.abs(numpy.diff(indices, axis=0)), axis=0).tolist()
    order = tuple(sorted(range(len(diff)), key=diff.__getitem__))
    if squeeze:
        order = tuple(i for i in order if diff[i] != 0)
    return order


def check_shape(
    page_shape: Sequence[int], series_shape: Sequence[int]
) -> bool:
    """Return if page and series shapes are compatible."""
    pi = product(page_shape)
    pj = product(series_shape)
    if pi == 0 and pj == 0:
        return True
    if pi == 0 or pj == 0:
        return False
    if pj % pi:
        return False

    series_shape = tuple(reversed(series_shape))
    a = 0
    pi = pj = 1
    for i in reversed(page_shape):
        pi *= i
        # if a == len(series_shape):
        #     return not pj % pi
        for j in series_shape[a:]:
            a += 1
            pj *= j
            if i == j or pi == pj:
                break
            if j == 1:
                continue
            if pj != pi:
                return False
    return True


def subresolution(
    a: TiffPage, b: TiffPage, /, p: int = 2, n: int = 16
) -> int | None: ...


def subresolution(
    a: TiffPageSeries, b: TiffPageSeries, /, p: int = 2, n: int = 16
) -> int | None: ...


def subresolution(
    a: TiffPage | TiffPageSeries,
    b: TiffPage | TiffPageSeries,
    /,
    p: int = 2,
    n: int = 16,
) -> int | None:
    """Return level of subresolution of series or page b vs a."""
    if a.axes != b.axes or a.dtype != b.dtype:
        return None
    level = None
    for ax, i, j in zip(a.axes.lower(), a.shape, b.shape, strict=True):
        if ax in 'xyz':
            if level is None:
                for r in range(n):
                    d = p**r
                    if d > i:
                        return None
                    if abs((i / d) - j) < 1.0:
                        level = r
                        break
                else:
                    return None
            else:
                d = p**level
                if d > i:
                    return None
                if abs((i / d) - j) >= 1.0:
                    return None
        elif i != j:
            return None
    return level


def unpack_rgb(
    data: bytes,
    /,
    dtype: DTypeLike | None = None,
    bitspersample: tuple[int, ...] | None = None,
    *,
    rescale: bool = True,
) -> NDArray[Any]:
    """Return array from bytes containing packed samples.

    Use to unpack RGB565 or RGB555 to RGB888 format.
    Works on little-endian platforms only.

    Parameters:
        data:
            Bytes to be decoded.
            Samples in each pixel are stored consecutively.
            Pixels are aligned to 8, 16, or 32 bit boundaries.
        dtype:
            Data type of samples.
            The byte order applies also to the data stream.
        bitspersample:
            Number of bits for each sample in pixel.
        rescale:
            Upscale samples to number of bits in dtype.

    Returns:
        Flattened array of unpacked samples of native dtype.

    Examples:
        >>> data = struct.pack('BBBB', 0x21, 0x08, 0xFF, 0xFF)
        >>> print(unpack_rgb(data, '<B', (5, 6, 5), rescale=False))
        [ 1  1  1 31 63 31]
        >>> print(unpack_rgb(data, '<B', (5, 6, 5)))
        [  8   4   8 255 255 255]
        >>> print(unpack_rgb(data, '<B', (5, 5, 5)))
        [ 16   8   8 255 255 255]

    """
    if bitspersample is None:
        bitspersample = (5, 6, 5)
    if dtype is None:
        dtype = '<B'
    dtype = numpy.dtype(dtype)
    bits = int(numpy.sum(bitspersample))
    if not (
        bits <= 32 and all(i <= dtype.itemsize * 8 for i in bitspersample)
    ):
        msg = f'sample size not supported: {bitspersample}'
        raise ValueError(msg)
    dt = next(i for i in 'BHI' if numpy.dtype(i).itemsize * 8 >= bits)
    data_array = numpy.frombuffer(data, dtype.byteorder + dt)
    result = numpy.empty((data_array.size, len(bitspersample)), dtype.char)
    for i, bps in enumerate(bitspersample):
        t = data_array >> int(numpy.sum(bitspersample[i + 1 :]))
        t &= int('0b' + '1' * bps, 2)
        if rescale:
            o = ((dtype.itemsize * 8) // bps + 1) * bps
            if o > data_array.dtype.itemsize * 8:
                t = t.astype('I')
            t *= (2**o - 1) // (2**bps - 1)
            t //= 2 ** (o - (dtype.itemsize * 8))
        result[:, i] = t
    return result.reshape(-1)


def apply_colormap(
    image: NDArray[Any], colormap: NDArray[Any], /, *, contig: bool = True
) -> NDArray[Any]:
    """Return palette-colored image.

    The image array values are used to index the colormap on axis 1.
    The returned image array is of shape `image.shape+colormap.shape[0]`
    and dtype `colormap.dtype`.

    Parameters:
        image:
            Array of indices into colormap.
        colormap:
            RGB lookup table aka palette of shape `(3, 2**bitspersample)`.
        contig:
            Return contiguous array.

    Examples:
        >>> import numpy
        >>> im = numpy.arange(256, dtype='uint8')
        >>> colormap = numpy.vstack([im, im, im]).astype('uint16') * 256
        >>> apply_colormap(im, colormap)[-1]
        array([65280, 65280, 65280], dtype=uint16)

    """
    image = numpy.take(colormap, image, axis=1)
    image = numpy.rollaxis(image, 0, image.ndim)
    if contig:
        image = numpy.ascontiguousarray(image)
    return image


def reorient(
    image: NDArray[Any], orientation: ORIENTATION | int | str, /
) -> NDArray[Any]:
    """Return reoriented view of image array.

    Parameters:
        image:
            Non-squeezed output of `asarray` functions.
            Axes -3 and -2 must be image length and width respectively.
        orientation:
            Value of Orientation tag.

    """
    orientation = cast(ORIENTATION, enumarg(ORIENTATION, orientation))

    match orientation:
        case ORIENTATION.TOPLEFT:
            return image
        case ORIENTATION.TOPRIGHT:
            return image[..., ::-1, :]
        case ORIENTATION.BOTLEFT:
            return image[..., ::-1, :, :]
        case ORIENTATION.BOTRIGHT:
            return image[..., ::-1, ::-1, :]
        case ORIENTATION.LEFTTOP:
            return numpy.swapaxes(image, -3, -2)
        case ORIENTATION.RIGHTTOP:
            return numpy.swapaxes(image, -3, -2)[..., ::-1, :]
        case ORIENTATION.RIGHTBOT:
            return numpy.swapaxes(image, -3, -2)[..., ::-1, :, :]
        case ORIENTATION.LEFTBOT:
            return numpy.swapaxes(image, -3, -2)[..., ::-1, ::-1, :]
        case _:
            return image


def iter_images(data: NDArray[Any], /) -> Iterator[NDArray[Any]]:
    """Return iterator over pages in data array of normalized shape."""
    yield from data

