# codecs.py

"""TIFF format, compression, and predictor codec classes."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, final

from .enums import COMPRESSION, PREDICTOR

if TYPE_CHECKING:
    from collections.abc import Iterator
    from typing import Any, Literal

try:
    import imagecodecs
except ImportError:
    try:
        from . import _imagecodecs as imagecodecs  # type: ignore[no-redef]
    except ImportError:
        import _imagecodecs as imagecodecs  # type: ignore[no-redef]

__all__ = [
    'CompressionCodec',
    'PredictorCodec',
    'TiffFormat',
]


def _identityfunc(arg: Any, /, *args: Any, **kwargs: Any) -> Any:
    """Single argument identity function."""
    return arg


def _indent(*args: Any) -> str:
    """Return joined string representations of objects with indented lines."""
    text = '\n'.join(str(arg) for arg in args)
    return '\n'.join(
        ('  ' + line if line else line) for line in text.splitlines() if line
    )[2:]


@final
class TiffFormat:
    """TIFF format properties."""

    __slots__ = (
        '_hash',
        'byteorder',
        'offsetformat',
        'offsetsize',
        'tagformat1',
        'tagformat2',
        'tagnoformat',
        'tagnosize',
        'tagoffsetthreshold',
        'tagsize',
        'version',
    )

    version: int
    """Version of TIFF header."""

    byteorder: Literal['>', '<']
    """Byteorder of TIFF header."""

    offsetsize: int
    """Size of offsets."""

    offsetformat: str
    """Struct format for offset values."""

    tagnosize: int
    """Size of `tagnoformat`."""

    tagnoformat: str
    """Struct format for number of TIFF tags."""

    tagsize: int
    """Size of `tagformat1` and `tagformat2`."""

    tagformat1: str
    """Struct format for code and dtype of TIFF tag."""

    tagformat2: str
    """Struct format for count and value of TIFF tag."""

    tagoffsetthreshold: int
    """Size of inline tag values."""

    _hash: int

    def __init__(
        self,
        version: int,
        byteorder: Literal['>', '<'],
        offsetsize: int,
        offsetformat: str,
        tagnosize: int,
        tagnoformat: str,
        tagsize: int,
        tagformat1: str,
        tagformat2: str,
        tagoffsetthreshold: int,
    ) -> None:
        self.version = version
        self.byteorder = byteorder
        self.offsetsize = offsetsize
        self.offsetformat = offsetformat
        self.tagnosize = tagnosize
        self.tagnoformat = tagnoformat
        self.tagsize = tagsize
        self.tagformat1 = tagformat1
        self.tagformat2 = tagformat2
        self.tagoffsetthreshold = tagoffsetthreshold
        self._hash = hash((version, byteorder, offsetsize))

    @property
    def is_bigtiff(self) -> bool:
        """Format is 64-bit BigTIFF."""
        return self.version == 43

    @property
    def is_ndpi(self) -> bool:
        """Format is 32-bit TIFF with 64-bit offsets used by NDPI."""
        return self.version == 42 and self.offsetsize == 8

    def __hash__(self) -> int:
        return self._hash

    def __repr__(self) -> str:
        bits = '32' if self.version == 42 else '64'
        endian = 'little' if self.byteorder == '<' else 'big'
        ndpi = ' with 64-bit offsets' if self.is_ndpi else ''
        return f'<tifffile.TiffFormat {bits}-bit {endian}-endian{ndpi}>'

    def __str__(self) -> str:
        return _indent(
            repr(self),
            *(
                f'{attr}: {getattr(self, attr)!r}'
                for attr in TiffFormat.__slots__
            ),
        )


class CompressionCodec(Mapping[int, Callable[..., object]]):
    """Map :py:class:`COMPRESSION` value to encode or decode function.

    Parameters:
        encode: If *True*, return encode functions, else decode functions.

    """

    _codecs: dict[int, Callable[..., Any]]
    _encode: bool

    def __init__(self, /, *, encode: bool) -> None:
        self._codecs = {1: _identityfunc}
        self._encode = bool(encode)

    def __getitem__(self, key: int, /) -> Callable[..., Any]:
        if key in self._codecs:
            return self._codecs[key]
        codec: Callable[..., Any]
        try:
            match key:
                case 5:
                    if self._encode:
                        codec = imagecodecs.lzw_encode
                    else:
                        codec = imagecodecs.lzw_decode
                case 6 | 7 | 33007:
                    if self._encode:
                        if key in {6, 33007}:
                            raise NotImplementedError
                        codec = imagecodecs.jpeg_encode
                    else:
                        codec = imagecodecs.jpeg_decode
                case 8 | 32946 | 50013:
                    if (
                        hasattr(imagecodecs, 'DEFLATE')
                        and imagecodecs.DEFLATE.available
                    ):
                        if self._encode:
                            codec = imagecodecs.deflate_encode
                        else:
                            codec = imagecodecs.deflate_decode
                    elif (
                        hasattr(imagecodecs, 'ZLIB')
                        and imagecodecs.ZLIB.available
                    ):
                        if self._encode:
                            codec = imagecodecs.zlib_encode
                        else:
                            codec = imagecodecs.zlib_decode
                    else:
                        try:
                            from . import _imagecodecs
                        except ImportError:
                            import _imagecodecs  # type: ignore[no-redef]
                        if self._encode:
                            codec = _imagecodecs.zlib_encode
                        else:
                            codec = _imagecodecs.zlib_decode
                case 32773:
                    if self._encode:
                        codec = imagecodecs.packbits_encode
                    else:
                        codec = imagecodecs.packbits_decode
                case 33003 | 33004 | 33005 | 34712:
                    if self._encode:
                        codec = imagecodecs.jpeg2k_encode
                    else:
                        codec = imagecodecs.jpeg2k_decode
                case 34887:
                    if self._encode:
                        codec = imagecodecs.lerc_encode
                    else:
                        codec = imagecodecs.lerc_decode
                case 34892:
                    if self._encode:
                        codec = imagecodecs.jpeg8_encode
                    else:
                        codec = imagecodecs.jpeg8_decode
                case 34925:
                    if (
                        hasattr(imagecodecs, 'LZMA')
                        and imagecodecs.LZMA.available
                    ):
                        if self._encode:
                            codec = imagecodecs.lzma_encode
                        else:
                            codec = imagecodecs.lzma_decode
                    else:
                        try:
                            from . import _imagecodecs
                        except ImportError:
                            import _imagecodecs  # type: ignore[no-redef]
                        if self._encode:
                            codec = _imagecodecs.lzma_encode
                        else:
                            codec = _imagecodecs.lzma_decode
                case 34933:
                    if self._encode:
                        codec = imagecodecs.png_encode
                    else:
                        codec = imagecodecs.png_decode
                case 34934 | 22610:
                    if self._encode:
                        codec = imagecodecs.jpegxr_encode
                    else:
                        codec = imagecodecs.jpegxr_decode
                case 48124:
                    if self._encode:
                        codec = imagecodecs.jetraw_encode
                    else:
                        codec = imagecodecs.jetraw_decode
                case 50000 | 34926:
                    if (
                        hasattr(imagecodecs, 'ZSTD')
                        and imagecodecs.ZSTD.available
                    ):
                        if self._encode:
                            codec = imagecodecs.zstd_encode
                        else:
                            codec = imagecodecs.zstd_decode
                    else:
                        try:
                            from . import _imagecodecs
                        except ImportError:
                            import _imagecodecs  # type: ignore[no-redef]
                        if self._encode:
                            codec = _imagecodecs.zstd_encode
                        else:
                            codec = _imagecodecs.zstd_decode
                case 50001 | 34927:
                    if self._encode:
                        codec = imagecodecs.webp_encode
                    else:
                        codec = imagecodecs.webp_decode
                case 65000 | 65001 | 65002 if not self._encode:
                    codec = imagecodecs.eer_decode
                case 50002 | 52546:
                    if self._encode:
                        codec = imagecodecs.jpegxl_encode
                    else:
                        codec = imagecodecs.jpegxl_decode
                case _:
                    try:
                        msg = f'{COMPRESSION(key)!r} not supported'
                    except ValueError:
                        msg = f'{key} is not a known COMPRESSION'
                    raise KeyError(msg)
        except (AttributeError, ImportError) as exc:
            msg = f"{COMPRESSION(key)!r} requires the 'imagecodecs' package"
            raise KeyError(msg) from exc
        except NotImplementedError as exc:
            msg = f'{COMPRESSION(key)!r} not implemented'
            raise KeyError(msg) from exc
        self._codecs[key] = codec
        return codec

    def __contains__(self, key: Any, /) -> bool:
        try:
            self[key]
        except KeyError:
            return False
        return True

    def __iter__(self) -> Iterator[int]:
        yield 1  # dummy

    def __len__(self) -> int:
        return 1  # dummy


@final
class PredictorCodec(Mapping[int, Callable[..., object]]):
    """Map :py:class:`PREDICTOR` value to encode or decode function.

    Parameters:
        encode: If *True*, return encode functions, else decode functions.

    """

    _codecs: dict[int, Callable[..., Any]]
    _encode: bool

    def __init__(self, /, *, encode: bool) -> None:
        self._codecs = {1: _identityfunc}
        self._encode = bool(encode)

    def __getitem__(self, key: int, /) -> Callable[..., Any]:
        if key in self._codecs:
            return self._codecs[key]
        codec: Callable[..., Any]
        try:
            match key:
                case 2:
                    if self._encode:
                        codec = imagecodecs.delta_encode
                    else:
                        codec = imagecodecs.delta_decode
                case 3:
                    if self._encode:
                        codec = imagecodecs.floatpred_encode
                    else:
                        codec = imagecodecs.floatpred_decode
                case 34892:
                    if self._encode:

                        def codec(data, axis=-1, out=None):
                            return imagecodecs.delta_encode(
                                data, axis=axis, out=out, dist=2
                            )

                    else:

                        def codec(data, axis=-1, out=None):
                            return imagecodecs.delta_decode(
                                data, axis=axis, out=out, dist=2
                            )

                case 34893:
                    if self._encode:

                        def codec(data, axis=-1, out=None):
                            return imagecodecs.delta_encode(
                                data, axis=axis, out=out, dist=4
                            )

                    else:

                        def codec(data, axis=-1, out=None):
                            return imagecodecs.delta_decode(
                                data, axis=axis, out=out, dist=4
                            )

                case 34894:
                    if self._encode:

                        def codec(data, axis=-1, out=None):
                            return imagecodecs.floatpred_encode(
                                data, axis=axis, out=out, dist=2
                            )

                    else:

                        def codec(data, axis=-1, out=None):
                            return imagecodecs.floatpred_decode(
                                data, axis=axis, out=out, dist=2
                            )

                case 34895:
                    if self._encode:

                        def codec(data, axis=-1, out=None):
                            return imagecodecs.floatpred_encode(
                                data, axis=axis, out=out, dist=4
                            )

                    else:

                        def codec(data, axis=-1, out=None):
                            return imagecodecs.floatpred_decode(
                                data, axis=axis, out=out, dist=4
                            )

                case _:
                    msg = f'{key} is not a known PREDICTOR'
                    raise KeyError(msg)
        except AttributeError as exc:
            msg = f"{PREDICTOR(key)!r} requires the 'imagecodecs' package"
            raise KeyError(msg) from exc
        except NotImplementedError as exc:
            msg = f'{PREDICTOR(key)!r} not implemented'
            raise KeyError(msg) from exc
        self._codecs[key] = codec
        return codec

    def __contains__(self, key: Any, /) -> bool:
        try:
            self[key]
        except KeyError:
            return False
        return True

    def __iter__(self) -> Iterator[int]:
        yield 1  # dummy

    def __len__(self) -> int:
        return 1  # dummy
