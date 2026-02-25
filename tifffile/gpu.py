# gpu.py

"""GPU acceleration support for tifffile.

Provides utilities for transferring TIFF image data to GPU devices
(CUDA) using PyTorch tensors. All GPU dependencies (torch, cupy, kvikio)
are lazy-imported so this module has zero cost when not used.

"""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING

import numpy

from .utils import logger

if TYPE_CHECKING:
    from typing import Any

    NDArray = numpy.ndarray[Any, Any]

# Numpy dtypes that torch supports natively (probed at runtime).
# On torch 2.3+, uint16/uint32/uint64 are supported; older versions
# lack them and will use the upcast fallback path.
_TORCH_DTYPE_MAP: dict[numpy.dtype[Any], Any] | None = None


def _get_torch_dtype_map() -> dict[numpy.dtype[Any], Any]:
    """Return mapping of numpy dtypes to torch dtype objects. Lazy init.

    Probes torch at runtime to detect which dtypes are actually supported,
    including uint16/uint32/uint64 added in torch 2.3+.
    """
    global _TORCH_DTYPE_MAP
    if _TORCH_DTYPE_MAP is not None:
        return _TORCH_DTYPE_MAP
    torch = _import_torch()
    _TORCH_DTYPE_MAP = {}
    for np_name in (
        'float16',
        'float32',
        'float64',
        'uint8',
        'int8',
        'int16',
        'int32',
        'int64',
        'bool',
        'complex64',
        'complex128',
        'uint16',
        'uint32',
        'uint64',
    ):
        np_dtype = numpy.dtype(np_name)
        torch_dtype = getattr(torch, np_name, None)
        if torch_dtype is None:
            continue
        try:
            torch.empty(1, dtype=torch_dtype)
            _TORCH_DTYPE_MAP[np_dtype] = torch_dtype
        except Exception:
            continue
    return _TORCH_DTYPE_MAP


def _import_torch() -> Any:
    """Import and return torch module. Raises ImportError with message."""
    try:
        import torch

        return torch
    except ImportError:
        msg = (
            "torch is required for device= parameter. "
            "Install with: pip install torch"
        )
        raise ImportError(msg) from None


def parse_device(device: str | None) -> Any:
    """Parse device string into a torch.device, or return None.

    Parameters:
        device:
            Device specification string such as ``'cuda'``, ``'cuda:0'``,
            or ``None``.

    Returns:
        A ``torch.device`` instance, or *None* if device is None.

    Raises:
        ImportError: If torch is not installed.
        ValueError: If device string is invalid.

    """
    if device is None:
        return None
    torch = _import_torch()
    return torch.device(device)


def numpy_to_tensor(array: NDArray, device: Any) -> Any:
    """Convert numpy array to a torch.Tensor on the given device.

    Uses pinned memory for efficient async CPU→GPU DMA transfer.

    On torch 2.3+, unsigned dtypes (uint16, uint32, uint64) are supported
    natively via zero-copy ``from_numpy``. On older torch versions, unsigned
    arrays are upcast to the next wider signed type on CPU before transfer.

    Parameters:
        array: NumPy array to convert.
        device: A ``torch.device`` instance (e.g. ``torch.device('cuda')``).

    Returns:
        A ``torch.Tensor`` on the specified device.

    """
    torch = _import_torch()

    # Ensure contiguous array for zero-copy torch.from_numpy
    if not array.flags['C_CONTIGUOUS']:
        array = numpy.ascontiguousarray(array)

    dtype_map = _get_torch_dtype_map()

    # If torch natively supports this dtype, use direct path
    if array.dtype in dtype_map:
        tensor = torch.from_numpy(array)
        if device.type == 'cuda':
            tensor = tensor.pin_memory()
            tensor = tensor.to(device)
        elif device.type != 'cpu':
            tensor = tensor.to(device)
        return tensor

    # Fallback for unsupported unsigned types (torch < 2.3)
    _UNSIGNED_UPCAST = {
        numpy.dtype('uint16'): numpy.int32,
        numpy.dtype('uint32'): numpy.int64,
        numpy.dtype('uint64'): numpy.int64,
    }
    upcast = _UNSIGNED_UPCAST.get(array.dtype)
    if upcast is not None:
        array = array.astype(upcast)
        return numpy_to_tensor(array, device)

    msg = f'numpy dtype {array.dtype} not supported by torch'
    raise TypeError(msg)


def read_to_gpu(
    fh: Any,
    device: Any,
    typecode: str,
    count: int,
    offset: int,
    shape: tuple[int, ...],
) -> Any:
    """Read raw bytes from file directly into a GPU tensor.

    For CUDA devices, attempts to use kvikio for GPUDirect Storage
    on Linux, falling back to pinned memory + async DMA.
    Non-CUDA devices fall back to the standard numpy path.

    Parameters:
        fh: FileHandle instance.
        device: A ``torch.device`` instance.
        typecode: NumPy typecode string (e.g. ``'<u2'``).
        count: Number of elements to read.
        offset: Byte offset in file.
        shape: Shape to reshape the resulting tensor.

    Returns:
        A ``torch.Tensor`` on the specified device, reshaped to ``shape``.

    """
    torch = _import_torch()
    np_dtype = numpy.dtype(typecode)

    # Try kvikio GPUDirect Storage on Linux
    if device.type == 'cuda' and sys.platform == 'linux':
        tensor = _read_kvikio(fh, device, np_dtype, count, offset, shape)
        if tensor is not None:
            return tensor

    # Pinned memory path (Windows + Linux fallback)
    if device.type == 'cuda':
        return _read_pinned(fh, device, np_dtype, count, offset, shape)

    # Non-CUDA device: read as numpy, convert
    result = fh.read_array(typecode, count, offset)
    result.shape = shape
    return numpy_to_tensor(result, device)


def _read_kvikio(
    fh: Any,
    device: Any,
    np_dtype: numpy.dtype[Any],
    count: int,
    offset: int,
    shape: tuple[int, ...],
) -> Any | None:
    """Try GPUDirect Storage via kvikio. Returns None if unavailable."""
    try:
        import cupy
        import kvikio
    except ImportError:
        return None

    torch = _import_torch()
    nbytes = count * np_dtype.itemsize

    filepath = fh.path
    if filepath is None:
        return None

    try:
        gpu_buf = cupy.empty(count, dtype=np_dtype)
        with kvikio.CuFile(filepath, 'r') as f:
            f.pread(gpu_buf, nbytes, offset)

        tensor = torch.as_tensor(gpu_buf, device=device)

        # Handle byte-swap if needed
        if np_dtype.byteorder == '>' or (
            np_dtype.byteorder == '=' and sys.byteorder == 'big'
        ):
            tensor = tensor.contiguous()
            # byteswap on GPU via view trick
            _gpu_byteswap(tensor, np_dtype.itemsize)

        tensor = tensor.reshape(shape)
        logger().debug('GPU read via kvikio GPUDirect Storage')
        return tensor
    except Exception as exc:
        logger().debug(f'kvikio read failed: {exc!r}, falling back to pinned')
        return None


def _read_pinned(
    fh: Any,
    device: Any,
    np_dtype: numpy.dtype[Any],
    count: int,
    offset: int,
    shape: tuple[int, ...],
) -> Any:
    """Read file data into pinned CPU memory, async DMA to GPU.

    On torch 2.3+, unsigned dtypes are read directly into pinned tensors
    of the native unsigned type. On older torch, falls back to reading
    via numpy and converting through ``numpy_to_tensor``.
    """
    torch = _import_torch()
    dtype_map = _get_torch_dtype_map()
    nbytes = count * np_dtype.itemsize

    if np_dtype in dtype_map:
        # Native path — read directly into pinned tensor
        torch_dtype = dtype_map[np_dtype]
        pinned = torch.empty(count, dtype=torch_dtype, pin_memory=True)
        buf = pinned.numpy().view(numpy.uint8)

        if fh._fh is None:
            raise ValueError('file handle is closed')
        fh._fh.seek(offset)
        n = fh._fh.readinto(buf)
        if n != nbytes:
            msg = f'expected {nbytes} bytes, got {n}'
            raise ValueError(msg)

        # Byte-swap if file byte order differs from native byte order
        needs_byteswap = np_dtype.byteorder == '>' or (
            np_dtype.byteorder == '<' and sys.byteorder == 'big'
        )
        if needs_byteswap:
            pinned.numpy().byteswap(inplace=True)

        return pinned.to(device).reshape(shape)

    # Fallback: read as numpy, convert (handles upcast for torch < 2.3)
    result = fh.read_array(np_dtype.str, count, offset)
    result.shape = shape
    return numpy_to_tensor(result, device)


def _gpu_byteswap(tensor: Any, itemsize: int) -> None:
    """In-place byte-swap on GPU tensor."""
    torch = _import_torch()
    if itemsize == 1:
        return
    flat = tensor.view(torch.uint8)
    flat_reshaped = flat.reshape(-1, itemsize)
    flipped = flat_reshaped.flip(1).contiguous()
    flat[:] = flipped.reshape(-1)


# --- Phase 3: GPU Codec Registry ---


class GpuCodecRegistry:
    """Registry mapping TIFF compression IDs to GPU decode functions.

    Probes for available GPU libraries (nvCOMP, nvImageCodec) at first use.
    Returns None for unsupported codecs, triggering CPU fallback.
    """

    _initialized: bool = False
    _codecs: dict[int, Any] = {}

    @classmethod
    def _init(cls) -> None:
        """Probe for available GPU codec libraries."""
        if cls._initialized:
            return
        cls._initialized = True

        # Try nvCOMP for Zstd
        try:
            from kvikio._lib import nvcomp  # noqa: F401

            cls._codecs[50000] = _gpu_decompress_zstd
            logger().debug('GPU codec: nvCOMP zstd available')
        except ImportError:
            pass

        # Try nvImageCodec for JPEG
        try:
            import nvidia.nvimgcodec as nvimgcodec  # noqa: F401

            for comp_id in (6, 7, 34892):
                cls._codecs[comp_id] = _gpu_decompress_jpeg
            logger().debug('GPU codec: nvImageCodec JPEG available')
        except ImportError:
            pass

        # Try nvImageCodec for JPEG2000
        try:
            import nvidia.nvimgcodec as nvimgcodec  # noqa: F401

            for comp_id in (33003, 33004, 33005):
                if comp_id not in cls._codecs:
                    cls._codecs[comp_id] = _gpu_decompress_jpeg2000
            logger().debug('GPU codec: nvImageCodec JPEG2000 available')
        except ImportError:
            pass

    @classmethod
    def get(cls, compression: int) -> Any | None:
        """Return GPU decode function for compression ID, or None."""
        cls._init()
        return cls._codecs.get(compression)


def _gpu_decompress_zstd(
    data: bytes,
    device: Any,
    out_nbytes: int,
) -> Any:
    """Decompress zstd data on GPU using nvCOMP."""
    import cupy
    from kvikio._lib import nvcomp

    torch = _import_torch()

    # Upload compressed data to GPU
    comp_gpu = cupy.array(numpy.frombuffer(data, dtype=numpy.uint8))
    decomp_gpu = cupy.empty(out_nbytes, dtype=numpy.uint8)

    nvcomp.zstd_decompress(comp_gpu, decomp_gpu)

    return torch.as_tensor(decomp_gpu, device=device)


_nvimgcodec_decoder: Any = None


def _get_nvimgcodec_decoder() -> Any:
    """Return cached nvImageCodec decoder instance."""
    global _nvimgcodec_decoder
    if _nvimgcodec_decoder is None:
        import nvidia.nvimgcodec as nvimgcodec

        _nvimgcodec_decoder = nvimgcodec.Decoder()
    return _nvimgcodec_decoder


def _gpu_decompress_jpeg(
    data: bytes,
    device: Any,
    out_nbytes: int,
) -> Any:
    """Decompress JPEG data on GPU using nvImageCodec."""
    torch = _import_torch()
    result = _get_nvimgcodec_decoder().decode(data)
    return torch.as_tensor(result, device=device)


def _gpu_decompress_jpeg2000(
    data: bytes,
    device: Any,
    out_nbytes: int,
) -> Any:
    """Decompress JPEG2000 data on GPU using nvImageCodec."""
    torch = _import_torch()
    result = _get_nvimgcodec_decoder().decode(data)
    return torch.as_tensor(result, device=device)


def gpu_cumsum_predictor(data: Any, axis: int = -1) -> Any:
    """Reverse horizontal differencing predictor on GPU.

    Equivalent to numpy cumsum for predictor=2 undo.

    Parameters:
        data: GPU tensor with delta-encoded data.
        axis: Axis along which to cumsum.

    Returns:
        GPU tensor with predictor reversed (in-place if possible).

    """
    torch = _import_torch()
    return torch.cumsum(data, dim=axis, out=data)


# --- GPU Writing Support ---


def _is_cuda_tensor(data: Any) -> bool:
    """Check if data is a CUDA tensor (torch or cupy)."""
    if hasattr(data, 'is_cuda'):
        return data.is_cuda  # torch tensor
    if hasattr(data, '__cuda_array_interface__'):
        return True  # cupy or similar
    return False


def _torch_dtype_to_numpy(torch_dtype: Any) -> numpy.dtype[Any]:
    """Convert torch dtype to numpy dtype."""
    dtype_map = _get_torch_dtype_map()
    for np_dt, t_dt in dtype_map.items():
        if t_dt == torch_dtype:
            return np_dt
    msg = f'unsupported torch dtype {torch_dtype}'
    raise TypeError(msg)


def tensor_to_numpy(tensor: Any) -> NDArray:
    """Convert torch tensor to numpy array (D2H transfer if CUDA)."""
    if hasattr(tensor, 'cpu'):
        if tensor.is_cuda:
            tensor = tensor.cpu()
        return tensor.detach().numpy()
    return numpy.asarray(tensor)


_UNSIGNED_WIDEN: dict[Any, Any] | None = None


def _get_unsigned_widen_map() -> dict[Any, Any]:
    """Return mapping of unsigned torch dtypes to wider signed types."""
    global _UNSIGNED_WIDEN
    if _UNSIGNED_WIDEN is not None:
        return _UNSIGNED_WIDEN
    torch = _import_torch()
    _UNSIGNED_WIDEN = {
        torch.uint8: torch.int16,
        torch.uint16: torch.int32,
        torch.uint32: torch.int64,
    }
    # uint64 has no wider signed type — handle explicitly if present
    if hasattr(torch, 'uint64'):
        _UNSIGNED_WIDEN[torch.uint64] = None  # sentinel: unsupported
    return _UNSIGNED_WIDEN


def gpu_delta_encode(tensor: Any, axis: int = -1) -> Any:
    """Apply horizontal differencing predictor on GPU (encode direction).

    This is the inverse of gpu_cumsum_predictor (decode direction).
    Result: [x[0], x[1]-x[0], x[2]-x[1], ...]

    Uses wrapping (modular) subtraction for unsigned dtypes, matching
    the TIFF predictor 2 specification.

    Parameters:
        tensor: GPU tensor with image data.
        axis: Axis along which to difference.

    Returns:
        GPU tensor with predictor applied.

    Raises:
        TypeError: If tensor dtype is uint64 (no wider signed type).

    """
    orig_dtype = tensor.dtype

    # torch doesn't support arithmetic on unsigned types (uint16, etc.)
    # Cast to wider signed type for subtraction, then cast back for wrapping
    widen_map = _get_unsigned_widen_map()
    work_dtype = widen_map.get(orig_dtype)
    if work_dtype is None and orig_dtype in widen_map:
        msg = f'gpu_delta_encode does not support {orig_dtype}'
        raise TypeError(msg)
    if work_dtype is not None:
        tensor = tensor.to(work_dtype)

    result = tensor.clone()
    idx_dst = [slice(None)] * tensor.ndim
    idx_src = [slice(None)] * tensor.ndim
    idx_prev = [slice(None)] * tensor.ndim
    idx_dst[axis] = slice(1, None)
    idx_src[axis] = slice(1, None)
    idx_prev[axis] = slice(None, -1)
    result[tuple(idx_dst)] = tensor[tuple(idx_src)] - tensor[tuple(idx_prev)]

    if work_dtype is not None:
        result = result.to(orig_dtype)
    return result


class GpuEncoderRegistry:
    """Registry for GPU compression/encoding functions.

    Probes for available GPU libraries (nvidia.nvcomp, nvImageCodec) at
    first use. Returns None for unsupported codecs, triggering CPU fallback.

    Uses ``nvidia.nvcomp`` (``pip install nvidia-nvcomp-cu12``) which
    provides GPU-accelerated zstd/lz4/deflate/snappy with standard-format
    output compatible with CPU decoders.

    GPU compression is only efficient for large buffers (>50 MB). The
    write path forces one strip per plane so each nvCOMP call processes
    a full image plane, avoiding per-tile kernel launch overhead.
    """

    _initialized: bool = False
    _codecs: dict[int, Any] = {}

    # Minimum plane size (bytes) for GPU encoding to be worthwhile.
    # Below this, kernel launch overhead dominates and CPU is faster.
    # Empirically ~50 MB crossover for zstd on RTX 4090 + PCIe Gen4.
    MIN_PLANE_BYTES: int = 48 * 1024 * 1024  # 48 MB

    @classmethod
    def _init(cls) -> None:
        """Probe for available GPU encoder libraries."""
        if cls._initialized:
            return
        cls._initialized = True

        # nvidia.nvcomp — official NVIDIA Python package (Windows + Linux)
        try:
            import nvidia.nvcomp as nvcomp  # noqa: F401

            # COMPRESSION tag -> nvcomp algorithm name.
            # Only algorithms faster than CPU are included.
            # nvCOMP deflate is ~50x slower than CPU — excluded.
            _NVCOMP_ALGOS: dict[int, str] = {
                # ZSTD
                50000: 'zstd',
                34926: 'zstd',
            }
            for comp_id, algo in _NVCOMP_ALGOS.items():
                cls._codecs[comp_id] = algo
            logger().debug(
                'GPU encoder: nvidia.nvcomp available '
                f'(algorithms: {set(_NVCOMP_ALGOS.values())})'
            )
        except ImportError:
            # Fall back to kvikio (Linux-only, older path)
            try:
                from kvikio._lib import nvcomp  # noqa: F401

                if hasattr(nvcomp, 'zstd_compress'):
                    for comp_id in (50000, 34926):
                        cls._codecs[comp_id] = 'kvikio_zstd'
                    logger().debug('GPU encoder: kvikio nvCOMP zstd available')
            except ImportError:
                pass

    @classmethod
    def get(cls, compression: int) -> str | None:
        """Return GPU encoder algorithm name for compression ID, or None."""
        cls._init()
        return cls._codecs.get(compression)


_nvcomp_codecs: dict[str, Any] = {}


def _get_nvcomp_codec(algorithm: str) -> Any:
    """Return cached nvcomp Codec for the given algorithm."""
    codec = _nvcomp_codecs.get(algorithm)
    if codec is None:
        import nvidia.nvcomp as nvcomp

        codec = nvcomp.Codec(
            algorithm=algorithm,
            bitstream_kind=nvcomp.BitstreamKind.RAW,
        )
        _nvcomp_codecs[algorithm] = codec
    return codec


def gpu_compress(
    data: Any,
    algorithm: str,
    **kwargs: Any,
) -> bytes:
    """Compress byte buffer on GPU using nvCOMP.

    Parameters:
        data: CUDA tensor or buffer to compress (viewed as uint8).
        algorithm: nvCOMP algorithm name ('zstd', 'lz4', etc.)
            or 'kvikio_zstd' for legacy kvikio path.

    Returns:
        Standard-format compressed bytes on CPU.

    """
    if algorithm == 'kvikio_zstd':
        return _gpu_compress_kvikio_zstd(data)

    import nvidia.nvcomp as nvcomp

    torch = _import_torch()

    # View tensor as flat uint8 for nvcomp
    flat = data.contiguous().view(torch.uint8)
    arr = nvcomp.as_array(flat)

    codec = _get_nvcomp_codec(algorithm)
    compressed = codec.encode(arr)
    return bytes(compressed.cpu())


def _gpu_compress_kvikio_zstd(data: Any) -> bytes:
    """Legacy kvikio path for zstd compression (Linux only)."""
    import cupy
    from kvikio._lib import nvcomp

    data_cupy = cupy.asarray(data).view(cupy.uint8)
    # Allocate with headroom for incompressible data (zstd framing overhead)
    out_size = data_cupy.nbytes + data_cupy.nbytes // 128 + 1024
    compressed = cupy.empty(out_size, dtype=cupy.uint8)
    nbytes = nvcomp.zstd_compress(data_cupy, compressed)
    return compressed[:nbytes].get().tobytes()


def gpu_encode_planes(
    tensor: Any,
    algorithm: str,
    storedshape: Any,
    predictortag: int,
    compressionaxis: int,
) -> Any:
    """Yield compressed bytes for each plane, encoded on GPU.

    Each plane (separate_samples x depth) is compressed as a single
    buffer in one GPU kernel call, avoiding per-tile/strip overhead.

    Iteration order matches iter_strips with rowsperstrip=full height:
    pages -> separate_samples -> depths -> (one strip = full plane)

    Parameters:
        tensor: CUDA tensor with image data.
        algorithm: nvCOMP algorithm name for gpu_compress().
        storedshape: StoredShape describing the image layout.
        predictortag: TIFF predictor tag (1=none, 2=horizontal diff).
        compressionaxis: Axis for predictor application.

    Yields:
        Compressed bytes for each plane.

    """
    data = tensor.reshape(storedshape.shape)

    for page in data:
        for plane in page:
            for depth in plane:
                # depth shape: (length, width, contig_samples)
                strip = depth.contiguous()
                if predictortag == 2:
                    strip = gpu_delta_encode(strip, axis=compressionaxis)
                yield gpu_compress(strip, algorithm)
