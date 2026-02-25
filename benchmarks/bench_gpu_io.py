"""Benchmark: GPU TIFF I/O with step-by-step timing.

Compares all viable compression codecs for large TIFF stacks to/from GPU,
reporting per-phase timing (D2H, compress, disk write, etc.) and file sizes.

Usage:
    PYTHONPATH=. .venv/Scripts/python.exe benchmarks/bench_gpu_io.py
"""

from __future__ import annotations

import gc
import io
import os
import shutil
import sys
import tempfile
import time
from typing import Any

import numpy
import torch

import tifffile
from tifffile.gpu import GpuEncoderRegistry, gpu_compress, tensor_to_numpy

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DATA_CONFIGS: list[tuple[str, tuple[int, ...], str]] = [
    # (label, shape, numpy dtype string)
    ('1x4K u16', (1, 4096, 4096), 'uint16'),
    ('1x8K u16', (1, 8192, 8192), 'uint16'),
    ('50x2K u16', (50, 2048, 2048), 'uint16'),
    ('50x2K f32', (50, 2048, 2048), 'float32'),
    ('10x4K u16', (10, 4096, 4096), 'uint16'),
]

# Codec name -> (compression arg for imwrite, compressionargs)
CODECS: dict[str, tuple[str | None, dict[str, Any]]] = {
    'none': (None, {}),
    'zstd': ('zstd', {'level': 1}),
    'zstd(cpu)': ('zstd', {'level': 1}),  # forced CPU path via numpy input
    'deflate': ('deflate', {'level': 1}),
    'lzw': ('lzw', {}),  # LZW has no level parameter
}

WARMUP = 2
REPEATS = 5

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_NP_TO_TORCH: dict[str, torch.dtype] = {
    'uint8': torch.uint8,
    'uint16': torch.uint16,
    'int16': torch.int16,
    'int32': torch.int32,
    'float32': torch.float32,
    'float64': torch.float64,
}


def format_size(nbytes: int | float) -> str:
    if nbytes >= 1e9:
        return f'{nbytes / 1e9:.1f} GB'
    return f'{nbytes / 1e6:.1f} MB'


def format_ms(ms: float | None) -> str:
    if ms is None:
        return '-'
    if ms >= 1000:
        return f'{ms / 1000:.2f}s'
    return f'{ms:.0f}ms'


def make_image(shape: tuple[int, ...], dtype: str) -> torch.Tensor:
    """Create realistic compressible GPU image (gradient + noise)."""
    h, w = shape[-2], shape[-1]
    y = torch.linspace(0, 1, h, device='cuda', dtype=torch.float32)
    x = torch.linspace(0, 0.3, w, device='cuda', dtype=torch.float32)
    base = y.unsqueeze(1) + x.unsqueeze(0)

    if dtype == 'float32':
        noise = torch.randn(h, w, device='cuda') * 0.01
        plane = base + noise
    elif dtype == 'float64':
        noise = torch.randn(h, w, device='cuda') * 0.01
        plane = (base + noise).to(torch.float64)
    else:
        maxval = 4095 if dtype == 'uint16' else 255
        noise = torch.randn(h, w, device='cuda') * (maxval * 0.01)
        plane = (base * maxval + noise).clamp(0, maxval)
        torch_dt = _NP_TO_TORCH[dtype]
        if torch_dt == torch.uint16:
            plane = plane.to(torch.int32).to(torch.int16).view(torch.uint16)
        else:
            plane = plane.to(torch_dt)

    n_pages = shape[0] if len(shape) >= 3 else 1
    if n_pages == 1:
        return plane.unsqueeze(0).contiguous()
    return plane.unsqueeze(0).expand(n_pages, h, w).contiguous()


def bench_median(fn: Any, warmup: int = WARMUP, repeats: int = REPEATS) -> float:
    """Return median wall-clock ms with CUDA sync."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(repeats):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)
    times.sort()
    return times[len(times) // 2]


def timed_once(fn: Any) -> float:
    """Time a single call with CUDA sync, return ms."""
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1000


# ---------------------------------------------------------------------------
# Component measurements
# ---------------------------------------------------------------------------


def measure_d2h(tensor: torch.Tensor) -> tuple[float, numpy.ndarray]:
    """Time D2H transfer, return (ms, numpy_array)."""
    # Warmup
    _ = tensor_to_numpy(tensor)
    torch.cuda.synchronize()

    times = []
    result = None
    for _ in range(REPEATS):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        result = tensor_to_numpy(tensor)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)
    times.sort()
    assert result is not None
    return times[len(times) // 2], result


def measure_h2d(arr: numpy.ndarray) -> tuple[float, torch.Tensor]:
    """Time H2D transfer, return (ms, cuda_tensor)."""
    from tifffile.gpu import numpy_to_tensor, parse_device

    dev = parse_device('cuda')

    # Warmup
    _ = numpy_to_tensor(arr, dev)
    torch.cuda.synchronize()

    times = []
    result = None
    for _ in range(REPEATS):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        result = numpy_to_tensor(arr, dev)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)
    times.sort()
    assert result is not None
    return times[len(times) // 2], result


def measure_cpu_compress_plane(
    arr_plane: numpy.ndarray, codec: str
) -> tuple[float, int]:
    """Time CPU compression of one plane, return (ms_per_plane, compressed_size).

    Uses imagecodecs directly for isolated timing.
    """
    import imagecodecs

    encode_fns: dict[str, Any] = {
        'zstd': imagecodecs.zstd_encode,
        'deflate': imagecodecs.deflate_encode,
        'lzw': imagecodecs.lzw_encode,
    }
    encode_fn = encode_fns.get(codec)
    if encode_fn is None:
        return 0.0, arr_plane.nbytes

    # LZW has no level parameter
    encode_kwargs: dict[str, Any] = {}
    if codec in ('zstd', 'deflate'):
        encode_kwargs['level'] = 1

    data = numpy.ascontiguousarray(arr_plane)

    # Warmup
    for _ in range(2):
        encode_fn(data, **encode_kwargs)

    times = []
    compressed = None
    for _ in range(REPEATS):
        t0 = time.perf_counter()
        compressed = encode_fn(data, **encode_kwargs)
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)
    times.sort()
    assert compressed is not None
    return times[len(times) // 2], len(compressed)


def measure_gpu_compress_plane(
    tensor_plane: torch.Tensor,
) -> tuple[float, int]:
    """Time GPU nvCOMP zstd compression of one plane, return (ms, size)."""
    algo = GpuEncoderRegistry.get(50000)
    if algo is None:
        return 0.0, 0

    flat = tensor_plane.contiguous()

    # Warmup
    for _ in range(2):
        gpu_compress(flat, algo)
    torch.cuda.synchronize()

    times = []
    compressed = None
    for _ in range(REPEATS):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        compressed = gpu_compress(flat, algo)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)
    times.sort()
    assert compressed is not None
    return times[len(times) // 2], len(compressed)


# ---------------------------------------------------------------------------
# End-to-end write/read benchmarks
# ---------------------------------------------------------------------------


def bench_write(
    tensor: torch.Tensor,
    np_data: numpy.ndarray,
    codec: str,
    tmpdir: str,
) -> dict[str, Any]:
    """Full write benchmark for one config+codec."""
    compression, compressionargs = CODECS[codec]
    is_gpu_zstd = codec == 'zstd' and GpuEncoderRegistry.get(50000) is not None
    n_pages = tensor.shape[0]
    plane_bytes = tensor[0].nelement() * tensor.element_size()
    gpu_eligible = is_gpu_zstd and plane_bytes >= GpuEncoderRegistry.MIN_PLANE_BYTES

    result: dict[str, Any] = {
        'codec': codec,
        'path': None,
        'file_size': 0,
        'ratio': 1.0,
        'd2h_ms': None,
        'compress_ms': None,
        'compress_label': None,
        'e2e_ram_ms': None,
        'e2e_ssd_ms': None,
        'write_path': '',
    }

    kwargs: dict[str, Any] = {}
    if compression is not None:
        kwargs['compression'] = compression
        kwargs.update(compressionargs)

    # --- Component: D2H ---
    if not gpu_eligible and codec != 'none':
        d2h_ms, _ = measure_d2h(tensor)
        result['d2h_ms'] = d2h_ms

    # --- Component: compress one plane ---
    if codec == 'none':
        result['compress_ms'] = None
        result['compress_label'] = '-'
    elif gpu_eligible:
        ms, _ = measure_gpu_compress_plane(tensor[0])
        result['compress_ms'] = ms
        result['compress_label'] = f'{ms:.0f}ms(GPU)'
    elif codec in ('zstd', 'zstd(cpu)', 'deflate', 'lzw'):
        real_codec = 'zstd' if codec == 'zstd(cpu)' else codec
        # Also need to measure zstd CPU if main zstd uses GPU
        if codec == 'zstd' and not gpu_eligible:
            real_codec = 'zstd'
        ms, _ = measure_cpu_compress_plane(np_data[0], real_codec)
        result['compress_ms'] = ms
        result['compress_label'] = f'{ms:.0f}ms(CPU)'

    # --- E2E imwrite to BytesIO ---
    if codec == 'zstd(cpu)':
        # Force CPU path: use numpy array
        def write_ram():
            buf = io.BytesIO()
            tifffile.imwrite(buf, np_data, **kwargs)
            return buf.tell()
    elif codec in ('zstd',) and gpu_eligible:
        # GPU path: use CUDA tensor
        def write_ram():
            buf = io.BytesIO()
            tifffile.imwrite(buf, tensor, **kwargs)
            return buf.tell()
    elif compression is not None:
        # CPU codec with CUDA tensor (D2H fallback inside imwrite)
        def write_ram():
            buf = io.BytesIO()
            tifffile.imwrite(buf, tensor, **kwargs)
            return buf.tell()
    else:
        # No compression: use CUDA tensor
        def write_ram():
            buf = io.BytesIO()
            tifffile.imwrite(buf, tensor, **kwargs)
            return buf.tell()

    result['e2e_ram_ms'] = bench_median(write_ram)

    # --- E2E imwrite to SSD ---
    ssd_path = os.path.join(tmpdir, f'bench_{codec}.tif')

    if codec == 'zstd(cpu)':
        def write_ssd():
            tifffile.imwrite(ssd_path, np_data, **kwargs)
    else:
        def write_ssd():
            tifffile.imwrite(ssd_path, tensor, **kwargs)

    result['e2e_ssd_ms'] = bench_median(write_ssd)
    result['file_size'] = os.path.getsize(ssd_path)
    raw_bytes = tensor.nelement() * tensor.element_size()
    result['ratio'] = raw_bytes / result['file_size'] if result['file_size'] > 0 else 1.0
    result['path'] = ssd_path

    # Determine write path label
    if codec == 'none':
        result['write_path'] = 'D2H+write'
    elif gpu_eligible:
        result['write_path'] = 'GPU nvCOMP'
    elif codec == 'zstd(cpu)':
        result['write_path'] = 'D2H+CPU zstd'
    elif codec == 'zstd':
        result['write_path'] = 'D2H+CPU zstd(small)'
    else:
        result['write_path'] = f'D2H+CPU {codec}'

    return result


def bench_read(
    path: str,
    raw_bytes: int,
    codec: str,
) -> dict[str, Any]:
    """Full read benchmark for one TIFF file."""
    result: dict[str, Any] = {
        'codec': codec,
        'open_ms': None,
        'decompress_ms': None,
        'h2d_ms': None,
        'e2e_cuda_ms': None,
        'e2e_numpy_ms': None,
    }

    # --- File open + IFD parse ---
    def do_open():
        with tifffile.TiffFile(path) as tif:
            _ = len(tif.pages)

    # Warmup
    for _ in range(WARMUP):
        do_open()
    times = []
    for _ in range(REPEATS):
        t0 = time.perf_counter()
        do_open()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)
    times.sort()
    result['open_ms'] = times[len(times) // 2]

    # --- E2E imread to numpy ---
    def read_numpy():
        return tifffile.imread(path)

    result['e2e_numpy_ms'] = bench_median(read_numpy)

    # --- CPU decompress + H2D ---
    arr = tifffile.imread(path)

    if codec != 'none':
        # Decompress time ≈ e2e_numpy - open
        result['decompress_ms'] = max(0, result['e2e_numpy_ms'] - result['open_ms'])

    h2d_ms, _ = measure_h2d(arr)
    result['h2d_ms'] = h2d_ms

    # --- E2E imread with device='cuda' ---
    def read_cuda():
        r = tifffile.imread(path, device='cuda')
        torch.cuda.synchronize()
        return r

    result['e2e_cuda_ms'] = bench_median(read_cuda)

    del arr
    gc.collect()
    torch.cuda.empty_cache()

    return result


# ---------------------------------------------------------------------------
# Per-config driver
# ---------------------------------------------------------------------------


def bench_config(
    label: str,
    shape: tuple[int, ...],
    dtype: str,
    tmpdir: str,
) -> None:
    """Run full read+write benchmark for one data configuration."""
    raw_bytes = 1
    for s in shape:
        raw_bytes *= s
    raw_bytes *= numpy.dtype(dtype).itemsize

    print(f'\n{"=" * 90}')
    print(f'  {label}  shape={shape}  dtype={dtype}  raw={format_size(raw_bytes)}')
    print(f'{"=" * 90}')

    # Generate data
    tensor = make_image(shape, dtype)
    np_data = tensor_to_numpy(tensor)
    torch.cuda.synchronize()

    codec_list = list(CODECS.keys())

    # ---- WRITE TABLE ----
    write_results: list[dict[str, Any]] = []
    for codec in codec_list:
        try:
            wr = bench_write(tensor, np_data, codec, tmpdir)
            write_results.append(wr)
        except Exception as e:
            print(f'  WRITE {codec}: FAILED — {e}')

    print(f'\n  WRITE: GPU tensor -> TIFF')
    hdr = (
        f'  {"Codec":<12s} {"FileSize":>9s} {"Ratio":>6s} '
        f'{"D2H":>8s} {"Compress":>12s} '
        f'{"E2E(RAM)":>10s} {"E2E(SSD)":>10s} {"Path"}'
    )
    print(hdr)
    print(f'  {"-" * (len(hdr) - 2)}')
    for wr in write_results:
        print(
            f'  {wr["codec"]:<12s} '
            f'{format_size(wr["file_size"]):>9s} '
            f'{wr["ratio"]:>5.1f}x '
            f'{format_ms(wr["d2h_ms"]):>8s} '
            f'{wr["compress_label"] or "-":>12s} '
            f'{format_ms(wr["e2e_ram_ms"]):>10s} '
            f'{format_ms(wr["e2e_ssd_ms"]):>10s} '
            f'{wr["write_path"]}'
        )

    # ---- READ TABLE ----
    read_results: list[dict[str, Any]] = []
    for wr in write_results:
        if wr['path'] is None:
            continue
        # Skip zstd(cpu) read — same file as zstd
        if wr['codec'] == 'zstd(cpu)':
            continue
        try:
            rd = bench_read(wr['path'], raw_bytes, wr['codec'])
            read_results.append(rd)
        except Exception as e:
            print(f'  READ {wr["codec"]}: FAILED — {e}')

    if read_results:
        print(f'\n  READ: TIFF -> GPU tensor')
        hdr2 = (
            f'  {"Codec":<12s} {"Decomp":>8s} {"H2D":>8s} '
            f'{"E2E(cuda)":>11s} {"E2E(numpy)":>11s}'
        )
        print(hdr2)
        print(f'  {"-" * (len(hdr2) - 2)}')
        for rd in read_results:
            print(
                f'  {rd["codec"]:<12s} '
                f'{format_ms(rd["decompress_ms"]):>8s} '
                f'{format_ms(rd["h2d_ms"]):>8s} '
                f'{format_ms(rd["e2e_cuda_ms"]):>11s} '
                f'{format_ms(rd["e2e_numpy_ms"]):>11s}'
            )

    # Cleanup write files
    for wr in write_results:
        if wr['path'] and os.path.exists(wr['path']):
            os.remove(wr['path'])

    del tensor, np_data
    gc.collect()
    torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# System info
# ---------------------------------------------------------------------------


def print_system_info() -> None:
    print('GPU TIFF I/O Benchmark - Step-by-Step Timing (level=1)')
    print('=' * 60)
    print(f'GPU:       {torch.cuda.get_device_name(0)}')
    print(f'CUDA:      {torch.version.cuda}')
    print(f'PyTorch:   {torch.__version__}')
    print(f'tifffile:  {tifffile.__version__}')

    try:
        import imagecodecs

        print(f'imagecodecs: {imagecodecs.__version__}')
    except ImportError:
        print('imagecodecs: NOT INSTALLED')

    try:
        import nvidia.nvcomp

        print(f'nvCOMP:    {nvidia.nvcomp.__version__}')
    except ImportError:
        print('nvCOMP:    not installed')

    algo = GpuEncoderRegistry.get(50000)
    print(f'Zstd GPU:  {algo or "unavailable"}')
    print(f'Min plane: {GpuEncoderRegistry.MIN_PLANE_BYTES / 1e6:.0f} MB')
    print(f'Platform:  {sys.platform}')

    # Quick disk speed estimate
    tmpdir = tempfile.mkdtemp(prefix='tiffbench_')
    try:
        test_data = numpy.zeros(100 * 1024 * 1024, dtype='uint8')  # 100 MB
        test_path = os.path.join(tmpdir, 'disktest.bin')
        t0 = time.perf_counter()
        with open(test_path, 'wb') as f:
            f.write(test_data.tobytes())
        t1 = time.perf_counter()
        write_bw = 100 / (t1 - t0)
        t0 = time.perf_counter()
        with open(test_path, 'rb') as f:
            f.read()
        t1 = time.perf_counter()
        read_bw = 100 / (t1 - t0)
        print(f'Disk:      ~{write_bw:.0f} MB/s write, ~{read_bw:.0f} MB/s read')
        os.remove(test_path)
    except Exception:
        pass
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

    print()
    print(f'Benchmark: {WARMUP} warmup, {REPEATS} repeats, median reported')


# ---------------------------------------------------------------------------
# Roundtrip verification
# ---------------------------------------------------------------------------


def verify_roundtrip(tmpdir: str) -> None:
    """Verify that all codec paths produce correct data."""
    print(f'\n{"=" * 60}')
    print('Roundtrip Verification')
    print(f'{"=" * 60}')

    test_configs = [
        ('uint16', torch.uint16, (2, 512, 512)),
        ('float32', torch.float32, (2, 512, 512)),
    ]

    for dt_name, dt_torch, shape in test_configs:
        tensor = make_image(shape, dt_name)
        np_orig = tensor_to_numpy(tensor)
        torch.cuda.synchronize()

        for codec in ('none', 'zstd', 'deflate', 'lzw'):
            compression, compressionargs = CODECS[codec]
            kwargs: dict[str, Any] = {}
            if compression:
                kwargs['compression'] = compression
                kwargs.update(compressionargs)

            # Write with tensor
            buf = io.BytesIO()
            tifffile.imwrite(buf, tensor, **kwargs)
            buf.seek(0)
            readback = tifffile.imread(buf)

            if numpy.array_equal(np_orig, readback):
                status = 'OK'
            else:
                diff = numpy.abs(
                    np_orig.astype('float64') - readback.astype('float64')
                )
                status = f'MISMATCH max_diff={diff.max():.6f}'

            # Also verify device='cuda' roundtrip
            buf.seek(0)
            readback_gpu = tifffile.imread(buf, device='cuda')
            torch.cuda.synchronize()
            readback_gpu_np = tensor_to_numpy(readback_gpu)
            if numpy.array_equal(np_orig, readback_gpu_np):
                gpu_status = 'OK'
            else:
                diff = numpy.abs(
                    np_orig.astype('float64') - readback_gpu_np.astype('float64')
                )
                gpu_status = f'MISMATCH max_diff={diff.max():.6f}'

            print(f'  {dt_name:>8s} {codec:>10s}:  write->numpy={status}  write->cuda={gpu_status}')

        del tensor, np_orig
        gc.collect()
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    print_system_info()

    tmpdir = tempfile.mkdtemp(prefix='tiffbench_gpu_io_')
    try:
        for label, shape, dtype in DATA_CONFIGS:
            bench_config(label, shape, dtype, tmpdir)

        verify_roundtrip(tmpdir)
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

    print(f'\n{"=" * 60}')
    print('Done.')
    print()
    print('Legend:')
    print('  D2H        = Device-to-Host transfer (CUDA tensor -> numpy)')
    print('  H2D        = Host-to-Device transfer (numpy -> CUDA tensor)')
    print('  Compress   = Single-plane compression time (CPU or GPU)')
    print('  E2E(RAM)   = imwrite to BytesIO (no disk I/O)')
    print('  E2E(SSD)   = imwrite to temp file (includes disk I/O)')
    print('  E2E(cuda)  = imread with device="cuda"')
    print('  E2E(numpy) = imread to numpy array (CPU only)')
    print('  Decomp     = estimated decompression (E2E numpy - file open)')


if __name__ == '__main__':
    main()
