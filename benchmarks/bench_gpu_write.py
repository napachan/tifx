"""Benchmark: GPU-accelerated TIFF writing via nvidia.nvcomp.

Compares three write paths for CUDA tensors:
  A) CPU baseline:  numpy array -> imwrite (CPU compression)
  B) GPU fallback:  CUDA tensor -> imwrite (D2H + CPU compression)
  C) GPU nvCOMP:    CUDA tensor -> imwrite (GPU compression, compressed D2H)

Usage:
    PYTHONPATH=. uv run --no-project python benchmarks/bench_gpu_write.py
"""

from __future__ import annotations

import io
import time
from typing import Any

import numpy
import torch

import tifffile
from tifffile.gpu import tensor_to_numpy


def bench(
    fn: Any, warmup: int = 3, repeats: int = 7
) -> tuple[float, float, Any]:
    """Return (median_ms, min_ms, last_result)."""
    result = None
    for _ in range(warmup):
        result = fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(repeats):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        result = fn()
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)
    times.sort()
    return times[len(times) // 2], times[0], result


def format_size(nbytes: int) -> str:
    if nbytes >= 1e9:
        return f'{nbytes / 1e9:.1f} GB'
    return f'{nbytes / 1e6:.1f} MB'


def make_image(shape: tuple[int, ...], dtype: torch.dtype) -> Any:
    """Create realistic compressible image on GPU."""
    h, w = shape[-2], shape[-1]
    y = torch.linspace(0, 1, h, device='cuda', dtype=torch.float32)
    x = torch.linspace(0, 0.3, w, device='cuda', dtype=torch.float32)
    base = y.unsqueeze(1) + x.unsqueeze(0)

    if dtype == torch.float32:
        noise = torch.randn(h, w, device='cuda') * 0.01
        plane = base + noise
    else:
        maxval = 4095 if dtype == torch.uint16 else 255
        noise = torch.randn(h, w, device='cuda') * (maxval * 0.01)
        plane = (base * maxval + noise).clamp(0, maxval)
        if dtype == torch.uint8:
            plane = plane.to(torch.uint8)
        elif dtype == torch.uint16:
            plane = plane.to(torch.int32).to(torch.int16).view(torch.uint16)

    if len(shape) == 2:
        return plane
    batch = shape[0] if len(shape) == 3 else 1
    return plane.unsqueeze(0).expand(batch, h, w).contiguous()


def run_e2e(
    label: str,
    shape: tuple[int, ...],
    dtype_torch: torch.dtype,
    compression: str | None,
) -> dict[str, Any]:
    """End-to-end imwrite benchmark."""
    tensor = make_image(shape, dtype_torch)
    np_data = tensor_to_numpy(tensor)
    raw_bytes = tensor.nelement() * tensor.element_size()
    torch.cuda.synchronize()

    kwargs: dict[str, Any] = {}
    if compression:
        kwargs['compression'] = compression

    # A) CPU baseline: numpy -> imwrite
    def cpu_write():
        buf = io.BytesIO()
        tifffile.imwrite(buf, np_data, **kwargs)
        return buf.tell()

    cpu_ms, _, fsize = bench(cpu_write, warmup=2, repeats=5)

    # B+C) GPU path: CUDA tensor -> imwrite
    # (uses GPU nvCOMP if available and plane large enough, else D2H fallback)
    def gpu_write():
        buf = io.BytesIO()
        tifffile.imwrite(buf, tensor, **kwargs)
        return buf.tell()

    gpu_ms, _, gpu_fsize = bench(gpu_write, warmup=2, repeats=5)

    return {
        'label': label,
        'raw_bytes': raw_bytes,
        'file_size': fsize,
        'gpu_file_size': gpu_fsize,
        'cpu_ms': cpu_ms,
        'gpu_ms': gpu_ms,
    }


def main() -> None:
    print('GPU Write Benchmark: End-to-End imwrite()')
    print('=' * 80)
    print(f'GPU: {torch.cuda.get_device_name(0)}')

    # Check nvcomp availability
    try:
        import nvidia.nvcomp

        print(f'nvCOMP: {nvidia.nvcomp.__version__} (GPU compression enabled)')
    except ImportError:
        print('nvCOMP: not installed (GPU path will use D2H + CPU fallback)')

    from tifffile.gpu import GpuEncoderRegistry

    algo = GpuEncoderRegistry.get(50000)
    print(f'Zstd GPU encoder: {algo or "unavailable"}')
    print(f'Min plane size: {GpuEncoderRegistry.MIN_PLANE_BYTES / 1e6:.0f} MB')
    print()

    configs = [
        # label, shape, dtype, compression
        # Below MIN_PLANE_BYTES threshold (should use CPU fallback)
        ('2K u8 zstd', (1, 2048, 2048), torch.uint8, 'zstd'),
        ('2K u16 zstd', (1, 2048, 2048), torch.uint16, 'zstd'),
        # Above threshold — GPU encoding
        ('4K u8 zstd', (1, 4096, 4096), torch.uint8, 'zstd'),
        ('4K u16 zstd', (1, 4096, 4096), torch.uint16, 'zstd'),
        ('4K f32 zstd', (1, 4096, 4096), torch.float32, 'zstd'),
        ('8K u8 zstd', (1, 8192, 8192), torch.uint8, 'zstd'),
        ('8K u16 zstd', (1, 8192, 8192), torch.uint16, 'zstd'),
        ('12K u16 zstd', (1, 12288, 12288), torch.uint16, 'zstd'),
        ('16K u8 zstd', (1, 16384, 16384), torch.uint8, 'zstd'),
        ('16K u16 zstd', (1, 16384, 16384), torch.uint16, 'zstd'),
        # Deflate (not GPU-accelerated — nvCOMP deflate is too slow)
        ('8K u8 deflate', (1, 8192, 8192), torch.uint8, 'deflate'),
        # Multi-page: individual planes below threshold -> CPU fallback
        ('10x4K u16 zstd', (10, 4096, 4096), torch.uint16, 'zstd'),
        # Multi-page: individual planes above threshold -> GPU
        ('4x8K u16 zstd', (4, 8192, 8192), torch.uint16, 'zstd'),
        # Uncompressed (no GPU benefit, just D2H)
        ('8K u8 none', (1, 8192, 8192), torch.uint8, None),
    ]

    hdr = (
        f'{"Config":<18s} {"Raw":>7s} {"Comp":>5s} '
        f'{"CPU(np)":>9s} {"GPU(cuda)":>10s} {"Speedup":>8s} {"Note":s}'
    )
    print(hdr)
    print('-' * len(hdr))

    for label, shape, dt, comp in configs:
        r = run_e2e(label, shape, dt, comp)
        speedup = r['cpu_ms'] / r['gpu_ms']
        ratio = (
            f'{r["raw_bytes"] / r["file_size"]:.1f}x'
            if r['file_size'] > 0
            else 'n/a'
        )
        plane_bytes = shape[-2] * shape[-1] * (
            4 if dt == torch.float32 else 2 if dt == torch.uint16 else 1
        )
        comp_tag = {
            'zstd': 50000, 'deflate': 8, 'lzw': 5,
        }.get(comp or '', 0)
        gpu_algo = GpuEncoderRegistry.get(comp_tag) if comp_tag else None
        if (
            gpu_algo
            and plane_bytes >= GpuEncoderRegistry.MIN_PLANE_BYTES
        ):
            note = 'GPU nvCOMP'
        elif comp:
            note = 'CPU fallback'
        else:
            note = 'uncompressed'

        sign = '+' if speedup >= 1 else ''
        print(
            f'{label:<18s} '
            f'{format_size(r["raw_bytes"]):>7s} '
            f'{ratio:>5s} '
            f'{r["cpu_ms"]:>8.1f}ms '
            f'{r["gpu_ms"]:>9.1f}ms '
            f'{sign}{speedup:>6.2f}x '
            f'{note}'
        )

    # Roundtrip verification
    print()
    print('Roundtrip Verification:')
    for dt_name, dt in [('uint8', torch.uint8), ('uint16', torch.uint16),
                         ('float32', torch.float32)]:
        t = make_image((1, 4096, 4096), dt)
        buf = io.BytesIO()
        tifffile.imwrite(buf, t, compression='zstd')
        buf.seek(0)
        readback = tifffile.imread(buf)
        np_orig = tensor_to_numpy(t)
        if numpy.array_equal(np_orig, readback):
            print(f'  {dt_name}: OK')
        else:
            diff = numpy.abs(np_orig.astype(float) - readback.astype(float))
            print(f'  {dt_name}: MISMATCH max_diff={diff.max():.6f}')


if __name__ == '__main__':
    main()
