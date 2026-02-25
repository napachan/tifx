"""Benchmark larger files for slow compression codecs: GPU vs numpy.

Tests JPEG XL, JPEG 2000, JPEG XR, WebP, LERC with larger stacks to see
how GPU transfer overhead scales relative to codec decode time.
"""
import gc
import os
import sys
import tempfile
import time

import numpy

# Use local tifffile
import tifffile


def make_file(path, data, **kwargs):
    """Write test file and return file size."""
    tifffile.imwrite(path, data, **kwargs)
    return os.path.getsize(path) / 1e6


def bench_interleaved(path, repeats=2):
    """Interleaved benchmark: numpy vs manual-to-GPU vs device='cuda'.

    Runs them in rotation to equalize OS cache effects.
    """
    import torch

    device = torch.device('cuda')

    # Warmup all paths
    a = tifffile.imread(path)
    t = torch.from_numpy(a.copy()).pin_memory().to(device, non_blocking=True)
    torch.cuda.synchronize()
    del t
    r = tifffile.imread(path, device='cuda')
    torch.cuda.synchronize()
    del r
    gc.collect()
    torch.cuda.empty_cache()

    np_times = []
    manual_times = []
    direct_times = []

    for _ in range(repeats):
        # numpy
        gc.collect()
        t0 = time.perf_counter()
        a = tifffile.imread(path)
        np_times.append(time.perf_counter() - t0)

        # manual: numpy → pin → GPU
        gc.collect()
        t0 = time.perf_counter()
        a2 = tifffile.imread(path)
        t = torch.from_numpy(numpy.ascontiguousarray(a2))
        t = t.pin_memory().to(device, non_blocking=True)
        torch.cuda.synchronize()
        manual_times.append(time.perf_counter() - t0)
        del t

        # device='cuda'
        gc.collect()
        torch.cuda.empty_cache()
        t0 = time.perf_counter()
        r = tifffile.imread(path, device='cuda')
        torch.cuda.synchronize()
        direct_times.append(time.perf_counter() - t0)
        del r

    return {
        'numpy': min(np_times) * 1000,
        'manual': min(manual_times) * 1000,
        'direct': min(direct_times) * 1000,
        'shape': a.shape,
        'dtype': str(a.dtype),
        'nbytes_mb': a.nbytes / 1e6,
    }


def print_result(label, file_mb, result):
    np_ms = result['numpy']
    man_ms = result['manual']
    dir_ms = result['direct']
    overhead = dir_ms - np_ms
    pct = (overhead / np_ms) * 100 if np_ms > 0 else 0
    shape_s = 'x'.join(str(x) for x in result['shape'])
    print(
        f'  {label:<42s} '
        f'{file_mb:>6.1f}MB  '
        f'{shape_s:<20s} '
        f'{result["dtype"]:<8s} '
        f'{np_ms:>8.1f}  '
        f'{man_ms:>8.1f}  '
        f'{dir_ms:>8.1f}  '
        f'{overhead:>+7.1f} ({pct:>+5.1f}%)',
        flush=True,
    )


def main():
    import torch

    print(f'GPU: {torch.cuda.get_device_name(0)}', flush=True)
    print(f'CUDA: {torch.version.cuda}', flush=True)
    print(flush=True)

    TD = tempfile.mkdtemp(prefix='tiffbench_gpu_')

    # Generate data of various sizes
    # u8 RGB for JPEG/WebP, u8 gray for JPEG XL/JPEG XR
    # u16 gray for JPEG 2000/LERC

    sizes = [
        # (label, pages, H, W)
        ('200p_1k', 200, 1024, 1024),
        ('500p_1k', 500, 1024, 1024),
    ]

    # The slow codecs to test
    codecs = {
        'jpegxl': {
            'compression': 'jpegxl',
            'dtypes': ['u8_gray', 'u16_gray'],
        },
        'jpeg2000': {
            'compression': 'jpeg2000',
            'dtypes': ['u8_gray', 'u16_gray'],
        },
        'jpegxr': {
            'compression': 'jpegxr',
            'dtypes': ['u8_gray', 'u16_gray'],
        },
        'webp': {
            'compression': 'webp',
            'dtypes': ['u8_rgb'],  # WebP only supports RGB
        },
        'lerc': {
            'compression': 'lerc',
            'dtypes': ['u8_gray', 'u16_gray'],
        },
    }

    header = (
        f'  {"Test":<42s} '
        f'{"File":>7s}  '
        f'{"Shape":<20s} '
        f'{"dtype":<8s} '
        f'{"numpy":>8s}  '
        f'{"manual":>8s}  '
        f'{"direct":>8s}  '
        f'{"overhead":>16s}'
    )

    for codec_name, codec_info in codecs.items():
        print('=' * 130, flush=True)
        print(f'  {codec_name.upper()} compression', flush=True)
        print('=' * 130, flush=True)
        print(header, flush=True)
        print(f'  {"-" * 128}', flush=True)

        compression = codec_info['compression']

        for size_label, pages, H, W in sizes:
            for dtype_label in codec_info['dtypes']:
                # Generate data
                if dtype_label == 'u8_gray':
                    data = numpy.random.randint(
                        0, 255, (pages, H, W), dtype='uint8'
                    )
                    photo = 'minisblack'
                elif dtype_label == 'u8_rgb':
                    data = numpy.random.randint(
                        0, 255, (pages, H, W, 3), dtype='uint8'
                    )
                    photo = 'rgb'
                elif dtype_label == 'u16_gray':
                    data = numpy.random.randint(
                        0, 4095, (pages, H, W), dtype='uint16'
                    )
                    photo = 'minisblack'
                else:
                    continue

                fname = f'{codec_name}_{size_label}_{dtype_label}.tif'
                fpath = os.path.join(TD, fname)
                label = f'{codec_name} {size_label} {dtype_label}'

                try:
                    file_mb = make_file(
                        fpath,
                        data,
                        photometric=photo,
                        compression=compression,
                        tile=(256, 256),
                    )
                except Exception as e:
                    print(f'  {label:<42s} WRITE FAILED: {e}')
                    continue

                try:
                    result = bench_interleaved(fpath)
                    print_result(label, file_mb, result)
                except Exception as e:
                    print(f'  {label:<42s} BENCH FAILED: {e}')
                finally:
                    # Clean up file to save disk space
                    if os.path.exists(fpath):
                        os.remove(fpath)

                del data
                gc.collect()

        print()

    # Cleanup temp dir
    import shutil

    shutil.rmtree(TD, ignore_errors=True)

    print('=' * 130)
    print('Done.')
    print()
    print('Legend:')
    print('  numpy   = tifffile.imread(path)                    [CPU only]')
    print('  manual  = tifffile.imread(path) → pin → .to(cuda)  [baseline]')
    print('  direct  = tifffile.imread(path, device="cuda")      [our path]')
    print('  overhead = direct - numpy  (GPU transfer cost)')


if __name__ == '__main__':
    main()
