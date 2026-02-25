"""Benchmark tifffile read performance with and without C++ extension."""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np


def benchmark(func, warmup=1, runs=5, label=''):
    """Run benchmark and return average time in ms."""
    for _ in range(warmup):
        func()
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        func()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)
    avg = sum(times) / len(times)
    std = (sum((t - avg) ** 2 for t in times) / len(times)) ** 0.5
    return avg, std


def find_test_files():
    """Find test files for benchmarking."""
    test_dir = Path('tests/data/public')
    files = {}

    candidates = {
        'gray_u1': 'imagecodecs/gray.u1.tif',
        'multipage_rgb': 'imageio/multipage_rgb.tif',
        'contig_tiled': 'GDAL/contig_tiled.tif',
        'generic_series': 'tifffile/generic_series.tif',
        'movie_30k': 'tifffile/movie.tif',
        '100k_pages': 'tifffile/100000_pages.tif',
        'circular_ifd': 'Tiff-Library-4J/IFD struct/Circular E.tif',
    }

    for key, relpath in candidates.items():
        path = test_dir / relpath
        if path.exists():
            files[key] = str(path)

    return files


def run_benchmarks(compare=False):
    """Run read benchmarks."""
    import tifffile
    import tifffile.page

    files = find_test_files()
    if not files:
        print('No test files found. Run tests/generate_tifffile_data.py first.')
        return

    cpp_available = tifffile._HAS_CPP

    print(f'tifffile version: {tifffile.__version__}')
    print(f'C++ extension: {"available" if cpp_available else "not available"}')
    print(f'Python: {sys.version}')
    print()

    results = {}

    if compare and cpp_available:
        modes = [('C++', True), ('Python', False)]
    elif cpp_available:
        modes = [('C++', True)]
    else:
        modes = [('Python', False)]

    for mode_name, use_cpp in modes:
        tifffile.page._HAS_CPP = use_cpp
        print(f'=== {mode_name} mode ===')
        results[mode_name] = {}

        # Benchmark 1: Open + len(pages) for large files
        for key in ['movie_30k', '100k_pages']:
            if key not in files:
                continue
            path = files[key]

            def bench_open_len(p=path):
                with tifffile.TiffFile(p) as tif:
                    return len(tif.pages)

            avg, std = benchmark(bench_open_len, warmup=1, runs=5)
            n = bench_open_len()
            print(f'  open + len(pages) [{key}, {n} pages]: '
                  f'{avg:.1f} +/- {std:.1f} ms')
            results[mode_name][f'open_len_{key}'] = avg

        # Benchmark 2: Open + read first page
        for key in ['gray_u1', 'contig_tiled']:
            if key not in files:
                continue
            path = files[key]

            def bench_read_page(p=path):
                with tifffile.TiffFile(p) as tif:
                    return tif.pages[0].asarray()

            avg, std = benchmark(bench_read_page, warmup=1, runs=10)
            data = bench_read_page()
            print(f'  open + read page [{key}, {data.shape}]: '
                  f'{avg:.1f} +/- {std:.1f} ms')
            results[mode_name][f'read_page_{key}'] = avg

        # Benchmark 3: Open + read all pages (multi-page)
        if 'generic_series' in files:
            path = files['generic_series']

            def bench_read_all(p=path):
                with tifffile.TiffFile(p) as tif:
                    return tif.asarray()

            avg, std = benchmark(bench_read_all, warmup=1, runs=5)
            data = bench_read_all()
            print(f'  open + read all [{os.path.basename(path)}, '
                  f'{data.shape}]: {avg:.1f} +/- {std:.1f} ms')
            results[mode_name][f'read_all_generic'] = avg

        # Benchmark 4: iterate pages (TiffFrame loading)
        for key in ['movie_30k']:
            if key not in files:
                continue
            path = files[key]

            def bench_iterate(p=path):
                with tifffile.TiffFile(p) as tif:
                    tif.pages.useframes = True
                    tif.pages.cache = True
                    count = 0
                    for page in tif.pages:
                        count += 1
                    return count

            avg, std = benchmark(bench_iterate, warmup=1, runs=3)
            n = bench_iterate()
            print(f'  iterate frames [{key}, {n} frames]: '
                  f'{avg:.1f} +/- {std:.1f} ms')
            results[mode_name][f'iterate_{key}'] = avg

        print()

    # Print comparison
    if compare and len(results) == 2:
        print('=== Comparison ===')
        cpp_results = results.get('C++', {})
        py_results = results.get('Python', {})
        for key in cpp_results:
            if key in py_results and py_results[key] > 0:
                speedup = py_results[key] / cpp_results[key]
                print(f'  {key}: {speedup:.2f}x speedup '
                      f'(C++ {cpp_results[key]:.1f}ms vs '
                      f'Python {py_results[key]:.1f}ms)')


if __name__ == '__main__':
    compare = '--compare' in sys.argv
    run_benchmarks(compare=compare)
