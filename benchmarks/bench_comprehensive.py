"""Comprehensive benchmark: C++ extension vs Python-only."""
from __future__ import annotations

import gc
import json
import os
import statistics
import subprocess
import sys
import time

BENCH_SCRIPT = r'''
import gc, json, os, statistics, sys, time

DATA = 'tests/data/public/tifffile'
BENCH = 'tests/data/bench'
WARMUP = 1
RUNS = 5

if os.environ.get('TIFFFILE_NO_CPP'):
    mode = 'Python-only'
else:
    mode = 'C++'

import tifffile
import numpy as np

def timeit(func, warmup=WARMUP, runs=RUNS):
    for _ in range(warmup):
        func()
        gc.collect()
    times = []
    for _ in range(runs):
        gc.collect()
        t0 = time.perf_counter()
        result = func()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)
    return statistics.median(times), result

def bench(path, op, page_idx=0):
    if not os.path.exists(path):
        return None
    if op == 'len':
        def run():
            with tifffile.TiffFile(path) as t:
                return len(t.pages)
        ms, val = timeit(run)
        return {'ms': ms, 'info': f'{val} pages'}
    elif op == 'series':
        def run():
            with tifffile.TiffFile(path) as t:
                s = t.series
                return len(s), s[0].shape if s else None
        ms, (ns, shape) = timeit(run)
        return {'ms': ms, 'info': f'{ns} series, {shape}'}
    elif op == 'iterate':
        def run():
            with tifffile.TiffFile(path) as t:
                _ = t.series
                return sum(1 for _ in t.pages)
        ms, count = timeit(run)
        return {'ms': ms, 'info': f'{count} pages'}
    elif op == 'read_page':
        def run():
            with tifffile.TiffFile(path) as t:
                return t.pages[page_idx].asarray().shape
        ms, shape = timeit(run)
        return {'ms': ms, 'info': f'page[{page_idx}] {shape}'}
    elif op == 'asarray':
        def run():
            with tifffile.TiffFile(path) as t:
                return t.asarray().shape
        ms, shape = timeit(run)
        return {'ms': ms, 'info': f'{shape}'}

benchmarks = []

# movie 30K
benchmarks.append(('movie 30K', 'open+len', bench(f'{DATA}/movie.tif', 'len')))
benchmarks.append(('movie 30K', 'open+series', bench(f'{DATA}/movie.tif', 'series')))
benchmarks.append(('movie 30K', 'open+series+iter', bench(f'{DATA}/movie.tif', 'iterate')))
benchmarks.append(('movie 30K', 'read page[0]', bench(f'{DATA}/movie.tif', 'read_page', 0)))
benchmarks.append(('movie 30K', 'read page[15000]', bench(f'{DATA}/movie.tif', 'read_page', 15000)))

# 100K pages
benchmarks.append(('100K pages', 'open+len', bench(f'{DATA}/100000_pages.tif', 'len')))
benchmarks.append(('100K pages', 'open+series', bench(f'{DATA}/100000_pages.tif', 'series')))
benchmarks.append(('100K pages', 'open+series+iter', bench(f'{DATA}/100000_pages.tif', 'iterate')))

# 50K pages
benchmarks.append(('50K pages', 'open+len', bench(f'{BENCH}/50k_pages.tif', 'len')))
benchmarks.append(('50K pages', 'open+series', bench(f'{BENCH}/50k_pages.tif', 'series')))
benchmarks.append(('50K pages', 'open+series+iter', bench(f'{BENCH}/50k_pages.tif', 'iterate')))

# 200K BigTIFF
benchmarks.append(('200K BigTIFF', 'open+len', bench(f'{BENCH}/200k_bigtiff.tif', 'len')))
benchmarks.append(('200K BigTIFF', 'open+series', bench(f'{BENCH}/200k_bigtiff.tif', 'series')))
benchmarks.append(('200K BigTIFF', 'open+series+iter', bench(f'{BENCH}/200k_bigtiff.tif', 'iterate')))

# tiled 4K
benchmarks.append(('tiled 4K', 'open+len', bench(f'{BENCH}/tiled_4k.tif', 'len')))
benchmarks.append(('tiled 4K', 'open+series', bench(f'{BENCH}/tiled_4k.tif', 'series')))
benchmarks.append(('tiled 4K', 'asarray()', bench(f'{BENCH}/tiled_4k.tif', 'asarray')))

# tiled 8K
benchmarks.append(('tiled 8K', 'open+len', bench(f'{BENCH}/tiled_8k.tif', 'len')))
benchmarks.append(('tiled 8K', 'open+series', bench(f'{BENCH}/tiled_8k.tif', 'series')))
benchmarks.append(('tiled 8K', 'asarray()', bench(f'{BENCH}/tiled_8k.tif', 'asarray')))

# pyramidal 16K
benchmarks.append(('pyramid 16K', 'open+len', bench(f'{BENCH}/pyramidal_16k.tif', 'len')))
benchmarks.append(('pyramid 16K', 'open+series', bench(f'{BENCH}/pyramidal_16k.tif', 'series')))
benchmarks.append(('pyramid 16K', 'read page[0]', bench(f'{BENCH}/pyramidal_16k.tif', 'read_page', 0)))

# multi-series (non-uniform)
benchmarks.append(('multi-series', 'open+len', bench(f'{BENCH}/multi_series.tif', 'len')))
benchmarks.append(('multi-series', 'open+series', bench(f'{BENCH}/multi_series.tif', 'series')))
benchmarks.append(('multi-series', 'open+series+iter', bench(f'{BENCH}/multi_series.tif', 'iterate')))

# OME pyramidal
benchmarks.append(('OME pyramid', 'open+len', bench(f'{DATA}/multiscene_pyramidal.ome.tif', 'len')))
benchmarks.append(('OME pyramid', 'open+series', bench(f'{DATA}/multiscene_pyramidal.ome.tif', 'series')))

# Output as JSON
output = []
for label, op, result in benchmarks:
    if result is not None:
        output.append({'label': label, 'op': op, 'ms': result['ms'], 'info': result['info']})
    else:
        output.append({'label': label, 'op': op, 'ms': None, 'info': 'N/A'})

print(json.dumps(output))
'''


def main():
    py = sys.executable
    env_cpp = {**os.environ, 'PYTHONPATH': '.'}
    env_py = {**os.environ, 'PYTHONPATH': '.', 'TIFFFILE_NO_CPP': '1'}

    # Remove TIFFFILE_NO_CPP from C++ env if present
    env_cpp.pop('TIFFFILE_NO_CPP', None)

    print('Running C++ benchmarks...')
    r1 = subprocess.run([py, '-c', BENCH_SCRIPT], capture_output=True, text=True,
                        env=env_cpp, timeout=600)
    if r1.returncode != 0:
        print(f'C++ benchmark error:\n{r1.stderr[:2000]}')
        return
    cpp_results = json.loads(r1.stdout.strip().split('\n')[-1])

    print('Running Python-only benchmarks...')
    r2 = subprocess.run([py, '-c', BENCH_SCRIPT], capture_output=True, text=True,
                        env=env_py, timeout=600)
    if r2.returncode != 0:
        print(f'Python-only benchmark error:\n{r2.stderr[:2000]}')
        return
    py_results = json.loads(r2.stdout.strip().split('\n')[-1])

    # Print comparison
    print()
    print(f'{"="*100}')
    print(f' BENCHMARK: C++ Extension vs Python-only')
    print(f'{"="*100}')
    print()
    print(f'{"File":<16} {"Operation":<20} {"Details":<32} {"C++(ms)":>9} {"Py(ms)":>9} {"Speedup":>9}')
    print(f'{"-"*98}')

    prev_label = None
    for cpp, py in zip(cpp_results, py_results):
        label = cpp['label']
        op = cpp['op']
        info = cpp['info']
        c_ms = cpp['ms']
        p_ms = py['ms']

        if label != prev_label and prev_label is not None:
            print()
        prev_label = label

        if c_ms is not None and p_ms is not None:
            speedup = p_ms / c_ms if c_ms > 0.01 else float('inf')
            marker = ' ***' if speedup > 10 else ' **' if speedup > 3 else ' *' if speedup > 1.5 else ''
            print(f'{label:<16} {op:<20} {info:<32} {c_ms:>8.1f}  {p_ms:>8.1f}  {speedup:>7.1f}x{marker}')
        elif c_ms is not None:
            print(f'{label:<16} {op:<20} {info:<32} {c_ms:>8.1f}  {"N/A":>8}  {"N/A":>7}')
        else:
            print(f'{label:<16} {op:<20} {info:<32} {"N/A":>8}  {"N/A":>8}  {"N/A":>7}')

    print(f'\n{"="*100}')
    print(' * >1.5x  ** >3x  *** >10x')


if __name__ == '__main__':
    main()
