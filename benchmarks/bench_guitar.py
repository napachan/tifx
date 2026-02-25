"""Benchmark guitar_stack.tif: C++ + mmap vs C++ only vs Python-only."""
from __future__ import annotations

import json
import os
import subprocess
import sys

BENCH_SCRIPT = r'''
import gc, json, os, statistics, sys, time
import numpy as np

fname = 'C:/Users/HEQ/Projects/tempdata/guitar_stack.tif'

import tifffile
if os.environ.get('TIFFFILE_NO_MMAP'):
    from tifffile.fileio import FileHandle
    FileHandle._get_mmap_view = lambda self: None

WARMUP = 1
RUNS = 3

def timeit(func):
    for _ in range(WARMUP):
        func()
        gc.collect()
    times = []
    for _ in range(RUNS):
        gc.collect()
        t0 = time.perf_counter()
        r = func()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)
    return statistics.median(times), r

results = []

def run(label, func):
    ms, r = timeit(func)
    results.append({"label": label, "ms": ms})

def f_open_len():
    with tifffile.TiffFile(fname) as t:
        return len(t.pages)
run("open+len(130K)", f_open_len)

def f_open_series():
    with tifffile.TiffFile(fname) as t:
        s = t.series
        return len(s)
run("open+series", f_open_series)

def f_iterate():
    with tifffile.TiffFile(fname) as t:
        _ = t.series
        return sum(1 for _ in t.pages)
run("iterate 130K pages", f_iterate)

def f_page0():
    with tifffile.TiffFile(fname) as t:
        return t.pages[0].asarray().shape
run("read page[0]", f_page0)

def f_page_mid():
    with tifffile.TiffFile(fname) as t:
        return t.pages[65000].asarray().shape
run("read page[65000]", f_page_mid)

def f_page_last():
    with tifffile.TiffFile(fname) as t:
        return t.pages[129999].asarray().shape
run("read page[129999]", f_page_last)

def f_stack10():
    with tifffile.TiffFile(fname) as t:
        return np.stack([t.pages[i].asarray() for i in range(10)]).shape
run("stack 10 pages", f_stack10)

def f_seq1000():
    with tifffile.TiffFile(fname) as t:
        _ = t.series
        for i in range(1000):
            t.pages[i].asarray()
run("sequential 1K pages", f_seq1000)

def f_asarray():
    with tifffile.TiffFile(fname) as t:
        return t.asarray().shape
run("asarray() all 130K", f_asarray)

print(json.dumps(results))
'''


def main():
    py = sys.executable
    env_base = {**os.environ, 'PYTHONPATH': '.'}

    configs = [
        ('C++ + mmap', {}),
        ('C++ only', {'TIFFFILE_NO_MMAP': '1'}),
        ('Python-only', {'TIFFFILE_NO_CPP': '1', 'TIFFFILE_NO_MMAP': '1'}),
    ]

    all_results = {}
    for name, extra_env in configs:
        print(f'Running: {name} ...', flush=True)
        env = dict(env_base)
        # Clean slate
        env.pop('TIFFFILE_NO_CPP', None)
        env.pop('TIFFFILE_NO_MMAP', None)
        env.update(extra_env)
        r = subprocess.run(
            [py, '-c', BENCH_SCRIPT],
            capture_output=True, text=True, env=env, timeout=600,
        )
        if r.returncode != 0:
            print(f'  ERROR: {r.stderr[:500]}')
            continue
        all_results[name] = json.loads(r.stdout.strip().split('\n')[-1])

    if len(all_results) < 3:
        print('Some benchmarks failed, aborting.')
        return

    labels = [r['label'] for r in all_results['C++ + mmap']]
    print()
    print('=' * 105)
    print(' guitar_stack.tif  |  130K pages, 15.9 GB BigTIFF, uint8 (256x512), uncompressed')
    print('=' * 105)
    print()
    hdr = (
        f'{"Operation":<25} {"C+++mmap":>10} {"C++ only":>10} '
        f'{"Python":>10}  {"C++mmap/Py":>11} {"C++/Py":>8} {"mmap/C++":>9}'
    )
    print(hdr)
    print('-' * 88)

    def marker(x):
        if x > 10:
            return ' ***'
        if x > 3:
            return ' **'
        if x > 1.5:
            return ' *'
        return ''

    for i, label in enumerate(labels):
        cm = all_results['C++ + mmap'][i]['ms']
        cn = all_results['C++ only'][i]['ms']
        pn = all_results['Python-only'][i]['ms']
        sp_full = pn / cm if cm > 0.01 else float('inf')
        sp_cpp = pn / cn if cn > 0.01 else float('inf')
        sp_mmap = cn / cm if cm > 0.01 else float('inf')
        print(
            f'{label:<25} {cm:>8.1f}ms {cn:>8.1f}ms {pn:>8.1f}ms'
            f'  {sp_full:>9.1f}x{marker(sp_full):<3}'
            f' {sp_cpp:>6.1f}x{marker(sp_cpp):<3}'
            f' {sp_mmap:>6.1f}x{marker(sp_mmap):<3}'
        )

    print(f'\n * >1.5x  ** >3x  *** >10x')


if __name__ == '__main__':
    main()
