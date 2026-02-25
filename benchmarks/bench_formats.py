import numpy, time, os, tempfile, shutil
import tifffile


def bench(label, func, repeats=3):
    func()
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        r = func()
        times.append(time.perf_counter() - t0)
    best = min(times)
    mb_out = r.nbytes / 1e6
    mb_s = mb_out / best if best > 0 else 0
    print(f'    {label:50s} {best*1000:8.1f} ms  {mb_out:7.2f} MB  {mb_s:7.1f} MB/s')


TD = tempfile.mkdtemp(prefix='tiffbench_')

# ================================================================
print('=' * 75)
print('1. OME-TIFF  (TZCYX uint16, zlib, tiled 128x128)')
print('=' * 75)
f1 = os.path.join(TD, 'ome.ome.tif')
d1 = numpy.random.randint(0, 4095, (20, 10, 3, 512, 512), dtype='uint16')
tifffile.imwrite(f1, d1, photometric='minisblack', compression='zlib',
                 tile=(128, 128), metadata={'axes': 'TZCYX'})
sz = os.path.getsize(f1) / 1e6
with tifffile.TiffFile(f1) as t:
    s = t.series[0]
    kf = s.keyframe
    print(f'  {sz:.1f} MB  shape={s.shape} axes={s.axes} dtype={s.dtype}'
          f'  tiles/page={len(kf.dataoffsets)}')
print()
bench('Full read', lambda: tifffile.imread(f1))
bench('T=5 (30 pages)', lambda: tifffile.imread(f1, selection={'T': 5}))
bench('T=5, Z=3, C=1 (1 page)',
      lambda: tifffile.imread(f1, selection={'T': 5, 'Z': 3, 'C': 1}))
bench('T=5, Z=3, C=1, ROI 128x128 (1/16 tiles)',
      lambda: tifffile.imread(f1, selection={'T': 5, 'Z': 3, 'C': 1,
                                              'Y': slice(0, 128),
                                              'X': slice(0, 128)}))
bench('T=0:5, Z=::3 (20 pages)',
      lambda: tifffile.imread(f1, selection={'T': slice(0, 5),
                                              'Z': slice(None, None, 3)}))

# ================================================================
print()
print('=' * 75)
print('2. BigTIFF  (uint16, uncompressed, contiguous)')
print('=' * 75)
f2 = os.path.join(TD, 'big.tif')
d2 = numpy.random.randint(0, 65535, (500, 512, 512), dtype='uint16')
tifffile.imwrite(f2, d2, bigtiff=True, photometric='minisblack')
sz = os.path.getsize(f2) / 1e6
with tifffile.TiffFile(f2) as t:
    s = t.series[0]
    print(f'  {sz:.1f} MB  shape={s.shape} axes={s.axes} dtype={s.dtype}'
          f'  contiguous={s.dataoffset is not None}  bigtiff={t.is_bigtiff}')
print()
bench('Full read', lambda: tifffile.imread(f2))
bench('Frame 0 (mmap)', lambda: tifffile.imread(f2, selection=(0,)))
bench('Frame 0:50 (mmap)', lambda: tifffile.imread(f2, selection=(slice(0, 50),)))
bench('Frame 0, Y=100:200 (sub-page mmap)',
      lambda: tifffile.imread(f2, selection=(0, slice(100, 200))))
bench('All frames, X=256 (strided MT)',
      lambda: tifffile.imread(f2, selection=(slice(None), slice(None), 256)))

# ================================================================
print()
print('=' * 75)
print('3. Tiled RGB  (zstd, 256x256 tiles, 2048x2048, 8 frames)')
print('=' * 75)
f3 = os.path.join(TD, 'wsi.tif')
d3 = numpy.random.randint(0, 255, (8, 2048, 2048, 3), dtype='uint8')
tifffile.imwrite(f3, d3, photometric='rgb', compression='zstd',
                 tile=(256, 256))
sz = os.path.getsize(f3) / 1e6
with tifffile.TiffFile(f3) as t:
    s = t.series[0]
    kf = s.keyframe
    print(f'  {sz:.1f} MB  shape={s.shape} axes={s.axes} dtype={s.dtype}'
          f'  tiles/page={len(kf.dataoffsets)}')
print()
bench('Full read', lambda: tifffile.imread(f3))
bench('Frame 0 (all 64 tiles)',
      lambda: tifffile.imread(f3, selection=(0,)))
bench('Frame 0, 256x256 ROI (1/64 tiles)',
      lambda: tifffile.imread(f3, selection=(0, slice(512, 768),
                                              slice(512, 768))))
bench('Frame 0, 512x512 ROI (4/64 tiles)',
      lambda: tifffile.imread(f3, selection=(0, slice(512, 1024),
                                              slice(512, 1024))))
bench('Frame 0:4, 256x256 ROI',
      lambda: tifffile.imread(f3, selection=(slice(0, 4), slice(512, 768),
                                              slice(512, 768))))

# ================================================================
print()
print('=' * 75)
print('4. LZW multi-strip  (float32, 32 strips/page, 100 frames)')
print('=' * 75)
f4 = os.path.join(TD, 'lzw.tif')
d4 = numpy.random.rand(100, 1024, 1024).astype('float32')
tifffile.imwrite(f4, d4, photometric='minisblack', compression='lzw',
                 rowsperstrip=32)
sz = os.path.getsize(f4) / 1e6
with tifffile.TiffFile(f4) as t:
    s = t.series[0]
    kf = s.keyframe
    print(f'  {sz:.1f} MB  shape={s.shape} axes={s.axes} dtype={s.dtype}'
          f'  strips/page={len(kf.dataoffsets)}')
print()
bench('Full read', lambda: tifffile.imread(f4))
bench('Frame 0 (all 32 strips)',
      lambda: tifffile.imread(f4, selection=(0,)))
bench('Frame 0, Y=0:32 (1/32 strips)',
      lambda: tifffile.imread(f4, selection=(0, slice(0, 32))))
bench('Frame 0, Y=0:128 (4/32 strips)',
      lambda: tifffile.imread(f4, selection=(0, slice(0, 128))))
bench('Frame 0:10, Y=0:128 (4/32 x 10)',
      lambda: tifffile.imread(f4, selection=(slice(0, 10),
                                              slice(0, 128))))

# ================================================================
print()
print('=' * 75)
print('5. Deflate + predictor  (uint16 volume, 64x64 tiles)')
print('=' * 75)
f5 = os.path.join(TD, 'vol.tif')
d5 = numpy.random.randint(0, 4095, (64, 256, 256), dtype='uint16')
tifffile.imwrite(f5, d5, photometric='minisblack', compression='deflate',
                 predictor=True, tile=(64, 64))
sz = os.path.getsize(f5) / 1e6
with tifffile.TiffFile(f5) as t:
    s = t.series[0]
    kf = s.keyframe
    print(f'  {sz:.1f} MB  shape={s.shape} axes={s.axes} dtype={s.dtype}'
          f'  tiles/page={len(kf.dataoffsets)}  predictor={kf.predictor}')
print()
bench('Full read', lambda: tifffile.imread(f5))
bench('Z=32 (1 slice)',
      lambda: tifffile.imread(f5, selection=(32,)))
bench('Z=0:8 (8 slices)',
      lambda: tifffile.imread(f5, selection=(slice(0, 8),)))
bench('Z=32, 64x64 ROI (1/16 tiles)',
      lambda: tifffile.imread(f5, selection=(32, slice(96, 160),
                                              slice(96, 160))))

# ================================================================
print()
print('=' * 75)
print('6. Shaped TIFF  (uncompressed float64, scientific)')
print('=' * 75)
f6 = os.path.join(TD, 'shaped.tif')
d6 = numpy.random.rand(200, 128, 128).astype('float64')
tifffile.imwrite(f6, d6, photometric='minisblack')
sz = os.path.getsize(f6) / 1e6
with tifffile.TiffFile(f6) as t:
    s = t.series[0]
    print(f'  {sz:.1f} MB  shape={s.shape} axes={s.axes} dtype={s.dtype}'
          f'  contiguous={s.dataoffset is not None}')
print()
bench('Full read', lambda: tifffile.imread(f6))
bench('Frame 50 (mmap)',
      lambda: tifffile.imread(f6, selection=(50,)))
bench('Frame 0:20 (mmap)',
      lambda: tifffile.imread(f6, selection=(slice(0, 20),)))
bench('Frame 50, Y=32:96 (sub-page)',
      lambda: tifffile.imread(f6, selection=(50, slice(32, 96))))

# ================================================================
ome = 'tests/data/public/tifffile/multiscene_pyramidal.ome.tif'
if os.path.exists(ome):
    print()
    print('=' * 75)
    print('7. Real OME  (multiscene_pyramidal.ome.tif)')
    print('=' * 75)
    sz = os.path.getsize(ome) / 1e6
    with tifffile.TiffFile(ome) as t:
        s = t.series[0]
        kf = s.keyframe
        print(f'  {sz:.1f} MB  shape={s.shape} axes={s.axes} dtype={s.dtype}'
              f'  tiles/page={len(kf.dataoffsets)}  compression={kf.compression}')
    print()
    bench('Full series 0', lambda: tifffile.imread(ome))
    bench('T=8 (64 pages)',
          lambda: tifffile.imread(ome, selection={'T': 8}))
    bench('T=8, Z=16, C=0 (1 page)',
          lambda: tifffile.imread(ome, selection={'T': 8, 'Z': 16, 'C': 0}))
    bench('T=8, Z=16, C=0, ROI 128x128',
          lambda: tifffile.imread(ome, selection={
              'T': 8, 'Z': 16, 'C': 0,
              'Y': slice(0, 128), 'X': slice(0, 128)}))
    bench('Level 1: T=0, Z=0, C=0',
          lambda: tifffile.imread(ome, series=0, level=1,
                                  selection={'T': 0, 'Z': 0, 'C': 0}))

shutil.rmtree(TD)
print()
print('=' * 75)
print('Done.')
