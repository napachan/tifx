"""Generate tifffile-specific public test data files.

Usage: python tests/generate_tifffile_data.py

Creates synthetic test TIFF files in tests/data/public/tifffile/.
These replicate the properties expected by test_tifffile.py.

Files that cannot be generated (require specific real-world data,
proprietary formats, or exact pixel values from external tools):
- MB231paxgfp_060214.lzw.lsm (real Zeiss LSM microscopy data)
- gimp_f2.tiff, gimp_f4.tiff, gimp_u2.tiff (GIMP-specific encoding)
- multiscene_pyramidal.ome.tif (complex OME pyramidal)
- test_FileHandle.bin (needs generic_series.tif + micromanager.tif)
- micromanager.tif (MicroManager format)
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy

# Add parent to path so we can import tifffile
sys.path.insert(0, str(Path(__file__).parent.parent))

from tifffile import TiffWriter, imwrite  # noqa: E402

HERE = Path(__file__).parent
TIFFFILE_DIR = HERE / 'data' / 'public' / 'tifffile'


def generate_rgb_tif():
    """Generate rgb.tif: 32x31x3 uint8 RGB image."""
    dest = TIFFFILE_DIR / 'rgb.tif'
    if dest.exists():
        print(f'  skip {dest.name} (exists)')
        return
    rng = numpy.random.default_rng(42)
    data = rng.integers(0, 255, (32, 31, 3), dtype=numpy.uint8)
    imwrite(str(dest), data, photometric='rgb')
    print(f'  wrote {dest.name} ({dest.stat().st_size} bytes)')


def generate_rgb_u1_npy():
    """Generate rgb.u1.npy: 32x31x3 RGB uint8 NumPy array.

    Used by test_write_codecs (expects imagewidth=31, imagelength=32)
    and test_write_compression_jpeg (sliced to [:32, :16]).

    Uses smooth gradient data (not random noise) so JPEG roundtrip
    stays within the tolerance expected by the compression tests.
    """
    dest = TIFFFILE_DIR / 'rgb.u1.npy'
    if dest.exists():
        print(f'  skip {dest.name} (exists)')
        return
    # Create smooth gradient image - JPEG-friendly (low spatial frequency)
    y = numpy.linspace(0, 255, 32, dtype=numpy.float64)
    x = numpy.linspace(0, 255, 31, dtype=numpy.float64)
    yy, xx = numpy.meshgrid(y, x, indexing='ij')
    data = numpy.zeros((32, 31, 3), dtype=numpy.uint8)
    data[:, :, 0] = numpy.clip(yy, 0, 255).astype(numpy.uint8)
    data[:, :, 1] = numpy.clip(xx, 0, 255).astype(numpy.uint8)
    data[:, :, 2] = numpy.clip((yy + xx) / 2, 0, 255).astype(numpy.uint8)
    numpy.save(str(dest), data)
    print(f'  wrote {dest.name} ({dest.stat().st_size} bytes)')


def generate_gray_f4_tif():
    """Generate gray.f4.tif: 83x111 float32 grayscale."""
    dest = TIFFFILE_DIR / 'gray.f4.tif'
    if dest.exists():
        print(f'  skip {dest.name} (exists)')
        return
    rng = numpy.random.default_rng(42)
    data = rng.random((83, 111), dtype=numpy.float32)
    imwrite(str(dest), data)
    print(f'  wrote {dest.name} ({dest.stat().st_size} bytes)')


def generate_generic_series_tif():
    """Generate generic_series.tif: 6 pages, 4 series.

    Series 0: 3x20x20 uint8, LZW, values at [:,9,9] = (19, 90, 206)
    Series 1: 10x10x3 float32 RGB, LZW, value at [9,9,1] = 214.5733642578125
    Series 2: 20x20x3 uint8 RGB, LZW
    Series 3: 10x10 uint16, LZW

    Uses metadata=None to avoid shaped descriptions -> 'generic' series kind.
    """
    dest = TIFFFILE_DIR / 'generic_series.tif'
    if dest.exists():
        print(f'  skip {dest.name} (exists)')
        return

    rng = numpy.random.default_rng(123)

    with TiffWriter(str(dest)) as tw:
        # Series 0: 3 pages of 20x20 uint8 grayscale, LZW
        for i, val in enumerate([19, 90, 206]):
            page = numpy.full((20, 20), val, dtype=numpy.uint8)
            page[9, 9] = val
            tw.write(page, compression='lzw', metadata=None)

        # Series 1: 10x10x3 float32 RGB, LZW
        data1 = rng.random((10, 10, 3)).astype(numpy.float32) * 255
        data1[9, 9, 1] = 214.5733642578125
        tw.write(data1, photometric='rgb', compression='lzw', metadata=None)

        # Series 2: 20x20x3 uint8 RGB, LZW
        # Value at [9,9,:] must be (19, 90, 206)
        data2 = rng.integers(0, 255, (20, 20, 3), dtype=numpy.uint8)
        data2[9, 9, :] = [19, 90, 206]
        tw.write(data2, photometric='rgb', compression='lzw', metadata=None)

        # Series 3: 10x10 float32 grayscale
        # Value at [9,9] must be 223.1648712158203
        data3 = rng.random((10, 10)).astype(numpy.float32) * 255
        data3[9, 9] = numpy.float32(223.1648712158203)
        tw.write(data3, compression='lzw', metadata=None)

    print(f'  wrote {dest.name} ({dest.stat().st_size} bytes)')


def generate_movie_tif():
    """Generate movie.tif: 30000x64x64 uint16, uncompressed, uniform.

    Value at [29999, 32, 32] must equal 460.
    Axes must be 'IYX' (uniform series), each frame a separate page.
    Little-endian, contiguous, memmappable.
    """
    dest = TIFFFILE_DIR / 'movie.tif'
    if dest.exists():
        print(f'  skip {dest.name} (exists)')
        return

    print(f'  writing {dest.name} (30000 frames, ~235 MB) ...', end='',
          flush=True)
    rng = numpy.random.default_rng(7)
    with TiffWriter(str(dest), bigtiff=False) as tw:
        for i in range(30000):
            frame = rng.integers(0, 1000, (64, 64), dtype=numpy.uint16)
            if i == 29999:
                frame[32, 32] = 460
            tw.write(frame, contiguous=True, metadata=None)
    print(f' {dest.stat().st_size} bytes')


def generate_chart_bl_tif():
    """Generate chart_bl.tif: 18710x13228, 1-bit, no compression.

    Value at [0,0] = True, [5000,5000] = False.
    Series kind must be 'uniform' (no shaped metadata).
    """
    dest = TIFFFILE_DIR / 'chart_bl.tif'
    if dest.exists():
        print(f'  skip {dest.name} (exists)')
        return

    print(f'  writing {dest.name} (18710x13228 1-bit, ~30 MB) ...',
          end='', flush=True)
    rng = numpy.random.default_rng(99)
    data = rng.random((18710, 13228)) > 0.5
    data[0, 0] = True
    data[5000, 5000] = False
    # Use metadata=None to avoid shaped description -> 'uniform' kind
    with TiffWriter(str(dest)) as tw:
        tw.write(
            data,
            photometric='minisblack',
            rowsperstrip=18710,
            metadata=None,
        )
    print(f' {dest.stat().st_size} bytes')


def generate_sgi_depth_tif():
    """Generate sgi_depth.tif: 128x128x128 float32, tiled volumetric.

    Value at [64,64,64] = 0.0.
    Software tag = 'MFL MeVis File Format Library, TIFF Module'.
    """
    dest = TIFFFILE_DIR / 'sgi_depth.tif'
    if dest.exists():
        print(f'  skip {dest.name} (exists)')
        return

    rng = numpy.random.default_rng(77)
    data = rng.random((128, 128, 128)).astype(numpy.float32)
    data[64, 64, 64] = 0.0
    imwrite(
        str(dest),
        data,
        tile=(128, 128),
        volumetric=True,
        metadata=None,
        software='MFL MeVis File Format Library, TIFF Module',
    )
    print(f'  wrote {dest.name} ({dest.stat().st_size} bytes)')


def generate_rrggbb_ij_tif():
    """Generate rrggbb.ij.tif: planar RGB ImageJ, LZW, 31x32 uint16.

    Shape (3, 32, 31), axes 'CYX'.
    Values at [:,15,15] = (812, 1755, 648).
    ImageJ metadata: channels=3, slices=1, frames=1, hyperstack=True.

    Written as a single page with planar RGB (RRGGBB layout).
    The ImageJ format normally doesn't support uint16 RGB in tifffile,
    but Bio-Formats writes it this way. We use low-level TiffWriter.
    """
    from tifffile import imagej_description

    dest = TIFFFILE_DIR / 'rrggbb.ij.tif'
    if dest.exists():
        print(f'  skip {dest.name} (exists)')
        return

    rng = numpy.random.default_rng(55)
    data = rng.integers(0, 4095, (3, 32, 31), dtype=numpy.uint16)
    data[0, 15, 15] = 812
    data[1, 15, 15] = 1755
    data[2, 15, 15] = 648

    # Build ImageJ description string manually.
    # The original was created by Bio-Formats with uint16 planar RGB,
    # which tifffile's ImageJ writer doesn't normally support.
    desc = (
        'ImageJ=\n'
        'images=3\n'
        'channels=3\n'
        'slices=1\n'
        'frames=1\n'
        'hyperstack=true\n'
        'mode=composite\n'
    )
    with TiffWriter(str(dest)) as tw:
        tw.write(
            data,
            photometric='rgb',
            planarconfig='separate',
            compression='lzw',
            description=desc,
            metadata=None,
        )
    print(f'  wrote {dest.name} ({dest.stat().st_size} bytes)')


def generate_temp_sequence_tifs():
    """Generate temp_C001T001.tif and temp_C001T002.tif for TiffSequence test."""
    for name in ('temp_C001T001.tif', 'temp_C001T002.tif'):
        dest = TIFFFILE_DIR / name
        if dest.exists():
            print(f'  skip {dest.name} (exists)')
            continue
        data = numpy.zeros((16, 16), dtype=numpy.uint8)
        imwrite(str(dest), data)
        print(f'  wrote {dest.name} ({dest.stat().st_size} bytes)')


def generate_100000_pages_tif():
    """Generate 100000_pages.tif: 100000x64x64 uint16, big-endian, ImageJ.

    Value at [7310, 25, 25] ≈ 100.
    ImageJ metadata: ImageJ='1.48g', max=119.0, min=86.0.
    Axes: 'TYX'. Series kind: 'imagej'.

    The test expects page._nextifd() == 819200206 (data starts at offset 206).
    To achieve this exact layout, we write a minimal ImageJ description
    and use the description parameter directly instead of imagej=True.
    """
    dest = TIFFFILE_DIR / '100000_pages.tif'
    if dest.exists():
        print(f'  skip {dest.name} (exists)')
        return

    print(f'  writing {dest.name} (100000 frames, ~780 MB) ...',
          end='', flush=True)

    # Exact description matching the original file's ImageJ metadata
    desc = (
        'ImageJ=1.48g\n'
        'images=100000\n'
        'frames=100000\n'
        'hyperstack=true\n'
        'mode=grayscale\n'
        'max=119.0\n'
        'min=86.0\n'
    )

    rng = numpy.random.default_rng(42)
    with TiffWriter(
        str(dest), bigtiff=True, byteorder='>'
    ) as tw:
        for i in range(100000):
            frame = rng.integers(86, 120, (64, 64), dtype=numpy.uint16)
            if i == 7310:
                frame[25, 25] = 100
            tw.write(
                frame,
                contiguous=True,
                description=desc if i == 0 else None,
                metadata=None,
            )
    print(f' {dest.stat().st_size} bytes')


def main() -> int:
    TIFFFILE_DIR.mkdir(parents=True, exist_ok=True)
    print(f'Generating test data in {TIFFFILE_DIR}\n')

    # Quick files first
    generate_rgb_tif()
    generate_rgb_u1_npy()
    generate_gray_f4_tif()
    generate_sgi_depth_tif()
    generate_generic_series_tif()
    generate_rrggbb_ij_tif()

    generate_temp_sequence_tifs()

    # Large files (slow)
    generate_chart_bl_tif()
    generate_movie_tif()

    # Very large file — only generate if SKIP_LARGE is not set
    if '--large' in sys.argv:
        generate_100000_pages_tif()
    else:
        print('  skip 100000_pages.tif (use --large to generate)')

    print('\nDone.')
    print('\nFiles NOT generated (require real-world data or special tools):')
    print('  - MB231paxgfp_060214.lzw.lsm (real Zeiss microscopy data)')
    print('  - test_FileHandle.bin (needs micromanager.tif + exact file sizes)')
    print('  - gimp_*.tiff (generated by separate script with manual TIFF construction)')
    print('  - multiscene_pyramidal.ome.tif (generated by separate script)')
    return 0


if __name__ == '__main__':
    sys.exit(main())
