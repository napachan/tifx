"""Download public test data files for tifffile tests.

Usage: python tests/fetch_public_data.py

Downloads test TIFF files from their upstream open-source repositories
into tests/data/public/<project>/.
"""

from __future__ import annotations

import io
import os
import sys
import zipfile
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import urlopen

HERE = Path(__file__).parent
PUBLIC_DIR = HERE / 'data' / 'public'

# Map: local_path -> download_url
FILES: dict[str, str] = {
    # GDAL (https://github.com/OSGeo/gdal)
    'GDAL/byte_zstd.tif':
        'https://raw.githubusercontent.com/OSGeo/gdal/master/autotest/gcore/data/byte_zstd.tif',
    'GDAL/contig_tiled.tif':
        'https://raw.githubusercontent.com/OSGeo/gdal/master/autotest/gcore/data/contig_tiled.tif',
    'GDAL/separate_tiled.tif':
        'https://raw.githubusercontent.com/OSGeo/gdal/master/autotest/gcore/data/separate_tiled.tif',
    'GDAL/tif_webp.tif':
        'https://raw.githubusercontent.com/OSGeo/gdal/master/autotest/gcore/data/tif_webp.tif',

    # OME (https://downloads.openmicroscopy.org/images/OME-TIFF/2016-06/)
    'OME/bioformats-artificial/4D-series.ome.tiff':
        'https://downloads.openmicroscopy.org/images/OME-TIFF/2016-06/bioformats-artificial/4D-series.ome.tiff',
    'OME/bioformats-artificial/multi-channel.ome.tiff':
        'https://downloads.openmicroscopy.org/images/OME-TIFF/2016-06/bioformats-artificial/multi-channel.ome.tiff',
    'OME/bioformats-artificial/multi-channel-z-series.ome.tiff':
        'https://downloads.openmicroscopy.org/images/OME-TIFF/2016-06/bioformats-artificial/multi-channel-z-series.ome.tiff',
    'OME/bioformats-artificial/multi-channel-time-series.ome.tiff':
        'https://downloads.openmicroscopy.org/images/OME-TIFF/2016-06/bioformats-artificial/multi-channel-time-series.ome.tiff',
    'OME/bioformats-artificial/multi-channel-4D-series.ome.tiff':
        'https://downloads.openmicroscopy.org/images/OME-TIFF/2016-06/bioformats-artificial/multi-channel-4D-series.ome.tiff',
    'OME/bioformats-artificial/single-channel.ome.tif':
        'https://downloads.openmicroscopy.org/images/OME-TIFF/2016-06/bioformats-artificial/single-channel.ome.tif',
    'OME/bioformats-artificial/time-series.ome.tiff':
        'https://downloads.openmicroscopy.org/images/OME-TIFF/2016-06/bioformats-artificial/time-series.ome.tiff',
    'OME/bioformats-artificial/z-series.ome.tif':
        'https://downloads.openmicroscopy.org/images/OME-TIFF/2016-06/bioformats-artificial/z-series.ome.tif',
    'OME/modulo/FLIM-ModuloAlongC.ome.tiff':
        'https://downloads.openmicroscopy.org/images/OME-TIFF/2016-06/modulo/FLIM-ModuloAlongC.ome.tiff',
    'OME/modulo/FLIM-ModuloAlongT-TSCPC.ome.tiff':
        'https://downloads.openmicroscopy.org/images/OME-TIFF/2016-06/modulo/FLIM-ModuloAlongT-TSCPC.ome.tiff',
    'OME/modulo/LAMBDA-ModuloAlongZ-ModuloAlongT.ome.tiff':
        'https://downloads.openmicroscopy.org/images/OME-TIFF/2016-06/modulo/LAMBDA-ModuloAlongZ-ModuloAlongT.ome.tiff',
    'OME/modulo/SPIM-ModuloAlongZ.ome.tiff':
        'https://downloads.openmicroscopy.org/images/OME-TIFF/2016-06/modulo/SPIM-ModuloAlongZ.ome.tiff',
    'OME/multifile/multifile-Z1.ome.tiff':
        'https://downloads.openmicroscopy.org/images/OME-TIFF/2016-06/companion/multifile-Z1.ome.tiff',
    'OME/tubhiswt-4D/tubhiswt_C0_TP10.ome.tif':
        'https://downloads.openmicroscopy.org/images/OME-TIFF/2016-06/tubhiswt-4D/tubhiswt_C0_TP10.ome.tif',
}

# tubhiswt-4D companion files (86 total: 2 channels x 43 timepoints)
_OME_TUBHISWT_BASE = 'https://downloads.openmicroscopy.org/images/OME-TIFF/2016-06/tubhiswt-4D'
for _c in range(2):
    for _tp in range(43):
        _name = f'tubhiswt_C{_c}_TP{_tp}.ome.tif'
        _key = f'OME/tubhiswt-4D/{_name}'
        if _key not in FILES:
            FILES[_key] = f'{_OME_TUBHISWT_BASE}/{_name}'

FILES.update({
    # Tiff-Library-4J (https://github.com/EasyinnovaSL/Tiff-Library-4J)
    'Tiff-Library-4J/IFD struct/Circular E.tif':
        'https://github.com/EasyinnovaSL/Tiff-Library-4J/raw/master/src/test/resources/IFD%20struct/Circular%20E.tif',
    'Tiff-Library-4J/IFD struct/SubIFDs array E.tif':
        'https://github.com/EasyinnovaSL/Tiff-Library-4J/raw/master/src/test/resources/IFD%20struct/SubIFDs%20array%20E.tif',

    # imagecodecs (https://github.com/cgohlke/imagecodecs)
    'imagecodecs/gray.u1.tif':
        'https://raw.githubusercontent.com/cgohlke/imagecodecs/master/tests/data/tiff/gray.u1.tif',
    # imagecodecs-samples (https://github.com/cgohlke/imagecodecs-samples)
    'imagecodecs/rgb.u2.lerc.tif':
        'https://raw.githubusercontent.com/cgohlke/imagecodecs-samples/main/images/rgb_u2_lerc.tif',

    # imageio (https://github.com/imageio/test_images)
    'imageio/multipage_rgb.tif':
        'https://raw.githubusercontent.com/imageio/test_images/main/multipage_rgb.tif',

    # juicypixels (https://github.com/Twinside/Juicy.Pixels)
    'juicypixels/caspian.tif':
        'https://raw.githubusercontent.com/Twinside/Juicy.Pixels/master/tests/tiff/caspian.tif',
    'juicypixels/cramps-tile.tif':
        'https://raw.githubusercontent.com/Twinside/Juicy.Pixels/master/tests/tiff/cramps-tile.tif',
    'juicypixels/cramps.tif':
        'https://raw.githubusercontent.com/Twinside/Juicy.Pixels/master/tests/tiff/cramps.tif',
    'juicypixels/jello.tif':
        'https://raw.githubusercontent.com/Twinside/Juicy.Pixels/master/tests/tiff/jello.tif',
    'juicypixels/oxford.tif':
        'https://raw.githubusercontent.com/Twinside/Juicy.Pixels/master/tests/tiff/oxford.tif',
    'juicypixels/quad-tile.tif':
        'https://raw.githubusercontent.com/Twinside/Juicy.Pixels/master/tests/tiff/quad-tile.tif',
    'juicypixels/strike.tif':
        'https://raw.githubusercontent.com/Twinside/Juicy.Pixels/master/tests/tiff/strike.tif',

    # libtiff (https://gitlab.com/libtiff/libtiff)
    'libtiff/quad-lzw-compat.tiff':
        'https://raw.githubusercontent.com/libsdl-org/libtiff/master/test/images/quad-lzw-compat.tiff',

    # pillow (https://github.com/python-pillow/Pillow)
    'pillow/tiff_gray_2_4_bpp/hopper2.tif':
        'https://raw.githubusercontent.com/python-pillow/Pillow/main/Tests/images/tiff_gray_2_4_bpp/hopper2.tif',
    'pillow/tiff_gray_2_4_bpp/hopper2I.tif':
        'https://raw.githubusercontent.com/python-pillow/Pillow/main/Tests/images/tiff_gray_2_4_bpp/hopper2I.tif',
    'pillow/tiff_gray_2_4_bpp/hopper2IR.tif':
        'https://raw.githubusercontent.com/python-pillow/Pillow/main/Tests/images/tiff_gray_2_4_bpp/hopper2IR.tif',
    'pillow/tiff_gray_2_4_bpp/hopper2R.tif':
        'https://raw.githubusercontent.com/python-pillow/Pillow/main/Tests/images/tiff_gray_2_4_bpp/hopper2R.tif',
    'pillow/tiff_gray_2_4_bpp/hopper4.tif':
        'https://raw.githubusercontent.com/python-pillow/Pillow/main/Tests/images/tiff_gray_2_4_bpp/hopper4.tif',
    'pillow/tiff_gray_2_4_bpp/hopper4I.tif':
        'https://raw.githubusercontent.com/python-pillow/Pillow/main/Tests/images/tiff_gray_2_4_bpp/hopper4I.tif',
    'pillow/tiff_gray_2_4_bpp/hopper4IR.tif':
        'https://raw.githubusercontent.com/python-pillow/Pillow/main/Tests/images/tiff_gray_2_4_bpp/hopper4IR.tif',
    'pillow/tiff_gray_2_4_bpp/hopper4R.tif':
        'https://raw.githubusercontent.com/python-pillow/Pillow/main/Tests/images/tiff_gray_2_4_bpp/hopper4R.tif',

    # twelvemonkeys (https://github.com/haraldk/TwelveMonkeys)
    'twelvemonkeys/bigtiff/BigTIFFSubIFD4.tif':
        'https://raw.githubusercontent.com/haraldk/TwelveMonkeys/master/imageio/imageio-tiff/src/test/resources/bigtiff/BigTIFFSubIFD4.tif',
    'twelvemonkeys/bigtiff/BigTIFFSubIFD8.tif':
        'https://raw.githubusercontent.com/haraldk/TwelveMonkeys/master/imageio/imageio-tiff/src/test/resources/bigtiff/BigTIFFSubIFD8.tif',
    'twelvemonkeys/tiff/lzw-full-12-bit-table.tif':
        'https://raw.githubusercontent.com/haraldk/TwelveMonkeys/master/imageio/imageio-tiff/src/test/resources/tiff/lzw-full-12-bit-table.tif',
})

# ZIP archives: local_dir -> (url, files_to_extract or None for all)
ZIPS: dict[str, tuple[str, list[str] | None]] = {
    'scif.io': (
        'https://samples.scif.io/2chZT.zip',
        ['2chZT.lsm'],
    ),
}


def download_file(url: str, dest: Path) -> bool:
    """Download a single file. Returns True on success."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        print(f'  skip {dest.relative_to(PUBLIC_DIR)} (exists)')
        return True
    try:
        print(f'  fetch {dest.relative_to(PUBLIC_DIR)} ...',
              end='', flush=True)
        with urlopen(url, timeout=30) as resp:
            data = resp.read()
        dest.write_bytes(data)
        print(f' {len(data)} bytes')
        return True
    except (HTTPError, URLError, TimeoutError, OSError) as exc:
        print(f' FAILED: {exc}')
        return False


def download_zip(
    url: str, dest_dir: Path, files: list[str] | None
) -> bool:
    """Download and extract files from a ZIP archive."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    if files and all((dest_dir / f).exists() for f in files):
        for f in files:
            print(f'  skip {(dest_dir / f).relative_to(PUBLIC_DIR)} (exists)')
        return True
    try:
        print(f'  fetch {url} ...', end='', flush=True)
        with urlopen(url, timeout=60) as resp:
            data = resp.read()
        print(f' {len(data)} bytes')
        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            members = files or zf.namelist()
            for m in members:
                target = dest_dir / m
                if target.exists():
                    print(f'  skip {target.relative_to(PUBLIC_DIR)} (exists)')
                    continue
                print(f'  extract {target.relative_to(PUBLIC_DIR)}')
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(zf.read(m))
        return True
    except (HTTPError, URLError, TimeoutError, OSError) as exc:
        print(f' FAILED: {exc}')
        return False


def main() -> int:
    ok = 0
    fail = 0

    print(f'Downloading public test data to {PUBLIC_DIR}\n')

    # Individual files
    for local_path, url in FILES.items():
        dest = PUBLIC_DIR / local_path.replace('/', os.sep)
        if download_file(url, dest):
            ok += 1
        else:
            fail += 1

    # ZIP archives
    for local_dir, (url, files) in ZIPS.items():
        dest_dir = PUBLIC_DIR / local_dir.replace('/', os.sep)
        if download_zip(url, dest_dir, files):
            ok += 1
        else:
            fail += 1

    print(f'\nDone: {ok} succeeded, {fail} failed')
    return 1 if fail else 0


if __name__ == '__main__':
    sys.exit(main())
