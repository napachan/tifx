# New Features in tifffile (Fork)

This document describes all features added to this fork of the tifffile library
beyond the original upstream monolithic `tifffile.py` (~24,000 lines). The fork
introduces a C++ nanobind extension for performance, a camera calibration
module, GPU acceleration support, mmap-based file I/O optimizations, and a
complete modular architecture that decomposes the original monolith into focused
single-responsibility modules.

---

## Table of Contents

1. [C++ Nanobind Extension (Performance)](#1-c-nanobind-extension-performance)
2. [Camera Calibration Module](#2-camera-calibration-module)
3. [GPU Acceleration Module](#3-gpu-acceleration-module)
4. [File I/O with mmap Optimization](#4-file-io-with-mmap-optimization)
5. [Modular Architecture](#5-modular-architecture)
6. [Series Parser Plugin System](#6-series-parser-plugin-system)
7. [Module Summary Table](#7-module-summary-table)

---

## 1. C++ Nanobind Extension (Performance)

### Overview

A C++17 extension module (`_tifffile_ext`) built with nanobind provides
order-of-magnitude speedups for the hot paths in TIFF file parsing: IFD chain
scanning, tag extraction, and series grouping. The extension is entirely
optional -- all functionality falls back gracefully to pure Python when the
extension is not available or is explicitly disabled.

### Build System

The extension uses **scikit-build-core** as the build backend with **CMake** and
**nanobind** for Python bindings.

- `pyproject.toml` declares `scikit-build-core>=0.10` and `nanobind>=2.4.0` as
  build requirements.
- `CMakeLists.txt` (53 lines) defines the `_tifffile_ext` nanobind module
  target, collecting source files from `tifffile/src/cpp/` and
  `tifffile/src/bindings/`.
- Build command: `.venv/Scripts/python.exe -m pip install . --no-build-isolation --no-cache-dir`
- Requires C++17 and Python 3.11+.
- The built `.pyd`/`.so` is installed into the `tifffile/` package directory.

### Source Organization (1,963 lines total)

```
tifffile/src/
  include/tifffile/
    common.hpp       (114 lines) - Byte order, endian helpers, TagValue variant
    enums.hpp        (221 lines) - C++ enum classes mirroring Python IntEnums
    tiff_format.hpp  (109 lines) - TiffFormat struct with static factories
    file_reader.hpp  ( 54 lines) - Memory-mapped file reader interface
    ifd_entry.hpp    ( 32 lines) - IfdEntry, IfdParseResult, ScanChainResult structs
    ifd_parser.hpp   ( 41 lines) - IfdParser static methods
    selection.hpp    ( 45 lines) - AxisSelection, SelectionResult, TileLayout structs
  cpp/
    common.cpp       (  3 lines) - Placeholder for future non-inline implementations
    ifd_parser.cpp   (211 lines) - IFD parsing and chain scanning implementation
    file_reader.cpp  (195 lines) - Cross-platform mmap file reader (Windows + POSIX)
    selection.cpp    (191 lines) - N-D selection index and tile position computation
  bindings/
    module.cpp         ( 26 lines) - Module definition, leak warnings suppression
    bind_enums.cpp     (146 lines) - 11 enum classes exposed to Python
    bind_format.cpp    ( 40 lines) - CppTiffFormat class binding
    bind_ifd.cpp       (103 lines) - parse_ifd, parse_ifd_filtered, scan_ifd_chain
    bind_file_reader.cpp (243 lines) - CppFileReader, scan_ifd_chain_file, bulk_extract_tag_values
    bind_selection.cpp (189 lines) - selection_to_page_indices, compute_segment_positions
```

### Exposed Python API

All functions are accessible from `tifffile._tifffile_ext`.

#### `CppTiffFormat`

TIFF format descriptor with static factories and derived properties.

- `CppTiffFormat.classic_le()` / `classic_be()` / `big_le()` / `big_be()` / `ndpi_le()`
  -- static factories for the 5 standard TIFF format variants.
- Read-only properties: `version`, `byteorder`, `offsetsize`, `tagnosize`,
  `tagsize`, `tagoffsetthreshold`, `is_bigtiff`, `is_ndpi`.
- Format string properties: `offsetformat`, `tagnoformat`, `tagformat1`, `tagformat2`.
- Hashable and equality-comparable for use as dict keys.

A module-level `_CPP_FORMATS` dict caches the 4 main variants (classic LE/BE,
big LE/BE) to avoid per-call allocation.

#### `CppFileReader`

Cross-platform memory-mapped file reader.

```python
reader = CppFileReader("/path/to/file.tif")
reader.size       # file size in bytes
reader.is_mmap    # always True for file-backed readers
data = reader.read(offset, count)  # returns bytes (copy)
```

- On Windows, uses `CreateFileA` + `CreateFileMappingA` + `MapViewOfFile`.
- On POSIX, uses `open` + `mmap` with `MAP_PRIVATE`.
- Non-copyable, movable. Proper RAII cleanup.

#### `parse_ifd(data, tagno, file_offset, fmt)`

Parse one IFD's tag entries from raw bytes. Returns a list of `IfdEntry`
objects, each containing: `code`, `dtype`, `count`, `valueoffset`,
`tag_file_offset`, `is_inline`, `inline_size`, and `get_inline_bytes()`.

#### `parse_ifd_filtered(data, tagno, file_offset, fmt, codes)`

Same as `parse_ifd` but returns only entries whose tag code is in the provided
`codes` set.

#### `scan_ifd_chain(file_data, first_offset, fmt, max_pages=0)`

Scan an IFD chain from an in-memory buffer. Returns packed `uint64_t` offsets
as `bytes`.

#### `scan_ifd_chain_file(reader, first_offset, fmt, max_pages=0)`

Scan IFD chain using a `CppFileReader` (mmap-based). **Releases the GIL**
during the scan. Returns a tuple of:
`(offsets_packed_bytes, next_page_offset, circular_ifd_index, circular_target_offset)`.

Circular IFD detection uses a two-tier strategy:
1. Quick per-iteration check: self-reference or loop to first IFD.
2. Periodic full check at page 100, then every 4096 pages.

#### `bulk_extract_tag_values(reader, ifd_offsets_packed, fmt, codes)`

Extract specific tag values for ALL pages in a single GIL-free pass over the
mmap'd file. Returns a dict mapping `tag_code -> list[tuple_of_values_per_page]`.
Pages missing a tag get an empty tuple.

This is the key function enabling the 41x series parsing speedup: instead of
creating TiffPage objects for every page, the GenericSeriesParser extracts only
the hash-relevant tags in bulk, computes hashes from raw values, and creates
lightweight TiffFrame objects for non-keyframe pages.

#### `selection_to_page_indices(frame_shape, frame_sel, frame_axes)`

Compute flat page indices from an N-D frame selection. Takes Python `int` or
`slice` objects per axis character in `frame_sel` dict, resolves them against
`frame_shape`, and returns `(flat_indices_ndarray_uint64, output_shape_tuple)`.
Releases the GIL for selections producing more than 1000 indices.

#### `compute_segment_positions(n_segments, stshape, imshape, is_tiled)`

Pre-compute all segment (tile/strip) positions and shapes for the decode
closure. Returns `(positions_ndarray_int32, shapes_ndarray_int32)` as flat
arrays. Releases the GIL. Used by `build_decode_closure()` in `decoders.py`.

### Performance Results

| Benchmark | C++ | Python | Speedup |
|---|---|---|---|
| Series parsing: 30K uniform pages | 46 ms | 1,872 ms | **41x** |
| `len(pages)` on 30K pages | 3.6 ms | 20 ms | **5.6x** |
| `len(pages)` on 100K pages | 14 ms | 73 ms | **5.1x** |
| Iterate 30K pages (after series) | 0.8 ms | 1,574 ms | **1,968x** |

### GIL Release Strategy

The following operations release the GIL:
- `scan_ifd_chain_file` -- IFD chain scanning
- `bulk_extract_tag_values` -- bulk tag extraction loop
- `selection_to_page_indices` -- for large selections (>1000 indices)
- `compute_segment_positions` -- tile/strip position pre-computation

### Fallback Behavior

- Setting the environment variable `TIFFFILE_NO_CPP=1` disables the C++
  extension entirely.
- If the `.pyd`/`.so` import fails, all C++ references are set to `None` and
  the pure Python paths are used.
- NDPI files always fall back to the Python parsing path (due to their
  non-standard 64-bit offset layout).
- The `__init__.py` exports `_HAS_CPP` to indicate extension availability.

### Enum Bindings

The extension exposes 11 C++ enum classes (`CppDatatype`, `CppCompression`,
`CppPredictor`, `CppPhotometric`, `CppPlanarconfig`, `CppSampleformat`,
`CppFillorder`, `CppOrientation`, `CppFiletype`, `CppExtrasample`,
`CppResunit`) used internally. Python's `enums.py` remains the canonical
source.

---

## 2. Camera Calibration Module

### Overview

`tifffile/camera.py` (826 lines) is an entirely new module for storing and
retrieving full computer-vision camera calibration data in TIFF files. There is
no existing TIFF standard for this -- OME-TIFF provides only 5 numeric fields
per plane and has no concept of lens distortion or rotation quaternions.

The module uses 7 private TIFF tags (65201-65207) with a COLMAP-compatible
convention: same model IDs, same parameter order, same quaternion layout.

### Tag Layout

All tags are written to the first IFD only (`writeonce=True`).

| Tag | TIFF Type | Count | Contents |
|---|---|---|---|
| 65201 | LONG (4) | 3 | `[model_id, image_width, image_height]` |
| 65202 | DOUBLE (12) | 4 | `[fx, fy, cx, cy]` intrinsics |
| 65203 | DOUBLE (12) | 12 | distortion coefficients, zero-padded |
| 65204 | DOUBLE (12) | N * 7 | per-frame extrinsics `[qx, qy, qz, qw, tx, ty, tz]` |
| 65205 | BYTE (1) | variable | per-frame source filenames, UTF-8 null-separated |
| 65206 | DOUBLE (12) | N | per-frame Unix epoch timestamps |
| 65207 | DOUBLE (12) | N * 3 | per-frame GPS `[latitude, longitude, altitude]` |

### CameraModel Enum

18 COLMAP-compatible camera models (IntEnum):

| ID | Name | Focal | Distortion Params |
|---|---|---|---|
| 0 | SIMPLE_PINHOLE | single | 0 |
| 1 | PINHOLE | dual | 0 |
| 2 | SIMPLE_RADIAL | single | 1 |
| 3 | RADIAL | single | 2 |
| 4 | OPENCV | dual | 4 |
| 5 | OPENCV_FISHEYE | dual | 4 |
| 6 | FULL_OPENCV | dual | 8 |
| 7 | FOV | dual | 1 |
| 8 | SIMPLE_RADIAL_FISHEYE | single | 1 |
| 9 | RADIAL_FISHEYE | single | 2 |
| 10 | THIN_PRISM_FISHEYE | dual | 8 |
| 11 | RADTAN_THIN_PRISM_FISHEYE | dual | 12 |
| 12 | SIMPLE_DIVISION | single | 1 |
| 13 | DIVISION | dual | 1 |
| 14 | SIMPLE_FISHEYE | single | 0 |
| 15 | FISHEYE | dual | 0 |
| 16 | EQUIRECTANGULAR | dual | 0 |
| 17 | CUBEMAP | dual | 0 |

IDs 0-15 match COLMAP exactly. IDs 16-17 are extensions for panoramic imaging.

### CameraData Dataclass

Returned by `read_camera()`. Fields:

- `model: CameraModel` -- camera model enum
- `width: int`, `height: int` -- image dimensions
- `intrinsics: ndarray (4,)` -- `[fx, fy, cx, cy]` as float64
- `distortion: ndarray (12,)` -- zero-padded distortion coefficients
- `extrinsics: ndarray (N, 7)` -- per-frame `[qx, qy, qz, qw, tx, ty, tz]`
- `filenames: list[str] | None` -- per-frame source filenames
- `timestamps: ndarray (N,) | None` -- per-frame Unix epoch timestamps
- `gps: ndarray (N, 3) | None` -- per-frame GPS `[lat, lon, alt]`

Properties: `num_frames`, `fx`, `fy`, `cx`, `cy`, `K` (3x3 intrinsic matrix).

Methods:

- `rotation_matrices()` -- returns `(N, 3, 3)` rotation matrices from quaternions
- `translation_vectors()` -- returns `(N, 3)` translation vectors
- `cam_from_world_matrix(frame)` -- 4x4 cam-from-world transform
- `world_from_cam_matrix(frame)` -- 4x4 world-from-cam transform (inverse)
- `camera_positions()` -- world-space camera positions as `(N, 3)`

### Bridge Methods

#### pycolmap

- `to_pycolmap()` -- returns `(pycolmap.Camera, list[pycolmap.Rigid3d])`.
  Converts quaternion order from `[qx, qy, qz, qw]` to pycolmap's
  `[qw, qx, qy, qz]`.
- `CameraData.from_pycolmap(camera, poses)` -- creates `CameraData` from
  pycolmap objects.

#### OpenCV

- `to_opencv()` -- returns `(K_3x3, dist_coeffs)` for `cv2.undistort`.
- `CameraData.from_opencv(K, dist_coeffs, width, height)` -- creates
  `CameraData` from OpenCV calibration results. Infers model from
  `len(dist_coeffs)`: 0 -> PINHOLE, <=4 -> OPENCV, >4 -> FULL_OPENCV.

#### transforms.json (NeRF/Nerfstudio)

- `to_transforms_json(image_paths=None)` -- returns a `transforms.json` dict.
  Converts cam_from_world (COLMAP) to world_from_cam (NeRF), flipping Y/Z axes
  (OpenCV to OpenGL convention).
- `CameraData.from_transforms_json(data)` -- creates `CameraData` from
  `transforms.json` file path, JSON string, or dict. Converts world_from_cam
  back to cam_from_world, handling both `fl_x`/`fl_y` and `camera_angle_x`
  focal length specifications.

### Key Functions

- `camera_extratags(model, width, height, intrinsics, distortion, extrinsics, *, filenames, timestamps, gps)` --
  builds extratag tuples for `TiffWriter.write()`.
- `read_camera(tif)` -- reads `CameraData` from a `TiffFile` or `TiffPage`.

### Usage Example

```python
import numpy as np
from tifffile import TiffWriter, TiffFile
from tifffile.camera import CameraModel, camera_extratags, read_camera

# Write
images = np.random.randint(0, 255, (10, 480, 640), dtype='uint8')
poses = np.zeros((10, 7), dtype='float64')
poses[:, 3] = 1.0  # identity rotation
tags = camera_extratags(
    model=CameraModel.OPENCV,
    width=640, height=480,
    intrinsics=[500.0, 500.0, 320.0, 240.0],
    distortion=[-0.1, 0.01, 0.001, -0.001],
    extrinsics=poses,
)
with TiffWriter('calibrated.tif') as tw:
    tw.write(images, extratags=tags)

# Read
with TiffFile('calibrated.tif') as tif:
    cam = read_camera(tif)
    print(cam.model, cam.fx, cam.num_frames)
    K, dist = cam.to_opencv()
```

---

## 3. GPU Acceleration Module

### Overview

`tifffile/gpu.py` (694 lines) provides GPU acceleration support for reading and
writing TIFF image data using PyTorch tensors, with optional NVIDIA library
integration (nvCOMP, nvImageCodec, kvikio). All GPU dependencies are
lazy-imported so the module has zero cost when not used.

### Key Components

#### Device Handling

- `parse_device(device)` -- parses a device string (e.g., `'cuda'`, `'cuda:0'`)
  into a `torch.device`. Returns `None` if device is `None`.

#### NumPy-to-Tensor Conversion

- `numpy_to_tensor(array, device)` -- converts a NumPy array to a
  `torch.Tensor` on the specified device. Uses pinned memory for efficient
  async CPU-to-GPU DMA transfer. On torch 2.3+, unsigned dtypes (`uint16`,
  `uint32`, `uint64`) are supported natively via zero-copy `from_numpy`. On
  older torch versions, unsigned arrays are upcast to the next wider signed
  type.

- `tensor_to_numpy(tensor)` -- converts a torch tensor to NumPy (D2H transfer
  if CUDA).

#### Direct File-to-GPU Reading

- `read_to_gpu(fh, device, typecode, count, offset, shape)` -- reads raw bytes
  from file directly into a GPU tensor. Three paths:
  1. **kvikio GPUDirect Storage** (Linux + CUDA): zero-copy file-to-GPU via
     `CuFile`. Handles byte-swapping on GPU.
  2. **Pinned memory** (Windows + fallback): reads into CPU pinned memory, then
     async DMA to GPU.
  3. **Non-CUDA**: reads as NumPy, converts via `numpy_to_tensor`.

#### GPU Codec Registry (Decode)

`GpuCodecRegistry` maps TIFF compression IDs to GPU decode functions. Probed
lazily at first use.

| Compression | GPU Library | Function |
|---|---|---|
| Zstd (50000) | nvCOMP (kvikio) | `_gpu_decompress_zstd` |
| JPEG (6, 7, 34892) | nvImageCodec | `_gpu_decompress_jpeg` |
| JPEG2000 (33003-33005) | nvImageCodec | `_gpu_decompress_jpeg2000` |

#### GPU Predictor Operations

- `gpu_cumsum_predictor(data, axis=-1)` -- reverses horizontal differencing
  predictor (decode direction) on GPU via `torch.cumsum`.
- `gpu_delta_encode(tensor, axis=-1)` -- applies horizontal differencing
  predictor (encode direction) on GPU. Uses wrapping (modular) subtraction for
  unsigned dtypes by casting to wider signed types.

#### GPU Encoder Registry (Write)

`GpuEncoderRegistry` maps TIFF compression IDs to GPU encode algorithms.
Supports `nvidia.nvcomp` (official package, Windows + Linux) and legacy
`kvikio` (Linux only).

- Minimum plane size threshold: 48 MB (below this, CPU is faster).
- Supported: Zstd (50000, 34926).

- `gpu_compress(data, algorithm)` -- compresses a CUDA tensor using nvCOMP.
  Returns standard-format compressed bytes on CPU.
- `gpu_encode_planes(tensor, algorithm, storedshape, predictortag, compressionaxis)` --
  yields compressed bytes for each plane, encoded on GPU. Forces one strip per
  plane for efficient GPU kernel utilization.

#### Utility Functions

- `_is_cuda_tensor(data)` -- detects CUDA tensors (torch or cupy).
- `_torch_dtype_to_numpy(torch_dtype)` -- converts torch dtype to numpy dtype.
- `_gpu_byteswap(tensor, itemsize)` -- in-place byte-swap on GPU.

---

## 4. File I/O with mmap Optimization

### Overview

`tifffile/fileio.py` (1,455 lines) contains the `FileHandle` class and related
I/O utilities, extracted from the monolith. The major enhancement is an mmap
zero-copy path for `read_segments()` that provides 2-3x faster reads for tiled
images.

### FileHandle Enhancements

#### mmap Zero-Copy `read_segments()`

The `read_segments()` method detects when an mmap view is available and uses
zero-copy `memoryview` slicing instead of seek+read syscalls. This eliminates
lock acquisition overhead and avoids copying data.

When the mmap path is active:
- No lock is needed (mmap is read-only, no seek state to protect).
- Segments are returned as `memoryview` slices pointing directly into the
  mapped file.
- Batching respects `buffersize` for non-flat mode.

The `read_array()` method also uses the mmap view when available, falling back
to the standard `readinto` path when needed.

#### Windows mmap Bug Workaround

A critical Windows bug was discovered and fixed: calling `mmap.mmap(fh.fileno(), ...)`
on the same file descriptor corrupts subsequent `seek+read` operations -- data
is correct up to the 8KB page boundary, then wraps to file offset 0.

The fix: `_get_mmap_view()` opens a **separate file descriptor** via `os.open()`
for the mmap, keeping the original `_fh` untouched. The separate fd is stored
in `_mmap_fd` and properly cleaned up in `close()`.

```python
# In _get_mmap_view():
fd = os.open(self._file, os.O_RDONLY | getattr(os, 'O_BINARY', 0))
self._mmap_fd = fd
self._mmap = mmap.mmap(fd, 0, access=mmap.ACCESS_READ)
self._mmap_view = memoryview(self._mmap)
```

Constraints: mmap is only used for on-disk read-only files with `_offset == 0`.

### StoredShape

A `Sequence[int]` subclass (lines 186-419) representing the normalized 6D shape
of image data: `(frames, separate_samples, depth, length, width, contig_samples)`.

Properties: `size`, `samples`, `photometric_samples`, `shape`, `page_shape`,
`page_size`, `squeezed`, `is_valid`, `is_planar`, `planarconfig`.

Used throughout the write path and GPU encoding for consistent shape handling.

### Other Classes

- **NullContext** -- a null context manager used as a dummy reentrant lock when
  thread safety is not needed. Avoids `threading.RLock` overhead.
- **Timer** -- a stopwatch context manager for timing execution. Supports
  `start()`, `stop()`, `print()`, and context manager protocol.
- **FileCache** -- keeps `FileHandle` instances open with LRU eviction.
  Supports `open()`, `close()`, `read()`, `write()`, `clear()` operations with
  thread-safe locking.

---

## 5. Modular Architecture

### Overview

The original monolithic `tifffile.py` (~24,394 lines) has been decomposed into
focused single-responsibility modules. The main file is now ~10,669 lines, with
~19,443 lines extracted into dedicated modules.

All modules use `from __future__ import annotations` and `TYPE_CHECKING` guards
for type-only imports. Circular imports are avoided via lazy imports inside
method bodies.

### `tifffile/enums.py` (367 lines)

13 IntEnum classes for TIFF constants:

- `DATATYPE` -- tag data types (BYTE through IFD8, 18 values)
- `COMPRESSION` -- compression schemes (70+ values including modern codecs:
  ZSTD, WEBP, JPEGXL, EER)
- `PREDICTOR` -- differencing predictors (7 values including x2/x4 variants)
- `PHOTOMETRIC` -- color space interpretations (16 values)
- `PLANARCONFIG` -- CONTIG / SEPARATE
- `SAMPLEFORMAT` -- UINT / INT / IEEEFP / VOID / COMPLEX variants
- `FILLORDER`, `ORIENTATION`, `FILETYPE`, `OFILETYPE`, `EXTRASAMPLE`,
  `RESUNIT`, `CHUNKMODE`

### `tifffile/tags.py` (1,251 lines)

- **TiffTag** -- represents a single TIFF tag with code, dtype, count, value,
  and offset. Includes `from_ifd_entry()` classmethod for constructing from C++
  `IfdEntry` objects.
- **TiffTags** -- dict-like container of `TiffTag` objects. Supports access by
  code or name. Used as the `tags` attribute of `TiffPage`.
- **TiffTagRegistry** -- bidirectional registry mapping tag codes to names and
  vice versa. Supports multiple names per code and multiple codes per name.

### `tifffile/codecs.py` (434 lines)

- **TiffFormat** -- TIFF format properties (version, byteorder, offset/tag
  sizes, struct format strings). Includes `is_bigtiff`, `is_ndpi` properties
  and `read_tagno()`/`read_offset()` methods.
- **CompressionCodec** -- maps `COMPRESSION` enum values to imagecodecs
  encode/decode functions.
- **PredictorCodec** -- maps `PREDICTOR` enum values to imagecodecs
  encode/decode functions.

### `tifffile/page.py` (3,699 lines)

- **TiffPage** -- full IFD representation with all tag parsing, property
  computation, and `asarray()` for reading image data.
  - C++ fast path: when `_HAS_CPP` is true and the file is not NDPI, IFD tag
    parsing uses `_cpp_parse_ifd()` instead of the Python struct-based parser.
  - Integrates with `_CppFileReader` and `_cpp_scan_ifd_chain_file` for chain
    scanning.
- **TiffFrame** -- lightweight proxy for pages in a uniform series. Shares the
  keyframe's metadata, stores only its own `dataoffsets` and `databytecounts`.
  Note: TiffFrame is NOT a subclass of TiffPage.
- **TiffPages** -- list-like container managing lazy loading of pages. Supports
  `set_keyframe()`, `useframes` mode, caching, and the fast `__iter__` path
  (yields from `_pages` when fully cached).

### `tifffile/decoders.py` (621 lines)

`build_decode_closure(page)` -- factory function that returns a decode closure
for a `TiffPage`. The closure decodes individual segments (strips or tiles) and
returns the decoded data, its position in the image array, and its shape.

Uses `compute_segment_positions` from the C++ extension when available for
pre-computing tile/strip positions.

### `tifffile/metadata.py` (3,437 lines)

Contains `OmeXml` class and all `read_*` metadata parsing functions:

- `OmeXml` -- OME-XML metadata parser with structured access.
- Metadata readers for: STK, FluoView, MicroManager, ScanImage, ImageJ, NIH,
  shaped descriptions, Pilatus, GDAL, LSM, SVS, SCN, NDPI, BIF, and many more.
- `jpeg_decode_colorspace()` -- determines JPEG color space for decoding.
- Uses a `_TIFFProxy` lazy singleton to avoid circular imports with
  `tifffile.py`.

### `tifffile/utils.py` (1,763 lines)

Shared utility functions and the `TiffFileError` exception class:

- `TiffFileError` -- base exception for invalid TIFF structures.
- Array/type utilities: `identityfunc`, `nullfunc`, `sequence`, `product`,
  `astype`, `unpack_rgb`.
- String utilities: `snipstr`, `stripnull`, `bytes2str`, `bytestr`,
  `natural_sorted`, `pformat`, `indent`.
- Shape utilities: `squeeze_axes`, `reshape_axes`, `check_shape`.
- Date parsing: `strptime`.
- XML: `xml2dict`.
- Logging: `logger()` returns a module-level logger.

---

## 6. Series Parser Plugin System

### Overview

Series parsing has been refactored from a monolithic method into a plugin-based
registry system. The base classes live in `tifffile/series.py` (162 lines), and
format-specific parsers live in `tifffile/series_parsers/` (7 files, ~2,616
lines total).

### SeriesParser Base Class

```python
class SeriesParser:
    kind: str = ''              # matches is_<kind> flag on TiffFile
    def can_parse(tiff) -> bool # check if parser applies
    def on_failure(tiff) -> bool # handle failure, return True to try next
    def parse(tiff) -> list[TiffPageSeries] | None
```

### SeriesParserRegistry

Priority-ordered registry that tries parsers in order:

```python
registry = SeriesParserRegistry()
registry.register(parser)        # format-specific
registry.register_generic(parser) # fallback
series = registry.parse(tiff)    # dispatch
```

`get_default_registry()` lazily creates the singleton registry with 21
format-specific parsers and 1 generic fallback, in the correct priority order.

### Format-Specific Parsers

| File | Parsers | Lines |
|---|---|---|
| `generic.py` | `GenericSeriesParser`, `ShapedSeriesParser`, `UniformSeriesParser` | 618 |
| `ome.py` | `OmeSeriesParser` | 470 |
| `imagej.py` | `ImageJSeriesParser` | 148 |
| `lsm.py` | `LsmSeriesParser` | 68 |
| `microscopy.py` | `FluoViewSeriesParser`, `StkSeriesParser`, `NdtiffSeriesParser`, `MmstackSeriesParser`, `ScanImageSeriesParser`, `NihSeriesParser` | 522 |
| `pathology.py` | `SvsSeriesParser`, `ScnSeriesParser`, `QpiSeriesParser`, `NdpiSeriesParser`, `BifSeriesParser`, `PhilipsSeriesParser` | 569 |
| `other.py` | `AvsSeriesParser`, `EerSeriesParser`, `MdgelSeriesParser`, `SisSeriesParser` | 171 |

### GenericSeriesParser._parse_fast()

The `GenericSeriesParser` has a `_parse_fast()` method that uses the C++
extension for bulk tag extraction on files with more than 100 pages:

1. Ensures all IFD offsets are known via `pages._seek(-1)`.
2. Packs all offsets into a `struct`-packed bytes buffer.
3. Calls `_cpp_bulk_extract()` to extract 21 hash-relevant tag codes for all
   pages in a single GIL-free pass.
4. Computes page hashes from raw values using the same hash formula as
   `TiffPage.hash`.
5. Groups pages by hash. Only creates full `TiffPage` for keyframes (first page
   of each group).
6. Creates lightweight `TiffFrame` for remaining pages using bulk-extracted
   `dataoffsets` and `databytecounts`.
7. Falls back to the standard path if SubIFDs are present.

This is responsible for the **41x speedup** on 30K-page files.

### Contiguous mmap Read Path

`tifffile/sequences.py` (2,098 lines) contains `TiffPageSeries` with an
optimized mmap read path (`_mmap_read_contiguous`) for contiguous series:

- Constructs a NumPy array directly from the mmap'd view without any file I/O.
- Supports page selection (slice, range, or fancy indexing).
- Supports combined page + spatial sub-selection (Y slicing, full spatial
  selection).
- Uses multithreaded copy for large strided access (>500 pages) to overlap
  memory latency.
- Handles byte-swapping to native order.

The C++ `selection_to_page_indices` function accelerates N-D page selection
computation in `TiffPageSeries`.

### Public API Surface

`tifffile/__init__.py` re-exports everything from `tifffile.tifffile` via
`from .tifffile import *`. The `__all__` list contains 75 public symbols. The
`_HAS_CPP` flag is available to check extension availability.

---

## 7. Module Summary Table

| Module | Lines | Contents |
|---|---|---|
| `tifffile.py` | 10,669 | Core TiffFile, TiffWriter, TIFF constants |
| `page.py` | 3,699 | TiffPage, TiffFrame, TiffPages with C++ fast paths |
| `metadata.py` | 3,437 | OmeXml, all `read_*` metadata parsers |
| `sequences.py` | 2,098 | TiffPageSeries, FileSequence, TiffSequence, TiledSequence, mmap read |
| `utils.py` | 1,763 | Helper functions, TiffFileError |
| `fileio.py` | 1,455 | FileHandle (mmap), FileCache, NullContext, Timer, StoredShape |
| `tags.py` | 1,251 | TiffTag, TiffTags, TiffTagRegistry |
| `camera.py` | 826 | Camera calibration: CameraModel, CameraData, extratags, bridges |
| `gpu.py` | 694 | GPU acceleration: read/write, codec registries, predictor ops |
| `decoders.py` | 621 | build_decode_closure with C++ segment positions |
| `codecs.py` | 434 | TiffFormat, CompressionCodec, PredictorCodec |
| `enums.py` | 367 | 13 IntEnum classes |
| `series.py` | 162 | SeriesParser base, SeriesParserRegistry |
| `series_parsers/` | 2,616 | 21 format-specific parsers across 7 files |
| **C++ extension** | **1,963** | **nanobind bindings, IFD parser, file reader, selection** |
| **Total** | **~32,055** | |
