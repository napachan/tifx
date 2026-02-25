# Camera Calibration in TIFF Files

Store and retrieve camera intrinsics, distortion coefficients, and per-frame
extrinsics in TIFF tags using a COLMAP-compatible convention.

```python
from tifffile.camera import CameraModel, CameraData, camera_extratags, read_camera
```

## Why

There is no existing TIFF standard for storing full computer-vision camera
calibration (intrinsics + distortion + per-frame extrinsics). OME-TIFF only
provides 5 numeric fields per plane (PositionX/Y/Z, DeltaT, ExposureTime) and
has no concept of lens distortion or rotation quaternions.

This module fills that gap by packing calibration data into four private TIFF
tags in the reusable range (65201-65204). The convention is fully
COLMAP-compatible: same model IDs, same parameter order, same quaternion layout.

## Tag Layout

All four tags are written to the **first IFD only** (`writeonce=True`).
This works with any compression mode since tags are stored in the IFD header,
not in the image data segments.

| Tag   | TIFF Type  | Count | Contents                                     |
|-------|------------|-------|----------------------------------------------|
| 65201 | LONG (4)   | 3     | `[model_id, image_width, image_height]`      |
| 65202 | DOUBLE (12)| 4     | `[fx, fy, cx, cy]`                           |
| 65203 | DOUBLE (12)| 12    | distortion coefficients, zero-padded to 12   |
| 65204 | DOUBLE (12)| N * 7 | per-frame extrinsics, 7 values per frame     |

**Tag 65201 (Camera Model):** Three unsigned 32-bit integers. The model ID
maps to a `CameraModel` enum value (see table below). Width and height are the
image dimensions the calibration was computed for.

**Tag 65202 (Intrinsics):** Four 64-bit floats. For pinhole-type models these
are `[fx, fy, cx, cy]`. Single-focal models (where fx = fy) can pass 3 values
`[f, cx, cy]` to `camera_extratags()` and the function normalizes to
`[f, f, cx, cy]`. For equirectangular: `[h_fov, v_fov, lon_offset,
lat_offset]`. For cubemap: `[face_size, 0, 0, 0]`.

**Tag 65203 (Distortion):** Twelve 64-bit floats. The number of meaningful
coefficients depends on the camera model (see table below). Unused slots are
zero-padded. Fixed length avoids IFD size ambiguity.

**Tag 65204 (Extrinsics):** `N * 7` 64-bit floats, where N is the number of
frames. Each frame has 7 values: `[qx, qy, qz, qw, tx, ty, tz]`.

## Extrinsics Convention

Follows the COLMAP convention:

- **Quaternion order:** `[qx, qy, qz, qw]` (Eigen convention, scalar-last)
- **Transform semantics:** `cam_from_world` &mdash; maps world points to camera
  coordinates: `x_cam = R * x_world + t`
- **Identity pose:** `[0, 0, 0, 1, 0, 0, 0]` (no rotation, no translation)

To convert to `world_from_cam` (camera position in world):

```python
R_w2c = cam.rotation_matrices()       # (N, 3, 3)
t_w2c = cam.translation_vectors()     # (N, 3)
R_c2w = R_w2c.transpose(0, 2, 1)     # inverse rotation
camera_position = -R_c2w @ t_w2c[:, :, None]  # (N, 3, 1)
```

## Camera Models

IDs 0-15 match COLMAP exactly. IDs 16-17 are extensions for panoramic imaging.

| ID | Name                       | Focal  | Distortion Params | Notes                      |
|----|----------------------------|--------|-------------------|----------------------------|
| 0  | SIMPLE_PINHOLE             | single | 0                 | f, cx, cy                  |
| 1  | PINHOLE                    | dual   | 0                 | fx, fy, cx, cy             |
| 2  | SIMPLE_RADIAL              | single | 1                 | + k1                       |
| 3  | RADIAL                     | single | 2                 | + k1, k2                   |
| 4  | OPENCV                     | dual   | 4                 | + k1, k2, p1, p2           |
| 5  | OPENCV_FISHEYE             | dual   | 4                 | + k1, k2, k3, k4           |
| 6  | FULL_OPENCV                | dual   | 8                 | + k1-k6, p1, p2            |
| 7  | FOV                        | dual   | 1                 | + omega                    |
| 8  | SIMPLE_RADIAL_FISHEYE      | single | 1                 | + k1                       |
| 9  | RADIAL_FISHEYE             | single | 2                 | + k1, k2                   |
| 10 | THIN_PRISM_FISHEYE         | dual   | 8                 | + k1-k4, p1, p2, s1, s2   |
| 11 | RADTAN_THIN_PRISM_FISHEYE  | dual   | 12                | + k1-k6, p1, p2, s1-s4    |
| 12 | SIMPLE_DIVISION            | single | 1                 | + lambda                   |
| 13 | DIVISION                   | dual   | 1                 | + lambda                   |
| 14 | SIMPLE_FISHEYE             | single | 0                 |                            |
| 15 | FISHEYE                    | dual   | 0                 |                            |
| 16 | EQUIRECTANGULAR            | dual   | 0                 | intrinsics = FOV + offsets |
| 17 | CUBEMAP                    | dual   | 0                 | intrinsics = [face_size, 0, 0, 0] |

**Single focal** means `fx = fy = f`. When writing, you can pass 3 intrinsics
`[f, cx, cy]` and the function expands to `[f, f, cx, cy]`.

**Dual focal** means `fx` and `fy` are independent. You must pass all 4
intrinsics `[fx, fy, cx, cy]`.

## API Reference

### `camera_extratags()`

Build TIFF extratag tuples for `TiffWriter.write()`.

```python
def camera_extratags(
    model: CameraModel | int,
    width: int,
    height: int,
    intrinsics: Sequence[float],
    distortion: Sequence[float] | None = None,
    extrinsics: numpy.ndarray | Sequence[Sequence[float]] | None = None,
) -> list[tuple[int, int, int, bytes, bool]]
```

**Parameters:**

- `model` &mdash; Camera model ID. Use `CameraModel.OPENCV`, integer `4`, etc.
- `width`, `height` &mdash; Image dimensions in pixels.
- `intrinsics` &mdash; `[fx, fy, cx, cy]` or `[f, cx, cy]` for single-focal
  models.
- `distortion` &mdash; Distortion coefficients. Must have at least as many
  values as the model requires. Zero-padded to 12. Defaults to all zeros.
- `extrinsics` &mdash; `(N, 7)` array of `[qx, qy, qz, qw, tx, ty, tz]`
  per frame. Defaults to a single identity pose.

**Returns:** List of extratag tuples to pass as `extratags=` to
`TiffWriter.write()`.

**Raises:** `ValueError` for wrong intrinsics count, insufficient distortion
params, or wrong extrinsics shape.

### `read_camera()`

Read camera calibration data from a TIFF file.

```python
def read_camera(tif: TiffFile | TiffPage) -> CameraData
```

Accepts either a `TiffFile` (reads from first page) or a `TiffPage` directly.

**Raises:** `KeyError` if camera tags are not present.

### `CameraData`

Dataclass returned by `read_camera()`.

| Field        | Type             | Description                                    |
|--------------|------------------|------------------------------------------------|
| `model`      | `CameraModel`    | Camera model enum                              |
| `width`      | `int`            | Image width in pixels                          |
| `height`     | `int`            | Image height in pixels                         |
| `intrinsics` | `ndarray (4,)`   | `[fx, fy, cx, cy]` as float64                  |
| `distortion` | `ndarray (12,)`  | Distortion coefficients, zero-padded, float64  |
| `extrinsics` | `ndarray (N, 7)` | Per-frame `[qx, qy, qz, qw, tx, ty, tz]`     |

**Properties:**

- `num_frames` &mdash; Number of frames (rows in extrinsics).
- `fx`, `fy`, `cx`, `cy` &mdash; Individual intrinsic values as `float`.

**Methods:**

- `rotation_matrices()` &mdash; Returns `(N, 3, 3)` float64 rotation matrices
  converted from the quaternions.
- `translation_vectors()` &mdash; Returns `(N, 3)` float64 translation vectors
  (copy of extrinsics columns 4-6).

### `CameraModel`

IntEnum with all 18 supported camera models. Use `CameraModel.OPENCV`,
`CameraModel(4)`, etc.

### Constants

- `DISTORTION_PARAMS: dict[CameraModel, int]` &mdash; Number of meaningful
  distortion parameters per model.
- `SINGLE_FOCAL: dict[CameraModel, bool]` &mdash; Whether the model uses a
  single focal length.

## Usage Examples

### Write a calibrated TIFF stack

```python
import numpy as np
from tifffile import TiffWriter
from tifffile.camera import CameraModel, camera_extratags

images = np.random.randint(0, 255, (100, 480, 640), dtype='uint8')

# camera calibration
poses = np.zeros((100, 7), dtype='float64')
poses[:, 3] = 1.0  # identity rotation for all frames

tags = camera_extratags(
    model=CameraModel.OPENCV,
    width=640,
    height=480,
    intrinsics=[525.0, 525.0, 319.5, 239.5],
    distortion=[-0.28, 0.07, 0.0008, -0.0003],
    extrinsics=poses,
)

with TiffWriter('calibrated.tif') as tw:
    tw.write(images, extratags=tags)
```

### Read calibration back

```python
from tifffile import TiffFile
from tifffile.camera import read_camera

with TiffFile('calibrated.tif') as tif:
    cam = read_camera(tif)

print(cam.model)          # CameraModel.OPENCV
print(cam.fx, cam.fy)     # 525.0 525.0
print(cam.num_frames)     # 100

R = cam.rotation_matrices()   # (100, 3, 3)
t = cam.translation_vectors() # (100, 3)
```

### Single-focal shorthand

For models where fx = fy, pass 3 values instead of 4:

```python
tags = camera_extratags(
    model=CameraModel.SIMPLE_PINHOLE,
    width=640, height=480,
    intrinsics=[500.0, 320.0, 240.0],  # [f, cx, cy] -> stored as [f, f, cx, cy]
)
```

### 360 equirectangular

```python
tags = camera_extratags(
    model=CameraModel.EQUIRECTANGULAR,
    width=4096, height=2048,
    intrinsics=[360.0, 180.0, 0.0, 0.0],  # [h_fov, v_fov, lon_off, lat_off]
)
```

### With compression

Camera tags are in the IFD header, not in image data. They work with any
compression:

```python
with TiffWriter('compressed.tif') as tw:
    tw.write(images, compression='zstd', extratags=tags)
```

### Construct an OpenCV camera matrix

```python
import numpy as np
from tifffile import TiffFile
from tifffile.camera import read_camera

with TiffFile('calibrated.tif') as tif:
    cam = read_camera(tif)

K = np.array([
    [cam.fx,    0,  cam.cx],
    [   0,  cam.fy, cam.cy],
    [   0,     0,      1  ],
])

# distortion coefficients for cv2.undistort
dist_coeffs = cam.distortion[:4]  # k1, k2, p1, p2 for OPENCV model
```

## Design Decisions

**Fixed-length fields.** Intrinsics are always 4 doubles, distortion always 12
doubles. This avoids ambiguity when parsing &mdash; readers don't need to know
the camera model to determine field boundaries. Zero-padding is cheap (96 bytes
of zeros at most).

**Private tag range.** Tags 65201-65204 are in the TIFF "reusable" private
range (65000-65535). These codes are not registered with any standards body and
may collide with other software using the same range. The tags are
self-describing (known type and count), so collisions are detectable.

**First-page-only storage.** All tags use `writeonce=True`, meaning they appear
only in the first IFD. This keeps multi-page TIFFs clean &mdash; calibration is
a per-file property, not per-page. The extrinsics array maps frames by index:
frame `i` corresponds to `extrinsics[i]`.

**COLMAP compatibility.** Model IDs 0-15 match COLMAP's `camera_models.h`
exactly. Quaternion order `[qx, qy, qz, qw]` matches COLMAP/Eigen convention
(not the `[qw, qx, qy, qz]` Hamilton convention used by some libraries).
Transform semantics are `cam_from_world`, matching COLMAP's image model.

**Extensions (16-17).** `EQUIRECTANGULAR` and `CUBEMAP` are not in COLMAP but
follow the same pattern: 4 intrinsic values with model-specific semantics, no
distortion.
