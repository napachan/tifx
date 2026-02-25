"""Camera calibration metadata for TIFF files.

Store and retrieve camera intrinsics, distortion, and per-frame extrinsics
in TIFF tags using a COLMAP-compatible convention.

Tag allocation (private reusable range 65201-65207):

    65201 (LONG × 3): camera model ID, image width, image height
    65202 (DOUBLE × 4): intrinsics [fx, fy, cx, cy]
    65203 (DOUBLE × 12): distortion coefficients, zero-padded
    65204 (DOUBLE × N*7): per-frame extrinsics [qx, qy, qz, qw, tx, ty, tz]
    65205 (BYTE): per-frame source filenames, UTF-8 null-separated
    65206 (DOUBLE × N): per-frame Unix epoch timestamps
    65207 (DOUBLE × N*3): per-frame GPS [latitude, longitude, altitude]

Extrinsics convention (COLMAP-compatible):
    Quaternion in [qx, qy, qz, qw] order (Eigen convention).
    Transform is cam_from_world: x_cam = R * x_world + t.

Examples:

    Write a TIFF stack with OPENCV camera model::

        >>> import numpy as np
        >>> from tifffile import TiffWriter
        >>> from tifffile.camera import CameraModel, camera_extratags, read_camera

        >>> images = np.random.randint(0, 255, (10, 480, 640), dtype='uint8')
        >>> poses = np.zeros((10, 7), dtype='float64')
        >>> poses[:, 3] = 1.0  # identity rotation (qw=1)
        >>> tags = camera_extratags(
        ...     model=CameraModel.OPENCV,
        ...     width=640, height=480,
        ...     intrinsics=[500.0, 500.0, 320.0, 240.0],
        ...     distortion=[-0.1, 0.01, 0.001, -0.001],
        ...     extrinsics=poses,
        ... )
        >>> with TiffWriter('_test_cam.tif') as tw:  # doctest: +SKIP
        ...     tw.write(images, extratags=tags)

    Read back::

        >>> from tifffile import TiffFile
        >>> with TiffFile('_test_cam.tif') as t:  # doctest: +SKIP
        ...     cam = read_camera(t)
        ...     print(cam.model, cam.intrinsics, cam.extrinsics.shape)

"""

from __future__ import annotations

import enum
import json
import os
import struct
from dataclasses import dataclass
from typing import Any, Sequence

import numpy

TAG_CAMERA_MODEL = 65201
TAG_INTRINSICS = 65202
TAG_DISTORTION = 65203
TAG_EXTRINSICS = 65204
TAG_FILENAMES = 65205
TAG_TIMESTAMPS = 65206
TAG_GPS = 65207

NUM_INTRINSICS = 4
NUM_DISTORTION = 12
NUM_EXTRINSICS = 7  # qx, qy, qz, qw, tx, ty, tz
NUM_GPS = 3  # lat, lon, alt


class CameraModel(enum.IntEnum):
    """Camera model IDs (COLMAP-compatible)."""

    SIMPLE_PINHOLE = 0
    PINHOLE = 1
    SIMPLE_RADIAL = 2
    RADIAL = 3
    OPENCV = 4
    OPENCV_FISHEYE = 5
    FULL_OPENCV = 6
    FOV = 7
    SIMPLE_RADIAL_FISHEYE = 8
    RADIAL_FISHEYE = 9
    THIN_PRISM_FISHEYE = 10
    RADTAN_THIN_PRISM_FISHEYE = 11
    SIMPLE_DIVISION = 12
    DIVISION = 13
    SIMPLE_FISHEYE = 14
    FISHEYE = 15
    EQUIRECTANGULAR = 16
    CUBEMAP = 17


#: Number of meaningful distortion parameters per model.
DISTORTION_PARAMS: dict[CameraModel, int] = {
    CameraModel.SIMPLE_PINHOLE: 0,
    CameraModel.PINHOLE: 0,
    CameraModel.SIMPLE_RADIAL: 1,
    CameraModel.RADIAL: 2,
    CameraModel.OPENCV: 4,
    CameraModel.OPENCV_FISHEYE: 4,
    CameraModel.FULL_OPENCV: 8,
    CameraModel.FOV: 1,
    CameraModel.SIMPLE_RADIAL_FISHEYE: 1,
    CameraModel.RADIAL_FISHEYE: 2,
    CameraModel.THIN_PRISM_FISHEYE: 8,
    CameraModel.RADTAN_THIN_PRISM_FISHEYE: 12,
    CameraModel.SIMPLE_DIVISION: 1,
    CameraModel.DIVISION: 1,
    CameraModel.SIMPLE_FISHEYE: 0,
    CameraModel.FISHEYE: 0,
    CameraModel.EQUIRECTANGULAR: 0,
    CameraModel.CUBEMAP: 0,
}

#: Whether the model uses a single focal length (fx=fy=f).
SINGLE_FOCAL: dict[CameraModel, bool] = {
    CameraModel.SIMPLE_PINHOLE: True,
    CameraModel.PINHOLE: False,
    CameraModel.SIMPLE_RADIAL: True,
    CameraModel.RADIAL: True,
    CameraModel.OPENCV: False,
    CameraModel.OPENCV_FISHEYE: False,
    CameraModel.FULL_OPENCV: False,
    CameraModel.FOV: False,
    CameraModel.SIMPLE_RADIAL_FISHEYE: True,
    CameraModel.RADIAL_FISHEYE: True,
    CameraModel.THIN_PRISM_FISHEYE: False,
    CameraModel.RADTAN_THIN_PRISM_FISHEYE: False,
    CameraModel.SIMPLE_DIVISION: True,
    CameraModel.DIVISION: False,
    CameraModel.SIMPLE_FISHEYE: True,
    CameraModel.FISHEYE: False,
    CameraModel.EQUIRECTANGULAR: False,
    CameraModel.CUBEMAP: False,
}


@dataclass
class CameraData:
    """Camera calibration data read from TIFF tags.

    Attributes:
        model: Camera model enum.
        width: Image width in pixels.
        height: Image height in pixels.
        intrinsics: Intrinsic parameters [fx, fy, cx, cy] as (4,) float64.
            For equirectangular: [h_fov, v_fov, lon_offset, lat_offset].
            For cubemap: [face_size, 0, 0, 0].
        distortion: Distortion coefficients as (12,) float64, zero-padded.
        extrinsics: Per-frame poses as (N, 7) float64.
            Each row: [qx, qy, qz, qw, tx, ty, tz].
            Convention: cam_from_world.
        filenames: Per-frame source filenames. Optional.
        timestamps: Per-frame Unix epoch seconds as (N,) float64. Optional.
        gps: Per-frame GPS as (N, 3) float64 [lat, lon, alt]. Optional.
    """

    model: CameraModel
    width: int
    height: int
    intrinsics: numpy.ndarray
    distortion: numpy.ndarray
    extrinsics: numpy.ndarray
    filenames: list[str] | None = None
    timestamps: numpy.ndarray | None = None
    gps: numpy.ndarray | None = None

    @property
    def num_frames(self) -> int:
        """Number of frames with extrinsics."""
        return self.extrinsics.shape[0]

    @property
    def fx(self) -> float:
        return float(self.intrinsics[0])

    @property
    def fy(self) -> float:
        return float(self.intrinsics[1])

    @property
    def cx(self) -> float:
        return float(self.intrinsics[2])

    @property
    def cy(self) -> float:
        return float(self.intrinsics[3])

    def rotation_matrices(self) -> numpy.ndarray:
        """Return rotation matrices as (N, 3, 3) float64."""
        return _quaternions_to_matrices(self.extrinsics[:, :4])

    def translation_vectors(self) -> numpy.ndarray:
        """Return translation vectors as (N, 3) float64."""
        return self.extrinsics[:, 4:7].copy()

    @property
    def K(self) -> numpy.ndarray:
        """3x3 intrinsic matrix."""
        return numpy.array(
            [
                [self.fx, 0.0, self.cx],
                [0.0, self.fy, self.cy],
                [0.0, 0.0, 1.0],
            ],
            dtype='float64',
        )

    def cam_from_world_matrix(self, frame: int) -> numpy.ndarray:
        """4x4 cam-from-world transform for given frame."""
        R = _quaternions_to_matrices(self.extrinsics[frame : frame + 1, :4])[0]
        t = self.extrinsics[frame, 4:7]
        T = numpy.eye(4, dtype='float64')
        T[:3, :3] = R
        T[:3, 3] = t
        return T

    def world_from_cam_matrix(self, frame: int) -> numpy.ndarray:
        """4x4 world-from-cam transform (inverse of cam-from-world)."""
        R = _quaternions_to_matrices(self.extrinsics[frame : frame + 1, :4])[0]
        t = self.extrinsics[frame, 4:7]
        T = numpy.eye(4, dtype='float64')
        T[:3, :3] = R.T
        T[:3, 3] = -R.T @ t
        return T

    def camera_positions(self) -> numpy.ndarray:
        """World-space camera positions as (N, 3), computed as -R^T @ t."""
        R = _quaternions_to_matrices(self.extrinsics[:, :4])  # (N, 3, 3)
        t = self.extrinsics[:, 4:7]  # (N, 3)
        return -numpy.einsum('nij,nj->ni', R.transpose(0, 2, 1), t)

    def to_pycolmap(self) -> tuple[Any, list[Any]]:
        """Return (pycolmap.Camera, list[pycolmap.Rigid3d]).

        Requires pycolmap to be installed.
        Quaternions are converted from [qx, qy, qz, qw] to
        pycolmap's [qw, qx, qy, qz] (Eigen/Hamilton) order.
        """
        import pycolmap

        params = _intrinsics_to_colmap_params(
            self.model, self.intrinsics, self.distortion
        )
        camera = pycolmap.Camera(
            model=int(self.model),
            width=self.width,
            height=self.height,
            params=params.tolist(),
        )
        poses = []
        for i in range(self.num_frames):
            qx, qy, qz, qw = self.extrinsics[i, :4]
            t = self.extrinsics[i, 4:7].copy()
            rot = pycolmap.Rotation3d(numpy.array([qw, qx, qy, qz]))
            rigid = pycolmap.Rigid3d(rotation=rot, translation=t)
            poses.append(rigid)
        return camera, poses

    @classmethod
    def from_pycolmap(
        cls,
        camera: Any,
        poses: Sequence[Any] | None = None,
    ) -> CameraData:
        """Create CameraData from pycolmap.Camera and optional Rigid3d poses.

        Parameters:
            camera: pycolmap.Camera instance.
            poses: List of pycolmap.Rigid3d (cam_from_world). Optional.
        """
        model = CameraModel(int(camera.model_id))
        params = numpy.asarray(camera.params, dtype='float64')
        intrinsics, distortion = _colmap_params_to_intrinsics(model, params)

        if poses:
            extrinsics = numpy.empty(
                (len(poses), NUM_EXTRINSICS), dtype='float64'
            )
            for i, rigid in enumerate(poses):
                q = rigid.rotation.quat  # [qw, qx, qy, qz]
                extrinsics[i, :4] = [q[1], q[2], q[3], q[0]]
                extrinsics[i, 4:] = rigid.translation
        else:
            extrinsics = numpy.array(
                [[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]]
            )

        return cls(
            model=model,
            width=camera.width,
            height=camera.height,
            intrinsics=intrinsics,
            distortion=distortion,
            extrinsics=extrinsics,
        )

    def to_opencv(self) -> tuple[numpy.ndarray, numpy.ndarray]:
        """Return (K_3x3, dist_coeffs) for cv2.undistort etc.

        dist_coeffs length depends on model:
        PINHOLE: 0, OPENCV: 4, FULL_OPENCV: 8.
        """
        n_dist = DISTORTION_PARAMS.get(self.model, 0)
        if n_dist > 0:
            dist_coeffs = self.distortion[:n_dist].copy()
        else:
            dist_coeffs = numpy.zeros(0, dtype='float64')
        return self.K, dist_coeffs

    @classmethod
    def from_opencv(
        cls,
        K: numpy.ndarray,
        dist_coeffs: numpy.ndarray | Sequence[float] | None,
        width: int,
        height: int,
        *,
        model: CameraModel | None = None,
        extrinsics: numpy.ndarray | None = None,
    ) -> CameraData:
        """Create CameraData from OpenCV calibration result.

        If model is None, inferred from len(dist_coeffs):
        0 -> PINHOLE, <=4 -> OPENCV, >4 -> FULL_OPENCV.
        """
        K = numpy.asarray(K, dtype='float64')
        intrinsics = numpy.array(
            [K[0, 0], K[1, 1], K[0, 2], K[1, 2]], dtype='float64'
        )

        if dist_coeffs is None:
            dist = numpy.zeros(0, dtype='float64')
        else:
            dist = numpy.asarray(dist_coeffs, dtype='float64').ravel()

        if model is None:
            n = len(dist)
            if n == 0:
                model = CameraModel.PINHOLE
            elif n <= 4:
                model = CameraModel.OPENCV
            else:
                model = CameraModel.FULL_OPENCV

        distortion = numpy.zeros(NUM_DISTORTION, dtype='float64')
        n = min(len(dist), NUM_DISTORTION)
        distortion[:n] = dist[:n]

        if extrinsics is None:
            extrinsics = numpy.array(
                [[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]]
            )

        return cls(
            model=model,
            width=width,
            height=height,
            intrinsics=intrinsics,
            distortion=distortion,
            extrinsics=extrinsics,
        )

    def to_transforms_json(
        self,
        image_paths: Sequence[str] | None = None,
    ) -> dict:
        """Return transforms.json dict (instant-ngp / nerfstudio convention).

        Converts cam_from_world (COLMAP) to world_from_cam (NeRF),
        flipping Y and Z axes (OpenCV to OpenGL convention).
        """
        data: dict[str, Any] = {
            'w': self.width,
            'h': self.height,
            'fl_x': self.fx,
            'fl_y': self.fy,
            'cx': self.cx,
            'cy': self.cy,
            'camera_model': self.model.name,
        }

        # distortion params as named keys
        n_dist = DISTORTION_PARAMS.get(self.model, 0)
        if n_dist > 0:
            if self.model == CameraModel.OPENCV_FISHEYE:
                keys = ('k1', 'k2', 'k3', 'k4')
            else:
                keys = ('k1', 'k2', 'p1', 'p2', 'k3', 'k4', 'k5', 'k6')
            for j, key in enumerate(keys[:n_dist]):
                data[key] = float(self.distortion[j])

        # convert cam_from_world to NeRF world_from_cam (OpenGL axes)
        flip = numpy.diag([1.0, -1.0, -1.0, 1.0])
        paths = image_paths if image_paths is not None else self.filenames
        frames = []
        for i in range(self.num_frames):
            c2w = self.world_from_cam_matrix(i)
            c2w_gl = c2w @ flip
            frame: dict[str, Any] = {
                'transform_matrix': c2w_gl.tolist(),
            }
            if paths is not None:
                frame['file_path'] = paths[i]
            if self.timestamps is not None:
                frame['timestamp'] = float(self.timestamps[i])
            if self.gps is not None:
                frame['gps'] = self.gps[i].tolist()
            frames.append(frame)
        data['frames'] = frames
        return data

    @classmethod
    def from_transforms_json(
        cls,
        data: dict | str | os.PathLike,
    ) -> CameraData:
        """Create CameraData from transforms.json file or dict.

        Accepts file path, JSON string, or pre-loaded dict.
        Converts world_from_cam (NeRF) to cam_from_world (COLMAP),
        flipping Y and Z axes (OpenGL to OpenCV convention).
        """
        if isinstance(data, dict):
            pass
        elif isinstance(data, (str, os.PathLike)):
            text = str(data)
            if text.lstrip().startswith('{'):
                data = json.loads(text)
            else:
                with open(text) as f:
                    data = json.load(f)
        else:
            raise TypeError(
                f'expected dict, str, or PathLike, got {type(data)}'
            )

        w = int(data['w'])
        h = int(data['h'])

        if 'fl_x' in data:
            fl_x = float(data['fl_x'])
            fl_y = float(data.get('fl_y', fl_x))
        elif 'camera_angle_x' in data:
            fl_x = 0.5 * w / numpy.tan(0.5 * float(data['camera_angle_x']))
            if 'camera_angle_y' in data:
                fl_y = 0.5 * h / numpy.tan(
                    0.5 * float(data['camera_angle_y'])
                )
            else:
                fl_y = fl_x
        else:
            raise ValueError(
                'transforms.json must contain fl_x or camera_angle_x'
            )

        cx = float(data.get('cx', w / 2.0))
        cy = float(data.get('cy', h / 2.0))
        intrinsics = numpy.array([fl_x, fl_y, cx, cy], dtype='float64')

        model_name = data.get('camera_model', 'OPENCV')
        model = CameraModel[model_name]

        distortion = numpy.zeros(NUM_DISTORTION, dtype='float64')
        if model == CameraModel.OPENCV_FISHEYE:
            keys = ('k1', 'k2', 'k3', 'k4')
        else:
            keys = ('k1', 'k2', 'p1', 'p2', 'k3', 'k4', 'k5', 'k6')
        for j, key in enumerate(keys):
            if key in data:
                distortion[j] = float(data[key])

        # convert NeRF world_from_cam (OpenGL) to cam_from_world (OpenCV)
        flip = numpy.diag([1.0, -1.0, -1.0, 1.0])
        frames = data.get('frames', [])
        filenames = None
        timestamps = None
        gps = None
        if frames:
            extrinsics = numpy.empty(
                (len(frames), NUM_EXTRINSICS), dtype='float64'
            )
            fnames = []
            ts_list = []
            gps_list = []
            for i, frame in enumerate(frames):
                c2w_gl = numpy.array(
                    frame['transform_matrix'], dtype='float64'
                )
                c2w_cv = c2w_gl @ flip
                w2c = numpy.linalg.inv(c2w_cv)
                q = _matrix_to_quaternion(w2c[:3, :3])
                extrinsics[i, :4] = q
                extrinsics[i, 4:] = w2c[:3, 3]
                if 'file_path' in frame:
                    fnames.append(frame['file_path'])
                if 'timestamp' in frame:
                    ts_list.append(float(frame['timestamp']))
                if 'gps' in frame:
                    gps_list.append(frame['gps'])
            if fnames:
                filenames = fnames
            if ts_list:
                timestamps = numpy.array(ts_list, dtype='float64')
            if gps_list:
                gps = numpy.array(gps_list, dtype='float64').reshape(
                    -1, NUM_GPS
                )
        else:
            extrinsics = numpy.array(
                [[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]]
            )

        return cls(
            model=model,
            width=w,
            height=h,
            intrinsics=intrinsics,
            distortion=distortion,
            extrinsics=extrinsics,
            filenames=filenames,
            timestamps=timestamps,
            gps=gps,
        )


def camera_extratags(
    model: CameraModel | int,
    width: int,
    height: int,
    intrinsics: Sequence[float],
    distortion: Sequence[float] | None = None,
    extrinsics: numpy.ndarray | Sequence[Sequence[float]] | None = None,
    *,
    filenames: Sequence[str] | None = None,
    timestamps: numpy.ndarray | Sequence[float] | None = None,
    gps: numpy.ndarray | Sequence[Sequence[float]] | None = None,
) -> list[tuple[int, int, int, bytes, bool]]:
    """Build TIFF extratags for camera calibration data.

    All tags are written to the first page only (writeonce=True).
    Pass the returned list as ``extratags`` to :py:meth:`TiffWriter.write`.

    Parameters:
        model:
            Camera model ID. See :py:class:`CameraModel`.
        width:
            Image width in pixels.
        height:
            Image height in pixels.
        intrinsics:
            Intrinsic parameters. For pinhole-type models: [fx, fy, cx, cy]
            or [f, cx, cy] for single-focal models (fx=fy=f).
            For equirectangular: [h_fov, v_fov, lon_offset, lat_offset].
            For cubemap: [face_size].
        distortion:
            Distortion coefficients. Length depends on model.
            Zero-padded to 12 values. Default is all zeros.
        extrinsics:
            Per-frame poses as (N, 7) array or list of [qx, qy, qz, qw,
            tx, ty, tz]. Convention: cam_from_world.
            Default is identity pose for one frame.
        filenames:
            Per-frame source filenames. Encoded as UTF-8 null-separated.
        timestamps:
            Per-frame Unix epoch seconds as (N,) float64.
        gps:
            Per-frame GPS as (N, 3) float64 [lat, lon, alt].

    Returns:
        List of extratag tuples for :py:meth:`TiffWriter.write`.

    """
    model = CameraModel(model)

    # normalize intrinsics to [fx, fy, cx, cy]
    intr = list(intrinsics)
    if SINGLE_FOCAL.get(model, False) and len(intr) == 3:
        # [f, cx, cy] -> [f, f, cx, cy]
        intr = [intr[0], intr[0], intr[1], intr[2]]
    if len(intr) != NUM_INTRINSICS:
        msg = f'expected {NUM_INTRINSICS} intrinsics, got {len(intr)}'
        raise ValueError(msg)

    # normalize distortion to 12 values
    if distortion is None:
        dist = [0.0] * NUM_DISTORTION
    else:
        dist = list(distortion)
        expected = DISTORTION_PARAMS.get(model, 0)
        if len(dist) < expected:
            msg = (
                f'model {model.name} expects at least '
                f'{expected} distortion params, got {len(dist)}'
            )
            raise ValueError(msg)
        if len(dist) > NUM_DISTORTION:
            msg = f'too many distortion params: {len(dist)} > {NUM_DISTORTION}'
            raise ValueError(msg)
        dist.extend([0.0] * (NUM_DISTORTION - len(dist)))

    # normalize extrinsics to (N, 7) float64
    if extrinsics is None:
        ext = numpy.array([[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]])
    else:
        ext = numpy.asarray(extrinsics, dtype='float64')
        if ext.ndim == 1:
            ext = ext.reshape(1, -1)
        if ext.shape[1] != NUM_EXTRINSICS:
            msg = (
                f'extrinsics must have {NUM_EXTRINSICS} columns, '
                f'got {ext.shape[1]}'
            )
            raise ValueError(msg)

    # DATATYPE.LONG = 4, DATATYPE.DOUBLE = 12
    tags: list[tuple[int, int, int, bytes, bool]] = [
        (
            TAG_CAMERA_MODEL,
            4,  # LONG
            3,
            struct.pack('3I', int(model), width, height),
            True,
        ),
        (
            TAG_INTRINSICS,
            12,  # DOUBLE
            NUM_INTRINSICS,
            struct.pack(f'{NUM_INTRINSICS}d', *intr),
            True,
        ),
        (
            TAG_DISTORTION,
            12,  # DOUBLE
            NUM_DISTORTION,
            struct.pack(f'{NUM_DISTORTION}d', *dist),
            True,
        ),
        (
            TAG_EXTRINSICS,
            12,  # DOUBLE
            ext.size,
            ext.tobytes(),
            True,
        ),
    ]

    if filenames is not None:
        blob = b'\x00'.join(
            name.encode('utf-8') for name in filenames
        ) + b'\x00'
        tags.append((TAG_FILENAMES, 1, len(blob), blob, True))  # BYTE

    if timestamps is not None:
        ts = numpy.asarray(timestamps, dtype='float64').ravel()
        tags.append((TAG_TIMESTAMPS, 12, ts.size, ts.tobytes(), True))

    if gps is not None:
        g = numpy.asarray(gps, dtype='float64').reshape(-1, NUM_GPS)
        tags.append((TAG_GPS, 12, g.size, g.tobytes(), True))

    return tags


def read_camera(tif: Any) -> CameraData:
    """Read camera calibration data from TIFF file.

    Parameters:
        tif:
            :py:class:`TiffFile` instance or :py:class:`TiffPage`.

    Returns:
        :py:class:`CameraData` with intrinsics, distortion, and extrinsics.

    Raises:
        KeyError: If camera tags are not found.

    """
    # accept TiffFile or TiffPage/TiffFrame
    pages = getattr(tif, 'pages', None)
    if pages is not None:
        page = pages.first
    else:
        page = tif

    tags = page.tags

    if TAG_CAMERA_MODEL not in tags:
        msg = f'tag {TAG_CAMERA_MODEL} (camera model) not found'
        raise KeyError(msg)

    model_val = tags[TAG_CAMERA_MODEL].value
    if len(model_val) < 3:
        msg = f'tag {TAG_CAMERA_MODEL} value too short: {len(model_val)}'
        raise ValueError(msg)
    model_id, width, height = model_val[0], model_val[1], model_val[2]
    model = CameraModel(model_id)

    intrinsics = numpy.array(tags[TAG_INTRINSICS].value, dtype='float64')
    distortion = numpy.array(tags[TAG_DISTORTION].value, dtype='float64')

    ext_values = numpy.array(tags[TAG_EXTRINSICS].value, dtype='float64')
    extrinsics = ext_values.reshape(-1, NUM_EXTRINSICS)

    filenames = None
    if TAG_FILENAMES in tags:
        raw = bytes(tags[TAG_FILENAMES].value)
        filenames = [
            s.decode('utf-8') for s in raw.rstrip(b'\x00').split(b'\x00')
        ]

    timestamps = None
    if TAG_TIMESTAMPS in tags:
        timestamps = numpy.array(
            tags[TAG_TIMESTAMPS].value, dtype='float64'
        )

    gps = None
    if TAG_GPS in tags:
        gps = numpy.array(tags[TAG_GPS].value, dtype='float64').reshape(
            -1, NUM_GPS
        )

    return CameraData(
        model=model,
        width=width,
        height=height,
        intrinsics=intrinsics,
        distortion=distortion,
        extrinsics=extrinsics,
        filenames=filenames,
        timestamps=timestamps,
        gps=gps,
    )


def _quaternions_to_matrices(quats: numpy.ndarray) -> numpy.ndarray:
    """Convert quaternions [qx, qy, qz, qw] to rotation matrices (N, 3, 3)."""
    qx, qy, qz, qw = quats[:, 0], quats[:, 1], quats[:, 2], quats[:, 3]
    R = numpy.empty((len(quats), 3, 3), dtype='float64')
    R[:, 0, 0] = 1 - 2 * (qy * qy + qz * qz)
    R[:, 0, 1] = 2 * (qx * qy - qz * qw)
    R[:, 0, 2] = 2 * (qx * qz + qy * qw)
    R[:, 1, 0] = 2 * (qx * qy + qz * qw)
    R[:, 1, 1] = 1 - 2 * (qx * qx + qz * qz)
    R[:, 1, 2] = 2 * (qy * qz - qx * qw)
    R[:, 2, 0] = 2 * (qx * qz - qy * qw)
    R[:, 2, 1] = 2 * (qy * qz + qx * qw)
    R[:, 2, 2] = 1 - 2 * (qx * qx + qy * qy)
    return R


def _matrix_to_quaternion(R: numpy.ndarray) -> numpy.ndarray:
    """Convert 3x3 rotation matrix to quaternion [qx, qy, qz, qw]."""
    trace = R[0, 0] + R[1, 1] + R[2, 2]
    if trace > 0:
        s = 0.5 / numpy.sqrt(trace + 1.0)
        qw = 0.25 / s
        qx = (R[2, 1] - R[1, 2]) * s
        qy = (R[0, 2] - R[2, 0]) * s
        qz = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * numpy.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        qw = (R[2, 1] - R[1, 2]) / s
        qx = 0.25 * s
        qy = (R[0, 1] + R[1, 0]) / s
        qz = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * numpy.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        qw = (R[0, 2] - R[2, 0]) / s
        qx = (R[0, 1] + R[1, 0]) / s
        qy = 0.25 * s
        qz = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * numpy.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        qw = (R[1, 0] - R[0, 1]) / s
        qx = (R[0, 2] + R[2, 0]) / s
        qy = (R[1, 2] + R[2, 1]) / s
        qz = 0.25 * s
    return numpy.array([qx, qy, qz, qw], dtype='float64')


def _intrinsics_to_colmap_params(
    model: CameraModel,
    intrinsics: numpy.ndarray,
    distortion: numpy.ndarray,
) -> numpy.ndarray:
    """Merge intrinsics and distortion into pycolmap flat params array."""
    n_dist = DISTORTION_PARAMS.get(model, 0)
    if SINGLE_FOCAL.get(model, False):
        params = [
            float(intrinsics[0]),
            float(intrinsics[2]),
            float(intrinsics[3]),
        ]
    else:
        params = [float(x) for x in intrinsics[:4]]
    params.extend(float(distortion[j]) for j in range(n_dist))
    return numpy.array(params, dtype='float64')


def _colmap_params_to_intrinsics(
    model: CameraModel,
    params: numpy.ndarray,
) -> tuple[numpy.ndarray, numpy.ndarray]:
    """Split pycolmap flat params into intrinsics (4,) and distortion (12,)."""
    if SINGLE_FOCAL.get(model, False):
        f = float(params[0])
        cx = float(params[1])
        cy = float(params[2])
        intrinsics = numpy.array([f, f, cx, cy], dtype='float64')
        dist_start = 3
    else:
        intrinsics = numpy.array(params[:4], dtype='float64')
        dist_start = 4

    distortion = numpy.zeros(NUM_DISTORTION, dtype='float64')
    n_dist = DISTORTION_PARAMS.get(model, 0)
    if n_dist > 0:
        distortion[:n_dist] = params[dist_start : dist_start + n_dist]
    return intrinsics, distortion
