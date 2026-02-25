"""Tests for native TIFF-based multi-dimensional slicing."""

from __future__ import annotations

import numpy
import pytest
import tempfile
import os

import tifffile
from tifffile.sequences import _split_axes, _normalize_selection, _selection_to_page_indices


# --- Unit tests for helper functions ---

class TestSplitAxes:
    """Tests for _split_axes."""

    def test_tzcyx(self):
        # 3T x 4Z x 2C x 64Y x 64X = shape, page_size = 2*64*64
        shape = (3, 4, 2, 64, 64)
        axes = 'TZCYX'
        page_size = 2 * 64 * 64
        fs, fa, ss, sa, si = _split_axes(shape, axes, page_size)
        assert fs == (3, 4)
        assert fa == 'TZ'
        assert ss == (2, 64, 64)
        assert sa == 'CYX'
        assert si == 2

    def test_tyx(self):
        shape = (10, 64, 64)
        axes = 'TYX'
        page_size = 64 * 64
        fs, fa, ss, sa, si = _split_axes(shape, axes, page_size)
        assert fs == (10,)
        assert fa == 'T'
        assert ss == (64, 64)
        assert sa == 'YX'
        assert si == 1

    def test_single_page(self):
        shape = (64, 64)
        axes = 'YX'
        page_size = 64 * 64
        fs, fa, ss, sa, si = _split_axes(shape, axes, page_size)
        assert fs == ()
        assert fa == ''
        assert ss == (64, 64)
        assert sa == 'YX'
        assert si == 0

    def test_tzcyxs(self):
        # ImageJ hyperstack: T=2, Z=3, C=1, Y=32, X=32, S=3
        # page_size = Y*X*S = 32*32*3 = 3072
        # C=1 so C*Y*X*S also = 3072, split finds YXS first
        shape = (2, 3, 1, 32, 32, 3)
        axes = 'TZCYXS'
        page_size = 32 * 32 * 3  # one page = YXS
        fs, fa, ss, sa, si = _split_axes(shape, axes, page_size)
        # Since C=1, product of CYXS = YXS = page_size, so split at i=3
        assert fs == (2, 3, 1)
        assert fa == 'TZC'
        assert ss == (32, 32, 3)
        assert sa == 'YXS'
        assert si == 3

    def test_tzcyxs_c2(self):
        # With C>1, split is different: page = CYX, frame = TZ
        shape = (2, 3, 2, 32, 32)
        axes = 'TZCYX'
        page_size = 2 * 32 * 32  # one page = CYX
        fs, fa, ss, sa, si = _split_axes(shape, axes, page_size)
        assert fs == (2, 3)
        assert fa == 'TZ'
        assert ss == (2, 32, 32)
        assert sa == 'CYX'
        assert si == 2


class TestNormalizeSelection:
    """Tests for _normalize_selection."""

    def test_dict_axis_codes(self):
        shape = (3, 4, 64, 64)
        axes = 'TZYX'
        split_idx = 2  # frame: TZ, spatial: YX
        sel = {'T': 1, 'Z': slice(0, 2)}
        frame, spatial = _normalize_selection(sel, shape, axes, split_idx)
        assert frame == {'T': 1, 'Z': slice(0, 2)}
        assert spatial is None

    def test_dict_axis_names(self):
        shape = (3, 4, 64, 64)
        axes = 'TZYX'
        split_idx = 2
        sel = {'time': 1, 'depth': slice(0, 2)}
        frame, spatial = _normalize_selection(sel, shape, axes, split_idx)
        assert frame == {'T': 1, 'Z': slice(0, 2)}
        assert spatial is None

    def test_dict_mixed_frame_and_spatial(self):
        shape = (3, 4, 64, 64)
        axes = 'TZYX'
        split_idx = 2
        sel = {'T': 1, 'Y': slice(10, 20)}
        frame, spatial = _normalize_selection(sel, shape, axes, split_idx)
        assert frame == {'T': 1}
        assert spatial == (slice(10, 20), slice(None))

    def test_tuple_positional(self):
        shape = (3, 4, 64, 64)
        axes = 'TZYX'
        split_idx = 2
        sel = (1, slice(0, 2))
        frame, spatial = _normalize_selection(sel, shape, axes, split_idx)
        assert frame == {'T': 1, 'Z': slice(0, 2)}
        assert spatial is None

    def test_tuple_with_spatial(self):
        shape = (3, 4, 64, 64)
        axes = 'TZYX'
        split_idx = 2
        sel = (1, slice(0, 2), slice(10, 20))
        frame, spatial = _normalize_selection(sel, shape, axes, split_idx)
        assert frame == {'T': 1, 'Z': slice(0, 2)}
        assert spatial == (slice(10, 20),)

    def test_unknown_axis(self):
        shape = (3, 4, 64, 64)
        axes = 'TZYX'
        split_idx = 2
        with pytest.raises(ValueError, match='unknown axis'):
            _normalize_selection({'W': 1}, shape, axes, split_idx)

    def test_too_many_elements(self):
        shape = (3, 64, 64)
        axes = 'TYX'
        split_idx = 1
        with pytest.raises(ValueError, match='selection has'):
            _normalize_selection((1, 2, 3, 4), shape, axes, split_idx)


class TestSelectionToPageIndices:
    """Tests for _selection_to_page_indices."""

    def test_single_index(self):
        frame_sel = {'T': 2, 'Z': 1}
        frame_shape = (3, 4)
        frame_axes = 'TZ'
        indices, out_shape = _selection_to_page_indices(
            frame_sel, frame_shape, frame_axes
        )
        # T=2, Z=1 -> flat index = 2*4 + 1 = 9
        assert list(indices) == [9]
        assert out_shape == ()

    def test_slice_selection(self):
        frame_sel = {'T': slice(0, 2), 'Z': slice(1, 3)}
        frame_shape = (3, 4)
        frame_axes = 'TZ'
        indices, out_shape = _selection_to_page_indices(
            frame_sel, frame_shape, frame_axes
        )
        assert out_shape == (2, 2)
        # T=[0,1] x Z=[1,2] -> (0,1),(0,2),(1,1),(1,2)
        expected = [0*4+1, 0*4+2, 1*4+1, 1*4+2]
        assert list(indices) == expected

    def test_mixed_int_and_slice(self):
        frame_sel = {'T': 1, 'Z': slice(0, 3)}
        frame_shape = (3, 4)
        frame_axes = 'TZ'
        indices, out_shape = _selection_to_page_indices(
            frame_sel, frame_shape, frame_axes
        )
        assert out_shape == (3,)  # T squeezed, Z kept
        expected = [1*4+0, 1*4+1, 1*4+2]
        assert list(indices) == expected

    def test_default_full_range(self):
        frame_sel = {'T': 1}  # Z not specified -> all Z
        frame_shape = (3, 4)
        frame_axes = 'TZ'
        indices, out_shape = _selection_to_page_indices(
            frame_sel, frame_shape, frame_axes
        )
        assert out_shape == (4,)
        expected = [1*4+0, 1*4+1, 1*4+2, 1*4+3]
        assert list(indices) == expected

    def test_negative_index(self):
        frame_sel = {'T': -1}
        frame_shape = (3,)
        frame_axes = 'T'
        indices, out_shape = _selection_to_page_indices(
            frame_sel, frame_shape, frame_axes
        )
        assert list(indices) == [2]
        assert out_shape == ()

    def test_out_of_range(self):
        frame_sel = {'T': 5}
        frame_shape = (3,)
        frame_axes = 'T'
        with pytest.raises(IndexError, match='out of range'):
            _selection_to_page_indices(frame_sel, frame_shape, frame_axes)

    def test_empty_selection(self):
        # No frame dims
        indices, out_shape = _selection_to_page_indices({}, (), '')
        assert list(indices) == [0]
        assert out_shape == ()


# --- Integration tests with real TIFF files ---

class TestSelectionIntegration:
    """Integration tests for selection-based reading."""

    @pytest.fixture
    def tzcyx_file(self, tmp_path):
        """Create a 5D TIFF (T=3, Z=4, C=2, Y=32, X=32)."""
        fname = str(tmp_path / 'tzcyx.tif')
        data = numpy.random.randint(
            0, 255, (3, 4, 2, 32, 32), dtype=numpy.uint8
        )
        tifffile.imwrite(
            fname,
            data,
            imagej=True,
            metadata={'axes': 'TZCYX'},
        )
        return fname, data

    @pytest.fixture
    def tyx_file(self, tmp_path):
        """Create a 3D TIFF (T=10, Y=64, X=64)."""
        fname = str(tmp_path / 'tyx.tif')
        data = numpy.random.randint(
            0, 255, (10, 64, 64), dtype=numpy.uint8
        )
        tifffile.imwrite(
            fname, data, imagej=True, metadata={'axes': 'TYX'}
        )
        return fname, data

    @pytest.fixture
    def yx_file(self, tmp_path):
        """Create a 2D TIFF (Y=64, X=64)."""
        fname = str(tmp_path / 'yx.tif')
        data = numpy.random.randint(
            0, 255, (64, 64), dtype=numpy.uint8
        )
        tifffile.imwrite(fname, data)
        return fname, data

    def test_dict_single_frame(self, tzcyx_file):
        fname, data = tzcyx_file
        result = tifffile.imread(fname, selection={'T': 1, 'Z': 2})
        expected = data[1, 2]
        numpy.testing.assert_array_equal(result, expected)

    def test_dict_slice_frame(self, tzcyx_file):
        fname, data = tzcyx_file
        result = tifffile.imread(fname, selection={'T': slice(0, 2)})
        expected = data[0:2]
        numpy.testing.assert_array_equal(result, expected)

    def test_dict_by_name(self, tzcyx_file):
        fname, data = tzcyx_file
        result = tifffile.imread(
            fname, selection={'time': 0, 'depth': slice(1, 3)}
        )
        expected = data[0, 1:3]
        numpy.testing.assert_array_equal(result, expected)

    def test_tuple_selection(self, tzcyx_file):
        fname, data = tzcyx_file
        result = tifffile.imread(fname, selection=(1, slice(0, 2)))
        expected = data[1, 0:2]
        numpy.testing.assert_array_equal(result, expected)

    def test_single_timepoint(self, tyx_file):
        fname, data = tyx_file
        result = tifffile.imread(fname, selection={'T': 5})
        expected = data[5]
        numpy.testing.assert_array_equal(result, expected)

    def test_negative_index(self, tyx_file):
        fname, data = tyx_file
        result = tifffile.imread(fname, selection={'T': -1})
        expected = data[-1]
        numpy.testing.assert_array_equal(result, expected)

    def test_single_page_with_spatial_selection(self, yx_file):
        fname, data = yx_file
        result = tifffile.imread(
            fname, selection={'Y': slice(10, 20), 'X': slice(5, 15)}
        )
        expected = data[10:20, 5:15]
        numpy.testing.assert_array_equal(result, expected)

    def test_key_and_selection_raises(self, tyx_file):
        fname, _ = tyx_file
        with pytest.raises(ValueError, match='cannot use both'):
            tifffile.imread(fname, key=0, selection={'T': 0})

    def test_tifffile_asarray_selection(self, tzcyx_file):
        fname, data = tzcyx_file
        with tifffile.TiffFile(fname) as tif:
            result = tif.asarray(selection={'T': 2, 'Z': 1})
            expected = data[2, 1]
            numpy.testing.assert_array_equal(result, expected)

    def test_series_asarray_selection(self, tzcyx_file):
        fname, data = tzcyx_file
        with tifffile.TiffFile(fname) as tif:
            s = tif.series[0]
            result = s.asarray(selection={'T': 0})
            expected = data[0]
            numpy.testing.assert_array_equal(result, expected)

    def test_squeeze_false(self, tzcyx_file):
        fname, data = tzcyx_file
        with tifffile.TiffFile(fname) as tif:
            s = tif.series[0]
            result = s.asarray(
                selection={'T': 0, 'Z': 0}, squeeze=False
            )
            # With squeeze=False, integer-selected frame axes are still
            # removed, but spatial dims with size=1 stay
            # The C dimension has size 2, so no squeezing there
            # Result should be (2, 32, 32) - the spatial shape
            expected = data[0, 0]
            numpy.testing.assert_array_equal(result.squeeze(), expected)

    def test_all_frames_selected(self, tzcyx_file):
        """Selecting all frames should match reading the full array."""
        fname, data = tzcyx_file
        result = tifffile.imread(
            fname,
            selection={'T': slice(None), 'Z': slice(None)},
        )
        expected = data
        numpy.testing.assert_array_equal(result, expected)

    def test_mixed_frame_and_spatial(self, tzcyx_file):
        """Select specific frames AND crop spatially."""
        fname, data = tzcyx_file
        result = tifffile.imread(
            fname,
            selection={
                'T': 1,
                'Z': slice(0, 2),
                'Y': slice(10, 20),
                'X': slice(5, 15),
            },
        )
        expected = data[1, 0:2, :, 10:20, 5:15]
        numpy.testing.assert_array_equal(result, expected)


    def test_step_slice(self, tzcyx_file):
        """Select every other Z plane."""
        fname, data = tzcyx_file
        result = tifffile.imread(fname, selection={'T': 1, 'Z': slice(0, 4, 2)})
        expected = data[1, 0:4:2]
        numpy.testing.assert_array_equal(result, expected)

    def test_step_slice_large(self, tyx_file):
        """Select every 3rd frame."""
        fname, data = tyx_file
        result = tifffile.imread(fname, selection={'T': slice(0, 10, 3)})
        expected = data[0:10:3]
        numpy.testing.assert_array_equal(result, expected)

    def test_negative_step(self, tzcyx_file):
        """Reverse Z order."""
        fname, data = tzcyx_file
        result = tifffile.imread(
            fname, selection={'T': 0, 'Z': slice(3, None, -1)}
        )
        expected = data[0, 3::-1]
        numpy.testing.assert_array_equal(result, expected)

    def test_tuple_with_step(self, tzcyx_file):
        """Tuple-based selection with step."""
        fname, data = tzcyx_file
        result = tifffile.imread(fname, selection=(slice(0, 3, 2), slice(None)))
        expected = data[0:3:2]
        numpy.testing.assert_array_equal(result, expected)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
