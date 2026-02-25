# tifffile.py

# Copyright (c) 2008-2026, Christoph Gohlke
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice,
#    this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

r"""Read and write TIFF files.

Tifffile is a Python library to

(1) store NumPy arrays in TIFF (Tagged Image File Format) files, and
(2) read image and metadata from TIFF-like files used in bioimaging.

Image and metadata can be read from TIFF, BigTIFF, OME-TIFF, GeoTIFF,
Adobe DNG, ZIF (Zoomable Image File Format), MetaMorph STK, Zeiss LSM,
ImageJ hyperstack, Micro-Manager MMStack and NDTiff, SGI, NIHImage,
Olympus FluoView and SIS, ScanImage, Molecular Dynamics GEL,
Aperio SVS, Leica SCN, Roche BIF, PerkinElmer QPTIFF (QPI, PKI),
Hamamatsu NDPI, Argos AVS, Philips DP, and ThermoFisher EER formatted files.

Image data can be read as NumPy arrays or Zarr arrays/groups from strips,
tiles, pages (IFDs), SubIFDs, higher-order series, and pyramidal levels.

Image data can be written to TIFF, BigTIFF, OME-TIFF, and ImageJ hyperstack
compatible files in multi-page, volumetric, pyramidal, memory-mappable,
tiled, predicted, or compressed form.

Many compression and predictor schemes are supported via the imagecodecs
library, including LZW, PackBits, Deflate, PIXTIFF, LZMA, LERC, Zstd,
JPEG (8 and 12-bit, lossless), JPEG 2000, JPEG XR, JPEG XL, WebP, PNG, EER,
Jetraw, 24-bit floating-point, and horizontal differencing.

Tifffile can also be used to inspect TIFF structures, read image data from
multi-dimensional file sequences, write fsspec ReferenceFileSystem for
TIFF files and image file sequences, patch TIFF tag values, and parse
many proprietary metadata formats.

:Author: `Christoph Gohlke <https://www.cgohlke.com>`_
:License: BSD-3-Clause
:Version: 2026.1.28
:DOI: `10.5281/zenodo.6795860 <https://doi.org/10.5281/zenodo.6795860>`_

Quickstart
----------

Install the tifffile package and all dependencies from the
`Python Package Index <https://pypi.org/project/tifffile/>`_::

    python -m pip install -U tifffile[all]

Tifffile is also available in other package repositories such as Anaconda,
Debian, and MSYS2.

The tifffile library is type annotated and documented via docstrings::

    python -c "import tifffile; help(tifffile)"

Tifffile can be used as a console script to inspect and preview TIFF files::

    python -m tifffile --help

See `Examples`_ for using the programming interface.

Source code and support are available on
`GitHub <https://github.com/cgohlke/tifffile>`_.

Support is also provided on the
`image.sc <https://forum.image.sc/tag/tifffile>`_ forum.

Requirements
------------

This revision was tested with the following requirements and dependencies
(other versions may work):

- `CPython <https://www.python.org>`_ 3.11.9, 3.12.10, 3.13.11, 3.14.2 64-bit
- `NumPy <https://pypi.org/project/numpy>`_ 2.4.1
- `Imagecodecs <https://pypi.org/project/imagecodecs/>`_ 2026.1.14
  (required for encoding or decoding LZW, JPEG, etc. compressed segments)
- `Matplotlib <https://pypi.org/project/matplotlib/>`_ 3.10.8
  (required for plotting)
- `Lxml <https://pypi.org/project/lxml/>`_ 6.0.2
  (required only for validating and printing XML)
- `Zarr <https://pypi.org/project/zarr/>`_ 3.1.5
  (required only for using Zarr stores; Zarr 2 is not compatible)
- `Kerchunk <https://pypi.org/project/kerchunk/>`_ 0.2.9
  (required only for opening ReferenceFileSystem files)

Revisions
---------

2026.1.28

- Pass 5128 tests.
- Deprecate colormaped parameter in imagej_description (use colormapped).
- Fix code review issues.

2026.1.14

- Improve code quality.

2025.12.20

- Do not initialize output arrays.

2025.12.12

- Improve code quality.

2025.10.16

- Add option to decode EER super-resolution sub-pixels (breaking, #313).
- Parse EER metadata to dict (breaking).

2025.10.4

- Fix parsing SVS description ending with "|".

2025.9.30

- Fix reading NDTiff series with unordered axes in index (#311).

2025.9.20

- Derive TiffFileError from ValueError.
- Natural-sort files in glob pattern passed to imread by default (breaking).
- Fix optional sorting of list of files passed to FileSequence and imread.

2025.9.9

- Consolidate Nuvu camera metadata.

2025.8.28

- Support DNG DCP files (#306).

2025.6.11

- Fix reading images with dimension length 1 through Zarr (#303).

2025.6.1

- Add experimental option to write iterator of bytes and bytecounts (#301).

2025.5.26

- Use threads in Zarr stores.

2025.5.24

- Fix incorrect tags created by Philips DP v1.1 (#299).
- Make Zarr stores partially listable.

2025.5.21

- Move Zarr stores to tifffile.zarr namespace (breaking).
- Require Zarr 3 for Zarr stores and remove support for Zarr 2 (breaking).
- Drop support for Python 3.10.

2025.5.10

- Raise ValueError when using Zarr 3 (#296).
- Fall back to compression.zstd on Python >= 3.14 if no imagecodecs.
- Remove doctest command line option.
- Support Python 3.14.

2025.3.30

- Fix for imagecodecs 2025.3.30.

2025.3.13

- …

Refer to the CHANGES file for older revisions.

Notes
-----

TIFF, the Tagged Image File Format, was created by the Aldus Corporation and
Adobe Systems Incorporated.

Tifffile supports a subset of the TIFF6 specification, mainly 8, 16, 32, and
64-bit integer, 16, 32, and 64-bit float, grayscale and multi-sample images.
Specifically, CCITT and OJPEG compression, chroma subsampling without JPEG
compression, color space transformations, samples with differing types, or
IPTC, ICC, and XMP metadata are not implemented.

Besides classic TIFF, tifffile supports several TIFF-like formats that do not
strictly adhere to the TIFF6 specification. Some formats allow file and data
sizes to exceed the 4 GB limit of the classic TIFF:

- **BigTIFF** is identified by version number 43 and uses different file
  header, IFD, and tag structures with 64-bit offsets. The format also adds
  64-bit data types. Tifffile can read and write BigTIFF files.
- **ImageJ hyperstacks** store all image data, which may exceed 4 GB,
  contiguously after the first IFD. Files > 4 GB contain one IFD only.
  The size and shape of the up to 6-dimensional image data can be determined
  from the ImageDescription tag of the first IFD, which is Latin-1 encoded.
  Tifffile can read and write ImageJ hyperstacks.
- **OME-TIFF** files store up to 8-dimensional image data in one or multiple
  TIFF or BigTIFF files. The UTF-8 encoded OME-XML metadata found in the
  ImageDescription tag of the first IFD defines the position of TIFF IFDs in
  the high-dimensional image data. Tifffile can read OME-TIFF files (except
  multi-file pyramidal) and write NumPy arrays to single-file OME-TIFF.
- **Micro-Manager NDTiff** stores multi-dimensional image data in one
  or more classic TIFF files. Metadata contained in a separate NDTiff.index
  binary file defines the position of the TIFF IFDs in the image array.
  Each TIFF file also contains metadata in a non-TIFF binary structure at
  offset 8. Downsampled image data of pyramidal datasets are stored in
  separate folders. Tifffile can read NDTiff files. Version 0 and 1 series,
  tiling, stitching, and multi-resolution pyramids are not supported.
- **Micro-Manager MMStack** stores 6-dimensional image data in one or more
  classic TIFF files. Metadata contained in non-TIFF binary structures and
  JSON strings define the image stack dimensions and the position of the image
  frame data in the file and the image stack. The TIFF structures and metadata
  are often corrupted or wrong. Tifffile can read MMStack files.
- **Carl Zeiss LSM** files store all IFDs below 4 GB and wrap around 32-bit
  StripOffsets pointing to image data above 4 GB. The StripOffsets of each
  series and position require separate unwrapping. The StripByteCounts tag
  contains the number of bytes for the uncompressed data. Tifffile can read
  LSM files of any size.
- **MetaMorph STK** files contain additional image planes stored
  contiguously after the image data of the first page. The total number of
  planes is equal to the count of the UIC2tag. Tifffile can read STK files.
- **ZIF**, the Zoomable Image File format, is a subspecification of BigTIFF
  with SGI's ImageDepth extension and additional compression schemes.
  Only little-endian, tiled, interleaved, 8-bit per sample images with
  JPEG, PNG, JPEG XR, and JPEG 2000 compression are allowed. Tifffile can
  read and write ZIF files.
- **Hamamatsu NDPI** files use some 64-bit offsets in the file header, IFD,
  and tag structures. Single, LONG typed tag values can exceed 32-bit.
  The high bytes of 64-bit tag values and offsets are stored after IFD
  structures. Tifffile can read NDPI files > 4 GB.
  JPEG compressed segments with dimensions >65530 or missing restart markers
  cannot be decoded with common JPEG libraries. Tifffile works around this
  limitation by separately decoding the MCUs between restart markers, which
  performs poorly. BitsPerSample, SamplesPerPixel, and
  PhotometricInterpretation tags may contain wrong values, which can be
  corrected using the value of tag 65441.
- **Philips TIFF** slides store padded ImageWidth and ImageLength tag values
  for tiled pages. The values can be corrected using the DICOM_PIXEL_SPACING
  attributes of the XML formatted description of the first page. Tile offsets
  and byte counts may be 0. Tifffile can read Philips slides.
- **Ventana/Roche BIF** slides store tiles and metadata in a BigTIFF container.
  Tiles may overlap and require stitching based on the TileJointInfo elements
  in the XMP tag. Volumetric scans are stored using the ImageDepth extension.
  Tifffile can read BIF and decode individual tiles but does not perform
  stitching.
- **ScanImage** optionally allows corrupted non-BigTIFF files > 2 GB.
  The values of StripOffsets and StripByteCounts can be recovered using the
  constant differences of the offsets of IFD and tag values throughout the
  file. Tifffile can read such files if the image data are stored contiguously
  in each page.
- **GeoTIFF sparse** files allow strip or tile offsets and byte counts to be 0.
  Such segments are implicitly set to 0 or the NODATA value on reading.
  Tifffile can read GeoTIFF sparse files.
- **Tifffile shaped** files store the array shape and user-provided metadata
  of multi-dimensional image series in JSON format in the ImageDescription tag
  of the first page of the series. The format allows multiple series,
  SubIFDs, sparse segments with zero offset and byte count, and truncated
  series, where only the first page of a series is present, and the image data
  are stored contiguously. No other software besides Tifffile supports the
  truncated format.

Other libraries for reading, writing, inspecting, or manipulating scientific
TIFF files from Python are
`bioio <https://github.com/bioio-devs/bioio>`_,
`aicsimageio <https://github.com/AllenCellModeling/aicsimageio>`_,
`apeer-ometiff-library
<https://github.com/apeer-micro/apeer-ometiff-library>`_,
`bigtiff <https://pypi.org/project/bigtiff>`_,
`fabio.TiffIO <https://github.com/silx-kit/fabio>`_,
`GDAL <https://github.com/OSGeo/gdal/>`_,
`imread <https://github.com/luispedro/imread>`_,
`large_image <https://github.com/girder/large_image>`_,
`openslide-python <https://github.com/openslide/openslide-python>`_,
`opentile <https://github.com/imi-bigpicture/opentile>`_,
`pylibtiff <https://github.com/pearu/pylibtiff>`_,
`pylsm <https://launchpad.net/pylsm>`_,
`pymimage <https://github.com/ardoi/pymimage>`_,
`python-bioformats <https://github.com/CellProfiler/python-bioformats>`_,
`pytiff <https://github.com/FZJ-INM1-BDA/pytiff>`_,
`scanimagetiffreader-python
<https://gitlab.com/vidriotech/scanimagetiffreader-python>`_,
`SimpleITK <https://github.com/SimpleITK/SimpleITK>`_,
`slideio <https://gitlab.com/bioslide/slideio>`_,
`tiffslide <https://github.com/bayer-science-for-a-better-life/tiffslide>`_,
`tifftools <https://github.com/DigitalSlideArchive/tifftools>`_,
`tyf <https://github.com/Moustikitos/tyf>`_,
`xtiff <https://github.com/BodenmillerGroup/xtiff>`_, and
`ndtiff <https://github.com/micro-manager/NDTiffStorage>`_.

References
----------

- TIFF 6.0 Specification and Supplements. Adobe Systems Incorporated.
  https://www.adobe.io/open/standards/TIFF.html
  https://download.osgeo.org/libtiff/doc/
- TIFF File Format FAQ. https://www.awaresystems.be/imaging/tiff/faq.html
- The BigTIFF File Format.
  https://www.awaresystems.be/imaging/tiff/bigtiff.html
- MetaMorph Stack (STK) Image File Format.
  http://mdc.custhelp.com/app/answers/detail/a_id/18862
- Image File Format Description LSM 5/7 Release 6.0 (ZEN 2010).
  Carl Zeiss MicroImaging GmbH. BioSciences. May 10, 2011
- The OME-TIFF format.
  https://docs.openmicroscopy.org/ome-model/latest/
- UltraQuant(r) Version 6.0 for Windows Start-Up Guide.
  http://www.ultralum.com/images%20ultralum/pdf/UQStart%20Up%20Guide.pdf
- Micro-Manager File Formats.
  https://micro-manager.org/wiki/Micro-Manager_File_Formats
- ScanImage BigTiff Specification.
  https://docs.scanimage.org/Appendix/ScanImage+BigTiff+Specification.html
- ZIF, the Zoomable Image File format. https://zif.photo/
- GeoTIFF File Format. https://gdal.org/drivers/raster/gtiff.html
- Cloud optimized GeoTIFF.
  https://github.com/cogeotiff/cog-spec/blob/master/spec.md
- Tags for TIFF and Related Specifications. Digital Preservation.
  https://www.loc.gov/preservation/digital/formats/content/tiff_tags.shtml
- CIPA DC-008-2016: Exchangeable image file format for digital still cameras:
  Exif Version 2.31.
  http://www.cipa.jp/std/documents/e/DC-008-Translation-2016-E.pdf
- The EER (Electron Event Representation) file format.
  https://github.com/fei-company/EerReaderLib
- Digital Negative (DNG) Specification. Version 1.7.1.0, September 2023.
  https://helpx.adobe.com/content/dam/help/en/photoshop/pdf/DNG_Spec_1_7_1_0.pdf
- Roche Digital Pathology. BIF image file format for digital pathology.
  https://diagnostics.roche.com/content/dam/diagnostics/Blueprint/en/pdf/rmd/Roche-Digital-Pathology-BIF-Whitepaper.pdf
- Astro-TIFF specification. https://astro-tiff.sourceforge.io/
- Aperio Technologies, Inc. Digital Slides and Third-Party Data Interchange.
  Aperio_Digital_Slides_and_Third-party_data_interchange.pdf
- PerkinElmer image format.
  https://downloads.openmicroscopy.org/images/Vectra-QPTIFF/perkinelmer/PKI_Image%20Format.docx
- NDTiffStorage. https://github.com/micro-manager/NDTiffStorage
- Argos AVS File Format.
  https://github.com/user-attachments/files/15580286/ARGOS.AVS.File.Format.pdf

Examples
--------

Write a NumPy array to a single-page RGB TIFF file:

>>> import numpy
>>> data = numpy.random.randint(0, 255, (256, 256, 3), 'uint8')
>>> imwrite('temp.tif', data, photometric='rgb')

Read the image from the TIFF file as NumPy array:

>>> image = imread('temp.tif')
>>> image.shape
(256, 256, 3)

Use the `photometric` and `planarconfig` arguments to write a 3x3x3 NumPy
array to an interleaved RGB, a planar RGB, or a 3-page grayscale TIFF:

>>> data = numpy.random.randint(0, 255, (3, 3, 3), 'uint8')
>>> imwrite('temp.tif', data, photometric='rgb')
>>> imwrite('temp.tif', data, photometric='rgb', planarconfig='separate')
>>> imwrite('temp.tif', data, photometric='minisblack')

Use the `extrasamples` argument to specify how extra components are
interpreted, for example, for an RGBA image with unassociated alpha channel:

>>> data = numpy.random.randint(0, 255, (256, 256, 4), 'uint8')
>>> imwrite('temp.tif', data, photometric='rgb', extrasamples=['unassalpha'])

Write a 3-dimensional NumPy array to a multi-page, 16-bit grayscale TIFF file:

>>> data = numpy.random.randint(0, 2**12, (64, 301, 219), 'uint16')
>>> imwrite('temp.tif', data, photometric='minisblack')

Read the whole image stack from the multi-page TIFF file as NumPy array:

>>> image_stack = imread('temp.tif')
>>> image_stack.shape
(64, 301, 219)
>>> image_stack.dtype
dtype('uint16')

Read the image from the first page in the TIFF file as NumPy array:

>>> image = imread('temp.tif', key=0)
>>> image.shape
(301, 219)

Read images from a selected range of pages:

>>> images = imread('temp.tif', key=range(4, 40, 2))
>>> images.shape
(18, 301, 219)

Iterate over all pages in the TIFF file and successively read images:

>>> with TiffFile('temp.tif') as tif:
...     for page in tif.pages:
...         image = page.asarray()
...

Get information about the image stack in the TIFF file without reading
any image data:

>>> tif = TiffFile('temp.tif')
>>> len(tif.pages)  # number of pages in the file
64
>>> page = tif.pages[0]  # get shape and dtype of image in first page
>>> page.shape
(301, 219)
>>> page.dtype
dtype('uint16')
>>> page.axes
'YX'
>>> series = tif.series[0]  # get shape and dtype of first image series
>>> series.shape
(64, 301, 219)
>>> series.dtype
dtype('uint16')
>>> series.axes
'QYX'
>>> tif.close()

Inspect the "XResolution" tag from the first page in the TIFF file:

>>> with TiffFile('temp.tif') as tif:
...     tag = tif.pages[0].tags['XResolution']
...
>>> tag.value
(1, 1)
>>> tag.name
'XResolution'
>>> tag.code
282
>>> tag.count
1
>>> tag.dtype
<DATATYPE.RATIONAL: 5>

Iterate over all tags in the TIFF file:

>>> with TiffFile('temp.tif') as tif:
...     for page in tif.pages:
...         for tag in page.tags:
...             tag_name, tag_value = tag.name, tag.value
...

Overwrite the value of an existing tag, for example, XResolution:

>>> with TiffFile('temp.tif', mode='r+') as tif:
...     _ = tif.pages[0].tags['XResolution'].overwrite((96000, 1000))
...

Write a 5-dimensional floating-point array using BigTIFF format, separate
color components, tiling, Zlib compression level 8, horizontal differencing
predictor, and additional metadata:

>>> data = numpy.random.rand(2, 5, 3, 301, 219).astype('float32')
>>> imwrite(
...     'temp.tif',
...     data,
...     bigtiff=True,
...     photometric='rgb',
...     planarconfig='separate',
...     tile=(32, 32),
...     compression='zlib',
...     compressionargs={'level': 8},
...     predictor=True,
...     metadata={'axes': 'TZCYX'},
... )

Write a 10 fps time series of volumes with xyz voxel size 2.6755x2.6755x3.9474
micron^3 to an ImageJ hyperstack formatted TIFF file:

>>> volume = numpy.random.randn(6, 57, 256, 256).astype('float32')
>>> image_labels = [f'{i}' for i in range(volume.shape[0] * volume.shape[1])]
>>> imwrite(
...     'temp.tif',
...     volume,
...     imagej=True,
...     resolution=(1.0 / 2.6755, 1.0 / 2.6755),
...     metadata={
...         'spacing': 3.947368,
...         'unit': 'um',
...         'finterval': 1 / 10,
...         'fps': 10.0,
...         'axes': 'TZYX',
...         'Labels': image_labels,
...     },
... )

Read the volume and metadata from the ImageJ hyperstack file:

>>> with TiffFile('temp.tif') as tif:
...     volume = tif.asarray()
...     axes = tif.series[0].axes
...     imagej_metadata = tif.imagej_metadata
...
>>> volume.shape
(6, 57, 256, 256)
>>> axes
'TZYX'
>>> imagej_metadata['slices']
57
>>> imagej_metadata['frames']
6

Memory-map the contiguous image data in the ImageJ hyperstack file:

>>> memmap_volume = memmap('temp.tif')
>>> memmap_volume.shape
(6, 57, 256, 256)
>>> del memmap_volume

Create a TIFF file containing an empty image and write to the memory-mapped
NumPy array (note: this does not work with compression or tiling):

>>> memmap_image = memmap(
...     'temp.tif', shape=(256, 256, 3), dtype='float32', photometric='rgb'
... )
>>> type(memmap_image)
<class 'numpy.memmap'>
>>> memmap_image[255, 255, 1] = 1.0
>>> memmap_image.flush()
>>> del memmap_image

Write two NumPy arrays to a multi-series TIFF file (note: other TIFF readers
will not recognize the two series; use the OME-TIFF format for better
interoperability):

>>> series0 = numpy.random.randint(0, 255, (32, 32, 3), 'uint8')
>>> series1 = numpy.random.randint(0, 255, (4, 256, 256), 'uint16')
>>> with TiffWriter('temp.tif') as tif:
...     tif.write(series0, photometric='rgb')
...     tif.write(series1, photometric='minisblack')
...

Read the second image series from the TIFF file:

>>> series1 = imread('temp.tif', series=1)
>>> series1.shape
(4, 256, 256)

Successively write the frames of one contiguous series to a TIFF file:

>>> data = numpy.random.randint(0, 255, (30, 301, 219), 'uint8')
>>> with TiffWriter('temp.tif') as tif:
...     for frame in data:
...         tif.write(frame, contiguous=True)
...

Append an image series to the existing TIFF file (note: this does not work
with ImageJ hyperstack or OME-TIFF files):

>>> data = numpy.random.randint(0, 255, (301, 219, 3), 'uint8')
>>> imwrite('temp.tif', data, photometric='rgb', append=True)

Create a TIFF file from a generator of tiles:

>>> data = numpy.random.randint(0, 2**12, (31, 33, 3), 'uint16')
>>> def tiles(data, tileshape):
...     for y in range(0, data.shape[0], tileshape[0]):
...         for x in range(0, data.shape[1], tileshape[1]):
...             yield data[y : y + tileshape[0], x : x + tileshape[1]]
...
>>> imwrite(
...     'temp.tif',
...     tiles(data, (16, 16)),
...     tile=(16, 16),
...     shape=data.shape,
...     dtype=data.dtype,
...     photometric='rgb',
... )

Write a multi-dimensional, multi-resolution (pyramidal), multi-series OME-TIFF
file with optional metadata. Sub-resolution images are written to SubIFDs.
Limit parallel encoding to 2 threads. Write a thumbnail image as a separate
image series:

>>> data = numpy.random.randint(0, 255, (8, 2, 512, 512, 3), 'uint8')
>>> subresolutions = 2
>>> pixelsize = 0.29  # micrometer
>>> with TiffWriter('temp.ome.tif', bigtiff=True) as tif:
...     metadata = {
...         'axes': 'TCYXS',
...         'SignificantBits': 8,
...         'TimeIncrement': 0.1,
...         'TimeIncrementUnit': 's',
...         'PhysicalSizeX': pixelsize,
...         'PhysicalSizeXUnit': 'µm',
...         'PhysicalSizeY': pixelsize,
...         'PhysicalSizeYUnit': 'µm',
...         'Channel': {'Name': ['Channel 1', 'Channel 2']},
...         'Plane': {'PositionX': [0.0] * 16, 'PositionXUnit': ['µm'] * 16},
...         'Description': 'A multi-dimensional, multi-resolution image',
...         'MapAnnotation': {  # for OMERO
...             'Namespace': 'openmicroscopy.org/PyramidResolution',
...             '1': '256 256',
...             '2': '128 128',
...         },
...     }
...     options = dict(
...         photometric='rgb',
...         tile=(128, 128),
...         compression='jpeg',
...         resolutionunit='CENTIMETER',
...         maxworkers=2,
...     )
...     tif.write(
...         data,
...         subifds=subresolutions,
...         resolution=(1e4 / pixelsize, 1e4 / pixelsize),
...         metadata=metadata,
...         **options,
...     )
...     # write pyramid levels to the two subifds
...     # in production use resampling to generate sub-resolution images
...     for level in range(subresolutions):
...         mag = 2 ** (level + 1)
...         tif.write(
...             data[..., ::mag, ::mag, :],
...             subfiletype=1,  # FILETYPE.REDUCEDIMAGE
...             resolution=(1e4 / mag / pixelsize, 1e4 / mag / pixelsize),
...             **options,
...         )
...     # add a thumbnail image as a separate series
...     # it is recognized by QuPath as an associated image
...     thumbnail = (data[0, 0, ::8, ::8] >> 2).astype('uint8')
...     tif.write(thumbnail, metadata={'Name': 'thumbnail'})
...

Access the image levels in the pyramidal OME-TIFF file:

>>> baseimage = imread('temp.ome.tif')
>>> second_level = imread('temp.ome.tif', series=0, level=1)
>>> with TiffFile('temp.ome.tif') as tif:
...     baseimage = tif.series[0].asarray()
...     second_level = tif.series[0].levels[1].asarray()
...     number_levels = len(tif.series[0].levels)  # includes base level
...

Iterate over and decode single JPEG compressed tiles in the TIFF file:

>>> with TiffFile('temp.ome.tif') as tif:
...     fh = tif.filehandle
...     for page in tif.pages:
...         for index, (offset, bytecount) in enumerate(
...             zip(page.dataoffsets, page.databytecounts)
...         ):
...             _ = fh.seek(offset)
...             data = fh.read(bytecount)
...             tile, indices, shape = page.decode(
...                 data, index, jpegtables=page.jpegtables
...             )
...

Use Zarr to read parts of the tiled, pyramidal images in the TIFF file:

>>> import zarr
>>> store = imread('temp.ome.tif', aszarr=True)
>>> z = zarr.open(store, mode='r')
>>> z
<Group ZarrTiffStore>
>>> z['0']  # base layer
 <Array ZarrTiffStore/0 shape=(8, 2, 512, 512, 3) dtype=uint8>
>>> z['0'][2, 0, 128:384, 256:].shape  # read a tile from the base layer
(256, 256, 3)
>>> store.close()

Load the base layer from the Zarr store as a dask array:

>>> import dask.array
>>> store = imread('temp.ome.tif', aszarr=True)
>>> dask.array.from_zarr(store, '0', zarr_format=2)
dask.array<...shape=(8, 2, 512, 512, 3)...chunksize=(1, 1, 128, 128, 3)...
>>> store.close()

Write the Zarr store to a fsspec ReferenceFileSystem in JSON format:

>>> store = imread('temp.ome.tif', aszarr=True)
>>> store.write_fsspec('temp.ome.tif.json', url='file://')
>>> store.close()

Open the fsspec ReferenceFileSystem as a Zarr group:

>>> from kerchunk.utils import refs_as_store
>>> import imagecodecs.numcodecs
>>> imagecodecs.numcodecs.register_codecs(verbose=False)
>>> z = zarr.open(refs_as_store('temp.ome.tif.json'), mode='r')
>>> z
<Group <FsspecStore(ReferenceFileSystem, /)>>

Create an OME-TIFF file containing an empty, tiled image series and write
to it via the Zarr interface (note: this does not work with compression):

>>> imwrite(
...     'temp2.ome.tif',
...     shape=(8, 800, 600),
...     dtype='uint16',
...     photometric='minisblack',
...     tile=(128, 128),
...     metadata={'axes': 'CYX'},
... )
>>> store = imread('temp2.ome.tif', mode='r+', aszarr=True)
>>> z = zarr.open(store, mode='r+')
>>> z
<Array ZarrTiffStore shape=(8, 800, 600) dtype=uint16>
>>> z[3, 100:200, 200:300:2] = 1024
>>> store.close()

Read images from a sequence of TIFF files as NumPy array using two I/O worker
threads:

>>> imwrite('temp_C001T001.tif', numpy.random.rand(64, 64))
>>> imwrite('temp_C001T002.tif', numpy.random.rand(64, 64))
>>> image_sequence = imread(
...     ['temp_C001T001.tif', 'temp_C001T002.tif'], ioworkers=2, maxworkers=1
... )
>>> image_sequence.shape
(2, 64, 64)
>>> image_sequence.dtype
dtype('float64')

Read an image stack from a series of TIFF files with a file name pattern
as NumPy or Zarr arrays:

>>> image_sequence = TiffSequence('temp_C0*.tif', pattern=r'_(C)(\d+)(T)(\d+)')
>>> image_sequence.shape
(1, 2)
>>> image_sequence.axes
'CT'
>>> data = image_sequence.asarray()
>>> data.shape
(1, 2, 64, 64)
>>> store = image_sequence.aszarr()
>>> zarr.open(store, mode='r', ioworkers=2, maxworkers=1)
<Array ZarrFileSequenceStore shape=(1, 2, 64, 64) dtype=float64>
>>> image_sequence.close()

Write the Zarr store to a fsspec ReferenceFileSystem in JSON format:

>>> store = image_sequence.aszarr()
>>> store.write_fsspec('temp.json', url='file://')

Open the fsspec ReferenceFileSystem as a Zarr array:

>>> from kerchunk.utils import refs_as_store
>>> import tifffile.numcodecs
>>> tifffile.numcodecs.register_codec()
>>> zarr.open(refs_as_store('temp.json'), mode='r')
<Array <FsspecStore(ReferenceFileSystem, /)> shape=(1, 2, 64, 64) ...>

Inspect the TIFF file from the command line::

    $ python -m tifffile temp.ome.tif

"""

from __future__ import annotations

__version__ = '2026.1.28'

__all__ = [
    'CHUNKMODE',
    'COMPRESSION',
    'DATATYPE',
    'EXTRASAMPLE',
    'FILETYPE',
    'FILLORDER',
    'OFILETYPE',
    'ORIENTATION',
    'PHOTOMETRIC',
    'PLANARCONFIG',
    'PREDICTOR',
    'RESUNIT',
    'SAMPLEFORMAT',
    'TIFF',
    '_TIFF',  # private
    'FileCache',
    'FileHandle',
    'FileSequence',
    'NullContext',
    'OmeXml',
    'OmeXmlError',
    'StoredShape',
    'TiffFile',
    'TiffFileError',
    'TiffFormat',
    'TiffFrame',
    'TiffPage',
    'TiffPageSeries',
    'TiffPages',
    'TiffReader',
    'TiffSequence',
    'TiffTag',
    'TiffTagRegistry',
    'TiffTags',
    'TiffWriter',
    'TiledSequence',
    'Timer',
    '__version__',
    'askopenfilename',
    'astype',
    'create_output',
    'enumarg',
    'enumstr',
    'format_size',
    'hexdump',
    'imagej_description',
    'imagej_metadata_tag',
    'imread',
    'imshow',
    'imwrite',
    'logger',
    'lsm2bin',
    'matlabstr2py',
    'memmap',
    'natural_sorted',
    'nullfunc',
    'parse_filenames',
    'parse_kwargs',
    'pformat',
    'product',
    'read_gdal_structural_metadata',
    'read_micromanager_metadata',
    'read_ndtiff_index',
    'read_scanimage_metadata',
    'repeat_nd',
    'reshape_axes',
    'reshape_nd',
    'stripnull',  # deprecated
    'strptime',
    'tiff2fsspec',
    'tiff2tiled',
    'tiff2pyramid',
    'tiffcomment',
    'transpose_axes',
    'update_kwargs',
    'validate_jhove',
    'xml2dict',
]

import binascii
import collections
import contextlib
import enum
import glob
import io
import json
import logging
import math
import os
import re
import struct
import sys
import threading
import time
import warnings
from collections.abc import Callable, Iterable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime as DateTime  # noqa: N812
from datetime import timedelta as TimeDelta  # noqa: N812
from functools import cached_property

import numpy

try:
    import imagecodecs
except ImportError:
    # load pure Python implementation of some codecs
    try:
        from . import _imagecodecs as imagecodecs  # type: ignore[no-redef]
    except ImportError:
        import _imagecodecs as imagecodecs  # type: ignore[no-redef]

from typing import IO, TYPE_CHECKING, cast, final, overload

if TYPE_CHECKING:
    from collections.abc import Collection, Container, Iterator
    from types import TracebackType
    from typing import Any, Literal, Self, TypeAlias

    from numpy.typing import ArrayLike, DTypeLike, NDArray

    ByteOrder: TypeAlias = Literal['>', '<']
    OutputType: TypeAlias = str | IO[bytes] | NDArray[Any] | None
    TagTuple: TypeAlias = tuple[int | str, int | str, int | None, Any, bool]

from .enums import (  # noqa: E402
    CHUNKMODE,
    COMPRESSION,
    DATATYPE,
    EXTRASAMPLE,
    FILETYPE,
    FILLORDER,
    OFILETYPE,
    ORIENTATION,
    PHOTOMETRIC,
    PLANARCONFIG,
    PREDICTOR,
    RESUNIT,
    SAMPLEFORMAT,
)
from .codecs import (  # noqa: E402
    CompressionCodec,
    PredictorCodec,
    TiffFormat,
)
from .utils import (  # noqa: E402
    asbool,
    astype,
    byteorder_compare,
    byteorder_isnative,
    bytes2str,
    bytestr,
    check_shape,
    clean_whitespace,
    enumarg,
    enumstr,
    epics_datetime,
    excel_datetime,
    format_size,
    hexdump,
    identityfunc,
    indent,
    isprintable,
    iter_images,
    julian_datetime,
    kwargs_notnone,
    logger,
    matlabstr2py,
    natural_sorted,
    nullfunc,
    order_axes,
    parse_kwargs,
    peek_iterator,
    pformat,
    pformat_xml,
    product,
    rational,
    recarray2dict,
    reorient,
    repeat_nd,
    reshape_axes,
    reshape_nd,
    sequence,
    snipstr,
    squeeze_axes,
    stripascii,
    stripnull,
    strptime,
    subresolution,
    transpose_axes,
    unique_strings,
    unpack_rgb,
    apply_colormap,
    update_kwargs,
    xml2dict,
)
from .fileio import (  # noqa: E402
    FileCache,
    FileHandle,
    NullContext,
    StoredShape,
    Timer,
)
from .metadata import (  # noqa: E402
    OmeXml,
    OmeXmlError,
    astrotiff_description_metadata,
    eer_xml_metadata,
    fluoview_description_metadata,
    imagej_description,
    imagej_description_metadata,
    imagej_metadata,
    imagej_metadata_tag,
    imagej_shape,
    jpeg_decode_colorspace,
    jpeg_shape,
    metaseries_description_metadata,
    ndpi_jpeg_tile,
    olympus_ini_metadata,
    pilatus_description_metadata,
    read_bytes,
    read_colormap,
    read_cz_lsminfo,
    read_cz_sem,
    read_exif_ifd,
    read_fei_metadata,
    read_gdal_structural_metadata,
    read_gps_ifd,
    read_interoperability_ifd,
    read_json,
    read_lsm_channelcolors,
    read_lsm_channeldatatypes,
    read_lsm_channelwavelength,
    read_lsm_eventlist,
    read_lsm_lookuptable,
    read_lsm_positions,
    read_lsm_scaninfo,
    read_lsm_timestamps,
    read_metaseries_catalog,
    read_micromanager_metadata,
    read_mm_header,
    read_mm_stamp,
    read_ndtiff_index,
    read_nih_image_header,
    read_numpy,
    read_scanimage_metadata,
    read_sis,
    read_sis_ini,
    read_tags,
    read_tvips_header,
    read_uic1tag,
    read_uic2tag,
    read_uic3tag,
    read_uic4tag,
    read_uic_tag,
    read_uic_property,
    read_utf8,
    scanimage_artist_metadata,
    scanimage_description_metadata,
    shaped_description,
    shaped_description_metadata,
    stk_description_metadata,
    streak_description_metadata,
    svs_description_metadata,
)
from .tags import TiffTag, TiffTagRegistry, TiffTags  # noqa: E402
from .sequences import (  # noqa: E402
    FileSequence,
    TiffPageSeries,
    TiffSequence,
    TiledSequence,
)


@overload
def imread(
    files: (
        str
        | os.PathLike[Any]
        | FileHandle
        | IO[bytes]
        | Sequence[str | os.PathLike[Any]]
        | None
    ) = None,
    *,
    selection: Any | None = None,  # TODO: type this
    aszarr: Literal[False] = ...,
    key: int | slice | Iterable[int] | None = None,
    series: int | None = None,
    level: int | None = None,
    squeeze: bool | None = None,
    maxworkers: int | None = None,
    buffersize: int | None = None,
    mode: Literal['r', 'r+'] | None = None,
    name: str | None = None,
    offset: int | None = None,
    size: int | None = None,
    pattern: str | None = None,
    axesorder: Sequence[int] | None = None,
    categories: dict[str, dict[str, int]] | None = None,
    imread: Callable[..., NDArray[Any]] | None = None,
    imreadargs: dict[str, Any] | None = None,
    sort: Callable[..., Any] | bool | None = None,
    container: str | os.PathLike[Any] | None = None,
    chunkshape: tuple[int, ...] | None = None,
    chunkdtype: DTypeLike | None = None,
    axestiled: dict[int, int] | Sequence[tuple[int, int]] | None = None,
    ioworkers: int | None = 1,
    chunkmode: CHUNKMODE | int | str | None = None,
    fillvalue: float | None = None,
    zattrs: dict[str, Any] | None = None,
    multiscales: bool | None = None,
    omexml: str | None = None,
    superres: int | None = None,
    out: OutputType = None,
    device: str | None = None,
    out_inplace: bool | None = None,
    _multifile: bool | None = None,
    _useframes: bool | None = None,
    **kwargs: Any,
) -> NDArray[Any]: ...


@overload
def imread(
    files: (
        str
        | os.PathLike[Any]
        | FileHandle
        | IO[bytes]
        | Sequence[str | os.PathLike[Any]]
        | None
    ) = None,
    *,
    selection: Any | None = None,  # TODO: type this
    aszarr: Literal[True],
    key: int | slice | Iterable[int] | None = None,
    series: int | None = None,
    level: int | None = None,
    squeeze: bool | None = None,
    maxworkers: int | None = None,
    buffersize: int | None = None,
    mode: Literal['r', 'r+'] | None = None,
    name: str | None = None,
    offset: int | None = None,
    size: int | None = None,
    pattern: str | None = None,
    axesorder: Sequence[int] | None = None,
    categories: dict[str, dict[str, int]] | None = None,
    imread: Callable[..., NDArray[Any]] | None = None,
    imreadargs: dict[str, Any] | None = None,
    sort: Callable[..., Any] | bool | None = None,
    container: str | os.PathLike[Any] | None = None,
    chunkshape: tuple[int, ...] | None = None,
    chunkdtype: DTypeLike | None = None,
    axestiled: dict[int, int] | Sequence[tuple[int, int]] | None = None,
    ioworkers: int | None = 1,
    chunkmode: CHUNKMODE | int | str | None = None,
    fillvalue: float | None = None,
    zattrs: dict[str, Any] | None = None,
    multiscales: bool | None = None,
    omexml: str | None = None,
    superres: int | None = None,
    out: OutputType = None,
    device: str | None = None,
    out_inplace: bool | None = None,
    _multifile: bool | None = None,
    _useframes: bool | None = None,
    **kwargs: Any,
) -> ZarrTiffStore | ZarrFileSequenceStore: ...


@overload
def imread(
    files: (
        str
        | os.PathLike[Any]
        | FileHandle
        | IO[bytes]
        | Sequence[str | os.PathLike[Any]]
        | None
    ) = None,
    *,
    selection: Any | None = None,  # TODO: type this
    aszarr: bool = False,
    key: int | slice | Iterable[int] | None = None,
    series: int | None = None,
    level: int | None = None,
    squeeze: bool | None = None,
    maxworkers: int | None = None,
    buffersize: int | None = None,
    mode: Literal['r', 'r+'] | None = None,
    name: str | None = None,
    offset: int | None = None,
    size: int | None = None,
    pattern: str | None = None,
    axesorder: Sequence[int] | None = None,
    categories: dict[str, dict[str, int]] | None = None,
    imread: Callable[..., NDArray[Any]] | None = None,
    imreadargs: dict[str, Any] | None = None,
    sort: Callable[..., Any] | bool | None = None,
    container: str | os.PathLike[Any] | None = None,
    chunkshape: tuple[int, ...] | None = None,
    chunkdtype: DTypeLike | None = None,
    axestiled: dict[int, int] | Sequence[tuple[int, int]] | None = None,
    ioworkers: int | None = 1,
    chunkmode: CHUNKMODE | int | str | None = None,
    fillvalue: float | None = None,
    zattrs: dict[str, Any] | None = None,
    multiscales: bool | None = None,
    omexml: str | None = None,
    superres: int | None = None,
    out: OutputType = None,
    device: str | None = None,
    out_inplace: bool | None = None,
    _multifile: bool | None = None,
    _useframes: bool | None = None,
    **kwargs: Any,
) -> NDArray[Any] | ZarrTiffStore | ZarrFileSequenceStore: ...


def imread(
    files: (
        str
        | os.PathLike[Any]
        | FileHandle
        | IO[bytes]
        | Sequence[str | os.PathLike[Any]]
        | None
    ) = None,
    *,
    selection: Any | None = None,  # TODO: type this
    aszarr: bool = False,
    key: int | slice | Iterable[int] | None = None,
    series: int | None = None,
    level: int | None = None,
    squeeze: bool | None = None,
    maxworkers: int | None = None,
    buffersize: int | None = None,
    mode: Literal['r', 'r+'] | None = None,
    name: str | None = None,
    offset: int | None = None,
    size: int | None = None,
    pattern: str | None = None,
    axesorder: Sequence[int] | None = None,
    categories: dict[str, dict[str, int]] | None = None,
    imread: Callable[..., NDArray[Any]] | None = None,
    imreadargs: dict[str, Any] | None = None,
    sort: Callable[..., Any] | bool | None = None,
    container: str | os.PathLike[Any] | None = None,
    chunkshape: tuple[int, ...] | None = None,
    chunkdtype: DTypeLike | None = None,
    axestiled: dict[int, int] | Sequence[tuple[int, int]] | None = None,
    ioworkers: int | None = 1,
    chunkmode: CHUNKMODE | int | str | None = None,
    fillvalue: float | None = None,
    zattrs: dict[str, Any] | None = None,
    multiscales: bool | None = None,
    omexml: str | None = None,
    superres: int | None = None,
    out: OutputType = None,
    device: str | None = None,
    out_inplace: bool | None = None,
    _multifile: bool | None = None,
    _useframes: bool | None = None,
    **kwargs: Any,
) -> NDArray[Any] | ZarrTiffStore | ZarrFileSequenceStore:
    """Return image from TIFF file(s) as NumPy array or Zarr store.

    The first image series in the file(s) is returned by default.

    Parameters:
        files:
            File name, seekable binary stream, glob pattern, or sequence of
            file names. May be *None* if `container` is specified.
        selection:
            Subset of image to be extracted.
            If not None, a Zarr array is created, indexed with the
            `selection` value, and returned as a NumPy array. Only segments
            that are part of the selection will be read from file.
            Refer to the Zarr documentation for valid selections.
            Depending on selection size, image size, and storage properties,
            it may be more efficient to read the whole image from file and
            then index it.
        aszarr:
            Return file sequences, series, or single pages as Zarr store
            instead of NumPy array if `selection` is None.
        mode, name, offset, size, superres, omexml, _multifile, _useframes:
            Passed to :py:class:`TiffFile`.
        key, series, level, squeeze, maxworkers, buffersize:
            Passed to :py:meth:`TiffFile.asarray`
            or :py:meth:`TiffFile.aszarr`.
        imread, container, sort, pattern, axesorder, axestiled, categories:
            Passed to :py:class:`FileSequence`.
        chunkmode, fillvalue, zattrs, multiscales:
            Passed to :py:class:`ZarrTiffStore`
            or :py:class:`ZarrFileSequenceStore`.
        chunkshape, chunkdtype, ioworkers:
            Passed to :py:meth:`FileSequence.asarray` or
            :py:class:`ZarrFileSequenceStore`.
        out_inplace:
            Passed to :py:meth:`FileSequence.asarray`
        out:
            Passed to :py:meth:`TiffFile.asarray`,
            :py:meth:`FileSequence.asarray`, or :py:func:`zarr_selection`.
        device:
            If not *None*, return a ``torch.Tensor`` on the specified device
            instead of a NumPy array. For example, ``'cuda'`` or ``'cuda:0'``.
            Requires PyTorch. Unsigned integer dtypes not supported by torch
            (uint16, uint32, uint64) are upcast to the next signed type.
            Has no effect when ``aszarr`` is *True*.
        imreadargs:
            Additional arguments passed to :py:attr:`FileSequence.imread`.
        **kwargs:
            Additional arguments passed to :py:class:`TiffFile` or
            :py:attr:`FileSequence.imread`.

    Returns:
        Images from specified files, series, or pages.
        Zarr store instances must be closed after use.
        See :py:meth:`TiffPage.asarray` for operations that are applied
        (or not) to the image data stored in the file.

    """
    store: ZarrStore
    # Route dict/tuple selections natively; other selections use zarr.
    # When key is specified alongside selection, fall through to zarr
    # since native selection works at the series level, not page level.
    native_selection = None
    if selection is not None and not aszarr:
        if isinstance(selection, (dict, tuple)) and key is None:
            native_selection = selection
            selection = None
        else:
            aszarr = True
    elif selection is not None:
        aszarr = True
    is_flags = parse_kwargs(kwargs, *(k for k in kwargs if k[:3] == 'is_'))

    if imread is None and kwargs:
        msg = 'imread() got unexpected keyword arguments ' + ', '.join(
            f"'{key}'" for key in kwargs
        )
        raise TypeError(msg)

    glob_pattern: str | None = None
    if container is None:
        if isinstance(files, str) and ('*' in files or '?' in files):
            glob_pattern = files
            files = glob.glob(files)
        if not files:
            msg = 'no files found'
            raise ValueError(msg)

        if (
            isinstance(files, Sequence)
            and not isinstance(files, str)
            and len(files) == 1
        ):
            files = files[0]

        if isinstance(files, str) or not isinstance(files, Sequence):
            with TiffFile(
                files,
                mode=mode,
                name=name,
                offset=offset,
                size=size,
                omexml=omexml,
                superres=superres,
                _multifile=_multifile,
                _useframes=_useframes,
                **is_flags,
            ) as tif:
                if aszarr:
                    assert key is None or isinstance(key, int)
                    store = tif.aszarr(
                        key=key,
                        series=series,
                        level=level,
                        squeeze=squeeze,
                        maxworkers=maxworkers,
                        buffersize=buffersize,
                        chunkmode=chunkmode,
                        fillvalue=fillvalue,
                        zattrs=zattrs,
                        multiscales=multiscales,
                    )
                    if selection is None:
                        return store

                    from .zarr import zarr_selection

                    return zarr_selection(store, selection, out=out)
                return tif.asarray(
                    key=key,
                    series=series,
                    level=level,
                    squeeze=squeeze,
                    selection=native_selection,
                    maxworkers=maxworkers,
                    buffersize=buffersize,
                    out=out,
                    device=device,
                )

    elif isinstance(files, (FileHandle, IO)):
        msg = 'BinaryIO not supported'
        raise ValueError(msg)

    imread_kwargs = kwargs_notnone(
        key=key,
        series=series,
        level=level,
        squeeze=squeeze,
        maxworkers=maxworkers,
        buffersize=buffersize,
        imreadargs=imreadargs,
        _multifile=_multifile,
        _useframes=_useframes,
        **is_flags,
        **kwargs,
    )

    if glob_pattern is not None:
        # TODO: this forces glob to be executed again
        files = glob_pattern

    with TiffSequence(
        files,
        pattern=pattern,
        axesorder=axesorder,
        categories=categories,
        container=container,
        sort=sort,
        **kwargs_notnone(imread=imread),
    ) as imseq:
        if aszarr:
            store = imseq.aszarr(
                axestiled=axestiled,
                chunkmode=chunkmode,
                chunkshape=chunkshape,
                chunkdtype=chunkdtype,
                fillvalue=fillvalue,
                ioworkers=ioworkers,
                zattrs=zattrs,
                **imread_kwargs,
            )
            if selection is None:
                return store

            from .zarr import zarr_selection

            return zarr_selection(store, selection, out=out)
        result = imseq.asarray(
            axestiled=axestiled,
            chunkshape=chunkshape,
            chunkdtype=chunkdtype,
            ioworkers=ioworkers,
            out_inplace=out_inplace,
            out=out,
            **imread_kwargs,
        )
        if device is not None:
            from .gpu import numpy_to_tensor, parse_device

            dev = parse_device(device)
            if dev is not None:
                return numpy_to_tensor(result, dev)
        return result


def imwrite(
    file: str | os.PathLike[Any] | FileHandle | IO[bytes],
    /,
    data: (
        ArrayLike
        | Iterator[NDArray[Any] | None]
        | Iterator[bytes]
        | Iterator[tuple[bytes, int]]
        | None
    ) = None,
    *,
    mode: Literal['w', 'x', 'r+'] | None = None,
    bigtiff: bool | None = None,
    byteorder: ByteOrder | None = None,
    imagej: bool = False,
    ome: bool | None = None,
    shaped: bool | None = None,
    append: bool = False,
    shape: Sequence[int] | None = None,
    dtype: DTypeLike | None = None,
    photometric: PHOTOMETRIC | int | str | None = None,
    planarconfig: PLANARCONFIG | int | str | None = None,
    extrasamples: Sequence[EXTRASAMPLE | int | str] | None = None,
    volumetric: bool = False,
    tile: Sequence[int] | None = None,
    rowsperstrip: int | None = None,
    bitspersample: int | None = None,
    compression: COMPRESSION | int | str | None = None,
    compressionargs: dict[str, Any] | None = None,
    predictor: PREDICTOR | int | str | bool | None = None,
    subsampling: tuple[int, int] | None = None,
    jpegtables: bytes | None = None,
    iccprofile: bytes | None = None,
    colormap: ArrayLike | None = None,
    description: str | bytes | None = None,
    datetime: str | bool | DateTime | None = None,
    resolution: (
        tuple[float | tuple[int, int], float | tuple[int, int]] | None
    ) = None,
    resolutionunit: RESUNIT | int | str | None = None,
    subfiletype: FILETYPE | int | None = None,
    software: str | bytes | bool | None = None,
    # subifds: int | Sequence[int] | None = None,
    metadata: dict[str, Any] | None = {},  # noqa: B006
    extratags: Sequence[TagTuple] | None = None,
    contiguous: bool = False,
    truncate: bool = False,
    align: int | None = None,
    maxworkers: int | None = None,
    buffersize: int | None = None,
    returnoffset: bool = False,
) -> tuple[int, int] | None:
    """Write NumPy array to TIFF file.

    A BigTIFF file is written if the data size is larger than 4 GB less
    32 MB for metadata, and `bigtiff` is not *False*, and `imagej`,
    `truncate` and `compression` are not enabled.
    Unless `byteorder` is specified, the TIFF file byte order is determined
    from the dtype of `data` or the `dtype` argument.

    Parameters:
        file:
            Passed to :py:class:`TiffWriter`.
        data, shape, dtype:
            Passed to :py:meth:`TiffWriter.write`.
        mode, append, byteorder, bigtiff, imagej, ome, shaped:
            Passed to :py:class:`TiffWriter`.
        photometric, planarconfig, extrasamples, volumetric, tile,\
        rowsperstrip, bitspersample, compression, compressionargs, predictor,\
        subsampling, jpegtables, iccprofile, colormap, description, datetime,\
        resolution, resolutionunit, subfiletype, software,\
        metadata, extratags, maxworkers, buffersize, \
        contiguous, truncate, align:
            Passed to :py:meth:`TiffWriter.write`.
        returnoffset:
            Return offset and number of bytes of memory-mappable image data
            in file.

    Returns:
        If `returnoffset` is *True* and the image data in the file are
        memory-mappable, the offset and number of bytes of the image
        data in the file.

    """
    if data is None:
        # write empty file
        if shape is None or dtype is None:
            msg = "missing required 'shape' or 'dtype' argument"
            raise ValueError(msg)
        dtype = numpy.dtype(dtype)
        shape = tuple(shape)
        datasize = product(shape) * dtype.itemsize
        if byteorder is None:
            byteorder = dtype.byteorder  # type: ignore[assignment]
    else:
        try:
            datasize = data.nbytes  # type: ignore[union-attr]
            if byteorder is None:
                byteorder = data.dtype.byteorder  # type: ignore[union-attr]
        except AttributeError:
            # torch tensors: .nbytes exists but .dtype.byteorder doesn't
            try:
                datasize = data.nelement() * data.element_size()
            except AttributeError:
                datasize = 0

    if bigtiff is None:
        bigtiff = (
            datasize > 2**32 - 2**25
            and not imagej
            and not truncate
            and compression in {None, 0, 1, 'NONE', 'None', 'none'}
        )

    with TiffWriter(
        file,
        mode=mode,
        bigtiff=bigtiff,
        byteorder=byteorder,
        append=append,
        imagej=imagej,
        ome=ome,
        shaped=shaped,
    ) as tif:
        return tif.write(
            data,
            shape=shape,
            dtype=dtype,
            photometric=photometric,
            planarconfig=planarconfig,
            extrasamples=extrasamples,
            volumetric=volumetric,
            tile=tile,
            rowsperstrip=rowsperstrip,
            bitspersample=bitspersample,
            compression=compression,
            compressionargs=compressionargs,
            predictor=predictor,
            subsampling=subsampling,
            jpegtables=jpegtables,
            iccprofile=iccprofile,
            colormap=colormap,
            description=description,
            datetime=datetime,
            resolution=resolution,
            resolutionunit=resolutionunit,
            subfiletype=subfiletype,
            software=software,
            metadata=metadata,
            extratags=extratags,
            contiguous=contiguous,
            truncate=truncate,
            align=align,
            maxworkers=maxworkers,
            buffersize=buffersize,
            returnoffset=returnoffset,
        )


def memmap(
    filename: str | os.PathLike[Any],
    /,
    *,
    shape: Sequence[int] | None = None,
    dtype: DTypeLike | None = None,
    page: int | None = None,
    series: int = 0,
    level: int = 0,
    mode: Literal['r+', 'r', 'c'] = 'r+',
    **kwargs: Any,
) -> numpy.memmap[Any, Any]:
    """Return memory-mapped NumPy array of image data stored in TIFF file.

    Memory-mapping requires the image data stored in native byte order,
    without tiling, compression, predictors, etc.
    If `shape` and `dtype` are provided, existing files are overwritten or
    appended to depending on the `append` argument.
    Else, the image data of a specified page or series in an existing
    file are memory-mapped. By default, the image data of the first
    series are memory-mapped.
    Call `flush` to write any changes in the array to the file.

    Parameters:
        filename:
            Name of TIFF file which stores array.
        shape:
            Shape of empty array.
        dtype:
            Datatype of empty array.
        page:
            Index of page which image data to memory-map.
        series:
            Index of page series which image data to memory-map.
        level:
            Index of pyramid level which image data to memory-map.
        mode:
            Memory-map file open mode. The default is 'r+', which opens
            existing file for reading and writing.
        **kwargs:
            Additional arguments passed to :py:func:`imwrite` or
            :py:class:`TiffFile`.

    Returns:
        Image in TIFF file as memory-mapped NumPy array.

    Raises:
        ValueError: Image data in TIFF file are not memory-mappable.

    """
    filename = os.fspath(filename)
    if shape is not None:
        shape = tuple(shape)
    if shape is not None and dtype is not None:
        # create a new, empty array
        dtype = numpy.dtype(dtype)
        if 'byteorder' in kwargs:
            dtype = dtype.newbyteorder(kwargs['byteorder'])
        kwargs.update(
            data=None,
            shape=shape,
            dtype=dtype,
            align=TIFF.ALLOCATIONGRANULARITY,
            returnoffset=True,
        )
        result = imwrite(filename, **kwargs)
        if result is None:
            # TODO: fail before creating file or writing data
            msg = 'image data are not memory-mappable'
            raise ValueError(msg)
        offset = result[0]
    else:
        # use existing file
        with TiffFile(filename, **kwargs) as tif:
            if page is None:
                tiffseries = tif.series[series].levels[level]
                if tiffseries.dataoffset is None:
                    msg = 'image data are not memory-mappable'
                    raise ValueError(msg)
                shape = tiffseries.shape
                dtype = tiffseries.dtype
                offset = tiffseries.dataoffset
            else:
                tiffpage = tif.pages[page]
                if not tiffpage.is_memmappable:
                    msg = 'image data are not memory-mappable'
                    raise ValueError(msg)
                offset = tiffpage.dataoffsets[0]
                shape = tiffpage.shape
                dtype = tiffpage.dtype
                assert dtype is not None
            dtype = numpy.dtype(tif.byteorder + dtype.char)
    return numpy.memmap(filename, dtype, mode, offset, shape, 'C')


from .utils import TiffFileError  # noqa: E402


@final
class TiffWriter:
    """Write NumPy arrays to TIFF file.

    TiffWriter's main purpose is to save multi-dimensional NumPy arrays in
    TIFF containers, not to create any possible TIFF format.
    Specifically, ExifIFD and GPSIFD tags are not supported.

    TiffWriter instances must be closed with :py:meth:`TiffWriter.close`,
    which is automatically called when using the 'with' context manager.

    TiffWriter instances are not thread-safe. All attributes are read-only.

    Parameters:
        file:
            Specifies file to write.
        mode:
            Binary file open mode if `file` is file name.
            The default is 'w', which opens files for writing, truncating
            existing files.
            'x' opens files for exclusive creation, failing on existing files.
            'r+' opens files for updating, enabling `append`.
        bigtiff:
            Write 64-bit BigTIFF formatted file, which can exceed 4 GB.
            By default, a classic 32-bit TIFF file is written, which is
            limited to 4 GB.
            If `append` is *True*, the format of the existing file is used.
        byteorder:
            Endianness of TIFF format. One of '<', '>', '=', or '|'.
            The default is the system's native byte order.
        append:
            If `file` is existing standard TIFF file, append image data
            and tags to file.
            Parameters `bigtiff` and `byteorder` set from existing file.
            Appending does not scale well with the number of pages already in
            the file and may corrupt specifically formatted TIFF files such as
            OME-TIFF, LSM, STK, ImageJ, or FluoView.
        imagej:
            Write ImageJ hyperstack compatible file if `ome` is not enabled.
            This format can handle data types uint8, uint16, or float32 and
            data shapes up to 6 dimensions in TZCYXS order.
            RGB images (S=3 or S=4) must be `uint8`.
            ImageJ's default byte order is big-endian, but this
            implementation uses the system's native byte order by default.
            ImageJ hyperstacks do not support BigTIFF or compression.
            The ImageJ file format is undocumented.
            Use FIJI's Bio-Formats import function for compressed files.
        ome:
            Write OME-TIFF compatible file.
            By default, the OME-TIFF format is used if the file name extension
            contains '.ome.', `imagej` is not enabled, and the `description`
            argument in the first call of :py:meth:`TiffWriter.write` is not
            specified.
            The format supports multiple image series up to 9 dimensions.
            The default axes order is TZC(S)YX(S).
            Refer to the OME model for restrictions of this format.
        shaped:
            Write tifffile "shaped" compatible file.
            The shape of multi-dimensional images is stored in JSON format in
            a ImageDescription tag of the first page of a series.
            This is the default format used by tifffile unless `imagej` or
            `ome` are enabled or ``metadata=None`` is passed to
            :py:meth:`TiffWriter.write`.

    Raises:
        ValueError:
            The TIFF file cannot be appended to. Use ``append='force'`` to
            force appending, which may result in a corrupted file.

    """

    tiff: TiffFormat
    """Format of TIFF file being written."""

    _fh: FileHandle
    _omexml: OmeXml | None
    _ome: bool | None  # writing OME-TIFF format
    _imagej: bool  # writing ImageJ format
    _tifffile: bool  # writing Tifffile shaped format
    _truncate: bool
    _metadata: dict[str, Any] | None
    _colormap: NDArray[numpy.uint16] | None
    _tags: list[tuple[int, bytes, Any, bool]] | None
    _datashape: tuple[int, ...] | None  # shape of data in consecutive pages
    _datadtype: numpy.dtype[Any] | None  # data type
    _dataoffset: int | None  # offset to data
    _databytecounts: list[int] | None  # byte counts per plane
    _dataoffsetstag: int | None  # strip or tile offset tag code
    _descriptiontag: TiffTag | None  # TiffTag for updating comment
    _ifdoffset: int
    _subifds: int  # number of subifds
    _subifdslevel: int  # index of current subifd level
    _subifdsoffsets: list[int]  # offsets to offsets to subifds
    _nextifdoffsets: list[int]  # offsets to offset to next ifd
    _ifdindex: int  # index of current ifd
    _storedshape: StoredShape | None  # normalized shape in consecutive pages

    def __init__(
        self,
        file: str | os.PathLike[Any] | FileHandle | IO[bytes],
        /,
        *,
        mode: Literal['w', 'x', 'r+'] | None = None,
        bigtiff: bool = False,
        byteorder: ByteOrder | None = None,
        append: bool | str = False,
        imagej: bool = False,
        ome: bool | None = None,
        shaped: bool | None = None,
    ) -> None:
        if mode in {'r+', 'r+b'} or (
            isinstance(file, FileHandle) and file._mode == 'r+b'
        ):
            mode = 'r+'
            append = True
        if append:
            # determine if file is an existing TIFF file that can be extended
            try:
                with FileHandle(file, mode='rb', size=0) as fh:
                    pos = fh.tell()
                    try:
                        with TiffFile(fh) as tif:
                            if append != 'force' and not tif.is_appendable:
                                msg = (
                                    'cannot append to file containing metadata'
                                )
                                raise ValueError(msg)
                            byteorder = tif.byteorder
                            bigtiff = tif.is_bigtiff
                            self._ifdoffset = cast(
                                int, tif.pages.next_page_offset
                            )
                    finally:
                        fh.seek(pos)
                    append = True
            except (OSError, FileNotFoundError):
                append = False

        if append:
            if mode not in {None, 'r+', 'r+b'}:
                msg = "append mode must be 'r+'"
                raise ValueError(msg)
            mode = 'r+'
        elif mode is None:
            mode = 'w'

        if byteorder is None or byteorder in {'=', '|'}:
            byteorder = '<' if sys.byteorder == 'little' else '>'
        elif byteorder not in {'<', '>'}:
            msg = f'invalid {byteorder=}'
            raise ValueError(msg)

        if byteorder == '<':
            self.tiff = TIFF.BIG_LE if bigtiff else TIFF.CLASSIC_LE
        else:
            self.tiff = TIFF.BIG_BE if bigtiff else TIFF.CLASSIC_BE

        self._truncate = False
        self._metadata = None
        self._colormap = None
        self._tags = None
        self._datashape = None
        self._datadtype = None
        self._dataoffset = None
        self._databytecounts = None
        self._dataoffsetstag = None
        self._descriptiontag = None
        self._subifds = 0
        self._subifdslevel = -1
        self._subifdsoffsets = []
        self._nextifdoffsets = []
        self._ifdindex = 0
        self._omexml = None
        self._storedshape = None

        self._fh = FileHandle(file, mode=mode, size=0)
        if append:
            self._fh.seek(0, os.SEEK_END)
        else:
            assert byteorder is not None
            self._fh.write(b'II' if byteorder == '<' else b'MM')
            if bigtiff:
                self._fh.write(struct.pack(byteorder + 'HHH', 43, 8, 0))
            else:
                self._fh.write(struct.pack(byteorder + 'H', 42))
            # first IFD
            self._ifdoffset = self._fh.tell()
            self._fh.write(struct.pack(self.tiff.offsetformat, 0))

        self._ome = None if ome is None else bool(ome)
        self._imagej = False if self._ome else bool(imagej)
        if self._imagej:
            self._ome = False
        if self._ome or self._imagej:
            self._tifffile = False
        else:
            self._tifffile = True if shaped is None else bool(shaped)

        if imagej and bigtiff:
            warnings.warn(
                f'{self!r} writing nonconformant BigTIFF ImageJ',
                UserWarning,
                stacklevel=2,
            )

    def write(
        self,
        data: (
            ArrayLike
            | Iterator[NDArray[Any] | None]
            | Iterator[bytes]
            | Iterator[tuple[bytes, int]]
            | None
        ) = None,
        *,
        shape: Sequence[int] | None = None,
        dtype: DTypeLike | None = None,
        photometric: PHOTOMETRIC | int | str | None = None,
        planarconfig: PLANARCONFIG | int | str | None = None,
        extrasamples: Sequence[EXTRASAMPLE | int | str] | None = None,
        volumetric: bool = False,
        tile: Sequence[int] | None = None,
        rowsperstrip: int | None = None,
        bitspersample: int | None = None,
        compression: COMPRESSION | int | str | bool | None = None,
        compressionargs: dict[str, Any] | None = None,
        predictor: PREDICTOR | int | str | bool | None = None,
        subsampling: tuple[int, int] | None = None,
        jpegtables: bytes | None = None,
        iccprofile: bytes | None = None,
        colormap: ArrayLike | None = None,
        description: str | bytes | None = None,
        datetime: str | bool | DateTime | None = None,
        resolution: (
            tuple[float | tuple[int, int], float | tuple[int, int]] | None
        ) = None,
        resolutionunit: RESUNIT | int | str | None = None,
        subfiletype: FILETYPE | int | None = None,
        software: str | bytes | bool | None = None,
        subifds: int | Sequence[int] | None = None,
        metadata: dict[str, Any] | None = {},  # noqa: B006
        extratags: Sequence[TagTuple] | None = None,
        contiguous: bool = False,
        truncate: bool = False,
        align: int | None = None,
        maxworkers: int | None = None,
        buffersize: int | None = None,
        returnoffset: bool = False,
    ) -> tuple[int, int] | None:
        r"""Write multi-dimensional image to series of TIFF pages.

        Metadata in JSON, ImageJ, or OME-XML format are written to the
        ImageDescription tag of the first page of a series by default,
        such that the image can later be read back as an array of the
        same shape.

        The values of the ImageWidth, ImageLength, ImageDepth, and
        SamplesPerPixel tags are inferred from the last dimensions of the
        data's shape.
        The value of the SampleFormat tag is inferred from the data's dtype.
        Image data are written uncompressed in one strip per plane by default.
        Dimensions higher than 2 to 4 (depending on photometric mode, planar
        configuration, and volumetric mode) are flattened and written as
        separate pages.
        If the data size is zero, write a single page with shape (0, 0).

        Parameters:
            data:
                Specifies image to write.
                If *None*, an empty image is written, which size and type must
                be specified using `shape` and `dtype` arguments.
                This option cannot be used with compression, predictors,
                packed integers, or bilevel images.
                A copy of array-like data is made if it is not a C-contiguous
                NumPy or dask array with the same byteorder as the TIFF file.
                Iterators must yield ndarrays or bytes compatible with the
                file's byteorder as well as the `shape` and `dtype` arguments.
                Iterator bytes must be compatible with the `compression`,
                `predictor`, `subsampling`, and `jpegtables` arguments.
                If `tile` is specified, iterator items must match the tile
                shape. Incomplete tiles are zero-padded.
                Iterators of non-tiled images must yield ndarrays of
                `shape[1:]` or strips as bytes. Iterators of strip ndarrays
                are not supported.
                Writing dask arrays might be excruciatingly slow for arrays
                with many chunks or files with many segments.
                (https://github.com/dask/dask/issues/8570).
            shape:
                Shape of image to write.
                The default is inferred from the `data` argument if possible.
                A ValueError is raised if the value is incompatible with
                the `data` or other arguments.
            dtype:
                NumPy data type of image to write.
                The default is inferred from the `data` argument if possible.
                A ValueError is raised if the value is incompatible with
                the `data` argument.
            photometric:
                Color space of image.
                The default is inferred from the data shape, dtype, and the
                `colormap` argument.
                A UserWarning is logged if RGB color space is auto-detected.
                Specify this parameter to silence the warning and to avoid
                ambiguities.
                *MINISBLACK*: for bilevel and grayscale images, 0 is black.
                *MINISWHITE*: for bilevel and grayscale images, 0 is white.
                *RGB*: the image contains red, green and blue samples.
                *SEPARATED*: the image contains CMYK samples.
                *PALETTE*: the image is used as an index into a colormap.
                *CFA*: the image is a Color Filter Array. The
                CFARepeatPatternDim, CFAPattern, and other DNG or TIFF/EP tags
                must be specified in `extratags` to produce a valid file.
                The value is written to the PhotometricInterpretation tag.
            planarconfig:
                Specifies if samples are stored interleaved or in separate
                planes.
                *CONTIG*: the last dimension contains samples.
                *SEPARATE*: the 3rd or 4th last dimension contains samples.
                The default is inferred from the data shape and `photometric`
                mode.
                If this parameter is set, extra samples are used to store
                grayscale images.
                The value is written to the PlanarConfiguration tag.
            extrasamples:
                Interpretation of extra components in pixels.
                *UNSPECIFIED*: no transparency information (default).
                *ASSOCALPHA*: true transparency with premultiplied color.
                *UNASSALPHA*: independent transparency masks.
                The values are written to the ExtraSamples tag.
            volumetric:
                Write volumetric image to single page (instead of multiple
                pages) using SGI ImageDepth tag.
                The volumetric format is not part of the TIFF specification,
                and few software can read it.
                OME and ImageJ formats are not compatible with volumetric
                storage.
            tile:
                Shape ([depth,] length, width) of image tiles to write.
                By default, image data are written in strips.
                The tile length and width must be a multiple of 16.
                If a tile depth is provided, the SGI ImageDepth and TileDepth
                tags are used to write volumetric data.
                Tiles cannot be used to write contiguous series, except if
                the tile shape matches the data shape.
                The values are written to the TileWidth, TileLength, and
                TileDepth tags.
            rowsperstrip:
                Number of rows per strip.
                By default, strips are about 256 KB if `compression` is
                enabled, else rowsperstrip is set to the image length.
                The value is written to the RowsPerStrip tag.
            bitspersample:
                Number of bits per sample.
                The default is the number of bits of the data's dtype.
                Different values per samples are not supported.
                Unsigned integer data are packed into bytes as tightly as
                possible.
                Valid values are 1-8 for uint8, 9-16 for uint16, and 17-32
                for uint32.
                This setting cannot be used with compression, contiguous
                series, or empty files.
                The value is written to the BitsPerSample tag.
            compression:
                Compression scheme used on image data.
                By default, image data are written uncompressed.
                Compression cannot be used to write contiguous series.
                Compressors may require certain data shapes, types or value
                ranges. For example, JPEG compression requires grayscale or
                RGB(A), uint8 or 12-bit uint16.
                JPEG compression is experimental. JPEG markers and TIFF tags
                may not match.
                Only a limited set of compression schemes are implemented.
                'ZLIB' is short for ADOBE_DEFLATE.
                The value is written to the Compression tag.
            compressionargs:
                Extra arguments passed to compression codec, for example,
                compression level. Refer to the Imagecodecs implementation
                for supported arguments.
            predictor:
                Horizontal differencing operator applied to image data before
                compression.
                By default, no operator is applied.
                Predictors can only be used with certain compression schemes
                and data types.
                The value is written to the Predictor tag.
            subsampling:
                Horizontal and vertical subsampling factors used for the
                chrominance components of images: (1, 1), (2, 1), (2, 2), or
                (4, 1). The default is *(2, 2)*.
                Currently applies to JPEG compression of RGB images only.
                Images are stored in YCbCr color space, the value of the
                PhotometricInterpretation tag is *YCBCR*.
                Segment widths must be a multiple of 8 times the horizontal
                factor. Segment lengths and rowsperstrip must be a multiple
                of 8 times the vertical factor.
                The values are written to the YCbCrSubSampling tag.
            jpegtables:
                JPEG quantization and/or Huffman tables.
                Use for copying pre-compressed JPEG segments.
                The value is written to the JPEGTables tag.
            iccprofile:
                International Color Consortium (ICC) device profile
                characterizing image color space.
                The value is written verbatim to the InterColorProfile tag.
            colormap:
                RGB color values for corresponding data value.
                The colormap array must be of shape
                `(3, 2\*\*(data.itemsize*8))` (or `(3, 256)` for ImageJ)
                and dtype uint16.
                The image's data type must be uint8 or uint16 (or float32
                for ImageJ) and the values are indices into the last
                dimension of the colormap.
                The value is written to the ColorMap tag.
            description:
                Subject of image. Must be 7-bit ASCII.
                Cannot be used with the ImageJ or OME formats.
                The value is written to the ImageDescription tag of the
                first page of a series.
            datetime:
                Date and time of image creation in ``%Y:%m:%d %H:%M:%S``
                format or datetime object.
                If *True*, the current date and time is used.
                The value is written to the DateTime tag of the first page
                of a series.
            resolution:
                Number of pixels per `resolutionunit` in X and Y directions
                as float or rational numbers.
                The default is (1.0, 1.0).
                The values are written to the YResolution and XResolution tags.
            resolutionunit:
                Unit of measurement for `resolution` values.
                The default is *NONE* if `resolution` is not specified and
                for ImageJ format, else *INCH*.
                The value is written to the ResolutionUnit tags.
            subfiletype:
                Bitfield to indicate kind of image.
                Set bit 0 if the image is a reduced-resolution version of
                another image.
                Set bit 1 if the image is part of a multi-page image.
                Set bit 2 if the image is transparency mask for another
                image (photometric must be MASK, SamplesPerPixel and
                bitspersample must be 1).
            software:
                Name of software used to create file.
                Must be 7-bit ASCII. The default is 'tifffile.py'.
                Unless *False*, the value is written to the Software tag of
                the first page of a series.
            subifds:
                Number of child IFDs.
                If greater than 0, the following `subifds` number of series
                are written as child IFDs of the current series.
                The number of IFDs written for each SubIFD level must match
                the number of IFDs written for the current series.
                All pages written to a certain SubIFD level of the current
                series must have the same hash.
                SubIFDs cannot be used with truncated or ImageJ files.
                SubIFDs in OME-TIFF files must be sub-resolutions of the
                main IFDs.
            metadata:
                Additional metadata describing image, written along
                with shape information in JSON, OME-XML, or ImageJ formats
                in ImageDescription or IJMetadata tags.
                Metadata do not determine, but must match, how image data
                is written to the file.
                By default, no additional metadata is written.
                If *None*, or the `shaped` argument to :py:class:`TiffWriter`
                is *False*, no information in JSON format is written to
                the ImageDescription tag.
                The 'axes' item defines the character codes for dimensions in
                `data` or `shape`.
                Refer to :py:class:`OmeXml` for supported keys when writing
                OME-TIFF.
                Refer to :py:func:`imagej_description` and
                :py:func:`imagej_metadata_tag` for items supported
                by the ImageJ format. Items 'Info', 'Labels', 'Ranges',
                'LUTs', 'Plot', 'ROI', and 'Overlays' are written to the
                IJMetadata and IJMetadataByteCounts tags.
                Strings must be 7-bit ASCII.
                Written with the first page of a series only.
            extratags:
                Additional tags to write. A list of tuples with 5 items:

                0. code (int): Tag Id.
                1. dtype (:py:class:`DATATYPE`):
                   Data type of items in `value`.
                2. count (int): Number of data values.
                   Not used for string or bytes values.
                3. value (Sequence[Any]): `count` values compatible with
                   `dtype`. Bytes must contain count values of dtype packed
                   as binary data.
                4. writeonce (bool): If *True*, write tag to first page
                   of a series only.

                Duplicate and select tags in TIFF.TAG_FILTERED are not written
                if the extratag is specified by integer code.
                Extratags cannot be used to write IFD type tags.

            contiguous:
                If *False* (default), write data to a new series.
                If *True* and the data and arguments are compatible with
                previous written ones (same shape, no compression, etc.),
                the image data are stored contiguously after the previous one.
                In that case, `photometric`, `planarconfig`, and
                `rowsperstrip` are ignored.
                Metadata such as `description`, `metadata`, `datetime`,
                and `extratags` are written to the first page of a contiguous
                series only.
                Contiguous mode cannot be used with the OME or ImageJ formats.
            truncate:
                If *True*, only write first page of contiguous series
                if possible (uncompressed, contiguous, not tiled).
                Other TIFF readers will only be able to read part of the data.
                Cannot be used with the OME or ImageJ formats.
            align:
                Byte boundary on which to align image data in file.
                The default is 16.
                Use mmap.ALLOCATIONGRANULARITY for memory-mapped data.
                Following contiguous writes are not aligned.
            maxworkers:
                Maximum number of threads to concurrently compress tiles
                or strips.
                If *None* or *0*, use up to :py:attr:`_TIFF.MAXWORKERS` CPU
                cores for compressing large segments.
                Using multiple threads can significantly speed up this
                function if the bottleneck is encoding the data, for example,
                in case of large JPEG compressed tiles.
                If the bottleneck is I/O or pure Python code, using multiple
                threads might be detrimental.
            buffersize:
                Approximate number of bytes to compress in one pass.
                The default is :py:attr:`_TIFF.BUFFERSIZE` * 2.
            returnoffset:
                Return offset and number of bytes of memory-mappable image
                data in file.

        Returns:
            If `returnoffset` is *True* and the image data in the file are
            memory-mappable, return the offset and number of bytes of the
            image data in the file.

        """
        fh: FileHandle
        storedshape: StoredShape = StoredShape(frames=-1)
        byteorder: Literal['>', '<']
        inputshape: tuple[int, ...]
        datashape: tuple[int, ...]
        dataarray: NDArray[Any] | None = None
        dataiter: Iterator[NDArray[Any] | bytes | None] | None = None
        dataoffsets: list[int] | None = None
        dataoffsetsoffset: tuple[int, int | None] | None = None
        databytecounts: list[int]
        databytecountsoffset: tuple[int, int | None] | None = None
        subifdsoffsets: tuple[int, int | None] | None = None
        datadtype: numpy.dtype[Any]
        bilevel: bool
        tiles: tuple[int, ...]
        ifdpos: int
        photometricsamples: int
        pos: int | None = None
        predictortag: int
        predictorfunc: Callable[..., Any] | None = None
        compressiontag: int
        compressionfunc: Callable[..., Any] | None = None
        tags: list[tuple[int, bytes, bytes | None, bool]]
        numtiles: int
        numstrips: int

        fh = self._fh
        byteorder = self.tiff.byteorder

        # detect CUDA tensor — extract shape/dtype, defer D2H transfer
        _gpu_tensor = None
        if (
            data is not None
            and not hasattr(data, '__next__')
            and not isinstance(data, numpy.ndarray)
        ):
            from .gpu import _is_cuda_tensor

            if _is_cuda_tensor(data):
                from .gpu import _torch_dtype_to_numpy

                _gpu_tensor = data
                tensor_np_dtype = _torch_dtype_to_numpy(data.dtype)
                if dtype is not None and numpy.dtype(dtype) != tensor_np_dtype:
                    msg = (
                        f'dtype {dtype!r} does not match '
                        f'tensor dtype {tensor_np_dtype}'
                    )
                    raise ValueError(msg)
                if shape is not None and tuple(shape) != tuple(data.shape):
                    msg = (
                        f'shape {shape!r} does not match '
                        f'tensor shape {tuple(data.shape)}'
                    )
                    raise ValueError(msg)
                if shape is None:
                    shape = tuple(data.shape)
                if dtype is None:
                    dtype = tensor_np_dtype
                data = None  # _resolve_write_data handles data=None

        # resolve data input
        dataarray, dataiter, datashape, datadtype = (
            self._resolve_write_data(data, shape, dtype, byteorder)
        )
        del data

        if any(size >= 4294967296 for size in datashape):
            msg = 'invalid data shape'
            raise ValueError(msg)

        bilevel = datadtype.char == '?'
        if bilevel:
            index = -1 if datashape[-1] > 1 else -2
            datasize = product(datashape[:index])
            if datashape[index] % 8:
                datasize *= datashape[index] // 8 + 1
            else:
                datasize *= datashape[index] // 8
        else:
            datasize = product(datashape) * datadtype.itemsize

        if datasize == 0:
            dataarray = None
            compression = False
            bitspersample = None
            if metadata is not None:
                truncate = True

        if not compression or (
            not isinstance(compression, bool)  # because True == 1
            and compression in ('NONE', 'None', 'none', 1)
        ):
            compression = False

        if not predictor or (
            not isinstance(predictor, bool)  # because True == 1
            and predictor in {'NONE', 'None', 'none', 1}
        ):
            predictor = False

        inputshape = datashape

        packints = (
            bitspersample is not None
            and bitspersample != datadtype.itemsize * 8
        )

        # just append contiguous data if possible
        if self._datashape is not None and self._datadtype is not None:
            if colormap is not None:
                colormap = numpy.asarray(colormap, dtype=byteorder + 'H')
            if (
                not contiguous
                or self._datashape[1:] != datashape
                or self._datadtype != datadtype
                or (colormap is None and self._colormap is not None)
                or (self._colormap is None and colormap is not None)
                or not numpy.array_equal(
                    colormap, self._colormap  # type: ignore[arg-type]
                )
            ):
                # incompatible shape, dtype, or colormap
                self._write_remaining_pages()

                if self._imagej:
                    msg = (
                        'the ImageJ format does not support '
                        'non-contiguous series'
                    )
                    raise ValueError(msg)
                if self._omexml is not None:
                    if self._subifdslevel < 0:
                        # add image to OME-XML
                        assert self._storedshape is not None
                        assert self._metadata is not None
                        self._omexml.addimage(
                            dtype=self._datadtype,
                            shape=self._datashape[
                                0 if self._datashape[0] != 1 else 1 :
                            ],
                            storedshape=self._storedshape.shape,
                            **self._metadata,
                        )
                elif metadata is not None:
                    self._write_image_description()
                    # description might have been appended to file
                    fh.seek(0, os.SEEK_END)

                if self._subifds:
                    if self._truncate or truncate:
                        msg = 'SubIFDs cannot be used with truncated series'
                        raise ValueError(msg)
                    self._subifdslevel += 1
                    if self._subifdslevel == self._subifds:
                        # done with writing SubIFDs
                        self._nextifdoffsets = []
                        self._subifdsoffsets = []
                        self._subifdslevel = -1
                        self._subifds = 0
                        self._ifdindex = 0
                    elif subifds:
                        msg = 'SubIFDs in SubIFDs are not supported'
                        raise ValueError(msg)

                self._datashape = None
                self._colormap = None

            elif compression or packints or tile:
                msg = (
                    'contiguous mode cannot be used with compression or tiles'
                )
                raise ValueError(msg)

            else:
                # consecutive mode
                # write all data, write IFDs/tags later
                self._datashape = (self._datashape[0] + 1, *datashape)
                offset = fh.tell()
                if _gpu_tensor is not None:
                    from .gpu import tensor_to_numpy

                    dataarray = tensor_to_numpy(_gpu_tensor).reshape(
                        datashape
                    )
                if dataarray is None:
                    fh.write_empty(datasize)
                else:
                    fh.write_array(dataarray, datadtype)
                if returnoffset:
                    return offset, datasize
                return None

        # format-specific validation and setup
        if self._ome is None:
            if description is None:
                self._ome = '.ome.' in fh.extension
            else:
                self._ome = False

        if self._tifffile or self._imagej:
            self._truncate = bool(truncate)
        elif truncate:
            msg = 'truncate can only be used with imagej or shaped formats'
            raise ValueError(msg)
        else:
            self._truncate = False

        if self._truncate and (compression or packints or tile):
            msg = (
                'truncate cannot be used with compression, packints, or tiles'
            )
            raise ValueError(msg)

        if datasize == 0:
            # write single placeholder TiffPage for arrays with size=0
            datashape = (0, 0)
            warnings.warn(
                f'{self!r} writing zero-size array to nonconformant TIFF',
                UserWarning,
                stacklevel=2,
            )
            # TODO: reconsider this
            # raise ValueError('cannot save zero size array')

        tagnoformat = self.tiff.tagnoformat
        offsetformat = self.tiff.offsetformat
        offsetsize = self.tiff.offsetsize
        tagsize = self.tiff.tagsize

        MINISBLACK = PHOTOMETRIC.MINISBLACK
        MINISWHITE = PHOTOMETRIC.MINISWHITE
        RGB = PHOTOMETRIC.RGB
        YCBCR = PHOTOMETRIC.YCBCR
        PALETTE = PHOTOMETRIC.PALETTE
        CONTIG = PLANARCONFIG.CONTIG
        SEPARATE = PLANARCONFIG.SEPARATE

        # parse input
        if photometric is not None:
            photometric = enumarg(PHOTOMETRIC, photometric)
        if planarconfig:
            planarconfig = enumarg(PLANARCONFIG, planarconfig)
        if extrasamples is not None:
            # TODO: deprecate non-sequence extrasamples
            extrasamples = tuple(
                int(enumarg(EXTRASAMPLE, x)) for x in sequence(extrasamples)
            )

        if compression:
            if isinstance(compression, str):
                compression = compression.upper()
                if compression == 'ZLIB':
                    compression = 8  # ADOBE_DEFLATE
            elif isinstance(compression, bool):
                compression = 8  # ADOBE_DEFLATE
            compressiontag = enumarg(COMPRESSION, compression).value
            compression = True
        else:
            compressiontag = 1
            compression = False

        if compressionargs is None:
            compressionargs = {}
        if compressiontag == 1:
            compressionargs = {}
        elif compressiontag in {33003, 33004, 33005, 34712}:
            # JPEG2000: use J2K instead of JP2
            compressionargs['codecformat'] = 0  # OPJ_CODEC_J2K

        assert compressionargs is not None

        if predictor:
            if not compression:
                msg = 'cannot use predictor without compression'
                raise ValueError(msg)
            if compressiontag in TIFF.IMAGE_COMPRESSIONS:
                # don't use predictor with JPEG, JPEG2000, WEBP, PNG, ...
                msg = (
                    'cannot use predictor with '
                    f'{COMPRESSION(compressiontag)!r}'
                )
                raise ValueError(msg)
            if isinstance(predictor, bool):
                if datadtype.kind == 'f':
                    predictortag = 3
                elif datadtype.kind in 'iu' and datadtype.itemsize <= 4:
                    predictortag = 2
                else:
                    msg = f'cannot use predictor with {datadtype!r}'
                    raise ValueError(msg)
            else:
                predictor = enumarg(PREDICTOR, predictor)
                if (
                    datadtype.kind in 'iu'
                    and predictor.value not in {2, 34892, 34893}
                    and datadtype.itemsize <= 4
                ) or (
                    datadtype.kind == 'f'
                    and predictor.value not in {3, 34894, 34895}
                ):
                    msg = f'cannot use {predictor!r} with {datadtype!r}'
                    raise ValueError(msg)
                predictortag = predictor.value
        else:
            predictortag = 1

        del predictor
        predictorfunc = TIFF.PREDICTORS[predictortag]

        if self._ome:
            if description is not None:
                warnings.warn(
                    f'{self!r} not writing description to OME-TIFF',
                    UserWarning,
                    stacklevel=2,
                )
                description = None
            if self._omexml is None:
                if metadata is None:
                    self._omexml = OmeXml()
                else:
                    self._omexml = OmeXml(**metadata)
            if volumetric or (tile and len(tile) > 2):
                msg = 'OME-TIFF does not support ImageDepth'
                raise ValueError(msg)
            volumetric = False

        elif self._imagej:
            # if tile is not None or predictor or compression:
            #     warnings.warn(
            #         f'{self!r} the ImageJ format does not support '
            #         'tiles, predictors, compression'
            #     )
            if description is not None:
                warnings.warn(
                    f'{self!r} not writing description to ImageJ file',
                    UserWarning,
                    stacklevel=2,
                )
                description = None
            if datadtype.char not in 'BHhf':
                msg = (
                    'the ImageJ format does not support data type '
                    f'{datadtype.char!r}'
                )
                raise ValueError(msg)
            if volumetric or (tile and len(tile) > 2):
                msg = 'the ImageJ format does not support ImageDepth'
                raise ValueError(msg)
            volumetric = False
            ijrgb = photometric == RGB if photometric else None
            if datadtype.char != 'B':
                if photometric == RGB:
                    msg = (
                        'the ImageJ format does not support '
                        f'data type {datadtype!r} for RGB'
                    )
                    raise ValueError(msg)
                ijrgb = False
            if colormap is not None:
                ijrgb = False
            axes = None if metadata is None else metadata.get('axes', None)
            ijshape = imagej_shape(datashape, rgb=ijrgb, axes=axes)
            if planarconfig == SEPARATE:
                msg = 'the ImageJ format does not support planar samples'
                raise ValueError(msg)
            if ijshape[-1] in {3, 4}:
                photometric = RGB
            elif photometric is None:
                if colormap is not None and datadtype.char == 'B':
                    photometric = PALETTE
                else:
                    photometric = MINISBLACK
                planarconfig = None
            planarconfig = CONTIG if ijrgb else None

        # verify colormap and indices
        if colormap is not None:
            colormap = numpy.asarray(colormap, dtype=byteorder + 'H')
            self._colormap = colormap
            if self._imagej:
                if colormap.shape != (3, 256):
                    msg = 'invalid colormap shape for ImageJ'
                    raise ValueError(msg)
                if datadtype.char == 'B' and photometric in {
                    MINISBLACK,
                    MINISWHITE,
                }:
                    photometric = PALETTE
                elif not (
                    (datadtype.char == 'B' and photometric == PALETTE)
                    or (
                        datadtype.char in 'Hf'
                        and photometric in {MINISBLACK, MINISWHITE}
                    )
                ):
                    warnings.warn(
                        f'{self!r} not writing colormap to ImageJ image with '
                        f'dtype={datadtype} and {photometric=}',
                        UserWarning,
                        stacklevel=2,
                    )
                    colormap = None
            elif photometric is None and datadtype.char in 'BH':
                photometric = PALETTE
                planarconfig = None
                if colormap.shape != (3, 2 ** (datadtype.itemsize * 8)):
                    msg = 'invalid colormap shape'
                    raise ValueError(msg)
            elif photometric == PALETTE:
                planarconfig = None
                if datadtype.char not in 'BH':
                    msg = 'invalid data dtype for palette-image'
                    raise ValueError(msg)
                if colormap.shape != (3, 2 ** (datadtype.itemsize * 8)):
                    msg = 'invalid colormap shape'
                    raise ValueError(msg)
            else:
                warnings.warn(
                    f'{self!r} not writing colormap with image of '
                    f'dtype={datadtype} and {photometric=}',
                    UserWarning,
                    stacklevel=2,
                )
                colormap = None

        # normalize tile shape and data dimensions
        if tile:
            # verify tile shape

            if (
                not 1 < len(tile) < 4
                or tile[-1] % 16
                or tile[-2] % 16
                or any(i < 1 for i in tile)
            ):
                msg = f'invalid tile shape {tile}'
                raise ValueError(msg)
            tile = tuple(int(i) for i in tile)
            if volumetric and len(tile) == 2:
                tile = (1, *tile)
            volumetric = len(tile) == 3
        else:
            tile = ()
            volumetric = bool(volumetric)
        assert isinstance(tile, tuple)  # for mypy

        # normalize data shape to 5D or 6D, depending on volume:
        #   (pages, separate_samples, [depth,] length, width, contig_samples)
        shape = reshape_nd(
            datashape,
            TIFF.PHOTOMETRIC_SAMPLES.get(
                photometric, 2  # type: ignore[arg-type]
            ),
        )
        ndim = len(shape)

        if volumetric and ndim < 3:
            volumetric = False

        # resolve photometric interpretation and planar configuration
        if photometric is None:
            deprecate = False
            photometric = MINISBLACK
            if bilevel:
                photometric = MINISWHITE
            elif planarconfig == CONTIG:
                if ndim > 2 and shape[-1] in {3, 4}:
                    photometric = RGB
                    deprecate = datadtype.char not in 'BH'
            elif planarconfig == SEPARATE:
                if (volumetric and ndim > 3 and shape[-4] in {3, 4}) or (
                    ndim > 2 and shape[-3] in {3, 4}
                ):
                    photometric = RGB
                    deprecate = True
            elif ndim > 2 and shape[-1] in {3, 4}:
                photometric = RGB
                planarconfig = CONTIG
                deprecate = datadtype.char not in 'BH'
            elif self._imagej or self._ome:
                photometric = MINISBLACK
                planarconfig = None
            elif (volumetric and ndim > 3 and shape[-4] in {3, 4}) or (
                ndim > 2 and shape[-3] in {3, 4}
            ):
                photometric = RGB
                planarconfig = SEPARATE
                deprecate = True

            if deprecate:
                if planarconfig == CONTIG:
                    msgs = 'contiguous samples', 'parameter is'
                else:
                    msgs = (
                        'separate component planes',
                        "and 'planarconfig' parameters are",
                    )
                warnings.warn(
                    f"<tifffile.TiffWriter.write> data with shape {datashape} "
                    f"and dtype '{datadtype}' are stored as RGB with {msgs[0]}"
                    '. Future versions will store such data as MINISBLACK in '
                    "separate pages by default, unless the 'photometric' "
                    f"{msgs[1]} specified.",
                    DeprecationWarning,
                    stacklevel=2,
                )
                del msgs
            del deprecate

        del datashape
        assert photometric is not None
        photometricsamples = TIFF.PHOTOMETRIC_SAMPLES[photometric]

        if planarconfig and len(shape) <= (3 if volumetric else 2):
            # TODO: raise error?
            planarconfig = None
            if photometricsamples > 1:
                photometric = MINISBLACK

        if photometricsamples > 1:
            if len(shape) < 3:
                msg = f'not a {photometric!r} image'
                raise ValueError(msg)
            if len(shape) < 4:
                volumetric = False
            if planarconfig is None:
                if photometric == RGB:
                    samples_set = {photometricsamples, 4}  # allow common alpha
                else:
                    samples_set = {photometricsamples}
                if shape[-1] in samples_set:
                    planarconfig = CONTIG
                elif shape[-4 if volumetric else -3] in samples_set:
                    planarconfig = SEPARATE
                elif shape[-1] > shape[-4 if volumetric else -3]:
                    # TODO: deprecated this?
                    planarconfig = SEPARATE
                else:
                    planarconfig = CONTIG
            if planarconfig == CONTIG:
                storedshape.contig_samples = shape[-1]
                storedshape.width = shape[-2]
                storedshape.length = shape[-3]
                if volumetric:
                    storedshape.depth = shape[-4]
            else:
                storedshape.width = shape[-1]
                storedshape.length = shape[-2]
                if volumetric:
                    storedshape.depth = shape[-3]
                    storedshape.separate_samples = shape[-4]
                else:
                    storedshape.separate_samples = shape[-3]
            if storedshape.samples > photometricsamples:
                storedshape.extrasamples = (
                    storedshape.samples - photometricsamples
                )

        elif photometric == PHOTOMETRIC.CFA:
            if len(shape) != 2:
                msg = 'invalid CFA image'
                raise ValueError(msg)
            volumetric = False
            planarconfig = None
            storedshape.width = shape[-1]
            storedshape.length = shape[-2]
            # if all(et[0] != 50706 for et in extratags):
            #     raise ValueError('must specify DNG tags for CFA image')

        elif planarconfig and len(shape) > (3 if volumetric else 2):
            if planarconfig == CONTIG:
                if extrasamples is None or len(extrasamples) > 0:
                    # use extrasamples
                    storedshape.contig_samples = shape[-1]
                    storedshape.width = shape[-2]
                    storedshape.length = shape[-3]
                    if volumetric:
                        storedshape.depth = shape[-4]
                else:
                    planarconfig = None
                    storedshape.contig_samples = 1
                    storedshape.width = shape[-1]
                    storedshape.length = shape[-2]
                    if volumetric:
                        storedshape.depth = shape[-3]
            else:
                storedshape.width = shape[-1]
                storedshape.length = shape[-2]
                if extrasamples is None or len(extrasamples) > 0:
                    # use extrasamples
                    if volumetric:
                        storedshape.depth = shape[-3]
                        storedshape.separate_samples = shape[-4]
                    else:
                        storedshape.separate_samples = shape[-3]
                else:
                    planarconfig = None
                    storedshape.separate_samples = 1
                    if volumetric:
                        storedshape.depth = shape[-3]
            storedshape.extrasamples = storedshape.samples - 1

        else:
            # photometricsamples == 1
            planarconfig = None
            if self._tifffile and (metadata or metadata == {}):
                # remove trailing 1s in shaped series
                while len(shape) > 2 and shape[-1] == 1:
                    shape = shape[:-1]
            elif self._imagej and len(shape) > 2 and shape[-1] == 1:
                # TODO: remove this and sync with ImageJ shape
                shape = shape[:-1]
            if len(shape) < 3:
                volumetric = False
            if not extrasamples:
                storedshape.width = shape[-1]
                storedshape.length = shape[-2]
                if volumetric:
                    storedshape.depth = shape[-3]
            else:
                storedshape.contig_samples = shape[-1]
                storedshape.width = shape[-2]
                storedshape.length = shape[-3]
                if volumetric:
                    storedshape.depth = shape[-4]
                storedshape.extrasamples = storedshape.samples - 1

        if not volumetric and tile and len(tile) == 3 and tile[0] > 1:
            msg = f'cannot write {storedshape!r} using volumetric tiles {tile}'
            raise ValueError(msg)

        if subfiletype is not None and subfiletype & 0b100:
            # FILETYPE_MASK
            if not (
                bilevel
                and storedshape.samples == 1
                and photometric in {0, 1, 4}
            ):
                msg = 'invalid SubfileType MASK'
                raise ValueError(msg)
            photometric = PHOTOMETRIC.MASK

        # resolve bits per sample and pack integers
        packints = False
        if bilevel:
            if bitspersample is not None and bitspersample != 1:
                msg = f'{bitspersample=} must be 1 for bilevel'
                raise ValueError(msg)
            bitspersample = 1
        elif compressiontag in {6, 7, 34892, 33007}:
            # JPEG
            # TODO: add bitspersample to compressionargs?
            if bitspersample is None:
                bitspersample = compressionargs.get(
                    'bitspersample', 12 if datadtype == 'uint16' else 8
                )
            if not 2 <= bitspersample <= 16:
                msg = f'{bitspersample=} invalid for JPEG compression'
                raise ValueError(msg)
        elif compressiontag in {33003, 33004, 33005, 34712, 50002, 52546}:
            # JPEG2K, JPEGXL
            # TODO: unify with JPEG?
            if bitspersample is None:
                bitspersample = compressionargs.get(
                    'bitspersample', datadtype.itemsize * 8
                )
            if not (
                bitspersample > {1: 0, 2: 8, 4: 16}[datadtype.itemsize]
                and bitspersample <= datadtype.itemsize * 8
            ):
                msg = f'{bitspersample=} out of range of {datadtype=}'
                raise ValueError(msg)
        elif bitspersample is None:
            bitspersample = datadtype.itemsize * 8
        elif (
            datadtype.kind != 'u' or datadtype.itemsize > 4
        ) and bitspersample != datadtype.itemsize * 8:
            msg = f'{bitspersample=} does not match {datadtype=}'
            raise ValueError(msg)
        elif not (
            bitspersample > {1: 0, 2: 8, 4: 16}[datadtype.itemsize]
            and bitspersample <= datadtype.itemsize * 8
        ):
            msg = f'{bitspersample=} out of range of {datadtype=}'
            raise ValueError(msg)
        elif compression:
            if bitspersample != datadtype.itemsize * 8:
                msg = f'{bitspersample=} cannot be used with compression'
                raise ValueError(msg)
        elif bitspersample != datadtype.itemsize * 8:
            packints = True

        if storedshape.frames == -1:
            s0 = storedshape.page_size
            storedshape.frames = 1 if s0 == 0 else product(inputshape) // s0

        if datasize > 0 and not storedshape.is_valid:
            msg = f'invalid {storedshape=!r}'
            raise RuntimeError(msg)

        if photometric == PALETTE:
            if storedshape.samples != 1 or storedshape.extrasamples > 0:
                msg = f'invalid {storedshape=!r} for palette mode'
                raise ValueError(msg)
        elif storedshape.samples < photometricsamples:
            msg = (
                f'not enough samples for {photometric!r}: '
                f'expected {photometricsamples}, got {storedshape.samples}'
            )
            raise ValueError(msg)

        if (
            planarconfig is not None
            and storedshape.planarconfig != planarconfig
        ):
            msg = f'{planarconfig!r} does not match {storedshape=!r}'
            raise ValueError(msg)
        del planarconfig

        if dataarray is not None:
            dataarray = dataarray.reshape(storedshape.shape)

        # detect GPU encoder to influence tile/strip layout
        _gpu_algorithm = self._detect_gpu_encoder(
            _gpu_tensor, compression, packints, compressiontag,
            predictortag, storedshape, datadtype,
        )
        if _gpu_algorithm is not None:
            if tile:
                logger().info(
                    'GPU encoder: overriding tiles to single-strip mode'
                )
                tile = None
            rowsperstrip = storedshape.length

        # build IFD tags
        tags = []  # list of (code, ifdentry, ifdvalue, writeonce)

        if tile:
            tagbytecounts = 325  # TileByteCounts
            tagoffsets = 324  # TileOffsets
        else:
            tagbytecounts = 279  # StripByteCounts
            tagoffsets = 273  # StripOffsets
        self._dataoffsetstag = tagoffsets

        pack = self._pack
        addtag = self._addtag

        if extratags is None:
            extratags = ()

        if description is not None:
            # ImageDescription: user provided description
            addtag(tags, 270, 2, 0, description, True)

        # write shape and metadata to ImageDescription
        self._metadata = {} if not metadata else metadata.copy()
        if self._omexml is not None:
            if len(self._omexml.images) == 0:
                # rewritten later at end of file
                description = '\x00\x00\x00\x00'
            else:
                description = None
        elif self._imagej:
            ijmetadata = parse_kwargs(
                self._metadata,
                'Info',
                'Labels',
                'Ranges',
                'LUTs',
                'Plot',
                'ROI',
                'Overlays',
                'Properties',
                'info',
                'labels',
                'ranges',
                'luts',
                'plot',
                'roi',
                'overlays',
                'prop',
            )

            for t in imagej_metadata_tag(ijmetadata, byteorder):
                addtag(tags, *t)
            description = imagej_description(
                inputshape,
                rgb=storedshape.contig_samples in {3, 4},
                colormapped=self._colormap is not None,
                **self._metadata,
            )
            description += '\x00' * 64  # add buffer for in-place update
        elif self._tifffile and (metadata or metadata == {}):
            if self._truncate:
                self._metadata.update(truncated=True)
            description = shaped_description(inputshape, **self._metadata)
            description += '\x00' * 16  # add buffer for in-place update
        # elif metadata is None and self._truncate:
        #     raise ValueError('cannot truncate without writing metadata')
        elif description is not None:
            if not isinstance(description, bytes):
                description = description.encode('ascii')
            self._descriptiontag = TiffTag(
                self, 0, 270, 2, len(description), description, 0
            )
            description = None

        if description is None:
            # disable shaped format if user disabled metadata
            self._tifffile = False
        else:
            description = description.encode('ascii')
            addtag(tags, 270, 2, 0, description, True)
            self._descriptiontag = TiffTag(
                self, 0, 270, 2, len(description), description, 0
            )
        del description

        if software is None:
            software = 'tifffile.py'
        if software:
            addtag(tags, 305, 2, 0, software, True)
        if datetime:
            if isinstance(datetime, str):
                if len(datetime) != 19 or datetime[16] != ':':
                    msg = 'invalid datetime string'
                    raise ValueError(msg)
            elif isinstance(datetime, DateTime):
                datetime = datetime.strftime('%Y:%m:%d %H:%M:%S')
            else:
                datetime = DateTime.now().strftime('%Y:%m:%d %H:%M:%S')
            addtag(tags, 306, 2, 0, datetime, True)
        addtag(tags, 259, 3, 1, compressiontag)  # Compression
        if compressiontag == 34887:
            # LERC
            if compressionargs.get('compression') is None:
                lerc_compression = 0
            elif compressionargs['compression'] == 'deflate':
                lerc_compression = 1
            elif compressionargs['compression'] == 'zstd':
                lerc_compression = 2
            else:
                msg = (
                    'invalid LERC compression'
                    f'{compressionargs["compression"]!r}'
                )
                raise ValueError(msg)
            addtag(tags, 50674, 4, 2, (4, lerc_compression))
            del lerc_compression
        if predictortag != 1:
            addtag(tags, 317, 3, 1, predictortag)
        addtag(tags, 256, 4, 1, storedshape.width)  # ImageWidth
        addtag(tags, 257, 4, 1, storedshape.length)  # ImageLength
        if tile:
            addtag(tags, 322, 4, 1, tile[-1])  # TileWidth
            addtag(tags, 323, 4, 1, tile[-2])  # TileLength
        if volumetric:
            addtag(tags, 32997, 4, 1, storedshape.depth)  # ImageDepth
            if tile:
                addtag(tags, 32998, 4, 1, tile[0])  # TileDepth
        if subfiletype is not None:
            addtag(tags, 254, 4, 1, subfiletype)  # NewSubfileType
        if (subifds or self._subifds) and self._subifdslevel < 0:
            if self._subifds:
                subifds = self._subifds
            elif hasattr(subifds, '__len__'):
                # allow TiffPage.subifds tuple
                subifds = len(subifds)  # type: ignore[arg-type]
            else:
                subifds = int(subifds)  # type: ignore[arg-type]
            self._subifds = subifds
            addtag(
                tags, 330, 18 if offsetsize > 4 else 13, subifds, [0] * subifds
            )
        if not bilevel and datadtype.kind != 'u':
            # SampleFormat
            sampleformat = {'u': 1, 'i': 2, 'f': 3, 'c': 6}[datadtype.kind]
            addtag(
                tags,
                339,
                3,
                storedshape.samples,
                (sampleformat,) * storedshape.samples,
            )
        if colormap is not None:
            addtag(tags, 320, 3, colormap.size, colormap)
        if iccprofile is not None:
            addtag(tags, 34675, 7, len(iccprofile), iccprofile)
        addtag(tags, 277, 3, 1, storedshape.samples)
        if bilevel:
            # PlanarConfiguration
            if storedshape.samples > 1:
                addtag(tags, 284, 3, 1, storedshape.planarconfig)
        elif storedshape.samples > 1:
            # PlanarConfiguration
            addtag(tags, 284, 3, 1, storedshape.planarconfig)
            # BitsPerSample
            addtag(
                tags,
                258,
                3,
                storedshape.samples,
                (bitspersample,) * storedshape.samples,
            )
        else:
            addtag(tags, 258, 3, 1, bitspersample)
        if storedshape.extrasamples > 0:
            if extrasamples is not None:
                if storedshape.extrasamples != len(extrasamples):
                    msg = (
                        'wrong number of extrasamples '
                        f'{storedshape.extrasamples} != {len(extrasamples)}'
                    )
                    raise ValueError(msg)
                addtag(tags, 338, 3, len(extrasamples), extrasamples)
            elif photometric == RGB and storedshape.extrasamples == 1:
                # Unassociated alpha channel
                addtag(tags, 338, 3, 1, 2)
            else:
                # Unspecified alpha channel
                addtag(
                    tags,
                    338,
                    3,
                    storedshape.extrasamples,
                    (0,) * storedshape.extrasamples,
                )

        if jpegtables is not None:
            addtag(tags, 347, 7, len(jpegtables), jpegtables)

        if (
            compressiontag == 7
            and storedshape.planarconfig == 1
            and photometric in {RGB, YCBCR}
        ):
            # JPEG compression with subsampling
            # TODO: use JPEGTables for multiple tiles or strips
            if subsampling is None:
                subsampling = (2, 2)
            elif subsampling not in {(1, 1), (2, 1), (2, 2), (4, 1)}:
                msg = f'invalid subsampling factors {subsampling!r}'
                raise ValueError(msg)
            maxsampling = max(subsampling) * 8
            if tile and (tile[-1] % maxsampling or tile[-2] % maxsampling):
                msg = f'tile shape not a multiple of {maxsampling}'
                raise ValueError(msg)
            if storedshape.extrasamples > 1:
                msg = 'JPEG subsampling requires RGB(A) images'
                raise ValueError(msg)
            addtag(tags, 530, 3, 2, subsampling)  # YCbCrSubSampling
            # use PhotometricInterpretation YCBCR by default
            outcolorspace = enumarg(
                PHOTOMETRIC, compressionargs.get('outcolorspace', 6)
            )
            compressionargs['subsampling'] = subsampling
            compressionargs['colorspace'] = photometric.name
            compressionargs['outcolorspace'] = outcolorspace.name
            addtag(tags, 262, 3, 1, outcolorspace)
            if outcolorspace == YCBCR and all(
                et[0] != 532 for et in extratags
            ):
                # ReferenceBlackWhite is required for YCBCR
                addtag(
                    tags,
                    532,
                    5,
                    6,
                    (0, 1, 255, 1, 128, 1, 255, 1, 128, 1, 255, 1),
                )
        else:
            if subsampling not in {None, (1, 1)}:
                logger().warning(
                    f'{self!r} cannot apply subsampling {subsampling!r}'
                )
            subsampling = None
            maxsampling = 1
            addtag(
                tags, 262, 3, 1, photometric.value
            )  # PhotometricInterpretation
            if photometric == YCBCR:
                # YCbCrSubSampling and ReferenceBlackWhite
                addtag(tags, 530, 3, 2, (1, 1))
                if all(et[0] != 532 for et in extratags):
                    addtag(
                        tags,
                        532,
                        5,
                        6,
                        (0, 1, 255, 1, 128, 1, 255, 1, 128, 1, 255, 1),
                    )

        if resolutionunit is not None:
            resolutionunit = enumarg(RESUNIT, resolutionunit)
        elif self._imagej or resolution is None:
            resolutionunit = RESUNIT.NONE
        else:
            resolutionunit = RESUNIT.INCH

        if resolution is not None:
            addtag(tags, 282, 5, 1, rational(resolution[0]))  # XResolution
            addtag(tags, 283, 5, 1, rational(resolution[1]))  # YResolution
            addtag(tags, 296, 3, 1, resolutionunit)  # ResolutionUnit
        else:
            addtag(tags, 282, 5, 1, (1, 1))  # XResolution
            addtag(tags, 283, 5, 1, (1, 1))  # YResolution
            addtag(tags, 296, 3, 1, resolutionunit)  # ResolutionUnit

        # calculate tile/strip layout and add offset/bytecount tags
        contiguous = not (compression or packints or bilevel)
        if tile:
            # one chunk per tile per plane
            if len(tile) == 2:
                tiles = (
                    (storedshape.length + tile[0] - 1) // tile[0],
                    (storedshape.width + tile[1] - 1) // tile[1],
                )
                contiguous = (
                    contiguous
                    and storedshape.length == tile[0]
                    and storedshape.width == tile[1]
                )
            else:
                tiles = (
                    (storedshape.depth + tile[0] - 1) // tile[0],
                    (storedshape.length + tile[1] - 1) // tile[1],
                    (storedshape.width + tile[2] - 1) // tile[2],
                )
                contiguous = (
                    contiguous
                    and storedshape.depth == tile[0]
                    and storedshape.length == tile[1]
                    and storedshape.width == tile[2]
                )
            numtiles = product(tiles) * storedshape.separate_samples
            databytecounts = [
                product(tile) * storedshape.contig_samples * datadtype.itemsize
            ] * numtiles
            bytecountformat = self._bytecount_format(
                databytecounts, compressiontag
            )
            addtag(
                tags, tagbytecounts, bytecountformat, numtiles, databytecounts
            )
            addtag(tags, tagoffsets, offsetformat, numtiles, [0] * numtiles)
            bytecountformat = f'{numtiles}{bytecountformat}'
            if not contiguous:
                if dataarray is not None:
                    dataiter = iter_tiles(dataarray, tile, tiles)
                elif dataiter is None and not (
                    compression or packints or bilevel
                ):

                    def dataiter_(
                        numtiles: int = numtiles * storedshape.frames,
                        bytecount: int = databytecounts[0],
                    ) -> Iterator[bytes]:
                        # yield empty tiles
                        chunk = bytes(bytecount)
                        for _ in range(numtiles):
                            yield chunk

                    dataiter = dataiter_()

            rowsperstrip = 0

        elif contiguous and (
            rowsperstrip is None or rowsperstrip >= storedshape.length
        ):
            count = storedshape.separate_samples * storedshape.depth
            databytecounts = [
                storedshape.length
                * storedshape.width
                * storedshape.contig_samples
                * datadtype.itemsize
            ] * count
            bytecountformat = self._bytecount_format(
                databytecounts, compressiontag
            )
            addtag(tags, tagbytecounts, bytecountformat, count, databytecounts)
            addtag(tags, tagoffsets, offsetformat, count, [0] * count)
            addtag(tags, 278, 4, 1, storedshape.length)  # RowsPerStrip
            bytecountformat = f'{count}{bytecountformat}'
            rowsperstrip = storedshape.length
            numstrips = count

        else:
            # use rowsperstrip
            rowsize = (
                storedshape.width
                * storedshape.contig_samples
                * datadtype.itemsize
            )
            if compressiontag == 48124:
                # Jetraw works on whole camera frame
                rowsperstrip = storedshape.length
            if rowsperstrip is None:
                # compress ~256 KB chunks by default
                # TIFF-EP requires <= 64 KB
                if compression:
                    rowsperstrip = 262144 // rowsize
                else:
                    rowsperstrip = storedshape.length
            if rowsperstrip < 1:
                rowsperstrip = maxsampling
            elif rowsperstrip > storedshape.length:
                rowsperstrip = storedshape.length
            elif subsampling and rowsperstrip % maxsampling:
                rowsperstrip = (
                    math.ceil(rowsperstrip / maxsampling) * maxsampling
                )
            assert rowsperstrip is not None
            addtag(tags, 278, 4, 1, rowsperstrip)  # RowsPerStrip

            numstrips1 = (
                storedshape.length + rowsperstrip - 1
            ) // rowsperstrip
            numstrips = (
                numstrips1 * storedshape.separate_samples * storedshape.depth
            )
            # TODO: save bilevel data with rowsperstrip
            stripsize = rowsperstrip * rowsize
            databytecounts = [stripsize] * numstrips
            laststripsize = stripsize - rowsize * (
                numstrips1 * rowsperstrip - storedshape.length
            )
            for i in range(numstrips1 - 1, numstrips, numstrips1):
                databytecounts[i] = laststripsize
            bytecountformat = self._bytecount_format(
                databytecounts, compressiontag
            )
            addtag(
                tags, tagbytecounts, bytecountformat, numstrips, databytecounts
            )
            addtag(tags, tagoffsets, offsetformat, numstrips, [0] * numstrips)
            bytecountformat = bytecountformat * numstrips

            if dataarray is not None and not contiguous:
                dataiter = iter_images(dataarray)

        if dataiter is None and not contiguous and _gpu_tensor is None:
            msg = 'cannot write non-contiguous empty file'
            raise ValueError(msg)

        # add extra tags from user; filter duplicate and select tags
        extratag: TagTuple
        tagset = {t[0] for t in tags}
        tagset.update(TIFF.TAG_FILTERED)
        for extratag in extratags:
            if extratag[0] in tagset:
                logger().warning(
                    f'{self!r} not writing extratag {extratag[0]}'
                )
            else:
                addtag(tags, *extratag)
        del tagset
        del extratags

        # TODO: check TIFFReadDirectoryCheckOrder warning in files containing
        #   multiple tags of same code
        # the entries in an IFD must be sorted in ascending order by tag code
        tags = sorted(tags, key=lambda x: x[0])

        # build compression function and encode pipeline
        compressionaxis: int = -2
        bytesiter: bool = False
        tupleiter: bool = False

        iteritem: NDArray[Any] | tuple[bytes, int] | bytes | None
        if dataiter is not None:
            iteritem, dataiter = peek_iterator(dataiter)
            if isinstance(iteritem, tuple):
                tupleiter = True
                iteritem, bytecount = iteritem
                if not isinstance(iteritem, bytes):
                    msg = f'{type(iteritem)=} != bytes'
                    raise TypeError(msg)
            bytesiter = isinstance(iteritem, bytes)
            if not bytesiter:
                iteritem = numpy.asarray(iteritem)
                if (
                    tile
                    and storedshape.contig_samples == 1
                    and iteritem.shape[-1] != 1
                ):
                    # issue 185
                    compressionaxis = -1
                if iteritem.dtype.char != datadtype.char:
                    msg = (
                        f'dtype of iterator {iteritem.dtype!r} '
                        f'does not match dtype {datadtype!r}'
                    )
                    raise ValueError(msg)
        else:
            iteritem = None

        compressionfunc = TiffWriter._build_compress_func(
            bilevel=bilevel,
            compression=compression,
            compressiontag=compressiontag,
            compressionaxis=compressionaxis,
            datadtype=datadtype,
            subsampling=subsampling,
            predictortag=predictortag,
            predictorfunc=predictorfunc,
            compressionargs=compressionargs,
            packints=packints,
            bitspersample=bitspersample,
        )

        del compression
        if not contiguous and not bytesiter and compressionfunc is not None:
            # create iterator of encoded tiles or strips
            bytesiter = True
            if _gpu_tensor is not None and _gpu_algorithm is not None:
                # GPU encoding: whole-plane compression via nvCOMP
                from .gpu import gpu_encode_planes

                dataiter = gpu_encode_planes(
                    _gpu_tensor,
                    _gpu_algorithm,
                    storedshape,
                    predictortag,
                    compressionaxis,
                )
                _gpu_tensor = None
            elif _gpu_tensor is not None:
                # GPU tensor but no GPU encoder — D2H + CPU path
                from .gpu import tensor_to_numpy

                logger().info(
                    'GPU encoder unavailable, falling back to CPU'
                )
                dataarray = tensor_to_numpy(_gpu_tensor).reshape(
                    storedshape.shape
                )
                _gpu_tensor = None
                if tile:
                    dataiter = iter_tiles(dataarray, tile, tiles)
                    tileshape = (*tile, storedshape.contig_samples)
                    tilesize = (
                        product(tileshape) * datadtype.itemsize
                    )
                    maxworkers = TiffWriter._maxworkers(
                        maxworkers,
                        numtiles * storedshape.frames,
                        tilesize,
                        compressiontag,
                    )
                    dataiter = encode_chunks(
                        numtiles * storedshape.frames,
                        dataiter,
                        compressionfunc,
                        tileshape,
                        datadtype,
                        maxworkers,
                        buffersize,
                        True,
                    )
                else:
                    dataiter = iter_images(dataarray)
                    maxworkers = TiffWriter._maxworkers(
                        maxworkers,
                        numstrips * storedshape.frames,
                        stripsize,
                        compressiontag,
                    )
                    dataiter = iter_strips(
                        dataiter,
                        storedshape.page_shape,
                        datadtype,
                        rowsperstrip,
                    )
                    dataiter = encode_chunks(
                        numstrips * storedshape.frames,
                        dataiter,
                        compressionfunc,
                        (
                            rowsperstrip,
                            storedshape.width,
                            storedshape.contig_samples,
                        ),
                        datadtype,
                        maxworkers,
                        buffersize,
                        False,
                    )
            elif tile:
                # dataiter yields tiles
                tileshape = (*tile, storedshape.contig_samples)
                tilesize = product(tileshape) * datadtype.itemsize
                maxworkers = TiffWriter._maxworkers(
                    maxworkers,
                    numtiles * storedshape.frames,
                    tilesize,
                    compressiontag,
                )
                # yield encoded tiles
                dataiter = encode_chunks(
                    numtiles * storedshape.frames,
                    dataiter,  # type: ignore[arg-type]
                    compressionfunc,
                    tileshape,
                    datadtype,
                    maxworkers,
                    buffersize,
                    True,
                )
            else:
                # dataiter yields frames
                maxworkers = TiffWriter._maxworkers(
                    maxworkers,
                    numstrips * storedshape.frames,
                    stripsize,
                    compressiontag,
                )
                # yield strips
                dataiter = iter_strips(
                    dataiter,  # type: ignore[arg-type]
                    storedshape.page_shape,
                    datadtype,
                    rowsperstrip,
                )
                # yield encoded strips
                dataiter = encode_chunks(
                    numstrips * storedshape.frames,
                    dataiter,
                    compressionfunc,
                    (
                        rowsperstrip,
                        storedshape.width,
                        storedshape.contig_samples,
                    ),
                    datadtype,
                    maxworkers,
                    buffersize,
                    False,
                )

        # D2H fallback for uncompressed GPU tensor writes
        if _gpu_tensor is not None and dataarray is None:
            from .gpu import tensor_to_numpy

            dataarray = tensor_to_numpy(_gpu_tensor).reshape(
                storedshape.shape
            )
            _gpu_tensor = None
            if tile and not contiguous:
                dataiter = iter_tiles(dataarray, tile, tiles)
            elif not contiguous:
                dataiter = iter_images(dataarray)

        # write IFDs and image data
        fhpos = fh.tell()
        # commented out to allow image data beyond 4GB in classic TIFF
        # if (
        #     not (
        #         offsetsize > 4
        #         or self._imagej or compressionfunc is not None
        #     )
        #     and fhpos + datasize > 2**32 - 1
        # ):
        #     raise ValueError('data too large for classic TIFF format')

        dataoffset: int = 0

        # if not compressed or multi-tiled, write the first IFD and then
        # all data contiguously; else, write all IFDs and data interleaved
        for pageindex in range(1 if contiguous else storedshape.frames):
            ifdpos = fhpos
            if ifdpos % 2:
                # position of IFD must begin on a word boundary
                fh.write(b'\x00')
                ifdpos += 1

            if self._subifdslevel < 0:
                # update pointer at ifdoffset
                fh.seek(self._ifdoffset)
                fh.write(pack(offsetformat, ifdpos))

            fh.seek(ifdpos)

            # create IFD in memory
            if pageindex < 2:
                subifdsoffsets = None
                ifd = io.BytesIO()
                ifd.write(pack(tagnoformat, len(tags)))
                tagoffset = ifd.tell()
                ifd.write(b''.join(t[1] for t in tags))
                ifdoffset = ifd.tell()
                ifd.write(pack(offsetformat, 0))  # offset to next IFD
                # write tag values and patch offsets in ifdentries
                for tagindex, tag in enumerate(tags):
                    offset = tagoffset + tagindex * tagsize + 4 + offsetsize
                    code = tag[0]
                    value = tag[2]
                    if value:
                        pos = ifd.tell()
                        if pos % 2:
                            # tag value is expected to begin on word boundary
                            ifd.write(b'\x00')
                            pos += 1
                        ifd.seek(offset)
                        ifd.write(pack(offsetformat, ifdpos + pos))
                        ifd.seek(pos)
                        ifd.write(value)
                        if code == tagoffsets:
                            dataoffsetsoffset = offset, pos
                        elif code == tagbytecounts:
                            databytecountsoffset = offset, pos
                        elif code == 270:
                            if (
                                self._descriptiontag is not None
                                and self._descriptiontag.offset == 0
                                and value.startswith(
                                    self._descriptiontag.value
                                )
                            ):
                                self._descriptiontag.offset = (
                                    ifdpos + tagoffset + tagindex * tagsize
                                )
                                self._descriptiontag.valueoffset = ifdpos + pos
                        elif code == 330:
                            subifdsoffsets = offset, pos
                    elif code == tagoffsets:
                        dataoffsetsoffset = offset, None
                    elif code == tagbytecounts:
                        databytecountsoffset = offset, None
                    elif code == 270:
                        if (
                            self._descriptiontag is not None
                            and self._descriptiontag.offset == 0
                            and self._descriptiontag.value in tag[1][-4:]
                        ):
                            self._descriptiontag.offset = (
                                ifdpos + tagoffset + tagindex * tagsize
                            )
                            self._descriptiontag.valueoffset = (
                                self._descriptiontag.offset + offsetsize + 4
                            )
                    elif code == 330:
                        subifdsoffsets = offset, None
                ifdsize = ifd.tell()
                if ifdsize % 2:
                    ifd.write(b'\x00')
                    ifdsize += 1

            # write IFD later when strip/tile bytecounts and offsets are known
            fh.seek(ifdsize, os.SEEK_CUR)

            # write image data
            dataoffset = fh.tell()
            if align is None:
                align = 16
            skip = (align - (dataoffset % align)) % align
            fh.seek(skip, os.SEEK_CUR)
            dataoffset += skip

            if contiguous:
                # write all image data contiguously
                if dataiter is not None:
                    byteswritten = 0
                    if bytesiter:
                        for iteritem in dataiter:
                            # assert isinstance(iteritem, bytes)
                            byteswritten += fh.write(
                                iteritem  # type: ignore[arg-type]
                            )
                            del iteritem
                    else:
                        pagesize = storedshape.page_size * datadtype.itemsize
                        for iteritem in dataiter:
                            if iteritem is None:
                                byteswritten += fh.write_empty(pagesize)
                            else:
                                # assert isinstance(iteritem, numpy.ndarray)
                                byteswritten += fh.write_array(
                                    iteritem,  # type: ignore[arg-type]
                                    datadtype,
                                )
                            del iteritem
                    if byteswritten != datasize:
                        msg = (
                            'iterator contains wrong number of bytes '
                            f'{byteswritten} != {datasize}'
                        )
                        raise ValueError(msg)
                elif dataarray is None:
                    fh.write_empty(datasize)
                else:
                    fh.write_array(dataarray, datadtype)

            elif tupleiter:
                # write tiles or strips from iterator of tuples
                assert dataiter is not None
                dataoffsets = [0] * (numtiles if tile else numstrips)
                offset = dataoffset
                for chunkindex in range(numtiles if tile else numstrips):
                    iteritem, bytecount = cast(
                        tuple[bytes, int], next(dataiter)
                    )
                    # assert bytecount >= len(iteritem)
                    databytecounts[chunkindex] = bytecount
                    dataoffsets[chunkindex] = offset
                    offset += len(iteritem)
                    fh.write(iteritem)
                    del iteritem

            elif bytesiter:
                # write tiles or strips
                assert dataiter is not None
                for chunkindex in range(numtiles if tile else numstrips):
                    iteritem = cast(bytes, next(dataiter))
                    # assert isinstance(iteritem, bytes)
                    databytecounts[chunkindex] = len(iteritem)
                    fh.write(iteritem)
                    del iteritem

            elif tile:
                # write uncompressed tiles
                assert dataiter is not None
                tileshape = (*tile, storedshape.contig_samples)
                tilesize = product(tileshape) * datadtype.itemsize
                for tileindex in range(numtiles):
                    iteritem = next(dataiter)
                    if iteritem is None:
                        databytecounts[tileindex] = 0
                        # fh.write_empty(tilesize)
                        continue
                    # assert not isinstance(iteritem, bytes)
                    iteritem = numpy.ascontiguousarray(iteritem, datadtype)
                    if iteritem.nbytes != tilesize:
                        # if iteritem.dtype != datadtype:
                        #     msg = 'dtype of tile does not match data'
                        #     raise ValueError()
                        if iteritem.nbytes > tilesize:
                            msg = 'tile is too large'
                            raise ValueError(msg)
                        pad = tuple(
                            (0, i - j)
                            for i, j in zip(
                                tileshape, iteritem.shape, strict=False
                            )
                        )
                        iteritem = numpy.pad(iteritem, pad)
                    fh.write_array(iteritem)
                    del iteritem

            else:
                msg = 'unreachable code'
                raise RuntimeError(msg)

            # update strip/tile offsets
            assert dataoffsetsoffset is not None
            offset, pos = dataoffsetsoffset
            ifd.seek(offset)
            if pos is not None:
                ifd.write(pack(offsetformat, ifdpos + pos))
                ifd.seek(pos)
                if dataoffsets is None:
                    offset = dataoffset
                    for size in databytecounts:
                        ifd.write(
                            pack(offsetformat, offset if size > 0 else 0)
                        )
                        offset += size
                else:
                    for offset in dataoffsets:
                        ifd.write(pack(offsetformat, offset))
            else:
                ifd.write(pack(offsetformat, dataoffset))

            if compressionfunc is not None or (tile and dataarray is None):
                # update strip/tile bytecounts
                assert databytecountsoffset is not None
                offset, pos = databytecountsoffset
                ifd.seek(offset)
                if pos is not None:
                    ifd.write(pack(offsetformat, ifdpos + pos))
                    ifd.seek(pos)
                ifd.write(pack(bytecountformat, *databytecounts))

            if subifdsoffsets is not None:
                # update and save pointer to SubIFDs tag values if necessary
                offset, pos = subifdsoffsets
                if pos is not None:
                    ifd.seek(offset)
                    ifd.write(pack(offsetformat, ifdpos + pos))
                    self._subifdsoffsets.append(ifdpos + pos)
                else:
                    self._subifdsoffsets.append(ifdpos + offset)

            fhpos = fh.tell()
            fh.seek(ifdpos)
            fh.write(ifd.getbuffer())
            fh.flush()

            if self._subifdslevel < 0:
                self._ifdoffset = ifdpos + ifdoffset
            else:
                # update SubIFDs tag values
                fh.seek(
                    self._subifdsoffsets[self._ifdindex]
                    + self._subifdslevel * offsetsize
                )
                fh.write(pack(offsetformat, ifdpos))

                # update SubIFD chain offsets
                if self._subifdslevel == 0:
                    self._nextifdoffsets.append(ifdpos + ifdoffset)
                else:
                    fh.seek(self._nextifdoffsets[self._ifdindex])
                    fh.write(pack(offsetformat, ifdpos))
                    self._nextifdoffsets[self._ifdindex] = ifdpos + ifdoffset
                self._ifdindex += 1
                self._ifdindex %= len(self._subifdsoffsets)

            fh.seek(fhpos)

            # remove tags that should be written only once
            if pageindex == 0:
                tags = [tag for tag in tags if not tag[-1]]

        assert dataoffset > 0

        self._datashape = (1, *inputshape)
        self._datadtype = datadtype
        self._dataoffset = dataoffset
        self._databytecounts = databytecounts
        self._storedshape = storedshape

        if contiguous:
            # write remaining IFDs/tags later
            self._tags = tags
            # return offset and size of image data
            if returnoffset:
                return dataoffset, sum(databytecounts)
        return None

    @staticmethod
    def _resolve_write_data(
        data: Any,
        shape: Sequence[int] | None,
        dtype: DTypeLike | None,
        byteorder: str,
    ) -> tuple[
        NDArray[Any] | None,
        Iterator[NDArray[Any] | bytes | None] | None,
        tuple[int, ...],
        numpy.dtype[Any],
    ]:
        """Return array, iterator, shape, dtype from write data arguments."""
        if data is None:
            # empty
            if shape is None or dtype is None:
                msg = "missing required 'shape' or 'dtype' arguments"
                raise ValueError(msg)
            return None, None, tuple(shape), numpy.dtype(dtype).newbyteorder(
                byteorder
            )

        if hasattr(data, '__next__'):
            # iterator/generator
            if shape is None or dtype is None:
                msg = "missing required 'shape' or 'dtype' arguments"
                raise ValueError(msg)
            return (
                None,
                data,
                tuple(shape),
                numpy.dtype(dtype).newbyteorder(byteorder),
            )

        if hasattr(data, 'dtype'):
            # numpy, zarr, or dask array
            data = cast(numpy.ndarray, data)
            dataarray: NDArray[Any] = data
            datadtype = numpy.dtype(data.dtype).newbyteorder(byteorder)
            if not hasattr(data, 'reshape'):
                # zarr array cannot be shape-normalized
                dataarray = numpy.asarray(data, datadtype, 'C')
            else:
                try:
                    # numpy array must be C contiguous
                    if data.flags.f_contiguous:
                        dataarray = numpy.asarray(data, datadtype, 'C')
                except AttributeError:
                    # not a numpy array
                    pass
            if dtype is not None and numpy.dtype(dtype) != datadtype:
                msg = (
                    f'dtype argument {dtype!r} does not match '
                    f'data dtype {datadtype}'
                )
                raise ValueError(msg)
            if shape is not None and shape != dataarray.shape:
                msg = (
                    f'shape argument {shape!r} does not match '
                    f'data shape {dataarray.shape}'
                )
                raise ValueError(msg)
            return dataarray, None, dataarray.shape, datadtype

        # scalar, list, tuple, etc
        datadtype = numpy.dtype(dtype).newbyteorder(byteorder)
        dataarray = numpy.asarray(data, datadtype, 'C')
        return dataarray, None, dataarray.shape, datadtype

    @staticmethod
    def _build_compress_func(
        *,
        bilevel: bool,
        compression: bool,
        compressiontag: int,
        compressionaxis: int,
        datadtype: numpy.dtype[Any],
        subsampling: tuple[int, int] | None,
        predictortag: int,
        predictorfunc: Callable[..., Any] | None,
        compressionargs: dict[str, Any],
        packints: bool,
        bitspersample: int,
    ) -> Callable[..., Any] | None:
        """Return compression function for write().

        Build a closure that compresses data for a single tile or strip.

        """
        if bilevel:
            if compressiontag == 1:

                def compressionfunc1(
                    data: Any, axis: int = compressionaxis
                ) -> bytes:
                    return numpy.packbits(data, axis=axis).tobytes()

                return compressionfunc1

            if compressiontag in {5, 32773, 8, 32946, 50013, 34925, 50000}:
                # LZW, PackBits, deflate, LZMA, ZSTD
                def compressionfunc2(
                    data: Any,
                    compressor: Any = TIFF.COMPRESSORS[compressiontag],
                    axis: int = compressionaxis,
                    kwargs: Any = compressionargs,
                ) -> bytes:
                    data = numpy.packbits(data, axis=axis).tobytes()
                    return compressor(data, **kwargs)

                return compressionfunc2

            msg = 'cannot compress bilevel image'
            raise NotImplementedError(msg)

        if compression:
            compressor = TIFF.COMPRESSORS[compressiontag]

            if compressiontag == 32773:
                # PackBits
                compressionargs['axis'] = compressionaxis

            if subsampling:
                # JPEG with subsampling
                def compressionfunc3(
                    data: Any,
                    compressor: Any = compressor,
                    kwargs: Any = compressionargs,
                ) -> bytes:
                    return compressor(data, **kwargs)

                return compressionfunc3

            if predictortag > 1:

                def compressionfunc4(
                    data: Any,
                    predictorfunc: Any = predictorfunc,
                    compressor: Any = compressor,
                    axis: int = compressionaxis,
                    kwargs: Any = compressionargs,
                ) -> bytes:
                    data = predictorfunc(data, axis=axis)
                    return compressor(data, **kwargs)

                return compressionfunc4

            if compressionargs:

                def compressionfunc5(
                    data: Any,
                    compressor: Any = compressor,
                    kwargs: Any = compressionargs,
                ) -> bytes:
                    return compressor(data, **kwargs)

                return compressionfunc5

            if compressiontag > 1:
                return compressor

            return None

        if packints:

            def compressionfunc6(
                data: Any,
                bps: Any = bitspersample,
                axis: int = compressionaxis,
            ) -> bytes:
                return imagecodecs.packints_encode(
                    data, bps, axis=axis
                )  # type: ignore[return-value]

            return compressionfunc6

        return None

    def overwrite_description(self, description: str, /) -> None:
        """Overwrite value of last ImageDescription tag.

        Can be used to write OME-XML after writing images.
        Ends a contiguous series.

        """
        if self._descriptiontag is None:
            msg = 'no ImageDescription tag found'
            raise ValueError(msg)
        self._write_remaining_pages()
        self._descriptiontag.overwrite(description, erase=False)
        self._descriptiontag = None

    @staticmethod
    def _detect_gpu_encoder(
        gpu_tensor: Any,
        compression: Any,
        packints: bool,
        compressiontag: int,
        predictortag: int,
        storedshape: Any,
        datadtype: Any,
    ) -> str | None:
        """Return GPU encoder algorithm name, or None if unavailable."""
        if gpu_tensor is None or not compression or packints:
            return None
        from .gpu import GpuEncoderRegistry

        algorithm = GpuEncoderRegistry.get(compressiontag)
        if algorithm is None:
            return None
        # float predictor not supported on GPU
        if predictortag in (3, 34894, 34895):
            return None
        plane_bytes = (
            storedshape.length
            * storedshape.width
            * storedshape.contig_samples
            * datadtype.itemsize
        )
        if plane_bytes < GpuEncoderRegistry.MIN_PLANE_BYTES:
            logger().info(
                f'GPU encoder skipped: plane too small '
                f'({plane_bytes} < '
                f'{GpuEncoderRegistry.MIN_PLANE_BYTES} bytes)'
            )
            return None
        return algorithm

    def close(self) -> None:
        """Write remaining pages and close file handle."""
        try:
            if not self._truncate:
                self._write_remaining_pages()
            self._write_image_description()
        finally:
            with contextlib.suppress(Exception):
                self._fh.close()

    @property
    def filehandle(self) -> FileHandle:
        """File handle to write file."""
        return self._fh

    def _write_remaining_pages(self) -> None:
        """Write outstanding IFDs and tags to file."""
        if not self._tags or self._truncate or self._datashape is None:
            return

        assert self._storedshape is not None
        assert self._databytecounts is not None
        assert self._dataoffset is not None

        pageno: int = self._storedshape.frames * self._datashape[0] - 1
        if pageno < 1:
            self._tags = None
            self._dataoffset = None
            self._databytecounts = None
            return

        fh = self._fh
        fhpos: int = fh.tell()
        if fhpos % 2:
            fh.write(b'\x00')
            fhpos += 1

        pack = struct.pack
        offsetformat: str = self.tiff.offsetformat
        offsetsize: int = self.tiff.offsetsize
        tagnoformat: str = self.tiff.tagnoformat
        tagsize: int = self.tiff.tagsize
        dataoffset: int = self._dataoffset
        pagedatasize: int = sum(self._databytecounts)
        subifdsoffsets: tuple[int, int | None] | None = None
        dataoffsetsoffset: tuple[int, int | None]
        pos: int | None
        offset: int

        # construct template IFD in memory
        # must patch offsets to next IFD and data before writing to file
        ifd = io.BytesIO()
        ifd.write(pack(tagnoformat, len(self._tags)))
        tagoffset = ifd.tell()
        ifd.write(b''.join(t[1] for t in self._tags))
        ifdoffset = ifd.tell()
        ifd.write(pack(offsetformat, 0))  # offset to next IFD
        # tag values
        for tagindex, tag in enumerate(self._tags):
            offset = tagoffset + tagindex * tagsize + offsetsize + 4
            code = tag[0]
            value = tag[2]
            if value:
                pos = ifd.tell()
                if pos % 2:
                    # tag value is expected to begin on word boundary
                    ifd.write(b'\x00')
                    pos += 1
                ifd.seek(offset)
                try:
                    ifd.write(pack(offsetformat, fhpos + pos))
                except Exception as exc:  # struct.error
                    if self._imagej:
                        warnings.warn(
                            f'{self!r} truncating ImageJ file',
                            UserWarning,
                            stacklevel=2,
                        )
                        self._truncate = True
                        return
                    msg = 'data too large for non-BigTIFF file'
                    raise ValueError(msg) from exc
                ifd.seek(pos)
                ifd.write(value)
                if code == self._dataoffsetstag:
                    # save strip/tile offsets for later updates
                    dataoffsetsoffset = offset, pos
                elif code == 330:
                    # save subifds offsets for later updates
                    subifdsoffsets = offset, pos
            elif code == self._dataoffsetstag:
                dataoffsetsoffset = offset, None
            elif code == 330:
                subifdsoffsets = offset, None

        ifdsize = ifd.tell()
        if ifdsize % 2:
            ifd.write(b'\x00')
            ifdsize += 1

        # check if all IFDs fit in file
        if offsetsize < 8 and fhpos + ifdsize * pageno > 2**32 - 32:
            if self._imagej:
                warnings.warn(
                    f'{self!r} truncating ImageJ file',
                    UserWarning,
                    stacklevel=2,
                )
                self._truncate = True
                return
            msg = 'data too large for non-BigTIFF file'
            raise ValueError(msg)

        # assemble IFD chain in memory from IFD template
        ifds = io.BytesIO(bytes(ifdsize * pageno))
        ifdpos = fhpos
        for _ in range(pageno):
            # update strip/tile offsets in IFD
            dataoffset += pagedatasize  # offset to image data
            offset, pos = dataoffsetsoffset
            ifd.seek(offset)
            if pos is not None:
                ifd.write(pack(offsetformat, ifdpos + pos))
                ifd.seek(pos)
                offset = dataoffset
                for size in self._databytecounts:
                    ifd.write(pack(offsetformat, offset))
                    offset += size
            else:
                ifd.write(pack(offsetformat, dataoffset))

            if subifdsoffsets is not None:
                offset, pos = subifdsoffsets
                self._subifdsoffsets.append(
                    ifdpos + (pos if pos is not None else offset)
                )

            if self._subifdslevel < 0:
                if subifdsoffsets is not None:
                    # update pointer to SubIFDs tag values if necessary
                    offset, pos = subifdsoffsets
                    if pos is not None:
                        ifd.seek(offset)
                        ifd.write(pack(offsetformat, ifdpos + pos))

                # update pointer at ifdoffset to point to next IFD in file
                ifdpos += ifdsize
                ifd.seek(ifdoffset)
                ifd.write(pack(offsetformat, ifdpos))

            else:
                # update SubIFDs tag values in file
                fh.seek(
                    self._subifdsoffsets[self._ifdindex]
                    + self._subifdslevel * offsetsize
                )
                fh.write(pack(offsetformat, ifdpos))

                # update SubIFD chain
                if self._subifdslevel == 0:
                    self._nextifdoffsets.append(ifdpos + ifdoffset)
                else:
                    fh.seek(self._nextifdoffsets[self._ifdindex])
                    fh.write(pack(offsetformat, ifdpos))
                    self._nextifdoffsets[self._ifdindex] = ifdpos + ifdoffset
                self._ifdindex += 1
                self._ifdindex %= len(self._subifdsoffsets)
                ifdpos += ifdsize

            # write IFD entry
            ifds.write(ifd.getbuffer())

        # terminate IFD chain
        ifdoffset += ifdsize * (pageno - 1)
        ifds.seek(ifdoffset)
        ifds.write(pack(offsetformat, 0))
        # write IFD chain to file
        fh.seek(fhpos)
        fh.write(ifds.getbuffer())

        if self._subifdslevel < 0:
            # update file to point to new IFD chain
            pos = fh.tell()
            fh.seek(self._ifdoffset)
            fh.write(pack(offsetformat, fhpos))
            fh.flush()
            fh.seek(pos)
            self._ifdoffset = fhpos + ifdoffset

        self._tags = None
        self._dataoffset = None
        self._databytecounts = None
        # do not reset _storedshape, _datashape, _datadtype

    def _write_image_description(self) -> None:
        """Write metadata to ImageDescription tag."""
        if self._datashape is None or self._descriptiontag is None:
            self._descriptiontag = None
            return

        assert self._storedshape is not None
        assert self._datadtype is not None

        if self._omexml is not None:
            if self._subifdslevel < 0:
                assert self._metadata is not None
                self._omexml.addimage(
                    dtype=self._datadtype,
                    shape=self._datashape[
                        0 if self._datashape[0] != 1 else 1 :
                    ],
                    storedshape=self._storedshape.shape,
                    **self._metadata,
                )
            description = self._omexml.tostring(declaration=True)
        elif self._datashape[0] == 1:
            # description already up-to-date
            self._descriptiontag = None
            return
        # elif self._subifdslevel >= 0:
        #     # don't write metadata to SubIFDs
        #     return
        elif self._imagej:
            assert self._metadata is not None
            colormapped = self._colormap is not None
            isrgb = self._storedshape.samples in {3, 4}
            description = imagej_description(
                self._datashape,
                rgb=isrgb,
                colormapped=colormapped,
                **self._metadata,
            )
        elif not self._tifffile:
            self._descriptiontag = None
            return
        else:
            assert self._metadata is not None
            description = shaped_description(self._datashape, **self._metadata)

        self._descriptiontag.overwrite(description.encode(), erase=False)
        self._descriptiontag = None

    def _addtag(
        self,
        tags: list[tuple[int, bytes, bytes | None, bool]],
        code: int | str,
        dtype: int | str,
        count: int | None,
        value: Any,
        writeonce: bool = False,  # noqa: FBT001, FBT002
        /,
    ) -> None:
        """Append (code, ifdentry, ifdvalue, writeonce) to tags list.

        Compute ifdentry and ifdvalue bytes from code, dtype, count, value.

        """
        pack = self._pack

        if not isinstance(code, int):
            code = TIFF.TAGS[code]
        try:
            datatype = cast(int, dtype)
            dataformat = TIFF.DATA_FORMATS[datatype][-1]
        except KeyError:
            try:
                dataformat = cast(str, dtype)
                if dataformat[0] in '<>':
                    dataformat = dataformat[1:]
                datatype = TIFF.DATA_DTYPES[dataformat]
            except (KeyError, TypeError):
                msg = f'unknown {dtype=}'
                raise ValueError(msg) from None
        del dtype

        rawcount = count
        if datatype == 2:
            # string
            if isinstance(value, str):
                # enforce 7-bit ASCII on Unicode strings
                try:
                    value = value.encode('ascii')
                except UnicodeEncodeError as exc:
                    msg = 'TIFF strings must be 7-bit ASCII'
                    raise ValueError(msg) from exc
            elif not isinstance(value, bytes):
                msg = 'TIFF strings must be 7-bit ASCII'
                raise ValueError(msg)

            if len(value) == 0 or value[-1:] != b'\x00':
                value += b'\x00'
            count = len(value)
            if code == 270:
                rawcount = int(value.find(b'\x00\x00'))
                if rawcount < 0:
                    rawcount = count
                else:
                    # length of string without buffer
                    rawcount = max(self.tiff.offsetsize + 1, rawcount + 1)
                    rawcount = min(count, rawcount)
            else:
                rawcount = count
            value = (value,)

        elif isinstance(value, bytes):
            # packed binary data
            itemsize = struct.calcsize(dataformat)
            if len(value) % itemsize:
                msg = 'invalid packed binary data'
                raise ValueError(msg)
            count = len(value) // itemsize
            rawcount = count

        elif count is None:
            msg = 'invalid count'
            raise ValueError(msg)
        else:
            count = int(count)

        if datatype in {5, 10}:  # rational
            count *= 2
            dataformat = dataformat[-1]

        ifdentry = [
            pack('HH', code, datatype),
            pack(self.tiff.offsetformat, rawcount),
        ]

        ifdvalue = None
        if struct.calcsize(dataformat) * count <= self.tiff.offsetsize:
            # value(s) can be written directly
            valueformat = f'{self.tiff.offsetsize}s'
            if isinstance(value, bytes):
                ifdentry.append(pack(valueformat, value))
            elif count == 1:
                if isinstance(value, (tuple, list, numpy.ndarray)):
                    value = value[0]
                ifdentry.append(pack(valueformat, pack(dataformat, value)))
            else:
                ifdentry.append(
                    pack(valueformat, pack(f'{count}{dataformat}', *value))
                )
        else:
            # use offset to value(s)
            ifdentry.append(pack(self.tiff.offsetformat, 0))
            if isinstance(value, bytes):
                ifdvalue = value
            elif isinstance(value, numpy.ndarray):
                if value.size != count:
                    msg = 'value.size != count'
                    raise RuntimeError(msg)
                if value.dtype.char != dataformat:
                    msg = 'value.dtype.char != dtype'
                    raise RuntimeError(msg)
                ifdvalue = value.tobytes()
            elif isinstance(value, (tuple, list)):
                ifdvalue = pack(f'{count}{dataformat}', *value)
            else:
                ifdvalue = pack(dataformat, value)
        tags.append((code, b''.join(ifdentry), ifdvalue, writeonce))

    def _pack(self, fmt: str, *val: Any) -> bytes:
        """Return values packed to bytes according to format."""
        if fmt[0] not in '<>':
            fmt = self.tiff.byteorder + fmt
        return struct.pack(fmt, *val)

    def _bytecount_format(
        self, bytecounts: Sequence[int], compression: int, /
    ) -> str:
        """Return small bytecount format."""
        if len(bytecounts) == 1:
            return self.tiff.offsetformat[1]
        bytecount = bytecounts[0]
        if compression > 1:
            bytecount = bytecount * 10
        if bytecount < 2**16:
            return 'H'
        if bytecount < 2**32:
            return 'I'
        return self.tiff.offsetformat[1]

    @staticmethod
    def _maxworkers(
        maxworkers: int | None,
        numchunks: int,
        chunksize: int,
        compression: int,
    ) -> int:
        """Return number of threads to encode segments."""
        if maxworkers is not None:
            return maxworkers
        if (
            # imagecodecs is None or
            compression <= 1
            or numchunks < 2
            or chunksize < 1024
            or compression == 48124  # Jetraw is not thread-safe?
        ):
            return 1
        # the following is based on benchmarking RGB tile sizes vs maxworkers
        # using a (8228, 11500, 3) uint8 WSI slide:
        if chunksize < 131072 and compression in {
            7,  # JPEG
            33007,  # ALT_JPG
            34892,  # JPEG_LOSSY
            32773,  # PackBits
            34887,  # LERC
        }:
            return 1
        if chunksize < 32768 and compression in {
            5,  # LZW
            8,  # zlib
            32946,  # zlib
            50000,  # zstd
            50013,  # zlib/pixtiff
        }:
            # zlib,
            return 1
        if chunksize < 8192 and compression in {
            34934,  # JPEG XR
            22610,  # JPEG XR
            34933,  # PNG
        }:
            return 1
        if chunksize < 2048 and compression in {
            33003,  # JPEG2000
            33004,  # JPEG2000
            33005,  # JPEG2000
            34712,  # JPEG2000
            50002,  # JPEG XL
            52546,  # JPEG XL DNG
        }:
            return 1
        if chunksize < 1024 and compression in {
            34925,  # LZMA
            50001,  # WebP
        }:
            return 1
        if compression == 34887:  # LERC
            # limit to 4 threads
            return min(numchunks, 4)
        return min(numchunks, TIFF.MAXWORKERS)

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()

    def __repr__(self) -> str:
        return f'<tifffile.TiffWriter {snipstr(self.filehandle.name, 32)!r}>'


@final
class TiffFile:
    """Read image and metadata from TIFF file.

    TiffFile instances must be closed with :py:meth:`TiffFile.close`, which
    is automatically called when using the 'with' context manager.

    TiffFile instances are not thread-safe. All attributes are read-only.

    Parameters:
        file:
            Specifies TIFF file to read.
            File objects must be open in binary mode and positioned at the
            TIFF header.
        mode:
            File open mode if `file` is file name. The default is 'rb'.
        name:
            Name of file if `file` is file handle.
        offset:
            Start position of embedded file.
            The default is the current file position.
        size:
            Size of embedded file. The default is the number of bytes
            from the `offset` to the end of the file.
        omexml:
            OME metadata in XML format, for example, from external companion
            file or sanitized XML overriding XML in file.
        superres:
            EER super-resolution level to decode.
            The default is 0 (no super-resolution).
        _multifile, _useframes, _parent:
            Internal use.
        **is_flags:
            Override `TiffFile.is_` flags, for example:

            ``is_ome=False``: disable processing of OME-XML metadata.
            ``is_lsm=False``: disable special handling of LSM files.
            ``is_ndpi=True``: force file to be NDPI format.

    Raises:
        TiffFileError: Invalid TIFF structure.

    """

    tiff: TiffFormat
    """Properties of TIFF file format."""

    pages: TiffPages
    """Sequence of pages in TIFF file."""

    _fh: FileHandle
    _multifile: bool
    _parent: TiffFile  # OME master file
    _files: dict[str | None, TiffFile]  # cache of TiffFile instances
    _omexml: str | None  # external OME-XML
    _superres: int  # EER super-resolution level
    _decoders: dict[  # cache of TiffPage.decode functions
        int,
        Callable[
            ...,
            tuple[
                NDArray[Any] | None,
                tuple[int, int, int, int, int],
                tuple[int, int, int, int],
            ],
        ],
    ]

    def __init__(
        self,
        file: str | os.PathLike[Any] | FileHandle | IO[bytes],
        /,
        *,
        mode: Literal['r', 'r+'] | None = None,
        name: str | None = None,
        offset: int | None = None,
        size: int | None = None,
        omexml: str | None = None,
        superres: int | None = None,
        _multifile: bool | None = None,
        _useframes: bool | None = None,
        _parent: TiffFile | None = None,
        **is_flags: bool | None,
    ) -> None:
        for key, value in is_flags.items():
            if key[:3] == 'is_' and key[3:] in TIFF.FILE_FLAGS:
                if value is not None:
                    setattr(self, key, bool(value))
            else:
                msg = f'unexpected keyword argument: {key}'
                raise TypeError(msg)

        if mode not in {None, 'r', 'r+', 'rb', 'r+b'}:
            msg = f'invalid {mode=}'
            raise ValueError(msg)

        self._omexml = None
        if omexml:
            if omexml.strip()[-4:] != 'OME>':
                msg = 'invalid OME-XML'
                raise ValueError(msg)
            self._omexml = omexml
            self.is_ome = True

        fh = FileHandle(file, mode=mode, name=name, offset=offset, size=size)
        self._fh = fh
        self._multifile = True if _multifile is None else bool(_multifile)
        self._files = {fh.name: self}
        self._decoders = {}
        self._parent = self if _parent is None else _parent
        self._superres = 0 if superres is None else max(0, int(superres))

        try:
            fh.seek(0)
            header = fh.read(4)
            try:
                byteorder = {b'II': '<', b'MM': '>', b'EP': '<'}[header[:2]]
            except KeyError:
                msg = f'not a TIFF file: {header=!r}'
                raise TiffFileError(msg) from None

            version = struct.unpack(byteorder + 'H', header[2:4])[0]
            if version == 43:
                # BigTiff
                offsetsize, zero = struct.unpack(byteorder + 'HH', fh.read(4))
                if zero != 0 or offsetsize != 8:
                    msg = f'invalid BigTIFF offset size {(offsetsize, zero)}'
                    raise TiffFileError(msg)
                if byteorder == '>':
                    self.tiff = TIFF.BIG_BE
                else:
                    self.tiff = TIFF.BIG_LE
            elif version == 42:
                # Classic TIFF
                if byteorder == '>':
                    self.tiff = TIFF.CLASSIC_BE
                elif is_flags.get('is_ndpi', fh.extension == '.ndpi'):
                    # NDPI uses 64 bit IFD offsets
                    if is_flags.get('is_ndpi', True):
                        self.tiff = TIFF.NDPI_LE
                    else:
                        self.tiff = TIFF.CLASSIC_LE
                else:
                    self.tiff = TIFF.CLASSIC_LE
            elif version == 0x4352:
                # DNG DCP
                if byteorder == '>':
                    self.tiff = TIFF.CLASSIC_BE
                else:
                    self.tiff = TIFF.CLASSIC_LE
            elif version == 0x4E31:
                # NIFF
                if byteorder == '>':
                    msg = 'invalid NIFF file'
                    raise TiffFileError(msg)
                logger().error(f'{self!r} NIFF format not supported')
                self.tiff = TIFF.CLASSIC_LE
            elif version in {0x55, 0x4F52, 0x5352}:
                # Panasonic or Olympus RAW
                logger().error(
                    f'{self!r} RAW format 0x{version:04X} not supported'
                )
                if byteorder == '>':
                    self.tiff = TIFF.CLASSIC_BE
                else:
                    self.tiff = TIFF.CLASSIC_LE
            else:
                msg = f'invalid TIFF {version=}'
                raise TiffFileError(msg)

            # file handle is at offset to offset to first page
            self.pages = TiffPages(self)

            if self.is_lsm and (
                self.filehandle.size >= 2**32
                or self.pages[0].compression != 1
                or self.pages[1].compression != 1
            ):
                self._lsm_load_pages()

            elif self.is_scanimage and not self.is_bigtiff:
                # ScanImage <= 2015
                try:
                    self.pages._load_virtual_frames()
                except Exception as exc:
                    logger().error(
                        f'{self!r} <TiffPages._load_virtual_frames> '
                        f'raised {exc!r:.128}'
                    )

            elif self.is_ndpi:
                try:
                    self._ndpi_load_pages()
                except Exception as exc:
                    logger().error(
                        f'{self!r} <_ndpi_load_pages> raised {exc!r:.128}'
                    )

            elif _useframes:
                self.pages.useframes = True

        except Exception:
            fh.close()
            raise

    @property
    def byteorder(self) -> Literal['>', '<']:
        """Byteorder of TIFF file."""
        return self.tiff.byteorder

    @property
    def filehandle(self) -> FileHandle:
        """File handle."""
        return self._fh

    @property
    def filename(self) -> str:
        """Name of file handle."""
        return self._fh.name

    @cached_property
    def fstat(self) -> Any:
        """Status of file handle's descriptor, if any."""
        try:
            return os.fstat(self._fh.fileno())
        except Exception:  # io.UnsupportedOperation
            return None

    def close(self) -> None:
        """Close open file handle(s)."""
        for tif in self._files.values():
            tif.filehandle.close()

    def asarray(
        self,
        key: int | slice | Iterable[int] | None = None,
        *,
        series: int | TiffPageSeries | None = None,
        level: int | None = None,
        squeeze: bool | None = None,
        selection: (
            dict[str, int | slice] | tuple[int | slice, ...] | None
        ) = None,
        out: OutputType = None,
        device: str | None = None,
        maxworkers: int | None = None,
        buffersize: int | None = None,
    ) -> NDArray[Any]:
        """Return images from select pages as NumPy array.

        By default, the image array from the first level of the first series
        is returned.

        Parameters:
            key:
                Specifies which pages to return as array.
                By default, the image of the specified `series` and `level`
                is returned.
                If not *None*, the images from the specified pages in the
                whole file (if `series` is *None*) or a specified series are
                returned as a stacked array.
                Requesting an array from multiple pages that are not
                compatible with respect to shape, dtype, compression etc.
                is undefined, that is, it may crash or return incorrect values.
            series:
                Specifies which series of pages to return as array.
                The default is 0.
            level:
                Specifies which level of multi-resolution series to return
                as array. The default is 0.
            squeeze:
                If *True*, remove all length-1 dimensions (except X and Y)
                from array.
                If *False*, single pages are returned as 5D array of shape
                :py:attr:`TiffPage.shaped`.
                For series, the shape of the returned array also includes
                singlet dimensions specified in some file formats.
                For example, ImageJ series and most commonly also OME series,
                are returned in TZCYXS order.
                By default, all but `"shaped"` series are squeezed.
            selection:
                Subset of multi-dimensional image to return.
                Cannot be used together with `key`.
                If a dict, maps axis codes (e.g. ``'T'``, ``'Z'``) or
                dimension names (e.g. ``'time'``, ``'depth'``) to integer
                indices or slices.
                If a tuple, positional selections matching the series axes
                order.
                Only pages matching the selection are read from disk.
                Spatial dimensions (within a page) are sliced in memory.
            out:
                Specifies how image array is returned.
                By default, a new NumPy array is created.
                If a *numpy.ndarray*, a writable array to which the image
                is copied.
                If *'memmap'*, directly memory-map the image data in the
                file if possible; else create a memory-mapped array in a
                temporary file.
                If a *string* or *open file*, the file used to create a
                memory-mapped array.
            device:
                If not *None*, return a ``torch.Tensor`` on the specified
                device instead of a NumPy array.
                For example, ``'cuda'`` or ``'cuda:0'``.
                Requires PyTorch. For uncompressed contiguous series, data
                may be transferred directly to GPU via pinned memory DMA
                or GPUDirect Storage (Linux with kvikio).
            maxworkers:
                Maximum number of threads to concurrently decode data from
                multiple pages or compressed segments.
                If *None* or *0*, use up to :py:attr:`_TIFF.MAXWORKERS`
                threads. Reading data from file is limited to the main thread.
                Using multiple threads can significantly speed up this
                function if the bottleneck is decoding compressed data,
                for example, in case of large LZW compressed LSM files or
                JPEG compressed tiled slides.
                If the bottleneck is I/O or pure Python code, using multiple
                threads might be detrimental.
            buffersize:
                Approximate number of bytes to read from file in one pass.
                The default is :py:attr:`_TIFF.BUFFERSIZE`.

        Returns:
            Images from specified pages. See `TiffPage.asarray`
            for operations that are applied (or not) to the image data
            stored in the file.

        """
        if not self.pages:
            return numpy.array([])

        if selection is not None:
            if key is not None:
                msg = 'cannot use both key and selection'
                raise ValueError(msg)
            if series is None:
                series = 0
            if not isinstance(series, TiffPageSeries):
                series = self.series[series]
            return series.asarray(
                level=level,
                selection=selection,
                squeeze=squeeze,
                out=out,
                device=device,
                maxworkers=maxworkers,
                buffersize=buffersize,
            )

        if key is None and series is None:
            series = 0

        pages: Any  # TiffPages | TiffPageSeries | list[TiffPage | TiffFrame]
        page0: TiffPage | TiffFrame | None

        if series is None:
            pages = self.pages
        else:
            if not isinstance(series, TiffPageSeries):
                series = self.series[series]
            if level is not None:
                series = series.levels[level]
            pages = series

        if key is None:
            pass
        elif series is None:
            pages = pages._getlist(key)
        elif isinstance(key, (int, numpy.integer)):
            pages = [pages[int(key)]]
        elif isinstance(key, slice):
            pages = pages[key]
        elif isinstance(key, Iterable) and not isinstance(key, str):
            pages = [pages[k] for k in key]
        else:
            msg = (
                f'key must be an integer, slice, or sequence, not {type(key)}'
            )
            raise TypeError(msg)

        if pages is None or len(pages) == 0:
            msg = 'no pages selected'
            raise ValueError(msg)

        if (
            key is None
            and series is not None
            and series.dataoffset is not None
        ):
            typecode = self.byteorder + series.dtype.char
            if device is not None:
                # GPU direct transfer for contiguous uncompressed data
                from .gpu import parse_device, read_to_gpu

                dev = parse_device(device)
                if dev is not None and dev.type == 'cuda':
                    shape = series.get_shape(squeeze=squeeze)
                    return read_to_gpu(
                        self.filehandle,
                        dev,
                        typecode,
                        series.size,
                        series.dataoffset,
                        shape,
                    )
            if (
                series.keyframe.is_memmappable
                and isinstance(out, str)
                and out == 'memmap'
            ):
                # direct mapping
                shape = series.get_shape(squeeze=squeeze)
                result = self.filehandle.memmap_array(
                    typecode, shape, series.dataoffset
                )
            else:
                # read into output
                shape = series.get_shape(squeeze=squeeze)
                if out is not None:
                    out = create_output(out, shape, series.dtype)
                result = self.filehandle.read_array(
                    typecode,
                    series.size,
                    series.dataoffset,
                    out=out,
                )
        elif (
            isinstance(key, (slice, range))
            and series is not None
            and series.dataoffset is not None
            and out is None
        ):
            # strided contiguous fast path: mmap view + stride
            # OS only faults in the pages actually accessed
            result = self._asarray_contiguous_strided(
                series, key, squeeze
            )
            if result is not None:
                return result
            result = stack_pages(
                pages,
                out=out,
                maxworkers=maxworkers,
                buffersize=buffersize,
            )
        elif len(pages) == 1:
            page0 = pages[0]
            if page0 is None:
                msg = 'page is None'
                raise ValueError(msg)
            result = page0.asarray(
                out=out, maxworkers=maxworkers, buffersize=buffersize
            )
        else:
            result = stack_pages(
                pages, out=out, maxworkers=maxworkers, buffersize=buffersize
            )

        assert result is not None

        if key is None:
            assert series is not None  # TODO: ?
            shape = series.get_shape(squeeze=squeeze)
            try:
                result.shape = shape
            except ValueError as exc:
                try:
                    logger().warning(
                        f'{self!r} <asarray> failed to reshape '
                        f'{result.shape} to {shape}, raised {exc!r:.128}'
                    )
                    # try series of expected shapes
                    result.shape = (-1, *shape)
                except ValueError:
                    # revert to generic shape
                    result.shape = (-1, *series.keyframe.shape)
        elif len(pages) == 1:
            if squeeze is None:
                squeeze = True
            page0 = pages[0]
            if page0 is None:
                msg = 'page is None'
                raise ValueError(msg)
            result = result.reshape(page0.shape if squeeze else page0.shaped)
        else:
            if squeeze is None:
                squeeze = True
            try:
                page0 = next(p for p in pages if p is not None)
            except StopIteration as exc:
                msg = 'pages are all None'
                raise ValueError(msg) from exc
            assert page0 is not None
            result = result.reshape(
                (-1, *page0.shape) if squeeze else (-1, *page0.shaped)
            )

        if device is not None:
            from .gpu import numpy_to_tensor, parse_device

            dev = parse_device(device)
            if dev is not None:
                return numpy_to_tensor(result, dev)
        return result

    def _asarray_contiguous_strided(
        self,
        series: TiffPageSeries,
        key: slice | range,
        squeeze: bool | None,
    ) -> NDArray[Any] | None:
        """Read strided pages from contiguous series via mmap.

        Returns None if mmap is not available or layout is incompatible.
        """
        from .sequences import _mmap_read_contiguous

        mv = self.filehandle._get_mmap_view()
        if mv is None:
            return None

        offset = series.dataoffset
        assert offset is not None
        page_elems = series.keyframe.size
        npages = series.size // page_elems
        if npages * page_elems != series.size:
            return None

        result = _mmap_read_contiguous(
            mv,
            offset,
            series.dtype.char,
            self.byteorder,
            npages,
            page_elems,
            page_selection=key,
        )
        if result is None:
            return None

        # reshape to (n_selected, *page_shape)
        page_shape = (
            series.keyframe.shape
            if squeeze is None or squeeze
            else series.keyframe.shaped
        )
        try:
            result = result.reshape(-1, *page_shape)
            if squeeze is None or squeeze:
                result = result.squeeze()
        except Exception:
            return None
        return result

    def aszarr(
        self,
        key: int | None = None,
        *,
        series: int | TiffPageSeries | None = None,
        level: int | None = None,
        **kwargs: Any,
    ) -> ZarrTiffStore:
        """Return images from select pages as Zarr store.

        By default, the images from the first series, including all levels,
        are wrapped as a Zarr store.

        Parameters:
            key:
                Index of page in file (if `series` is None) or series to wrap
                as Zarr store.
                By default, a series is wrapped.
            series:
                Index of series to wrap as Zarr store.
                The default is 0 (if `key` is None).
            level:
                Index of pyramid level in series to wrap as Zarr store.
                By default, all levels are included as a multi-scale group.
            **kwargs:
                Additional arguments passed to :py:meth:`TiffPage.aszarr`
                or :py:meth:`TiffPageSeries.aszarr`.

        """
        if not self.pages:
            msg = 'empty Zarr arrays not supported'
            raise NotImplementedError(msg)
        if key is None and series is None:
            return self.series[0].aszarr(level=level, **kwargs)

        pages: Any
        if series is None:
            pages = self.pages
        else:
            if not isinstance(series, TiffPageSeries):
                series = self.series[series]
            if key is None:
                return series.aszarr(level=level, **kwargs)
            if level is not None:
                series = series.levels[level]
            pages = series

        if isinstance(key, (int, numpy.integer)):
            page: TiffPage | TiffFrame = pages[key]
            return page.aszarr(**kwargs)
        msg = 'key must be an integer index'
        raise TypeError(msg)

    @cached_property
    def series(self) -> list[TiffPageSeries]:
        """Series of pages with compatible shape and data type.

        Side effect: after accessing this property, `TiffFile.pages` might
        contain `TiffPage` and `TiffFrame` instead of only `TiffPage`
        instances.

        """
        if not self.pages:
            return []
        assert self.pages.keyframe is not None
        useframes = self.pages.useframes
        keyframe = self.pages.keyframe.index

        from .series import get_default_registry

        series = get_default_registry().parse(self)

        self.pages.useframes = useframes
        self.pages.set_keyframe(keyframe)

        # remove empty series, for example, in MD Gel files
        # series = [s for s in series if product(s.shape) > 0]
        assert series is not None
        for i, s in enumerate(series):
            s._index = i
        return series

    def _lsm_load_pages(self) -> None:
        """Read and fix all pages from LSM file."""
        # cache all pages to preserve corrected values
        pages = self.pages
        pages.cache = True
        pages.useframes = True
        # use first and second page as keyframes
        pages.set_keyframe(1)
        pages.set_keyframe(0)
        # load remaining pages as frames
        pages._load(None)
        # fix offsets and bytecounts first
        # TODO: fix multiple conversions between lists and tuples
        self._lsm_fix_strip_offsets()
        self._lsm_fix_strip_bytecounts()
        # assign keyframes for data and thumbnail series
        keyframe = self.pages.first
        for page in pages._pages[::2]:
            page.keyframe = keyframe  # type: ignore[union-attr]
        keyframe = cast(TiffPage, pages[1])
        for page in pages._pages[1::2]:
            page.keyframe = keyframe  # type: ignore[union-attr]

    def _lsm_fix_strip_offsets(self) -> None:
        """Unwrap strip offsets for LSM files greater than 4 GB.

        Each series and position require separate unwrapping (undocumented).

        """
        if self.filehandle.size < 2**32:
            return

        indices: NDArray[Any]
        pages = self.pages
        npages = len(pages)
        series = self.series[0]
        axes = series.axes

        # find positions
        positions = 1
        for i in 0, 1:
            if series.axes[i] in 'PM':
                positions *= series.shape[i]

        # make time axis first
        if positions > 1:
            ntimes = 0
            for i in 1, 2:
                if axes[i] == 'T':
                    ntimes = series.shape[i]
                    break
            if ntimes:
                div, mod = divmod(npages, 2 * positions * ntimes)
                if mod != 0:
                    msg = 'mod != 0'
                    raise RuntimeError(msg)
                shape = (positions, ntimes, div, 2)
                indices = numpy.arange(product(shape)).reshape(shape)
                indices = numpy.moveaxis(indices, 1, 0)
            else:
                indices = numpy.arange(npages).reshape((-1, 2))
        else:
            indices = numpy.arange(npages).reshape((-1, 2))

        # images of reduced page might be stored first
        if pages[0].dataoffsets[0] > pages[1].dataoffsets[0]:
            indices = indices[..., ::-1]

        # unwrap offsets
        wrap = 0
        previousoffset = 0
        for npi in indices.flat:
            page = pages[int(npi)]
            dataoffsets = []
            if all(i <= 0 for i in page.dataoffsets):
                logger().warning(
                    f'{self!r} LSM file incompletely written at {page}'
                )
                break
            for currentoffset in page.dataoffsets:
                if currentoffset < previousoffset:
                    wrap += 2**32
                dataoffsets.append(currentoffset + wrap)
                previousoffset = currentoffset
            page.dataoffsets = tuple(dataoffsets)

    def _lsm_fix_strip_bytecounts(self) -> None:
        """Set databytecounts to size of compressed data.

        The StripByteCounts tag in LSM files contains the number of bytes
        for the uncompressed data.

        """
        if self.pages.first.compression == 1:
            return
        # sort pages by first strip offset
        pages = sorted(self.pages, key=lambda p: p.dataoffsets[0])
        npages = len(pages) - 1
        for i, page in enumerate(pages):
            if page.index % 2:
                continue
            offsets = page.dataoffsets
            bytecounts = page.databytecounts
            if i < npages:
                lastoffset = pages[i + 1].dataoffsets[0]
            else:
                # LZW compressed strips might be longer than uncompressed
                lastoffset = min(
                    offsets[-1] + 2 * bytecounts[-1], self._fh.size
                )
            bytecount_list = list(bytecounts)
            for j in range(len(bytecounts) - 1):
                bytecount_list[j] = offsets[j + 1] - offsets[j]
            bytecount_list[-1] = lastoffset - offsets[-1]
            page.databytecounts = tuple(bytecount_list)

    def _ndpi_load_pages(self) -> None:
        """Read and fix pages from NDPI slide file if CaptureMode > 6.

        If the value of the CaptureMode tag is greater than 6, change the
        attributes of TiffPage instances that are part of the pyramid to
        match 16-bit grayscale data. TiffTag values are not corrected.

        """
        pages = self.pages
        capturemode = self.pages.first.tags.valueof(65441)
        if capturemode is None or capturemode < 6:
            return

        pages.cache = True
        pages.useframes = False
        pages._load()

        for page in pages:
            assert isinstance(page, TiffPage)
            mag = page.tags.valueof(65421)
            if mag is None or mag > 0:
                page.photometric = PHOTOMETRIC.MINISBLACK
                page.sampleformat = SAMPLEFORMAT.UINT
                page.samplesperpixel = 1
                page.bitspersample = 16
                page.dtype = page._dtype = numpy.dtype(numpy.uint16)
                if page.shaped[-1] > 1:
                    page.axes = page.axes[:-1]
                    page.shape = page.shape[:-1]
                    page.shaped = (*page.shaped[:-1], 1)

    def __getattr__(self, name: str, /) -> bool:
        """Return `is_flag` attributes from first page."""
        if name[3:] in TIFF.PAGE_FLAGS:
            if not self.pages:
                return False
            value = bool(getattr(self.pages.first, name))
            setattr(self, name, value)
            return value
        msg = f'{self.__class__.__name__!r} object has no attribute {name!r}'
        raise AttributeError(msg)

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()

    def __repr__(self) -> str:
        return f'<tifffile.TiffFile {snipstr(self._fh.name, 32)!r}>'

    def __str__(self) -> str:
        return self._str()

    def _str(self, detail: int = 0, width: int = 79) -> str:
        """Return string containing information about TiffFile.

        The `detail` parameter specifies the level of detail returned:

        0: file only.
        1: all series, first page of series and its tags.
        2: large tag values and file metadata.
        3: all pages.

        """
        info_list = [
            "TiffFile '{}'",
            format_size(self._fh.size),
            (
                ''
                if byteorder_isnative(self.byteorder)
                else {'<': 'little-endian', '>': 'big-endian'}[self.byteorder]
            ),
        ]
        if self.is_bigtiff:
            info_list.append('BigTiff')
        if len(self.pages) > 1:
            info_list.append(f'{len(self.pages)} Pages')
        if len(self.series) > 1:
            info_list.append(f'{len(self.series)} Series')
        if len(self._files) > 1:
            info_list.append(f'{len(self._files)} Files')
        flags = self.flags
        if 'uniform' in flags and len(self.pages) == 1:
            flags.discard('uniform')
        info_list.append('|'.join(f.lower() for f in sorted(flags)))
        info = '  '.join(info_list)
        info = info.replace('    ', '  ').replace('   ', '  ')
        info = info.format(
            snipstr(self._fh.name, max(12, width + 2 - len(info)))
        )
        if detail <= 0:
            return info
        info_list = [info]
        info_list.append('\n'.join(str(s) for s in self.series))
        if detail >= 3:
            for page in self.pages:
                info_list.append(page._str(detail=detail, width=width))
                if page.pages is not None:
                    for subifd in page.pages:
                        info_list.append(
                            subifd._str(detail=detail, width=width)
                        )
        elif self.series:
            info_list.extend(
                s.keyframe._str(detail=detail, width=width)
                for s in self.series
                if not s.keyframe.parent.filehandle.closed  # avoid warning
            )
        elif self.pages:  # and self.pages.first:
            info_list.append(self.pages.first._str(detail=detail, width=width))
        if detail >= 2:
            for name in sorted(self.flags):
                if hasattr(self, name + '_metadata'):
                    m = getattr(self, name + '_metadata')
                    if m:
                        info_list.append(
                            f'{name.upper()}_METADATA\n'
                            f'{pformat(m, width=width, height=detail * 24)}'
                        )
        return '\n\n'.join(info_list).replace('\n\n\n', '\n\n')

    @cached_property
    def flags(self) -> set[str]:
        """Set of file flags (a potentially expensive operation)."""
        return {
            name.lower()
            for name in TIFF.FILE_FLAGS
            if getattr(self, 'is_' + name)
        }

    @cached_property
    def is_uniform(self) -> bool:
        """File contains uniform series of pages."""
        # the hashes of IFDs 0, 7, and -1 are the same
        pages = self.pages
        try:
            page = self.pages.first
        except IndexError:
            return False
        if page.subifds:
            return False
        if page.is_scanimage or page.is_nih:
            return True
        i = 0
        useframes = pages.useframes
        try:
            pages.useframes = False
            h = page.hash
            for i in (1, 7, -1):
                if pages[i].aspage().hash != h:
                    return False
        except IndexError:
            return i == 1  # single page TIFF is uniform
        finally:
            pages.useframes = useframes
        return True

    @property
    def is_appendable(self) -> bool:
        """Pages can be appended to file without corrupting."""
        # TODO: check other formats
        return not (
            self.is_ome
            or self.is_lsm
            or self.is_stk
            or self.is_imagej
            or self.is_fluoview
            or self.is_micromanager
        )

    @property
    def is_bigtiff(self) -> bool:
        """File has BigTIFF format."""
        return self.tiff.is_bigtiff

    @cached_property
    def is_ndtiff(self) -> bool:
        """File has NDTiff format."""
        # file should be accompanied by NDTiff.index
        meta = self.micromanager_metadata
        if meta is not None and meta.get('MajorVersion', 0) >= 2:
            self.is_uniform = True
            return True
        return False

    @cached_property
    def is_mmstack(self) -> bool:
        """File has Micro-Manager stack format."""
        meta = self.micromanager_metadata
        if (
            meta is not None
            and 'Summary' in meta
            and 'IndexMap' in meta
            and meta.get('MajorVersion', 1) == 0
            # and 'MagellanStack' not in self.filename:
        ):
            self.is_uniform = True
            return True
        return False

    @cached_property
    def is_mdgel(self) -> bool:
        """File has MD Gel format."""
        # side effect: add second page, if exists, to cache
        try:
            ismdgel = (
                self.pages.first.is_mdgel
                or self.pages.get(1, cache=True).is_mdgel
            )
        except IndexError:
            return False
        if ismdgel:
            self.is_uniform = False
        return ismdgel

    @property
    def is_sis(self) -> bool:
        """File is Olympus SIS format."""
        try:
            return (
                self.pages.first.is_sis
                and not self.filename.lower().endswith('.vsi')
            )
        except IndexError:
            return False

    @cached_property
    def shaped_metadata(self) -> tuple[dict[str, Any], ...] | None:
        """Tifffile metadata from JSON formatted ImageDescription tags."""
        if not self.is_shaped:
            return None
        result = []
        for s in self.series:
            if s.kind.lower() != 'shaped':
                continue
            page = s.pages[0]
            if (
                not isinstance(page, TiffPage)
                or page.shaped_description is None
            ):
                continue
            result.append(shaped_description_metadata(page.shaped_description))
        return tuple(result)

    @property
    def ome_metadata(self) -> str | None:
        """OME XML metadata from ImageDescription tag."""
        if not self.is_ome:
            return None
        # return xml2dict(self.pages.first.description)['OME']
        if self._omexml:
            return self._omexml
        return self.pages.first.description

    @property
    def scn_metadata(self) -> str | None:
        """Leica SCN XML metadata from ImageDescription tag."""
        if not self.is_scn:
            return None
        return self.pages.first.description

    @property
    def philips_metadata(self) -> str | None:
        """Philips DP XML metadata from ImageDescription tag."""
        if not self.is_philips:
            return None
        return self.pages.first.description

    @property
    def indica_metadata(self) -> str | None:
        """IndicaLabs XML metadata from ImageDescription tag."""
        if not self.is_indica:
            return None
        return self.pages.first.description

    @property
    def avs_metadata(self) -> str | None:
        """Argos AVS XML metadata from tag 65000."""
        if not self.is_avs:
            return None
        return self.pages.first.tags.valueof(65000)

    @property
    def lsm_metadata(self) -> dict[str, Any] | None:
        """LSM metadata from CZ_LSMINFO tag."""
        if not self.is_lsm:
            return None
        return self.pages.first.tags.valueof(34412)  # CZ_LSMINFO

    @cached_property
    def stk_metadata(self) -> dict[str, Any] | None:
        """STK metadata from UIC tags."""
        if not self.is_stk:
            return None
        page = self.pages.first
        tags = page.tags
        result: dict[str, Any] = {}
        if page.description:
            result['PlaneDescriptions'] = page.description.split('\x00')
        tag = tags.get(33629)  # UIC2tag
        result['NumberPlanes'] = 1 if tag is None else tag.count
        value = tags.valueof(33628)  # UIC1tag
        if value is not None:
            result.update(value)
        value = tags.valueof(33630)  # UIC3tag
        if value is not None:
            result.update(value)  # wavelengths
        value = tags.valueof(33631)  # UIC4tag
        if value is not None:
            result.update(value)  # override UIC1 tags
        uic2tag = tags.valueof(33629)
        if uic2tag is not None:
            result['ZDistance'] = uic2tag['ZDistance']
            result['TimeCreated'] = uic2tag['TimeCreated']
            result['TimeModified'] = uic2tag['TimeModified']
            for key in ('Created', 'Modified'):
                try:
                    result['Datetime' + key] = numpy.array(
                        [
                            julian_datetime(*dt)
                            for dt in zip(
                                uic2tag['Date' + key],
                                uic2tag['Time' + key],
                                strict=True,
                            )
                        ],
                        dtype='datetime64[ns]',
                    )
                except Exception as exc:
                    result['Datetime' + key] = None
                    logger().warning(
                        f'{self!r} STK Datetime{key} raised {exc!r:.128}'
                    )
        return result

    @cached_property
    def imagej_metadata(self) -> dict[str, Any] | None:
        """ImageJ metadata from ImageDescription and IJMetadata tags."""
        if not self.is_imagej:
            return None
        page = self.pages.first
        if page.imagej_description is None:
            return None
        result = imagej_description_metadata(page.imagej_description)
        value = page.tags.valueof(50839)  # IJMetadata
        if value is not None:
            with contextlib.suppress(TypeError):
                result.update(value)
        return result

    @cached_property
    def fluoview_metadata(self) -> dict[str, Any] | None:
        """FluoView metadata from MM_Header and MM_Stamp tags."""
        if not self.is_fluoview:
            return None
        result = {}
        page = self.pages.first
        value = page.tags.valueof(34361)  # MM_Header
        if value is not None:
            result.update(value)
        # TODO: read stamps from all pages
        value = page.tags.valueof(34362)  # MM_Stamp
        if value is not None:
            result['Stamp'] = value
        # skip parsing image description; not reliable
        # try:
        #     t = fluoview_description_metadata(page.image_description)
        #     if t is not None:
        #         result['ImageDescription'] = t
        # except Exception as exc:
        #     logger().warning(
        #         f'{self!r} <fluoview_description_metadata> '
        #         f'raised {exc!r:.128}'
        #     )
        return result

    @property
    def nih_metadata(self) -> dict[str, Any] | None:
        """NIHImage metadata from NIHImageHeader tag."""
        if not self.is_nih:
            return None
        return self.pages.first.tags.valueof(43314)  # NIHImageHeader

    @property
    def fei_metadata(self) -> dict[str, Any] | None:
        """FEI metadata from SFEG or HELIOS tags."""
        if not self.is_fei:
            return None
        tags = self.pages.first.tags
        result = {}
        with contextlib.suppress(TypeError):
            result.update(tags.valueof(34680))  # FEI_SFEG
        with contextlib.suppress(TypeError):
            result.update(tags.valueof(34682))  # FEI_HELIOS
        return result

    @property
    def sem_metadata(self) -> dict[str, Any] | None:
        """SEM metadata from CZ_SEM tag."""
        if not self.is_sem:
            return None
        return self.pages.first.tags.valueof(34118)

    @property
    def sis_metadata(self) -> dict[str, Any] | None:
        """Olympus SIS metadata from OlympusSIS and OlympusINI tags."""
        if not self.pages.first.is_sis:
            return None
        tags = self.pages.first.tags
        result = {}
        with contextlib.suppress(TypeError):
            result.update(tags.valueof(33471))  # OlympusINI
        with contextlib.suppress(TypeError):
            result.update(tags.valueof(33560))  # OlympusSIS
        return result if result else None

    @cached_property
    def mdgel_metadata(self) -> dict[str, Any] | None:
        """MD-GEL metadata from MDFileTag tags."""
        if not self.is_mdgel:
            return None
        if 33445 in self.pages.first.tags:
            tags = self.pages.first.tags
        else:
            page = cast(TiffPage, self.pages[1])
            if 33445 in page.tags:
                tags = page.tags
            else:
                return None
        result = {}
        for code in range(33445, 33453):
            if code not in tags:
                continue
            name = TIFF.TAGS[code]
            result[name[2:]] = tags.valueof(code)
        return result

    @property
    def eer_metadata(self) -> dict[str, Any] | None:
        """EER metadata from tags 65001-65009."""
        if not self.is_eer:
            return None
        return self.pages.first.eer_tags

    @property
    def nuvu_metadata(self) -> dict[str, Any] | None:
        """Nuvu metadata from tags >= 65000."""
        if not self.is_nuvu:
            return None
        return self.pages.first.nuvu_tags

    @property
    def andor_metadata(self) -> dict[str, Any] | None:
        """Andor metadata from Andor tags."""
        return self.pages.first.andor_tags

    @property
    def epics_metadata(self) -> dict[str, Any] | None:
        """EPICS metadata from areaDetector tags."""
        return self.pages.first.epics_tags

    @property
    def tvips_metadata(self) -> dict[str, Any] | None:
        """TVIPS metadata from tag."""
        if not self.is_tvips:
            return None
        return self.pages.first.tags.valueof(37706)

    @cached_property
    def metaseries_metadata(self) -> dict[str, Any] | None:
        """MetaSeries metadata from ImageDescription tag of first tag."""
        # TODO: remove this? It is a per page property
        if not self.is_metaseries:
            return None
        return metaseries_description_metadata(self.pages.first.description)

    @cached_property
    def pilatus_metadata(self) -> dict[str, Any] | None:
        """Pilatus metadata from ImageDescription tag."""
        if not self.is_pilatus:
            return None
        return pilatus_description_metadata(self.pages.first.description)

    @cached_property
    def micromanager_metadata(self) -> dict[str, Any] | None:
        """Non-TIFF Micro-Manager metadata."""
        if not self.is_micromanager:
            return None
        return read_micromanager_metadata(self._fh)

    @cached_property
    def gdal_structural_metadata(self) -> dict[str, Any] | None:
        """Non-TIFF GDAL structural metadata."""
        return read_gdal_structural_metadata(self._fh)

    @cached_property
    def scanimage_metadata(self) -> dict[str, Any] | None:
        """ScanImage non-varying frame and ROI metadata.

        The returned dict may contain 'FrameData', 'RoiGroups', and 'version'
        keys.

        Varying frame data can be found in the ImageDescription tags.

        """
        if not self.is_scanimage:
            return None
        result: dict[str, Any] = {}
        try:
            framedata, roidata, version = read_scanimage_metadata(self._fh)
            result['version'] = version
            result['FrameData'] = framedata
            result.update(roidata)
        except (TypeError, ValueError):
            pass
        return result

    @property
    def geotiff_metadata(self) -> dict[str, Any] | None:
        """GeoTIFF metadata from tags."""
        if not self.is_geotiff:
            return None
        return self.pages.first.geotiff_tags

    @property
    def gdal_metadata(self) -> dict[str, Any] | None:
        """GDAL XML metadata from GDAL_METADATA tag."""
        if not self.is_gdal:
            return None
        return self.pages.first.tags.valueof(42112)

    @cached_property
    def astrotiff_metadata(self) -> dict[str, Any] | None:
        """AstroTIFF metadata from ImageDescription tag."""
        if not self.is_astrotiff:
            return None
        return astrotiff_description_metadata(self.pages.first.description)

    @cached_property
    def streak_metadata(self) -> dict[str, Any] | None:
        """Hamamatsu streak metadata from ImageDescription tag."""
        if not self.is_streak:
            return None
        return streak_description_metadata(
            self.pages.first.description, self.filehandle
        )


from .page import TiffPage, TiffFrame, TiffPages  # noqa: E402


class _TIFF:
    """Delay-loaded constants, accessible via :py:attr:`TIFF` instance."""

    @cached_property
    def CLASSIC_LE(self) -> TiffFormat:
        """32-bit little-endian TIFF format."""
        return TiffFormat(
            version=42,
            byteorder='<',
            offsetsize=4,
            offsetformat='<I',
            tagnosize=2,
            tagnoformat='<H',
            tagsize=12,
            tagformat1='<HH',
            tagformat2='<I4s',
            tagoffsetthreshold=4,
        )

    @cached_property
    def CLASSIC_BE(self) -> TiffFormat:
        """32-bit big-endian TIFF format."""
        return TiffFormat(
            version=42,
            byteorder='>',
            offsetsize=4,
            offsetformat='>I',
            tagnosize=2,
            tagnoformat='>H',
            tagsize=12,
            tagformat1='>HH',
            tagformat2='>I4s',
            tagoffsetthreshold=4,
        )

    @cached_property
    def BIG_LE(self) -> TiffFormat:
        """64-bit little-endian TIFF format."""
        return TiffFormat(
            version=43,
            byteorder='<',
            offsetsize=8,
            offsetformat='<Q',
            tagnosize=8,
            tagnoformat='<Q',
            tagsize=20,
            tagformat1='<HH',
            tagformat2='<Q8s',
            tagoffsetthreshold=8,
        )

    @cached_property
    def BIG_BE(self) -> TiffFormat:
        """64-bit big-endian TIFF format."""
        return TiffFormat(
            version=43,
            byteorder='>',
            offsetsize=8,
            offsetformat='>Q',
            tagnosize=8,
            tagnoformat='>Q',
            tagsize=20,
            tagformat1='>HH',
            tagformat2='>Q8s',
            tagoffsetthreshold=8,
        )

    @cached_property
    def NDPI_LE(self) -> TiffFormat:
        """32-bit little-endian TIFF format with 64-bit offsets."""
        return TiffFormat(
            version=42,
            byteorder='<',
            offsetsize=8,  # NDPI uses 8 bytes IFD and tag offsets
            offsetformat='<Q',
            tagnosize=2,
            tagnoformat='<H',
            tagsize=12,  # 16 after patching
            tagformat1='<HH',
            tagformat2='<I8s',  # after patching
            tagoffsetthreshold=4,
        )

    @cached_property
    def TAGS(self) -> TiffTagRegistry:
        """Registry of TIFF tag codes and names from TIFF6, TIFF/EP, EXIF."""
        # TODO: divide into baseline, exif, private, ... tags
        return TiffTagRegistry(
            (
                (11, 'ProcessingSoftware'),
                (254, 'NewSubfileType'),
                (255, 'SubfileType'),
                (256, 'ImageWidth'),
                (257, 'ImageLength'),
                (258, 'BitsPerSample'),
                (259, 'Compression'),
                (262, 'PhotometricInterpretation'),
                (263, 'Thresholding'),
                (264, 'CellWidth'),
                (265, 'CellLength'),
                (266, 'FillOrder'),
                (269, 'DocumentName'),
                (270, 'ImageDescription'),
                (271, 'Make'),
                (272, 'Model'),
                (273, 'StripOffsets'),
                (274, 'Orientation'),
                (277, 'SamplesPerPixel'),
                (278, 'RowsPerStrip'),
                (279, 'StripByteCounts'),
                (280, 'MinSampleValue'),
                (281, 'MaxSampleValue'),
                (282, 'XResolution'),
                (283, 'YResolution'),
                (284, 'PlanarConfiguration'),
                (285, 'PageName'),
                (286, 'XPosition'),
                (287, 'YPosition'),
                (288, 'FreeOffsets'),
                (289, 'FreeByteCounts'),
                (290, 'GrayResponseUnit'),
                (291, 'GrayResponseCurve'),
                (292, 'T4Options'),
                (293, 'T6Options'),
                (296, 'ResolutionUnit'),
                (297, 'PageNumber'),
                (300, 'ColorResponseUnit'),
                (301, 'TransferFunction'),
                (305, 'Software'),
                (306, 'DateTime'),
                (315, 'Artist'),
                (316, 'HostComputer'),
                (317, 'Predictor'),
                (318, 'WhitePoint'),
                (319, 'PrimaryChromaticities'),
                (320, 'ColorMap'),
                (321, 'HalftoneHints'),
                (322, 'TileWidth'),
                (323, 'TileLength'),
                (324, 'TileOffsets'),
                (325, 'TileByteCounts'),
                (326, 'BadFaxLines'),
                (327, 'CleanFaxData'),
                (328, 'ConsecutiveBadFaxLines'),
                (330, 'SubIFDs'),
                (332, 'InkSet'),
                (333, 'InkNames'),
                (334, 'NumberOfInks'),
                (336, 'DotRange'),
                (337, 'TargetPrinter'),
                (338, 'ExtraSamples'),
                (339, 'SampleFormat'),
                (340, 'SMinSampleValue'),
                (341, 'SMaxSampleValue'),
                (342, 'TransferRange'),
                (343, 'ClipPath'),
                (344, 'XClipPathUnits'),
                (345, 'YClipPathUnits'),
                (346, 'Indexed'),
                (347, 'JPEGTables'),
                (351, 'OPIProxy'),
                (400, 'GlobalParametersIFD'),
                (401, 'ProfileType'),
                (402, 'FaxProfile'),
                (403, 'CodingMethods'),
                (404, 'VersionYear'),
                (405, 'ModeNumber'),
                (433, 'Decode'),
                (434, 'DefaultImageColor'),
                (435, 'T82Options'),
                (437, 'JPEGTables'),  # 347
                (512, 'JPEGProc'),
                (513, 'JPEGInterchangeFormat'),
                (514, 'JPEGInterchangeFormatLength'),
                (515, 'JPEGRestartInterval'),
                (517, 'JPEGLosslessPredictors'),
                (518, 'JPEGPointTransforms'),
                (519, 'JPEGQTables'),
                (520, 'JPEGDCTables'),
                (521, 'JPEGACTables'),
                (529, 'YCbCrCoefficients'),
                (530, 'YCbCrSubSampling'),
                (531, 'YCbCrPositioning'),
                (532, 'ReferenceBlackWhite'),
                (559, 'StripRowCounts'),
                (700, 'XMP'),  # XMLPacket
                (769, 'GDIGamma'),  # GDI+
                (770, 'ICCProfileDescriptor'),  # GDI+
                (771, 'SRGBRenderingIntent'),  # GDI+
                (800, 'ImageTitle'),  # GDI+
                (907, 'SiffCompress'),  # https://github.com/MaimonLab/SiffPy
                (999, 'USPTO_Miscellaneous'),
                (4864, 'AndorId'),  # TODO, Andor Technology 4864 - 5030
                (4869, 'AndorTemperature'),
                (4876, 'AndorExposureTime'),
                (4878, 'AndorKineticCycleTime'),
                (4879, 'AndorAccumulations'),
                (4881, 'AndorAcquisitionCycleTime'),
                (4882, 'AndorReadoutTime'),
                (4884, 'AndorPhotonCounting'),
                (4885, 'AndorEmDacLevel'),
                (4890, 'AndorFrames'),
                (4896, 'AndorHorizontalFlip'),
                (4897, 'AndorVerticalFlip'),
                (4898, 'AndorClockwise'),
                (4899, 'AndorCounterClockwise'),
                (4904, 'AndorVerticalClockVoltage'),
                (4905, 'AndorVerticalShiftSpeed'),
                (4907, 'AndorPreAmpSetting'),
                (4908, 'AndorCameraSerial'),
                (4911, 'AndorActualTemperature'),
                (4912, 'AndorBaselineClamp'),
                (4913, 'AndorPrescans'),
                (4914, 'AndorModel'),
                (4915, 'AndorChipSizeX'),
                (4916, 'AndorChipSizeY'),
                (4944, 'AndorBaselineOffset'),
                (4966, 'AndorSoftwareVersion'),
                (18246, 'Rating'),
                (18247, 'XP_DIP_XML'),
                (18248, 'StitchInfo'),
                (18249, 'RatingPercent'),
                (20481, 'ResolutionXUnit'),  # GDI+
                (20482, 'ResolutionYUnit'),  # GDI+
                (20483, 'ResolutionXLengthUnit'),  # GDI+
                (20484, 'ResolutionYLengthUnit'),  # GDI+
                (20485, 'PrintFlags'),  # GDI+
                (20486, 'PrintFlagsVersion'),  # GDI+
                (20487, 'PrintFlagsCrop'),  # GDI+
                (20488, 'PrintFlagsBleedWidth'),  # GDI+
                (20489, 'PrintFlagsBleedWidthScale'),  # GDI+
                (20490, 'HalftoneLPI'),  # GDI+
                (20491, 'HalftoneLPIUnit'),  # GDI+
                (20492, 'HalftoneDegree'),  # GDI+
                (20493, 'HalftoneShape'),  # GDI+
                (20494, 'HalftoneMisc'),  # GDI+
                (20495, 'HalftoneScreen'),  # GDI+
                (20496, 'JPEGQuality'),  # GDI+
                (20497, 'GridSize'),  # GDI+
                (20498, 'ThumbnailFormat'),  # GDI+
                (20499, 'ThumbnailWidth'),  # GDI+
                (20500, 'ThumbnailHeight'),  # GDI+
                (20501, 'ThumbnailColorDepth'),  # GDI+
                (20502, 'ThumbnailPlanes'),  # GDI+
                (20503, 'ThumbnailRawBytes'),  # GDI+
                (20504, 'ThumbnailSize'),  # GDI+
                (20505, 'ThumbnailCompressedSize'),  # GDI+
                (20506, 'ColorTransferFunction'),  # GDI+
                (20507, 'ThumbnailData'),
                (20512, 'ThumbnailImageWidth'),  # GDI+
                (20513, 'ThumbnailImageHeight'),  # GDI+
                (20514, 'ThumbnailBitsPerSample'),  # GDI+
                (20515, 'ThumbnailCompression'),
                (20516, 'ThumbnailPhotometricInterp'),  # GDI+
                (20517, 'ThumbnailImageDescription'),  # GDI+
                (20518, 'ThumbnailEquipMake'),  # GDI+
                (20519, 'ThumbnailEquipModel'),  # GDI+
                (20520, 'ThumbnailStripOffsets'),  # GDI+
                (20521, 'ThumbnailOrientation'),  # GDI+
                (20522, 'ThumbnailSamplesPerPixel'),  # GDI+
                (20523, 'ThumbnailRowsPerStrip'),  # GDI+
                (20524, 'ThumbnailStripBytesCount'),  # GDI+
                (20525, 'ThumbnailResolutionX'),
                (20526, 'ThumbnailResolutionY'),
                (20527, 'ThumbnailPlanarConfig'),  # GDI+
                (20528, 'ThumbnailResolutionUnit'),
                (20529, 'ThumbnailTransferFunction'),
                (20530, 'ThumbnailSoftwareUsed'),  # GDI+
                (20531, 'ThumbnailDateTime'),  # GDI+
                (20532, 'ThumbnailArtist'),  # GDI+
                (20533, 'ThumbnailWhitePoint'),  # GDI+
                (20534, 'ThumbnailPrimaryChromaticities'),  # GDI+
                (20535, 'ThumbnailYCbCrCoefficients'),  # GDI+
                (20536, 'ThumbnailYCbCrSubsampling'),  # GDI+
                (20537, 'ThumbnailYCbCrPositioning'),
                (20538, 'ThumbnailRefBlackWhite'),  # GDI+
                (20539, 'ThumbnailCopyRight'),  # GDI+
                (20545, 'InteroperabilityIndex'),
                (20546, 'InteroperabilityVersion'),
                (20624, 'LuminanceTable'),
                (20625, 'ChrominanceTable'),
                (20736, 'FrameDelay'),  # GDI+
                (20737, 'LoopCount'),  # GDI+
                (20738, 'GlobalPalette'),  # GDI+
                (20739, 'IndexBackground'),  # GDI+
                (20740, 'IndexTransparent'),  # GDI+
                (20752, 'PixelUnit'),  # GDI+
                (20753, 'PixelPerUnitX'),  # GDI+
                (20754, 'PixelPerUnitY'),  # GDI+
                (20755, 'PaletteHistogram'),  # GDI+
                (28672, 'SonyRawFileType'),  # Sony ARW
                (28722, 'VignettingCorrParams'),  # Sony ARW
                (28725, 'ChromaticAberrationCorrParams'),  # Sony ARW
                (28727, 'DistortionCorrParams'),  # Sony ARW
                # Private tags >= 32768
                (32781, 'ImageID'),
                (32931, 'WangTag1'),
                (32932, 'WangAnnotation'),
                (32933, 'WangTag3'),
                (32934, 'WangTag4'),
                (32953, 'ImageReferencePoints'),
                (32954, 'RegionXformTackPoint'),
                (32955, 'WarpQuadrilateral'),
                (32956, 'AffineTransformMat'),
                (32995, 'Matteing'),
                (32996, 'DataType'),  # use SampleFormat
                (32997, 'ImageDepth'),
                (32998, 'TileDepth'),
                (33300, 'ImageFullWidth'),
                (33301, 'ImageFullLength'),
                (33302, 'TextureFormat'),
                (33303, 'TextureWrapModes'),
                (33304, 'FieldOfViewCotangent'),
                (33305, 'MatrixWorldToScreen'),
                (33306, 'MatrixWorldToCamera'),
                (33405, 'Model2'),
                (33421, 'CFARepeatPatternDim'),
                (33422, 'CFAPattern'),
                (33423, 'BatteryLevel'),
                (33424, 'KodakIFD'),
                (33434, 'ExposureTime'),
                (33437, 'FNumber'),
                (33432, 'Copyright'),
                (33445, 'MDFileTag'),
                (33446, 'MDScalePixel'),
                (33447, 'MDColorTable'),
                (33448, 'MDLabName'),
                (33449, 'MDSampleInfo'),
                (33450, 'MDPrepDate'),
                (33451, 'MDPrepTime'),
                (33452, 'MDFileUnits'),
                (33465, 'NiffRotation'),  # NIFF
                (33466, 'NiffNavyCompression'),  # NIFF
                (33467, 'NiffTileIndex'),  # NIFF
                (33471, 'OlympusINI'),
                (33550, 'ModelPixelScaleTag'),
                (33560, 'OlympusSIS'),  # see also 33471 and 34853
                (33589, 'AdventScale'),
                (33590, 'AdventRevision'),
                (33628, 'UIC1tag'),  # Metamorph  Universal Imaging Corp STK
                (33629, 'UIC2tag'),
                (33630, 'UIC3tag'),
                (33631, 'UIC4tag'),
                (33723, 'IPTCNAA'),
                (33858, 'ExtendedTagsOffset'),  # DEFF points IFD with tags
                (33918, 'IntergraphPacketData'),  # INGRPacketDataTag
                (33919, 'IntergraphFlagRegisters'),  # INGRFlagRegisters
                (33920, 'IntergraphMatrixTag'),  # IrasBTransformationMatrix
                (33921, 'INGRReserved'),
                (33922, 'ModelTiepointTag'),
                (33923, 'LeicaMagic'),
                (34016, 'Site'),  # 34016..34032 ANSI IT8 TIFF/IT
                (34017, 'ColorSequence'),
                (34018, 'IT8Header'),
                (34019, 'RasterPadding'),
                (34020, 'BitsPerRunLength'),
                (34021, 'BitsPerExtendedRunLength'),
                (34022, 'ColorTable'),
                (34023, 'ImageColorIndicator'),
                (34024, 'BackgroundColorIndicator'),
                (34025, 'ImageColorValue'),
                (34026, 'BackgroundColorValue'),
                (34027, 'PixelIntensityRange'),
                (34028, 'TransparencyIndicator'),
                (34029, 'ColorCharacterization'),
                (34030, 'HCUsage'),
                (34031, 'TrapIndicator'),
                (34032, 'CMYKEquivalent'),
                (34118, 'CZ_SEM'),  # Zeiss SEM
                (34152, 'AFCP_IPTC'),
                (34232, 'PixelMagicJBIGOptions'),  # EXIF, also TI FrameCount
                (34263, 'JPLCartoIFD'),
                (34122, 'IPLAB'),  # number of images
                (34264, 'ModelTransformationTag'),
                (34306, 'WB_GRGBLevels'),  # Leaf MOS
                (34310, 'LeafData'),
                (34361, 'MM_Header'),
                (34362, 'MM_Stamp'),
                (34363, 'MM_Unknown'),
                (34377, 'ImageResources'),  # Photoshop
                (34386, 'MM_UserBlock'),
                (34412, 'CZ_LSMINFO'),
                (34665, 'ExifTag'),
                (34675, 'InterColorProfile'),  # ICCProfile
                (34680, 'FEI_SFEG'),
                (34682, 'FEI_HELIOS'),
                (34683, 'FEI_TITAN'),
                (34687, 'FXExtensions'),
                (34688, 'MultiProfiles'),
                (34689, 'SharedData'),
                (34690, 'T88Options'),
                (34710, 'MarCCD'),  # offset to MarCCD header
                (34732, 'ImageLayer'),
                (34735, 'GeoKeyDirectoryTag'),
                (34736, 'GeoDoubleParamsTag'),
                (34737, 'GeoAsciiParamsTag'),
                (34750, 'JBIGOptions'),
                (34821, 'PIXTIFF'),  # ? Pixel Translations Inc
                (34850, 'ExposureProgram'),
                (34852, 'SpectralSensitivity'),
                (34853, 'GPSTag'),  # GPSIFD  also OlympusSIS2
                (34853, 'OlympusSIS2'),
                (34855, 'ISOSpeedRatings'),
                (34855, 'PhotographicSensitivity'),
                (34856, 'OECF'),  # optoelectric conversion factor
                (34857, 'Interlace'),  # TIFF/EP
                (34858, 'TimeZoneOffset'),  # TIFF/EP
                (34859, 'SelfTimerMode'),  # TIFF/EP
                (34864, 'SensitivityType'),
                (34865, 'StandardOutputSensitivity'),
                (34866, 'RecommendedExposureIndex'),
                (34867, 'ISOSpeed'),
                (34868, 'ISOSpeedLatitudeyyy'),
                (34869, 'ISOSpeedLatitudezzz'),
                (34908, 'HylaFAXFaxRecvParams'),
                (34909, 'HylaFAXFaxSubAddress'),
                (34910, 'HylaFAXFaxRecvTime'),
                (34911, 'FaxDcs'),
                (34929, 'FedexEDR'),
                (34954, 'LeafSubIFD'),
                (34959, 'Aphelion1'),
                (34960, 'Aphelion2'),
                (34961, 'AphelionInternal'),  # ADCIS
                (36864, 'ExifVersion'),
                (36867, 'DateTimeOriginal'),
                (36868, 'DateTimeDigitized'),
                (36873, 'GooglePlusUploadCode'),
                (36880, 'OffsetTime'),
                (36881, 'OffsetTimeOriginal'),
                (36882, 'OffsetTimeDigitized'),
                # TODO, Pilatus/CHESS/TV6 36864..37120 conflicting with Exif
                (36864, 'TVX_Unknown'),
                (36865, 'TVX_NumExposure'),
                (36866, 'TVX_NumBackground'),
                (36867, 'TVX_ExposureTime'),
                (36868, 'TVX_BackgroundTime'),
                (36870, 'TVX_Unknown'),
                (36873, 'TVX_SubBpp'),
                (36874, 'TVX_SubWide'),
                (36875, 'TVX_SubHigh'),
                (36876, 'TVX_BlackLevel'),
                (36877, 'TVX_DarkCurrent'),
                (36878, 'TVX_ReadNoise'),
                (36879, 'TVX_DarkCurrentNoise'),
                (36880, 'TVX_BeamMonitor'),
                (37120, 'TVX_UserVariables'),  # A/D values
                (37121, 'ComponentsConfiguration'),
                (37122, 'CompressedBitsPerPixel'),
                (37377, 'ShutterSpeedValue'),
                (37378, 'ApertureValue'),
                (37379, 'BrightnessValue'),
                (37380, 'ExposureBiasValue'),
                (37381, 'MaxApertureValue'),
                (37382, 'SubjectDistance'),
                (37383, 'MeteringMode'),
                (37384, 'LightSource'),
                (37385, 'Flash'),
                (37386, 'FocalLength'),
                (37387, 'FlashEnergy'),  # TIFF/EP
                (37388, 'SpatialFrequencyResponse'),  # TIFF/EP
                (37389, 'Noise'),  # TIFF/EP
                (37390, 'FocalPlaneXResolution'),  # TIFF/EP
                (37391, 'FocalPlaneYResolution'),  # TIFF/EP
                (37392, 'FocalPlaneResolutionUnit'),  # TIFF/EP
                (37393, 'ImageNumber'),  # TIFF/EP
                (37394, 'SecurityClassification'),  # TIFF/EP
                (37395, 'ImageHistory'),  # TIFF/EP
                (37396, 'SubjectLocation'),  # TIFF/EP
                (37397, 'ExposureIndex'),  # TIFF/EP
                (37398, 'TIFFEPStandardID'),  # TIFF/EP
                (37399, 'SensingMethod'),  # TIFF/EP
                (37434, 'CIP3DataFile'),
                (37435, 'CIP3Sheet'),
                (37436, 'CIP3Side'),
                (37439, 'StoNits'),
                (37500, 'MakerNote'),
                (37510, 'UserComment'),
                (37520, 'SubsecTime'),
                (37521, 'SubsecTimeOriginal'),
                (37522, 'SubsecTimeDigitized'),
                (37679, 'MODIText'),  # Microsoft Office Document Imaging
                (37680, 'MODIOLEPropertySetStorage'),
                (37681, 'MODIPositioning'),
                (37701, 'AgilentBinary'),  # private structure
                (37702, 'AgilentString'),  # file description
                (37706, 'TVIPS'),  # offset to TemData structure
                (37707, 'TVIPS1'),
                (37708, 'TVIPS2'),  # same TemData structure as undefined
                (37724, 'ImageSourceData'),  # Photoshop
                (37888, 'Temperature'),
                (37889, 'Humidity'),
                (37890, 'Pressure'),
                (37891, 'WaterDepth'),
                (37892, 'Acceleration'),
                (37893, 'CameraElevationAngle'),
                (40000, 'XPos'),  # Janelia
                (40001, 'YPos'),
                (40002, 'ZPos'),
                (40001, 'MC_IpWinScal'),  # Media Cybernetics
                (40001, 'RecipName'),  # MS FAX
                (40002, 'RecipNumber'),
                (40003, 'SenderName'),
                (40004, 'Routing'),
                (40005, 'CallerId'),
                (40006, 'TSID'),
                (40007, 'CSID'),
                (40008, 'FaxTime'),
                (40100, 'MC_IdOld'),
                (40106, 'MC_Unknown'),
                (40965, 'InteroperabilityTag'),  # InteropOffset
                (40091, 'XPTitle'),
                (40092, 'XPComment'),
                (40093, 'XPAuthor'),
                (40094, 'XPKeywords'),
                (40095, 'XPSubject'),
                (40960, 'FlashpixVersion'),
                (40961, 'ColorSpace'),
                (40962, 'PixelXDimension'),
                (40963, 'PixelYDimension'),
                (40964, 'RelatedSoundFile'),
                (40976, 'SamsungRawPointersOffset'),
                (40977, 'SamsungRawPointersLength'),
                (41217, 'SamsungRawByteOrder'),
                (41218, 'SamsungRawUnknown'),
                (41483, 'FlashEnergy'),
                (41484, 'SpatialFrequencyResponse'),
                (41485, 'Noise'),  # 37389
                (41486, 'FocalPlaneXResolution'),  # 37390
                (41487, 'FocalPlaneYResolution'),  # 37391
                (41488, 'FocalPlaneResolutionUnit'),  # 37392
                (41489, 'ImageNumber'),  # 37393
                (41490, 'SecurityClassification'),  # 37394
                (41491, 'ImageHistory'),  # 37395
                (41492, 'SubjectLocation'),  # 37395
                (41493, 'ExposureIndex '),  # 37397
                (41494, 'TIFF-EPStandardID'),
                (41495, 'SensingMethod'),  # 37399
                (41728, 'FileSource'),
                (41729, 'SceneType'),
                (41730, 'CFAPattern'),  # 33422
                (41985, 'CustomRendered'),
                (41986, 'ExposureMode'),
                (41987, 'WhiteBalance'),
                (41988, 'DigitalZoomRatio'),
                (41989, 'FocalLengthIn35mmFilm'),
                (41990, 'SceneCaptureType'),
                (41991, 'GainControl'),
                (41992, 'Contrast'),
                (41993, 'Saturation'),
                (41994, 'Sharpness'),
                (41995, 'DeviceSettingDescription'),
                (41996, 'SubjectDistanceRange'),
                (42016, 'ImageUniqueID'),
                (42032, 'CameraOwnerName'),
                (42033, 'BodySerialNumber'),
                (42034, 'LensSpecification'),
                (42035, 'LensMake'),
                (42036, 'LensModel'),
                (42037, 'LensSerialNumber'),
                (42080, 'CompositeImage'),
                (42081, 'SourceImageNumberCompositeImage'),
                (42082, 'SourceExposureTimesCompositeImage'),
                (42112, 'GDAL_METADATA'),
                (42113, 'GDAL_NODATA'),
                (42240, 'Gamma'),
                (43314, 'NIHImageHeader'),
                (44992, 'ExpandSoftware'),
                (44993, 'ExpandLens'),
                (44994, 'ExpandFilm'),
                (44995, 'ExpandFilterLens'),
                (44996, 'ExpandScanner'),
                (44997, 'ExpandFlashLamp'),
                (48129, 'PixelFormat'),  # HDP and WDP
                (48130, 'Transformation'),
                (48131, 'Uncompressed'),
                (48132, 'ImageType'),
                (48256, 'ImageWidth'),  # 256
                (48257, 'ImageHeight'),
                (48258, 'WidthResolution'),
                (48259, 'HeightResolution'),
                (48320, 'ImageOffset'),
                (48321, 'ImageByteCount'),
                (48322, 'AlphaOffset'),
                (48323, 'AlphaByteCount'),
                (48324, 'ImageDataDiscard'),
                (48325, 'AlphaDataDiscard'),
                (50003, 'KodakAPP3'),
                (50215, 'OceScanjobDescription'),
                (50216, 'OceApplicationSelector'),
                (50217, 'OceIdentificationNumber'),
                (50218, 'OceImageLogicCharacteristics'),
                (50255, 'Annotations'),
                (50288, 'MC_Id'),  # Media Cybernetics
                (50289, 'MC_XYPosition'),
                (50290, 'MC_ZPosition'),
                (50291, 'MC_XYCalibration'),
                (50292, 'MC_LensCharacteristics'),
                (50293, 'MC_ChannelName'),
                (50294, 'MC_ExcitationWavelength'),
                (50295, 'MC_TimeStamp'),
                (50296, 'MC_FrameProperties'),
                (50341, 'PrintImageMatching'),
                (50495, 'PCO_RAW'),  # TODO, PCO CamWare
                (50547, 'OriginalFileName'),
                (50560, 'USPTO_OriginalContentType'),  # US Patent Office
                (50561, 'USPTO_RotationCode'),
                (50648, 'CR2Unknown1'),
                (50649, 'CR2Unknown2'),
                (50656, 'CR2CFAPattern'),
                (50674, 'LercParameters'),  # ESGI 50674 .. 50677
                (50706, 'DNGVersion'),  # DNG 50706 .. 51114
                (50707, 'DNGBackwardVersion'),
                (50708, 'UniqueCameraModel'),
                (50709, 'LocalizedCameraModel'),
                (50710, 'CFAPlaneColor'),
                (50711, 'CFALayout'),
                (50712, 'LinearizationTable'),
                (50713, 'BlackLevelRepeatDim'),
                (50714, 'BlackLevel'),
                (50715, 'BlackLevelDeltaH'),
                (50716, 'BlackLevelDeltaV'),
                (50717, 'WhiteLevel'),
                (50718, 'DefaultScale'),
                (50719, 'DefaultCropOrigin'),
                (50720, 'DefaultCropSize'),
                (50721, 'ColorMatrix1'),
                (50722, 'ColorMatrix2'),
                (50723, 'CameraCalibration1'),
                (50724, 'CameraCalibration2'),
                (50725, 'ReductionMatrix1'),
                (50726, 'ReductionMatrix2'),
                (50727, 'AnalogBalance'),
                (50728, 'AsShotNeutral'),
                (50729, 'AsShotWhiteXY'),
                (50730, 'BaselineExposure'),
                (50731, 'BaselineNoise'),
                (50732, 'BaselineSharpness'),
                (50733, 'BayerGreenSplit'),
                (50734, 'LinearResponseLimit'),
                (50735, 'CameraSerialNumber'),
                (50736, 'LensInfo'),
                (50737, 'ChromaBlurRadius'),
                (50738, 'AntiAliasStrength'),
                (50739, 'ShadowScale'),
                (50740, 'DNGPrivateData'),
                (50741, 'MakerNoteSafety'),
                (50752, 'RawImageSegmentation'),
                (50778, 'CalibrationIlluminant1'),
                (50779, 'CalibrationIlluminant2'),
                (50780, 'BestQualityScale'),
                (50781, 'RawDataUniqueID'),
                (50784, 'AliasLayerMetadata'),
                (50827, 'OriginalRawFileName'),
                (50828, 'OriginalRawFileData'),
                (50829, 'ActiveArea'),
                (50830, 'MaskedAreas'),
                (50831, 'AsShotICCProfile'),
                (50832, 'AsShotPreProfileMatrix'),
                (50833, 'CurrentICCProfile'),
                (50834, 'CurrentPreProfileMatrix'),
                (50838, 'IJMetadataByteCounts'),
                (50839, 'IJMetadata'),
                (50844, 'RPCCoefficientTag'),
                (50879, 'ColorimetricReference'),
                (50885, 'SRawType'),
                (50898, 'PanasonicTitle'),
                (50899, 'PanasonicTitle2'),
                (50908, 'RSID'),  # DGIWG
                (50909, 'GEO_METADATA'),  # DGIWG XML
                (50931, 'CameraCalibrationSignature'),
                (50932, 'ProfileCalibrationSignature'),
                (50933, 'ProfileIFD'),  # EXTRACAMERAPROFILES
                (50934, 'AsShotProfileName'),
                (50935, 'NoiseReductionApplied'),
                (50936, 'ProfileName'),
                (50937, 'ProfileHueSatMapDims'),
                (50938, 'ProfileHueSatMapData1'),
                (50939, 'ProfileHueSatMapData2'),
                (50940, 'ProfileToneCurve'),
                (50941, 'ProfileEmbedPolicy'),
                (50942, 'ProfileCopyright'),
                (50964, 'ForwardMatrix1'),
                (50965, 'ForwardMatrix2'),
                (50966, 'PreviewApplicationName'),
                (50967, 'PreviewApplicationVersion'),
                (50968, 'PreviewSettingsName'),
                (50969, 'PreviewSettingsDigest'),
                (50970, 'PreviewColorSpace'),
                (50971, 'PreviewDateTime'),
                (50972, 'RawImageDigest'),
                (50973, 'OriginalRawFileDigest'),
                (50974, 'SubTileBlockSize'),
                (50975, 'RowInterleaveFactor'),
                (50981, 'ProfileLookTableDims'),
                (50982, 'ProfileLookTableData'),
                (51008, 'OpcodeList1'),
                (51009, 'OpcodeList2'),
                (51022, 'OpcodeList3'),
                (51023, 'FibicsXML'),
                (51041, 'NoiseProfile'),
                (51043, 'TimeCodes'),
                (51044, 'FrameRate'),
                (51058, 'TStop'),
                (51081, 'ReelName'),
                (51089, 'OriginalDefaultFinalSize'),
                (51090, 'OriginalBestQualitySize'),
                (51091, 'OriginalDefaultCropSize'),
                (51105, 'CameraLabel'),
                (51107, 'ProfileHueSatMapEncoding'),
                (51108, 'ProfileLookTableEncoding'),
                (51109, 'BaselineExposureOffset'),
                (51110, 'DefaultBlackRender'),
                (51111, 'NewRawImageDigest'),
                (51112, 'RawToPreviewGain'),
                (51113, 'CacheBlob'),
                (51114, 'CacheVersion'),
                (51123, 'MicroManagerMetadata'),
                (51125, 'DefaultUserCrop'),
                (51159, 'ZIFmetadata'),  # Objective Pathology Services
                (51160, 'ZIFannotations'),  # Objective Pathology Services
                (51177, 'DepthFormat'),
                (51178, 'DepthNear'),
                (51179, 'DepthFar'),
                (51180, 'DepthUnits'),
                (51181, 'DepthMeasureType'),
                (51182, 'EnhanceParams'),
                (52525, 'ProfileGainTableMap'),  # DNG 1.6
                (52526, 'SemanticName'),  # DNG 1.6
                (52528, 'SemanticInstanceID'),  # DNG 1.6
                (52536, 'MaskSubArea'),  # DNG 1.6
                (52543, 'RGBTables'),  # DNG 1.6
                (52529, 'CalibrationIlluminant3'),  # DNG 1.6
                (52531, 'ColorMatrix3'),  # DNG 1.6
                (52530, 'CameraCalibration3'),  # DNG 1.6
                (52538, 'ReductionMatrix3'),  # DNG 1.6
                (52537, 'ProfileHueSatMapData3'),  # DNG 1.6
                (52532, 'ForwardMatrix3'),  # DNG 1.6
                (52533, 'IlluminantData1'),  # DNG 1.6
                (52534, 'IlluminantData2'),  # DNG 1.6
                (53535, 'IlluminantData3'),  # DNG 1.6
                (52544, 'ProfileGainTableMap2'),  # DNG 1.7
                (52547, 'ColumnInterleaveFactor'),  # DNG 1.7
                (52548, 'ImageSequenceInfo'),  # DNG 1.7
                (52550, 'ImageStats'),  # DNG 1.7
                (52551, 'ProfileDynamicRange'),  # DNG 1.7
                (52552, 'ProfileGroupName'),  # DNG 1.7
                (52553, 'JXLDistance'),  # DNG 1.7
                (52554, 'JXLEffort'),  # DNG 1.7
                (52555, 'JXLDecodeSpeed'),  # DNG 1.7
                (55000, 'AperioUnknown55000'),
                (55001, 'AperioMagnification'),
                (55002, 'AperioMPP'),
                (55003, 'AperioScanScopeID'),
                (55004, 'AperioDate'),
                (59932, 'Padding'),
                (59933, 'OffsetSchema'),
                # Reusable Tags 65000-65535
                # (65000, 'DimapDocumentXML'),
                # EER metadata:
                # (65001, 'AcquisitionMetadata'),
                # (65002, 'FrameMetadata'),
                # (65005, 'ImageMetadata'),  # ?
                # (65006, 'ImageMetadata'),
                # (65007, 'PosSkipBits'),
                # (65008, 'HorzSubBits'),
                # (65009, 'VertSubBits'),
                # Photoshop Camera RAW EXIF tags:
                # (65000, 'OwnerName'),
                # (65001, 'SerialNumber'),
                # (65002, 'Lens'),
                # (65024, 'KodakKDCPrivateIFD'),
                # (65100, 'RawFile'),
                # (65101, 'Converter'),
                # (65102, 'WhiteBalance'),
                # (65105, 'Exposure'),
                # (65106, 'Shadows'),
                # (65107, 'Brightness'),
                # (65108, 'Contrast'),
                # (65109, 'Saturation'),
                # (65110, 'Sharpness'),
                # (65111, 'Smoothness'),
                # (65112, 'MoireFilter'),
                # JEOL TEM metadata
                # (65006, 'JEOL_DOUBLE1'),
                # (65007, 'JEOL_DOUBLE2'),
                # (65009, 'JEOL_DOUBLE3'),
                # (65010, 'JEOL_DOUBLE4'),
                # (65015, 'JEOL_SLONG1'),
                # (65016, 'JEOL_SLONG2'),
                # (65024, 'JEOL_DOUBLE5'),
                # (65025, 'JEOL_DOUBLE6'),
                # (65026, 'JEOL_SLONG3'),
                (65027, 'JEOL_Header'),
                (65200, 'FlexXML'),
            )
        )

    @cached_property
    def TAG_READERS(
        self,
    ) -> dict[int, Callable[[FileHandle, ByteOrder, int, int, int], Any]]:
        # map tag codes to import functions
        return {
            301: read_colormap,
            320: read_colormap,
            # 700: read_bytes,  # read_utf8,
            # 34377: read_bytes,
            33723: read_bytes,
            # 34675: read_bytes,
            33628: read_uic1tag,  # Universal Imaging Corp STK
            33629: read_uic2tag,
            33630: read_uic3tag,
            33631: read_uic4tag,
            34118: read_cz_sem,  # Carl Zeiss SEM
            34361: read_mm_header,  # Olympus FluoView
            34362: read_mm_stamp,
            34363: read_numpy,  # MM_Unknown
            34386: read_numpy,  # MM_UserBlock
            34412: read_cz_lsminfo,  # Carl Zeiss LSM
            34680: read_fei_metadata,  # S-FEG
            34682: read_fei_metadata,  # Helios NanoLab
            37706: read_tvips_header,  # TVIPS EMMENU
            37724: read_bytes,  # ImageSourceData
            33923: read_bytes,  # read_leica_magic
            43314: read_nih_image_header,
            # 40001: read_bytes,
            40100: read_bytes,
            50288: read_bytes,
            50296: read_bytes,
            50839: read_bytes,
            51123: read_json,
            33471: read_sis_ini,
            33560: read_sis,
            34665: read_exif_ifd,
            34853: read_gps_ifd,  # conflicts with OlympusSIS
            40965: read_interoperability_ifd,
            65426: read_numpy,  # NDPI McuStarts
            65432: read_numpy,  # NDPI McuStartsHighBytes
            65439: read_numpy,  # NDPI unknown
            65459: read_bytes,  # NDPI bytes, not string
        }

    @cached_property
    def TAG_LOAD(self) -> frozenset[int]:
        # tags whose values are not delay loaded
        return frozenset(
            (
                258,  # BitsPerSample
                270,  # ImageDescription
                273,  # StripOffsets
                277,  # SamplesPerPixel
                279,  # StripByteCounts
                282,  # XResolution
                283,  # YResolution
                # 301,  # TransferFunction
                305,  # Software
                # 306,  # DateTime
                # 320,  # ColorMap
                324,  # TileOffsets
                325,  # TileByteCounts
                330,  # SubIFDs
                338,  # ExtraSamples
                339,  # SampleFormat
                347,  # JPEGTables
                513,  # JPEGInterchangeFormat
                514,  # JPEGInterchangeFormatLength
                530,  # YCbCrSubSampling
                33628,  # UIC1tag
                42113,  # GDAL_NODATA
                50838,  # IJMetadataByteCounts
                50839,  # IJMetadata
            )
        )

    @cached_property
    def TAG_FILTERED(self) -> frozenset[int]:
        # tags filtered from extratags in :py:meth:`TiffWriter.write`
        return frozenset(
            (
                256,  # ImageWidth
                257,  # ImageLength
                258,  # BitsPerSample
                259,  # Compression
                262,  # PhotometricInterpretation
                266,  # FillOrder
                273,  # StripOffsets
                277,  # SamplesPerPixel
                278,  # RowsPerStrip
                279,  # StripByteCounts
                284,  # PlanarConfiguration
                317,  # Predictor
                322,  # TileWidth
                323,  # TileLength
                324,  # TileOffsets
                325,  # TileByteCounts
                330,  # SubIFDs,
                338,  # ExtraSamples
                339,  # SampleFormat
                400,  # GlobalParametersIFD
                32997,  # ImageDepth
                32998,  # TileDepth
                34665,  # ExifTag
                34853,  # GPSTag
                40965,  # InteroperabilityTag
            )
        )

    @cached_property
    def TAG_TUPLE(self) -> frozenset[int]:
        # tags whose values must be stored as tuples
        return frozenset(
            (
                273,
                279,
                282,
                283,
                324,
                325,
                330,
                338,
                513,
                514,
                530,
                531,
                34736,
                50838,
            )
        )

    @cached_property
    def TAG_ATTRIBUTES(self) -> dict[int, str]:
        # map tag codes to TiffPage attribute names
        return {
            254: 'subfiletype',
            256: 'imagewidth',
            257: 'imagelength',
            # 258: 'bitspersample',  # set manually
            259: 'compression',
            262: 'photometric',
            266: 'fillorder',
            270: 'description',
            277: 'samplesperpixel',
            278: 'rowsperstrip',
            284: 'planarconfig',
            # 301: 'transferfunction',  # delay load
            305: 'software',
            # 320: 'colormap',  # delay load
            317: 'predictor',
            322: 'tilewidth',
            323: 'tilelength',
            330: 'subifds',
            338: 'extrasamples',
            # 339: 'sampleformat',  # set manually
            347: 'jpegtables',
            530: 'subsampling',
            32997: 'imagedepth',
            32998: 'tiledepth',
        }

    @cached_property
    def TAG_ENUM(self) -> dict[int, type[enum.Enum]]:
        # map tag codes to Enums
        return {
            254: FILETYPE,
            255: OFILETYPE,
            259: COMPRESSION,
            262: PHOTOMETRIC,
            # 263: THRESHOLD,
            266: FILLORDER,
            274: ORIENTATION,
            284: PLANARCONFIG,
            # 290: GRAYRESPONSEUNIT,
            # 292: GROUP3OPT
            # 293: GROUP4OPT
            296: RESUNIT,
            # 300: COLORRESPONSEUNIT,
            317: PREDICTOR,
            338: EXTRASAMPLE,
            339: SAMPLEFORMAT,
            # 512: JPEGPROC
            # 531: YCBCRPOSITION
        }

    @cached_property
    def EXIF_TAGS(self) -> TiffTagRegistry:
        """Registry of EXIF tags, including private Photoshop Camera RAW."""
        # 65000 - 65112  Photoshop Camera RAW EXIF tags
        tags = TiffTagRegistry(
            (
                (65000, 'OwnerName'),
                (65001, 'SerialNumber'),
                (65002, 'Lens'),
                (65100, 'RawFile'),
                (65101, 'Converter'),
                (65102, 'WhiteBalance'),
                (65105, 'Exposure'),
                (65106, 'Shadows'),
                (65107, 'Brightness'),
                (65108, 'Contrast'),
                (65109, 'Saturation'),
                (65110, 'Sharpness'),
                (65111, 'Smoothness'),
                (65112, 'MoireFilter'),
            )
        )
        tags.update(TIFF.TAGS)
        return tags

    @cached_property
    def NDPI_TAGS(self) -> TiffTagRegistry:
        """Registry of private TIFF tags for Hamamatsu NDPI (65420-65458)."""
        # TODO: obtain specification
        return TiffTagRegistry(
            (
                (65324, 'OffsetHighBytes'),
                (65325, 'ByteCountHighBytes'),
                (65420, 'FileFormat'),
                (65421, 'Magnification'),  # SourceLens
                (65422, 'XOffsetFromSlideCenter'),
                (65423, 'YOffsetFromSlideCenter'),
                (65424, 'ZOffsetFromSlideCenter'),  # FocalPlane
                (65425, 'TissueIndex'),
                (65426, 'McuStarts'),
                (65427, 'SlideLabel'),
                (65428, 'AuthCode'),  # ?
                (65429, '65429'),
                (65430, '65430'),
                (65431, '65431'),
                (65432, 'McuStartsHighBytes'),
                (65433, '65433'),
                (65434, 'Fluorescence'),  # FilterSetName, Channel
                (65435, 'ExposureRatio'),
                (65436, 'RedMultiplier'),
                (65437, 'GreenMultiplier'),
                (65438, 'BlueMultiplier'),
                (65439, 'FocusPoints'),
                (65440, 'FocusPointRegions'),
                (65441, 'CaptureMode'),
                (65442, 'ScannerSerialNumber'),
                (65443, '65443'),
                (65444, 'JpegQuality'),
                (65445, 'RefocusInterval'),
                (65446, 'FocusOffset'),
                (65447, 'BlankLines'),
                (65448, 'FirmwareVersion'),
                (65449, 'Comments'),  # PropertyMap, CalibrationInfo
                (65450, 'LabelObscured'),
                (65451, 'Wavelength'),
                (65452, '65452'),
                (65453, 'LampAge'),
                (65454, 'ExposureTime'),
                (65455, 'FocusTime'),
                (65456, 'ScanTime'),
                (65457, 'WriteTime'),
                (65458, 'FullyAutoFocus'),
                (65500, 'DefaultGamma'),
            )
        )

    @cached_property
    def GPS_TAGS(self) -> TiffTagRegistry:
        """Registry of GPS IFD tags."""
        return TiffTagRegistry(
            (
                (0, 'GPSVersionID'),
                (1, 'GPSLatitudeRef'),
                (2, 'GPSLatitude'),
                (3, 'GPSLongitudeRef'),
                (4, 'GPSLongitude'),
                (5, 'GPSAltitudeRef'),
                (6, 'GPSAltitude'),
                (7, 'GPSTimeStamp'),
                (8, 'GPSSatellites'),
                (9, 'GPSStatus'),
                (10, 'GPSMeasureMode'),
                (11, 'GPSDOP'),
                (12, 'GPSSpeedRef'),
                (13, 'GPSSpeed'),
                (14, 'GPSTrackRef'),
                (15, 'GPSTrack'),
                (16, 'GPSImgDirectionRef'),
                (17, 'GPSImgDirection'),
                (18, 'GPSMapDatum'),
                (19, 'GPSDestLatitudeRef'),
                (20, 'GPSDestLatitude'),
                (21, 'GPSDestLongitudeRef'),
                (22, 'GPSDestLongitude'),
                (23, 'GPSDestBearingRef'),
                (24, 'GPSDestBearing'),
                (25, 'GPSDestDistanceRef'),
                (26, 'GPSDestDistance'),
                (27, 'GPSProcessingMethod'),
                (28, 'GPSAreaInformation'),
                (29, 'GPSDateStamp'),
                (30, 'GPSDifferential'),
                (31, 'GPSHPositioningError'),
            )
        )

    @cached_property
    def IOP_TAGS(self) -> TiffTagRegistry:
        """Registry of Interoperability IFD tags."""
        return TiffTagRegistry(
            (
                (1, 'InteroperabilityIndex'),
                (2, 'InteroperabilityVersion'),
                (4096, 'RelatedImageFileFormat'),
                (4097, 'RelatedImageWidth'),
                (4098, 'RelatedImageLength'),
            )
        )

    @cached_property
    def PHOTOMETRIC_SAMPLES(self) -> dict[int, int]:
        """Map :py:class:`PHOTOMETRIC` to number of photometric samples."""
        return {
            0: 1,  # MINISWHITE
            1: 1,  # MINISBLACK
            2: 3,  # RGB
            3: 1,  # PALETTE
            4: 1,  # MASK
            5: 4,  # SEPARATED
            6: 3,  # YCBCR
            8: 3,  # CIELAB
            9: 3,  # ICCLAB
            10: 3,  # ITULAB
            32803: 1,  # CFA
            32844: 1,  # LOGL ?
            32845: 3,  # LOGLUV
            34892: 3,  # LINEAR_RAW ?
            51177: 1,  # DEPTH_MAP ?
            52527: 1,  # SEMANTIC_MASK ?
        }

    @cached_property
    def DATA_FORMATS(self) -> dict[int, str]:
        """Map :py:class:`DATATYPE` to Python struct formats."""
        return {
            1: '1B',
            2: '1s',
            3: '1H',
            4: '1I',
            5: '2I',
            6: '1b',
            7: '1B',
            8: '1h',
            9: '1i',
            10: '2i',
            11: '1f',
            12: '1d',
            13: '1I',
            # 14: '',
            # 15: '',
            16: '1Q',
            17: '1q',
            18: '1Q',
        }

    @cached_property
    def DATA_DTYPES(self) -> dict[str, int]:
        """Map NumPy dtype to :py:class:`DATATYPE`."""
        return {
            'B': 1,
            's': 2,
            'H': 3,
            'I': 4,
            '2I': 5,
            'b': 6,
            'h': 8,
            'i': 9,
            '2i': 10,
            'f': 11,
            'd': 12,
            'Q': 16,
            'q': 17,
        }

    @cached_property
    def SAMPLE_DTYPES(self) -> dict[tuple[int, int | tuple[int, ...]], str]:
        """Map :py:class:`SAMPLEFORMAT` and BitsPerSample to NumPy dtype."""
        return {
            # UINT
            (1, 1): '?',  # bitmap
            (1, 2): 'B',
            (1, 3): 'B',
            (1, 4): 'B',
            (1, 5): 'B',
            (1, 6): 'B',
            (1, 7): 'B',
            (1, 8): 'B',
            (1, 9): 'H',
            (1, 10): 'H',
            (1, 11): 'H',
            (1, 12): 'H',
            (1, 13): 'H',
            (1, 14): 'H',
            (1, 15): 'H',
            (1, 16): 'H',
            (1, 17): 'I',
            (1, 18): 'I',
            (1, 19): 'I',
            (1, 20): 'I',
            (1, 21): 'I',
            (1, 22): 'I',
            (1, 23): 'I',
            (1, 24): 'I',
            (1, 25): 'I',
            (1, 26): 'I',
            (1, 27): 'I',
            (1, 28): 'I',
            (1, 29): 'I',
            (1, 30): 'I',
            (1, 31): 'I',
            (1, 32): 'I',
            (1, 64): 'Q',
            # VOID : treat as UINT
            (4, 1): '?',  # bitmap
            (4, 2): 'B',
            (4, 3): 'B',
            (4, 4): 'B',
            (4, 5): 'B',
            (4, 6): 'B',
            (4, 7): 'B',
            (4, 8): 'B',
            (4, 9): 'H',
            (4, 10): 'H',
            (4, 11): 'H',
            (4, 12): 'H',
            (4, 13): 'H',
            (4, 14): 'H',
            (4, 15): 'H',
            (4, 16): 'H',
            (4, 17): 'I',
            (4, 18): 'I',
            (4, 19): 'I',
            (4, 20): 'I',
            (4, 21): 'I',
            (4, 22): 'I',
            (4, 23): 'I',
            (4, 24): 'I',
            (4, 25): 'I',
            (4, 26): 'I',
            (4, 27): 'I',
            (4, 28): 'I',
            (4, 29): 'I',
            (4, 30): 'I',
            (4, 31): 'I',
            (4, 32): 'I',
            (4, 64): 'Q',
            # INT
            (2, 8): 'b',
            (2, 16): 'h',
            (2, 32): 'i',
            (2, 64): 'q',
            # IEEEFP
            (3, 16): 'e',
            (3, 24): 'f',  # float24 bit not supported by numpy
            (3, 32): 'f',
            (3, 64): 'd',
            # COMPLEXIEEEFP
            (6, 64): 'F',
            (6, 128): 'D',
            # RGB565
            (1, (5, 6, 5)): 'B',
            # COMPLEXINT : not supported by numpy
            (5, 16): 'E',
            (5, 32): 'F',
            (5, 64): 'D',
        }

    @cached_property
    def PREDICTORS(self) -> Mapping[int, Callable[..., Any]]:
        """Map :py:class:`PREDICTOR` value to encode function."""
        return PredictorCodec(encode=True)

    @cached_property
    def UNPREDICTORS(self) -> Mapping[int, Callable[..., Any]]:
        """Map :py:class:`PREDICTOR` value to decode function."""
        return PredictorCodec(encode=False)

    @cached_property
    def COMPRESSORS(self) -> Mapping[int, Callable[..., Any]]:
        """Map :py:class:`COMPRESSION` value to compress function."""
        return CompressionCodec(encode=True)

    @cached_property
    def DECOMPRESSORS(self) -> Mapping[int, Callable[..., Any]]:
        """Map :py:class:`COMPRESSION` value to decompress function."""
        return CompressionCodec(encode=False)

    @cached_property
    def IMAGE_COMPRESSIONS(self) -> set[int]:
        # set of compression to encode/decode images
        # encode/decode preserves shape and dtype
        # cannot be used with predictors or fillorder
        return {
            6,  # jpeg
            7,  # jpeg
            22610,  # jpegxr
            33003,  # jpeg2k
            33004,  # jpeg2k
            33005,  # jpeg2k
            33007,  # alt_jpeg
            34712,  # jpeg2k
            34892,  # jpeg
            34933,  # png
            34934,  # jpegxr ZIF
            48124,  # jetraw
            50001,  # webp
            50002,  # jpegxl
            52546,  # jpegxl DNG
            65000,  # EER
            65001,  # EER
            65002,  # EER
        }

    @cached_property
    def AXES_NAMES(self) -> dict[str, str]:
        """Map axes character codes to dimension names.

        - **X : width** (image width)
        - **Y : height** (image length)
        - **Z : depth** (image depth)
        - **S : sample** (color space and extra samples)
        - **I : sequence** (generic sequence of images, frames, planes, pages)
        - **T : time** (time series)
        - **C : channel** (acquisition path or emission wavelength)
        - **A : angle** (OME)
        - **P : phase** (OME. In LSM, **P** maps to **position**)
        - **R : tile** (OME. Region, position, or mosaic)
        - **H : lifetime** (OME. Histogram)
        - **E : lambda** (OME. Excitation wavelength)
        - **Q : other** (OME)
        - **L : exposure** (FluoView)
        - **V : event** (FluoView)
        - **M : mosaic** (LSM 6)
        - **J : column** (NDTiff)
        - **K : row** (NDTiff)

        There is no universal standard for dimension codes or names.
        This mapping mainly follows TIFF, OME-TIFF, ImageJ, LSM, and FluoView
        conventions.

        """
        return {
            'X': 'width',
            'Y': 'height',
            'Z': 'depth',
            'S': 'sample',
            'I': 'sequence',
            # 'F': 'file',
            'T': 'time',
            'C': 'channel',
            'A': 'angle',
            'P': 'phase',
            'R': 'tile',
            'H': 'lifetime',
            'E': 'lambda',
            'L': 'exposure',
            'V': 'event',
            'M': 'mosaic',
            'Q': 'other',
            'J': 'column',
            'K': 'row',
        }

    @cached_property
    def AXES_CODES(self) -> dict[str, str]:
        """Map dimension names to axes character codes.

        Reverse mapping of :py:attr:`AXES_NAMES`.

        """
        codes = {name: code for code, name in TIFF.AXES_NAMES.items()}
        codes['z'] = 'Z'  # NDTiff
        codes['position'] = 'R'  # NDTiff
        return codes

    @cached_property
    def GEO_KEYS(self) -> type[enum.IntEnum]:
        """:py:class:`geodb.GeoKeys`."""
        try:
            from .geodb import GeoKeys
        except ImportError:

            class GeoKeys(enum.IntEnum):  # type: ignore[no-redef]
                pass

        return GeoKeys

    @cached_property
    def GEO_CODES(self) -> dict[int, type[enum.IntEnum]]:
        """Map :py:class:`geodb.GeoKeys` to GeoTIFF codes."""
        try:
            from .geodb import GEO_CODES
        except ImportError:
            GEO_CODES = {}
        return GEO_CODES

    @cached_property
    def PAGE_FLAGS(self) -> set[str]:
        # TiffFile and TiffPage 'is_\*' attributes
        exclude = {
            'reduced',
            'mask',
            'final',
            'memmappable',
            'contiguous',
            'tiled',
            'subsampled',
            'jfif',
        }
        return {
            a[3:]
            for a in dir(TiffPage)
            if a[:3] == 'is_' and a[3:] not in exclude
        }

    @cached_property
    def FILE_FLAGS(self) -> set[str]:
        # TiffFile 'is_\*' attributes
        exclude = {'bigtiff', 'appendable'}
        return {
            a[3:]
            for a in dir(TiffFile)
            if a[:3] == 'is_' and a[3:] not in exclude
        }.union(TIFF.PAGE_FLAGS)

    @property
    def FILE_PATTERNS(self) -> dict[str, str]:
        # predefined FileSequence patterns
        return {'axes': r"""(?ix)
                # matches Olympus OIF and Leica TIFF series
                _?(?:(q|l|p|a|c|t|x|y|z|ch|tp)(\d{1,4}))
                _?(?:(q|l|p|a|c|t|x|y|z|ch|tp)(\d{1,4}))?
                _?(?:(q|l|p|a|c|t|x|y|z|ch|tp)(\d{1,4}))?
                _?(?:(q|l|p|a|c|t|x|y|z|ch|tp)(\d{1,4}))?
                _?(?:(q|l|p|a|c|t|x|y|z|ch|tp)(\d{1,4}))?
                _?(?:(q|l|p|a|c|t|x|y|z|ch|tp)(\d{1,4}))?
                _?(?:(q|l|p|a|c|t|x|y|z|ch|tp)(\d{1,4}))?
                """}

    @property
    def FILE_EXTENSIONS(self) -> tuple[str, ...]:
        """Known TIFF file extensions."""
        return (
            'tif',
            'tiff',
            'ome.tif',
            'lsm',
            'stk',
            'qpi',
            'pcoraw',
            'qptiff',
            'ptiff',
            'ptif',
            'gel',
            'seq',
            'svs',
            'avs',
            'scn',
            'zif',
            'ndpi',
            'bif',
            'tf8',
            'tf2',
            'btf',
            'eer',
        )

    @property
    def FILEOPEN_FILTER(self) -> list[tuple[str, str]]:
        # string for use in Windows File Open box
        return [
            (f'{ext.upper()} files', f'*.{ext}')
            for ext in TIFF.FILE_EXTENSIONS
        ] + [('All files', '*')]

    @property
    def CZ_LSMINFO(self) -> list[tuple[str, str]]:
        # numpy data type of LSMINFO structure
        return [
            ('MagicNumber', 'u4'),
            ('StructureSize', 'i4'),
            ('DimensionX', 'i4'),
            ('DimensionY', 'i4'),
            ('DimensionZ', 'i4'),
            ('DimensionChannels', 'i4'),
            ('DimensionTime', 'i4'),
            ('DataType', 'i4'),  # DATATYPES
            ('ThumbnailX', 'i4'),
            ('ThumbnailY', 'i4'),
            ('VoxelSizeX', 'f8'),
            ('VoxelSizeY', 'f8'),
            ('VoxelSizeZ', 'f8'),
            ('OriginX', 'f8'),
            ('OriginY', 'f8'),
            ('OriginZ', 'f8'),
            ('ScanType', 'u2'),
            ('SpectralScan', 'u2'),
            ('TypeOfData', 'u4'),  # TYPEOFDATA
            ('OffsetVectorOverlay', 'u4'),
            ('OffsetInputLut', 'u4'),
            ('OffsetOutputLut', 'u4'),
            ('OffsetChannelColors', 'u4'),
            ('TimeIntervall', 'f8'),  # typo in LSM spec
            ('OffsetChannelDataTypes', 'u4'),
            ('OffsetScanInformation', 'u4'),  # SCANINFO
            ('OffsetKsData', 'u4'),
            ('OffsetTimeStamps', 'u4'),
            ('OffsetEventList', 'u4'),
            ('OffsetRoi', 'u4'),
            ('OffsetBleachRoi', 'u4'),
            ('OffsetNextRecording', 'u4'),
            # LSM 2.0 ends here
            ('DisplayAspectX', 'f8'),
            ('DisplayAspectY', 'f8'),
            ('DisplayAspectZ', 'f8'),
            ('DisplayAspectTime', 'f8'),
            ('OffsetMeanOfRoisOverlay', 'u4'),
            ('OffsetTopoIsolineOverlay', 'u4'),
            ('OffsetTopoProfileOverlay', 'u4'),
            ('OffsetLinescanOverlay', 'u4'),
            ('ToolbarFlags', 'u4'),
            ('OffsetChannelWavelength', 'u4'),
            ('OffsetChannelFactors', 'u4'),
            ('ObjectiveSphereCorrection', 'f8'),
            ('OffsetUnmixParameters', 'u4'),
            # LSM 3.2, 4.0 end here
            ('OffsetAcquisitionParameters', 'u4'),
            ('OffsetCharacteristics', 'u4'),
            ('OffsetPalette', 'u4'),
            ('TimeDifferenceX', 'f8'),
            ('TimeDifferenceY', 'f8'),
            ('TimeDifferenceZ', 'f8'),
            ('InternalUse1', 'u4'),
            ('DimensionP', 'i4'),
            ('DimensionM', 'i4'),
            ('DimensionsReserved', '16i4'),
            ('OffsetTilePositions', 'u4'),
            ('', '9u4'),  # Reserved
            ('OffsetPositions', 'u4'),
            # ('', '21u4'),  # must be 0
        ]

    @property
    def CZ_LSMINFO_READERS(
        self,
    ) -> dict[str, Callable[[FileHandle], Any] | None]:
        # import functions for CZ_LSMINFO sub-records
        # TODO: read more CZ_LSMINFO sub-records
        return {
            'ScanInformation': read_lsm_scaninfo,
            'TimeStamps': read_lsm_timestamps,
            'EventList': read_lsm_eventlist,
            'ChannelColors': read_lsm_channelcolors,
            'Positions': read_lsm_positions,
            'TilePositions': read_lsm_positions,
            'VectorOverlay': None,
            'InputLut': read_lsm_lookuptable,
            'OutputLut': read_lsm_lookuptable,
            'TimeIntervall': None,  # typo in LSM spec
            'ChannelDataTypes': read_lsm_channeldatatypes,
            'KsData': None,
            'Roi': None,
            'BleachRoi': None,
            'NextRecording': None,  # read with TiffFile(fh, offset=)
            'MeanOfRoisOverlay': None,
            'TopoIsolineOverlay': None,
            'TopoProfileOverlay': None,
            'ChannelWavelength': read_lsm_channelwavelength,
            'SphereCorrection': None,
            'ChannelFactors': None,
            'UnmixParameters': None,
            'AcquisitionParameters': None,
            'Characteristics': None,
        }

    @property
    def CZ_LSMINFO_SCANTYPE(self) -> dict[int, str]:
        # map CZ_LSMINFO.ScanType to dimension order
        return {
            0: 'ZCYX',  # Stack, normal x-y-z-scan
            1: 'CZX',  # Z-Scan, x-z-plane
            2: 'CTX',  # Line or Time Series Line
            3: 'TCYX',  # Time Series Plane, x-y
            4: 'TCZX',  # Time Series z-Scan, x-z
            5: 'CTX',  # Time Series Mean-of-ROIs
            6: 'TZCYX',  # Time Series Stack, x-y-z
            7: 'TZCYX',  # TODO: Spline Scan
            8: 'CZX',  # Spline Plane, x-z
            9: 'TCZX',  # Time Series Spline Plane, x-z
            10: 'CTX',  # Point or Time Series Point
        }

    @property
    def CZ_LSMINFO_DIMENSIONS(self) -> dict[str, str]:
        # map dimension codes to CZ_LSMINFO attribute
        return {
            'X': 'DimensionX',
            'Y': 'DimensionY',
            'Z': 'DimensionZ',
            'C': 'DimensionChannels',
            'T': 'DimensionTime',
            'P': 'DimensionP',
            'M': 'DimensionM',
        }

    @property
    def CZ_LSMINFO_DATATYPES(self) -> dict[int, str]:
        # description of CZ_LSMINFO.DataType
        return {
            0: 'varying data types',
            1: '8 bit unsigned integer',
            2: '12 bit unsigned integer',
            5: '32 bit float',
        }

    @property
    def CZ_LSMINFO_TYPEOFDATA(self) -> dict[int, str]:
        # description of CZ_LSMINFO.TypeOfData
        return {
            0: 'Original scan data',
            1: 'Calculated data',
            2: '3D reconstruction',
            3: 'Topography height map',
        }

    @property
    def CZ_LSMINFO_SCANINFO_ARRAYS(self) -> dict[int, str]:
        return {
            0x20000000: 'Tracks',
            0x30000000: 'Lasers',
            0x60000000: 'DetectionChannels',
            0x80000000: 'IlluminationChannels',
            0xA0000000: 'BeamSplitters',
            0xC0000000: 'DataChannels',
            0x11000000: 'Timers',
            0x13000000: 'Markers',
        }

    @property
    def CZ_LSMINFO_SCANINFO_STRUCTS(self) -> dict[int, str]:
        return {
            # 0x10000000: 'Recording',
            0x40000000: 'Track',
            0x50000000: 'Laser',
            0x70000000: 'DetectionChannel',
            0x90000000: 'IlluminationChannel',
            0xB0000000: 'BeamSplitter',
            0xD0000000: 'DataChannel',
            0x12000000: 'Timer',
            0x14000000: 'Marker',
        }

    @property
    def CZ_LSMINFO_SCANINFO_ATTRIBUTES(self) -> dict[int, str]:
        return {
            # Recording
            0x10000001: 'Name',
            0x10000002: 'Description',
            0x10000003: 'Notes',
            0x10000004: 'Objective',
            0x10000005: 'ProcessingSummary',
            0x10000006: 'SpecialScanMode',
            0x10000007: 'ScanType',
            0x10000008: 'ScanMode',
            0x10000009: 'NumberOfStacks',
            0x1000000A: 'LinesPerPlane',
            0x1000000B: 'SamplesPerLine',
            0x1000000C: 'PlanesPerVolume',
            0x1000000D: 'ImagesWidth',
            0x1000000E: 'ImagesHeight',
            0x1000000F: 'ImagesNumberPlanes',
            0x10000010: 'ImagesNumberStacks',
            0x10000011: 'ImagesNumberChannels',
            0x10000012: 'LinscanXySize',
            0x10000013: 'ScanDirection',
            0x10000014: 'TimeSeries',
            0x10000015: 'OriginalScanData',
            0x10000016: 'ZoomX',
            0x10000017: 'ZoomY',
            0x10000018: 'ZoomZ',
            0x10000019: 'Sample0X',
            0x1000001A: 'Sample0Y',
            0x1000001B: 'Sample0Z',
            0x1000001C: 'SampleSpacing',
            0x1000001D: 'LineSpacing',
            0x1000001E: 'PlaneSpacing',
            0x1000001F: 'PlaneWidth',
            0x10000020: 'PlaneHeight',
            0x10000021: 'VolumeDepth',
            0x10000023: 'Nutation',
            0x10000034: 'Rotation',
            0x10000035: 'Precession',
            0x10000036: 'Sample0time',
            0x10000037: 'StartScanTriggerIn',
            0x10000038: 'StartScanTriggerOut',
            0x10000039: 'StartScanEvent',
            0x10000040: 'StartScanTime',
            0x10000041: 'StopScanTriggerIn',
            0x10000042: 'StopScanTriggerOut',
            0x10000043: 'StopScanEvent',
            0x10000044: 'StopScanTime',
            0x10000045: 'UseRois',
            0x10000046: 'UseReducedMemoryRois',
            0x10000047: 'User',
            0x10000048: 'UseBcCorrection',
            0x10000049: 'PositionBcCorrection1',
            0x10000050: 'PositionBcCorrection2',
            0x10000051: 'InterpolationY',
            0x10000052: 'CameraBinning',
            0x10000053: 'CameraSupersampling',
            0x10000054: 'CameraFrameWidth',
            0x10000055: 'CameraFrameHeight',
            0x10000056: 'CameraOffsetX',
            0x10000057: 'CameraOffsetY',
            0x10000059: 'RtBinning',
            0x1000005A: 'RtFrameWidth',
            0x1000005B: 'RtFrameHeight',
            0x1000005C: 'RtRegionWidth',
            0x1000005D: 'RtRegionHeight',
            0x1000005E: 'RtOffsetX',
            0x1000005F: 'RtOffsetY',
            0x10000060: 'RtZoom',
            0x10000061: 'RtLinePeriod',
            0x10000062: 'Prescan',
            0x10000063: 'ScanDirectionZ',
            # Track
            0x40000001: 'MultiplexType',  # 0 After Line; 1 After Frame
            0x40000002: 'MultiplexOrder',
            0x40000003: 'SamplingMode',  # 0 Sample; 1 Line Avg; 2 Frame Avg
            0x40000004: 'SamplingMethod',  # 1 Mean; 2 Sum
            0x40000005: 'SamplingNumber',
            0x40000006: 'Acquire',
            0x40000007: 'SampleObservationTime',
            0x4000000B: 'TimeBetweenStacks',
            0x4000000C: 'Name',
            0x4000000D: 'Collimator1Name',
            0x4000000E: 'Collimator1Position',
            0x4000000F: 'Collimator2Name',
            0x40000010: 'Collimator2Position',
            0x40000011: 'IsBleachTrack',
            0x40000012: 'IsBleachAfterScanNumber',
            0x40000013: 'BleachScanNumber',
            0x40000014: 'TriggerIn',
            0x40000015: 'TriggerOut',
            0x40000016: 'IsRatioTrack',
            0x40000017: 'BleachCount',
            0x40000018: 'SpiCenterWavelength',
            0x40000019: 'PixelTime',
            0x40000021: 'CondensorFrontlens',
            0x40000023: 'FieldStopValue',
            0x40000024: 'IdCondensorAperture',
            0x40000025: 'CondensorAperture',
            0x40000026: 'IdCondensorRevolver',
            0x40000027: 'CondensorFilter',
            0x40000028: 'IdTransmissionFilter1',
            0x40000029: 'IdTransmission1',
            0x40000030: 'IdTransmissionFilter2',
            0x40000031: 'IdTransmission2',
            0x40000032: 'RepeatBleach',
            0x40000033: 'EnableSpotBleachPos',
            0x40000034: 'SpotBleachPosx',
            0x40000035: 'SpotBleachPosy',
            0x40000036: 'SpotBleachPosz',
            0x40000037: 'IdTubelens',
            0x40000038: 'IdTubelensPosition',
            0x40000039: 'TransmittedLight',
            0x4000003A: 'ReflectedLight',
            0x4000003B: 'SimultanGrabAndBleach',
            0x4000003C: 'BleachPixelTime',
            # Laser
            0x50000001: 'Name',
            0x50000002: 'Acquire',
            0x50000003: 'Power',
            # DetectionChannel
            0x70000001: 'IntegrationMode',
            0x70000002: 'SpecialMode',
            0x70000003: 'DetectorGainFirst',
            0x70000004: 'DetectorGainLast',
            0x70000005: 'AmplifierGainFirst',
            0x70000006: 'AmplifierGainLast',
            0x70000007: 'AmplifierOffsFirst',
            0x70000008: 'AmplifierOffsLast',
            0x70000009: 'PinholeDiameter',
            0x7000000A: 'CountingTrigger',
            0x7000000B: 'Acquire',
            0x7000000C: 'PointDetectorName',
            0x7000000D: 'AmplifierName',
            0x7000000E: 'PinholeName',
            0x7000000F: 'FilterSetName',
            0x70000010: 'FilterName',
            0x70000013: 'IntegratorName',
            0x70000014: 'ChannelName',
            0x70000015: 'DetectorGainBc1',
            0x70000016: 'DetectorGainBc2',
            0x70000017: 'AmplifierGainBc1',
            0x70000018: 'AmplifierGainBc2',
            0x70000019: 'AmplifierOffsetBc1',
            0x70000020: 'AmplifierOffsetBc2',
            0x70000021: 'SpectralScanChannels',
            0x70000022: 'SpiWavelengthStart',
            0x70000023: 'SpiWavelengthStop',
            0x70000026: 'DyeName',
            0x70000027: 'DyeFolder',
            # IlluminationChannel
            0x90000001: 'Name',
            0x90000002: 'Power',
            0x90000003: 'Wavelength',
            0x90000004: 'Aquire',  # typo in LSM spec
            0x90000005: 'DetchannelName',
            0x90000006: 'PowerBc1',
            0x90000007: 'PowerBc2',
            # BeamSplitter
            0xB0000001: 'FilterSet',
            0xB0000002: 'Filter',
            0xB0000003: 'Name',
            # DataChannel
            0xD0000001: 'Name',
            0xD0000003: 'Acquire',
            0xD0000004: 'Color',
            0xD0000005: 'SampleType',
            0xD0000006: 'BitsPerSample',
            0xD0000007: 'RatioType',
            0xD0000008: 'RatioTrack1',
            0xD0000009: 'RatioTrack2',
            0xD000000A: 'RatioChannel1',
            0xD000000B: 'RatioChannel2',
            0xD000000C: 'RatioConst1',
            0xD000000D: 'RatioConst2',
            0xD000000E: 'RatioConst3',
            0xD000000F: 'RatioConst4',
            0xD0000010: 'RatioConst5',
            0xD0000011: 'RatioConst6',
            0xD0000012: 'RatioFirstImages1',
            0xD0000013: 'RatioFirstImages2',
            0xD0000014: 'DyeName',
            0xD0000015: 'DyeFolder',
            0xD0000016: 'Spectrum',
            0xD0000017: 'Acquire',
            # Timer
            0x12000001: 'Name',
            0x12000002: 'Description',
            0x12000003: 'Interval',
            0x12000004: 'TriggerIn',
            0x12000005: 'TriggerOut',
            0x12000006: 'ActivationTime',
            0x12000007: 'ActivationNumber',
            # Marker
            0x14000001: 'Name',
            0x14000002: 'Description',
            0x14000003: 'TriggerIn',
            0x14000004: 'TriggerOut',
        }

    @cached_property
    def CZ_LSM_LUTTYPE(self):  # TODO: type this
        class CZ_LSM_LUTTYPE(enum.IntEnum):
            NORMAL = 0
            ORIGINAL = 1
            RAMP = 2
            POLYLINE = 3
            SPLINE = 4
            GAMMA = 5

        return CZ_LSM_LUTTYPE

    @cached_property
    def CZ_LSM_SUBBLOCK_TYPE(self):  # TODO: type this
        class CZ_LSM_SUBBLOCK_TYPE(enum.IntEnum):
            END = 0
            GAMMA = 1
            BRIGHTNESS = 2
            CONTRAST = 3
            RAMP = 4
            KNOTS = 5
            PALETTE_12_TO_12 = 6

        return CZ_LSM_SUBBLOCK_TYPE

    @property
    def NIH_IMAGE_HEADER(self):  # TODO: type this
        return [
            ('FileID', 'S8'),
            ('nLines', 'i2'),
            ('PixelsPerLine', 'i2'),
            ('Version', 'i2'),
            ('OldLutMode', 'i2'),
            ('OldnColors', 'i2'),
            ('Colors', 'u1', (3, 32)),
            ('OldColorStart', 'i2'),
            ('ColorWidth', 'i2'),
            ('ExtraColors', 'u2', (6, 3)),
            ('nExtraColors', 'i2'),
            ('ForegroundIndex', 'i2'),
            ('BackgroundIndex', 'i2'),
            ('XScale', 'f8'),
            ('Unused2', 'i2'),
            ('Unused3', 'i2'),
            ('UnitsID', 'i2'),  # NIH_UNITS_TYPE
            ('p1', [('x', 'i2'), ('y', 'i2')]),
            ('p2', [('x', 'i2'), ('y', 'i2')]),
            ('CurveFitType', 'i2'),  # NIH_CURVEFIT_TYPE
            ('nCoefficients', 'i2'),
            ('Coeff', 'f8', 6),
            ('UMsize', 'u1'),
            ('UM', 'S15'),
            ('UnusedBoolean', 'u1'),
            ('BinaryPic', 'b1'),
            ('SliceStart', 'i2'),
            ('SliceEnd', 'i2'),
            ('ScaleMagnification', 'f4'),
            ('nSlices', 'i2'),
            ('SliceSpacing', 'f4'),
            ('CurrentSlice', 'i2'),
            ('FrameInterval', 'f4'),
            ('PixelAspectRatio', 'f4'),
            ('ColorStart', 'i2'),
            ('ColorEnd', 'i2'),
            ('nColors', 'i2'),
            ('Fill1', '3u2'),
            ('Fill2', '3u2'),
            ('Table', 'u1'),  # NIH_COLORTABLE_TYPE
            ('LutMode', 'u1'),  # NIH_LUTMODE_TYPE
            ('InvertedTable', 'b1'),
            ('ZeroClip', 'b1'),
            ('XUnitSize', 'u1'),
            ('XUnit', 'S11'),
            ('StackType', 'i2'),  # NIH_STACKTYPE_TYPE
            # ('UnusedBytes', 'u1', 200)
        ]

    @property
    def NIH_COLORTABLE_TYPE(self) -> tuple[str, ...]:
        return (
            'CustomTable',
            'AppleDefault',
            'Pseudo20',
            'Pseudo32',
            'Rainbow',
            'Fire1',
            'Fire2',
            'Ice',
            'Grays',
            'Spectrum',
        )

    @property
    def NIH_LUTMODE_TYPE(self) -> tuple[str, ...]:
        return (
            'PseudoColor',
            'OldAppleDefault',
            'OldSpectrum',
            'GrayScale',
            'ColorLut',
            'CustomGrayscale',
        )

    @property
    def NIH_CURVEFIT_TYPE(self) -> tuple[str, ...]:
        return (
            'StraightLine',
            'Poly2',
            'Poly3',
            'Poly4',
            'Poly5',
            'ExpoFit',
            'PowerFit',
            'LogFit',
            'RodbardFit',
            'SpareFit1',
            'Uncalibrated',
            'UncalibratedOD',
        )

    @property
    def NIH_UNITS_TYPE(self) -> tuple[str, ...]:
        return (
            'Nanometers',
            'Micrometers',
            'Millimeters',
            'Centimeters',
            'Meters',
            'Kilometers',
            'Inches',
            'Feet',
            'Miles',
            'Pixels',
            'OtherUnits',
        )

    @property
    def TVIPS_HEADER_V1(self) -> list[tuple[str, str]]:
        # TVIPS TemData structure from EMMENU Help file
        return [
            ('Version', 'i4'),
            ('CommentV1', 'S80'),
            ('HighTension', 'i4'),
            ('SphericalAberration', 'i4'),
            ('IlluminationAperture', 'i4'),
            ('Magnification', 'i4'),
            ('PostMagnification', 'i4'),
            ('FocalLength', 'i4'),
            ('Defocus', 'i4'),
            ('Astigmatism', 'i4'),
            ('AstigmatismDirection', 'i4'),
            ('BiprismVoltage', 'i4'),
            ('SpecimenTiltAngle', 'i4'),
            ('SpecimenTiltDirection', 'i4'),
            ('IlluminationTiltDirection', 'i4'),
            ('IlluminationTiltAngle', 'i4'),
            ('ImageMode', 'i4'),
            ('EnergySpread', 'i4'),
            ('ChromaticAberration', 'i4'),
            ('ShutterType', 'i4'),
            ('DefocusSpread', 'i4'),
            ('CcdNumber', 'i4'),
            ('CcdSize', 'i4'),
            ('OffsetXV1', 'i4'),
            ('OffsetYV1', 'i4'),
            ('PhysicalPixelSize', 'i4'),
            ('Binning', 'i4'),
            ('ReadoutSpeed', 'i4'),
            ('GainV1', 'i4'),
            ('SensitivityV1', 'i4'),
            ('ExposureTimeV1', 'i4'),
            ('FlatCorrected', 'i4'),
            ('DeadPxCorrected', 'i4'),
            ('ImageMean', 'i4'),
            ('ImageStd', 'i4'),
            ('DisplacementX', 'i4'),
            ('DisplacementY', 'i4'),
            ('DateV1', 'i4'),
            ('TimeV1', 'i4'),
            ('ImageMin', 'i4'),
            ('ImageMax', 'i4'),
            ('ImageStatisticsQuality', 'i4'),
        ]

    @property
    def TVIPS_HEADER_V2(self) -> list[tuple[str, str]]:
        return [
            ('ImageName', 'V160'),  # utf16
            ('ImageFolder', 'V160'),
            ('ImageSizeX', 'i4'),
            ('ImageSizeY', 'i4'),
            ('ImageSizeZ', 'i4'),
            ('ImageSizeE', 'i4'),
            ('ImageDataType', 'i4'),
            ('Date', 'i4'),
            ('Time', 'i4'),
            ('Comment', 'V1024'),
            ('ImageHistory', 'V1024'),
            ('Scaling', '16f4'),
            ('ImageStatistics', '16c16'),
            ('ImageType', 'i4'),
            ('ImageDisplayType', 'i4'),
            ('PixelSizeX', 'f4'),  # distance between two px in x, [nm]
            ('PixelSizeY', 'f4'),  # distance between two px in y, [nm]
            ('ImageDistanceZ', 'f4'),
            ('ImageDistanceE', 'f4'),
            ('ImageMisc', '32f4'),
            ('TemType', 'V160'),
            ('TemHighTension', 'f4'),
            ('TemAberrations', '32f4'),
            ('TemEnergy', '32f4'),
            ('TemMode', 'i4'),
            ('TemMagnification', 'f4'),
            ('TemMagnificationCorrection', 'f4'),
            ('PostMagnification', 'f4'),
            ('TemStageType', 'i4'),
            ('TemStagePosition', '5f4'),  # x, y, z, a, b
            ('TemImageShift', '2f4'),
            ('TemBeamShift', '2f4'),
            ('TemBeamTilt', '2f4'),
            ('TilingParameters', '7f4'),  # 0: tiling? 1:x 2:y 3: max x
            #                               4: max y 5: overlap x 6: overlap y
            ('TemIllumination', '3f4'),  # 0: spotsize 1: intensity
            ('TemShutter', 'i4'),
            ('TemMisc', '32f4'),
            ('CameraType', 'V160'),
            ('PhysicalPixelSizeX', 'f4'),
            ('PhysicalPixelSizeY', 'f4'),
            ('OffsetX', 'i4'),
            ('OffsetY', 'i4'),
            ('BinningX', 'i4'),
            ('BinningY', 'i4'),
            ('ExposureTime', 'f4'),
            ('Gain', 'f4'),
            ('ReadoutRate', 'f4'),
            ('FlatfieldDescription', 'V160'),
            ('Sensitivity', 'f4'),
            ('Dose', 'f4'),
            ('CamMisc', '32f4'),
            ('FeiMicroscopeInformation', 'V1024'),
            ('FeiSpecimenInformation', 'V1024'),
            ('Magic', 'u4'),
        ]

    @property
    def MM_HEADER(self) -> list[tuple[Any, ...]]:
        # Olympus FluoView MM_Header
        MM_DIMENSION = [
            ('Name', 'S16'),
            ('Size', 'i4'),
            ('Origin', 'f8'),
            ('Resolution', 'f8'),
            ('Unit', 'S64'),
        ]
        return [
            ('HeaderFlag', 'i2'),
            ('ImageType', 'u1'),
            ('ImageName', 'S257'),
            ('OffsetData', 'u4'),
            ('PaletteSize', 'i4'),
            ('OffsetPalette0', 'u4'),
            ('OffsetPalette1', 'u4'),
            ('CommentSize', 'i4'),
            ('OffsetComment', 'u4'),
            ('Dimensions', MM_DIMENSION, 10),
            ('OffsetPosition', 'u4'),
            ('MapType', 'i2'),
            ('MapMin', 'f8'),
            ('MapMax', 'f8'),
            ('MinValue', 'f8'),
            ('MaxValue', 'f8'),
            ('OffsetMap', 'u4'),
            ('Gamma', 'f8'),
            ('Offset', 'f8'),
            ('GrayChannel', MM_DIMENSION),
            ('OffsetThumbnail', 'u4'),
            ('VoiceField', 'i4'),
            ('OffsetVoiceField', 'u4'),
        ]

    @property
    def MM_DIMENSIONS(self) -> dict[str, str]:
        # map FluoView MM_Header.Dimensions to axes characters
        return {
            'X': 'X',
            'Y': 'Y',
            'Z': 'Z',
            'T': 'T',
            'CH': 'C',
            'WAVELENGTH': 'C',
            'TIME': 'T',
            'XY': 'R',
            'EVENT': 'V',
            'EXPOSURE': 'L',
        }

    @property
    def UIC_TAGS(self) -> list[tuple[str, Any]]:
        # map Universal Imaging Corporation MetaMorph internal tag ids to
        # name and type
        from fractions import Fraction

        return [
            ('AutoScale', int),
            ('MinScale', int),
            ('MaxScale', int),
            ('SpatialCalibration', int),
            ('XCalibration', Fraction),
            ('YCalibration', Fraction),
            ('CalibrationUnits', str),
            ('Name', str),
            ('ThreshState', int),
            ('ThreshStateRed', int),
            ('tagid_10', None),  # undefined
            ('ThreshStateGreen', int),
            ('ThreshStateBlue', int),
            ('ThreshStateLo', int),
            ('ThreshStateHi', int),
            ('Zoom', int),
            ('CreateTime', julian_datetime),
            ('LastSavedTime', julian_datetime),
            ('currentBuffer', int),
            ('grayFit', None),
            ('grayPointCount', None),
            ('grayX', Fraction),
            ('grayY', Fraction),
            ('grayMin', Fraction),
            ('grayMax', Fraction),
            ('grayUnitName', str),
            ('StandardLUT', int),
            ('wavelength', int),
            ('StagePosition', '(%i,2,2)u4'),  # N xy positions as fract
            ('CameraChipOffset', '(%i,2,2)u4'),  # N xy offsets as fract
            ('OverlayMask', None),
            ('OverlayCompress', None),
            ('Overlay', None),
            ('SpecialOverlayMask', None),
            ('SpecialOverlayCompress', None),
            ('SpecialOverlay', None),
            ('ImageProperty', read_uic_property),
            ('StageLabel', '%ip'),  # N str
            ('AutoScaleLoInfo', Fraction),
            ('AutoScaleHiInfo', Fraction),
            ('AbsoluteZ', '(%i,2)u4'),  # N fractions
            ('AbsoluteZValid', '(%i,)u4'),  # N long
            ('Gamma', 'I'),  # 'I' uses offset
            ('GammaRed', 'I'),
            ('GammaGreen', 'I'),
            ('GammaBlue', 'I'),
            ('CameraBin', '2I'),
            ('NewLUT', int),
            ('ImagePropertyEx', None),
            ('PlaneProperty', int),
            ('UserLutTable', '(256,3)u1'),
            ('RedAutoScaleInfo', int),
            ('RedAutoScaleLoInfo', Fraction),
            ('RedAutoScaleHiInfo', Fraction),
            ('RedMinScaleInfo', int),
            ('RedMaxScaleInfo', int),
            ('GreenAutoScaleInfo', int),
            ('GreenAutoScaleLoInfo', Fraction),
            ('GreenAutoScaleHiInfo', Fraction),
            ('GreenMinScaleInfo', int),
            ('GreenMaxScaleInfo', int),
            ('BlueAutoScaleInfo', int),
            ('BlueAutoScaleLoInfo', Fraction),
            ('BlueAutoScaleHiInfo', Fraction),
            ('BlueMinScaleInfo', int),
            ('BlueMaxScaleInfo', int),
            # ('OverlayPlaneColor', read_uic_overlay_plane_color),
        ]

    @property
    def PILATUS_HEADER(self) -> dict[str, Any]:
        # PILATUS CBF Header Specification, Version 1.4
        # map key to [value_indices], type
        return {
            'Detector': ([slice(1, None)], str),
            'Pixel_size': ([1, 4], float),
            'Silicon': ([3], float),
            'Exposure_time': ([1], float),
            'Exposure_period': ([1], float),
            'Tau': ([1], float),
            'Count_cutoff': ([1], int),
            'Threshold_setting': ([1], float),
            'Gain_setting': ([1, 2], str),
            'N_excluded_pixels': ([1], int),
            'Excluded_pixels': ([1], str),
            'Flat_field': ([1], str),
            'Trim_file': ([1], str),
            'Image_path': ([1], str),
            # optional
            'Wavelength': ([1], float),
            'Energy_range': ([1, 2], float),
            'Detector_distance': ([1], float),
            'Detector_Voffset': ([1], float),
            'Beam_xy': ([1, 2], float),
            'Flux': ([1], str),
            'Filter_transmission': ([1], float),
            'Start_angle': ([1], float),
            'Angle_increment': ([1], float),
            'Detector_2theta': ([1], float),
            'Polarization': ([1], float),
            'Alpha': ([1], float),
            'Kappa': ([1], float),
            'Phi': ([1], float),
            'Phi_increment': ([1], float),
            'Chi': ([1], float),
            'Chi_increment': ([1], float),
            'Oscillation_axis': ([slice(1, None)], str),
            'N_oscillations': ([1], int),
            'Start_position': ([1], float),
            'Position_increment': ([1], float),
            'Shutter_time': ([1], float),
            'Omega': ([1], float),
            'Omega_increment': ([1], float),
        }

    @cached_property
    def ALLOCATIONGRANULARITY(self) -> int:
        # alignment for writing contiguous data to TIFF
        import mmap

        return mmap.ALLOCATIONGRANULARITY

    @cached_property
    def MAXWORKERS(self) -> int:
        """Default maximum number of threads for de/compressing segments.

        The value of the ``TIFFFILE_NUM_THREADS`` environment variable if set,
        else half the CPU cores up to 32.

        """
        if 'TIFFFILE_NUM_THREADS' in os.environ:
            return max(1, int(os.environ['TIFFFILE_NUM_THREADS']))
        cpu_count: int | None
        try:
            cpu_count = len(
                os.sched_getaffinity(0)  # type: ignore[attr-defined]
            )
        except AttributeError:
            cpu_count = os.cpu_count()
        if cpu_count is None:
            return 1
        return min(32, max(1, cpu_count // 2))

    @cached_property
    def MAXIOWORKERS(self) -> int:
        """Default maximum number of I/O threads for reading file sequences.

        The value of the ``TIFFFILE_NUM_IOTHREADS`` environment variable if
        set, else 4 more than the number of CPU cores up to 32.

        """
        if 'TIFFFILE_NUM_IOTHREADS' in os.environ:
            return max(1, int(os.environ['TIFFFILE_NUM_IOTHREADS']))
        cpu_count: int | None
        try:
            cpu_count = len(
                os.sched_getaffinity(0)  # type: ignore[attr-defined]
            )
        except AttributeError:
            cpu_count = os.cpu_count()
        if cpu_count is None:
            return 5
        return min(32, cpu_count + 4)

    BUFFERSIZE: int = 268435456
    """Default number of bytes to read or encode in one pass (256 MB)."""


TIFF = _TIFF()


























































































































def parse_filenames(
    files: Sequence[str],
    /,
    pattern: str | None = None,
    axesorder: Sequence[int] | None = None,
    categories: dict[str, dict[str, int]] | None = None,
    *,
    _shape: Sequence[int] | None = None,
) -> tuple[
    tuple[str, ...], tuple[int, ...], list[tuple[int, ...]], Sequence[str]
]:
    r"""Return shape and axes from sequence of file names matching pattern.

    Parameters:
        files:
            Sequence of file names to parse.
        pattern:
            Regular expression pattern matching axes names and chunk indices
            in file names.
            By default, no pattern matching is performed.
            Axes names can be specified by matching groups preceding the index
            groups in the file name, be provided as group names for the index
            groups, or be omitted.
            The predefined 'axes' pattern matches Olympus OIF and Leica TIFF
            series.
        axesorder:
            Indices of axes in pattern. By default, axes are returned in the
            order they appear in pattern.
        categories:
            Map of index group matches to integer indices.
            `{'axislabel': {'category': index}}`
        _shape:
            Shape of file sequence. The default is
            `maximum - minimum + 1` of the parsed indices for each dimension.

    Returns:
        - Axes names for each dimension.
        - Shape of file series.
        - Index of each file in shape.
        - Filtered sequence of file names.

    Examples:
        >>> parse_filenames(
        ...     ['c1001.ext', 'c2002.ext'], r'([^\d])(\d)(?P<t>\d+)\.ext'
        ... )
        (('c', 't'), (2, 2), [(0, 0), (1, 1)], ['c1001.ext', 'c2002.ext'])

    """
    # TODO: add option to filter files that do not match pattern

    shape = None if _shape is None else tuple(_shape)
    if pattern is None:
        if shape is not None and (len(shape) != 1 or shape[0] < len(files)):
            msg = f'shape {(len(files),)} does not fit provided shape {shape}'
            raise ValueError(msg)
        return (
            ('I',),
            (len(files),),
            [(i,) for i in range(len(files))],
            files,
        )

    pattern = TIFF.FILE_PATTERNS.get(pattern, pattern)
    if not pattern:
        msg = 'invalid pattern'
        raise ValueError(msg)
    pattern_compiled: Any
    if isinstance(pattern, str):
        pattern_compiled = re.compile(pattern)
    elif hasattr(pattern, 'groupindex'):
        pattern_compiled = pattern
    else:
        msg = 'invalid pattern'
        raise ValueError(msg)

    if categories is None:
        categories = {}

    def parse(filename: str, /) -> tuple[tuple[str, ...], tuple[int, ...]]:
        # return axes names and indices from file name
        assert categories is not None
        dims: list[str] = []
        indices: list[int] = []
        groupindex = {v: k for k, v in pattern_compiled.groupindex.items()}
        matches = pattern_compiled.search(filename)
        if matches is None:
            msg = f'pattern does not match file name {filename!r}'
            raise ValueError(msg)
        ax = None
        for i, match in enumerate(matches.groups()):
            m = match
            if m is None:
                continue
            if i + 1 in groupindex:
                ax = groupindex[i + 1]
            elif m[0].isalpha():
                ax = m  # axis label for next index
                continue
            if ax is None:
                ax = 'Q'  # no preceding axis letter
            try:
                m = int(categories[ax][m] if ax in categories else m)
            except Exception as exc:
                msg = f'invalid index {m!r}'
                raise ValueError(msg) from exc
            indices.append(m)
            dims.append(ax)
            ax = None
        return tuple(dims), tuple(indices)

    normpaths = [os.path.normpath(f) for f in files]
    if len(normpaths) == 1:
        prefix_str = os.path.dirname(normpaths[0])
    else:
        prefix_str = os.path.commonpath(normpaths)
    prefix = len(prefix_str)

    dims: tuple[str, ...] | None = None
    indices: list[tuple[int, ...]] = []
    for filename in normpaths:
        lbl, idx = parse(filename[prefix:])
        if dims is None:
            dims = lbl
            if axesorder is not None and (
                len(axesorder) != len(dims)
                or any(i not in axesorder for i in range(len(dims)))
            ):
                msg = f'invalid axesorder {axesorder!r} for {dims!r}'
                raise ValueError(msg)
        elif dims != lbl:
            msg = 'dims do not match within image sequence'
            raise ValueError(msg)
        if axesorder is not None:
            idx = tuple(idx[i] for i in axesorder)
        indices.append(idx)

    assert dims is not None
    if axesorder is not None:
        dims = tuple(dims[i] for i in axesorder)

    # determine shape
    indices_array = numpy.array(indices, dtype=numpy.intp)
    parsedshape = numpy.max(indices, axis=0)

    if shape is None:
        startindex = numpy.min(indices_array, axis=0)
        indices_array -= startindex
        parsedshape -= startindex
        parsedshape += 1
        shape = tuple(int(i) for i in parsedshape.tolist())
    elif len(parsedshape) != len(shape) or any(
        i > j for i, j in zip(shape, parsedshape, strict=True)
    ):
        msg = f'parsed shape {parsedshape} does not fit provided shape {shape}'
        raise ValueError(msg)

    indices_list: list[list[int]] = indices_array.tolist()
    indices = [tuple(index) for index in indices_list]

    return dims, shape, indices, files




def iter_strips(
    pageiter: Iterator[NDArray[Any] | None],
    shape: tuple[int, ...],
    dtype: numpy.dtype[Any],
    rowsperstrip: int,
    /,
) -> Iterator[NDArray[Any]]:
    """Return iterator over strips in pages."""
    numstrips = (shape[-3] + rowsperstrip - 1) // rowsperstrip

    for iteritem in pageiter:
        if iteritem is None:
            # for _ in range(numstrips):
            #     yield None
            # continue
            pagedata = numpy.zeros(shape, dtype)
        else:
            pagedata = iteritem.reshape(shape)
        for plane in pagedata:
            for depth in plane:
                for i in range(numstrips):
                    yield depth[i * rowsperstrip : (i + 1) * rowsperstrip]


def iter_tiles(
    data: NDArray[Any],
    tile: tuple[int, ...],
    tiles: tuple[int, ...],
    /,
) -> Iterator[NDArray[Any]]:
    """Return iterator over full tiles in data array of normalized shape.

    Tiles are zero-padded if necessary.

    """
    if not 1 < len(tile) < 4 or len(tile) != len(tiles):
        msg = 'invalid tile or tiles shape'
        raise ValueError(msg)
    chunkshape = (*tile, data.shape[-1])
    chunksize = product(chunkshape)
    dtype = data.dtype
    sz, sy, sx = data.shape[2:5]
    if len(tile) == 2:
        y, x = tile
        for page in data:
            for plane in page:
                for iy in range(tiles[0]):
                    ty = iy * y
                    cy = min(y, sy - ty)
                    for ix in range(tiles[1]):
                        tx = ix * x
                        cx = min(x, sx - tx)
                        chunk = plane[0, ty : ty + cy, tx : tx + cx]
                        if chunk.size != chunksize:
                            chunk_ = numpy.zeros(chunkshape, dtype)
                            chunk_[:cy, :cx] = chunk
                            chunk = chunk_
                        yield chunk
    else:
        z, y, x = tile
        for page in data:
            for plane in page:
                for iz in range(tiles[0]):
                    tz = iz * z
                    cz = min(z, sz - tz)
                    for iy in range(tiles[1]):
                        ty = iy * y
                        cy = min(y, sy - ty)
                        for ix in range(tiles[2]):
                            tx = ix * x
                            cx = min(x, sx - tx)
                            chunk = plane[
                                tz : tz + cz, ty : ty + cy, tx : tx + cx
                            ]
                            if chunk.size != chunksize:
                                chunk_ = numpy.zeros(chunkshape, dtype)
                                chunk_[:cz, :cy, :cx] = chunk
                                chunk = chunk_
                            yield chunk[0] if z == 1 else chunk


def encode_chunks(
    numchunks: int,
    chunkiter: Iterator[NDArray[Any] | None],
    encode: Callable[[NDArray[Any]], bytes],
    shape: Sequence[int],
    dtype: numpy.dtype[Any],
    maxworkers: int | None,
    buffersize: int | None,
    tiled: bool,  # noqa: FBT001
    /,
) -> Iterator[bytes]:
    """Return iterator over encoded chunks."""
    if numchunks <= 0:
        return

    chunksize = product(shape) * dtype.itemsize

    if tiled:
        # pad tiles
        def func(chunk: NDArray[Any] | None, /) -> bytes:
            if chunk is None:
                return b''
            chunk = numpy.ascontiguousarray(chunk, dtype)
            if chunk.nbytes != chunksize:
                # if chunk.dtype != dtype:
                #     raise ValueError('dtype of chunk does not match data')
                pad = tuple(
                    (0, i - j)
                    for i, j in zip(shape, chunk.shape, strict=False)
                )
                chunk = numpy.pad(chunk, pad)
            return encode(chunk)

    else:
        # strips
        def func(chunk: NDArray[Any] | None, /) -> bytes:
            if chunk is None:
                return b''
            chunk = numpy.ascontiguousarray(chunk, dtype)
            return encode(chunk)

    if maxworkers is None or maxworkers < 2 or numchunks < 2:
        for _ in range(numchunks):
            chunk = next(chunkiter)
            # assert chunk is None or isinstance(chunk, numpy.ndarray)
            yield func(chunk)
            del chunk
        return

    # because ThreadPoolExecutor.map is not collecting items lazily, reduce
    # memory overhead by processing chunks iterator maxchunks items at a time
    if buffersize is None:
        buffersize = TIFF.BUFFERSIZE * 2
    maxchunks = max(maxworkers, buffersize // chunksize)

    if numchunks <= maxchunks:

        def chunks() -> Iterator[NDArray[Any] | None]:
            for _ in range(numchunks):
                chunk = next(chunkiter)
                # assert chunk is None or isinstance(chunk, numpy.ndarray)
                yield chunk
                del chunk

        with ThreadPoolExecutor(maxworkers) as executor:
            yield from executor.map(func, chunks())
        return

    with ThreadPoolExecutor(maxworkers) as executor:
        count = 1
        chunk_list = []
        for _ in range(numchunks):
            chunk = next(chunkiter)
            if chunk is not None:
                count += 1
            # assert chunk is None or isinstance(chunk, numpy.ndarray)
            chunk_list.append(chunk)
            if count == maxchunks:
                yield from executor.map(func, chunk_list)
                chunk_list.clear()
                count = 0
        if chunk_list:
            yield from executor.map(func, chunk_list)




































def pyramidize_series(
    series: list[TiffPageSeries], /, *, reduced: bool = False
) -> None:
    """Pyramidize list of TiffPageSeries in-place.

    TiffPageSeries that are a subresolution of another TiffPageSeries are
    appended to the other's TiffPageSeries levels and removed from the list.
    Levels are to be ordered by size using the same downsampling factor.
    TiffPageSeries of subifds cannot be pyramid top levels.

    """
    samplingfactors = (2, 3, 4)
    i = 0
    while i < len(series):
        a = series[i]
        p = None
        j = i + 1
        if a.keyframe.is_subifd:
            # subifds cannot be pyramid top levels
            i += 1
            continue
        while j < len(series):
            b = series[j]
            if reduced and not b.keyframe.is_reduced:
                # pyramid levels must be reduced
                j += 1
                continue  # not a pyramid level
            if p is None:
                for f in samplingfactors:
                    if subresolution(a.levels[-1], b, p=f) == 1:
                        p = f
                        break  # not a pyramid level
                else:
                    j += 1
                    continue  # not a pyramid level
            elif subresolution(a.levels[-1], b, p=p) != 1:
                j += 1
                continue
            a.levels.append(b)
            del series[j]
        i += 1


def stack_pages(
    pages: Sequence[TiffPage | TiffFrame | None],
    /,
    *,
    tiled: TiledSequence | None = None,
    lock: threading.RLock | NullContext | None = None,
    device: str | None = None,
    maxworkers: int | None = None,
    out: OutputType = None,
    crop: tuple[slice, ...] | None = None,
    segment_filter: Sequence[int] | None = None,
    **kwargs: Any,
) -> NDArray[Any]:
    """Return vertically stacked image arrays from sequence of TIFF pages.

    Parameters:
        pages:
            TIFF pages or frames to stack.
        tiled:
            Organize pages in non-overlapping grid.
        lock:
            Reentrant lock to synchronize seeks and reads from file.
        device:
            If not *None*, return a ``torch.Tensor`` on the specified
            device instead of a NumPy array.
        maxworkers:
            Maximum number of threads to concurrently decode pages or segments.
            By default, use up to :py:attr:`_TIFF.MAXWORKERS` threads.
        out:
            Specifies how image array is returned.
            By default, a new NumPy array is created.
            If a *numpy.ndarray*, a writable array to which the images
            are copied.
            If a string or open file, the file used to create a memory-mapped
            array.
        crop:
            Spatial slices to apply to each decoded page.
            If provided, the output shape is
            ``(npages, *cropped_page_shape)`` and each worker decodes
            the page then copies only the cropped region to the output.
            Only affects the per-page spatial dimensions (not the page
            axis).
        segment_filter:
            Indices of segments (strips/tiles) to decode per page.
            Segments not in this list are skipped.  Combined with
            ``crop``, this avoids both decompression and memory for
            unneeded regions.
        **kwargs:
            Additional arguments passed to :py:meth:`TiffPage.asarray`.

    """
    npages = len(pages)
    if npages == 0:
        msg = 'no pages'
        raise ValueError(msg)

    if npages == 1 and crop is None and segment_filter is None:
        kwargs['maxworkers'] = maxworkers
        assert pages[0] is not None
        return pages[0].asarray(out=out, device=device, **kwargs)

    try:
        page0 = next(p.keyframe for p in pages if p is not None)
    except StopIteration:
        msg = 'pages are all None'
        raise ValueError(msg) from None
    assert page0 is not None

    # Compute output shape
    if crop is not None:
        # Determine cropped page shape
        cropped_shape = tuple(
            len(range(*s.indices(dim)))
            for s, dim in zip(crop, page0.shape, strict=False)
        )
        # Pad with remaining dimensions if crop has fewer dims
        if len(crop) < len(page0.shape):
            cropped_shape = cropped_shape + page0.shape[len(crop):]
        shape = (npages, *cropped_shape) if tiled is None else tiled.shape
    else:
        shape = (npages, *page0.shape) if tiled is None else tiled.shape
    dtype = page0.dtype
    assert dtype is not None
    out = create_output(out, shape, dtype)

    # TODO: benchmark and optimize this
    if maxworkers is None or maxworkers < 1:
        # auto-detect
        page_maxworkers = page0.maxworkers
        maxworkers = min(npages, TIFF.MAXWORKERS)
        if maxworkers == 1 or page_maxworkers < 1:
            maxworkers = page_maxworkers = 1
        elif npages < 3 or (
            page_maxworkers <= 2
            and page0.compression == 1
            and page0.fillorder == 1
            and page0.predictor == 1
        ):
            maxworkers = 1
        else:
            page_maxworkers = 1
    elif maxworkers == 1:
        maxworkers = page_maxworkers = 1
    elif npages > maxworkers or page0.maxworkers < 2:
        page_maxworkers = 1
    else:
        page_maxworkers = maxworkers
        maxworkers = 1

    kwargs['maxworkers'] = page_maxworkers

    fh = page0.parent.filehandle
    if lock is None:
        haslock = fh.has_lock
        if (not haslock and maxworkers > 1) or page_maxworkers > 1:
            fh.set_lock(True)
        lock = fh.lock
    else:
        haslock = True
    filecache = FileCache(size=max(4, maxworkers), lock=lock)

    if segment_filter is not None:
        # Segment-filtered path: decode only specified segments per page,
        # optionally crop the result.

        def func_filtered(
            page: TiffPage | TiffFrame | None,
            index: int,
            out: Any = out,
            filecache: FileCache = filecache,
            kwargs: dict[str, Any] = kwargs,
            keyframe: TiffPage = page0,
            seg_filter: Sequence[int] = segment_filter,
            crop_slices: tuple[slice, ...] | None = crop,
            /,
        ) -> None:
            if page is None:
                out[index].fill(0)
                return
            filecache.open(page.parent.filehandle)
            # Allocate full-page buffer for segment placement
            page_buf = create_output(None, keyframe.shaped, keyframe._dtype)

            def copy_segment(
                decoderesult: tuple[Any, ...],
                kf: Any = keyframe,
                buf: Any = page_buf,
            ) -> None:
                segment, (s, d, h, w, _), shape = decoderesult
                if segment is None:
                    buf[
                        s, d : d + shape[0],
                        h : h + shape[1], w : w + shape[2],
                    ] = kf.nodata
                else:
                    buf[
                        s, d : d + shape[0],
                        h : h + shape[1], w : w + shape[2],
                    ] = segment[
                        : kf.imagedepth - d,
                        : kf.imagelength - h,
                        : kf.imagewidth - w,
                    ]

            for _ in page.segments(
                func=copy_segment,
                lock=lock,
                sort=True,
                _fullsize=False,
                segment_filter=list(seg_filter),
                **kwargs,
            ):
                pass

            # Reshape to page shape and optionally crop
            try:
                page_data = page_buf.reshape(keyframe.shape)
            except ValueError:
                page_data = page_buf.squeeze()
            if crop_slices is not None:
                out[index] = page_data[crop_slices]
            else:
                out[index] = page_data
            filecache.close(page.parent.filehandle)

        if maxworkers < 2:
            for index, page in enumerate(pages):
                func_filtered(page, index)
        else:
            page0.decode  # noqa: B018 - init TiffPage.decode function
            with ThreadPoolExecutor(maxworkers) as executor:
                for _ in executor.map(func_filtered, pages, range(npages)):
                    pass

    elif crop is not None:
        # Crop-only path (no segment filter): decode full page, crop result

        def func_crop(
            page: TiffPage | TiffFrame | None,
            index: int,
            out: Any = out,
            filecache: FileCache = filecache,
            kwargs: dict[str, Any] = kwargs,
            crop_slices: tuple[slice, ...] = crop,
            /,
        ) -> None:
            if page is None:
                out[index].fill(0)
            else:
                filecache.open(page.parent.filehandle)
                page_data = page.asarray(lock=lock, **kwargs)
                out[index] = page_data[crop_slices]
                filecache.close(page.parent.filehandle)

        if maxworkers < 2:
            for index, page in enumerate(pages):
                func_crop(page, index)
        else:
            page0.decode  # noqa: B018 - init TiffPage.decode function
            with ThreadPoolExecutor(maxworkers) as executor:
                for _ in executor.map(func_crop, pages, range(npages)):
                    pass

    elif tiled is None:

        def func(
            page: TiffPage | TiffFrame | None,
            index: int,
            out: Any = out,
            filecache: FileCache = filecache,
            kwargs: dict[str, Any] = kwargs,
            /,
        ) -> None:
            # read, decode, and copy page data
            if page is None:
                out[index].fill(0)
            else:
                filecache.open(page.parent.filehandle)
                page.asarray(lock=lock, out=out[index], **kwargs)
                filecache.close(page.parent.filehandle)

        if maxworkers < 2:
            for index, page in enumerate(pages):
                func(page, index)
        else:
            page0.decode  # noqa: B018 - init TiffPage.decode function
            with ThreadPoolExecutor(maxworkers) as executor:
                for _ in executor.map(func, pages, range(npages)):
                    pass

    else:
        # TODO: not used or tested

        def func_tiled(
            page: TiffPage | TiffFrame | None,
            index: tuple[int | slice, ...],
            out: Any = out,
            filecache: FileCache = filecache,
            kwargs: dict[str, Any] = kwargs,
            /,
        ) -> None:
            # read, decode, and copy page data
            if page is None:
                out[index].fill(0)
            else:
                filecache.open(page.parent.filehandle)
                out[index] = page.asarray(lock=lock, **kwargs)
                filecache.close(page.parent.filehandle)

        if maxworkers < 2:
            for index_tiled, page in zip(tiled.slices(), pages, strict=True):
                func_tiled(page, index_tiled)
        else:
            page0.decode  # noqa: B018 - init TiffPage.decode function
            with ThreadPoolExecutor(maxworkers) as executor:
                for _ in executor.map(func_tiled, pages, tiled.slices()):
                    pass

    filecache.clear()
    if not haslock:
        fh.set_lock(False)

    if device is not None:
        from .gpu import numpy_to_tensor, parse_device

        dev = parse_device(device)
        if dev is not None:
            return numpy_to_tensor(out, dev)
    return out


def create_output(
    out: OutputType,
    /,
    shape: Sequence[int],
    dtype: DTypeLike | None,
    *,
    mode: Literal['r+', 'w+', 'r', 'c'] = 'w+',
    suffix: str | None = None,
    fillvalue: float | None = None,
) -> NDArray[Any] | numpy.memmap[Any, Any]:
    """Return NumPy array where data of shape and dtype can be copied.

    Parameters:
        out:
            Specifies kind of array of `shape` and `dtype` to return:

                `None`:
                    Return new array.
                `numpy.ndarray`:
                    Return view of existing array.
                `'memmap'` or `'memmap:tempdir'`:
                    Return memory-map to array stored in temporary binary file.
                `str` or open file:
                    Return memory-map to array stored in specified binary file.
        shape:
            Shape of array to return.
        dtype:
            Data type of array to return.
            If `out` is an existing array, `dtype` must be castable to its
            data type.
        mode:
            File mode to create memory-mapped array.
            The default is 'w+' to create new, or overwrite existing file for
            reading and writing.
        suffix:
            Suffix of `NamedTemporaryFile` if `out` is `'memmap'`.
            The default is '.memmap'.
        fillvalue:
            Value to initialize output array.
            By default, return uninitialized array.

    Returns:
        NumPy array or memory-mapped array of `shape` and `dtype`.

    Raises:
        ValueError:
            Existing array cannot be reshaped to `shape` or cast to `dtype`.

    """
    shape = tuple(shape)
    dtype = numpy.dtype(dtype)
    if out is None:
        if fillvalue is None:
            return numpy.empty(shape, dtype)
        if fillvalue:
            return numpy.full(shape, fillvalue, dtype)
        return numpy.zeros(shape, dtype)
    if isinstance(out, numpy.ndarray):
        if product(shape) != product(out.shape):
            msg = f'cannot reshape {shape} to {out.shape}'
            raise ValueError(msg)
        if not numpy.can_cast(dtype, out.dtype):
            msg = f'cannot cast {dtype} to {out.dtype}'
            raise ValueError(msg)
        out = out.reshape(shape)
        if fillvalue is not None:
            out.fill(fillvalue)
        return out
    if isinstance(out, str) and out[:6] == 'memmap':
        import tempfile

        tempdir = out[7:] if len(out) > 7 else None
        if suffix is None:
            suffix = '.memmap'
        with tempfile.NamedTemporaryFile(dir=tempdir, suffix=suffix) as fh:
            out = numpy.memmap(fh, shape=shape, dtype=dtype, mode=mode)
            if fillvalue is not None:
                out.fill(fillvalue)
            return out
    out = numpy.memmap(out, shape=shape, dtype=dtype, mode=mode)
    if fillvalue is not None:
        out.fill(fillvalue)
    return out
















































































def validate_jhove(
    filename: str,
    /,
    jhove: str | None = None,
    ignore: Collection[str] | None = None,
) -> None:
    """Validate TIFF file with ``jhove -m TIFF-hul``.

    JHOVE does not support the BigTIFF format, more than 50 IFDs, and
    many TIFF extensions.

    Parameters:
        filename:
            Name of TIFF file to validate.
        jhove:
            Path of jhove app. The default is 'jhove'.
        ignore:
            Jhove error message to ignore.

    Raises:
        ValueError:
            Jhove printed error message and did not contain one of strings
            in `ignore`.

    References:
        - `JHOVE TIFF-hul Module <http://jhove.sourceforge.net/tiff-hul.html>`_

    """
    import subprocess

    if ignore is None:
        ignore = {'More than 50 IFDs', 'Predictor value out of range'}
    if jhove is None:
        jhove = 'jhove'
    out = subprocess.check_output(  # # noqa: S603
        [jhove, filename, '-m', 'TIFF-hul']
    )
    if b'ErrorMessage: ' in out:
        for line_full in out.splitlines():
            line = line_full.strip()
            if line.startswith(b'ErrorMessage: '):
                error = line[14:].decode()
                for i in ignore:
                    if i in error:
                        break
                else:
                    raise ValueError(error)
                break


def tiffcomment(
    arg: str | os.PathLike[Any] | FileHandle | IO[bytes],
    /,
    comment: str | bytes | None = None,
    pageindex: int | None = None,
    tagcode: int | str | None = None,
) -> str | None:
    """Return or replace ImageDescription value in first page of TIFF file.

    Parameters:
        arg:
            Specifies TIFF file to open.
        comment:
            7-bit ASCII string or bytes to replace existing tag value.
            The existing value is zeroed.
        pageindex:
            Index of page which ImageDescription tag value to
            read or replace. The default is 0.
        tagcode:
            Code of tag which value to read or replace.
            The default is 270 (ImageDescription).

    Returns:
        None, if `comment` is specified. Else, the current value of the
        specified tag in the specified page.


    """
    if pageindex is None:
        pageindex = 0
    if tagcode is None:
        tagcode = 270
    mode: Any = None if comment is None else 'r+'
    with TiffFile(arg, mode=mode) as tif:
        page = tif.pages[pageindex]
        if not isinstance(page, TiffPage):
            msg = f'TiffPage {pageindex} not found'
            raise IndexError(msg)
        tag = page.tags.get(tagcode, None)
        if tag is None:
            msg = f'no {TIFF.TAGS[tagcode]} tag found'
            raise ValueError(msg)
        if comment is None:
            return tag.value
        tag.overwrite(comment)
        return None


def tiff2fsspec(
    filename: str | os.PathLike[Any],
    /,
    url: str,
    *,
    out: str | None = None,
    key: int | None = None,
    series: int | None = None,
    level: int | None = None,
    chunkmode: CHUNKMODE | int | str | None = None,
    fillvalue: float | None = None,
    zattrs: dict[str, Any] | None = None,
    squeeze: bool | None = None,
    groupname: str | None = None,
    version: int | None = None,
) -> None:
    """Write fsspec ReferenceFileSystem in JSON format for data in TIFF file.

    By default, the first series, including all levels, is exported.

    Parameters:
        filename:
            Name of TIFF file to reference.
        url:
            Remote location of TIFF file without file name(s).
        out:
            Name of output JSON file.
            The default is the `filename` with a '.json' extension.
        key, series, level, chunkmode, fillvalue, zattrs, squeeze:
            Passed to :py:meth:`TiffFile.aszarr`.
        groupname, version:
            Passed to :py:meth:`ZarrTiffStore.write_fsspec`.

    """
    if out is None:
        out = os.fspath(filename) + '.json'
    with TiffFile(filename) as tif:
        store: ZarrTiffStore
        with tif.aszarr(
            key=key,
            series=series,
            level=level,
            chunkmode=chunkmode,
            fillvalue=fillvalue,
            zattrs=zattrs,
            squeeze=squeeze,
        ) as store:
            store.write_fsspec(out, url, groupname=groupname, version=version)


def tiff2tiled(
    filename: str | os.PathLike[Any],
    /,
    out: str | os.PathLike[Any] | None = None,
    *,
    series: int | None = None,
    tile: Sequence[int] | None = (128, 128),
    compression: COMPRESSION | int | str | None = 'zstd',
    compressionargs: dict[str, Any] | None = None,
    predictor: PREDICTOR | int | str | bool | None = None,
    maxworkers: int | None = None,
    bigtiff: bool | None = None,
    progress: bool = False,
) -> None:
    """Write tiled OME-TIFF from image data in TIFF file.

    Read image data from a series in a TIFF file and write it as a tiled
    OME-TIFF file. Tiled files enable efficient spatial region-of-interest
    reads.

    Parameters:
        filename:
            Name of TIFF file to read.
        out:
            Name of output OME-TIFF file.
            The default is ``filename`` with ``'.tiled.ome.tif'`` appended.
        series:
            Index of series to convert. The default is 0.
        tile, compression, compressionargs, predictor, maxworkers:
            Passed to :py:meth:`TiffWriter.write`.
        bigtiff:
            Passed to :py:class:`TiffWriter`.
            By default, BigTIFF is used if the data exceeds 4 GB.
        progress:
            If *True*, print progress to stdout.

    Examples:
        Convert to tiled OME-TIFF with default settings:

        >>> # tiff2tiled('input.tif', 'output.ome.tif')  # doctest: +SKIP

    """
    import numpy

    filename = os.fspath(filename)
    if out is None:
        base, ext = os.path.splitext(filename)
        out = base + '.tiled.ome.tif'
    out = os.fspath(out)
    if series is None:
        series = 0

    with TiffFile(filename) as tif:
        src = tif.series[series]
        axes = src.axes
        shape = src.shape
        dtype = src.dtype
        keyframe = src.keyframe

        # determine photometric
        photometric: str | PHOTOMETRIC
        s_ax = axes.find('S')
        if keyframe.photometric == PHOTOMETRIC.RGB:
            photometric = 'rgb'
        elif s_ax >= 0 and shape[s_ax] in (3, 4):
            photometric = 'rgb'
        else:
            photometric = 'minisblack'

        if progress:
            nbytes = numpy.dtype(dtype).itemsize
            for s in shape:
                nbytes *= s
            print(
                f'tiff2tiled: {filename}\n'
                f'  shape={shape} axes={axes} dtype={dtype}'
                f' ({nbytes / 1e6:.1f} MB)'
            )

        # use memmap for contiguous series to avoid loading all into memory
        if src.dataoffset is not None:
            data = numpy.memmap(
                tif.filehandle.path,
                dtype=dtype,
                mode='r',
                offset=src.dataoffset,
                shape=shape,
            )
            if progress:
                print('  using memory-mapped read')
        else:
            if progress:
                print('  reading...', end='\r')
            data = src.asarray()
            if progress:
                print(f'  read {data.nbytes / 1e6:.1f} MB          ')

    # auto bigtiff
    if bigtiff is None:
        nbytes = numpy.dtype(dtype).itemsize
        for s in shape:
            nbytes *= s
        bigtiff = nbytes > 2**31

    write_opts: dict[str, Any] = {
        'photometric': photometric,
        'compression': compression,
        'predictor': predictor,
    }
    if tile is not None:
        write_opts['tile'] = tile
    if compressionargs is not None:
        write_opts['compressionargs'] = compressionargs
    if maxworkers is not None:
        write_opts['maxworkers'] = maxworkers

    metadata: dict[str, Any] = {'axes': axes}

    with TiffWriter(out, bigtiff=bigtiff, ome=True) as tw:
        tw.write(data, metadata=metadata, **write_opts)

    del data

    if progress:
        out_size = os.path.getsize(out) / 1e6
        print(
            f'  tile={tile}, compression={compression}\n'
            f'  output: {out} ({out_size:.1f} MB)'
        )


def tiff2pyramid(
    filename: str | os.PathLike[Any],
    /,
    out: str | os.PathLike[Any] | None = None,
    *,
    series: int | None = None,
    selection: dict[str, int | slice] | None = None,
    subresolutions: int | None = None,
    minsize: int = 256,
    resample: str | Callable[..., Any] = 'area',
    tile: Sequence[int] | None = (128, 128),
    compression: COMPRESSION | int | str | None = 'zstd',
    compressionargs: dict[str, Any] | None = None,
    predictor: PREDICTOR | int | str | bool | None = None,
    maxworkers: int | None = None,
    bigtiff: bool | None = None,
    split: bool = False,
    progress: bool = False,
) -> None:
    """Write pyramidal OME-TIFF from image data in TIFF file.

    Read image data from a series in a TIFF file, optionally select a
    subset, and write a tiled, multi-resolution pyramidal OME-TIFF file.
    Sub-resolution images are written to SubIFDs.
    Only the spatial dimensions (Y, X) are downsampled for pyramid levels.

    Parameters:
        filename:
            Name of TIFF file to read.
        out:
            Name of output OME-TIFF file.
            The default is ``filename`` with ``'.pyramid.ome.tif'`` appended.
            When *split* is *True*, ``out`` is used as a template: the
            base name is appended with ``'_HxW'`` spatial dimensions.
        series:
            Index of series to convert. The default is 0.
        selection:
            Subset of data to include in the output.
            A dict mapping axis codes to integer indices or slices.
            For example, ``{'T': 5}`` includes only timepoint 5, and
            ``{'Z': slice(None, None, 2)}`` takes every other Z slice.
            Spatial axes (Y, X, S) in the dict are ignored.
        subresolutions:
            Number of sub-resolution levels to write.
            By default, levels are added until the smallest spatial
            dimension is less than *minsize*.
        minsize:
            Minimum spatial dimension for the smallest pyramid level.
            Ignored if *subresolutions* is specified. The default is 256.
        resample:
            Resampling method for generating sub-resolution levels:

            ``'area'``:
                Area averaging (default). Requires ``scikit-image``.
            ``'nearest'``:
                Nearest-neighbor. Suitable for label/mask images.
            ``'mean'``:
                Block mean. Fast, no extra dependencies.
            A callable:
                ``f(data, factor) -> downsampled`` where *factor*
                is the integer downsampling factor. *data* has shape
                ``(Y, X)`` or ``(Y, X, S)``.
        tile, compression, compressionargs, predictor, maxworkers:
            Passed to :py:meth:`TiffWriter.write`.
        bigtiff:
            Passed to :py:class:`TiffWriter`.
            By default, BigTIFF is used if the data exceeds 4 GB.
        split:
            If *True*, write each resolution level as a separate file.
            File names include spatial dimensions, for example,
            ``'out_512x512.ome.tif'``, ``'out_256x256.ome.tif'``.
        progress:
            If *True*, print progress to stdout.

    Examples:
        Convert to pyramidal OME-TIFF with default settings:

        >>> # tiff2pyramid('input.tif', 'output.ome.tif')  # doctest: +SKIP

        Use nearest-neighbor resampling for a label image:

        >>> # tiff2pyramid('labels.tif', resample='nearest')  # doctest: +SKIP

        Select a subset of Z and T before pyramidizing:

        >>> # tiff2pyramid('volume.ome.tif',
        ... #     selection={'T': slice(0, 5),
        ... #                'Z': slice(None, None, 2)})  # doctest: +SKIP

        Write each resolution level as a separate file:

        >>> # tiff2pyramid('input.tif', split=True)  # doctest: +SKIP

    """
    import numpy

    filename = os.fspath(filename)
    if out is None:
        base, ext = os.path.splitext(filename)
        out = base + '.pyramid.ome.tif'
    out = os.fspath(out)
    if series is None:
        series = 0

    # resolve resampling function
    resample_func: Callable[..., Any]
    if callable(resample):
        resample_func = resample
    elif resample == 'area':
        try:
            from skimage.transform import downscale_local_mean
        except ImportError:
            resample = 'mean'
            resample_func = _resample_mean
        else:
            def resample_func(data: Any, factor: int) -> Any:
                factors = tuple(
                    factor if i < 2 else 1
                    for i in range(data.ndim)
                )
                return downscale_local_mean(data, factors).astype(data.dtype)
    if resample == 'nearest':
        def resample_func(data: Any, factor: int) -> Any:
            idx: tuple[Any, ...] = (slice(None, None, factor),) * min(
                2, data.ndim
            )
            if data.ndim > 2:
                idx = idx + (slice(None),)
            return numpy.ascontiguousarray(data[idx])
    elif resample == 'mean':
        resample_func = _resample_mean

    with TiffFile(filename) as tif:
        src = tif.series[series]
        axes = src.axes
        dtype = src.dtype
        keyframe = src.keyframe

        # identify spatial dimensions
        y_ax = axes.find('Y')
        x_ax = axes.find('X')
        if y_ax < 0 or x_ax < 0:
            msg = f'cannot identify spatial axes in {axes!r}'
            raise ValueError(msg)

        # determine photometric
        photometric: str | PHOTOMETRIC
        s_ax = axes.find('S')
        if keyframe.photometric == PHOTOMETRIC.RGB:
            photometric = 'rgb'
        elif s_ax >= 0 and src.shape[s_ax] in (3, 4):
            photometric = 'rgb'
        else:
            photometric = 'minisblack'

        # read source data, applying selection
        if progress:
            print(
                f'tiff2pyramid: {filename}\n'
                f'  shape={src.shape} axes={axes} dtype={dtype}'
            )
            print('  reading...', end='\r')
        base_data = src.asarray()
        if progress:
            mb = base_data.nbytes / 1e6
            print(f'  read {mb:.1f} MB          ')

        # apply selection (subset non-spatial axes)
        if selection:
            sel_idx: list[int | slice] = [slice(None)] * len(axes)
            collapsed: set[int] = set()
            for ax_code, sel_val in selection.items():
                ax_code_u = ax_code.upper()
                if ax_code_u in ('Y', 'X', 'S'):
                    continue
                ax_pos = axes.find(ax_code_u)
                if ax_pos < 0:
                    msg = (
                        f'selection axis {ax_code!r} '
                        f'not found in axes {axes!r}'
                    )
                    raise ValueError(msg)
                sel_idx[ax_pos] = sel_val
                if isinstance(sel_val, int):
                    collapsed.add(ax_pos)
            base_data = numpy.ascontiguousarray(base_data[tuple(sel_idx)])
            # update axes and axis positions for collapsed dimensions
            if collapsed:
                axes = ''.join(
                    a for i, a in enumerate(axes) if i not in collapsed
                )
                y_ax = axes.find('Y')
                x_ax = axes.find('X')
            if progress:
                mb = base_data.nbytes / 1e6
                print(f'  selected shape={base_data.shape} ({mb:.1f} MB)')

        height = base_data.shape[y_ax]
        width = base_data.shape[x_ax]

        # determine number of sub-resolution levels
        if subresolutions is None:
            subresolutions = 0
            h, w = height, width
            while h >= minsize * 2 and w >= minsize * 2:
                h //= 2
                w //= 2
                subresolutions += 1
        subresolutions = max(0, subresolutions)
        num_levels = subresolutions + 1

        # auto bigtiff
        from .utils import product as _product
        if bigtiff is None:
            element_size = numpy.dtype(dtype).itemsize
            total_elements = _product(base_data.shape) * num_levels
            bigtiff = total_elements * element_size > 2**31

        if progress:
            print(
                f'  {num_levels} levels, tile={tile}, '
                f'compression={compression}'
            )

        write_opts: dict[str, Any] = {
            'photometric': photometric,
            'compression': compression,
            'predictor': predictor,
        }
        if tile is not None:
            write_opts['tile'] = tile
        if compressionargs is not None:
            write_opts['compressionargs'] = compressionargs
        if maxworkers is not None:
            write_opts['maxworkers'] = maxworkers

        # build OME metadata
        metadata: dict[str, Any] = {'axes': axes}

        # prepare all level data: list of (data, height, width)
        levels_data: list[tuple[Any, int, int]] = [
            (base_data, height, width)
        ]
        for lvl in range(1, num_levels):
            factor = 2 ** lvl
            down = _downsample_spatial(
                base_data, factor, y_ax, x_ax, resample_func
            )
            levels_data.append(
                (down, down.shape[y_ax], down.shape[x_ax])
            )

        if split:
            # write each level as a separate file
            out_base, out_ext = os.path.splitext(out)
            if out_base.lower().endswith('.ome'):
                out_base = out_base[:-4]
                out_ext = '.ome' + out_ext
            for lvl, (data, lvl_h, lvl_w) in enumerate(levels_data):
                lvl_path = f'{out_base}_{lvl_h}x{lvl_w}{out_ext}'
                lvl_bigtiff = bigtiff
                if lvl_bigtiff is None:
                    lvl_bigtiff = data.nbytes > 2**31
                with TiffWriter(
                    lvl_path, bigtiff=lvl_bigtiff, ome=True
                ) as tw:
                    tw.write(data, metadata=metadata, **write_opts)
                if progress:
                    fsz = os.path.getsize(lvl_path) / 1e6
                    print(
                        f'  level {lvl} ({lvl_h}x{lvl_w}) -> '
                        f'{lvl_path} ({fsz:.1f} MB)'
                    )
                del data
        else:
            # write single pyramidal file with SubIFDs
            with TiffWriter(out, bigtiff=bigtiff, ome=True) as tw:
                tw.write(
                    levels_data[0][0],
                    subifds=subresolutions,
                    metadata=metadata,
                    **write_opts,
                )
                if progress:
                    print(f'  base level written ({height}x{width})')

                for lvl in range(1, num_levels):
                    data, lvl_h, lvl_w = levels_data[lvl]
                    tw.write(
                        data,
                        subfiletype=1,
                        **write_opts,
                    )
                    if progress:
                        print(
                            f'  level {lvl} ({2 ** lvl}x) written '
                            f'({lvl_h}x{lvl_w})'
                        )
                    del data

            if progress:
                out_size = os.path.getsize(out) / 1e6
                print(f'  output: {out} ({out_size:.1f} MB)')

        del levels_data


def _downsample_spatial(
    data: Any,
    factor: int,
    y_ax: int,
    x_ax: int,
    resample_func: Callable[..., Any],
) -> Any:
    """Downsample a multi-dimensional array in Y and X dimensions only."""
    import numpy

    ndim = data.ndim
    if ndim == 2:
        # simple YX
        return resample_func(data, factor)
    if ndim == 3 and y_ax == 0 and x_ax == 1:
        # YXS
        return resample_func(data, factor)

    # general case: move Y,X to last positions, downsample per page, move back
    # build list of spatial and frame axes
    spatial = {y_ax, x_ax}
    s_ax = -1
    # check for samples axis right after X
    if x_ax + 1 < ndim and x_ax + 1 not in (y_ax,):
        # could be S axis
        s_ax = x_ax + 1
        spatial.add(s_ax)

    frame_axes = [i for i in range(ndim) if i not in spatial]
    frame_shape = tuple(data.shape[i] for i in frame_axes)

    # flatten frame axes, keep spatial at end
    if s_ax >= 0:
        spatial_order = (y_ax, x_ax, s_ax)
    else:
        spatial_order = (y_ax, x_ax)
    perm = (*frame_axes, *spatial_order)
    transposed = numpy.transpose(data, perm)
    n_spatial = len(spatial_order)
    spatial_shape = transposed.shape[len(frame_axes):]
    nframes = 1
    for s in frame_shape:
        nframes *= s

    flat = transposed.reshape(nframes, *spatial_shape)

    # downsample each frame
    down_first = resample_func(flat[0], factor)
    result = numpy.empty(
        (nframes, *down_first.shape), dtype=data.dtype
    )
    result[0] = down_first
    for i in range(1, nframes):
        result[i] = resample_func(flat[i], factor)

    # reshape back to frame dims + downsampled spatial
    result = result.reshape(*frame_shape, *down_first.shape)

    # transpose back to original axis order
    inv_perm = [0] * ndim
    for new_pos, old_pos in enumerate(perm):
        inv_perm[old_pos] = new_pos
    result = numpy.transpose(result, inv_perm)
    return numpy.ascontiguousarray(result)


def _resample_mean(data: Any, factor: int) -> Any:
    """Downsample by block mean without external dependencies."""
    import numpy

    h, w = data.shape[0], data.shape[1]
    new_h = h // factor
    new_w = w // factor
    cropped = data[: new_h * factor, : new_w * factor]
    if data.ndim == 2:
        blocks = cropped.reshape(new_h, factor, new_w, factor)
        return blocks.mean(axis=(1, 3)).astype(data.dtype)
    else:
        # (Y, X, S)
        s = data.shape[2]
        blocks = cropped.reshape(new_h, factor, new_w, factor, s)
        return blocks.mean(axis=(1, 3)).astype(data.dtype)


def lsm2bin(
    lsmfile: str,
    /,
    binfile: str | None = None,
    *,
    tile: tuple[int, int] | None = None,
    verbose: bool = True,
) -> None:
    """Convert [MP]TZCYX LSM file to series of BIN files.

    One BIN file containing 'ZCYX' data is created for each position, time,
    and tile. The position, time, and tile indices are encoded at the end
    of the filenames.

    Parameters:
        lsmfile:
            Name of LSM file to convert.
        binfile:
            Common name of output BIN files.
            The default is the name of the LSM file without extension.
        tile:
            Y and X dimension sizes of BIN files.
            The default is (256, 256).
        verbose:
            Print status of conversion.

    """
    prints: Any = print if verbose else nullfunc

    if tile is None:
        tile = (256, 256)

    if binfile is None:
        binfile = lsmfile
    elif binfile.lower() == 'none':
        binfile = None
    if binfile:
        binfile += '_(z%ic%iy%ix%i)_m%%ip%%it%%03iy%%ix%%i.bin'

    prints('\nOpening LSM file... ', end='', flush=True)
    timer = Timer()

    with TiffFile(lsmfile) as lsm:
        if not lsm.is_lsm:
            prints('\n', lsm, flush=True)
            msg = 'not a LSM file'
            raise ValueError(msg)
        series = lsm.series[0]  # first series contains the image
        shape = series.get_shape(squeeze=False)
        axes = series.get_axes(squeeze=False)
        dtype = series.dtype
        size = product(shape) * dtype.itemsize

        prints(timer)
        # verbose(lsm, flush=True)
        prints(
            indent(
                'Image',
                f'axes:  {axes}',
                f'shape: {shape}',
                f'dtype: {dtype}',
                f'size:  {size}',
            ),
            flush=True,
        )
        if axes == 'CYX':
            shape = (1, 1, *shape)
        elif axes == 'ZCYX':
            shape = (1, *shape)
        elif axes == 'MPCYX':
            shape = (*shape[:2], 1, 1, *shape[2:])
        elif axes == 'MPZCYX':
            shape = (*shape[:2], 1, *shape[2:])
        elif not axes.endswith('TZCYX'):
            msg = 'not a *TZCYX LSM file'
            raise ValueError(msg)

        prints('Copying image from LSM to BIN files', end='', flush=True)
        timer.start()
        tiles = shape[-2] // tile[-2], shape[-1] // tile[-1]
        if binfile:
            binfile = binfile % (shape[-4], shape[-3], tile[0], tile[1])
        shape = (1,) * (7 - len(shape)) + shape
        # cache for ZCYX stacks and output files
        data = numpy.empty(shape[3:], dtype=dtype)
        out = numpy.empty(
            (shape[-4], shape[-3], tile[0], tile[1]), dtype=dtype
        )
        # iterate over Tiff pages containing data
        pages = iter(series.pages)
        for m in range(shape[0]):  # mosaic axis
            for p in range(shape[1]):  # position axis
                for t in range(shape[2]):  # time axis
                    for z in range(shape[3]):  # z slices
                        page = next(pages)
                        assert page is not None
                        data[z] = page.asarray()
                    for y in range(tiles[0]):  # tile y
                        for x in range(tiles[1]):  # tile x
                            out[:] = data[
                                ...,
                                y * tile[0] : (y + 1) * tile[0],
                                x * tile[1] : (x + 1) * tile[1],
                            ]
                            if binfile:
                                out.tofile(binfile % (m, p, t, y, x))
                            prints('.', end='', flush=True)
        prints(timer, flush=True)


def imshow(
    data: NDArray[Any],
    /,
    *,
    photometric: PHOTOMETRIC | int | str | None = None,
    planarconfig: PLANARCONFIG | int | str | None = None,
    bitspersample: int | None = None,
    nodata: float = 0,
    interpolation: str | None = None,
    cmap: Any | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
    figure: Any = None,
    subplot: Any = None,
    title: str | bytes | None = None,
    window_title: str | None = None,
    dpi: int = 96,
    maxdim: int | None = None,
    background: tuple[float, float, float] | str | None = None,
    show: bool = False,
    **kwargs: Any,
) -> tuple[Any, Any, Any]:
    """Plot n-dimensional images with `matplotlib.pyplot`.

    Parameters:
        data:
            Image array to display.
        photometric:
            Color space of image.
        planarconfig:
            How components of each pixel are stored.
        bitspersample:
            Number of bits per channel in integer RGB images.
        interpolation:
            Image interpolation method used in `matplotlib.imshow`.
           The default is 'nearest' for image dimensions > 512,
           else 'bilinear'.
        cmap:
            Colormap mapping non-RGBA scalar data to colors.
            See `matplotlib.colors.Colormap`.
        vmin:
            Minimum of data range covered by colormap.
            By default, the complete range of the data is covered.
        vmax:
            Maximum of data range covered by colormap.
            By default, the complete range of the data is covered.
        figure:
            Matplotlib figure to use for plotting.
            See `matplotlib.figure.Figure`.
        subplot:
            A `matplotlib.pyplot.subplot` axis.
        title:
            Subplot title.
        window_title:
            Window title.
        dpi:
            Resolution of figure.
        maxdim:
            Maximum image width and length.
        background:
            Background color.
        show:
            Display figure.
        **kwargs:
            Additional arguments passed to :py:func:`matplotlib.pyplot.imshow`.

    Returns:
        Matplotlib figure, subplot, and plot axis.

    """
    # TODO: rewrite detection of isrgb, iscontig
    # TODO: use planarconfig
    if photometric is None:
        photometric = 'RGB'
    if maxdim is None:
        maxdim = 2**16
    isrgb = photometric in {'RGB', 'YCBCR'}  # 'PALETTE', 'YCBCR'

    if data.dtype == 'float16':
        data = data.astype(numpy.float32)

    if data.dtype.kind == 'b':
        isrgb = False

    if isrgb and not (
        data.shape[-1] in {3, 4}
        or (data.ndim > 2 and data.shape[-3] in {3, 4})
    ):
        isrgb = False
        photometric = 'MINISBLACK'

    data = data.squeeze()
    if photometric in {
        None,
        'MINISWHITE',
        'MINISBLACK',
        'CFA',
        'MASK',
        'PALETTE',
        'LOGL',
        'LOGLUV',
        'DEPTH_MAP',
        'SEMANTIC_MASK',
    }:
        data = reshape_nd(data, 2)
    else:
        data = reshape_nd(data, 3)

    dims = data.ndim
    if dims < 2:
        msg = 'not an image'
        raise ValueError(msg)
    if dims == 2:
        dims = 0
        isrgb = False
    else:
        if isrgb and data.shape[-3] in {3, 4} and data.shape[-1] not in {3, 4}:
            data = numpy.swapaxes(data, -3, -2)
            data = numpy.swapaxes(data, -2, -1)
        elif not isrgb and (
            data.shape[-1] < data.shape[-2] // 8
            and data.shape[-1] < data.shape[-3] // 8
        ):
            data = numpy.swapaxes(data, -3, -1)
            data = numpy.swapaxes(data, -2, -1)
        isrgb = isrgb and data.shape[-1] in {3, 4}
        dims -= 3 if isrgb else 2

    if interpolation is None:
        threshold = 512
    elif isinstance(interpolation, int):
        threshold = interpolation
    else:
        threshold = 0

    if isrgb:
        data = data[..., :maxdim, :maxdim, :maxdim]
        if threshold:
            if data.shape[-2] > threshold or data.shape[-3] > threshold:
                interpolation = 'bilinear'
            else:
                interpolation = 'nearest'
    else:
        data = data[..., :maxdim, :maxdim]
        if threshold:
            if data.shape[-1] > threshold or data.shape[-2] > threshold:
                interpolation = 'bilinear'
            else:
                interpolation = 'nearest'

    if photometric == 'PALETTE' and isrgb:
        try:
            datamax = numpy.max(data)
        except ValueError:
            datamax = 1
        if datamax > 255:
            data = data >> 8  # possible precision loss
        data = data.astype('B', copy=False)
    elif data.dtype.kind in 'ui':
        if not (isrgb and data.dtype.itemsize <= 1) or bitspersample is None:
            try:
                bitspersample = math.ceil(math.log2(data.max()))
            except Exception:
                bitspersample = data.dtype.itemsize * 8
        elif not isinstance(bitspersample, (int, numpy.integer)):
            # bitspersample can be tuple, such as (5, 6, 5)
            bitspersample = data.dtype.itemsize * 8
        assert bitspersample is not None
        datamax = 2**bitspersample
        if isrgb:
            if bitspersample < 8:
                data = data << (8 - bitspersample)
            elif bitspersample > 8:
                data = data >> (bitspersample - 8)  # precision loss
            data = data.astype('B', copy=False)
    elif data.dtype.kind == 'f':
        if nodata:
            data = data.copy()
            data[data == nodata] = numpy.nan
        try:
            datamax = numpy.nanmax(data)
        except ValueError:
            datamax = 1
        if isrgb and datamax > 1.0:
            if data.dtype.char == 'd':
                data = data.astype('f')
                data /= datamax
            else:
                data = data / datamax
    elif data.dtype.kind == 'b':
        datamax = 1
    elif data.dtype.kind == 'c':
        data = numpy.absolute(data)
        try:
            datamax = numpy.nanmax(data)
        except ValueError:
            datamax = 1

    if isrgb:
        vmin = 0
    else:
        if vmax is None:
            vmax = datamax
        if vmin is None:
            if data.dtype.kind == 'i':
                imin = numpy.iinfo(data.dtype).min
                try:
                    vmin = numpy.min(data)
                except ValueError:
                    vmin = -1
                if vmin == imin:
                    vmin = numpy.min(data[data > imin])
            elif data.dtype.kind == 'f':
                fmin = float(numpy.finfo(data.dtype).min)
                try:
                    vmin = numpy.nanmin(data)
                except ValueError:
                    vmin = 0.0
                if vmin == fmin:
                    vmin = numpy.nanmin(data[data > fmin])
            else:
                vmin = 0

    from matplotlib import pyplot
    from matplotlib.widgets import Slider

    if figure is None:
        pyplot.rc('font', family='sans-serif', weight='normal', size=8)
        figure = pyplot.figure(
            dpi=dpi,
            figsize=(10.3, 6.3),
            frameon=True,
            facecolor='1.0',
            edgecolor='w',
        )
        if window_title is not None:
            with contextlib.suppress(Exception):
                figure.canvas.manager.window.title(window_title)
        size = len(title.splitlines()) if title else 1
        pyplot.subplots_adjust(
            bottom=0.03 * (dims + 2),
            top=0.98 - size * 0.03,
            left=0.1,
            right=0.95,
            hspace=0.05,
            wspace=0.0,
        )
    if subplot is None:
        subplot = 111
    subplot = pyplot.subplot(subplot)
    if background is None:
        background = (0.382, 0.382, 0.382)
    subplot.set_facecolor(background)

    if title:
        if isinstance(title, bytes):
            title = title.decode('Windows-1252')
        pyplot.title(title, size=11)

    if cmap is None:
        if data.dtype.char == '?':
            cmap = 'gray'
        elif data.dtype.kind in 'buf' or vmin == 0:
            cmap = 'viridis'
        else:
            cmap = 'coolwarm'
        if photometric == 'MINISWHITE':
            cmap += '_r'

    image = pyplot.imshow(
        numpy.atleast_2d(data[(0,) * dims].squeeze()),
        vmin=vmin,
        vmax=vmax,
        cmap=cmap,
        interpolation=interpolation,
        **kwargs,
    )

    if not isrgb:
        pyplot.colorbar()  # panchor=(0.55, 0.5), fraction=0.05

    def format_coord(x: float, y: float, /) -> str:
        # callback function to format coordinate display in toolbar
        x = int(x + 0.5)
        y = int(y + 0.5)
        try:
            if dims:
                return f'{curaxdat[1][y, x]} @ {current} [{y:4}, {x:4}]'
            return f'{data[y, x]} @ [{y:4}, {x:4}]'
        except IndexError:
            return ''

    def none(event: Any) -> str:
        return ''

    subplot.format_coord = format_coord
    image.get_cursor_data = none  # type: ignore[assignment, method-assign]
    image.format_cursor_data = none  # type: ignore[assignment, method-assign]

    if dims:
        current = list((0,) * dims)
        curaxdat = [0, data[tuple(current)].squeeze()]
        sliders = [
            Slider(
                ax=pyplot.axes((0.125, 0.03 * (axis + 1), 0.725, 0.025)),
                label=f'Dimension {axis}',
                valmin=0,
                valmax=data.shape[axis] - 1,
                valinit=0,
                valfmt=f'%.0f [{data.shape[axis]}]',
            )
            for axis in range(dims)
        ]
        for slider in sliders:
            slider.drawon = False

        def set_image(current, sliders=sliders, data=data):
            # change image and redraw canvas
            curaxdat[1] = data[tuple(current)].squeeze()
            image.set_data(curaxdat[1])
            for ctrl, index in zip(sliders, current, strict=True):
                ctrl.eventson = False
                ctrl.set_val(index)
                ctrl.eventson = True
            figure.canvas.draw()

        def on_changed(index, axis, data=data, current=current):
            # callback function for slider change event
            index = round(index)
            curaxdat[0] = axis
            if index == current[axis]:
                return
            if index >= data.shape[axis]:
                index = 0
            elif index < 0:
                index = data.shape[axis] - 1
            current[axis] = index
            set_image(current)

        def on_keypressed(event, data=data, current=current):
            # callback function for key press event
            key = event.key
            axis = curaxdat[0]
            if str(key) in '0123456789':
                on_changed(key, axis)
            elif key == 'right':
                on_changed(current[axis] + 1, axis)
            elif key == 'left':
                on_changed(current[axis] - 1, axis)
            elif key == 'up':
                curaxdat[0] = 0 if axis == len(data.shape) - 1 else axis + 1
            elif key == 'down':
                curaxdat[0] = len(data.shape) - 1 if axis == 0 else axis - 1
            elif key == 'end':
                on_changed(data.shape[axis] - 1, axis)
            elif key == 'home':
                on_changed(0, axis)

        figure.canvas.mpl_connect('key_press_event', on_keypressed)
        for axis, ctrl in enumerate(sliders):
            ctrl.on_changed(
                lambda k, a=axis: on_changed(k, a)  # type: ignore[misc]
            )

    if show:
        pyplot.show()

    return figure, subplot, image


def askopenfilename(**kwargs: Any) -> str:
    """Return file name(s) from Tkinter's file open dialog."""
    from tkinter import Tk, filedialog

    root = Tk()
    root.withdraw()
    root.update()
    print(kwargs)
    filenames = filedialog.askopenfilename(**kwargs)
    root.destroy()
    return filenames


def main() -> int:
    """Tifffile command line usage main function."""
    import optparse  # TODO: use argparse

    logger().setLevel(logging.INFO)

    parser = optparse.OptionParser(
        usage='usage: %prog [options] path',
        description='Display image and metadata in TIFF file.',
        version=f'%prog {__version__}',
        prog='tifffile',
    )
    opt = parser.add_option
    opt(
        '-p',
        '--page',
        dest='page',
        type='int',
        default=-1,
        help='display single page',
    )
    opt(
        '-s',
        '--series',
        dest='series',
        type='int',
        default=-1,
        help='display select series',
    )
    opt(
        '-l',
        '--level',
        dest='level',
        type='int',
        default=-1,
        help='display pyramid level of series',
    )
    opt(
        '--nomultifile',
        dest='nomultifile',
        action='store_true',
        default=False,
        help='do not read OME series from multiple files',
    )
    opt(
        '--maxplots',
        dest='maxplots',
        type='int',
        default=10,
        help='maximum number of plot windows',
    )
    opt(
        '--interpol',
        dest='interpol',
        metavar='INTERPOL',
        default=None,
        help='image interpolation method',
    )
    opt('--dpi', dest='dpi', type='int', default=96, help='plot resolution')
    opt(
        '--vmin',
        dest='vmin',
        type='int',
        default=None,
        help='minimum value for colormapping',
    )
    opt(
        '--vmax',
        dest='vmax',
        type='int',
        default=None,
        help='maximum value for colormapping',
    )
    opt(
        '--cmap',
        dest='cmap',
        type='str',
        default=None,
        help='colormap name used to map data to colors',
    )
    opt(
        '--maxworkers',
        dest='maxworkers',
        type='int',
        default=0,
        help='maximum number of threads',
    )
    opt(
        '--debug',
        dest='debug',
        action='store_true',
        default=False,
        help='raise exception on failures',
    )
    opt('-v', '--detail', dest='detail', type='int', default=2)
    opt('-q', '--quiet', dest='quiet', action='store_true')

    settings, path_list = parser.parse_args()
    path = ' '.join(path_list)

    if not path:
        path = askopenfilename(
            title='Select a TIFF file', filetypes=TIFF.FILEOPEN_FILTER
        )
        if not path:
            parser.error('No file specified')

    if any(i in path for i in '?*'):
        path_list = glob.glob(path)
        if not path_list:
            print('No files match the pattern')
            return 0
        # TODO: handle image sequences
        path = path_list[0]

    if not settings.quiet:
        print('\nReading TIFF header:', end=' ', flush=True)
    timer = Timer()
    try:
        tif = TiffFile(path, _multifile=not settings.nomultifile)
    except Exception as exc:
        if settings.debug:
            raise
        print(f'\n\n{exc.__class__.__name__}: {exc}')
        return 0

    if not settings.quiet:
        print(timer)

    if tif.is_ome:
        settings.norgb = True

    images: list[tuple[Any, Any, Any]] = []
    if settings.maxplots > 0:
        if not settings.quiet:
            print('Reading image data:', end=' ', flush=True)

        def notnone(x: Any, /) -> Any:
            return next(i for i in x if i is not None)

        timer.start()
        try:
            if settings.page >= 0:
                images = [
                    (
                        tif.asarray(
                            key=settings.page, maxworkers=settings.maxworkers
                        ),
                        tif.pages[settings.page],
                        None,
                    )
                ]
            elif settings.series >= 0:
                series = tif.series[settings.series]
                if settings.level >= 0:
                    level = settings.level
                elif series.is_pyramidal and product(series.shape) > 2**32:
                    level = -1
                    for r in series.levels:
                        level += 1
                        if product(r.shape) < 2**32:
                            break
                else:
                    level = 0
                images = [
                    (
                        tif.asarray(
                            series=settings.series,
                            level=level,
                            maxworkers=settings.maxworkers,
                        ),
                        notnone(tif.series[settings.series]._pages),
                        tif.series[settings.series],
                    )
                ]
            else:
                for i, s in enumerate(tif.series[: settings.maxplots]):
                    if settings.level < 0:
                        level = -1
                        for r in s.levels:
                            level += 1
                            if product(r.shape) < 2**31:
                                break
                    else:
                        level = settings.level
                    try:
                        images.append(
                            (
                                tif.asarray(
                                    series=i,
                                    level=level,
                                    maxworkers=settings.maxworkers,
                                ),
                                notnone(s._pages),
                                tif.series[i],
                            )
                        )
                    except Exception as exc:
                        images.append((None, notnone(s.pages), None))
                        if settings.debug:
                            raise
                        print(f'\nSeries {i} raised {exc!r:.128}... ', end='')
        except Exception as exc:
            if settings.debug:
                raise
            print(f'{exc.__class__.__name__}: {exc}')

        if not settings.quiet:
            print(timer)

    if not settings.quiet:
        print('Generating report:', end='   ', flush=True)
        timer.start()
        try:
            width = os.get_terminal_size()[0]
        except Exception:
            width = 80
        info = tif._str(detail=int(settings.detail), width=width - 1)
        print(timer)
        print()
        print(info)
        print()

    if images and settings.maxplots > 0:
        try:
            from matplotlib import pyplot
        except ImportError as exc:
            logger().warning(f'<tifffile.main> raised {exc!r:.128}')
        else:
            for img, page, series in images:
                if img is None:
                    continue
                keyframe = page.keyframe
                vmin, vmax = settings.vmin, settings.vmax
                if keyframe.nodata:
                    try:
                        if img.dtype.kind == 'f':
                            img[img == keyframe.nodata] = numpy.nan
                            vmin = numpy.nanmin(img)
                        else:
                            vmin = numpy.min(img[img > keyframe.nodata])
                    except ValueError:
                        pass
                if tif.is_stk:
                    try:
                        vmin = tif.stk_metadata[
                            'MinScale'  # type: ignore[index]
                        ]
                        vmax = tif.stk_metadata[
                            'MaxScale'  # type: ignore[index]
                        ]
                    except KeyError:
                        pass
                    else:
                        if vmax <= vmin:
                            vmin, vmax = settings.vmin, settings.vmax
                if series:
                    title = f'{tif}\n{page}\n{series}'
                    window_title = f'{tif.filename} series {series.index}'
                else:
                    title = f'{tif}\n{page}'
                    window_title = f'{tif.filename} page {page.index}'
                photometric = 'MINISBLACK'
                if keyframe.photometric != 3:
                    photometric = PHOTOMETRIC(keyframe.photometric).name
                imshow(
                    img,
                    title=title,
                    window_title=window_title,
                    vmin=vmin,
                    vmax=vmax,
                    cmap=settings.cmap,
                    bitspersample=keyframe.bitspersample,
                    nodata=keyframe.nodata,
                    photometric=photometric,
                    interpolation=settings.interpol,
                    dpi=settings.dpi,
                    show=False,
                )
            pyplot.show()

    tif.close()
    return 0






# aliases and deprecated
TiffReader = TiffFile

if TYPE_CHECKING:
    from .zarr import ZarrFileSequenceStore, ZarrStore, ZarrTiffStore

if __name__ == '__main__':
    sys.exit(main())

# mypy: allow-untyped-defs, allow-untyped-calls
# mypy: disable-error-code="no-any-return, redundant-expr, unreachable"
