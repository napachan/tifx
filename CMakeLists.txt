cmake_minimum_required(VERSION 3.17)
project(tifffile_ext LANGUAGES CXX)

# C++17 required
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find Python and nanobind
find_package(Python 3.11
    REQUIRED COMPONENTS Interpreter Development.Module)

execute_process(
    COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
    OUTPUT_STRIP_TRAILING_WHITESPACE
    OUTPUT_VARIABLE nanobind_ROOT
    RESULT_VARIABLE _nanobind_result
)
if(NOT _nanobind_result EQUAL 0)
    message(FATAL_ERROR "Could not find nanobind cmake directory")
endif()

find_package(nanobind CONFIG REQUIRED)

# Collect source files
set(TIFFFILE_CPP_SOURCES
    tifffile/src/cpp/common.cpp
    tifffile/src/cpp/ifd_parser.cpp
    tifffile/src/cpp/file_reader.cpp
    tifffile/src/cpp/selection.cpp
)

set(TIFFFILE_BINDING_SOURCES
    tifffile/src/bindings/module.cpp
    tifffile/src/bindings/bind_enums.cpp
    tifffile/src/bindings/bind_format.cpp
    tifffile/src/bindings/bind_ifd.cpp
    tifffile/src/bindings/bind_file_reader.cpp
    tifffile/src/bindings/bind_selection.cpp
)

# Build the extension module
nanobind_add_module(_tifffile_ext
    ${TIFFFILE_BINDING_SOURCES}
    ${TIFFFILE_CPP_SOURCES}
)

target_include_directories(_tifffile_ext PRIVATE
    tifffile/src/include
)

# Install into the tifffile package directory
install(TARGETS _tifffile_ext LIBRARY DESTINATION tifffile)
