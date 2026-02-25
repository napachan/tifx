#include "tifffile/file_reader.hpp"

#include <algorithm>
#include <cstring>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#else
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace tifffile {

#ifdef _WIN32

FileReader::FileReader(const std::string& path) {
    HANDLE hFile = CreateFileA(
        path.c_str(),
        GENERIC_READ,
        FILE_SHARE_READ | FILE_SHARE_WRITE,
        NULL,
        OPEN_EXISTING,
        FILE_ATTRIBUTE_NORMAL,
        NULL
    );
    if (hFile == INVALID_HANDLE_VALUE) {
        throw TiffFileError("cannot open file: " + path);
    }
    handle1_ = reinterpret_cast<intptr_t>(hFile);

    LARGE_INTEGER fileSize;
    if (!GetFileSizeEx(hFile, &fileSize)) {
        CloseHandle(hFile);
        handle1_ = -1;
        throw TiffFileError("cannot get file size: " + path);
    }
    size_ = static_cast<size_t>(fileSize.QuadPart);

    if (size_ == 0) {
        data_ = nullptr;
        owns_mmap_ = false;
        return;
    }

    HANDLE hMapping = CreateFileMappingA(hFile, NULL, PAGE_READONLY, 0, 0, NULL);
    if (hMapping == NULL) {
        CloseHandle(hFile);
        handle1_ = -1;
        throw TiffFileError("cannot create file mapping: " + path);
    }
    handle2_ = reinterpret_cast<intptr_t>(hMapping);

    data_ = static_cast<const uint8_t*>(
        MapViewOfFile(hMapping, FILE_MAP_READ, 0, 0, 0)
    );
    if (data_ == nullptr) {
        CloseHandle(hMapping);
        handle2_ = 0;
        CloseHandle(hFile);
        handle1_ = -1;
        throw TiffFileError("cannot map file: " + path);
    }
    owns_mmap_ = true;
}

void FileReader::cleanup() noexcept {
    if (owns_mmap_ && data_ != nullptr) {
        UnmapViewOfFile(data_);
    }
    if (handle2_ != 0) {
        CloseHandle(reinterpret_cast<HANDLE>(handle2_));
        handle2_ = 0;
    }
    if (handle1_ != -1) {
        CloseHandle(reinterpret_cast<HANDLE>(handle1_));
        handle1_ = -1;
    }
    data_ = nullptr;
    size_ = 0;
    owns_mmap_ = false;
}

#else

FileReader::FileReader(const std::string& path) {
    int fd = ::open(path.c_str(), O_RDONLY);
    if (fd < 0) {
        throw TiffFileError("cannot open file: " + path);
    }
    handle1_ = static_cast<intptr_t>(fd);

    struct stat st;
    if (fstat(fd, &st) != 0) {
        ::close(fd);
        handle1_ = -1;
        throw TiffFileError("cannot stat file: " + path);
    }
    size_ = static_cast<size_t>(st.st_size);

    if (size_ == 0) {
        data_ = nullptr;
        owns_mmap_ = false;
        return;
    }

    void* mapped = ::mmap(nullptr, size_, PROT_READ, MAP_PRIVATE, fd, 0);
    if (mapped == MAP_FAILED) {
        ::close(fd);
        handle1_ = -1;
        throw TiffFileError("cannot mmap file: " + path);
    }
    data_ = static_cast<const uint8_t*>(mapped);
    owns_mmap_ = true;
}

void FileReader::cleanup() noexcept {
    if (owns_mmap_ && data_ != nullptr) {
        ::munmap(const_cast<uint8_t*>(data_), size_);
    }
    int fd = static_cast<int>(handle1_);
    if (fd >= 0) {
        ::close(fd);
        handle1_ = -1;
    }
    data_ = nullptr;
    size_ = 0;
    owns_mmap_ = false;
}

#endif

// Common implementation

FileReader::FileReader(const uint8_t* data, size_t size)
    : data_(data), size_(size), owns_mmap_(false) {}

FileReader::~FileReader() {
    cleanup();
}

FileReader::FileReader(FileReader&& other) noexcept
    : data_(other.data_),
      size_(other.size_),
      owns_mmap_(other.owns_mmap_),
      handle1_(other.handle1_),
      handle2_(other.handle2_)
{
    other.data_ = nullptr;
    other.size_ = 0;
    other.owns_mmap_ = false;
    other.handle1_ = -1;
    other.handle2_ = 0;
}

FileReader& FileReader::operator=(FileReader&& other) noexcept {
    if (this != &other) {
        cleanup();
        data_ = other.data_;
        size_ = other.size_;
        owns_mmap_ = other.owns_mmap_;
        handle1_ = other.handle1_;
        handle2_ = other.handle2_;
        other.data_ = nullptr;
        other.size_ = 0;
        other.owns_mmap_ = false;
        other.handle1_ = -1;
        other.handle2_ = 0;
    }
    return *this;
}

std::vector<uint8_t> FileReader::read(uint64_t offset, size_t count) const {
    if (data_ == nullptr || offset >= size_) {
        return {};
    }
    size_t avail = size_ - static_cast<size_t>(offset);
    size_t n = std::min(count, avail);
    std::vector<uint8_t> buf(n);
    std::memcpy(buf.data(), data_ + offset, n);
    return buf;
}

const uint8_t* FileReader::ptr_at(uint64_t offset, size_t count) const {
    if (data_ == nullptr || offset + count > size_) {
        throw TiffFileError("read beyond end of file");
    }
    return data_ + offset;
}

}  // namespace tifffile
