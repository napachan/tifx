#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "tifffile/common.hpp"

namespace tifffile {

class FileReader {
public:
    // Open and memory-map a file by path
    explicit FileReader(const std::string& path);

    // Wrap an existing buffer (does not own the data)
    FileReader(const uint8_t* data, size_t size);

    ~FileReader();

    // Non-copyable
    FileReader(const FileReader&) = delete;
    FileReader& operator=(const FileReader&) = delete;

    // Movable
    FileReader(FileReader&& other) noexcept;
    FileReader& operator=(FileReader&& other) noexcept;

    // Access
    const uint8_t* data() const { return data_; }
    size_t size() const { return size_; }
    bool is_mmap() const { return owns_mmap_; }

    // Read a range (returns a copy)
    std::vector<uint8_t> read(uint64_t offset, size_t count) const;

    // Direct access to a range (zero-copy, only valid while FileReader alive)
    const uint8_t* ptr_at(uint64_t offset, size_t count) const;

private:
    void cleanup() noexcept;

    const uint8_t* data_ = nullptr;
    size_t size_ = 0;
    bool owns_mmap_ = false;

    // Platform-specific handles stored as intptr_t
    // Windows: handle1_ = HANDLE for file, handle2_ = HANDLE for mapping
    // POSIX: handle1_ = file descriptor
    intptr_t handle1_ = -1;
    intptr_t handle2_ = 0;
};

}  // namespace tifffile
