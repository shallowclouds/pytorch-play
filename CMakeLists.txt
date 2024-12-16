cmake_minimum_required(VERSION 3.18)
project(pytorch_static_example)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Force static linking for C++ runtime and exclude duplicate symbols
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -static-libstdc++ -static-libgcc -Wl,--exclude-libs,ALL -Wl,--unresolved-symbols=ignore-in-object-files")
set(BUILD_SHARED_LIBS OFF)

# Add compile definitions at top level to ensure they affect all targets
add_definitions(
    -DUSE_MKL
    -DUSE_MKLDNN
    -DMKL_ILP64
    -DTH_BLAS_MKL
    -DAT_PARALLEL_OPENMP=1
    -DAT_MKL_ENABLED=1
    -DAT_MKLDNN_ENABLED=1
    -DUSE_DISTRIBUTED=0
    -DUSE_RPC=0
    -DUSE_TENSORPIPE=0
    -DC10_MOBILE=1
    -DUSE_TENSORPIPE_UNDEF=1
    -DUSE_DISTRIBUTED_UNDEF=1
    -DUSE_RPC_UNDEF=1
    -DUSE_TENSORPIPE_STATIC=0
    -DUSE_DISTRIBUTED_STATIC=0
    -DUSE_RPC_STATIC=0
)

# Set MKL paths and configuration
set(MKL_ROOT "/usr/lib/x86_64-linux-gnu")
set(MKL_INCLUDE_DIR "/usr/include/mkl")

# Add include directories
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/libtorch/include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/libtorch/include/torch/csrc/api/include)
include_directories(${MKL_INCLUDE_DIR})

# Find OpenMP
find_package(OpenMP REQUIRED)

add_executable(example example.cpp)

# Set properties for example executable
set_target_properties(example PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    CXX_VISIBILITY_PRESET hidden
)

# Add OpenMP flags to compilation
target_compile_options(example PRIVATE ${OpenMP_CXX_FLAGS})

# Link against libraries with specific order and linking control
target_link_libraries(example
    # Core PyTorch libraries (CPU only, static)
    -Wl,--whole-archive
    ${CMAKE_CURRENT_SOURCE_DIR}/libtorch/lib/libprotobuf.a
    ${CMAKE_CURRENT_SOURCE_DIR}/libtorch/lib/libc10.a
    ${CMAKE_CURRENT_SOURCE_DIR}/libtorch/lib/libtorch_cpu.a
    ${CMAKE_CURRENT_SOURCE_DIR}/libtorch/lib/libcaffe2_protos.a
    ${CMAKE_CURRENT_SOURCE_DIR}/libtorch/lib/libasmjit.a
    ${CMAKE_CURRENT_SOURCE_DIR}/libtorch/lib/libfbgemm.a
    ${CMAKE_CURRENT_SOURCE_DIR}/libtorch/lib/libcpuinfo.a
    ${CMAKE_CURRENT_SOURCE_DIR}/libtorch/lib/libpthreadpool.a
    ${CMAKE_CURRENT_SOURCE_DIR}/libtorch/lib/libdnnl.a
    ${CMAKE_CURRENT_SOURCE_DIR}/libtorch/lib/sleef/libsleef.a
    ${CMAKE_CURRENT_SOURCE_DIR}/libtorch/lib/libclog.a
    ${CMAKE_CURRENT_SOURCE_DIR}/libtorch/lib/libfmt.a
    -Wl,--no-whole-archive

    # MKL libraries (static)
    -Wl,--start-group
    ${MKL_ROOT}/libmkl_intel_ilp64.a
    ${MKL_ROOT}/libmkl_gnu_thread.a
    ${MKL_ROOT}/libmkl_core.a
    -Wl,--end-group

    # System libraries (dynamic)
    pthread
    numa
    z
    m
    dl
    rt
    ${OpenMP_CXX_LIBRARIES}
)
