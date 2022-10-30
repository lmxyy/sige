# We do not need cmake for PyTorch, this CMakeLists is just for debugging
cmake_minimum_required(VERSION 3.21)

set(CMAKE_CUDA_COMPILER "/usr/local/cuda/bin/nvcc")
project(sige LANGUAGES C CXX CUDA)

add_library(
        sige-cuda SHARED
        sige/cuda/gather.cpp
        sige/cuda/gather_kernel.cu
        sige/cuda/scatter.cpp
        sige/cuda/scatter_kernel.cu
        sige/cuda/common_cuda.cu
        sige/cuda/scatter_gather.cpp
        sige/cuda/scatter_gather_kernel.cu
        sige/cuda/pybind_cuda.cpp
        sige/common.cpp)

add_library(
        sige-cpu SHARED
        sige/cpu/gather.cpp
        sige/cpu/gather.h
        sige/cpu/scatter.cpp
        sige/cpu/scatter.h
        sige/cpu/scatter_gather.cpp
        sige/cpu/scatter_gather.h
        sige/cpu/common_cpu.cpp
        sige/cpu/pybind_cpu.cpp
        sige/common.cpp)

list(APPEND CMAKE_PREFIX_PATH
        "/home/lmxyy/.conda/envs/sige/lib/python3.8/site-packages/torch")

find_package(Torch REQUIRED)

target_include_directories(
        sige-cuda
        PRIVATE ${TORCH_INCLUDE_DIRS}
        PRIVATE /home/lmxyy/.conda/envs/sige/include/python3.8
        PRIVATE CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES)

target_include_directories(
        sige-cpu
        PRIVATE ${TORCH_INCLUDE_DIRS}
        PRIVATE /home/lmxyy/.conda/envs/sige/include/python3.8)
