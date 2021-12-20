cmake_minimum_required(VERSION 3.10)
project(spECK LANGUAGES CXX CUDA VERSION 1.0)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Release")
  message(STATUS "Build type not specified: Use Release by default")
endif(NOT CMAKE_BUILD_TYPE)

# Choose for which CC to build and if to enable Debug
option(CUDA_BUILD_CC50 "Build with compute capability 5.0 support" FALSE)
option(CUDA_BUILD_CC52 "Build with compute capability 5.2 support" FALSE)
option(CUDA_BUILD_CC61 "Build with compute capability 6.1 support" FALSE)
option(CUDA_BUILD_CC70 "Build with compute capability 7.0 support" FALSE)
option(CUDA_BUILD_CC75 "Build with compute capability 7.5 support" FALSE)
option(CUDA_BUILD_CC80 "Build with compute capability 8.0 support" FALSE)
option(CUDA_BUILD_CC86 "Build with compute capability 8.6 support" TRUE)
option(CUDA_BUILD_INFO "Build with kernel statistics and line numbers" FALSE)
option(CUDA_BUILD_DEBUG "Build with kernel debug" FALSE)


# Include Directories
INCLUDE_DIRECTORIES(include)
INCLUDE_DIRECTORIES(externals)
INCLUDE_DIRECTORIES(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})


##########################################################################
# CUDA Flags
if (CUDA_BUILD_CC50)
	string(APPEND CMAKE_CUDA_FLAGS " -gencode=arch=compute_50,code=sm_50")
endif ()
if (CUDA_BUILD_CC52)
	string(APPEND CMAKE_CUDA_FLAGS " -gencode=arch=compute_52,code=sm_52")
endif ()
if (CUDA_BUILD_CC61)
	string(APPEND CMAKE_CUDA_FLAGS " -gencode=arch=compute_61,code=sm_61")
endif ()
if (CUDA_BUILD_CC70)
	string(APPEND CMAKE_CUDA_FLAGS " -gencode=arch=compute_61,code=sm_70")
endif ()
if (CUDA_BUILD_CC75)
	string(APPEND CMAKE_CUDA_FLAGS " -gencode=arch=compute_61,code=sm_75")
endif ()
if (CUDA_BUILD_CC80)
	string(APPEND CMAKE_CUDA_FLAGS " -gencode=arch=compute_61,code=sm_80")
endif ()
if (CUDA_BUILD_CC86)
	string(APPEND CMAKE_CUDA_FLAGS " -gencode=arch=compute_61,code=sm_86")
endif ()

string(APPEND CMAKE_CUDA_FLAGS "  -Xcompiler -Wall -D_FORCE_INLINES --expt-extended-lambda -use_fast_math --expt-relaxed-constexpr")

if (CUDA_BUILD_INFO)
	string(APPEND CMAKE_CUDA_FLAGS " -keep --ptxas-options=-v -lineinfo")
endif ()

if (CUDA_BUILD_DEBUG)
	string(APPEND CMAKE_CUDA_FLAGS " -G")
endif ()

##########################################################################
# CXX Flags
if(WIN32)
set(CUDA_PROPAGATE_HOST_FLAGS ON)
if(MSVC)
  string(APPEND CMAKE_CXX_FLAGS "/wd4464 /wd4514 /wd4820 /wd4668 /wd4574 /wd4571 /wd4324 /wd4710 /wd4711 /wd4365 /wd4515 /wd4201 /wd4267 /wd5027 /wd4626")
endif()
else()
set(CUDA_PROPAGATE_HOST_FLAGS ON)
SET(GCC_COVERAGE_LINK_FLAGS  "-lstdc++fs")
# string(APPEND CMAKE_CXX_FLAGS "-std=c++17 -Wno-unknown-pragmas")
string(APPEND CMAKE_CXX_FLAGS "-Wno-unknown-pragmas")
endif()

SET(SPECKHEADERS
	include/common.cuh 
	include/common.h 
	include/Compare.h 
	include/Config.h
	include/spECKConfig.h
	include/INIReader.h
	include/COO.h 
	include/CSR.h 
	include/dCSR.h 
	include/HashMap.cuh 
	include/meta_utils.h 
	include/multi_arch_build.h 
	include/Multiply.h 
	include/Transpose.h 
	include/Vector.h
	include/GPU/spECK_HashSpGEMM.cuh 
	include/GPU/spECK_HashLoadBalancer.cuh
	include/GPU/BlockRange.cuh 
	include/GPU/consistent_gpu_memory.h 
	include/GPU/HelperFunctions.cuh
	include/GPU/limits.cuh
	include/GPU/Hash.cuh 
	include/GPU/spECKKernels.h 
	include/GPU/scan_largearray_kernel.cuh
	include/CUDATools/error.h 
	include/CUDATools/event.h
	include/CUDATools/memory.h
	include/CUDATools/memory_space.h
	include/CUDATools/stream.h 
	include/CUDATools/unique_handle.h
	include/WorkDistribution.h
)

# spECK Library
add_library(spECKLib
				source/GPU/Transpose.cu
				source/GPU/memory.cpp
				source/GPU/Compare.cu
				source/COO.cpp
				source/CSR.cpp
				source/dCSR.cpp
				source/Config.cpp
				source/GPU/Multiply.cu
				${SPECKHEADERS}
)
set_property(TARGET spECKLib PROPERTY CUDA_STANDARD 17)

# For transposing matrices
SET(COMP_SOURCES "externals/cusparse/source/cuSparseMultiply.cu")

# Executable to run spECK
add_executable(runspECK
				${COMP_SOURCES}
				source/DataLoader.cpp
				source/RunConfig.cpp
				source/Executor.cpp
				source/runspECK.cpp
				include/DataLoader.h
				include/Executor.h
				include/RunConfig.h
				include/spECKConfig.h
)
set_property(TARGET runspECK PROPERTY CUDA_STANDARD 17)

# Link Libraries
TARGET_LINK_LIBRARIES(runspECK spECKLib cusparse ${GCC_COVERAGE_LINK_FLAGS})
